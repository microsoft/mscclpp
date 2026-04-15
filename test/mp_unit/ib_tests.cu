// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <mpi.h>

#include <atomic>
#include <mscclpp/atomic_device.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <thread>

#include "gdr.hpp"
#include "mp_unit_tests.hpp"
#include "utils_internal.hpp"

void IbTestBase::SetUp() {
  MSCCLPP_CUDATHROW(cudaGetDeviceCount(&cudaDevNum));
  cudaDevId = (gEnv->rank % gEnv->nRanksPerNode) % cudaDevNum;
  MSCCLPP_CUDATHROW(cudaSetDevice(cudaDevId));

  int ibDevId = (gEnv->rank % gEnv->nRanksPerNode) % mscclpp::getIBDeviceCount();
  ibDevName = mscclpp::getIBDeviceName(ibIdToTransport(ibDevId));
}

void IbPeerToPeerTest::SetUp() {
  REQUIRE_IBVERBS;

  IbTestBase::SetUp();

  mscclpp::UniqueId id;

  if (gEnv->rank < 2) {
    // This test needs only two ranks
    bootstrap = std::make_shared<mscclpp::TcpBootstrap>(gEnv->rank, 2);
    if (bootstrap->getRank() == 0) id = bootstrap->createUniqueId();
  }
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  if (gEnv->rank >= 2) {
    // This test needs only two ranks
    return;
  }

  bootstrap->initialize(id);

  int ib_gid_index = std::stoi(gEnv->args["ib_gid_index"]);

  ibCtx = std::make_shared<mscclpp::IbCtx>(ibDevName);
  bool noAtomic = !ibCtx->supportsRdmaAtomics();
  // When atomics are not supported, the MemoryConsistency test uses
  // write-with-imm which requires recv WRs on the receiver side.
  int maxRecvWr = noAtomic ? 64 : 0;
  qp = ibCtx->createQp(-1, ib_gid_index, 1024, 1, 8192, maxRecvWr, 64, noAtomic);

  qpInfo[gEnv->rank] = qp->getInfo();
  bootstrap->allGather(qpInfo.data(), sizeof(mscclpp::IbQpInfo));
}

void IbPeerToPeerTest::registerBufferAndConnect(void* buf, size_t size) {
  bufSize = size;
  mr = ibCtx->registerMr(buf, size);
  mrInfo[gEnv->rank] = mr->getInfo();
  bootstrap->allGather(mrInfo.data(), sizeof(mscclpp::IbMrInfo));

  for (int i = 0; i < bootstrap->getNranks(); ++i) {
    if (i == gEnv->rank) continue;
    qp->rtr(qpInfo[i]);
    qp->rts();
    break;
  }
  bootstrap->barrier();
}

void IbPeerToPeerTest::stageSendWrite(uint32_t size, uint64_t wrId, uint64_t srcOffset, uint64_t dstOffset,
                                      bool signaled) {
  const mscclpp::IbMrInfo& remoteMrInfo = mrInfo[(gEnv->rank == 1) ? 0 : 1];
  qp->stageSendWrite(mr.get(), remoteMrInfo, size, wrId, srcOffset, dstOffset, signaled);
}

void IbPeerToPeerTest::stageSendAtomicAdd(uint64_t wrId, uint64_t dstOffset, uint64_t addVal, bool signaled) {
  const mscclpp::IbMrInfo& remoteMrInfo = mrInfo[(gEnv->rank == 1) ? 0 : 1];
  qp->stageSendAtomicAdd(mr.get(), remoteMrInfo, wrId, dstOffset, addVal, signaled);
}

void IbPeerToPeerTest::stageSendWriteWithImm(uint32_t size, uint64_t wrId, uint64_t srcOffset, uint64_t dstOffset,
                                             bool signaled, unsigned int immData) {
  const mscclpp::IbMrInfo& remoteMrInfo = mrInfo[(gEnv->rank == 1) ? 0 : 1];
  qp->stageSendWriteWithImm(mr.get(), remoteMrInfo, size, wrId, srcOffset, dstOffset, signaled, immData);
}

PERF_TEST(IbPeerToPeerTest, SimpleSendRecv) {
  if (gEnv->rank >= 2) {
    // This test needs only two ranks
    return;
  }

  mscclpp::Timer timeout(3);

  const int maxIter = 100000;
  const int nelem = 1;
  auto data = mscclpp::detail::gpuCallocUnique<uint64_t>(nelem);

  registerBufferAndConnect(data.get(), sizeof(uint64_t) * nelem);

  if (gEnv->rank == 1) {
    mscclpp::Timer timer;
    for (int iter = 0; iter < maxIter; ++iter) {
      stageSendWrite(sizeof(uint64_t) * nelem, 0, 0, 0, true);
      qp->postSend();
      bool waiting = true;
      int spin = 0;
      while (waiting) {
        int wcNum = qp->pollSendCq();
        ASSERT_GE(wcNum, 0);
        for (int i = 0; i < wcNum; ++i) {
          int status = qp->getSendWcStatus(i);
          EXPECT_EQ(status, static_cast<int>(mscclpp::WsStatus::Success));
          waiting = false;
          break;
        }
        if (spin++ > 10000000) {
          FAIL() << "Polling is stuck.";
        }
      }
    }
    float us = (float)timer.elapsed();
    ::mscclpp::test::reportPerfResult("latency", us / maxIter, "us/iter");
  }
  bootstrap->barrier();
}

__global__ void kernelMemoryConsistency(uint64_t* data, volatile uint64_t* curIter, volatile int* result,
                                        uint64_t nelem, uint64_t maxIter) {
  if (blockIdx.x != 0) return;

  __shared__ int errs[1024];

  constexpr int FlagWrong = 1;
  constexpr int FlagAbort = 2;

  volatile uint64_t* ptr = data;
  for (uint64_t iter = 1; iter < maxIter + 1; ++iter) {
    int err = 0;

    if (threadIdx.x == 0) {
      *curIter = iter;

      // Wait for the first element arrival (expect equal to iter). Expect that the first element is delivered in
      // a special way that guarantees all other elements are completely delivered.
      uint64_t spin = 0;
      while (ptr[0] != iter) {
        if (spin++ == 1000000) {
          // Assume the program is stuck. Set the abort flag and escape the loop.
          *result |= FlagAbort;
          err = 1;
          break;
        }
      }
    }
    __syncthreads();

    // Check results (expect equal to iter) in backward that is more likely to see the wrong result.
    for (size_t i = nelem - 1 + threadIdx.x; i >= blockDim.x; i -= blockDim.x) {
      if (data[i - blockDim.x] != iter) {
#if 1
        *result |= FlagWrong;
        err = 1;
        break;
#else
        // For debugging purposes: try waiting for the correct result.
        uint64_t spin = 0;
        while (ptr[i - blockDim.x] != iter) {
          if (spin++ == 1000000) {
            *result |= FlagAbort;
            err = 1;
            break;
          }
        }
        if (spin >= 1000000) {
          break;
        }
#endif
      }
    }

    errs[threadIdx.x] = err;
    __threadfence();
    __syncthreads();

    // Check if any error is detected.
    int total_err = 0;
    for (size_t i = 0; i < blockDim.x; ++i) {
      total_err += errs[i];
    }

    if (total_err > 0) {
      // Exit if any error is detected.
      return;
    }
  }
  if (threadIdx.x == 0) {
    *curIter = maxIter + 1;
  }
}

TEST(IbPeerToPeerTest, MemoryConsistency) {
  if (gEnv->rank >= 2) {
    // This test needs only two ranks
    return;
  }

  // Use atomic path if supported by the IB device.
  bool useAtomic = ibCtx->supportsRdmaAtomics();

  const uint64_t signalPeriod = 1024;
  const uint64_t maxIter = 10000;
  const uint64_t nelem = 65536 + 1;
  auto data = mscclpp::detail::gpuCallocUnique<uint64_t>(nelem);

  // For no-atomic mode: allocate a separate signal buffer for write-with-imm destination.
  // The sender writes-with-imm to this buffer; the receiver's CPU thread reads the imm_data
  // from the recv CQ and writes the iteration value to data[0] via GDRCopy atomicStore.
  std::shared_ptr<uint64_t> signalBuf;
  std::unique_ptr<const mscclpp::IbMr> signalMr;
  std::array<mscclpp::IbMrInfo, 2> signalMrInfo{};
  if (!useAtomic) {
    signalBuf = mscclpp::detail::gpuCallocShared<uint64_t>(1);
    signalMr = ibCtx->registerMr(signalBuf.get(), sizeof(uint64_t));
    signalMrInfo[gEnv->rank] = signalMr->getInfo();
    bootstrap->allGather(signalMrInfo.data(), sizeof(mscclpp::IbMrInfo));

    // Pre-post recv WRs for write-with-imm on both ranks
    for (int i = 0; i < 64; ++i) {
      qp->stageRecv(0);
    }
    qp->postRecv();
  }

  registerBufferAndConnect(data.get(), sizeof(uint64_t) * nelem);

  uint64_t res = 0;
  uint64_t iter = 0;

  if (gEnv->rank == 0) {
    // Receiver
    auto curIter = mscclpp::detail::gpuCallocHostUnique<uint64_t>();
    auto result = mscclpp::detail::gpuCallocHostUnique<int>();

    volatile uint64_t* ptrCurIter = (volatile uint64_t*)curIter.get();
    volatile int* ptrResult = (volatile int*)result.get();

    ASSERT_NE(ptrCurIter, nullptr);
    ASSERT_NE(ptrResult, nullptr);
    ASSERT_EQ(*ptrCurIter, 0);
    ASSERT_EQ(*ptrResult, 0);

    // For no-atomic mode: create a GDRCopy mapping for data[0] and start a CPU thread that
    // polls recv CQ and forwards the signal via GDRCopy BAR1 write — the same mechanism
    // used by IBConnection::recvThreadFunc for port channels.
    std::atomic<bool> stopRecvThread(false);
    std::thread recvThread;
    std::unique_ptr<mscclpp::GdrMap> dataGdrMap;
    if (!useAtomic) {
      if (!mscclpp::gdrEnabled()) {
        SKIP_TEST() << "No-atomic mode requires GDRCopy but it is not available.";
      }
      // Create GDRCopy BAR1 mapping for data[0] — same as how connection.cc maps inboundToken_
      dataGdrMap =
          std::make_unique<mscclpp::GdrMap>(std::shared_ptr<void>(data.get(), [](void*) {}),  // non-owning shared_ptr
                                            cudaDevId);

      recvThread = std::thread([&]() {
        while (!stopRecvThread.load(std::memory_order_relaxed)) {
          int wcNum = qp->pollRecvCq();
          if (wcNum <= 0) continue;
          for (int i = 0; i < wcNum; ++i) {
            int status = qp->getRecvWcStatus(i);
            if (status != static_cast<int>(mscclpp::WsStatus::Success)) continue;
            uint64_t val = static_cast<uint64_t>(qp->getRecvWcImmData(i));
            // Write the iteration value to data[0] via GDRCopy BAR1 atomicStore —
            // same pattern as IBConnection::recvThreadFunc.
            mscclpp::atomicStore(dataGdrMap->hostPtr(), val, mscclpp::memoryOrderRelaxed);
            // Re-post recv
            qp->stageRecv(0);
            qp->postRecv();
          }
        }
      });
    }

    kernelMemoryConsistency<<<1, 1024>>>(data.get(), ptrCurIter, ptrResult, nelem, maxIter);
    MSCCLPP_CUDATHROW(cudaGetLastError());

    for (iter = 1; iter < maxIter + 1; ++iter) {
      mscclpp::Timer timeout(5);

      while (*ptrCurIter != iter + 1) {
        res = *ptrResult;
        if (res != 0) break;
      }

      // Send the result to the sender
      res = *ptrResult;
      uint64_t tmp[2];
      tmp[0] = res;
      bootstrap->allGather(tmp, sizeof(uint64_t));

      if (res != 0) break;
    }

    MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

    if (!useAtomic) {
      stopRecvThread.store(true, std::memory_order_relaxed);
      if (recvThread.joinable()) recvThread.join();
    }
  } else if (gEnv->rank == 1) {
    // Sender
    std::vector<uint64_t> hostBuffer(nelem, 0);

    for (iter = 1; iter < maxIter + 1; ++iter) {
      mscclpp::Timer timeout(5);

      // Set data
      for (uint64_t i = 0; i < nelem; i++) {
        hostBuffer[i] = iter;
      }
      mscclpp::gpuMemcpy<uint64_t>(data.get(), hostBuffer.data(), nelem, cudaMemcpyHostToDevice);

      // Need to signal from time to time to empty the IB send queue
      bool signaled = (iter % signalPeriod == 0);

      // Send from the second element to the last
      stageSendWrite(sizeof(uint64_t) * (nelem - 1), 0, sizeof(uint64_t), sizeof(uint64_t), signaled);
      qp->postSend();

      if (useAtomic) {
        // Send the first element using AtomicAdd. The non-posted PCIe atomic operation
        // provides end-to-end ordering: data[1..N] are guaranteed visible when data[0] updates.
        stageSendAtomicAdd(0, 0, 1, false);
        qp->postSend();
      } else {
        // No-atomic mode: send a 0-byte WRITE_WITH_IMM carrying the iteration in imm_data.
        // The receiver's CPU thread polls the recv CQ and writes the value to data[0]
        // via GDRCopy atomicStore.
        // QP ordering guarantees data[1..N] WRITE completes before this write-with-imm.
        const mscclpp::IbMrInfo& remoteSignalMrInfo = signalMrInfo[(gEnv->rank == 1) ? 0 : 1];
        qp->stageSendWriteWithImm(nullptr, remoteSignalMrInfo, 0, 0, 0, 0, false, static_cast<unsigned int>(iter));
        qp->postSend();
      }

      if (signaled) {
        int wcNum = qp->pollSendCq();
        while (wcNum == 0) {
          wcNum = qp->pollSendCq();
        }
        ASSERT_EQ(wcNum, 1);
        int status = qp->getSendWcStatus(0);
        ASSERT_EQ(status, static_cast<int>(mscclpp::WsStatus::Success));
      }

      // Get the result from the receiver
      uint64_t tmp[2];
      bootstrap->allGather(tmp, sizeof(uint64_t));
      res = tmp[0];

      if (res != 0) break;
    }
  }

  if (useAtomic) {
    // With RDMA atomics, memory consistency must be guaranteed.
    if (res & 2) {
      FAIL() << "The receiver is stuck at iteration " << iter << ".";
    }
    EXPECT_EQ(res, 0);
  } else {
    if (res == 0) {
      // No-atomic path works correctly here.
    } else if (res & 2) {
      SKIP_TEST() << "No-atomic signal forwarding: receiver stuck at iteration " << iter
                  << ". NIC DMA and CPU writes are not ordered on this platform.";
    } else {
      SKIP_TEST() << "No-atomic signal forwarding: memory inconsistency detected at iteration " << iter
                  << ". NIC DMA and CPU writes are not ordered on this platform.";
    }
  }
}

PERF_TEST(IbPeerToPeerTest, SimpleAtomicAdd) {
  if (gEnv->rank >= 2) {
    // This test needs only two ranks
    return;
  }
  if (!ibCtx->supportsRdmaAtomics()) {
    SKIP_TEST() << "This test requires RDMA atomics support.";
  }

  mscclpp::Timer timeout(3);

  const int maxIter = 100000;
  const int nelem = 1;
  auto data = mscclpp::detail::gpuCallocUnique<uint64_t>(nelem);

  registerBufferAndConnect(data.get(), sizeof(uint64_t) * nelem);

  if (gEnv->rank == 1) {
    mscclpp::Timer timer;
    for (int iter = 0; iter < maxIter; ++iter) {
      stageSendAtomicAdd(0, 0, 1, true);
      qp->postSend();
      bool waiting = true;
      int spin = 0;
      while (waiting) {
        int wcNum = qp->pollSendCq();
        ASSERT_GE(wcNum, 0);
        for (int i = 0; i < wcNum; ++i) {
          int status = qp->getSendWcStatus(i);
          if (status != static_cast<int>(mscclpp::WsStatus::Success)) {
            FAIL() << "Work completion status error: " << qp->getSendWcStatusString(i);
          }
          waiting = false;
          break;
        }
        if (spin++ > 1000000) {
          FAIL() << "Polling is stuck.";
        }
      }
    }
    float us = (float)timer.elapsed();
    ::mscclpp::test::reportPerfResult("latency", us / maxIter, "us/iter");
  }
  bootstrap->barrier();
}
