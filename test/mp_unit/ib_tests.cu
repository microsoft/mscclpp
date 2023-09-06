// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mpi.h>

#include <mscclpp/cuda_utils.hpp>

#include "infiniband/verbs.h"
#include "mp_unit_tests.hpp"

void IbTestBase::SetUp() {
  MSCCLPP_CUDATHROW(cudaGetDeviceCount(&cudaDevNum));
  cudaDevId = (gEnv->rank % gEnv->nRanksPerNode) % cudaDevNum;
  MSCCLPP_CUDATHROW(cudaSetDevice(cudaDevId));

  int ibDevId = (gEnv->rank % gEnv->nRanksPerNode) / mscclpp::getIBDeviceCount();
  ibDevName = mscclpp::getIBDeviceName(ibIdToTransport(ibDevId));
}

void IbPeerToPeerTest::SetUp() {
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

  ibCtx = std::make_shared<mscclpp::IbCtx>(ibDevName);
  qp = ibCtx->createQp(1024, 1, 8192, 0, 64, 4);

  int remoteRank = (gEnv->rank == 0) ? 1 : 0;

  mscclpp::IbQpInfo localQpInfo = qp->getInfo();
  bootstrap->send(&localQpInfo, sizeof(mscclpp::IbQpInfo), remoteRank, /*tag=*/0);
  bootstrap->recv(&remoteQpInfo, sizeof(mscclpp::IbQpInfo), remoteRank, /*tag=*/0);
}

void IbPeerToPeerTest::registerBuffersAndConnect(const std::vector<void*>& bufList,
                                                 const std::vector<uint32_t>& sizeList) {
  size_t numMrs = bufList.size();
  if (numMrs != sizeList.size()) {
    throw std::runtime_error("bufList.size() != sizeList.size()");
  }

  // Assume the remote side registers the same number of MRs
  std::vector<mscclpp::IbMrInfo> localMrInfo;
  for (size_t i = 0; i < numMrs; ++i) {
    const mscclpp::IbMr* mr = ibCtx->registerMr(bufList[i], sizeList[i]);
    localMrList.push_back(mr);
    localMrInfo.emplace_back(mr->getInfo());
  }

  int remoteRank = (gEnv->rank == 0) ? 1 : 0;

  // Send the number of MRs and the MR info to the remote side
  bootstrap->send(&numMrs, sizeof(numMrs), remoteRank, /*tag=*/0);
  bootstrap->send(localMrInfo.data(), sizeof(mscclpp::IbMrInfo) * numMrs, remoteRank, /*tag=*/1);

  // Receive the number of MRs and the MR info from the remote side
  size_t numRemoteMrs;
  bootstrap->recv(&numRemoteMrs, sizeof(numRemoteMrs), remoteRank, /*tag=*/0);
  remoteMrInfoList.resize(numRemoteMrs);
  bootstrap->recv(remoteMrInfoList.data(), sizeof(mscclpp::IbMrInfo) * numRemoteMrs, remoteRank, /*tag=*/1);

  qp->rtr(remoteQpInfo);
  qp->rts();

  bootstrap->barrier();
}

void IbPeerToPeerTest::stageSend(uint32_t size, uint64_t wrId, uint64_t srcOffset, uint64_t dstOffset, bool signaled) {
  qp->stageSend(localMrList[0], remoteMrInfoList[0], size, wrId, srcOffset, dstOffset, signaled);
}

void IbPeerToPeerTest::stageAtomicAdd(uint64_t wrId, uint64_t dstOffset, uint64_t addVal) {
  qp->stageAtomicAdd(localMrList[0], remoteMrInfoList[0], wrId, dstOffset, addVal);
}

void IbPeerToPeerTest::stageSendWithImm(uint32_t size, uint64_t wrId, uint64_t srcOffset, uint64_t dstOffset,
                                        bool signaled, unsigned int immData) {
  qp->stageSendWithImm(localMrList[0], remoteMrInfoList[0], size, wrId, srcOffset, dstOffset, signaled, immData);
}

TEST_F(IbPeerToPeerTest, SimpleSendRecv) {
  if (gEnv->rank >= 2) {
    // This test needs only two ranks
    return;
  }

  mscclpp::Timer timeout(3);

  const int maxIter = 100000;
  const int nelem = 1;
  auto data = mscclpp::allocUniqueCuda<int>(nelem);

  registerBuffersAndConnect({data.get()}, {sizeof(int) * nelem});

  if (gEnv->rank == 1) {
    mscclpp::Timer timer;
    for (int iter = 0; iter < maxIter; ++iter) {
      stageSend(sizeof(int) * nelem, 0, 0, 0, true);
      qp->postSend();
      bool waiting = true;
      int spin = 0;
      while (waiting) {
        int wcNum = qp->pollCq();
        ASSERT_GE(wcNum, 0);
        for (int i = 0; i < wcNum; ++i) {
          const ibv_wc* wc = qp->getWc(i);
          EXPECT_EQ(wc->status, IBV_WC_SUCCESS);
          waiting = false;
          break;
        }
        if (spin++ > 1000000) {
          FAIL() << "Polling is stuck.";
        }
      }
    }
    float us = (float)timer.elapsed();
    std::cout << "IbPeerToPeerTest.SimpleSendRecv: " << us / maxIter << " us/iter" << std::endl;
  }
  bootstrap->barrier();
}

__global__ void kernelMemoryConsistency(uint64_t* data, volatile uint64_t* curIter, volatile int* result,
                                        uint64_t nelem, uint64_t maxIter) {
  if (blockIdx.x != 0) return;

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
    __threadfence();
    __syncthreads();

    // Shuffle err
    for (int i = 16; i > 0; i /= 2) {
      err += __shfl_xor_sync(0xffffffff, err, i);
    }

    if (err > 0) {
      // Exit if any error is detected.
      return;
    }
  }
  if (threadIdx.x == 0) {
    *curIter = maxIter + 1;
  }
}

TEST_F(IbPeerToPeerTest, MemoryConsistency) {
  if (gEnv->rank >= 2) {
    // This test needs only two ranks
    return;
  }

  const uint64_t signalPeriod = 1024;
  const uint64_t maxIter = 10000;
  const uint64_t nelem = 65536 + 1;
  auto data = mscclpp::allocUniqueCuda<uint64_t>(nelem);

  registerBuffersAndConnect({data.get()}, {sizeof(uint64_t) * nelem});

  uint64_t res = 0;
  uint64_t iter = 0;

  if (gEnv->rank == 0) {
    // Receiver
    auto curIter = mscclpp::makeUniqueCudaHost<uint64_t>(0);
    auto result = mscclpp::makeUniqueCudaHost<int>(0);

    volatile uint64_t* ptrCurIter = (volatile uint64_t*)curIter.get();
    volatile int* ptrResult = (volatile int*)result.get();

    ASSERT_EQ(*ptrCurIter, 0);
    ASSERT_EQ(*ptrResult, 0);

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
  } else if (gEnv->rank == 1) {
    // Sender
    std::vector<uint64_t> hostBuffer(nelem, 0);

    for (iter = 1; iter < maxIter + 1; ++iter) {
      mscclpp::Timer timeout(5);

      // Set data
      for (uint64_t i = 0; i < nelem; i++) {
        hostBuffer[i] = iter;
      }
      mscclpp::memcpyCuda<uint64_t>(data.get(), hostBuffer.data(), nelem, cudaMemcpyHostToDevice);

      // Need to signal from time to time to empty the IB send queue
      bool signaled = (iter % signalPeriod == 0);

      // Send from the second element to the last
      stageSend(sizeof(uint64_t) * (nelem - 1), 0, sizeof(uint64_t), sizeof(uint64_t), signaled);
      qp->postSend();

#if 0
      // Send the first element using a normal send. This should occasionally see the wrong result.
      stageSend(sizeof(uint64_t), 0, 0, 0, false);
      qp->postSend();
#else
      // For reference: send the first element using AtomicAdd. This should see the correct result.
      stageAtomicAdd(0, 0, 1);
      qp->postSend();
#endif

      if (signaled) {
        int wcNum = qp->pollCq();
        while (wcNum == 0) {
          wcNum = qp->pollCq();
        }
        ASSERT_EQ(wcNum, 1);
        const ibv_wc* wc = qp->getWc(0);
        ASSERT_EQ(wc->status, IBV_WC_SUCCESS);
      }

      // Get the result from the receiver
      uint64_t tmp[2];
      bootstrap->allGather(tmp, sizeof(uint64_t));
      res = tmp[0];

      if (res != 0) break;
    }
  }

  if (res & 2) {
    FAIL() << "The receiver is stuck at iteration " << iter << ".";
  } else if (res != 0 && res != 1) {
    FAIL() << "Unknown error is detected at iteration " << iter << ". res =" << res;
  }

  EXPECT_EQ(res, 0);
}

TEST_F(IbPeerToPeerTest, SendGather) {
  if (gEnv->rank >= 2) {
    // This test needs only two ranks
    return;
  }

  mscclpp::Timer timeout(3);

  const int numDataSrcs = 4;
  const int nelemPerMr = 1024;

  // Gather send from rank 0 to 1
  if (gEnv->rank == 0) {
    std::vector<mscclpp::UniqueCudaPtr<int>> dataList;
    for (int i = 0; i < numDataSrcs; ++i) {
      auto data = mscclpp::allocUniqueCuda<int>(nelemPerMr);
      // Fill in data for correctness check
      std::vector<int> hostData(nelemPerMr, i + 1);
      mscclpp::memcpyCuda<int>(data.get(), hostData.data(), nelemPerMr);
      dataList.emplace_back(std::move(data));
    }

    std::vector<void*> dataRefList;
    for (int i = 0; i < numDataSrcs; ++i) {
      dataRefList.emplace_back(dataList[i].get());
    }

    // For sending a completion signal to the remote side
    uint64_t outboundSema = 1;

    dataRefList.push_back(&outboundSema);

    std::vector<uint32_t> sizeList(numDataSrcs, sizeof(int) * nelemPerMr);
    sizeList.push_back(sizeof(outboundSema));

    registerBuffersAndConnect(dataRefList, sizeList);

    auto& remoteDataMrInfo = remoteMrInfoList[0];
    auto& remoteSemaMrInfo = remoteMrInfoList[1];
    auto& localSemaMr = localMrList[numDataSrcs];

    std::vector<const mscclpp::IbMr*> gatherLocalMrList;
    for (int i = 0; i < numDataSrcs; ++i) {
      gatherLocalMrList.emplace_back(localMrList[i]);
    }
    std::vector<uint32_t> gatherSizeList(numDataSrcs, sizeof(int) * nelemPerMr);
    std::vector<uint32_t> gatherOffsetList(numDataSrcs, 0);

    qp->stageSendGather(gatherLocalMrList, remoteDataMrInfo, gatherSizeList, /*wrId=*/0, gatherOffsetList,
                        /*dstOffset=*/0, /*signaled=*/true);
    qp->postSend();

    qp->stageAtomicAdd(localSemaMr, remoteSemaMrInfo, /*wrId=*/0, /*dstOffset=*/0, /*addVal=*/1);
    qp->postSend();

    // Wait for send completion
    bool waiting = true;
    int spin = 0;
    while (waiting) {
      int wcNum = qp->pollCq();
      ASSERT_GE(wcNum, 0);
      for (int i = 0; i < wcNum; ++i) {
        const ibv_wc* wc = qp->getWc(i);
        EXPECT_EQ(wc->status, IBV_WC_SUCCESS);
        waiting = false;
        break;
      }
      if (spin++ > 1000000) {
        FAIL() << "Polling is stuck.";
      }
    }
  } else {
    // Data array to receive
    auto data = mscclpp::allocUniqueCuda<int>(nelemPerMr * numDataSrcs);

    // For receiving a completion signal from the remote side
    uint64_t inboundSema = 0;

    registerBuffersAndConnect({data.get(), &inboundSema},
                              {sizeof(int) * nelemPerMr * numDataSrcs, sizeof(inboundSema)});

    // Wait for a signal from the remote side
    volatile uint64_t* ptrInboundSema = &inboundSema;
    int spin = 0;
    while (*ptrInboundSema == 0) {
      if (spin++ > 1000000) {
        FAIL() << "Polling is stuck.";
      }
    }

    // Correctness check
    std::vector<int> hostData(nelemPerMr * numDataSrcs);
    mscclpp::memcpyCuda<int>(hostData.data(), data.get(), nelemPerMr * numDataSrcs);
    for (int i = 0; i < numDataSrcs; ++i) {
      for (int j = 0; j < nelemPerMr; ++j) {
        EXPECT_EQ(hostData[i * nelemPerMr + j], i + 1);
      }
    }
  }

  bootstrap->barrier();
}
