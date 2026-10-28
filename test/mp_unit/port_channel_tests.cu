// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mscclpp/concurrency_device.hpp>
#include <numeric>
#include <thread>

#include "gdr.hpp"
#include "mp_unit_tests.hpp"
#include "utils_internal.hpp"

#if defined(MSCCLPP_USE_GPUNETIO)
#include <cooperative_groups.h>
#endif

// Skip the current test if the given IB mode will require GDRCopy on CUDA but it is unavailable.
// On CUDA, HostNoAtomic requires GDRCopy for BAR1 signal forwarding. When IbMode::Host or
// IbMode::Default is used and the IB device does not support RDMA atomics, the endpoint falls
// back to no-atomic mode, which also requires GDRCopy.
// On ROCm, no-atomic mode uses direct volatile writes and does not need GDRCopy.
#if defined(MSCCLPP_USE_CUDA)
inline void requireGdrForIbMode(IbMode mode, mscclpp::Transport ibTransport) {
  if (mscclpp::gdrEnabled()) return;  // GDRCopy available — nothing to skip.
  if (mode == IbMode::HostNoAtomic) {
    SKIP_TEST() << "HostNoAtomic requires GDRCopy on CUDA: " << mscclpp::gdrStatusMessage();
  }
  // For Host/Default modes: check whether the IB device lacks RDMA atomics,
  // which would cause an automatic fallback to no-atomic mode.
  if (mode == IbMode::Host || mode == IbMode::Default) {
    std::string devName = mscclpp::getIBDeviceName(ibTransport);
    mscclpp::IbCtx ibCtx(devName);
    if (!ibCtx.supportsRdmaAtomics()) {
      SKIP_TEST() << "IB device " << devName
                  << " lacks RDMA atomics; Host mode falls back to HostNoAtomic which requires GDRCopy: "
                  << mscclpp::gdrStatusMessage();
    }
  }
}
#define REQUIRE_GDR_FOR_IB_MODE(mode) requireGdrForIbMode((mode), ibTransport)

inline void requireGdrForHostNoAtomicCollective(int firstRank, int secondRank) {
  bool participates = gEnv->rank == firstRank || gEnv->rank == secondRank;
  int localReady = (!participates || mscclpp::gdrEnabled()) ? 1 : 0;
  int allReady = 0;
  MPI_Allreduce(&localReady, &allReady, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (!allReady) {
    SKIP_TEST() << "HostNoAtomic rejection requires GDRCopy on every participating CUDA rank";
  }
}
#else
#define REQUIRE_GDR_FOR_IB_MODE(mode)  // No extra requirements on non-CUDA platforms.
inline void requireGdrForHostNoAtomicCollective(int, int) {}
#endif

inline void requireCudaIpcRankZeroPeers(int lastPeer) {
  uint64_t localHost = mscclpp::getHostHash();
  std::vector<uint64_t> hosts(gEnv->worldSize);
  MPI_Allgather(&localHost, sizeof(localHost), MPI_BYTE, hosts.data(), sizeof(localHost), MPI_BYTE, MPI_COMM_WORLD);

  bool needsFabric = false;
  for (int peer = 1; peer <= lastPeer; ++peer) needsFabric |= hosts[peer] != hosts[0];
  if (!needsFabric) return;

  int localFabric = 0;
  try {
    localFabric = mscclpp::isFabricMemHandleAvailable() ? 1 : 0;
  } catch (...) {
    // A failed capability query is unavailable; every rank makes the same decision below.
  }
  std::vector<int> fabricAvailable(gEnv->worldSize);
  MPI_Allgather(&localFabric, 1, MPI_INT, fabricAvailable.data(), 1, MPI_INT, MPI_COMM_WORLD);
  for (int peer = 1; peer <= lastPeer; ++peer) {
    if (hosts[peer] != hosts[0] && (!fabricAvailable[0] || !fabricAvailable[peer])) {
      SKIP_TEST() << "CudaIpc requires usable fabric memory handles on rank 0 and every cross-host peer";
    }
  }
}

#define REQUIRE_CUDA_IPC_AVAILABLE requireCudaIpcRankZeroPeers(/*lastPeer=*/1)

inline void requireIbRdmaAtomics(mscclpp::Transport ibTransport) {
  int localSupport = 0;
  try {
    std::string devName = mscclpp::getIBDeviceName(ibTransport);
    mscclpp::IbCtx ibCtx(devName);
    localSupport = ibCtx.supportsRdmaAtomics() ? 1 : 0;
  } catch (...) {
    // Treat an unavailable or unusable local IB device as lacking atomic support. The collective
    // below then makes every rank skip together.
  }
  int allSupport = 0;
  MPI_Allreduce(&localSupport, &allSupport, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (!allSupport) {
    SKIP_TEST() << "Positive IB accumulate tests require RDMA atomic support on every rank";
  }
}

template <typename Func>
bool rejectsInvalidUsage(Func func, const char* message) {
  try {
    func();
  } catch (const mscclpp::Error& e) {
    return e.getErrorCode() == mscclpp::ErrorCode::InvalidUsage &&
           std::string(e.what()).find(message) != std::string::npos;
  } catch (...) {
  }
  return false;
}

void PortChannelOneToOneTest::SetUp() {
  // Use only two ranks
  setNumRanksToUse(2);
  CommunicatorTestBase::SetUp();
  proxyService = std::make_shared<mscclpp::ProxyService>();
}

void PortChannelOneToOneTest::TearDown() { CommunicatorTestBase::TearDown(); }

void PortChannelOneToOneTest::setupMeshConnections(std::vector<mscclpp::PortChannel>& portChannels, bool useIPC,
                                                   bool useIb, bool useEthernet, void* sendBuff, size_t sendBuffBytes,
                                                   void* recvBuff, size_t recvBuffBytes, IbMode ibMode) {
  const int rank = communicator->bootstrap()->getRank();
  const int worldSize = communicator->bootstrap()->getNranks();
  const bool isInPlace = (recvBuff == nullptr);
  mscclpp::TransportFlags transport;

  if (useIPC) transport |= mscclpp::Transport::CudaIpc;
  if (useIb) transport |= ibTransport;
  if (useEthernet) transport |= mscclpp::Transport::Ethernet;

  std::vector<std::shared_future<mscclpp::Connection>> connectionFutures(worldSize);
  std::vector<std::shared_future<mscclpp::RegisteredMemory>> remoteMemFutures(worldSize);

  mscclpp::RegisteredMemory sendBufRegMem = communicator->registerMemory(sendBuff, sendBuffBytes, transport);
  mscclpp::RegisteredMemory recvBufRegMem;
  if (!isInPlace) {
    recvBufRegMem = communicator->registerMemory(recvBuff, recvBuffBytes, transport);
  }

  for (int r = 0; r < worldSize; r++) {
    if (r == rank) {
      continue;
    }
    mscclpp::EndpointConfig cfg;
    if (useIPC) {
      // CudaIpc works intra-node always, and cross-node on MNNVL systems (GB200 NVL72 + IMEX)
      // via fabric handles. Tests that exercise CudaIpc across nodes on non-MNNVL hardware should
      // gate themselves with REQUIRE_CUDA_IPC_AVAILABLE; we always request CudaIpc here when asked.
      cfg.transport = mscclpp::Transport::CudaIpc;
    } else if (useIb) {
      cfg.transport = ibTransport;
      cfg.ib.gidIndex = std::stoi(gEnv->args["ib_gid_index"]);
      cfg.ib.mode = ibMode;
    } else if (useEthernet) {
      cfg.transport = mscclpp::Transport::Ethernet;
    }
    connectionFutures[r] = communicator->connect(cfg, r);

    if (isInPlace) {
      communicator->sendMemory(sendBufRegMem, r);
    } else {
      communicator->sendMemory(recvBufRegMem, r);
    }
    remoteMemFutures[r] = communicator->recvMemory(r);
  }

  for (int r = 0; r < worldSize; r++) {
    if (r == rank) {
      continue;
    }
    auto sema = communicator->buildSemaphore(connectionFutures[r].get(), r).get();

    mscclpp::SemaphoreId cid = proxyService->addSemaphore(sema);

    portChannels.emplace_back(proxyService->portChannel(cid, proxyService->addMemory(remoteMemFutures[r].get()),
                                                        proxyService->addMemory(sendBufRegMem)));
  }
  // Keep memory reference
  registeredMemories.push_back(recvBufRegMem);
}

__constant__ DeviceHandle<mscclpp::PortChannel> gChannelOneToOneTestConstPortChans;

__global__ void kernelProxyPingPong(int* buff, int rank, int nElem, bool waitWithPoll, int nTries, int* ret) {
  DeviceHandle<mscclpp::PortChannel>& portChan = gChannelOneToOneTestConstPortChans;
  volatile int* sendBuff = (volatile int*)buff;
  int flusher = 0;
  int rank1Offset = 10000000;
  for (int i = 0; i < nTries; i++) {
    if (rank == 0) {
      if (i > 0) {
        if (threadIdx.x == 0) {
          if (waitWithPoll) {
            int spin = 1000000;
            while (!portChan.poll() && spin > 0) {
              spin--;
            }
            if (spin == 0) {
              // printf("rank 0 ERROR: poll timeout\n");
              *ret = 1;
            }
          } else {
            portChan.wait();
          }
        }
        __syncthreads();
        for (int j = threadIdx.x; j < nElem; j += blockDim.x) {
          if (sendBuff[j] != rank1Offset + i - 1 + j) {
            // printf("rank 0 ERROR: sendBuff[%d] = %d, expected %d\n", j, sendBuff[j], rank1Offset + i - 1 + j);
            *ret = 1;
            break;
          }
        }
      }
      for (int j = threadIdx.x; j < nElem; j += blockDim.x) {
        sendBuff[j] = i + j;
      }
      __syncthreads();
      // __threadfence_system(); // not necessary if we make sendBuff volatile
      if (threadIdx.x == 0) portChan.putWithSignal(0, nElem * sizeof(int));
    }
    if (rank == 1) {
      if (threadIdx.x == 0) {
        if (waitWithPoll) {
          int spin = 1000000;
          while (!portChan.poll() && spin > 0) {
            spin--;
          }
          if (spin == 0) {
            // printf("rank 0 ERROR: poll timeout\n");
            *ret = 1;
          }
        } else {
          portChan.wait();
        }
      }
      __syncthreads();
      for (int j = threadIdx.x; j < nElem; j += blockDim.x) {
        if (sendBuff[j] != i + j) {
          // printf("rank 1 ERROR: sendBuff[%d] = %d, expected %d\n", j, sendBuff[j], i + j);
          *ret = 1;
          break;
        }
      }
      if (i < nTries - 1) {
        for (int j = threadIdx.x; j < nElem; j += blockDim.x) {
          sendBuff[j] = rank1Offset + i + j;
        }
        __syncthreads();
        // __threadfence_system(); // not necessary if we make sendBuff volatile
        if (threadIdx.x == 0) portChan.putWithSignal(0, nElem * sizeof(int));
      }
    }
    flusher++;
    if (flusher == 1) {
      if (threadIdx.x == 0) portChan.flush();
      flusher = 0;
    }
  }
}

void PortChannelOneToOneTest::testPingPong(PingPongTestParams params) {
  if (gEnv->rank >= numRanksToUse) return;

  const int nElem = 4 * 1024 * 1024;

  std::vector<mscclpp::PortChannel> portChannels;
  std::shared_ptr<int> buff = mscclpp::GpuBuffer<int>(nElem).memory();
  setupMeshConnections(portChannels, params.useIPC, params.useIB, params.useEthernet, buff.get(), nElem * sizeof(int),
                       nullptr, 0, params.ibMode);

  std::vector<DeviceHandle<mscclpp::PortChannel>> portChannelHandles;
  for (auto& ch : portChannels) portChannelHandles.push_back(ch.deviceHandle());

  ASSERT_EQ(portChannels.size(), 1);
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestConstPortChans, portChannelHandles.data(),
                                       sizeof(DeviceHandle<mscclpp::PortChannel>)));

  proxyService->startProxy();

  std::shared_ptr<int> ret = mscclpp::detail::gpuCallocHostShared<int>();

  const int nTries = 1000;
  for (int nElem : {1, 1024, 1024 * 1024, 4 * 1024 * 1024}) {
    *ret = 0;
    kernelProxyPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, nElem, params.waitWithPoll, nTries, ret.get());
    MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
    EXPECT_EQ(*ret, 0);
    // Rank 0's next launch writes to rank 1 without waiting; keep it from racing rank 1's final check.
    communicator->bootstrap()->barrier();
  }

  proxyService->stopProxy();
}

void PortChannelOneToOneTest::testPingPongPerf(PingPongTestParams params) {
  if (gEnv->rank >= numRanksToUse) return;

  const int nElem = 4 * 1024 * 1024;

  std::vector<mscclpp::PortChannel> portChannels;
  std::shared_ptr<int> buff = mscclpp::GpuBuffer<int>(nElem).memory();
  setupMeshConnections(portChannels, params.useIPC, params.useIB, params.useEthernet, buff.get(), nElem * sizeof(int),
                       nullptr, 0, params.ibMode);

  std::vector<DeviceHandle<mscclpp::PortChannel>> portChannelHandles;
  for (auto& ch : portChannels) portChannelHandles.push_back(ch.deviceHandle());

  ASSERT_EQ(portChannels.size(), 1);
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestConstPortChans, portChannelHandles.data(),
                                       sizeof(DeviceHandle<mscclpp::PortChannel>)));

  proxyService->startProxy();

  std::shared_ptr<int> ret = mscclpp::detail::gpuCallocHostShared<int>();

  const std::string testName = ::mscclpp::test::currentTestName();
  const int nTries = 1000;

  // Warm-up
  kernelProxyPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1, params.waitWithPoll, nTries, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  communicator->bootstrap()->barrier();

  // Measure latency
  mscclpp::Timer timer;
  kernelProxyPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1, params.waitWithPoll, nTries, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  communicator->bootstrap()->barrier();

  if (gEnv->rank == 0) {
    ::mscclpp::test::reportPerfResult("latency", (float)timer.elapsed() / (float)nTries, "us/iter");
  }

  proxyService->stopProxy();
}

TEST(PortChannelOneToOneTest, PingPong) {
  REQUIRE_CUDA_IPC_AVAILABLE;
  testPingPong(PingPongTestParams{
      .useIPC = true, .useIB = false, .useEthernet = false, .waitWithPoll = false, .ibMode = IbMode::Default});
}

TEST(PortChannelOneToOneTest, PingPongIbHostMode) {
  REQUIRE_IBVERBS;
  REQUIRE_GDR_FOR_IB_MODE(IbMode::Host);
  testPingPong(PingPongTestParams{
      .useIPC = false, .useIB = true, .useEthernet = false, .waitWithPoll = false, .ibMode = IbMode::Host});
}

TEST(PortChannelOneToOneTest, PingPongEthernet) {
  testPingPong(PingPongTestParams{
      .useIPC = false, .useIB = false, .useEthernet = true, .waitWithPoll = false, .ibMode = IbMode::Default});
}

TEST(PortChannelOneToOneTest, PingPongWithPoll) {
  REQUIRE_CUDA_IPC_AVAILABLE;
  testPingPong(PingPongTestParams{
      .useIPC = true, .useIB = false, .useEthernet = false, .waitWithPoll = true, .ibMode = IbMode::Default});
}

TEST(PortChannelOneToOneTest, PingPongIbHostModeWithPoll) {
  REQUIRE_IBVERBS;
  REQUIRE_GDR_FOR_IB_MODE(IbMode::Host);
  testPingPong(PingPongTestParams{
      .useIPC = false, .useIB = true, .useEthernet = false, .waitWithPoll = true, .ibMode = IbMode::Host});
}

PERF_TEST(PortChannelOneToOneTest, PingPongPerf) {
  REQUIRE_CUDA_IPC_AVAILABLE;
  testPingPongPerf(PingPongTestParams{
      .useIPC = true, .useIB = false, .useEthernet = false, .waitWithPoll = false, .ibMode = IbMode::Default});
}

PERF_TEST(PortChannelOneToOneTest, PingPongPerfIbHostMode) {
  REQUIRE_IBVERBS;
  REQUIRE_GDR_FOR_IB_MODE(IbMode::Host);
  testPingPongPerf(PingPongTestParams{
      .useIPC = false, .useIB = true, .useEthernet = false, .waitWithPoll = false, .ibMode = IbMode::Host});
}

PERF_TEST(PortChannelOneToOneTest, PingPongPerfIbHostNoAtomicMode) {
  REQUIRE_IBVERBS;
  REQUIRE_GDR_FOR_IB_MODE(IbMode::HostNoAtomic);
  testPingPongPerf(PingPongTestParams{
      .useIPC = false, .useIB = true, .useEthernet = false, .waitWithPoll = false, .ibMode = IbMode::HostNoAtomic});
}

PERF_TEST(PortChannelOneToOneTest, PingPongPerfEthernet) {
  testPingPongPerf(PingPongTestParams{
      .useIPC = false, .useIB = false, .useEthernet = true, .waitWithPoll = false, .ibMode = IbMode::Default});
}

__device__ mscclpp::DeviceSyncer gChannelOneToOneTestPortChansSyncer;

template <bool CheckCorrectness, bool RotateSubmitter = false>
__global__ void kernelProxyLLPingPong(int* buff, mscclpp::LLPacket* putPktBuf, mscclpp::LLPacket* getPktBuf, int rank,
                                      int nElem, int nTries, int* ret) {
  if (rank > 1) return;

  DeviceHandle<mscclpp::PortChannel>& portChan = gChannelOneToOneTestConstPortChans;
  volatile int* buffPtr = (volatile int*)buff;
  int putOffset = (rank == 0) ? 0 : 10000000;
  int getOffset = (rank == 0) ? 10000000 : 0;
  int threadId = threadIdx.x + blockIdx.x * blockDim.x;
  int numThreads = blockDim.x * gridDim.x;
  int flusher = 0;
  const int nPkt = nElem / 2;
  for (int i = 0; i < nTries; i++) {
    uint64_t flag = (uint64_t)i + 1;
    const int submitter = RotateSubmitter ? i % numThreads : 0;

    // rank=0: 0, 1, 0, 1, ...
    // rank=1: 1, 0, 1, 0, ...
    if ((rank ^ (i & 1)) == 0) {
      if constexpr (CheckCorrectness) {
        // If each thread writes 8 bytes at once, we don't need a barrier before copyToPackets().
        for (int j = threadId; j < nPkt; j += numThreads) {
          buffPtr[2 * j] = putOffset + i + 2 * j;
          buffPtr[2 * j + 1] = putOffset + i + 2 * j + 1;
        }
        // __syncthreads();
      }
      mscclpp::copyToPackets(putPktBuf, buff, nElem * sizeof(int), threadId, numThreads, flag);
      gChannelOneToOneTestPortChansSyncer.sync(gridDim.x);
      if (threadId == submitter) {
        // Send data from the local putPacketBuffer to the remote getPacketBuffer
        portChan.put(0, nPkt * sizeof(mscclpp::LLPacket));
      }
      flusher++;
      if (flusher == 64) {
        if (threadId == submitter) portChan.flush();
        flusher = 0;
      }
    } else {
      mscclpp::copyFromPackets(buff, getPktBuf, nElem * sizeof(int), threadId, numThreads, flag);
      if constexpr (CheckCorrectness) {
        // If each thread reads 8 bytes at once, we don't need a barrier after copyFromPackets().
        // __syncthreads();
        for (int j = threadId; j < nPkt; j += numThreads) {
          if (buffPtr[2 * j] != getOffset + i + 2 * j) {
            // printf("ERROR: rank = %d, buffPtr[%d] = %d, expected %d. Skipping following errors\n", rank, 2 * j,
            //        buffPtr[2 * j], getOffset + i + 2 * j);
            *ret = 1;
            break;
          }
          if (buffPtr[2 * j + 1] != getOffset + i + 2 * j + 1) {
            // printf("ERROR: rank = %d, buffPtr[%d] = %d, expected %d. Skipping following errors\n", rank, 2 * j + 1,
            //        buffPtr[2 * j + 1], getOffset + i + 2 * j + 1);
            *ret = 1;
            break;
          }
        }
      }
      // Make sure all threads are done in this iteration
      gChannelOneToOneTestPortChansSyncer.sync(gridDim.x);
    }
  }
}

void PortChannelOneToOneTest::testPacketPingPong(bool useIb, IbMode ibMode) {
  if (gEnv->rank >= numRanksToUse) return;

  const int nElem = 4 * 1024 * 1024;

  std::vector<mscclpp::PortChannel> portChannels;
  std::shared_ptr<int> buff = mscclpp::GpuBuffer<int>(nElem).memory();

  const size_t nPacket = (nElem * sizeof(int) + sizeof(uint64_t) - 1) / sizeof(uint64_t);
  auto putPacketBuffer = mscclpp::GpuBuffer<mscclpp::LLPacket>(nPacket).memory();
  auto getPacketBuffer = mscclpp::GpuBuffer<mscclpp::LLPacket>(nPacket).memory();

  setupMeshConnections(portChannels, !useIb, useIb, false, putPacketBuffer.get(), nPacket * sizeof(mscclpp::LLPacket),
                       getPacketBuffer.get(), nPacket * sizeof(mscclpp::LLPacket), ibMode);

  ASSERT_EQ(portChannels.size(), 1);

  std::vector<DeviceHandle<mscclpp::PortChannel>> portChannelHandles;
  for (auto& portChannel : portChannels) {
    portChannelHandles.push_back(portChannel.deviceHandle());
  }

  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestConstPortChans, portChannelHandles.data(),
                                       sizeof(DeviceHandle<mscclpp::PortChannel>)));

  mscclpp::DeviceSyncer syncer = {};
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestPortChansSyncer, &syncer, sizeof(mscclpp::DeviceSyncer)));

  proxyService->startProxy();

  std::shared_ptr<int> ret = mscclpp::detail::gpuCallocHostShared<int>();

  const int nTries = 1000;
  // The least nelem is 2 for packet ping pong
  for (int nElem : {2, 1024, 1024 * 1024, 4 * 1024 * 1024}) {
    *ret = 0;
    if (ibMode == IbMode::GpuNetIo) {
      kernelProxyLLPingPong<true, true>
          <<<1, 1024>>>(buff.get(), putPacketBuffer.get(), getPacketBuffer.get(), gEnv->rank, nElem, nTries, ret.get());
    } else {
      kernelProxyLLPingPong<true>
          <<<1, 1024>>>(buff.get(), putPacketBuffer.get(), getPacketBuffer.get(), gEnv->rank, nElem, nTries, ret.get());
    }
    MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
    EXPECT_EQ(*ret, 0);
  }

  communicator->bootstrap()->barrier();

  proxyService->stopProxy();
}

void PortChannelOneToOneTest::testPacketPingPongPerf(bool useIb, IbMode ibMode) {
  if (gEnv->rank >= numRanksToUse) return;

  const int nElem = 4 * 1024 * 1024;

  std::vector<mscclpp::PortChannel> portChannels;
  std::shared_ptr<int> buff = mscclpp::GpuBuffer<int>(nElem).memory();

  const size_t nPacket = (nElem * sizeof(int) + sizeof(uint64_t) - 1) / sizeof(uint64_t);
  auto putPacketBuffer = mscclpp::GpuBuffer<mscclpp::LLPacket>(nPacket).memory();
  auto getPacketBuffer = mscclpp::GpuBuffer<mscclpp::LLPacket>(nPacket).memory();

  setupMeshConnections(portChannels, !useIb, useIb, false, putPacketBuffer.get(), nPacket * sizeof(mscclpp::LLPacket),
                       getPacketBuffer.get(), nPacket * sizeof(mscclpp::LLPacket), ibMode);

  ASSERT_EQ(portChannels.size(), 1);

  std::vector<DeviceHandle<mscclpp::PortChannel>> portChannelHandles;
  for (auto& portChannel : portChannels) {
    portChannelHandles.push_back(portChannel.deviceHandle());
  }

  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestConstPortChans, portChannelHandles.data(),
                                       sizeof(DeviceHandle<mscclpp::PortChannel>)));

  mscclpp::DeviceSyncer syncer = {};
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestPortChansSyncer, &syncer, sizeof(mscclpp::DeviceSyncer)));

  proxyService->startProxy();

  const std::string testName = ::mscclpp::test::currentTestName();
  const int nTries = 1000000;

  // Warm-up
  kernelProxyLLPingPong<false>
      <<<1, 1024>>>(buff.get(), putPacketBuffer.get(), getPacketBuffer.get(), gEnv->rank, 2, nTries, nullptr);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  communicator->bootstrap()->barrier();

  // Measure latency
  mscclpp::Timer timer;
  kernelProxyLLPingPong<false>
      <<<1, 1024>>>(buff.get(), putPacketBuffer.get(), getPacketBuffer.get(), gEnv->rank, 2, nTries, nullptr);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  communicator->bootstrap()->barrier();

  if (gEnv->rank == 0) {
    ::mscclpp::test::reportPerfResult("latency", (float)timer.elapsed() / (float)nTries, "us/iter");
  }

  proxyService->stopProxy();
}

TEST(PortChannelOneToOneTest, PacketPingPong) {
  REQUIRE_CUDA_IPC_AVAILABLE;
  testPacketPingPong(false, IbMode::Default);
}

TEST(PortChannelOneToOneTest, PacketPingPongIbHostMode) {
  REQUIRE_IBVERBS;
  REQUIRE_GDR_FOR_IB_MODE(IbMode::Host);
  testPacketPingPong(true, IbMode::Host);
}

PERF_TEST(PortChannelOneToOneTest, PacketPingPongPerf) {
  REQUIRE_CUDA_IPC_AVAILABLE;
  testPacketPingPongPerf(false, IbMode::Default);
}

PERF_TEST(PortChannelOneToOneTest, PacketPingPongPerfIbHostMode) {
  REQUIRE_IBVERBS;
  REQUIRE_GDR_FOR_IB_MODE(IbMode::Host);
  testPacketPingPongPerf(true, IbMode::Host);
}

PERF_TEST(PortChannelOneToOneTest, PacketPingPongPerfIbHostNoAtomicMode) {
  REQUIRE_IBVERBS;
  REQUIRE_GDR_FOR_IB_MODE(IbMode::HostNoAtomic);
  testPacketPingPongPerf(true, IbMode::HostNoAtomic);
}

TEST(PortChannelOneToOneTest, PingPongIbHostNoAtomicMode) {
  REQUIRE_IBVERBS;
  REQUIRE_GDR_FOR_IB_MODE(IbMode::HostNoAtomic);
  testPingPong(PingPongTestParams{
      .useIPC = false, .useIB = true, .useEthernet = false, .waitWithPoll = false, .ibMode = IbMode::HostNoAtomic});
}

TEST(PortChannelOneToOneTest, PacketPingPongIbHostNoAtomicMode) {
  REQUIRE_IBVERBS;
  REQUIRE_GDR_FOR_IB_MODE(IbMode::HostNoAtomic);
  testPacketPingPong(true, IbMode::HostNoAtomic);
}

// Bandwidth test: bidirectional bulk transfer matching the tutorial pattern.
// Both ranks do signal+wait+putWithSignal+wait per iteration.
__global__ void kernelBandwidthBidir(int* buff, int nElem, int nIters, int rank) {
  DeviceHandle<mscclpp::PortChannel>& portChan = gChannelOneToOneTestConstPortChans;
  if (threadIdx.x != 0) return;
  const uint64_t srcOffset = rank * nElem * sizeof(int);
  const uint64_t dstOffset = srcOffset;
  for (int i = 0; i < nIters; i++) {
    portChan.signal();
    portChan.wait();
    portChan.putWithSignal(dstOffset, srcOffset, nElem * sizeof(int));
    portChan.wait();
  }
}

void PortChannelOneToOneTest::testBandwidth(PingPongTestParams params) {
  if (gEnv->rank >= numRanksToUse) return;

  const int maxElem = 32 * 1024 * 1024;  // 128 MB per direction
  const int bufElem = maxElem * 2;       // 2x for bidirectional

  std::vector<mscclpp::PortChannel> portChannels;
  std::shared_ptr<int> buff = mscclpp::GpuBuffer<int>(bufElem).memory();
  setupMeshConnections(portChannels, params.useIPC, params.useIB, params.useEthernet, buff.get(), bufElem * sizeof(int),
                       nullptr, 0, params.ibMode);

  std::vector<DeviceHandle<mscclpp::PortChannel>> portChannelHandles;
  for (auto& ch : portChannels) portChannelHandles.push_back(ch.deviceHandle());

  ASSERT_EQ(portChannels.size(), 1);
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestConstPortChans, portChannelHandles.data(),
                                       sizeof(DeviceHandle<mscclpp::PortChannel>)));

  proxyService->startProxy();

  const std::string testName = ::mscclpp::test::currentTestName();
  const int nIters = 1000;

  for (int nElem : {256, 16 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024, 32 * 1024 * 1024}) {
    // Warm-up
    kernelBandwidthBidir<<<1, 1024>>>(buff.get(), nElem, 10, gEnv->rank);
    MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
    communicator->bootstrap()->barrier();

    // Measure
    mscclpp::Timer timer;
    kernelBandwidthBidir<<<1, 1024>>>(buff.get(), nElem, nIters, gEnv->rank);
    MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
    double elapsedUs = timer.elapsed();
    communicator->bootstrap()->barrier();

    if (gEnv->rank == 0) {
      double copyBytes = (double)nElem * sizeof(int);
      double elapsedMsPerIter = elapsedUs / 1e3 / nIters;
      double gbps = copyBytes / elapsedMsPerIter * 1e-6;
      double sizeKB = copyBytes / 1024.0;
      std::string label =
          (sizeKB >= 1024.0) ? (std::to_string((int)(sizeKB / 1024.0)) + " MB") : (std::to_string((int)sizeKB) + " KB");
      ::mscclpp::test::reportPerfResult(label, gbps, "GB/s");
    }
  }

  proxyService->stopProxy();
}

PERF_TEST(PortChannelOneToOneTest, Bandwidth) {
  REQUIRE_CUDA_IPC_AVAILABLE;
  testBandwidth(PingPongTestParams{
      .useIPC = true, .useIB = false, .useEthernet = false, .waitWithPoll = false, .ibMode = IbMode::Default});
}

PERF_TEST(PortChannelOneToOneTest, BandwidthIbHostMode) {
  REQUIRE_IBVERBS;
  REQUIRE_GDR_FOR_IB_MODE(IbMode::Host);
  testBandwidth(PingPongTestParams{
      .useIPC = false, .useIB = true, .useEthernet = false, .waitWithPoll = false, .ibMode = IbMode::Host});
}

PERF_TEST(PortChannelOneToOneTest, BandwidthIbHostNoAtomicMode) {
  REQUIRE_IBVERBS;
  REQUIRE_GDR_FOR_IB_MODE(IbMode::HostNoAtomic);
  testBandwidth(PingPongTestParams{
      .useIPC = false, .useIB = true, .useEthernet = false, .waitWithPoll = false, .ibMode = IbMode::HostNoAtomic});
}

// The high-word coefficients do not cancel: correct net = 2*2^32+4 per block. Reconstructing
// either or both operands from only their low 32 bits produces a different net.
static constexpr int64_t kAccumulatePositive = 3 * (int64_t{1} << 32) + 7;
static constexpr int64_t kAccumulateNegative = -((int64_t{1} << 32) + 3);
static constexpr int64_t kAccumulateNet = kAccumulatePositive + kAccumulateNegative;
static_assert(kAccumulateNet > 0 && kAccumulateNet <= std::numeric_limits<int64_t>::max() / (32 * 20));

__global__ void kernelPortChannelAccumulate(int64_t* localBuff, int nTries, mscclpp::DeviceSyncer* syncer, int* ret) {
  auto& portChan = gChannelOneToOneTestConstPortChans;
  const int numBlocks = gridDim.x;

  for (int iter = 0; iter < nTries; ++iter) {
    portChan.accumulate(0, 0);
    portChan.accumulate(0, kAccumulatePositive);
    portChan.accumulate(0, kAccumulateNegative);
    syncer->sync(numBlocks);

    if (blockIdx.x == 0) {
      // The signal/flush after every block's additions validates proxy ordering before the peer wait.
      portChan.signal();
      portChan.flush();
      portChan.wait();

      const int64_t perIter = static_cast<int64_t>(numBlocks) * kAccumulateNet;
      const int64_t expected = static_cast<int64_t>(iter + 1) * perIter;
      const int64_t observed = *(volatile int64_t*)localBuff;
      const bool finalIter = iter + 1 == nTries;
      if (observed < expected || (finalIter && observed != expected)) {
        printf("iter %d (final %d): buff = %lld, expected %lld\n", iter, (int)finalIter, (long long)observed,
               (long long)expected);
        *ret = 1;
      }
    }
    syncer->sync(numBlocks);
  }
}

__global__ void kernelPortChannelAccumulateWrap(uint64_t* localBuff, int* ret) {
  auto& portChan = gChannelOneToOneTestConstPortChans;
  portChan.accumulate(0, 1);
  portChan.signal();
  portChan.flush();
  portChan.wait();
  if (*(volatile uint64_t*)localBuff != 0) *ret = 1;
}

void PortChannelOneToOneTest::testAccumulate(bool useIPC, bool useIb, bool useEthernet, IbMode ibMode) {
  if (gEnv->rank >= numRanksToUse) return;

  const int nElem = 1;
  std::vector<mscclpp::PortChannel> portChannels;
  auto buff = mscclpp::GpuBuffer<int64_t>(nElem);
  MSCCLPP_CUDATHROW(cudaMemset(buff.memory().get(), 0, nElem * sizeof(int64_t)));

  setupMeshConnections(portChannels, useIPC, useIb, useEthernet, buff.memory().get(), nElem * sizeof(int64_t), nullptr,
                       0, ibMode);
  ASSERT_EQ(portChannels.size(), 1);

  auto handle = portChannels[0].deviceHandle();
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestConstPortChans, &handle, sizeof(handle)));
  auto syncer = mscclpp::detail::gpuCallocShared<mscclpp::DeviceSyncer>();
  auto ret = mscclpp::detail::gpuCallocHostShared<int>();
  *ret = 0;

  proxyService->startProxy();
  kernelPortChannelAccumulate<<<ibMode == IbMode::GpuNetIo ? 1 : 32, 1>>>(buff.memory().get(), 20, syncer.get(),
                                                                          ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  EXPECT_EQ(*ret, 0);

  // Every transport defines accumulation modulo 2^64.
  MSCCLPP_CUDATHROW(cudaMemset(buff.memory().get(), 0xff, sizeof(uint64_t)));
  // cudaMemset may still be pending; complete it before the peer's accumulate can arrive.
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  *ret = 0;
  communicator->bootstrap()->barrier();
  kernelPortChannelAccumulateWrap<<<1, 1>>>(reinterpret_cast<uint64_t*>(buff.memory().get()), ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  proxyService->stopProxy();
  EXPECT_EQ(*ret, 0);
}

void PortChannelOneToOneTest::testAccumulateRejected(mscclpp::Transport transport, IbMode ibMode, int tag,
                                                     const char* backendMessage, bool checkHugeOffset) {
  const bool participates = gEnv->rank < numRanksToUse;
  bool rejectsBackend = false;
  bool rejectsHuge = !checkHugeOffset;
  if (participates) {
    const int peer = 1 - gEnv->rank;
    auto buff = mscclpp::GpuBuffer<int64_t>(1).memory();
    mscclpp::EndpointConfig cfg;
    cfg.transport = transport;
    if (transport != mscclpp::Transport::CudaIpc) {
      cfg.ib.gidIndex = std::stoi(gEnv->args["ib_gid_index"]);
      cfg.ib.mode = ibMode;
    }
    auto connFuture = communicator->connect(cfg, peer);
    auto localMem = communicator->registerMemory(buff.get(), sizeof(int64_t), transport);
    communicator->sendMemory(localMem, peer, tag);
    auto remoteFuture = communicator->recvMemory(peer, tag);
    auto conn = connFuture.get();
    auto remoteMem = remoteFuture.get();

    rejectsBackend = rejectsInvalidUsage([&] { conn.accumulate(remoteMem, 0, 1); }, backendMessage);
    if (checkHugeOffset) {
      rejectsHuge = rejectsInvalidUsage([&] { conn.accumulate(remoteMem, std::numeric_limits<uint64_t>::max(), 1); },
                                        "out of bounds");
    }
  }

  // Only ranks 0 and 1 own the fixture bootstrap; synchronize the full test world after teardown.
  MPI_Barrier(MPI_COMM_WORLD);
  if (participates) {
    EXPECT_TRUE(rejectsBackend);
    EXPECT_TRUE(rejectsHuge);
  }
}

#if defined(__HIP_PLATFORM_AMD__)
TEST(PortChannelOneToOneTest, AccumulateCudaIpc) {
  REQUIRE_CUDA_IPC_AVAILABLE;
  testAccumulate(true, false, false);
}
#else
TEST(PortChannelOneToOneTest, AccumulateCudaIpcRejected) {
  REQUIRE_CUDA_IPC_AVAILABLE;
  testAccumulateRejected(mscclpp::Transport::CudaIpc, IbMode::Default, /*tag=*/78, "not supported over CudaIpc on CUDA",
                         /*checkHugeOffset=*/false);
}
#endif

TEST(PortChannelOneToOneTest, AccumulateIb) {
  REQUIRE_IBVERBS;
  requireIbRdmaAtomics(ibTransport);
  REQUIRE_GDR_FOR_IB_MODE(IbMode::Host);
  testAccumulate(false, true, false, IbMode::Host);
}

TEST(PortChannelOneToOneTest, AccumulateEthernet) { testAccumulate(false, false, true); }

TEST(PortChannelOneToOneTest, AccumulateEthernetRejectsInvalidTargets) {
  const bool participates = gEnv->rank < numRanksToUse;
  bool rejectsUndersized = false;
  bool rejectsExactEnd = false;
  bool rejectsStraddling = false;
  bool rejectsMisalignment = false;

  if (participates) {
    const int peer = 1 - gEnv->rank;
    // The allocations are deliberately larger than their registrations, so a missing bounds check
    // remains within an allocation while the test reports the failure.
    auto backing = mscclpp::GpuBuffer<uint8_t>(64).memory();
    auto smallBacking = mscclpp::GpuBuffer<uint8_t>(64).memory();

    mscclpp::EndpointConfig cfg;
    cfg.transport = mscclpp::Transport::Ethernet;
    auto connFuture = communicator->connect(cfg, peer);
    auto localMem = communicator->registerMemory(backing.get(), 16, mscclpp::Transport::Ethernet);
    auto smallLocalMem = communicator->registerMemory(smallBacking.get(), 7, mscclpp::Transport::Ethernet);
    communicator->sendMemory(localMem, peer, /*tag=*/79);
    communicator->sendMemory(smallLocalMem, peer, /*tag=*/80);
    auto remoteFuture = communicator->recvMemory(peer, /*tag=*/79);
    auto smallRemoteFuture = communicator->recvMemory(peer, /*tag=*/80);

    auto conn = connFuture.get();
    auto remoteMem = remoteFuture.get();
    auto smallRemoteMem = smallRemoteFuture.get();
    rejectsUndersized = rejectsInvalidUsage([&] { conn.accumulate(smallRemoteMem, 0, 1); }, "out of bounds");
    rejectsExactEnd = rejectsInvalidUsage([&] { conn.accumulate(remoteMem, 16, 1); }, "out of bounds");
    rejectsStraddling = rejectsInvalidUsage([&] { conn.accumulate(remoteMem, 12, 1); }, "out of bounds");
    rejectsMisalignment = rejectsInvalidUsage([&] { conn.accumulate(remoteMem, 1, 1); }, "aligned");
  }

  // Only ranks 0 and 1 own the fixture bootstrap; synchronize the full test world after teardown.
  MPI_Barrier(MPI_COMM_WORLD);
  if (participates) {
    EXPECT_TRUE(rejectsUndersized);
    EXPECT_TRUE(rejectsExactEnd);
    EXPECT_TRUE(rejectsStraddling);
    EXPECT_TRUE(rejectsMisalignment);
  }
}

TEST(PortChannelOneToOneTest, AccumulateIbHostNoAtomicRejected) {
  REQUIRE_IBVERBS;
  requireGdrForHostNoAtomicCollective(/*firstRank=*/0, /*secondRank=*/1);
  testAccumulateRejected(ibTransport, IbMode::HostNoAtomic, /*tag=*/77, "not supported in IB no-atomic mode",
                         /*checkHugeOffset=*/true);
}

static constexpr int kMaxQps = 4;
__constant__ DeviceHandle<mscclpp::PortChannel> gMultiQpPortChans[kMaxQps];

// Multi-QP bandwidth kernel: one thread per QP, putWithSignal per QP, parallel waits.
__global__ void kernelMultiQpBandwidth(int nElemPerChan, int nIters, int numQps) {
  int q = threadIdx.x;
  if (q >= numQps) return;
  for (int i = 0; i < nIters; i++) {
    if (q == 0) {
      gMultiQpPortChans[0].signal();
      gMultiQpPortChans[0].wait();
    }
    __syncthreads();
    gMultiQpPortChans[q].putWithSignal(0, nElemPerChan * sizeof(int));
    gMultiQpPortChans[q].wait();
    __syncthreads();
  }
}

// Multi-QP setup helper: bootstrap N parallel IB connections + port channels in two
// futures-based phases (issue all async ops before resolving any, to avoid deadlock).
// tagBase: distinct base used by each caller so concurrent tests don't clash on tags.
void PortChannelOneToOneTest::setupMultiQpChannels(int numQps, size_t elemsPerChan, IbMode ibMode, int tagBase,
                                                   std::vector<std::shared_ptr<int>>& sendBuffs,
                                                   std::vector<mscclpp::RegisteredMemory>& localMems,
                                                   std::vector<mscclpp::RegisteredMemory>& remoteMems,
                                                   std::vector<mscclpp::PortChannel>& portChannels) {
  const int peer = 1 - communicator->bootstrap()->getRank();
  sendBuffs.assign(numQps, nullptr);
  localMems.assign(numQps, mscclpp::RegisteredMemory{});
  remoteMems.assign(numQps, mscclpp::RegisteredMemory{});
  portChannels.clear();

  std::vector<std::shared_future<mscclpp::Connection>> connFutures(numQps);
  std::vector<std::shared_future<mscclpp::RegisteredMemory>> remoteMemFutures(numQps);

  for (int q = 0; q < numQps; q++) {
    sendBuffs[q] = mscclpp::GpuBuffer<int>(elemsPerChan).memory();
    localMems[q] = communicator->registerMemory(sendBuffs[q].get(), elemsPerChan * sizeof(int), ibTransport);

    mscclpp::EndpointConfig cfg;
    cfg.transport = ibTransport;
    cfg.ib.gidIndex = std::stoi(gEnv->args["ib_gid_index"]);
    cfg.ib.mode = ibMode;

    connFutures[q] = communicator->connect(cfg, peer, tagBase + q);
    communicator->sendMemory(localMems[q], peer, tagBase + numQps + q);
    remoteMemFutures[q] = communicator->recvMemory(peer, tagBase + numQps + q);
  }

  for (int q = 0; q < numQps; q++) {
    auto conn = connFutures[q].get();
    remoteMems[q] = remoteMemFutures[q].get();
    auto sema = communicator->buildSemaphore(conn, peer, tagBase + 2 * numQps + q).get();
    mscclpp::SemaphoreId cid = proxyService->addSemaphore(sema);
    portChannels.emplace_back(
        proxyService->portChannel(cid, proxyService->addMemory(remoteMems[q]), proxyService->addMemory(localMems[q])));
  }
}

void PortChannelOneToOneTest::testMultiQpBandwidth(IbMode ibMode, int numQps) {
  if (gEnv->rank >= numRanksToUse) return;

  const int rank = communicator->bootstrap()->getRank();
  const int maxElemPerChan = 32 * 1024 * 1024;  // 128 MB per channel

  std::vector<std::shared_ptr<int>> sendBuffs;
  std::vector<mscclpp::RegisteredMemory> localMems;
  std::vector<mscclpp::RegisteredMemory> remoteMems;
  std::vector<mscclpp::PortChannel> portChannels;
  setupMultiQpChannels(numQps, maxElemPerChan, ibMode, /*tagBase=*/100, sendBuffs, localMems, remoteMems, portChannels);

  std::vector<DeviceHandle<mscclpp::PortChannel>> handles;
  for (auto& ch : portChannels) handles.push_back(ch.deviceHandle());
  ASSERT_EQ(handles.size(), static_cast<size_t>(numQps));
  ASSERT_LE(numQps, kMaxQps);  // numQps must not exceed __constant__ array size (kMaxQps)
  MSCCLPP_CUDATHROW(
      cudaMemcpyToSymbol(gMultiQpPortChans, handles.data(), numQps * sizeof(DeviceHandle<mscclpp::PortChannel>)));

  proxyService->startProxy();

  const std::string testName = ::mscclpp::test::currentTestName();
  const std::string qpLabel = std::to_string(numQps) + " QP" + (numQps > 1 ? "s" : "");

  for (int nElemPerChan :
       {256, 16 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024, 32 * 1024 * 1024}) {
    int nIters = 200;
    // Warm-up
    kernelMultiQpBandwidth<<<1, numQps>>>(nElemPerChan, 10, numQps);
    MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
    communicator->bootstrap()->barrier();

    // Measure
    mscclpp::Timer timer;
    kernelMultiQpBandwidth<<<1, numQps>>>(nElemPerChan, nIters, numQps);
    MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
    double elapsedUs = timer.elapsed();
    communicator->bootstrap()->barrier();

    if (rank == 0) {
      double totalBytes = (double)nElemPerChan * sizeof(int) * numQps;
      double elapsedMsPerIter = elapsedUs / 1e3 / nIters;
      double gbps = totalBytes / elapsedMsPerIter * 1e-6;
      double totalSizeKB = totalBytes / 1024.0;
      std::string label;
      if (totalSizeKB >= 1024.0)
        label = std::to_string((int)(totalSizeKB / 1024.0)) + " MB";
      else
        label = std::to_string((int)totalSizeKB) + " KB";
      ::mscclpp::test::reportPerfResult(label + " (" + qpLabel + ")", gbps, "GB/s");
    }
  }

  proxyService->stopProxy();

  for (auto& m : localMems) registeredMemories.push_back(m);
  for (auto& m : remoteMems) registeredMemories.push_back(m);
}

PERF_TEST(PortChannelOneToOneTest, SingleQpBandwidthIbHostMode) {
  REQUIRE_IBVERBS;
  REQUIRE_GDR_FOR_IB_MODE(IbMode::Host);
  testMultiQpBandwidth(IbMode::Host, /*numQps=*/1);
}

PERF_TEST(PortChannelOneToOneTest, TwoQpBandwidthIbHostMode) {
  REQUIRE_IBVERBS;
  REQUIRE_GDR_FOR_IB_MODE(IbMode::Host);
  testMultiQpBandwidth(IbMode::Host, /*numQps=*/2);
}

PERF_TEST(PortChannelOneToOneTest, MultiQpBandwidthIbHostMode) {
  REQUIRE_IBVERBS;
  REQUIRE_GDR_FOR_IB_MODE(IbMode::Host);
  testMultiQpBandwidth(IbMode::Host, /*numQps=*/4);
}

PERF_TEST(PortChannelOneToOneTest, SingleQpBandwidthIbHostNoAtomicMode) {
  REQUIRE_IBVERBS;
  REQUIRE_GDR_FOR_IB_MODE(IbMode::HostNoAtomic);
  testMultiQpBandwidth(IbMode::HostNoAtomic, /*numQps=*/1);
}

PERF_TEST(PortChannelOneToOneTest, TwoQpBandwidthIbHostNoAtomicMode) {
  REQUIRE_IBVERBS;
  REQUIRE_GDR_FOR_IB_MODE(IbMode::HostNoAtomic);
  testMultiQpBandwidth(IbMode::HostNoAtomic, /*numQps=*/2);
}

PERF_TEST(PortChannelOneToOneTest, ThreeQpBandwidthIbHostNoAtomicMode) {
  REQUIRE_IBVERBS;
  REQUIRE_GDR_FOR_IB_MODE(IbMode::HostNoAtomic);
  testMultiQpBandwidth(IbMode::HostNoAtomic, /*numQps=*/3);
}

PERF_TEST(PortChannelOneToOneTest, MultiQpBandwidthIbHostNoAtomicMode) {
  REQUIRE_IBVERBS;
  REQUIRE_GDR_FOR_IB_MODE(IbMode::HostNoAtomic);
  testMultiQpBandwidth(IbMode::HostNoAtomic, /*numQps=*/4);
}

// Multi-QP flush-stress kernel: one thread per QP, all calling putWithSignalAndFlush
// concurrently so all N CQ drains are in flight on the proxy thread at once.
// This is the concurrent-flush worst case the async-progress design protects against.
__global__ void kernelMultiQpFlushStress(int nElemPerChan, int nIters, int numQps) {
  int q = threadIdx.x;
  if (q >= numQps) return;
  for (int i = 0; i < nIters; i++) {
    if (q == 0) {
      gMultiQpPortChans[0].signal();
      gMultiQpPortChans[0].wait();
    }
    __syncthreads();
    gMultiQpPortChans[q].putWithSignalAndFlush(0, nElemPerChan * sizeof(int));
    gMultiQpPortChans[q].wait();
    __syncthreads();
  }
}

void PortChannelOneToOneTest::testMultiQpFlushStress(IbMode ibMode, int numQps) {
  if (gEnv->rank >= numRanksToUse) return;

  const int rank = communicator->bootstrap()->getRank();
  const int maxElemPerChan = 8 * 1024 * 1024;

  std::vector<std::shared_ptr<int>> sendBuffs;
  std::vector<mscclpp::RegisteredMemory> localMems;
  std::vector<mscclpp::RegisteredMemory> remoteMems;
  std::vector<mscclpp::PortChannel> portChannels;
  setupMultiQpChannels(numQps, maxElemPerChan, ibMode, /*tagBase=*/400, sendBuffs, localMems, remoteMems, portChannels);

  std::vector<DeviceHandle<mscclpp::PortChannel>> handles;
  for (auto& ch : portChannels) handles.push_back(ch.deviceHandle());
  ASSERT_EQ(handles.size(), static_cast<size_t>(numQps));
  ASSERT_LE(numQps, kMaxQps);
  MSCCLPP_CUDATHROW(
      cudaMemcpyToSymbol(gMultiQpPortChans, handles.data(), numQps * sizeof(DeviceHandle<mscclpp::PortChannel>)));

  proxyService->startProxy();

  const std::string qpLabel = std::to_string(numQps) + " QP" + (numQps > 1 ? "s" : "");

  for (int nElemPerChan : {256, 4 * 1024, 64 * 1024, 256 * 1024, 1024 * 1024, 4 * 1024 * 1024, 8 * 1024 * 1024}) {
    int nIters = (nElemPerChan >= 256 * 1024) ? 200 : 2000;
    kernelMultiQpFlushStress<<<1, numQps>>>(nElemPerChan, 10, numQps);
    MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
    communicator->bootstrap()->barrier();

    mscclpp::Timer timer;
    kernelMultiQpFlushStress<<<1, numQps>>>(nElemPerChan, nIters, numQps);
    MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
    double elapsedUs = timer.elapsed();
    communicator->bootstrap()->barrier();

    if (rank == 0) {
      double usPerIter = elapsedUs / nIters;
      double usPerIterPerQp = usPerIter / numQps;
      int bytesPerChan = nElemPerChan * (int)sizeof(int);
      std::string sizeLabel = (bytesPerChan >= 1024) ? (std::to_string(bytesPerChan / 1024) + " KB")
                                                     : (std::to_string(bytesPerChan) + " B");
      double aggGbps = ((double)bytesPerChan * numQps) / usPerIter * 1e-3;  // bytes/us = MB/s × 1e-3 = GB/s
      ::mscclpp::test::reportPerfResult(sizeLabel + " (" + qpLabel + ") per-iter", usPerIter, "us");
      ::mscclpp::test::reportPerfResult(sizeLabel + " (" + qpLabel + ") per-iter/QP", usPerIterPerQp, "us");
      ::mscclpp::test::reportPerfResult(sizeLabel + " (" + qpLabel + ") aggregate", aggGbps, "GB/s");
    }
  }

  proxyService->stopProxy();

  for (auto& m : localMems) registeredMemories.push_back(m);
  for (auto& m : remoteMems) registeredMemories.push_back(m);
}

PERF_TEST(PortChannelOneToOneTest, SingleQpFlushStressIbHostMode) {
  REQUIRE_IBVERBS;
  REQUIRE_GDR_FOR_IB_MODE(IbMode::Host);
  testMultiQpFlushStress(IbMode::Host, /*numQps=*/1);
}

PERF_TEST(PortChannelOneToOneTest, TwoQpFlushStressIbHostMode) {
  REQUIRE_IBVERBS;
  REQUIRE_GDR_FOR_IB_MODE(IbMode::Host);
  testMultiQpFlushStress(IbMode::Host, /*numQps=*/2);
}

PERF_TEST(PortChannelOneToOneTest, MultiQpFlushStressIbHostMode) {
  REQUIRE_IBVERBS;
  REQUIRE_GDR_FOR_IB_MODE(IbMode::Host);
  testMultiQpFlushStress(IbMode::Host, /*numQps=*/4);
}

PERF_TEST(PortChannelOneToOneTest, SingleQpFlushStressIbHostNoAtomicMode) {
  REQUIRE_IBVERBS;
  REQUIRE_GDR_FOR_IB_MODE(IbMode::HostNoAtomic);
  testMultiQpFlushStress(IbMode::HostNoAtomic, /*numQps=*/1);
}

PERF_TEST(PortChannelOneToOneTest, TwoQpFlushStressIbHostNoAtomicMode) {
  REQUIRE_IBVERBS;
  REQUIRE_GDR_FOR_IB_MODE(IbMode::HostNoAtomic);
  testMultiQpFlushStress(IbMode::HostNoAtomic, /*numQps=*/2);
}

PERF_TEST(PortChannelOneToOneTest, MultiQpFlushStressIbHostNoAtomicMode) {
  REQUIRE_IBVERBS;
  REQUIRE_GDR_FOR_IB_MODE(IbMode::HostNoAtomic);
  testMultiQpFlushStress(IbMode::HostNoAtomic, /*numQps=*/4);
}

// Same-channel concurrent-flush kernel: N GPU threads on the same PortChannel each call
// putWithSignalAndFlush in lockstep. Stresses the FIFO-position-based wait target so that
// each caller waits on its own TriggerFlush rather than on a globally-incrementing counter
// that could be assigned out-of-order relative to the FIFO push order.
__constant__ DeviceHandle<mscclpp::PortChannel> gSingleChanForConcurrentFlush;

template <bool Serialize>
__global__ void kernelSameChanConcurrentFlush(int nIters) {
  auto& chan = gSingleChanForConcurrentFlush;
  int tid = threadIdx.x;
  for (int i = 0; i < nIters; i++) {
    for (int issuer = 0; issuer < (Serialize ? blockDim.x : 1); issuer++) {
      if (!Serialize || tid == issuer) {
        uint64_t offset = tid * sizeof(int);
        chan.put(offset, offset, sizeof(int));
        chan.putWithSignalAndFlush(offset, offset, sizeof(int));
        chan.wait();
      }
      if constexpr (Serialize) __syncthreads();
    }
  }
}

void PortChannelOneToOneTest::testSameChanConcurrentFlush(IbMode ibMode) {
  if (gEnv->rank >= numRanksToUse) return;

  constexpr int nThreads = 4;
  std::vector<std::shared_ptr<int>> sendBuffs;
  std::vector<mscclpp::RegisteredMemory> localMems;
  std::vector<mscclpp::RegisteredMemory> remoteMems;
  std::vector<mscclpp::PortChannel> portChannels;
  setupMultiQpChannels(/*numQps=*/1, /*elemsPerChan=*/nThreads, ibMode, /*tagBase=*/700, sendBuffs, localMems,
                       remoteMems, portChannels);

  DeviceHandle<mscclpp::PortChannel> handle = portChannels[0].deviceHandle();
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gSingleChanForConcurrentFlush, &handle, sizeof(handle)));

  proxyService->startProxy();

  // Warm-up
  if (ibMode == IbMode::GpuNetIo) {
    kernelSameChanConcurrentFlush<true><<<1, nThreads>>>(10);
  } else {
    kernelSameChanConcurrentFlush<false><<<1, nThreads>>>(10);
  }
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  communicator->bootstrap()->barrier();

  // Measure: a successful completion (no deadlock, no CQ error) validates that each
  // concurrent-flush caller waited on its own TriggerFlush (not someone else's earlier one).
  const int nIters = 500;
  mscclpp::Timer timer;
  if (ibMode == IbMode::GpuNetIo) {
    kernelSameChanConcurrentFlush<true><<<1, nThreads>>>(nIters);
  } else {
    kernelSameChanConcurrentFlush<false><<<1, nThreads>>>(nIters);
  }
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  double elapsedUs = timer.elapsed();
  communicator->bootstrap()->barrier();

  if (communicator->bootstrap()->getRank() == 0) {
    double usPerIter = elapsedUs / nIters;
    ::mscclpp::test::reportPerfResult(std::to_string(nThreads) + " threads same-chan per-iter", usPerIter, "us");
  }

  proxyService->stopProxy();
  for (auto& m : localMems) registeredMemories.push_back(m);
  for (auto& m : remoteMems) registeredMemories.push_back(m);
}

TEST(PortChannelOneToOneTest, SameChanConcurrentFlushIbHostMode) {
  REQUIRE_IBVERBS;
  REQUIRE_GDR_FOR_IB_MODE(IbMode::Host);
  testSameChanConcurrentFlush(IbMode::Host);
}

void PortChannelFanInTest::SetUp() {
  CommunicatorTestBase::SetUp();
  proxyService = std::make_shared<mscclpp::ProxyService>();
}

void PortChannelFanInTest::TearDown() { CommunicatorTestBase::TearDown(); }

// Each rank other than 0 pushes nTries accumulates at rank 0's single counter.
__global__ void kernelFanInAccumulate(int nTries) {
  DeviceHandle<mscclpp::PortChannel>& portChan = gChannelOneToOneTestConstPortChans;
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  for (int i = 0; i < nTries; i++) {
    portChan.accumulate(0, kAccumulatePositive);
  }
  portChan.flush();
}

void PortChannelFanInTest::testFanIn(bool useIPC, bool useIb, bool useEthernet, IbMode ibMode) {
  const int worldSize = communicator->bootstrap()->getNranks();
  const int rank = communicator->bootstrap()->getRank();
  if (worldSize < 3) {
    SKIP_TEST() << "Fan-in test needs at least 3 ranks to have more than one writer.";
    return;
  }
  const int nTries = 200;

  auto buff = mscclpp::GpuBuffer<int64_t>(1);
  MSCCLPP_CUDATHROW(cudaMemset(buff.memory().get(), 0, sizeof(int64_t)));

  // Rank 0 is the target; every other rank connects to it.
  mscclpp::TransportFlags transport;
  if (useIPC) transport |= mscclpp::Transport::CudaIpc;
  if (useIb) transport |= ibTransport;
  if (useEthernet) transport |= mscclpp::Transport::Ethernet;

  mscclpp::EndpointConfig cfg;
  if (useIPC) {
    cfg.transport = mscclpp::Transport::CudaIpc;
  } else if (useIb) {
    cfg.transport = ibTransport;
    cfg.ib.gidIndex = std::stoi(gEnv->args["ib_gid_index"]);
    cfg.ib.mode = ibMode;
  } else {
    cfg.transport = mscclpp::Transport::Ethernet;
  }

  mscclpp::RegisteredMemory localMem = communicator->registerMemory(buff.memory().get(), sizeof(int64_t), transport);
  registeredMemories.push_back(localMem);

  std::vector<std::shared_future<mscclpp::Connection>> connFutures(worldSize);
  std::vector<std::shared_future<mscclpp::RegisteredMemory>> remoteMemFutures(worldSize);
  if (rank == 0) {
    for (int r = 1; r < worldSize; r++) {
      connFutures[r] = communicator->connect(cfg, r);
      communicator->sendMemory(localMem, r);
      remoteMemFutures[r] = communicator->recvMemory(r);
    }
  } else {
    connFutures[0] = communicator->connect(cfg, 0);
    communicator->sendMemory(localMem, 0);
    remoteMemFutures[0] = communicator->recvMemory(0);
  }

  std::vector<mscclpp::PortChannel> portChannels;
  if (rank == 0) {
    for (int r = 1; r < worldSize; r++) {
      auto sema = communicator->buildSemaphore(connFutures[r].get(), r).get();
      mscclpp::SemaphoreId cid = proxyService->addSemaphore(sema);
      portChannels.emplace_back(proxyService->portChannel(cid, proxyService->addMemory(remoteMemFutures[r].get()),
                                                          proxyService->addMemory(localMem)));
      registeredMemories.push_back(remoteMemFutures[r].get());
    }
  } else {
    auto sema = communicator->buildSemaphore(connFutures[0].get(), 0).get();
    mscclpp::SemaphoreId cid = proxyService->addSemaphore(sema);
    portChannels.emplace_back(proxyService->portChannel(cid, proxyService->addMemory(remoteMemFutures[0].get()),
                                                        proxyService->addMemory(localMem)));
    registeredMemories.push_back(remoteMemFutures[0].get());
  }

  proxyService->startProxy();

  if (rank != 0) {
    std::vector<DeviceHandle<mscclpp::PortChannel>> handles;
    handles.push_back(portChannels[0].deviceHandle());
    MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestConstPortChans, handles.data(),
                                         sizeof(DeviceHandle<mscclpp::PortChannel>)));
    kernelFanInAccumulate<<<1, 1>>>(nTries);
    MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  }

  communicator->bootstrap()->barrier();

  if (rank == 0) {
    // EthernetConnection::flush() is a no-op, so wait up to one overall deadline for the receiver
    // to apply all updates. Temporary inactivity is not treated as completion.
    const int64_t expected = (int64_t)(worldSize - 1) * nTries * kAccumulatePositive;
    int64_t observed = 0;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    for (;;) {
      mscclpp::gpuMemcpy(reinterpret_cast<char*>(&observed), reinterpret_cast<char*>(buff.memory().get()),
                         sizeof(int64_t), cudaMemcpyDeviceToHost);
      if (observed == expected || std::chrono::steady_clock::now() >= deadline) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (observed != expected) {
      std::cout << "fan-in lost " << (expected - observed) / kAccumulatePositive << " of "
                << (int64_t)(worldSize - 1) * nTries << " accumulates" << std::endl;
    }
    EXPECT_EQ(observed, expected);
  }

  communicator->bootstrap()->barrier();
  proxyService->stopProxy();
  communicator->bootstrap()->barrier();
}

#if defined(__HIP_PLATFORM_AMD__)
// CudaIpc supports many writers on ROCm: the kernel is a real read-modify-write, so writers in
// separate processes do not lose updates.
TEST(PortChannelFanInTest, AccumulateCudaIpc) {
  requireCudaIpcRankZeroPeers(gEnv->worldSize - 1);
  testFanIn(true, false, false);
}
#endif  // defined(__HIP_PLATFORM_AMD__)

TEST(PortChannelFanInTest, AccumulateIb) {
  REQUIRE_IBVERBS;
  requireIbRdmaAtomics(ibTransport);
  REQUIRE_GDR_FOR_IB_MODE(IbMode::Host);
  testFanIn(false, true, false, IbMode::Host);
}

TEST(PortChannelFanInTest, AccumulateEthernet) { testFanIn(false, false, true); }

// GPUNetIO (IbMode::GpuNetIo): the same PortChannel workflows with GPU threads posting RDMA work requests directly.
#if defined(MSCCLPP_USE_GPUNETIO)
#define REQUIRE_GPUNETIO \
  REQUIRE_IBVERBS;       \
  requireIbRdmaAtomics(ibTransport)
#else  // !defined(MSCCLPP_USE_GPUNETIO)
#define REQUIRE_GPUNETIO SKIP_TEST() << "MSCCL++ was built with MSCCLPP_USE_GPUNETIO=OFF"
#endif  // !defined(MSCCLPP_USE_GPUNETIO)

TEST(PortChannelOneToOneTest, PingPongIbGpuNetIoMode) {
  REQUIRE_GPUNETIO;
  testPingPong(PingPongTestParams{
      .useIPC = false, .useIB = true, .useEthernet = false, .waitWithPoll = false, .ibMode = IbMode::GpuNetIo});
}

TEST(PortChannelOneToOneTest, PingPongIbGpuNetIoModeWithPoll) {
  REQUIRE_GPUNETIO;
  testPingPong(PingPongTestParams{
      .useIPC = false, .useIB = true, .useEthernet = false, .waitWithPoll = true, .ibMode = IbMode::GpuNetIo});
}

TEST(PortChannelOneToOneTest, PacketPingPongIbGpuNetIoMode) {
  REQUIRE_GPUNETIO;
  testPacketPingPong(true, IbMode::GpuNetIo);
}

TEST(PortChannelOneToOneTest, AccumulateIbGpuNetIoMode) {
  REQUIRE_GPUNETIO;
  testAccumulate(false, true, false, IbMode::GpuNetIo);
}

TEST(PortChannelOneToOneTest, SameChanSerializedFlushIbGpuNetIoMode) {
  REQUIRE_GPUNETIO;
  testSameChanConcurrentFlush(IbMode::GpuNetIo);
}

TEST(PortChannelOneToOneTest, HostAccumulateIbGpuNetIoModeRejected) {
  REQUIRE_GPUNETIO;
  testAccumulateRejected(ibTransport, IbMode::GpuNetIo, /*tag=*/79, "GPU-initiated", /*checkHugeOffset=*/false);
}

TEST(PortChannelOneToOneTest, DirectConstructionIbGpuNetIoModeRejected) {
  REQUIRE_GPUNETIO;
  const bool participates = gEnv->rank < numRanksToUse;
  bool rejectsDirect = false;
  bool factoryOk = false;
  if (participates) {
    const int peer = 1 - gEnv->rank;
    auto buff = mscclpp::GpuBuffer<int64_t>(1).memory();
    mscclpp::EndpointConfig cfg;
    cfg.transport = ibTransport;
    cfg.ib.gidIndex = std::stoi(gEnv->args["ib_gid_index"]);
    cfg.ib.mode = IbMode::GpuNetIo;
    auto conn = communicator->connect(cfg, peer).get();
    auto localMem = communicator->registerMemory(buff.get(), sizeof(int64_t), ibTransport);
    mscclpp::ProxyService service;
    auto semaphoreId = service.buildAndAddSemaphore(*communicator, conn);
    auto memoryId = service.addMemory(localMem);

    mscclpp::PortChannel direct(semaphoreId, service.semaphore(semaphoreId), nullptr, memoryId, memoryId);
    rejectsDirect = rejectsInvalidUsage([&] { direct.deviceHandle(); }, "ProxyService::portChannel()");
    try {
      service.portChannel(semaphoreId, memoryId, memoryId).deviceHandle();
      factoryOk = true;
    } catch (const std::exception& e) {
      std::cerr << "ProxyService::portChannel() failed: " << e.what() << std::endl;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (participates) {
    EXPECT_TRUE(rejectsDirect);
    EXPECT_TRUE(factoryOk);
  }
}

PERF_TEST(PortChannelOneToOneTest, PingPongPerfIbGpuNetIoMode) {
  REQUIRE_GPUNETIO;
  testPingPongPerf(PingPongTestParams{
      .useIPC = false, .useIB = true, .useEthernet = false, .waitWithPoll = false, .ibMode = IbMode::GpuNetIo});
}

PERF_TEST(PortChannelOneToOneTest, PacketPingPongPerfIbGpuNetIoMode) {
  REQUIRE_GPUNETIO;
  testPacketPingPongPerf(true, IbMode::GpuNetIo);
}

PERF_TEST(PortChannelOneToOneTest, BandwidthIbGpuNetIoMode) {
  REQUIRE_GPUNETIO;
  testBandwidth(PingPongTestParams{
      .useIPC = false, .useIB = true, .useEthernet = false, .waitWithPoll = false, .ibMode = IbMode::GpuNetIo});
}

PERF_TEST(PortChannelOneToOneTest, SingleQpBandwidthIbGpuNetIoMode) {
  REQUIRE_GPUNETIO;
  testMultiQpBandwidth(IbMode::GpuNetIo, /*numQps=*/1);
}

PERF_TEST(PortChannelOneToOneTest, MultiQpBandwidthIbGpuNetIoMode) {
  REQUIRE_GPUNETIO;
  testMultiQpBandwidth(IbMode::GpuNetIo, /*numQps=*/4);
}

#if defined(MSCCLPP_USE_GPUNETIO)
namespace {
__global__ void kernelEpPairedStreams(int chunkBytes, int chunks, int qps, int halfBytes, int iterations,
                                      uint64_t* durations) {
  const int q = threadIdx.x / 32;
  const bool owner = threadIdx.x % 32 == 0 && q < qps;
  __shared__ uint64_t start;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    __syncthreads();
    if (owner) {
      auto channel = gMultiQpPortChans[q];
      // Each QP has one owner warp and a distinct channelId/WQE scratch slot.
      // All chunks are posted before any completion wait; sources are immutable.
      for (int chunk = q; chunk < chunks; chunk += qps) {
        const size_t offset = static_cast<size_t>(chunk / qps) * chunkBytes;
        channel.putWithSignal(halfBytes + offset, offset, chunkBytes);
      }
      for (int chunk = q; chunk < chunks; chunk += qps) channel.wait(-1);
      channel.flush(-1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      uint64_t end;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(end));
      durations[iteration] = end - start;
    }
    __syncthreads();
  }
}

template <bool FinalSignalOnly, bool RecordStages>
__global__ void kernelEpSignalAblation(int chunkBytes, int chunks, int qps, int halfBytes, int iterations,
                                       uint64_t* durations, uint64_t* stages) {
  const int q = threadIdx.x / 32;
  const bool owner = threadIdx.x % 32 == 0 && q < qps;
  __shared__ uint64_t start;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    __syncthreads();
    if (owner) {
      auto channel = gMultiQpPortChans[q];
      for (int chunk = q; chunk < chunks; chunk += qps) {
        const size_t offset = static_cast<size_t>(chunk / qps) * chunkBytes;
        if constexpr (FinalSignalOnly) {
          // Keep the last signal strongly ordered after every payload on this QP.
          // Only the readiness granularity changes; no payload or CQ drain is omitted.
          if (chunk + qps < chunks)
            channel.put(halfBytes + offset, offset, chunkBytes);
          else
            channel.putWithSignal(halfBytes + offset, offset, chunkBytes);
        } else {
          channel.putWithSignal(halfBytes + offset, offset, chunkBytes);
        }
      }
      const size_t record = (static_cast<size_t>(iteration) * qps + q) * 3;
      if constexpr (RecordStages) {
        uint64_t now;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
        stages[record] = now - start;
      }
      if constexpr (FinalSignalOnly) {
        if (q < chunks) channel.wait(-1);
      } else {
        for (int chunk = q; chunk < chunks; chunk += qps) channel.wait(-1);
      }
      if constexpr (RecordStages) {
        uint64_t now;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
        stages[record + 1] = now - start;
      }
      channel.flush(-1);
      if constexpr (RecordStages) {
        uint64_t now;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
        stages[record + 2] = now - start;
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      uint64_t end;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(end));
      durations[iteration] = end - start;
    }
    __syncthreads();
  }
}
}  // namespace

PERF_TEST(PortChannelOneToOneTest, EpPairedStreamsIbGpuNetIoMode) {
  REQUIRE_GPUNETIO;
  if (gEnv->worldSize != 2) {
    SKIP_TEST() << "Requires exactly two ranks; launch separate pairs for concurrent NIC load.";
  }
  if (gEnv->rank >= numRanksToUse) return;
  constexpr int HalfBytes = 3 * 229376;
  constexpr int HalfWords = HalfBytes / sizeof(int);
  constexpr int Iterations = 1000;
  const bool ablation = gEnv->args.count("ep_signal_ablation") && gEnv->args["ep_signal_ablation"] == "1";
  const bool recordStages = ablation && gEnv->args.count("ep_signal_stages") && gEnv->args["ep_signal_stages"] == "1";
  const int rank = communicator->bootstrap()->getRank();
  std::vector<std::shared_ptr<int>> buffers;
  std::vector<mscclpp::RegisteredMemory> local, remote;
  std::vector<mscclpp::PortChannel> channels;
  setupMultiQpChannels(4, 2 * HalfWords, IbMode::GpuNetIo, 3000, buffers, local, remote, channels);
  std::vector<mscclpp::PortChannelDeviceHandle> handles;
  int valid = 1, allValid = 0;
  for (int q = 0; q < 4; ++q) {
    handles.push_back(channels[q].deviceHandle());
    for (int previous = 0; previous < q; ++previous)
      valid &= handles[q].gpuNetIo_.channelId != handles[previous].gpuNetIo_.channelId &&
               handles[q].gpuNetIo_.qp != handles[previous].gpuNetIo_.qp;
    std::vector<uint32_t> source(HalfWords);
    for (int i = 0; i < HalfWords; ++i) source[i] = static_cast<uint32_t>(rank + 1) * 0x9e3779b9u ^ (q * HalfWords + i);
    MSCCLPP_CUDATHROW(cudaMemcpy(buffers[q].get(), source.data(), HalfBytes, cudaMemcpyHostToDevice));
  }
  MPI_Allreduce(&valid, &allValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  ASSERT_EQ(allValid, 1);
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gMultiQpPortChans, handles.data(), handles.size() * sizeof(handles[0])));
  auto deviceDurations = mscclpp::GpuBuffer<uint64_t>(Iterations).memory();
  auto deviceStages =
      recordStages ? mscclpp::GpuBuffer<uint64_t>(Iterations * 4 * 3).memory() : std::shared_ptr<uint64_t>{};
  std::vector<uint64_t> durations(Iterations), allDurations(rank == 0 ? Iterations * 2 : 0);
  std::vector<uint64_t> stages(recordStages ? Iterations * 4 * 3 : 0);
  std::vector<uint32_t> received(HalfWords);
  std::ofstream samples, phaseSamples;
  if (gEnv->args.count("perf_output")) {
    samples.open(gEnv->args["perf_output"] + ".rank" + std::to_string(rank) + ".csv");
    samples << (ablation ? "mode,stages,chunk_bytes,chunks,qps,repetition,round,total_ns\n"
                         : "chunk_bytes,chunks,qps,repetition,round,total_ns\n");
    if (recordStages) {
      phaseSamples.open(gEnv->args["perf_output"] + ".phases.rank" + std::to_string(rank) + ".csv");
      phaseSamples << "mode,chunk_bytes,chunks,qps,repetition,round,qp,posted_ns,markers_ns,cq_ns\n";
    }
  }
  cudaEvent_t begin, end;
  MSCCLPP_CUDATHROW(cudaEventCreate(&begin));
  MSCCLPP_CUDATHROW(cudaEventCreate(&end));
  struct Shape {
    int bytes, chunks, qps;
  };
  int phase = 0;
  for (const Shape shape : {Shape{14336, 1, 1},
                            {115712, 1, 1},
                            {229376, 1, 1},
                            {115712, 8, 4},
                            {115712, 12, 4},
                            {229376, 8, 4},
                            {229376, 12, 4}}) {
    if (ablation && (shape.bytes != 115712 || shape.qps != 4)) continue;
    for (int mode = 0; mode < (ablation ? 2 : 1); ++mode) {
      const bool finalSignalOnly = mode == 1;
      auto launch = [&](int iterations) {
        if (!ablation) {
          kernelEpPairedStreams<<<1, 128>>>(shape.bytes, shape.chunks, shape.qps, HalfBytes, iterations,
                                            deviceDurations.get());
        } else if (recordStages && finalSignalOnly) {
          kernelEpSignalAblation<true, true><<<1, 128>>>(shape.bytes, shape.chunks, shape.qps, HalfBytes, iterations,
                                                         deviceDurations.get(), deviceStages.get());
        } else if (recordStages) {
          kernelEpSignalAblation<false, true><<<1, 128>>>(shape.bytes, shape.chunks, shape.qps, HalfBytes, iterations,
                                                          deviceDurations.get(), deviceStages.get());
        } else if (finalSignalOnly) {
          kernelEpSignalAblation<true, false>
              <<<1, 128>>>(shape.bytes, shape.chunks, shape.qps, HalfBytes, iterations, deviceDurations.get(), nullptr);
        } else {
          kernelEpSignalAblation<false, false>
              <<<1, 128>>>(shape.bytes, shape.chunks, shape.qps, HalfBytes, iterations, deviceDurations.get(), nullptr);
        }
      };
      for (int repetition = 0; repetition < 3; ++repetition) {
        for (auto& buffer : buffers) MSCCLPP_CUDATHROW(cudaMemset(buffer.get() + HalfWords, 0, HalfBytes));
        MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
        communicator->bootstrap()->barrier();
        launch(32);
        MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
        communicator->bootstrap()->barrier();
        // Independent two-rank jobs can align their load windows without adding MPI traffic to timing.
        if (gEnv->args.count("roofline_start_ms")) {
          const auto startMs = std::stoll(gEnv->args["roofline_start_ms"]) + phase * 1000;
          std::this_thread::sleep_until(std::chrono::system_clock::time_point(std::chrono::milliseconds(startMs)));
        }
        ++phase;
        const auto hostBegin = std::chrono::system_clock::now();
        MSCCLPP_CUDATHROW(cudaEventRecord(begin));
        launch(Iterations);
        MSCCLPP_CUDATHROW(cudaEventRecord(end));
        MSCCLPP_CUDATHROW(cudaEventSynchronize(end));
        const auto hostEnd = std::chrono::system_clock::now();
        float milliseconds = 0;
        MSCCLPP_CUDATHROW(cudaEventElapsedTime(&milliseconds, begin, end));
        communicator->bootstrap()->barrier();
        uint64_t errors = 0, allErrors = 0;
        for (int q = 0; q < 4; ++q) {
          MSCCLPP_CUDATHROW(
              cudaMemcpy(received.data(), buffers[q].get() + HalfWords, HalfBytes, cudaMemcpyDeviceToHost));
          const int writtenWords =
              q < shape.qps ? (shape.chunks + shape.qps - 1 - q) / shape.qps * shape.bytes / sizeof(int) : 0;
          for (int i = 0; i < HalfWords; ++i) {
            const uint32_t expected =
                i < writtenWords ? static_cast<uint32_t>(2 - rank) * 0x9e3779b9u ^ (q * HalfWords + i) : 0;
            errors += received[i] != expected;
          }
        }
        MPI_Allreduce(&errors, &allErrors, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        EXPECT_EQ(allErrors, 0u);
        MSCCLPP_CUDATHROW(
            cudaMemcpy(durations.data(), deviceDurations.get(), Iterations * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        double phaseSum[3] = {}, allPhaseSum[3] = {};
        if (recordStages) {
          MSCCLPP_CUDATHROW(
              cudaMemcpy(stages.data(), deviceStages.get(), stages.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost));
          for (int i = 0; i < Iterations; ++i) {
            uint64_t last[3] = {};
            for (int q = 0; q < shape.qps; ++q) {
              const auto* record = stages.data() + (i * shape.qps + q) * 3;
              for (int field = 0; field < 3; ++field) last[field] = std::max(last[field], record[field]);
              if (phaseSamples)
                phaseSamples << mode << ',' << shape.bytes << ',' << shape.chunks << ',' << shape.qps << ','
                             << repetition << ',' << i << ',' << q << ',' << record[0] << ',' << record[1] << ','
                             << record[2] << '\n';
            }
            for (int field = 0; field < 3; ++field) phaseSum[field] += last[field] / (1000.0 * Iterations);
          }
        }
        MPI_Reduce(phaseSum, allPhaseSum, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        for (int i = 0; i < Iterations; ++i) {
          if (samples) {
            if (ablation) samples << mode << ',' << recordStages << ',';
            samples << shape.bytes << ',' << shape.chunks << ',' << shape.qps << ',' << repetition << ',' << i << ','
                    << durations[i] << '\n';
          }
        }
        MPI_Gather(durations.data(), Iterations, MPI_UINT64_T, allDurations.data(), Iterations, MPI_UINT64_T, 0,
                   MPI_COMM_WORLD);
        double eventUs = milliseconds * 1000.0 / Iterations, sumEventUs = 0;
        MPI_Reduce(&eventUs, &sumEventUs, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
          std::sort(allDurations.begin(), allDurations.end());
          const double mean =
              std::accumulate(allDurations.begin(), allDurations.end(), 0.0) / allDurations.size() / 1000;
          const size_t bytes = static_cast<size_t>(shape.bytes) * shape.chunks;
          std::cout << std::setprecision(9) << "[IB_PAIRED_STREAMS] repetition=" << repetition
                    << " chunk_bytes=" << shape.bytes << " chunks=" << shape.chunks << " qps=" << shape.qps
                    << " bytes=" << bytes << " event_mean_us=" << sumEventUs / 2 << " round_mean_us=" << mean
                    << " p50_us=" << allDurations[allDurations.size() / 2] / 1000.0
                    << " p95_us=" << allDurations[allDurations.size() * 95 / 100] / 1000.0
                    << " GBps=" << bytes / (sumEventUs / 2) / 1000 << " errors=" << allErrors << " host_begin_ns="
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(hostBegin.time_since_epoch()).count()
                    << " host_end_ns="
                    << std::chrono::duration_cast<std::chrono::nanoseconds>(hostEnd.time_since_epoch()).count()
                    << (ablation ? " ablation=1" : " ablation=0") << " final_signal_only=" << finalSignalOnly
                    << " stages=" << recordStages << " post_mean_us=" << allPhaseSum[0] / 2
                    << " markers_mean_us=" << allPhaseSum[1] / 2 << " cq_mean_us=" << allPhaseSum[2] / 2 << std::endl;
          ::mscclpp::test::reportPerfResult(std::to_string(shape.bytes) + "B x" + std::to_string(shape.chunks) +
                                                " rep" + std::to_string(repetition) +
                                                (ablation ? " finalSignalOnly=" + std::to_string(mode) : ""),
                                            bytes / (sumEventUs / 2) / 1000, "GB/s");
        }
      }
    }
  }
  MSCCLPP_CUDATHROW(cudaEventDestroy(begin));
  MSCCLPP_CUDATHROW(cudaEventDestroy(end));
  for (auto& memory : local) registeredMemories.push_back(memory);
  for (auto& memory : remote) registeredMemories.push_back(memory);
}

class PortChannelBipartiteTest : public CommunicatorTestBase {};

namespace {
constexpr int BipartiteChannels = 64;
constexpr uint64_t BipartitePayloadBytes = 8 * 7168 * sizeof(uint16_t);
constexpr uint64_t BipartiteMetadataBytes = 8 * 8 * sizeof(uint32_t);
constexpr uint64_t BipartiteSlotBytes = BipartitePayloadBytes + 2 * BipartiteMetadataBytes;

template <bool Distributed>
__device__ void bipartiteSync() {
  if constexpr (Distributed) {
    cooperative_groups::this_grid().sync();
  } else {
    __syncthreads();
  }
}

template <bool Distributed>
__global__ void kernelBipartiteTransport(mscclpp::PortChannelDeviceHandle* channels, int rank, int iterations,
                                         int remotePeers, int qps, bool metadata, uint64_t* durations,
                                         uint64_t* tickets) {
  const int channelIndex = Distributed ? static_cast<int>(blockIdx.x) : static_cast<int>(threadIdx.x);
  const bool active = !Distributed || threadIdx.x == 0;
  const bool leader = channelIndex == 0 && active;
  auto channel = channels[channelIndex];
  auto handle = channel.gpuNetIo_;
  const uint64_t sourceBase = static_cast<uint64_t>(channelIndex) * BipartiteSlotBytes;
  const uint64_t destinationBase =
      static_cast<uint64_t>((rank % remotePeers) * qps + channelIndex % qps) * BipartiteSlotBytes;
  uint64_t sourceOffsets[3] = {sourceBase, sourceBase + BipartitePayloadBytes,
                               sourceBase + BipartitePayloadBytes + BipartiteMetadataBytes};
  uint64_t destinationOffsets[3] = {destinationBase, destinationBase + BipartitePayloadBytes,
                                    destinationBase + BipartitePayloadBytes + BipartiteMetadataBytes};
  uint64_t sizes[3] = {BipartitePayloadBytes, BipartiteMetadataBytes, BipartiteMetadataBytes};
  mscclpp::GpuNetIoMemoryDeviceHandle sources[3], destinations[3];
  for (int i = 0; i < 3; ++i) {
    sources[i] = handle.memory(channel.src_, true, sourceOffsets[i], sizes[i]);
    destinations[i] = handle.memory(channel.dst_, false, destinationOffsets[i], sizes[i]);
  }
  __shared__ uint64_t start;
  if (leader) tickets[0] = handle.reservedTicket();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    if (leader) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    bipartiteSync<Distributed>();
    // Sources are immutable and their host-to-device copy completed before this kernel.
    if (active) {
      if (metadata) {
        handle.putBatched3(destinations, destinationOffsets, sources, sourceOffsets, sizes);
        handle.signal();
      } else {
        handle.putWithSignal(destinations[0], destinationOffsets[0], sources[0], sourceOffsets[0], sizes[0]);
      }
    }
    bipartiteSync<Distributed>();
    if (leader) {
      uint64_t now;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
      durations[2 * iteration + 1] = now - start;
    }
    if (active) {
      channel.wait(-1);
      handle.flush(-1);
    }
    bipartiteSync<Distributed>();
    if (leader) {
      uint64_t now;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
      durations[2 * iteration] = now - start;
    }
    bipartiteSync<Distributed>();
  }
  if (leader) tickets[1] = handle.reservedTicket();
}
}  // namespace

// Opposite rank halves exchange 112 KiB on each of 64 QPs/GPU: 1x64, 4x16, or 16x4 peers/QPs.
// Distributed submission is the default; --bipartite_compare_packed also measures one 64-thread CTA.
PERF_TEST(PortChannelBipartiteTest, EpGranularityIbGpuNetIoMode) {
  REQUIRE_GPUNETIO;
  if (gEnv->worldSize != 2 && gEnv->worldSize != 8 && gEnv->worldSize != 32) {
    SKIP_TEST() << "Requires 2, 8, or 32 ranks; opposite rank halves communicate only over IB";
  }
  const int remotePeers = gEnv->worldSize / 2;
  const int qps = BipartiteChannels / remotePeers;
  const bool comparePacked = gEnv->args.count("bipartite_compare_packed") != 0;
  int cooperative = 0, allCooperative = 0;
  MSCCLPP_CUDATHROW(cudaDeviceGetAttribute(&cooperative, cudaDevAttrCooperativeLaunch, rankToLocalRank(gEnv->rank)));
  MPI_Allreduce(&cooperative, &allCooperative, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (!allCooperative) {
    SKIP_TEST() << "Distributed-QP diagnostic requires cooperative launches on every rank";
  }
  const int rank = gEnv->rank;
  constexpr size_t PoolBytes = BipartiteChannels * BipartiteSlotBytes;
  constexpr size_t PoolWords = PoolBytes / sizeof(uint32_t);
  constexpr size_t SlotWords = BipartiteSlotBytes / sizeof(uint32_t);
  constexpr size_t PayloadWords = BipartitePayloadBytes / sizeof(uint32_t);
  auto sendBuffer = mscclpp::GpuBuffer<uint32_t>(PoolWords).memory();
  auto receiveBuffer = mscclpp::GpuBuffer<uint32_t>(PoolWords).memory();
  std::vector<uint32_t> hostSource(PoolWords);
  for (size_t i = 0; i < PoolWords; ++i) hostSource[i] = static_cast<uint32_t>(rank + 1) * 0x9e3779b9u ^ i;
  MSCCLPP_CUDATHROW(cudaMemcpy(sendBuffer.get(), hostSource.data(), PoolBytes, cudaMemcpyHostToDevice));
  auto localSource = communicator->registerMemory(sendBuffer.get(), PoolBytes, ibTransport);
  auto localDestination = communicator->registerMemory(receiveBuffer.get(), PoolBytes, ibTransport);
  registeredMemories.push_back(localSource);
  registeredMemories.push_back(localDestination);
  auto proxy = std::make_shared<mscclpp::ProxyService>();
  std::vector<std::shared_future<mscclpp::Connection>> connectionFutures(BipartiteChannels);
  std::vector<std::shared_future<mscclpp::RegisteredMemory>> memoryFutures(remotePeers);
  std::vector<std::shared_future<mscclpp::Semaphore>> semaphoreFutures(BipartiteChannels);
  const int remoteBase = rank < remotePeers ? remotePeers : 0;
  mscclpp::EndpointConfig config;
  config.transport = ibTransport;
  config.ib.gidIndex = std::stoi(gEnv->args["ib_gid_index"]);
  config.ib.mode = IbMode::GpuNetIo;
  for (int peerIndex = 0; peerIndex < remotePeers; ++peerIndex) {
    const int peer = remoteBase + peerIndex;
    communicator->sendMemory(localDestination, peer, 900);
    memoryFutures[peerIndex] = communicator->recvMemory(peer, 900);
    for (int q = 0; q < qps; ++q)
      connectionFutures[peerIndex * qps + q] = communicator->connect(config, peer, 1000 + q);
  }
  for (int channel = 0; channel < BipartiteChannels; ++channel) {
    semaphoreFutures[channel] = communicator->buildSemaphore(connectionFutures[channel].get(),
                                                             remoteBase + channel / qps, 2000 + channel % qps);
  }
  std::vector<mscclpp::PortChannel> portChannels;
  std::vector<mscclpp::PortChannelDeviceHandle> handles;
  std::vector<bool> usedSlots(BipartiteChannels, false);
  int slotsValid = 1, allSlotsValid = 0;
  const auto sourceId = proxy->addMemory(localSource);
  for (int peer = 0; peer < remotePeers; ++peer) {
    auto remoteMemory = memoryFutures[peer].get();
    registeredMemories.push_back(remoteMemory);
    const auto destinationId = proxy->addMemory(remoteMemory);
    for (int q = 0; q < qps; ++q) {
      const auto semaphoreId = proxy->addSemaphore(semaphoreFutures[peer * qps + q].get());
      portChannels.push_back(proxy->portChannel(semaphoreId, destinationId, sourceId));
      handles.push_back(portChannels.back().deviceHandle());
      const auto slot = handles.back().gpuNetIo_.channelId;
      if (slot >= BipartiteChannels || usedSlots[slot]) {
        slotsValid = 0;
      } else {
        usedSlots[slot] = true;
      }
    }
  }
  MPI_Allreduce(&slotsValid, &allSlotsValid, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  ASSERT_EQ(allSlotsValid, 1);
  auto deviceChannels = mscclpp::GpuBuffer<mscclpp::PortChannelDeviceHandle>(BipartiteChannels).memory();
  MSCCLPP_CUDATHROW(
      cudaMemcpy(deviceChannels.get(), handles.data(), handles.size() * sizeof(handles[0]), cudaMemcpyHostToDevice));
  constexpr int Iterations = 4096;
  auto deviceDurations = mscclpp::GpuBuffer<uint64_t>(2 * Iterations).memory();
  auto deviceTickets = mscclpp::GpuBuffer<uint64_t>(2).memory();
  std::vector<uint64_t> durations(2 * Iterations);
  std::vector<uint32_t> received(PoolWords);
  std::ofstream samples;
  if (gEnv->args.count("perf_output")) {
    samples.open(gEnv->args["perf_output"] + ".rank" + std::to_string(rank) + ".csv");
    EXPECT_TRUE(samples.good());
    samples << "distributed,repetition,metadata,round,total_ns,post_ns,qp0_end_ticket\n";
  }
  cudaEvent_t begin, end;
  MSCCLPP_CUDATHROW(cudaEventCreate(&begin));
  MSCCLPP_CUDATHROW(cudaEventCreate(&end));
  auto launch = [&](int iterations, bool metadata, bool distributed) {
    if (distributed) {
      auto* channelData = deviceChannels.get();
      auto* durationData = deviceDurations.get();
      auto* ticketData = deviceTickets.get();
      int rankArgument = rank;
      int remotePeersArgument = remotePeers, qpsArgument = qps;
      void* arguments[] = {&channelData, &rankArgument, &iterations,   &remotePeersArgument,
                           &qpsArgument, &metadata,     &durationData, &ticketData};
      MSCCLPP_CUDATHROW(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(kernelBipartiteTransport<true>),
                                                    dim3(BipartiteChannels), dim3(32), arguments));
    } else {
      kernelBipartiteTransport<false><<<1, BipartiteChannels>>>(deviceChannels.get(), rank, iterations, remotePeers,
                                                                qps, metadata, deviceDurations.get(),
                                                                deviceTickets.get());
      MSCCLPP_CUDATHROW(cudaGetLastError());
    }
  };
  for (bool distributed : {false, true}) {
    if (!distributed && !comparePacked) continue;
    for (int repetition = 0; repetition < 3; ++repetition) {
      for (bool metadata : {false, true}) {
        MSCCLPP_CUDATHROW(cudaMemset(receiveBuffer.get(), 0, PoolBytes));
        MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
        communicator->bootstrap()->barrier();
        launch(16, metadata, distributed);
        MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
        communicator->bootstrap()->barrier();
        MSCCLPP_CUDATHROW(cudaMemset(receiveBuffer.get(), 0, PoolBytes));
        MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
        communicator->bootstrap()->barrier();
        MSCCLPP_CUDATHROW(cudaEventRecord(begin));
        launch(Iterations, metadata, distributed);
        MSCCLPP_CUDATHROW(cudaEventRecord(end));
        MSCCLPP_CUDATHROW(cudaEventSynchronize(end));
        float elapsedMs;
        MSCCLPP_CUDATHROW(cudaEventElapsedTime(&elapsedMs, begin, end));
        MSCCLPP_CUDATHROW(cudaMemcpy(durations.data(), deviceDurations.get(), durations.size() * sizeof(uint64_t),
                                     cudaMemcpyDeviceToHost));
        uint64_t tickets[2];
        MSCCLPP_CUDATHROW(cudaMemcpy(tickets, deviceTickets.get(), sizeof(tickets), cudaMemcpyDeviceToHost));
        communicator->bootstrap()->barrier();
        MSCCLPP_CUDATHROW(cudaMemcpy(received.data(), receiveBuffer.get(), PoolBytes, cudaMemcpyDeviceToHost));
        uint64_t errors = 0;
        for (size_t i = 0; i < PoolWords; ++i) {
          const int sourceRank = remoteBase + i / SlotWords / qps;
          const size_t sourceSlot = (rank % remotePeers) * qps + i / SlotWords % qps;
          const size_t word = i % SlotWords;
          const uint32_t expected = metadata || word < PayloadWords
                                        ? static_cast<uint32_t>(sourceRank + 1) * 0x9e3779b9u ^
                                              static_cast<uint32_t>(sourceSlot * SlotWords + word)
                                        : 0;
          errors += received[i] != expected;
        }
        uint64_t allErrors = 0;
        MPI_Allreduce(&errors, &allErrors, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        EXPECT_EQ(allErrors, 0u);
        EXPECT_EQ(tickets[1] - tickets[0], static_cast<uint64_t>(Iterations) * (metadata ? 4 : 2));
        std::vector<double> localDurations(Iterations);
        double postUs = 0;
        for (int i = 0; i < Iterations; ++i) {
          localDurations[i] = static_cast<double>(durations[2 * i]) / 1000;
          postUs += static_cast<double>(durations[2 * i + 1]) / 1000 / Iterations;
          if (samples)
            samples << distributed << ',' << repetition << ',' << metadata << ',' << i << ',' << durations[2 * i] << ','
                    << durations[2 * i + 1] << ',' << tickets[0] + static_cast<uint64_t>(i + 1) * (metadata ? 4 : 2)
                    << '\n';
        }
        std::vector<double> allDurations(rank == 0 ? Iterations * gEnv->worldSize : 0);
        MPI_Gather(localDurations.data(), Iterations, MPI_DOUBLE, allDurations.data(), Iterations, MPI_DOUBLE, 0,
                   MPI_COMM_WORLD);
        double eventUs = static_cast<double>(elapsedMs) * 1000 / Iterations;
        double meanEventUs = 0, meanPostUs = 0;
        MPI_Reduce(&eventUs, &meanEventUs, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&postUs, &meanPostUs, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
          std::sort(allDurations.begin(), allDurations.end());
          const double mean = std::accumulate(allDurations.begin(), allDurations.end(), 0.0) / allDurations.size();
          std::cout << std::setprecision(9) << "[IB_BIPARTITE] distributed=" << distributed
                    << " repetition=" << repetition << " metadata=" << metadata << " ranks=" << gEnv->worldSize
                    << " remote_peers=" << remotePeers << " qps_per_peer=" << qps
                    << " payload_bytes=" << BipartiteChannels * BipartitePayloadBytes
                    << " event_mean_us=" << meanEventUs / gEnv->worldSize << " round_mean_us=" << mean
                    << " p50_us=" << allDurations[allDurations.size() / 2]
                    << " p95_us=" << allDurations[allDurations.size() * 95 / 100]
                    << " p99_us=" << allDurations[allDurations.size() * 99 / 100] << " max_us=" << allDurations.back()
                    << " post_mean_us=" << meanPostUs / gEnv->worldSize
                    << " payload_GBps=" << BipartiteChannels * BipartitePayloadBytes / mean / 1000
                    << " errors=" << allErrors << std::endl;
        }
      }
    }
  }
  MSCCLPP_CUDATHROW(cudaEventDestroy(begin));
  MSCCLPP_CUDATHROW(cudaEventDestroy(end));
}
#endif

PERF_TEST(PortChannelOneToOneTest, MultiQpFlushStressIbGpuNetIoMode) {
  REQUIRE_GPUNETIO;
  testMultiQpFlushStress(IbMode::GpuNetIo, /*numQps=*/4);
}

TEST(PortChannelFanInTest, AccumulateIbGpuNetIoMode) {
  REQUIRE_GPUNETIO;
  testFanIn(false, true, false, IbMode::GpuNetIo);
}
