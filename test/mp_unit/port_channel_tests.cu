// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cstdint>
#include <mscclpp/concurrency_device.hpp>

#include "gdr.hpp"
#include "mp_unit_tests.hpp"
#include "utils_internal.hpp"

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
#else
#define REQUIRE_GDR_FOR_IB_MODE(mode)  // No extra requirements on non-CUDA platforms.
#endif

// Skip an IPC-only PortChannel test (useIPC=true, useIB=false, useEthernet=false) when CudaIpc
// cannot connect this rank pair. CudaIpc works intra-node always, and cross-node only on MNNVL
// systems (GB200 NVL72 + IMEX). The combined check is "at least 2 ranks per node" OR "fabric
// (MNNVL) handles are usable on this system".
#define REQUIRE_CUDA_IPC_AVAILABLE                                           \
  do {                                                                       \
    if (gEnv->nRanksPerNode < 2 && !mscclpp::isFabricMemHandleAvailable()) { \
      SKIP_TEST() << "CudaIpc requires intra-node ranks (nRanksPerNode>=2) or MNNVL fabric handles, \
both unavailable here.";                                                     \
    }                                                                        \
  } while (0)

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

template <bool CheckCorrectness>
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
      if (threadId == 0) {
        // Send data from the local putPacketBuffer to the remote getPacketBuffer
        portChan.put(0, nPkt * sizeof(mscclpp::LLPacket));
      }
      flusher++;
      if (flusher == 64) {
        if (threadId == 0) portChan.flush();
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
    kernelProxyLLPingPong<true>
        <<<1, 1024>>>(buff.get(), putPacketBuffer.get(), getPacketBuffer.get(), gEnv->rank, nElem, nTries, ret.get());
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

// Concurrent atomicAdd test kernel.
// Each rank launches numBlocks thread blocks. Every block atomicAdds +1 to the remote buffer.
// Block 0 polls the local buffer (written by the remote) until it reaches the expected value,
// then releases all blocks for the next iteration. This creates a ping-pong pattern where
// both ranks simultaneously send numBlocks atomic adds per iteration.
__global__ void kernelPortChannelAtomicAddConcurrent(int64_t* localBuff, int nTries, mscclpp::DeviceSyncer* syncer,
                                                     int* ret) {
  DeviceHandle<mscclpp::PortChannel>& portChan = gChannelOneToOneTestConstPortChans;
  const int numBlocks = gridDim.x;

  for (int iter = 0; iter < nTries; iter++) {
    // Step 1: Every block atomicAdds +1 to the remote buffer via port channel.
    portChan.atomicAdd(0, (int64_t)1);

    // Step 2: Grid barrier — all blocks must have pushed their atomicAdd.
    syncer->sync(numBlocks);

    // Step 3: Block 0 signals remote that all adds are done, flushes, then waits for remote.
    if (blockIdx.x == 0) {
      portChan.signal();
      portChan.flush();
      portChan.wait();
    }

    // Step 4: Grid barrier — ensure signal/wait complete before next iteration.
    syncer->sync(numBlocks);
  }

  // Verify final value: each of nTries iterations adds numBlocks from the remote.
  if (blockIdx.x == 0) {
    int64_t expected = (int64_t)nTries * numBlocks;
    if (*localBuff != expected) {
      printf("buff = %lld, expected = %lld\n", (long long)*localBuff, (long long)expected);
      *ret = 1;
    }
  }
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

void PortChannelOneToOneTest::testAtomicAdd(bool useIPC, bool useIb, bool useEthernet, IbMode ibMode) {
  if (gEnv->rank >= numRanksToUse) return;

  const int nElem = 1;
  const int numBlocks = 64;
  const int nTries = 50;

  std::vector<mscclpp::PortChannel> portChannels;
  auto buff = mscclpp::GpuBuffer<int64_t>(nElem);
  MSCCLPP_CUDATHROW(cudaMemset(buff.memory().get(), 0, nElem * sizeof(int64_t)));

  setupMeshConnections(portChannels, useIPC, useIb, useEthernet, buff.memory().get(), nElem * sizeof(int64_t), nullptr,
                       0, ibMode);

  ASSERT_EQ(portChannels.size(), 1);

  std::vector<DeviceHandle<mscclpp::PortChannel>> portChannelHandles;
  for (auto& ch : portChannels) portChannelHandles.push_back(ch.deviceHandle());

  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestConstPortChans, portChannelHandles.data(),
                                       sizeof(DeviceHandle<mscclpp::PortChannel>)));

  // Allocate DeviceSyncer for grid barrier (device memory, zero-initialized).
  auto syncer = mscclpp::detail::gpuCallocShared<mscclpp::DeviceSyncer>();

  proxyService->startProxy();

  auto ret = mscclpp::detail::gpuCallocHostShared<int>();
  *ret = 0;

  // Use a dedicated stream + cudaStreamSynchronize instead of cudaDeviceSynchronize
  // to avoid deadlocking the proxy's atomicAdd kernel (which runs on a separate stream).
  cudaStream_t testStream;
  MSCCLPP_CUDATHROW(cudaStreamCreateWithFlags(&testStream, cudaStreamNonBlocking));
  kernelPortChannelAtomicAddConcurrent<<<numBlocks, 1, 0, testStream>>>(buff.memory().get(), nTries, syncer.get(),
                                                                        ret.get());
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(testStream));
  MSCCLPP_CUDATHROW(cudaStreamDestroy(testStream));

  EXPECT_EQ(*ret, 0);

  proxyService->stopProxy();
}

TEST(PortChannelOneToOneTest, AtomicAdd) { testAtomicAdd(true, false, false); }

TEST(PortChannelOneToOneTest, AtomicAddIb) {
  REQUIRE_IBVERBS;
  testAtomicAdd(false, true, false, IbMode::Host);
}

TEST(PortChannelOneToOneTest, AtomicAddEthernet) { testAtomicAdd(false, false, true); }

TEST(PortChannelOneToOneTest, AtomicAddIbHostNoAtomicRejected) {
  REQUIRE_IBVERBS;
  REQUIRE_GDR_FOR_IB_MODE(IbMode::HostNoAtomic);
  if (gEnv->rank >= numRanksToUse) return;

  const int peer = 1 - gEnv->rank;
  auto buff = mscclpp::GpuBuffer<int64_t>(1).memory();
  mscclpp::RegisteredMemory localMem;
  mscclpp::RegisteredMemory remoteMem;

  mscclpp::EndpointConfig cfg;
  cfg.transport = ibTransport;
  cfg.ib.gidIndex = std::stoi(gEnv->args["ib_gid_index"]);
  cfg.ib.mode = IbMode::HostNoAtomic;

  auto connFuture = communicator->connect(cfg, peer);
  localMem = communicator->registerMemory(buff.get(), sizeof(int64_t), ibTransport);
  communicator->sendMemory(localMem, peer, /*tag=*/77);
  auto remoteFuture = communicator->recvMemory(peer, /*tag=*/77);

  auto conn = connFuture.get();
  remoteMem = remoteFuture.get();
  registeredMemories.push_back(localMem);

  try {
    conn.atomicAdd(remoteMem, 0, 1);
    FAIL() << "Expected atomicAdd in IB HostNoAtomic mode to throw InvalidUsage";
  } catch (const mscclpp::Error& e) {
    EXPECT_TRUE(e.getErrorCode() == mscclpp::ErrorCode::InvalidUsage);
  }

  communicator->bootstrap()->barrier();
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
// each caller waits on its own TriggerSync rather than on a globally-incrementing counter
// that could be assigned out-of-order relative to the FIFO push order.
__constant__ DeviceHandle<mscclpp::PortChannel> gSingleChanForConcurrentFlush;

__global__ void kernelSameChanConcurrentFlush(int nIters) {
  auto& chan = gSingleChanForConcurrentFlush;
  int tid = threadIdx.x;
  for (int i = 0; i < nIters; i++) {
    // Each thread writes to a distinct slot (so puts don't overlap on remote side),
    // then concurrently flushes on the same channel.
    uint64_t offset = tid * sizeof(int);
    chan.putWithSignalAndFlush(offset, offset, sizeof(int));
    // Each thread waits for one signal from the remote rank's symmetric putWithSignalAndFlush.
    chan.wait();
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
  kernelSameChanConcurrentFlush<<<1, nThreads>>>(10);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  communicator->bootstrap()->barrier();

  // Measure: a successful completion (no deadlock, no CQ error) validates that each
  // concurrent-flush caller waited on its own TriggerSync (not someone else's earlier one).
  const int nIters = 500;
  mscclpp::Timer timer;
  kernelSameChanConcurrentFlush<<<1, nThreads>>>(nIters);
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
