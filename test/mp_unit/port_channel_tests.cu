// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cstdint>
#include <mscclpp/concurrency_device.hpp>

#include "mp_unit_tests.hpp"
#include "utils_internal.hpp"

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
    if ((rankToNode(r) == rankToNode(gEnv->rank)) && useIPC) {
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

  kernelProxyPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1, params.waitWithPoll, nTries, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);

  kernelProxyPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1024, params.waitWithPoll, nTries, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);

  kernelProxyPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1024 * 1024, params.waitWithPoll, nTries, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);

  kernelProxyPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 4 * 1024 * 1024, params.waitWithPoll, nTries, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);

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
    std::cout << testName << ": " << std::setprecision(4) << (float)timer.elapsed() / (float)nTries << " us/iter\n";
  }

  proxyService->stopProxy();
}

TEST_F(PortChannelOneToOneTest, PingPong) {
  testPingPong(PingPongTestParams{
      .useIPC = true, .useIB = false, .useEthernet = false, .waitWithPoll = false, .ibMode = IbMode::Default});
}

TEST_F(PortChannelOneToOneTest, PingPongIbHostMode) {
#if defined(USE_IBVERBS)
  testPingPong(PingPongTestParams{
      .useIPC = false, .useIB = true, .useEthernet = false, .waitWithPoll = false, .ibMode = IbMode::Host});
#else   // !defined(USE_IBVERBS)
  SKIP_TEST() << "This test requires IBVerbs that the current build does not support.";
#endif  // !defined(USE_IBVERBS)
}

TEST_F(PortChannelOneToOneTest, PingPongEthernet) {
  testPingPong(PingPongTestParams{
      .useIPC = false, .useIB = false, .useEthernet = true, .waitWithPoll = false, .ibMode = IbMode::Default});
}

TEST_F(PortChannelOneToOneTest, PingPongWithPoll) {
  testPingPong(PingPongTestParams{
      .useIPC = true, .useIB = false, .useEthernet = false, .waitWithPoll = true, .ibMode = IbMode::Default});
}

TEST_F(PortChannelOneToOneTest, PingPongIbHostModeWithPoll) {
#if defined(USE_IBVERBS)
  testPingPong(PingPongTestParams{
      .useIPC = false, .useIB = true, .useEthernet = false, .waitWithPoll = true, .ibMode = IbMode::Host});
#else   // !defined(USE_IBVERBS)
  SKIP_TEST() << "This test requires IBVerbs that the current build does not support.";
#endif  // !defined(USE_IBVERBS)
}

TEST_F(PortChannelOneToOneTest, PingPongPerf) {
  testPingPongPerf(PingPongTestParams{
      .useIPC = true, .useIB = false, .useEthernet = false, .waitWithPoll = false, .ibMode = IbMode::Default});
}

TEST_F(PortChannelOneToOneTest, PingPongPerfIbHostMode) {
#if defined(USE_IBVERBS)
  testPingPongPerf(PingPongTestParams{
      .useIPC = false, .useIB = true, .useEthernet = false, .waitWithPoll = false, .ibMode = IbMode::Host});
#else   // !defined(USE_IBVERBS)
  SKIP_TEST() << "This test requires IBVerbs that the current build does not support.";
#endif  // !defined(USE_IBVERBS)
}

TEST_F(PortChannelOneToOneTest, PingPongPerfIbHostNoAtomicMode) {
#if defined(USE_IBVERBS)
  testPingPongPerf(PingPongTestParams{
      .useIPC = false, .useIB = true, .useEthernet = false, .waitWithPoll = false, .ibMode = IbMode::HostNoAtomic});
#else   // !defined(USE_IBVERBS)
  SKIP_TEST() << "This test requires IBVerbs that the current build does not support.";
#endif  // !defined(USE_IBVERBS)
}

TEST_F(PortChannelOneToOneTest, PingPongPerfEthernet) {
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
  kernelProxyLLPingPong<true>
      <<<1, 1024>>>(buff.get(), putPacketBuffer.get(), getPacketBuffer.get(), gEnv->rank, 2, nTries, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelProxyLLPingPong<true>
      <<<1, 1024>>>(buff.get(), putPacketBuffer.get(), getPacketBuffer.get(), gEnv->rank, 1024, nTries, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelProxyLLPingPong<true><<<1, 1024>>>(buff.get(), putPacketBuffer.get(), getPacketBuffer.get(), gEnv->rank,
                                           1024 * 1024, nTries, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelProxyLLPingPong<true><<<1, 1024>>>(buff.get(), putPacketBuffer.get(), getPacketBuffer.get(), gEnv->rank,
                                           4 * 1024 * 1024, nTries, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);

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
    std::cout << testName << ": " << std::setprecision(4) << (float)timer.elapsed() / (float)nTries << " us/iter\n";
  }

  proxyService->stopProxy();
}

TEST_F(PortChannelOneToOneTest, PacketPingPong) { testPacketPingPong(false, IbMode::Default); }

TEST_F(PortChannelOneToOneTest, PacketPingPongIbHostMode) {
#if defined(USE_IBVERBS)
  testPacketPingPong(true, IbMode::Host);
#else   // !defined(USE_IBVERBS)
  SKIP_TEST() << "This test requires IBVerbs that the current build does not support.";
#endif  // !defined(USE_IBVERBS)
}

TEST_F(PortChannelOneToOneTest, PacketPingPongPerf) { testPacketPingPongPerf(false, IbMode::Default); }

TEST_F(PortChannelOneToOneTest, PacketPingPongPerfIbHostMode) {
#if defined(USE_IBVERBS)
  testPacketPingPongPerf(true, IbMode::Host);
#else   // !defined(USE_IBVERBS)
  SKIP_TEST() << "This test requires IBVerbs that the current build does not support.";
#endif  // !defined(USE_IBVERBS)
}

TEST_F(PortChannelOneToOneTest, PacketPingPongPerfIbHostNoAtomicMode) {
#if defined(USE_IBVERBS)
  testPacketPingPongPerf(true, IbMode::HostNoAtomic);
#else   // !defined(USE_IBVERBS)
  SKIP_TEST() << "This test requires IBVerbs that the current build does not support.";
#endif  // !defined(USE_IBVERBS)
}

TEST_F(PortChannelOneToOneTest, PingPongIbHostNoAtomicMode) {
#if defined(USE_IBVERBS)
  testPingPong(PingPongTestParams{
      .useIPC = false, .useIB = true, .useEthernet = false, .waitWithPoll = false, .ibMode = IbMode::HostNoAtomic});
#else   // !defined(USE_IBVERBS)
  SKIP_TEST() << "This test requires IBVerbs that the current build does not support.";
#endif  // !defined(USE_IBVERBS)
}

TEST_F(PortChannelOneToOneTest, PacketPingPongIbHostNoAtomicMode) {
#if defined(USE_IBVERBS)
  testPacketPingPong(true, IbMode::HostNoAtomic);
#else   // !defined(USE_IBVERBS)
  SKIP_TEST() << "This test requires IBVerbs that the current build does not support.";
#endif  // !defined(USE_IBVERBS)
}
