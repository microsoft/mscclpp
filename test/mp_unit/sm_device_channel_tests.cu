#include "mp_unit_tests.hpp"

void SmDeviceChannelOneToOneTest::SetUp() {
  // Use only two ranks
  setNumRanksToUse(2);
  CommunicatorTestBase::SetUp();
  channelService = std::make_shared<mscclpp::channel::SmDeviceChannelService>(*communicator.get());
}

void SmDeviceChannelOneToOneTest::TearDown() { CommunicatorTestBase::TearDown(); }

void SmDeviceChannelOneToOneTest::setupMeshConnections(
    std::vector<mscclpp::channel::SimpleSmDeviceChannel>& smDevChannels, bool useIbOnly, void* inputBuff,
    size_t inputBuffBytes) {
  const int rank = communicator->bootstrapper()->getRank();
  const int worldSize = communicator->bootstrapper()->getNranks();
  mscclpp::TransportFlags transport = (useIbOnly) ? ibTransport : (mscclpp::Transport::CudaIpc | ibTransport);

  const size_t nPacket = (inputBuffBytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);
  putPacketBuffer = mscclpp::allocSharedCuda<mscclpp::channel::ChannelPacket>(nPacket);
  getPacketBuffer = mscclpp::allocSharedCuda<mscclpp::channel::ChannelPacket>(nPacket);

  mscclpp::RegisteredMemory putPacketBufRegMem =
      communicator->registerMemory(putPacketBuffer.get(), nPacket * sizeof(mscclpp::channel::ChannelPacket), transport);
  mscclpp::RegisteredMemory getPacketBufRegMem =
      communicator->registerMemory(getPacketBuffer.get(), nPacket * sizeof(mscclpp::channel::ChannelPacket), transport);

  for (int r = 0; r < worldSize; r++) {
    if (r == rank) {
      continue;
    }
    std::shared_ptr<mscclpp::Connection> conn;
    if ((rankToNode(r) == rankToNode(gEnv->rank)) && !useIbOnly) {
      conn = communicator->connectOnSetup(r, 0, mscclpp::Transport::CudaIpc);
    } else {
      conn = communicator->connectOnSetup(r, 0, ibTransport);
    }
    connections[r] = conn;

    communicator->sendMemoryOnSetup(getPacketBufRegMem, r, 0);
    auto remoteMemory = communicator->recvMemoryOnSetup(r, 0);

    communicator->setup();

    uint32_t eid = channelService->addEpoch(conn);
    communicator->setup();

    smDevChannels.emplace_back(channelService->deviceChannel(eid), channelService->addMemory(remoteMemory.get()),
                               channelService->addMemory(putPacketBufRegMem), putPacketBufRegMem.data(),
                               getPacketBufRegMem.data());
  }
}

// SimpleSmDeviceChannel cannot be on the constant memory because it has a DeviceSyncer internally.
__device__ mscclpp::channel::SimpleSmDeviceChannel gChannelOneToOneTestSmDevChans;
__device__ mscclpp::DeviceSyncer gChannelOneToOneTestSmDevChansSyncer;

template <bool CheckCorrectness>
__global__ void kernelSmDevicePacketPingPong(int* buff, int rank, int nElem, int nTries, int* ret) {
  if (rank > 1) return;

  mscclpp::channel::SimpleSmDeviceChannel& smDevChan = gChannelOneToOneTestSmDevChans;
  volatile int* buffPtr = (volatile int*)buff;
  int putOffset = (rank == 0) ? 0 : 10000000;
  int getOffset = (rank == 0) ? 10000000 : 0;
  int threadId = threadIdx.x + blockIdx.x * blockDim.x;
  int numThreads = blockDim.x * gridDim.x;
  int flusher = 0;
  const size_t nPkt = nElem / 2;
  for (int i = 0; i < nTries; i++) {
    uint64_t flag = (uint64_t)i + 1;

    // rank=0: 0, 1, 0, 1, ...
    // rank=1: 1, 0, 1, 0, ...
    if ((rank ^ (i & 1)) == 0) {
      if (CheckCorrectness) {
        // If each thread writes 8 bytes at once, we don't need a barrier before putPacket().
        for (int j = threadId; j < nPkt; j += numThreads) {
          buffPtr[2 * j] = putOffset + i + 2 * j;
          buffPtr[2 * j + 1] = putOffset + i + 2 * j + 1;
        }
        // __syncthreads();
      }
      mscclpp::channel::putPackets(smDevChan.putPacketBuffer_, 0, buff, 0, nElem * sizeof(int), threadId, numThreads,
                                   flag);
      gChannelOneToOneTestSmDevChansSyncer.sync(gridDim.x);
      if (threadId == 0) {
        // Send data from the local putPacketBuffer to the remote getPacketBuffer
        smDevChan.put(0, nPkt * sizeof(mscclpp::channel::ChannelPacket));
      }
      flusher++;
      if (flusher == 64) {
        if (threadId == 0) smDevChan.flush();
        flusher = 0;
      }
    } else {
      mscclpp::channel::getPackets(buff, 0, smDevChan.getPacketBuffer_, 0, nElem * sizeof(int), threadId, numThreads,
                                   flag);
      if (CheckCorrectness) {
        // If each thread reads 8 bytes at once, we don't need a barrier after getPacket().
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
      gChannelOneToOneTestSmDevChansSyncer.sync(gridDim.x);
    }
  }
}

void SmDeviceChannelOneToOneTest::testPacketPingPong(bool useIbOnly) {
  if (gEnv->rank >= numRanksToUse) return;

  const int nElem = 4 * 1024 * 1024;

  std::vector<mscclpp::channel::SimpleSmDeviceChannel> smDevChannels;
  std::shared_ptr<int> buff = mscclpp::allocSharedCuda<int>(nElem);

  setupMeshConnections(smDevChannels, useIbOnly, buff.get(), nElem * sizeof(int));

  ASSERT_EQ(smDevChannels.size(), 1);
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestSmDevChans, smDevChannels.data(),
                                       sizeof(mscclpp::channel::SimpleSmDeviceChannel)));

  mscclpp::DeviceSyncer syncer = {};
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestSmDevChansSyncer, &syncer, sizeof(mscclpp::DeviceSyncer)));

  channelService->startProxy();

  std::shared_ptr<int> ret = mscclpp::makeSharedCudaHost<int>(0);

  const int nTries = 1000;

  // The least nelem is 2 for packet ping pong
  kernelSmDevicePacketPingPong<true><<<1, 1024>>>(buff.get(), gEnv->rank, 2, nTries, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelSmDevicePacketPingPong<true><<<1, 1024>>>(buff.get(), gEnv->rank, 1024, nTries, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelSmDevicePacketPingPong<true><<<1, 1024>>>(buff.get(), gEnv->rank, 1024 * 1024, nTries, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelSmDevicePacketPingPong<true><<<1, 1024>>>(buff.get(), gEnv->rank, 4 * 1024 * 1024, nTries, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);

  communicator->bootstrapper()->barrier();

  channelService->stopProxy();
}

void SmDeviceChannelOneToOneTest::testPacketPingPongPerf(bool useIbOnly) {
  if (gEnv->rank >= numRanksToUse) return;

  const int nElem = 4 * 1024 * 1024;

  std::vector<mscclpp::channel::SimpleSmDeviceChannel> smDevChannels;
  std::shared_ptr<int> buff = mscclpp::allocSharedCuda<int>(nElem);

  setupMeshConnections(smDevChannels, useIbOnly, buff.get(), nElem * sizeof(int));

  ASSERT_EQ(smDevChannels.size(), 1);
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestSmDevChans, smDevChannels.data(),
                                       sizeof(mscclpp::channel::SimpleSmDeviceChannel)));

  mscclpp::DeviceSyncer syncer = {};
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestSmDevChansSyncer, &syncer, sizeof(mscclpp::DeviceSyncer)));

  channelService->startProxy();

  auto* testInfo = ::testing::UnitTest::GetInstance()->current_test_info();
  const std::string testName = std::string(testInfo->test_suite_name()) + "." + std::string(testInfo->name());
  const int nTries = 1000;

  // Warm-up
  kernelSmDevicePacketPingPong<false><<<1, 1024>>>(buff.get(), gEnv->rank, 2, nTries, nullptr);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  communicator->bootstrapper()->barrier();

  // Measure latency
  mscclpp::Timer timer;
  kernelSmDevicePacketPingPong<false><<<1, 1024>>>(buff.get(), gEnv->rank, 2, nTries, nullptr);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  communicator->bootstrapper()->barrier();

  if (gEnv->rank == 0) {
    std::cout << testName << ": " << std::setprecision(4) << (float)timer.elapsed() / (float)nTries << " us/iter\n";
  }

  channelService->stopProxy();
}

TEST_F(SmDeviceChannelOneToOneTest, PacketPingPong) { testPacketPingPong(false); }

TEST_F(SmDeviceChannelOneToOneTest, PacketPingPongIb) { testPacketPingPong(true); }

TEST_F(SmDeviceChannelOneToOneTest, PacketPingPongPerf) { testPacketPingPongPerf(false); }

TEST_F(SmDeviceChannelOneToOneTest, PacketPingPongPerfIb) { testPacketPingPongPerf(true); }
