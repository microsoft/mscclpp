#include "mp_unit_tests.hpp"

void DirectChannelOneToOneTest::SetUp() {
  // Need at least two ranks within a node
  if (gEnv->nRanksPerNode < 2) {
    GTEST_SKIP();
  }
  // Use only two ranks
  setNumRanksToUse(2);
  CommunicatorTestBase::SetUp();
}

void DirectChannelOneToOneTest::TearDown() { CommunicatorTestBase::TearDown(); }

void DirectChannelOneToOneTest::setupMeshConnections(std::vector<mscclpp::channel::DirectChannel>& dirChannels,
                                                     void* inputBuff, size_t inputBuffBytes, void* outputBuff,
                                                     size_t outputBuffBytes) {
  const int rank = communicator->bootstrapper()->getRank();
  const int worldSize = communicator->bootstrapper()->getNranks();
  const bool isInPlace = (outputBuff == nullptr);
  mscclpp::TransportFlags transport = mscclpp::Transport::CudaIpc | ibTransport;

  mscclpp::RegisteredMemory inputBufRegMem = communicator->registerMemory(inputBuff, inputBuffBytes, transport);
  mscclpp::RegisteredMemory outputBufRegMem;
  if (!isInPlace) {
    outputBufRegMem = communicator->registerMemory(outputBuff, outputBuffBytes, transport);
  }

  for (int r = 0; r < worldSize; r++) {
    if (r == rank) {
      continue;
    }
    std::shared_ptr<mscclpp::Connection> conn;
    if (rankToNode(r) == rankToNode(gEnv->rank)) {
      conn = communicator->connectOnSetup(r, 0, mscclpp::Transport::CudaIpc);
    } else {
      conn = communicator->connectOnSetup(r, 0, ibTransport);
    }
    connections[r] = conn;

    if (isInPlace) {
      communicator->sendMemoryOnSetup(inputBufRegMem, r, 0);
    } else {
      communicator->sendMemoryOnSetup(outputBufRegMem, r, 0);
    }
    auto remoteMemory = communicator->recvMemoryOnSetup(r, 0);

    communicator->setup();

    directEpochs[r] = std::make_shared<mscclpp::DirectEpoch>(*communicator, conn);

    communicator->setup();

    dirChannels.emplace_back(directEpochs[r]->deviceHandle(), remoteMemory.get(), inputBufRegMem.data(),
                             (isInPlace ? nullptr : outputBufRegMem.data()));
  }
}

__constant__ mscclpp::channel::DirectChannel gChannelOneToOneTestConstDirChans;

__global__ void kernelDirectPingPong(int* buff, int rank, int nElem, int* ret) {
  mscclpp::channel::DirectChannel& dirChan = gChannelOneToOneTestConstDirChans;
  volatile int* sendBuff = (volatile int*)buff;
  int nTries = 1000;
  int rank1Offset = 10000000;
  for (int i = 0; i < nTries; i++) {
    if (rank == 0) {
      if (i > 0) {
        if (threadIdx.x == 0) dirChan.wait();
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
      dirChan.put(0, 0, nElem * sizeof(int), threadIdx.x, blockDim.x);
      if (threadIdx.x == 0) dirChan.signal();
    }
    if (rank == 1) {
      if (threadIdx.x == 0) dirChan.wait();
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
        dirChan.put(0, 0, nElem * sizeof(int), threadIdx.x, blockDim.x);
        if (threadIdx.x == 0) dirChan.signal();
      }
    }
  }
}

TEST_F(DirectChannelOneToOneTest, PingPong) {
  if (gEnv->rank >= numRanksToUse) return;

  const int nElem = 4 * 1024 * 1024;

  std::vector<mscclpp::channel::DirectChannel> dirChannels;
  std::shared_ptr<int> buff = mscclpp::allocSharedCuda<int>(nElem);
  setupMeshConnections(dirChannels, buff.get(), nElem * sizeof(int));

  ASSERT_EQ(dirChannels.size(), 1);
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestConstDirChans, dirChannels.data(),
                                       sizeof(mscclpp::channel::DirectChannel)));

  std::shared_ptr<int> ret = mscclpp::makeSharedCudaHost<int>(0);

  kernelDirectPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelDirectPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1024, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelDirectPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1024 * 1024, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelDirectPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 4 * 1024 * 1024, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
}

__global__ void kernelDirectPacketPingPong(int* buff, int rank, int nElem, int* ret) {
  if (rank > 1) return;

  mscclpp::channel::DirectChannel& dirChan = gChannelOneToOneTestConstDirChans;
  volatile int* sendBuff = (volatile int*)buff;
  int nTries = 1000;
  int putOffset = (rank == 0) ? 0 : 10000000;
  int getOffset = (rank == 0) ? 10000000 : 0;
  for (int i = 0; i < nTries; i++) {
    uint64_t flag = (uint64_t)i + 1;

    // rank=0: 0, 1, 0, 1, ...
    // rank=1: 1, 0, 1, 0, ...
    if ((rank ^ (i & 1)) == 0) {
      // If each thread writes 8 bytes at once, we don't need a barrier before putPacket().
      for (int j = threadIdx.x; j < nElem / 2; j += blockDim.x) {
        sendBuff[2 * j] = putOffset + i + 2 * j;
        sendBuff[2 * j + 1] = putOffset + i + 2 * j + 1;
      }
      // __syncthreads();
      dirChan.putPacket(0, 0, nElem * sizeof(int), threadIdx.x, blockDim.x, flag);
    } else {
      dirChan.getPacket(0, 0, nElem * sizeof(int), threadIdx.x, blockDim.x, flag);
      // If each thread reads 8 bytes at once, we don't need a barrier after getPacket().
      // __syncthreads();
      for (int j = threadIdx.x; j < nElem / 2; j += blockDim.x) {
        if (sendBuff[2 * j] != getOffset + i + 2 * j) {
          // printf("ERROR: rank = %d, sendBuff[%d] = %d, expected %d. Skipping following errors\n", rank, 2 * j,
          //        sendBuff[2 * j], getOffset + i + 2 * j);
          *ret = 1;
          break;
        }
        if (sendBuff[2 * j + 1] != getOffset + i + 2 * j + 1) {
          // printf("ERROR: rank = %d, sendBuff[%d] = %d, expected %d. Skipping following errors\n", rank, 2 * j + 1,
          //        sendBuff[2 * j + 1], getOffset + i + 2 * j + 1);
          *ret = 1;
          break;
        }
      }
    }
    // Make sure all threads are done in this iteration
    __syncthreads();
  }
}

TEST_F(DirectChannelOneToOneTest, PacketPingPong) {
  if (gEnv->rank >= numRanksToUse) return;

  const int nElem = 4 * 1024 * 1024;

  std::vector<mscclpp::channel::DirectChannel> dirChannels;
  std::shared_ptr<int> buff = mscclpp::allocSharedCuda<int>(nElem);
  std::shared_ptr<int> intermBuff = mscclpp::allocSharedCuda<int>(nElem * 2);
  setupMeshConnections(dirChannels, buff.get(), nElem * sizeof(int), intermBuff.get(), nElem * 2 * sizeof(int));

  ASSERT_EQ(dirChannels.size(), 1);
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestConstDirChans, dirChannels.data(),
                                       sizeof(mscclpp::channel::DirectChannel)));

  std::shared_ptr<int> ret = mscclpp::makeSharedCudaHost<int>(0);

  // The least nelem is 2 for packet ping pong
  kernelDirectPacketPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 2, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelDirectPacketPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1024, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelDirectPacketPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1024 * 1024, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelDirectPacketPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 4 * 1024 * 1024, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
}
