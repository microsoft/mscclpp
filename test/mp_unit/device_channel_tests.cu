// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "mp_unit_tests.hpp"

void DeviceChannelOneToOneTest::SetUp() {
  // Use only two ranks
  setNumRanksToUse(2);
  CommunicatorTestBase::SetUp();
  channelService = std::make_shared<mscclpp::channel::DeviceChannelService>(*communicator.get());
}

void DeviceChannelOneToOneTest::TearDown() { CommunicatorTestBase::TearDown(); }

void DeviceChannelOneToOneTest::setupMeshConnections(std::vector<mscclpp::channel::SimpleDeviceChannel>& devChannels,
                                                     bool useIbOnly, void* sendBuff, size_t sendBuffBytes,
                                                     void* recvBuff, size_t recvBuffBytes) {
  const int rank = communicator->bootstrapper()->getRank();
  const int worldSize = communicator->bootstrapper()->getNranks();
  const bool isInPlace = (recvBuff == nullptr);
  mscclpp::TransportFlags transport = mscclpp::Transport::CudaIpc | ibTransport;

  connectMesh(useIbOnly);

  for (int r = 0; r < worldSize; r++) {
    if (r == rank) {
      continue;
    }
    mscclpp::RegisteredMemory sendMemory;
    mscclpp::RegisteredMemory remoteMemory;

    if (isInPlace) {
      registerMemoryPair(sendBuff, sendBuffBytes, transport, 0, r, sendMemory, remoteMemory);
    } else {
      sendMemory = communicator->registerMemory(recvBuff, recvBuffBytes, transport);
      mscclpp::RegisteredMemory recvMemory;
      registerMemoryPair(recvBuff, recvBuffBytes, transport, 0, r, recvMemory, remoteMemory);
    }

    mscclpp::channel::ChannelId cid = channelService->addChannel(connections[r]);
    communicator->setup();

    devChannels.emplace_back(channelService->deviceChannel(cid), channelService->addMemory(remoteMemory),
                             channelService->addMemory(sendMemory));
  }
}

__constant__ mscclpp::channel::SimpleDeviceChannel gChannelOneToOneTestConstDevChans;

__global__ void kernelDevicePingPong(int* buff, int rank, int nElem, int* ret) {
  mscclpp::channel::SimpleDeviceChannel& devChan = gChannelOneToOneTestConstDevChans;
  volatile int* sendBuff = (volatile int*)buff;
  int nTries = 1000;
  int flusher = 0;
  int rank1Offset = 10000000;
  for (int i = 0; i < nTries; i++) {
    if (rank == 0) {
      if (i > 0) {
        if (threadIdx.x == 0) devChan.wait();
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
      if (threadIdx.x == 0) devChan.putWithSignal(0, nElem * sizeof(int));
    }
    if (rank == 1) {
      if (threadIdx.x == 0) devChan.wait();
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
        if (threadIdx.x == 0) devChan.putWithSignal(0, nElem * sizeof(int));
      }
    }
    flusher++;
    if (flusher == 100) {
      if (threadIdx.x == 0) devChan.flush();
      flusher = 0;
    }
  }
}

TEST_F(DeviceChannelOneToOneTest, PingPongIb) {
  if (gEnv->rank >= numRanksToUse) return;

  const int nElem = 4 * 1024 * 1024;

  std::vector<mscclpp::channel::SimpleDeviceChannel> devChannels;
  std::shared_ptr<int> buff = mscclpp::allocSharedCuda<int>(nElem);
  setupMeshConnections(devChannels, true, buff.get(), nElem * sizeof(int));

  ASSERT_EQ(devChannels.size(), 1);
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestConstDevChans, devChannels.data(),
                                       sizeof(mscclpp::channel::SimpleDeviceChannel)));

  channelService->startProxy();

  std::shared_ptr<int> ret = mscclpp::makeSharedCudaHost<int>(0);

  kernelDevicePingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);

  kernelDevicePingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1024, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);

  kernelDevicePingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1024 * 1024, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);

  kernelDevicePingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 4 * 1024 * 1024, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);

  channelService->stopProxy();
}
