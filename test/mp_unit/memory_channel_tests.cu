// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <algorithm>

#include "mp_unit_tests.hpp"

void MemoryChannelOneToOneTest::SetUp() {
  // Need at least two ranks within a node
  if (gEnv->nRanksPerNode < 2) {
    GTEST_SKIP();
  }
  // Use only two ranks
  setNumRanksToUse(2);
  CommunicatorTestBase::SetUp();
}

void MemoryChannelOneToOneTest::TearDown() { CommunicatorTestBase::TearDown(); }

void MemoryChannelOneToOneTest::setupMeshConnections(std::vector<mscclpp::MemoryChannel>& memoryChannels,
                                                     void* inputBuff, size_t inputBuffBytes, void* outputBuff,
                                                     size_t outputBuffBytes) {
  const int rank = communicator->bootstrap()->getRank();
  const int worldSize = communicator->bootstrap()->getNranks();
  const bool isInPlace = (outputBuff == nullptr);
  mscclpp::TransportFlags transport = mscclpp::Transport::CudaIpc;

  std::vector<std::shared_future<mscclpp::Connection>> connectionFutures(worldSize);
  std::vector<std::shared_future<mscclpp::RegisteredMemory>> remoteMemFutures(worldSize);

  mscclpp::RegisteredMemory inputBufRegMem = communicator->registerMemory(inputBuff, inputBuffBytes, transport);
  mscclpp::RegisteredMemory outputBufRegMem;
  if (!isInPlace) {
    outputBufRegMem = communicator->registerMemory(outputBuff, outputBuffBytes, transport);
  }

  for (int r = 0; r < worldSize; r++) {
    if (r == rank) {
      continue;
    }
    // No IB for MemoryChannel tests
    connectionFutures[r] = communicator->connect(mscclpp::Transport::CudaIpc, r);

    if (isInPlace) {
      communicator->sendMemory(inputBufRegMem, r);
    } else {
      communicator->sendMemory(outputBufRegMem, r);
    }
    remoteMemFutures[r] = communicator->recvMemory(r);
  }

  for (int r = 0; r < worldSize; r++) {
    if (r == rank) {
      continue;
    }
    auto sema = communicator->buildSemaphore(connectionFutures[r].get(), r).get();

    memoryChannels.emplace_back(sema, remoteMemFutures[r].get(), inputBufRegMem,
                                (isInPlace ? nullptr : outputBufRegMem.data()));
  }
  // keep the registered memories alive until TearDown
  if (!isInPlace) {
    registeredMemories.push_back(outputBufRegMem);
  }
}

__constant__ DeviceHandle<mscclpp::MemoryChannel> gChannelOneToOneTestConstMemChans;

void MemoryChannelOneToOneTest::packetPingPongTest(const std::string testName,
                                                   PacketPingPongKernelWrapper kernelWrapper) {
  if (gEnv->rank >= numRanksToUse) return;

  const int nElem = 4 * 1024 * 1024;
  const int defaultNTries = 1000;

  std::vector<mscclpp::MemoryChannel> memoryChannels;
  std::shared_ptr<int> buff = mscclpp::GpuBuffer<int>(nElem).memory();
  std::shared_ptr<int> intermBuff = mscclpp::GpuBuffer<int>(nElem * 2).memory();
  setupMeshConnections(memoryChannels, buff.get(), nElem * sizeof(int), intermBuff.get(), nElem * 2 * sizeof(int));
  std::vector<DeviceHandle<mscclpp::MemoryChannel>> deviceHandles(memoryChannels.size());
  std::transform(memoryChannels.begin(), memoryChannels.end(), deviceHandles.begin(),
                 [](const mscclpp::MemoryChannel& memChan) { return mscclpp::deviceHandle(memChan); });

  ASSERT_EQ(memoryChannels.size(), 1);
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestConstMemChans, deviceHandles.data(),
                                       sizeof(DeviceHandle<mscclpp::MemoryChannel>)));

  std::shared_ptr<int> ret = mscclpp::detail::gpuCallocHostShared<int>();

  // The least nelem is 2 for packet ping pong
  kernelWrapper(buff.get(), gEnv->rank, 2, ret.get(), defaultNTries);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  *ret = 0;

  kernelWrapper(buff.get(), gEnv->rank, 1024, ret.get(), defaultNTries);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelWrapper(buff.get(), gEnv->rank, 1024 * 1024, ret.get(), defaultNTries);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelWrapper(buff.get(), gEnv->rank, 4 * 1024 * 1024, ret.get(), defaultNTries);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  int nTries = 1000000;
  communicator->bootstrap()->barrier();
  mscclpp::Timer timer;
  kernelWrapper(buff.get(), gEnv->rank, 1024, ret.get(), nTries);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  communicator->bootstrap()->barrier();

  if (gEnv->rank == 0) {
    std::cout << testName << ": " << std::setprecision(4) << (float)timer.elapsed() / (float)(nTries) << " us/iter\n";
  }
}

__global__ void kernelMemPutPingPong(int* buff, int rank, int nElem, int* ret) {
  DeviceHandle<mscclpp::MemoryChannel>& memChan = gChannelOneToOneTestConstMemChans;
  volatile int* sendBuff = (volatile int*)buff;
  int nTries = 1000;
  int rank1Offset = 10000000;
  for (int i = 0; i < nTries; i++) {
    if (rank == 0) {
      if (i > 0) {
        if (threadIdx.x == 0) memChan.wait();
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
      memChan.put(0, 0, nElem * sizeof(int), threadIdx.x, blockDim.x);
      if (threadIdx.x == 0) memChan.signal();
    }
    if (rank == 1) {
      if (threadIdx.x == 0) memChan.wait();
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
        memChan.put(0, 0, nElem * sizeof(int), threadIdx.x, blockDim.x);
        if (threadIdx.x == 0) memChan.signal();
      }
    }
  }
}

TEST_F(MemoryChannelOneToOneTest, PutPingPong) {
  if (gEnv->rank >= numRanksToUse) return;

  const int nElem = 4 * 1024 * 1024;

  std::vector<mscclpp::MemoryChannel> memoryChannels;
  std::shared_ptr<int> buff = mscclpp::GpuBuffer<int>(nElem).memory();
  setupMeshConnections(memoryChannels, buff.get(), nElem * sizeof(int));
  std::vector<DeviceHandle<mscclpp::MemoryChannel>> deviceHandles(memoryChannels.size());
  std::transform(memoryChannels.begin(), memoryChannels.end(), deviceHandles.begin(),
                 [](const mscclpp::MemoryChannel& memChan) { return mscclpp::deviceHandle(memChan); });

  ASSERT_EQ(memoryChannels.size(), 1);
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestConstMemChans, deviceHandles.data(),
                                       sizeof(DeviceHandle<mscclpp::MemoryChannel>)));

  std::shared_ptr<int> ret = mscclpp::detail::gpuCallocHostShared<int>();

  kernelMemPutPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelMemPutPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1024, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelMemPutPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1024 * 1024, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelMemPutPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 4 * 1024 * 1024, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
}

__global__ void kernelMemGetPingPong(int* buff, int rank, int nElem, int* ret) {
  if (rank > 1) return;

  DeviceHandle<mscclpp::MemoryChannel>& memChan = gChannelOneToOneTestConstMemChans;
  volatile int* buffPtr = (volatile int*)buff;
  int offset0 = (rank == 0) ? 0 : 10000000;
  int offset1 = (rank == 0) ? 10000000 : 0;
  int nTries = 1000;

  for (int i = 0; i < nTries; i++) {
    // rank=0: 0, 1, 0, 1, ...
    // rank=1: 1, 0, 1, 0, ...
    if ((rank ^ (i & 1)) == 0) {
      for (int j = threadIdx.x; j < nElem; j += blockDim.x) {
        buffPtr[j] = offset0 + i + j;
      }
      if (threadIdx.x == 0) {
        memChan.signal();
      }
    } else {
      if (threadIdx.x == 0) {
        memChan.wait();
      }
      __syncthreads();
      memChan.get(0, 0, nElem * sizeof(int), threadIdx.x, blockDim.x);
      __syncthreads();
      for (int j = threadIdx.x; j < nElem; j += blockDim.x) {
        if (buffPtr[j] != offset1 + i + j) {
          // printf("rank %d ERROR: buff[%d] = %d, expected %d\n", rank, j, buffPtr[j], offset1 + i + j);
          *ret = 1;
          break;
        }
      }
    }
  }
}

TEST_F(MemoryChannelOneToOneTest, GetPingPong) {
  if (gEnv->rank >= numRanksToUse) return;

  const int nElem = 4 * 1024 * 1024;

  std::vector<mscclpp::MemoryChannel> memoryChannels;
  std::shared_ptr<int> buff = mscclpp::GpuBuffer<int>(nElem).memory();
  setupMeshConnections(memoryChannels, buff.get(), nElem * sizeof(int));
  std::vector<DeviceHandle<mscclpp::MemoryChannel>> deviceHandles(memoryChannels.size());
  std::transform(memoryChannels.begin(), memoryChannels.end(), deviceHandles.begin(),
                 [](const mscclpp::MemoryChannel& memChan) { return mscclpp::deviceHandle(memChan); });

  ASSERT_EQ(deviceHandles.size(), 1);
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestConstMemChans, deviceHandles.data(),
                                       sizeof(DeviceHandle<mscclpp::MemoryChannel>)));

  std::shared_ptr<int> ret = mscclpp::detail::gpuCallocHostShared<int>();

  kernelMemGetPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelMemGetPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1024, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelMemGetPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 1024 * 1024, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
  *ret = 0;

  kernelMemGetPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, 4 * 1024 * 1024, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  EXPECT_EQ(*ret, 0);
}

__global__ void kernelMemLL8PacketPingPong(int* buff, int rank, int nElem, int* ret, int nTries) {
  if (rank > 1) return;

  DeviceHandle<mscclpp::MemoryChannel>& memChan = gChannelOneToOneTestConstMemChans;
  volatile int* sendBuff = (volatile int*)buff;
  int putOffset = (rank == 0) ? 0 : 10000000;
  int getOffset = (rank == 0) ? 10000000 : 0;
  for (int i = 0; i < nTries; i++) {
    uint64_t flag = (uint64_t)i + 1;

    // rank=0: 0, 1, 0, 1, ...
    // rank=1: 1, 0, 1, 0, ...
    if ((rank ^ (i & 1)) == 0) {
      // If each thread writes 8 bytes at once, we don't need a barrier before putPackets().
      for (int j = threadIdx.x; j < nElem; j += blockDim.x) {
        sendBuff[j] = putOffset + i + j;
        // sendBuff[2 * j + 1] = putOffset + i + 2 * j + 1;
      }
      // __syncthreads();
      memChan.putPackets<mscclpp::LL8Packet>(0, 0, nElem * sizeof(int), threadIdx.x, blockDim.x, flag);
    } else {
      memChan.unpackPackets<mscclpp::LL8Packet>(0, 0, nElem * sizeof(int), threadIdx.x, blockDim.x, flag);
      // If each thread reads 8 bytes at once, we don't need a barrier after unpackPackets().
      // __syncthreads();
      for (int j = threadIdx.x; j < nElem; j += blockDim.x) {
        if (sendBuff[j] != getOffset + i + j) {
          // printf("ERROR: rank = %d, sendBuff[%d] = %d, expected %d. Skipping following errors\n", rank, 2 * j,
          //        sendBuff[2 * j], getOffset + i + 2 * j);
          *ret = 1;
          break;
        }
      }
    }
    // Make sure all threads are done in this iteration
    __syncthreads();
  }
}

__global__ void kernelMemLL16PacketPingPong(int* buff, int rank, int nElem, int* ret, int nTries) {
  if (rank > 1) return;

  DeviceHandle<mscclpp::MemoryChannel>& memChan = gChannelOneToOneTestConstMemChans;
  volatile int* sendBuff = (volatile int*)buff;
  int putOffset = (rank == 0) ? 0 : 10000000;
  int getOffset = (rank == 0) ? 10000000 : 0;
  for (int i = 0; i < nTries; i++) {
    uint64_t flag = (uint64_t)i + 1;
    // rank=0: 0, 1, 0, 1, ...
    // rank=1: 1, 0, 1, 0, ...
    if ((rank ^ (i & 1)) == 0) {
      // If each thread writes 8 bytes at once, we don't need a barrier before putPackets().
      for (int j = threadIdx.x; j < nElem / 2; j += blockDim.x) {
        sendBuff[2 * j] = putOffset + i + 2 * j;
        sendBuff[2 * j + 1] = putOffset + i + 2 * j + 1;
      }
      // __syncthreads();
      memChan.putPackets<mscclpp::LL16Packet>(0, 0, nElem * sizeof(int), threadIdx.x, blockDim.x, flag);
    } else {
      memChan.unpackPackets<mscclpp::LL16Packet>(0, 0, nElem * sizeof(int), threadIdx.x, blockDim.x, flag);
      // If each thread reads 8 bytes at once, we don't need a barrier after unpackPackets().
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

TEST_F(MemoryChannelOneToOneTest, LL8PacketPingPong) {
  auto kernelMemLL8PacketPingPongWrapper = [](int* buff, int rank, int nElem, int* ret, int nTries) {
    kernelMemLL8PacketPingPong<<<1, 1024>>>(buff, rank, nElem, ret, nTries);
  };
  packetPingPongTest("memoryLL8PacketPingPong", kernelMemLL8PacketPingPongWrapper);
}

TEST_F(MemoryChannelOneToOneTest, LL16PacketPingPong) {
  auto kernelMemLL16PacketPingPongWrapper = [](int* buff, int rank, int nElem, int* ret, int nTries) {
    kernelMemLL16PacketPingPong<<<1, 1024>>>(buff, rank, nElem, ret, nTries);
  };
  packetPingPongTest("memoryLL16PacketPingPong", kernelMemLL16PacketPingPongWrapper);
}
