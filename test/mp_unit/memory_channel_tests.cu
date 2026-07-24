// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>

#include "mp_unit_tests.hpp"

void MemoryChannelOneToOneTest::SetUp() {
  // Need at least two ranks within a node
  if (gEnv->nRanksPerNode < 2) {
    SKIP_TEST();
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
  for (int nElem : {2, 1024, 1024 * 1024, 4 * 1024 * 1024}) {
    *ret = 0;
    kernelWrapper(buff.get(), gEnv->rank, nElem, ret.get(), defaultNTries);
    MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
    EXPECT_EQ(*ret, 0);
  }

  int nTries = 1000000;
  communicator->bootstrap()->barrier();
  mscclpp::Timer timer;
  kernelWrapper(buff.get(), gEnv->rank, 1024, ret.get(), nTries);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  communicator->bootstrap()->barrier();

  if (gEnv->rank == 0) {
    ::mscclpp::test::reportPerfResult("latency", (float)timer.elapsed() / (float)(nTries), "us/iter");
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

TEST(MemoryChannelOneToOneTest, PutPingPong) {
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

  for (int nElem : {1, 1024, 1024 * 1024, 4 * 1024 * 1024}) {
    *ret = 0;
    kernelMemPutPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, nElem, ret.get());
    MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
    EXPECT_EQ(*ret, 0);
  }
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

TEST(MemoryChannelOneToOneTest, GetPingPong) {
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

  for (int nElem : {1, 1024, 1024 * 1024, 4 * 1024 * 1024}) {
    *ret = 0;
    kernelMemGetPingPong<<<1, 1024>>>(buff.get(), gEnv->rank, nElem, ret.get());
    MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
    EXPECT_EQ(*ret, 0);
  }
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

PERF_TEST(MemoryChannelOneToOneTest, LL8PacketPingPong) {
  auto kernelMemLL8PacketPingPongWrapper = [](int* buff, int rank, int nElem, int* ret, int nTries) {
    kernelMemLL8PacketPingPong<<<1, 1024>>>(buff, rank, nElem, ret, nTries);
  };
  packetPingPongTest("memoryLL8PacketPingPong", kernelMemLL8PacketPingPongWrapper);
}

PERF_TEST(MemoryChannelOneToOneTest, LL16PacketPingPong) {
  auto kernelMemLL16PacketPingPongWrapper = [](int* buff, int rank, int nElem, int* ret, int nTries) {
    kernelMemLL16PacketPingPong<<<1, 1024>>>(buff, rank, nElem, ret, nTries);
  };
  packetPingPongTest("memoryLL16PacketPingPong", kernelMemLL16PacketPingPongWrapper);
}

// ------------------------------------------------------------------------------------------------
// Bulk load tests (NVIDIA sm_90+), modeled on the MoE expert-parallel combine pattern (PR #852):
// gather several peer contributions into shared memory through the channel, then reduce them.
// Everything goes through the MemoryChannel handle (getBulk + a BulkLoad completion object); no
// architecture-specific names appear in user code. Skipped where the channel reports no bulk-load
// support.
// ------------------------------------------------------------------------------------------------

static constexpr uint32_t kBulkTile = 16384;
static constexpr uint32_t kBulkTilePipe = 8192;

// True if the current device supports channel bulk loads (compute capability >= 9.0).
static bool bulkSupported() {
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return false;
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return false;
  return prop.major >= 9;
}

using MemChan = DeviceHandle<mscclpp::MemoryChannel>;

// Single-source stage: rank 1 stages each remote tile into shared memory with the channel's
// getBulk() and copies it out (a stand-in for a fused reduction), then verifies. One BulkLoad is
// reset once and reused across chunks.
template <uint32_t TILE>
__global__ void kernelMemGetBulkFused(int* buff, int* out, int rank, int nElem, int* ret) {
  if (rank > 1) return;

  MemChan& memChan = gChannelOneToOneTestConstMemChans;
  __shared__ mscclpp::BulkScratch<TILE> scratch;

  if (rank == 0) {
    for (int j = threadIdx.x; j < nElem; j += blockDim.x) {
      buff[j] = 7 + j;
    }
    __syncthreads();
    if (threadIdx.x == 0) memChan.signal();
    if (threadIdx.x == 0) memChan.wait();  // keep buffer/channel alive until rank 1 finishes reading
    return;
  }

  if (threadIdx.x == 0) memChan.wait();
  __syncthreads();
  {
    const uint64_t bytes = (uint64_t)nElem * sizeof(int);
    if (threadIdx.x == 0) scratch.load.reset();  // prepare once; reused across chunks
    for (uint64_t off = 0; off < bytes; off += TILE) {
      const uint64_t remain = bytes - off;
      const uint32_t chunk = remain < TILE ? (uint32_t)remain : TILE;
      if (threadIdx.x == 0) {
        memChan.getBulk(scratch.tile(), scratch.load, off, chunk);  // remote (dst_) -> smem
        scratch.load.wait();
      }
      __syncthreads();  // publish the staged tile to all threads
      const int* smemInt = reinterpret_cast<const int*>(scratch.tile());
      const uint32_t nInt = chunk / sizeof(int);
      const uint64_t base = off / sizeof(int);
      for (uint32_t k = threadIdx.x; k < nInt; k += blockDim.x) {
        out[base + k] = smemInt[k];  // fuse point: here a real kernel would reduce from smem
      }
      __syncthreads();  // done reading the tile before the next getBulk overwrites it
    }
    for (int j = threadIdx.x; j < nElem; j += blockDim.x) {
      if (out[j] != 7 + j) {
        *ret = 1;
        break;
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) memChan.signal();  // release rank 0
  }
}

TEST(MemoryChannelOneToOneTest, GetBulkFused) {
  if (gEnv->rank >= numRanksToUse) return;
  if (!bulkSupported()) {
    SKIP_TEST() << "Channel bulk load requires compute capability >= 9.0.";
    return;
  }

  const int kMaxElem = 4 * 1024 * 1024;

  std::vector<mscclpp::MemoryChannel> memoryChannels;
  std::shared_ptr<int> buff = mscclpp::GpuBuffer<int>(kMaxElem).memory();
  std::shared_ptr<int> out = mscclpp::GpuBuffer<int>(kMaxElem).memory();
  setupMeshConnections(memoryChannels, buff.get(), kMaxElem * sizeof(int));
  std::vector<MemChan> deviceHandles(memoryChannels.size());
  std::transform(memoryChannels.begin(), memoryChannels.end(), deviceHandles.begin(),
                 [](const mscclpp::MemoryChannel& memChan) { return mscclpp::deviceHandle(memChan); });

  ASSERT_EQ(deviceHandles.size(), 1);
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestConstMemChans, deviceHandles.data(), sizeof(MemChan)));

  std::shared_ptr<int> ret = mscclpp::detail::gpuCallocHostShared<int>();

  for (int nElem : {1024, 1024 * 1024, 4 * 1024 * 1024}) {
    *ret = 0;
    kernelMemGetBulkFused<kBulkTile><<<1, 1024>>>(buff.get(), out.get(), gEnv->rank, nElem, ret.get());
    MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
    EXPECT_EQ(*ret, 0);
  }
}

// Multi-source gather + fused reduce (the EP combine pattern). The peer buffer holds two
// contributor slots; rank 1 issues one channel getBulk() per slot into a distinct shared tile, all
// tracked by ONE BulkLoad, then waits once and reduces both contributors from shared memory. The
// BulkLoad is reset once and reused across chunks. All communication is expressed through the
// channel handle.
template <uint32_t TILE, int NumContrib>
__global__ void kernelMemGatherReduceBulk(int* buff, int* out, int rank, int nElem, int* ret) {
  if (rank > 1) return;

  MemChan& memChan = gChannelOneToOneTestConstMemChans;
  __shared__ mscclpp::BulkScratch<TILE, NumContrib> scratch;
  const uint64_t slotBytes = (uint64_t)nElem * sizeof(int);

  if (rank == 0) {
    // Fill NumContrib slots; slot c holds value (c + 1) * 100 + j.
    for (int c = 0; c < NumContrib; ++c) {
      int* slot = buff + (int64_t)c * nElem;
      for (int j = threadIdx.x; j < nElem; j += blockDim.x) slot[j] = (c + 1) * 100 + j;
    }
    __syncthreads();
    if (threadIdx.x == 0) memChan.signal();
    if (threadIdx.x == 0) memChan.wait();  // keep buffer stable until rank 1 finishes reading
    return;
  }

  if (threadIdx.x == 0) memChan.wait();
  __syncthreads();

  const uint64_t bytes = slotBytes;
  if (threadIdx.x == 0) scratch.load.reset();  // prepare once; reused across chunks
  for (uint64_t off = 0; off < bytes; off += TILE) {
    const uint64_t remain = bytes - off;
    const uint32_t chunk = remain < TILE ? (uint32_t)remain : TILE;
    if (threadIdx.x == 0) {
      for (int c = 0; c < NumContrib; ++c) {
        memChan.getBulk(scratch.tile(c), scratch.load, (uint64_t)c * slotBytes + off, chunk);  // slot c -> smem
      }
      scratch.load.wait();  // one completion object tracks all loads
    }
    __syncthreads();
    const uint32_t nInt = chunk / sizeof(int);
    const uint64_t base = off / sizeof(int);
    for (uint32_t k = threadIdx.x; k < nInt; k += blockDim.x) {
      int acc = 0;
#pragma unroll
      for (int c = 0; c < NumContrib; ++c) acc += reinterpret_cast<const int*>(scratch.tile(c))[k];
      out[base + k] = acc;  // fused reduce across contributors
    }
    __syncthreads();
  }

  // Expected: sum_c ((c+1)*100 + j) = 100*NumContrib*(NumContrib+1)/2 + NumContrib*j.
  const int base = 100 * NumContrib * (NumContrib + 1) / 2;
  for (int j = threadIdx.x; j < nElem; j += blockDim.x) {
    if (out[j] != base + NumContrib * j) {
      *ret = 1;
      break;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) memChan.signal();  // release rank 0
}

TEST(MemoryChannelOneToOneTest, GatherReduceBulk) {
  if (gEnv->rank >= numRanksToUse) return;
  if (!bulkSupported()) {
    SKIP_TEST() << "Channel bulk load requires compute capability >= 9.0.";
    return;
  }

  constexpr int kNumContrib = 2;
  const int kMaxElem = 4 * 1024 * 1024;

  std::vector<mscclpp::MemoryChannel> memoryChannels;
  std::shared_ptr<int> buff = mscclpp::GpuBuffer<int>(kMaxElem * kNumContrib).memory();  // NumContrib slots
  std::shared_ptr<int> out = mscclpp::GpuBuffer<int>(kMaxElem).memory();
  setupMeshConnections(memoryChannels, buff.get(), (size_t)kMaxElem * kNumContrib * sizeof(int));
  std::vector<MemChan> deviceHandles(memoryChannels.size());
  std::transform(memoryChannels.begin(), memoryChannels.end(), deviceHandles.begin(),
                 [](const mscclpp::MemoryChannel& memChan) { return mscclpp::deviceHandle(memChan); });

  ASSERT_EQ(deviceHandles.size(), 1);
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestConstMemChans, deviceHandles.data(), sizeof(MemChan)));

  std::shared_ptr<int> ret = mscclpp::detail::gpuCallocHostShared<int>();

  for (int nElem : {1024, 1024 * 1024, 4 * 1024 * 1024}) {
    *ret = 0;
    kernelMemGatherReduceBulk<kBulkTile, kNumContrib><<<1, 1024>>>(buff.get(), out.get(), gEnv->rank, nElem, ret.get());
    MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
    EXPECT_EQ(*ret, 0);
  }
}

// Double-buffered (two-stage) gather + reduce, mirroring the EP combine's software pipeline: while
// reducing one chunk, prefetch the next chunk's loads into the other stage's scratch. Each stage's
// BulkLoad is reset per issue, so every wait is phase 0.
template <uint32_t TILE, int NumContrib>
__global__ void kernelMemPipelinedGatherReduceBulk(int* buff, int* out, int rank, int nElem, int* ret) {
  if (rank > 1) return;
  constexpr int NumStages = 2;

  MemChan& memChan = gChannelOneToOneTestConstMemChans;
  __shared__ mscclpp::BulkScratch<TILE, NumContrib> stages[NumStages];
  const uint64_t slotBytes = (uint64_t)nElem * sizeof(int);

  if (rank == 0) {
    for (int c = 0; c < NumContrib; ++c) {
      int* slot = buff + (int64_t)c * nElem;
      for (int j = threadIdx.x; j < nElem; j += blockDim.x) slot[j] = (c + 1) * 100 + j;
    }
    __syncthreads();
    if (threadIdx.x == 0) memChan.signal();
    if (threadIdx.x == 0) memChan.wait();
    return;
  }

  if (threadIdx.x == 0) memChan.wait();
  __syncthreads();

  const uint64_t bytes = slotBytes;
  const uint64_t nChunks = (bytes + TILE - 1) / TILE;

  auto issue = [&](int stage, uint64_t chunkIdx) {
    const uint64_t off = chunkIdx * TILE;
    const uint64_t remain = bytes - off;
    const uint32_t chunk = remain < TILE ? (uint32_t)remain : TILE;
    stages[stage].load.reset();
    for (int c = 0; c < NumContrib; ++c) {
      memChan.getBulk(stages[stage].tile(c), stages[stage].load, (uint64_t)c * slotBytes + off, chunk);
    }
  };

  if (threadIdx.x == 0 && nChunks > 0) issue(0, 0);  // prime stage 0

  for (uint64_t chunkIdx = 0; chunkIdx < nChunks; ++chunkIdx) {
    const int stage = (int)(chunkIdx % NumStages);
    if (threadIdx.x == 0) {
      if (chunkIdx + 1 < nChunks) issue((int)((chunkIdx + 1) % NumStages), chunkIdx + 1);  // prefetch
      stages[stage].load.wait();
    }
    __syncthreads();
    const uint64_t off = chunkIdx * TILE;
    const uint64_t remain = bytes - off;
    const uint32_t chunk = remain < TILE ? (uint32_t)remain : TILE;
    const uint32_t nInt = chunk / sizeof(int);
    const uint64_t base = off / sizeof(int);
    for (uint32_t k = threadIdx.x; k < nInt; k += blockDim.x) {
      int acc = 0;
#pragma unroll
      for (int c = 0; c < NumContrib; ++c) acc += reinterpret_cast<const int*>(stages[stage].tile(c))[k];
      out[base + k] = acc;
    }
    __syncthreads();  // finished consuming this stage's tiles before they are reused
  }

  const int vbase = 100 * NumContrib * (NumContrib + 1) / 2;
  for (int j = threadIdx.x; j < nElem; j += blockDim.x) {
    if (out[j] != vbase + NumContrib * j) {
      *ret = 1;
      break;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) memChan.signal();
}

TEST(MemoryChannelOneToOneTest, PipelinedGatherReduceBulk) {
  if (gEnv->rank >= numRanksToUse) return;
  if (!bulkSupported()) {
    SKIP_TEST() << "Channel bulk load requires compute capability >= 9.0.";
    return;
  }

  constexpr int kNumContrib = 2;
  const int kMaxElem = 4 * 1024 * 1024;

  std::vector<mscclpp::MemoryChannel> memoryChannels;
  std::shared_ptr<int> buff = mscclpp::GpuBuffer<int>(kMaxElem * kNumContrib).memory();
  std::shared_ptr<int> out = mscclpp::GpuBuffer<int>(kMaxElem).memory();
  setupMeshConnections(memoryChannels, buff.get(), (size_t)kMaxElem * kNumContrib * sizeof(int));
  std::vector<MemChan> deviceHandles(memoryChannels.size());
  std::transform(memoryChannels.begin(), memoryChannels.end(), deviceHandles.begin(),
                 [](const mscclpp::MemoryChannel& memChan) { return mscclpp::deviceHandle(memChan); });

  ASSERT_EQ(deviceHandles.size(), 1);
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gChannelOneToOneTestConstMemChans, deviceHandles.data(), sizeof(MemChan)));

  std::shared_ptr<int> ret = mscclpp::detail::gpuCallocHostShared<int>();

  for (int nElem : {1024, 1024 * 1024, 4 * 1024 * 1024}) {
    *ret = 0;
    kernelMemPipelinedGatherReduceBulk<kBulkTilePipe, kNumContrib>
        <<<1, 1024>>>(buff.get(), out.get(), gEnv->rank, nElem, ret.get());
    MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
    EXPECT_EQ(*ret, 0);
  }
}
