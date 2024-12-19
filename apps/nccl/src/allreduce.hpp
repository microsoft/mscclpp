// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ALLREDUCE_HPP_
#define ALLREDUCE_HPP_

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu.hpp>
#include <mscclpp/gpu_data_types.hpp>
#include <mscclpp/packet_device.hpp>
#include <mscclpp/sm_channel.hpp>
#include <mscclpp/sm_channel_device.hpp>

#include "common.hpp"

template <typename To, typename From>
__forceinline__ __device__ To bit_cast(const From& src) {
  static_assert(sizeof(To) == sizeof(From), "Size mismatch for bit_cast");

  union {
    From f;
    To t;
  } u;
  u.f = src;
  return u.t;
}

template <typename T>
__forceinline__ __device__ T clip(T val) {
  return val;
}

template <>
__forceinline__ __device__ __half clip(__half val) {
  val = __hmax(val, bit_cast<__half, unsigned short>(0xfbff));
  val = __hmin(val, bit_cast<__half, unsigned short>(0x7bff));

  return val;
}

template <>
__forceinline__ __device__ __half2 clip(__half2 val) {
  val.x = __hmax(val.x, bit_cast<__half, unsigned short>(0xfbff));
  val.x = __hmin(val.x, bit_cast<__half, unsigned short>(0x7bff));
  val.y = __hmax(val.y, bit_cast<__half, unsigned short>(0xfbff));
  val.y = __hmin(val.y, bit_cast<__half, unsigned short>(0x7bff));
  return val;
}

template <>
__forceinline__ __device__ __bfloat16 clip(__bfloat16 val) {
  val = __hmax(val, bit_cast<__bfloat16, unsigned short>(0xff80));
  val = __hmin(val, bit_cast<__bfloat16, unsigned short>(0x7f80));
  return val;
}

template <>
__forceinline__ __device__ __bfloat162 clip(__bfloat162 val) {
  val.x = __hmax(val.x, bit_cast<__bfloat16, unsigned short>(0xff80));
  val.x = __hmin(val.x, bit_cast<__bfloat16, unsigned short>(0x7f80));
  val.y = __hmax(val.y, bit_cast<__bfloat16, unsigned short>(0xff80));
  val.y = __hmin(val.y, bit_cast<__bfloat16, unsigned short>(0x7f80));
  return val;
}

template <typename T>
__forceinline__ __device__ T add_elements(T a, T b) {
  return clip(a + b);
}

template <>
__forceinline__ __device__ __half2 add_elements(__half2 a, __half2 b) {
  return clip(__hadd2(a, b));
}

template <>
__forceinline__ __device__ __bfloat162 add_elements(__bfloat162 a, __bfloat162 b) {
  return clip(__hadd2(a, b));
}

template <typename T>
__forceinline__ __device__ int4 add_vectors_helper(int4 a, int4 b) {
  int4 ret;
  ret.w = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.w), bit_cast<T, int>(b.w)));
  ret.x = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.x), bit_cast<T, int>(b.x)));
  ret.y = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.y), bit_cast<T, int>(b.y)));
  ret.z = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.z), bit_cast<T, int>(b.z)));
  return ret;
}

template <typename T>
__forceinline__ __device__ int4 add_vectors(int4 a, int4 b) {
  return add_vectors_helper<T>(a, b);
}

template <>
__forceinline__ __device__ int4 add_vectors<__half>(int4 a, int4 b) {
  return add_vectors_helper<__half2>(a, b);
}

template <>
__forceinline__ __device__ int4 add_vectors<__bfloat16>(int4 a, int4 b) {
  return add_vectors_helper<__bfloat162>(a, b);
}

template <typename T>
__forceinline__ __device__ uint2 add_vectors_helper(uint2 a, uint2 b) {
  uint2 ret;
  ret.x = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.x), bit_cast<T, int>(b.x)));
  ret.y = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.y), bit_cast<T, int>(b.y)));
  return ret;
}

template <typename T>
__forceinline__ __device__ uint2 add_vectors(uint2 a, uint2 b) {
  return add_vectors_helper<T>(a, b);
}

template <>
__forceinline__ __device__ uint2 add_vectors<__half>(uint2 a, uint2 b) {
  return add_vectors_helper<__half2>(a, b);
}

template <>
__forceinline__ __device__ uint2 add_vectors<__bfloat16>(uint2 a, uint2 b) {
  return add_vectors_helper<__bfloat162>(a, b);
}

template <typename T>
__forceinline__ __device__ int add_vectors_helper(int a, int b) {
  return bit_cast<int, T>(add_elements(bit_cast<T, int>(a), bit_cast<T, int>(b)));
}

template <typename T>
__forceinline__ __device__ int add_vectors(int a, int b) {
  return add_vectors_helper<T>(a, b);
}

template <>
__forceinline__ __device__ int add_vectors<__half>(int a, int b) {
  return add_vectors_helper<__half2>(a, b);
}

template <>
__forceinline__ __device__ int add_vectors<__bfloat16>(int a, int b) {
  return add_vectors_helper<__bfloat162>(a, b);
}

template <typename T>
__forceinline__ __device__ uint32_t add_vectors_helper(uint32_t a, uint32_t b) {
  return bit_cast<uint32_t, T>(add_elements(bit_cast<T, uint32_t>(a), bit_cast<T, uint32_t>(b)));
}

template <typename T>
__forceinline__ __device__ uint32_t add_vectors(uint32_t a, uint32_t b) {
  return add_vectors_helper<T>(a, b);
}

template <>
__forceinline__ __device__ uint32_t add_vectors<__half>(uint32_t a, uint32_t b) {
  return add_vectors_helper<__half2>(a, b);
}

template <>
__forceinline__ __device__ uint32_t add_vectors<__bfloat16>(uint32_t a, uint32_t b) {
  return add_vectors_helper<__bfloat162>(a, b);
}

template <typename T>
__forceinline__ __device__ void vectorSum(T* dst, T* src, size_t nElem, int blockId, int nBlocks) {
  size_t nInt4 = nElem / 4;
  size_t nLastInts = nElem % 4;
  int4* dst4 = (int4*)dst;
  int4* src4 = (int4*)src;
  for (size_t i = threadIdx.x + blockId * blockDim.x; i < nInt4; i += blockDim.x * nBlocks) {
    dst4[i] = add_vectors<T>(dst4[i], src4[i]);
  }
  if (nLastInts > 0) {
    int* dstLast = ((int*)dst) + nInt4 * 4;
    int* srcLast = ((int*)src) + nInt4 * 4;
    for (size_t i = threadIdx.x + blockId * blockDim.x; i < nLastInts; i += blockDim.x * nBlocks) {
      dstLast[i] = add_vectors<T>(dstLast[i], srcLast[i]);
    }
  }
}

template <typename T>
__forceinline__ __device__ void vectorSum(T* dst, T* src, size_t nElem) {
  vectorSum(dst, src, nElem, blockIdx.x, gridDim.x);
}

template <typename T>
__global__ void __launch_bounds__(32, 1)
    allreduceAllToAll(T* buff, T* scratch, T* resultBuff, mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels,
                      size_t channelDataOffset, size_t channelScratchOffset, int rank, int nRanksPerNode, int worldSize,
                      size_t nelems, uint32_t flag) {
  // This version of allreduce only works for single nodes
  if (worldSize != nRanksPerNode) return;
  if (sizeof(T) == 2) nelems = (nelems * sizeof(T) + sizeof(T)) / sizeof(int);
  const int nPeers = nRanksPerNode - 1;
  const int nBlocksPerPeer = gridDim.x / nPeers;
  const int localBlockIdx = blockIdx.x % nBlocksPerPeer;
  const int tid = threadIdx.x + localBlockIdx * blockDim.x;
  const int peerIdx = blockIdx.x / nBlocksPerPeer;
  size_t srcOffset = channelDataOffset;
  size_t scratchOffset = channelScratchOffset + rank * nelems * sizeof(mscclpp::LL8Packet);
  void* scratchBuff = (void*)((char*)scratch + channelScratchOffset);
  uint32_t* src = (uint32_t*)((char*)buff);
  uint32_t* dst = (uint32_t*)((char*)resultBuff);

  __shared__ mscclpp::DeviceHandle<mscclpp::SmChannel> channels[NRANKS_PER_NODE - 1];
  const int lid = tid % WARP_SIZE;
  if (lid < nPeers) {
    channels[lid] = smChannels[lid];
  }
  __syncwarp();

  // step 1: write data to each peer's scratch buffer
  channels[peerIdx].putPackets<mscclpp::LL8Packet>(scratchOffset, srcOffset, nelems * sizeof(uint32_t), tid,
                                                   blockDim.x * nBlocksPerPeer, flag);

  // step 2: Reduce Data
  for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nelems; idx += blockDim.x * gridDim.x) {
    uint32_t data = 0;
    for (int index = 0; index < nPeers; index++) {
      const int remoteRank = index < rank ? index : index + 1;
      mscclpp::LL8Packet* dstPkt = (mscclpp::LL8Packet*)scratchBuff + remoteRank * nelems;
      uint32_t val = dstPkt[idx].read(flag, -1);
      data = add_vectors<T>(val, data);
    }
    data = add_vectors<T>(data, src[idx]);
    dst[idx] = data;
  }
}

template <typename T>
__global__ void __launch_bounds__(1024, 1)
    allreduce7(T* buff, T* scratch, T* resultBuff, mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels,
               size_t channelDataOffset, size_t channelScratchOffset, int rank, int nRanksPerNode, int worldSize,
               size_t nelems, uint32_t flag) {
  // This version of allreduce only works for single nodes
  if (worldSize != nRanksPerNode) return;

  if (sizeof(T) == 2)
    nelems = (nelems * sizeof(T) + sizeof(T)) / sizeof(int);
  else
    nelems = nelems / (sizeof(int) / sizeof(T));

  const int nPeers = nRanksPerNode - 1;
  const size_t nPkts = nelems / 2;

  int nelemsPerRank = nelems / worldSize;
  if ((nelemsPerRank % 2)) nelemsPerRank = (nelemsPerRank * sizeof(T) + sizeof(T)) / sizeof(T);

  const int nPktsPerRank = nelemsPerRank / 2;
  // thread block & channel info
  const int nBlocksPerPeer = gridDim.x / nPeers;
  const int localBlockIdx = blockIdx.x % nBlocksPerPeer;
  const int peerIdx = blockIdx.x / nBlocksPerPeer;
  const int remoteRank = peerIdx < rank ? peerIdx : peerIdx + 1;
  const int tid = threadIdx.x + localBlockIdx * blockDim.x;
  void* scratchBuff = (void*)((char*)scratch + channelScratchOffset);
  size_t scratchOffset = channelScratchOffset + rank * nPktsPerRank * sizeof(mscclpp::LLPacket);
  size_t scratchResultOffset = channelScratchOffset + 2 * nPkts * sizeof(mscclpp::LLPacket);
  size_t srcOffset = remoteRank * nelemsPerRank * sizeof(int) + channelDataOffset;

  uint2* src = (uint2*)((char*)buff + rank * nelemsPerRank * sizeof(int));
  uint2* dst = (uint2*)((char*)resultBuff + rank * nelemsPerRank * sizeof(int));

  // Put channels into shared memory, read channel info from global memory is unexpectable slow.
  __shared__ mscclpp::DeviceHandle<mscclpp::SmChannel> channels[NRANKS_PER_NODE - 1];
  const int lid = tid % WARP_SIZE;
  if (lid < nPeers) {
    channels[lid] = smChannels[lid];
  }
  __syncwarp();

  // step 1: write to scratch buffer
  channels[peerIdx].putPackets<mscclpp::LLPacket>(scratchOffset, srcOffset, nelemsPerRank * sizeof(int), tid,
                                                  blockDim.x * nBlocksPerPeer, flag);
  // step 2: get data from scratch buffer, reduce data and write result to remote scratch buffer
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nPktsPerRank; idx += blockDim.x * gridDim.x) {
    uint2 data = src[idx];
    for (int index = 0; index < NPEERS; index++) {
      const int remoteRank = index < rank ? index : index + 1;
      mscclpp::LLPacket* dstPkt = (mscclpp::LLPacket*)scratchBuff + remoteRank * nPktsPerRank;
      uint2 val = dstPkt[idx].read(flag);
      data.x = add_vectors<T>(val.x, data.x);
      data.y = add_vectors<T>(val.y, data.y);
    }

    dst[idx].x = data.x;
    dst[idx].y = data.y;

    mscclpp::LLPacket packet;
    packet.data1 = data.x;
    packet.flag1 = flag;
    packet.data2 = data.y;
    packet.flag2 = flag;
    size_t offset = scratchResultOffset / sizeof(mscclpp::LLPacket) + (idx + rank * nPktsPerRank);
    for (int index = 0; index < NPEERS; index++) {
      channels[index].write(offset, packet);
    }
  }
  // step 3: get data result from scratch buffer
  mscclpp::LLPacket* dstPkt = (mscclpp::LLPacket*)((char*)scratch + scratchResultOffset);
  const int dstOffset = remoteRank * nPktsPerRank;
  uint2* result = (uint2*)((char*)resultBuff + remoteRank * nelemsPerRank * sizeof(int));
  for (int idx = threadIdx.x + localBlockIdx * blockDim.x; idx < nPktsPerRank; idx += blockDim.x * nBlocksPerPeer) {
    uint2 data = dstPkt[idx + dstOffset].read(flag, -1);
    result[idx].x = data.x;
    result[idx].y = data.y;
  }
}

template <typename T>
__global__ void __launch_bounds__(512, 1)
    allreduce8(T* buff, T* scratch, T* resultBuff, mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels,
               mscclpp::DeviceHandle<mscclpp::SmChannel>* smOutChannels, size_t channelOutDataOffset,
               size_t channelScratchOffset, int rank, int nRanksPerNode, int worldSize, size_t nelems) {
  const int nPeer = nRanksPerNode - 1;
  const size_t chanOffset = nPeer * blockIdx.x;
  // assume (nelems * sizeof(T)) is divisible by (16 * worldSize)
  const size_t nInt4 = nelems * sizeof(T) / sizeof(int4);
  const size_t nInt4PerRank = nInt4 / worldSize;
  auto smChans = smChannels + chanOffset;
  auto smOutChans = smOutChannels + chanOffset;

  int4* buff4 = reinterpret_cast<int4*>(buff);
  int4* scratch4 = reinterpret_cast<int4*>((char*)scratch + channelScratchOffset);
  int4* resultBuff4 = reinterpret_cast<int4*>(resultBuff);

  // Distribute `nInt4PerRank` across all blocks with the unit size `unitNInt4`
  constexpr size_t unitNInt4 = 512;
  const size_t maxNInt4PerBlock =
      (((nInt4PerRank + gridDim.x - 1) / gridDim.x) + unitNInt4 - 1) / unitNInt4 * unitNInt4;
  size_t offsetOfThisBlock = maxNInt4PerBlock * blockIdx.x;
  size_t nInt4OfThisBlock = maxNInt4PerBlock;
  size_t nNeededBlocks = (nInt4PerRank + maxNInt4PerBlock - 1) / maxNInt4PerBlock;
  constexpr size_t nInt4PerChunk = 1024 * 256 / sizeof(int4);  // 256KB
  if (blockIdx.x >= nNeededBlocks) {
    nInt4OfThisBlock = 0;
  } else if (blockIdx.x == nNeededBlocks - 1) {
    nInt4OfThisBlock = nInt4PerRank - maxNInt4PerBlock * (nNeededBlocks - 1);
  }
  const size_t nItrs = nInt4OfThisBlock / nInt4PerChunk;
  const size_t restNInt4 = nInt4OfThisBlock % nInt4PerChunk;
  const size_t chunkSizePerRank = nNeededBlocks * nInt4PerChunk;
  const size_t blockOffset = nInt4PerChunk * blockIdx.x;
  const size_t scratchChunkRankOffset = chunkSizePerRank * rank;
  const size_t scratchBaseOffsetInt4 = channelScratchOffset / sizeof(int4);

  __shared__ mscclpp::DeviceHandle<mscclpp::SmChannel> channels[NRANKS_PER_NODE - 1];
  __shared__ mscclpp::DeviceHandle<mscclpp::SmChannel> outChannels[NRANKS_PER_NODE - 1];
  const int lid = threadIdx.x % WARP_SIZE;
  if (lid < nPeer) {
    channels[lid] = smChans[lid];
    outChannels[lid] = smOutChans[lid];
  }
  __syncwarp();

  // we can use double buffering to hide synchronization overhead
  for (size_t itr = 0; itr < nItrs; itr++) {
    if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
      outChannels[threadIdx.x].signal();
      outChannels[threadIdx.x].wait();
    }
    __syncthreads();
    // Starts allgather
    for (size_t idx = threadIdx.x; idx < nInt4PerChunk; idx += blockDim.x) {
      for (int i = 0; i < NPEERS; i++) {
        const int peerIdx = (i + blockIdx.x) % nPeer;
        const int remoteRank = (peerIdx < rank) ? peerIdx : peerIdx + 1;
        int4 val = buff4[nInt4PerRank * remoteRank + idx + offsetOfThisBlock];
        channels[peerIdx].write(scratchBaseOffsetInt4 + scratchChunkRankOffset + blockOffset + idx, val);
      }
    }

    /// Starts reduce-scatter
    // Ensure that all writes of this block have been issued before issuing the signal
    __syncthreads();
    if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
      outChannels[threadIdx.x].signal();
      outChannels[threadIdx.x].wait();
    }
    __syncthreads();

    for (size_t idx = threadIdx.x; idx < nInt4PerChunk; idx += blockDim.x) {
      int4 data = buff4[nInt4PerRank * rank + idx + offsetOfThisBlock];
      for (int peerIdx = 0; peerIdx < NPEERS; peerIdx++) {
        const int remoteRank = (peerIdx < rank) ? peerIdx : peerIdx + 1;
        int4 val = scratch4[chunkSizePerRank * remoteRank + blockOffset + idx];
        data = add_vectors<T>(val, data);
      }
      resultBuff4[nInt4PerRank * rank + idx + offsetOfThisBlock] = data;
      for (int peerIdx = 0; peerIdx < NPEERS; peerIdx++) {
        outChannels[peerIdx].write(nInt4PerRank * rank + idx + offsetOfThisBlock + channelOutDataOffset / sizeof(int4),
                                   data);
      }
    }
    offsetOfThisBlock += nInt4PerChunk;
    // Ensure all threads have consumed data from scratch buffer before signaling re-use in next iteration
    __syncthreads();
  }
  if (restNInt4 > 0) {
    if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
      outChannels[threadIdx.x].signal();
      outChannels[threadIdx.x].wait();
    }
    __syncthreads();
    for (size_t idx = threadIdx.x; idx < restNInt4; idx += blockDim.x) {
      for (int i = 0; i < NPEERS; i++) {
        const int peerIdx = (i + blockIdx.x) % nPeer;
        const int remoteRank = (peerIdx < rank) ? peerIdx : peerIdx + 1;
        int4 val = buff4[nInt4PerRank * remoteRank + idx + offsetOfThisBlock];
        channels[peerIdx].write(scratchBaseOffsetInt4 + scratchChunkRankOffset + blockOffset + idx, val);
      }
    }

    // Ensure that all writes of this block have been issued before issuing the signal
    __syncthreads();
    if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
      outChannels[threadIdx.x].signal();
      outChannels[threadIdx.x].wait();
    }
    __syncthreads();

    for (size_t idx = threadIdx.x; idx < restNInt4; idx += blockDim.x) {
      int4 data = buff4[nInt4PerRank * rank + idx + offsetOfThisBlock];
      for (int peerIdx = 0; peerIdx < NPEERS; peerIdx++) {
        const int remoteRank = (peerIdx < rank) ? peerIdx : peerIdx + 1;
        int4 val = scratch4[chunkSizePerRank * remoteRank + blockOffset + idx];
        data = add_vectors<T>(val, data);
      }
      resultBuff4[nInt4PerRank * rank + idx + offsetOfThisBlock] = data;
      for (int peerIdx = 0; peerIdx < NPEERS; peerIdx++) {
        outChannels[peerIdx].write(nInt4PerRank * rank + idx + offsetOfThisBlock + channelOutDataOffset / sizeof(int4),
                                   data);
      }
    }
    // Ensure all threads have issued writes to outChannel
    __syncthreads();
  }
  // Threads are already synchronized
  // So all writes to outChannel have been issued before signal is being issued
  if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
    outChannels[threadIdx.x].signal();
    outChannels[threadIdx.x].wait();
  }
}

template <typename T>
cudaError_t allreduce(T* buff, T* scratch, T* resultBuff, mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels,
                      mscclpp::DeviceHandle<mscclpp::SmChannel>* smOutChannels, size_t channelInOffset,
                      size_t channelOutOffset, size_t channelScratchOffset, int rank, int nRanksPerNode, int worldSize,
                      size_t nelems, cudaStream_t stream) {
  static uint32_t flag = 1;

  if (sizeof(T) * nelems < worldSize * sizeof(int)) {
    int nBlocks = 7;
    int nThreadsPerBlock = 32;
    allreduceAllToAll<<<nBlocks, nThreadsPerBlock, 0, stream>>>(buff, scratch, resultBuff, smChannels, channelInOffset,
                                                                channelScratchOffset, rank, nRanksPerNode, worldSize,
                                                                nelems, flag++);
  } else if (sizeof(T) * nelems <= (1 << 20)) {
    int nBlocks = 28;
    int nThreadsPerBlock = 1024;
    if (nelems >= 8192) {
      nBlocks = 56;
      nThreadsPerBlock = (nelems <= 76800) ? 512 : 1024;
    }
    allreduce7<<<nBlocks, nThreadsPerBlock, 0, stream>>>(buff, scratch, resultBuff, smChannels, channelInOffset,
                                                         channelScratchOffset, rank, nRanksPerNode, worldSize, nelems,
                                                         flag++);
  } else {
    int nBlocks = 35;
    int nThreadsPerBlock = 512;
    allreduce8<<<nBlocks, nThreadsPerBlock, 0, stream>>>(buff, scratch, resultBuff, smChannels, smOutChannels,
                                                         channelOutOffset, channelScratchOffset, rank, nRanksPerNode,
                                                         worldSize, nelems);
  }

  return cudaGetLastError();
}

#endif  // ALLREDUCE_KERNEL_H
