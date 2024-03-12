// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ALLREDUCE_HPP_
#define ALLREDUCE_HPP_

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/gpu.hpp>
#include <mscclpp/packet_device.hpp>
#include <mscclpp/sm_channel_device.hpp>

#include "common.hpp"

extern __device__ mscclpp::DeviceSyncer deviceSyncer;

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
__forceinline__ __device__ T add_elements(T a, T b) {
  return a + b;
}

template <>
__forceinline__ __device__ __half2 add_elements(__half2 a, __half2 b) {
  return __hadd2(a, b);
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
__global__ void __launch_bounds__(1024, 1)
    allreduce6(T* buff, T* scratch, T* resultBuff, mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels, int rank,
               int nRanksPerNode, int worldSize, size_t nelems, uint32_t flag) {
  // This version of allreduce only works for single nodes
  if (worldSize != nRanksPerNode) return;
  nelems = nelems / (sizeof(int) / sizeof(T));
  const int nPeers = nRanksPerNode - 1;
  const int nPkts = nelems / 2;
  const int nelemsPerRank = nelems / worldSize;
  const int nPktsPerRank = nelemsPerRank / 2;
  // thread block & channel info
  const int nBlocksPerPeer = gridDim.x / nPeers;
  const int localBlockIdx = blockIdx.x % nBlocksPerPeer;
  const int peerIdx = blockIdx.x / nBlocksPerPeer;
  const int remoteRank = peerIdx < rank ? peerIdx : peerIdx + 1;
  mscclpp::SmChannelDeviceHandle smChan = smChannels[peerIdx];
  const int tid = threadIdx.x + localBlockIdx * blockDim.x;
  // double buffering
  size_t scratchBaseOffset = (flag & 1) ? 0 : nPkts * sizeof(mscclpp::LLPacket);
  void* scratchBuff = (void*)((char*)scratch + scratchBaseOffset);
  size_t scratchOffset = scratchBaseOffset + rank * nPktsPerRank * sizeof(mscclpp::LLPacket);
  size_t scratchResultOffset =
      (flag & 1) ? 2 * nPkts * sizeof(mscclpp::LLPacket) : 3 * nPkts * sizeof(mscclpp::LLPacket);
  size_t srcOffset = remoteRank * nelemsPerRank * sizeof(int);
  uint2* src = (uint2*)((char*)buff + rank * nelemsPerRank * sizeof(int));
  uint2* dst = (uint2*)((char*)resultBuff + rank * nelemsPerRank * sizeof(int));

  // step 1: write to scratch buffer
  smChan.putPackets(scratchOffset, srcOffset, nelemsPerRank * sizeof(int), tid, blockDim.x * nBlocksPerPeer, flag);
  // step 2: get data from scratch buffer, reduce data and write result to remote scratch buffer
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nPktsPerRank; idx += blockDim.x * gridDim.x) {
    uint2 data = make_uint2(0, 0);
    for (int index = 0; index < nPeers; index++) {
      const int remoteRank = index < rank ? index : index + 1;
      mscclpp::LLPacket* dstPkt = (mscclpp::LLPacket*)scratchBuff + remoteRank * nPktsPerRank;
      uint2 val = dstPkt[idx].read(flag);
      data = add_vectors<T>(val, data);
    }
    data = add_vectors<T>(data, src[idx]);
    dst[idx].x = data.x;
    dst[idx].y = data.y;
    for (int index = 0; index < nPeers; index++) {
      mscclpp::LLPacket* dstPkt = (mscclpp::LLPacket*)((char*)smChannels[index].dst_ + scratchResultOffset);
      dstPkt[idx + rank * nPktsPerRank].write(data.x, data.y, flag);
    }
  }
  // step 3: get data result from scratch buffer
  mscclpp::LLPacket* dstPkt = (mscclpp::LLPacket*)((char*)scratch + scratchResultOffset);
  const int dstOffset = remoteRank * nPktsPerRank;
  uint2* result = (uint2*)((char*)resultBuff + remoteRank * nelemsPerRank * sizeof(int));
  for (int idx = threadIdx.x + localBlockIdx * blockDim.x; idx < nPktsPerRank; idx += blockDim.x * nBlocksPerPeer) {
    uint2 data = dstPkt[idx + dstOffset].read(flag);
    result[idx].x = data.x;
    result[idx].y = data.y;
  }
}

template <typename T>
__global__ void __launch_bounds__(1024, 1)
    allreduce1(T* src, T* dst, mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels,
               mscclpp::DeviceHandle<mscclpp::SmChannel>* smOutChannels, int rank, int nranks, size_t nelems) {
  const size_t chunkSize = nelems / nranks;
  if (nranks == 1) return;
  const int nPeer = nranks - 1;
  const size_t indexOffset = rank * chunkSize;
  const size_t vectorSize = sizeof(int4) / sizeof(T);
  const size_t indexOffset4 = indexOffset / vectorSize;
  int4* src4 = (int4*)src;
  int4* dst4 = (int4*)dst;
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // synchronize everyone
  if (tid == 0) {
    __threadfence_system();
  }
  __syncthreads();
  if (tid < nPeer) {
    smChannels[tid].relaxedSignal();
  }
  if (tid >= nPeer && tid < nPeer * 2) {
    smChannels[tid - nPeer].wait();
  }
  deviceSyncer.sync(gridDim.x);

  // use int4 as much as possible
  const size_t nInt4 = chunkSize / vectorSize;
  for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nInt4; idx += blockDim.x * gridDim.x) {
    int4 tmp = src4[indexOffset4 + idx];
    for (int index = 0; index < nPeer; ++index) {
      int4 val;
      int peerIdx = (index + rank);
      if (peerIdx >= nPeer) peerIdx -= nPeer;
      val = smChannels[peerIdx].read<int4>(indexOffset4 + idx);
      tmp = add_vectors<T>(tmp, val);
    }
    dst4[indexOffset4 + idx] = tmp;
  }

  // use the given TYPE for the rest
  size_t processed = nInt4 * vectorSize * nranks;
  const size_t nRemElems = nelems - processed;
  const size_t startIdx = processed + (nRemElems * rank) / nranks;
  const size_t endIdx = processed + (nRemElems * (rank + 1)) / nranks;
  for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x + startIdx; idx < endIdx; idx += blockDim.x * gridDim.x) {
    T tmp = src[idx];
    for (int index = 0; index < nPeer; ++index) {
      int peerIdx = (index + rank);
      if (peerIdx >= nPeer) peerIdx -= nPeer;
      T val = smChannels[peerIdx].read<T>(idx);
      tmp += val;
    }
    dst[idx] = tmp;
  }

  // synchronize everyone again
  deviceSyncer.sync(gridDim.x);
  if (tid == 0) {
    __threadfence_system();
  }
  __syncthreads();
  if (tid < nPeer) {
    smChannels[tid].relaxedSignal();
  }
  if (tid >= nPeer && tid < nPeer * 2) {
    smChannels[tid - nPeer].wait();
  }

  deviceSyncer.sync(gridDim.x);
  for (int i = 0; i < nPeer; ++i) {
    int peerIdx = (i + rank);
    if (peerIdx >= nPeer) peerIdx -= nPeer;
    const int remoteRank = (peerIdx < rank ? peerIdx : peerIdx + 1);
    size_t offset = chunkSize * remoteRank * sizeof(T);
    smOutChannels[peerIdx].get(offset, chunkSize * sizeof(T), tid, blockDim.x * gridDim.x);
  }
}

template <typename T>
__global__ void __launch_bounds__(1024, 1)
    allreduce7(T* buff, T* scratch, T* resultBuff, mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels, int rank,
               int nRanksPerNode, int worldSize, size_t nelems, uint32_t flag) {
  // This version of allreduce only works for single nodes
  if (worldSize != nRanksPerNode) return;
  nelems = nelems / (sizeof(int) / sizeof(T));
  const int nPeers = nRanksPerNode - 1;
  const size_t nPkts = nelems;
  const int nelemsPerRank = nelems / worldSize;
  const int nPktsPerRank = nelemsPerRank;
  // thread block & channel info
  const int nBlocksPerPeer = gridDim.x / nPeers;
  const int localBlockIdx = blockIdx.x % nBlocksPerPeer;
  const int peerIdx = blockIdx.x / nBlocksPerPeer;
  const int remoteRank = peerIdx < rank ? peerIdx : peerIdx + 1;
  const int tid = threadIdx.x + localBlockIdx * blockDim.x;
  // double buffering
  size_t scratchBaseOffset = (flag & 1) ? 0 : nPkts * sizeof(mscclpp::LL8Packet);
  void* scratchBuff = (void*)((char*)scratch + scratchBaseOffset);
  size_t scratchOffset = scratchBaseOffset + rank * nPktsPerRank * sizeof(mscclpp::LL8Packet);
  size_t scratchResultOffset =
      (flag & 1) ? 2 * nPkts * sizeof(mscclpp::LL8Packet) : 3 * nPkts * sizeof(mscclpp::LL8Packet);
  size_t srcOffset = remoteRank * nelemsPerRank * sizeof(int);
  uint32_t* src = (uint32_t*)((char*)buff + rank * nelemsPerRank * sizeof(int));
  uint32_t* dst = (uint32_t*)((char*)resultBuff + rank * nelemsPerRank * sizeof(int));

  // Put channels into shared memory, read channel info from global memory is unexpectable slow.
  __shared__ mscclpp::DeviceHandle<mscclpp::SmChannel> channels[NRANKS_PER_NODE - 1];
  const int lid = tid % WARP_SIZE;
  if (lid < nPeers) {
    channels[lid] = smChannels[lid];
  }
  __syncwarp();

  // step 1: write to scratch buffer
  channels[peerIdx].putPackets<mscclpp::LL8Packet>(scratchOffset, srcOffset, nelemsPerRank * sizeof(int), tid,
                                                          blockDim.x * nBlocksPerPeer, flag);
  // step 2: get data from scratch buffer, reduce data and write result to remote scratch buffer
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nPktsPerRank; idx += blockDim.x * gridDim.x) {
    uint32_t data = 0;
    for (int index = 0; index < nPeers; index++) {
      const int remoteRank = index < rank ? index : index + 1;
      mscclpp::LL8Packet* dstPkt = (mscclpp::LL8Packet*)scratchBuff + remoteRank * nPktsPerRank;
      uint32_t val = dstPkt[idx].read(flag);
      data = add_vectors<T>(val, data);
    }
    data = add_vectors<T>(data, src[idx]);
    dst[idx] = data;

    mscclpp::LL8Packet packet;
    packet.data = data;
    packet.flag = flag;
    size_t offset = scratchResultOffset / sizeof(mscclpp::LL8Packet) + (idx + rank * nPktsPerRank);
    for (int index = 0; index < nPeers; index++) {
      channels[index].write(offset, packet);
    }
  }
  // step 3: get data result from scratch buffer
  mscclpp::LL8Packet* dstPkt = (mscclpp::LL8Packet*)((char*)scratch + scratchResultOffset);
  const int dstOffset = remoteRank * nPktsPerRank;
  uint32_t* result = (uint32_t*)((char*)resultBuff + remoteRank * nelemsPerRank * sizeof(int));
  for (int idx = threadIdx.x + localBlockIdx * blockDim.x; idx < nPktsPerRank; idx += blockDim.x * nBlocksPerPeer) {
    uint32_t data = dstPkt[idx + dstOffset].read(flag);
    result[idx] = data;
  }
}

template <typename T>
__global__ void __launch_bounds__(1024, 1)
    allreduce8(T* buff, T* scratch, T* resultBuff, mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels,
               mscclpp::DeviceHandle<mscclpp::SmChannel>* smOutChannels, int rank, int nRanksPerNode, int worldSize,
               size_t nelems) {
  const int nPeer = nRanksPerNode - 1;
  const size_t chanOffset = nPeer * blockIdx.x;
  // assume (nelems * sizeof(T)) is divisible by (16 * worldSize)
  const size_t nInt4 = nelems * sizeof(T) / sizeof(int4);
  const size_t nInt4PerRank = nInt4 / worldSize;
  auto smChans = smChannels + chanOffset;
  auto smOutChans = smOutChannels + chanOffset;

  int4* buff4 = reinterpret_cast<int4*>(buff);
  int4* scratch4 = reinterpret_cast<int4*>(scratch);
  int4* resultBuff4 = reinterpret_cast<int4*>(resultBuff);

  // Distribute `nInt4PerRank` across all blocks with the unit size `unitNInt4`
  constexpr size_t unitNInt4 = 1024;
  const size_t maxNInt4PerBlock = (((nInt4PerRank + gridDim.x - 1) / gridDim.x) + unitNInt4 - 1) / unitNInt4 * unitNInt4;
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
      for (int i = 0; i < nPeer; i++) {
        const int peerIdx = (i + blockIdx.x) % nPeer;
        const int remoteRank = (peerIdx < rank) ? peerIdx : peerIdx + 1;
        int4 val = buff4[nInt4PerRank * remoteRank + idx + offsetOfThisBlock];
        channels[peerIdx].write(scratchChunkRankOffset + blockOffset + idx, val);
      }
    }

    /// Starts reduce-scatter
    if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
      outChannels[threadIdx.x].signal();
      outChannels[threadIdx.x].wait();
    }
    __syncthreads();

    for (size_t idx =  threadIdx.x; idx < nInt4PerChunk; idx += blockDim.x) {
      int4 data = buff4[nInt4PerRank * rank + idx + offsetOfThisBlock];
      for (int peerIdx = 0; peerIdx < nPeer; peerIdx++) {
        const int remoteRank = (peerIdx < rank) ? peerIdx : peerIdx + 1;
        int4 val = scratch4[chunkSizePerRank * remoteRank + blockOffset + idx];
        data = add_vectors<T>(val, data);
      }
      resultBuff4[nInt4PerRank * rank + idx + offsetOfThisBlock] = data;
      for (int peerIdx = 0; peerIdx < nPeer; peerIdx++) {
        outChannels[peerIdx].write(nInt4PerRank * rank + idx + offsetOfThisBlock, data);
      }
    }
    offsetOfThisBlock += nInt4PerChunk;
  }
  if (restNInt4 > 0) {
    if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
      outChannels[threadIdx.x].signal();
      outChannels[threadIdx.x].wait();
    }
    __syncthreads();
    for (size_t idx = threadIdx.x; idx < restNInt4; idx += blockDim.x) {
      for (int i = 0; i < nPeer; i++) {
        const int peerIdx = (i + blockIdx.x) % nPeer;
        const int remoteRank = (peerIdx < rank) ? peerIdx : peerIdx + 1;
        int4 val = buff4[nInt4PerRank * remoteRank + idx + offsetOfThisBlock];
        channels[peerIdx].write(scratchChunkRankOffset + blockOffset + idx, val);
      }
    }

    if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
      outChannels[threadIdx.x].signal();
      outChannels[threadIdx.x].wait();
    }
    __syncthreads();

    for (size_t idx = threadIdx.x; idx < restNInt4; idx += blockDim.x) {
      int4 data = buff4[nInt4PerRank * rank + idx + offsetOfThisBlock];
      for (int peerIdx = 0; peerIdx < nPeer; peerIdx++) {
        const int remoteRank = (peerIdx < rank) ? peerIdx : peerIdx + 1;
        int4 val = scratch4[chunkSizePerRank * remoteRank + blockOffset + idx];
        data = add_vectors<T>(val, data);
      }
      resultBuff4[nInt4PerRank * rank + idx + offsetOfThisBlock] = data;
      for (int peerIdx = 0; peerIdx < nPeer; peerIdx++) {
        outChannels[peerIdx].write(nInt4PerRank * rank + idx + offsetOfThisBlock, data);
      }
    }
  }
}

template <typename T>
cudaError_t allreduce(T* buff, T* scratch, T* resultBuff, mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels,
                      mscclpp::DeviceHandle<mscclpp::SmChannel>* smOutChannels, int rank, int nRanksPerNode,
                      int worldSize, size_t nelems, cudaStream_t stream) {
  static uint32_t flag = 1;
#if defined(__HIP_PLATFORM_AMD__)
  if (sizeof(T) * nelems <= (1 << 20)) {
    int nBlocks = 28;
    int nThreadsPerBlock = 1024;
    if (nelems >= 8192) {
      nBlocks = 56;
      nThreadsPerBlock = (nelems <= 76800) ? 512 : 1024;
    }
    allreduce7<<<nBlocks, nThreadsPerBlock, 0, stream>>>(buff, scratch, resultBuff, smChannels, rank, nRanksPerNode,
                                                         worldSize, nelems, flag++);
  } else {
    int nBlocks = 35;
    int nThreadsPerBlock = 512;
    allreduce8<<<nBlocks, nThreadsPerBlock, 0, stream>>>(buff, scratch, resultBuff, smChannels, smOutChannels, rank, nRanksPerNode,
                                                         worldSize, nelems);
  }
#else
  if (sizeof(T) * nelems <= (1 << 20)) {
    allreduce6<<<21, 512, 0, stream>>>(buff, scratch, resultBuff, smChannels, rank, nRanksPerNode, worldSize, nelems,
                                       flag++);
  } else {
    allreduce1<<<24, 1024, 0, stream>>>(buff, resultBuff, smChannels, smOutChannels, rank, worldSize, nelems);
  }
#endif
  return cudaGetLastError();
}

#endif  // ALLREDUCE_KERNEL_H
