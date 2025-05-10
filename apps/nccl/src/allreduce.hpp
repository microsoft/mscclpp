// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ALLREDUCE_HPP_
#define ALLREDUCE_HPP_

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu.hpp>
#include <mscclpp/gpu_data_types.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/memory_channel_device.hpp>
#include <mscclpp/nvls.hpp>
#include <mscclpp/packet_device.hpp>
#include <type_traits>

#if defined(ENABLE_NPKIT)
#include <mscclpp/npkit/npkit.hpp>
#endif

#include "common.hpp"

enum Op {
  SUM = 0,
  MIN = 3,
};

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

template <typename T, bool UseClip = true>
__forceinline__ __device__ T add_elements(T a, T b) {
  if constexpr (UseClip) {
    return clip(a + b);
  } else {
    return a + b;
  }
}

template <bool UseClip = true>
__forceinline__ __device__ __half2 add_elements(__half2 a, __half2 b) {
  if constexpr (UseClip) {
    return clip(__hadd2(a, b));
  } else {
    return __hadd2(a, b);
  }
}

template <bool UseClip = true>
__forceinline__ __device__ __bfloat162 add_elements(__bfloat162 a, __bfloat162 b) {
  if constexpr (UseClip) {
    return clip(__hadd2(a, b));
  } else {
    return __hadd2(a, b);
  }
}

template <typename T>
__forceinline__ __device__ T min_elements(T a, T b) {
  return (a < b ? a : b);
}

template <>
__forceinline__ __device__ __half2 min_elements(__half2 a, __half2 b) {
#if defined(__HIP_PLATFORM_AMD__)
  __half2 val;
  val.x = __hmin(a.x, b.x);
  val.y = __hmin(a.y, b.y);
  return val;
#else
  return __hmin2(a, b);
#endif
}

template <>
__forceinline__ __device__ __bfloat162 min_elements(__bfloat162 a, __bfloat162 b) {
  return __hmin2(a, b);
}

template <typename T, Op OpType>
__forceinline__ __device__ T cal_elements(T a, T b) {
  if constexpr (OpType == SUM) {
    return add_elements(a, b);
  } else if constexpr (OpType == MIN) {
    return min_elements(a, b);
  }
  // Should never reach here
  return a;
}

template <typename T, Op OpType>
__forceinline__ __device__ int4 cal_vectors_helper(int4 a, int4 b) {
  int4 ret;
  ret.w = bit_cast<int, T>(cal_elements<T, OpType>(bit_cast<T, int>(a.w), bit_cast<T, int>(b.w)));
  ret.x = bit_cast<int, T>(cal_elements<T, OpType>(bit_cast<T, int>(a.x), bit_cast<T, int>(b.x)));
  ret.y = bit_cast<int, T>(cal_elements<T, OpType>(bit_cast<T, int>(a.y), bit_cast<T, int>(b.y)));
  ret.z = bit_cast<int, T>(cal_elements<T, OpType>(bit_cast<T, int>(a.z), bit_cast<T, int>(b.z)));
  return ret;
}

template <typename T, Op OpType>
__forceinline__ __device__ uint2 cal_vectors_helper(uint2 a, uint2 b) {
  uint2 ret;
  ret.x = bit_cast<int, T>(cal_elements<T, OpType>(bit_cast<T, int>(a.x), bit_cast<T, int>(b.x)));
  ret.y = bit_cast<int, T>(cal_elements<T, OpType>(bit_cast<T, int>(a.y), bit_cast<T, int>(b.y)));
  return ret;
}

template <typename T, Op OpType>
__forceinline__ __device__ int cal_vectors_helper(int a, int b) {
  return bit_cast<int, T>(cal_elements<T, OpType>(bit_cast<T, int>(a), bit_cast<T, int>(b)));
}

template <typename T, Op OpType, typename DataType>
__forceinline__ __device__ DataType cal_vectors(DataType a, DataType b) {
  using CompType = typename std::conditional_t<std::is_same_v<T, __half>, __half2,
                                               std::conditional_t<std::is_same_v<T, __bfloat16>, __bfloat162, T>>;
  return cal_vectors_helper<CompType, OpType>(a, b);
}

template <typename T>
struct VectorType {
  using type = T;
  using nvls_type = T;
  using nvls_type2 = T;
};

template <>
struct VectorType<__half> {
  using type = __half2;
  using nvls_type = uint4;
  using nvls_type2 = uint1;
};

template <>
struct VectorType<__bfloat16> {
  using type = __bfloat162;
  using nvls_type = uint4;
  using nvls_type2 = uint1;
};

template <>
struct VectorType<float> {
  using type = float;
  using nvls_type = uint4;
  using nvls_type2 = uint1;
};

template <Op OpType, typename T>
__global__ void allreduceAllPairs(T* buff, T* scratch, T* resultBuff,
                                  mscclpp::DeviceHandle<mscclpp::MemoryChannel>* memoryChannels,
                                  size_t channelDataOffset, size_t channelScratchOffset, int rank, int nRanksPerNode,
                                  int worldSize, size_t nelems, uint32_t* deviceFlag, uint32_t numScratchBuff) {
  // This version of allreduce only works for single nodes
  if (worldSize != nRanksPerNode) return;
  if (sizeof(T) == 2) nelems = (nelems * sizeof(T) + sizeof(T)) / sizeof(int);
  const int nPeers = nRanksPerNode - 1;

  uint32_t flag = deviceFlag[blockIdx.x];

  size_t scratchBaseOffset = (flag % numScratchBuff) ? SCRATCH_SIZE / numScratchBuff : 0;
  channelScratchOffset = scratchBaseOffset;

  const int nBlocksPerPeer = gridDim.x / nPeers;
  const int localBlockIdx = blockIdx.x % nBlocksPerPeer;
  const int tid = threadIdx.x + localBlockIdx * blockDim.x;
  const int peerIdx = blockIdx.x / nBlocksPerPeer;
  size_t srcOffset = channelDataOffset;
  size_t scratchOffset = channelScratchOffset + rank * nelems * sizeof(mscclpp::LL8Packet);
  void* scratchBuff = (void*)((char*)scratch + channelScratchOffset);
  uint32_t* src = (uint32_t*)((char*)buff);
  uint32_t* dst = (uint32_t*)((char*)resultBuff);

  // step 1: write data to each peer's scratch buffer
  memoryChannels[peerIdx].putPackets<mscclpp::LL8Packet>(scratchOffset, srcOffset, nelems * sizeof(uint32_t), tid,
                                                         blockDim.x * nBlocksPerPeer, flag);

  // step 2: Reduce Data
  for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nelems; idx += blockDim.x * gridDim.x) {
    uint32_t data = src[idx];
    for (int index = 0; index < nPeers; index++) {
      const int remoteRank = index < rank ? index : index + 1;
      mscclpp::LL8Packet* dstPkt = (mscclpp::LL8Packet*)scratchBuff + remoteRank * nelems;
      uint32_t val = dstPkt[idx].read(flag, -1);
      data = cal_vectors<T, OpType>(val, data);
    }
    dst[idx] = data;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    deviceFlag[blockIdx.x] = deviceFlag[blockIdx.x] + 1;
  }
}

template <Op OpType, typename T>
__global__ void __launch_bounds__(1024, 1)
    allreduce7(T* buff, T* scratch, T* resultBuff, mscclpp::DeviceHandle<mscclpp::MemoryChannel>* memoryChannels,
               size_t channelDataOffset, size_t channelScratchOffset, int rank, int nRanksPerNode, int worldSize,
               size_t nelems, uint32_t* deviceFlag, uint32_t numScratchBuff
#if defined(ENABLE_NPKIT)
               ,
               NpKitEventCollectContext* npKitEventCollectContexts, uint64_t* cpuTimestamp) {
#else
    ) {
#endif
  // This version of allreduce only works for single nodes
  if (worldSize != nRanksPerNode) return;

#if defined(ENABLE_NPKIT)
  extern __shared__ int4 NpkitSharedMem[];
  NpKitEvent* event_buffer = (NpKitEvent*)((char*)NpkitSharedMem);
  uint64_t event_buffer_head = 0;
#if defined(ENABLE_NPKIT_EVENT_KERNEL_ALLREDUCE_ENTRY) && defined(ENABLE_NPKIT_EVENT_KERNEL_ALLREDUCE_EXIT)
  uint64_t npkit_timestamp_entry = 0;
  if (threadIdx.x == 0) {
    npkit_timestamp_entry = NPKIT_GET_GPU_TIMESTAMP();
  }
#endif
#endif
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_CPU)
#if defined(MSCCLPP_DEVICE_HIP)
  NpKit::CollectGpuEventShm(NPKIT_EVENT_TIME_SYNC_CPU, 0, 0,
                            NPKIT_LOAD_CPU_TIMESTAMP_PER_BLOCK(cpuTimestamp, blockIdx.x),
#else
  NpKit::CollectGpuEventShm(NPKIT_EVENT_TIME_SYNC_CPU, 0, 0, *cpuTimestamp,
#endif
                            event_buffer, &event_buffer_head);
#endif
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_GPU)
  NpKit::CollectGpuEventShm(NPKIT_EVENT_TIME_SYNC_GPU, 0, 0, NPKIT_GET_GPU_TIMESTAMP(), event_buffer,
                            &event_buffer_head);
#endif

  if (sizeof(T) == 2)
    nelems = (nelems * sizeof(T) + sizeof(T)) / sizeof(int);
  else
    nelems = nelems / (sizeof(int) / sizeof(T));

  const int nPeers = nRanksPerNode - 1;
  const size_t nPkts = nelems / 2;

  uint32_t flag = (uint32_t)deviceFlag[blockIdx.x];

  size_t scratchBaseOffset = (flag % numScratchBuff) ? SCRATCH_SIZE / numScratchBuff : 0;
  channelScratchOffset = scratchBaseOffset;

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
  __shared__ mscclpp::DeviceHandle<mscclpp::MemoryChannel> channels[NRANKS_PER_NODE - 1];
  const int lid = tid % WARP_SIZE;
  if (lid < nPeers) {
    channels[lid] = memoryChannels[lid];
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
      data.x = cal_vectors<T, OpType>(val.x, data.x);
      data.y = cal_vectors<T, OpType>(val.y, data.y);
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

  __syncthreads();
#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_KERNEL_ALLREDUCE_ENTRY) && \
    defined(ENABLE_NPKIT_EVENT_KERNEL_ALLREDUCE_EXIT)
  NpKit::CollectGpuEventShm(NPKIT_EVENT_KERNEL_ALLREDUCE_ENTRY, 0, 0, npkit_timestamp_entry, event_buffer,
                            &event_buffer_head);
  NpKit::CollectGpuEventShm(NPKIT_EVENT_KERNEL_ALLREDUCE_EXIT, 0, 0, NPKIT_GET_GPU_TIMESTAMP(), event_buffer,
                            &event_buffer_head);
#endif
#if defined(ENABLE_NPKIT)
  NpKit::StoreGpuEventShm(npKitEventCollectContexts, event_buffer, event_buffer_head);
#endif
  if (threadIdx.x == 0) {
    deviceFlag[blockIdx.x] = deviceFlag[blockIdx.x] + 1;
  }
}

template <Op OpType, typename T>
__global__ void __launch_bounds__(512, 1)
    allreduce8(T* buff, T* scratch, T* resultBuff, mscclpp::DeviceHandle<mscclpp::MemoryChannel>* memoryChannels,
               mscclpp::DeviceHandle<mscclpp::MemoryChannel>* memoryOutChannels, size_t channelOutDataOffset,
               size_t channelScratchOffset, int rank, int nRanksPerNode, int worldSize, size_t nelems) {
  const int nPeer = nRanksPerNode - 1;
  const size_t chanOffset = nPeer * blockIdx.x;
  // assume (nelems * sizeof(T)) is divisible by (16 * worldSize)
  const size_t nInt4 = nelems * sizeof(T) / sizeof(int4);
  const size_t nInt4PerRank = nInt4 / worldSize;
  auto memoryChans = memoryChannels + chanOffset;
  auto memoryOutChans = memoryOutChannels + chanOffset;

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

  __shared__ mscclpp::DeviceHandle<mscclpp::MemoryChannel> channels[NRANKS_PER_NODE - 1];
  __shared__ mscclpp::DeviceHandle<mscclpp::MemoryChannel> outChannels[NRANKS_PER_NODE - 1];
  const int lid = threadIdx.x % WARP_SIZE;
  if (lid < nPeer) {
    channels[lid] = memoryChans[lid];
    outChannels[lid] = memoryOutChans[lid];
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

    // Starts reduce-scatter
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
        data = cal_vectors<T, OpType>(val, data);
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
        data = cal_vectors<T, OpType>(val, data);
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

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
template <typename T>
MSCCLPP_DEVICE_INLINE void handleMultiLoadReduceStore(T* src, T* dst, uint32_t srcOffset, uint32_t dstOffset,
                                                      size_t size, int tid, int nThreads) {
  using vectorType = typename VectorType<T>::type;
  using nvlsType = typename VectorType<T>::nvls_type;
  // nvls can only handle 4 bytes alignment
  assert(size % sizeof(vectorType) == 0);
  const size_t nInt4 = size / sizeof(nvlsType);
  const size_t srcOffset4 = srcOffset / sizeof(nvlsType);
  const size_t dstOffset4 = dstOffset / sizeof(nvlsType);
  nvlsType* src4 = (nvlsType*)src;
  nvlsType* dst4 = (nvlsType*)dst;
  for (size_t idx = tid; idx < nInt4; idx += nThreads) {
    nvlsType val;
    mscclpp::DeviceMulticastPointerDeviceHandle::multimemLoadReduce(val, (vectorType*)(src4 + srcOffset4 + idx));
    mscclpp::DeviceMulticastPointerDeviceHandle::multimemStore(val, (vectorType*)(dst4 + dstOffset4 + idx));
  }
  // handle rest of data
  size_t processed = nInt4 * sizeof(nvlsType);
  using nvlsType2 = typename VectorType<T>::nvls_type2;
  const size_t startIdx = (srcOffset + processed) / sizeof(nvlsType2);
  const size_t endIdx = (dstOffset + size) / sizeof(nvlsType2);
  for (size_t idx = tid + startIdx; idx < endIdx; idx += nThreads) {
    nvlsType2 val;
    mscclpp::DeviceMulticastPointerDeviceHandle::multimemLoadReduce(val, (vectorType*)src + idx);
    mscclpp::DeviceMulticastPointerDeviceHandle::multimemStore(val, (vectorType*)dst + idx);
  }
}
#endif

template <typename T>
__global__ void __launch_bounds__(1024, 1)
    allreduce9([[maybe_unused]] mscclpp::DeviceHandle<mscclpp::MemoryChannel>* memoryChannels,
               [[maybe_unused]] mscclpp::DeviceHandle<mscclpp::NvlsConnection::DeviceMulticastPointer>* multicast,
               [[maybe_unused]] mscclpp::DeviceHandle<mscclpp::NvlsConnection::DeviceMulticastPointer>* multicastOut,
               [[maybe_unused]] size_t channelInOffset, [[maybe_unused]] size_t channelOutOffset,
               [[maybe_unused]] size_t size, [[maybe_unused]] int rank) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  int nBlocks = gridDim.x;
  int bid = blockIdx.x;
  size_t sizePerRank = size / 8;
  size_t sizePerBlock = sizePerRank / nBlocks;
  size_t rankOffset = sizePerRank * rank;
  size_t blockOffset = sizePerBlock * bid + rankOffset;
  mscclpp::DeviceHandle<mscclpp::NvlsConnection::DeviceMulticastPointer>* multicastPtr = multicast + bid;
  mscclpp::DeviceHandle<mscclpp::NvlsConnection::DeviceMulticastPointer>* multicastOutPtr = multicastOut + bid;

  const size_t chanOffset = (NRANKS_PER_NODE - 1) * blockIdx.x;
  auto memoryChans = memoryChannels + chanOffset;
  __shared__ mscclpp::DeviceHandle<mscclpp::MemoryChannel> channels[NRANKS_PER_NODE - 1];
  const int lid = threadIdx.x % WARP_SIZE;
  if (lid < NRANKS_PER_NODE - 1) {
    channels[lid] = memoryChans[lid];
  }
  __syncwarp();
  if (threadIdx.x < NPEERS) {
    channels[threadIdx.x].relaxedSignal();
    channels[threadIdx.x].relaxedWait();
  }
  __syncthreads();
  T* src = (T*)multicastPtr->mcPtr;
  T* dst = (T*)multicastOutPtr->mcPtr;
  handleMultiLoadReduceStore(src, dst, blockOffset + channelInOffset, blockOffset + channelOutOffset, sizePerBlock,
                             threadIdx.x, blockDim.x);
  __syncthreads();
  if (threadIdx.x < NPEERS) {
    channels[threadIdx.x].relaxedSignal();
    channels[threadIdx.x].relaxedWait();
  }
#endif
}

template <typename T>
__global__ void __launch_bounds__(1024, 1)
    allreduce10([[maybe_unused]] const void* src, [[maybe_unused]] void* scratch, [[maybe_unused]] void* dst,
                [[maybe_unused]] mscclpp::DeviceHandle<mscclpp::MemoryChannel>* memoryChannels,
                [[maybe_unused]] mscclpp::DeviceHandle<mscclpp::NvlsConnection::DeviceMulticastPointer>* multicast,
                [[maybe_unused]] size_t size, [[maybe_unused]] int rank) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  int nBlocks = gridDim.x;
  int nBlocksPerNvlsConn = nBlocks / NUM_NVLS_CONNECTION;
  int bid = blockIdx.x;
  size_t sizePerRank = size / NRANKS_PER_NODE;
  constexpr size_t scratchSizePerRank = SCRATCH_SIZE / NRANKS_PER_NODE;
  const size_t maxSizePerBlock = (sizePerRank + nBlocks - 1) / nBlocks;
  size_t start = bid * maxSizePerBlock;
  size_t end = min(start + maxSizePerBlock, sizePerRank);
  size_t sizePerBlock = end - start;
  mscclpp::DeviceHandle<mscclpp::NvlsConnection::DeviceMulticastPointer>* multicastPtr =
      multicast + bid / nBlocksPerNvlsConn;
  size_t copyPerIter = 1024 * 16;
  if (sizePerBlock >= 1024 * 64) {
    copyPerIter = 1024 * 32;
  }
  size_t scratchSizePerBlock = (scratchSizePerRank / nBlocks) / copyPerIter * copyPerIter;
  size_t blockScratchOffset = scratchSizePerBlock * bid + scratchSizePerRank * rank;
  constexpr int NCOPY_WARPS = 14;
  constexpr int NREDUCE_WARPS = 4;
  constexpr int NRECV_COPY_WARPS = 14;
  constexpr int endCopyWid = NCOPY_WARPS;
  constexpr int startRecvCopyWid = NCOPY_WARPS;
  constexpr int endRecvCopyWid = NCOPY_WARPS + NRECV_COPY_WARPS;
  constexpr int endReduceWid = NCOPY_WARPS + NREDUCE_WARPS + NRECV_COPY_WARPS;
  const int warpId = threadIdx.x / WARP_SIZE;
  size_t nIter = sizePerBlock / copyPerIter;
  size_t lastIterSize = copyPerIter;
  if (sizePerBlock % copyPerIter != 0) {
    nIter += 1;
    lastIterSize = sizePerBlock % copyPerIter;
  }

  const size_t chanOffset = (NRANKS_PER_NODE - 1) * blockIdx.x * 2;
  auto memoryChans = memoryChannels + chanOffset;
  __shared__ mscclpp::DeviceHandle<mscclpp::MemoryChannel> channels[(NRANKS_PER_NODE - 1) * 2];
  const int lid = threadIdx.x % WARP_SIZE;
  if (lid < (NRANKS_PER_NODE - 1) * 2) {
    channels[lid] = memoryChans[lid];
  }
  __syncwarp();
  for (int it = 0; it < nIter; it++) {
    const size_t iterSize = (it == nIter - 1) ? lastIterSize : copyPerIter;
    if (warpId < endCopyWid) {
      int tidInCopy = threadIdx.x;
      for (int i = 0; i < NRANKS_PER_NODE; i++) {
        size_t offset = i * sizePerRank + sizePerBlock * bid + it * copyPerIter;
        size_t offsetScratch =
            i * scratchSizePerRank + scratchSizePerBlock * bid + (it * copyPerIter) % scratchSizePerBlock;
        char* srcData = (char*)src + offset;
        char* dstData = (char*)scratch + offsetScratch;
        mscclpp::copy(dstData, srcData, iterSize, tidInCopy, NCOPY_WARPS * WARP_SIZE);
      }
      asm volatile("bar.sync %0, %1;" ::"r"(0), "r"(NCOPY_WARPS * WARP_SIZE) : "memory");
      if (tidInCopy < NPEERS) {
        channels[tidInCopy].signal();
        channels[tidInCopy].wait();
      }
      asm volatile("bar.sync %0, %1;" ::"r"(1), "r"((NCOPY_WARPS + NREDUCE_WARPS) * WARP_SIZE) : "memory");
    }
    if (warpId >= endRecvCopyWid && warpId < endReduceWid) {
      int tidInReduce = threadIdx.x - endRecvCopyWid * WARP_SIZE;
      asm volatile("bar.sync %0, %1;" ::"r"(1), "r"((NCOPY_WARPS + NREDUCE_WARPS) * WARP_SIZE) : "memory");
      T* mcBuff = (T*)multicastPtr->mcPtr;
      size_t offset = blockScratchOffset + (it * copyPerIter) % scratchSizePerBlock;
      handleMultiLoadReduceStore(mcBuff, mcBuff, offset, offset, iterSize, tidInReduce, NREDUCE_WARPS * WARP_SIZE);
      asm volatile("bar.sync %0, %1;" ::"r"(2), "r"((NRECV_COPY_WARPS + NREDUCE_WARPS) * WARP_SIZE) : "memory");
    }
    if (warpId >= startRecvCopyWid && warpId < endRecvCopyWid) {
      int tidInRecvCopy = threadIdx.x - startRecvCopyWid * WARP_SIZE;
      asm volatile("bar.sync %0, %1;" ::"r"(2), "r"((NRECV_COPY_WARPS + NREDUCE_WARPS) * WARP_SIZE) : "memory");
      if (tidInRecvCopy < NPEERS) {
        channels[tidInRecvCopy + NPEERS].signal();
        channels[tidInRecvCopy + NPEERS].wait();
      }
      asm volatile("bar.sync %0, %1;" ::"r"(3), "r"((NRECV_COPY_WARPS)*WARP_SIZE) : "memory");
      for (int i = 0; i < NRANKS_PER_NODE; i++) {
        size_t offset = i * sizePerRank + sizePerBlock * bid + it * copyPerIter;
        size_t offsetScratch =
            i * scratchSizePerRank + scratchSizePerBlock * bid + (it * copyPerIter) % scratchSizePerBlock;
        char* srcData = (char*)scratch + offsetScratch;
        char* dstData = (char*)dst + offset;
        mscclpp::copy(dstData, srcData, iterSize, tidInRecvCopy, NRECV_COPY_WARPS * WARP_SIZE);
      }
    }
  }
#endif
}

template <typename T>
__global__ void __launch_bounds__(1024, 1)
    allreduce11([[maybe_unused]] const void* src, [[maybe_unused]] void* scratch, [[maybe_unused]] void* dst,
                [[maybe_unused]] mscclpp::DeviceHandle<mscclpp::MemoryChannel>* memoryChannels,
                [[maybe_unused]] mscclpp::DeviceHandle<mscclpp::NvlsConnection::DeviceMulticastPointer>* multicast,
                [[maybe_unused]] size_t size, [[maybe_unused]] int rank) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  constexpr int nBlocksForCopy = 16;
  constexpr int nBlocksForReduce = 8;
  constexpr int copyDeviceReduceRatio = nBlocksForCopy / nBlocksForReduce;
  constexpr size_t scratchSizePerRank = SCRATCH_SIZE / NRANKS_PER_NODE;
  size_t sizePerRank = size / NRANKS_PER_NODE;
  const size_t sizePerBlock = sizePerRank / nBlocksForCopy / sizeof(T) * sizeof(T);
  const int unitSize = 1 << 17;
  int nIter = sizePerBlock / unitSize;
  int bid = blockIdx.x;
  size_t lastIterSize = unitSize;
  if (sizePerBlock % unitSize != 0) {
    nIter += 1;
    lastIterSize = sizePerBlock % unitSize;
  }
  size_t scratchSizePerBlock = (scratchSizePerRank / nBlocksForCopy) / unitSize * unitSize;
  size_t maxItersForScatch = scratchSizePerBlock / unitSize;
  if (bid < nBlocksForCopy && threadIdx.x == 0) {
    deviceSemaphore[bid + 2 * nBlocksForCopy].set(maxItersForScatch);
  }
  for (int it = 0; it < nIter; it++) {
    const size_t iterSize = (it == nIter - 1) ? lastIterSize : unitSize;
    const uint32_t scratchIt = it % maxItersForScatch;
    if (bid < nBlocksForCopy) {
      if (threadIdx.x == 0) {
        deviceSemaphore[bid + 2 * nBlocksForCopy].acquire();
      }
      __syncthreads();
      for (int i = 0; i < NRANKS_PER_NODE; i++) {
        uint32_t blockOffset = it * unitSize + bid * sizePerBlock + i * sizePerRank;
        uint32_t scratchOffset = scratchIt * unitSize + bid * scratchSizePerBlock + i * scratchSizePerRank;
        char* srcData = (char*)src + blockOffset;
        char* dstData = (char*)scratch + scratchOffset;
        mscclpp::copy(dstData, srcData, iterSize, threadIdx.x, blockDim.x);
      }
      __syncthreads();
      if (threadIdx.x < NPEERS) {
        int chanId = bid * NPEERS + threadIdx.x;
        mscclpp::DeviceHandle<mscclpp::MemoryChannel>* channels = memoryChannels + chanId;
        channels->signal();
        channels->wait();
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        deviceSemaphore[bid].release();
      }
    }
    if (bid >= nBlocksForCopy && bid < nBlocksForCopy + nBlocksForReduce) {
      int bidForReduce = bid - nBlocksForCopy;
      mscclpp::DeviceHandle<mscclpp::NvlsConnection::DeviceMulticastPointer>* multicastPtr = multicast + bidForReduce;
      int tid = threadIdx.x;
      T* mcBuff = (T*)multicastPtr->mcPtr;
      for (int i = 0; i < copyDeviceReduceRatio; i++) {
        size_t offset = rank * scratchSizePerRank + scratchIt * unitSize +
                        (bidForReduce * copyDeviceReduceRatio + i) * scratchSizePerBlock;
        if (tid == 0) {
          deviceSemaphore[bidForReduce * copyDeviceReduceRatio + i].acquire();
        }
        __syncthreads();
        handleMultiLoadReduceStore(mcBuff, mcBuff, offset, offset, iterSize, tid, blockDim.x);
        __syncthreads();
        if (tid == 0) {
          deviceSemaphore[nBlocksForCopy + bidForReduce * copyDeviceReduceRatio + i].release();
        }
      }
    }
    if (bid >= nBlocksForCopy + nBlocksForReduce) {
      int bidForCopy = bid - nBlocksForCopy - nBlocksForReduce;
      if (threadIdx.x == 0) {
        deviceSemaphore[bid - nBlocksForReduce].acquire();
      }
      __syncthreads();
      if (threadIdx.x < NPEERS) {
        int chanId = (bid - nBlocksForReduce) * NPEERS + threadIdx.x;
        mscclpp::DeviceHandle<mscclpp::MemoryChannel>* channels = memoryChannels + chanId;
        channels->signal();
        channels->wait();
      }
      __syncthreads();
      for (int i = 0; i < NRANKS_PER_NODE; i++) {
        uint32_t blockOffset =
            it * unitSize + (bid - nBlocksForCopy - nBlocksForReduce) * sizePerBlock + i * sizePerRank;
        uint32_t scratchOffset = scratchIt * unitSize +
                                 (bid - nBlocksForCopy - nBlocksForReduce) * scratchSizePerBlock +
                                 i * scratchSizePerRank;
        char* srcData = (char*)scratch + scratchOffset;
        char* dstData = (char*)dst + blockOffset;
        mscclpp::copy(dstData, srcData, iterSize, threadIdx.x, blockDim.x);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        deviceSemaphore[bidForCopy + 2 * nBlocksForCopy].release();
      }
    }
  }
  if (bid < nBlocksForCopy && threadIdx.x == 0) {
    deviceSemaphore[bid + 2 * nBlocksForCopy].set(0);
  }
#endif
}

template <Op OpType, typename T>
cudaError_t allreduce(const void* buff, void* scratch, void* resultBuff,
                      mscclpp::DeviceHandle<mscclpp::MemoryChannel>* memoryChannels,
                      mscclpp::DeviceHandle<mscclpp::MemoryChannel>* memoryOutChannels,
                      mscclpp::DeviceHandle<mscclpp::NvlsConnection::DeviceMulticastPointer>* nvlsChannels,
                      mscclpp::DeviceHandle<mscclpp::NvlsConnection::DeviceMulticastPointer>* nvlsOutChannels,
                      size_t channelInOffset, size_t channelOutOffset, size_t channelScratchOffset, int rank,
                      int nRanksPerNode, int worldSize, size_t nelems, cudaStream_t stream, uint32_t* deviceFlag7,
                      uint32_t* deviceFlag28, uint32_t* deviceFlag56, uint32_t numScratchBuff) {
  bool useNvlsWithZeroCopy = mscclpp::isNvlsSupported() && !mscclppDisableChannelCache;

  if (sizeof(T) * nelems < worldSize * sizeof(int)) {
    int nBlocks = 7;
    int nThreadsPerBlock = 32;
    allreduceAllPairs<OpType><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
        (T*)buff, (T*)scratch, (T*)resultBuff, memoryChannels, channelInOffset, channelScratchOffset, rank,
        nRanksPerNode, worldSize, nelems, deviceFlag7, numScratchBuff);
  } else if (sizeof(T) * nelems <= (1 << 14)) {
    int nBlocks = 28;
    int nThreadsPerBlock = 512;
    allreduceAllPairs<OpType><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
        (T*)buff, (T*)scratch, (T*)resultBuff, memoryChannels, channelInOffset, channelScratchOffset, rank,
        nRanksPerNode, worldSize, nelems, deviceFlag28, numScratchBuff);
  } else if (sizeof(T) * nelems <= (1 << 16) || (sizeof(T) * nelems <= (1 << 20) && !useNvlsWithZeroCopy)) {
    int nBlocks = 28;
    int nThreadsPerBlock = 1024;
    uint32_t* deviceFlag = deviceFlag28;
    if (nelems >= 8192) {
      nBlocks = 56;
      nThreadsPerBlock = (nelems <= 76800) ? 512 : 1024;
      deviceFlag = deviceFlag56;
    }
#if defined(ENABLE_NPKIT)
    size_t NpkitSharedMemSize = NPKIT_SHM_NUM_EVENTS * sizeof(NpKitEvent);
    allreduce7<OpType><<<nBlocks, nThreadsPerBlock, NpkitSharedMemSize, stream>>>(
        (T*)buff, (T*)scratch, (T*)resultBuff, memoryChannels, channelInOffset, channelScratchOffset, rank,
        nRanksPerNode, worldSize, nelems, deviceFlag, numScratchBuff, NpKit::GetGpuEventCollectContexts(),
        NpKit::GetCpuTimestamp());
#else
    allreduce7<OpType><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
        (T*)buff, (T*)scratch, (T*)resultBuff, memoryChannels, channelInOffset, channelScratchOffset, rank,
        nRanksPerNode, worldSize, nelems, deviceFlag, numScratchBuff);
#endif
  } else if (useNvlsWithZeroCopy) {
    int nBlocks = 8;
    int nThreadsPerBlock = 1024;
    allreduce9<T><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
        memoryChannels, nvlsChannels, nvlsOutChannels, channelInOffset, channelOutOffset, nelems * sizeof(T), rank);
  } else if (mscclpp::isNvlsSupported()) {
    int nBlocks = 40;
    int nThreadsPerBlock = 1024;
    allreduce11<T><<<nBlocks, nThreadsPerBlock, 0, stream>>>(buff, scratch, resultBuff, memoryChannels, nvlsChannels,
                                                             nelems * sizeof(T), rank);
  } else {
    int nBlocks = 35;
    int nThreadsPerBlock = 512;
    allreduce8<OpType><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
        (T*)buff, (T*)scratch, (T*)resultBuff, memoryChannels, memoryOutChannels, channelOutOffset,
        channelScratchOffset, rank, nRanksPerNode, worldSize, nelems);
  }

  return cudaGetLastError();
}

#endif  // ALLREDUCE_KERNEL_H
