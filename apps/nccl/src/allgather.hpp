// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ALLGATHER_HPP_
#define ALLGATHER_HPP_

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/memory_channel_device.hpp>

#include "common.hpp"

template <bool IsOutOfPlace>
__global__ void __launch_bounds__(1024, 1)
    allgather6(void* sendbuff, mscclpp::DeviceHandle<mscclpp::MemoryChannel>* memoryChannels, size_t channelOutOffset,
               size_t rank, [[maybe_unused]] size_t worldSize, size_t nRanksPerNode, size_t nelemsPerGPU) {
  const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t lid = tid % WARP_SIZE;
  const size_t wid = tid / WARP_SIZE;

  const size_t nThread = blockDim.x * gridDim.x;
  const size_t nWarp = nThread / WARP_SIZE;
  const size_t nPeer = nRanksPerNode - 1;
  const size_t chanOffset = nPeer * blockIdx.x;
  auto memChans = memoryChannels + chanOffset;

  if (threadIdx.x < nPeer) {
    memChans[threadIdx.x].relaxedSignal();
    memChans[threadIdx.x].wait();
  }
  __syncthreads();

  const size_t bytesPerGPU = nelemsPerGPU * sizeof(int);
  const size_t bytes = bytesPerGPU * nPeer;
  size_t unitBytesPerThread;
  if (bytes >= nThread * 64) {
    unitBytesPerThread = 64;
  } else {
    unitBytesPerThread = 16;
  }
  const size_t unitBytesPerWarp = unitBytesPerThread * WARP_SIZE;
  const size_t unitBytes = unitBytesPerWarp * nWarp;
  const size_t nLoop = bytes / unitBytes;

  if (nLoop > 0) {
    // First loop unrolling
    const size_t peerIdx = wid % nPeer;
    const size_t offset = bytesPerGPU * rank + (wid / nPeer) * unitBytesPerWarp;
    if constexpr (IsOutOfPlace) {
      char* dst = reinterpret_cast<char*>(memChans[peerIdx].dst_);
      char* src = reinterpret_cast<char*>(memChans[peerIdx].src_);
      char* buff = reinterpret_cast<char*>(sendbuff);
      const size_t offsetWithinRank = (wid / nPeer) * unitBytesPerWarp;
      memChans[peerIdx].copy<16, false>(src + offset + channelOutOffset, buff + offsetWithinRank, unitBytesPerWarp, lid,
                                        WARP_SIZE);
      memChans[peerIdx].copy<16, false>(dst + offset + channelOutOffset, buff + offsetWithinRank, unitBytesPerWarp, lid,
                                        WARP_SIZE);
    } else {
      memChans[peerIdx].put<16, false>(offset + channelOutOffset, unitBytesPerWarp, lid, WARP_SIZE);
    }
  }

  for (size_t i = 1; i < nLoop; ++i) {
    const size_t gWid = wid + i * nWarp;
    const size_t peerIdx = gWid % nPeer;
    const size_t offset = bytesPerGPU * rank + (gWid / nPeer) * unitBytesPerWarp;
    if constexpr (IsOutOfPlace) {
      char* dst = reinterpret_cast<char*>(memChans[peerIdx].dst_);
      char* src = reinterpret_cast<char*>(memChans[peerIdx].src_);
      char* buff = reinterpret_cast<char*>(sendbuff);
      const size_t offsetWithinRank = (gWid / nPeer) * unitBytesPerWarp;
      memChans[peerIdx].copy<16, false>(src + offset + channelOutOffset, buff + offsetWithinRank, unitBytesPerWarp, lid,
                                        WARP_SIZE);
      memChans[peerIdx].copy<16, false>(dst + offset + channelOutOffset, buff + offsetWithinRank, unitBytesPerWarp, lid,
                                        WARP_SIZE);
    } else {
      memChans[peerIdx].put<16, false>(offset + channelOutOffset, unitBytesPerWarp, lid, WARP_SIZE);
    }
  }

  if (bytes % unitBytes > 0) {
    const size_t gWid = wid + nLoop * nWarp;
    const size_t peerIdx = gWid % nPeer;
    const size_t offsetWithinRank = (gWid / nPeer) * unitBytesPerWarp;
    const size_t offset = bytesPerGPU * rank + offsetWithinRank;
    const size_t remainBytes = (offsetWithinRank + unitBytesPerWarp > bytesPerGPU)
                                   ? ((bytesPerGPU > offsetWithinRank) ? (bytesPerGPU - offsetWithinRank) : 0)
                                   : unitBytesPerWarp;
    if (remainBytes > 0) {
      if constexpr (IsOutOfPlace) {
        char* dst = reinterpret_cast<char*>(memChans[peerIdx].dst_);
        char* src = reinterpret_cast<char*>(memChans[peerIdx].src_);
        char* buff = reinterpret_cast<char*>(sendbuff);
        memChans[peerIdx].copy<16, true>(src + offset + channelOutOffset, buff + offsetWithinRank, remainBytes, lid,
                                         WARP_SIZE);
        memChans[peerIdx].copy<16, true>(dst + offset + channelOutOffset, buff + offsetWithinRank, remainBytes, lid,
                                         WARP_SIZE);
      } else {
        memChans[peerIdx].put<16, true>(offset + channelOutOffset, remainBytes, lid, WARP_SIZE);
      }
    }
  }

  deviceSyncer.sync(gridDim.x);

  if (threadIdx.x < nPeer) {
    memChans[threadIdx.x].relaxedSignal();
    memChans[threadIdx.x].wait();
  }
}

template <bool IsOutOfPlace>
__global__ void __launch_bounds__(1024, 1)
    allgather8(void* buff, void* scratch, void* resultBuff, mscclpp::DeviceHandle<mscclpp::MemoryChannel>* memoryChannels,
               size_t channelInputOffset, int rank, int nRanksPerNode, int worldSize, size_t nelems) {
  const int nPeer = nRanksPerNode - 1;
  const size_t chanOffset = nPeer * blockIdx.x;
  // assume (nelems * sizeof(T)) is divisible by 16
  const size_t nInt4 = nelems * sizeof(int) / sizeof(int4);
  auto memoryChans = memoryChannels + chanOffset;

  int4* buff4 = reinterpret_cast<int4*>(buff);
  int4* scratch4 = reinterpret_cast<int4*>(scratch);
  int4* resultBuff4 = reinterpret_cast<int4*>(resultBuff);


  // Distribute `nInt4` across all blocks with the unit size `unitNInt4`
  constexpr size_t unitNInt4 = 1024;
  const size_t maxNInt4PerBlock = (((nInt4 + gridDim.x - 1) / gridDim.x) + unitNInt4 - 1) / unitNInt4 * unitNInt4;
  size_t offsetOfThisBlock = maxNInt4PerBlock * blockIdx.x;
  size_t nInt4OfThisBlock = maxNInt4PerBlock;
  size_t nNeededBlocks = (nInt4 + maxNInt4PerBlock - 1) / maxNInt4PerBlock;
  constexpr size_t nInt4PerChunk = 1024 * 256 / sizeof(int4);  // 256KB
  if (blockIdx.x >= nNeededBlocks) {
    nInt4OfThisBlock = 0;
  } else if (blockIdx.x == nNeededBlocks - 1) {
    nInt4OfThisBlock = nInt4 - maxNInt4PerBlock * (nNeededBlocks - 1);
  }
  const size_t nItrs = nInt4OfThisBlock / nInt4PerChunk;
  const size_t restNInt4 = nInt4OfThisBlock % nInt4PerChunk;
  const size_t chunkSizePerRank = nNeededBlocks * nInt4PerChunk;
  const size_t blockOffset = nInt4PerChunk * blockIdx.x;
  const size_t scratchChunkRankOffset = chunkSizePerRank * rank;

  __shared__ mscclpp::DeviceHandle<mscclpp::MemoryChannel> channels[NRANKS_PER_NODE - 1];
  const int lid = threadIdx.x % WARP_SIZE;
  if (lid < nPeer) {
    channels[lid] = memoryChans[lid];
  }
  __syncwarp();
  // we can use double buffering to hide synchronization overhead
  for (size_t itr = 0; itr < nItrs; itr++) {
    if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
      channels[threadIdx.x].signal();
      channels[threadIdx.x].wait();
    }
    __syncthreads();
    // Starts allgather
    for (int i = 0; i < NPEERS; i++) {
    for (size_t idx = threadIdx.x; idx < nInt4PerChunk; idx += blockDim.x) {
      int4 val = buff4[idx + offsetOfThisBlock];
        const int peerIdx = (i + rank) % nPeer;
        const int remoteRank = (peerIdx < rank) ? peerIdx : peerIdx + 1;
        channels[peerIdx].write(scratchChunkRankOffset + blockOffset + idx, val);
      }
    }
    // Ensure that all writes of this block have been issued before issuing the signal
    __syncthreads();
    if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
      channels[threadIdx.x].signal();
      channels[threadIdx.x].wait();
    }
    __syncthreads();
    for (int peerIdx = 0; peerIdx < NPEERS; peerIdx++) {
      const int remoteRank = (peerIdx < rank) ? peerIdx : peerIdx + 1;
      for (size_t idx = threadIdx.x; idx < nInt4PerChunk; idx += blockDim.x) {
        int4 val = scratch4[chunkSizePerRank * remoteRank + blockOffset + idx];
        resultBuff4[nInt4 * remoteRank + idx + offsetOfThisBlock] = val;
      }
    }
  }

  // if (restNInt4 > 0) {
  //   if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
  //     channels[threadIdx.x].signal();
  //     channels[threadIdx.x].wait();
  //   }
  //   __syncthreads();
  //   for (size_t idx = threadIdx.x; idx < restNInt4; idx += blockDim.x) {
  //     for (int i = 0; i < NPEERS; i++) {
  //       const int peerIdx = (i + blockIdx.x) % nPeer;
  //       const int remoteRank = (peerIdx < rank) ? peerIdx : peerIdx + 1;
  //       int4 val = buff4[nInt4PerRank * remoteRank + idx + offsetOfThisBlock];
  //       channels[peerIdx].write(scratchBaseOffsetInt4 + scratchChunkRankOffset + blockOffset + idx, val);
  //     }
  //   }

  //   // Ensure that all writes of this block have been issued before issuing the signal
  //   __syncthreads();
  //   if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
  //     channels[threadIdx.x].signal();
  //     channels[threadIdx.x].wait();
  //   }
  //   __syncthreads();

  //   for (size_t idx = threadIdx.x; idx < restNInt4; idx += blockDim.x) {
  //     int4 data = buff4[nInt4PerRank * rank + idx + offsetOfThisBlock];
  //     resultBuff4[nInt4PerRank * rank + idx + offsetOfThisBlock] = data;
  //   }
  //   // Ensure all threads have issued writes to outChannel
  //   __syncthreads();
  // }
  // // Threads are already synchronized
  // // So all writes to outChannel have been issued before signal is being issued
  // if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
  //   channels[threadIdx.x].signal();
  //   channels[threadIdx.x].wait();
  // }
}

template <bool IsOutOfPlace, typename T>
cudaError_t allgather(T* buff, T* scratch, T* resultBuff, mscclpp::DeviceHandle<mscclpp::MemoryChannel>* memoryChannels,
                      size_t channelInOffset, size_t channelOutOffset, int rank, int nRanksPerNode, int worldSize,
                      size_t nelems, cudaStream_t stream) {
  int nBlocks = 28;
  if (nelems * sizeof(T) <= 64 * (1 << 20)) {
    if (nelems <= 4096) {
      nBlocks = 7;
    } else if (nelems <= 32768) {
      nBlocks = 14;
    } else if (nelems >= 2097152) {
      nBlocks = 35;
    }
    allgather6<IsOutOfPlace><<<nBlocks, 1024, 0, stream>>>((void*)buff, memoryChannels, channelOutOffset, rank,
                                                           worldSize, nRanksPerNode, nelems * sizeof(T) / sizeof(int));
  } else {
    nBlocks = 64;
    allgather8<IsOutOfPlace><<<nBlocks, 1024, 0, stream>>>((void*)buff, (void*)scratch, (void*)resultBuff,
                                                           memoryChannels, channelInOffset, rank, nRanksPerNode,
                                                           worldSize, nelems * sizeof(T) / sizeof(int));
  }
  return cudaGetLastError();
}

#endif  // ALLGATHER_HPP_
