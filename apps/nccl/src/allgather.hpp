// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ALLGATHER_HPP_
#define ALLGATHER_HPP_

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu.hpp>
#include <mscclpp/sm_channel.hpp>
#include <mscclpp/sm_channel_device.hpp>

#include "common.hpp"

template <bool IsOutOfPlace>
__global__ void __launch_bounds__(1024, 1)
    allgather6(void* sendbuff, mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels, size_t channelOutOffset,
               size_t rank, [[maybe_unused]] size_t worldSize, size_t nRanksPerNode, size_t nelemsPerGPU) {
  const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t lid = tid % WARP_SIZE;
  const size_t wid = tid / WARP_SIZE;

  const size_t nThread = blockDim.x * gridDim.x;
  const size_t nWarp = nThread / WARP_SIZE;
  const size_t nPeer = nRanksPerNode - 1;
  const size_t chanOffset = nPeer * blockIdx.x;
  auto smChans = smChannels + chanOffset;

  if (threadIdx.x < nPeer) {
    smChans[threadIdx.x].relaxedSignal();
    smChans[threadIdx.x].wait();
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
      char* dst = reinterpret_cast<char*>(smChans[peerIdx].dst_);
      char* src = reinterpret_cast<char*>(smChans[peerIdx].src_);
      char* buff = reinterpret_cast<char*>(sendbuff);
      const size_t offsetWithinRank = (wid / nPeer) * unitBytesPerWarp;
      smChans[peerIdx].copy<16, false>(src + offset + channelOutOffset, buff + offsetWithinRank, unitBytesPerWarp, lid,
                                       WARP_SIZE);
      smChans[peerIdx].copy<16, false>(dst + offset + channelOutOffset, buff + offsetWithinRank, unitBytesPerWarp, lid,
                                       WARP_SIZE);
    } else {
      smChans[peerIdx].put<16, false>(offset + channelOutOffset, unitBytesPerWarp, lid, WARP_SIZE);
    }
  }

  for (size_t i = 1; i < nLoop; ++i) {
    const size_t gWid = wid + i * nWarp;
    const size_t peerIdx = gWid % nPeer;
    const size_t offset = bytesPerGPU * rank + (gWid / nPeer) * unitBytesPerWarp;
    if constexpr (IsOutOfPlace) {
      char* dst = reinterpret_cast<char*>(smChans[peerIdx].dst_);
      char* src = reinterpret_cast<char*>(smChans[peerIdx].src_);
      char* buff = reinterpret_cast<char*>(sendbuff);
      const size_t offsetWithinRank = (gWid / nPeer) * unitBytesPerWarp;
      smChans[peerIdx].copy<16, false>(src + offset + channelOutOffset, buff + offsetWithinRank, unitBytesPerWarp, lid,
                                       WARP_SIZE);
      smChans[peerIdx].copy<16, false>(dst + offset + channelOutOffset, buff + offsetWithinRank, unitBytesPerWarp, lid,
                                       WARP_SIZE);
    } else {
      smChans[peerIdx].put<16, false>(offset + channelOutOffset, unitBytesPerWarp, lid, WARP_SIZE);
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
        char* dst = reinterpret_cast<char*>(smChans[peerIdx].dst_);
        char* src = reinterpret_cast<char*>(smChans[peerIdx].src_);
        char* buff = reinterpret_cast<char*>(sendbuff);
        smChans[peerIdx].copy<16, true>(src + offset + channelOutOffset, buff + offsetWithinRank, remainBytes, lid,
                                        WARP_SIZE);
        smChans[peerIdx].copy<16, true>(dst + offset + channelOutOffset, buff + offsetWithinRank, remainBytes, lid,
                                        WARP_SIZE);
      } else {
        smChans[peerIdx].put<16, true>(offset + channelOutOffset, remainBytes, lid, WARP_SIZE);
      }
    }
  }

  deviceSyncer.sync(gridDim.x);

  if (threadIdx.x < nPeer) {
    smChans[threadIdx.x].relaxedSignal();
    smChans[threadIdx.x].wait();
  }
}

template <bool IsOutOfPlace, typename T>
cudaError_t allgather(T* buff, [[maybe_unused]] T* scratch, [[maybe_unused]] T* resultBuff,
                      mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels, size_t channelOutOffset, int rank,
                      int nRanksPerNode, int worldSize, size_t nelems, cudaStream_t stream) {
  int nBlocks = 28;
  if (nelems <= 4096) {
    nBlocks = 7;
  } else if (nelems <= 32768) {
    nBlocks = 14;
  } else if (nelems >= 2097152) {
    nBlocks = 35;
  }
  allgather6<IsOutOfPlace><<<nBlocks, 1024, 0, stream>>>((void*)buff, smChannels, channelOutOffset, rank, worldSize,
                                                         nRanksPerNode, nelems * sizeof(T) / sizeof(int));
  return cudaGetLastError();
}

#endif  // ALLGATHER_HPP_
