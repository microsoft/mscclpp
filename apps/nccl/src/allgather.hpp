// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ALLGATHER_HPP_
#define ALLGATHER_HPP_

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/gpu.hpp>
#include <mscclpp/sm_channel_device.hpp>

#include "common.hpp"

template <bool IsOutOfPlace>
__global__ void __launch_bounds__(1024, 1)
    allgather6(void* sendbuff, mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels, size_t rank,
               [[maybe_unused]] size_t worldSize, size_t nRanksPerNode, size_t nelemsPerGPU) {
  const size_t nBlock = gridDim.x;
  if (blockIdx.x >= nBlock) return;

  const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t lid = tid % WARP_SIZE;
  const size_t wid = tid / WARP_SIZE;

  const size_t nThread = blockDim.x * nBlock;
  const size_t nWarp = nThread / WARP_SIZE;
  const size_t nPeer = nRanksPerNode - 1;
  const size_t chanOffset = nPeer * blockIdx.x;
  auto smChans = smChannels + chanOffset;

  if (wid < nPeer && lid == 0) {
    smChans[wid].relaxedSignal();
    smChans[wid].wait();
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
      smChans[peerIdx].copy<16, false>(src + offset, buff + offsetWithinRank, unitBytesPerWarp, lid, WARP_SIZE);
      smChans[peerIdx].copy<16, false>(dst + offset, buff + offsetWithinRank, unitBytesPerWarp, lid, WARP_SIZE);
    } else {
      smChans[peerIdx].put<16, false>(offset, unitBytesPerWarp, lid, WARP_SIZE);
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
      smChans[peerIdx].copy<16, false>(src + offset, buff + offsetWithinRank, unitBytesPerWarp, lid, WARP_SIZE);
      smChans[peerIdx].copy<16, false>(dst + offset, buff + offsetWithinRank, unitBytesPerWarp, lid, WARP_SIZE);
    } else {
      smChans[peerIdx].put<16, false>(offset, unitBytesPerWarp, lid, WARP_SIZE);
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
        smChans[peerIdx].copy<16, true>(src + offset, buff + offsetWithinRank, remainBytes, lid, WARP_SIZE);
        smChans[peerIdx].copy<16, true>(dst + offset, buff + offsetWithinRank, remainBytes, lid, WARP_SIZE);
      } else {
        smChans[peerIdx].put<16, true>(offset, remainBytes, lid, WARP_SIZE);
      }
    }
  }
}

template <bool IsOutOfPlace, typename T>
cudaError_t allgather(T* buff, [[maybe_unused]] T* scratch, [[maybe_unused]] T* resultBuff,
                      mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels, int rank, int nRanksPerNode, int worldSize,
                      size_t nelems, cudaStream_t stream) {
  allgather6<IsOutOfPlace><<<28, 1024, 0, stream>>>((void*)buff, smChannels, rank, worldSize, nRanksPerNode,
                                                    nelems * sizeof(T) / sizeof(int));
  return cudaGetLastError();
}

#endif // ALLGATHER_HPP_
