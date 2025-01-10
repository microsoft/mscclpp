// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef BROADCAST_HPP_
#define BROADCAST_HPP_

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu.hpp>
#include <mscclpp/sm_channel.hpp>
#include <mscclpp/sm_channel_device.hpp>

#include "common.hpp"

template <bool IsOutOfPlace>
__global__ void __launch_bounds__(1024, 1)
    broadcast6(void* sendbuff, void* scratchbuff, void* recvbuff, mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels,
               [[maybe_unused]] size_t channelOutOffset, size_t rank, [[maybe_unused]] size_t worldSize, size_t root,
               size_t nRanksPerNode, size_t nelemsPerGPU) {
  const size_t bid = blockIdx.x;

  const size_t nThread = blockDim.x * gridDim.x;
  const unsigned int nPeer = nRanksPerNode - 1;
  const size_t chanOffset = nPeer * blockIdx.x;

  __shared__ mscclpp::DeviceHandle<mscclpp::SmChannel> smChans[NRANKS_PER_NODE - 1];
  if (threadIdx.x < nPeer) {
    smChans[threadIdx.x] = smChannels[chanOffset + threadIdx.x];
    smChans[threadIdx.x].relaxedSignal();
    smChans[threadIdx.x].wait();
  }
  __syncthreads();

  const unsigned int peerRootIdx = (root == rank) ? nPeer : ((root < rank) ? root : (root - 1));

  const size_t bytesPerGPU = nelemsPerGPU * sizeof(int);
  const size_t bytes = bytesPerGPU;
  size_t unitBytesPerThread;
  if (bytes >= nThread * 64 * nPeer) {
    unitBytesPerThread = 64;
  } else {
    unitBytesPerThread = 16;
  }
  const size_t unitBytesPerBlock = unitBytesPerThread * blockDim.x;
  const size_t unitBytes = unitBytesPerBlock * gridDim.x * nPeer;
  const size_t nLoop = bytes / unitBytes;

  const size_t maxScratchSizeToUse = (SCRATCH_SIZE - unitBytes);
  const size_t nLoopToSync = (maxScratchSizeToUse / unitBytes) + 1;

  size_t scratchOffset = 0;
  for (size_t i = 0; i < nLoop; ++i) {
    if (i % nLoopToSync == 0 && i > 0) {
      scratchOffset -= nLoopToSync * unitBytes;
      deviceSyncer.sync(gridDim.x);
      if (threadIdx.x < nPeer) {
        smChans[threadIdx.x].signal();
        smChans[threadIdx.x].wait();
      }
    }
    if (rank == root) {
      unsigned int peerIdx = bid % nPeer;
      const size_t offset = blockIdx.x * unitBytesPerBlock * nPeer + i * unitBytes;
      char* send = reinterpret_cast<char*>(sendbuff);
      char* dst = reinterpret_cast<char*>(smChans[peerIdx].dst_);

      smChans[peerIdx].copy<16, false>(dst + offset + scratchOffset, send + offset, nPeer * unitBytesPerBlock,
                                       threadIdx.x, blockDim.x);
      __syncthreads();
      if (threadIdx.x == peerIdx) smChans[threadIdx.x].signal();
      if (IsOutOfPlace) {
        char* recv = reinterpret_cast<char*>(recvbuff);
        smChans[peerIdx].copy<16, false>(recv + offset, send + offset, nPeer * unitBytesPerBlock, threadIdx.x,
                                         blockDim.x);
      }
    } else {  // rank != root.
      int rankIndexInRoot = (rank < root) ? rank : (rank - 1);
      if (bid % nPeer == rankIndexInRoot && threadIdx.x == peerRootIdx) smChans[peerRootIdx].wait();
      deviceSyncer.sync(gridDim.x);  // All blocks in the GPU wait.

      // Step 2.
      char* recv_ = reinterpret_cast<char*>(recvbuff);
      char* scratch_ = reinterpret_cast<char*>(scratchbuff);

      const int chunkId = bid % nPeer;
      const int chunkGroundId = bid / nPeer;
      const size_t offset = chunkId * unitBytesPerBlock +
                            unitBytesPerBlock * nPeer * (chunkGroundId * nPeer + rankIndexInRoot) + i * unitBytes;
      for (unsigned int j = 0; j < nPeer; ++j) {
        unsigned int peerIdx = (bid + j) % nPeer;
        if (peerIdx != peerRootIdx) {
          char* dst = reinterpret_cast<char*>(smChans[peerIdx].dst_);  // Peer's scratchbuff.
          smChans[peerIdx].copy<16, false>(dst + offset + scratchOffset, scratch_ + offset + scratchOffset,
                                           unitBytesPerBlock, threadIdx.x, blockDim.x);
        }
      }
      __syncthreads();
      if (threadIdx.x != peerRootIdx && threadIdx.x < nPeer) {
        smChans[threadIdx.x].signal();
        smChans[threadIdx.x].wait();
      }
      __syncthreads();
      for (unsigned int peerId = 0; peerId < nPeer; ++peerId) {
        const size_t offset =
            chunkId * unitBytesPerBlock + (peerId + chunkGroundId * nPeer) * unitBytesPerBlock * nPeer + i * unitBytes;
        smChans[0].copy<16, false>(recv_ + offset, scratch_ + offset + scratchOffset, unitBytesPerBlock, threadIdx.x,
                                   blockDim.x);
      }
    }
  }

  if (bytes % unitBytes > 0) {
    if (rank == root) {
      unsigned int peerIdx = bid % nPeer;
      const size_t offset = blockIdx.x * unitBytesPerBlock * nPeer + nLoop * unitBytes;
      const size_t remainBytes =
          offset < bytes ? ((bytes - offset) > unitBytesPerBlock * nPeer ? unitBytesPerBlock * nPeer : (bytes - offset))
                         : 0;
      char* send = reinterpret_cast<char*>(sendbuff);
      char* dst = reinterpret_cast<char*>(smChans[peerIdx].dst_);

      smChans[peerIdx].copy<16, true>(dst + offset + scratchOffset, send + offset, remainBytes, threadIdx.x, blockDim.x);
      __syncthreads();
      if (threadIdx.x == peerIdx) smChans[threadIdx.x].signal();
      if constexpr (IsOutOfPlace) {
        char* recv = reinterpret_cast<char*>(recvbuff);
        smChans[peerIdx].copy<16, true>(recv + offset, send + offset, remainBytes, threadIdx.x, blockDim.x);
      }

    } else {
      int rankIndexInRoot = (rank < root) ? rank : (rank - 1);
      if (bid % nPeer == rankIndexInRoot && threadIdx.x == peerRootIdx) smChans[peerRootIdx].wait();
      deviceSyncer.sync(gridDim.x);

      // Step 2.
      char* recv_ = reinterpret_cast<char*>(recvbuff);
      char* scratch_ = reinterpret_cast<char*>(scratchbuff);

      const int chunkId = bid % nPeer;
      const int chunkGroundId = bid / nPeer;
      const size_t offset = chunkId * unitBytesPerBlock +
                            unitBytesPerBlock * nPeer * (chunkGroundId * nPeer + rankIndexInRoot) + nLoop * unitBytes;
      const size_t remainBytes =
          (offset < bytes) ? ((bytes - offset) > unitBytesPerBlock ? unitBytesPerBlock : (bytes - offset)) : 0;

      for (size_t j = 0; j < nPeer; ++j) {
        unsigned peerIdx = (bid + j) % nPeer;
        if (peerIdx != peerRootIdx) {
          char* dst = reinterpret_cast<char*>(smChans[peerIdx].dst_);  // Peer's scratchbuff.
          smChans[peerIdx].copy<16, true>(dst + offset + scratchOffset, scratch_ + offset + scratchOffset, remainBytes,
                                           threadIdx.x, blockDim.x);
        }
      }
      __syncthreads();
      if (threadIdx.x != peerRootIdx && threadIdx.x < nPeer) {
        smChans[threadIdx.x].signal();
        smChans[threadIdx.x].wait();
      }
      __syncthreads();
      for (unsigned int peerId = 0; peerId < nPeer; ++peerId) {
        const size_t offset = chunkId * unitBytesPerBlock +
                              (peerId + chunkGroundId * nPeer) * unitBytesPerBlock * nPeer + nLoop * unitBytes;
        const size_t remainBytes =
            (offset < bytes) ? ((bytes - offset) > unitBytesPerBlock ? unitBytesPerBlock : (bytes - offset)) : 0;
        smChans[0].copy<16, true>(recv_ + offset, scratch_ + offset + scratchOffset, remainBytes, threadIdx.x,
                                  blockDim.x);
      }
    }
  }
}

template <bool IsOutOfPlace, typename T>
cudaError_t broadcast(T* buff, T* scratch, T* resultBuff, mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels,
                      size_t channelOutOffset, int rank, int nRanksPerNode, int root, int worldSize, size_t nelems,
                      cudaStream_t stream) {
  int nBlocks = 7;
  if (nelems <= 4096) {
    nBlocks = 7;
  } else if (nelems >= 32768) {
    nBlocks = 14;
  } else if (nelems >= 5242880) {
    nBlocks = 28;
  }
  broadcast6<IsOutOfPlace><<<nBlocks, 1024, 0, stream>>>((void*)buff, (void*)scratch, (void*)resultBuff, smChannels,
                                                         channelOutOffset, rank, worldSize, root, nRanksPerNode,
                                                         nelems * sizeof(T) / sizeof(int));
  return cudaGetLastError();
}

#endif  // BROADCAST_HPP_
