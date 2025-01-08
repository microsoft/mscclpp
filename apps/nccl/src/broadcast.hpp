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
               size_t channelOutOffset, size_t rank, [[maybe_unused]] size_t worldSize, size_t root,
               size_t nRanksPerNode, size_t nelemsPerGPU) {
  const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t lid = tid % WARP_SIZE;
  const size_t wid = tid / WARP_SIZE;

  const size_t nThread = blockDim.x * gridDim.x;
  const size_t nWarp = nThread / WARP_SIZE;
  const size_t nPeer = nRanksPerNode - 1;
  const size_t chanOffset = nPeer * blockIdx.x;
  const size_t peerIdx = blockIdx.x;  // Stores the peerIdx.

  __shared__ mscclpp::DeviceHandle<mscclpp::SmChannel> smChans[NRANKS_PER_NODE - 1];
  if (threadIdx.x < nPeer) {
    smChans[threadIdx.x] = smChannels[chanOffset + threadIdx.x];
    smChans[threadIdx.x].relaxedSignal();
    smChans[threadIdx.x].wait();
  }
  __syncthreads();

  const size_t peerRootIdx = (root == rank) ? nPeer : ((root < rank) ? root : (root - 1));
  const size_t rootsmaller = (root < rank) ? 1 : 0;

  const size_t bytesPerGPU = nelemsPerGPU * sizeof(int);
  const size_t bytes = bytesPerGPU;
  size_t unitBytesPerThread;
  if (bytes >= nThread * 64) {
    unitBytesPerThread = 64;
    // unitBytesPerThread = 16;
    // unitBytesPerThread = 32;
  } else {
    unitBytesPerThread = 16;
  }
  const size_t unitBytesPerBlock = unitBytesPerThread * blockDim.x;
  const size_t unitBytes = unitBytesPerBlock * gridDim.x;
  const size_t nLoop = bytes / unitBytes;

  const size_t maxScratchSizeToUse = (SCRATCH_SIZE - unitBytes);
  const size_t nLoopToSync = (maxScratchSizeToUse / unitBytes) + 1;

  size_t scratchSub = 0;

  // printf("nLoop = %ld, bytes = %ld, unitBytes = %ld, bytes mod unitBytes = %ld \n", nLoop, bytes, unitBytes,
  //        bytes % unitBytes);

  // First loop will always fit the scratch size.
  if (nLoop > 0) {
    // First loop unrolling
    if (rank == root) {
      const size_t offset = blockIdx.x * unitBytesPerBlock;
      char* send_ = reinterpret_cast<char*>(sendbuff);
      char* dst = reinterpret_cast<char*>(smChans[peerIdx].dst_);  // Peer's scratchbuff.
      smChans[peerIdx].copy<16, false>(dst + offset, send_ + offset, unitBytesPerBlock, threadIdx.x, blockDim.x);
      __syncthreads();
      if (threadIdx.x == peerIdx) smChans[threadIdx.x].signal();
      if constexpr (IsOutOfPlace) {
        char* recv_ = reinterpret_cast<char*>(recvbuff);
        smChans[0].copy<16, false>(recv_ + offset, send_ + offset, unitBytesPerBlock, threadIdx.x, blockDim.x);
      }
    } else {  // rank != root.
      const size_t offset = (rank - rootsmaller) * unitBytesPerBlock;
      if (blockIdx.x == (rank - rootsmaller) && threadIdx.x == peerRootIdx) smChans[peerRootIdx].wait();
      deviceSyncer.sync(gridDim.x);  // All blocks in the GPU wait.

      // Step 2.
      char* recv_ = reinterpret_cast<char*>(recvbuff);
      char* scratch_ = reinterpret_cast<char*>(scratchbuff);       // My scratchbuff.
      char* dst = reinterpret_cast<char*>(smChans[peerIdx].dst_);  // Peer's scratchbuff.
      if (peerIdx != peerRootIdx) {
        smChans[peerIdx].copy<16, false>(dst + offset, scratch_ + offset, unitBytesPerBlock, threadIdx.x, blockDim.x);
      }
      __syncthreads();
      if (threadIdx.x != peerRootIdx && threadIdx.x < nPeer) {
        smChans[threadIdx.x].signal();
        smChans[threadIdx.x].wait();
      }
      deviceSyncer.sync(gridDim.x);  // All blocks in the GPU wait.
      //__syncthreads();
      {
        const size_t offset = blockIdx.x * unitBytesPerBlock;
        smChans[peerIdx].copy<16, false>(recv_ + offset, scratch_ + offset, unitBytesPerBlock, threadIdx.x, blockDim.x);
      }
    }
  }

  for (size_t i = 1; i < nLoop; ++i) {
    if (i % nLoopToSync == 0) {  // Sync to reuse scratch buff
      scratchSub = -i * unitBytes;
      deviceSyncer.sync(gridDim.x);
      if (threadIdx.x < nPeer) {
        smChans[threadIdx.x].signal();
        smChans[threadIdx.x].wait();
      }
    }
    if (rank == root) {
      const size_t offset = blockIdx.x * unitBytesPerBlock + i * unitBytes;
      char* send_ = reinterpret_cast<char*>(sendbuff);
      char* dst = reinterpret_cast<char*>(smChans[peerIdx].dst_);  // Peer's scratchbuff.

      smChans[peerIdx].copy<16, false>(dst + offset + scratchSub, send_ + offset, unitBytesPerBlock, threadIdx.x,
                                       blockDim.x);
      __syncthreads();
      if (threadIdx.x == peerIdx) smChans[threadIdx.x].signal();
      if constexpr (IsOutOfPlace) {
        char* recv_ = reinterpret_cast<char*>(recvbuff);
        smChans[0].copy<16, false>(recv_ + offset, send_ + offset, unitBytesPerBlock, threadIdx.x, blockDim.x);
      }
    } else {  // rank != root.
      const size_t offset = (rank - rootsmaller) * unitBytesPerBlock + i * unitBytes;
      if (blockIdx.x == (rank - rootsmaller) && threadIdx.x == peerRootIdx) smChans[peerRootIdx].wait();
      deviceSyncer.sync(gridDim.x);  // All blocks in the GPU wait.

      // Step 2.
      char* recv_ = reinterpret_cast<char*>(recvbuff);
      char* scratch_ = reinterpret_cast<char*>(scratchbuff);       // My scratchbuff.
      char* dst = reinterpret_cast<char*>(smChans[peerIdx].dst_);  // Peer's scratchbuff.
      if (peerIdx != peerRootIdx) {
        smChans[peerIdx].copy<16, false>(dst + offset + scratchSub, scratch_ + offset + scratchSub, unitBytesPerBlock,
                                         threadIdx.x, blockDim.x);
      }
      __syncthreads();
      if (threadIdx.x != peerRootIdx && threadIdx.x < nPeer) {
        smChans[threadIdx.x].signal();
        smChans[threadIdx.x].wait();
      }
      deviceSyncer.sync(gridDim.x);  // All blocks in the GPU wait.
      {
        const size_t offset = blockIdx.x * unitBytesPerBlock + i * unitBytes;
        smChans[peerIdx].copy<16, false>(recv_ + offset, scratch_ + offset + scratchSub, unitBytesPerBlock, threadIdx.x,
                                         blockDim.x);
      }
    }
  }

  // Remainder loop will also fit the scratch buff since we subtract unitBytes from SCRATCH_SIZE.
  if (bytes % unitBytes > 0) {  // remainder.
                                // const size_t remainTotalBytes = bytes - nLoop * unitBytes;
    // const size_t nblocks_to_use_base = remainTotalBytes / unitBytesPerBlock;
    // const size_t nblocks_to_use =
    //     (remainTotalBytes % unitBytesPerBlock) ? nblocks_to_use_base + 1 : nblocks_to_use_base;

    // printf("nLoop = %ld, bytes = %ld, nblocks_to_use = %ld\n", nLoop, bytes, nblocks_to_use);

    // if (blockIdx.x < nblocks_to_use) {
    if (rank == root) {
      const size_t offset = blockIdx.x * unitBytesPerBlock + nLoop * unitBytes;
      const size_t remainBytes =
          offset < bytes ? ((bytes - offset) > unitBytesPerBlock ? unitBytesPerBlock : (bytes - offset)) : 0;
      char* send_ = reinterpret_cast<char*>(sendbuff);
      char* dst = reinterpret_cast<char*>(smChans[peerIdx].dst_);  // Peer's scratchbuff.

      smChans[peerIdx].copy<16, true>(dst + offset + scratchSub, send_ + offset, remainBytes, threadIdx.x, blockDim.x);
      __syncthreads();
      if (threadIdx.x == peerIdx) smChans[threadIdx.x].signal();
      if constexpr (IsOutOfPlace) {
        char* recv_ = reinterpret_cast<char*>(recvbuff);
        smChans[0].copy<16, true>(recv_ + offset, send_ + offset, remainBytes, threadIdx.x, blockDim.x);
      }

    } else {  // rank != root.
      const size_t offset = (rank - rootsmaller) * unitBytesPerBlock + nLoop * unitBytes;
      const size_t remainBytes =
          (offset < bytes) ? ((bytes - offset) > unitBytesPerBlock ? unitBytesPerBlock : (bytes - offset)) : 0;

      if (blockIdx.x == (rank - rootsmaller) && threadIdx.x == peerRootIdx) smChans[peerRootIdx].wait();
      deviceSyncer.sync(gridDim.x);  // All blocks in the GPU wait.
      __syncthreads();

      // Step 2.
      char* recv_ = reinterpret_cast<char*>(recvbuff);
      char* scratch_ = reinterpret_cast<char*>(scratchbuff);       // My scratchbuff.
      char* dst = reinterpret_cast<char*>(smChans[peerIdx].dst_);  // Peer's scratchbuff.
      if (peerIdx != peerRootIdx) {
        smChans[peerIdx].copy<16, true>(dst + offset + scratchSub, scratch_ + offset + scratchSub, remainBytes,
                                        threadIdx.x, blockDim.x);
      }
      __syncthreads();
      if (threadIdx.x != peerRootIdx && threadIdx.x < nPeer) {
        smChans[threadIdx.x].signal();
        smChans[threadIdx.x].wait();
      }
      deviceSyncer.sync(gridDim.x);  // All blocks in the GPU wait.
      {
        const size_t offset = blockIdx.x * unitBytesPerBlock + nLoop * unitBytes;
        const size_t remainBytes =
            (offset < bytes) ? ((bytes - offset) > unitBytesPerBlock ? unitBytesPerBlock : (bytes - offset)) : 0;
        smChans[peerIdx].copy<16, true>(recv_ + offset, scratch_ + offset + scratchSub, remainBytes, threadIdx.x,
                                        blockDim.x);
      }
    }
    //}  // remainBytes > 0.
  }

  deviceSyncer.sync(gridDim.x);

  if (threadIdx.x < nPeer) {
    smChans[threadIdx.x].relaxedSignal();
    smChans[threadIdx.x].wait();
  }
}

template <bool IsOutOfPlace, typename T>
cudaError_t broadcast(T* buff, T* scratch, T* resultBuff, mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannels,
                      size_t channelOutOffset, int rank, int nRanksPerNode, int root, int worldSize, size_t nelems,
                      cudaStream_t stream) {
  int nBlocks = 7;
  // if (nelems <= 4096) {
  //   nBlocks = 7;
  // } else if (nelems <= 32768) {
  //   nBlocks = 14;
  // } else if (nelems >= 2097152) {
  //   nBlocks = 35;
  // }
  broadcast6<IsOutOfPlace><<<nBlocks, 1024, 0, stream>>>((void*)buff, (void*)scratch, (void*)resultBuff, smChannels,
                                                         channelOutOffset, rank, worldSize, root, nRanksPerNode,
                                                         nelems * sizeof(T) / sizeof(int));
  return cudaGetLastError();
}

#endif  // BROADCAST_HPP_
