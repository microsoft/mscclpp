// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <mscclpp/memory_channel_device.hpp>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/copy_device.hpp>

namespace mscclpp {
namespace collective {

#if defined(__HIP_PLATFORM_AMD__)
#define ALLTOALLV_WARP_SIZE 64
#else
#define ALLTOALLV_WARP_SIZE 32
#endif

/**
 * High-performance AllToAllV kernel using maximum thread parallelism.
 *
 * Processes each peer sequentially but uses ALL block threads (1024) for each
 * data transfer to maximize copy bandwidth. This provides much better performance
 * than the warp-per-peer approach for large message sizes.
 *
 * Launch config: <<<1, 1024>>> for maximum bandwidth within a single block.
 *
 * @param memoryChannels Array of MemoryChannel handles for each peer (worldSize-1 channels)
 * @param rank Current rank
 * @param worldSize Total number of ranks
 * @param sendBuff Source buffer containing data to send
 * @param recvBuff Destination buffer for received data
 * @param sendCounts Array of send counts for each rank (in bytes)
 * @param sendDispls Array of send displacements for each rank (in bytes)
 * @param recvCounts Array of receive counts for each rank (in bytes)
 * @param recvDispls Array of receive displacements for each rank (in bytes)
 */
__global__ void __launch_bounds__(1024)
    alltoallvKernel(DeviceHandle<MemoryChannel>* memoryChannels,
                    int rank,
                    int worldSize,
                    const void* sendBuff,
                    void* recvBuff,
                    const size_t* sendCounts,
                    const size_t* sendDispls,
                    const size_t* recvCounts,
                    const size_t* recvDispls) {
  int tid = threadIdx.x;
  int nThreads = blockDim.x;
  int nPeers = worldSize - 1;

  // Step 1: Copy local data using ALL threads for maximum bandwidth
  if (sendCounts[rank] > 0) {
    mscclpp::copy((char*)recvBuff + recvDispls[rank],
                  (void*)((const char*)sendBuff + sendDispls[rank]),
                  sendCounts[rank], tid, nThreads);
  }
  __syncthreads();

  // Step 2: Process each peer sequentially, but use ALL threads for each transfer
  // This maximizes bandwidth for each transfer compared to warp-per-peer approach
  for (int peerIdx = 0; peerIdx < nPeers; peerIdx++) {
    int peer = peerIdx < rank ? peerIdx : peerIdx + 1;
    int chanIdx = peerIdx;

    if (sendCounts[peer] > 0) {
      // Use all threads for maximum copy throughput
      memoryChannels[chanIdx].put(
          recvDispls[rank],       // dst offset in peer's buffer
          sendDispls[peer],       // src offset in our buffer
          sendCounts[peer],       // size
          tid,                    // thread id
          nThreads                // total threads
      );
    }
    __syncthreads();

    // Only one thread signals per peer
    if (tid == 0) {
      memoryChannels[chanIdx].signal();
    }
    __syncthreads();

    // Wait for incoming data from this peer
    if (tid == 0 && recvCounts[peer] > 0) {
      memoryChannels[chanIdx].wait();
    }
    __syncthreads();
  }
}

/**
 * Ring-based AllToAllV kernel with maximum thread parallelism.
 *
 * Uses step-by-step ring pattern with ALL threads for maximum bandwidth.
 * Better for larger world sizes to avoid congestion.
 */
__global__ void __launch_bounds__(1024)
    alltoallvRingKernel(DeviceHandle<MemoryChannel>* memoryChannels,
                        int rank,
                        int worldSize,
                        const void* sendBuff,
                        void* recvBuff,
                        const size_t* sendCounts,
                        const size_t* sendDispls,
                        const size_t* recvCounts,
                        const size_t* recvDispls) {
  int tid = threadIdx.x;
  int nThreads = blockDim.x;

  // Copy local data first using ALL threads
  if (sendCounts[rank] > 0) {
    mscclpp::copy((char*)recvBuff + recvDispls[rank],
                  (void*)((const char*)sendBuff + sendDispls[rank]),
                  sendCounts[rank], tid, nThreads);
  }
  __syncthreads();

  // Ring-based exchange - all threads participate in copy
  for (int step = 1; step < worldSize; step++) {
    int sendPeer = (rank + step) % worldSize;
    int recvPeer = (rank - step + worldSize) % worldSize;

    int sendChanIdx = sendPeer < rank ? sendPeer : sendPeer - 1;
    int recvChanIdx = recvPeer < rank ? recvPeer : recvPeer - 1;

    // Send data to sendPeer using ALL threads
    if (sendCounts[sendPeer] > 0) {
      memoryChannels[sendChanIdx].put(
          recvDispls[rank],
          sendDispls[sendPeer],
          sendCounts[sendPeer],
          tid,
          nThreads
      );
    }
    __syncthreads();

    // Signal completion
    if (tid == 0 && sendCounts[sendPeer] > 0) {
      memoryChannels[sendChanIdx].signal();
    }
    __syncthreads();

    // Wait for data from recvPeer
    if (tid == 0 && recvCounts[recvPeer] > 0) {
      memoryChannels[recvChanIdx].wait();
    }
    __syncthreads();
  }
}

#undef ALLTOALLV_WARP_SIZE
}  // namespace collective
}  // namespace mscclpp