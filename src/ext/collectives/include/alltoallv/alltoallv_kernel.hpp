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
 * AllToAllV kernel implementation using parallel warp-based communication with MemoryChannel.
 *
 * Each warp handles communication with one peer. Data is copied in parallel using all threads
 * in the warp, which significantly improves throughput for large messages.
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
  int nPeers = worldSize - 1;

  // Step 1: Copy local data (rank's own portion) using all threads
  if (sendCounts[rank] > 0) {
    mscclpp::copy((char*)recvBuff + recvDispls[rank],
                  (void*)((const char*)sendBuff + sendDispls[rank]),
                  sendCounts[rank], tid, blockDim.x);
  }
  __syncthreads();

  // Step 2: Each warp handles one peer for sending (parallel copy within warp)
  int warpId = tid / ALLTOALLV_WARP_SIZE;
  int laneId = tid % ALLTOALLV_WARP_SIZE;

  if (warpId < nPeers) {
    // Determine which peer this warp handles
    int peer = warpId < rank ? warpId : warpId + 1;
    int chanIdx = warpId;

    if (sendCounts[peer] > 0) {
      // Use parallel put with all threads in the warp
      // targetOffset: recvDispls[rank] - where peer should receive our data
      // originOffset: sendDispls[peer] - where our data for this peer starts
      memoryChannels[chanIdx].put(
          recvDispls[rank],       // dst offset in peer's buffer
          sendDispls[peer],       // src offset in our buffer
          sendCounts[peer],       // size
          laneId,                 // thread id within warp
          ALLTOALLV_WARP_SIZE     // number of threads
      );
    }
  }
  __syncthreads();

  // Step 3: Signal completion to all peers
  if (warpId < nPeers && laneId == 0) {
    memoryChannels[warpId].signal();
  }
  __syncthreads();

  // Step 4: Wait for all incoming data
  if (warpId < nPeers && laneId == 0) {
    int peer = warpId < rank ? warpId : warpId + 1;
    if (recvCounts[peer] > 0) {
      memoryChannels[warpId].wait();
    }
  }
  __syncthreads();
}

/**
 * Ring-based AllToAllV kernel for serialized communication with MemoryChannel.
 *
 * Uses step-by-step ring pattern to exchange data, sending to (rank+step) and
 * receiving from (rank-step) in each step. All threads participate in the copy
 * for better throughput.
 *
 * This kernel is more robust for larger world sizes.
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

  // Copy local data first using all threads
  if (sendCounts[rank] > 0) {
    mscclpp::copy((char*)recvBuff + recvDispls[rank],
                  (void*)((const char*)sendBuff + sendDispls[rank]),
                  sendCounts[rank], tid, blockDim.x);
  }
  __syncthreads();

  // Ring-based exchange - all threads participate in copy
  for (int step = 1; step < worldSize; step++) {
    int sendPeer = (rank + step) % worldSize;
    int recvPeer = (rank - step + worldSize) % worldSize;

    int sendChanIdx = sendPeer < rank ? sendPeer : sendPeer - 1;
    int recvChanIdx = recvPeer < rank ? recvPeer : recvPeer - 1;

    // Send data to sendPeer using all threads
    if (sendCounts[sendPeer] > 0) {
      memoryChannels[sendChanIdx].put(
          recvDispls[rank],
          sendDispls[sendPeer],
          sendCounts[sendPeer],
          tid,
          blockDim.x
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