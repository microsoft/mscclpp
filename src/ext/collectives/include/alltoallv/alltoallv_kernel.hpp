// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <mscclpp/port_channel_device.hpp>
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
 * AllToAllV kernel implementation using parallel warp-based communication.
 *
 * Each warp handles communication with one peer. All sends happen in parallel,
 * followed by flushes and waits.
 *
 * @param portChannels Array of PortChannel handles for each peer (worldSize-1 channels)
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
    alltoallvKernel(DeviceHandle<PortChannel>* portChannels,
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

  // Step 1: Copy local data (rank's own portion)
  if (tid == 0 && sendCounts[rank] > 0) {
    const char* src = (const char*)sendBuff + sendDispls[rank];
    char* dst = (char*)recvBuff + recvDispls[rank];
    memcpy(dst, src, sendCounts[rank]);
  }
  __syncthreads();

  // Step 2: Each warp handles one peer for sending
  int warpId = tid / ALLTOALLV_WARP_SIZE;
  int laneId = tid % ALLTOALLV_WARP_SIZE;

  if (warpId < nPeers && laneId == 0) {
    // Determine which peer this warp handles
    int peer = warpId < rank ? warpId : warpId + 1;
    int chanIdx = warpId;

    if (sendCounts[peer] > 0) {
      portChannels[chanIdx].putWithSignal(
          recvDispls[rank],       // dst offset in peer's buffer
          sendDispls[peer],       // src offset in our buffer
          sendCounts[peer]        // size
      );
    }
  }
  __syncthreads();

  // Step 3: Flush all pending operations
  if (warpId < nPeers && laneId == 0) {
    int peer = warpId < rank ? warpId : warpId + 1;
    if (sendCounts[peer] > 0) {
      portChannels[warpId].flush();
    }
  }
  __syncthreads();

  // Step 4: Wait for all incoming data
  if (warpId < nPeers && laneId == 0) {
    int peer = warpId < rank ? warpId : warpId + 1;
    if (recvCounts[peer] > 0) {
      portChannels[warpId].wait();
    }
  }
  __syncthreads();
}

/**
 * Ring-based AllToAllV kernel for serialized communication.
 *
 * Uses step-by-step ring pattern to exchange data, sending to (rank+step) and
 * receiving from (rank-step) in each step. Single thread handles all communication
 * to avoid race conditions.
 *
 * This kernel is more robust but slower than the parallel version.
 */
__global__ void __launch_bounds__(1024)
    alltoallvRingKernel(DeviceHandle<PortChannel>* portChannels,
                        int rank,
                        int worldSize,
                        const void* sendBuff,
                        void* recvBuff,
                        const size_t* sendCounts,
                        const size_t* sendDispls,
                        const size_t* recvCounts,
                        const size_t* recvDispls) {
  // Copy local data first
  if (threadIdx.x == 0) {
    if (sendCounts[rank] > 0) {
      const char* src = (const char*)sendBuff + sendDispls[rank];
      char* dst = (char*)recvBuff + recvDispls[rank];
      memcpy(dst, src, sendCounts[rank]);
    }
  }
  __syncthreads();

  // Ring-based exchange - single thread handles communication
  if (threadIdx.x == 0) {
    for (int step = 1; step < worldSize; step++) {
      int sendPeer = (rank + step) % worldSize;
      int recvPeer = (rank - step + worldSize) % worldSize;

      int sendChanIdx = sendPeer < rank ? sendPeer : sendPeer - 1;
      int recvChanIdx = recvPeer < rank ? recvPeer : recvPeer - 1;

      // Send data to sendPeer
      if (sendCounts[sendPeer] > 0) {
        portChannels[sendChanIdx].putWithSignal(
            recvDispls[rank],
            sendDispls[sendPeer],
            sendCounts[sendPeer]
        );
        portChannels[sendChanIdx].flush();
      }

      // Wait for data from recvPeer
      if (recvCounts[recvPeer] > 0) {
        portChannels[recvChanIdx].wait();
      }
    }
  }
}

#undef ALLTOALLV_WARP_SIZE

}  // namespace collective
}  // namespace mscclpp
