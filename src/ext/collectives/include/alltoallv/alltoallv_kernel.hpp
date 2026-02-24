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

// Chunk size for pipelined transfers (1MB)
// Large enough to amortize overhead, small enough for good memory patterns
constexpr size_t ALLTOALLV_CHUNK_SIZE = 1 << 20;

// Default number of blocks for multi-block kernels.
// Tuned for H100 (132 SMs). Enough to saturate NVLink bandwidth without
// excessive DeviceSyncer overhead.
constexpr int ALLTOALLV_DEFAULT_NBLOCKS = 24;

// Default blocks per peer for the peer-parallel kernel.
// Controls how many thread blocks cooperate on each peer's data transfer.
constexpr int ALLTOALLV_DEFAULT_BLOCKS_PER_PEER = 16;

/**
 * Peer-parallel AllToAllV kernel for maximum throughput with multiple GPUs.
 *
 * Unlike the sequential multi-block kernel that processes one peer at a time,
 * this kernel assigns blocks to peers round-robin so ALL NVLink connections
 * are active simultaneously. This is critical for 4+ GPU systems where the
 * per-peer bandwidth is a fraction of aggregate NVLink bandwidth.
 *
 * Block assignment: block i handles peer (i % nPeers). Blocks assigned to the
 * same peer cooperate using local thread IDs within the peer's block group.
 *
 * Signal/wait is also parallelized: each peer's primary block (localBlockIdx==0)
 * independently signals and waits, overlapping wait latencies across peers:
 * total wait = O(max) instead of O(sum).
 *
 * For small messages: launch with nPeers blocks (1 per peer, __syncthreads only)
 * For large messages: launch with nPeers*K blocks (K per peer, DeviceSyncer barrier)
 *
 * Launch config: <<<numBlocks, 1024>>> where numBlocks >= nPeers
 */
__global__ void __launch_bounds__(1024)
    alltoallvPeerParallelKernel(DeviceHandle<MemoryChannel>* memoryChannels,
                                DeviceSyncer* syncer,
                                int rank,
                                int worldSize,
                                const void* sendBuff,
                                void* recvBuff,
                                const size_t* sendCounts,
                                const size_t* sendDispls,
                                const size_t* recvCounts,
                                const size_t* recvDispls,
                                const size_t* remoteRecvDispls) {
  const int nPeers = worldSize - 1;

  // Handle trivial case (single rank, no peers)
  if (nPeers == 0) {
    const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
    const int nThreads = blockDim.x * gridDim.x;
    if (sendCounts[rank] > 0) {
      mscclpp::copy((char*)recvBuff + recvDispls[rank],
                    (void*)((const char*)sendBuff + sendDispls[rank]),
                    sendCounts[rank], gtid, nThreads);
    }
    return;
  }

  // Phase 1: Local copy — all blocks cooperate using global thread IDs
  const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  const int nThreads = blockDim.x * gridDim.x;
  if (sendCounts[rank] > 0) {
    mscclpp::copy((char*)recvBuff + recvDispls[rank],
                  (void*)((const char*)sendBuff + sendDispls[rank]),
                  sendCounts[rank], gtid, nThreads);
  }

  // Phase 2: Per-peer remote puts — blocks assigned round-robin to peers.
  // Block i handles peer (i % nPeers). Multiple blocks for the same peer
  // cooperate using local thread IDs within the peer's block group.
  const int myPeerIdx = blockIdx.x % nPeers;
  const int localBlockIdx = blockIdx.x / nPeers;
  const int nBlocksForMyPeer = ((int)gridDim.x - myPeerIdx + nPeers - 1) / nPeers;

  const int localTid = threadIdx.x + localBlockIdx * blockDim.x;
  const int nLocalThreads = nBlocksForMyPeer * blockDim.x;

  const int peer = myPeerIdx < rank ? myPeerIdx : myPeerIdx + 1;

  if (sendCounts[peer] > 0) {
    memoryChannels[myPeerIdx].put(
        remoteRecvDispls[peer],  // dst offset in peer's buffer
        sendDispls[peer],        // src offset in our buffer
        sendCounts[peer],        // size
        localTid,                // thread id within peer's block group
        nLocalThreads            // total threads for this peer
    );
  }

  // Phase 3: Synchronization
  // - Multiple blocks per peer (gridDim.x > nPeers): grid-wide barrier to ensure
  //   all blocks' put contributions complete before any signaling.
  // - Exactly one block per peer: __syncthreads() suffices (no cross-block deps).
  if ((int)gridDim.x > nPeers) {
    syncer->sync(gridDim.x);
  } else {
    __syncthreads();
  }

  // Phase 4: Signal and wait — parallelized across peers.
  // Each peer's primary block (localBlockIdx==0, thread 0) independently
  // signals and waits. Wait latencies overlap: O(max) instead of O(sum).
  if (threadIdx.x == 0 && localBlockIdx == 0) {
    memoryChannels[myPeerIdx].signal();
    if (recvCounts[peer] > 0) {
      memoryChannels[myPeerIdx].wait();
    }
  }
}

/**
 * Legacy multi-block AllToAllV kernel (sequential peers).
 *
 * All thread blocks cooperate on each peer's data transfer using global thread IDs.
 * Peers are processed sequentially. Kept for comparison; prefer alltoallvPeerParallelKernel.
 *
 * Launch config: <<<nBlocks, 1024>>>
 */
__global__ void __launch_bounds__(1024)
    alltoallvMultiBlockKernel(DeviceHandle<MemoryChannel>* memoryChannels,
                              DeviceSyncer* syncer,
                              int rank,
                              int worldSize,
                              const void* sendBuff,
                              void* recvBuff,
                              const size_t* sendCounts,
                              const size_t* sendDispls,
                              const size_t* recvCounts,
                              const size_t* recvDispls,
                              const size_t* remoteRecvDispls) {
  const int gtid = threadIdx.x + blockIdx.x * blockDim.x;
  const int nThreads = blockDim.x * gridDim.x;
  const int nPeers = worldSize - 1;

  // Phase 1: Local copy — all threads across all blocks cooperate
  if (sendCounts[rank] > 0) {
    mscclpp::copy((char*)recvBuff + recvDispls[rank],
                  (void*)((const char*)sendBuff + sendDispls[rank]),
                  sendCounts[rank], gtid, nThreads);
  }

  // Phase 2: Remote puts — all blocks cooperate on each peer's transfer
  for (int peerIdx = 0; peerIdx < nPeers; peerIdx++) {
    int peer = peerIdx < rank ? peerIdx : peerIdx + 1;
    int chanIdx = peerIdx;

    if (sendCounts[peer] > 0) {
      memoryChannels[chanIdx].put(
          remoteRecvDispls[peer],
          sendDispls[peer],
          sendCounts[peer],
          gtid,
          nThreads
      );
    }
  }

  // Phase 3: Grid-wide barrier
  syncer->sync(gridDim.x);

  // Phase 4: Signal all peers, then wait (single thread)
  if (gtid == 0) {
    for (int peerIdx = 0; peerIdx < nPeers; peerIdx++) {
      memoryChannels[peerIdx].signal();
    }
    for (int peerIdx = 0; peerIdx < nPeers; peerIdx++) {
      int peer = peerIdx < rank ? peerIdx : peerIdx + 1;
      if (recvCounts[peer] > 0) {
        memoryChannels[peerIdx].wait();
      }
    }
  }
}

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
                    const size_t* recvDispls,
                    const size_t* remoteRecvDispls) {
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
          remoteRecvDispls[peer], // dst offset in peer's buffer (peer's recvDispls[rank])
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
 * Pipelined AllToAllV kernel for imbalanced workloads.
 *
 * For large messages, breaks transfers into chunks to improve memory access
 * patterns, but avoids excessive signaling overhead by signaling only once
 * per peer after all chunks are sent.
 *
 * Optimized for MoE workloads where message sizes can vary by 100x+ between ranks.
 *
 * Launch config: <<<1, 1024>>>
 */
__global__ void __launch_bounds__(1024)
    alltoallvPipelinedKernel(DeviceHandle<MemoryChannel>* memoryChannels,
                             int rank,
                             int worldSize,
                             const void* sendBuff,
                             void* recvBuff,
                             const size_t* sendCounts,
                             const size_t* sendDispls,
                             const size_t* recvCounts,
                             const size_t* recvDispls,
                             const size_t* remoteRecvDispls) {
  int tid = threadIdx.x;
  int nThreads = blockDim.x;
  int nPeers = worldSize - 1;

  // Step 1: Copy local data
  if (sendCounts[rank] > 0) {
    mscclpp::copy((char*)recvBuff + recvDispls[rank],
                  (void*)((const char*)sendBuff + sendDispls[rank]),
                  sendCounts[rank], tid, nThreads);
  }
  __syncthreads();

  // Step 2: Process each peer - send all data in chunks, then signal once
  for (int peerIdx = 0; peerIdx < nPeers; peerIdx++) {
    int peer = peerIdx < rank ? peerIdx : peerIdx + 1;
    int chanIdx = peerIdx;

    size_t sendSize = sendCounts[peer];
    size_t recvSize = recvCounts[peer];
    size_t dstOffset = remoteRecvDispls[peer]; // peer's recvDispls[rank]
    size_t srcOffset = sendDispls[peer];

    // Send data in chunks for better memory access patterns
    // But only signal ONCE after all chunks are sent (avoids signaling overhead)
    if (sendSize > 0) {
      for (size_t offset = 0; offset < sendSize; offset += ALLTOALLV_CHUNK_SIZE) {
        size_t chunkSize = (sendSize - offset < ALLTOALLV_CHUNK_SIZE)
                           ? (sendSize - offset) : ALLTOALLV_CHUNK_SIZE;
        memoryChannels[chanIdx].put(
            dstOffset + offset,
            srcOffset + offset,
            chunkSize,
            tid,
            nThreads
        );
        __syncthreads();
      }
    }

    // Signal ONCE after all data is sent
    if (tid == 0 && sendSize > 0) {
      memoryChannels[chanIdx].signal();
    }
    __syncthreads();

    // Wait ONCE for all peer's data
    if (tid == 0 && recvSize > 0) {
      memoryChannels[chanIdx].wait();
    }
    __syncthreads();
  }
}

/**
 * Ring-based AllToAllV kernel with maximum thread parallelism.
 *
 * Uses step-by-step ring pattern with ALL threads for maximum bandwidth.
 * Each step processes one peer pair, with correct semaphore handling.
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
                        const size_t* recvDispls,
                        const size_t* remoteRecvDispls) {
  int tid = threadIdx.x;
  int nThreads = blockDim.x;

  // Copy local data first using ALL threads
  if (sendCounts[rank] > 0) {
    mscclpp::copy((char*)recvBuff + recvDispls[rank],
                  (void*)((const char*)sendBuff + sendDispls[rank]),
                  sendCounts[rank], tid, nThreads);
  }
  __syncthreads();

  // Ring-based exchange - process each peer sequentially
  // Key fix: use the SAME channel for both signal and wait (peer-pair symmetry)
  for (int step = 1; step < worldSize; step++) {
    int sendPeer = (rank + step) % worldSize;
    int chanIdx = sendPeer < rank ? sendPeer : sendPeer - 1;

    // Send data to sendPeer using ALL threads
    if (sendCounts[sendPeer] > 0) {
      memoryChannels[chanIdx].put(
          remoteRecvDispls[sendPeer], // dst offset in peer's buffer (peer's recvDispls[rank])
          sendDispls[sendPeer],
          sendCounts[sendPeer],
          tid,
          nThreads
      );
    }
    __syncthreads();

    // Signal completion on the SAME channel we'll wait on
    if (tid == 0) {
      memoryChannels[chanIdx].signal();
    }
    __syncthreads();

    // Wait for peer's data on the SAME channel (correct semaphore pairing)
    if (tid == 0 && recvCounts[sendPeer] > 0) {
      memoryChannels[chanIdx].wait();
    }
    __syncthreads();
  }
}

#undef ALLTOALLV_WARP_SIZE
}  // namespace collective
}  // namespace mscclpp