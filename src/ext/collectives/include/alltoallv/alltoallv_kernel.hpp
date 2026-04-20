// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <mscclpp/memory_channel_device.hpp>
#include <mscclpp/port_channel_device.hpp>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/copy_device.hpp>

namespace mscclpp {
namespace collective {

// Default blocks per peer for the peer-parallel kernel.
// Controls how many thread blocks cooperate on each peer's data transfer.
constexpr int ALLTOALLV_DEFAULT_BLOCKS_PER_PEER = 16;

/**
 * Hybrid AllToAllV kernel for multi-node: MemoryChannel (intra-node) + PortChannel (inter-node).
 *
 * Each block handles one peer (1 block per peer). For intra-node peers, all threads
 * cooperate on a MemoryChannel put (multi-threaded NVLink copy). For inter-node peers,
 * thread 0 pushes a PortChannel put descriptor to the CPU proxy FIFO (single-threaded),
 * which triggers an RDMA transfer.
 *
 * Key design points:
 * - MemoryChannel uses peerIdx-based dense indexing (only intra-node peers have MemoryChannels)
 *   but we need the SAME peerIdx ordering as the connection array.
 *   In practice, memoryChannels[] are created only for CudaIpc connections and are dense.
 *   We use a separate peerToMemChIdx mapping from peerIsLocal.
 * - PortChannel uses separate dense indexing via peerToPortChannelIdx.
 * - Signal/wait is done per-peer by thread 0 of each block.
 *
 * Launch config: <<<nPeers, 1024>>>
 */
__global__ void __launch_bounds__(1024)
    alltoallvHybridKernel(DeviceHandle<MemoryChannel>* memoryChannels,
                          PortChannelDeviceHandle* portChannels,
                          const int* peerIsLocal,
                          const int* peerToPortChannelIdx,
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

  // Handle trivial case (single rank)
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

  // Phase 2: Per-peer data transfer.
  // Each block handles one peer: blockIdx.x == peerIdx
  const int peerIdx = blockIdx.x;
  if (peerIdx >= nPeers) return;

  const int peer = peerIdx < rank ? peerIdx : peerIdx + 1;

  if (peerIsLocal[peerIdx]) {
    // Intra-node: MemoryChannel — all threads cooperate on multi-threaded put
    // MemoryChannels are densely indexed for CudaIpc connections only.
    // We need to compute the MemoryChannel index from peerIdx.
    // Count how many local peers are before this peerIdx.
    int memChIdx = 0;
    for (int i = 0; i < peerIdx; i++) {
      if (peerIsLocal[i]) memChIdx++;
    }

    if (sendCounts[peer] > 0) {
      memoryChannels[memChIdx].put(
          remoteRecvDispls[peer],  // dst offset in peer's buffer
          sendDispls[peer],        // src offset in our buffer
          sendCounts[peer],        // size
          threadIdx.x,             // thread id within block
          blockDim.x               // total threads for this peer
      );
    }
    __syncthreads();

    // Signal and wait (thread 0 only)
    if (threadIdx.x == 0) {
      memoryChannels[memChIdx].signal();
      memoryChannels[memChIdx].wait();
      __threadfence_system();
    }
  } else {
    // Inter-node: PortChannel — single-threaded FIFO push
    int portChIdx = peerToPortChannelIdx[peerIdx];

    if (threadIdx.x == 0 && sendCounts[peer] > 0) {
      portChannels[portChIdx].putWithSignalAndFlush(
          remoteRecvDispls[peer],  // dst offset
          sendDispls[peer],        // src offset
          sendCounts[peer]         // size
      );
    }
    __syncthreads();

    // Wait for incoming data from remote peer
    if (threadIdx.x == 0 && recvCounts[peer] > 0) {
      portChannels[portChIdx].wait();
    }
  }
}

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
    memoryChannels[myPeerIdx].wait();
  }

  // ALL threads/blocks must execute the fence before kernel exit.
  // Only the primary block does signal/wait, but ALL blocks did put() —
  // their NVLink writes may still be in flight. The fence ensures every
  // SM's write buffer is flushed before the kernel is marked "complete".
  __threadfence_system();
}



}  // namespace collective
}  // namespace mscclpp