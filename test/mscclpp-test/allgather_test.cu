// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <algorithm>
#include <cstring>
#include <mscclpp/concurrency_device.hpp>
#include <string>

#include "common.hpp"

#if defined(__HIP_PLATFORM_AMD__)
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif

namespace {
auto isUsingHostOffload = [](int kernelNum) { return kernelNum == 3; };
constexpr uint64_t MAGIC = 0xdeadbeef;
}  // namespace

template <class T>
using DeviceHandle = mscclpp::DeviceHandle<T>;
__constant__ DeviceHandle<mscclpp::PortChannel> constPortChans[16];
__constant__ DeviceHandle<mscclpp::BasePortChannel> constRawPortChan[16];

__constant__ DeviceHandle<mscclpp::MemoryChannel> constMemChans[512];
__constant__ DeviceHandle<mscclpp::MemoryChannel> constMemOutOfPlaceChans[16];
__device__ uint64_t globalFlag;

__global__ void __launch_bounds__(1024) allgather0(int rank, size_t nelemsPerGPU) {
  int warpId = threadIdx.x / WARP_SIZE;

  // Each warp is responsible for one of the remote ranks
  DeviceHandle<mscclpp::PortChannel> portChan = constPortChans[warpId];

  // this allgather is really simple and implemented as an alltoall

  // this thread's role is a sender role
  // put your data asynchronously
  if (threadIdx.x % WARP_SIZE == 0) {
    portChan.putWithSignal(rank * nelemsPerGPU * sizeof(int), nelemsPerGPU * sizeof(int));
  }
  // make sure everyone is put their data before some thread randomly blocks everyone else in signal
  __syncthreads();
  // push with flag and sync to make sure the data is received
  if (threadIdx.x % WARP_SIZE == 0) portChan.flush();

  // this thread's role is a receiver role. wait on the semaphore to make sure the data is ready
  if (threadIdx.x % WARP_SIZE == 0) portChan.wait();
}

__device__ void localAllGather(DeviceHandle<mscclpp::PortChannel> portChan, int rank, int nRanksPerNode, int remoteRank,
                               uint64_t offset, uint64_t size, bool flushAfterSignal = true) {
  // this allgather algorithm works as follows:
  // Step 1: GPU rank i sends data to GPU rank (i+1) % nRanksPerNode
  // and waits for data from GPU rank (i-1) % nRanksPerNode
  // Step 2: GPU rank i sends data to GPU rank (i+2) % nRanksPerNode
  // ...
  // This order is much better for DMA engine for NVLinks
  for (int i = 1; i < nRanksPerNode; i++) {
    if ((remoteRank % nRanksPerNode) == ((rank + i) % nRanksPerNode)) {
      // put your data to GPU (rank+i) % nRanksPerNode and signal in one call
      if (flushAfterSignal && (threadIdx.x % WARP_SIZE) == 0) portChan.putWithSignalAndFlush(offset, size);
      if (!flushAfterSignal && (threadIdx.x % WARP_SIZE) == 0) portChan.putWithSignal(offset, size);
    }
    // wait for the data from GPU (rank-i) % nRanksPerNode to arrive
    if ((remoteRank % nRanksPerNode) == ((rank - i + nRanksPerNode) % nRanksPerNode)) {
      if ((threadIdx.x % WARP_SIZE) == 0) portChan.wait();
    }
#if defined(__HIP_PLATFORM_AMD__)
    // NOTE: we actually need a group barrier here for better performance, but __syncthreads() is still correct.
    __syncthreads();
#else
    asm volatile("bar.sync %0, %1;" ::"r"(11), "r"((nRanksPerNode - 1) * WARP_SIZE) : "memory");
#endif
  }
}

__device__ mscclpp::DeviceSyncer deviceSyncer;

// This kernel is the most performant when the number of blocks is a multiple of (nRanksPerNode - 1).
__device__ void localAllGatherMem(int rank, int nRanksPerNode, int startRankChunkIndex, uint64_t offsetInRankChunk,
                                  uint64_t rankChunkSize, uint64_t size, size_t nBlocks) {
  if (nRanksPerNode == 1) return;
  if (blockIdx.x >= nBlocks) return;
  const size_t nPeer = nRanksPerNode - 1;
  const size_t peerIdx = blockIdx.x % nPeer;
  const size_t nBlockForThisPeer = nBlocks / nPeer + (nBlocks % nPeer > peerIdx ? 1 : 0);
  const size_t peerLocalBlockIdx = blockIdx.x / nPeer;
  const size_t rankLocalIndex = rank % nRanksPerNode;
  const int remoteRankLocalIndex = (peerIdx < rankLocalIndex ? peerIdx : peerIdx + 1);

  // Split the data into chunks for aligned data access. Ignore the remainder here and let the last block handle it.
  constexpr size_t chunkBytes = 128;  // heuristic value
  const size_t nChunk = size / chunkBytes;
  const size_t nMinChunkPerBlock = nChunk / nBlockForThisPeer;
  const size_t nRemainderChunk = nChunk % nBlockForThisPeer;

  // Distribute chunks to blocks
  size_t nChunkForThisBlock;
  size_t offsetForThisBlock;
  if (peerLocalBlockIdx < nRemainderChunk) {
    nChunkForThisBlock = nMinChunkPerBlock + 1;
    offsetForThisBlock = (nMinChunkPerBlock + 1) * peerLocalBlockIdx;
  } else {
    nChunkForThisBlock = nMinChunkPerBlock;
    offsetForThisBlock =
        (nMinChunkPerBlock + 1) * nRemainderChunk + (peerLocalBlockIdx - nRemainderChunk) * nMinChunkPerBlock;
  }
  offsetForThisBlock *= chunkBytes;

  // Calculate the size of the data for this block
  size_t sizeForThisBlock = nChunkForThisBlock * chunkBytes;
  const size_t lastChunkSize = size - nChunk * chunkBytes;
  if (lastChunkSize > 0 && peerLocalBlockIdx == nBlockForThisPeer - 1) {
    sizeForThisBlock += lastChunkSize;
  }
  if (threadIdx.x == 0 && peerLocalBlockIdx == 0) {
    constMemChans[peerIdx].signal();
    constMemChans[peerIdx].wait();
  }
  deviceSyncer.sync(nBlocks);
  size_t offset = rankChunkSize * (startRankChunkIndex + remoteRankLocalIndex) + offsetInRankChunk;
  constMemChans[peerIdx].get(offset + offsetForThisBlock, sizeForThisBlock, threadIdx.x, blockDim.x);
}

__global__ void __launch_bounds__(1024) allgather1(int rank, int nRanksPerNode, size_t nelemsPerGPU) {
  int warpId = threadIdx.x / WARP_SIZE;
  int remoteRank = (warpId < rank) ? warpId : warpId + 1;

  // Each warp is responsible for one of the remote ranks
  DeviceHandle<mscclpp::PortChannel> portChan = constPortChans[warpId];

  localAllGather(portChan, rank, nRanksPerNode, remoteRank, rank * nelemsPerGPU * sizeof(int),
                 nelemsPerGPU * sizeof(int));
}

__global__ void __launch_bounds__(1024) allgather2(int rank, int worldSize, int nRanksPerNode, size_t nelemsPerGPU) {
  int warpId = threadIdx.x / WARP_SIZE;
  int remoteRank = (warpId < rank) ? warpId : warpId + 1;

  // Each warp is responsible for one of the remote ranks
  DeviceHandle<mscclpp::PortChannel> portChan = constPortChans[warpId];

  // this allgather is a pipelined and hierarchical one and only works for two nodes
  // it is implemented as follows:
  // Step 1: each node does a local allgather and concurrently,
  // local GPU i exchange (piplineSize-1)/pipelineSize portion of their data with
  // its cross-node neighbor (local GPU i on the other node) via IB
  // Step 2: each node does a local allgather again with the data just received from its
  // cross-node neighbor in step 1, and concurrently, exchange the rest of the data with
  // its cross-node neighbor
  // Step 3: each node does a local allgather for the last time with the rest of the data

  int pipelineSize = 3;

  // Step 1
  // local allgather
  if (remoteRank / nRanksPerNode == rank / nRanksPerNode) {
    localAllGather(portChan, rank, nRanksPerNode, remoteRank, rank * nelemsPerGPU * sizeof(int),
                   nelemsPerGPU * sizeof(int), false);
  }
  // cross-node exchange
  if (remoteRank % nRanksPerNode == rank % nRanksPerNode) {
    // opposite side
    if ((threadIdx.x % WARP_SIZE) == 0)
      portChan.putWithSignal(rank * nelemsPerGPU * sizeof(int),
                             (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize * sizeof(int));
    if ((threadIdx.x % WARP_SIZE) == 0) portChan.wait();
  }

  // sync here to make sure IB flush dose not block the CUDA IPC traffic
  __syncthreads();
  // need to flush ib channel here to avoid cq overflow. since we won't change send suffer after send, we don't need
  // to flush for IPC channel.
  if (remoteRank % nRanksPerNode == rank % nRanksPerNode) {
    if ((threadIdx.x % WARP_SIZE) == 0) portChan.flush();
  }
  __syncthreads();

  // Step 2
  // local allgather
  int otherNghr = (rank + nRanksPerNode) % worldSize;
  if (remoteRank / nRanksPerNode == rank / nRanksPerNode) {
    localAllGather(portChan, rank, nRanksPerNode, remoteRank, otherNghr * nelemsPerGPU * sizeof(int),
                   (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize * sizeof(int), false);
  }

  // cross-node exchange
  if (remoteRank % nRanksPerNode == rank % nRanksPerNode) {
    // opposite side
    if ((threadIdx.x % WARP_SIZE) == 0)
      portChan.putWithSignal((rank * nelemsPerGPU + (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize) * sizeof(int),
                             nelemsPerGPU / pipelineSize * sizeof(int));
    if ((threadIdx.x % WARP_SIZE) == 0) portChan.wait();
  }

  __syncthreads();
  if (remoteRank % nRanksPerNode == rank % nRanksPerNode) {
    if ((threadIdx.x % WARP_SIZE) == 0) portChan.flush();
  }
  __syncthreads();

  // Step 3
  // local allgather
  if (remoteRank / nRanksPerNode == rank / nRanksPerNode) {
    localAllGather(portChan, rank, nRanksPerNode, remoteRank,
                   (otherNghr * nelemsPerGPU + (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize) * sizeof(int),
                   nelemsPerGPU / pipelineSize * sizeof(int));
  }
}

__global__ void __launch_bounds__(1024) allgather3() {
  int warpId = threadIdx.x / WARP_SIZE;

  // Each warp is responsible for one of the remote ranks
  DeviceHandle<mscclpp::BasePortChannel> portChan = constRawPortChan[warpId];

  int tid = threadIdx.x;
  __syncthreads();
  if (tid == 0) {
    mscclpp::ProxyTrigger trigger;
    trigger.fst = MAGIC;
    trigger.snd = 0;
    // offload all the work to the proxy
    uint64_t currentFifoHead = portChan.fifo_.push(trigger);
    // wait for the work to be done in cpu side
    portChan.fifo_.sync(currentFifoHead);
  }
  if (tid % WARP_SIZE == 0) {
    portChan.wait();
  }
}

__global__ void __launch_bounds__(1024) allgather4(int rank, int worldSize, int nRanksPerNode, size_t nelemsPerGPU) {
  // this allgather is a pipelined and hierarchical one and only works for two nodes
  // it is implemented as follows:
  // Step 1: each node does a local allgather and concurrently,
  // local GPU i exchange (piplineSize-1)/pipelineSize portion of their data with
  // its cross-node neighbor (local GPU i on the other node) via IB
  // Step 2: each node does a local allgather again with the data just received from its
  // cross-node neighbor in step 1, and concurrently, exchange the rest of the data with
  // its cross-node neighbor
  // Step 3: each node does a local allgather for the last time with the rest of the data

  int pipelineSize = 3;
  int peerRank = (rank + nRanksPerNode) % worldSize;
  int peerNodeId = peerRank / nRanksPerNode;
  int peer = (peerRank < rank) ? peerRank : peerRank - 1;
  DeviceHandle<mscclpp::PortChannel>& portChan = constPortChans[peer];
  const size_t nBlocksForLocalAllGather = gridDim.x;
  const size_t rankChunkSize = nelemsPerGPU * sizeof(int);
  const int startRankIndexInLocalNode = (rank / nRanksPerNode) * nRanksPerNode;
  const int startRankIndexInPeerNode = (peerRank / nRanksPerNode) * nRanksPerNode;

  if (peerNodeId == rank / nRanksPerNode) {
    localAllGatherMem(rank, nRanksPerNode, 0, 0, rankChunkSize, rankChunkSize, nBlocksForLocalAllGather);
    return;
  }

  constexpr size_t alignment = 128;
  size_t step1Bytes = (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize * sizeof(int);
  step1Bytes = step1Bytes / alignment * alignment;
  const size_t step2Bytes = nelemsPerGPU * sizeof(int) - step1Bytes;

  // Step 1
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    portChan.putWithSignal(rank * nelemsPerGPU * sizeof(int), step1Bytes);
  }
  localAllGatherMem(rank, nRanksPerNode, startRankIndexInLocalNode, 0, rankChunkSize, rankChunkSize,
                    nBlocksForLocalAllGather);
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    portChan.wait();
    portChan.flush();
  }
  deviceSyncer.sync(nBlocksForLocalAllGather);
  // Step 2
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    portChan.putWithSignal(rank * nelemsPerGPU * sizeof(int) + step1Bytes, step2Bytes);
  }
  localAllGatherMem(rank, nRanksPerNode, startRankIndexInPeerNode, 0, rankChunkSize, step1Bytes,
                    nBlocksForLocalAllGather);
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    portChan.wait();
    portChan.flush();
  }
  deviceSyncer.sync(nBlocksForLocalAllGather);
  // Step 3
  localAllGatherMem(rank, nRanksPerNode, startRankIndexInPeerNode, step1Bytes, rankChunkSize, step2Bytes,
                    nBlocksForLocalAllGather);
}

__global__ void __launch_bounds__(1024, 1)
    allgather5(size_t rank, [[maybe_unused]] size_t worldSize, size_t nRanksPerNode, size_t nelemsPerGPU) {
  const size_t nBlock = gridDim.x;
  if (blockIdx.x >= nBlock) return;

  const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t lid = tid % WARP_SIZE;
  const size_t wid = tid / WARP_SIZE;

  const size_t nThread = blockDim.x * nBlock;
  const size_t nWarp = nThread / WARP_SIZE;
  const size_t nPeer = nRanksPerNode - 1;
  const size_t chanOffset = nPeer * blockIdx.x;
  auto memChans = constMemChans + chanOffset;

  if (wid < nPeer && lid == 0) {
    memChans[wid].relaxedSignal();
    memChans[wid].wait();
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
    const size_t remoteRankLocalIndex = (peerIdx < rank ? peerIdx : peerIdx + 1);
    const size_t offset = bytesPerGPU * remoteRankLocalIndex + (wid / nPeer) * unitBytesPerWarp;
    memChans[peerIdx].get<16, false>(offset, unitBytesPerWarp, lid, WARP_SIZE);
  }

  for (size_t i = 1; i < nLoop; ++i) {
    const size_t gWid = wid + i * nWarp;
    const size_t peerIdx = gWid % nPeer;
    const size_t remoteRankLocalIndex = (peerIdx < rank ? peerIdx : peerIdx + 1);
    const size_t offset = bytesPerGPU * remoteRankLocalIndex + (gWid / nPeer) * unitBytesPerWarp;
    memChans[peerIdx].get<16, false>(offset, unitBytesPerWarp, lid, WARP_SIZE);
  }

  if (bytes % unitBytes > 0) {
    const size_t gWid = wid + nLoop * nWarp;
    const size_t peerIdx = gWid % nPeer;
    const size_t remoteRankLocalIndex = (peerIdx < rank ? peerIdx : peerIdx + 1);
    const size_t offsetWithinRank = (gWid / nPeer) * unitBytesPerWarp;
    const size_t offset = bytesPerGPU * remoteRankLocalIndex + offsetWithinRank;
    const size_t remainBytes = (offsetWithinRank + unitBytesPerWarp > bytesPerGPU)
                                   ? ((bytesPerGPU > offsetWithinRank) ? (bytesPerGPU - offsetWithinRank) : 0)
                                   : unitBytesPerWarp;
    if (remainBytes > 0) {
      memChans[peerIdx].get<16, true>(offset, remainBytes, lid, WARP_SIZE);
    }
  }
}

__global__ void __launch_bounds__(1024, 1)
    allgather6(size_t rank, [[maybe_unused]] size_t worldSize, size_t nRanksPerNode, size_t nelemsPerGPU) {
  const size_t nBlock = gridDim.x;
  if (blockIdx.x >= nBlock) return;

  const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t lid = tid % WARP_SIZE;
  const size_t wid = tid / WARP_SIZE;

  const size_t nThread = blockDim.x * nBlock;
  const size_t nWarp = nThread / WARP_SIZE;
  const size_t nPeer = nRanksPerNode - 1;
  const size_t chanOffset = nPeer * blockIdx.x;
  auto memChans = constMemChans + chanOffset;

  if (wid < nPeer && lid == 0) {
    memChans[wid].relaxedSignal();
    memChans[wid].wait();
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
    memChans[peerIdx].put<16, false>(offset, unitBytesPerWarp, lid, WARP_SIZE);
  }

  for (size_t i = 1; i < nLoop; ++i) {
    const size_t gWid = wid + i * nWarp;
    const size_t peerIdx = gWid % nPeer;
    const size_t offset = bytesPerGPU * rank + (gWid / nPeer) * unitBytesPerWarp;
    memChans[peerIdx].put<16, false>(offset, unitBytesPerWarp, lid, WARP_SIZE);
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
      memChans[peerIdx].put<16, true>(offset, remainBytes, lid, WARP_SIZE);
    }
  }
}

__global__ void __launch_bounds__(1024, 1)
    allgather7(size_t rank, [[maybe_unused]] size_t worldSize, size_t nRanksPerNode, size_t nelemsPerGPU) {
  const size_t nBlock = gridDim.x;
  if (blockIdx.x >= nBlock) return;

  const size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  const size_t lid = tid % WARP_SIZE;
  const size_t wid = tid / WARP_SIZE;

  const size_t nThread = blockDim.x * nBlock;
  const size_t nWarp = nThread / WARP_SIZE;
  const size_t nPeer = nRanksPerNode - 1;
  auto memChans = constMemOutOfPlaceChans;

  const uint32_t flag = (uint32_t)globalFlag;
  const size_t bytesPerGPU = nelemsPerGPU * sizeof(int);
  const size_t bytes = bytesPerGPU * nPeer;
  size_t unitBytesPerThread = 8;
  const size_t unitBytesPerWarp = unitBytesPerThread * WARP_SIZE;
  const size_t unitBytes = unitBytesPerWarp * nWarp;
  const size_t nLoop = bytes / unitBytes;

  // double buffering
  const size_t scratchOffset = (flag & 1) ? 0 : bytesPerGPU * nRanksPerNode * 2;

  if (nLoop > 0) {
    // First loop unrolling
    const size_t peerIdx = wid % nPeer;
    const size_t offset = bytesPerGPU * rank + (wid / nPeer) * unitBytesPerWarp;
    memChans[peerIdx].putPackets(scratchOffset + offset * 2, offset, unitBytesPerWarp, lid, WARP_SIZE, flag);
  }

  if (nLoop > 0) {
    // First loop unrolling
    const size_t peerIdx = wid % nPeer;
    const size_t remoteRankLocalIndex = (peerIdx < rank ? peerIdx : peerIdx + 1);
    const size_t offset = bytesPerGPU * remoteRankLocalIndex + (wid / nPeer) * unitBytesPerWarp;
    memChans[peerIdx].unpackPackets(scratchOffset + offset * 2, offset, unitBytesPerWarp, lid, WARP_SIZE, flag);
  }

  for (size_t i = 1; i < nLoop; ++i) {
    const size_t gWid = wid + i * nWarp;
    const size_t peerIdx = gWid % nPeer;
    const size_t offset = bytesPerGPU * rank + (gWid / nPeer) * unitBytesPerWarp;
    memChans[peerIdx].putPackets(scratchOffset + offset * 2, offset, unitBytesPerWarp, lid, WARP_SIZE, flag);
  }

  for (size_t i = 1; i < nLoop; ++i) {
    const size_t gWid = wid + i * nWarp;
    const size_t peerIdx = gWid % nPeer;
    const size_t remoteRankLocalIndex = (peerIdx < rank ? peerIdx : peerIdx + 1);
    const size_t offset = bytesPerGPU * remoteRankLocalIndex + (gWid / nPeer) * unitBytesPerWarp;
    memChans[peerIdx].unpackPackets(scratchOffset + offset * 2, offset, unitBytesPerWarp, lid, WARP_SIZE, flag);
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
      memChans[peerIdx].putPackets(scratchOffset + offset * 2, offset, remainBytes, lid, WARP_SIZE, flag);
    }
  }
  if (bytes % unitBytes > 0) {
    const size_t gWid = wid + nLoop * nWarp;
    const size_t peerIdx = gWid % nPeer;
    const size_t remoteRankLocalIndex = (peerIdx < rank ? peerIdx : peerIdx + 1);
    const size_t offsetWithinRank = (gWid / nPeer) * unitBytesPerWarp;
    const size_t offset = bytesPerGPU * remoteRankLocalIndex + offsetWithinRank;
    const size_t remainBytes = (offsetWithinRank + unitBytesPerWarp > bytesPerGPU)
                                   ? ((bytesPerGPU > offsetWithinRank) ? (bytesPerGPU - offsetWithinRank) : 0)
                                   : unitBytesPerWarp;
    if (remainBytes > 0) {
      memChans[peerIdx].unpackPackets(scratchOffset + offset * 2, offset, remainBytes, lid, WARP_SIZE, flag);
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    globalFlag += 1;
  }
}

class AllGatherProxyService : public mscclpp::BaseProxyService {
 public:
  AllGatherProxyService(int worldSize, int rank, int cudaDevice);
  void startProxy(bool blocking = false) override { proxy_->start(blocking); }
  void stopProxy() override { proxy_->stop(); }
  void setSendBytes(size_t sendBytes) { this->sendBytes_ = sendBytes; }
  void addRemoteMemory(mscclpp::RegisteredMemory memory) { remoteMemories_.push_back(memory); }
  void setLocalMemory(mscclpp::RegisteredMemory memory) { localMemory_ = memory; }
  mscclpp::SemaphoreId buildAndAddSemaphore(mscclpp::Communicator& communicator,
                                            const mscclpp::Connection& connection) {
    semaphores_.push_back(std::make_shared<mscclpp::Host2DeviceSemaphore>(communicator, connection));
    return semaphores_.size() - 1;
  }
  std::vector<DeviceHandle<mscclpp::BasePortChannel>> portChannels() {
    std::vector<DeviceHandle<mscclpp::BasePortChannel>> result;
    for (auto& semaphore : semaphores_) {
      result.push_back(mscclpp::deviceHandle(mscclpp::BasePortChannel(0, semaphore, proxy_)));
    }
    return result;
  }

 private:
  int worldSize_;
  int rank_;
  int cudaDevice_;
  size_t sendBytes_;

  std::shared_ptr<mscclpp::Proxy> proxy_;
  std::vector<std::shared_ptr<mscclpp::Host2DeviceSemaphore>> semaphores_;
  std::vector<mscclpp::RegisteredMemory> remoteMemories_;
  mscclpp::RegisteredMemory localMemory_;

  mscclpp::ProxyHandlerResult handleTrigger(mscclpp::ProxyTrigger triggerRaw);
};

AllGatherProxyService::AllGatherProxyService(int worldSize, int rank, int cudaDevice)
    : worldSize_(worldSize), rank_(rank), cudaDevice_(cudaDevice), sendBytes_(0) {
  MSCCLPP_CUDATHROW(cudaSetDevice(cudaDevice));
  auto handlerFunc = [&](mscclpp::ProxyTrigger triggerRaw) { return handleTrigger(triggerRaw); };
  proxy_ = std::make_shared<mscclpp::Proxy>(handlerFunc);
}

mscclpp::ProxyHandlerResult AllGatherProxyService::handleTrigger(mscclpp::ProxyTrigger triggerRaw) {
  size_t offset = rank_ * sendBytes_;
  if (triggerRaw.fst != MAGIC) {
    // this is not a valid trigger
    throw std::runtime_error("Invalid trigger");
  }
  for (int i = 0; i < worldSize_; i++) {
    int r = (rank_ + i) % worldSize_;
    if (r == rank_) {
      continue;
    }
    int index = (r < rank_) ? r : r - 1;
    semaphores_[index]->connection().write(remoteMemories_[index], offset, localMemory_, offset, sendBytes_);
    semaphores_[index]->signal();
  }
  bool flushIpc = false;
  for (auto& semaphore : semaphores_) {
    auto& conn = semaphore->connection();
    if (conn.transport() == mscclpp::Transport::CudaIpc && !flushIpc) {
      // since all the cudaIpc channels are using the same cuda stream, we only need to flush one of them
      conn.flush();
      flushIpc = true;
    }
    if (mscclpp::AllIBTransports.has(conn.transport())) {
      conn.flush();
    }
  }
  return mscclpp::ProxyHandlerResult::Continue;
}

class AllGatherTestColl : public BaseTestColl {
 public:
  AllGatherTestColl() = default;
  ~AllGatherTestColl() override = default;

  void runColl(const TestArgs& args, cudaStream_t stream) override;
  void initData(const TestArgs& args, std::vector<void*> sendBuff, void* expectedBuff) override;
  void getBw(const double deltaSec, double& algBw /*OUT*/, double& busBw /*OUT*/) override;
  void setupCollTest(size_t size) override;
  std::vector<KernelRestriction> getKernelRestrictions() override;
};

void AllGatherTestColl::runColl(const TestArgs& args, cudaStream_t stream) {
  const int worldSize = args.totalRanks;
  const int rank = args.rank;
  const int nRanksPerNode = args.nRanksPerNode;
  const int kernelNum = args.kernelNum;
  int nBlocks;
  int nThreads;
  if (kernelNum == 4) {
    nBlocks = 21;
    nThreads = 1024;
  } else if (kernelNum == 5) {
    nBlocks = 24;
    nThreads = 1024;
  } else if (kernelNum == 6) {
    nBlocks = 24;
    nThreads = 1024;
  } else if (kernelNum == 7) {
    nBlocks = 4;
    nThreads = 896;
  } else {
    nBlocks = 1;
    nThreads = WARP_SIZE * (worldSize - 1);
  }
  if (kernelNum == 0) {
    allgather0<<<nBlocks, nThreads, 0, stream>>>(rank, paramCount_);
  } else if (kernelNum == 1) {
    allgather1<<<nBlocks, nThreads, 0, stream>>>(rank, nRanksPerNode, paramCount_);
  } else if (kernelNum == 2) {
    allgather2<<<nBlocks, nThreads, 0, stream>>>(rank, worldSize, nRanksPerNode, paramCount_);
  } else if (kernelNum == 3) {
    allgather3<<<nBlocks, nThreads, 0, stream>>>();
  } else if (kernelNum == 4) {
    allgather4<<<nBlocks, nThreads, 0, stream>>>(rank, worldSize, nRanksPerNode, paramCount_);
  } else if (kernelNum == 5) {
    allgather5<<<nBlocks, nThreads, 0, stream>>>(rank, worldSize, nRanksPerNode, paramCount_);
  } else if (kernelNum == 6) {
    allgather6<<<nBlocks, nThreads, 0, stream>>>(rank, worldSize, nRanksPerNode, paramCount_);
  } else if (kernelNum == 7) {
    allgather7<<<nBlocks, nThreads, 0, stream>>>(rank, worldSize, nRanksPerNode, paramCount_);
  }
}

void AllGatherTestColl::initData(const TestArgs& args, std::vector<void*> sendBuff, void* expectedBuff) {
  if (sendBuff.size() != 1) std::runtime_error("unexpected error");
  int rank = args.rank;
  std::vector<int> dataHost(std::max(sendCount_, recvCount_), 0);
  for (size_t i = 0; i < recvCount_; i++) {
    int val = i + 1;
    if (i / sendCount_ == (size_t)rank) {
      dataHost[i] = val;
    } else {
      dataHost[i] = 0;
    }
  }
  CUDATHROW(cudaMemcpy(sendBuff[0], dataHost.data(), recvCount_ * typeSize_, cudaMemcpyHostToDevice));

  for (size_t i = 0; i < recvCount_; i++) {
    dataHost[i] = static_cast<int>(i) + 1;
  }
  std::memcpy(expectedBuff, dataHost.data(), recvCount_ * typeSize_);
}

void AllGatherTestColl::getBw(const double deltaSec, double& algBw, double& busBw) {
  double baseBw = (double)(paramCount_ * typeSize_ * worldSize_) / 1.0E9 / deltaSec;

  algBw = baseBw;
  double factor = ((double)(worldSize_ - 1)) / ((double)worldSize_);
  busBw = baseBw * factor;
}

void AllGatherTestColl::setupCollTest(size_t size) {
  size_t count = size / typeSize_;
  size_t base = (count / worldSize_);
  sendCount_ = base;
  recvCount_ = base * worldSize_;
  paramCount_ = base;
  expectedCount_ = recvCount_;
  if (isUsingHostOffload(kernelNum_)) {
    auto service = std::dynamic_pointer_cast<AllGatherProxyService>(chanService_);
    service->setSendBytes(sendCount_ * typeSize_);
  }
  mscclpp::DeviceSyncer syncer = {};
  CUDATHROW(cudaMemcpyToSymbol(deviceSyncer, &syncer, sizeof(mscclpp::DeviceSyncer)));
}

std::vector<KernelRestriction> AllGatherTestColl::getKernelRestrictions() {
  return {// {kernelNum, kernelName, compatibleWithMultiNodes, countDivisorForMultiNodes, alignedBytes}
          {0, "allgather0", true, 1, 4 * worldSize_},
          {1, "allgather1", false, 1, 4 * worldSize_},
          {2, "allgather2", true, 3, 4 * worldSize_},
          {3, "allgather3", true, 1, 4 * worldSize_},
          {4, "allgather4", true, 3, 16 * worldSize_ /*use ulong2 to transfer data*/},
          {5, "allgather5", false, 1, 16 * worldSize_ /*use ulong2 to transfer data*/},
          {6, "allgather6", false, 1, 16 * worldSize_ /*use ulong2 to transfer data*/},
          {7, "allgather7", false, 1, 16 * worldSize_ /*use ulong2 to transfer data*/}};
}

class AllGatherTestEngine : public BaseTestEngine {
 public:
  AllGatherTestEngine(const TestArgs& args);
  ~AllGatherTestEngine() override = default;

  void allocateBuffer() override;
  void setupConnections() override;

  std::vector<void*> getSendBuff() override;
  void* getRecvBuff() override;
  void* getScratchBuff() override;
  std::shared_ptr<mscclpp::BaseProxyService> createProxyService() override;

 private:
  void* getExpectedBuff() override;

  std::shared_ptr<int> sendBuff_;
  std::shared_ptr<int[]> expectedBuff_;
  std::shared_ptr<mscclpp::LLPacket> scratchPacketBuff_;
  std::vector<mscclpp::MemoryChannel> memoryChannels_;
  std::vector<mscclpp::MemoryChannel> memoryOutOfPlaceChannels_;
};

AllGatherTestEngine::AllGatherTestEngine(const TestArgs& args) : BaseTestEngine(args, "allgather") {}

void AllGatherTestEngine::allocateBuffer() {
  sendBuff_ = mscclpp::GpuBuffer<int>(args_.maxBytes / sizeof(int)).memory();
  expectedBuff_ = std::shared_ptr<int[]>(new int[args_.maxBytes / sizeof(int)]);
  if (args_.kernelNum == 7) {
    const size_t nPacket = (args_.maxBytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);
    // 2x for double-buffering, scratchBuff used to store original data and reduced results
    const size_t scratchBuffNelem = nPacket * 2 /*original data & reduced result */ * 2 /* double buffering*/;
    scratchPacketBuff_ = mscclpp::GpuBuffer<mscclpp::LLPacket>(scratchBuffNelem).memory();
  }
}

void AllGatherTestEngine::setupConnections() {
  std::vector<DeviceHandle<mscclpp::PortChannel>> devPortChannels;
  if (!isUsingHostOffload(args_.kernelNum)) {
    setupMeshConnections(devPortChannels, sendBuff_.get(), args_.maxBytes);
    if (devPortChannels.size() > sizeof(constPortChans) / sizeof(DeviceHandle<mscclpp::PortChannel>)) {
      std::runtime_error("unexpected error");
    }
    CUDATHROW(cudaMemcpyToSymbol(constPortChans, devPortChannels.data(),
                                 sizeof(DeviceHandle<mscclpp::PortChannel>) * devPortChannels.size()));

    setupMeshConnections(memoryChannels_, sendBuff_.get(), args_.maxBytes, nullptr, 0, ChannelSemantic::PUT, 64);
    std::vector<DeviceHandle<mscclpp::MemoryChannel>> memoryChannelHandles(memoryChannels_.size());
    if (memoryChannels_.size() > sizeof(constMemChans) / sizeof(DeviceHandle<mscclpp::MemoryChannel>)) {
      std::runtime_error("unexpected error");
    }
    std::transform(memoryChannels_.begin(), memoryChannels_.end(), memoryChannelHandles.begin(),
                   [](const mscclpp::MemoryChannel& memoryChannel) { return mscclpp::deviceHandle(memoryChannel); });
    CUDATHROW(cudaMemcpyToSymbol(constMemChans, memoryChannelHandles.data(),
                                 sizeof(DeviceHandle<mscclpp::MemoryChannel>) * memoryChannelHandles.size()));

    if (args_.kernelNum == 7) {
      const size_t nPacket = (args_.maxBytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);
      const size_t scratchPacketBuffBytes = nPacket * 2 * 2 * sizeof(mscclpp::LLPacket);
      setupMeshConnections(memoryOutOfPlaceChannels_, sendBuff_.get(), args_.maxBytes, scratchPacketBuff_.get(),
                           scratchPacketBuffBytes);
      std::vector<DeviceHandle<mscclpp::MemoryChannel>> memoryOutOfPlaceChannelHandles(
          memoryOutOfPlaceChannels_.size());
      if (memoryOutOfPlaceChannels_.size() >
          sizeof(constMemOutOfPlaceChans) / sizeof(DeviceHandle<mscclpp::MemoryChannel>)) {
        std::runtime_error("unexpected error");
      }
      std::transform(memoryOutOfPlaceChannels_.begin(), memoryOutOfPlaceChannels_.end(),
                     memoryOutOfPlaceChannelHandles.begin(),
                     [](const mscclpp::MemoryChannel& memoryChannel) { return mscclpp::deviceHandle(memoryChannel); });
      CUDATHROW(
          cudaMemcpyToSymbol(constMemOutOfPlaceChans, memoryOutOfPlaceChannelHandles.data(),
                             sizeof(DeviceHandle<mscclpp::MemoryChannel>) * memoryOutOfPlaceChannelHandles.size()));
    }
  } else {
    auto service = std::dynamic_pointer_cast<AllGatherProxyService>(chanService_);
    setupMeshConnections(devPortChannels, sendBuff_.get(), args_.maxBytes, nullptr, 0,
                         [&](std::vector<mscclpp::Connection> conns,
                             std::vector<std::shared_future<mscclpp::RegisteredMemory>>& remoteMemories,
                             const mscclpp::RegisteredMemory& localMemory) {
                           std::vector<mscclpp::SemaphoreId> semaphoreIds;
                           for (size_t i = 0; i < conns.size(); ++i) {
                             service->buildAndAddSemaphore(*comm_, conns[i]);
                             service->addRemoteMemory(remoteMemories[i].get());
                           }
                           service->setLocalMemory(localMemory);
                         });
    auto portChannels = service->portChannels();
    if (portChannels.size() > sizeof(constRawPortChan) / sizeof(DeviceHandle<mscclpp::BasePortChannel>)) {
      std::runtime_error("unexpected error");
    }
    CUDATHROW(cudaMemcpyToSymbol(constRawPortChan, portChannels.data(),
                                 sizeof(DeviceHandle<mscclpp::BasePortChannel>) * portChannels.size()));
  }
}

std::shared_ptr<mscclpp::BaseProxyService> AllGatherTestEngine::createProxyService() {
  if (isUsingHostOffload(args_.kernelNum)) {
    return std::make_shared<AllGatherProxyService>(args_.totalRanks, args_.rank, args_.gpuNum);
  } else {
    return std::make_shared<mscclpp::ProxyService>();
  }
}

std::vector<void*> AllGatherTestEngine::getSendBuff() { return {sendBuff_.get()}; }

void* AllGatherTestEngine::getExpectedBuff() { return expectedBuff_.get(); }

void* AllGatherTestEngine::getRecvBuff() {
  // in-place operation reuse the send buffer
  return sendBuff_.get();
}

void* AllGatherTestEngine::getScratchBuff() { return nullptr; }

std::shared_ptr<BaseTestEngine> getTestEngine(const TestArgs& args) {
  return std::make_shared<AllGatherTestEngine>(args);
}

std::shared_ptr<BaseTestColl> getTestColl() { return std::make_shared<AllGatherTestColl>(); }
