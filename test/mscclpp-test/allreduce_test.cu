// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>
#include <mscclpp/concurrency.hpp>
#include <mscclpp/packet.hpp>
#include <vector>

#include "common.hpp"

#define ALIGN 4
#define BLOCKS_PER_PEER 1

__constant__ mscclpp::SimpleProxyChannel constDevFstRoundChans[16];
__constant__ mscclpp::SimpleProxyChannel constDevSndRoundChans[16];

__constant__ mscclpp::SmChannel constSmInPlaceChans[8];
__constant__ mscclpp::SmChannel constSmOutOfPlaceChans[8];
__device__ uint64_t globalFlag;

// TODO(chhwang): need an interface for this.
static void* inputBuff = nullptr;
static void* resultBuff = nullptr;
static void* scratchBuff = nullptr;
static void* scratchPacketBuff = nullptr;
static void* putPacketBuff = nullptr;
static void* getPacketBuff = nullptr;

struct Chunk {
  size_t offset;
  size_t size;
};

__host__ __device__ Chunk getChunk(size_t dataCount, size_t numChunks, size_t chunkIdx) {
  size_t remainder = dataCount % numChunks;
  size_t smallChunkSize = dataCount / numChunks;
  size_t largeChunkSize = smallChunkSize + 1;
  size_t numRemainedLargeChunks = chunkIdx < remainder ? remainder - chunkIdx : 0;
  size_t offset = (remainder - numRemainedLargeChunks) * largeChunkSize +
                  (chunkIdx > remainder ? chunkIdx - remainder : 0) * smallChunkSize;
  return Chunk{offset, chunkIdx < remainder ? largeChunkSize : smallChunkSize};
}

__forceinline__ __device__ void vectorSum(int* dst, int* src, size_t nElem) {
  size_t nInt4 = nElem / 4;
  size_t nLastInts = nElem % 4;
  int4* dst4 = (int4*)dst;
  int4* src4 = (int4*)src;
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < nInt4; i += blockDim.x * gridDim.x) {
    dst4[i].w += src4[i].w;
    dst4[i].x += src4[i].x;
    dst4[i].y += src4[i].y;
    dst4[i].z += src4[i].z;
  }
  if (nLastInts > 0) {
    int* dstLast = dst + nInt4 * 4;
    int* srcLast = src + nInt4 * 4;
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < nLastInts; i += blockDim.x * gridDim.x) {
      dstLast[i] += srcLast[i];
    }
  }
}

__device__ void vectorSumSingleBlock(int* dst, int* src, size_t nElem) {
  for (int i = threadIdx.x; i < nElem; i += blockDim.x) {
    dst[i] += src[i];
  }
}

__device__ mscclpp::DeviceSyncer deviceSyncer;

__device__ void localReduceScatter(int* buff, int* scratch, int rank, int nRanksPerNode, int startChunkIndex,
                                   size_t offsetInChunk, size_t chunkSize, size_t nelems) {
  if (nRanksPerNode == 1) {
    return;
  }
  int isComm = (threadIdx.x == 0) && (blockIdx.x == 0);
  int startRankInNode = (rank / nRanksPerNode) * nRanksPerNode;
  int rankIdexInNode = rank % nRanksPerNode;

  for (int i = 1; i < nRanksPerNode; ++i) {
    int remoteSendToRank = (rank + i) % nRanksPerNode + startRankInNode;
    int remoteRecvFromRank = (rank + nRanksPerNode - i) % nRanksPerNode + startRankInNode;
    int peerSendId = (remoteSendToRank < rank) ? remoteSendToRank : remoteSendToRank - 1;
    int peerRecvId = (remoteRecvFromRank < rank) ? remoteRecvFromRank : remoteRecvFromRank - 1;

    mscclpp::SimpleProxyChannel& devFstSendChan = constDevFstRoundChans[peerSendId];
    mscclpp::SimpleProxyChannel& devFstRecvChan = constDevFstRoundChans[peerRecvId];
    size_t srcOffset =
        (((rankIdexInNode + i) % nRanksPerNode + startChunkIndex) * chunkSize + offsetInChunk) * sizeof(int);
    size_t dstOffset = rank * chunkSize * sizeof(int);

    if (i == 1) {
      if (isComm) {
        devFstSendChan.putWithSignal(dstOffset, srcOffset, nelems * sizeof(int));
      }
    } else {
      int pre = i - 1;
      int preRemoteRecvFromRank = (rank + nRanksPerNode - pre) % nRanksPerNode + startRankInNode;
      int prePeerRecvId = (preRemoteRecvFromRank < rank) ? preRemoteRecvFromRank : preRemoteRecvFromRank - 1;

      // overlap communication and computation
      mscclpp::SimpleProxyChannel& preDevFstRecvChan = constDevFstRoundChans[prePeerRecvId];
      if (isComm) {
        preDevFstRecvChan.wait();
        devFstSendChan.putWithSignal(dstOffset, srcOffset, nelems * sizeof(int));
      }

      deviceSyncer.sync(gridDim.x);
      size_t offset = ((startChunkIndex + rankIdexInNode) * chunkSize + offsetInChunk) * sizeof(int);
      size_t scratchOffset = preRemoteRecvFromRank * chunkSize * sizeof(int);
      int* dst = (int*)((char*)buff + offset);
      int* src = (int*)((char*)scratch + scratchOffset);
      vectorSum(dst, src, nelems);
    }
    // for last iteration, wait for the last send
    if (i == nRanksPerNode - 1) {
      if (isComm) {
        devFstRecvChan.wait();
      }
      deviceSyncer.sync(gridDim.x);
      size_t offset = ((startChunkIndex + rankIdexInNode) * chunkSize + offsetInChunk) * sizeof(int);
      size_t scratchOffset = remoteRecvFromRank * chunkSize * sizeof(int);
      int* dst = (int*)((char*)buff + offset);
      int* src = (int*)((char*)scratch + scratchOffset);
      vectorSum(dst, src, nelems);
    }
  }
}

__device__ void reduceScatter(int* buff, int* scratch, int rank, int nRanksPerNode, int worldSize,
                              size_t nelems  // must be divisible by 3
) {
  // this reduce-scatter algorithm works as follows:
  // Step 1: each node does a local reduce-scatter on peer node data chunks with 1/pipeline portion of chunk data. For
  // example, 2 nodes and each node has 2 ranks. rank 0 and rank 1 perform reduce-scatter on chunk 2 and chunk 3, with
  // 1/pipeline portion of the data.
  // Step 2: each node does a local reduce-scatter on peers data chunks with (pipeline-1)/pipeline portion of chunk
  // data. Meanwhile, exchange the reduced data of the previous step with its cross-node neighbor (same local rank
  // number on the other node) via IB. Then performs a reduce operation.
  // Step 3:  each node does a local reduce-scatter on local ranks, meanwhile exchange the reduced data of the previous
  // step with its cross-node neighbor (same local rank number on the other node) via IB. Then performs a reduce
  // operation.
  int pipelineSize = 3;
  const size_t chunkSize = nelems / worldSize;
  int peerRank = (rank + nRanksPerNode) % worldSize;
  int peerNodeId = peerRank / nRanksPerNode;
  int isComm = (threadIdx.x == 0) && (blockIdx.x == 0);
  int peer = (peerRank < rank) ? peerRank : peerRank - 1;
  mscclpp::SimpleProxyChannel& devChan = constDevFstRoundChans[peer];
  if (peerNodeId == rank / nRanksPerNode) {
    localReduceScatter(buff, scratch, rank, nRanksPerNode, 0, 0, chunkSize, chunkSize);
    return;
  }

  // step 1: local reduce
  int startChunkIndex = peerNodeId * nRanksPerNode;
  localReduceScatter(buff, scratch, rank, nRanksPerNode, startChunkIndex, 0, chunkSize, chunkSize / pipelineSize);
  deviceSyncer.sync(gridDim.x);

  // step 2: local reduce and exchange data with neighbor
  if (isComm) {
    size_t offset = (peerRank * chunkSize) * sizeof(int);
    // opposite side
    devChan.putWithSignal(offset, (chunkSize / pipelineSize * sizeof(int)));
  }
  localReduceScatter(buff, scratch, rank, nRanksPerNode, startChunkIndex, chunkSize / pipelineSize, chunkSize,
                     2 * chunkSize / pipelineSize);
  if (isComm) {
    devChan.wait();
  }
  deviceSyncer.sync(gridDim.x);
  // reduce data received from peer to related rank
  size_t offset = rank * chunkSize * sizeof(int);
  int* dst = (int*)((char*)buff + offset);
  int* src = (int*)((char*)scratch + offset);
  vectorSum(dst, src, chunkSize / pipelineSize);
  if (isComm) {
    devChan.flush();
  }
  deviceSyncer.sync(gridDim.x);

  // step 3: local reduce and exchange data with neighbor
  startChunkIndex = (rank / nRanksPerNode) * nRanksPerNode;
  if (isComm) {
    size_t offset = (peerRank * chunkSize + chunkSize / pipelineSize) * sizeof(int);
    devChan.putWithSignal(offset, (pipelineSize - 1) * chunkSize / pipelineSize * sizeof(int));
  }
  localReduceScatter(buff, scratch, rank, nRanksPerNode, startChunkIndex, 0, chunkSize, chunkSize);
  if (isComm) {
    devChan.wait();
  }
  deviceSyncer.sync(gridDim.x);
  // reduce to related rank
  offset = (rank * chunkSize + chunkSize / pipelineSize) * sizeof(int);
  dst = (int*)((char*)buff + offset);
  src = (int*)((char*)scratch + offset);
  vectorSum(dst, src, 2 * chunkSize / pipelineSize);
  if (isComm) {
    devChan.flush();
  }
}

// Run with a single thread only.
__device__ void localAllGather(int rank, int nRanksPerNode, uint64_t offset, uint64_t size) {
  // this allgather algorithm works as follows:
  // Step 1: GPU rank i sends data to GPU rank (i+1) % nranksPerNode
  // and waits for data from GPU rank (i-1) % nranksPerNode
  // Step 2: GPU rank i sends data to GPU rank (i+2) % nranksPerNode
  // ...
  // This order is much better for DMA engine for NVLinks
  if (nRanksPerNode == 1) return;

  int startRankInNode = (rank / nRanksPerNode) * nRanksPerNode;
  for (int i = 1; i < nRanksPerNode; i++) {
    int remoteSendToRank = (rank + i) % nRanksPerNode + startRankInNode;
    int remoteRecvFromRank = (rank + nRanksPerNode - i) % nRanksPerNode + startRankInNode;
    int peerSendId = (remoteSendToRank < rank) ? remoteSendToRank : remoteSendToRank - 1;
    int peerRecvId = (remoteRecvFromRank < rank) ? remoteRecvFromRank : remoteRecvFromRank - 1;

    mscclpp::SimpleProxyChannel& devSendChan = constDevSndRoundChans[peerSendId];
    mscclpp::SimpleProxyChannel& devRecvChan = constDevSndRoundChans[peerRecvId];
    // wait for the data from GPU (rank-i) % nranksPerNode to arrive
    devSendChan.putWithSignal(offset, size);
    devRecvChan.wait();
  }
}

// Run with a single thread only.
__device__ void allGather(int rank, int worldSize, int nRanksPerNode, size_t nelemsPerGPU) {
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
  mscclpp::SimpleProxyChannel& devChan = constDevSndRoundChans[peer];

  if (peerNodeId == rank / nRanksPerNode) {
    localAllGather(rank, nRanksPerNode, rank * nelemsPerGPU * sizeof(int), nelemsPerGPU * sizeof(int));
    return;
  }

  // Step 1
  devChan.putWithSignal(rank * nelemsPerGPU * sizeof(int),
                        (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize * sizeof(int));
  localAllGather(rank, nRanksPerNode, rank * nelemsPerGPU * sizeof(int), nelemsPerGPU * sizeof(int));
  devChan.wait();
  devChan.flush();
  // Step 2
  devChan.putWithSignal((rank * nelemsPerGPU + (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize) * sizeof(int),
                        nelemsPerGPU / pipelineSize * sizeof(int));
  localAllGather(rank, nRanksPerNode, peerRank * nelemsPerGPU * sizeof(int),
                 (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize * sizeof(int));
  devChan.wait();
  devChan.flush();
  // Step 3
  localAllGather(rank, nRanksPerNode,
                 (peerRank * nelemsPerGPU + (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize) * sizeof(int),
                 nelemsPerGPU / pipelineSize * sizeof(int));
}

__device__ void localReduceScatterSm(int* buff, int* scratch, int rank, int nRanksPerNode, int startChunkIndex,
                                     size_t offsetInChunk, size_t chunkSize, size_t nelems) {
  if (nRanksPerNode == 1) return;

  const size_t size = nelems * sizeof(int);
  const int nPeer = nRanksPerNode - 1;
  const size_t peerIdx = blockIdx.x % nPeer;
  const size_t nBlockForThisPeer = gridDim.x / nPeer + (gridDim.x % nPeer > peerIdx ? 1 : 0);
  const size_t peerLocalBlockIdx = blockIdx.x / nPeer;

  // Split the data into micro-chunks for aligned data access. Ignore the remainder here and let the last block handle
  // it.
  constexpr size_t microChunkBytes = 128;  // heuristic value
  const size_t nMicroChunk = size / microChunkBytes;
  const size_t nMinMicroChunkPerBlock = nMicroChunk / nBlockForThisPeer;
  const size_t nRemainderMicroChunk = nMicroChunk % nBlockForThisPeer;

  // Distribute micro-chunks to blocks
  size_t nMicroChunkForThisBlock;
  size_t offsetForThisBlock;
  if (peerLocalBlockIdx < nRemainderMicroChunk) {
    nMicroChunkForThisBlock = nMinMicroChunkPerBlock + 1;
    offsetForThisBlock = (nMinMicroChunkPerBlock + 1) * peerLocalBlockIdx;
  } else {
    nMicroChunkForThisBlock = nMinMicroChunkPerBlock;
    offsetForThisBlock = (nMinMicroChunkPerBlock + 1) * nRemainderMicroChunk +
                         (peerLocalBlockIdx - nRemainderMicroChunk) * nMinMicroChunkPerBlock;
  }
  offsetForThisBlock *= microChunkBytes;

  // Calculate the size of the data for this block
  size_t sizeForThisBlock = nMicroChunkForThisBlock * microChunkBytes;
  const size_t lastMicroChunkSize = size - nMicroChunk * microChunkBytes;
  if (lastMicroChunkSize > 0 && peerLocalBlockIdx == nBlockForThisPeer - 1) {
    sizeForThisBlock += lastMicroChunkSize;
  }

  int remoteRank = (peerIdx < rank ? peerIdx : peerIdx + 1);
  size_t srcOffset = ((remoteRank + startChunkIndex) * chunkSize + offsetInChunk) * sizeof(int);
  size_t dstOffset = (rank * chunkSize + offsetInChunk) * sizeof(int);
  constSmOutOfPlaceChans[peerIdx].put(dstOffset + offsetForThisBlock, srcOffset + offsetForThisBlock, sizeForThisBlock,
                                      threadIdx.x, blockDim.x);
  if (peerLocalBlockIdx == 0 && threadIdx.x == 0) {
    constSmOutOfPlaceChans[peerIdx].signal();
    constSmOutOfPlaceChans[peerIdx].wait();
  }
  deviceSyncer.sync(gridDim.x);

  int rankIndexInNode = rank % nRanksPerNode;
  for (int i = 1; i < nRanksPerNode; ++i) {
    int remoteRank = (rankIndexInNode + i) % nRanksPerNode + startChunkIndex;
    size_t offset = ((startChunkIndex + rankIndexInNode) * chunkSize + offsetInChunk) * sizeof(int);
    size_t scratchOffset = (remoteRank * chunkSize + offsetInChunk) * sizeof(int);
    int* dst = (int*)((char*)buff + offset);
    int* src = (int*)((char*)scratch + scratchOffset);
    vectorSum(dst, src, nelems);
  }
}

__device__ void reduceScatterSm(int* buff, int* scratch, int rank, int nRanksPerNode, int worldSize,
                                size_t nelems  // must be divisible by 3
) {
  // this reduce-scatter algorithm works as follows:
  // Step 1: each node does a local reduce-scatter on peer node data chunks with 1/pipeline portion of chunk data. For
  // example, 2 nodes and each node has 2 ranks. rank 0 and rank 1 perform reduce-scatter on chunk 2 and chunk 3, with
  // 1/pipeline portion of the data.
  // Step 2: each node does a local reduce-scatter on peers data chunks with (pipeline-1)/pipeline portion of chunk
  // data. Meanwhile, exchange the reduced data of the previous step with its cross-node neighbor (same local rank
  // number on the other node) via IB. Then performs a reduce operation.
  // Step 3:  each node does a local reduce-scatter on local ranks, meanwhile exchange the reduced data of the previous
  // step with its cross-node neighbor (same local rank number on the other node) via IB. Then performs a reduce
  // operation.
  int pipelineSize = 3;
  const size_t chunkSize = nelems / worldSize;
  int peerRank = (rank + nRanksPerNode) % worldSize;
  int peerNodeId = peerRank / nRanksPerNode;
  int isComm = (threadIdx.x == 0) && (blockIdx.x == 0);
  int peer = (peerRank < rank) ? peerRank : peerRank - 1;
  mscclpp::SimpleProxyChannel& devChan = constDevFstRoundChans[peer];
  if (peerNodeId == rank / nRanksPerNode) {
    localReduceScatterSm(buff, scratch, rank, nRanksPerNode, 0, 0, chunkSize, chunkSize);
    return;
  }

  // step 1: local reduce
  int startChunkIndex = peerNodeId * nRanksPerNode;
  localReduceScatterSm(buff, scratch, rank, nRanksPerNode, startChunkIndex, 0, chunkSize, chunkSize / pipelineSize);
  deviceSyncer.sync(gridDim.x);

  // step 2: local reduce and exchange data with neighbor
  if (isComm) {
    size_t offset = (peerRank * chunkSize) * sizeof(int);
    // opposite side
    devChan.putWithSignal(offset, (chunkSize / pipelineSize * sizeof(int)));
  }
  localReduceScatterSm(buff, scratch, rank, nRanksPerNode, startChunkIndex, chunkSize / pipelineSize, chunkSize,
                       2 * chunkSize / pipelineSize);
  if (isComm) {
    devChan.wait();
  }
  deviceSyncer.sync(gridDim.x);
  // reduce data received from peer to related rank
  size_t offset = rank * chunkSize * sizeof(int);
  int* dst = (int*)((char*)buff + offset);
  int* src = (int*)((char*)scratch + offset);
  vectorSum(dst, src, chunkSize / pipelineSize);
  if (isComm) {
    devChan.flush();
  }
  deviceSyncer.sync(gridDim.x);

  // step 3: local reduce and exchange data with neighbor
  startChunkIndex = (rank / nRanksPerNode) * nRanksPerNode;
  if (isComm) {
    size_t offset = (peerRank * chunkSize + chunkSize / pipelineSize) * sizeof(int);
    devChan.putWithSignal(offset, (pipelineSize - 1) * chunkSize / pipelineSize * sizeof(int));
  }
  localReduceScatterSm(buff, scratch, rank, nRanksPerNode, startChunkIndex, 0, chunkSize, chunkSize);
  if (isComm) {
    devChan.wait();
  }
  deviceSyncer.sync(gridDim.x);
  // reduce to related rank
  offset = (rank * chunkSize + chunkSize / pipelineSize) * sizeof(int);
  dst = (int*)((char*)buff + offset);
  src = (int*)((char*)scratch + offset);
  vectorSum(dst, src, 2 * chunkSize / pipelineSize);
  if (isComm) {
    devChan.flush();
  }
}

// This kernel is the most performant when the number of blocks is a multiple of (nRanksPerNode - 1).
__device__ void localAllGatherSm(int rank, int nRanksPerNode, uint64_t offset, uint64_t size) {
  if (nRanksPerNode == 1) return;
  const size_t nPeer = nRanksPerNode - 1;
  const size_t peerIdx = blockIdx.x % nPeer;
  const size_t nBlockForThisPeer = gridDim.x / nPeer + (gridDim.x % nPeer > peerIdx ? 1 : 0);
  const size_t peerLocalBlockIdx = blockIdx.x / nPeer;

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

  constSmInPlaceChans[peerIdx].put(offset + offsetForThisBlock, sizeForThisBlock, threadIdx.x, blockDim.x);
  deviceSyncer.sync(gridDim.x);
  if (threadIdx.x == 0 && peerLocalBlockIdx == 0) {
    constSmInPlaceChans[peerIdx].signal();
    constSmInPlaceChans[peerIdx].wait();
  }
}

// This is an allgather4 equivalent
__device__ void allGatherSm(int rank, int worldSize, int nRanksPerNode, size_t nelemsPerGPU) {
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
  mscclpp::SimpleProxyChannel& devChan = constDevSndRoundChans[peer];
  if (peerNodeId == rank / nRanksPerNode) {
    localAllGatherSm(rank, nRanksPerNode, rank * nelemsPerGPU * sizeof(int), nelemsPerGPU * sizeof(int));
    return;
  }

  constexpr size_t alignment = 128;
  size_t step1Bytes = (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize * sizeof(int);
  step1Bytes = step1Bytes / alignment * alignment;
  const size_t step2Bytes = nelemsPerGPU * sizeof(int) - step1Bytes;

  // Step 1
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    devChan.putWithSignal(rank * nelemsPerGPU * sizeof(int), step1Bytes);
  }
  localAllGatherSm(rank, nRanksPerNode, rank * nelemsPerGPU * sizeof(int), nelemsPerGPU * sizeof(int));
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    devChan.wait();
    devChan.flush();
  }
  deviceSyncer.sync(gridDim.x);
  // Step 2
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    devChan.putWithSignal(rank * nelemsPerGPU * sizeof(int) + step1Bytes, step2Bytes);
  }
  localAllGatherSm(rank, nRanksPerNode, peerRank * nelemsPerGPU * sizeof(int), step1Bytes);
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    devChan.wait();
    devChan.flush();
  }
  deviceSyncer.sync(gridDim.x);
  // Step 3
  localAllGatherSm(rank, nRanksPerNode, (peerRank * nelemsPerGPU * sizeof(int) + step1Bytes), step2Bytes);
}

__device__ void allreduce0(int* buff, int* scratch, int rank, int worldSize, size_t nelems, size_t scratchDataCount) {
  int peerId = blockIdx.x / BLOCKS_PER_PEER;
  int isComm = (threadIdx.x == 0) && (blockIdx.x % BLOCKS_PER_PEER == 0);
  int remoteRank = (peerId < rank) ? peerId : peerId + 1;

  // 1st communication phase: send data to the scratch buffer of the peer associated with this block
  mscclpp::SimpleProxyChannel& devFstRoundChan = constDevFstRoundChans[peerId];
  Chunk toPeerChunk = getChunk(nelems, worldSize, remoteRank);
  // Now we need to figure out the offset of this chunk in the scratch buffer of the destination.
  // The destination will have allocated a scratch buffer of size numPeers() * toPeerChunk.size and
  // inside that each of the destination's peers send to the nth chunk, where n is the index of the
  // source peer from the destination's perspective.
  size_t dstOffset = (rank < remoteRank ? rank : rank - 1) * toPeerChunk.size;
  if (isComm) {
    // Write data to the peer
    devFstRoundChan.putWithSignalAndFlush(dstOffset * sizeof(int), toPeerChunk.offset * sizeof(int),
                                          toPeerChunk.size * sizeof(int));
    // Wait for data from the peer
    devFstRoundChan.wait();
  }

  deviceSyncer.sync(gridDim.x);

  // Local reduction: every block reduces a slice of each chunk in the scratch buffer into the user buffer
  mscclpp::SimpleProxyChannel& devSndRoundChan = constDevSndRoundChans[peerId];
  Chunk rankChunk = getChunk(nelems, worldSize, rank);
  int* chunk = buff + rankChunk.offset;
  int numPeers = gridDim.x / BLOCKS_PER_PEER;
  int numBlocks = gridDim.x;
  Chunk blockUserChunk = getChunk(rankChunk.size, numBlocks, blockIdx.x);
  size_t scratchDataCountPerPeer = scratchDataCount / numPeers;
  Chunk blockScratchChunk = getChunk(scratchDataCountPerPeer, numBlocks, blockIdx.x);
  for (int peerIdx = 0; peerIdx < numPeers; ++peerIdx) {
    int* scratchChunk = scratch + peerIdx * scratchDataCountPerPeer;
    vectorSumSingleBlock(chunk + blockUserChunk.offset, scratchChunk + blockScratchChunk.offset,
                         blockScratchChunk.size);
  }

  deviceSyncer.sync(gridDim.x);

  // 2nd communication phase: send the now reduced data between the user buffers
  Chunk collectionChunk = getChunk(nelems, worldSize, rank);
  if (isComm) {
    // Write data to the peer
    devSndRoundChan.putWithSignalAndFlush(collectionChunk.offset * sizeof(int), collectionChunk.offset * sizeof(int),
                                          collectionChunk.size * sizeof(int));
    // Wait for data from the peer
    devSndRoundChan.wait();
  }
}

__device__ void allreduce1(int* buff, int* scratch, int rank, int worldSize, size_t nelems, size_t scratchDataCount) {
  int isComm = (threadIdx.x == 0) && (blockIdx.x == 0);
  int remoteSendRank = (rank + 1) % worldSize;
  int remoteRecvRank = (rank + worldSize - 1) % worldSize;
  int peerSendId = (remoteSendRank < rank) ? remoteSendRank : remoteSendRank - 1;
  int peerRecvId = (remoteRecvRank < rank) ? remoteRecvRank : remoteRecvRank - 1;

  mscclpp::SimpleProxyChannel& devFstSendChan = constDevFstRoundChans[peerSendId];
  mscclpp::SimpleProxyChannel& devFstRecvChan = constDevFstRoundChans[peerRecvId];
  mscclpp::SimpleProxyChannel& devSndSendChan = constDevSndRoundChans[peerSendId];
  mscclpp::SimpleProxyChannel& devSndRecvChan = constDevSndRoundChans[peerRecvId];

  // Step 1
  size_t chunkIndex = (rank + worldSize - 1) % worldSize;
  size_t chunkNelem = nelems / worldSize;
  size_t offset = chunkIndex * chunkNelem * sizeof(int);
  if (isComm) {
    if (chunkNelem > 1) {
      devFstSendChan.putWithSignal(offset, chunkNelem / 2 * sizeof(int));
    }
  }

  // Step 2 ~ Step n-1
  for (int step = 2; step < worldSize; ++step) {
    if (isComm) {
      if (chunkNelem > 1) {
        devFstRecvChan.wait();
        devFstSendChan.flush();
      }
      devFstSendChan.putWithSignal(offset + chunkNelem / 2 * sizeof(int), (chunkNelem - chunkNelem / 2) * sizeof(int));
    }
    deviceSyncer.sync(gridDim.x);

    // Reduce
    chunkIndex = (rank + worldSize - step) % worldSize;
    offset = chunkIndex * chunkNelem * sizeof(int);
    int* dst = (int*)((char*)buff + offset);
    int* src = (int*)((char*)scratch + offset);
    vectorSum(dst, src, chunkNelem / 2);

    if (isComm) {
      devFstRecvChan.wait();
      devFstSendChan.flush();
      if (chunkNelem > 1) {
        devFstSendChan.putWithSignal(offset, chunkNelem / 2 * sizeof(int));
      }
    }
    deviceSyncer.sync(gridDim.x);

    dst += chunkNelem / 2;
    src += chunkNelem / 2;
    vectorSum(dst, src, chunkNelem - chunkNelem / 2);
  }

  // Step n
  if (isComm) {
    if (chunkNelem > 1) {
      devFstRecvChan.wait();
      devFstSendChan.flush();
    }
    devFstSendChan.putWithSignal(offset + chunkNelem / 2 * sizeof(int), (chunkNelem - chunkNelem / 2) * sizeof(int));
  }
  deviceSyncer.sync(gridDim.x);

  offset = rank * chunkNelem * sizeof(int);
  int* dst = (int*)((char*)buff + offset);
  int* src = (int*)((char*)scratch + offset);
  vectorSum(dst, src, chunkNelem / 2);

  if (isComm) {
    devFstRecvChan.wait();
    devFstSendChan.flush();
    if (chunkNelem > 1) {
      devSndSendChan.putWithSignal(offset, chunkNelem / 2 * sizeof(int));
    }
  }
  deviceSyncer.sync(gridDim.x);

  dst += chunkNelem / 2;
  src += chunkNelem / 2;
  vectorSum(dst, src, chunkNelem - chunkNelem / 2);

  if (isComm) {
    if (chunkNelem > 1) {
      devSndRecvChan.wait();
      devSndSendChan.flush();
    }
    devSndSendChan.putWithSignalAndFlush(offset + chunkNelem / 2 * sizeof(int),
                                         (chunkNelem - chunkNelem / 2) * sizeof(int));
  }

  // Step n+1 ~ Step 2n-2
  for (int i = 1; i < worldSize - 1; ++i) {
    if (isComm) {
      devSndRecvChan.wait();
    }
    deviceSyncer.sync(gridDim.x);

    // Copy
    chunkIndex = (rank + worldSize - i) % worldSize;
    if (isComm) {
      devSndSendChan.putWithSignalAndFlush(chunkIndex * chunkNelem * sizeof(int), chunkNelem * sizeof(int));
    }
  }

  // Final receive
  if (isComm) {
    devSndRecvChan.wait();
  }
}

__device__ void allreduce2(int* buff, void* scratch, void* putPktBuf, void* getPktBuf, void* result, int rank,
                           int nRanksPerNode, int worldSize, size_t nelems) {
  int numPeersPerNode = nRanksPerNode - 1;
  size_t nPkts = nelems / 2;  // 2 elems per packet, assume nelems is even
  size_t pktBytes = nPkts * sizeof(mscclpp::LLPacket);

  // Channel to a local peer
  int smChanIdx = blockIdx.x / BLOCKS_PER_PEER;
  mscclpp::SmChannel smChan = constSmOutOfPlaceChans[smChanIdx];

  // Channel to a remote peer that has the same local rank as me
  int localRank = rank % nRanksPerNode;
  mscclpp::SimpleProxyChannel devChan = constDevFstRoundChans[localRank];

  // Flag for packets. Initially 1
  uint32_t flag = (uint32_t)globalFlag;

  int2* src = (int2*)buff;
  int2* res = (int2*)result;
  // double buffering
  size_t scratchOffset = (flag & 1) ? 0 : nPkts * max(numPeersPerNode, 1) * sizeof(mscclpp::LLPacket);
  mscclpp::LLPacket* scratchPtr = (mscclpp::LLPacket*)((char*)scratch + scratchOffset);
  size_t pktBufOffset = (flag & 1) ? 0 : nPkts * sizeof(mscclpp::LLPacket);
  mscclpp::LLPacket* getPktPtr = (mscclpp::LLPacket*)((char*)getPktBuf + pktBufOffset);
  mscclpp::LLPacket* putPktPtr = (mscclpp::LLPacket*)((char*)putPktBuf + pktBufOffset);

  // Phase 1: Local AllReduce. Read from buff, write to putPktBuf (for single node) or to result (for 2 nodes)
  if (numPeersPerNode == 0) {
    // One rank per node: write data to putPktBuf directly
    for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nPkts; idx += blockDim.x * gridDim.x) {
      putPktPtr[idx].write(src[idx].x, src[idx].y, flag);
    }
  } else {
    // Offset of the input data (buff) to read from
    size_t srcOffset =
        ((blockIdx.x % BLOCKS_PER_PEER) * nelems * sizeof(int) / BLOCKS_PER_PEER);  // offset for this block
    // Offset of the peer's scratch buffer (scratch) to write on
    size_t dstOffset = (scratchOffset) +                                                   // double buffering
                       ((smChanIdx < localRank ? localRank - 1 : localRank) * pktBytes) +  // offset for this rank
                       (srcOffset * 2);  // offset for this block: twice of srcOffset because 2 elems per packet
    // Write data to the peer's scratch
    smChan.putPackets(dstOffset, srcOffset, nelems / BLOCKS_PER_PEER * sizeof(int), threadIdx.x, blockDim.x, flag);
    // Read data from my scratch, reduce data with my buff, and write the result to my putPktBuf or to result
    const bool isSingleNode = (worldSize == nRanksPerNode);
    for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nPkts; idx += blockDim.x * gridDim.x) {
      int x = 0;
      int y = 0;
      for (int peerIdx = 0; peerIdx < numPeersPerNode / 2; ++peerIdx) {
        mscclpp::LLPacket* pkt0 = scratchPtr + 2 * peerIdx * nPkts;
        mscclpp::LLPacket* pkt1 = scratchPtr + (2 * peerIdx + 1) * nPkts;
        uint2 data0 = pkt0[idx].read(flag);
        uint2 data1 = pkt1[idx].read(flag);
        x += (int)data0.x;
        y += (int)data0.y;
        x += (int)data1.x;
        y += (int)data1.y;
      }
      if (numPeersPerNode & 1) {
        mscclpp::LLPacket* pkt = scratchPtr + (numPeersPerNode - 1) * nPkts;
        uint2 data = pkt[idx].read(flag);
        x += (int)data.x;
        y += (int)data.y;
      }
      if (isSingleNode) {
        res[idx].x = src[idx].x + x;
        res[idx].y = src[idx].y + y;
      } else {
        putPktPtr[idx].write(src[idx].x + x, src[idx].y + y, flag);
      }
    }
  }

  // If this is single node AllReduce, we are done.
  if (worldSize != nRanksPerNode) {
    // Phase 2: Inter-node AllReduce. Supports only 2 nodes. Read from putPktBuf, write to result

    // Wait for all threads to finish writing to putPktBuf in Phase 1
    deviceSyncer.sync(gridDim.x);

    // Phase 2 may need less blocks than Phase 1.
    constexpr int nBlocksPhase2 = 1;
    if (blockIdx.x >= nBlocksPhase2) return;

    // Write my putPktBuf to the remote peer's getPktBuf
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      devChan.put(pktBufOffset, pktBytes);
      if ((flag & 63) == 0) {
        devChan.flush();
      }
    }

    // Read data from my getPktBuf, reduce data with my putPktBuf, and write the result to result
    for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nPkts; idx += blockDim.x * nBlocksPhase2) {
      uint2 data0 = putPktPtr[idx].read(flag);
      uint2 data1 = getPktPtr[idx].read(flag);
      res[idx].x = (int)data0.x + (int)data1.x;
      res[idx].y = (int)data0.y + (int)data1.y;
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    globalFlag += 1;
  }
}

__device__ void allreduce3(int* buff, int* scratch, void* result, int rank, int nRanksPerNode, int worldSize,
                           size_t nelems) {
  reduceScatter(buff, scratch, rank, nRanksPerNode, worldSize, nelems);
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    allGather(rank, worldSize, nRanksPerNode, nelems / worldSize);
  }
}

__device__ void allreduce4(int* buff, int* scratch, void* result, int rank, int nRanksPerNode, int worldSize,
                           size_t nelems) {
  reduceScatterSm(buff, scratch, rank, nRanksPerNode, worldSize, nelems);
  deviceSyncer.sync(gridDim.x);
  allGatherSm(rank, worldSize, nRanksPerNode, nelems / worldSize);
}

__global__ void kernel(void* buff, void* scratch, void* result, void* putPktBuf, void* getPktBuf, int rank,
                       int nRanksPerNode, int worldSize, size_t nelems, size_t scratchDataCount, int kernel) {
  if (kernel == 0)
    allreduce0((int*)buff, (int*)scratch, rank, worldSize, nelems, scratchDataCount);
  else if (kernel == 1)
    allreduce1((int*)buff, (int*)scratch, rank, worldSize, nelems, scratchDataCount);
  else if (kernel == 2)
    allreduce2((int*)buff, scratch, putPktBuf, getPktBuf, result, rank, nRanksPerNode, worldSize, nelems);
  else if (kernel == 3)
    allreduce3((int*)buff, (int*)scratch, result, rank, nRanksPerNode, worldSize, nelems);
  else if (kernel == 4)
    allreduce4((int*)buff, (int*)scratch, result, rank, nRanksPerNode, worldSize, nelems);
}

class AllReduceTestColl : public BaseTestColl {
 public:
  AllReduceTestColl() = default;
  ~AllReduceTestColl() = default;

  void runColl(const TestArgs& args, cudaStream_t stream) override;
  void initData(const TestArgs& args, std::vector<void*> sendBuff, void* expectedBuff) override;
  void getBw(const double deltaSec, double& algBw /*OUT*/, double& busBw /*OUT*/) override;
  void setupCollTest(size_t size) override;
  std::vector<KernelRestriction> getKernelRestrictions() override;
};

void AllReduceTestColl::runColl(const TestArgs& args, cudaStream_t stream) {
  const int worldSize = args.totalRanks;
  const int rank = args.rank;
  const int kernelNum = args.kernelNum;
  const int nPeers = worldSize - 1;
  const Chunk chunk = getChunk(paramCount_, worldSize, rank);
  const size_t scratchDataCount = chunk.size * nPeers;

  int nBlocks;
  void* tmpBuff;
  if (kernelNum == 0) {
    nBlocks = nPeers * BLOCKS_PER_PEER;
    tmpBuff = scratchBuff;
  } else if (kernelNum == 1 || kernelNum == 3) {
    nBlocks = 24;
    tmpBuff = scratchBuff;
  } else if (kernelNum == 4) {
    nBlocks = 14;
    tmpBuff = scratchBuff;
  } else {
    nBlocks = std::max(args.nRanksPerNode - 1, 1) * BLOCKS_PER_PEER;
    tmpBuff = scratchPacketBuff;
  }
  kernel<<<nBlocks, 1024, 0, stream>>>(inputBuff, tmpBuff, resultBuff, putPacketBuff, getPacketBuff, rank,
                                       args.nRanksPerNode, worldSize, paramCount_, scratchDataCount, kernelNum);
}

void AllReduceTestColl::initData(const TestArgs& args, std::vector<void*> sendBuff, void* expectedBuff) {
  assert(sendBuff.size() == 1);
  const int rank = args.rank;
  const int worldSize = args.totalRanks;
  std::vector<int> dataHost(std::max(sendCount_, recvCount_), rank);
  CUDATHROW(cudaMemcpy(sendBuff[0], dataHost.data(), sendCount_ * typeSize_, cudaMemcpyHostToDevice));

  for (size_t i = 0; i < recvCount_; i++) {
    dataHost[i] = worldSize * (worldSize - 1) / 2;
  }
  std::memcpy(expectedBuff, dataHost.data(), recvCount_ * typeSize_);
}

void AllReduceTestColl::getBw(const double deltaSec, double& algBw /*OUT*/, double& busBw /*OUT*/) {
  double baseBw = (double)(paramCount_ * typeSize_) / 1.0E9 / deltaSec;
  algBw = baseBw;
  double factor = (2 * (double)(worldSize_ - 1)) / ((double)worldSize_);
  busBw = baseBw * factor;
}

void AllReduceTestColl::setupCollTest(size_t size) {
  size_t count = size / typeSize_;
  sendCount_ = count;
  recvCount_ = count;
  paramCount_ = count;
  expectedCount_ = count;

  mscclpp::DeviceSyncer syncer = {};
  uint64_t initFlag = 1;
  CUDATHROW(cudaMemcpyToSymbol(deviceSyncer, &syncer, sizeof(mscclpp::DeviceSyncer)));
  CUDATHROW(cudaMemcpyToSymbol(globalFlag, &initFlag, sizeof(uint64_t)));
}

std::vector<KernelRestriction> AllReduceTestColl::getKernelRestrictions() {
  return {// {kernelNum, kernelName, compatibleWithMultiNodes, countDivisorForMultiNodes}
          {0, "allreduce0", true, 1},
          {1, "allreduce1", true, 1},
          {2, "allreduce2", true, 1},
          {3, "allreduce3", true, 3},
          {4, "allreduce4", true, 3}};
}

class AllReduceTestEngine : public BaseTestEngine {
 public:
  AllReduceTestEngine(const TestArgs& args);
  ~AllReduceTestEngine() = default;

  void allocateBuffer() override;
  void setupConnections() override;

  bool isUsePacket() const;
  bool isInPlace() const;

  std::vector<void*> getSendBuff() override;
  void* getRecvBuff() override;
  void* getScratchBuff() override;

 private:
  void* getExpectedBuff() override;

  std::shared_ptr<int> inputBuff_;
  std::shared_ptr<int> scratchBuff_;
  std::shared_ptr<int> resultBuff_;
  std::shared_ptr<mscclpp::LLPacket> scratchPacketBuff_;
  std::shared_ptr<mscclpp::LLPacket> putPacketBuff_;
  std::shared_ptr<mscclpp::LLPacket> getPacketBuff_;
  std::shared_ptr<int[]> expectedBuff_;
};

AllReduceTestEngine::AllReduceTestEngine(const TestArgs& args) : BaseTestEngine(args, "allreduce") {
  inPlace_ = isInPlace();
}

bool AllReduceTestEngine::isUsePacket() const { return (args_.kernelNum == 2); }

bool AllReduceTestEngine::isInPlace() const { return (args_.kernelNum != 2); }

void AllReduceTestEngine::allocateBuffer() {
  inputBuff_ = mscclpp::allocSharedCuda<int>(args_.maxBytes / sizeof(int));
  resultBuff_ = mscclpp::allocSharedCuda<int>(args_.maxBytes / sizeof(int));
  inputBuff = inputBuff_.get();
  resultBuff = resultBuff_.get();

  if (args_.kernelNum == 0 || args_.kernelNum == 1 || args_.kernelNum == 3 || args_.kernelNum == 4) {
    scratchBuff_ = mscclpp::allocSharedCuda<int>(args_.maxBytes / sizeof(int));
    scratchBuff = scratchBuff_.get();
  } else if (args_.kernelNum == 2) {
    const size_t nPacket = (args_.maxBytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);
    // 2x for double-buffering
    const size_t scratchBuffNelem = nPacket * std::max(args_.nRanksPerNode - 1, 1) * 2;
    scratchPacketBuff_ = mscclpp::allocSharedCuda<mscclpp::LLPacket>(scratchBuffNelem);
    scratchPacketBuff = scratchPacketBuff_.get();
    const size_t packetBuffNelem = nPacket * 2;
    putPacketBuff_ = mscclpp::allocSharedCuda<mscclpp::LLPacket>(packetBuffNelem);
    getPacketBuff_ = mscclpp::allocSharedCuda<mscclpp::LLPacket>(packetBuffNelem);
    putPacketBuff = putPacketBuff_.get();
    getPacketBuff = getPacketBuff_.get();
  }

  expectedBuff_ = std::shared_ptr<int[]>(new int[args_.maxBytes / sizeof(int)]);
}

void AllReduceTestEngine::setupConnections() {
  if (isUsePacket()) {
    std::vector<mscclpp::SmChannel> smChannels;
    std::vector<mscclpp::SimpleProxyChannel> devChannels;

    const size_t nPacket = (args_.maxBytes + sizeof(uint64_t) - 1) / sizeof(uint64_t);
    const size_t scratchPacketBuffBytes =
        nPacket * std::max(args_.nRanksPerNode - 1, 1) * 2 * sizeof(mscclpp::LLPacket);
    const size_t packetBuffBytes = nPacket * 2 * sizeof(mscclpp::LLPacket);
    setupMeshConnections(smChannels, devChannels, inputBuff_.get(), args_.maxBytes, putPacketBuff_.get(),
                         packetBuffBytes, getPacketBuff_.get(), packetBuffBytes, scratchPacketBuff_.get(),
                         scratchPacketBuffBytes);

    assert(smChannels.size() < sizeof(constSmOutOfPlaceChans) / sizeof(mscclpp::SmChannel));
    assert(devChannels.size() < sizeof(constDevFstRoundChans) / sizeof(mscclpp::SimpleProxyChannel));
    CUDATHROW(
        cudaMemcpyToSymbol(constSmOutOfPlaceChans, smChannels.data(), sizeof(mscclpp::SmChannel) * smChannels.size()));
    CUDATHROW(cudaMemcpyToSymbol(constDevFstRoundChans, devChannels.data(),
                                 sizeof(mscclpp::SimpleProxyChannel) * devChannels.size()));
  } else {
    std::vector<mscclpp::SimpleProxyChannel> fstRoundChannels;
    std::vector<mscclpp::SimpleProxyChannel> sndRoundChannels;

    std::vector<mscclpp::SmChannel> smOutOfPlaceChannels;
    std::vector<mscclpp::SmChannel> smInPlaceChannels;

    // Send data from local inputBuff to remote scratchBuff (out-of-place)
    setupMeshConnections(fstRoundChannels, inputBuff_.get(), args_.maxBytes, scratchBuff_.get(), args_.maxBytes);
    assert(fstRoundChannels.size() < sizeof(constDevFstRoundChans) / sizeof(mscclpp::SimpleProxyChannel));
    CUDATHROW(cudaMemcpyToSymbol(constDevFstRoundChans, fstRoundChannels.data(),
                                 sizeof(mscclpp::SimpleProxyChannel) * fstRoundChannels.size()));

    // Send data from local inputBuff to remote inputBuff (in-place)
    setupMeshConnections(sndRoundChannels, inputBuff_.get(), args_.maxBytes);
    assert(sndRoundChannels.size() < sizeof(constDevSndRoundChans) / sizeof(mscclpp::SimpleProxyChannel));
    CUDATHROW(cudaMemcpyToSymbol(constDevSndRoundChans, sndRoundChannels.data(),
                                 sizeof(mscclpp::SimpleProxyChannel) * sndRoundChannels.size()));

    setupMeshConnections(smOutOfPlaceChannels, inputBuff_.get(), args_.maxBytes, scratchBuff_.get(), args_.maxBytes);
    assert(smOutOfPlaceChannels.size() < sizeof(constSmOutOfPlaceChans) / sizeof(mscclpp::SmChannel));
    CUDATHROW(cudaMemcpyToSymbol(constSmOutOfPlaceChans, smOutOfPlaceChannels.data(),
                                 sizeof(mscclpp::SmChannel) * smOutOfPlaceChannels.size()));

    setupMeshConnections(smInPlaceChannels, inputBuff_.get(), args_.maxBytes);
    assert(smInPlaceChannels.size() < sizeof(constSmInPlaceChans) / sizeof(mscclpp::SmChannel));
    CUDATHROW(cudaMemcpyToSymbol(constSmInPlaceChans, smInPlaceChannels.data(),
                                 sizeof(mscclpp::SmChannel) * smInPlaceChannels.size()));
  }
}

std::vector<void*> AllReduceTestEngine::getSendBuff() { return {inputBuff_.get()}; }

void* AllReduceTestEngine::getExpectedBuff() { return expectedBuff_.get(); }

void* AllReduceTestEngine::getRecvBuff() { return isInPlace() ? inputBuff_.get() : resultBuff_.get(); }

void* AllReduceTestEngine::getScratchBuff() { return scratchBuff_.get(); }

std::shared_ptr<BaseTestEngine> getTestEngine(const TestArgs& args) {
  return std::make_shared<AllReduceTestEngine>(args);
}

std::shared_ptr<BaseTestColl> getTestColl() { return std::make_shared<AllReduceTestColl>(); }
