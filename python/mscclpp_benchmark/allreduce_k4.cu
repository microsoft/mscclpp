// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "common.cuh"

__device__ mscclpp::DeviceSyncer deviceSyncer;
__device__ mscclpp::DeviceSyncer allGatherDeviceSyncer;
__device__ mscclpp::DeviceSyncer reduceScatterDeviceSyncer;
__device__ mscclpp::DeviceSyncer ibDeviceSyncer;

// -------------------------------------------
// AllReduce4
// 2-node
// -------------------------------------------
__device__ void localReduceScatterSm(mscclpp::SmChannelDeviceHandle* smChans, TYPE* buff, int rank, int nRanksPerNode,
                                     int startChunkIndex, size_t offsetInChunk, size_t chunkSize, size_t nelems,
                                     int nBlocks) {
  if (nRanksPerNode == 1) return;
  if (blockIdx.x >= nBlocks) return;
  const int nPeer = nRanksPerNode - 1;

  const size_t localRankIndexInNode = rank % nRanksPerNode;
  const size_t indexOffset = ((localRankIndexInNode + startChunkIndex) * chunkSize + offsetInChunk);
  const size_t indexOffset4 = indexOffset / 4;

  int4* buff4 = (int4*)buff;

  for (int peerIdx = threadIdx.x + blockIdx.x * blockDim.x; peerIdx < nPeer; peerIdx += blockDim.x * nBlocks) {
    smChans[peerIdx].relaxedSignal();
  }
  for (int peerIdx = threadIdx.x + blockIdx.x * blockDim.x; peerIdx < nPeer; peerIdx += blockDim.x * nBlocks) {
    smChans[peerIdx].wait();
  }
  reduceScatterDeviceSyncer.sync(nBlocks);

  const size_t nInt4 = nelems / 4;
  for (int idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nInt4; idx += blockDim.x * nBlocks) {
    int4 tmp = buff4[indexOffset4 + idx];
    for (int index = 0; index < nPeer; ++index) {
      int4 val;
      int peerIdx = index + localRankIndexInNode;
      if (peerIdx >= nPeer) peerIdx -= nPeer;
      val = smChans[peerIdx].read<int4>(indexOffset4 + idx);
      tmp = add_vectors<TYPE>(tmp, val);
    }
    buff4[indexOffset4 + idx] = tmp;
  }

  // TODO: deal with rest elements
}

// This kernel is the most performant when the number of blocks is a multiple of (nRanksPerNode - 1).
__device__ void localAllGatherSm(mscclpp::SmChannelDeviceHandle* smChans, int rank, int nRanksPerNode,
                                 int startRankChunkIndex, uint64_t offsetInRankChunk, uint64_t rankChunkSize,
                                 uint64_t size, size_t nBlocks) {
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
    smChans[peerIdx].relaxedSignal();
    smChans[peerIdx].wait();
  }
  allGatherDeviceSyncer.sync(nBlocks);
  size_t offset = rankChunkSize * (startRankChunkIndex + remoteRankLocalIndex) + offsetInRankChunk;
  smChans[peerIdx].get(offset + offsetForThisBlock, sizeForThisBlock, threadIdx.x, blockDim.x);
}

__device__ void localAllGatherAllPairsSm(mscclpp::SmChannelDeviceHandle* smChans, int rank, int nRanksPerNode,
                                         uint64_t size, size_t nBlocks) {
  if (nRanksPerNode == 1) return;
  if (blockIdx.x >= nBlocks) return;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int nPeer = nRanksPerNode - 1;

  if (tid < nPeer) {
    smChans[tid].signal();
  }
  int waitStart = nBlocks * blockDim.x - nPeer;
  if (tid >= waitStart && tid < nBlocks * blockDim.x) {
    smChans[tid - waitStart].wait();
  }
  allGatherDeviceSyncer.sync(nBlocks);
  for (int i = 0; i < nPeer; ++i) {
    int peerIdx = (i + rank) % nPeer;
    const int remoteRankLocalIndex = (peerIdx < rank ? peerIdx : peerIdx + 1);
    size_t offset = size * remoteRankLocalIndex;
    smChans[peerIdx].get(offset, size, tid, blockDim.x * nBlocks);
  }
}

// This is an allgather4 equivalent
__device__ void allGatherSm(mscclpp::SmChannelDeviceHandle* smChans,
                            mscclpp::SimpleProxyChannelDeviceHandle* proxyChans, int rank, int worldSize,
                            int nRanksPerNode, size_t nelemsPerGPU, int pipelineDepth) {
  // this allgather is a pipelined and hierarchical one and only works for two nodes
  // it is implemented as follows:
  // Step 1: each node does a local allgather and concurrently,
  // local GPU i exchange (piplineSize-1)/pipelineSize portion of their data with
  // its cross-node neighbor (local GPU i on the other node) via IB
  // Step 2: each node does a local allgather again with the data just received from its
  // cross-node neighbor in step 1, and concurrently, exchange the rest of the data with
  // its cross-node neighbor
  // Step 3: each node does a local allgather for the last time with the rest of the data

  int pipelineSize = pipelineDepth;
  int peerRank = (rank + nRanksPerNode) % worldSize;
  int peerNodeId = peerRank / nRanksPerNode;
  int peer = (peerRank < rank) ? peerRank : peerRank - 1;
  mscclpp::SimpleProxyChannelDeviceHandle proxyChan = proxyChans[peer];
  const size_t nBlocksForLocalAllGather = gridDim.x / (nRanksPerNode - 1) * (nRanksPerNode - 1);
  const size_t rankChunkSize = nelemsPerGPU * sizeof(int);
  const int startRankIndexInLocalNode = (rank / nRanksPerNode) * nRanksPerNode;
  const int startRankIndexInPeerNode = (peerRank / nRanksPerNode) * nRanksPerNode;

  if (peerNodeId == rank / nRanksPerNode) {
    localAllGatherSm(smChans, rank, nRanksPerNode, 0, 0, rankChunkSize, rankChunkSize, gridDim.x);
    return;
  }

  constexpr size_t alignment = 128;
  size_t step1Bytes = (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize * sizeof(int);
  step1Bytes = step1Bytes / alignment * alignment;
  const size_t step2Bytes = nelemsPerGPU * sizeof(int) - step1Bytes;

  // Step 1
  if (threadIdx.x == 0 && blockIdx.x == 0 && step1Bytes > 0) {
    proxyChan.putWithSignal(rank * nelemsPerGPU * sizeof(int), step1Bytes);
  }
  localAllGatherSm(smChans, rank, nRanksPerNode, startRankIndexInLocalNode, 0, rankChunkSize, rankChunkSize,
                   nBlocksForLocalAllGather);
  if (threadIdx.x == 0 && blockIdx.x == 0 && step1Bytes > 0) {
    proxyChan.wait();
    proxyChan.flush();
  }
  deviceSyncer.sync(gridDim.x);
  // Step 2
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    proxyChan.putWithSignal(rank * nelemsPerGPU * sizeof(int) + step1Bytes, step2Bytes);
  }
  if (step1Bytes > 0)
    localAllGatherSm(smChans, rank, nRanksPerNode, startRankIndexInPeerNode, 0, rankChunkSize, step1Bytes,
                     nBlocksForLocalAllGather);
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    proxyChan.wait();
    proxyChan.flush();
  }
  deviceSyncer.sync(gridDim.x);
  // Step 3
  localAllGatherSm(smChans, rank, nRanksPerNode, startRankIndexInPeerNode, step1Bytes, rankChunkSize, step2Bytes,
                   nBlocksForLocalAllGather);
}

__device__ void reduceScatterSm(mscclpp::SmChannelDeviceHandle* smChans,
                                mscclpp::SimpleProxyChannelDeviceHandle* proxyChans, TYPE* buff, TYPE* scratch,
                                int rank, int nRanksPerNode, int worldSize,
                                size_t nelems,  // must be divisible by 3
                                int pipelineDepth) {
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
  int pipelineSize = pipelineDepth;
  float nBlocksForReduceScatterRatio = 0.8;
  const size_t chunkSize = nelems / worldSize;
  const int peerRank = (rank + nRanksPerNode) % worldSize;
  int peerNodeId = peerRank / nRanksPerNode;
  int nBlocksForReduceScatter =
      (int)(nBlocksForReduceScatterRatio * gridDim.x) / (nRanksPerNode - 1) * (nRanksPerNode - 1);
  int isComm = (threadIdx.x == 0) && (blockIdx.x == nBlocksForReduceScatter);
  int peer = (peerRank < rank) ? peerRank : peerRank - 1;
  int nBlocksRemain = gridDim.x - nBlocksForReduceScatter;
  mscclpp::SimpleProxyChannelDeviceHandle proxyChan = proxyChans[peer];
  if (peerNodeId == rank / nRanksPerNode) {
    localReduceScatterSm(smChans, buff, rank, nRanksPerNode, 0, 0, chunkSize, chunkSize, gridDim.x);
    return;
  }

  // step 1: local reduce
  int startChunkIndex = peerNodeId * nRanksPerNode;
  localReduceScatterSm(smChans, buff, rank, nRanksPerNode, startChunkIndex, 0, chunkSize, chunkSize / pipelineSize,
                       nBlocksForReduceScatter);
  deviceSyncer.sync(gridDim.x);

  // step 2: local reduce and exchange data with neighbor
  if (isComm) {
    size_t offset = (peerRank * chunkSize) * sizeof(int);
    // opposite side
    proxyChan.putWithSignal(offset, (chunkSize / pipelineSize * sizeof(int)));
  }
  if (pipelineSize > 1)
    localReduceScatterSm(smChans, buff, rank, nRanksPerNode, startChunkIndex, chunkSize / pipelineSize, chunkSize,
                         (pipelineSize - 1) * chunkSize / pipelineSize, nBlocksForReduceScatter);
  if (isComm) {
    proxyChan.wait();
  }
  if (blockIdx.x >= nBlocksForReduceScatter) {
    ibDeviceSyncer.sync(nBlocksRemain);
    // reduce data received from peer to related rank
    size_t offset = rank * chunkSize * sizeof(int);
    int* dst = (int*)((char*)buff + offset);
    int* src = (int*)((char*)scratch + offset);
    vectorSum((TYPE*)dst, (TYPE*)src, chunkSize / pipelineSize, blockIdx.x - nBlocksForReduceScatter, nBlocksRemain);
  }
  if (isComm) {
    proxyChan.flush();
  }
  deviceSyncer.sync(gridDim.x);

  // step 3: local reduce and exchange data with neighbor
  startChunkIndex = (rank / nRanksPerNode) * nRanksPerNode;
  if (isComm && pipelineSize > 1) {
    size_t offset = (peerRank * chunkSize + chunkSize / pipelineSize) * sizeof(int);
    proxyChan.putWithSignal(offset, (pipelineSize - 1) * chunkSize / pipelineSize * sizeof(int));
  }
  localReduceScatterSm(smChans, buff, rank, nRanksPerNode, startChunkIndex, 0, chunkSize, chunkSize,
                       nBlocksForReduceScatter);
  if (isComm && pipelineSize > 1) {
    proxyChan.wait();
  }
  deviceSyncer.sync(gridDim.x);
  // reduce to related rank, can not overlap since localReduceScatter also calculate the sum
  size_t offset = (rank * chunkSize + chunkSize / pipelineSize) * sizeof(int);
  int* dst = (int*)((char*)buff + offset);
  int* src = (int*)((char*)scratch + offset);
  if (pipelineSize > 1) vectorSum((TYPE*)dst, (TYPE*)src, (pipelineSize - 1) * chunkSize / pipelineSize);
  if (isComm) {
    proxyChan.flush();
  }
}

extern "C" __global__ void __launch_bounds__(1024, 1) __global__
    allreduce4(mscclpp::SmChannelDeviceHandle* smChans,
               mscclpp::SimpleProxyChannelDeviceHandle* reduceScatterProxyChans,
               mscclpp::SimpleProxyChannelDeviceHandle* allGatherProxyChans, TYPE* buff, TYPE* scratch, int rank,
               int nRanksPerNode, int worldSize, size_t nelems, int pipelineDepth) {
  nelems = nelems / (sizeof(int) / sizeof(TYPE));
  reduceScatterSm(smChans, reduceScatterProxyChans, buff, scratch, rank, nRanksPerNode, worldSize, nelems,
                  pipelineDepth);
  deviceSyncer.sync(gridDim.x);
  allGatherSm(smChans, allGatherProxyChans, rank, worldSize, nRanksPerNode, nelems / worldSize, pipelineDepth);
}
