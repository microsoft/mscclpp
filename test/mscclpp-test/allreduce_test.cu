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

__constant__ mscclpp::SmChannel constSmChans[8];
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
  mscclpp::SmChannel smChan = constSmChans[smChanIdx];

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

__global__ void kernel(void* buff, void* scratch, void* result, void* putPktBuf, void* getPktBuf, int rank,
                       int nRanksPerNode, int worldSize, size_t nelems, size_t scratchDataCount, int kernel) {
  if (kernel == 0)
    allreduce0((int*)buff, (int*)scratch, rank, worldSize, nelems, scratchDataCount);
  else if (kernel == 1)
    allreduce1((int*)buff, (int*)scratch, rank, worldSize, nelems, scratchDataCount);
  else if (kernel == 2)
    allreduce2((int*)buff, scratch, putPktBuf, getPktBuf, result, rank, nRanksPerNode, worldSize, nelems);
}

class AllReduceTestColl : public BaseTestColl {
 public:
  AllReduceTestColl() = default;
  ~AllReduceTestColl() = default;

  void runColl(const TestArgs& args, cudaStream_t stream) override;
  void initData(const TestArgs& args, std::vector<void*> sendBuff, void* expectedBuff) override;
  void getBw(const double deltaSec, double& algBw /*OUT*/, double& busBw /*OUT*/) override;
  void setupCollTest(size_t size) override;
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
  } else if (kernelNum == 1) {
    nBlocks = 24;
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

class AllReduceTestEngine : public BaseTestEngine {
 public:
  AllReduceTestEngine(const TestArgs& args);
  ~AllReduceTestEngine() = default;

  void allocateBuffer() override;
  std::shared_ptr<mscclpp::BaseProxyService> createChannelService() override;
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

  if (args_.kernelNum == 0 || args_.kernelNum == 1) {
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

std::shared_ptr<mscclpp::BaseProxyService> AllReduceTestEngine::createChannelService() {
  if (isUsePacket()) {
    return std::make_shared<mscclpp::ProxyService>(*comm_);
  } else {
    return std::make_shared<mscclpp::ProxyService>(*comm_);
  }
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

    assert(smChannels.size() < sizeof(constSmChans) / sizeof(mscclpp::SmChannel));
    assert(devChannels.size() < sizeof(constDevFstRoundChans) / sizeof(mscclpp::SimpleProxyChannel));
    CUDATHROW(cudaMemcpyToSymbol(constSmChans, smChannels.data(), sizeof(mscclpp::SmChannel) * smChannels.size()));
    CUDATHROW(cudaMemcpyToSymbol(constDevFstRoundChans, devChannels.data(),
                                 sizeof(mscclpp::SimpleProxyChannel) * devChannels.size()));
  } else {
    std::vector<mscclpp::SimpleProxyChannel> fstRoundChannels;
    std::vector<mscclpp::SimpleProxyChannel> sndRoundChannels;

    // Send data from local sendBuff to remote scratchBuff (out-of-place)
    setupMeshConnections(fstRoundChannels, inputBuff_.get(), args_.maxBytes, scratchBuff_.get(), args_.maxBytes);
    assert(fstRoundChannels.size() < sizeof(constDevFstRoundChans) / sizeof(mscclpp::SimpleProxyChannel));
    CUDATHROW(cudaMemcpyToSymbol(constDevFstRoundChans, fstRoundChannels.data(),
                                 sizeof(mscclpp::SimpleProxyChannel) * fstRoundChannels.size()));

    // Send data from local sendBuff to remote sendBuff (in-place)
    setupMeshConnections(sndRoundChannels, inputBuff_.get(), args_.maxBytes);
    assert(sndRoundChannels.size() < sizeof(constDevSndRoundChans) / sizeof(mscclpp::SimpleProxyChannel));
    CUDATHROW(cudaMemcpyToSymbol(constDevSndRoundChans, sndRoundChannels.data(),
                                 sizeof(mscclpp::SimpleProxyChannel) * sndRoundChannels.size()));
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
