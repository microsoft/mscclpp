#include <cassert>
#include <mscclpp/concurrency.hpp>
#include <vector>

#include "common.hpp"

#define ALIGN 4
#define BLOCKS_PER_PEER 15

__constant__ mscclpp::channel::SimpleDeviceChannel constDevFstRoundChans[16];
__constant__ mscclpp::channel::SimpleDeviceChannel constDevSndRoundChans[16];

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

__device__ void reduceSum(int* dst, int* src, size_t size) {
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    dst[i] += src[i];
  }
}

__device__ mscclpp::DeviceSyncer deviceSyncer;

__device__ void allreduce0(int rank, int worldSize, size_t nelems, size_t scratchDataCount) {
  int peerId = blockIdx.x / BLOCKS_PER_PEER;
  int isComm = (threadIdx.x == 0) && (blockIdx.x % BLOCKS_PER_PEER == 0);
  int remoteRank = (peerId < rank) ? peerId : peerId + 1;

  // 1st communication phase: send data to the scratch buffer of the peer associated with this block
  mscclpp::channel::SimpleDeviceChannel& devFstRoundChan = constDevFstRoundChans[peerId];
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
  mscclpp::channel::SimpleDeviceChannel& devSndRoundChan = constDevSndRoundChans[peerId];
  Chunk rankChunk = getChunk(nelems, worldSize, rank);
  int* chunk = (int*)devSndRoundChan.srcPtr_ + rankChunk.offset;
  int numPeers = gridDim.x / BLOCKS_PER_PEER;
  int numBlocks = gridDim.x;
  Chunk blockUserChunk = getChunk(rankChunk.size, numBlocks, blockIdx.x);
  size_t scratchDataCountPerPeer = scratchDataCount / numPeers;
  Chunk blockScratchChunk = getChunk(scratchDataCountPerPeer, numBlocks, blockIdx.x);
  for (int peerIdx = 0; peerIdx < numPeers; ++peerIdx) {
    int* scratchChunk = (int*)devFstRoundChan.tmpPtr_ + peerIdx * scratchDataCountPerPeer;
    reduceSum(chunk + blockUserChunk.offset, scratchChunk + blockScratchChunk.offset, blockScratchChunk.size);
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

__device__ void allreduce1(int rank, int worldSize, size_t nelems, size_t scratchDataCount) {
  int isComm = (threadIdx.x == 0) && (blockIdx.x == 0);
  int remoteSendRank = (rank + 1) % worldSize;
  int remoteRecvRank = (rank + worldSize - 1) % worldSize;
  int peerSendId = (remoteSendRank < rank) ? remoteSendRank : remoteSendRank - 1;
  int peerRecvId = (remoteRecvRank < rank) ? remoteRecvRank : remoteRecvRank - 1;

  mscclpp::channel::SimpleDeviceChannel& devFstSendChan = constDevFstRoundChans[peerSendId];
  mscclpp::channel::SimpleDeviceChannel& devFstRecvChan = constDevFstRoundChans[peerRecvId];
  mscclpp::channel::SimpleDeviceChannel& devSndSendChan = constDevSndRoundChans[peerSendId];
  mscclpp::channel::SimpleDeviceChannel& devSndRecvChan = constDevSndRoundChans[peerRecvId];

  // Step 1
  size_t chunkIndex = (rank + worldSize - 1) % worldSize;
  size_t chunkSize = nelems / worldSize * sizeof(int);
  size_t offset = chunkIndex * chunkSize;
  if (isComm) {
    devFstSendChan.putWithSignalAndFlush(offset, chunkSize);
  }

  // Step 2 ~ Step n-1
  for (int i = 2; i < worldSize; ++i) {
    if (isComm) {
      devFstRecvChan.wait();
    }
    deviceSyncer.sync(gridDim.x);

    // Reduce
    chunkIndex = (rank + worldSize - i) % worldSize;
    offset = chunkIndex * chunkSize;
    int* dst = (int*)((char*)devFstSendChan.srcPtr_ + offset);
    int* src = (int*)((char*)devFstRecvChan.tmpPtr_ + offset);
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < chunkSize / sizeof(int); i += blockDim.x * gridDim.x) {
      dst[i] += src[i];
    }
    deviceSyncer.sync(gridDim.x);

    if (isComm) {
      devFstSendChan.putWithSignalAndFlush(offset, chunkSize);
    }
  }

  // Step n
  if (isComm) {
    devFstRecvChan.wait();
  }
  deviceSyncer.sync(gridDim.x);
  offset = rank * chunkSize;
  int* dst = (int*)((char*)devFstSendChan.srcPtr_ + offset);
  int* src = (int*)((char*)devFstRecvChan.tmpPtr_ + offset);
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < chunkSize / sizeof(int); i += blockDim.x * gridDim.x) {
    dst[i] += src[i];
  }
  deviceSyncer.sync(gridDim.x);
  if (isComm) {
    devSndSendChan.putWithSignalAndFlush(offset, chunkSize);
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
      devSndSendChan.putWithSignalAndFlush(chunkIndex * chunkSize, chunkSize);
    }
  }

  // Final receive
  if (isComm) {
    devSndRecvChan.wait();
  }
}

__global__ void kernel(int rank, int worldSize, size_t nelems, size_t scratchDataCount, int kernel) {
  if (kernel == 0) allreduce0(rank, worldSize, nelems, scratchDataCount);
  else if (kernel == 1) allreduce1(rank, worldSize, nelems, scratchDataCount);
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
  const int nBlocks = (kernelNum == 0) ? nPeers * BLOCKS_PER_PEER : 32;
  kernel<<<nBlocks, 1024, 0, stream>>>(rank, worldSize, paramCount_, scratchDataCount, kernelNum);
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
  size_t base = (count / ALIGN) * ALIGN;
  sendCount_ = base;
  recvCount_ = base;
  paramCount_ = base;
  recvCount_ = base;

  mscclpp::DeviceSyncer syncer = {};
  CUDATHROW(cudaMemcpyToSymbol(deviceSyncer, &syncer, sizeof(mscclpp::DeviceSyncer)));
}

class AllReduceTestEngine : public BaseTestEngine {
 public:
  AllReduceTestEngine() = default;
  ~AllReduceTestEngine() = default;

  void allocateBuffer() override;
  void setupConnections() override;

 private:
  std::vector<void*> getSendBuff() override;
  void* getExpectedBuff() override;
  void* getRecvBuff() override;

  std::shared_ptr<int> sendBuff_;
  std::shared_ptr<int> scratchBuff_;
  std::shared_ptr<int[]> expectedBuff_;
};

void AllReduceTestEngine::allocateBuffer() {
  sendBuff_ = mscclpp::allocSharedCuda<int>(args_.maxBytes / sizeof(int));
  scratchBuff_ = mscclpp::allocSharedCuda<int>(args_.maxBytes / sizeof(int));
  expectedBuff_ = std::shared_ptr<int[]>(new int[args_.maxBytes / sizeof(int)]);
}

void AllReduceTestEngine::setupConnections() {
  std::vector<mscclpp::channel::SimpleDeviceChannel> fstRoundChannels;
  std::vector<mscclpp::channel::SimpleDeviceChannel> sndRoundChannels;

  // Send data from local sendBuff to remote scratchBuff (out-of-place)
  setupMeshConnections(fstRoundChannels, sendBuff_.get(), args_.maxBytes, scratchBuff_.get(), args_.maxBytes);
  assert(fstRoundChannels.size() < sizeof(constDevFstRoundChans) / sizeof(mscclpp::channel::SimpleDeviceChannel));
  CUDATHROW(cudaMemcpyToSymbol(constDevFstRoundChans, fstRoundChannels.data(),
                               sizeof(mscclpp::channel::SimpleDeviceChannel) * fstRoundChannels.size()));

  // Send data from local sendBuff to remote sendBuff (in-place)
  setupMeshConnections(sndRoundChannels, sendBuff_.get(), args_.maxBytes);
  assert(sndRoundChannels.size() < sizeof(constDevSndRoundChans) / sizeof(mscclpp::channel::SimpleDeviceChannel));
  CUDATHROW(cudaMemcpyToSymbol(constDevSndRoundChans, sndRoundChannels.data(),
                               sizeof(mscclpp::channel::SimpleDeviceChannel) * sndRoundChannels.size()));
}

std::vector<void*> AllReduceTestEngine::getSendBuff() { return {sendBuff_.get()}; }

void* AllReduceTestEngine::getExpectedBuff() { return expectedBuff_.get(); }

void* AllReduceTestEngine::getRecvBuff() {
  // in-place operation reuse the send buffer
  return sendBuff_.get();
}

std::shared_ptr<BaseTestEngine> getTestEngine() { return std::make_shared<AllReduceTestEngine>(); }
std::shared_ptr<BaseTestColl> getTestColl() { return std::make_shared<AllReduceTestColl>(); }
