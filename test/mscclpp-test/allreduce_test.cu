#include <cassert>
#include <mscclpp/concurrency.hpp>
#include <vector>

#include "common.hpp"

#define ALIGN 4
__constant__ mscclpp::channel::SimpleDeviceChannel constDevFstRoundChans[16];
__constant__ mscclpp::channel::SimpleDeviceChannel constDevSndRoundChans[16];

struct Chunk {
  size_t offset;
  size_t size;
};

__host__ __device__ Chunk getChunk(size_t dataCount, size_t numChunks, size_t chunkIdx)
{
  size_t remainder = dataCount % numChunks;
  size_t smallChunkSize = dataCount / numChunks;
  size_t largeChunkSize = smallChunkSize + 1;
  size_t numRemainedLargeChunks = chunkIdx < remainder ? remainder - chunkIdx : 0;
  size_t offset = (remainder - numRemainedLargeChunks) * largeChunkSize +
                  (chunkIdx > remainder ? chunkIdx - remainder : 0) * smallChunkSize;
  return Chunk{offset, chunkIdx < remainder ? largeChunkSize : smallChunkSize};
}

__device__ void send(mscclpp::channel::SimpleDeviceChannel& chan, size_t dstOffset, size_t srcOffset, size_t size) {
  if (threadIdx.x == 0) {
    chan.putWithSignalAndFlush(dstOffset, srcOffset, size);
  }
  __syncthreads();
}

__device__ void recv(mscclpp::channel::SimpleDeviceChannel& chan) {
  if (threadIdx.x == 0) {
    chan.wait();
  }
  __syncthreads();
}

__device__ void reduceSum(int* dst, int* src, size_t size) {
  for (int i = threadIdx.x; i < size; i += blockDim.x) {
    dst[i] += src[i];
  }
}

__device__ mscclpp::DeviceSyncer deviceSyncer;

__device__ void allreduce0(int rank, int worldSize, size_t nelems, size_t scratchDataCount) {
  int remoteRank = (blockIdx.x < rank) ? blockIdx.x : blockIdx.x + 1;

  // 1st communication phase: send data to the scratch buffer of the peer associated with this block
  mscclpp::channel::SimpleDeviceChannel devFstRoundChan = constDevFstRoundChans[blockIdx.x];
  Chunk toPeerChunk = getChunk(nelems, worldSize, remoteRank);
  // Now we need to figure out the offset of this chunk in the scratch buffer of the destination.
  // The destination will have allocated a scratch buffer of size numPeers() * toPeerChunk.size and
  // inside that each of the destination's peers send to the nth chunk, where n is the index of the
  // source peer from the destination's perspective.
  size_t dstOffset = (rank < remoteRank ? rank : rank - 1) * toPeerChunk.size;
  send(devFstRoundChan, dstOffset * sizeof(int), toPeerChunk.offset * sizeof(int), toPeerChunk.size * sizeof(int));
  recv(devFstRoundChan);

  deviceSyncer.sync(gridDim.x);

  // Local reduction: every block reduces a slice of each chunk in the scratch buffer into the user buffer
  mscclpp::channel::SimpleDeviceChannel devSndRoundChan = constDevSndRoundChans[blockIdx.x];
  Chunk rankChunk = getChunk(nelems, worldSize, rank);
  int* chunk = (int*)devSndRoundChan.srcPtr_ + rankChunk.offset;
  int numPeers = gridDim.x;
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
  send(devSndRoundChan, collectionChunk.offset * sizeof(int), collectionChunk.offset * sizeof(int),
       collectionChunk.size * sizeof(int));
  recv(devSndRoundChan);
}

__global__ void kernel(int rank, int worldSize, size_t nelems, size_t scratchDataCount, int kernel) {
  if (kernel == 0) allreduce0(rank, worldSize, nelems, scratchDataCount);
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
  kernel<<<worldSize - 1, 1024, 0, stream>>>(rank, worldSize, paramCount_, scratchDataCount, kernelNum);
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
  sendBuff_ = mscclpp::makeSharedCuda<int>(args_.maxBytes / sizeof(int));
  scratchBuff_ = mscclpp::makeSharedCuda<int>(args_.maxBytes / sizeof(int));
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
