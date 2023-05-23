#include <vector>
#include <cassert>
#include "common.hpp"

#define ALIGN 4
__constant__ mscclpp::channel::SimpleDeviceChannel constDevFstRoundChans[16];
__constant__ mscclpp::channel::SimpleDeviceChannel constDevSndRoundChans[16];

struct Chunk
{
  size_t offset;
  size_t size;
};

__host__ __device__ Chunk getChunk(size_t dataCount, size_t numChunks, size_t chunkIdx, size_t chunkCount)
{
  size_t remainder = dataCount % numChunks;
  size_t smallChunkSize = dataCount / numChunks;
  size_t largeChunkSize = smallChunkSize + 1;
  size_t numLargeChunks = chunkIdx < remainder ? remainder - chunkIdx : 0;
  size_t numSmallChunks = chunkCount - numLargeChunks;
  size_t offset =
    (remainder - numLargeChunks) * largeChunkSize + (chunkIdx > remainder ? chunkIdx - remainder : 0) * smallChunkSize;
  return Chunk{offset, numLargeChunks * largeChunkSize + numSmallChunks * smallChunkSize};
}

__device__ void allreduce0(int rank, int worldSize, size_t nelems, size_t scratchDataCount)
{
}

__global__ void kernel(int rank, int worldSize, size_t nelems, size_t scratchDataCount, int kernel) {
  if (kernel == 0)
    allreduce0(rank, worldSize, nelems, scratchDataCount);
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
  const Chunk chunk = getChunk(paramCount_, worldSize, rank, 1);
  const size_t scratchDataCount = chunk.size * nPeers;
  kernel<<<worldSize - 1, 256, 0, stream>>>(rank, worldSize, paramCount_, scratchDataCount, kernelNum);
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

void AllReduceTestColl::getBw(const double deltaSec, double& algBw /*OUT*/, double& busBw /*OUT*/)
{
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

void AllReduceTestEngine::setupConnections()
{
  std::vector<mscclpp::channel::SimpleDeviceChannel> fstRoundChannels;
  std::vector<mscclpp::channel::SimpleDeviceChannel> sndRoundChannels;

  setupMeshConnections(fstRoundChannels, scratchBuff_.get(), args_.maxBytes);
  assert(fstRoundChannels.size() < sizeof(constDevFstRoundChans) / sizeof(mscclpp::channel::SimpleDeviceChannel));
  CUDATHROW(cudaMemcpyToSymbol(constDevFstRoundChans, fstRoundChannels.data(),
                               sizeof(mscclpp::channel::SimpleDeviceChannel) * fstRoundChannels.size()));

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
