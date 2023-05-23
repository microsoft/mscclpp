#include <cuda_runtime.h>

#include <cassert>
#include <string>

#include "common.hpp"

#define ALIGN 4
__constant__ mscclpp::channel::SimpleDeviceChannel constDevChans[16];

__device__ void allgather0(mscclpp::channel::SimpleDeviceChannel devChan, int rank, int worldSize, int remoteRank,
                           size_t nelemsPerGPU) {
  // this allgather is really simple and implemented as an alltoall

  // this thread's role is a sender role
  // put your data asynchronously
  if (threadIdx.x % 32 == 0) devChan.putWithSignal(rank * nelemsPerGPU * sizeof(int), nelemsPerGPU * sizeof(int));
  // make sure everyone is put their data before some thread randomly blocks everyone else in signal
  __syncthreads();
  // push with flag and sync to make sure the data is received
  if (threadIdx.x % 32 == 0) devChan.flush();

  // this thread's role is a receiver role. wait on the semaphore to make sure the data is ready
  if (threadIdx.x % 32 == 0) devChan.wait();
}

__device__ void localAllGather(mscclpp::channel::SimpleDeviceChannel devChan, int rank, int worldSize,
                               int nranksPerNode, int remoteRank, uint64_t offset, uint64_t size) {
  // this allgather algorithm works as follows:
  // Step 1: GPU rank i sends data to GPU rank (i+1) % nranksPerNode
  // and waits for data from GPU rank (i-1) % nranksPerNode
  // Step 2: GPU rank i sends data to GPU rank (i+2) % nranksPerNode
  // ...
  // This order is much better for DMA engine for NVLinks
  for (int i = 1; i < nranksPerNode; i++) {
    if ((remoteRank % nranksPerNode) == ((rank + i) % nranksPerNode)) {
      // put your data to GPU (rank+i) % nranksPerNode and signal in one call
      if ((threadIdx.x % 32) == 0) devChan.putWithSignalAndFlush(offset, size);
    }
    // wait for the data from GPU (rank-i) % nranksPerNode to arrive
    if ((remoteRank % nranksPerNode) == ((rank - i + nranksPerNode) % nranksPerNode)) {
      if ((threadIdx.x % 32) == 0) devChan.wait();
    }
    asm volatile("bar.sync %0, %1;" ::"r"(11), "r"((nranksPerNode - 1) * 32) : "memory");
  }
}

__device__ void allgather1(mscclpp::channel::SimpleDeviceChannel devChan, int rank, int worldSize, int nranksPerNode,
                           int remoteRank, size_t nelemsPerGPU) {
  localAllGather(devChan, rank, worldSize, nranksPerNode, remoteRank, rank * nelemsPerGPU * sizeof(int),
                 nelemsPerGPU * sizeof(int));
}

__device__ void allgather2(mscclpp::channel::SimpleDeviceChannel devChan, int rank, int worldSize, int nranksPerNode,
                           int remoteRank, size_t nelemsPerGPU) {
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
  if (remoteRank / nranksPerNode == rank / nranksPerNode) {
    localAllGather(devChan, rank, worldSize, nranksPerNode, remoteRank, rank * nelemsPerGPU * sizeof(int),
                   nelemsPerGPU * sizeof(int));
  }
  // cross-node exchange
  if (remoteRank % nranksPerNode == rank % nranksPerNode) {
    // opposite side
    if ((threadIdx.x % 32) == 0)
      devChan.putWithSignalAndFlush(rank * nelemsPerGPU * sizeof(int),
                                    (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize * sizeof(int));
    if ((threadIdx.x % 32) == 0) devChan.wait();
  }

  __syncthreads();

  // Step 2
  // local allgather
  int otherNghr = (rank + nranksPerNode) % worldSize;
  if (remoteRank / nranksPerNode == rank / nranksPerNode) {
    localAllGather(devChan, rank, worldSize, nranksPerNode, remoteRank, otherNghr * nelemsPerGPU * sizeof(int),
                   (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize * sizeof(int));
  }

  // cross-node exchange
  if (remoteRank % nranksPerNode == rank % nranksPerNode) {
    // opposite side
    if ((threadIdx.x % 32) == 0)
      devChan.putWithSignalAndFlush(
          (rank * nelemsPerGPU + (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize) * sizeof(int),
          nelemsPerGPU / pipelineSize * sizeof(int));
    if ((threadIdx.x % 32) == 0) devChan.wait();
  }

  __syncthreads();

  // Step 3
  // local allgather
  if (remoteRank / nranksPerNode == rank / nranksPerNode) {
    localAllGather(devChan, rank, worldSize, nranksPerNode, remoteRank,
                   (otherNghr * nelemsPerGPU + (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize) * sizeof(int),
                   nelemsPerGPU / pipelineSize * sizeof(int));
  }
}

__global__ void kernel(int rank, int worldSize, int nranksPerNode, size_t nelemsPerGPU, int kernel) {
  // find the mapping between remoteRank and devConns
  int warpId = threadIdx.x / 32;
  int remoteRank = (warpId < rank) ? warpId : warpId + 1;
  // Each warp is responsible for one of the remote ranks
  mscclpp::channel::SimpleDeviceChannel devChan = constDevChans[warpId];

  if (kernel == 0)
    allgather0(devChan, rank, worldSize, remoteRank, nelemsPerGPU);
  else if (kernel == 1)
    allgather1(devChan, rank, worldSize, nranksPerNode, remoteRank, nelemsPerGPU);
  else if (kernel == 2)
    allgather2(devChan, rank, worldSize, nranksPerNode, remoteRank, nelemsPerGPU);
}

class AllGatherTestColl : public BaseTestColl {
 public:
  AllGatherTestColl() = default;
  ~AllGatherTestColl() override = default;

  void runColl(const TestArgs& args, cudaStream_t stream) override;
  void initData(const TestArgs& args, std::vector<void*> sendBuff, void* expectedBuff) override;
  void getBw(const double deltaSec, double& algBW /*OUT*/, double& busBw /*OUT*/) override;
  void setupCollTest(size_t size) override;
};

void AllGatherTestColl::runColl(const TestArgs& args, cudaStream_t stream) {
  const int worldSize = args.totalRanks;
  const int rank = args.rank;
  const int nRanksPerNode = args.nRanksPerNode;
  const int kernelNum = args.kernelNum;
  kernel<<<1, 32 * (worldSize - 1), 0, stream>>>(rank, worldSize, nRanksPerNode, paramCount_, kernelNum);
}

void AllGatherTestColl::initData(const TestArgs& args, std::vector<void*> sendBuff, void* expectedBuff) {
  assert(sendBuff.size() == 1);
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
  size_t base = (count / (ALIGN * worldSize_)) * ALIGN;
  sendCount_ = base;
  recvCount_ = base * worldSize_;
  paramCount_ = base;
  expectedCount_ = recvCount_;
}

class AllGatherTestEngine : public BaseTestEngine {
 public:
  AllGatherTestEngine() = default;
  ~AllGatherTestEngine() override = default;

  void allocateBuffer() override;
  void setupConnections() override;

 private:
  std::vector<void*> getSendBuff() override;
  void* getExpectedBuff() override;
  void* getRecvBuff() override;

  std::shared_ptr<int> sendBuff_;
  std::shared_ptr<int[]> expectedBuff_;
};

void AllGatherTestEngine::allocateBuffer() {
  sendBuff_ = mscclpp::makeSharedCuda<int>(args_.maxBytes / sizeof(int));
  expectedBuff_ = std::shared_ptr<int[]>(new int[args_.maxBytes / sizeof(int)]);
}

void AllGatherTestEngine::setupConnections() {
  const int worldSize = args_.totalRanks;
  const int rank = args_.rank;
  const int nRanksPerNode = args_.nRanksPerNode;
  const int thisNode = rank / nRanksPerNode;
  const mscclpp::Transport ibTransport = IBs[args_.gpuNum];

  std::vector<mscclpp::channel::ChannelId> channelIds;
  std::vector<mscclpp::RegisteredMemory> localMemories;
  std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteMemories;

  auto rankToNode = [&](int rank) { return rank / nRanksPerNode; };
  for (int r = 0; r < worldSize; r++) {
    if (r == rank) {
      continue;
    }
    mscclpp::Transport transport;
    if (rankToNode(r) == thisNode) {
      transport = mscclpp::Transport::CudaIpc;
    } else {
      transport = ibTransport;
    }
    // Connect with all other ranks
    channelIds.push_back(chanService_->addChannel(comm_->connectOnSetup(r, 0, transport)));
    auto memory = comm_->registerMemory(sendBuff_.get(), args_.maxBytes, mscclpp::Transport::CudaIpc | ibTransport);
    localMemories.push_back(memory);
    comm_->sendMemoryOnSetup(memory, r, 0);
    remoteMemories.push_back(comm_->recvMemoryOnSetup(r, 0));
  }
  comm_->setup();

  std::vector<mscclpp::channel::SimpleDeviceChannel> devChannels;
  for (size_t i = 0; i < channelIds.size(); ++i) {
    devChannels.push_back(mscclpp::channel::SimpleDeviceChannel(chanService_->deviceChannel(channelIds[i]),
                                                                chanService_->addMemory(remoteMemories[i].get()),
                                                                chanService_->addMemory(localMemories[i])));
  }

  assert(devChannels.size() < sizeof(constDevChans) / sizeof(mscclpp::channel::SimpleDeviceChannel));
  CUDATHROW(cudaMemcpyToSymbol(constDevChans, devChannels.data(),
                               sizeof(mscclpp::channel::SimpleDeviceChannel) * devChannels.size()));
}

std::vector<void*> AllGatherTestEngine::getSendBuff() { return {sendBuff_.get()}; }

void* AllGatherTestEngine::getExpectedBuff() { return expectedBuff_.get(); }

void* AllGatherTestEngine::getRecvBuff() {
  // in-place operation reuse the send buffer
  return sendBuff_.get();
}

std::shared_ptr<BaseTestEngine> getTestEngine() { return std::make_shared<AllGatherTestEngine>(); }
std::shared_ptr<BaseTestColl> getTestColl() { return std::make_shared<AllGatherTestColl>(); }
