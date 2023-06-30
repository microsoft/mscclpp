// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cassert>
#include <mscclpp/concurrency.hpp>

#include "common.hpp"

#define ALIGN 4
__constant__ mscclpp::SimpleProxyChannel constDevChans[16];
__device__ mscclpp::DeviceSyncer deviceSyncer;
void* localRecvBuff;
void* localSendBuff;

__device__ void localAlltoall(int rank, int nRanksPerNode, size_t nElements) {
  int remoteRank = (blockIdx.x < rank) ? blockIdx.x : blockIdx.x + 1;
  for (int i = 1; i < nRanksPerNode; i++) {
    mscclpp::SimpleProxyChannel devChan = constDevChans[blockIdx.x];
    if (threadIdx.x == 0 && remoteRank % nRanksPerNode == (rank + i) % nRanksPerNode) {
      devChan.putWithSignalAndFlush(rank * nElements * sizeof(int), remoteRank * nElements * sizeof(int),
                                    nElements * sizeof(int));
    }
    // wait for the data from GPU (rank-i) % nranksPerNode to arrive
    if (threadIdx.x == 0 && remoteRank % nRanksPerNode == (rank - i + nRanksPerNode) % nRanksPerNode) {
      devChan.wait();
    }
    deviceSyncer.sync(nRanksPerNode - 1);
  }
}

__device__ void alltoall0(int rank, int worldSize, size_t nElements) {
  int remoteRank = (blockIdx.x < rank) ? blockIdx.x : blockIdx.x + 1;
  mscclpp::SimpleProxyChannel devChan = constDevChans[blockIdx.x];
  if (threadIdx.x == 0) {
    devChan.putWithSignal(rank * nElements * sizeof(int), remoteRank * nElements * sizeof(int),
                          nElements * sizeof(int));
  }

  deviceSyncer.sync(gridDim.x);
  if (threadIdx.x == 0) {
    devChan.flush();
    devChan.wait();
  }
}

__device__ void alltoall1(int rank, int nRanksPerNode, size_t nElements) {
  localAlltoall(rank, nRanksPerNode, nElements);
}

__global__ void kernel(int rank, int worldSize, size_t nElements, int nRanksPerNode, int kernelNum) {
  if (kernelNum == 0) {
    alltoall0(rank, worldSize, nElements);
  }
  if (kernelNum == 1) {
    alltoall1(rank, nRanksPerNode, nElements);
  }
}

class AllToAllTestColl : public BaseTestColl {
 public:
  AllToAllTestColl() = default;
  ~AllToAllTestColl() override = default;

  void runColl(const TestArgs& args, cudaStream_t stream) override;
  void initData(const TestArgs& args, std::vector<void*> sendBuff, void* expectedBuff) override;
  void getBw(const double deltaSec, double& algBw /*OUT*/, double& busBw /*OUT*/) override;
  void setupCollTest(size_t size) override;
  std::vector<KernelRestriction> getKernelRestrictions() override;
};

void AllToAllTestColl::runColl(const TestArgs& args, cudaStream_t stream) {
  const int worldSize = args.totalRanks;
  const int rank = args.rank;
  const int kernelNum = args.kernelNum;
  const int nRanksPerNode = args.nRanksPerNode;
  CUDATHROW(cudaMemcpyAsync((int*)localRecvBuff + paramCount_ * rank, (int*)localSendBuff + paramCount_ * rank,
                            paramCount_ * sizeof(int), cudaMemcpyDeviceToDevice, stream));
  kernel<<<worldSize - 1, 32, 0, stream>>>(rank, worldSize, paramCount_, nRanksPerNode, kernelNum);
}

void AllToAllTestColl::initData(const TestArgs& args, std::vector<void*> sendBuff, void* expectedBuff) {
  assert(sendBuff.size() == 1);
  const int rank = args.rank;
  std::vector<int> dataHost(recvCount_, 0);
  // For rank 0, the data is 0, 1, 2 ... recvCount_ - 1, for rank 1, the data is recvCount_, recvCount_ + 1, ...
  for (size_t i = 0; i < recvCount_; i++) {
    dataHost[i] = rank * recvCount_ + i;
  }
  CUDATHROW(cudaMemcpy(sendBuff[0], dataHost.data(), sendCount_ * typeSize_, cudaMemcpyHostToDevice));
  for (size_t i = 0; i < recvCount_ / paramCount_; i++) {
    for (int j = 0; j < paramCount_; j++) {
      dataHost[i * paramCount_ + j] = i * recvCount_ + rank * paramCount_ + j;
    }
  }
  std::memcpy(expectedBuff, dataHost.data(), recvCount_ * typeSize_);
}

void AllToAllTestColl::getBw(const double deltaSec, double& algBw, double& busBw) {
  double baseBw = (double)(paramCount_ * typeSize_ * worldSize_) / 1.0E9 / deltaSec;
  algBw = baseBw;
  double factor = ((double)(worldSize_ - 1)) / ((double)worldSize_);
  busBw = baseBw * factor;
}

void AllToAllTestColl::setupCollTest(size_t size) {
  size_t count = size / typeSize_;
  size_t base = (count / worldSize_ / (ALIGN)) * ALIGN * worldSize_;
  sendCount_ = base;
  recvCount_ = base;
  paramCount_ = base / worldSize_;
  expectedCount_ = base;

  mscclpp::DeviceSyncer syncer = {};
  CUDATHROW(cudaMemcpyToSymbol(deviceSyncer, &syncer, sizeof(mscclpp::DeviceSyncer)));
}

std::vector<KernelRestriction> AllToAllTestColl::getKernelRestrictions() {
  return {// {kernelNum, kernelName, compatibleWithMultiNodes, countDivisorForMultiNodes}
          {0, "alltoall0", true, 1},
          {1, "alltoall1", false, 1}};
}

class AllToAllTestEngine : public BaseTestEngine {
 public:
  AllToAllTestEngine(const TestArgs& args);
  ~AllToAllTestEngine() override = default;

  void allocateBuffer() override;
  void setupConnections() override;

  std::vector<void*> getSendBuff() override;
  void* getRecvBuff() override;
  void* getScratchBuff() override;

 private:
  void* getExpectedBuff() override;

  std::shared_ptr<int> sendBuff_;
  std::shared_ptr<int> recvBuff_;
  std::shared_ptr<int[]> expectedBuff_;
};

AllToAllTestEngine::AllToAllTestEngine(const TestArgs& args) : BaseTestEngine(args, "alltoall") { inPlace_ = false; }

void AllToAllTestEngine::allocateBuffer() {
  sendBuff_ = mscclpp::allocSharedCuda<int>(args_.maxBytes / sizeof(int));
  recvBuff_ = mscclpp::allocSharedCuda<int>(args_.maxBytes / sizeof(int));
  expectedBuff_ = std::shared_ptr<int[]>(new int[args_.maxBytes / sizeof(int)]);

  localSendBuff = sendBuff_.get();
  localRecvBuff = recvBuff_.get();
}

void AllToAllTestEngine::setupConnections() {
  std::vector<mscclpp::SimpleProxyChannel> devChannels;
  setupMeshConnections(devChannels, sendBuff_.get(), args_.maxBytes, recvBuff_.get(), args_.maxBytes);

  assert(devChannels.size() < sizeof(constDevChans) / sizeof(mscclpp::SimpleProxyChannel));
  CUDATHROW(
      cudaMemcpyToSymbol(constDevChans, devChannels.data(), sizeof(mscclpp::SimpleProxyChannel) * devChannels.size()));
}

std::vector<void*> AllToAllTestEngine::getSendBuff() { return {sendBuff_.get()}; }
void* AllToAllTestEngine::getExpectedBuff() { return expectedBuff_.get(); }
void* AllToAllTestEngine::getRecvBuff() { return recvBuff_.get(); }
void* AllToAllTestEngine::getScratchBuff() { return nullptr; }

std::shared_ptr<BaseTestEngine> getTestEngine(const TestArgs& args) {
  return std::make_shared<AllToAllTestEngine>(args);
}
std::shared_ptr<BaseTestColl> getTestColl() { return std::make_shared<AllToAllTestColl>(); }
