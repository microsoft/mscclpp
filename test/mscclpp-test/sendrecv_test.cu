// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/semaphore.hpp>
#include <string>
#include <vector>

#include "common.hpp"

constexpr size_t BLOCK_THREADS_NUM = 1024;
// Try to use more blocks if per-block data size exceeds this threshold
constexpr size_t THRES_BYTES_PER_BLOCK = 8192;
// Let it no more than the number of SMs on a GPU
constexpr size_t MAX_BLOCKS_NUM = 32;

#define ALIGN 4

template <class T>
using DeviceHandle = mscclpp::DeviceHandle<T>;
__constant__ DeviceHandle<mscclpp::MemoryChannel> constMemChans[2];

inline int getBlockNum(size_t count) {
  return std::min((count + THRES_BYTES_PER_BLOCK - 1) / THRES_BYTES_PER_BLOCK, MAX_BLOCKS_NUM);
}

inline mscclpp::Transport getTransport(int rank, int peerRank, int nRanksPerNode, mscclpp::Transport ibDevice) {
  return rank / nRanksPerNode == peerRank / nRanksPerNode ? mscclpp::Transport::CudaIpc : ibDevice;
}

__device__ mscclpp::DeviceSyncer deviceSyncer;

__global__ void __launch_bounds__(1024) kernel(size_t dataSize, size_t dataPerBlock) {
  size_t startIndex = blockIdx.x * dataPerBlock;
  size_t blockDataSize = min(dataSize - startIndex, dataPerBlock);
  int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

  DeviceHandle<mscclpp::MemoryChannel> sendConn = constMemChans[0];
  DeviceHandle<mscclpp::MemoryChannel> recvConn = constMemChans[1];

  sendConn.put(startIndex, startIndex, blockDataSize, threadIdx.x, blockDim.x);
  deviceSyncer.sync(gridDim.x);
  if (globalIndex == 0) {
    sendConn.signal();
    recvConn.wait();
  }
}

class SendRecvTestColl : public BaseTestColl {
 public:
  SendRecvTestColl() = default;
  ~SendRecvTestColl() override = default;

  void runColl(const TestArgs& args, cudaStream_t stream) override;
  void initData(const TestArgs& args, std::vector<void*> sendBuff, void* expectedBuff) override;
  void getBw(const double deltaSec, double& algBw /*OUT*/, double& busBw /*OUT*/) override;
  void setupCollTest(size_t size) override;
  std::vector<KernelRestriction> getKernelRestrictions() override;
};

void SendRecvTestColl::runColl(const TestArgs&, cudaStream_t stream) {
  size_t sendBytes = sendCount_ * typeSize_;
  int blockNum = getBlockNum(sendBytes);
  size_t bytesPerBlock = (sendBytes + blockNum - 1) / blockNum;
  if (kernelNum_ == 0) {
    kernel<<<blockNum, BLOCK_THREADS_NUM, 0, stream>>>(sendBytes, bytesPerBlock);
  }
}

void SendRecvTestColl::getBw(const double deltaSec, double& algBw /*OUT*/, double& busBw /*OUT*/) {
  double baseBw = (double)(paramCount_ * typeSize_) / 1.0E9 / deltaSec;
  algBw = baseBw;
  double factor = 1;
  busBw = baseBw * factor;
}

std::vector<KernelRestriction> SendRecvTestColl::getKernelRestrictions() {
  return {// {kernelNum, kernelName, compatibleWithMultiNodes, countDivisorForMultiNodes, alignedBytes}
          {0, "sendrecv0", false, 1, 16 /*use ulong2 to transfer data*/}};
}

void SendRecvTestColl::initData(const TestArgs& args, std::vector<void*> sendBuff, void* expectedBuff) {
  int rank = args.rank;
  if (sendBuff.size() != 1) std::runtime_error("unexpected error");
  MSCCLPP_CUDATHROW(cudaMemset(sendBuff[0], 0, sendCount_ * typeSize_));

  // TODO: The type should not limited to int.
  std::vector<int> dataHost(std::max(sendCount_, recvCount_), rank);
  MSCCLPP_CUDATHROW(cudaMemcpy(sendBuff[0], dataHost.data(), sendCount_ * typeSize_, cudaMemcpyHostToDevice));

  int peerRank = (rank - 1 + args.totalRanks) % args.totalRanks;
  for (size_t i = 0; i < recvCount_; i++) {
    dataHost[i] = peerRank;
  }

  std::memcpy(expectedBuff, dataHost.data(), recvCount_ * typeSize_);
}

void SendRecvTestColl::setupCollTest(size_t size) {
  size_t count = size / typeSize_;
  size_t base = (count / ALIGN) * ALIGN;
  sendCount_ = base;
  recvCount_ = base;
  paramCount_ = base;
  expectedCount_ = base;

  mscclpp::DeviceSyncer syncer = {};
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(deviceSyncer, &syncer, sizeof(mscclpp::DeviceSyncer)));
}

class SendRecvTestEngine : public BaseTestEngine {
 public:
  SendRecvTestEngine(const TestArgs& args);
  ~SendRecvTestEngine() override = default;

  void allocateBuffer() override;
  void setupConnections() override;

  std::vector<void*> getSendBuff() override;
  void* getRecvBuff() override;
  void* getScratchBuff() override;

 private:
  void* getExpectedBuff() override;

  std::vector<std::shared_ptr<int>> devicePtrs_;
  std::shared_ptr<int[]> expectedBuff_;
  std::vector<mscclpp::MemoryChannel> memoryChannels_;
};

SendRecvTestEngine::SendRecvTestEngine(const TestArgs& args) : BaseTestEngine(args, "sendrecv") { inPlace_ = false; }

void SendRecvTestEngine::allocateBuffer() {
  std::shared_ptr<int> sendBuff = mscclpp::GpuBuffer<int>(args_.maxBytes / sizeof(int)).memory();
  std::shared_ptr<int> recvBuff = mscclpp::GpuBuffer<int>(args_.maxBytes / sizeof(int)).memory();
  devicePtrs_.push_back(sendBuff);
  devicePtrs_.push_back(recvBuff);

  expectedBuff_ = std::shared_ptr<int[]>(new int[args_.maxBytes / sizeof(int)]);
}

void SendRecvTestEngine::setupConnections() {
  auto ibDevice = IBs[args_.localRank];
  int worldSize = args_.totalRanks;
  int sendToRank = (args_.rank + 1) % worldSize;
  int recvFromRank = (args_.rank - 1 + worldSize) % worldSize;
  std::array<int, 2> ranks = {sendToRank, recvFromRank};
  auto service = std::dynamic_pointer_cast<mscclpp::ProxyService>(chanService_);

  std::vector<mscclpp::Semaphore> semaphores;
  if (recvFromRank != sendToRank) {
    auto sendTransport = getTransport(args_.rank, sendToRank, args_.nRanksPerNode, ibDevice);
    auto recvTransport = getTransport(args_.rank, recvFromRank, args_.nRanksPerNode, ibDevice);
    auto connFutures = {comm_->connect(sendTransport, sendToRank), comm_->connect(recvTransport, recvFromRank)};
    auto semaFutures = {comm_->buildSemaphore(connFutures.begin()->get(), sendToRank),
                        comm_->buildSemaphore((connFutures.begin() + 1)->get(), recvFromRank)};
    semaphores.emplace_back(semaFutures.begin()->get());
    semaphores.emplace_back((semaFutures.begin() + 1)->get());
  } else {
    auto sendTransport = getTransport(args_.rank, sendToRank, args_.nRanksPerNode, ibDevice);
    auto connFuture = comm_->connect(sendTransport, sendToRank);
    semaphores.emplace_back(comm_->buildSemaphore(connFuture.get(), sendToRank).get());
    semaphores.emplace_back(semaphores[0]);  // reuse the semaphore if worldSize is 2
  }

  std::vector<mscclpp::RegisteredMemory> localMemories;
  std::vector<std::shared_future<mscclpp::RegisteredMemory>> futureRemoteMemory;

  for (int i : {0, 1}) {
    auto regMem = comm_->registerMemory(devicePtrs_[i].get(), args_.maxBytes, mscclpp::Transport::CudaIpc | ibDevice);
    comm_->sendMemory(regMem, ranks[i]);
    localMemories.push_back(regMem);
    futureRemoteMemory.push_back(comm_->recvMemory(ranks[1 - i]));
  }

  // swap to make sure devicePtrs_[0] in local rank write to devicePtrs_[1] in remote rank
  std::swap(futureRemoteMemory[0], futureRemoteMemory[1]);
  std::vector<DeviceHandle<mscclpp::MemoryChannel>> memoryChannelHandles(2);
  for (int i : {0, 1}) {
    // We assume ranks in the same node
    memoryChannels_.emplace_back(semaphores[i], futureRemoteMemory[i].get(), (void*)localMemories[i].data());
  }
  std::transform(memoryChannels_.begin(), memoryChannels_.end(), memoryChannelHandles.begin(),
                 [](const mscclpp::MemoryChannel& memoryChannel) { return memoryChannel.deviceHandle(); });
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(constMemChans, memoryChannelHandles.data(),
                                       sizeof(DeviceHandle<mscclpp::MemoryChannel>) * memoryChannelHandles.size()));
}

std::vector<void*> SendRecvTestEngine::getSendBuff() { return {devicePtrs_[0].get()}; }

void* SendRecvTestEngine::getExpectedBuff() { return expectedBuff_.get(); }

void* SendRecvTestEngine::getRecvBuff() { return devicePtrs_[1].get(); }

void* SendRecvTestEngine::getScratchBuff() { return nullptr; }

std::shared_ptr<BaseTestEngine> getTestEngine(const TestArgs& args) {
  return std::make_shared<SendRecvTestEngine>(args);
}

std::shared_ptr<BaseTestColl> getTestColl() { return std::make_shared<SendRecvTestColl>(); }
