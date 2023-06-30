// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cuda_runtime.h>

#include <cassert>
#include <mscclpp/concurrency.hpp>
#include <string>

#include "common.hpp"

#define ALIGN 4

namespace {
auto isUsingHostOffload = [](int kernelNum) { return kernelNum == 3; };
constexpr uint64_t MAGIC = 0xdeadbeef;
}  // namespace

__constant__ mscclpp::SimpleProxyChannel constDevChans[16];
__constant__ mscclpp::ProxyChannel constRawDevChan[16];

__constant__ mscclpp::SmChannel constSmChans[8];

__device__ void allgather0(mscclpp::SimpleProxyChannel devChan, int rank, int worldSize, int remoteRank,
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

__device__ void localAllGather(mscclpp::SimpleProxyChannel devChan, int rank, int worldSize, int nRanksPerNode,
                               int remoteRank, uint64_t offset, uint64_t size, bool flushAfterSignal = true) {
  // this allgather algorithm works as follows:
  // Step 1: GPU rank i sends data to GPU rank (i+1) % nRanksPerNode
  // and waits for data from GPU rank (i-1) % nRanksPerNode
  // Step 2: GPU rank i sends data to GPU rank (i+2) % nRanksPerNode
  // ...
  // This order is much better for DMA engine for NVLinks
  for (int i = 1; i < nRanksPerNode; i++) {
    if ((remoteRank % nRanksPerNode) == ((rank + i) % nRanksPerNode)) {
      // put your data to GPU (rank+i) % nRanksPerNode and signal in one call
      if (flushAfterSignal && (threadIdx.x % 32) == 0) devChan.putWithSignalAndFlush(offset, size);
      if (!flushAfterSignal && (threadIdx.x % 32) == 0) devChan.putWithSignal(offset, size);
    }
    // wait for the data from GPU (rank-i) % nRanksPerNode to arrive
    if ((remoteRank % nRanksPerNode) == ((rank - i + nRanksPerNode) % nRanksPerNode)) {
      if ((threadIdx.x % 32) == 0) devChan.wait();
    }
    asm volatile("bar.sync %0, %1;" ::"r"(11), "r"((nRanksPerNode - 1) * 32) : "memory");
  }
}

__device__ mscclpp::DeviceSyncer deviceSyncer;

// This kernel is the most performant when the number of blocks is a multiple of (nRanksPerNode - 1).
__device__ void localAllGatherSm(int rank, int nRanksPerNode, uint64_t offset, uint64_t size) {
  if (nRanksPerNode == 1) return;

  const size_t nBlocks = gridDim.x;
  const size_t nPeer = nRanksPerNode - 1;
  const size_t peerIdx = blockIdx.x % nPeer;
  const size_t nBlockForThisPeer = nBlocks / nPeer + (nBlocks % nPeer > peerIdx ? 1 : 0);
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

  constSmChans[peerIdx].put(offset + offsetForThisBlock, sizeForThisBlock, threadIdx.x, blockDim.x);
}

__device__ void allgather1(mscclpp::SimpleProxyChannel devChan, int rank, int worldSize, int nRanksPerNode,
                           int remoteRank, size_t nelemsPerGPU) {
  localAllGather(devChan, rank, worldSize, nRanksPerNode, remoteRank, rank * nelemsPerGPU * sizeof(int),
                 nelemsPerGPU * sizeof(int));
}

__device__ void allgather2(mscclpp::SimpleProxyChannel devChan, int rank, int worldSize, int nRanksPerNode,
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
  if (remoteRank / nRanksPerNode == rank / nRanksPerNode) {
    localAllGather(devChan, rank, worldSize, nRanksPerNode, remoteRank, rank * nelemsPerGPU * sizeof(int),
                   nelemsPerGPU * sizeof(int), false);
  }
  // cross-node exchange
  if (remoteRank % nRanksPerNode == rank % nRanksPerNode) {
    // opposite side
    if ((threadIdx.x % 32) == 0)
      devChan.putWithSignal(rank * nelemsPerGPU * sizeof(int),
                            (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize * sizeof(int));
    if ((threadIdx.x % 32) == 0) devChan.wait();
  }

  // sync here to make sure IB flush dose not block the CUDA IPC traffic
  __syncthreads();
  // need to flush ib channel here to avoid cq overflow. since we won't change send suffer after send, we don't need
  // to flush for IPC channel.
  if (remoteRank % nRanksPerNode == rank % nRanksPerNode) {
    if ((threadIdx.x % 32) == 0) devChan.flush();
  }
  __syncthreads();

  // Step 2
  // local allgather
  int otherNghr = (rank + nRanksPerNode) % worldSize;
  if (remoteRank / nRanksPerNode == rank / nRanksPerNode) {
    localAllGather(devChan, rank, worldSize, nRanksPerNode, remoteRank, otherNghr * nelemsPerGPU * sizeof(int),
                   (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize * sizeof(int), false);
  }

  // cross-node exchange
  if (remoteRank % nRanksPerNode == rank % nRanksPerNode) {
    // opposite side
    if ((threadIdx.x % 32) == 0)
      devChan.putWithSignal((rank * nelemsPerGPU + (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize) * sizeof(int),
                            nelemsPerGPU / pipelineSize * sizeof(int));
    if ((threadIdx.x % 32) == 0) devChan.wait();
  }

  __syncthreads();
  if (remoteRank % nRanksPerNode == rank % nRanksPerNode) {
    if ((threadIdx.x % 32) == 0) devChan.flush();
  }
  __syncthreads();

  // Step 3
  // local allgather
  if (remoteRank / nRanksPerNode == rank / nRanksPerNode) {
    localAllGather(devChan, rank, worldSize, nRanksPerNode, remoteRank,
                   (otherNghr * nelemsPerGPU + (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize) * sizeof(int),
                   nelemsPerGPU / pipelineSize * sizeof(int));
  }
}

__device__ void allgather3(mscclpp::ProxyChannel devChan, int rank, int worldSize) {
  int tid = threadIdx.x;
  __syncthreads();
  if (tid == 0) {
    mscclpp::ProxyTrigger trigger;
    trigger.fst = MAGIC;
    // offload all the work to the proxy
    uint64_t currentFifoHead = devChan.fifo_.push(trigger);
    // wait for the work to be done in cpu side
    devChan.fifo_.sync(currentFifoHead);
  }
  if (tid % 32 == 0) {
    devChan.wait();
  }
}

// Run with a single thread only.
__device__ void allgather4(int rank, int worldSize, int nRanksPerNode, size_t nelemsPerGPU) {
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
  mscclpp::SimpleProxyChannel& devChan = constDevChans[peer];

  if (peerNodeId == rank / nRanksPerNode) {
    localAllGatherSm(rank, nRanksPerNode, rank * nelemsPerGPU * sizeof(int), nelemsPerGPU * sizeof(int));
    return;
  }

  // Step 1
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    devChan.putWithSignal(rank * nelemsPerGPU * sizeof(int),
                          (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize * sizeof(int));
  }
  localAllGatherSm(rank, nRanksPerNode, rank * nelemsPerGPU * sizeof(int), nelemsPerGPU * sizeof(int));
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    devChan.wait();
    devChan.flush();
  }
  deviceSyncer.sync(gridDim.x);
  // Step 2
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    devChan.putWithSignal((rank * nelemsPerGPU + (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize) * sizeof(int),
                          nelemsPerGPU / pipelineSize * sizeof(int));
  }
  localAllGatherSm(rank, nRanksPerNode, peerRank * nelemsPerGPU * sizeof(int),
                   (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize * sizeof(int));
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    devChan.wait();
    devChan.flush();
  }
  deviceSyncer.sync(gridDim.x);
  // Step 3
  localAllGatherSm(rank, nRanksPerNode,
                   (peerRank * nelemsPerGPU + (nelemsPerGPU * (pipelineSize - 1)) / pipelineSize) * sizeof(int),
                   nelemsPerGPU / pipelineSize * sizeof(int));
}

__global__ void kernel(int rank, int worldSize, int nRanksPerNode, size_t nelemsPerGPU, int kernel) {
  // find the mapping between remoteRank and devConns
  int warpId = threadIdx.x / 32;
  int remoteRank = (warpId < rank) ? warpId : warpId + 1;
  // Each warp is responsible for one of the remote ranks
  mscclpp::SimpleProxyChannel devChan = constDevChans[warpId];

  if (kernel == 0) {
    allgather0(devChan, rank, worldSize, remoteRank, nelemsPerGPU);
  } else if (kernel == 1) {
    allgather1(devChan, rank, worldSize, nRanksPerNode, remoteRank, nelemsPerGPU);
  } else if (kernel == 2) {
    allgather2(devChan, rank, worldSize, nRanksPerNode, remoteRank, nelemsPerGPU);
  } else if (kernel == 3) {
    mscclpp::ProxyChannel devChan = constRawDevChan[warpId];
    allgather3(devChan, rank, worldSize);
  } else if (kernel == 4) {
    allgather4(rank, worldSize, nRanksPerNode, nelemsPerGPU);
  }
}

class AllGatherChannelService : public mscclpp::BaseProxyService {
 public:
  AllGatherChannelService(mscclpp::Communicator& communicator, int worldSize, int rank, int cudaDevice);
  void startProxy() override { proxy_.start(); }
  void stopProxy() override { proxy_.stop(); }
  void setSendBytes(size_t sendBytes) { this->sendBytes_ = sendBytes; }
  void addRemoteMemory(mscclpp::RegisteredMemory memory) { remoteMemories_.push_back(memory); }
  void setLocalMemory(mscclpp::RegisteredMemory memory) { localMemory_ = memory; }
  mscclpp::SemaphoreId addSemaphore(std::shared_ptr<mscclpp::Connection> connection) {
    semaphores_.push_back(std::make_shared<mscclpp::Host2DeviceSemaphore>(communicator_, connection));
    return semaphores_.size() - 1;
  }
  std::vector<mscclpp::ProxyChannel> deviceChannels() {
    std::vector<mscclpp::ProxyChannel> result;
    for (auto& semaphore : semaphores_) {
      result.push_back(mscclpp::ProxyChannel(0, semaphore->deviceHandle(), proxy_.fifo().deviceFifo()));
    }
    return result;
  }

 private:
  int worldSize_;
  int rank_;
  int cudaDevice_;
  size_t sendBytes_;

  mscclpp::Proxy proxy_;
  mscclpp::Communicator& communicator_;
  std::vector<std::shared_ptr<mscclpp::Host2DeviceSemaphore>> semaphores_;
  std::vector<mscclpp::RegisteredMemory> remoteMemories_;
  mscclpp::RegisteredMemory localMemory_;

  mscclpp::ProxyHandlerResult handleTrigger(mscclpp::ProxyTrigger triggerRaw);
};

AllGatherChannelService::AllGatherChannelService(mscclpp::Communicator& communicator, int worldSize, int rank,
                                                 int cudaDevice)
    : communicator_(communicator),
      worldSize_(worldSize),
      sendBytes_(0),
      rank_(rank),
      cudaDevice_(cudaDevice),
      proxy_([&](mscclpp::ProxyTrigger triggerRaw) { return handleTrigger(triggerRaw); },
             [&]() {
               int deviceNumaNode = getDeviceNumaNode(cudaDevice_);
               numaBind(deviceNumaNode);
             }) {}

mscclpp::ProxyHandlerResult AllGatherChannelService::handleTrigger(mscclpp::ProxyTrigger triggerRaw) {
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
    semaphores_[index]->connection()->write(remoteMemories_[index], offset, localMemory_, offset, sendBytes_);
    semaphores_[index]->signal();
  }
  bool flushIpc = false;
  for (auto& semaphore : semaphores_) {
    auto conn = semaphore->connection();
    if (conn->transport() == mscclpp::Transport::CudaIpc && !flushIpc) {
      // since all the cudaIpc channels are using the same cuda stream, we only need to flush one of them
      conn->flush();
      flushIpc = true;
    }
    if (mscclpp::AllIBTransports.has(conn->transport())) {
      conn->flush();
    }
  }
  return mscclpp::ProxyHandlerResult::FlushFifoTailAndContinue;
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
    nBlocks = 56;
    nThreads = 1024;
  } else {
    nBlocks = 1;
    nThreads = 32 * (worldSize - 1);
  }
  kernel<<<nBlocks, nThreads, 0, stream>>>(rank, worldSize, nRanksPerNode, paramCount_, kernelNum);
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
  if (isUsingHostOffload(kernelNum_)) {
    auto service = std::dynamic_pointer_cast<AllGatherChannelService>(chanService_);
    service->setSendBytes(sendCount_ * typeSize_);
  }
  mscclpp::DeviceSyncer syncer = {};
  CUDATHROW(cudaMemcpyToSymbol(deviceSyncer, &syncer, sizeof(mscclpp::DeviceSyncer)));
}

std::vector<KernelRestriction> AllGatherTestColl::getKernelRestrictions() {
  return {// {kernelNum, kernelName, compatibleWithMultiNodes, countDivisorForMultiNodes}
          {0, "allgather0", true, 1},
          {1, "allgather1", false, 1},
          {2, "allgather2", true, 3},
          {3, "allgather3", true, 1}};
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
  std::shared_ptr<mscclpp::BaseProxyService> createChannelService() override;

 private:
  void* getExpectedBuff() override;

  std::shared_ptr<int> sendBuff_;
  std::shared_ptr<int[]> expectedBuff_;
};

AllGatherTestEngine::AllGatherTestEngine(const TestArgs& args) : BaseTestEngine(args, "allgather") {}

void AllGatherTestEngine::allocateBuffer() {
  sendBuff_ = mscclpp::allocSharedCuda<int>(args_.maxBytes / sizeof(int));
  expectedBuff_ = std::shared_ptr<int[]>(new int[args_.maxBytes / sizeof(int)]);
}

void AllGatherTestEngine::setupConnections() {
  std::vector<mscclpp::SimpleProxyChannel> devChannels;
  if (!isUsingHostOffload(args_.kernelNum)) {
    setupMeshConnections(devChannels, sendBuff_.get(), args_.maxBytes);
    assert(devChannels.size() < sizeof(constDevChans) / sizeof(mscclpp::SimpleProxyChannel));
    CUDATHROW(cudaMemcpyToSymbol(constDevChans, devChannels.data(),
                                 sizeof(mscclpp::SimpleProxyChannel) * devChannels.size()));

    std::vector<mscclpp::SmChannel> smChannels;
    setupMeshConnections(smChannels, sendBuff_.get(), args_.maxBytes);
    assert(smChannels.size() < sizeof(constSmChans) / sizeof(mscclpp::SmChannel));
    CUDATHROW(cudaMemcpyToSymbol(constSmChans, smChannels.data(), sizeof(mscclpp::SmChannel) * smChannels.size()));
  } else {
    auto service = std::dynamic_pointer_cast<AllGatherChannelService>(chanService_);
    setupMeshConnections(devChannels, sendBuff_.get(), args_.maxBytes, nullptr, 0,
                         [&](std::vector<std::shared_ptr<mscclpp::Connection>> conns,
                             std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>>& remoteMemories,
                             const mscclpp::RegisteredMemory& localMemory) {
                           std::vector<mscclpp::SemaphoreId> semaphoreIds;
                           for (int i = 0; i < conns.size(); ++i) {
                             service->addSemaphore(conns[i]);
                             service->addRemoteMemory(remoteMemories[i].get());
                           }
                           service->setLocalMemory(localMemory);
                           comm_->setup();
                         });
    auto devChannels = service->deviceChannels();
    assert(devChannels.size() < sizeof(constRawDevChan) / sizeof(mscclpp::ProxyChannel));
    CUDATHROW(
        cudaMemcpyToSymbol(constRawDevChan, devChannels.data(), sizeof(mscclpp::ProxyChannel) * devChannels.size()));
  }
}

std::shared_ptr<mscclpp::BaseProxyService> AllGatherTestEngine::createChannelService() {
  if (isUsingHostOffload(args_.kernelNum)) {
    return std::make_shared<AllGatherChannelService>(*comm_, args_.totalRanks, args_.rank, args_.gpuNum);
  } else {
    return std::make_shared<mscclpp::ProxyService>(*comm_);
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
