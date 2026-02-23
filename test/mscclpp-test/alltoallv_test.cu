// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// AllToAllV test - tests variable-length alltoall operations
// This test validates the alltoallv kernel that handles variable element counts per rank.
// Uses the kernel implementations from src/ext/collectives/include/alltoallv/

#include <cstdint>
#include <cstring>
#include <numeric>
#include <mscclpp/concurrency_device.hpp>

#include "common.hpp"

// Include the alltoallv kernel implementations from src/ext/collectives
#include "alltoallv/alltoallv_kernel.hpp"

template <class T>
using DeviceHandle = mscclpp::DeviceHandle<T>;
__device__ mscclpp::DeviceSyncer deviceSyncerV;

static void* localRecvBuffV;
static void* localSendBuffV;

// Device arrays for variable counts and displacements
static size_t* d_sendCounts;
static size_t* d_sendDispls;
static size_t* d_recvCounts;
static size_t* d_recvDispls;
static size_t* d_remoteRecvDispls;  // peer's recvDispls[rank] for each peer

// Device array for memory channels (used by library kernels)
static DeviceHandle<mscclpp::MemoryChannel>* d_memoryChannels;

class AllToAllVTestColl : public BaseTestColl {
 public:
  AllToAllVTestColl() = default;
  ~AllToAllVTestColl() override = default;

  void runColl(const TestArgs& args, cudaStream_t stream) override;
  void initData(const TestArgs& args, std::vector<void*> sendBuff, void* expectedBuff) override;
  void getBw(const double deltaSec, double& algBw /*OUT*/, double& busBw /*OUT*/) override;
  void setupCollTest(size_t size) override;
  std::vector<KernelRestriction> getKernelRestrictions() override;

 private:
  // Host-side counts and displacements
  std::vector<size_t> sendCounts_;
  std::vector<size_t> sendDispls_;
  std::vector<size_t> recvCounts_;
  std::vector<size_t> recvDispls_;
  size_t totalSendBytes_;
  size_t totalRecvBytes_;
};

void AllToAllVTestColl::runColl(const TestArgs& args, cudaStream_t stream) {
  const int worldSize = args.totalRanks;
  const int rank = args.rank;
  const int kernelNum = args.kernelNum;

  // Use maximum threads (1024) for best bandwidth utilization
  const int nThreads = 1024;

  if (kernelNum == 0) {
    // Use high-throughput kernel with all threads participating in each transfer
    mscclpp::collective::alltoallvKernel<<<1, nThreads, 0, stream>>>(
        d_memoryChannels,
        rank, worldSize,
        localSendBuffV, localRecvBuffV,
        d_sendCounts, d_sendDispls,
        d_recvCounts, d_recvDispls,
        d_remoteRecvDispls);
  } else if (kernelNum == 1) {
    // Use ring-based kernel for larger world sizes
    mscclpp::collective::alltoallvRingKernel<<<1, nThreads, 0, stream>>>(
        d_memoryChannels,
        rank, worldSize,
        localSendBuffV, localRecvBuffV,
        d_sendCounts, d_sendDispls,
        d_recvCounts, d_recvDispls,
        d_remoteRecvDispls);
  } else if (kernelNum == 2) {
    // Use pipelined kernel for imbalanced workloads (MoE)
    mscclpp::collective::alltoallvPipelinedKernel<<<1, nThreads, 0, stream>>>(
        d_memoryChannels,
        rank, worldSize,
        localSendBuffV, localRecvBuffV,
        d_sendCounts, d_sendDispls,
        d_recvCounts, d_recvDispls,
        d_remoteRecvDispls);
  }
}

void AllToAllVTestColl::initData(const TestArgs& args, std::vector<void*> sendBuff, void* expectedBuff) {
  if (sendBuff.size() != 1) throw std::runtime_error("unexpected error");
  const int rank = args.rank;
  const int worldSize = args.totalRanks;

  // Create send data: each segment has values identifying source and destination
  std::vector<int> sendData(totalSendBytes_ / sizeof(int), 0);
  for (int peer = 0; peer < worldSize; peer++) {
    size_t offset = sendDispls_[peer] / sizeof(int);
    size_t count = sendCounts_[peer] / sizeof(int);
    for (size_t i = 0; i < count; i++) {
      // Encode: rank * 10000 + peer * 100 + position
      sendData[offset + i] = rank * 10000 + peer * 100 + i;
    }
  }
  CUDATHROW(cudaMemcpy(sendBuff[0], sendData.data(), totalSendBytes_, cudaMemcpyHostToDevice));

  // Create expected data: we receive from each peer
  std::vector<int> expectedData(totalRecvBytes_ / sizeof(int), 0);
  for (int peer = 0; peer < worldSize; peer++) {
    size_t offset = recvDispls_[peer] / sizeof(int);
    size_t count = recvCounts_[peer] / sizeof(int);
    for (size_t i = 0; i < count; i++) {
      // We receive data sent by peer to us
      expectedData[offset + i] = peer * 10000 + rank * 100 + i;
    }
  }
  std::memcpy(expectedBuff, expectedData.data(), totalRecvBytes_);

  // Copy counts and displacements to device
  CUDATHROW(cudaMemcpy(d_sendCounts, sendCounts_.data(), worldSize * sizeof(size_t), cudaMemcpyHostToDevice));
  CUDATHROW(cudaMemcpy(d_sendDispls, sendDispls_.data(), worldSize * sizeof(size_t), cudaMemcpyHostToDevice));
  CUDATHROW(cudaMemcpy(d_recvCounts, recvCounts_.data(), worldSize * sizeof(size_t), cudaMemcpyHostToDevice));
  CUDATHROW(cudaMemcpy(d_recvDispls, recvDispls_.data(), worldSize * sizeof(size_t), cudaMemcpyHostToDevice));
  // remoteRecvDispls[peer] = peer's recvDispls[rank] = where our data goes in peer's output.
  // For equal splits, all ranks have the same layout, so peer's recvDispls[rank] = our recvDispls[rank].
  std::vector<size_t> remoteRecvDispls(worldSize);
  for (int peer = 0; peer < worldSize; peer++) {
    remoteRecvDispls[peer] = recvDispls_[rank];
  }
  CUDATHROW(cudaMemcpy(d_remoteRecvDispls, remoteRecvDispls.data(), worldSize * sizeof(size_t), cudaMemcpyHostToDevice));
}

void AllToAllVTestColl::getBw(const double deltaSec, double& algBw, double& busBw) {
  double baseBw = (double)(totalRecvBytes_) / 1.0E9 / deltaSec;
  algBw = baseBw;
  double factor = ((double)(worldSize_ - 1)) / ((double)worldSize_);
  busBw = baseBw * factor;
}

void AllToAllVTestColl::setupCollTest(size_t size) {
  // For alltoallv, we use variable sizes per peer
  // For testing: rank i sends (rank + 1) * baseCount elements to each peer
  // Each peer j sends (j + 1) * baseCount elements to us
  
  size_t baseBytes = size / (worldSize_ * worldSize_);  // Base unit for variable sizing
  if (baseBytes < sizeof(int)) baseBytes = sizeof(int);
  baseBytes = (baseBytes / sizeof(int)) * sizeof(int);  // Align to int size

  sendCounts_.resize(worldSize_);
  sendDispls_.resize(worldSize_);
  recvCounts_.resize(worldSize_);
  recvDispls_.resize(worldSize_);

  // Each rank sends the same amount to each peer (for simplicity in this test)
  // In a real MOE scenario, these would be variable
  totalSendBytes_ = 0;
  totalRecvBytes_ = 0;

  for (int peer = 0; peer < worldSize_; peer++) {
    sendCounts_[peer] = baseBytes;
    sendDispls_[peer] = totalSendBytes_;
    totalSendBytes_ += sendCounts_[peer];

    recvCounts_[peer] = baseBytes;
    recvDispls_[peer] = totalRecvBytes_;
    totalRecvBytes_ += recvCounts_[peer];
  }

  sendCount_ = totalSendBytes_ / typeSize_;
  recvCount_ = totalRecvBytes_ / typeSize_;
  paramCount_ = sendCount_;
  expectedCount_ = recvCount_;

  mscclpp::DeviceSyncer syncer = {};
  CUDATHROW(cudaMemcpyToSymbol(deviceSyncerV, &syncer, sizeof(mscclpp::DeviceSyncer)));
}

std::vector<KernelRestriction> AllToAllVTestColl::getKernelRestrictions() {
  return {
      {0, "alltoallvKernel", true, 1, 4 * worldSize_},
      {1, "alltoallvRingKernel", true, 1, 4 * worldSize_},
      {2, "alltoallvPipelinedKernel", true, 1, 4 * worldSize_}
  };
}

class AllToAllVTestEngine : public BaseTestEngine {
 public:
  AllToAllVTestEngine(const TestArgs& args);
  ~AllToAllVTestEngine() override = default;

  void allocateBuffer() override;
  void setupConnections() override;

  std::vector<void*> getSendBuff() override;
  void* getRecvBuff() override;
  void* getScratchBuff() override;

 private:
  void* getExpectedBuff() override;
  bool isInPlace() const;

  std::shared_ptr<int> sendBuff_;
  std::shared_ptr<int> recvBuff_;
  std::shared_ptr<int[]> expectedBuff_;
};

bool AllToAllVTestEngine::isInPlace() const { return false; }

AllToAllVTestEngine::AllToAllVTestEngine(const TestArgs& args) : BaseTestEngine(args, "alltoallv") { inPlace_ = false; }

void AllToAllVTestEngine::allocateBuffer() {
  sendBuff_ = mscclpp::GpuBuffer<int>(args_.maxBytes / sizeof(int)).memory();
  recvBuff_ = mscclpp::GpuBuffer<int>(args_.maxBytes / sizeof(int)).memory();
  expectedBuff_ = std::shared_ptr<int[]>(new int[args_.maxBytes / sizeof(int)]);

  localSendBuffV = sendBuff_.get();
  localRecvBuffV = recvBuff_.get();

  // Allocate device arrays for counts and displacements
  CUDATHROW(cudaMalloc(&d_sendCounts, args_.totalRanks * sizeof(size_t)));
  CUDATHROW(cudaMalloc(&d_sendDispls, args_.totalRanks * sizeof(size_t)));
  CUDATHROW(cudaMalloc(&d_recvCounts, args_.totalRanks * sizeof(size_t)));
  CUDATHROW(cudaMalloc(&d_recvDispls, args_.totalRanks * sizeof(size_t)));
  CUDATHROW(cudaMalloc(&d_remoteRecvDispls, args_.totalRanks * sizeof(size_t)));

  // Allocate device array for memory channels
  CUDATHROW(cudaMalloc(&d_memoryChannels, args_.totalRanks * sizeof(DeviceHandle<mscclpp::MemoryChannel>)));
}

void AllToAllVTestEngine::setupConnections() {
  std::vector<mscclpp::MemoryChannel> memoryChannels;
  // Setup MemoryChannels: we write to peer's recv buffer from our send buffer
  setupMeshConnections(memoryChannels, sendBuff_.get(), args_.maxBytes, recvBuff_.get(), args_.maxBytes,
                       ChannelSemantic::PUT, 1);

  // Convert to device handles and copy to device memory
  std::vector<DeviceHandle<mscclpp::MemoryChannel>> memoryChannelHandles;
  for (auto& channel : memoryChannels) {
    memoryChannelHandles.push_back(mscclpp::deviceHandle(channel));
  }
  CUDATHROW(cudaMemcpy(d_memoryChannels, memoryChannelHandles.data(),
                       sizeof(DeviceHandle<mscclpp::MemoryChannel>) * memoryChannelHandles.size(),
                       cudaMemcpyHostToDevice));
}

std::vector<void*> AllToAllVTestEngine::getSendBuff() { return {sendBuff_.get()}; }
void* AllToAllVTestEngine::getExpectedBuff() { return expectedBuff_.get(); }
void* AllToAllVTestEngine::getRecvBuff() {
  if (this->isInPlace())
    return sendBuff_.get();
  else
    return recvBuff_.get();
}
void* AllToAllVTestEngine::getScratchBuff() { return nullptr; }

std::shared_ptr<BaseTestEngine> getTestEngine(const TestArgs& args) {
  return std::make_shared<AllToAllVTestEngine>(args);
}
std::shared_ptr<BaseTestColl> getTestColl() { return std::make_shared<AllToAllVTestColl>(); }
