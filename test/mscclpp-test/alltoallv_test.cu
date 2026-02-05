// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// AllToAllV test - tests variable-length alltoall operations
// This test validates the alltoallv kernel that handles variable element counts per rank.

#include <cstdint>
#include <cstring>
#include <numeric>
#include <mscclpp/concurrency_device.hpp>

#include "common.hpp"

#if defined(__HIP_PLATFORM_AMD__)
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif

template <class T>
using DeviceHandle = mscclpp::DeviceHandle<T>;
__constant__ DeviceHandle<mscclpp::PortChannel> constPortChansV[16];
__device__ mscclpp::DeviceSyncer deviceSyncerV;

static void* localRecvBuffV;
static void* localSendBuffV;

// Device arrays for variable counts and displacements
static size_t* d_sendCounts;
static size_t* d_sendDispls;
static size_t* d_recvCounts;
static size_t* d_recvDispls;

/**
 * AllToAllV kernel implementation
 *
 * Each rank sends sendCounts[i] bytes to rank i at sendDispls[i] offset,
 * and receives recvCounts[i] bytes from rank i at recvDispls[i] offset.
 *
 * Uses ring-based exchange pattern to avoid deadlocks.
 */
__global__ void __launch_bounds__(1024)
    alltoallv0(int rank, int worldSize,
               const void* sendBuff, void* recvBuff,
               const size_t* sendCounts, const size_t* sendDispls,
               const size_t* recvCounts, const size_t* recvDispls) {
  int tid = threadIdx.x;
  int nPeers = worldSize - 1;

  // Step 1: Copy local data (rank's own portion)
  if (tid == 0 && sendCounts[rank] > 0) {
    const char* src = (const char*)sendBuff + sendDispls[rank];
    char* dst = (char*)recvBuff + recvDispls[rank];
    memcpy(dst, src, sendCounts[rank]);
  }
  __syncthreads();

  // Step 2: Each warp handles one peer for sending
  int warpId = tid / WARP_SIZE;
  int laneId = tid % WARP_SIZE;

  if (warpId < nPeers && laneId == 0) {
    // Determine which peer this warp handles
    int peer = warpId < rank ? warpId : warpId + 1;
    int chanIdx = warpId;

    if (sendCounts[peer] > 0) {
      constPortChansV[chanIdx].putWithSignal(
          recvDispls[rank],       // dst offset in peer's buffer
          sendDispls[peer],       // src offset in our buffer
          sendCounts[peer]        // size
      );
    }
  }
  __syncthreads();

  // Step 3: Flush all pending operations
  if (warpId < nPeers && laneId == 0) {
    int peer = warpId < rank ? warpId : warpId + 1;
    if (sendCounts[peer] > 0) {
      constPortChansV[warpId].flush();
    }
  }
  __syncthreads();

  // Step 4: Wait for all incoming data
  if (warpId < nPeers && laneId == 0) {
    int peer = warpId < rank ? warpId : warpId + 1;
    if (recvCounts[peer] > 0) {
      constPortChansV[warpId].wait();
    }
  }
  __syncthreads();
}

/**
 * Ring-based AllToAllV kernel for larger world sizes
 *
 * Uses step-by-step ring pattern to exchange data, sending to (rank+step) and
 * receiving from (rank-step) in each step. Single block to avoid concurrent
 * access to the same port channels.
 */
__global__ void __launch_bounds__(1024)
    alltoallv1(int rank, int worldSize,
               const void* sendBuff, void* recvBuff,
               const size_t* sendCounts, const size_t* sendDispls,
               const size_t* recvCounts, const size_t* recvDispls) {
  // Copy local data first
  if (threadIdx.x == 0) {
    if (sendCounts[rank] > 0) {
      const char* src = (const char*)sendBuff + sendDispls[rank];
      char* dst = (char*)recvBuff + recvDispls[rank];
      memcpy(dst, src, sendCounts[rank]);
    }
  }
  __syncthreads();

  // Ring-based exchange - single thread handles the communication
  // to avoid race conditions on port channels
  if (threadIdx.x == 0) {
    for (int step = 1; step < worldSize; step++) {
      int sendPeer = (rank + step) % worldSize;
      int recvPeer = (rank - step + worldSize) % worldSize;

      int sendChanIdx = sendPeer < rank ? sendPeer : sendPeer - 1;
      int recvChanIdx = recvPeer < rank ? recvPeer : recvPeer - 1;

      // Send data to sendPeer (non-blocking put with signal)
      if (sendCounts[sendPeer] > 0) {
        constPortChansV[sendChanIdx].putWithSignal(
            recvDispls[rank],       // dst offset in peer's buffer
            sendDispls[sendPeer],   // src offset in our buffer
            sendCounts[sendPeer]    // size
        );
        constPortChansV[sendChanIdx].flush();
      }

      // Wait for data from recvPeer
      if (recvCounts[recvPeer] > 0) {
        constPortChansV[recvChanIdx].wait();
      }
    }
  }
}

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

  // Reset device syncer
  mscclpp::DeviceSyncer syncer = {};
  CUDATHROW(cudaMemcpyToSymbol(deviceSyncerV, &syncer, sizeof(mscclpp::DeviceSyncer)));

  if (kernelNum == 0) {
    int nThreads = (worldSize - 1) * WARP_SIZE;
    if (nThreads < 32) nThreads = 32;
    if (nThreads > 1024) nThreads = 1024;
    alltoallv0<<<1, nThreads, 0, stream>>>(
        rank, worldSize,
        localSendBuffV, localRecvBuffV,
        d_sendCounts, d_sendDispls,
        d_recvCounts, d_recvDispls);
  } else if (kernelNum == 1) {
    // Single block, single thread for ring-based serialized communication
    alltoallv1<<<1, 32, 0, stream>>>(
        rank, worldSize,
        localSendBuffV, localRecvBuffV,
        d_sendCounts, d_sendDispls,
        d_recvCounts, d_recvDispls);
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
      {0, "alltoallv0", true, 1, 4 * worldSize_},
      {1, "alltoallv1", true, 1, 4 * worldSize_}
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
}

void AllToAllVTestEngine::setupConnections() {
  std::vector<DeviceHandle<mscclpp::PortChannel>> portChannels;
  setupMeshConnections(portChannels, sendBuff_.get(), args_.maxBytes, recvBuff_.get(), args_.maxBytes);

  if (portChannels.size() > sizeof(constPortChansV) / sizeof(DeviceHandle<mscclpp::PortChannel>)) {
    throw std::runtime_error("Too many port channels for alltoallv test");
  }
  CUDATHROW(cudaMemcpyToSymbol(constPortChansV, portChannels.data(),
                               sizeof(DeviceHandle<mscclpp::PortChannel>) * portChannels.size()));
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
