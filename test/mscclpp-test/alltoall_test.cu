// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cstdint>
#include <cstring>
#include <mscclpp/concurrency_device.hpp>

#include "common.hpp"

#if defined(__HIP_PLATFORM_AMD__)
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif

template <class T>
using DeviceHandle = mscclpp::DeviceHandle<T>;
__constant__ DeviceHandle<mscclpp::PortChannel> constPortChans[16];
__constant__ DeviceHandle<mscclpp::MemoryChannel> constMemChans[512];
__device__ mscclpp::DeviceSyncer deviceSyncer;

static void* localRecvBuff;
static void* localSendBuff;
static void* localScratchBuff;

__device__ void localAlltoall(int rank, int nRanksPerNode, size_t nElements) {
  int remoteRank = ((int)blockIdx.x < rank) ? blockIdx.x : blockIdx.x + 1;
  for (int i = 1; i < nRanksPerNode; i++) {
    DeviceHandle<mscclpp::PortChannel> portChan = constPortChans[blockIdx.x];
    if (threadIdx.x == 0 && remoteRank % nRanksPerNode == (rank + i) % nRanksPerNode) {
      portChan.putWithSignalAndFlush(rank * nElements * sizeof(int), remoteRank * nElements * sizeof(int),
                                     nElements * sizeof(int));
    }
    // wait for the data from GPU (rank-i) % nranksPerNode to arrive
    if (threadIdx.x == 0 && remoteRank % nRanksPerNode == (rank - i + nRanksPerNode) % nRanksPerNode) {
      portChan.wait();
    }
    deviceSyncer.sync(nRanksPerNode - 1);
  }
}

__global__ void __launch_bounds__(1024) alltoall0(int rank, size_t nElements) {
  int remoteRank = ((int)blockIdx.x < rank) ? blockIdx.x : blockIdx.x + 1;
  DeviceHandle<mscclpp::PortChannel> portChan = constPortChans[blockIdx.x];
  if (threadIdx.x == 0) {
    portChan.putWithSignal(rank * nElements * sizeof(int), remoteRank * nElements * sizeof(int),
                           nElements * sizeof(int));
  }

  deviceSyncer.sync(gridDim.x);
  if (threadIdx.x == 0) {
    portChan.flush();
    portChan.wait();
  }
}

__global__ void __launch_bounds__(1024) alltoall1(int rank, int nRanksPerNode, size_t nElements) {
  localAlltoall(rank, nRanksPerNode, nElements);
}

__global__ void __launch_bounds__(1024) alltoall2(int rank, int nRanksPerNode, size_t nElements, void* inputBuffer,
                                                  void* scratchBuffer, void* resultBuffer) {
#if defined(__CUDA_ARCH__)
  constexpr int nWarpForPut = 16;
  constexpr int nWarpForCopy = 16;
  constexpr int putStartWid = 0;
  constexpr int putEndWid = putStartWid + nWarpForPut;
  constexpr int copyStartWid = putEndWid;
  constexpr int copyEndWid = copyStartWid + nWarpForCopy;
  constexpr size_t unit = 1 << 18;  // 256K

  size_t totalCount = nElements * sizeof(int);
  size_t nBytesPerBlock = (totalCount + (gridDim.x - 1)) / gridDim.x;
  nBytesPerBlock = nBytesPerBlock / 16 * 16;  // alignment
  size_t nBytesForLastBlock = totalCount - (nBytesPerBlock * (gridDim.x - 1));
  size_t totalBytesForCurrentBlock = nBytesPerBlock;
  if (blockIdx.x == gridDim.x - 1) {
    totalBytesForCurrentBlock = nBytesForLastBlock;
  }
  size_t nIters = (totalBytesForCurrentBlock + unit - 1) / unit;
  int wid = threadIdx.x / WARP_SIZE;
  int lid = threadIdx.x % WARP_SIZE;
  DeviceHandle<mscclpp::MemoryChannel>* memoryChannels = constMemChans + blockIdx.x * (nRanksPerNode - 1);
  if (wid == 0 && lid < nRanksPerNode - 1) {
    memoryChannels[lid].relaxedSignal();
    memoryChannels[lid].relaxedWait();
  }
  __syncthreads();
  size_t lastIterSize = totalBytesForCurrentBlock - (nIters - 1) * unit;
  mscclpp::copy((char*)resultBuffer + rank * totalCount, (char*)inputBuffer + rank * totalCount, totalCount,
                threadIdx.x + blockIdx.x * blockDim.x, gridDim.x * blockDim.x);
  for (int step = 0; step < nRanksPerNode - 1; step++) {
    int peer = (rank + step + 1) % nRanksPerNode;
    int peerToCopy = (rank - (step + 1) + nRanksPerNode) % nRanksPerNode;
    int peerId = peer < rank ? peer : peer - 1;
    int peerIdForCopy = peerToCopy < rank ? peerToCopy : peerToCopy - 1;
    size_t startOffset = peer * totalCount + blockIdx.x * nBytesPerBlock;
    size_t startCopyOffset = peerToCopy * totalCount + blockIdx.x * nBytesPerBlock;
    size_t dstOffset = rank * totalCount + blockIdx.x * nBytesPerBlock;
    size_t size = unit;
    if (wid >= putStartWid && wid < putEndWid) {
      int tidInPut = wid * WARP_SIZE + lid - putStartWid * WARP_SIZE;
      for (size_t i = 0; i < nIters; i++) {
        if (i == nIters - 1) {
          size = lastIterSize;
        }
        mscclpp::copy((char*)memoryChannels[peerId].dst_ + dstOffset + i * unit,
                      (char*)inputBuffer + startOffset + i * unit, size, wid * WARP_SIZE + lid,
                      nWarpForPut * WARP_SIZE);
        asm volatile("bar.sync %0, %1;" ::"r"(15), "r"(nWarpForPut * WARP_SIZE) : "memory");
        if (tidInPut == 0) {
          memoryChannels[peerId].signal();
        }
      }
    } else if (wid >= copyStartWid && wid < copyEndWid) {
      int tidInCopy = wid * WARP_SIZE + lid - copyStartWid * WARP_SIZE;
      for (size_t i = 0; i < nIters; i++) {
        if (tidInCopy == 0) {
          memoryChannels[peerIdForCopy].wait();
        }
        if (i == nIters - 1) {
          size = lastIterSize;
        }
        // barrier for n warp
        asm volatile("bar.sync %0, %1;" ::"r"(14), "r"(nWarpForCopy * WARP_SIZE) : "memory");
        mscclpp::copy((char*)resultBuffer + startCopyOffset + i * unit,
                      (char*)scratchBuffer + startCopyOffset + i * unit, size, tidInCopy, nWarpForCopy * WARP_SIZE);
      }
    }
  }
#endif
}

__global__ void __launch_bounds__(1024) alltoall3(int rank, int nRanksPerNode, size_t nElements, void* inputBuffer,
                                                  void* scratchBuffer, void* resultBuffer) {
#if defined(__CUDA_ARCH__)
  constexpr int nWarpForCopy = 16;
  constexpr int nWarpForGet = 16;
  constexpr int copyStartWid = 0;
  constexpr int copyEndWid = copyStartWid + nWarpForCopy;
  constexpr int getStartWid = copyEndWid;
  constexpr int getEndWid = getStartWid + nWarpForGet;
  constexpr size_t unit = 1 << 18;  // 256K

  size_t totalCount = nElements * sizeof(int);
  size_t nBytesPerBlock = (totalCount + (gridDim.x - 1)) / gridDim.x;
  nBytesPerBlock = nBytesPerBlock / 16 * 16;  // alignment
  size_t nBytesForLastBlock = totalCount - (nBytesPerBlock * (gridDim.x - 1));
  size_t totalBytesForCurrentBlock = nBytesPerBlock;
  if (blockIdx.x == gridDim.x - 1) {
    totalBytesForCurrentBlock = nBytesForLastBlock;
  }

  size_t nIters = (totalBytesForCurrentBlock + unit - 1) / unit;
  int wid = threadIdx.x / WARP_SIZE;
  int lid = threadIdx.x % WARP_SIZE;
  DeviceHandle<mscclpp::MemoryChannel>* memoryChannels = constMemChans + blockIdx.x * (nRanksPerNode - 1);
  size_t lastIterSize = totalBytesForCurrentBlock - (nIters - 1) * unit;
  mscclpp::copy((char*)resultBuffer + rank * totalCount, (char*)inputBuffer + rank * totalCount, totalCount,
                threadIdx.x + blockIdx.x * blockDim.x, gridDim.x * blockDim.x);
  for (int step = 0; step < nRanksPerNode - 1; step++) {
    int peer = (rank + step + 1) % nRanksPerNode;
    int peerToGet = (rank - (step + 1) + nRanksPerNode) % nRanksPerNode;
    int peerId = peer < rank ? peer : peer - 1;
    int peerIdForGet = peerToGet < rank ? peerToGet : peerToGet - 1;
    size_t startOffset = peer * totalCount + blockIdx.x * nBytesPerBlock;
    size_t startOffsetForGet = peerToGet * totalCount + blockIdx.x * nBytesPerBlock;
    size_t dstOffset = rank * totalCount + blockIdx.x * nBytesPerBlock;
    size_t size = unit;
    if (wid >= copyStartWid && wid < copyEndWid) {
      int tidInCopy = wid * WARP_SIZE + lid - copyStartWid * WARP_SIZE;
      for (size_t i = 0; i < nIters; i++) {
        if (i == nIters - 1) {
          size = lastIterSize;
        }
        mscclpp::copy((char*)scratchBuffer + startOffset + i * unit, (char*)inputBuffer + startOffset + i * unit, size,
                      tidInCopy, nWarpForCopy * WARP_SIZE);
        asm volatile("bar.sync %0, %1;" ::"r"(15), "r"(nWarpForCopy * WARP_SIZE) : "memory");
        if (tidInCopy == 0) {
          memoryChannels[peerId].signal();
        }
      }
    } else if (wid >= getStartWid && wid < getEndWid) {
      int tidInGet = wid * WARP_SIZE + lid - getStartWid * WARP_SIZE;
      for (size_t i = 0; i < nIters; i++) {
        if (tidInGet == 0) {
          memoryChannels[peerIdForGet].wait();
        }
        if (i == nIters - 1) {
          size = lastIterSize;
        }
        // barrier for n warp
        asm volatile("bar.sync %0, %1;" ::"r"(14), "r"(nWarpForGet * WARP_SIZE) : "memory");
        mscclpp::copy((char*)resultBuffer + startOffsetForGet + i * unit,
                      (char*)memoryChannels[peerIdForGet].dst_ + dstOffset + i * unit, size, tidInGet,
                      nWarpForGet * WARP_SIZE);
      }
    }
  }
  __syncthreads();
  if (wid == 0 && lid < nRanksPerNode - 1) {
    memoryChannels[lid].relaxedSignal();
    memoryChannels[lid].relaxedWait();
  }
#endif
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
  auto isNeedCopyKernel = [kernelNum]() { return kernelNum == 2 || kernelNum == 3; };
  if (!isNeedCopyKernel()) {
    CUDATHROW(cudaMemcpyAsync((int*)localRecvBuff + paramCount_ * rank, (int*)localSendBuff + paramCount_ * rank,
                              paramCount_ * sizeof(int), cudaMemcpyDeviceToDevice, stream));
  }
  if (kernelNum == 0) {
    alltoall0<<<worldSize - 1, 32, 0, stream>>>(rank, paramCount_);
  } else if (kernelNum == 1) {
    alltoall1<<<worldSize - 1, 32, 0, stream>>>(rank, nRanksPerNode, paramCount_);
  } else if (kernelNum == 2) {
    alltoall2<<<64, 1024, 0, stream>>>(rank, nRanksPerNode, paramCount_, localSendBuff, localScratchBuff,
                                       localRecvBuff);
  } else if (kernelNum == 3) {
    alltoall3<<<64, 1024, 0, stream>>>(rank, nRanksPerNode, paramCount_, localSendBuff, localScratchBuff,
                                       localRecvBuff);
  }
}

void AllToAllTestColl::initData(const TestArgs& args, std::vector<void*> sendBuff, void* expectedBuff) {
  if (sendBuff.size() != 1) throw std::runtime_error("unexpected error");
  const int rank = args.rank;
  std::vector<int> dataHost(recvCount_, 0);
  // For rank 0, the data is 0, 1, 2 ... recvCount_ - 1, for rank 1, the data is recvCount_, recvCount_ + 1, ...
  for (size_t i = 0; i < recvCount_; i++) {
    dataHost[i] = rank * recvCount_ + i;
  }
  CUDATHROW(cudaMemcpy(sendBuff[0], dataHost.data(), sendCount_ * typeSize_, cudaMemcpyHostToDevice));
  for (size_t i = 0; i < recvCount_ / paramCount_; i++) {
    for (size_t j = 0; j < paramCount_; j++) {
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
  size_t base = count;
  sendCount_ = base;
  recvCount_ = base;
  paramCount_ = base / worldSize_;
  expectedCount_ = base;

  mscclpp::DeviceSyncer syncer = {};
  CUDATHROW(cudaMemcpyToSymbol(deviceSyncer, &syncer, sizeof(mscclpp::DeviceSyncer)));
}

std::vector<KernelRestriction> AllToAllTestColl::getKernelRestrictions() {
  return {// {kernelNum, kernelName, compatibleWithMultiNodes, countDivisorForMultiNodes, alignedBytes}
          {0, "alltoall0", true, 1, 4 * worldSize_},
          {1, "alltoall1", false, 1, 4 * worldSize_},
          {2, "alltoall2", false, 1, 4 * worldSize_},
          {3, "alltoall3", false, 1, 4 * worldSize_}};
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
  bool isInPlace() const;

  std::shared_ptr<int> sendBuff_;
  std::shared_ptr<int> recvBuff_;
  std::shared_ptr<int[]> expectedBuff_;
  std::shared_ptr<int> scratchBuff_;

  std::vector<mscclpp::MemoryChannel> memoryChannels;
};

bool AllToAllTestEngine::isInPlace() const { return false; }

AllToAllTestEngine::AllToAllTestEngine(const TestArgs& args) : BaseTestEngine(args, "alltoall") { inPlace_ = false; }

void AllToAllTestEngine::allocateBuffer() {
  sendBuff_ = mscclpp::GpuBuffer<int>(args_.maxBytes / sizeof(int)).memory();
  recvBuff_ = mscclpp::GpuBuffer<int>(args_.maxBytes / sizeof(int)).memory();
  expectedBuff_ = std::shared_ptr<int[]>(new int[args_.maxBytes / sizeof(int)]);
  scratchBuff_ = mscclpp::GpuBuffer<int>(args_.maxBytes / sizeof(int)).memory();

  localSendBuff = sendBuff_.get();
  localRecvBuff = recvBuff_.get();
  localScratchBuff = scratchBuff_.get();
}

void AllToAllTestEngine::setupConnections() {
  std::vector<DeviceHandle<mscclpp::PortChannel>> portChannels;
  std::vector<DeviceHandle<mscclpp::MemoryChannel>> memoryChannelHandles;
  setupMeshConnections(portChannels, sendBuff_.get(), args_.maxBytes, recvBuff_.get(), args_.maxBytes);
  setupMeshConnections(this->memoryChannels, sendBuff_.get(), args_.maxBytes, scratchBuff_.get(), args_.maxBytes,
                       ChannelSemantic::PUT, 64);

  if (portChannels.size() > sizeof(constPortChans) / sizeof(DeviceHandle<mscclpp::PortChannel>)) {
    throw std::runtime_error("unexpected error");
  }
  CUDATHROW(cudaMemcpyToSymbol(constPortChans, portChannels.data(),
                               sizeof(DeviceHandle<mscclpp::PortChannel>) * portChannels.size()));
  std::transform(this->memoryChannels.begin(), this->memoryChannels.end(), std::back_inserter(memoryChannelHandles),
                 [](const mscclpp::MemoryChannel& channel) { return mscclpp::deviceHandle(channel); });
  if (memoryChannelHandles.size() > sizeof(constMemChans) / sizeof(DeviceHandle<mscclpp::MemoryChannel>)) {
    throw std::runtime_error("unexpected error");
  }
  CUDATHROW(cudaMemcpyToSymbol(constMemChans, memoryChannelHandles.data(),
                               sizeof(DeviceHandle<mscclpp::MemoryChannel>) * memoryChannelHandles.size()));
}

std::vector<void*> AllToAllTestEngine::getSendBuff() { return {sendBuff_.get()}; }
void* AllToAllTestEngine::getExpectedBuff() { return expectedBuff_.get(); }
void* AllToAllTestEngine::getRecvBuff() {
  if (this->isInPlace())
    return sendBuff_.get();
  else
    return recvBuff_.get();
}
void* AllToAllTestEngine::getScratchBuff() { return nullptr; }

std::shared_ptr<BaseTestEngine> getTestEngine(const TestArgs& args) {
  return std::make_shared<AllToAllTestEngine>(args);
}
std::shared_ptr<BaseTestColl> getTestColl() { return std::make_shared<AllToAllTestColl>(); }