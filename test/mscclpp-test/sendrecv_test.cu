#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mscclpp/channel.hpp>
#include <mscclpp/epoch.hpp>
#include <string>
#include <vector>

#include "common.hpp"

constexpr size_t BLOCK_THREADS_NUM = 1024;
// Try to use more blocks if per-block data size exceeds this threshold
constexpr size_t THRES_BYTES_PER_BLOCK = 8192;
// Let it no more than the number of SMs on a GPU
constexpr size_t MAX_BLOCKS_NUM = 32;

#define ALIGN 4

__constant__ mscclpp::channel::SimpleDeviceChannel constDevChans[2];

struct SyncGpuState
{
  volatile int flag;
  int cnt;
  int is_add;
};

// Synchronize multiple thread blocks inside a kernel. Guarantee that all
// previous work of all threads in cooperating blocks is finished and
// visible to all threads in the device.
__forceinline__ __device__ void sync_gpu(SyncGpuState& state, int blockNum)
{
  int maxOldCnt = blockNum - 1;
  __syncthreads();
  if (threadIdx.x == 0) {
    int is_add_ = state.is_add ^ 1;
    if (is_add_) {
      if (atomicAdd(&state.cnt, 1) == maxOldCnt) {
        state.flag = 1;
      }
      while (!state.flag) {
      }
    } else {
      if (atomicSub(&state.cnt, 1) == 1) {
        state.flag = 0;
      }
      while (state.flag) {
      }
    }
    state.is_add = is_add_;
  }
  // We need sync here because only a single thread is checking whether
  // the flag is flipped.
  __syncthreads();
}

inline int getBlockNum(size_t count)
{
  return std::min((count + THRES_BYTES_PER_BLOCK - 1) / THRES_BYTES_PER_BLOCK, MAX_BLOCKS_NUM);
}

inline mscclpp::Transport getTransport(int rank, int peerRank, int nRanksPerNode, mscclpp::Transport ibDevice) {
  return rank / nRanksPerNode == peerRank / nRanksPerNode ? mscclpp::Transport::CudaIpc : ibDevice;
}

__device__ SyncGpuState GLOBAL_SYNC_STATE;

__global__ void kernel(int rank, size_t dataSize, size_t dataPerBlock)
{
  size_t startIndex = blockIdx.x * dataPerBlock;
  size_t blockDataSize = min(dataSize - startIndex, dataPerBlock);
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  mscclpp::channel::SimpleDeviceChannel sendConn = constDevChans[0];
  mscclpp::channel::SimpleDeviceChannel recvConn = constDevChans[1];

  sendConn.putDirect(startIndex, blockDataSize, threadIdx.x, blockDim.x);
  sync_gpu(GLOBAL_SYNC_STATE, gridDim.x);
  if (tid == 0) {
    sendConn.signalDirect();
    recvConn.wait();
  }
}

class SendRecvTestColl : public BaseTestColl {
 public:
  SendRecvTestColl() = default;
  ~SendRecvTestColl() override = default;

  void runColl(const TestArgs& args, cudaStream_t stream) override;
  void initData(const TestArgs& args, std::vector<void*> sendBuff, void* expectedBuff) override;
  void getBw(const double deltaSec, double& algBW /*OUT*/, double& busBw /*OUT*/) override;
  void setupCollTest(size_t size, size_t typeSize) override;
};

void SendRecvTestColl::runColl(const TestArgs& args, cudaStream_t stream) {
  size_t sendBytes = sendCount_ * typeSize_;
  int blockNum = getBlockNum(sendBytes);
  size_t bytesPerBlock = (sendBytes + blockNum - 1) / blockNum;
  kernel<<<blockNum, BLOCK_THREADS_NUM, 0, stream>>>(args.rank, sendBytes, bytesPerBlock);
}

void SendRecvTestColl::getBw(const double deltaSec, double& algBW /*OUT*/, double& busBw /*OUT*/) {
  double baseBw = (double)(paramCount_ * typeSize_) / 1.0E9 / deltaSec;
  algBW = baseBw;
  double factor = 1;
  busBw = baseBw * factor;
}

void SendRecvTestColl::initData(const TestArgs& args, std::vector<void*> sendBuff, void* expectedBuff) {
  int rank = args.rank;
  assert(sendBuff.size() == 1);
  CUDATHROW(cudaMemset(sendBuff[0], 0, sendCount_ * typeSize_));

  // TODO: The type should not limited to int.
  std::vector<int> dataHost(std::max(sendCount_, recvCount_), rank);
  CUDATHROW(cudaMemcpy(sendBuff[0], dataHost.data(), sendCount_ * typeSize_, cudaMemcpyHostToDevice));

  int peerRank = (rank - 1 + args.totalRanks) % args.totalRanks;
  for (size_t i = 0; i < recvCount_; i++) {
    dataHost[i] = peerRank;
  }

  std::memcpy(expectedBuff, dataHost.data(), recvCount_ * typeSize_);
}

void SendRecvTestColl::setupCollTest(size_t size, size_t typeSize) {
  size_t count = size / typeSize;
  size_t base = (count / ALIGN) * ALIGN;
  sendCount_ = base;
  recvCount_ = base;
  paramCount_ = base;
  expectedCount_ = base;
  typeSize_ = typeSize;

  SyncGpuState state = {};
  CUDATHROW(cudaMemcpyToSymbol(GLOBAL_SYNC_STATE, &state, sizeof(SyncGpuState)));
}

class SendRecvTestEngine : public BaseTestEngine {
 public:
  SendRecvTestEngine() : BaseTestEngine(false) {};
  ~SendRecvTestEngine() override = default;

  void allocateBuffer() override;
  void setupConnections() override;
  void teardown() override;

 private:
  std::vector<void*> getSendBuff() override;
  void* getExpectedBuff() override;
  void* getRecvBuff() override;

  std::vector<void*> devicePtrs_;
  std::shared_ptr<int[]> expectedBuff_;

  std::vector<std::shared_ptr<mscclpp::Connection>> conns_;
};

void SendRecvTestEngine::allocateBuffer() {
  int *sendBuff = nullptr, *recvBuff = nullptr;
  CUDATHROW(cudaMalloc(&sendBuff, args_.maxBytes));
  CUDATHROW(cudaMalloc(&recvBuff, args_.maxBytes));
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

  std::vector<mscclpp::channel::ChannelId> chanIds;

  chanIds.push_back(chanService_->addChannel(
      comm_->connectOnSetup(sendToRank, 0, getTransport(args_.rank, sendToRank, args_.nRanksPerNode, ibDevice))));
  if (recvFromRank != sendToRank) {
    chanIds.push_back(chanService_->addChannel(
        comm_->connectOnSetup(recvFromRank, 0, getTransport(args_.rank, recvFromRank, args_.nRanksPerNode, ibDevice))));
  } else {
    // reuse the send channel if worldSize is 2
    chanIds.push_back(chanIds[0]);
  }
  comm_->setup();

  std::vector<mscclpp::RegisteredMemory> localMemories;
  std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> futureRemoteMemory;

  for (int i : {0, 1}) {
    auto regMem = comm_->registerMemory(devicePtrs_[i], args_.maxBytes, mscclpp::Transport::CudaIpc | ibDevice);
    comm_->sendMemoryOnSetup(regMem, ranks[i], 0);
    localMemories.push_back(regMem);
    futureRemoteMemory.push_back(comm_->recvMemoryOnSetup(ranks[1 - i], 0));
    comm_->setup();
  }

  // swap to make sure devicePtrs_[0] in local rank write to devicePtrs_[1] in remote rank
  std::swap(futureRemoteMemory[0], futureRemoteMemory[1]);
  std::vector<mscclpp::channel::SimpleDeviceChannel> devChannels;
  for (int i : {0, 1}) {
    // We assume ranks in the same node
    devChannels.push_back(mscclpp::channel::SimpleDeviceChannel(
        chanService_->deviceChannel(chanIds[i]), futureRemoteMemory[i].get().data(), localMemories[i].data()));
  }
  cudaMemcpyToSymbol(constDevChans, devChannels.data(), sizeof(mscclpp::channel::SimpleDeviceChannel) * devChannels.size());
}

void SendRecvTestEngine::teardown() {
  for (auto& ptr : devicePtrs_) {
    CUDATHROW(cudaFree(ptr));
  }
  devicePtrs_.clear();
}

std::vector<void*> SendRecvTestEngine::getSendBuff() { return {devicePtrs_[0]}; }

void* SendRecvTestEngine::getExpectedBuff() { return expectedBuff_.get(); }

void* SendRecvTestEngine::getRecvBuff() { return devicePtrs_[1]; }

#pragma weak testEngine = sendRecvTestEngine
#pragma weak testColl = sendRecvTestColl

std::shared_ptr<BaseTestEngine> sendRecvTestEngine = std::make_shared<SendRecvTestEngine>();
std::shared_ptr<BaseTestColl> sendRecvTestColl = std::make_shared<SendRecvTestColl>();
