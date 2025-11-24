// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "algorithm_utils.hpp"
#include "allreduce_allpair_packet.hpp"
#include "allreduce_common.hpp"
#include "debug.h"

namespace mscclpp {

template <Op OpType, typename T>
__global__ void allreduceAllPairs(T* buff, T* scratch, T* resultBuff,
                                  mscclpp::DeviceHandle<mscclpp::MemoryChannel>* memoryChannels,
                                  size_t channelDataOffset, size_t scratchBufferSize, int rank, int nRanksPerNode,
                                  int worldSize, size_t nelems, uint32_t* deviceFlag, uint32_t numScratchBuff) {
  // This version of allreduce only works for single nodes
  if (worldSize != nRanksPerNode) return;

  if (sizeof(T) == 2 || sizeof(T) == 1) nelems = (nelems * sizeof(T) + sizeof(T)) / sizeof(int);
  const int nPeers = nRanksPerNode - 1;

  uint32_t flag = deviceFlag[blockIdx.x];

  size_t scratchBaseOffset = (flag % numScratchBuff) ? scratchBufferSize / numScratchBuff : 0;
  size_t channelScratchOffset = scratchBaseOffset;

  const int nBlocksPerPeer = gridDim.x / nPeers;
  const int localBlockIdx = blockIdx.x % nBlocksPerPeer;
  const int tid = threadIdx.x + localBlockIdx * blockDim.x;
  const int peerIdx = blockIdx.x / nBlocksPerPeer;
  size_t srcOffset = channelDataOffset;
  size_t scratchOffset = channelScratchOffset + rank * nelems * sizeof(mscclpp::LL8Packet);
  void* scratchBuff = (void*)((char*)scratch + channelScratchOffset);
  uint32_t* src = (uint32_t*)((char*)buff);
  uint32_t* dst = (uint32_t*)((char*)resultBuff);

  // step 1: write data to each peer's scratch buffer
  memoryChannels[peerIdx].putPackets<mscclpp::LL8Packet>(scratchOffset, srcOffset, nelems * sizeof(uint32_t), tid,
                                                         blockDim.x * nBlocksPerPeer, flag);

  // step 2: Reduce Data
  for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nelems; idx += blockDim.x * gridDim.x) {
    uint32_t data = src[idx];
    for (int index = 0; index < nPeers; index++) {
      const int remoteRank = index < rank ? index : index + 1;
      mscclpp::LL8Packet* dstPkt = (mscclpp::LL8Packet*)scratchBuff + remoteRank * nelems;
      uint32_t val = dstPkt[idx].read(flag, -1);
      data = cal_vectors<T, OpType>(val, data);
    }
    dst[idx] = data;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    deviceFlag[blockIdx.x] = deviceFlag[blockIdx.x] + 1;
  }
}

inline std::pair<int, int> getDefaultBlockNumAndThreadNum(size_t inputSize, int worldSize) {
  if (inputSize < worldSize * sizeof(int)) {
    return {worldSize - 1, 32};
  }
  return {(worldSize - 1) * 4, 512};
}

template <Op OpType, typename T>
struct AllpairAdapter {
  static cudaError_t call(const void* buff, void* scratch, void* resultBuff, void* memoryChannels, void*,
                          mscclpp::DeviceHandle<mscclpp::SwitchChannel>*,
                          mscclpp::DeviceHandle<mscclpp::SwitchChannel>*, size_t channelInOffset, size_t,
                          size_t scratchBufferSize, int rank, int nRanksPerNode, int worldSize, size_t inputSize,
                          cudaStream_t stream, uint32_t* deviceFlag7, uint32_t* deviceFlag28, uint32_t* deviceFlag56,
                          uint32_t numScratchBuff, int nBlocks = 0, int nThreadsPerBlock = 0) {
    using ChannelType = mscclpp::DeviceHandle<mscclpp::MemoryChannel>;
    const size_t nelems = inputSize / sizeof(T);
    allreduceAllPairs<OpType><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
        (T*)buff, (T*)scratch, (T*)resultBuff, (ChannelType*)memoryChannels, channelInOffset, scratchBufferSize, rank,
        nRanksPerNode, worldSize, nelems, deviceFlag28, numScratchBuff);
    return cudaGetLastError();
  }
};

void AllreducePacket::initialize(std::shared_ptr<mscclpp::Communicator> comm) {
  deviceFlag7_ = mscclpp::detail::gpuCallocShared<uint32_t>(7);
  deviceFlag28_ = mscclpp::detail::gpuCallocShared<uint32_t>(28);
  deviceFlag56_ = mscclpp::detail::gpuCallocShared<uint32_t>(56);
  std::vector<uint32_t> initFlag(56);
  for (int i = 0; i < 56; ++i) {
    initFlag[i] = 1;
  }
  mscclpp::gpuMemcpy<uint32_t>(deviceFlag7_.get(), initFlag.data(), 7, cudaMemcpyHostToDevice);
  mscclpp::gpuMemcpy<uint32_t>(deviceFlag28_.get(), initFlag.data(), 28, cudaMemcpyHostToDevice);
  mscclpp::gpuMemcpy<uint32_t>(deviceFlag56_.get(), initFlag.data(), 56, cudaMemcpyHostToDevice);
  this->conns_ = setupConnections(comm);
}

CommResult AllreducePacket::allreduceKernelFunc(const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, const void* input,
                                                void* output, size_t inputSize,
                                                [[maybe_unused]] mscclpp::DataType dtype, cudaStream_t stream,
                                                std::unordered_map<std::string, uintptr_t>& extras) {
  Algorithm::Op op = *reinterpret_cast<Algorithm::Op*>(extras.at("op"));
  std::pair<int, int> blockAndThreadNum = getBlockNumAndThreadNum(extras);
  if (blockAndThreadNum.first == 0 || blockAndThreadNum.second == 0) {
    blockAndThreadNum = getDefaultBlockNumAndThreadNum(inputSize, ctx->workSize);
  }

  size_t sendBytes;
  CUdeviceptr sendBasePtr;
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&sendBasePtr, &sendBytes, (CUdeviceptr)input));
  size_t channelInOffset = (char*)input - (char*)sendBasePtr;

  AllreduceFunc allreduce = dispatch<AllpairAdapter>(op, dtype);
  if (!allreduce) {
    WARN("Unsupported operation or data type for allreduce: op=%d, dtype=%d", op, static_cast<int>(dtype));
    return CommResult::commInvalidArgument;
  }
  cudaError_t error =
      allreduce(input, this->scratchBuffer_.lock().get(), output, ctx->memoryChannelDeviceHandles.get(), nullptr,
                nullptr, nullptr, channelInOffset, 0, this->scratchBufferSize_, ctx->rank, ctx->nRanksPerNode,
                ctx->workSize, inputSize, stream, deviceFlag7_.get(), deviceFlag28_.get(), deviceFlag56_.get(),
                this->nSegmentsForScratchBuffer_, blockAndThreadNum.first, blockAndThreadNum.second);
  if (error != cudaSuccess) {
    WARN("AllreducePacket failed with error: %s", cudaGetErrorString(error));
    return CommResult::commUnhandledCudaError;
  }
  return CommResult::commSuccess;
}

std::shared_ptr<mscclpp::AlgorithmCtx> AllreducePacket::initAllreduceContext(
    std::shared_ptr<mscclpp::Communicator> comm, const void* input, void*, size_t, mscclpp::DataType) {
  auto ctx = std::make_shared<mscclpp::AlgorithmCtx>();
  const int nChannelsPerConnection = maxBlockNum_;
  ctx->rank = comm->bootstrap()->getRank();
  ctx->workSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();
  if (this->ctx_ == nullptr) {
    // setup semaphores
    ctx->memorySemaphores = setupMemorySemaphores(comm, this->conns_, nChannelsPerConnection);
    // setup registered memories
    mscclpp::RegisteredMemory scratchMemory =
        comm->registerMemory(this->scratchBuffer_.lock().get(), this->scratchBufferSize_, mscclpp::Transport::CudaIpc);
    std::vector<mscclpp::RegisteredMemory> remoteMemories = setupRemoteMemories(comm, ctx->rank, scratchMemory);
    ctx->registeredMemories = std::move(remoteMemories);
    ctx->registeredMemories.push_back(scratchMemory);
  } else {
    ctx->memorySemaphores = ctx_->memorySemaphores;
    ctx->registeredMemories = ctx_->registeredMemories;
    ctx->registeredMemories.pop_back();  // remove the local memory from previous context
  }

  size_t sendBytes;
  CUdeviceptr sendBasePtr;
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&sendBasePtr, &sendBytes, (CUdeviceptr)input));
  mscclpp::RegisteredMemory localMemory =
      comm->registerMemory((void*)sendBasePtr, sendBytes, mscclpp::Transport::CudaIpc);

  // setup channels
  ctx->memoryChannels = setupMemoryChannels(this->conns_, ctx->memorySemaphores, ctx->registeredMemories, localMemory,
                                            nChannelsPerConnection);
  ctx->memoryChannelDeviceHandles = setupMemoryChannelDeviceHandles(ctx->memoryChannels);
  ctx->registeredMemories.emplace_back(localMemory);

  this->ctx_ = ctx;

  return ctx;
}

mscclpp::AlgorithmCtxKey AllreducePacket::generateAllreduceContextKey(const void* input, void*, size_t,
                                                                      mscclpp::DataType) {
  size_t sendBytes;
  CUdeviceptr sendBasePtr;
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&sendBasePtr, &sendBytes, (CUdeviceptr)input));
  return mscclpp::AlgorithmCtxKey{(void*)sendBasePtr, nullptr, sendBytes, 0, 0};
}

std::shared_ptr<mscclpp::Algorithm> AllreducePacket::build() {
  auto self = std::make_shared<AllreducePacket>(scratchBuffer_.lock(), scratchBufferSize_);
  return std::make_shared<mscclpp::NativeAlgorithm>(
      "default_allreduce_packet", "allreduce",
      [self](std::shared_ptr<mscclpp::Communicator> comm) { self->initialize(comm); },
      [self](const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, const void* input, void* output, size_t inputSize,
             [[maybe_unused]] size_t outputSize, mscclpp::DataType dtype, cudaStream_t stream,
             std::unordered_map<std::string, uintptr_t>& extras) {
        return self->allreduceKernelFunc(ctx, input, output, inputSize, dtype, stream, extras);
      },
      [self](std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output, size_t inputSize,
             [[maybe_unused]] size_t outputSize,
             mscclpp::DataType dtype) { return self->initAllreduceContext(comm, input, output, inputSize, dtype); },
      [self](const void* input, void* output, size_t inputSize, [[maybe_unused]] size_t outputSize,
             mscclpp::DataType dtype) { return self->generateAllreduceContextKey(input, output, inputSize, dtype); });
}
}  // namespace mscclpp