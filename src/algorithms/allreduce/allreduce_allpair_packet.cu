// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "algorithm_utils.hpp"
#include "allreduce_allpair_packet.hpp"
#include "allreduce_common.hpp"
#include "debug.h"

namespace mscclpp {
namespace algorithm {

__device__ uint32_t deviceFlag = 1;

template <Op OpType, typename T>
__global__ void allreduceAllPairs(T* buff, T* scratch, T* resultBuff, DeviceHandle<MemoryChannel>* memoryChannels,
                                  size_t channelDataOffset, size_t scratchBufferSize, int rank, int nRanksPerNode,
                                  int worldSize, size_t nelems, uint32_t numScratchBuff, LL8Packet* flags) {
  // This version of allreduce only works for single nodes
  if (worldSize != nRanksPerNode) return;

  if (sizeof(T) == 2 || sizeof(T) == 1) nelems = (nelems * sizeof(T) + sizeof(T)) / sizeof(int);
  const int nPeers = nRanksPerNode - 1;

  uint32_t flag = deviceFlag;
  if (threadIdx.x == 0) {
    flags[blockIdx.x].write(0, flag);
  }

  size_t scratchBaseOffset = (flag % numScratchBuff) ? scratchBufferSize / numScratchBuff : 0;
  size_t channelScratchOffset = scratchBaseOffset;

  const int nBlocksPerPeer = gridDim.x / nPeers;
  const int localBlockIdx = blockIdx.x % nBlocksPerPeer;
  const int tid = threadIdx.x + localBlockIdx * blockDim.x;
  const int peerIdx = blockIdx.x / nBlocksPerPeer;
  size_t srcOffset = channelDataOffset;
  size_t scratchOffset = channelScratchOffset + rank * nelems * sizeof(LL8Packet);
  void* scratchBuff = (void*)((char*)scratch + channelScratchOffset);
  uint32_t* src = (uint32_t*)((char*)buff);
  uint32_t* dst = (uint32_t*)((char*)resultBuff);

  // step 1: write data to each peer's scratch buffer
  memoryChannels[peerIdx].putPackets<LL8Packet>(scratchOffset, srcOffset, nelems * sizeof(uint32_t), tid,
                                                blockDim.x * nBlocksPerPeer, flag);

  // step 2: Reduce Data
  for (size_t idx = threadIdx.x + blockIdx.x * blockDim.x; idx < nelems; idx += blockDim.x * gridDim.x) {
    uint32_t data = src[idx];
    for (int index = 0; index < nPeers; index++) {
      const int remoteRank = index < rank ? index : index + 1;
      LL8Packet* dstPkt = (LL8Packet*)scratchBuff + remoteRank * nelems;
      uint32_t val = dstPkt[idx].read(flag, -1);
      data = cal_vectors<T, OpType>(val, data);
    }
    dst[idx] = data;
  }
  // Make sure all threadblocks have finished reading before incrementing the flag
  if (blockIdx.x == 0 && threadIdx.x < gridDim.x) {
    flags[threadIdx.x].read(flag, -1);
  }
  __syncthreads();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    deviceFlag++;
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
                          DeviceHandle<SwitchChannel>*, DeviceHandle<SwitchChannel>*, DeviceHandle<SwitchChannel>*,
                          size_t channelInOffset, size_t, size_t scratchBufferSize, int rank, int nRanksPerNode,
                          int worldSize, size_t inputSize, cudaStream_t stream, uint32_t* deviceFlag7,
                          uint32_t* deviceFlag28, uint32_t* deviceFlag56, uint32_t numScratchBuff, int nBlocks = 0,
                          int nThreadsPerBlock = 0) {
    using ChannelType = DeviceHandle<MemoryChannel>;
    const size_t nelems = inputSize / sizeof(T);
    allreduceAllPairs<OpType><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
        (T*)buff, (T*)scratch, (T*)resultBuff, (ChannelType*)memoryChannels, channelInOffset, scratchBufferSize, rank,
        nRanksPerNode, worldSize, nelems, deviceFlag28, numScratchBuff);
    return cudaGetLastError();
  }
};

void AllreduceAllpairPacket::initialize(std::shared_ptr<Communicator> comm) {
  conns_ = setupConnections(comm);
  memorySemaphores_ = setupMemorySemaphores(comm, conns_, maxBlockNum_);
  RegisteredMemory scratchMemory = comm->registerMemory(scratchBuffer_, scratchBufferSize_, Transport::CudaIpc);
  registeredMemories_ = setupRemoteMemories(comm, comm->bootstrap()->getRank(), scratchMemory);
  registeredMemories_.push_back(scratchMemory);
}

CommResult AllreduceAllpairPacket::allreduceKernelFunc(const std::shared_ptr<AlgorithmCtx> ctx, const void* input,
                                                       void* output, size_t inputSize, [[maybe_unused]] DataType dtype,
                                                       cudaStream_t stream,
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
      allreduce(input, this->scratchBuffer_, output, ctx->memoryChannelDeviceHandles.get(), nullptr, nullptr, nullptr,
                channelInOffset, 0, this->scratchBufferSize_, ctx->rank, ctx->nRanksPerNode, ctx->workSize, inputSize,
                stream, this->nSegmentsForScratchBuffer_, blockAndThreadNum.first, blockAndThreadNum.second);
  if (error != cudaSuccess) {
    WARN("AllreducePacket failed with error: %s", cudaGetErrorString(error));
    return CommResult::commUnhandledCudaError;
  }
  return CommResult::commSuccess;
}

std::shared_ptr<AlgorithmCtx> AllreduceAllpairPacket::initAllreduceContext(std::shared_ptr<Communicator> comm,
                                                                           const void* input, void*, size_t, DataType) {
  auto ctx = std::make_shared<AlgorithmCtx>();
  const int nChannelsPerConnection = maxBlockNum_;
  ctx->rank = comm->bootstrap()->getRank();
  ctx->workSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();
  ctx->memorySemaphores = this->memorySemaphores_;
  ctx->registeredMemories = this->registeredMemories_;
  ctx->registeredMemories.pop_back();  // remove the local memory from previous context

  size_t sendBytes;
  CUdeviceptr sendBasePtr;
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&sendBasePtr, &sendBytes, (CUdeviceptr)input));
  RegisteredMemory localMemory = comm->registerMemory((void*)sendBasePtr, sendBytes, Transport::CudaIpc);

  // setup channels
  ctx->memoryChannels = setupMemoryChannels(this->conns_, ctx->memorySemaphores, ctx->registeredMemories, localMemory,
                                            nChannelsPerConnection);
  ctx->memoryChannelDeviceHandles = setupMemoryChannelDeviceHandles(ctx->memoryChannels);
  ctx->registeredMemories.emplace_back(localMemory);
  return ctx;
}

AlgorithmCtxKey AllreduceAllpairPacket::generateAllreduceContextKey(const void* input, void*, size_t, DataType) {
  size_t sendBytes;
  CUdeviceptr sendBasePtr;
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&sendBasePtr, &sendBytes, (CUdeviceptr)input));
  return AlgorithmCtxKey{(void*)sendBasePtr, nullptr, sendBytes, 0, 0};
}

std::shared_ptr<Algorithm> AllreduceAllpairPacket::build() {
  auto self = std::make_shared<AllreduceAllpairPacket>(scratchBuffer_, scratchBufferSize_);
  return std::make_shared<NativeAlgorithm>(
      "default_allreduce_allpair_packet", "allreduce",
      [self](std::shared_ptr<Communicator> comm) { self->initialize(comm); },
      [self](const std::shared_ptr<AlgorithmCtx> ctx, const void* input, void* output, size_t inputSize,
             [[maybe_unused]] size_t outputSize, DataType dtype, cudaStream_t stream,
             std::unordered_map<std::string, uintptr_t>& extras) {
        return self->allreduceKernelFunc(ctx, input, output, inputSize, dtype, stream, extras);
      },
      [self](std::shared_ptr<Communicator> comm, const void* input, void* output, size_t inputSize,
             [[maybe_unused]] size_t outputSize,
             DataType dtype) { return self->initAllreduceContext(comm, input, output, inputSize, dtype); },
      [self](const void* input, void* output, size_t inputSize, [[maybe_unused]] size_t outputSize, DataType dtype) {
        return self->generateAllreduceContextKey(input, output, inputSize, dtype);
      });
}
}  // namespace algorithm
}  // namespace mscclpp