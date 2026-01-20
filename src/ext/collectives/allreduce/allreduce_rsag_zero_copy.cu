// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "allreduce/allreduce_rsag_zero_copy.hpp"
#include "allreduce/common.hpp"
#include "collective_utils.hpp"
#include "debug.h"

namespace mscclpp {
namespace collective {

__device__ mscclpp::DeviceSyncer globalSyncer;

template <int NRanksPerNode, ReduceOp OpType, typename T>
__global__ void __launch_bounds__(1024, 1)
    allreduceRsAgZeroCopy(T* buff, T* scratch, T* resultBuff, DeviceHandle<BaseMemoryChannel>* memoryChannels,
                          DeviceHandle<SwitchChannel>* switchChannels, void* remoteMemories, int rank,
                          int worldSize, size_t nelems) {
  int blockId = blockIdx.x;

  assert((uintptr_t)buff % sizeof(int4) == 0);
  assert((uintptr_t)resultBuff % sizeof(int4) == 0);

  constexpr int NPeers = NRanksPerNode - 1;           
  constexpr uint32_t nelemsPerInt4 = sizeof(int4) / sizeof(T);
  const uint32_t outputRemoteBufferOffset = NRanksPerNode - 1;
  uint32_t alignedNelems = ((nelems + NRanksPerNode - 1) / NRanksPerNode + nelemsPerInt4 - 1) / nelemsPerInt4 *
                           nelemsPerInt4 * NRanksPerNode;
  uint32_t nelemsPerRank = alignedNelems / NRanksPerNode;
  uint32_t nInt4PerRank = nelemsPerRank / nelemsPerInt4;
  uint32_t nInt4Total = (nelems + nelemsPerInt4 - 1) / nelemsPerInt4;

  int4* resultBuff4 = reinterpret_cast<int4*>((char*)resultBuff);
  int4* buff4 = reinterpret_cast<int4*>((char*)buff);
  DeviceHandle<BaseMemoryChannel>* memoryChannelsLocal = memoryChannels + blockId * NPeers;

  uint32_t nInt4PerBlock = nInt4PerRank / gridDim.x;
  uint32_t remainderForBlock = nInt4PerRank % gridDim.x;
  uint32_t offset4 = blockId * nInt4PerBlock;
  if (blockId == gridDim.x - 1) {
    nInt4PerBlock += remainderForBlock;
  }
  if (nInt4PerBlock == 0) return;

  if (threadIdx.x < NPeers) {
    memoryChannelsLocal[threadIdx.x].relaxedSignal();
    memoryChannelsLocal[threadIdx.x].relaxedWait();
  }
  __syncthreads();
  int4 data[NPeers];
  for (uint32_t idx = threadIdx.x; idx < nInt4PerBlock; idx += blockDim.x) {
    uint32_t offset = idx + offset4 + rank * nInt4PerRank;
    if (offset >= nInt4Total) continue;
    int4 tmp = buff4[offset];
    #pragma unroll
    for (int i = 0; i < NPeers; i++) {
      int rankIdx = (rank + i + 1) % NRanksPerNode;
      int peerIdx = rankIdx < rank ? rankIdx : rankIdx - 1;
      data[i] = mscclpp::read<int4>(((void**)remoteMemories)[peerIdx], offset);
    }
    for (int i = 0; i < NPeers; i++) {
      tmp = cal_vectors<T, OpType>(data[i], tmp);
    }
    #pragma unroll
    for (int i = 0; i < NPeers; i++) {
      int rankIdx = (rank + i + 1) % NRanksPerNode;
      int peerIdx = rankIdx < rank ? rankIdx : rankIdx - 1;
      mscclpp::write<int4>(((void**)remoteMemories)[outputRemoteBufferOffset + peerIdx], offset, tmp);
    }
    resultBuff4[offset] = tmp;
  }
  globalSyncer.sync(gridDim.x);
  if (blockIdx.x == 0 && threadIdx.x < NPeers) {
    memoryChannelsLocal[threadIdx.x].signal();
    memoryChannelsLocal[threadIdx.x].wait();
  }
}

template <ReduceOp OpType, typename T>
struct AllreduceRsAgZeroCopyAdapter {
  static cudaError_t call(const void* input, void* scratch, void* output, void* memoryChannels, void* remoteMemories,
                          DeviceHandle<SwitchChannel>* switchChannel, DeviceHandle<SwitchChannel>*, size_t, size_t,
                          size_t, int rank, int nRanksPerNode, int worldSize, size_t inputSize, cudaStream_t stream,
                          void*, uint32_t, int nBlocks, int nThreadsPerBlock) {
    using ChannelType = DeviceHandle<BaseMemoryChannel>;
    size_t nelems = inputSize / sizeof(T);
    if (nBlocks == 0 || nThreadsPerBlock == 0) {
      nThreadsPerBlock = 1024;
      nBlocks = 64;
      if (inputSize >= (1 << 26)) {
        nBlocks = 128;
      }
    }
    if (nRanksPerNode == 4) {
      allreduceRsAgZeroCopy<4, OpType, T>
          <<<nBlocks, nThreadsPerBlock, 0, stream>>>((T*)input, (T*)scratch, (T*)output, (ChannelType*)memoryChannels,
                                                     switchChannel, remoteMemories, rank, worldSize, nelems);
    }
    return cudaGetLastError();
  }
};

void AllreduceRsAgZeroCopy::initialize(std::shared_ptr<Communicator> comm) {
  this->conns_ = setupConnections(comm);
  nChannelsPerConnection_ = 128;
  comm_ = comm;
  // setup semaphores
  this->semaphores_ = setupMemorySemaphores(comm, this->conns_, nChannelsPerConnection_);
  this->baseChannels_ = setupBaseMemoryChannels(this->conns_, this->semaphores_, nChannelsPerConnection_);
  this->baseMemoryChannelHandles_ = setupBaseMemoryChannelDeviceHandles(baseChannels_);
}

CommResult AllreduceRsAgZeroCopy::allreduceKernelFunc(const std::shared_ptr<void> ctx, const void* input, void* output,
                                                      size_t inputSize, DataType dtype, ReduceOp op,
                                                      cudaStream_t stream, int nBlocks, int nThreadsPerBlock,
                                                      const std::unordered_map<std::string, uintptr_t>&) {
  auto algoCtx = std::static_pointer_cast<AlgorithmCtx>(ctx);
  AllreduceFunc allreduce = dispatch<AllreduceRsAgZeroCopyAdapter>(op, dtype);
  if (!allreduce) {
    WARN("Unsupported operation or data type for allreduce: op=%d, dtype=%d", static_cast<int>(op),
         static_cast<int>(dtype));
    return CommResult::CommInvalidArgument;
  }
  std::pair<int, int> numBlocksAndThreads = {nBlocks, nThreadsPerBlock};
  cudaError_t error = allreduce(input, nullptr, output, this->baseMemoryChannelHandles_.get(),
                                algoCtx->remoteMemoryHandles.get(), nullptr, nullptr,
                                0, 0, 0, algoCtx->rank, algoCtx->nRanksPerNode, algoCtx->workSize, inputSize, stream,
                                nullptr, 0, numBlocksAndThreads.first, numBlocksAndThreads.second);
  if (error != cudaSuccess) {
    WARN("AllreduceAllconnect failed with error: %s", cudaGetErrorString(error));
    return CommResult::CommUnhandledCudaError;
  }
  return CommResult::CommSuccess;
}

AlgorithmCtxKey AllreduceRsAgZeroCopy::generateAllreduceContextKey(const void* inputBuffer, void* outputBuffer, size_t size, DataType) {
  return AlgorithmCtxKey{(void*)inputBuffer, outputBuffer, size, size, 0};
}

std::shared_ptr<void> AllreduceRsAgZeroCopy::initAllreduceContext(std::shared_ptr<Communicator> comm, const void* input,
                                                                  void* output, size_t size, DataType) {
  auto ctx = std::make_shared<AlgorithmCtx>();
  ctx->rank = comm->bootstrap()->getRank();
  ctx->workSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();

  ctx->memorySemaphores = this->semaphores_;

  // register input and output memories
  RegisteredMemory inputMemory = comm->registerMemory((void*)input, size, Transport::CudaIpc);
  RegisteredMemory outputMemory = comm->registerMemory(output, size, Transport::CudaIpc);
  this->inputMemories_.push_back(inputMemory);
  this->outputMemories_.push_back(outputMemory);

  auto remoteInputMemories = setupRemoteMemories(comm, ctx->rank, inputMemory);
  auto remoteOutputMemories = setupRemoteMemories(comm, ctx->rank, outputMemory);
  ctx->registeredMemories.insert(ctx->registeredMemories.end(), remoteInputMemories.begin(), remoteInputMemories.end());
  ctx->registeredMemories.insert(ctx->registeredMemories.end(), remoteOutputMemories.begin(),
                                 remoteOutputMemories.end());
  std::vector<void*> remoteMemorieHandles;
  for (const auto& remoteMemory : ctx->registeredMemories) {
    remoteMemorieHandles.push_back(remoteMemory.data());
  }
  ctx->remoteMemoryHandles = detail::gpuCallocShared<void*>(remoteMemorieHandles.size());
  gpuMemcpy(ctx->remoteMemoryHandles.get(), remoteMemorieHandles.data(), remoteMemorieHandles.size(),
            cudaMemcpyHostToDevice);

  size_t recvBytes;
  CUdeviceptr recvBasePtr;
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&recvBasePtr, &recvBytes, (CUdeviceptr)output));

  // store local registered memories to ctx for lifetime management
  ctx->registeredMemories.push_back(inputMemory);
  ctx->registeredMemories.push_back(outputMemory);
  return ctx;
}

std::shared_ptr<Algorithm> AllreduceRsAgZeroCopy::build() {
  auto self = std::make_shared<AllreduceRsAgZeroCopy>();
  return std::make_shared<NativeAlgorithm>(
      "default_allreduce_rsag_zero_copy", "allreduce",
      [self](std::shared_ptr<mscclpp::Communicator> comm) { self->initialize(comm); },
      [self](const std::shared_ptr<void> ctx, const void* input, void* output, size_t inputSize,
             [[maybe_unused]] size_t outputSize, DataType dtype, ReduceOp op, cudaStream_t stream, int nBlocks,
             int nThreadsPerBlock, const std::unordered_map<std::string, uintptr_t>& extras) -> CommResult {
        return self->allreduceKernelFunc(ctx, input, output, inputSize, dtype, op, stream, nBlocks, nThreadsPerBlock,
                                         extras);
      },
      [self](std::shared_ptr<Communicator> comm, const void* input, void* output, size_t inputSize,
             [[maybe_unused]] size_t outputSize,
             DataType dtype) { return self->initAllreduceContext(comm, input, output, inputSize, dtype); },
      [self](const void* input, void* output, size_t inputSize, [[maybe_unused]] size_t outputSize, DataType dtype) {
        return self->generateAllreduceContextKey(input, output, inputSize, dtype);
      });
}
}  // namespace collective
}  // namespace mscclpp