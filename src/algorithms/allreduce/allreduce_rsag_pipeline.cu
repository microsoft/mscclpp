// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "algorithms/allreduce/allreduce_rsag_pipeline.hpp"
#include "algorithms/allreduce/common.hpp"
#include "algorithms/utils.hpp"
#include "debug.h"

namespace mscclpp {
namespace algorithm {

constexpr int NBLOCKS_FOR_PUT = 32;
constexpr int NBLOCKS_FOR_RECV = 32;
constexpr int NBLOCKS_FOR_REDUCE = 64;
constexpr int PIPELINE_DEPTH = 4;
constexpr int REDUCE_COPY_RATIO = NBLOCKS_FOR_REDUCE / NBLOCKS_FOR_PUT;
__device__ DeviceSemaphore semaphoreForSend[NBLOCKS_FOR_REDUCE];
__device__ DeviceSemaphore semaphoreForRecv[NBLOCKS_FOR_REDUCE];

template <ReduceOp OpType, typename T>
__global__ void __launch_bounds__(1024, 1)
    allreduceRsAgPipeline(T* buff, T* scratch, T* resultBuff, DeviceHandle<BaseMemoryChannel>* memoryChannels,
                          DeviceHandle<SwitchChannel>* switchChannels, void* remoteMemories, int rank,
                          int nRanksPerNode, int worldSize, size_t nelems) {
  int bid = blockIdx.x;
  const uint32_t nStepsPerIter = 8;
  uint32_t nInt4 = nelems * sizeof(T) / sizeof(int4);
  uint32_t nInt4PerRank = (nInt4 + worldSize - 1) / worldSize;
  // 32 K * sizeof(int4) = 32K * 16B = 512KB
  uint32_t nInt4PerIter = NBLOCKS_FOR_REDUCE * blockDim.x * nStepsPerIter;
  uint32_t nIters = (nInt4PerRank + nInt4PerIter - 1) / nInt4PerIter;
  uint32_t nPeers = nRanksPerNode - 1;
  int4* buff4 = reinterpret_cast<int4*>((char*)buff);
  int4* scratch4 = reinterpret_cast<int4*>((char*)scratch);
  int4* result4 = reinterpret_cast<int4*>((char*)resultBuff);
  if (bid < NBLOCKS_FOR_PUT) {
    DeviceHandle<BaseMemoryChannel>* localMemoryChannels = memoryChannels + bid * nPeers;
    for (int iter = 0; iter < nIters; iter++) {
      int threadIdInPut = bid * blockDim.x + threadIdx.x;
      for (int peer = 0; peer < nPeers; peer++) {
        int remoteRankId = (rank + peer + 1) % nRanksPerNode;
        int peerId = remoteRankId < rank ? remoteRankId : remoteRankId - 1;
        // Read chunk[remoteRankId] from local buff, write to peer's scratch[rank] (sender's slot)
        uint32_t srcOffset = iter * nInt4PerIter * worldSize + remoteRankId * nInt4PerIter;
        uint32_t dstOffset = iter * nInt4PerIter * worldSize + rank * nInt4PerIter;
        int4 tmp[nStepsPerIter * REDUCE_COPY_RATIO];
#pragma unroll
        for (int step = 0; step < nStepsPerIter * REDUCE_COPY_RATIO; step++) {
          uint32_t offset = srcOffset + threadIdInPut + step * blockDim.x * NBLOCKS_FOR_PUT;
          tmp[step] = buff4[offset];
        }
#pragma unroll
        for (int step = 0; step < nStepsPerIter * REDUCE_COPY_RATIO; step++) {
          uint32_t offset = dstOffset + threadIdInPut + step * blockDim.x * NBLOCKS_FOR_PUT;
          mscclpp::write<int4>(((void**)remoteMemories)[peerId], offset, tmp[step]);
        }
      }
      __syncthreads();
      if (threadIdx.x < REDUCE_COPY_RATIO) {
        semaphoreForSend[bid * REDUCE_COPY_RATIO + threadIdx.x].release();
      }
    }
  } else if (bid < NBLOCKS_FOR_PUT + NBLOCKS_FOR_REDUCE) {
    uint32_t bidInReduce = bid - NBLOCKS_FOR_PUT;
    DeviceHandle<BaseMemoryChannel>* localMemoryChannels = memoryChannels + bidInReduce * nPeers;
    // Map REDUCE blocks to PUT blocks: REDUCE blocks 0,1 handle PUT block 0's data
    uint32_t putBlockId = bidInReduce / REDUCE_COPY_RATIO;
    uint32_t subBlockId = bidInReduce % REDUCE_COPY_RATIO;
    for (int iter = 0; iter < nIters; iter++) {
      if (threadIdx.x == 0) {
        semaphoreForSend[bidInReduce].acquire();
      }
      uint32_t baseOffset = nInt4PerIter * worldSize * iter;
      // Use same thread mapping as PUT: putBlockId * blockDim.x + threadIdx.x
      uint32_t threadIdInPut = putBlockId * blockDim.x + threadIdx.x;
      __syncthreads();
      if (threadIdx.x < nPeers) {
        localMemoryChannels[threadIdx.x].signal();
        localMemoryChannels[threadIdx.x].wait();
      }
      __syncthreads();
#pragma unroll nStepsPerIter
      for (int step = 0; step < nStepsPerIter; step++) {
        // Map to PUT's step pattern: each REDUCE block handles nStepsPerIter steps
        // subBlockId determines which subset of the REDUCE_COPY_RATIO * nStepsPerIter steps
        uint32_t putStep = subBlockId * nStepsPerIter + step;
        uint32_t myChunkOffset =
            baseOffset + rank * nInt4PerIter + threadIdInPut + putStep * blockDim.x * NBLOCKS_FOR_PUT;
        int4 tmp = buff4[myChunkOffset];
        // Add data from each peer's slot in scratch (peer sent their chunk[rank] to our scratch[peer])
        for (int peer = 0; peer < nPeers; peer++) {
          int remoteRankId = (rank + peer + 1) % nRanksPerNode;
          uint32_t peerSlotOffset =
              baseOffset + remoteRankId * nInt4PerIter + threadIdInPut + putStep * blockDim.x * NBLOCKS_FOR_PUT;
          int4 data = scratch4[peerSlotOffset];
          tmp = cal_vectors<T, OpType>(data, tmp);
        }
        result4[myChunkOffset] = tmp;
        // Broadcast reduced result to all peers' scratch at rank's slot
        for (int i = 0; i < nPeers; i++) {
          int peerIdx = (rank + i + 1) % nRanksPerNode;
          int index = peerIdx < rank ? peerIdx : peerIdx - 1;
          mscclpp::write<int4>(((void**)remoteMemories)[index], myChunkOffset, tmp);
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        semaphoreForRecv[bidInReduce].release();
      }
    }
  } else if (bid < NBLOCKS_FOR_PUT + NBLOCKS_FOR_REDUCE + NBLOCKS_FOR_RECV) {
    uint32_t bidInRecv = bid - NBLOCKS_FOR_PUT - NBLOCKS_FOR_REDUCE;
    DeviceHandle<BaseMemoryChannel>* localMemoryChannels = memoryChannels + (NBLOCKS_FOR_REDUCE + bidInRecv) * nPeers;
    for (int iter = 0; iter < nIters; iter++) {
      if (threadIdx.x < REDUCE_COPY_RATIO) {
        semaphoreForRecv[bidInRecv * REDUCE_COPY_RATIO + threadIdx.x].acquire();
      }
      uint32_t baseOffset = nInt4PerIter * worldSize * iter;
      int threadIdInRecv = bidInRecv * blockDim.x + threadIdx.x;
      __syncthreads();
      if (threadIdx.x < nPeers) {
        localMemoryChannels[threadIdx.x].signal();
        localMemoryChannels[threadIdx.x].wait();
      }
      __syncthreads();
      // Copy other ranks' reduced chunks from scratch to result
      for (int peer = 0; peer < nPeers; peer++) {
        int remoteRankId = (rank + peer + 1) % nRanksPerNode;
        for (uint32_t step = 0; step < nStepsPerIter * REDUCE_COPY_RATIO; step++) {
          uint32_t offset =
              baseOffset + remoteRankId * nInt4PerIter + threadIdInRecv + step * blockDim.x * NBLOCKS_FOR_RECV;
          result4[offset] = scratch4[offset];
        }
      }
    }
  }
}

template <ReduceOp OpType, typename T>
struct AllreduceRsAgPipelineAdapter {
  static cudaError_t call(const void* input, void* scratch, void* output, void* memoryChannels, void* remoteMemories,
                          DeviceHandle<SwitchChannel>* switchChannel, DeviceHandle<SwitchChannel>*, size_t, size_t,
                          size_t, int rank, int nRanksPerNode, int worldSize, size_t inputSize, cudaStream_t stream,
                          void*, uint32_t, int nBlocks, int nThreadsPerBlock) {
    using ChannelType = DeviceHandle<BaseMemoryChannel>;
    size_t nelems = inputSize / sizeof(T);
    if (nBlocks == 0 || nThreadsPerBlock == 0) {
      nThreadsPerBlock = 1024;
      nBlocks = NBLOCKS_FOR_PUT + NBLOCKS_FOR_REDUCE + NBLOCKS_FOR_RECV;
    }
    allreduceRsAgPipeline<OpType, T><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
        (T*)input, (T*)scratch, (T*)output, (ChannelType*)memoryChannels, switchChannel, remoteMemories, rank,
        nRanksPerNode, worldSize, nelems);
    return cudaGetLastError();
  }
};

void AllreduceRsAgPipeline::initialize(std::shared_ptr<Communicator> comm) {
  this->conns_ = setupConnections(comm);
  nChannelsPerConnection_ = 96;
  comm_ = comm;
  // setup semaphores
  this->scratchSemaphores_ = setupMemorySemaphores(comm, this->conns_, nChannelsPerConnection_);
  RegisteredMemory localMemory = comm->registerMemory(scratchBuffer_, scratchBufferSize_, Transport::CudaIpc);
  this->remoteScratchMemories_ = setupRemoteMemories(comm, comm->bootstrap()->getRank(), localMemory);
  localScratchMemory_ = std::move(localMemory);

  this->baseChannels_ = setupBaseMemoryChannels(this->conns_, this->scratchSemaphores_, nChannelsPerConnection_);
  this->baseMemoryChannelHandles_ = setupBaseMemoryChannelDeviceHandles(baseChannels_);
  std::vector<void*> remoteMemorieHandles;
  for (const auto& remoteMemory : this->remoteScratchMemories_) {
    remoteMemorieHandles.push_back(remoteMemory.data());
  }
  this->remoteMemorieHandles_ = detail::gpuCallocShared<void*>(remoteMemorieHandles.size());
  gpuMemcpy(this->remoteMemorieHandles_.get(), remoteMemorieHandles.data(), remoteMemorieHandles.size(),
            cudaMemcpyHostToDevice);
}

CommResult AllreduceRsAgPipeline::allreduceKernelFunc(const std::shared_ptr<AlgorithmCtx> ctx, const void* input,
                                                      void* output, size_t inputSize, DataType dtype, ReduceOp op,
                                                      cudaStream_t stream, int nBlocks, int nThreadsPerBlock,
                                                      const std::unordered_map<std::string, uintptr_t>&) {
  AllreduceFunc allreduce = dispatch<AllreduceRsAgPipelineAdapter>(op, dtype);
  if (!allreduce) {
    WARN("Unsupported operation or data type for allreduce: op=%d, dtype=%d", static_cast<int>(op),
         static_cast<int>(dtype));
    return CommResult::commInvalidArgument;
  }
  if (inputSize > this->scratchBufferSize_) {
    WARN("Input size %zu exceeds scratch buffer size %zu", inputSize, this->scratchBufferSize_);
    return CommResult::commInvalidArgument;
  }
  std::pair<int, int> numBlocksAndThreads = {nBlocks, nThreadsPerBlock};
  cudaError_t error =
      allreduce(input, this->scratchBuffer_, output, this->baseMemoryChannelHandles_.get(),
                this->remoteMemorieHandles_.get(), nullptr, nullptr, 0, 0, 0, ctx->rank, ctx->nRanksPerNode,
                ctx->workSize, inputSize, stream, nullptr, 0, numBlocksAndThreads.first, numBlocksAndThreads.second);
  if (error != cudaSuccess) {
    WARN("AllreduceAllconnect failed with error: %s", cudaGetErrorString(error));
    return CommResult::commUnhandledCudaError;
  }
  return CommResult::commSuccess;
}

AlgorithmCtxKey AllreduceRsAgPipeline::generateAllreduceContextKey(const void*, void*, size_t, DataType) {
  return AlgorithmCtxKey{nullptr, nullptr, 0, 0, 0};
}

std::shared_ptr<AlgorithmCtx> AllreduceRsAgPipeline::initAllreduceContext(std::shared_ptr<Communicator> comm,
                                                                          const void*, void*, size_t, DataType) {
  auto ctx = std::make_shared<AlgorithmCtx>();
  ctx->rank = comm->bootstrap()->getRank();
  ctx->workSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();

  ctx->memorySemaphores = this->scratchSemaphores_;
  ctx->registeredMemories = this->remoteScratchMemories_;
  return ctx;
}

std::shared_ptr<Algorithm> AllreduceRsAgPipeline::build() {
  auto self = std::make_shared<AllreduceRsAgPipeline>((uintptr_t)scratchBuffer_, scratchBufferSize_);
  return std::make_shared<NativeAlgorithm>(
      "default_allreduce_rsag_pipeline", "allreduce",
      [self](std::shared_ptr<mscclpp::Communicator> comm) { self->initialize(comm); },
      [self](const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, const void* input, void* output, size_t inputSize,
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
}  // namespace algorithm
}  // namespace mscclpp