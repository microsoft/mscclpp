// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "algorithms/allreduce/allreduce_next.hpp"
#include "algorithms/allreduce/common.hpp"
#include "algorithms/utils.hpp"
#include "debug.h"

namespace mscclpp {
namespace algorithm {

constexpr int NBLOCKS_FOR_COPY = 8;
constexpr int NBLOCKS_FOR_REDUCE = 24;
constexpr int PIPELINE_DEPTH = 4;
constexpr int REDUCE_COPY_RATIO = 8;

__device__ DeviceSemaphore semaphoreForSend[NBLOCKS_FOR_REDUCE];
__device__ DeviceSemaphore semaphoreForRecv;

constexpr uint32_t NBYTES_COPY_PER_THREAD = sizeof(int4);

// template <ReduceOp OpType, typename T>
// __global__ void __launch_bounds__(1024, 1)
//     allreduceHybrid(T* buff, T* scratch, T* resultBuff, DeviceHandle<BaseMemoryChannel>* memoryChannels,
//                     DeviceHandle<SwitchChannel>* switchChannels, void* remoteMemories, int rank, int nRanksPerNode,
//                     int worldSize, size_t nelems) {
//   int blockId = blockIdx.x;
//   int nPeers = nRanksPerNode - 1;
//   uint32_t nelemsPerRank = nelems / nRanksPerNode;
//   uint32_t nInt4 = nelemsPerRank * sizeof(T) / sizeof(int4);
//   uint32_t rest = nelemsPerRank * sizeof(T) - nInt4 * sizeof(int4);
//   int4* scratch4 = reinterpret_cast<int4*>((char*)scratch);
//   int4* resultBuff4 = reinterpret_cast<int4*>((char*)resultBuff);
//   int4* buff4 = reinterpret_cast<int4*>((char*)buff);

//   uint32_t nInt4PerIter = (blockDim.x * NBLOCKS_FOR_REDUCE);
//   uint32_t nIters = (nInt4 + nInt4PerIter - 1) / nInt4PerIter;
//   uint32_t totalScratchSize4 = PIPELINE_DEPTH * nInt4PerIter;

//   if (blockId < NBLOCKS_FOR_COPY) {
//     DeviceHandle<BaseMemoryChannel>* memoryChannelsLocal = memoryChannels + blockId * nPeers;
//     for (int iter = 0; iter < nIters; iter++) {
//       uint32_t offset4 = (iter * nInt4PerIter) % totalScratchSize4;
//       int threadIdInCopy = blockId * blockDim.x + threadIdx.x;
//       for (int rank = 0; rank < worldSize; rank++) {
//         for (uint32_t idx = threadIdInCopy; idx < nInt4PerIter; idx += blockDim.x * NBLOCKS_FOR_COPY) {
//           scratch4[rank * nInt4 + idx + offset4] = buff4[idx + offset4];
//         }
//       }
//       __syncthreads();
//       if (blockIdx.x == NBLOCKS_FOR_COPY && threadIdx.x < nPeers) {
//         memoryChannels[threadIdx.x].signal();
//         memoryChannels[threadIdx.x].wait();
//       }
//       __syncthreads();
//       if (threadIdx.x < REDUCE_COPY_RATIO) {
//         semaphoreForSend[blockId * REDUCE_COPY_RATIO + threadIdx.x].release();
//       }
//     }
//   } else if (blockId < NBLOCKS_FOR_COPY + NBLOCKS_FOR_REDUCE) {
//     uint32_t bidInReduce = blockId - NBLOCKS_FOR_COPY;
//     for (int iter = 0; iter < nIters; iter++) {
//       if (threadIdx.x == 0) {
//         semaphoreForSend[bidInReduce].acquire();
//       }
//       __syncthreads();
//       uint32_t offset4 = (iter * nInt4PerIter) % totalScratchSize4;
//       int threadIdInReduce = (blockId - NBLOCKS_FOR_COPY) * blockDim.x + threadIdx.x;
//       for (uint32_t idx = threadIdInReduce; idx < nInt4PerIter; idx += blockDim.x * NBLOCKS_FOR_REDUCE) {
//         int4 tmp = scratch4[idx + offset4];
//         int4 data = tmp;
//         for (int i = 0; i < nPeers; i++) {
//           int peerIdx = (rank + i + 1) % nRanksPerNode;
//           int index = peerIdx < rank ? peerIdx : peerIdx - 1;
//           int4 data = mscclpp::read<int4>(((void**)remoteMemories)[index], idx + offset4);
//           tmp = cal_vectors<T, OpType>(data, tmp);
//         }
//         resultBuff4[idx + offset4] = tmp;
//         for (int i = 0; i < nPeers; i++) {
//           int peerIdx = (rank + i + 1) % nRanksPerNode;
//           int index = peerIdx < rank ? peerIdx : peerIdx - 1;
//           mscclpp::write<int4>(((void**)remoteMemories)[index], idx + offset4, tmp);
//         }
//       }
//       __syncthreads();
//     }
//   }
//   // if (blockIdx.x == 0 && threadIdx.x < nPeers) {
//   //   memoryChannels[threadIdx.x].signal();
//   //   memoryChannels[threadIdx.x].wait();
//   // }
// }

template <ReduceOp OpType, typename T>
__global__ void __launch_bounds__(1024, 1)
    allreduceHybrid(T* buff, T* scratch, T* resultBuff, DeviceHandle<BaseMemoryChannel>* memoryChannels,
                    DeviceHandle<SwitchChannel>* switchChannels, void* remoteMemories, int rank, int nRanksPerNode,
                    int worldSize, size_t nelems) {
  int blockId = blockIdx.x;
  int nPeers = nRanksPerNode - 1;
  uint32_t nelemsPerRank = nelems / nRanksPerNode;
  uint32_t nInt4 = nelemsPerRank * sizeof(T) / sizeof(int4);
  uint32_t rest = nelemsPerRank * sizeof(T) - nInt4 * sizeof(int4);
  int4* scratch4 = reinterpret_cast<int4*>((char*)scratch);
  int4* resultBuff4 = reinterpret_cast<int4*>((char*)resultBuff);
  int4* buff4 = reinterpret_cast<int4*>((char*)buff);
  DeviceHandle<BaseMemoryChannel>* memoryChannelsLocal = memoryChannels + blockId * nPeers;

  uint32_t nInt4PerBlock = nInt4 / gridDim.x;
  uint32_t remainderForBlock = nInt4 % gridDim.x;
  uint32_t offset4 = blockId * nInt4PerBlock;
  if (blockId == gridDim.x - 1) {
    nInt4PerBlock += remainderForBlock;
  }
  uint32_t nInt4ForCopy = nInt4PerBlock * worldSize;
  for (uint32_t idx = threadIdx.x; idx < nInt4ForCopy; idx += blockDim.x) {
    int rankIdx = idx / nInt4PerBlock;
    uint32_t offsetIdx = rankIdx * nInt4 + offset4 + (idx % nInt4PerBlock);
    scratch4[offsetIdx] = buff4[offsetIdx];
  }
  __syncthreads();
  if (threadIdx.x < nPeers) {
    memoryChannelsLocal[threadIdx.x].signal();
    memoryChannelsLocal[threadIdx.x].wait();
  }
  __syncthreads();
  for (uint32_t idx = threadIdx.x; idx < nInt4PerBlock; idx += blockDim.x) {
    int4 tmp = scratch4[idx + offset4 + rank * nInt4];
    for (int i = 0; i < nPeers; i++) {
      int rankIdx = (rank + i + 1) % nRanksPerNode;
      int peerIdx = rankIdx < rank ? rankIdx : rankIdx - 1;
      int4 data = mscclpp::read<int4>(((void**)remoteMemories)[peerIdx], idx + offset4 + rank * nInt4);
      tmp = cal_vectors<T, OpType>(data, tmp);
    }
    resultBuff4[rank * nInt4 + idx + offset4] = tmp;
    for (int i = 0; i < nPeers; i++) {
      int rankIdx = (rank + i + 1) % nRanksPerNode;
      int peerIdx = rankIdx < rank ? rankIdx : rankIdx - 1;
      mscclpp::write<int4>(((void**)remoteMemories)[peerIdx], idx + offset4 + rank * nInt4, tmp);
    }
  }
  __syncthreads();
  if (threadIdx.x < nPeers) {
    memoryChannelsLocal[threadIdx.x].signal();
    memoryChannelsLocal[threadIdx.x].wait();
  }
  __syncthreads();
  for (uint32_t idx = threadIdx.x; idx < nInt4ForCopy; idx += blockDim.x) {
    int rankIdx = idx / nInt4PerBlock;
    if (rankIdx == rank) continue;
    uint32_t offsetIdx = rankIdx * nInt4 + offset4 + (idx % nInt4PerBlock);
    resultBuff4[offsetIdx] = scratch4[offsetIdx];
  }
}

template <ReduceOp OpType, typename T>
struct AllreduceHybridAdapter {
  static cudaError_t call(const void* input, void* scratch, void* output, void* memoryChannels, void* remoteMemories,
                          DeviceHandle<SwitchChannel>* switchChannel, DeviceHandle<SwitchChannel>*, size_t,
                          size_t, size_t, int rank, int nRanksPerNode, int worldSize,
                          size_t inputSize, cudaStream_t stream, void*, uint32_t, int nBlocks, int nThreadsPerBlock) {
    using ChannelType = DeviceHandle<BaseMemoryChannel>;
    size_t nelems = inputSize / sizeof(T);
    if (nBlocks == 0 || nThreadsPerBlock == 0) {
      nThreadsPerBlock = 1024;
      nBlocks = NBLOCKS_FOR_REDUCE;
    }
    allreduceHybrid<OpType, T><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
        (T*)input, (T*)scratch, (T*)output, (ChannelType*)memoryChannels, switchChannel, remoteMemories, rank,
        nRanksPerNode, worldSize, nelems);
    return cudaGetLastError();
  }
};

void AllreduceNext::initialize(std::shared_ptr<Communicator> comm) {
  this->conns_ = setupConnections(comm);
  nChannelsPerConnection_ = 64;
  comm_ = comm;
  // setup semaphores
  this->scratchSemaphores_ = setupMemorySemaphores(comm, this->conns_, nChannelsPerConnection_);
  RegisteredMemory localMemory = comm->registerMemory(scratchBuffer_, scratchBufferSize_, Transport::CudaIpc);
  this->remoteScratchMemories_ = setupRemoteMemories(comm, comm->bootstrap()->getRank(), localMemory);
  localScratchMemory_ = std::move(localMemory);

  this->baseChannels_ = setupBaseMemoryChannels(this->conns_, this->scratchSemaphores_, nChannelsPerConnection_);
  this->baseMemoryChannelHandles_ = setupBaseMemoryChannelDeviceHandles(baseChannels_);
  for (const auto& remoteMemory : this->remoteScratchMemories_) {
    this->remoteMemorieHandles_.push_back(remoteMemory.data());
  }
}

CommResult AllreduceNext::allreduceKernelFunc(const std::shared_ptr<AlgorithmCtx> ctx, const void* input, void* output,
                                              size_t inputSize, DataType dtype, ReduceOp op, cudaStream_t stream,
                                              int nBlocks, int nThreadsPerBlock,
                                              const std::unordered_map<std::string, uintptr_t>&) {
  AllreduceFunc allreduce = dispatch<AllreduceHybridAdapter>(op, dtype);
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
                this->remoteMemorieHandles_.data(), nullptr, nullptr, 0, 0, 0, ctx->rank, ctx->nRanksPerNode, ctx->workSize,
                inputSize, stream, nullptr, 0, numBlocksAndThreads.first, numBlocksAndThreads.second);
  if (error != cudaSuccess) {
    WARN("AllreduceAllconnect failed with error: %s", cudaGetErrorString(error));
    return CommResult::commUnhandledCudaError;
  }
  return CommResult::commSuccess;
}

AlgorithmCtxKey AllreduceNext::generateAllreduceContextKey(const void*, void*, size_t, DataType) {
  return AlgorithmCtxKey{nullptr,nullptr, 0, 0, 0};
}

std::shared_ptr<AlgorithmCtx> AllreduceNext::initAllreduceContext(std::shared_ptr<Communicator> comm, const void*,
                                                                  void*, size_t, DataType) {
  auto ctx = std::make_shared<AlgorithmCtx>();
  ctx->rank = comm->bootstrap()->getRank();
  ctx->workSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerNode = comm->bootstrap()->getNranksPerNode();

  ctx->memorySemaphores = this->scratchSemaphores_;
  ctx->registeredMemories = this->remoteScratchMemories_;
  return ctx;
}

std::shared_ptr<Algorithm> AllreduceNext::build() {
  auto self = std::make_shared<AllreduceNext>((uintptr_t)scratchBuffer_, scratchBufferSize_);
  return std::make_shared<NativeAlgorithm>(
      "default_allreduce_next", "allreduce",
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