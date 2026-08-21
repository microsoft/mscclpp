// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "allgather/allgather_fullmesh.hpp"
#include "collective_utils.hpp"
#include "debug.h"

namespace mscclpp {
namespace collective {

namespace {
constexpr int kMaxBlocks = 56;
constexpr int kMaxThreadsPerBlock = 1024;
}  // namespace

__device__ __forceinline__ size_t outputIndex(size_t localIndex, int sourceRank, size_t nInt4,
                                              int worldSize, size_t rowWidthInt4, int layoutMode) {
  if (layoutMode == 1) {
    const size_t row = localIndex / rowWidthInt4;
    const size_t column = localIndex % rowWidthInt4;
    return row * rowWidthInt4 * worldSize + sourceRank * rowWidthInt4 + column;
  }
  return nInt4 * sourceRank + localIndex;
}

template <bool IsOutOfPlace>
__global__ void __launch_bounds__(1024, 1)
    allgatherFullmesh(void* buff, void* scratch, void* resultBuff, DeviceHandle<MemoryChannel>* memoryChannels,
                      int rank, int nRanksPerIpcDomain, int worldSize, size_t nelems, size_t rowCount,
                      size_t rowWidthInt4, int layoutMode, unsigned long long* executeCounter) {
  const int nPeer = nRanksPerIpcDomain - 1;
  const size_t chanOffset = nPeer * blockIdx.x;
  if (executeCounter != nullptr && blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(executeCounter, 1ULL);
  if (layoutMode == 1 && (rowCount == 0 || rowWidthInt4 * rowCount != nelems * sizeof(int) / sizeof(int4))) return;
  // assume (nelems * sizeof(T)) is divisible by 16
  const size_t nInt4 = nelems * sizeof(int) / sizeof(int4);
  auto memoryChans = memoryChannels + chanOffset;

  int4* buff4 = reinterpret_cast<int4*>(buff);
  int4* scratch4 = reinterpret_cast<int4*>(scratch);
  int4* resultBuff4 = reinterpret_cast<int4*>(resultBuff);

  const size_t unitNInt4 = blockDim.x * gridDim.x;  // The number of int4 transfers at once
  const size_t nInt4PerChunk = unitNInt4 * 4;       // 4 instructions per thread to make it more efficient
  const size_t nItrs = nInt4 / nInt4PerChunk;
  const size_t restNInt4 = nInt4 % nInt4PerChunk;
  const size_t scratchChunkRankOffset = nInt4PerChunk * rank;

  __shared__ DeviceHandle<MemoryChannel> channels[MAX_IPC_DOMAIN_NRANKS - 1];
  const int lid = threadIdx.x % WARP_SIZE;
  // Peer count may exceed WARP_SIZE on MNNVL.
  for (int i = lid; i < nPeer; i += WARP_SIZE) {
    channels[i] = memoryChans[i];
  }
  __syncwarp();
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // we can use double buffering to hide synchronization overhead
  for (size_t itr = 0; itr < nItrs; itr++) {
    if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
      channels[threadIdx.x].signal();
      channels[threadIdx.x].wait();
    }
    __syncthreads();
    // Starts allgather
    for (size_t idx = tid; idx < nInt4PerChunk; idx += blockDim.x * gridDim.x) {
      int4 val = buff4[itr * nInt4PerChunk + idx];
      for (int i = 0; i < nPeer; i++) {
        const int peerIdx = (i + rank) % nPeer;
        channels[peerIdx].write(idx + scratchChunkRankOffset, val);
      }
      if constexpr (IsOutOfPlace) {
        const size_t localIndex = idx + itr * nInt4PerChunk;
        resultBuff4[outputIndex(localIndex, rank, nInt4, worldSize, rowWidthInt4, layoutMode)] = val;
      }
    }
    // Ensure that all writes of this block have been issued before issuing the signal
    __syncthreads();
    if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
      channels[threadIdx.x].signal();
      channels[threadIdx.x].wait();
    }
    __syncthreads();
    for (int peerIdx = 0; peerIdx < nPeer; peerIdx++) {
      const int remoteRank = (peerIdx < rank) ? peerIdx : peerIdx + 1;
      for (size_t idx = tid; idx < nInt4PerChunk; idx += blockDim.x * gridDim.x) {
        int4 val = scratch4[nInt4PerChunk * remoteRank + idx];
        const size_t localIndex = idx + itr * nInt4PerChunk;
        resultBuff4[outputIndex(localIndex, remoteRank, nInt4, worldSize, rowWidthInt4, layoutMode)] = val;
      }
    }
  }

  if (restNInt4 > 0) {
    if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
      channels[threadIdx.x].signal();
      channels[threadIdx.x].wait();
    }
    __syncthreads();
    for (size_t idx = tid; idx < restNInt4; idx += blockDim.x * gridDim.x) {
      int4 val = buff4[nItrs * nInt4PerChunk + idx];
      for (int i = 0; i < nPeer; i++) {
        const int peerIdx = (i + rank) % nPeer;
        channels[peerIdx].write(idx + scratchChunkRankOffset, val);
      }
      if constexpr (IsOutOfPlace) {
        const size_t localIndex = idx + nItrs * nInt4PerChunk;
        resultBuff4[outputIndex(localIndex, rank, nInt4, worldSize, rowWidthInt4, layoutMode)] = val;
      }
    }
    // Ensure that all writes of this block have been issued before issuing the signal
    __syncthreads();
    if (threadIdx.x < static_cast<uint32_t>(nPeer)) {
      channels[threadIdx.x].signal();
      channels[threadIdx.x].wait();
    }
    __syncthreads();
    for (int peerIdx = 0; peerIdx < nPeer; peerIdx++) {
      const int remoteRank = (peerIdx < rank) ? peerIdx : peerIdx + 1;
      for (size_t idx = tid; idx < restNInt4; idx += blockDim.x * gridDim.x) {
        int4 val = scratch4[nInt4PerChunk * remoteRank + idx];
        const size_t localIndex = idx + nItrs * nInt4PerChunk;
        resultBuff4[outputIndex(localIndex, remoteRank, nInt4, worldSize, rowWidthInt4, layoutMode)] = val;
      }
    }
  }
}

void AllgatherFullmesh::initialize(std::shared_ptr<mscclpp::Communicator> comm) {
  this->conns_ = setupConnections(comm);
}

CommResult AllgatherFullmesh::allgatherKernelFunc(const std::shared_ptr<void> ctx_void, const void* input, void* output,
                                                  size_t inputSize, cudaStream_t stream, int nBlocks,
                                                  int nThreadsPerBlock,
                                                  const std::unordered_map<std::string, uintptr_t>& extras) {
  auto ctx = std::static_pointer_cast<AlgorithmCtx>(ctx_void);
  int rank = ctx->rank;
  const size_t nElem = inputSize / sizeof(int);
  std::pair<int, int> numBlocksAndThreads = {nBlocks, nThreadsPerBlock};
  if (numBlocksAndThreads.first == 0 || numBlocksAndThreads.second == 0) {
    numBlocksAndThreads = {kMaxBlocks, kMaxThreadsPerBlock};
  }
  if (numBlocksAndThreads.first > kMaxBlocks || numBlocksAndThreads.second > kMaxThreadsPerBlock) {
    WARN(
        "AllgatherFullmesh: number of blocks must be no more than %d and threads per block must be no more than %d; "
        "got nBlocks=%d, nThreadsPerBlock=%d",
        kMaxBlocks, kMaxThreadsPerBlock, numBlocksAndThreads.first, numBlocksAndThreads.second);
    return CommResult::CommInvalidArgument;
  }
  if (numBlocksAndThreads.second % WARP_SIZE != 0) {
    WARN("AllgatherFullmesh: threads per block must be a multiple of warp size %d", WARP_SIZE);
    return CommResult::CommInvalidArgument;
  }
  auto getExtra = [&extras](const char* name, uintptr_t fallback) {
    auto it = extras.find(name);
    return it == extras.end() ? fallback : it->second;
  };
  const size_t rowCount = static_cast<size_t>(getExtra("rowCount", 1));
  const size_t localRowBytes = static_cast<size_t>(getExtra("localRowBytes", inputSize));
  const int layoutMode = static_cast<int>(getExtra("layoutMode", 0));
  auto* executeCounter = reinterpret_cast<unsigned long long*>(getExtra("executeCounter", 0));
  if (rowCount == 0 || localRowBytes == 0 || rowCount * localRowBytes != inputSize || localRowBytes % 16 != 0) {
    return CommResult::CommInvalidArgument;
  }
  const size_t rowWidthInt4 = localRowBytes / sizeof(int4);
  if ((char*)input == (char*)output + rank * inputSize && layoutMode == 0) {
    allgatherFullmesh<false><<<numBlocksAndThreads.first, numBlocksAndThreads.second, 0, stream>>>(
        (void*)input, this->scratchBuffer_, (void*)output, ctx->memoryChannelDeviceHandles.get(), rank,
        ctx->nRanksPerIpcDomain, ctx->worldSize, nElem, rowCount, rowWidthInt4, layoutMode, executeCounter);
  } else {
    allgatherFullmesh<true><<<numBlocksAndThreads.first, numBlocksAndThreads.second, 0, stream>>>(
        (void*)input, this->scratchBuffer_, (void*)output, ctx->memoryChannelDeviceHandles.get(), rank,
        ctx->nRanksPerIpcDomain, ctx->worldSize, nElem, rowCount, rowWidthInt4, layoutMode, executeCounter);
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    WARN("AllgatherFullmesh failed with error %d", err);
    return mscclpp::CommResult::CommInternalError;
  }
  return mscclpp::CommResult::CommSuccess;
}

std::shared_ptr<void> AllgatherFullmesh::initAllgatherContext(std::shared_ptr<Communicator> comm, const void* input,
                                                              void*, size_t inputSize, DataType) {
  auto ctx = std::make_shared<AlgorithmCtx>();
  ctx->rank = comm->bootstrap()->getRank();
  ctx->worldSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerIpcDomain = comm->bootstrap()->getNranksPerIpcDomain();

  // setup semaphores
  ctx->memorySemaphores = setupMemorySemaphores(comm, this->conns_, kMaxBlocks);

  // register the memory for the broadcast operation
  RegisteredMemory localMemory = comm->registerMemory((void*)input, inputSize, Transport::CudaIpc);
  RegisteredMemory scratchMemory = comm->registerMemory(this->scratchBuffer_, scratchBufferSize_, Transport::CudaIpc);
  std::vector<RegisteredMemory> remoteMemories = setupRemoteMemories(comm, ctx->rank, scratchMemory);

  // setup channels
  ctx->memoryChannels =
      setupMemoryChannels(this->conns_, ctx->memorySemaphores, remoteMemories, localMemory, kMaxBlocks);
  ctx->memoryChannelDeviceHandles = setupMemoryChannelDeviceHandles(ctx->memoryChannels);

  // keep registered memories reference
  ctx->registeredMemories = std::move(remoteMemories);
  ctx->registeredMemories.push_back(localMemory);
  ctx->registeredMemories.push_back(scratchMemory);

  return ctx;
}

AlgorithmCtxKey AllgatherFullmesh::generateAllgatherContextKey(const void* input, void* output, size_t inputSize,
                                                                size_t outputSize, DataType dtype,
                                                                bool symmetricMemory) {
  // Buffer registration and MemoryChannel handles are pointer- and size-specific.
  // NativeAlgorithm augments this key with communicator/device identity, element
  // count, dtype, and symmetric-memory mode before cache lookup.
  return AlgorithmCtxKey{const_cast<void*>(input), output, inputSize, outputSize,
                         symmetricMemory ? 1 : 0};
}

std::shared_ptr<Algorithm> AllgatherFullmesh::build() {
  auto self = std::make_shared<AllgatherFullmesh>(reinterpret_cast<uintptr_t>(scratchBuffer_), scratchBufferSize_);
  return std::make_shared<mscclpp::NativeAlgorithm>(
      "default_allgather_fullmesh", "allgather",
      [self](std::shared_ptr<mscclpp::Communicator> comm) { self->initialize(comm); },
      [self](const std::shared_ptr<void> ctx, const void* input, void* output, size_t inputSize,
             [[maybe_unused]] size_t outputSize, [[maybe_unused]] DataType dtype, [[maybe_unused]] ReduceOp op,
             cudaStream_t stream, int nBlocks, int nThreadsPerBlock,
             const std::unordered_map<std::string, uintptr_t>& extras,
             [[maybe_unused]] DataType accumDtype) -> CommResult {
        return self->allgatherKernelFunc(ctx, input, output, inputSize, stream, nBlocks, nThreadsPerBlock, extras);
      },
      [self](std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output, size_t inputSize,
             [[maybe_unused]] size_t outputSize,
             DataType dtype) { return self->initAllgatherContext(comm, input, output, inputSize, dtype); },
      [self](const void* input, void* output, size_t inputSize, [[maybe_unused]] size_t outputSize, DataType dtype,
             bool symmetricMemory) {
        return self->generateAllgatherContextKey(input, output, inputSize, outputSize, dtype, symmetricMemory);
      });
}
}  // namespace collective
}  // namespace mscclpp
