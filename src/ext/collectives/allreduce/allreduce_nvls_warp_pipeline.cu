// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include <mscclpp/algorithm.hpp>
#include <mscclpp/errors.hpp>

#include "allreduce/allreduce_nvls_warp_pipeline.hpp"
#include "allreduce/common.hpp"
#include "collective_utils.hpp"
#include "debug.h"

namespace mscclpp {
namespace collective {

template <typename T>
__global__ void __launch_bounds__(1024, 1)
    allreduceNvlsWarpPipeline([[maybe_unused]] const void* src, [[maybe_unused]] void* scratch,
                              [[maybe_unused]] void* dst,
                              [[maybe_unused]] DeviceHandle<BaseMemoryChannel>* memoryChannels,
                              [[maybe_unused]] DeviceHandle<SwitchChannel>* multicast, [[maybe_unused]] size_t size,
                              [[maybe_unused]] size_t scratchBufferSize, [[maybe_unused]] int rank,
                              [[maybe_unused]] int ipcDomainNranks) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  constexpr int alignment = 16;
  int nPeers = ipcDomainNranks - 1;
  int nBlocks = gridDim.x;
  int nBlocksPerNvlsConn = nBlocks / NUM_NVLS_CONNECTION;
  int bid = blockIdx.x;
  size_t sizePerRank = size / ipcDomainNranks;
  size_t scratchSizePerRank = scratchBufferSize / ipcDomainNranks;
  const size_t maxSizePerBlock = ((sizePerRank + nBlocks - 1) / nBlocks + alignment - 1) / alignment * alignment;
  size_t start = bid * maxSizePerBlock;
  size_t end = min(start + maxSizePerBlock, sizePerRank);
  size_t sizePerBlock = end - start;
  auto* multicastPtr = multicast + bid / nBlocksPerNvlsConn;
  size_t copyPerIter = 1024 * 16;
  if (sizePerBlock >= 1024 * 64) {
    copyPerIter = 1024 * 32;
  }
  size_t scratchSizePerBlock = (scratchSizePerRank / nBlocks) / copyPerIter * copyPerIter;
  size_t blockScratchOffset = scratchSizePerBlock * bid + scratchSizePerRank * rank;
  constexpr int NCOPY_WARPS = 14;
  constexpr int NREDUCE_WARPS = 4;
  constexpr int NRECV_COPY_WARPS = 14;
  constexpr int endCopyWid = NCOPY_WARPS;
  constexpr int startRecvCopyWid = NCOPY_WARPS;
  constexpr int endRecvCopyWid = NCOPY_WARPS + NRECV_COPY_WARPS;
  constexpr int endReduceWid = NCOPY_WARPS + NREDUCE_WARPS + NRECV_COPY_WARPS;
  const int warpId = threadIdx.x / WARP_SIZE;
  size_t nIter = sizePerBlock / copyPerIter;
  size_t lastIterSize = copyPerIter;
  if (sizePerBlock % copyPerIter != 0) {
    nIter += 1;
    lastIterSize = sizePerBlock % copyPerIter;
  }

  const size_t chanOffset = (ipcDomainNranks - 1) * blockIdx.x * 2;
  auto memoryChans = memoryChannels + chanOffset;
  __shared__ DeviceHandle<BaseMemoryChannel> channels[(MAX_IPC_DOMAIN_NRANKS - 1) * 2];
  const int lid = threadIdx.x % WARP_SIZE;
  // Peer count may exceed WARP_SIZE on MNNVL.
  for (int i = lid; i < nPeers * 2; i += WARP_SIZE) {
    channels[i] = memoryChans[i];
  }
  __syncwarp();
  for (int it = 0; it < nIter; it++) {
    const size_t iterSize = (it == nIter - 1) ? lastIterSize : copyPerIter;
    if (warpId < endCopyWid) {
      int tidInCopy = threadIdx.x;
      for (int i = 0; i < ipcDomainNranks; i++) {
        size_t offset = i * sizePerRank + maxSizePerBlock * bid + it * copyPerIter;
        size_t offsetScratch =
            i * scratchSizePerRank + scratchSizePerBlock * bid + (it * copyPerIter) % scratchSizePerBlock;
        char* srcData = (char*)src + offset;
        char* dstData = (char*)scratch + offsetScratch;
        mscclpp::copy(dstData, srcData, iterSize, tidInCopy, NCOPY_WARPS * WARP_SIZE);
      }
      asm volatile("bar.sync %0, %1;" ::"r"(0), "r"(NCOPY_WARPS * WARP_SIZE) : "memory");
      if (tidInCopy < nPeers) {
        channels[tidInCopy].signal();
        channels[tidInCopy].wait();
      }
      asm volatile("bar.sync %0, %1;" ::"r"(1), "r"((NCOPY_WARPS + NREDUCE_WARPS) * WARP_SIZE) : "memory");
    }
    if (warpId >= endRecvCopyWid && warpId < endReduceWid) {
      int tidInReduce = threadIdx.x - endRecvCopyWid * WARP_SIZE;
      asm volatile("bar.sync %0, %1;" ::"r"(1), "r"((NCOPY_WARPS + NREDUCE_WARPS) * WARP_SIZE) : "memory");
      T* mcBuff = (T*)multicastPtr->mcPtr;
      size_t offset = blockScratchOffset + (it * copyPerIter) % scratchSizePerBlock;
      handleMultiLoadReduceStore(mcBuff, mcBuff, offset, offset, iterSize, tidInReduce, NREDUCE_WARPS * WARP_SIZE);
      asm volatile("bar.sync %0, %1;" ::"r"(2), "r"((NRECV_COPY_WARPS + NREDUCE_WARPS) * WARP_SIZE) : "memory");
    }
    if (warpId >= startRecvCopyWid && warpId < endRecvCopyWid) {
      int tidInRecvCopy = threadIdx.x - startRecvCopyWid * WARP_SIZE;
      asm volatile("bar.sync %0, %1;" ::"r"(2), "r"((NRECV_COPY_WARPS + NREDUCE_WARPS) * WARP_SIZE) : "memory");
      if (tidInRecvCopy < nPeers) {
        channels[tidInRecvCopy + nPeers].signal();
        channels[tidInRecvCopy + nPeers].wait();
      }
      asm volatile("bar.sync %0, %1;" ::"r"(3), "r"((NRECV_COPY_WARPS)*WARP_SIZE) : "memory");
      for (int i = 0; i < ipcDomainNranks; i++) {
        size_t offset = i * sizePerRank + maxSizePerBlock * bid + it * copyPerIter;
        size_t offsetScratch =
            i * scratchSizePerRank + scratchSizePerBlock * bid + (it * copyPerIter) % scratchSizePerBlock;
        char* srcData = (char*)scratch + offsetScratch;
        char* dstData = (char*)dst + offset;
        mscclpp::copy(dstData, srcData, iterSize, tidInRecvCopy, NRECV_COPY_WARPS * WARP_SIZE);
      }
    }
  }
#endif
}

template <ReduceOp OpType, typename T, typename AccumT = T>
struct NvlsWarpPipelineAdapter {
  static cudaError_t call(const void* input, void* scratch, void* output, void* memoryChannels, void*,
                          DeviceHandle<SwitchChannel>* nvlsChannels, DeviceHandle<SwitchChannel>*, size_t, size_t,
                          size_t scratchBufferSize, int rank, int ipcDomainNranks, int, size_t inputSize,
                          cudaStream_t stream, void*, uint32_t, uint32_t, int nBlocks, int nThreadsPerBlock) {
    // uint8_t is not supported for NVLS (no hardware support for byte-level reduction)
    if constexpr (std::is_same_v<T, uint8_t>) {
      return cudaErrorNotSupported;
    } else if constexpr (std::is_same_v<T, __fp8_e4m3b15>) {
      // fp8_e4m3b15 is a software-only type with no hardware NVLS support.
      return cudaErrorNotSupported;
    } else
#if defined(__CUDA_ARCH__)  // Skip the __CUDA_ARCH__ < 1000 since FP8 has not been supported for NVLS
      if constexpr (std::is_same_v<T, __fp8_e4m3> || std::is_same_v<T, __fp8_e5m2>) {
        return cudaErrorNotSupported;
      } else
#endif
      {
        using ChannelType = DeviceHandle<BaseMemoryChannel>;
        allreduceNvlsWarpPipeline<T><<<nBlocks, nThreadsPerBlock, 0, stream>>>(
            input, scratch, output, (ChannelType*)memoryChannels, nvlsChannels, inputSize, scratchBufferSize, rank,
            ipcDomainNranks);
        return cudaGetLastError();
      }
  }
};

void AllreduceNvlsWarpPipeline::initialize(std::shared_ptr<Communicator> comm) {
  nSwitchChannels_ = NUM_NVLS_CONNECTION;
  ipcDomainNranks_ = getIpcDomainNranks(comm);
  // The warp-pipeline kernel addresses 2 * nPeers entries per block in `memoryChannels`,
  // so per-peer base channel allocation must be at least `2 * nBlocks`. Default
  // nBlocks = 4 * ipcDomainNranks (see allreduceKernelFunc), so size accordingly.
  nBaseChannels_ = std::max(64, 8 * ipcDomainNranks_);
  this->conns_ = setupConnections(comm);
  // setup semaphores
  std::vector<std::shared_ptr<MemoryDevice2DeviceSemaphore>> memorySemaphores =
      setupMemorySemaphores(comm, this->conns_, nBaseChannels_);
  // setup base memory channels
  this->baseChannels_ = setupBaseMemoryChannels(this->conns_, memorySemaphores, nBaseChannels_);
  this->memoryChannelsDeviceHandle_ = setupBaseMemoryChannelDeviceHandles(this->baseChannels_);
  this->nvlsConnections_ = setupNvlsConnections(comm, nvlsBufferSize_, nSwitchChannels_);
}

CommResult AllreduceNvlsWarpPipeline::allreduceKernelFunc(
    const std::shared_ptr<void> ctx_void, const void* input, void* output, size_t inputSize, DataType dtype,
    ReduceOp op, cudaStream_t stream, int nBlocks, int nThreadsPerBlock,
    [[maybe_unused]] const std::unordered_map<std::string, uintptr_t>& extras, DataType accumDtype) {
  auto ctx = std::static_pointer_cast<AlgorithmCtx>(ctx_void);
  AllreduceFunc allreduce = dispatch<NvlsWarpPipelineAdapter>(op, dtype, accumDtype);
  if (!allreduce) {
    WARN("Unsupported operation or data type for allreduce, dtype=%d", static_cast<int>(dtype));
    return CommResult::CommInvalidArgument;
  }
  std::pair<int, int> blockAndThreadNum = {nBlocks, nThreadsPerBlock};
  if (blockAndThreadNum.first == 0) {
    // Default to 4 * ipcDomainNranks blocks, rounded up to a multiple of NUM_NVLS_CONNECTION
    // so that nBlocks / NUM_NVLS_CONNECTION partitioning in the kernel is well-defined.
    int defaultBlocks = ctx->ipcDomainNranks * 4;
    defaultBlocks = ((defaultBlocks + NUM_NVLS_CONNECTION - 1) / NUM_NVLS_CONNECTION) * NUM_NVLS_CONNECTION;
    blockAndThreadNum.first = std::max(defaultBlocks, NUM_NVLS_CONNECTION);
  }
  if (blockAndThreadNum.second == 0) blockAndThreadNum.second = 1024;
  // The kernel computes nBlocksPerNvlsConn = nBlocks / NUM_NVLS_CONNECTION and indexes the
  // multicast handle array with bid / nBlocksPerNvlsConn; both must be safe.
  if (blockAndThreadNum.first < NUM_NVLS_CONNECTION || blockAndThreadNum.first % NUM_NVLS_CONNECTION != 0) {
    WARN("AllreduceNvlsWarpPipeline requires nBlocks to be a positive multiple of %d (got %d)", NUM_NVLS_CONNECTION,
         blockAndThreadNum.first);
    return CommResult::CommInvalidArgument;
  }
  // Each block uses 2 * nPeers consecutive entries in `memoryChannels`, so the per-peer
  // base-channel allocation must support 2 * nBlocks distinct entries.
  if (2 * blockAndThreadNum.first > this->nBaseChannels_) {
    WARN(
        "AllreduceNvlsWarpPipeline: nBlocks %d exceeds channel allocation (nBaseChannels=%d, "
        "ipcDomainNranks=%d). Increase MSCCLPP_IPC_DOMAIN_NRANKS-aware sizing or reduce nBlocks.",
        blockAndThreadNum.first, this->nBaseChannels_, ctx->ipcDomainNranks);
    return CommResult::CommInvalidArgument;
  }
  // The kernel hard-codes 14 + 4 + 14 = 32 warps per block and bar.sync member counts
  // computed from these constants; deviating from 1024 threads breaks those barriers.
  if (blockAndThreadNum.second != 1024) {
    WARN("AllreduceNvlsWarpPipeline requires nThreadsPerBlock == 1024 (got %d)", blockAndThreadNum.second);
    return CommResult::CommInvalidArgument;
  }
  // Validate input divisibility by ipcDomainNranks (kernel computes size / ipcDomainNranks).
  if (inputSize % static_cast<size_t>(ctx->ipcDomainNranks) != 0) {
    WARN("AllreduceNvlsWarpPipeline requires inputSize %% ipcDomainNranks == 0 (got inputSize=%zu, ipcDomainNranks=%d)",
         inputSize, ctx->ipcDomainNranks);
    return CommResult::CommInvalidArgument;
  }
  // Validate scratch is large enough for at least one pipeline iteration. The kernel
  // computes scratchSizePerBlock = (scratchSizePerRank / nBlocks) aligned down to copyPerIter;
  // if this is 0 the modulo offset arithmetic divides by zero.
  const size_t sizePerRank = inputSize / static_cast<size_t>(ctx->ipcDomainNranks);
  const size_t maxSizePerBlock = ((sizePerRank + blockAndThreadNum.first - 1) / blockAndThreadNum.first + 15) / 16 * 16;
  const size_t copyPerIter = (maxSizePerBlock >= 1024 * 64) ? (1024 * 32) : (1024 * 16);
  const size_t scratchSizePerRank = this->scratchBufferSize_ / static_cast<size_t>(ctx->ipcDomainNranks);
  const size_t scratchSizePerBlock =
      (scratchSizePerRank / static_cast<size_t>(blockAndThreadNum.first)) / copyPerIter * copyPerIter;
  if (scratchSizePerBlock < copyPerIter) {
    WARN(
        "AllreduceNvlsWarpPipeline scratch buffer too small for ipcDomainNranks=%d, nBlocks=%d, inputSize=%zu "
        "(scratchBufferSize=%zu, need at least ~%zu bytes)",
        ctx->ipcDomainNranks, blockAndThreadNum.first, inputSize, this->scratchBufferSize_,
        static_cast<size_t>(ctx->ipcDomainNranks) * static_cast<size_t>(blockAndThreadNum.first) * copyPerIter);
    return CommResult::CommInvalidArgument;
  }
  cudaError_t error = allreduce(input, this->scratchBuffer_, output, this->memoryChannelsDeviceHandle_.get(), nullptr,
                                ctx->switchChannelDeviceHandles.get(), nullptr, 0, 0, this->scratchBufferSize_,
                                ctx->rank, ctx->ipcDomainNranks, ctx->workSize, inputSize, stream, nullptr, 0, 0,
                                blockAndThreadNum.first, blockAndThreadNum.second);
  if (error != cudaSuccess) {
    WARN("AllreduceNvlsWarpPipeline failed with error: %s", cudaGetErrorString(error));
    return CommResult::CommUnhandledCudaError;
  }
  return CommResult::CommSuccess;
}

AlgorithmCtxKey AllreduceNvlsWarpPipeline::generateAllreduceContextKey(const void*, void*, size_t, DataType, bool) {
  return AlgorithmCtxKey{nullptr, nullptr, 0, 0, 0};
}

std::shared_ptr<void> AllreduceNvlsWarpPipeline::initAllreduceContext(std::shared_ptr<Communicator> comm, const void*,
                                                                      void*, size_t, DataType) {
  auto ctx = std::make_shared<AlgorithmCtx>();
  ctx->rank = comm->bootstrap()->getRank();
  ctx->workSize = comm->bootstrap()->getNranks();
  ctx->ipcDomainNranks = getIpcDomainNranks(comm);

  // setup channels
  ctx->switchChannels =
      setupNvlsChannels(this->nvlsConnections_, this->scratchBuffer_, scratchBufferSize_, nSwitchChannels_);
  ctx->switchChannelDeviceHandles = setupNvlsChannelDeviceHandles(ctx->switchChannels);
  return ctx;
}

std::shared_ptr<Algorithm> AllreduceNvlsWarpPipeline::build() {
  auto self =
      std::make_shared<AllreduceNvlsWarpPipeline>(reinterpret_cast<uintptr_t>(scratchBuffer_), scratchBufferSize_);
  return std::make_shared<NativeAlgorithm>(
      "default_allreduce_nvls_warp_pipeline", "allreduce",
      [self](std::shared_ptr<Communicator> comm) { self->initialize(comm); },
      [self](const std::shared_ptr<void> ctx, const void* input, void* output, size_t inputSize,
             [[maybe_unused]] size_t outputSize, DataType dtype, ReduceOp op, cudaStream_t stream, int nBlocks,
             int nThreadsPerBlock, const std::unordered_map<std::string, uintptr_t>& extras, DataType accumDtype) {
        return self->allreduceKernelFunc(ctx, input, output, inputSize, dtype, op, stream, nBlocks, nThreadsPerBlock,
                                         extras, accumDtype);
      },
      [self](std::shared_ptr<Communicator> comm, const void* input, void* output, size_t inputSize,
             [[maybe_unused]] size_t outputSize,
             DataType dtype) { return self->initAllreduceContext(comm, input, output, inputSize, dtype); },
      [self](const void* input, void* output, size_t inputSize, [[maybe_unused]] size_t outputSize, DataType dtype,
             bool symmetricMemory) {
        return self->generateAllreduceContextKey(input, output, inputSize, dtype, symmetricMemory);
      });
}
}  // namespace collective
}  // namespace mscclpp
