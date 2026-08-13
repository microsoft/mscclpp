// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <mscclpp/core.hpp>
#include <mscclpp/gpu_data_types.hpp>
#include <mscclpp/switch_channel_device.hpp>

#include "allgather/allgather_nvls_zero_copy.hpp"
#include "collective_utils.hpp"
#include "debug.h"

namespace mscclpp {
namespace collective {

constexpr int MAX_NBLOCKS = 32;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
// Broadcast `size` bytes from the local `src` buffer to the multicast destination `mcDst` using
// NVLS multimem stores. Allgather is a pure copy (no reduction), so the operation is
// dtype-agnostic: bulk 16-byte (f32x4) stores with a 4-byte (f32x1) aligned tail. NVLS requires
// at least 4-byte alignment; callers guarantee 16-byte-aligned offsets for the bulk path.
MSCCLPP_DEVICE_INLINE void multimemBroadcast(const char* src, char* mcDst, size_t size, int tid, int nThreads) {
  constexpr size_t vecBytes = sizeof(f32x4);
  const size_t nVec = size / vecBytes;
  for (size_t i = tid; i < nVec; i += nThreads) {
    f32x4 val = *(reinterpret_cast<const f32x4*>(src) + i);
    SwitchChannelDeviceHandle::multimemStore(val, reinterpret_cast<f32x4*>(mcDst) + i);
  }
  const size_t tailStart = nVec * vecBytes;
  const size_t nTail = (size - tailStart) / sizeof(f32x1);
  for (size_t i = tid; i < nTail; i += nThreads) {
    f32x1 val = *(reinterpret_cast<const f32x1*>(src + tailStart) + i);
    SwitchChannelDeviceHandle::multimemStore(val, reinterpret_cast<f32x1*>(mcDst + tailStart) + i);
  }
}
#endif

__global__ void __launch_bounds__(1024, 1)
    allgatherNvls([[maybe_unused]] mscclpp::DeviceHandle<mscclpp::BaseMemoryChannel>* memoryChannels,
                  [[maybe_unused]] mscclpp::DeviceHandle<mscclpp::SwitchChannel>* multicast,
                  [[maybe_unused]] const void* sendbuff, [[maybe_unused]] size_t channelOutOffset,
                  [[maybe_unused]] size_t bytesPerRank, [[maybe_unused]] int rank,
                  [[maybe_unused]] int nRanksPerIpcDomain) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const int nPeers = nRanksPerIpcDomain - 1;
  const int nBlocks = gridDim.x;
  const int bid = blockIdx.x;
  const size_t minAlign = 16;
  // Align the per-block slice to 16 bytes so bulk f32x4 stores stay aligned.
  size_t sizePerBlock = (bytesPerRank + nBlocks - 1) / nBlocks;
  sizePerBlock = (sizePerBlock + minAlign - 1) / minAlign * minAlign;
  const size_t blockStart = sizePerBlock * bid;
  size_t curBlockSize = 0;
  if (blockStart < bytesPerRank) {
    curBlockSize = min(sizePerBlock, bytesPerRank - blockStart);
  }

  mscclpp::DeviceHandle<mscclpp::SwitchChannel>* multicastPtr = multicast + bid;

  const size_t chanOffset = (nRanksPerIpcDomain - 1) * blockIdx.x;
  auto memoryChans = memoryChannels + chanOffset;
  __shared__ mscclpp::DeviceHandle<mscclpp::BaseMemoryChannel> channels[MAX_IPC_DOMAIN_NRANKS - 1];
  const int lid = threadIdx.x % WARP_SIZE;
  if (lid < nRanksPerIpcDomain - 1) {
    channels[lid] = memoryChans[lid];
  }
  __syncwarp();
  if (threadIdx.x < nPeers) {
    channels[threadIdx.x].relaxedSignal();
    channels[threadIdx.x].relaxedWait();
  }
  __syncthreads();

  if (curBlockSize > 0) {
    const char* src = reinterpret_cast<const char*>(sendbuff) + blockStart;
    char* mcDst = reinterpret_cast<char*>(multicastPtr->mcPtr) + channelOutOffset + bytesPerRank * rank + blockStart;
    multimemBroadcast(src, mcDst, curBlockSize, threadIdx.x, blockDim.x);
  }
  __syncthreads();
  if (threadIdx.x < nPeers) {
    channels[threadIdx.x].relaxedSignal();
    channels[threadIdx.x].relaxedWait();
  }
#endif
}

void AllgatherNvls::initialize(std::shared_ptr<mscclpp::Communicator> comm) {
  int device;
  MSCCLPP_CUDATHROW(cudaGetDevice(&device));
  cudaDeviceProp deviceProp;
  MSCCLPP_CUDATHROW(cudaGetDeviceProperties(&deviceProp, device));
  computeCapabilityMajor_ = deviceProp.major;
  nSwitchChannels_ = 32;
  this->conns_ = setupConnections(comm);
  std::vector<std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>> memorySemaphores =
      setupMemorySemaphores(comm, this->conns_, nSwitchChannels_);
  this->baseChannels_ = setupBaseMemoryChannels(this->conns_, memorySemaphores, nSwitchChannels_);
  this->memoryChannelsDeviceHandle_ = setupBaseMemoryChannelDeviceHandles(this->baseChannels_);
  this->nvlsConnections_ = setupNvlsConnections(comm, nvlsBufferSize_, nSwitchChannels_);
}

CommResult AllgatherNvls::allgatherKernelFunc(const std::shared_ptr<void> ctx_void, const void* input, void* output,
                                              size_t inputSize, cudaStream_t stream, int nBlocks, int nThreadsPerBlock,
                                              const std::unordered_map<std::string, uintptr_t>&) {
  if (!symmetricMemory_) {
    WARN("AllgatherNvls requires symmetric memory for now.");
    return CommResult::CommInvalidArgument;
  }
  auto ctx = std::static_pointer_cast<AlgorithmCtx>(ctx_void);

  size_t recvBytes;
  CUdeviceptr recvBasePtr;
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&recvBasePtr, &recvBytes, (CUdeviceptr)output));
  size_t channelOutOffset = (char*)output - (char*)recvBasePtr;

  mscclpp::DeviceHandle<mscclpp::SwitchChannel>* nvlsChannels = ctx->switchChannelDeviceHandles.get();

  std::pair<int, int> numBlocksAndThreads = {nBlocks, nThreadsPerBlock};
  if (numBlocksAndThreads.first == 0 || numBlocksAndThreads.second == 0) {
    numBlocksAndThreads = {::min(ctx->nRanksPerIpcDomain, MAX_NBLOCKS), 1024};
    // For GB200 devices with MNNVLS, scale the number of blocks inversely with the number of GPUs
    // (empirically 128 / nGPUs, clamped to [1, MAX_NBLOCKS]), mirroring the NVLS allreduce heuristic.
    if (computeCapabilityMajor_ == 10) {
      numBlocksAndThreads.first = ::max(1, ::min(128 / ctx->worldSize, MAX_NBLOCKS));
    }
  }
  if (numBlocksAndThreads.first > MAX_NBLOCKS) {
    WARN("Number of blocks exceeds maximum supported value of %d", MAX_NBLOCKS);
    return CommResult::CommInvalidArgument;
  }

  allgatherNvls<<<numBlocksAndThreads.first, numBlocksAndThreads.second, 0, stream>>>(
      this->memoryChannelsDeviceHandle_.get(), nvlsChannels, input, channelOutOffset, inputSize, ctx->rank,
      ctx->nRanksPerIpcDomain);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    WARN("AllgatherNvls failed with error: %s", cudaGetErrorString(error));
    return CommResult::CommUnhandledCudaError;
  }
  return CommResult::CommSuccess;
}

mscclpp::AlgorithmCtxKey AllgatherNvls::generateAllgatherContextKey(const void*, void* output, size_t, mscclpp::DataType,
                                                                    bool symmetricMemory) {
  static int tag = 0;
  symmetricMemory_ = symmetricMemory;
  if (!symmetricMemory_) {
    // Always return a fresh key if symmetric memory is not enabled.
    return mscclpp::AlgorithmCtxKey{nullptr, nullptr, 0, 0, tag++};
  }
  size_t recvBytes;
  CUdeviceptr recvBasePtr;
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&recvBasePtr, &recvBytes, (CUdeviceptr)output));
  return mscclpp::AlgorithmCtxKey{nullptr, (void*)recvBasePtr, 0, recvBytes, 0};
}

std::shared_ptr<void> AllgatherNvls::initAllgatherContext(std::shared_ptr<mscclpp::Communicator> comm, const void*,
                                                          void* output, size_t, mscclpp::DataType) {
  auto ctx = std::make_shared<AlgorithmCtx>();
  ctx->rank = comm->bootstrap()->getRank();
  ctx->worldSize = comm->bootstrap()->getNranks();
  ctx->nRanksPerIpcDomain = comm->bootstrap()->getNranksPerIpcDomain();

  size_t recvBytes;
  CUdeviceptr recvBasePtr;
  MSCCLPP_CUTHROW(cuMemGetAddressRange(&recvBasePtr, &recvBytes, (CUdeviceptr)output));

  // NVLS multicast channels over the output buffer (each rank stores its chunk to all ranks).
  ctx->switchChannels = setupNvlsChannels(comm, this->nvlsConnections_, (void*)recvBasePtr, recvBytes, nSwitchChannels_);
  ctx->switchChannelDeviceHandles = setupNvlsChannelDeviceHandles(ctx->switchChannels);
  return ctx;
}

std::shared_ptr<mscclpp::Algorithm> AllgatherNvls::build() {
  auto self = std::make_shared<AllgatherNvls>();
  return std::make_shared<mscclpp::NativeAlgorithm>(
      "default_allgather_nvls_zero_copy", "allgather",
      [self](std::shared_ptr<mscclpp::Communicator> comm) { self->initialize(comm); },
      [self](const std::shared_ptr<void> ctx, const void* input, void* output, size_t inputSize,
             [[maybe_unused]] size_t outputSize, [[maybe_unused]] mscclpp::DataType dtype, [[maybe_unused]] ReduceOp op,
             cudaStream_t stream, int nBlocks, int nThreadsPerBlock,
             const std::unordered_map<std::string, uintptr_t>& extras,
             [[maybe_unused]] mscclpp::DataType accumDtype) -> mscclpp::CommResult {
        return self->allgatherKernelFunc(ctx, input, output, inputSize, stream, nBlocks, nThreadsPerBlock, extras);
      },
      [self](std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output, size_t inputSize,
             [[maybe_unused]] size_t outputSize,
             mscclpp::DataType dtype) { return self->initAllgatherContext(comm, input, output, inputSize, dtype); },
      [self](const void* input, void* output, size_t inputSize, [[maybe_unused]] size_t outputSize,
             mscclpp::DataType dtype, bool symmetricMemory) {
        return self->generateAllgatherContextKey(input, output, inputSize, dtype, symmetricMemory);
      });
}

}  // namespace collective
}  // namespace mscclpp
