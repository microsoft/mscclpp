// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "moe_runtime.hpp"

#include <cuda.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <future>
#include <mscclpp/concurrency_device.hpp>
#include <string>

#include "api.cuh"
#include "constants.cuh"
#include "exception.cuh"

namespace mscclpp {
namespace ep {

namespace {

int localWorldSize() {
  int localWorldSize = NUM_MAX_NVL_PEERS;
  if (const char* env = std::getenv("MSCCLPP_EP_LOCAL_WORLD_SIZE")) {
    int v = std::atoi(env);
    if (v > 0 && v <= NUM_MAX_NVL_PEERS) localWorldSize = v;
  }
  return localWorldSize;
}

bool resolveFabricIpcSupported() {
  if (const char* env = std::getenv("MSCCLPP_EP_FABRIC_IPC")) {
    std::string v(env);
    for (auto& c : v) c = std::tolower(static_cast<unsigned char>(c));
    if (v == "0" || v == "off" || v == "false" || v == "no") return false;
    if (v == "1" || v == "on" || v == "true" || v == "yes" || v == "force") return true;
  }
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return false;
  CUdevice cu_dev;
  if (cuDeviceGet(&cu_dev, dev) != CUDA_SUCCESS) return false;
  int supported = 0;
  if (cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, cu_dev) != CUDA_SUCCESS) {
    return false;
  }
  if (!supported) return false;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return false;
  return prop.major >= 10;
}

}  // namespace

MoERuntime::MoERuntime(mscclpp::Communicator& communicator, int64_t numNvlBytes, int64_t numRdmaBytes, MoEMode mode)
    : rank_(communicator.bootstrap()->getRank()),
      numRanks_(communicator.bootstrap()->getNranks()),
      numNvlBytes_(numNvlBytes),
      numRdmaBytes_(numRdmaBytes),
      mode_(mode),
      communicator_(&communicator) {
  EP_HOST_ASSERT(mode_ == MoEMode::LOW_LATENCY);
  EP_HOST_ASSERT(communicator_ != nullptr);
  EP_HOST_ASSERT(numNvlBytes_ == 0);
  EP_HOST_ASSERT(numRdmaBytes_ % NUM_BUFFER_ALIGNMENT_BYTES == 0);
  EP_HOST_ASSERT(rank_ >= 0 && rank_ < numRanks_);

  CUDA_CHECK(cudaGetDevice(&deviceId_));
  int lws = localWorldSize();
  rdmaRank_ = rank_ / lws;
  nvlRank_ = rank_ % lws;
  numRdmaRanks_ = std::max(1, numRanks_ / lws);
  numNvlRanks_ = std::min(numRanks_, lws);

  CUDA_CHECK(cudaMalloc(&workspace_, NUM_WORKSPACE_BYTES));
  CUDA_CHECK(cudaMemset(workspace_, 0, NUM_WORKSPACE_BYTES));
  setup();
}

MoERuntime::~MoERuntime() noexcept(false) {
  CUDA_CHECK(cudaDeviceSynchronize());
  if (peerRdmaBasesGpu_ != nullptr) CUDA_CHECK(cudaFree(peerRdmaBasesGpu_));
  if (workspace_ != nullptr) CUDA_CHECK(cudaFree(workspace_));
  if (rdmaBufferPtr_ != nullptr) {
    // gpuCallocPhysical allocations are intentionally process-lifetime in the
    // existing EP runtime. cudaFree is safe for the fallback cudaMalloc path.
    // Ignore failures from fabric allocations during process teardown.
    cudaFree(rdmaBufferPtr_);
  }
}

bool MoERuntime::isAvailable() const { return available_; }
bool MoERuntime::isInternodeAvailable() const { return isAvailable() && numRanks_ > NUM_MAX_NVL_PEERS; }
int MoERuntime::getNumRdmaRanks() const { return numRdmaRanks_; }
int MoERuntime::getRdmaRank() const { return rdmaRank_; }
int MoERuntime::getRootRdmaRank(bool global) const { return global ? nvlRank_ : 0; }
int MoERuntime::getLocalDeviceId() const { return deviceId_; }
std::string MoERuntime::getLocalIpcHandle() const { return {}; }

void MoERuntime::setup() {
  EP_HOST_ASSERT(!available_);
  EP_HOST_ASSERT(communicator_ != nullptr);

  const auto ipcTransport = mscclpp::Transport::CudaIpc;
  const bool fabricIpcSupported = resolveFabricIpcSupported();
  const bool useFabricIpcAlloc = mscclpp::isNvlsSupported() && fabricIpcSupported;
  if (useFabricIpcAlloc) {
    rdmaBufferPtr_ = mscclpp::detail::gpuCallocPhysical(numRdmaBytes_);
  } else {
    CUDA_CHECK(cudaMalloc(&rdmaBufferPtr_, numRdmaBytes_));
  }
  CUDA_CHECK(cudaMemset(rdmaBufferPtr_, 0, numRdmaBytes_));
  communicator_->bootstrap()->barrier();
  CUDA_CHECK(cudaDeviceSynchronize());

  const mscclpp::EndpointConfig ipcConfig(ipcTransport);
  const int ipcDomainSize = useFabricIpcAlloc ? numRanks_ : numNvlRanks_;
  auto isIpcPeer = [&](int peer) {
    return peer != rank_ && ipcDomainSize > 1 && rank_ / ipcDomainSize == peer / ipcDomainSize;
  };

  constexpr int IpcTag = 1;
  peerRdmaMemories_.resize(numRanks_);
  peerRdmaMemories_[rank_] = communicator_->registerMemory(rdmaBufferPtr_, numRdmaBytes_, ipcTransport);
  std::vector<std::shared_future<mscclpp::RegisteredMemory>> remoteFutures(numRanks_);
  std::vector<std::shared_future<mscclpp::Connection>> connectionFutures(numRanks_);
  for (int r = 0; r < numRanks_; ++r) {
    if (!isIpcPeer(r)) continue;
    communicator_->sendMemory(peerRdmaMemories_[rank_], r, IpcTag);
    remoteFutures[r] = communicator_->recvMemory(r, IpcTag);
    connectionFutures[r] = communicator_->connect(ipcConfig, r, IpcTag);
  }

  peerRdmaBases_.assign(numRanks_, nullptr);
  peerRdmaBases_[rank_] = rdmaBufferPtr_;
  std::vector<mscclpp::BaseMemoryChannelDeviceHandle> baseMemoryChannelHandles(numRanks_);
  for (int r = 0; r < numRanks_; ++r) {
    if (!isIpcPeer(r)) continue;
    peerRdmaMemories_[r] = remoteFutures[r].get();
    peerRdmaBases_[r] = peerRdmaMemories_[r].data();
    auto semaphore =
        std::make_shared<mscclpp::MemoryDevice2DeviceSemaphore>(*communicator_, connectionFutures[r].get());
    baseMemoryChannels_.emplace_back(semaphore);
    baseMemoryChannelHandles[r] = baseMemoryChannels_.back().deviceHandle();
  }

  CUDA_CHECK(cudaMalloc(&peerRdmaBasesGpu_, sizeof(void*) * numRanks_));
  CUDA_CHECK(cudaMemcpy(peerRdmaBasesGpu_, peerRdmaBases_.data(), sizeof(void*) * numRanks_, cudaMemcpyHostToDevice));
  baseMemoryChannelHandles_ = mscclpp::detail::gpuCallocShared<mscclpp::BaseMemoryChannelDeviceHandle>(numRanks_);
  mscclpp::gpuMemcpy<mscclpp::BaseMemoryChannelDeviceHandle>(
      baseMemoryChannelHandles_.get(), baseMemoryChannelHandles.data(), numRanks_, cudaMemcpyHostToDevice);

  int maxSharedMemoryPerBlock;
  int numSms;
  CUDA_CHECK(cudaDeviceGetAttribute(&maxSharedMemoryPerBlock, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceId_));
  CUDA_CHECK(cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, deviceId_));
  commContext_ = {.rdmaBufferBase_ = rdmaBufferPtr_,
                  .baseMemoryChannels_ = baseMemoryChannelHandles_.get(),
                  .peerBases_ = peerRdmaBasesGpu_,
                  .maxSharedMemoryPerBlock_ = maxSharedMemoryPerBlock,
                  .numSms_ = numSms,
                  .deviceId_ = deviceId_,
                  .rank_ = rank_,
                  .numRanks_ = numRanks_};
  available_ = ipcDomainSize >= numRanks_;
}

void MoERuntime::dispatch(void* output, float* outputScales, int* outputSrcInfo, int64_t* outputLayout,
                          int* outputCount, const void* input, const int64_t* topkIdx, const float* topkWeights,
                          int numTokens, int hidden, int numTopk, int maxTokensPerRank, int numExperts,
                          low_latency::DispatchDataType dispatchDataType, int numBlocks, cudaStream_t stream) {
  EP_HOST_ASSERT(mode_ == MoEMode::LOW_LATENCY);
  EP_HOST_ASSERT(available_);
  EP_HOST_ASSERT(numTokens <= maxTokensPerRank);
  EP_HOST_ASSERT(numExperts % numRanks_ == 0);
  EP_HOST_ASSERT(numBlocks - low_latency::DispatchControlBlocks >= numRanks_ &&
                 numBlocks <= low_latency::MaxDispatchBlocks);

  low_latency::Layout layout(rdmaBufferPtr_, maxTokensPerRank, hidden, numRanks_, numExperts, numTopk);
  EP_HOST_ASSERT(layout.totalBytes_ <= static_cast<size_t>(numRdmaBytes_));
  void* recvBuffer = layout.buffers_[lowLatencyBufferIdx_].dispatchData_;
  lowLatencyBufferIdx_ ^= 1;

  const low_latency::Workload workload{.numTokens_ = numTokens,
                                       .hidden_ = hidden,
                                       .numTopk_ = numTopk,
                                       .numExperts_ = numExperts,
                                       .maxTokensPerRank_ = maxTokensPerRank,
                                       .dispatchDataType_ = dispatchDataType};
  const size_t workspaceBytes = low_latency::workspaceSize(numRanks_, numExperts);
  EP_HOST_ASSERT(workspaceBytes <= NUM_WORKSPACE_BYTES);
  low_latency::dispatch(output, outputScales, outputSrcInfo, outputLayout, outputCount, input, topkIdx, topkWeights,
                        workload, recvBuffer, commContext_, workspace_, numBlocks, stream);
}

void MoERuntime::combine(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights,
                         const int* srcInfo, const int64_t* layoutRange, int numTokens, int hidden, int numTopk,
                         int maxTokensPerRank, int numExperts, low_latency::DispatchDataType dispatchDataType,
                         low_latency::CombineMode mode, int numBlocks, cudaStream_t stream) {
  EP_HOST_ASSERT(mode_ == MoEMode::LOW_LATENCY);
  EP_HOST_ASSERT(available_);
  EP_HOST_ASSERT(numExperts % numRanks_ == 0);
  EP_HOST_ASSERT(numBlocks > 0 && numBlocks <= low_latency::MaxWorkerBlocks);

  low_latency::Layout layout(rdmaBufferPtr_, maxTokensPerRank, hidden, numRanks_, numExperts, numTopk);
  EP_HOST_ASSERT(layout.totalBytes_ <= static_cast<size_t>(numRdmaBytes_));
  void* recvBuffer = layout.buffers_[lowLatencyBufferIdx_].combineData_;
  lowLatencyBufferIdx_ ^= 1;
  void* dispatchRecvBuffer = layout.buffers_[lowLatencyBufferIdx_].dispatchData_;

  const low_latency::Workload workload{.numTokens_ = numTokens,
                                       .hidden_ = hidden,
                                       .numTopk_ = numTopk,
                                       .numExperts_ = numExperts,
                                       .maxTokensPerRank_ = maxTokensPerRank,
                                       .dispatchDataType_ = dispatchDataType};
  low_latency::combine(output, input, topkIdx, topkWeights, srcInfo, layoutRange, workload, recvBuffer,
                       dispatchRecvBuffer, commContext_, workspace_, numBlocks, mode, stream);
}

}  // namespace ep
}  // namespace mscclpp
