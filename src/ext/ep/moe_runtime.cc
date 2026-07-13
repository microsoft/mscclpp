// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "moe_runtime.hpp"

#include <cuda.h>

#include <algorithm>
#include <future>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/utils.hpp>

#include "api.cuh"
#include "constants.cuh"
#include "exception.cuh"

namespace mscclpp {
namespace ep {

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
  numNvlRanks_ = std::min(numRanks_, communicator.bootstrap()->getNranksPerNode());
  numRanksPerIpcDomain_ =
      std::max(numNvlRanks_, std::min(numRanks_, communicator.bootstrap()->getNranksPerIpcDomain()));
  EP_HOST_ASSERT(numNvlRanks_ > 0 && numRanks_ % numNvlRanks_ == 0);
  EP_HOST_ASSERT(numRanks_ % numRanksPerIpcDomain_ == 0);
  rdmaRank_ = rank_ / numNvlRanks_;
  nvlRank_ = rank_ % numNvlRanks_;
  numRdmaRanks_ = numRanks_ / numNvlRanks_;

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
bool MoERuntime::isInternodeAvailable() const { return isAvailable() && numRdmaRanks_ > 1; }
int MoERuntime::getNumRdmaRanks() const { return numRdmaRanks_; }
int MoERuntime::getRdmaRank() const { return rdmaRank_; }
int MoERuntime::getRootRdmaRank(bool global) const { return global ? nvlRank_ : 0; }
int MoERuntime::getLocalDeviceId() const { return deviceId_; }
std::string MoERuntime::getLocalIpcHandle() const { return {}; }

void MoERuntime::setup() {
  EP_HOST_ASSERT(!available_);
  EP_HOST_ASSERT(communicator_ != nullptr);

  const auto ipcTransport = mscclpp::Transport::CudaIpc;
  const bool spansHosts = numRanksPerIpcDomain_ > numNvlRanks_;
  std::vector<int> fabricCapabilities(numRanks_, 0);
  fabricCapabilities[rank_] = spansHosts && mscclpp::isFabricMemHandleAvailable() && mscclpp::isNvlsSupported();
  communicator_->bootstrap()->allGather(fabricCapabilities.data(), sizeof(int));
  const bool useFabricIpcAlloc =
      spansHosts && std::all_of(fabricCapabilities.begin(), fabricCapabilities.end(), [](int value) { return value; });
  if (useFabricIpcAlloc) {
    rdmaBufferPtr_ = mscclpp::detail::gpuCallocPhysical(numRdmaBytes_);
  } else {
    CUDA_CHECK(cudaMalloc(&rdmaBufferPtr_, numRdmaBytes_));
  }
  CUDA_CHECK(cudaMemset(rdmaBufferPtr_, 0, numRdmaBytes_));
  communicator_->bootstrap()->barrier();
  CUDA_CHECK(cudaDeviceSynchronize());

  const mscclpp::EndpointConfig ipcConfig(ipcTransport);
  const int ipcDomainSize = useFabricIpcAlloc ? numRanksPerIpcDomain_ : numNvlRanks_;
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
