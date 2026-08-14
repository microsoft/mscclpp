// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "ll_runtime.hpp"

#include <cuda.h>

#include <algorithm>
#include <future>
#include <mscclpp/concurrency_device.hpp>

#include "api.cuh"
#include "exception.cuh"

namespace mscclpp {
namespace ep {

MoELowLatencyRuntime::MoELowLatencyRuntime(mscclpp::Communicator& communicator, int maxTokensPerRank, int hidden,
                                           int numExperts, int numTopk, DispatchLayout outputLayout)
    : MoERuntime(communicator),
      maxTokensPerRank_(maxTokensPerRank),
      hidden_(hidden),
      numExperts_(numExperts),
      numTopk_(numTopk),
      outputLayout_(outputLayout),
      symmetricBufferBytes_(static_cast<int64_t>(low_latency::symmetricBufferSize(
          maxTokensPerRank, hidden, numRanks_, numExperts, numTopk, outputLayout == DispatchLayout::RANK_MAJOR))),
      workspaceBytes_(low_latency::workspaceSize(numRanks_, numExperts, maxTokensPerRank, numTopk)),
      communicator_(&communicator) {
  EP_HOST_ASSERT(communicator_ != nullptr);
  EP_HOST_ASSERT(symmetricBufferBytes_ % BufferAlignmentBytes == 0);
  EP_HOST_ASSERT(maxTokensPerRank > 0);
  EP_HOST_ASSERT(numExperts > 0 && numExperts % numRanks_ == 0);
  EP_HOST_ASSERT(numTopk > 0 && numTopk <= 32);
  EP_HOST_ASSERT(outputLayout == DispatchLayout::EXPERT_MAJOR || outputLayout == DispatchLayout::RANK_MAJOR);

  CUDA_CHECK(cudaGetDevice(&deviceId_));
  EP_HOST_ASSERT(numRanks_ % numNvlRanks_ == 0);
  EP_HOST_ASSERT(numRanks_ % numRanksPerIpcDomain_ == 0);

  CUDA_CHECK(cudaMalloc(&workspace_, workspaceBytes_));
  CUDA_CHECK(cudaMemset(workspace_, 0, workspaceBytes_));
  setup();
}

MoELowLatencyRuntime::~MoELowLatencyRuntime() noexcept(false) {
  CUDA_CHECK(cudaDeviceSynchronize());
  if (peerMappedBufferBasesGpu_ != nullptr) CUDA_CHECK(cudaFree(peerMappedBufferBasesGpu_));
  if (workspace_ != nullptr) CUDA_CHECK(cudaFree(workspace_));
  if (symmetricBuffer_ != nullptr) {
    mscclpp::detail::gpuFreePhysical(symmetricBuffer_);
  }
}

void* MoELowLatencyRuntime::outputTopkIdsBuffer() const {
  return low_latency::Layout(symmetricBuffer_, maxTokensPerRank_, hidden_, numRanks_, numExperts_, numTopk_,
                             outputLayout_ == DispatchLayout::RANK_MAJOR)
      .rankMajorTopkIdsBuffer_;
}
void* MoELowLatencyRuntime::outputTopkWeightsBuffer() const {
  return low_latency::Layout(symmetricBuffer_, maxTokensPerRank_, hidden_, numRanks_, numExperts_, numTopk_,
                             outputLayout_ == DispatchLayout::RANK_MAJOR)
      .rankMajorTopkWeightsBuffer_;
}
void* MoELowLatencyRuntime::outputTokensBuffer() const {
  return low_latency::Layout(symmetricBuffer_, maxTokensPerRank_, hidden_, numRanks_, numExperts_, numTopk_,
                             outputLayout_ == DispatchLayout::RANK_MAJOR)
      .rankMajorTokenBuffer_;
}
void* MoELowLatencyRuntime::expertOutputBuffer() const {
  return low_latency::Layout(symmetricBuffer_, maxTokensPerRank_, hidden_, numRanks_, numExperts_, numTopk_,
                             outputLayout_ == DispatchLayout::RANK_MAJOR)
      .rankMajorExpertOutputBuffer_;
}

void MoELowLatencyRuntime::setup() {
  EP_HOST_ASSERT(!available_);
  EP_HOST_ASSERT(communicator_ != nullptr);

  const auto ipcTransport = mscclpp::Transport::CudaIpc;
  const size_t allocationGranularity = mscclpp::detail::getCuAllocationGranularity(CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  symmetricBuffer_ =
      mscclpp::detail::gpuCallocPhysical(symmetricBufferBytes_, allocationGranularity, allocationGranularity);

  const mscclpp::EndpointConfig ipcConfig(ipcTransport);
  const int ipcDomainSize = numRanksPerIpcDomain_;
  auto isMappedPeer = [&](int peer) {
    return peer != rank_ && ipcDomainSize > 1 && rank_ / ipcDomainSize == peer / ipcDomainSize;
  };

  constexpr int IpcTag = 1;
  peerBufferMemories_.resize(numRanks_);
  peerBufferMemories_[rank_] = communicator_->registerMemory(symmetricBuffer_, symmetricBufferBytes_, ipcTransport);
  std::vector<std::shared_future<mscclpp::RegisteredMemory>> remoteFutures(numRanks_);
  std::vector<std::shared_future<mscclpp::Connection>> connectionFutures(numRanks_);
  for (int r = 0; r < numRanks_; ++r) {
    if (!isMappedPeer(r)) continue;
    communicator_->sendMemory(peerBufferMemories_[rank_], r, IpcTag);
    remoteFutures[r] = communicator_->recvMemory(r, IpcTag);
    connectionFutures[r] = communicator_->connect(ipcConfig, r, IpcTag);
  }

  peerMappedBufferBases_.assign(numRanks_, nullptr);
  peerMappedBufferBases_[rank_] = symmetricBuffer_;
  std::vector<mscclpp::BaseMemoryChannelDeviceHandle> baseMemoryChannelHandles(numRanks_);
  for (int r = 0; r < numRanks_; ++r) {
    if (!isMappedPeer(r)) continue;
    peerBufferMemories_[r] = remoteFutures[r].get();
    peerMappedBufferBases_[r] = peerBufferMemories_[r].data();
    auto semaphore =
        std::make_shared<mscclpp::MemoryDevice2DeviceSemaphore>(*communicator_, connectionFutures[r].get());
    baseMemoryChannels_.emplace_back(semaphore);
    baseMemoryChannelHandles[r] = baseMemoryChannels_.back().deviceHandle();
  }

  CUDA_CHECK(cudaMalloc(&peerMappedBufferBasesGpu_, sizeof(void*) * numRanks_));
  CUDA_CHECK(cudaMemcpy(peerMappedBufferBasesGpu_, peerMappedBufferBases_.data(), sizeof(void*) * numRanks_,
                        cudaMemcpyHostToDevice));
  baseMemoryChannelHandles_ = mscclpp::detail::gpuCallocShared<mscclpp::BaseMemoryChannelDeviceHandle>(numRanks_);
  mscclpp::gpuMemcpy<mscclpp::BaseMemoryChannelDeviceHandle>(
      baseMemoryChannelHandles_.get(), baseMemoryChannelHandles.data(), numRanks_, cudaMemcpyHostToDevice);

  int maxSharedMemoryPerBlock;
  int numSms;
  CUDA_CHECK(cudaDeviceGetAttribute(&maxSharedMemoryPerBlock, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceId_));
  CUDA_CHECK(cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, deviceId_));
  commContext_ = {.symmetricBufferBase_ = symmetricBuffer_,
                  .baseMemoryChannels_ = baseMemoryChannelHandles_.get(),
                  .peerMappedBufferBases_ = peerMappedBufferBasesGpu_,
                  .maxSharedMemoryPerBlock_ = maxSharedMemoryPerBlock,
                  .numSms_ = numSms,
                  .deviceId_ = deviceId_,
                  .rank_ = rank_,
                  .numRanks_ = numRanks_};
  available_ = ipcDomainSize >= numRanks_;
}

uint32_t MoELowLatencyRuntime::dispatch(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                                        float* outputTopkWeights, int64_t* outputLayout, int* outputCount,
                                        const void* input, const int64_t* topkIdx, const float* topkWeights,
                                        int numTokens, int hidden, int numTopk, int maxTokensPerRank, int numExperts,
                                        int invalidTokenExpertId, DispatchLayout dispatchLayout,
                                        low_latency::DispatchDataType dispatchDataType, int numBlocks,
                                        cudaStream_t stream) {
  EP_HOST_ASSERT(available_);
  EP_HOST_ASSERT(maxTokensPerRank > 0 && maxTokensPerRank <= maxTokensPerRank_);
  EP_HOST_ASSERT(numTokens <= maxTokensPerRank);
  EP_HOST_ASSERT(numExperts % numRanks_ == 0);
  EP_HOST_ASSERT(invalidTokenExpertId < 0 || invalidTokenExpertId >= numExperts);
  EP_HOST_ASSERT(numBlocks - low_latency::DispatchControlBlocks >= numRanks_ &&
                 numBlocks <= low_latency::MaxDispatchBlocks);
  EP_HOST_ASSERT(dispatchLayout == outputLayout_);

  low_latency::Layout allocationLayout(symmetricBuffer_, maxTokensPerRank_, hidden, numRanks_, numExperts, numTopk,
                                       outputLayout_ == DispatchLayout::RANK_MAJOR);
  EP_HOST_ASSERT(allocationLayout.totalBytes_ <= static_cast<size_t>(symmetricBufferBytes_));
  void* dispatchRecvBuffer = allocationLayout.dispatchRecvBuffer_;
  if (dispatchLayout == DispatchLayout::RANK_MAJOR) {
    EP_HOST_ASSERT(output == allocationLayout.rankMajorTokenBuffer_);
    EP_HOST_ASSERT(outputTopkIdx == allocationLayout.rankMajorTopkIdsBuffer_);
    EP_HOST_ASSERT(outputTopkWeights == allocationLayout.rankMajorTopkWeightsBuffer_);
  }

  const uint32_t dispatchEpoch = ++dispatchEpoch_;
  const low_latency::Workload workload{.dispatchEpoch_ = dispatchEpoch,
                                       .numTokens_ = numTokens,
                                       .hidden_ = hidden,
                                       .numTopk_ = numTopk,
                                       .numExperts_ = numExperts,
                                       .invalidTokenExpertId_ = invalidTokenExpertId,
                                       .maxTokensPerRank_ = maxTokensPerRank,
                                       .outputLayout_ = dispatchLayout,
                                       .dispatchDataType_ = dispatchDataType};
  const size_t workspaceBytes = low_latency::workspaceSize(numRanks_, numExperts, maxTokensPerRank, numTopk);
  EP_HOST_ASSERT(workspaceBytes <= workspaceBytes_);
  low_latency::dispatch(output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout,
                        outputCount, input, topkIdx, topkWeights, workload, dispatchRecvBuffer, commContext_,
                        workspace_, numBlocks, stream);
  return dispatchEpoch;
}

void MoELowLatencyRuntime::combine(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights,
                                   const int* srcInfo, const int64_t* layoutRange, uint32_t dispatchEpoch,
                                   int numTokens, int hidden, int numTopk, int maxTokensPerRank, int numExperts,
                                   DispatchLayout dispatchLayout, low_latency::DispatchDataType dispatchDataType,
                                   low_latency::CombineMode mode, int numBlocks, cudaStream_t stream) {
  EP_HOST_ASSERT(available_);
  EP_HOST_ASSERT(maxTokensPerRank > 0 && maxTokensPerRank <= maxTokensPerRank_);
  EP_HOST_ASSERT(numExperts % numRanks_ == 0);
  EP_HOST_ASSERT(numBlocks > 0 && numBlocks <= low_latency::MaxWorkerBlocks);
  EP_HOST_ASSERT(dispatchLayout == outputLayout_);

  low_latency::Layout allocationLayout(symmetricBuffer_, maxTokensPerRank_, hidden, numRanks_, numExperts, numTopk,
                                       outputLayout_ == DispatchLayout::RANK_MAJOR);
  EP_HOST_ASSERT(allocationLayout.totalBytes_ <= static_cast<size_t>(symmetricBufferBytes_));
  void* combineRecvBuffer = allocationLayout.combineRecvBuffer_;
  void* dispatchRecvBuffer = allocationLayout.dispatchRecvBuffer_;
  if (dispatchLayout == DispatchLayout::RANK_MAJOR) {
    EP_HOST_ASSERT(input == allocationLayout.rankMajorExpertOutputBuffer_);
  }

  const low_latency::Workload workload{.dispatchEpoch_ = dispatchEpoch,
                                       .numTokens_ = numTokens,
                                       .hidden_ = hidden,
                                       .numTopk_ = numTopk,
                                       .numExperts_ = numExperts,
                                       .invalidTokenExpertId_ = numExperts,
                                       .maxTokensPerRank_ = maxTokensPerRank,
                                       .outputLayout_ = dispatchLayout,
                                       .dispatchDataType_ = dispatchDataType};
  const size_t workspaceBytes = low_latency::workspaceSize(numRanks_, numExperts, maxTokensPerRank, numTopk);
  EP_HOST_ASSERT(workspaceBytes <= workspaceBytes_);
  low_latency::combine(output, input, topkIdx, topkWeights, srcInfo, layoutRange, workload, combineRecvBuffer,
                       dispatchRecvBuffer, commContext_, workspace_, numBlocks, mode, stream);
}

}  // namespace ep
}  // namespace mscclpp
