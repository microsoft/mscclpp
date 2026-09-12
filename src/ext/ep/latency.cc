// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cuda.h>

#include <future>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/ext/ep/moe_runtime.hpp>

#include "exception.hpp"
#include "kernels.hpp"
#include "moe_runtime_context.hpp"

namespace mscclpp {
namespace ep {

LatencyRuntimeContext::LatencyRuntimeContext(mscclpp::Communicator& communicator, int rank, int numRanks,
                                             int numNvlRanks, int numRanksPerIpcDomain, int maxTokensPerRank,
                                             int hidden, int numExperts, int numTopk, DispatchLayout outputLayout,
                                             CombineMode combineMode)
    : rank_(rank),
      numRanks_(numRanks),
      numRanksPerIpcDomain_(numRanksPerIpcDomain),
      maxTokensPerRank_(maxTokensPerRank),
      hidden_(hidden),
      numExperts_(numExperts),
      numTopk_(numTopk),
      outputLayout_(outputLayout),
      combineMode_(combineMode),
      symmetricBufferBytes_(0),
      workspaceBytes_(0),
      communicator_(communicator) {
  if (!isSupportedHidden(hidden)) {
    EP_THROW("Unsupported latency hidden size: " + std::to_string(hidden));
  }
  EP_HOST_ASSERT(maxTokensPerRank > 0);
  EP_HOST_ASSERT(numExperts > 0 && numExperts % numRanks_ == 0);
  EP_HOST_ASSERT(numTopk > 0 && numTopk <= MaxNumTopk);
  EP_HOST_ASSERT(outputLayout == DispatchLayout::EXPERT_MAJOR || outputLayout == DispatchLayout::RANK_MAJOR);
  EP_HOST_ASSERT(combineMode == CombineMode::RANK_LOCAL_REDUCE || combineMode == CombineMode::DIRECT_SEND);

  symmetricBufferBytes_ = static_cast<int64_t>(
      latencyStorageSize(maxTokensPerRank, hidden, numRanks_, numExperts, numTopk, outputLayout, combineMode));
  workspaceBytes_ = workspaceSize(numRanks_, numExperts, maxTokensPerRank, numTopk);
  EP_HOST_ASSERT(symmetricBufferBytes_ % BufferAlignmentBytes == 0);

  MSCCLPP_CUDATHROW(cudaGetDevice(&deviceId_));
  EP_HOST_ASSERT(numRanks_ % numNvlRanks == 0);
  EP_HOST_ASSERT(numRanks_ % numRanksPerIpcDomain_ == 0);
  available_ = numRanksPerIpcDomain_ >= numRanks_;
}

LatencyRuntimeContext::~LatencyRuntimeContext() noexcept(false) {
  CudaDeviceGuard deviceGuard(deviceId_);
  if (deviceContext_.devicePtr_ != nullptr) MSCCLPP_CUDATHROW(cudaFree(deviceContext_.devicePtr_));
  if (peerMappedBufferBasesGpu_ != nullptr) MSCCLPP_CUDATHROW(cudaFree(peerMappedBufferBasesGpu_));
  if (workspace_ != nullptr) MSCCLPP_CUDATHROW(cudaFree(workspace_));
  if (symmetricBuffer_ != nullptr) {
    mscclpp::detail::gpuFreePhysical(symmetricBuffer_);
  }
}

void LatencyRuntimeContext::initialize() {
  EP_HOST_ASSERT(available_);
  EP_HOST_ASSERT(symmetricBuffer_ == nullptr);
  AvoidCudaGraphCaptureGuard captureGuard;

  workspace_ = mscclpp::detail::gpuCalloc(workspaceBytes_);

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
  peerBufferMemories_[rank_] = communicator_.registerMemory(symmetricBuffer_, symmetricBufferBytes_, ipcTransport);
  std::vector<std::shared_future<mscclpp::RegisteredMemory>> remoteFutures(numRanks_);
  std::vector<std::shared_future<mscclpp::Connection>> connectionFutures(numRanks_);
  for (int r = 0; r < numRanks_; ++r) {
    if (!isMappedPeer(r)) continue;
    communicator_.sendMemory(peerBufferMemories_[rank_], r, IpcTag);
    remoteFutures[r] = communicator_.recvMemory(r, IpcTag);
    connectionFutures[r] = communicator_.connect(ipcConfig, r, IpcTag);
  }

  peerMappedBufferBases_.assign(numRanks_, nullptr);
  peerMappedBufferBases_[rank_] = symmetricBuffer_;
  std::vector<mscclpp::BaseMemoryChannelDeviceHandle> baseMemoryChannelHandles(numRanks_);
  for (int r = 0; r < numRanks_; ++r) {
    if (!isMappedPeer(r)) continue;
    peerBufferMemories_[r] = remoteFutures[r].get();
    peerMappedBufferBases_[r] = peerBufferMemories_[r].data();
    auto semaphore = std::make_shared<mscclpp::MemoryDevice2DeviceSemaphore>(communicator_, connectionFutures[r].get());
    baseMemoryChannels_.emplace_back(semaphore);
    baseMemoryChannelHandles[r] = baseMemoryChannels_.back().deviceHandle();
  }

  peerMappedBufferBasesGpu_ =
      static_cast<void**>(mscclpp::detail::gpuCalloc(sizeof(void*) * static_cast<size_t>(numRanks_)));
  mscclpp::gpuMemcpy<void*>(peerMappedBufferBasesGpu_, peerMappedBufferBases_.data(), numRanks_,
                            cudaMemcpyHostToDevice);
  baseMemoryChannelHandles_ = mscclpp::detail::gpuCallocShared<mscclpp::BaseMemoryChannelDeviceHandle>(numRanks_);
  mscclpp::gpuMemcpy<mscclpp::BaseMemoryChannelDeviceHandle>(
      baseMemoryChannelHandles_.get(), baseMemoryChannelHandles.data(), numRanks_, cudaMemcpyHostToDevice);

  int maxSharedMemoryPerBlock;
  int numSms;
  MSCCLPP_CUDATHROW(
      cudaDeviceGetAttribute(&maxSharedMemoryPerBlock, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceId_));
  MSCCLPP_CUDATHROW(cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, deviceId_));
  deviceContext_ = {.localBufferBase_ = symmetricBuffer_,
                    .peerBufferBases_ = peerMappedBufferBasesGpu_,
                    .channels_ = baseMemoryChannelHandles_.get(),
                    .workspace_ = workspace_,
                    .maxSharedMemoryPerBlock_ = maxSharedMemoryPerBlock,
                    .numSms_ = numSms,
                    .deviceId_ = deviceId_,
                    .rank_ = rank_,
                    .numRanks_ = numRanks_};
  deviceContext_.devicePtr_ = static_cast<DeviceContext*>(mscclpp::detail::gpuCalloc(sizeof(DeviceContext)));
  mscclpp::gpuMemcpy<DeviceContext>(deviceContext_.devicePtr_, &deviceContext_, 1, cudaMemcpyHostToDevice);
}

void* MoERuntime::outputTopkIdsBuffer() const {
  requireMode(MoEMode::LATENCY);
  const auto& context = *latencyContext_;
  EP_HOST_ASSERT(context.outputLayout_ == DispatchLayout::RANK_MAJOR);
  EP_HOST_ASSERT(context.symmetricBuffer_ != nullptr);
  return LatencyStorageLayout(context.symmetricBuffer_, context.maxTokensPerRank_, context.hidden_, context.numRanks_,
                              context.numExperts_, context.numTopk_, context.outputLayout_, context.combineMode_)
      .rankMajorTopkIdsBuffer_;
}

void* MoERuntime::outputTopkWeightsBuffer() const {
  requireMode(MoEMode::LATENCY);
  const auto& context = *latencyContext_;
  EP_HOST_ASSERT(context.outputLayout_ == DispatchLayout::RANK_MAJOR);
  EP_HOST_ASSERT(context.symmetricBuffer_ != nullptr);
  return LatencyStorageLayout(context.symmetricBuffer_, context.maxTokensPerRank_, context.hidden_, context.numRanks_,
                              context.numExperts_, context.numTopk_, context.outputLayout_, context.combineMode_)
      .rankMajorTopkWeightsBuffer_;
}

void* MoERuntime::combineInputBuffer() const {
  switch (mode_) {
    case MoEMode::LATENCY: {
      const auto& context = *latencyContext_;
      EP_HOST_ASSERT(context.outputLayout_ == DispatchLayout::RANK_MAJOR);
      EP_HOST_ASSERT(context.symmetricBuffer_ != nullptr);
      return LatencyStorageLayout(context.symmetricBuffer_, context.maxTokensPerRank_, context.hidden_,
                                  context.numRanks_, context.numExperts_, context.numTopk_, context.outputLayout_,
                                  context.combineMode_)
          .combineBuffer_;
    }
    case MoEMode::THROUGHPUT: {
      const auto& context = *throughputContext_;
      EP_HOST_ASSERT(context.deviceContext_.devicePtr_ != nullptr);
      return context.storageLayout().recvBuffer_;
    }
    default:
      EP_THROW("Unsupported MoE runtime mode");
  }
}

DispatchHandle MoERuntime::launchLatencyDispatch(const LatencyDispatchRequest& request) {
  auto& context = *latencyContext_;
  void* output = request.output;
  void* outputScales = request.outputScales;
  int* outputSrcInfo = request.outputSrcInfo;
  int* outputTopkIdx = request.outputTopkIdx;
  float* outputTopkWeights = request.outputTopkWeights;
  int64_t* outputLayout = request.outputLayoutRange;
  int* outputCount = request.outputCount;
  const void* input = request.input;
  const int64_t* topkIdx = request.topkIdx;
  const float* topkWeights = request.topkWeights;
  const int numTokens = request.numTokens;
  const int hidden = context.hidden_;
  const int numTopk = context.numTopk_;
  const int maxTokensPerRank = request.maxTokensPerRank;
  const int numExperts = context.numExperts_;
  const int invalidTokenExpertId = request.invalidTokenExpertId;
  const DispatchLayout dispatchLayout = context.outputLayout_;
  const DispatchDataType dispatchDataType = request.dispatchDataType;
  const int numBlocks = request.numBlocks;
  const cudaStream_t stream = request.stream;

  EP_HOST_ASSERT(context.available_);
  EP_HOST_ASSERT(context.deviceContext_.devicePtr_ != nullptr);
  if (maxTokensPerRank <= 0 || maxTokensPerRank > context.maxTokensPerRank_ || numTokens < 0 ||
      numTokens > maxTokensPerRank) {
    EP_THROW("Latency requests require 0 <= numTokens <= maxTokensPerRank <= runtime capacity");
  }
  EP_HOST_ASSERT(invalidTokenExpertId < 0 || invalidTokenExpertId >= numExperts);
  EP_HOST_ASSERT(numBlocks - DispatchControlBlocks >= numRanks_ && numBlocks <= MaxDispatchBlocks);

  LatencyStorageLayout allocationLayout(context.symmetricBuffer_, context.maxTokensPerRank_, hidden, context.numRanks_,
                                        numExperts, numTopk, context.outputLayout_, context.combineMode_);
  EP_HOST_ASSERT(allocationLayout.totalBytes_ <= static_cast<size_t>(context.symmetricBufferBytes_));
  void* dispatchRecvBuffer = allocationLayout.dispatchRecvBuffer_;
  if (dispatchLayout == DispatchLayout::RANK_MAJOR) {
    EP_HOST_ASSERT(output == allocationLayout.dispatchOutputBuffer_);
    EP_HOST_ASSERT(outputTopkIdx == allocationLayout.rankMajorTopkIdsBuffer_);
    EP_HOST_ASSERT(outputTopkWeights == allocationLayout.rankMajorTopkWeightsBuffer_);
  }

  const Workload workload{.epoch_ = context.epoch_ + 1,
                          .numTokens_ = numTokens,
                          .hidden_ = hidden,
                          .numTopk_ = numTopk,
                          .numExperts_ = numExperts,
                          .invalidTokenExpertId_ = invalidTokenExpertId,
                          .maxTokensPerRank_ = maxTokensPerRank,
                          .outputLayout_ = dispatchLayout,
                          .dispatchDataType_ = dispatchDataType};
  const size_t workspaceBytes = workspaceSize(context.numRanks_, numExperts, maxTokensPerRank, numTopk);
  EP_HOST_ASSERT(workspaceBytes <= context.workspaceBytes_);
  DispatchHandle handle(std::make_shared<const DispatchHandle::Impl>(latencyContext_, workload.epoch_, request));
  if (dispatchLayout == DispatchLayout::RANK_MAJOR) {
    rankMajorDispatch(output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount,
                      input, topkIdx, topkWeights, workload, dispatchRecvBuffer, context.deviceContext_, numBlocks,
                      stream);
  } else {
    expertMajorDispatch(output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout,
                        outputCount, input, topkIdx, topkWeights, workload, dispatchRecvBuffer, context.deviceContext_,
                        numBlocks, stream);
  }
  context.epoch_ = workload.epoch_;
  return handle;
}

void MoERuntime::launchLatencyCombine(const LatencyCombineRequest& request) {
  auto& context = *latencyContext_;
  if (!request.handle.impl_) {
    EP_THROW("Invalid or expired dispatch handle");
  }
  const auto& handle = *request.handle.impl_;
  const auto owner = handle.owner_.lock();
  if (!owner) {
    EP_THROW("Invalid or expired dispatch handle");
  }
  if (owner != latencyContext_) {
    EP_THROW("Dispatch handle belongs to a different runtime");
  }
  if (handle.epoch_ != context.epoch_) {
    EP_THROW("Stale dispatch handle: a newer dispatch has replaced its metadata");
  }

  const auto* latencyMetadata = std::get_if<DispatchHandle::Impl::LatencyMetadata>(&handle.metadata_);
  if (latencyMetadata == nullptr) {
    EP_THROW("Dispatch handle does not contain latency metadata");
  }

  void* output = request.output;
  const void* input = request.input;
  const int64_t* topkIdx = latencyMetadata->topkIdx_;
  const float* topkWeights = latencyMetadata->topkWeights_;
  const int* srcInfo = latencyMetadata->srcInfo_;
  const int64_t* layoutRange = latencyMetadata->layoutRange_;
  const int numTokens = handle.numTokens_;
  const int hidden = context.hidden_;
  const int numTopk = context.numTopk_;
  const int maxTokensPerRank = handle.maxTokensPerRank_;
  const int numExperts = context.numExperts_;
  const DispatchLayout dispatchLayout = context.outputLayout_;
  const DispatchDataType dispatchDataType = handle.dispatchDataType_;
  const CombineMode mode = context.combineMode_;
  const int numBlocks = request.numBlocks;
  const cudaStream_t stream = request.stream;

  EP_HOST_ASSERT(context.available_);
  EP_HOST_ASSERT(context.deviceContext_.devicePtr_ != nullptr);
  const Workload workload{.epoch_ = handle.epoch_,
                          .numTokens_ = numTokens,
                          .hidden_ = hidden,
                          .numTopk_ = numTopk,
                          .numExperts_ = numExperts,
                          .invalidTokenExpertId_ = numExperts,
                          .maxTokensPerRank_ = maxTokensPerRank,
                          .outputLayout_ = dispatchLayout,
                          .dispatchDataType_ = dispatchDataType};
  EP_HOST_ASSERT(numBlocks > 0 && numBlocks <= MaxWorkerBlocks);

  LatencyStorageLayout allocationLayout(context.symmetricBuffer_, context.maxTokensPerRank_, hidden, context.numRanks_,
                                        numExperts, numTopk, context.outputLayout_, context.combineMode_);
  EP_HOST_ASSERT(allocationLayout.totalBytes_ <= static_cast<size_t>(context.symmetricBufferBytes_));
  void* combineBuffer = allocationLayout.combineBuffer_;
  void* dispatchRecvBuffer = allocationLayout.dispatchRecvBuffer_;
  if (dispatchLayout == DispatchLayout::RANK_MAJOR) {
    EP_HOST_ASSERT(input == allocationLayout.combineBuffer_);
  }

  if (dispatchLayout == DispatchLayout::RANK_MAJOR) {
    if (mode == CombineMode::DIRECT_SEND) {
      rankMajorDirectSendCombine(output, input, topkIdx, workload, combineBuffer, dispatchRecvBuffer,
                                 context.deviceContext_, numBlocks, stream);
    } else {
      EP_HOST_ASSERT(mode == CombineMode::RANK_LOCAL_REDUCE);
      rankMajorGatherReduceCombine(output, input, topkIdx, workload, combineBuffer, dispatchRecvBuffer,
                                   context.deviceContext_, numBlocks, stream);
    }
  } else if (mode == CombineMode::DIRECT_SEND) {
    expertMajorDirectSendCombine(output, input, topkIdx, topkWeights, srcInfo, layoutRange, workload, combineBuffer,
                                 dispatchRecvBuffer, context.deviceContext_, numBlocks, stream);
  } else {
    expertMajorLocalReduceCombine(output, input, topkIdx, topkWeights, srcInfo, layoutRange, workload, combineBuffer,
                                  dispatchRecvBuffer, context.deviceContext_, numBlocks, stream);
  }
}

}  // namespace ep
}  // namespace mscclpp
