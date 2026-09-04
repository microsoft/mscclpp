// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cuda.h>

#include <algorithm>
#include <future>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/ext/ep/moe_runtime.hpp>

#include "exception.hpp"
#include "kernels.hpp"
#include "moe_runtime_context.hpp"

namespace mscclpp {
namespace ep {

LatencyContext::LatencyContext(mscclpp::Communicator& communicator, int rank, int numRanks, int numNvlRanks,
                               int numRanksPerIpcDomain, int maxTokensPerRank, int hidden, int numExperts, int numTopk,
                               DispatchLayout outputLayout, CombineMode combineMode)
    : rank_(rank),
      numRanks_(numRanks),
      numRanksPerIpcDomain_(numRanksPerIpcDomain),
      maxTokensPerRank_(maxTokensPerRank),
      hidden_(hidden),
      numExperts_(numExperts),
      numTopk_(numTopk),
      outputLayout_(outputLayout),
      combineMode_(combineMode),
      symmetricBufferBytes_(static_cast<int64_t>(
          latencyStorageSize(maxTokensPerRank, hidden, numRanks_, numExperts, numTopk, outputLayout, combineMode))),
      workspaceBytes_(workspaceSize(numRanks_, numExperts, maxTokensPerRank, numTopk)),
      communicator_(&communicator) {
  EP_HOST_ASSERT(communicator_ != nullptr);
  EP_HOST_ASSERT(symmetricBufferBytes_ % BufferAlignmentBytes == 0);
  EP_HOST_ASSERT(maxTokensPerRank > 0);
  EP_HOST_ASSERT(numExperts > 0 && numExperts % numRanks_ == 0);
  EP_HOST_ASSERT(numTopk > 0 && numTopk <= 32);
  EP_HOST_ASSERT(outputLayout == DispatchLayout::EXPERT_MAJOR || outputLayout == DispatchLayout::RANK_MAJOR ||
                 outputLayout == DispatchLayout::TOKEN_MAJOR);

  CUDA_CHECK(cudaGetDevice(&deviceId_));
  EP_HOST_ASSERT(numRanks_ % numNvlRanks == 0);
  EP_HOST_ASSERT(numRanks_ % numRanksPerIpcDomain_ == 0);
  available_ = numRanksPerIpcDomain_ >= numRanks_;
}

LatencyContext::~LatencyContext() noexcept(false) {
  CUDA_CHECK(cudaDeviceSynchronize());
  if (deviceContext_.devicePtr_ != nullptr) CUDA_CHECK(cudaFree(deviceContext_.devicePtr_));
  if (peerMappedBufferBasesGpu_ != nullptr) CUDA_CHECK(cudaFree(peerMappedBufferBasesGpu_));
  if (workspace_ != nullptr) CUDA_CHECK(cudaFree(workspace_));
  if (symmetricBuffer_ != nullptr) {
    mscclpp::detail::gpuFreePhysical(symmetricBuffer_);
  }
}

void LatencyContext::initialize() {
  EP_HOST_ASSERT(available_);
  EP_HOST_ASSERT(symmetricBuffer_ == nullptr);
  EP_HOST_ASSERT(communicator_ != nullptr);
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

  peerMappedBufferBasesGpu_ =
      static_cast<void**>(mscclpp::detail::gpuCalloc(sizeof(void*) * static_cast<size_t>(numRanks_)));
  mscclpp::gpuMemcpy<void*>(peerMappedBufferBasesGpu_, peerMappedBufferBases_.data(), numRanks_,
                            cudaMemcpyHostToDevice);
  baseMemoryChannelHandles_ = mscclpp::detail::gpuCallocShared<mscclpp::BaseMemoryChannelDeviceHandle>(numRanks_);
  mscclpp::gpuMemcpy<mscclpp::BaseMemoryChannelDeviceHandle>(
      baseMemoryChannelHandles_.get(), baseMemoryChannelHandles.data(), numRanks_, cudaMemcpyHostToDevice);

  int maxSharedMemoryPerBlock;
  int numSms;
  CUDA_CHECK(cudaDeviceGetAttribute(&maxSharedMemoryPerBlock, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceId_));
  CUDA_CHECK(cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, deviceId_));
  deviceContext_ = {.localBufferBase_ = symmetricBuffer_,
                    .peerBufferBases_ = peerMappedBufferBasesGpu_,
                    .peerPayloadBases_ = nullptr,
                    .channels_ = baseMemoryChannelHandles_.get(),
                    .workspace_ = workspace_,
                    .combineRecvIdx_ = nullptr,
                    .mappedRecvCounter_ = nullptr,
                    .mappedRecvExpertCounters_ = nullptr,
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
  EP_HOST_ASSERT(context.symmetricBuffer_ != nullptr);
  return LatencyStorageLayout(context.symmetricBuffer_, context.maxTokensPerRank_, context.hidden_, context.numRanks_,
                              context.numExperts_, context.numTopk_, context.outputLayout_, context.combineMode_)
      .rankMajorTopkIdsBuffer_;
}

void* MoERuntime::outputTopkWeightsBuffer() const {
  requireMode(MoEMode::LATENCY);
  const auto& context = *latencyContext_;
  EP_HOST_ASSERT(context.symmetricBuffer_ != nullptr);
  return LatencyStorageLayout(context.symmetricBuffer_, context.maxTokensPerRank_, context.hidden_, context.numRanks_,
                              context.numExperts_, context.numTopk_, context.outputLayout_, context.combineMode_)
      .rankMajorTopkWeightsBuffer_;
}

void* MoERuntime::combineInputBuffer() const {
  requireMode(MoEMode::LATENCY);
  const auto& context = *latencyContext_;
  EP_HOST_ASSERT(context.outputLayout_ == DispatchLayout::RANK_MAJOR);
  EP_HOST_ASSERT(context.symmetricBuffer_ != nullptr);
  return LatencyStorageLayout(context.symmetricBuffer_, context.maxTokensPerRank_, context.hidden_, context.numRanks_,
                              context.numExperts_, context.numTopk_, context.outputLayout_, context.combineMode_)
      .combineRecvBuffer_;
}

void MoERuntime::launchLatencyDispatch(const LatencyDispatchRequest& request) {
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
  const int hidden = request.hidden;
  const int numTopk = request.numTopk;
  const int maxTokensPerRank = request.maxTokensPerRank;
  const int numExperts = request.numExperts;
  const int invalidTokenExpertId = request.invalidTokenExpertId;
  const DispatchLayout dispatchLayout = request.dispatchLayout;
  const DispatchDataType dispatchDataType = request.dispatchDataType;
  const int numBlocks = request.numBlocks;
  const cudaStream_t stream = request.stream;

  auto& context = *latencyContext_;
  EP_HOST_ASSERT(context.available_);
  EP_HOST_ASSERT(context.deviceContext_.devicePtr_ != nullptr);
  EP_HOST_ASSERT(maxTokensPerRank > 0 && maxTokensPerRank <= context.maxTokensPerRank_);
  EP_HOST_ASSERT(numTokens <= maxTokensPerRank);
  EP_HOST_ASSERT(numExperts % context.numRanks_ == 0);
  EP_HOST_ASSERT(invalidTokenExpertId < 0 || invalidTokenExpertId >= numExperts);
  EP_HOST_ASSERT(numBlocks - DispatchControlBlocks >= numRanks_ && numBlocks <= MaxDispatchBlocks);
  EP_HOST_ASSERT(dispatchLayout == context.outputLayout_);

  LatencyStorageLayout allocationLayout(context.symmetricBuffer_, context.maxTokensPerRank_, hidden, context.numRanks_,
                                        numExperts, numTopk, context.outputLayout_, context.combineMode_);
  EP_HOST_ASSERT(allocationLayout.totalBytes_ <= static_cast<size_t>(context.symmetricBufferBytes_));
  void* dispatchRecvBuffer = allocationLayout.dispatchRecvBuffer_;
  if (dispatchLayout == DispatchLayout::RANK_MAJOR) {
    EP_HOST_ASSERT(output == allocationLayout.dispatchOutputBuffer_);
    EP_HOST_ASSERT(outputTopkIdx == allocationLayout.rankMajorTopkIdsBuffer_);
    EP_HOST_ASSERT(outputTopkWeights == allocationLayout.rankMajorTopkWeightsBuffer_);
  } else if (dispatchLayout == DispatchLayout::TOKEN_MAJOR) {
    EP_HOST_ASSERT(output == allocationLayout.dispatchOutputBuffer_);
    EP_HOST_ASSERT(outputTopkIdx == allocationLayout.rankMajorTopkIdsBuffer_);
    EP_HOST_ASSERT(outputTopkWeights == allocationLayout.rankMajorTopkWeightsBuffer_);
  }

  ++context.epoch_;
  const Workload workload{.epoch_ = context.epoch_,
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
  if (dispatchLayout == DispatchLayout::RANK_MAJOR) {
    rankMajorDispatch(output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount,
                      input, topkIdx, topkWeights, workload, dispatchRecvBuffer, context.deviceContext_, numBlocks,
                      stream);
  } else if (dispatchLayout == DispatchLayout::TOKEN_MAJOR) {
    tokenMajorDispatch(output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount,
                       input, topkIdx, topkWeights, workload, dispatchRecvBuffer, context.deviceContext_, numBlocks,
                       stream);
  } else {
    expertMajorDispatch(output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout,
                        outputCount, input, topkIdx, topkWeights, workload, dispatchRecvBuffer, context.deviceContext_,
                        numBlocks, stream);
  }
}

void MoERuntime::launchLatencyCombine(const LatencyCombineRequest& request) {
  void* output = request.output;
  const void* input = request.input;
  const int64_t* topkIdx = request.topkIdx;
  const float* topkWeights = request.topkWeights;
  const int* srcInfo = request.srcInfo;
  const int64_t* layoutRange = request.layoutRange;
  const int numTokens = request.numTokens;
  const int hidden = request.hidden;
  const int numTopk = request.numTopk;
  const int maxTokensPerRank = request.maxTokensPerRank;
  const int numExperts = request.numExperts;
  const DispatchLayout dispatchLayout = request.dispatchLayout;
  const DispatchDataType dispatchDataType = request.dispatchDataType;
  const CombineMode mode = request.combineMode;
  const int numBlocks = request.numBlocks;
  const cudaStream_t stream = request.stream;

  auto& context = *latencyContext_;
  EP_HOST_ASSERT(context.available_);
  EP_HOST_ASSERT(context.deviceContext_.devicePtr_ != nullptr);
  EP_HOST_ASSERT(maxTokensPerRank > 0 && maxTokensPerRank <= context.maxTokensPerRank_);
  EP_HOST_ASSERT(numExperts % context.numRanks_ == 0);
  EP_HOST_ASSERT(numBlocks > 0 && numBlocks <= MaxWorkerBlocks);
  EP_HOST_ASSERT(dispatchLayout == context.outputLayout_);
  EP_HOST_ASSERT(mode == context.combineMode_);

  LatencyStorageLayout allocationLayout(context.symmetricBuffer_, context.maxTokensPerRank_, hidden, context.numRanks_,
                                        numExperts, numTopk, context.outputLayout_, context.combineMode_);
  EP_HOST_ASSERT(allocationLayout.totalBytes_ <= static_cast<size_t>(context.symmetricBufferBytes_));
  void* combineRecvBuffer = allocationLayout.combineRecvBuffer_;
  void* dispatchRecvBuffer = allocationLayout.dispatchRecvBuffer_;
  if (dispatchLayout == DispatchLayout::RANK_MAJOR) {
    EP_HOST_ASSERT(input == allocationLayout.combineRecvBuffer_);
  } else if (dispatchLayout == DispatchLayout::TOKEN_MAJOR) {
    EP_HOST_ASSERT(input == allocationLayout.dispatchOutputBuffer_);
  }

  const Workload workload{.epoch_ = context.epoch_,
                          .numTokens_ = numTokens,
                          .hidden_ = hidden,
                          .numTopk_ = numTopk,
                          .numExperts_ = numExperts,
                          .invalidTokenExpertId_ = numExperts,
                          .maxTokensPerRank_ = maxTokensPerRank,
                          .outputLayout_ = dispatchLayout,
                          .dispatchDataType_ = dispatchDataType};
  const size_t workspaceBytes = workspaceSize(context.numRanks_, numExperts, maxTokensPerRank, numTopk);
  EP_HOST_ASSERT(workspaceBytes <= context.workspaceBytes_);
  if (dispatchLayout == DispatchLayout::RANK_MAJOR) {
    if (mode == CombineMode::DIRECT_SEND) {
      rankMajorDirectSendCombine(output, input, topkIdx, workload, combineRecvBuffer, dispatchRecvBuffer,
                                 context.deviceContext_, numBlocks, stream);
    } else {
      EP_HOST_ASSERT(mode == CombineMode::RANK_LOCAL_REDUCE);
      rankMajorGatherReduceCombine(output, input, topkIdx, topkWeights, srcInfo, layoutRange, workload,
                                   combineRecvBuffer, dispatchRecvBuffer, context.deviceContext_, numBlocks, stream);
    }
  } else if (dispatchLayout == DispatchLayout::TOKEN_MAJOR) {
    EP_HOST_ASSERT(mode == CombineMode::RANK_LOCAL_REDUCE);
    tokenMajorGatherReduceCombine(output, input, topkIdx, topkWeights, workload, combineRecvBuffer, dispatchRecvBuffer,
                                  context.deviceContext_, numBlocks, stream);
  } else if (mode == CombineMode::DIRECT_SEND) {
    expertMajorDirectSendCombine(output, input, topkIdx, topkWeights, srcInfo, layoutRange, workload, combineRecvBuffer,
                                 dispatchRecvBuffer, context.deviceContext_, numBlocks, stream);
  } else {
    expertMajorLocalReduceCombine(output, input, topkIdx, topkWeights, srcInfo, layoutRange, workload,
                                  combineRecvBuffer, dispatchRecvBuffer, context.deviceContext_, numBlocks, stream);
  }
}

}  // namespace ep
}  // namespace mscclpp
