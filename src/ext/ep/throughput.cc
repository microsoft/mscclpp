// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP)
// branch `chhwang/dev-atomic-add-cleanup`. Licensed under the MIT License.

#include <cuda.h>

#include <future>
#include <limits>
#include <mscclpp/ext/ep/moe_runtime.hpp>
#include <mscclpp/gpu_utils.hpp>

#include "exception.hpp"
#include "kernels.hpp"
#include "moe_runtime_context.hpp"

namespace mscclpp {
namespace ep {

ThroughputRuntimeContext::ThroughputRuntimeContext(mscclpp::Communicator& communicator, int rank, int numRanks,
                                                   int numRanksPerIpcDomain, int maxTokensPerRank, int hidden,
                                                   int numExperts, int numTopk, DispatchLayout outputLayout)
    : rank_(rank),
      numRanks_(numRanks),
      numRanksPerIpcDomain_(numRanksPerIpcDomain),
      bootstrap_(communicator.bootstrap()),
      maxTokensPerRank_(maxTokensPerRank),
      hidden_(hidden),
      numExperts_(numExperts),
      numTopk_(numTopk),
      outputLayout_(outputLayout),
      communicator_(communicator) {
  EP_HOST_ASSERT(hidden_ > 0);
  EP_HOST_ASSERT(numExperts_ > 0 && numExperts_ % numRanks_ == 0);
  EP_HOST_ASSERT(numTopk_ > 0 && numTopk_ <= MaxNumTopk);
  EP_HOST_ASSERT(outputLayout_ == DispatchLayout::TOKEN_MAJOR || outputLayout_ == DispatchLayout::RANK_MAJOR);
  EP_HOST_ASSERT(maxTokensPerRank_ > 0);
  EP_HOST_ASSERT(static_cast<int64_t>(hidden_) * sizeof(Bf16) % sizeof(int4) == 0);

  if (!isSupportedThroughputRanks(numRanks_) || numRanksPerIpcDomain_ < numRanks_) {
    return;
  }

  available_ = fitsReceiveBuffer(maxTokensPerRank_);
  if (!available_) return;
  symmetricBufferBytes_ = storageLayout().totalBytes_;
  workspaceBytes_ = throughputWorkspaceSize(maxTokensPerRank_, numRanks_, numExperts_);
}

ThroughputRuntimeContext::~ThroughputRuntimeContext() noexcept(false) {
  if (deviceContext_.devicePtr_ == nullptr) return;

  CudaDeviceGuard deviceGuard(deviceContext_.deviceId_);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  bootstrap_->barrier();

  if (prepareEvent_ != nullptr) MSCCLPP_CUDATHROW(cudaEventDestroy(prepareEvent_));
  MSCCLPP_CUDATHROW(cudaFree(deviceContext_.devicePtr_));
  if (peerMappedBufferBasesGpu_ != nullptr) MSCCLPP_CUDATHROW(cudaFree(peerMappedBufferBasesGpu_));
  if (workspace_ != nullptr) MSCCLPP_CUDATHROW(cudaFree(workspace_));

  peerBufferMemories_.clear();
  if (symmetricBuffer_ != nullptr) mscclpp::detail::gpuFreePhysical(symmetricBuffer_);
}

void ThroughputRuntimeContext::initialize() {
  EP_HOST_ASSERT(available_);
  EP_HOST_ASSERT(symmetricBuffer_ == nullptr);
  AvoidCudaGraphCaptureGuard captureGuard;

  workspace_ = mscclpp::detail::gpuCalloc(workspaceBytes_);
  const size_t allocationGranularity = mscclpp::detail::getCuAllocationGranularity(CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  symmetricBuffer_ =
      mscclpp::detail::gpuCallocPhysical(symmetricBufferBytes_, allocationGranularity, allocationGranularity);

  constexpr int BufferTag = 17;
  constexpr int ConnectionTag = 19;
  const auto transport = mscclpp::Transport::CudaIpc;
  const mscclpp::EndpointConfig ipcConfig(transport);
  peerBufferMemories_.resize(numRanks_);
  peerBufferMemories_[rank_] = communicator_.registerMemory(symmetricBuffer_, symmetricBufferBytes_, transport);
  std::vector<std::shared_future<mscclpp::RegisteredMemory>> remoteMemories(numRanks_);
  std::vector<std::shared_future<mscclpp::Connection>> connections(numRanks_);
  for (int peer = 0; peer < numRanks_; ++peer) {
    if (peer == rank_) continue;
    communicator_.sendMemory(peerBufferMemories_[rank_], peer, BufferTag);
    remoteMemories[peer] = communicator_.recvMemory(peer, BufferTag);
    connections[peer] = communicator_.connect(ipcConfig, peer, ConnectionTag);
  }

  peerMappedBufferBases_.resize(numRanks_);
  baseMemoryChannels_.reserve(numRanks_ - 1);
  std::vector<mscclpp::BaseMemoryChannelDeviceHandle> baseMemoryChannelHandles(numRanks_);
  for (int peer = 0; peer < numRanks_; ++peer) {
    if (peer != rank_) {
      peerBufferMemories_[peer] = remoteMemories[peer].get();
    }
    peerMappedBufferBases_[peer] = peer == rank_ ? symmetricBuffer_ : peerBufferMemories_[peer].data();
    if (peer != rank_) {
      auto semaphore = std::make_shared<mscclpp::MemoryDevice2DeviceSemaphore>(communicator_, connections[peer].get());
      baseMemoryChannels_.emplace_back(semaphore);
      baseMemoryChannelHandles[peer] = baseMemoryChannels_.back().deviceHandle();
    }
  }

  peerMappedBufferBasesGpu_ =
      static_cast<void**>(mscclpp::detail::gpuCalloc(sizeof(void*) * static_cast<size_t>(numRanks_)));
  mscclpp::gpuMemcpy<void*>(peerMappedBufferBasesGpu_, peerMappedBufferBases_.data(), numRanks_,
                            cudaMemcpyHostToDevice);
  baseMemoryChannelHandles_ = mscclpp::detail::gpuCallocShared<mscclpp::BaseMemoryChannelDeviceHandle>(numRanks_);
  mscclpp::gpuMemcpy<mscclpp::BaseMemoryChannelDeviceHandle>(
      baseMemoryChannelHandles_.get(), baseMemoryChannelHandles.data(), numRanks_, cudaMemcpyHostToDevice);
  int deviceId;
  int maxSharedMemoryPerBlock;
  int numSms;
  MSCCLPP_CUDATHROW(cudaGetDevice(&deviceId));
  MSCCLPP_CUDATHROW(
      cudaDeviceGetAttribute(&maxSharedMemoryPerBlock, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceId));
  MSCCLPP_CUDATHROW(cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, deviceId));
  deviceContext_ = {.localBufferBase_ = symmetricBuffer_,
                    .peerBufferBases_ = peerMappedBufferBasesGpu_,
                    .channels_ = baseMemoryChannelHandles_.get(),
                    .workspace_ = workspace_,
                    .maxSharedMemoryPerBlock_ = maxSharedMemoryPerBlock,
                    .numSms_ = numSms,
                    .deviceId_ = deviceId,
                    .rank_ = rank_,
                    .numRanks_ = numRanks_};
  deviceContext_.devicePtr_ = static_cast<DeviceContext*>(mscclpp::detail::gpuCalloc(sizeof(DeviceContext)));
  mscclpp::gpuMemcpy<DeviceContext>(deviceContext_.devicePtr_, &deviceContext_, 1, cudaMemcpyHostToDevice);
  MSCCLPP_CUDATHROW(cudaEventCreateWithFlags(&prepareEvent_, cudaEventDisableTiming));
}

bool ThroughputRuntimeContext::fitsReceiveBuffer(int maxTokensPerRank) const {
  if (maxTokensPerRank <= 0 || maxTokensPerRank > maxTokensPerRank_) return false;
  const uint64_t hiddenBytes = static_cast<uint64_t>(hidden_) * sizeof(Bf16);
  const uint64_t maxRows = static_cast<uint64_t>(numRanks_) * maxTokensPerRank;
  // Kernel row sizes, route indices, and expert counters use signed int arithmetic.
  constexpr uint64_t MaxIndex = std::numeric_limits<int>::max();
  return hiddenBytes <= MaxIndex && maxRows * static_cast<uint64_t>(numTopk_) <= MaxIndex;
}

ThroughputStorageLayout ThroughputRuntimeContext::storageLayout() const {
  return {symmetricBuffer_, maxTokensPerRank_, hidden_, numRanks_, numExperts_, numTopk_};
}

Workload ThroughputRuntimeContext::makeWorkload(int numTokens, int maxTokensPerRank, DispatchDataType dataType) const {
  return {.epoch_ = epoch_,
          .numTokens_ = numTokens,
          .hidden_ = hidden_,
          .numTopk_ = numTopk_,
          .numExperts_ = numExperts_,
          .invalidTokenExpertId_ = -1,
          .maxTokensPerRank_ = maxTokensPerRank,
          .outputLayout_ = outputLayout_,
          .dispatchDataType_ = dataType};
}

void ThroughputRuntimeContext::validatePrepareRequest(const PrepareRequest& request) const {
  EP_HOST_ASSERT(available_);
  EP_HOST_ASSERT(deviceContext_.devicePtr_ != nullptr);
  if (request.maxTokensPerRank <= 0 || request.maxTokensPerRank > maxTokensPerRank_ || request.numTokens < 0 ||
      request.numTokens > request.maxTokensPerRank) {
    EP_THROW("Throughput requests require 0 <= numTokens <= maxTokensPerRank <= runtime capacity");
  }
  EP_HOST_ASSERT(request.numBlocks > 0 && request.numBlocks <= MaxDispatchBlocks);
  EP_HOST_ASSERT(numExperts_ / numRanks_ <= ThroughputCountThreads && numRanks_ <= ThroughputCountThreads);
  EP_HOST_ASSERT(request.numBlocks <= maxCooperativeThroughputDispatchBlocks(outputLayout_, deviceContext_));
  EP_HOST_ASSERT(request.topkIdx != nullptr || request.numTokens == 0);
  if (!fitsReceiveBuffer(request.maxTokensPerRank)) {
    EP_THROW("Throughput receive-buffer capacity exceeded for this runtime configuration");
  }
}

PrepareHandle MoERuntime::prepare(const PrepareRequest& request) {
  requireMode(MoEMode::THROUGHPUT);
  auto& context = *throughputContext_;
  context.validatePrepareRequest(request);
  cudaStreamCaptureStatus captureStatus;
  MSCCLPP_CUDATHROW(cudaStreamIsCapturing(request.stream, &captureStatus));
  const ThroughputWorkspaceLayout workspaceLayout(context.workspace_, context.maxTokensPerRank_, context.numRanks_,
                                                  context.numExperts_);
  EP_HOST_ASSERT(workspaceLayout.totalBytes_ <= context.workspaceBytes_);
  auto metadata = std::make_shared<PrepareHandle::Impl>(throughputContext_, context.prepareEpoch_ + 1, request);

  // Preparation replaces shared routing metadata, but reusing a preparation only
  // replaces dispatch results. Track those lifetimes independently.
  ++context.prepareEpoch_;
  ++context.epoch_;
  const Workload workload = context.makeWorkload(request.numTokens, request.maxTokensPerRank);
  throughputCountRoutes(request.topkIdx, workspaceLayout, workload, context.deviceContext_, request.stream);
  throughputExchangeCounts(workspaceLayout, workload, context.deviceContext_, request.stream);
  // Keep readiness on the device, including when preparation is captured in a graph.
  MSCCLPP_CUDATHROW(cudaEventRecordWithFlags(
      context.prepareEvent_, request.stream,
      captureStatus == cudaStreamCaptureStatusActive ? cudaEventRecordExternal : cudaEventRecordDefault));
  return PrepareHandle(std::move(metadata));
}

DispatchHandle MoERuntime::launchThroughputDispatch(const ThroughputDispatchRequest& request) {
  auto& context = *throughputContext_;
  const PrepareRequest prepareRequest{request.topkIdx, request.numTokens, request.maxTokensPerRank, request.numBlocks,
                                      request.stream};
  context.validatePrepareRequest(prepareRequest);
  EP_HOST_ASSERT(request.output != nullptr || request.numTokens == 0);
  EP_HOST_ASSERT(request.input != nullptr || request.numTokens == 0);
  EP_HOST_ASSERT(isSupportedDispatchDataType(request.dispatchDataType));

  const int elementBytes = dispatchElementBytes(request.dispatchDataType);
  const int hiddenBytes = context.hidden_ * elementBytes;
  if (hiddenBytes % static_cast<int>(sizeof(int4)) != 0) {
    EP_THROW("Throughput dispatch requires the row byte size to be a multiple of int4");
  }
  if (request.dispatchDataType == DispatchDataType::FP8_E4M3) {
    EP_HOST_ASSERT(context.hidden_ % dispatchElementsPerScale(request.dispatchDataType) == 0);
    EP_HOST_ASSERT(request.inputScales != nullptr || request.numTokens == 0);
  }

  const bool reusePreparation = request.prepareHandle.impl_ != nullptr;
  PrepareHandle preparation = request.prepareHandle;
  if (reusePreparation) {
    const auto& metadata = *preparation.impl_;
    const auto owner = metadata.owner_.lock();
    if (!owner) {
      EP_THROW("Expired preparation handle");
    }
    if (owner != throughputContext_) {
      EP_THROW("Preparation handle belongs to a different runtime");
    }
    if (metadata.epoch_ != context.prepareEpoch_) {
      EP_THROW("Stale preparation handle: a newer preparation has replaced its metadata");
    }
    if (metadata.topkIdx_ != request.topkIdx || metadata.numTokens_ != request.numTokens ||
        metadata.maxTokensPerRank_ != request.maxTokensPerRank || metadata.numBlocks_ != request.numBlocks) {
      EP_THROW("Dispatch routing IDs, token counts, capacity, and block count must match the preparation");
    }
  } else {
    preparation = prepare(prepareRequest);
  }

  const ThroughputWorkspaceLayout workspaceLayout(context.workspace_, context.maxTokensPerRank_, context.numRanks_,
                                                  context.numExperts_);
  if (reusePreparation) {
    cudaStreamCaptureStatus captureStatus;
    MSCCLPP_CUDATHROW(cudaStreamIsCapturing(request.stream, &captureStatus));
    MSCCLPP_CUDATHROW(cudaStreamWaitEvent(
        request.stream, context.prepareEvent_,
        captureStatus == cudaStreamCaptureStatusActive ? cudaEventWaitExternal : cudaEventWaitDefault));
    // Replays still need a peer handshake before overwriting the previous payload.
    throughputSynchronizePeers(context.deviceContext_, request.stream);
  }
  if (request.outputCount != nullptr) {
    const int numOutputCounts = context.outputLayout_ == DispatchLayout::TOKEN_MAJOR
                                    ? context.numExperts_ / context.numRanks_
                                    : context.numRanks_;
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(request.outputCount, workspaceLayout.recvCounts_,
                                      sizeof(int) * static_cast<size_t>(numOutputCounts), cudaMemcpyDeviceToDevice,
                                      request.stream));
  }

  const ThroughputStorageLayout storageLayout = context.storageLayout();
  const Workload workload = context.makeWorkload(request.numTokens, request.maxTokensPerRank, request.dispatchDataType);
  throughputDispatch(request.output, request.outputTopkIdx, request.outputTopkWeights,
                     static_cast<float*>(request.outputScales), request.input, request.topkIdx, request.topkWeights,
                     request.inputScales, workload, workspaceLayout, storageLayout.payload_, storageLayout.recvBuffer_,
                     context.deviceContext_, request.numBlocks, request.stream);

  ++context.epoch_;
  return DispatchHandle(std::make_shared<const DispatchHandle::Impl>(throughputContext_, context.epoch_, request));
}

void MoERuntime::launchThroughputCombine(const ThroughputCombineRequest& request) {
  auto& context = *throughputContext_;
  if (!request.handle.impl_) {
    EP_THROW("Invalid or expired dispatch handle");
  }
  const auto& handle = *request.handle.impl_;
  const auto owner = handle.owner_.lock();
  if (!owner) {
    EP_THROW("Invalid or expired dispatch handle");
  }
  if (owner != throughputContext_) {
    EP_THROW("Dispatch handle belongs to a different runtime");
  }
  if (handle.epoch_ != context.epoch_) {
    EP_THROW("Stale dispatch handle: a newer dispatch has replaced its metadata");
  }
  if (!std::holds_alternative<std::monostate>(handle.metadata_)) {
    EP_THROW("Dispatch handle does not contain throughput metadata");
  }

  EP_HOST_ASSERT(context.available_);
  EP_HOST_ASSERT(context.deviceContext_.devicePtr_ != nullptr);
  EP_HOST_ASSERT(request.numBlocks > 0 && request.numBlocks <= MaxWorkerBlocks);
  EP_HOST_ASSERT(request.output != nullptr || handle.numTokens_ == 0);

  const ThroughputStorageLayout storageLayout = context.storageLayout();
  const ThroughputWorkspaceLayout workspaceLayout(context.workspace_, context.maxTokensPerRank_, context.numRanks_,
                                                  context.numExperts_);
  const Workload workload = context.makeWorkload(handle.numTokens_, handle.maxTokensPerRank_, handle.dispatchDataType_);
  throughputReduceCombine(request.output, request.outputTopkWeights, request.input, workload, workspaceLayout,
                          storageLayout.payload_, storageLayout.recvBuffer_, context.deviceContext_, request.numBlocks,
                          request.stream);
}

}  // namespace ep
}  // namespace mscclpp
