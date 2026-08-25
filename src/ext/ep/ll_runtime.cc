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

namespace {

bool isSupportedLowLatencyHidden(int hidden) {
  switch (hidden) {
    case 2048:
    case 4096:
    case 6144:
    case 6656:
    case 7168:
    case 8192:
    case 8704:
    case 9216:
      return true;
    default:
      return false;
  }
}

}  // namespace

MoELowLatencyRuntime::MoELowLatencyRuntime(mscclpp::Communicator& communicator, int maxTokensPerRank, int hidden,
                                           int numExperts, int numTopk)
    : MoERuntime(communicator),
      maxTokensPerRank_(maxTokensPerRank),
      hidden_(hidden),
      numExperts_(numExperts),
      numTopk_(numTopk),
      symmetricBufferBytes_(static_cast<int64_t>(
          low_latency::symmetricBufferSize(maxTokensPerRank, hidden, numRanks_, numExperts, numTopk))),
      workspaceBytes_(low_latency::workspaceSize(numRanks_, numExperts, maxTokensPerRank, numTopk)),
      communicator_(&communicator) {
  EP_HOST_ASSERT(communicator_ != nullptr);
  EP_HOST_ASSERT(symmetricBufferBytes_ % BufferAlignmentBytes == 0);
  EP_HOST_ASSERT(maxTokensPerRank > 0);
  EP_HOST_ASSERT(isSupportedLowLatencyHidden(hidden));
  EP_HOST_ASSERT(hidden % low_latency::Fp8DeepGemmScaleBlockSize == 0);
  EP_HOST_ASSERT(numExperts > 0 && numExperts % numRanks_ == 0);
  EP_HOST_ASSERT(numTopk > 0 && numTopk <= 32);

  CUDA_CHECK(cudaGetDevice(&deviceId_));
  EP_HOST_ASSERT(numRanks_ % numNvlRanks_ == 0);
  EP_HOST_ASSERT(numRanks_ % numRanksPerIpcDomain_ == 0);

  CUDA_CHECK(cudaMalloc(&workspace_, workspaceBytes_));
  CUDA_CHECK(cudaMemset(workspace_, 0, workspaceBytes_));
  low_latency::ExecutionReceipt initialReceipt{};
  initialReceipt.abiVersion_ = low_latency::Fp8DeepGemmAbi;
  initialReceipt.lastScaleBlockSize_ = low_latency::Fp8DeepGemmScaleBlockSize;
  CUDA_CHECK(cudaMemcpy(workspace_, &initialReceipt, sizeof(initialReceipt), cudaMemcpyHostToDevice));
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
  return low_latency::Layout(symmetricBuffer_, maxTokensPerRank_, hidden_, numRanks_, numExperts_, numTopk_)
      .rankMajorTopkIdsBuffer_;
}
void* MoELowLatencyRuntime::outputTopkWeightsBuffer() const {
  return low_latency::Layout(symmetricBuffer_, maxTokensPerRank_, hidden_, numRanks_, numExperts_, numTopk_)
      .rankMajorTopkWeightsBuffer_;
}
void* MoELowLatencyRuntime::outputTokensBuffer() const {
  return low_latency::Layout(symmetricBuffer_, maxTokensPerRank_, hidden_, numRanks_, numExperts_, numTopk_)
      .rankMajorTokenBuffer_;
}
void* MoELowLatencyRuntime::expertOutputBuffer() const {
  return low_latency::Layout(symmetricBuffer_, maxTokensPerRank_, hidden_, numRanks_, numExperts_, numTopk_)
      .rankMajorExpertOutputBuffer_;
}

low_latency::ExecutionReceipt MoELowLatencyRuntime::executionReceipt(cudaStream_t stream) const {
  EP_HOST_ASSERT(workspace_ != nullptr);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  low_latency::ExecutionReceipt receipt{};
  CUDA_CHECK(cudaMemcpy(&receipt, workspace_, sizeof(receipt), cudaMemcpyDeviceToHost));
  EP_HOST_ASSERT(receipt.abiVersion_ == low_latency::Fp8DeepGemmAbi);
  return receipt;
}

void MoELowLatencyRuntime::setup() {
  EP_HOST_ASSERT(!available_);
  EP_HOST_ASSERT(communicator_ != nullptr);

  const auto ipcTransport = mscclpp::Transport::CudaIpc;
  symmetricBuffer_ = mscclpp::detail::gpuCallocPhysical(symmetricBufferBytes_);

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

void MoELowLatencyRuntime::dispatch(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                                    float* outputTopkWeights, int64_t* outputLayout, int* outputCount,
                                    const void* input, const int64_t* topkIdx, const float* topkWeights, int numTokens,
                                    int hidden, int numTopk, int maxTokensPerRank, int numExperts,
                                    int invalidTokenExpertId, DispatchLayout dispatchLayout,
                                    low_latency::DispatchDataType dispatchDataType, int numBlocks,
                                    cudaStream_t stream) {
  EP_HOST_ASSERT(available_);
  EP_HOST_ASSERT(numTokens >= 0 && numTokens <= maxTokensPerRank);
  EP_HOST_ASSERT(hidden == hidden_ && numTopk == numTopk_ && maxTokensPerRank == maxTokensPerRank_ &&
                 numExperts == numExperts_);
  EP_HOST_ASSERT(numExperts % numRanks_ == 0);
  EP_HOST_ASSERT(invalidTokenExpertId < 0 || invalidTokenExpertId >= numExperts);
  EP_HOST_ASSERT(numBlocks - low_latency::DispatchControlBlocks >= numRanks_ &&
                 numBlocks <= low_latency::MaxDispatchBlocks);
  if (dispatchDataType == low_latency::DispatchDataType::FP8_E4M3) {
    EP_HOST_ASSERT(dispatchLayout == DispatchLayout::EXPERT_MAJOR);
    EP_HOST_ASSERT(hidden % low_latency::Fp8DeepGemmScaleBlockSize == 0);
    EP_HOST_ASSERT(outputScales != nullptr);
  }

  low_latency::Layout layout(symmetricBuffer_, maxTokensPerRank, hidden, numRanks_, numExperts, numTopk);
  EP_HOST_ASSERT(layout.totalBytes_ <= static_cast<size_t>(symmetricBufferBytes_));
  void* dispatchRecvBuffer = layout.dispatchRecvBuffer_;
  if (dispatchLayout == DispatchLayout::RANK_MAJOR) {
    EP_HOST_ASSERT(output == layout.rankMajorTokenBuffer_);
    EP_HOST_ASSERT(outputTopkIdx == layout.rankMajorTopkIdsBuffer_);
    EP_HOST_ASSERT(outputTopkWeights == layout.rankMajorTopkWeightsBuffer_);
  }

  const low_latency::Workload workload{.numTokens_ = numTokens,
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
}

void MoELowLatencyRuntime::combine(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights,
                                   const int* srcInfo, const int64_t* layoutRange, int numTokens, int hidden,
                                   int numTopk, int maxTokensPerRank, int numExperts, DispatchLayout dispatchLayout,
                                   low_latency::DispatchDataType dispatchDataType, low_latency::CombineMode mode,
                                   int numBlocks, cudaStream_t stream) {
  EP_HOST_ASSERT(available_);
  EP_HOST_ASSERT(numTokens >= 0 && numTokens <= maxTokensPerRank);
  EP_HOST_ASSERT(hidden == hidden_ && numTopk == numTopk_ && maxTokensPerRank == maxTokensPerRank_ &&
                 numExperts == numExperts_);
  EP_HOST_ASSERT(numExperts % numRanks_ == 0);
  EP_HOST_ASSERT(numBlocks > 0 && numBlocks <= low_latency::MaxWorkerBlocks);
  if (dispatchDataType == low_latency::DispatchDataType::FP8_E4M3) {
    EP_HOST_ASSERT(dispatchLayout == DispatchLayout::EXPERT_MAJOR);
    EP_HOST_ASSERT(hidden % low_latency::Fp8DeepGemmScaleBlockSize == 0);
  }

  low_latency::Layout layout(symmetricBuffer_, maxTokensPerRank, hidden, numRanks_, numExperts, numTopk);
  EP_HOST_ASSERT(layout.totalBytes_ <= static_cast<size_t>(symmetricBufferBytes_));
  void* combineRecvBuffer = layout.combineRecvBuffer_;
  void* dispatchRecvBuffer = layout.dispatchRecvBuffer_;
  if (dispatchLayout == DispatchLayout::RANK_MAJOR) {
    EP_HOST_ASSERT(input == layout.rankMajorExpertOutputBuffer_);
  }

  const low_latency::Workload workload{.numTokens_ = numTokens,
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
