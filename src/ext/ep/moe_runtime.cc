// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "moe_runtime.hpp"

#include <cuda.h>

#include <algorithm>
#include <future>
#include <mscclpp/concurrency_device.hpp>

#include "api.cuh"
#include "constants.cuh"
#include "exception.cuh"

namespace mscclpp {
namespace ep {

MoERuntime::MoERuntime(mscclpp::Communicator& communicator, int maxTokensPerRank, int hidden, int numExperts,
                       int numTopk, bool initializeTokenMajorPadding)
    : rank_(communicator.bootstrap()->getRank()),
      numRanks_(communicator.bootstrap()->getNranks()),
      symmetricBufferBytes_(static_cast<int64_t>(
          low_latency::symmetricBufferSize(maxTokensPerRank, hidden, numRanks_, numExperts, numTopk))),
      initializeTokenMajorPadding_(initializeTokenMajorPadding),
      communicator_(&communicator) {
  EP_HOST_ASSERT(communicator_ != nullptr);
  EP_HOST_ASSERT(symmetricBufferBytes_ % NUM_BUFFER_ALIGNMENT_BYTES == 0);
  EP_HOST_ASSERT(maxTokensPerRank > 0);
  EP_HOST_ASSERT(numExperts > 0 && numExperts % numRanks_ == 0);
  EP_HOST_ASSERT(numTopk > 0 && numTopk <= 32);
  EP_HOST_ASSERT(rank_ >= 0 && rank_ < numRanks_);

  CUDA_CHECK(cudaGetDevice(&deviceId_));
  numNvlRanks_ = std::min(numRanks_, communicator.bootstrap()->getNranksPerNode());
  numRanksPerIpcDomain_ =
      std::max(numNvlRanks_, std::min(numRanks_, communicator.bootstrap()->getNranksPerIpcDomain()));
  EP_HOST_ASSERT(numNvlRanks_ > 0 && numRanks_ % numNvlRanks_ == 0);
  EP_HOST_ASSERT(numRanks_ % numRanksPerIpcDomain_ == 0);

  CUDA_CHECK(cudaMalloc(&workspace_, NUM_WORKSPACE_BYTES));
  CUDA_CHECK(cudaMemset(workspace_, 0, NUM_WORKSPACE_BYTES));
  setup();
}

MoERuntime::~MoERuntime() noexcept(false) {
  CUDA_CHECK(cudaDeviceSynchronize());
  if (peerMappedBufferBasesGpu_ != nullptr) CUDA_CHECK(cudaFree(peerMappedBufferBasesGpu_));
  if (workspace_ != nullptr) CUDA_CHECK(cudaFree(workspace_));
  if (symmetricBuffer_ != nullptr) {
    mscclpp::detail::gpuFreePhysical(symmetricBuffer_);
  }
}

bool MoERuntime::isAvailable() const { return available_; }
bool MoERuntime::isInternodeAvailable() const { return isAvailable() && numRanks_ > numNvlRanks_; }

void MoERuntime::setup() {
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

void MoERuntime::dispatch(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                          float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                          const int64_t* topkIdx, const float* topkWeights, int numTokens, int hidden, int numTopk,
                          int maxTokensPerRank, int numExperts, int invalidTokenExpertId, DispatchLayout dispatchLayout,
                          low_latency::DispatchDataType dispatchDataType, int numBlocks, cudaStream_t stream) {
  EP_HOST_ASSERT(available_);
  EP_HOST_ASSERT(numTokens <= maxTokensPerRank);
  EP_HOST_ASSERT(numExperts % numRanks_ == 0);
  EP_HOST_ASSERT(invalidTokenExpertId < 0 || invalidTokenExpertId >= numExperts);
  EP_HOST_ASSERT(numBlocks - low_latency::DispatchControlBlocks >= numRanks_ &&
                 numBlocks <= low_latency::MaxDispatchBlocks);

  low_latency::Layout layout(symmetricBuffer_, maxTokensPerRank, hidden, numRanks_, numExperts, numTopk);
  EP_HOST_ASSERT(layout.totalBytes_ <= static_cast<size_t>(symmetricBufferBytes_));
  void* dispatchRecvBuffer = layout.dispatchRecvBuffer_;

  const low_latency::Workload workload{.numTokens_ = numTokens,
                                       .hidden_ = hidden,
                                       .numTopk_ = numTopk,
                                       .numExperts_ = numExperts,
                                       .invalidTokenExpertId_ = invalidTokenExpertId,
                                       .maxTokensPerRank_ = maxTokensPerRank,
                                       .outputLayout_ = dispatchLayout,
                                       .initializeTokenMajorPadding_ = initializeTokenMajorPadding_,
                                       .dispatchDataType_ = dispatchDataType};
  const size_t workspaceBytes = low_latency::workspaceSize(numRanks_, numExperts);
  EP_HOST_ASSERT(workspaceBytes <= NUM_WORKSPACE_BYTES);
  low_latency::dispatch(output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout,
                        outputCount, input, topkIdx, topkWeights, workload, dispatchRecvBuffer, commContext_,
                        workspace_, numBlocks, stream);
}

void MoERuntime::combine(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights,
                         const int* srcInfo, const int64_t* layoutRange, int numTokens, int hidden, int numTopk,
                         int maxTokensPerRank, int numExperts, DispatchLayout dispatchLayout,
                         low_latency::DispatchDataType dispatchDataType, low_latency::CombineMode mode, int numBlocks,
                         cudaStream_t stream) {
  EP_HOST_ASSERT(available_);
  EP_HOST_ASSERT(numExperts % numRanks_ == 0);
  EP_HOST_ASSERT(numBlocks > 0 && numBlocks <= low_latency::MaxWorkerBlocks);

  low_latency::Layout layout(symmetricBuffer_, maxTokensPerRank, hidden, numRanks_, numExperts, numTopk);
  EP_HOST_ASSERT(layout.totalBytes_ <= static_cast<size_t>(symmetricBufferBytes_));
  void* combineRecvBuffer = layout.combineRecvBuffer_;
  void* dispatchRecvBuffer = layout.dispatchRecvBuffer_;

  const low_latency::Workload workload{.numTokens_ = numTokens,
                                       .hidden_ = hidden,
                                       .numTopk_ = numTopk,
                                       .numExperts_ = numExperts,
                                       .invalidTokenExpertId_ = numExperts,
                                       .maxTokensPerRank_ = maxTokensPerRank,
                                       .outputLayout_ = dispatchLayout,
                                       .initializeTokenMajorPadding_ = false,
                                       .dispatchDataType_ = dispatchDataType};
  low_latency::combine(output, input, topkIdx, topkWeights, srcInfo, layoutRange, workload, combineRecvBuffer,
                       dispatchRecvBuffer, commContext_, workspace_, numBlocks, mode, stream);
}

}  // namespace ep
}  // namespace mscclpp
