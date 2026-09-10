// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP)
// branch `chhwang/dev-atomic-add-cleanup`. Licensed under the MIT License.

#include <chrono>
#include <future>
#include <mscclpp/ext/ep/moe_runtime.hpp>
#include <mscclpp/gpu_utils.hpp>

#include "exception.hpp"
#include "kernels.hpp"
#include "moe_runtime_context.hpp"

namespace mscclpp {
namespace ep {
namespace {

constexpr auto ReceiveCountTimeout = std::chrono::seconds(100);

int outputRows(DispatchLayout outputLayout, int numRanks, int numRecvTokens, int maxTokensPerRank) {
  return outputLayout == DispatchLayout::RANK_MAJOR ? numRanks * maxTokensPerRank : numRecvTokens;
}

void waitForReceiveCounts(const volatile int* recvCounter, const volatile int* recvExpertCounter, int numLocalExperts) {
  const auto start = std::chrono::steady_clock::now();
  while (true) {
    const int numRecvTokens = static_cast<int>(*recvCounter);
    bool ready = numRecvTokens >= 0;
    for (int i = 0; i < numLocalExperts && ready; ++i) ready &= recvExpertCounter[i] >= 0;
    if (ready) return;
    if (std::chrono::steady_clock::now() - start >= ReceiveCountTimeout) {
      EP_THROW("MSCCL++ EP throughput receive-count timeout");
    }
  }
}

}  // namespace

ThroughputRuntimeContext::ThroughputRuntimeContext(mscclpp::Communicator& communicator, int rank, int numRanks,
                                                   int numNvlRanks, int numRanksPerIpcDomain, int maxTokensPerRank,
                                                   int hidden, int numExperts, int numTopk, DispatchLayout outputLayout)
    : rank_(rank),
      numRanks_(numRanks),
      numNvlRanks_(numNvlRanks),
      numRanksPerIpcDomain_(numRanksPerIpcDomain),
      bootstrap_(communicator.bootstrap()),
      maxTokensPerRank_(maxTokensPerRank),
      hidden_(hidden),
      numExperts_(numExperts),
      numTopk_(numTopk),
      maxHiddenBytes_(static_cast<int64_t>(hidden) * sizeof(Bf16)),
      outputLayout_(outputLayout),
      communicator_(communicator) {
  EP_HOST_ASSERT(hidden_ > 0);
  EP_HOST_ASSERT(numExperts_ > 0 && numExperts_ % numRanks_ == 0);
  EP_HOST_ASSERT(numTopk_ > 0 && numTopk_ <= RecvPoolConfig::MaxTopk);
  EP_HOST_ASSERT(outputLayout_ == DispatchLayout::TOKEN_MAJOR || outputLayout_ == DispatchLayout::RANK_MAJOR);
  EP_HOST_ASSERT(maxTokensPerRank_ > 0);
  EP_HOST_ASSERT(maxHiddenBytes_ % sizeof(int4) == 0);

  if ((numRanks_ != 2 && numRanks_ != 4 && numRanks_ != 8 && numRanks_ != 16) || numRanksPerIpcDomain_ < numRanks_) {
    return;
  }

  controlBufferBytes_ = RecvPoolConfig::controlBufferBytes(numRanks_);
  symmetricBufferBytes_ = configAlign<size_t>(controlBufferBytes_, BufferAlignmentBytes);
  physicalControlBuffer_ = numRanks_ > numNvlRanks_;
  recvPoolBytes_ = RecvPoolConfig::recvPoolBytes(numRanks_);
  workspaceBytes_ = throughputStorageSize(maxTokensPerRank_, numRanks_, numExperts_, MaxDispatchBlocks);
  available_ = canUseDirectRecvPool(maxTokensPerRank_);
}

ThroughputRuntimeContext::~ThroughputRuntimeContext() noexcept(false) {
  if (deviceContext_.devicePtr_ == nullptr) return;

  CudaDeviceGuard deviceGuard(deviceContext_.deviceId_);
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  bootstrap_->barrier();

  MSCCLPP_CUDATHROW(cudaFree(deviceContext_.devicePtr_));
  if (combineRecvIdxGpu_ != nullptr) MSCCLPP_CUDATHROW(cudaFree(combineRecvIdxGpu_));
  if (recvPoolPtrsGpu_ != nullptr) MSCCLPP_CUDATHROW(cudaFree(recvPoolPtrsGpu_));
  if (bufferPtrsGpu_ != nullptr) MSCCLPP_CUDATHROW(cudaFree(bufferPtrsGpu_));
  if (workspace_ != nullptr) MSCCLPP_CUDATHROW(cudaFree(workspace_));
  if (moeRecvExpertCounter_ != nullptr) MSCCLPP_CUDATHROW(cudaFreeHost(const_cast<int*>(moeRecvExpertCounter_)));
  if (moeRecvCounter_ != nullptr) MSCCLPP_CUDATHROW(cudaFreeHost(const_cast<int*>(moeRecvCounter_)));

  recvPoolMemories_.clear();
  peerMemories_.clear();
  if (recvPool_ != nullptr) mscclpp::detail::gpuFreePhysical(recvPool_);
  if (symmetricBuffer_ != nullptr) {
    if (physicalControlBuffer_) {
      mscclpp::detail::gpuFreePhysical(symmetricBuffer_);
    } else {
      MSCCLPP_CUDATHROW(cudaFree(symmetricBuffer_));
    }
  }
}

void ThroughputRuntimeContext::initialize() {
  EP_HOST_ASSERT(available_);
  EP_HOST_ASSERT(symmetricBuffer_ == nullptr);
  AvoidCudaGraphCaptureGuard captureGuard;

  if (physicalControlBuffer_) {
    symmetricBuffer_ = mscclpp::detail::gpuCallocPhysical(symmetricBufferBytes_);
  } else {
    symmetricBuffer_ = mscclpp::detail::gpuCalloc(symmetricBufferBytes_);
  }
  recvPool_ = mscclpp::detail::gpuCallocPhysical(recvPoolBytes_);
  workspace_ = mscclpp::detail::gpuCalloc(workspaceBytes_);

  constexpr int ControlBufferTag = 17;
  constexpr int RecvPoolTag = 18;
  constexpr int BarrierConnectionTag = 19;
  const auto transport = mscclpp::Transport::CudaIpc;
  const mscclpp::EndpointConfig ipcConfig(transport);
  peerMemories_.resize(numRanks_);
  peerMemories_[rank_] = communicator_.registerMemory(symmetricBuffer_, symmetricBufferBytes_, transport);
  std::vector<std::shared_future<mscclpp::RegisteredMemory>> remoteMemories(numRanks_);
  recvPoolMemories_.resize(numRanks_);
  recvPoolMemories_[rank_] = communicator_.registerMemory(recvPool_, recvPoolBytes_, transport);
  std::vector<std::shared_future<mscclpp::RegisteredMemory>> remoteRecvPools(numRanks_);
  std::vector<std::shared_future<mscclpp::Connection>> barrierConnections(numRanks_);
  for (int peer = 0; peer < numRanks_; ++peer) {
    if (peer == rank_) continue;
    communicator_.sendMemory(peerMemories_[rank_], peer, ControlBufferTag);
    remoteMemories[peer] = communicator_.recvMemory(peer, ControlBufferTag);
    communicator_.sendMemory(recvPoolMemories_[rank_], peer, RecvPoolTag);
    remoteRecvPools[peer] = communicator_.recvMemory(peer, RecvPoolTag);
    barrierConnections[peer] = communicator_.connect(ipcConfig, peer, BarrierConnectionTag);
  }

  bufferPtrs_.resize(numRanks_);
  recvPoolPtrs_.resize(numRanks_);
  barrierChannels_.reserve(numRanks_ - 1);
  std::vector<mscclpp::BaseMemoryChannelDeviceHandle> barrierChannelHandles(numRanks_);
  for (int peer = 0; peer < numRanks_; ++peer) {
    if (peer != rank_) {
      peerMemories_[peer] = remoteMemories[peer].get();
      recvPoolMemories_[peer] = remoteRecvPools[peer].get();
    }
    bufferPtrs_[peer] = peer == rank_ ? symmetricBuffer_ : peerMemories_[peer].data();
    recvPoolPtrs_[peer] = peer == rank_ ? recvPool_ : recvPoolMemories_[peer].data();
    if (peer != rank_) {
      auto semaphore =
          std::make_shared<mscclpp::MemoryDevice2DeviceSemaphore>(communicator_, barrierConnections[peer].get());
      barrierChannels_.emplace_back(semaphore);
      barrierChannelHandles[peer] = barrierChannels_.back().deviceHandle();
    }
  }

  bufferPtrsGpu_ = static_cast<void**>(mscclpp::detail::gpuCalloc(sizeof(void*) * static_cast<size_t>(numRanks_)));
  mscclpp::gpuMemcpy<void*>(bufferPtrsGpu_, bufferPtrs_.data(), numRanks_, cudaMemcpyHostToDevice);
  recvPoolPtrsGpu_ = static_cast<void**>(mscclpp::detail::gpuCalloc(sizeof(void*) * static_cast<size_t>(numRanks_)));
  mscclpp::gpuMemcpy<void*>(recvPoolPtrsGpu_, recvPoolPtrs_.data(), numRanks_, cudaMemcpyHostToDevice);
  barrierChannelHandles_ = mscclpp::detail::gpuCallocShared<mscclpp::BaseMemoryChannelDeviceHandle>(numRanks_);
  mscclpp::gpuMemcpy<mscclpp::BaseMemoryChannelDeviceHandle>(barrierChannelHandles_.get(), barrierChannelHandles.data(),
                                                             numRanks_, cudaMemcpyHostToDevice);
  combineRecvIdxGpu_ =
      static_cast<int*>(mscclpp::detail::gpuCalloc(sizeof(int) * static_cast<size_t>(maxTokensPerRank_) * numRanks_));
  moeRecvCounter_ = static_cast<volatile int*>(mscclpp::detail::gpuCallocHost(sizeof(int), cudaHostAllocMapped));
  MSCCLPP_CUDATHROW(cudaHostGetDevicePointer(&moeRecvCounterMapped_, const_cast<int*>(moeRecvCounter_), 0));
  moeRecvExpertCounter_ = static_cast<volatile int*>(
      mscclpp::detail::gpuCallocHost(sizeof(int) * static_cast<size_t>(numExperts_ / numRanks_), cudaHostAllocMapped));
  MSCCLPP_CUDATHROW(cudaHostGetDevicePointer(&moeRecvExpertCounterMapped_, const_cast<int*>(moeRecvExpertCounter_), 0));
  *moeRecvCounter_ = -1;
  for (int i = 0; i < numExperts_ / numRanks_; ++i) moeRecvExpertCounter_[i] = -1;

  int deviceId;
  int maxSharedMemoryPerBlock;
  int numSms;
  MSCCLPP_CUDATHROW(cudaGetDevice(&deviceId));
  MSCCLPP_CUDATHROW(
      cudaDeviceGetAttribute(&maxSharedMemoryPerBlock, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceId));
  MSCCLPP_CUDATHROW(cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, deviceId));
  deviceContext_ = {.localBufferBase_ = symmetricBuffer_,
                    .peerBufferBases_ = bufferPtrsGpu_,
                    .peerPayloadBases_ = recvPoolPtrsGpu_,
                    .channels_ = barrierChannelHandles_.get(),
                    .workspace_ = workspace_,
                    .combineRecvIdx_ = combineRecvIdxGpu_,
                    .mappedRecvCounter_ = moeRecvCounterMapped_,
                    .mappedRecvExpertCounters_ = moeRecvExpertCounterMapped_,
                    .maxSharedMemoryPerBlock_ = maxSharedMemoryPerBlock,
                    .numSms_ = numSms,
                    .deviceId_ = deviceId,
                    .rank_ = rank_,
                    .numRanks_ = numRanks_};
  deviceContext_.devicePtr_ = static_cast<DeviceContext*>(mscclpp::detail::gpuCalloc(sizeof(DeviceContext)));
  mscclpp::gpuMemcpy<DeviceContext>(deviceContext_.devicePtr_, &deviceContext_, 1, cudaMemcpyHostToDevice);
}

bool ThroughputRuntimeContext::canUseDirectRecvPool(int maxTokensPerRank) const {
  if (maxTokensPerRank <= 0 || maxTokensPerRank > maxTokensPerRank_) return false;
  if (maxHiddenBytes_ <= 0 || maxHiddenBytes_ > RecvPoolConfig::RecvPoolMaxHiddenBytes) return false;
  const int maxRows = numRanks_ * maxTokensPerRank;
  return maxRows <= RecvPoolConfig::RecvPoolMaxTokens &&
         static_cast<size_t>(maxRows) * static_cast<size_t>(maxHiddenBytes_) <=
             RecvPoolConfig::recvPoolHiddenBytes(numRanks_);
}

DispatchHandle MoERuntime::launchThroughputDispatch(const ThroughputDispatchRequest& request) {
  auto& context = *throughputContext_;
  EP_HOST_ASSERT(context.available_);
  EP_HOST_ASSERT(context.deviceContext_.devicePtr_ != nullptr);
  if (request.maxTokensPerRank <= 0 || request.maxTokensPerRank > context.maxTokensPerRank_ || request.numTokens < 0 ||
      request.numTokens > request.maxTokensPerRank) {
    EP_THROW("Throughput requests require 0 <= numTokens <= maxTokensPerRank <= runtime capacity");
  }
  EP_HOST_ASSERT(request.numBlocks > 0 && request.numBlocks <= MaxDispatchBlocks);
  EP_HOST_ASSERT(request.output != nullptr || request.numTokens == 0);
  EP_HOST_ASSERT(request.input != nullptr || request.numTokens == 0);
  EP_HOST_ASSERT(request.topkIdx != nullptr || request.numTokens == 0);
  if (!context.canUseDirectRecvPool(request.maxTokensPerRank)) {
    EP_THROW("Throughput receive-pool capacity exceeded for this runtime configuration");
  }

  const int elementBytes = dispatchElementBytes(request.dispatchDataType);
  const int hiddenBytes = context.hidden_ * elementBytes;
  if (hiddenBytes % static_cast<int>(sizeof(int4)) != 0) {
    EP_THROW("Throughput dispatch requires the row byte size to be a multiple of int4");
  }
  const int numScales = dispatchNumScales(request.dispatchDataType, context.hidden_);
  if (request.dispatchDataType == DispatchDataType::FP8_E4M3) {
    EP_HOST_ASSERT(context.hidden_ % dispatchScaleBlockSize(request.dispatchDataType) == 0);
    EP_HOST_ASSERT(request.inputScales != nullptr || request.numTokens == 0);
  }

  const ThroughputStorageLayout layout(context.workspace_, context.maxTokensPerRank_, context.numRanks_,
                                       context.numExperts_, MaxDispatchBlocks);
  EP_HOST_ASSERT(layout.totalBytes_ <= context.workspaceBytes_);
  *context.moeRecvCounter_ = -1;
  for (int i = 0; i < context.numExperts_ / context.numRanks_; ++i) context.moeRecvExpertCounter_[i] = -1;
  throughputPrepare(request.topkIdx, layout.numTokensPerRank_, layout.numTokensPerExpert_, layout.isTokenInRank_,
                    request.numTokens, context.numTopk_, context.numExperts_, context.deviceContext_, request.stream);
  throughputExchangeCounts(layout.numTokensPerRank_, layout.numTokensPerExpert_, context.numExperts_, request.numTokens,
                           layout.isTokenInRank_, layout.channelPrefixMatrix_, layout.rankPrefixMatrix_, 1,
                           context.deviceContext_, request.stream, request.numBlocks);
  waitForReceiveCounts(context.moeRecvCounter_, context.moeRecvExpertCounter_, context.numExperts_ / context.numRanks_);
  const int numRecvTokens = static_cast<int>(*context.moeRecvCounter_);
  EP_HOST_ASSERT(numRecvTokens >= 0 && numRecvTokens <= RecvPoolConfig::RecvPoolMaxTokens);

  if (request.outputCount != nullptr) {
    if (context.outputLayout_ == DispatchLayout::TOKEN_MAJOR) {
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(request.outputCount, const_cast<int*>(context.moeRecvExpertCounter_),
                                        sizeof(int) * static_cast<size_t>(context.numExperts_ / context.numRanks_),
                                        cudaMemcpyHostToDevice, request.stream));
    } else {
      std::vector<int> hostRankPrefix(static_cast<size_t>(context.numRanks_) * context.numRanks_);
      std::vector<int> hostRankCounts(context.numRanks_);
      MSCCLPP_CUDATHROW(cudaMemcpy(hostRankPrefix.data(), layout.rankPrefixMatrix_, sizeof(int) * hostRankPrefix.size(),
                                   cudaMemcpyDeviceToHost));
      for (int srcRank = 0; srcRank < context.numRanks_; ++srcRank) {
        const int prefix = hostRankPrefix[srcRank * context.numRanks_ + context.rank_];
        const int previous = srcRank == 0 ? 0 : hostRankPrefix[(srcRank - 1) * context.numRanks_ + context.rank_];
        hostRankCounts[srcRank] = prefix - previous;
      }
      MSCCLPP_CUDATHROW(cudaMemcpyAsync(request.outputCount, hostRankCounts.data(), sizeof(int) * hostRankCounts.size(),
                                        cudaMemcpyHostToDevice, request.stream));
    }
  }

  const size_t recvPoolHeaderBytes = RecvPoolConfig::recvPoolHeaderBytes(context.numRanks_);
  const size_t recvPoolMetadataOffset = RecvPoolConfig::recvPoolMetadataOffset(context.numRanks_);
  void* localRecvPoolX = static_cast<uint8_t*>(context.recvPoolPtrs_[context.rank_]) + recvPoolHeaderBytes;
  throughputDispatch(layout.sendHead_, request.input, request.topkIdx, request.topkWeights, request.inputScales,
                     layout.isTokenInRank_, layout.channelPrefixMatrix_, request.numTokens, numRecvTokens,
                     hiddenBytes / static_cast<int>(sizeof(int4)), context.numTopk_, context.numExperts_, numScales,
                     request.outputTopkIdx, request.outputTopkWeights, static_cast<float*>(request.outputScales),
                     request.numBlocks, static_cast<int64_t>(recvPoolHeaderBytes),
                     static_cast<int64_t>(recvPoolMetadataOffset), RecvPoolConfig::RecvPoolMetaBytes,
                     context.outputLayout_, request.maxTokensPerRank, context.deviceContext_, request.stream);

  const int rows = outputRows(context.outputLayout_, context.numRanks_, numRecvTokens, request.maxTokensPerRank);
  if (rows > 0 && request.output != localRecvPoolX) {
    MSCCLPP_CUDATHROW(cudaMemcpyAsync(request.output, localRecvPoolX,
                                      static_cast<size_t>(rows) * static_cast<size_t>(hiddenBytes),
                                      cudaMemcpyDeviceToDevice, request.stream));
  }

  ++context.epoch_;
  return DispatchHandle(std::make_shared<const DispatchHandle::Impl>(throughputContext_, context.epoch_, request,
                                                                     layout.sendHead_, numRecvTokens));
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
  const auto* throughputMetadata = std::get_if<DispatchHandle::Impl::ThroughputMetadata>(&handle.metadata_);
  if (throughputMetadata == nullptr) {
    EP_THROW("Dispatch handle does not contain throughput metadata");
  }

  EP_HOST_ASSERT(context.available_);
  EP_HOST_ASSERT(context.deviceContext_.devicePtr_ != nullptr);
  EP_HOST_ASSERT(request.numBlocks > 0 && request.numBlocks <= MaxWorkerBlocks);
  EP_HOST_ASSERT(request.output != nullptr || handle.numTokens_ == 0);
  EP_HOST_ASSERT(request.input != nullptr || throughputMetadata->numRecvTokens_ == 0);

  const size_t recvPoolHeaderBytes = RecvPoolConfig::recvPoolHeaderBytes(context.numRanks_);
  const size_t recvPoolMetadataOffset = RecvPoolConfig::recvPoolMetadataOffset(context.numRanks_);
  void* localRecvPoolX = static_cast<uint8_t*>(context.recvPoolPtrs_[context.rank_]) + recvPoolHeaderBytes;
  if (throughputMetadata->numRecvTokens_ > 0 && request.input != localRecvPoolX) {
    MSCCLPP_CUDATHROW(
        cudaMemcpyAsync(localRecvPoolX, request.input,
                        static_cast<size_t>(throughputMetadata->numRecvTokens_) * context.hidden_ * sizeof(Bf16),
                        cudaMemcpyDeviceToDevice, request.stream));
  }

  throughputReduceCombine(request.output, request.outputTopkWeights, throughputMetadata->sendHead_, handle.numTokens_,
                          context.hidden_, context.numTopk_, static_cast<int64_t>(recvPoolHeaderBytes),
                          static_cast<int64_t>(recvPoolMetadataOffset), RecvPoolConfig::RecvPoolMetaBytes,
                          request.numBlocks, context.deviceContext_, request.stream);
}

}  // namespace ep
}  // namespace mscclpp
