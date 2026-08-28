// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "ll_runtime.hpp"

#include <cuda.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <future>
#include <iostream>
#include <mscclpp/concurrency_device.hpp>
#include <string>
#include <thread>

#include "api.cuh"
#include "exception.cuh"

namespace mscclpp {
namespace ep {

namespace {
constexpr int DebugProgressWords = 160;

bool envEnabled(const char* name) {
  const char* value = std::getenv(name);
  return value != nullptr && value[0] != '\0' && value[0] != '0';
}

int envInt(const char* name, int defaultValue) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') return defaultValue;
  return std::atoi(value);
}

low_latency::DispatchProfileMode dispatchProfileModeFromEnv(DispatchLayout dispatchLayout) {
  const char* rawMode = std::getenv("MSCCLPP_EP_DISPATCH_PROFILE_MODE");
  if (rawMode == nullptr || rawMode[0] == '\0') return low_latency::DispatchProfileMode::DISABLED;
  EP_HOST_ASSERT(dispatchLayout == DispatchLayout::EXPERT_MAJOR || dispatchLayout == DispatchLayout::KI_RAGGED);

  const std::string mode(rawMode);
  if (mode == "disabled") return low_latency::DispatchProfileMode::DISABLED;
  if (mode == "send_notify_rank_counts") return low_latency::DispatchProfileMode::SEND_NOTIFY_RANK_COUNTS;
  if (mode == "send_notify_rank_wait") return low_latency::DispatchProfileMode::SEND_NOTIFY_RANK_WAIT;
  if (mode == "send_notify_layout") return low_latency::DispatchProfileMode::SEND_NOTIFY_LAYOUT;
  if (mode == "skip_output_store") return low_latency::DispatchProfileMode::SKIP_OUTPUT_STORE;
  if (mode == "send_counts_from_send") return low_latency::DispatchProfileMode::SEND_COUNTS_FROM_SEND;
  EP_HOST_ASSERT(false && "unsupported MSCCLPP_EP_DISPATCH_PROFILE_MODE");
  return low_latency::DispatchProfileMode::DISABLED;
}
}  // namespace

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
  EP_HOST_ASSERT(outputLayout == DispatchLayout::EXPERT_MAJOR || outputLayout == DispatchLayout::RANK_MAJOR ||
                 outputLayout == DispatchLayout::TOKEN_MAJOR || outputLayout == DispatchLayout::KI_RAGGED);

  CUDA_CHECK(cudaGetDevice(&deviceId_));
  debugProgressEnabled_ = envEnabled("MSCCLPP_EP_DEBUG_PROGRESS");
  if (debugProgressEnabled_) {
    CUDA_CHECK(cudaHostAlloc(&debugProgressHost_, sizeof(uint64_t) * DebugProgressWords, cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&debugProgressDevice_, debugProgressHost_, 0));
    std::fill(debugProgressHost_, debugProgressHost_ + DebugProgressWords, 0);
    std::cerr << "[MSCCLPP_EP_DEBUG_PROGRESS] runtime ctor rank=" << rank_ << " device=" << deviceId_
              << " world=" << numRanks_ << std::endl;
  }
  EP_HOST_ASSERT(numRanks_ % numNvlRanks_ == 0);
  EP_HOST_ASSERT(numRanks_ % numRanksPerIpcDomain_ == 0);

  CUDA_CHECK(cudaMalloc(&workspace_, workspaceBytes_));
  CUDA_CHECK(cudaMemset(workspace_, 0, workspaceBytes_));
  setup();
}

MoELowLatencyRuntime::~MoELowLatencyRuntime() noexcept(false) {
  CUDA_CHECK(cudaDeviceSynchronize());
  if (peerMappedBufferBasesGpu_ != nullptr) CUDA_CHECK(cudaFree(peerMappedBufferBasesGpu_));
  if (debugProgressHost_ != nullptr) CUDA_CHECK(cudaFreeHost(debugProgressHost_));
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
void* MoELowLatencyRuntime::tokenMajorTokenBuffer() const {
  return low_latency::Layout(symmetricBuffer_, maxTokensPerRank_, hidden_, numRanks_, numExperts_, numTopk_)
      .tokenMajorTokenBuffer_;
}
void* MoELowLatencyRuntime::kiRaggedTokenBuffer() const {
  return low_latency::Layout(symmetricBuffer_, maxTokensPerRank_, hidden_, numRanks_, numExperts_, numTopk_)
      .kiRaggedTokenBuffer_;
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

void MoELowLatencyRuntime::dispatch(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                                    float* outputTopkWeights, int64_t* outputLayout, int* outputCount,
                                    const void* input, const int64_t* topkIdx, const float* topkWeights, int numTokens,
                                    int hidden, int numTopk, int maxTokensPerRank, int numExperts,
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
  if (dispatchLayout == DispatchLayout::TOKEN_MAJOR) {
    EP_HOST_ASSERT(output == allocationLayout.tokenMajorTokenBuffer_);
    EP_HOST_ASSERT(outputTopkIdx == allocationLayout.rankMajorTopkIdsBuffer_);
    EP_HOST_ASSERT(outputTopkWeights == allocationLayout.rankMajorTopkWeightsBuffer_);
  }
  if (dispatchLayout == DispatchLayout::KI_RAGGED) {
    EP_HOST_ASSERT(output == allocationLayout.kiRaggedTokenBuffer_);
  }

  ++epoch_;
  if (debugProgressEnabled_) {
    std::cerr << "[MSCCLPP_EP_DEBUG_PROGRESS] dispatch enter rank=" << rank_ << " epoch=" << epoch_
              << " tokens=" << numTokens << " hidden=" << hidden << " topk=" << numTopk
              << " experts=" << numExperts << " layout=" << static_cast<int>(dispatchLayout)
              << " dtype=" << static_cast<int>(dispatchDataType) << std::endl;
    std::fill(debugProgressHost_, debugProgressHost_ + DebugProgressWords, 0);
    CUDA_CHECK(cudaMemsetAsync(debugProgressDevice_, 0, sizeof(uint64_t) * DebugProgressWords, stream));
  }
  const low_latency::Workload workload{.epoch_ = epoch_,
                                       .numTokens_ = numTokens,
                                       .hidden_ = hidden,
                                       .numTopk_ = numTopk,
                                       .numExperts_ = numExperts,
                                       .invalidTokenExpertId_ = invalidTokenExpertId,
                                       .maxTokensPerRank_ = maxTokensPerRank,
                                       .outputLayout_ = dispatchLayout,
                                       .dispatchDataType_ = dispatchDataType,
                                       .dispatchProfileMode_ = dispatchProfileModeFromEnv(dispatchLayout),
                                       .debugProgress_ = debugProgressDevice_};
  const size_t workspaceBytes = low_latency::workspaceSize(numRanks_, numExperts, maxTokensPerRank, numTopk);
  EP_HOST_ASSERT(workspaceBytes <= workspaceBytes_);
  low_latency::dispatch(output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout,
                        outputCount, input, topkIdx, topkWeights, workload, dispatchRecvBuffer, commContext_,
                        workspace_, numBlocks, stream);
  if (debugProgressEnabled_) {
    std::cerr << "[MSCCLPP_EP_DEBUG_PROGRESS] dispatch launched rank=" << rank_ << " epoch=" << epoch_ << std::endl;
    const int timeoutMs = envInt("MSCCLPP_EP_DEBUG_PROGRESS_TIMEOUT_MS", 30000);
    const int printRank = envInt("MSCCLPP_EP_DEBUG_PROGRESS_RANK", 0);
    const bool printAllRanks = envEnabled("MSCCLPP_EP_DEBUG_PROGRESS_ALL_RANKS");
    const bool shouldPrint = printAllRanks || rank_ == printRank;
    auto start = std::chrono::steady_clock::now();
    uint64_t lastStage = UINT64_MAX;
    while (true) {
      cudaError_t query = cudaStreamQuery(stream);
      if (query == cudaSuccess) break;
      if (query != cudaErrorNotReady) CUDA_CHECK(query);
      uint64_t stage = debugProgressHost_[0];
      if (shouldPrint && stage != lastStage) {
        std::cerr << "[MSCCLPP_EP_DEBUG_PROGRESS] rank=" << rank_ << " epoch=" << epoch_ << " stage=" << stage
                  << " block=" << debugProgressHost_[1] << " a=" << debugProgressHost_[2]
                  << " b=" << debugProgressHost_[3] << " c=" << debugProgressHost_[4]
                  << " d=" << debugProgressHost_[5] << " e=" << debugProgressHost_[6] << std::endl;
        lastStage = stage;
      }
      auto elapsedMs =
          std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
      if (elapsedMs > timeoutMs) {
        if (shouldPrint) {
          std::cerr << "[MSCCLPP_EP_DEBUG_PROGRESS] rank=" << rank_ << " timed out after " << elapsedMs
                    << " ms epoch=" << epoch_ << " stage=" << debugProgressHost_[0]
                    << " block=" << debugProgressHost_[1] << " a=" << debugProgressHost_[2]
                    << " b=" << debugProgressHost_[3] << " c=" << debugProgressHost_[4]
                    << " d=" << debugProgressHost_[5] << " e=" << debugProgressHost_[6] << " words=[";
          for (int word = 0; word < DebugProgressWords; ++word) {
            if (word != 0) std::cerr << ",";
            std::cerr << debugProgressHost_[word];
          }
          std::cerr << "] ready=[";
          for (int sourceRank = 0; sourceRank < numRanks_ && 16 + sourceRank < DebugProgressWords; ++sourceRank) {
            if (sourceRank != 0) std::cerr << ",";
            std::cerr << debugProgressHost_[16 + sourceRank];
          }
          std::cerr << "] send=[";
          for (int dstRank = 0; dstRank < numRanks_ && 48 + dstRank < DebugProgressWords; ++dstRank) {
            if (dstRank != 0) std::cerr << ",";
            std::cerr << debugProgressHost_[48 + dstRank];
          }
          std::cerr << "] sender=[";
          for (int dstRank = 0; dstRank < numRanks_ && 80 + dstRank < DebugProgressWords; ++dstRank) {
            if (dstRank != 0) std::cerr << ",";
            std::cerr << debugProgressHost_[80 + dstRank];
          }
          std::cerr << "] sender_loop=[" << debugProgressHost_[112] << "," << debugProgressHost_[113] << ","
                    << debugProgressHost_[114] << "," << debugProgressHost_[115] << ","
                    << debugProgressHost_[116] << "] send_steps=[" << debugProgressHost_[117] << ","
                    << debugProgressHost_[118] << "," << debugProgressHost_[119] << "] lane_dst=[";
          for (int lane = 0; lane < 8; ++lane) {
            if (lane != 0) std::cerr << ",";
            std::cerr << debugProgressHost_[120 + lane];
          }
          std::cerr << "] lane_slot=[";
          for (int lane = 0; lane < 8; ++lane) {
            if (lane != 0) std::cerr << ",";
            std::cerr << debugProgressHost_[128 + lane];
          }
          std::cerr << "] recv=[" << debugProgressHost_[136] << "," << debugProgressHost_[137] << ","
                    << debugProgressHost_[138] << "," << debugProgressHost_[139] << ","
                    << debugProgressHost_[140] << "," << debugProgressHost_[141] << ","
                    << debugProgressHost_[142] << "," << debugProgressHost_[143];
          std::cerr << "] worker_wait=[" << debugProgressHost_[144] << "," << debugProgressHost_[145] << ","
                    << debugProgressHost_[146] << "]" << std::endl;
        }
        std::abort();
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
}

void MoELowLatencyRuntime::combine(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights,
                                   const int* srcInfo, const int64_t* layoutRange, int numTokens, int hidden,
                                   int numTopk, int maxTokensPerRank, int numExperts, DispatchLayout dispatchLayout,
                                   low_latency::DispatchDataType dispatchDataType, low_latency::CombineMode mode,
                                   int numBlocks, cudaStream_t stream) {
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
  if (dispatchLayout == DispatchLayout::TOKEN_MAJOR) {
    // Token-major combine reads per-slot expert outputs from the symmetric token
    // buffer (in-place expert output; identity when no GEMM is applied).
    EP_HOST_ASSERT(input == allocationLayout.tokenMajorTokenBuffer_);
  }

  const low_latency::Workload workload{.epoch_ = epoch_,
                                       .numTokens_ = numTokens,
                                       .hidden_ = hidden,
                                       .numTopk_ = numTopk,
                                       .numExperts_ = numExperts,
                                       .invalidTokenExpertId_ = numExperts,
                                       .maxTokensPerRank_ = maxTokensPerRank,
                                       .outputLayout_ = dispatchLayout,
                                       .dispatchDataType_ = dispatchDataType,
                                       .dispatchProfileMode_ = low_latency::DispatchProfileMode::DISABLED,
                                       .debugProgress_ = nullptr};
  const size_t workspaceBytes = low_latency::workspaceSize(numRanks_, numExperts, maxTokensPerRank, numTopk);
  EP_HOST_ASSERT(workspaceBytes <= workspaceBytes_);
  low_latency::combine(output, input, topkIdx, topkWeights, srcInfo, layoutRange, workload, combineRecvBuffer,
                       dispatchRecvBuffer, commContext_, workspace_, numBlocks, mode, stream);
}

}  // namespace ep
}  // namespace mscclpp
