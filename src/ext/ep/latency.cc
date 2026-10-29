// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cuda.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/ext/ep/moe_runtime.hpp>
#include <string>

#include "exception.hpp"
#include "kernels.hpp"
#include "moe_runtime_context.hpp"

#if defined(MSCCLPP_USE_GPUNETIO)
#include "gpu_net_io.hpp"
#endif

namespace mscclpp {
namespace ep {

namespace {

// Parse a 0/1 switch: unset keeps fallback, anything else is invalid (-1).
int envSwitch(const char* name, int fallback) {
  const char* requested = std::getenv(name);
  if (requested == nullptr) return fallback;
  const std::string value(requested);
  return value == "0" ? 0 : value == "1" ? 1 : -1;
}

// Parse a small unsigned integer: unset keeps fallback, anything else is invalid (-1).
int envCount(const char* name, int fallback, size_t maxDigits) {
  const char* requested = std::getenv(name);
  if (requested == nullptr) return fallback;
  const std::string value(requested);
  const bool numeric = !value.empty() && value.size() <= maxDigits &&
                       std::all_of(value.begin(), value.end(), [](char digit) { return digit >= '0' && digit <= '9'; });
  return numeric ? std::stoi(value) : -1;
}

}  // namespace

LatencyRuntimeContext::LatencyRuntimeContext(mscclpp::Communicator& communicator, int rank, int numRanks,
                                             int numRanksPerIpcDomain, int maxTokensPerRank, int hidden, int numExperts,
                                             int numTopk, DispatchLayout outputLayout, CombineMode combineMode)
    : rank_(rank),
      numRanks_(numRanks),
      numRanksPerIpcDomain_(numRanksPerIpcDomain),
      numRanksPerNode_(std::min(numRanks, communicator.bootstrap()->getNranksPerNode())),
      maxTokensPerRank_(maxTokensPerRank),
      hidden_(hidden),
      numExperts_(numExperts),
      numTopk_(numTopk),
      outputLayout_(outputLayout),
      combineMode_(combineMode),
      symmetricBufferBytes_(0),
      workspaceBytes_(0),
      communicator_(communicator) {
  EP_HOST_ASSERT(maxTokensPerRank > 0);
  EP_HOST_ASSERT(numExperts > 0 && numExperts % numRanks_ == 0);
  EP_HOST_ASSERT(numTopk > 0 && numTopk <= MaxNumTopk);
  EP_HOST_ASSERT(outputLayout == DispatchLayout::EXPERT_MAJOR || outputLayout == DispatchLayout::RANK_MAJOR);
  EP_HOST_ASSERT(combineMode == CombineMode::RANK_LOCAL_REDUCE || combineMode == CombineMode::DIRECT_SEND);
  // Optionally shrink the CUDA IPC domain (for example to one node) so that traffic
  // between domains is served by GPUNetIO instead of multi-node NVLink.
  if (const char* forced = std::getenv("MSCCLPP_EP_IPC_DOMAIN_SIZE")) {
    const int size = std::atoi(forced);
    if (size < 1 || size > numRanksPerIpcDomain_ || numRanksPerIpcDomain_ % size != 0) {
      EP_THROW("MSCCLPP_EP_IPC_DOMAIN_SIZE must be a positive divisor of the detected IPC domain size");
    }
    numRanksPerIpcDomain_ = size;
  }
  EP_HOST_ASSERT(numRanks_ % numRanksPerIpcDomain_ == 0);
  MSCCLPP_CUDATHROW(cudaGetDevice(&deviceId_));

#if defined(MSCCLPP_USE_GPUNETIO)
  if (const char* enable = std::getenv("MSCCLPP_EP_ENABLE_GPUNETIO")) {
    useGpuNetIo_ = std::atoi(enable) != 0 && numRanksPerIpcDomain_ < numRanks_ &&
                   outputLayout == DispatchLayout::RANK_MAJOR && combineMode == CombineMode::RANK_LOCAL_REDUCE;
  }
#endif
  if (useGpuNetIo_) {
    const int aggregateCombine = envSwitch("MSCCLPP_EP_GPUNETIO_AGGREGATE_COMBINE", 0);
    // Retain syntax validity until collective initialization so every rank rejects bad configuration.
    networkConfigValid_ = aggregateCombine >= 0;
    deviceContext_.gpuNetIoAggregateCombine_ = aggregateCombine == 1;
  }
  if (!isSupportedHidden(hidden, useGpuNetIo_)) {
    EP_THROW("Unsupported latency hidden size: " + std::to_string(hidden));
  }
  symmetricBufferBytes_ = static_cast<int64_t>(storageLayout().totalBytes_);
  workspaceBytes_ = workspaceSize(numRanks_, numExperts, maxTokensPerRank, numTopk);
  EP_HOST_ASSERT(symmetricBufferBytes_ % BufferAlignmentBytes == 0);
  available_ = isSupportedRanks(numRanks_) && (numRanksPerIpcDomain_ >= numRanks_ || useGpuNetIo_);
}

LatencyRuntimeContext::~LatencyRuntimeContext() noexcept(false) {
  CudaDeviceGuard deviceGuard(deviceId_);
  // Complete all network accesses before deregistering or freeing the symmetric buffer.
  if (gpuNetIoService_ != nullptr) MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  gpuNetIoService_.reset();
  if (deviceContext_.devicePtr_ != nullptr) MSCCLPP_CUDATHROW(cudaFree(deviceContext_.devicePtr_));
  if (peerMappedBufferBasesGpu_ != nullptr) MSCCLPP_CUDATHROW(cudaFree(peerMappedBufferBasesGpu_));
  if (workspace_ != nullptr) MSCCLPP_CUDATHROW(cudaFree(workspace_));
  symmetricBuffer_.reset();
}

LatencyStorageLayout LatencyRuntimeContext::storageLayout() const {
  return LatencyStorageLayout(symmetricBuffer_.get(), maxTokensPerRank_, hidden_, numRanks_, numExperts_, numTopk_,
                              outputLayout_, combineMode_, useGpuNetIo_, deviceContext_.gpuNetIoAggregateCombine_);
}

void LatencyRuntimeContext::initialize() {
  EP_HOST_ASSERT(available_);
  EP_HOST_ASSERT(symmetricBuffer_ == nullptr);
  AvoidCudaGraphCaptureGuard captureGuard;

  workspace_ = mscclpp::detail::gpuCalloc(workspaceBytes_);

  const auto ipcTransport = mscclpp::Transport::CudaIpc;
  const size_t allocationGranularity = mscclpp::detail::getCuAllocationGranularity(CU_MEM_ALLOC_GRANULARITY_MINIMUM);
  symmetricBuffer_ = mscclpp::detail::gpuCallocPhysicalUnique<uint8_t>(static_cast<size_t>(symmetricBufferBytes_),
                                                                       allocationGranularity, allocationGranularity);

  const mscclpp::EndpointConfig ipcConfig(ipcTransport);
  const int ipcDomainSize = numRanksPerIpcDomain_;
  auto isMappedPeer = [&](int peer) {
    return peer != rank_ && ipcDomainSize > 1 && rank_ / ipcDomainSize == peer / ipcDomainSize;
  };

  constexpr int IpcTag = 1;
  peerBufferMemories_.resize(numRanks_);
  peerBufferMemories_[rank_] =
      communicator_.registerMemory(symmetricBuffer_.get(), symmetricBufferBytes_, ipcTransport);
  std::vector<std::shared_future<mscclpp::RegisteredMemory>> remoteFutures(numRanks_);
  std::vector<std::shared_future<mscclpp::Connection>> connectionFutures(numRanks_);
  for (int r = 0; r < numRanks_; ++r) {
    if (!isMappedPeer(r)) continue;
    communicator_.sendMemory(peerBufferMemories_[rank_], r, IpcTag);
    remoteFutures[r] = communicator_.recvMemory(r, IpcTag);
    connectionFutures[r] = communicator_.connect(ipcConfig, r, IpcTag);
  }

  peerMappedBufferBases_.assign(numRanks_, nullptr);
  peerMappedBufferBases_[rank_] = symmetricBuffer_.get();
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
  deviceContext_.localBufferBase_ = symmetricBuffer_.get();
  deviceContext_.peerBufferBases_ = peerMappedBufferBasesGpu_;
  deviceContext_.channels_ = baseMemoryChannelHandles_.get();
  deviceContext_.workspace_ = workspace_;
  deviceContext_.maxSharedMemoryPerBlock_ = maxSharedMemoryPerBlock;
  deviceContext_.numSms_ = numSms;
  deviceContext_.deviceId_ = deviceId_;
  deviceContext_.rank_ = rank_;
  deviceContext_.numRanks_ = numRanks_;
  if (useGpuNetIo_) initializeNetwork();
  deviceContext_.devicePtr_ = static_cast<DeviceContext*>(mscclpp::detail::gpuCalloc(sizeof(DeviceContext)));
  mscclpp::gpuMemcpy<DeviceContext>(deviceContext_.devicePtr_, &deviceContext_, 1, cudaMemcpyHostToDevice);
}

void LatencyRuntimeContext::initializeNetwork() {
  const bool aggregateCombine = deviceContext_.gpuNetIoAggregateCombine_;
  const int domainSize = numRanksPerIpcDomain_;
  const int pipelineChunkRows =
      envCount("MSCCLPP_EP_GPUNETIO_DISPATCH_CHUNK_ROWS", GpuNetIoDistributeDefaultChunkRows, 3);
  int rowFence = GpuNetIoRowFenceGpuPoster;
  if (const char* requested = std::getenv("MSCCLPP_EP_GPUNETIO_ROW_FENCE")) {
    const std::string value(requested);
    rowFence = value == "sys"          ? GpuNetIoRowFenceSystem
               : value == "gpu"        ? GpuNetIoRowFenceGpu
               : value == "gpu+poster" ? GpuNetIoRowFenceGpuPoster
                                       : -1;
  }
  const int combineChunkRows = envCount("MSCCLPP_EP_GPUNETIO_COMBINE_CHUNK_ROWS", aggregateCombine ? 8 : 0, 3);

  // All ranks must agree before registering network buffers or exchanging protocol state.
  struct StorageConfig {
    uint64_t bytes;
    int ipcDomainSize;
    int pipelineChunkRows;
    int rowFence;
    int combineChunkRows;
    int aggregateCombine;
    int valid;
  };
  std::vector<StorageConfig> configs(numRanks_);
  configs[rank_] = {static_cast<uint64_t>(symmetricBufferBytes_),
                    domainSize,
                    pipelineChunkRows,
                    rowFence,
                    combineChunkRows,
                    aggregateCombine,
                    networkConfigValid_};
  communicator_.bootstrap()->allGather(configs.data(), sizeof(StorageConfig));
  for (const auto& config : configs) {
    EP_HOST_ASSERT(config.bytes == static_cast<uint64_t>(symmetricBufferBytes_) && config.ipcDomainSize == domainSize &&
                   config.pipelineChunkRows == pipelineChunkRows && config.rowFence == rowFence &&
                   config.combineChunkRows == combineChunkRows && config.aggregateCombine == aggregateCombine &&
                   config.valid);
  }
  if (aggregateCombine && (numTopk_ > 8 || combineChunkRows < 1 || combineChunkRows > 128)) {
    EP_THROW("MSCCLPP_EP_GPUNETIO_AGGREGATE_COMBINE requires topk <= 8 and COMBINE_CHUNK_ROWS in [1,128]");
  }
  if (pipelineChunkRows < 1 || pipelineChunkRows > 128) {
    EP_THROW("Hybrid dispatch requires DISPATCH_CHUNK_ROWS in [1,128]");
  }
  if (!aggregateCombine && combineChunkRows != 0) {
    EP_THROW("Direct Combine requires COMBINE_CHUNK_ROWS=0; enable AGGREGATE_COMBINE for chunked returns");
  }
  EP_HOST_ASSERT(rowFence >= 0);

  deviceContext_.gpuNetIoPipelineChunkRows_ = pipelineChunkRows;
  deviceContext_.gpuNetIoRowFence_ = rowFence;
  deviceContext_.gpuNetIoDistributeStats_ = envSwitch("MSCCLPP_EP_GPUNETIO_SEND_DISTRIBUTE_STATS", 0) == 1;
  deviceContext_.gpuNetIoCombineChunkRows_ = combineChunkRows;
  {
    // Validate the actual IPC cliques, then pair their sorted members as gateways.
    std::vector<uint64_t> mappings(numRanks_, 0);
    if (numRanks_ <= 1 || numRanks_ > 64 || numRanks_ % 2 != 0 || numRanks_ != 2 * domainSize) {
      EP_THROW("Hybrid dispatch requires exactly two equal, fully IPC-mapped domains");
    }
    for (int peer = 0; peer < numRanks_; ++peer) {
      if (peer == rank_ || peerMappedBufferBases_[peer] != nullptr) mappings[rank_] |= uint64_t{1} << peer;
    }
    communicator_.bootstrap()->allGather(mappings.data(), sizeof(uint64_t));
    const uint64_t all = ~uint64_t{0} >> (64 - numRanks_);
    const uint64_t first = mappings[0];
    const uint64_t second = all ^ first;
    std::vector<int> domains[2];
    bool valid = true;
    for (int peer = 0; peer < numRanks_; ++peer) {
      const int domain = (first & (uint64_t{1} << peer)) ? 0 : 1;
      valid &= mappings[peer] == (domain == 0 ? first : second);
      domains[domain].push_back(peer);
    }
    if (!valid || domains[0].size() != domains[1].size() || domains[0].size() != static_cast<size_t>(domainSize)) {
      EP_THROW("Hybrid dispatch requires exactly two equal, fully IPC-mapped domains");
    }
    const int domain = (first & (uint64_t{1} << rank_)) ? 0 : 1;
    const auto index = std::find(domains[domain].begin(), domains[domain].end(), rank_) - domains[domain].begin();
    deviceContext_.gpuNetIoGateway_ = domains[1 - domain][index];
    deviceContext_.gpuNetIoDistributeCapacity_ = maxTokensPerRank_;
    distributeScratch_ = mscclpp::detail::gpuCallocShared<int>(3 + configCellDiv(maxTokensPerRank_, pipelineChunkRows));
    deviceContext_.gpuNetIoDistributeScratch_ = distributeScratch_.get();
    if (aggregateCombine) {
      // Private caches survive remote reuse of the registered dispatch landing.
      aggregateMetadata_ =
          mscclpp::detail::gpuCallocShared<int>(1 + static_cast<size_t>(maxTokensPerRank_) * (1 + 2 * numTopk_));
      deviceContext_.gpuNetIoAggregateIncomingRows_ = aggregateMetadata_.get();
      deviceContext_.gpuNetIoAggregateTokenRows_ = aggregateMetadata_.get() + 1;
      deviceContext_.gpuNetIoAggregateRoutes_ = aggregateMetadata_.get() + 1 + maxTokensPerRank_;
      aggregateLocal_ = mscclpp::detail::gpuCallocShared<float>(static_cast<size_t>(maxTokensPerRank_) * hidden_);
      deviceContext_.gpuNetIoAggregateLocal_ = aggregateLocal_.get();
      aggregateChunkReady_ = mscclpp::detail::gpuCallocShared<int>(configCellDiv(maxTokensPerRank_, combineChunkRows));
      deviceContext_.gpuNetIoAggregateChunkReady_ = aggregateChunkReady_.get();
    }
  }
  if (rank_ == 0) {
    const char* rowFenceName = rowFence == GpuNetIoRowFenceGpu         ? "gpu"
                               : rowFence == GpuNetIoRowFenceGpuPoster ? "gpu+poster"
                                                                       : "sys";
    std::fprintf(stderr,
                 "[EPGPUNETIO] ranks=%d ipcDomain=%d dispatch=hybrid_dispatch aggregateCombine=%d "
                 "dispatchChunkRows=%d rowFence=%s combineChunkRows=%d\n",
                 numRanks_, domainSize, aggregateCombine, pipelineChunkRows, rowFenceName, combineChunkRows);
    std::fflush(stderr);
  }

#if defined(MSCCLPP_USE_GPUNETIO)
  std::string hca;
  if (const char* devices = std::getenv("MSCCLPP_EP_GPUNETIO_HCAS")) {
    hca = devices;
  }
  gpuNetIoService_ = std::make_shared<EpGpuNetIoService>(communicator_.bootstrap(), hca, deviceId_);
  gpuNetIoService_->setup(symmetricBuffer_.get(), symmetricBufferBytes_);
  deviceContext_.gpuNetIo_ = gpuNetIoService_->deviceContext();
  const auto storage = storageLayout();
  deviceContext_.gpuNetIoCombineFlagsBuffer_ = storage.gpuNetIoCombineFlagsBuffer_;
  deviceContext_.gpuNetIoCombineLandingBuffer_ = storage.gpuNetIoCombineLandingBuffer_;
  deviceContext_.gpuNetIoAggregateSend_ = storage.gpuNetIoAggregateSend_;
  deviceContext_.gpuNetIoAggregateRecv_ = storage.gpuNetIoAggregateRecv_;
  deviceContext_.gpuNetIoAggregateControl_ = storage.gpuNetIoAggregateControl_;
  {
    EpGpuNetIoDeviceContext network{};
    mscclpp::gpuMemcpy<EpGpuNetIoDeviceContext>(&network, gpuNetIoService_->deviceContext(), 1, cudaMemcpyDeviceToHost);
    if (network.numQpsPerPeer != 1 && network.numQpsPerPeer != 4) {
      EP_THROW("Hybrid dispatch requires 1 or 4 QPs per peer");
    }
    deviceContext_.gpuNetIoDistributeSend_ = storage.gpuNetIoDistributeSend_;
    deviceContext_.gpuNetIoDistributeRecv_ = storage.gpuNetIoDistributeRecv_;
    deviceContext_.gpuNetIoDistributeControl_ = storage.gpuNetIoDistributeControl_;
    std::fprintf(stderr,
                 "[EPGPUNETIO] rank=%d dispatch=hybrid_dispatch gateway=%d chunkRows=%d qps=%d wireRowBytes=%zu "
                 "sendBytes=%zu recvBytes=%zu ack=round combine=%s\n",
                 rank_, deviceContext_.gpuNetIoGateway_, pipelineChunkRows, network.numQpsPerPeer,
                 GpuNetIoDistributeRow::stride(hidden_, numTopk_),
                 maxTokensPerRank_ * GpuNetIoDistributeRow::stride(hidden_, numTopk_),
                 maxTokensPerRank_ * GpuNetIoDistributeRow::stride(hidden_, numTopk_),
                 aggregateCombine ? "aggregate_fp32" : "direct");
    std::fflush(stderr);
  }
#endif

  if (std::getenv("MSCCLPP_EP_DEBUG_TOPO") != nullptr) {
    std::string nvlinkPeerMap(static_cast<size_t>(numRanks_), '0');
    for (int r = 0; r < numRanks_; ++r) {
      if (r == rank_ || peerMappedBufferBases_[r] != nullptr) nvlinkPeerMap[r] = '1';
    }
    std::fprintf(stderr,
                 "[EPTOPO] rank=%d numRanks=%d numRanksPerIpcDomain=%d numRanksPerNode=%d gpuNetIo=%d "
                 "nvlinkPeerMap=%s\n",
                 rank_, numRanks_, numRanksPerIpcDomain_, numRanksPerNode_, deviceContext_.gpuNetIo_ != nullptr,
                 nvlinkPeerMap.c_str());
    std::fflush(stderr);
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
  const bool emitLocalExpertIds = request.emitLocalExpertIds;
  const DispatchLayout dispatchLayout = context.outputLayout_;
  const DispatchDataType dispatchDataType = request.quant.dispatchDataType;
  const float dispatchScale = request.quant.dispatchScale;
  const QuantConfig::InputMode inputMode = request.quant.inputMode;
  const DispatchDataType combineDataType = request.quant.combineDataType;
  const int numBlocks = request.numBlocks == 0 ? DefaultLatencyDispatchBlocks : request.numBlocks;
  const cudaStream_t stream = request.stream;

  EP_HOST_ASSERT(context.available_);
  EP_HOST_ASSERT(context.deviceContext_.devicePtr_ != nullptr);
  if (maxTokensPerRank <= 0 || maxTokensPerRank > context.maxTokensPerRank_ || numTokens < 0 ||
      numTokens > maxTokensPerRank) {
    EP_THROW("Latency requests require 0 <= numTokens <= maxTokensPerRank <= runtime capacity");
  }
  EP_HOST_ASSERT(invalidTokenExpertId < 0 || invalidTokenExpertId >= numExperts);
  if (emitLocalExpertIds && dispatchLayout != DispatchLayout::RANK_MAJOR) {
    EP_THROW("Local expert IDs are supported only by latency RANK_MAJOR dispatch");
  }
  EP_HOST_ASSERT(numBlocks - DispatchControlBlocks >= numRanks_ && numBlocks <= MaxDispatchBlocks);
  const bool staticFp8RankMajor =
      dispatchLayout == DispatchLayout::RANK_MAJOR && dispatchDataType == DispatchDataType::FP8_E4M3_STATIC;
  if (inputMode != QuantConfig::InputMode::PREQUANTIZED && inputMode != QuantConfig::InputMode::QUANTIZE) {
    EP_THROW("Unsupported static FP8 input mode");
  }
  if (staticFp8RankMajor) {
    if (!std::isfinite(dispatchScale) || dispatchScale <= 0.0f) {
      EP_THROW("Latency RANK_MAJOR static FP8 requires a finite positive quantization scale");
    }
    if (combineDataType == DispatchDataType::FP8_E4M3_STATIC && !std::isfinite(1.0f / dispatchScale)) {
      EP_THROW("Latency RANK_MAJOR static FP8 combine requires a quantization scale with a finite reciprocal");
    }
    EP_HOST_ASSERT(outputScales == nullptr);
    if (inputMode == QuantConfig::InputMode::QUANTIZE) {
      EP_HOST_ASSERT(reinterpret_cast<uintptr_t>(input) % alignof(mscclpp::bf16x8) == 0);
    }
  } else if (dispatchScale != 1.0f) {
    EP_THROW("A scalar quantization scale is supported only by FP8_E4M3_STATIC");
  } else if (inputMode != QuantConfig::InputMode::PREQUANTIZED) {
    EP_THROW("Static FP8 input mode is supported only by FP8_E4M3_STATIC");
  }
  if (dispatchDataType == DispatchDataType::FP8_E4M3 && dispatchLayout == DispatchLayout::RANK_MAJOR) {
    EP_THROW("Latency RANK_MAJOR block-scaled FP8 is unsupported; use FP8_E4M3_STATIC");
  }
  if (dispatchDataType == DispatchDataType::FP8_E4M3_STATIC && dispatchLayout != DispatchLayout::RANK_MAJOR) {
    EP_THROW("FP8_E4M3_STATIC dispatch requires latency RANK_MAJOR");
  }
  if (dispatchDataType == DispatchDataType::MXFP8_E4M3 && dispatchLayout != DispatchLayout::RANK_MAJOR) {
    EP_THROW("MXFP8 dispatch requires RANK_MAJOR layout");
  }
  if (!isSupportedCombineDataType(combineDataType)) {
    EP_THROW("Combine input supports only BF16 or FP8_E4M3_STATIC");
  }
  if (combineDataType == DispatchDataType::FP8_E4M3_STATIC && !staticFp8RankMajor) {
    EP_THROW("FP8_E4M3_STATIC combine input requires matching static FP8 dispatch");
  }

  const Workload workload{.epoch_ = context.epoch_,
                          .numTokens_ = numTokens,
                          .hidden_ = hidden,
                          .numTopk_ = numTopk,
                          .numExperts_ = numExperts,
                          .invalidTokenExpertId_ = invalidTokenExpertId,
                          .maxTokensPerRank_ = maxTokensPerRank,
                          .outputLayout_ = dispatchLayout,
                          .dispatchDataType_ = dispatchDataType,
                          .combineDataType_ = combineDataType,
                          .dispatchScale_ = dispatchScale,
                          .emitLocalExpertIds_ = emitLocalExpertIds,
                          .inputMode_ = inputMode};
  const auto allocationLayout = context.storageLayout();
  EP_HOST_ASSERT(allocationLayout.totalBytes_ <= static_cast<size_t>(context.symmetricBufferBytes_));
  const size_t workspaceBytes = workspaceSize(context.numRanks_, numExperts, maxTokensPerRank, numTopk);
  EP_HOST_ASSERT(workspaceBytes <= context.workspaceBytes_);
  if (context.useGpuNetIo_) {
    if (dispatchDataType != DispatchDataType::BF16 || combineDataType != DispatchDataType::BF16) {
      EP_THROW("GPUNetIO latency dispatch supports BF16 only");
    }
    if (emitLocalExpertIds) {
      EP_THROW("Local expert IDs are unsupported by GPUNetIO latency dispatch");
    }
    EP_HOST_ASSERT(output == allocationLayout.dispatchOutputBuffer_);
    EP_HOST_ASSERT(outputTopkIdx == allocationLayout.rankMajorTopkIdsBuffer_);
    EP_HOST_ASSERT(outputTopkWeights == allocationLayout.rankMajorTopkWeightsBuffer_);
    DispatchHandle handle(std::make_shared<const DispatchHandle::Impl>(latencyContext_, workload.epoch_, request));
    hybridDispatch(output, outputTopkIdx, outputTopkWeights, outputCount, input, topkIdx, topkWeights, workload,
                   allocationLayout.dispatchRecvBuffer_, context.deviceContext_, numBlocks, stream);
    return handle;
  }

  void* dispatchRecvBuffer = allocationLayout.dispatchRecvBuffer_;
  if (dispatchLayout == DispatchLayout::RANK_MAJOR) {
    EP_HOST_ASSERT(output == allocationLayout.dispatchOutputBuffer_);
    EP_HOST_ASSERT(outputTopkIdx == allocationLayout.rankMajorTopkIdsBuffer_);
    EP_HOST_ASSERT(outputTopkWeights == allocationLayout.rankMajorTopkWeightsBuffer_);
    if (dispatchDataType == DispatchDataType::MXFP8_E4M3) {
      EP_HOST_ASSERT(outputScales == allocationLayout.rankMajorScalesBuffer_);
      EP_HOST_ASSERT(reinterpret_cast<uintptr_t>(input) % alignof(mscclpp::bf16x8) == 0);
    }
  }

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
    EP_THROW("Stale dispatch handle: its dispatch has already been combined");
  }

  const auto* metadata = std::get_if<DispatchHandle::Impl::LatencyMetadata>(&handle.metadata_);
  if (metadata == nullptr) {
    EP_THROW("Dispatch handle does not contain latency metadata");
  }

  void* output = request.output;
  const void* input = request.input;
  const int64_t* topkIdx = metadata->topkIdx_;
  const float* topkWeights = request.applyRouterWeights ? metadata->topkWeights_ : nullptr;
  const int* srcInfo = metadata->srcInfo_;
  const int64_t* layoutRange = metadata->layoutRange_;
  const int numTokens = handle.numTokens_;
  const int hidden = context.hidden_;
  const int numTopk = context.numTopk_;
  const int maxTokensPerRank = handle.maxTokensPerRank_;
  const int numExperts = context.numExperts_;
  const DispatchLayout dispatchLayout = context.outputLayout_;
  const DispatchDataType dispatchDataType = handle.dispatchDataType_;
  const DispatchDataType combineDataType = handle.combineDataType_;
  const float dispatchScale = handle.dispatchScale_;
  const CombineMode mode = context.combineMode_;
  const int numBlocks = request.numBlocks == 0 ? DefaultLatencyCombineBlocks : request.numBlocks;
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
                          .dispatchDataType_ = dispatchDataType,
                          .combineDataType_ = combineDataType,
                          .dispatchScale_ = dispatchScale,
                          .emitLocalExpertIds_ = false,
                          .inputMode_ = QuantConfig::InputMode::PREQUANTIZED};
  EP_HOST_ASSERT(numBlocks > 0 && numBlocks <= MaxWorkerBlocks);

  const auto allocationLayout = context.storageLayout();
  EP_HOST_ASSERT(allocationLayout.totalBytes_ <= static_cast<size_t>(context.symmetricBufferBytes_));
  if (context.useGpuNetIo_) {
    EP_HOST_ASSERT(numBlocks >= context.numRanks_);
    EP_HOST_ASSERT(input == allocationLayout.combineBuffer_);
    hybridCombine(output, input, topkIdx, workload, allocationLayout.combineBuffer_,
                  allocationLayout.dispatchRecvBuffer_, context.deviceContext_, numBlocks, stream);
    ++context.epoch_;
    return;
  }

  void* combineBuffer = allocationLayout.combineBuffer_;
  void* dispatchRecvBuffer = allocationLayout.dispatchRecvBuffer_;
  if (dispatchLayout == DispatchLayout::RANK_MAJOR) EP_HOST_ASSERT(input == combineBuffer);

  if (dispatchLayout == DispatchLayout::RANK_MAJOR) {
    if (mode == CombineMode::DIRECT_SEND) {
      rankMajorDirectSendCombine(output, input, topkIdx, topkWeights, workload, combineBuffer, dispatchRecvBuffer,
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
  // Graph replay does not advance this host epoch. Rank-major latency graphs
  // therefore require at least two captured dispatch/combine pairs so adjacent
  // dispatches use different recorded epochs.
  ++context.epoch_;
}

}  // namespace ep
}  // namespace mscclpp
