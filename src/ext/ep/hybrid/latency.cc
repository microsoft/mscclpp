// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mscclpp/gpu_utils.hpp>
#include <string>

#include "hybrid/include/kernels.hpp"
#include "hybrid/include/runtime.hpp"
#include "include/exception.hpp"

#if defined(MSCCLPP_USE_GPUNETIO)
#include "hybrid/include/gpu_net_io.hpp"
#endif  // defined(MSCCLPP_USE_GPUNETIO)

namespace mscclpp {
namespace ep {
namespace hybrid {

namespace {

const char* gpuNetIoRowFenceName(int rowFence) {
  return rowFence == GpuNetIoRowFenceGpu ? "gpu" : rowFence == GpuNetIoRowFenceGpuPoster ? "gpu+poster" : "sys";
}

// Parse a 0/1 switch: unset keeps @p fallback, anything else is invalid (-1).
int envSwitch(const char* name, int fallback) {
  const char* requested = std::getenv(name);
  if (requested == nullptr) return fallback;
  const std::string value(requested);
  return value == "0" ? 0 : value == "1" ? 1 : -1;
}

// Parse a small unsigned integer: unset keeps @p fallback, anything else is invalid (-1).
int envCount(const char* name, int fallback, size_t maxDigits) {
  const char* requested = std::getenv(name);
  if (requested == nullptr) return fallback;
  const std::string value(requested);
  const bool numeric = !value.empty() && value.size() <= maxDigits &&
                       std::all_of(value.begin(), value.end(), [](char digit) { return digit >= '0' && digit <= '9'; });
  return numeric ? std::stoi(value) : -1;
}

}  // namespace

bool isSupportedHidden(int hidden) {
  return hidden == 2048 || hidden == 4096 || hidden == 4352 || hidden == 5120 || hidden == 6656 || hidden == 7168 ||
         hidden == 8192 || hidden == 8704 || hidden == 9216;
}

bool requiresHybrid(DispatchLayout outputLayout, bool useGpuNetIo) {
  return useGpuNetIo || outputLayout == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED;
}

bool gpuNetIoRequested(int numRanks, int numRanksPerIpcDomain, DispatchLayout outputLayout, CombineMode combineMode) {
#if defined(MSCCLPP_USE_GPUNETIO)
  if (numRanksPerIpcDomain >= numRanks) return false;
  const char* enable = std::getenv("MSCCLPP_EP_ENABLE_GPUNETIO");
  if (enable == nullptr || std::atoi(enable) == 0) return false;
  // The network protocol carries one pre-reduced BF16 row per rank (or per route for
  // TOPK_EXPANDED); expert-major and DIRECT_SEND stay IPC-only.
  return (outputLayout == DispatchLayout::RANK_MAJOR || outputLayout == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED) &&
         combineMode == CombineMode::RANK_LOCAL_REDUCE;
#else
  (void)numRanks;
  (void)numRanksPerIpcDomain;
  (void)outputLayout;
  (void)combineMode;
  return false;
#endif
}

LatencyRuntime::LatencyRuntime(int rank, int numRanks, int numRanksPerIpcDomain, int numRanksPerNode,
                               int maxTokensPerRank, int hidden, int numExperts, int numTopk,
                               DispatchLayout outputLayout, CombineMode combineMode, bool useGpuNetIo)
    : rank_(rank),
      numRanks_(numRanks),
      numRanksPerIpcDomain_(numRanksPerIpcDomain),
      numRanksPerNode_(numRanksPerNode),
      maxTokensPerRank_(maxTokensPerRank),
      hidden_(hidden),
      numExperts_(numExperts),
      numTopk_(numTopk),
      outputLayout_(outputLayout),
      combineMode_(combineMode),
      useGpuNetIo_(useGpuNetIo) {
  if (!isSupportedHidden(hidden)) {
    EP_THROW("Unsupported hybrid latency hidden size: " + std::to_string(hidden));
  }
  EP_HOST_ASSERT(outputLayout == DispatchLayout::RANK_MAJOR ||
                 outputLayout == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED);
  EP_HOST_ASSERT(combineMode == CombineMode::RANK_LOCAL_REDUCE);
  if (outputLayout == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED) {
    EP_HOST_ASSERT(numTopk <= 9);
    EP_HOST_ASSERT(static_cast<int64_t>(maxTokensPerRank) * numTopk <= INT32_MAX);
  }
  symmetricBufferBytes_ = latencyStorageSize(maxTokensPerRank, hidden, numRanks, numExperts, numTopk, outputLayout,
                                             combineMode, useGpuNetIo);
  workspaceBytes_ = workspaceSize(numRanks, numExperts, maxTokensPerRank, numTopk);
  EP_HOST_ASSERT(symmetricBufferBytes_ % BufferAlignmentBytes == 0);
}

LatencyRuntime::~LatencyRuntime() noexcept(false) {
  if (gpuNetIoService_ != nullptr) MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  gpuNetIoService_.reset();
  if (deviceContext_.devicePtr_ != nullptr) MSCCLPP_CUDATHROW(cudaFree(deviceContext_.devicePtr_));
}

LatencyStorageLayout LatencyRuntime::layout(void* symmetricBuffer) const {
  return LatencyStorageLayout(symmetricBuffer, maxTokensPerRank_, hidden_, numRanks_, numExperts_, numTopk_,
                              outputLayout_, combineMode_, useGpuNetIo_);
}

void LatencyRuntime::initialize(mscclpp::Communicator& communicator, void* symmetricBuffer,
                                const std::vector<void*>& peerMappedBufferBases, void** peerMappedBufferBasesGpu,
                                mscclpp::BaseMemoryChannelDeviceHandle* channels, void* workspace,
                                int maxSharedMemoryPerBlock, int numSms, int deviceId) {
  EP_HOST_ASSERT(deviceContext_.devicePtr_ == nullptr);
  const bool gpuNetIoRankMajor = useGpuNetIo_ && outputLayout_ == DispatchLayout::RANK_MAJOR;
  const int numNodes = numRanksPerNode_ > 0 && numRanks_ % numRanksPerNode_ == 0 ? numRanks_ / numRanksPerNode_ : 0;

  // Rank-major GPUNetIO pipeline knobs. Parallel staging, the GPU-scope row fence and the
  // deferred send flush default on with batched dispatch. They apply only to GPUNetIO
  // RANK_MAJOR, so a shared environment does not affect IPC or TOPK_EXPANDED runtimes.
  const bool knobs = gpuNetIoRankMajor;
  const int batchedDispatch = knobs ? envSwitch("MSCCLPP_EP_GPUNETIO_BATCHED_DISPATCH", 0) : 0;
  const int batchAllQps = knobs ? envSwitch("MSCCLPP_EP_GPUNETIO_BATCH_ALL_QPS", 0) : 0;
  // Default dispatch chunk rows: 4 rows for 2-7 nodes, 16 rows for >= 8 nodes.
  const int defaultChunkRows = batchedDispatch == 1 && numNodes >= 2 ? (numNodes >= 8 ? 16 : 4) : 0;
  const int pipelineChunkRows = knobs ? envCount("MSCCLPP_EP_GPUNETIO_DISPATCH_CHUNK_ROWS", defaultChunkRows, 3) : 0;
  const int asyncDispatch = knobs ? envSwitch("MSCCLPP_EP_GPUNETIO_ASYNC_DISPATCH", 0) : 0;
  const int parallelStage = knobs ? envCount("MSCCLPP_EP_GPUNETIO_PARALLEL_STAGE", batchedDispatch == 1 ? 1 : 0, 2) : 0;
  int rowFence = batchedDispatch == 1 ? GpuNetIoRowFenceGpu : GpuNetIoRowFenceSystem;
  if (const char* requested = knobs ? std::getenv("MSCCLPP_EP_GPUNETIO_ROW_FENCE") : nullptr) {
    const std::string value(requested);
    rowFence = value == "sys"          ? GpuNetIoRowFenceSystem
               : value == "gpu"        ? GpuNetIoRowFenceGpu
               : value == "gpu+poster" ? GpuNetIoRowFenceGpuPoster
                                       : -1;
  }
  const int deferFlush = knobs ? envSwitch("MSCCLPP_EP_GPUNETIO_DEFER_FLUSH", batchedDispatch == 1 ? 1 : 0) : 0;
  const int defaultCombineChunkRows = knobs && numNodes >= 2 ? std::min(32, maxTokensPerRank_) : 0;
  const int combineChunkRows =
      knobs ? envCount("MSCCLPP_EP_GPUNETIO_COMBINE_CHUNK_ROWS", defaultCombineChunkRows, 3) : 0;

  // Every rank must agree on the storage size and on every protocol knob.
  struct StorageConfig {
    uint64_t bytes;
    int ipcDomainSize;
    int useGpuNetIo;
    int batchedDispatch;
    int batchAllQps;
    int pipelineChunkRows;
    int asyncDispatch;
    int parallelStage;
    int rowFence;
    int deferFlush;
    int combineChunkRows;
  };
  std::vector<StorageConfig> storageConfigs(numRanks_);
  storageConfigs[rank_] = {static_cast<uint64_t>(symmetricBufferBytes_),
                           numRanksPerIpcDomain_,
                           useGpuNetIo_ ? 1 : 0,
                           batchedDispatch,
                           batchAllQps,
                           pipelineChunkRows,
                           asyncDispatch,
                           parallelStage,
                           rowFence,
                           deferFlush,
                           combineChunkRows};
  communicator.bootstrap()->allGather(storageConfigs.data(), sizeof(StorageConfig));
  for (const auto& config : storageConfigs) {
    EP_HOST_ASSERT(config.bytes == static_cast<uint64_t>(symmetricBufferBytes_) &&
                   config.ipcDomainSize == numRanksPerIpcDomain_ && config.useGpuNetIo == (useGpuNetIo_ ? 1 : 0) &&
                   config.batchedDispatch == batchedDispatch && config.batchAllQps == batchAllQps &&
                   config.pipelineChunkRows == pipelineChunkRows && config.asyncDispatch == asyncDispatch &&
                   config.parallelStage == parallelStage && config.rowFence == rowFence &&
                   config.deferFlush == deferFlush && config.combineChunkRows == combineChunkRows);
  }
  EP_HOST_ASSERT(batchedDispatch >= 0 && batchAllQps >= 0 && asyncDispatch >= 0 && deferFlush >= 0);
  EP_HOST_ASSERT(pipelineChunkRows >= 0 && pipelineChunkRows <= 128);
  EP_HOST_ASSERT(asyncDispatch == 0 || (batchedDispatch == 1 && pipelineChunkRows > 0));
  EP_HOST_ASSERT(pipelineChunkRows == 0 || batchedDispatch == 1);
  EP_HOST_ASSERT(batchAllQps == 0 || batchedDispatch == 1);
  EP_HOST_ASSERT(parallelStage >= 0 && parallelStage <= 16);
  EP_HOST_ASSERT(parallelStage == 0 || batchedDispatch == 1);
  EP_HOST_ASSERT(rowFence >= 0);
  EP_HOST_ASSERT(rowFence == GpuNetIoRowFenceSystem || batchedDispatch == 1);
  EP_HOST_ASSERT(batchedDispatch == 0 ||
                 (gpuNetIoRankMajor && static_cast<size_t>(numRanks_) * maxTokensPerRank_ <= GpuNetIoStagingSlots));
  EP_HOST_ASSERT(combineChunkRows >= 0 && combineChunkRows <= maxTokensPerRank_);

  deviceContext_ = {.localBufferBase_ = symmetricBuffer,
                    .peerBufferBases_ = peerMappedBufferBasesGpu,
                    .peerPayloadBases_ = nullptr,
                    .channels_ = channels,
                    .workspace_ = workspace,
                    .combineRecvIdx_ = nullptr,
                    .mappedRecvCounter_ = nullptr,
                    .mappedRecvExpertCounters_ = nullptr,
                    .maxSharedMemoryPerBlock_ = maxSharedMemoryPerBlock,
                    .numSms_ = numSms,
                    .deviceId_ = deviceId,
                    .rank_ = rank_,
                    .numRanks_ = numRanks_};
  deviceContext_.gpuNetIoBatchedDispatch_ = batchedDispatch == 1;
  deviceContext_.gpuNetIoBatchAllQps_ = batchAllQps == 1;
  deviceContext_.gpuNetIoPipelineChunkRows_ = pipelineChunkRows;
  deviceContext_.gpuNetIoParallelStage_ = parallelStage;
  deviceContext_.gpuNetIoRowFence_ = rowFence;
  deviceContext_.gpuNetIoAsyncDispatch_ = asyncDispatch == 1;
  // Deferred send drain: the next dispatch waits for this dispatch's send tickets.
  deviceContext_.gpuNetIoDeferFlush_ = gpuNetIoRankMajor && asyncDispatch == 0 && deferFlush == 1;
  if (gpuNetIoRankMajor && combineChunkRows > 0) {
    deviceContext_.gpuNetIoCombineChunkRows_ = combineChunkRows;
    combineChunkBase_ =
        mscclpp::detail::gpuCallocShared<uint64_t>(static_cast<size_t>(numRanks_) * GpuNetIoMaxQpsPerPeer);
    deviceContext_.gpuNetIoCombineChunkBase_ = combineChunkBase_.get();
  }
  if (const char* noBarrier = std::getenv("MSCCLPP_EP_GPUNETIO_COMBINE_NO_BARRIER")) {
    deviceContext_.gpuNetIoCombineNoBarrier_ = useGpuNetIo_ && std::string(noBarrier) == "1";
  }
  if (asyncDispatch == 1) {
    dispatchPeers_ = mscclpp::detail::gpuCallocShared<GpuNetIoDispatchPeerState>(numRanks_);
    deviceContext_.gpuNetIoDispatchPeers_ = dispatchPeers_.get();
  } else if (gpuNetIoRankMajor) {
    dispatchDone_ = mscclpp::detail::gpuCallocShared<int>(1);
    deviceContext_.gpuNetIoDispatchDone_ = dispatchDone_.get();
  }
  if (deviceContext_.gpuNetIoDeferFlush_) {
    dispatchTickets_ =
        mscclpp::detail::gpuCallocShared<uint64_t>(static_cast<size_t>(numRanks_) * GpuNetIoMaxQpsPerPeer);
    deviceContext_.gpuNetIoDispatchTickets_ = dispatchTickets_.get();
  }
  if (pipelineChunkRows > 0) {
    const size_t counters = static_cast<size_t>(numRanks_) * configCellDiv(maxTokensPerRank_, pipelineChunkRows);
    // Early tail post: producer-posted chunks only (not async).
    const char* earlyTail = std::getenv("MSCCLPP_EP_GPUNETIO_EARLY_TAIL");
    const bool tailRows = earlyTail != nullptr && std::string(earlyTail) == "1" && gpuNetIoRankMajor &&
                          asyncDispatch == 0 && deviceContext_.gpuNetIoBatchedDispatch_;
    chunkReady_ = mscclpp::detail::gpuCallocShared<int>(counters + (tailRows ? numRanks_ : 0));
    deviceContext_.gpuNetIoTailRows_ = tailRows ? chunkReady_.get() : nullptr;
    deviceContext_.gpuNetIoChunkReady_ = chunkReady_.get() + (tailRows ? numRanks_ : 0);
  }
  if (const char* tma = std::getenv("MSCCLPP_EP_GPUNETIO_TMA_STAGE");
      tma != nullptr && (std::string(tma) == "1" || std::string(tma) == "2")) {
    deviceContext_.gpuNetIoTmaStage_ = (useGpuNetIo_ && deviceContext_.gpuNetIoBatchedDispatch_) ? std::atoi(tma) : 0;
  }
  if (useGpuNetIo_ && rank_ == 0) {
    std::fprintf(stderr,
                 "[EPGPUNETIO] ranks=%d ipcDomain=%d nodes=%d batched=%d stripe=%s dispatchChunkRows=%d async=%d "
                 "parallelStage=%d rowFence=%s deferFlush=%d combineChunkRows=%d\n",
                 numRanks_, numRanksPerIpcDomain_, numNodes, batchedDispatch, batchAllQps ? "all_qps" : "hca",
                 pipelineChunkRows, asyncDispatch, parallelStage, gpuNetIoRowFenceName(rowFence),
                 deviceContext_.gpuNetIoDeferFlush_ ? 1 : 0, deviceContext_.gpuNetIoCombineChunkRows_);
    std::fflush(stderr);
  }

#if defined(MSCCLPP_USE_GPUNETIO)
  // Bring up GPU-initiated networking for peers outside this rank's CUDA IPC domain and
  // publish its device context and symmetric staging/flag regions.
  if (useGpuNetIo_) {
    std::string hca;
    if (const char* devices = std::getenv("MSCCLPP_EP_GPUNETIO_HCAS")) {
      hca = devices;
    } else if (const char* device = std::getenv("MSCCLPP_EP_GPUNETIO_HCA")) {
      hca = device;
    }
    auto service = std::make_shared<EpGpuNetIoService>(communicator.bootstrap(), hca, deviceId);
    service->setup(symmetricBuffer, symmetricBufferBytes_);
    gpuNetIoService_ = service;
    deviceContext_.gpuNetIo_ = service->deviceContext();
    const LatencyStorageLayout storage = layout(symmetricBuffer);
    deviceContext_.gpuNetIoStagingBuffer_ = storage.gpuNetIoStagingBuffer_;
    deviceContext_.gpuNetIoFlagsBuffer_ = storage.gpuNetIoFlagsBuffer_;
    deviceContext_.gpuNetIoCombineFlagsBuffer_ = storage.gpuNetIoCombineFlagsBuffer_;
    deviceContext_.gpuNetIoCombineLandingBuffer_ = storage.gpuNetIoCombineLandingBuffer_;
    deviceContext_.gpuNetIoSlotStride_ = storage.gpuNetIoSlotStride_;
    static_assert(EpGpuNetIoDeviceContext::SignalQueues == GpuNetIoMaxQpsPerPeer);
    const auto offsetOf = [symmetricBuffer](void* pointer) {
      return pointer == nullptr ? EpGpuNetIoDeviceContext::NoOffset
                                : static_cast<uint64_t>(static_cast<uint8_t*>(pointer) -
                                                        static_cast<uint8_t*>(symmetricBuffer));
    };
    const uint64_t flagsBytes =
        storage.gpuNetIoFlagsBuffer_ == nullptr || storage.gpuNetIoCombineFlagsBuffer_ == nullptr
            ? 0
            : static_cast<uint64_t>(static_cast<uint8_t*>(storage.gpuNetIoCombineFlagsBuffer_) -
                                    static_cast<uint8_t*>(storage.gpuNetIoFlagsBuffer_));
    service->setSignalRegions(offsetOf(storage.gpuNetIoFlagsBuffer_), flagsBytes,
                              offsetOf(storage.expandedSyncFlags_));
  }
#endif  // defined(MSCCLPP_USE_GPUNETIO)

  if (outputLayout_ == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED) {
    // Expanded fast paths are enabled only when every rank agrees.
    std::vector<int> enabled(numRanks_, 0);
    if (numRanksPerIpcDomain_ >= numRanks_) {
      const char* requested = std::getenv("MSCCLPP_EP_EXPANDED_IPC_FASTPATH");
      const bool allMapped = std::all_of(peerMappedBufferBases.begin(), peerMappedBufferBases.end(),
                                         [](void* base) { return base != nullptr; });
      enabled[rank_] =
          (requested == nullptr || std::atoi(requested) != 0) && allMapped && deviceContext_.gpuNetIo_ == nullptr;
    } else {
      const char* requested = std::getenv("MSCCLPP_EP_EXPANDED_GPUNETIO_FASTPATH");
      enabled[rank_] = (requested == nullptr || std::atoi(requested) != 0) && deviceContext_.gpuNetIo_ != nullptr;
    }
    communicator.bootstrap()->allGather(enabled.data(), sizeof(int));
    const bool allEnabled = std::all_of(enabled.begin(), enabled.end(), [](int value) { return value != 0; });
    if (numRanksPerIpcDomain_ >= numRanks_) {
      deviceContext_.expandedIpcFastPath_ = allEnabled;
    } else {
      deviceContext_.expandedGpuNetIoFastPath_ = allEnabled;
    }
  }
  if (std::getenv("MSCCLPP_EP_DEBUG_TOPO") != nullptr) {
    // nvlinkPeerMap[r] = '1' when peer r is IPC-mapped and '0' when it is served over GPUNetIO.
    std::string nvlinkPeerMap(static_cast<size_t>(numRanks_), '0');
    for (int r = 0; r < numRanks_; ++r) {
      if (r == rank_ || peerMappedBufferBases[r] != nullptr) nvlinkPeerMap[r] = '1';
    }
    std::fprintf(stderr,
                 "[EPTOPO] rank=%d numRanks=%d numRanksPerIpcDomain=%d numRanksPerNode=%d gpuNetIo=%d "
                 "expandedIpcFastPath=%d expandedGpuNetIoFastPath=%d nvlinkPeerMap=%s\n",
                 rank_, numRanks_, numRanksPerIpcDomain_, numRanksPerNode_, deviceContext_.gpuNetIo_ != nullptr ? 1 : 0,
                 deviceContext_.expandedIpcFastPath_ ? 1 : 0, deviceContext_.expandedGpuNetIoFastPath_ ? 1 : 0,
                 nvlinkPeerMap.c_str());
    std::fflush(stderr);
  }
  deviceContext_.devicePtr_ = static_cast<DeviceContext*>(mscclpp::detail::gpuCalloc(sizeof(DeviceContext)));
  mscclpp::gpuMemcpy<DeviceContext>(deviceContext_.devicePtr_, &deviceContext_, 1, cudaMemcpyHostToDevice);
}

void LatencyRuntime::dispatch(void* output, int* outputTopkIdx, float* outputTopkWeights, int* outputCount,
                              const void* input, const int64_t* topkIdx, const float* topkWeights,
                              const Workload& workload, void* dispatchRecvBuffer, int numBlocks, cudaStream_t stream) {
  EP_HOST_ASSERT(deviceContext_.devicePtr_ != nullptr);
  EP_HOST_ASSERT(workspaceSize(numRanks_, numExperts_, workload.maxTokensPerRank_, numTopk_) <= workspaceBytes_);
  if (outputLayout_ == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED) {
    EP_HOST_ASSERT(workload.maxTokensPerRank_ == maxTokensPerRank_);
    topk_expanded::dispatch(output, outputTopkIdx, outputTopkWeights, outputCount, input, topkIdx, topkWeights,
                            workload, deviceContext_, numBlocks, stream);
  } else {
    rankMajorDispatch(output, outputTopkIdx, outputTopkWeights, outputCount, input, topkIdx, topkWeights, workload,
                      dispatchRecvBuffer, deviceContext_, numBlocks, stream);
  }
}

void LatencyRuntime::combine(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights,
                             const Workload& workload, void* combineRecvBuffer, void* dispatchRecvBuffer, int numBlocks,
                             cudaStream_t stream) {
  EP_HOST_ASSERT(deviceContext_.devicePtr_ != nullptr);
  if (deviceContext_.gpuNetIo_ != nullptr) EP_HOST_ASSERT(numBlocks >= numRanks_);
  if (outputLayout_ == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED) {
    EP_HOST_ASSERT(workload.maxTokensPerRank_ == maxTokensPerRank_ && numBlocks > numRanks_);
    topk_expanded::combine(output, input, topkIdx, topkWeights, workload, deviceContext_, numBlocks, stream);
  } else {
    rankMajorGatherReduceCombine(output, input, topkIdx, workload, combineRecvBuffer, dispatchRecvBuffer,
                                 deviceContext_, numBlocks, stream);
  }
}

}  // namespace hybrid
}  // namespace ep
}  // namespace mscclpp
