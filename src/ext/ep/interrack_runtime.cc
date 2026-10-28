// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "interrack_runtime.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mscclpp/gpu_utils.hpp>
#include <string>

#include "include/exception.hpp"
#include "kernels.hpp"

#if defined(MSCCLPP_USE_GPUNETIO)
#include "gpu_net_io.hpp"
#endif  // defined(MSCCLPP_USE_GPUNETIO)

namespace mscclpp {
namespace ep {
namespace interrack {

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

bool gpuNetIoRequested(int numRanks, int numRanksPerIpcDomain, DispatchLayout outputLayout, CombineMode combineMode) {
#if defined(MSCCLPP_USE_GPUNETIO)
  if (numRanksPerIpcDomain >= numRanks) return false;
  const char* enable = std::getenv("MSCCLPP_EP_ENABLE_GPUNETIO");
  if (enable == nullptr || std::atoi(enable) == 0) return false;
  // The network protocol carries one pre-reduced BF16 row per rank.
  return outputLayout == DispatchLayout::RANK_MAJOR && combineMode == CombineMode::RANK_LOCAL_REDUCE;
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
      useGpuNetIo_(useGpuNetIo),
      sendDistribute_(useGpuNetIo && outputLayout == DispatchLayout::RANK_MAJOR &&
                      envSwitch("MSCCLPP_EP_GPUNETIO_SEND_DISTRIBUTE", 0) == 1),
      aggregateCombine_(sendDistribute_ && envSwitch("MSCCLPP_EP_GPUNETIO_AGGREGATE_COMBINE", 0) == 1) {
  if (!isSupportedHidden(hidden)) {
    EP_THROW("Unsupported GPUNetIO latency hidden size: " + std::to_string(hidden));
  }
  EP_HOST_ASSERT(outputLayout == DispatchLayout::RANK_MAJOR);
  EP_HOST_ASSERT(combineMode == CombineMode::RANK_LOCAL_REDUCE);
  symmetricBufferBytes_ = latencyStorageSize(maxTokensPerRank, hidden, numRanks, numExperts, numTopk, outputLayout,
                                             combineMode, useGpuNetIo, sendDistribute_, aggregateCombine_);
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
                              outputLayout_, combineMode_, useGpuNetIo_, sendDistribute_, aggregateCombine_);
}

void LatencyRuntime::initialize(mscclpp::Communicator& communicator, void* symmetricBuffer,
                                const std::vector<void*>& peerMappedBufferBases, void** peerMappedBufferBasesGpu,
                                mscclpp::BaseMemoryChannelDeviceHandle* channels, void* workspace,
                                int maxSharedMemoryPerBlock, int numSms, int deviceId) {
  EP_HOST_ASSERT(deviceContext_.devicePtr_ == nullptr);
  const bool gpuNetIoRankMajor = useGpuNetIo_ && outputLayout_ == DispatchLayout::RANK_MAJOR;
  const int domainSize = numRanksPerIpcDomain_ > 0 ? numRanksPerIpcDomain_ : numRanksPerNode_;
  const int numNodes = domainSize > 0 && numRanks_ % domainSize == 0 ? numRanks_ / domainSize : 0;

  // Rank-major GPUNetIO pipeline knobs. Batched dispatch uses parallel staging,
  // GPU-scope row fences and dedicated receive CTAs. They apply only to GPUNetIO
  // RANK_MAJOR, so a shared environment does not affect IPC runtimes.
  const bool knobs = gpuNetIoRankMajor && !sendDistribute_;
  const int sendDistribute = envSwitch("MSCCLPP_EP_GPUNETIO_SEND_DISTRIBUTE", 0);
  const int aggregateCombine = envSwitch("MSCCLPP_EP_GPUNETIO_AGGREGATE_COMBINE", 0);
  const int defaultBatched =
      knobs && static_cast<size_t>(numRanks_) * maxTokensPerRank_ <= GpuNetIoStagingSlots ? 1 : 0;
  const int batchedDispatch = knobs ? envSwitch("MSCCLPP_EP_GPUNETIO_BATCHED_DISPATCH", defaultBatched) : 0;
  // Default dispatch chunk rows: 8 rows for 2-7 nodes, 16 rows for >= 8 nodes.
  const int defaultChunkRows = sendDistribute_                         ? GpuNetIoDistributeDefaultChunkRows
                               : batchedDispatch == 1 && numNodes >= 2 ? (numNodes >= 8 ? 16 : 8)
                                                                       : 0;
  const int requestedChunkRows =
      gpuNetIoRankMajor ? envCount("MSCCLPP_EP_GPUNETIO_DISPATCH_CHUNK_ROWS", defaultChunkRows, 3) : 0;
  // Whole-destination batching is one capacity-sized chunk. Dedicated receive
  // CTAs and per-peer completion apply to every batched dispatch, including zero rows.
  const bool asyncDispatch = batchedDispatch == 1;
  const int pipelineChunkRows = asyncDispatch && requestedChunkRows == 0 ? maxTokensPerRank_ : requestedChunkRows;
  const int parallelStage = knobs ? envCount("MSCCLPP_EP_GPUNETIO_PARALLEL_STAGE", batchedDispatch == 1 ? 1 : 0, 2) : 0;
  int rowFence = sendDistribute_        ? GpuNetIoRowFenceGpuPoster
                 : batchedDispatch == 1 ? GpuNetIoRowFenceGpu
                                        : GpuNetIoRowFenceSystem;
  if (const char* requested = gpuNetIoRankMajor ? std::getenv("MSCCLPP_EP_GPUNETIO_ROW_FENCE") : nullptr) {
    const std::string value(requested);
    rowFence = value == "sys"          ? GpuNetIoRowFenceSystem
               : value == "gpu"        ? GpuNetIoRowFenceGpu
               : value == "gpu+poster" ? GpuNetIoRowFenceGpuPoster
                                       : -1;
  }
  const int deferFlush = knobs ? envSwitch("MSCCLPP_EP_GPUNETIO_DEFER_FLUSH", batchedDispatch == 1 ? 1 : 0) : 0;
  const int defaultCombineChunkRows = aggregateCombine_ ? 8 : 0;
  const int combineChunkRows =
      gpuNetIoRankMajor ? envCount("MSCCLPP_EP_GPUNETIO_COMBINE_CHUNK_ROWS", defaultCombineChunkRows, 3) : 0;

  // Every rank must agree on the storage size and on every protocol knob.
  struct StorageConfig {
    uint64_t bytes;
    int ipcDomainSize;
    int useGpuNetIo;
    int batchedDispatch;
    int pipelineChunkRows;
    int parallelStage;
    int rowFence;
    int deferFlush;
    int combineChunkRows;
    int sendDistribute;
    int aggregateCombine;
  };
  std::vector<StorageConfig> storageConfigs(numRanks_);
  storageConfigs[rank_] = {static_cast<uint64_t>(symmetricBufferBytes_),
                           numRanksPerIpcDomain_,
                           useGpuNetIo_ ? 1 : 0,
                           batchedDispatch,
                           pipelineChunkRows,
                           parallelStage,
                           rowFence,
                           deferFlush,
                           combineChunkRows,
                           sendDistribute,
                           aggregateCombine};
  communicator.bootstrap()->allGather(storageConfigs.data(), sizeof(StorageConfig));
  for (const auto& config : storageConfigs) {
    EP_HOST_ASSERT(config.bytes == static_cast<uint64_t>(symmetricBufferBytes_) &&
                   config.ipcDomainSize == numRanksPerIpcDomain_ && config.useGpuNetIo == (useGpuNetIo_ ? 1 : 0) &&
                   config.batchedDispatch == batchedDispatch && config.pipelineChunkRows == pipelineChunkRows &&
                   config.parallelStage == parallelStage && config.rowFence == rowFence &&
                   config.deferFlush == deferFlush && config.combineChunkRows == combineChunkRows &&
                   config.sendDistribute == sendDistribute && config.aggregateCombine == aggregateCombine);
  }
  EP_HOST_ASSERT(sendDistribute >= 0);
  EP_HOST_ASSERT(aggregateCombine >= 0);
  if (gpuNetIoRankMajor && aggregateCombine == 1 && !sendDistribute_) {
    EP_THROW("MSCCLPP_EP_GPUNETIO_AGGREGATE_COMBINE requires SEND_DISTRIBUTE=1");
  }
  if (aggregateCombine_ && (numTopk_ > 8 || combineChunkRows < 1 || combineChunkRows > 128)) {
    EP_THROW("MSCCLPP_EP_GPUNETIO_AGGREGATE_COMBINE requires topk <= 8 and COMBINE_CHUNK_ROWS in [1,128]");
  }
  EP_HOST_ASSERT(batchedDispatch >= 0 && deferFlush >= 0);
  EP_HOST_ASSERT(requestedChunkRows >= 0 && requestedChunkRows <= 128);
  EP_HOST_ASSERT(pipelineChunkRows == 0 || batchedDispatch == 1 || sendDistribute_);
  EP_HOST_ASSERT(!sendDistribute_ || pipelineChunkRows > 0);
  EP_HOST_ASSERT(parallelStage >= 0 && parallelStage <= 16);
  EP_HOST_ASSERT(parallelStage == 0 || batchedDispatch == 1);
  EP_HOST_ASSERT(rowFence >= 0);
  EP_HOST_ASSERT(rowFence == GpuNetIoRowFenceSystem || batchedDispatch == 1 || sendDistribute_);
  EP_HOST_ASSERT(batchedDispatch == 0 ||
                 (gpuNetIoRankMajor && static_cast<size_t>(numRanks_) * maxTokensPerRank_ <= GpuNetIoStagingSlots));
  EP_HOST_ASSERT(combineChunkRows >= 0 && (aggregateCombine_ || combineChunkRows <= maxTokensPerRank_));

  deviceContext_ = {.localBufferBase_ = symmetricBuffer,
                    .peerBufferBases_ = peerMappedBufferBasesGpu,
                    .channels_ = channels,
                    .workspace_ = workspace,
                    .maxSharedMemoryPerBlock_ = maxSharedMemoryPerBlock,
                    .numSms_ = numSms,
                    .deviceId_ = deviceId,
                    .rank_ = rank_,
                    .numRanks_ = numRanks_};
  deviceContext_.gpuNetIoBatchedDispatch_ = batchedDispatch == 1;
  deviceContext_.gpuNetIoPipelineChunkRows_ = pipelineChunkRows;
  deviceContext_.gpuNetIoParallelStage_ = parallelStage;
  deviceContext_.gpuNetIoRowFence_ = rowFence;
  deviceContext_.gpuNetIoSendDistribute_ = sendDistribute_;
  deviceContext_.gpuNetIoAggregateCombine_ = aggregateCombine_;
  deviceContext_.gpuNetIoDistributeStats_ =
      sendDistribute_ && envSwitch("MSCCLPP_EP_GPUNETIO_SEND_DISTRIBUTE_STATS", 0) == 1;
  if (sendDistribute_) {
    // Validate the complete IPC mapping matrix, not rank/domain arithmetic. Sorted
    // members of the two cliques are paired, including non-contiguous memberships.
    std::vector<uint64_t> mappings(numRanks_, 0);
    EP_HOST_ASSERT(numRanks_ > 1 && numRanks_ <= 64 && numRanks_ % 2 == 0);
    for (int peer = 0; peer < numRanks_; ++peer) {
      if (peer == rank_ || peerMappedBufferBases[peer] != nullptr) mappings[rank_] |= uint64_t{1} << peer;
    }
    communicator.bootstrap()->allGather(mappings.data(), sizeof(uint64_t));
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
      EP_THROW("MSCCLPP_EP_GPUNETIO_SEND_DISTRIBUTE requires exactly two equal, fully IPC-mapped domains");
    }
    const int domain = (first & (uint64_t{1} << rank_)) ? 0 : 1;
    const auto index = std::find(domains[domain].begin(), domains[domain].end(), rank_) - domains[domain].begin();
    deviceContext_.gpuNetIoGateway_ = domains[1 - domain][index];
    deviceContext_.gpuNetIoDistributeCapacity_ = maxTokensPerRank_;
    distributeScratch_ = mscclpp::detail::gpuCallocShared<int>(3 + configCellDiv(maxTokensPerRank_, pipelineChunkRows));
    deviceContext_.gpuNetIoDistributeScratch_ = distributeScratch_.get();
    if (combineChunkRows != 0 && !aggregateCombine_) {
      EP_THROW("MSCCLPP_EP_GPUNETIO_SEND_DISTRIBUTE requires unchunked Combine (COMBINE_CHUNK_ROWS=0)");
    }
    if (aggregateCombine_) {
      // All caches are GPU-private: a source may already overwrite its gateway's
      // registered dispatch landing while that gateway is still combining.
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
  } else if (sendDistribute == 1 && rank_ == 0) {
    std::fprintf(stderr, "[EPGPUNETIO] sendDistribute=0 (only cross-domain RANK_MAJOR is supported)\n");
  }
  if (aggregateCombine == 1 && !aggregateCombine_ && rank_ == 0) {
    std::fprintf(stderr, "[EPGPUNETIO] aggregateCombine=0 (only cross-domain RANK_MAJOR is supported)\n");
  }
  // Deferred send drain: the next dispatch waits for this dispatch's send tickets.
  deviceContext_.gpuNetIoDeferFlush_ = gpuNetIoRankMajor && asyncDispatch == 0 && deferFlush == 1;
  if (gpuNetIoRankMajor && combineChunkRows > 0) {
    deviceContext_.gpuNetIoCombineChunkRows_ = combineChunkRows;
    if (!aggregateCombine_) {
      combineChunkBase_ =
          mscclpp::detail::gpuCallocShared<uint64_t>(static_cast<size_t>(numRanks_) * GpuNetIoMaxQpsPerPeer);
      deviceContext_.gpuNetIoCombineChunkBase_ = combineChunkBase_.get();
    }
  }
  if (asyncDispatch == 1) {
    dispatchPeers_ = mscclpp::detail::gpuCallocShared<GpuNetIoDispatchPeerState>(numRanks_);
    deviceContext_.gpuNetIoDispatchPeers_ = dispatchPeers_.get();
  } else if (gpuNetIoRankMajor && !sendDistribute_) {
    dispatchDone_ = mscclpp::detail::gpuCallocShared<int>(1);
    deviceContext_.gpuNetIoDispatchDone_ = dispatchDone_.get();
  }
  if (deviceContext_.gpuNetIoDeferFlush_) {
    dispatchTickets_ =
        mscclpp::detail::gpuCallocShared<uint64_t>(static_cast<size_t>(numRanks_) * GpuNetIoMaxQpsPerPeer);
    deviceContext_.gpuNetIoDispatchTickets_ = dispatchTickets_.get();
  }
  if (pipelineChunkRows > 0 && !sendDistribute_) {
    const size_t counters = static_cast<size_t>(numRanks_) * configCellDiv(maxTokensPerRank_, pipelineChunkRows);
    chunkReady_ = mscclpp::detail::gpuCallocShared<int>(counters);
    deviceContext_.gpuNetIoChunkReady_ = chunkReady_.get();
  }
  if (const char* tma = std::getenv("MSCCLPP_EP_GPUNETIO_TMA_STAGE");
      tma != nullptr && (std::string(tma) == "1" || std::string(tma) == "2")) {
    deviceContext_.gpuNetIoTmaStage_ = (useGpuNetIo_ && deviceContext_.gpuNetIoBatchedDispatch_) ? std::atoi(tma) : 0;
  }
  if (useGpuNetIo_ && rank_ == 0) {
    std::fprintf(
        stderr,
        "[EPGPUNETIO] ranks=%d ipcDomain=%d nodes=%d sendDistribute=%d aggregateCombine=%d batched=%d stripe=all_qps "
        "dispatchChunkRows=%d async=%d "
        "parallelStage=%d rowFence=%s deferFlush=%d combineChunkRows=%d\n",
        numRanks_, numRanksPerIpcDomain_, numNodes, sendDistribute_ ? 1 : 0, aggregateCombine_ ? 1 : 0, batchedDispatch,
        pipelineChunkRows, asyncDispatch ? 1 : 0, parallelStage, gpuNetIoRowFenceName(rowFence),
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
    deviceContext_.gpuNetIoAggregateSend_ = storage.gpuNetIoAggregateSend_;
    deviceContext_.gpuNetIoAggregateRecv_ = storage.gpuNetIoAggregateRecv_;
    deviceContext_.gpuNetIoAggregateControl_ = storage.gpuNetIoAggregateControl_;
    if (sendDistribute_) {
      EpGpuNetIoDeviceContext network{};
      mscclpp::gpuMemcpy<EpGpuNetIoDeviceContext>(&network, service->deviceContext(), 1, cudaMemcpyDeviceToHost);
      if (network.numQpsPerPeer != 1 && network.numQpsPerPeer != 4) {
        EP_THROW("MSCCLPP_EP_GPUNETIO_SEND_DISTRIBUTE requires 1 or 4 QPs per peer");
      }
      deviceContext_.gpuNetIoDistributeSend_ = storage.gpuNetIoDistributeSend_;
      deviceContext_.gpuNetIoDistributeRecv_ = storage.gpuNetIoDistributeRecv_;
      deviceContext_.gpuNetIoDistributeControl_ = storage.gpuNetIoDistributeControl_;
      std::fprintf(stderr,
                   "[EPGPUNETIO] rank=%d sendDistribute=1 gateway=%d chunkRows=%d qps=%d wireRowBytes=%zu "
                   "sendBytes=%zu recvBytes=%zu ack=round combine=%s\n",
                   rank_, deviceContext_.gpuNetIoGateway_, pipelineChunkRows, network.numQpsPerPeer,
                   GpuNetIoDistributeRow::stride(hidden_, numTopk_),
                   maxTokensPerRank_ * GpuNetIoDistributeRow::stride(hidden_, numTopk_),
                   maxTokensPerRank_ * GpuNetIoDistributeRow::stride(hidden_, numTopk_),
                   aggregateCombine_ ? "aggregate_fp32" : "direct");
      std::fflush(stderr);
    }
  }
#endif  // defined(MSCCLPP_USE_GPUNETIO)

  if (std::getenv("MSCCLPP_EP_DEBUG_TOPO") != nullptr) {
    // nvlinkPeerMap[r] = '1' when peer r is IPC-mapped and '0' when it is served over GPUNetIO.
    std::string nvlinkPeerMap(static_cast<size_t>(numRanks_), '0');
    for (int r = 0; r < numRanks_; ++r) {
      if (r == rank_ || peerMappedBufferBases[r] != nullptr) nvlinkPeerMap[r] = '1';
    }
    std::fprintf(stderr,
                 "[EPTOPO] rank=%d numRanks=%d numRanksPerIpcDomain=%d numRanksPerNode=%d gpuNetIo=%d "
                 "nvlinkPeerMap=%s\n",
                 rank_, numRanks_, numRanksPerIpcDomain_, numRanksPerNode_, deviceContext_.gpuNetIo_ != nullptr ? 1 : 0,
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
  if (sendDistribute_) EP_HOST_ASSERT(workload.maxTokensPerRank_ <= maxTokensPerRank_);
  gpuNetIoRankMajorDispatch(output, outputTopkIdx, outputTopkWeights, outputCount, input, topkIdx, topkWeights,
                            workload, dispatchRecvBuffer, deviceContext_, numBlocks, stream);
}

void LatencyRuntime::combine(void* output, const void* input, const int64_t* topkIdx, const Workload& workload,
                             void* combineRecvBuffer, void* dispatchRecvBuffer, int numBlocks, cudaStream_t stream) {
  EP_HOST_ASSERT(deviceContext_.devicePtr_ != nullptr);
  if (deviceContext_.gpuNetIo_ != nullptr) EP_HOST_ASSERT(numBlocks >= numRanks_);
  gpuNetIoRankMajorCombine(output, input, topkIdx, workload, combineRecvBuffer, dispatchRecvBuffer, deviceContext_,
                           numBlocks, stream);
}

}  // namespace interrack
}  // namespace ep
}  // namespace mscclpp
