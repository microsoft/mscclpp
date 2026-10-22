// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cuda.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <future>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/ext/ep/moe_runtime.hpp>
#include <string>

#include "exception.hpp"
#include "kernels.hpp"
#include "moe_runtime_context.hpp"

#if defined(MSCCLPP_USE_GPUNETIO)
#include <mscclpp/utils.hpp>

#include "host/gpu_net_io_service.hpp"
#endif  // defined(MSCCLPP_USE_GPUNETIO)

namespace mscclpp {
namespace ep {

namespace {
const char* gpuNetIoRowFenceName(int rowFence) {
  return rowFence == GpuNetIoRowFenceGpu ? "gpu" : rowFence == GpuNetIoRowFenceGpuPoster ? "gpu+poster" : "sys";
}
}  // namespace

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
      symmetricBufferBytes_(0),
      workspaceBytes_(workspaceSize(numRanks_, numExperts, maxTokensPerRank, numTopk)),
      communicator_(&communicator) {
  EP_HOST_ASSERT(communicator_ != nullptr);
  EP_HOST_ASSERT(maxTokensPerRank > 0);
  EP_HOST_ASSERT(numExperts > 0 && numExperts % numRanks_ == 0);
  EP_HOST_ASSERT(numTopk > 0 && numTopk <= 32);
  EP_HOST_ASSERT(outputLayout == DispatchLayout::EXPERT_MAJOR || outputLayout == DispatchLayout::RANK_MAJOR ||
                 outputLayout == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED);
  if (outputLayout == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED) {
    EP_HOST_ASSERT(combineMode == CombineMode::RANK_LOCAL_REDUCE);
    EP_HOST_ASSERT(numTopk <= 9);
    EP_HOST_ASSERT(static_cast<int64_t>(maxTokensPerRank) * numTopk <= INT32_MAX);
  }

  CUDA_CHECK(cudaGetDevice(&deviceId_));
  EP_HOST_ASSERT(numRanks_ % numNvlRanks == 0);
  numNvlRanks_ = numNvlRanks;
  // Optional override to shrink the IPC/NVLink domain (e.g. to the physical node)
  // so cross-physical-node traffic is served by GPUNetIO instead of MNNVL IPC.
  if (const char* e = std::getenv("MSCCLPP_EP_IPC_DOMAIN_SIZE")) {
    const int forced = std::atoi(e);
    if (forced >= 1 && forced <= numRanks_ && numRanks_ % forced == 0) numRanksPerIpcDomain_ = forced;
  }
  EP_HOST_ASSERT(numRanks_ % numRanksPerIpcDomain_ == 0);
  available_ = numRanksPerIpcDomain_ >= numRanks_;
#if defined(MSCCLPP_USE_GPUNETIO)
  // Cross-domain GPUNetIO makes the runtime available even when the IPC domain
  // does not span all ranks, provided the backend is explicitly enabled.
  if (numRanksPerIpcDomain_ < numRanks_) {
    const char* enableGpuNetIo = std::getenv("MSCCLPP_EP_ENABLE_GPUNETIO");
    if (enableGpuNetIo != nullptr && std::atoi(enableGpuNetIo) != 0) {
      available_ =
          outputLayout_ == DispatchLayout::RANK_MAJOR || outputLayout_ == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED;
      useGpuNetIo_ = available_;
    }
  }
#endif  // defined(MSCCLPP_USE_GPUNETIO)
  symmetricBufferBytes_ = static_cast<int64_t>(latencyStorageSize(maxTokensPerRank_, hidden_, numRanks_, numExperts_,
                                                                  numTopk_, outputLayout_, combineMode_, useGpuNetIo_));
  EP_HOST_ASSERT(symmetricBufferBytes_ % BufferAlignmentBytes == 0);
}

LatencyContext::~LatencyContext() noexcept(false) {
  CUDA_CHECK(cudaDeviceSynchronize());
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  dumpTiming();
#endif
#if defined(MSCCLPP_USE_GPUNETIO)
  gpuNetIoService_.reset();
#endif
  if (deviceContext_.devicePtr_ != nullptr) CUDA_CHECK(cudaFree(deviceContext_.devicePtr_));
  if (peerMappedBufferBasesGpu_ != nullptr) CUDA_CHECK(cudaFree(peerMappedBufferBasesGpu_));
  if (workspace_ != nullptr) CUDA_CHECK(cudaFree(workspace_));
  if (symmetricBuffer_ != nullptr) {
    mscclpp::detail::gpuFreePhysical(symmetricBuffer_);
  }
}

void LatencyContext::initialize() {
  EP_HOST_ASSERT(!initializeStarted_ &&
                 "LatencyContext::initialize() is not retryable after a partial failure; report the first error");
  EP_HOST_ASSERT(communicator_ != nullptr);
  struct StorageConfig {
    uint64_t bytes;
    int ipcDomainSize;
    int useGpuNetIo;
    int batchedDispatch;
    int batchAllQps;
    int pipelineChunkRows;
    int asyncDispatch;
    int dedicatedPoster;
  };
  int batchedDispatch = 0;
  if (const char* requested = std::getenv("MSCCLPP_EP_GPUNETIO_BATCHED_DISPATCH")) {
    const std::string value(requested);
    batchedDispatch = value == "0" ? 0 : value == "1" ? 1 : -1;
  }
  int batchAllQps = 0;
  if (const char* requested = std::getenv("MSCCLPP_EP_GPUNETIO_BATCH_ALL_QPS")) {
    const std::string value(requested);
    batchAllQps = value == "0" ? 0 : value == "1" ? 1 : -1;
  }
  std::vector<StorageConfig> storageConfigs(numRanks_);
  int pipelineChunkRows = 0;
  if (const char* requested = std::getenv("MSCCLPP_EP_GPUNETIO_DISPATCH_CHUNK_ROWS")) {
    const std::string value(requested);
    pipelineChunkRows =
        !value.empty() && value.size() <= 3 &&
                std::all_of(value.begin(), value.end(), [](char digit) { return digit >= '0' && digit <= '9'; })
            ? std::stoi(value)
            : -1;
  }
  // Default pipeline chunk rows by node count when the env var is unset: 4 rows for 2-7 nodes, 16 rows for >=8 nodes.
  const int numNodes = numNvlRanks_ > 0 && numRanks_ % numNvlRanks_ == 0 ? numRanks_ / numNvlRanks_ : 0;
  if (std::getenv("MSCCLPP_EP_GPUNETIO_DISPATCH_CHUNK_ROWS") == nullptr && batchedDispatch == 1 && useGpuNetIo_ &&
      outputLayout_ == DispatchLayout::RANK_MAJOR && numNodes >= 2) {
    pipelineChunkRows = numNodes >= 8 ? 16 : 4;
    if (rank_ == 0)
      std::fprintf(stderr, "[EPGPUNETIO_DISPATCH] default pipelineChunkRows=%d numNodes=%d\n", pipelineChunkRows,
                   numNodes);
  }
  int asyncDispatch = 0;
  if (const char* requested = std::getenv("MSCCLPP_EP_GPUNETIO_ASYNC_DISPATCH")) {
    const std::string value(requested);
    asyncDispatch = value == "0" ? 0 : value == "1" ? 1 : -1;
  }
  int dedicatedPoster = 0;
  if (const char* requested = std::getenv("MSCCLPP_EP_GPUNETIO_DEDICATED_POSTER")) {
    const std::string value(requested);
    dedicatedPoster = value == "0" ? 0 : value == "1" ? 1 : -1;
  }
  // Batched-dispatch defaults (measured faster): parallel staging and GPU-scope row fence.
  int parallelStage = batchedDispatch == 1 ? 1 : 0;
  if (const char* requested = std::getenv("MSCCLPP_EP_GPUNETIO_PARALLEL_STAGE")) {
    const std::string value(requested);
    const bool numeric = !value.empty() && value.size() <= 2 && std::all_of(value.begin(), value.end(), [](char digit) {
      return digit >= '0' && digit <= '9';
    });
    parallelStage = numeric ? std::stoi(value) : -1;
  }
  int rowFence = batchedDispatch == 1 ? GpuNetIoRowFenceGpu : GpuNetIoRowFenceSystem;
  if (const char* requested = std::getenv("MSCCLPP_EP_GPUNETIO_ROW_FENCE")) {
    const std::string value(requested);
    rowFence = value == "sys"          ? GpuNetIoRowFenceSystem
               : value == "gpu"        ? GpuNetIoRowFenceGpu
               : value == "gpu+poster" ? GpuNetIoRowFenceGpuPoster
                                       : -1;
  }
  storageConfigs[rank_] = {static_cast<uint64_t>(symmetricBufferBytes_),
                           numRanksPerIpcDomain_,
                           useGpuNetIo_ ? 1 : 0,
                           batchedDispatch,
                           batchAllQps,
                           pipelineChunkRows,
                           asyncDispatch,
                           dedicatedPoster};
  communicator_->bootstrap()->allGather(storageConfigs.data(), sizeof(StorageConfig));
  for (const auto& config : storageConfigs) {
    EP_HOST_ASSERT(
        config.bytes == static_cast<uint64_t>(symmetricBufferBytes_) && config.ipcDomainSize == numRanksPerIpcDomain_ &&
        config.useGpuNetIo == (useGpuNetIo_ ? 1 : 0) && config.batchedDispatch == batchedDispatch &&
        config.batchedDispatch >= 0 && config.batchAllQps == batchAllQps && config.batchAllQps >= 0 &&
        config.pipelineChunkRows == pipelineChunkRows && config.pipelineChunkRows >= 0 &&
        config.pipelineChunkRows <= 128 && config.asyncDispatch == asyncDispatch && config.asyncDispatch >= 0 &&
        config.dedicatedPoster == dedicatedPoster && config.dedicatedPoster >= 0);
  }
  EP_HOST_ASSERT(asyncDispatch == 0 || (batchedDispatch == 1 && pipelineChunkRows > 0));
  EP_HOST_ASSERT(dedicatedPoster == 0 ||
                 (asyncDispatch == 0 && batchedDispatch == 1 && pipelineChunkRows > 0 && batchAllQps == 1));
  EP_HOST_ASSERT(pipelineChunkRows == 0 || batchedDispatch == 1);
  EP_HOST_ASSERT(batchAllQps == 0 || batchedDispatch == 1);
  EP_HOST_ASSERT(parallelStage >= 0 && parallelStage <= 16);
  EP_HOST_ASSERT(parallelStage == 0 || batchedDispatch == 1);
  EP_HOST_ASSERT(rowFence >= 0);
  EP_HOST_ASSERT(rowFence == GpuNetIoRowFenceSystem || batchedDispatch == 1);
  EP_HOST_ASSERT(batchedDispatch == 0 || (useGpuNetIo_ && outputLayout_ == DispatchLayout::RANK_MAJOR &&
                                          static_cast<size_t>(numRanks_) * maxTokensPerRank_ <= GpuNetIoStagingSlots));
  EP_HOST_ASSERT(available_);
  // Resources below are not released on failure, so a second attempt cannot succeed.
  initializeStarted_ = true;
  AvoidCudaGraphCaptureGuard captureGuard;
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  initializeTiming();
#endif
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
  deviceContext_.gpuNetIoBatchedDispatch_ = batchedDispatch == 1;
  deviceContext_.gpuNetIoBatchAllQps_ = batchAllQps == 1;
  deviceContext_.gpuNetIoPipelineChunkRows_ = pipelineChunkRows;
  deviceContext_.gpuNetIoParallelStage_ = parallelStage;
  deviceContext_.gpuNetIoRowFence_ = rowFence;
  deviceContext_.timing_ = timingDevice_.get();
  deviceContext_.gpuNetIoAsyncDispatch_ = asyncDispatch == 1;
  deviceContext_.gpuNetIoDedicatedPoster_ =
      useGpuNetIo_ && outputLayout_ == DispatchLayout::RANK_MAJOR && dedicatedPoster == 1;
  {
    // Deferred send drain defaults on for batched dispatch; MSCCLPP_EP_GPUNETIO_DEFER_FLUSH=0/1 overrides.
    const char* deferFlush = std::getenv("MSCCLPP_EP_GPUNETIO_DEFER_FLUSH");
    const bool deferRequested = deferFlush != nullptr ? std::string(deferFlush) == "1" : batchedDispatch == 1;
    deviceContext_.gpuNetIoDeferFlush_ =
        useGpuNetIo_ && outputLayout_ == DispatchLayout::RANK_MAJOR && asyncDispatch == 0 && deferRequested;
    if (deviceContext_.gpuNetIoDeferFlush_) std::fprintf(stderr, "[EPGPUNETIO_DISPATCH] rank=%d deferFlush=1\n", rank_);
  }
  if (deviceContext_.gpuNetIoDedicatedPoster_)
    std::fprintf(stderr, "[EPGPUNETIO_DISPATCH] rank=%d dedicatedPoster=1 maxRowsPerPost=%d\n", rank_,
                 pipelineChunkRows);
  {
    // Default combine chunk rows when the env var is unset: 32 rows for multi-node GPUNetIO rank-major runs.
    int combineChunkRows = numNodes >= 2 ? std::min(32, maxTokensPerRank_) : 0;
    if (const char* requested = std::getenv("MSCCLPP_EP_GPUNETIO_COMBINE_CHUNK_ROWS")) {
      const std::string value(requested);
      const bool numeric = !value.empty() && value.size() <= 3 &&
                           std::all_of(value.begin(), value.end(), [](char digit) { return digit >= '0' && digit <= '9'; });
      combineChunkRows = numeric ? std::stoi(value) : -1;
    }
    EP_HOST_ASSERT(combineChunkRows >= 0 && combineChunkRows <= maxTokensPerRank_);
    if (useGpuNetIo_ && outputLayout_ == DispatchLayout::RANK_MAJOR && combineChunkRows > 0) {
      deviceContext_.gpuNetIoCombineChunkRows_ = combineChunkRows;
      combineChunkBase_ =
          mscclpp::detail::gpuCallocShared<uint64_t>(static_cast<size_t>(numRanks_) * GpuNetIoMaxQpsPerPeer);
      deviceContext_.gpuNetIoCombineChunkBase_ = combineChunkBase_.get();
      std::fprintf(stderr, "[EPGPUNETIO_COMBINE] rank=%d combineChunkRows=%d\n", rank_, combineChunkRows);
    }
  }
  if (const char* noBarrier = std::getenv("MSCCLPP_EP_GPUNETIO_COMBINE_NO_BARRIER")) {
    deviceContext_.gpuNetIoCombineNoBarrier_ = useGpuNetIo_ && std::string(noBarrier) == "1";
    if (deviceContext_.gpuNetIoCombineNoBarrier_)
      std::fprintf(stderr, "[EPGPUNETIO_COMBINE] rank=%d noBarrier=1\n", rank_);
  }
  if (asyncDispatch == 1) {
    dispatchPeers_ = mscclpp::detail::gpuCallocShared<GpuNetIoDispatchPeerState>(numRanks_);
    deviceContext_.gpuNetIoDispatchPeers_ = dispatchPeers_.get();
  } else if (useGpuNetIo_ && outputLayout_ == DispatchLayout::RANK_MAJOR) {
    dispatchDone_ = mscclpp::detail::gpuCallocShared<int>(1);
    deviceContext_.gpuNetIoDispatchDone_ = dispatchDone_.get();
  }
  if (deviceContext_.gpuNetIoDeferFlush_) {
    dispatchTickets_ =
        mscclpp::detail::gpuCallocShared<uint64_t>(static_cast<size_t>(numRanks_) * GpuNetIoMaxQpsPerPeer);
    deviceContext_.gpuNetIoDispatchTickets_ = dispatchTickets_.get();
  }
  if (pipelineChunkRows > 0) {
    const size_t counters =
        static_cast<size_t>(numRanks_) *
        (deviceContext_.gpuNetIoDedicatedPoster_ ? maxTokensPerRank_ : configCellDiv(maxTokensPerRank_, pipelineChunkRows));
    // Early tail post: producer-posted chunks only (not async, not dedicated posters).
    const char* earlyTail = std::getenv("MSCCLPP_EP_GPUNETIO_EARLY_TAIL");
    const bool tailRows = earlyTail && std::string(earlyTail) == "1" && useGpuNetIo_ &&
                          outputLayout_ == DispatchLayout::RANK_MAJOR && asyncDispatch == 0 &&
                          !deviceContext_.gpuNetIoDedicatedPoster_ && deviceContext_.gpuNetIoBatchedDispatch_;
    chunkReady_ = mscclpp::detail::gpuCallocShared<int>(counters + (tailRows ? numRanks_ : 0));
    deviceContext_.gpuNetIoTailRows_ = tailRows ? chunkReady_.get() : nullptr;
    deviceContext_.gpuNetIoChunkReady_ = chunkReady_.get() + (tailRows ? numRanks_ : 0);
    if (earlyTail) std::fprintf(stderr, "[EPGPUNETIO_DISPATCH] rank=%d earlyTail=%d\n", rank_, static_cast<int>(tailRows));
  }
  if (const char* tma = std::getenv("MSCCLPP_EP_GPUNETIO_TMA_STAGE"); tma && (std::string(tma) == "1" || std::string(tma) == "2" || std::string(tma) == "3" || std::string(tma) == "4" || std::string(tma) == "5" || std::string(tma) == "6" || std::string(tma) == "7")) {
    deviceContext_.gpuNetIoTmaStage_ = (useGpuNetIo_ && deviceContext_.gpuNetIoBatchedDispatch_) ? std::atoi(tma) : 0;
    std::fprintf(stderr, "[EPGPUNETIO_DISPATCH] rank=%d tmaStage=%d\n", rank_,
                 static_cast<int>(deviceContext_.gpuNetIoTmaStage_));
  }
  if (const char* zc = std::getenv("MSCCLPP_EP_GPUNETIO_ZERO_COPY"); zc && zc[0] >= '1' && zc[0] <= '4') {
    if (useGpuNetIo_ && deviceContext_.gpuNetIoBatchedDispatch_ && pipelineChunkRows > 0 &&
        outputLayout_ == DispatchLayout::RANK_MAJOR && asyncDispatch == 0 && !deviceContext_.gpuNetIoDedicatedPoster_) {
      zcRowSrc_ = mscclpp::detail::gpuCallocShared<int>(static_cast<size_t>(numRanks_) * maxTokensPerRank_);
      deviceContext_.gpuNetIoZcRowSrc_ = zcRowSrc_.get();
      deviceContext_.gpuNetIoZeroCopy_ = zc[0] - '0';
    }
    std::fprintf(stderr, "[EPGPUNETIO_DISPATCH] rank=%d zeroCopy=%d\n", rank_, deviceContext_.gpuNetIoZeroCopy_);
  }
  if (deviceContext_.gpuNetIoBatchedDispatch_) {
    std::fprintf(
        stderr,
        "[EPGPUNETIO_DISPATCH] rank=%d batched=1 ranks=%d capacity=%d stripeMode=%s pipelineChunkRows=%d async=%d\n",
        rank_, numRanks_, maxTokensPerRank_, deviceContext_.gpuNetIoBatchAllQps_ ? "all_qps" : "hca", pipelineChunkRows,
        asyncDispatch);
    if (deviceContext_.gpuNetIoParallelStage_ > 0)
      std::fprintf(stderr, "[EPGPUNETIO_DISPATCH] rank=%d parallelStage=%d\n", rank_,
                   deviceContext_.gpuNetIoParallelStage_);
    if (deviceContext_.gpuNetIoRowFence_ != GpuNetIoRowFenceSystem)
      std::fprintf(stderr, "[EPGPUNETIO_DISPATCH] rank=%d rowFence=%s\n", rank_,
                   gpuNetIoRowFenceName(deviceContext_.gpuNetIoRowFence_));
    std::fflush(stderr);
  }

#if defined(MSCCLPP_USE_GPUNETIO)
  // Bring up the GPU-initiated networking service for peers outside this rank's
  // NVLink/IPC domain (opt-in via MSCCLPP_EP_ENABLE_GPUNETIO). Publishes the
  // device context + symmetric staging/flags regions onto deviceContext_.
  if (useGpuNetIo_) {
    std::string hca;
    if (const char* devices = std::getenv("MSCCLPP_EP_GPUNETIO_HCAS")) {
      hca = devices;
    } else if (const char* device = std::getenv("MSCCLPP_EP_GPUNETIO_HCA")) {
      hca = device;
    }
    auto svc = std::make_shared<mscclpp::GpuNetIoService>(communicator_->bootstrap(), hca, deviceId_);
    svc->setup(symmetricBuffer_, static_cast<size_t>(symmetricBufferBytes_));
    gpuNetIoService_ = svc;
    deviceContext_.gpuNetIo_ = svc->deviceContext();
    LatencyStorageLayout layout(symmetricBuffer_, maxTokensPerRank_, hidden_, numRanks_, numExperts_, numTopk_,
                                outputLayout_, combineMode_, useGpuNetIo_);
    deviceContext_.gpuNetIoStagingBuffer_ = layout.gpuNetIoStagingBuffer_;
    deviceContext_.gpuNetIoFlagsBuffer_ = layout.gpuNetIoFlagsBuffer_;
    deviceContext_.gpuNetIoCombineFlagsBuffer_ = layout.gpuNetIoCombineFlagsBuffer_;
    deviceContext_.gpuNetIoCombineLandingBuffer_ = layout.gpuNetIoCombineLandingBuffer_;
    deviceContext_.gpuNetIoSlotStride_ = layout.gpuNetIoSlotStride_;
  }
#endif  // defined(MSCCLPP_USE_GPUNETIO)

  if (outputLayout_ == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED && ipcDomainSize >= numRanks_) {
    const char* requested = std::getenv("MSCCLPP_EP_EXPANDED_IPC_FASTPATH");
    const bool allMapped = std::all_of(peerMappedBufferBases_.begin(), peerMappedBufferBases_.end(),
                                       [](void* base) { return base != nullptr; });
    std::vector<int> enabled(numRanks_, 0);
    enabled[rank_] =
        (requested == nullptr || std::atoi(requested) != 0) && allMapped && deviceContext_.gpuNetIo_ == nullptr;
    communicator_->bootstrap()->allGather(enabled.data(), sizeof(int));
    deviceContext_.expandedIpcFastPath_ =
        std::all_of(enabled.begin(), enabled.end(), [](int value) { return value != 0; });
  }
  if (outputLayout_ == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED && ipcDomainSize < numRanks_) {
    const char* requested = std::getenv("MSCCLPP_EP_EXPANDED_GPUNETIO_FASTPATH");
    std::vector<int> enabled(numRanks_, 0);
    enabled[rank_] = (requested == nullptr || std::atoi(requested) != 0) && deviceContext_.gpuNetIo_ != nullptr;
    communicator_->bootstrap()->allGather(enabled.data(), sizeof(int));
    deviceContext_.expandedGpuNetIoFastPath_ =
        std::all_of(enabled.begin(), enabled.end(), [](int value) { return value != 0; });
  }
  // Host-side topology dump (opt-in via MSCCLPP_EP_DEBUG_TOPO): nvlinkPeerMap[r]
  // ='1' => peer r is NVLink/IPC-mapped; '0' => cross-domain (served via GPUNetIO).
  if (std::getenv("MSCCLPP_EP_DEBUG_TOPO") != nullptr) {
    std::string nvmap(static_cast<size_t>(numRanks_), '0');
    for (int r = 0; r < numRanks_; ++r) {
      if (r == rank_ || peerMappedBufferBases_[r] != nullptr) nvmap[r] = '1';
    }
#if defined(MSCCLPP_USE_GPUNETIO)
    const int ginOn = (gpuNetIoService_ != nullptr) ? 1 : 0;
#else
    const int ginOn = 0;
#endif
    std::fprintf(stderr,
                 "[EPTOPO] rank=%d numRanks=%d nRanksPerIpcDomain=%d numNvlRanks=%d crossDomain=%d ginOn=%d "
                 "available=%d nvlinkPeerMap=%s\n",
                 rank_, numRanks_, numRanksPerIpcDomain_, numNvlRanks_, (numRanksPerIpcDomain_ < numRanks_) ? 1 : 0,
                 ginOn, available_ ? 1 : 0, nvmap.c_str());
    std::fflush(stderr);
    if (outputLayout_ == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED) {
      std::fprintf(stderr, "[EPEXPANDED] rank=%d ipcFastPath=%d hidden=%d topk=%d\n", rank_,
                   deviceContext_.expandedIpcFastPath_ ? 1 : 0, hidden_, numTopk_);
      std::fprintf(stderr, "[EPEXPANDED_GPUNETIO] rank=%d gpuNetIoFastPath=%d\n", rank_,
                   deviceContext_.expandedGpuNetIoFastPath_ ? 1 : 0);
      std::fflush(stderr);
    }
  }
  deviceContext_.devicePtr_ = static_cast<DeviceContext*>(mscclpp::detail::gpuCalloc(sizeof(DeviceContext)));
  mscclpp::gpuMemcpy<DeviceContext>(deviceContext_.devicePtr_, &deviceContext_, 1, cudaMemcpyHostToDevice);
}

void LatencyContext::initializeTiming() {
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  const char* prefix = std::getenv("MSCCLPP_EP_TIMING_FILE");
  if (prefix == nullptr || *prefix == '\0') return;
  EP_HOST_ASSERT(outputLayout_ == DispatchLayout::RANK_MAJOR);
  unsigned long capacity = 2048;
  if (const char* requested = std::getenv("MSCCLPP_EP_TIMING_RECORDS_PER_BLOCK")) {
    size_t consumed = 0;
    capacity = std::stoul(requested, &consumed);
    EP_HOST_ASSERT(consumed == std::string(requested).size());
  }
  EP_HOST_ASSERT(capacity >= 1 && capacity <= 4096);
  static std::atomic<unsigned long> nextContext{0};
  timingPath_ = std::string(prefix) + ".rank" + std::to_string(rank_) + ".pid" + std::to_string(getpid()) + ".ctx" +
                std::to_string(nextContext.fetch_add(1)) + ".csv";
  std::ofstream probe(timingPath_, std::ios::out);
  EP_HOST_ASSERT(probe.good());
  probe.close();
  const size_t recordCount = static_cast<size_t>(TimingOperations) * TimingMaxBlocks * capacity;
  timingRecords_ = mscclpp::detail::gpuCallocShared<TimingRecord>(recordCount);
  timingDevice_ = mscclpp::detail::gpuCallocShared<BufferedTiming>(1);
  auto initial = std::make_unique<BufferedTiming>();
  if (const char* requested = std::getenv("MSCCLPP_EP_TIMING_SUBMIT_COUNTS")) {
    const std::string value(requested);
    EP_HOST_ASSERT(value == "0" || value == "1");
    initial->submitCounts = value == "1";
  }
  if (const char* requested = std::getenv("MSCCLPP_EP_TIMING_SUBMIT_EVERY")) {
    const std::string value(requested);
    EP_HOST_ASSERT(!value.empty() && value.size() <= 5 &&
                   std::all_of(value.begin(), value.end(), [](char digit) { return digit >= '0' && digit <= '9'; }));
    const unsigned long every = std::stoul(value);
    EP_HOST_ASSERT(every <= 65536);
    initial->submitEvery = static_cast<unsigned int>(every);
  }
  initial->records = timingRecords_.get();
  initial->capacityPerBlock = static_cast<unsigned int>(capacity);
  mscclpp::gpuMemcpy<BufferedTiming>(timingDevice_.get(), initial.get(), 1, cudaMemcpyHostToDevice);
  deviceContext_.timing_ = timingDevice_.get();
  std::fprintf(stderr,
               "[EPTIMING] rank=%d file=%s capacityPerBlock=%lu clock=globaltimer_ns submitEvery=%u submitCounts=%d\n",
               rank_, timingPath_.c_str(), capacity, initial->submitEvery, initial->submitCounts);
  std::fflush(stderr);
#endif
}

void LatencyContext::dumpTiming() {
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  if (!timingDevice_) return;
  auto captured = std::make_unique<BufferedTiming>();
  mscclpp::gpuMemcpy<BufferedTiming>(captured.get(), timingDevice_.get(), 1, cudaMemcpyDeviceToHost);
  std::ofstream output(timingPath_, std::ios::out);
  EP_HOST_ASSERT(output.good());
  output
      << "# clock=globaltimer_ns; local GPU only; invocation is per operation/CTA/thread; epoch is captured host "
         "epoch\n"
      << "# retained=first capacityPerBlock records per operation/CTA; serialized communicator use required\n"
      << "# instrumented durations include tracing overhead; not uninstrumented performance\n"
      << "# kind=0 DispatchBlock: "
         "v0=send_ns,v1=flush_ns,v2=barrier_ns,v3=recv_ns,v4=tokens,v5=topk,v6=blocks,v7=hidden\n"
      << "# kind=1 DispatchSend: "
         "v0=stage_ns,v1=fence_ns,v2=post_wait_or_chunk_publish_ns,v3=sends,v4=tma_and_route_ns,v5=reuse_flush_ns,v6="
         "tokens,v7="
         "warp_group\n"
      << "# kind=2 DispatchNotify: v0=count_ns,v1=count_write_ns; remaining span=payload_publish\n"
      << "# kind=3 DispatchQueue: v0=action(0=marker,1=drain,2=arrival_wait),v1=arrival_target\n"
      << "# kind=4 CombineBlock: v0=entry_ns,v1=push_ns,v2=recv_ns,v3=drain_ns,v4=tokens,v5=topk,v6=blocks,v7=hidden; "
         "remaining span=tail_sync\n"
      << "# kind=5 CombineSend and kind=6 CombineDrain: v0=stripe,v1=stripe_rows,v2=owner_rows\n"
      << "# kind=7 CombineRecv(vector fallback): v0=wait_ns,v1=barrier_ns; remaining span=reduce\n"
      << "# kind=8 CombineReady: v0=arrival_baseline; span=route_scan_and_peer_wait_and_publication\n"
      << "# kind=9 DispatchPost: v0=reserve_ns,v1=prepare_ns,v2=ordered_ready_ns,v3=submit_ns,v4=token,v5=bytes\n"
      << "# kind=10 DispatchBatch: "
         "v0=reserve_ns,v1=prepare_ns,v2=ordered_ready_ns,v3=submit_ns,v4=row_begin,v5=rows,v6=bytes,v7=destination_"
         "rows\n"
      << "# kind=11 DispatchChunk: "
         "v0=reserve_ns,v1=prepare_ns,v2=ordered_ready_ns,v3=submit_ns,v4=row_begin,v5=rows,v6=bytes,v7=tail(0="
         "producer_full,1=post_join_tail,2=producer_tail)\n"
      << "# kind=12 DispatchSubmit: v0=handler_raw(2=GPU_SM_DB,10=GPU_SM_NO_DBR,other=AUTO_proxy_fallback),"
         "v1=lock_wait_ns,v2=producer_index_or_proxy_fence_ns,v3=first_doorbell_or_proxy_publish_ns,"
         "v4=dbrec_ns,v5=second_doorbell_ns,v6=lock_held_ns,v7=publication_issued\n"
      << "# Submit details: zero can mean stage not executed or below timer resolution; publication_issued=0 means a "
         "covered/skipped DB call, "
         "not a missing payload; proxy publication is a request, not proof the NIC doorbell rang\n"
      << "# Match kind=12 to kind=9/10/11 by file/rank/operation/block/thread/invocation/epoch/peer/queue/end_ns; "
         "lock_held_ns overlaps producer-index/doorbell/DBREC stages and excludes lock/unlock; remaining span "
         "includes selector, unlock, branch and timestamp overhead\n"
      << "# Coalescing: covered calls may finish without acquiring the SQ lock; lock_wait_ns then includes "
         "waiting for completed publication and lock_held_ns is zero\n"
      << "# Submit tracing adds timestamps inside the SQ lock and may amplify contention; submitEvery="
      << captured->submitEvery << " selects invocation ordinals 1,1+N,... per operation/CTA/thread\n"
      << "# kind=13 DispatchSubmitCounts: v0=handler_raw,v1=publication_issued,v2=lock_acquired,"
         "v3=completed_watermark_bypass,v4=requested_wqes,v5=published_frontier_advance,"
         "v6=extra_new_wqes_beyond_caller,v7=coalesced_path\n"
      << "# submitCounts=" << captured->submitCounts
      << "; scope=dispatch_putBatched3_only; one record per call, "
         "no sampling or submit-stage timestamps; other records suppressed in counts-only mode\n"
      << "# Counts are not whole-QP traffic: frontier advances can cover other ready WQEs; normal marker/count/"
         "combine submit calls are not recorded; proxy publications are requests, not observed NIC doorbells\n"
      << "# kind=14 DispatchRecv: thread-0 receive-handler span; peer=source rank,v0=path(0=self,1=NVLink,2=network)\n"
      << "# kind=15 DispatchTopology: v0=ranks,v1=first_producer_CTA,v2=producer_CTAs,v3=warp_groups_per_producer,"
         "v4=control_CTA,v5=QPs_per_peer,v6=remote_peer_mask,v7=async_dispatch\n"
      << "# kind=16 DispatchStage (stage-trace builds): one remote destination row; peer=destination,"
         "queue=order_in_token,v0=copy_ns,v1=gpu_fence_ns,v2=sys_fence_ns,v3=publish_ns,"
         "v4=posted_chunk(0/1,2=async),v5=token,v6=warps_per_row(0=leader_serial),v7=destination_slot\n"
      << "# parallel staging: kind=1 v0-v3 sum helper-warp intervals, which overlap\n"
      << "# Timeline endpoints are GPU-local observations, not exact kernel completion or proof of a hardware "
         "bottleneck; "
         "DispatchSend includes inline posting, DispatchBlock end is thread 0 only\n"
      << "# batched dispatch: DispatchSend counts packed destinations, not posts; DispatchBlock barrier_ns includes "
         "producer-local completion or control-only producer wait, plus control-CTA posting/local join; no grid join\n"
      << "# GPUNetIO dispatch: receiver CTAs proceed after their own producer work; remote arrivals still require "
         "all-QP completion markers; flush_ns remains control-CTA only\n"
      << "# async dispatch: dedicated receive CTAs; per-peer markers follow all posted rows; full and tail chunks "
         "are producer-posted; no all-producer join; control flush_ns includes local marker readiness waits\n"
      << "# rank=" << rank_ << " ranks=" << numRanks_ << " hidden=" << hidden_ << " topk=" << numTopk_
      << " capacity=" << maxTokensPerRank_ << " network=" << useGpuNetIo_
      << " parallelStage=" << deviceContext_.gpuNetIoParallelStage_
      << " rowFence=" << gpuNetIoRowFenceName(deviceContext_.gpuNetIoRowFence_) << '\n'
      << "rank,operation,block,invocation,epoch,thread,kind,peer,queue,begin_ns,end_ns,v0,v1,v2,v3,v4,v5,v6,v7\n";
  std::vector<TimingRecord> records(captured->capacityPerBlock);
  uint64_t totalDropped = 0;
  for (int operation = 0; operation < TimingOperations; ++operation) {
    for (int block = 0; block < TimingMaxBlocks; ++block) {
      const auto& state = captured->blocks[operation * TimingMaxBlocks + block];
      if (state.attempted == 0) continue;
      const auto count = std::min<unsigned long long>(state.attempted, captured->capacityPerBlock);
      totalDropped += state.attempted - count;
      output << "# operation=" << operation << " block=" << block << " attempted=" << state.attempted
             << " dropped=" << state.attempted - count << '\n';
      mscclpp::gpuMemcpy<TimingRecord>(
          records.data(), timingRecords_.get() + timingRecordIndex(operation, block, 0, captured->capacityPerBlock),
          count, cudaMemcpyDeviceToHost);
      for (size_t index = 0; index < count; ++index) {
        const auto& record = records[index];
        output << rank_ << ',' << operation << ',' << block << ',' << record.invocation << ',' << record.epoch << ','
               << record.thread << ',' << static_cast<uint32_t>(record.kind) << ',' << record.peer << ','
               << record.queue << ',' << record.beginNs << ',' << record.endNs;
        for (auto value : record.values) output << ',' << value;
        output << '\n';
      }
    }
  }
  output.close();
  EP_HOST_ASSERT(output.good());
  std::fprintf(stderr, "[EPTIMING] rank=%d saved=%s dropped=%llu\n", rank_, timingPath_.c_str(),
               static_cast<unsigned long long>(totalDropped));
  std::fflush(stderr);
#endif
}

void* MoERuntime::outputTopkIdsBuffer() const {
  requireMode(MoEMode::LATENCY);
  const auto& context = *latencyContext_;
  EP_HOST_ASSERT(context.symmetricBuffer_ != nullptr);
  return LatencyStorageLayout(context.symmetricBuffer_, context.maxTokensPerRank_, context.hidden_, context.numRanks_,
                              context.numExperts_, context.numTopk_, context.outputLayout_, context.combineMode_,
                              context.useGpuNetIo_)
      .rankMajorTopkIdsBuffer_;
}

void* MoERuntime::outputTopkWeightsBuffer() const {
  requireMode(MoEMode::LATENCY);
  const auto& context = *latencyContext_;
  EP_HOST_ASSERT(context.symmetricBuffer_ != nullptr);
  return LatencyStorageLayout(context.symmetricBuffer_, context.maxTokensPerRank_, context.hidden_, context.numRanks_,
                              context.numExperts_, context.numTopk_, context.outputLayout_, context.combineMode_,
                              context.useGpuNetIo_)
      .rankMajorTopkWeightsBuffer_;
}

void* MoERuntime::combineInputBuffer() const {
  requireMode(MoEMode::LATENCY);
  const auto& context = *latencyContext_;
  EP_HOST_ASSERT(context.outputLayout_ == DispatchLayout::RANK_MAJOR ||
                 context.outputLayout_ == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED);
  EP_HOST_ASSERT(context.symmetricBuffer_ != nullptr);
  return LatencyStorageLayout(context.symmetricBuffer_, context.maxTokensPerRank_, context.hidden_, context.numRanks_,
                              context.numExperts_, context.numTopk_, context.outputLayout_, context.combineMode_,
                              context.useGpuNetIo_)
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
  const bool deduplicateExpandedRoutes = request.deduplicateExpandedRoutes;
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
  EP_HOST_ASSERT(!deduplicateExpandedRoutes || dispatchLayout == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED);

  if (dispatchLayout == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED) {
    EP_HOST_ASSERT(maxTokensPerRank == context.maxTokensPerRank_ && hidden == context.hidden_ &&
                   numTopk == context.numTopk_ && numExperts == context.numExperts_);
    EP_HOST_ASSERT(dispatchDataType == DispatchDataType::BF16);
  }
  LatencyStorageLayout allocationLayout(context.symmetricBuffer_, context.maxTokensPerRank_, hidden, context.numRanks_,
                                        numExperts, numTopk, context.outputLayout_, context.combineMode_,
                                        context.useGpuNetIo_);
  EP_HOST_ASSERT(allocationLayout.totalBytes_ <= static_cast<size_t>(context.symmetricBufferBytes_));
  void* dispatchRecvBuffer = allocationLayout.dispatchRecvBuffer_;
  if (dispatchLayout == DispatchLayout::RANK_MAJOR || dispatchLayout == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED) {
    EP_HOST_ASSERT(output == allocationLayout.dispatchOutputBuffer_);
    EP_HOST_ASSERT(outputTopkIdx == allocationLayout.rankMajorTopkIdsBuffer_);
    EP_HOST_ASSERT(outputTopkWeights == allocationLayout.rankMajorTopkWeightsBuffer_);
  }

#if defined(MSCCLPP_USE_GPUNETIO)
  if (context.deviceContext_.gpuNetIoZeroCopy_ != 0 && context.deviceContext_.gpuNetIoZeroCopy_ != 2) {
    EP_HOST_ASSERT(dispatchDataType == DispatchDataType::BF16);
    const size_t inputBytes = static_cast<size_t>(numTokens > 0 ? numTokens : 1) * hidden * 2;
    if (input != context.zcInput_ || inputBytes > context.zcInputBytes_) {
      cudaStreamCaptureStatus captureStatus = cudaStreamCaptureStatusNone;
      CUDA_CHECK(cudaStreamIsCapturing(stream, &captureStatus));
      if (captureStatus != cudaStreamCaptureStatusNone)
        throw std::runtime_error("zero-copy dispatch: input changed during CUDA graph capture; run eagerly first");
      const uint32_t* lkeys = context.gpuNetIoService_->registerLocalBuffer(input, inputBytes);
      context.zcInput_ = input;
      context.zcInputBytes_ = inputBytes;
      context.deviceContext_.gpuNetIoZcBase_ = reinterpret_cast<uintptr_t>(input);
      context.deviceContext_.gpuNetIoZcLkeys_ = lkeys;
      CUDA_CHECK(cudaDeviceSynchronize());
      mscclpp::gpuMemcpy<DeviceContext>(context.deviceContext_.devicePtr_, &context.deviceContext_, 1,
                                        cudaMemcpyHostToDevice);
    }
  }
#endif
  ++context.epoch_;
  if (std::getenv("MSCCLPP_EP_DEBUG_EPOCH") != nullptr) {
    std::fprintf(stderr, "[EPOCHDBG] rank=%d dispatch epoch=%u numTokens=%d\n", rank_, context.epoch_, numTokens);
    std::fflush(stderr);
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
                          .deduplicateExpandedRoutes_ = deduplicateExpandedRoutes};
  const size_t workspaceBytes = workspaceSize(context.numRanks_, numExperts, maxTokensPerRank, numTopk);
  EP_HOST_ASSERT(workspaceBytes <= context.workspaceBytes_);
  if (dispatchLayout == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED) {
    rankMajorTopkExpandedDispatch(output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout,
                                  outputCount, input, topkIdx, topkWeights, workload, dispatchRecvBuffer,
                                  context.deviceContext_, numBlocks, stream);
  } else if (dispatchLayout == DispatchLayout::RANK_MAJOR) {
    rankMajorDispatch(output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount,
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
  if (dispatchLayout == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED) {
    EP_HOST_ASSERT(maxTokensPerRank == context.maxTokensPerRank_ && hidden == context.hidden_ &&
                   numTopk == context.numTopk_ && numExperts == context.numExperts_);
    EP_HOST_ASSERT(dispatchDataType == DispatchDataType::BF16 && mode == CombineMode::RANK_LOCAL_REDUCE);
    EP_HOST_ASSERT(numTokens >= 0 && numTokens <= maxTokensPerRank && numBlocks > context.numRanks_);
  }
  if (context.deviceContext_.gpuNetIo_ != nullptr) {
    EP_HOST_ASSERT(
        dispatchLayout == DispatchLayout::RANK_MAJOR ||
        (dispatchLayout == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED && mode == CombineMode::RANK_LOCAL_REDUCE));
    EP_HOST_ASSERT(numBlocks >= context.numRanks_);
  }

  LatencyStorageLayout allocationLayout(context.symmetricBuffer_, context.maxTokensPerRank_, hidden, context.numRanks_,
                                        numExperts, numTopk, context.outputLayout_, context.combineMode_,
                                        context.useGpuNetIo_);
  EP_HOST_ASSERT(allocationLayout.totalBytes_ <= static_cast<size_t>(context.symmetricBufferBytes_));
  void* combineRecvBuffer = allocationLayout.combineRecvBuffer_;
  void* dispatchRecvBuffer = allocationLayout.dispatchRecvBuffer_;
  if (dispatchLayout == DispatchLayout::RANK_MAJOR || dispatchLayout == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED) {
    EP_HOST_ASSERT(input == allocationLayout.combineRecvBuffer_);
  }

  if (std::getenv("MSCCLPP_EP_DEBUG_EPOCH") != nullptr) {
    std::fprintf(stderr, "[EPOCHDBG] rank=%d combine  epoch=%u\n", rank_, context.epoch_);
    std::fflush(stderr);
  }
  const Workload workload{.epoch_ = context.epoch_,
                          .numTokens_ = numTokens,
                          .hidden_ = hidden,
                          .numTopk_ = numTopk,
                          .numExperts_ = numExperts,
                          .invalidTokenExpertId_ = numExperts,
                          .maxTokensPerRank_ = maxTokensPerRank,
                          .outputLayout_ = dispatchLayout,
                          .dispatchDataType_ = dispatchDataType,
                          .deduplicateExpandedRoutes_ = false};
  const size_t workspaceBytes = workspaceSize(context.numRanks_, numExperts, maxTokensPerRank, numTopk);
  EP_HOST_ASSERT(workspaceBytes <= context.workspaceBytes_);
  if (dispatchLayout == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED) {
    topk_expanded::combine(output, input, topkIdx, topkWeights, workload, context.deviceContext_, numBlocks, stream);
  } else if (dispatchLayout == DispatchLayout::RANK_MAJOR) {
    if (mode == CombineMode::DIRECT_SEND) {
      rankMajorDirectSendCombine(output, input, topkIdx, topkWeights, workload, combineRecvBuffer, dispatchRecvBuffer,
                                 context.deviceContext_, numBlocks, stream);
    } else {
      EP_HOST_ASSERT(mode == CombineMode::RANK_LOCAL_REDUCE);
      rankMajorGatherReduceCombine(output, input, topkIdx, topkWeights, srcInfo, layoutRange, workload,
                                   combineRecvBuffer, dispatchRecvBuffer, context.deviceContext_, numBlocks, stream);
    }
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
