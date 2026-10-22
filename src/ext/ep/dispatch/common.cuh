// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_DISPATCH_COMMON_CUH_
#define MSCCLPP_EP_DISPATCH_COMMON_CUH_
#include <mscclpp/bulk_device.hpp>
#include <mscclpp/memory_channel_device.hpp>

#include "common/device_helpers.cuh"
#include "common/latency.cuh"
#include "common/quantization.cuh"
#include "exception.hpp"
#include "kernels.hpp"

#if defined(MSCCLPP_USE_GPUNETIO)
#include <mscclpp/port_channel_gpunetio_device.hpp>
#endif  // defined(MSCCLPP_USE_GPUNETIO)

namespace mscclpp {
namespace ep {

#if MSCCLPP_BULK_AVAILABLE

struct RankMajorRoute {
  int dstRank;
  int destinationSlot;
  bool isLeader;
};

MSCCLPP_DEVICE_INLINE RankMajorRoute prepareRankMajorRoute(WorkspaceView& workspaceView,
                                                           const int64_t* __restrict__ topkIndices, int tokenIdx,
                                                           int nTopk, int nLocalExperts, int maxTokensPerRank,
                                                           int laneId) {
  const int routedExpertIdx = laneId < nTopk ? static_cast<int>(topkIndices[tokenIdx * nTopk + laneId]) : -1;
  const int dstRank = routedExpertIdx >= 0 ? routedExpertIdx / nLocalExperts : -1;
  const bool firstLaneForRank = isFirstLaneForRank(dstRank, laneId);
  int destinationSlot = -1;
  if (dstRank >= 0 && firstLaneForRank) {
    destinationSlot = atomicAdd(workspaceView.dispatchRankPayloadSlots_ + dstRank, 1);
    EP_DEVICE_ASSERT(destinationSlot < maxTokensPerRank);
  }

  const unsigned matchMask = __match_any_sync(0xffffffff, dstRank);
  const int firstLane = __ffs(matchMask) - 1;
  destinationSlot = __shfl_sync(0xffffffff, destinationSlot, firstLane);
  if (laneId < nTopk) {
    workspaceView.rankMajorSendIndices_[tokenIdx * nTopk + laneId] = dstRank >= 0 ? destinationSlot : -1;
  }
  return {dstRank, destinationSlot, firstLaneForRank};
}

MSCCLPP_DEVICE_INLINE void sendRankMajorMetadata(const TransportView& transport, int* outputTopkIdx,
                                                 float* outputTopkWeights, const int64_t* __restrict__ topkIndices,
                                                 const float* __restrict__ topkWeights, const RankMajorRoute& route,
                                                 int tokenIdx, int nTopk, int nLocalExperts, int maxTokensPerRank,
                                                 int invalidTokenExpertId) {
  const int laneId = get_lane_id();
  const int candidateExpert =
      laneId < nTopk ? static_cast<int>(topkIndices[tokenIdx * nTopk + laneId]) : invalidTokenExpertId;
  const float candidateWeight =
      laneId < nTopk ? (topkWeights == nullptr ? 1.0f : topkWeights[tokenIdx * nTopk + laneId]) : 0.0f;
  unsigned int leaderMask = __ballot_sync(0xffffffff, route.dstRank >= 0 && route.isLeader);
  while (leaderMask != 0) {
    const int leaderLane = __ffs(leaderMask) - 1;
    const int destinationRank = __shfl_sync(0xffffffff, route.dstRank, leaderLane);
    const int destinationSlot = __shfl_sync(0xffffffff, route.destinationSlot, leaderLane);
    if (laneId < nTopk) {
      auto* destinationTopkIdx = reinterpret_cast<int*>(transport.mappedBuffer(outputTopkIdx, destinationRank));
      auto* destinationTopkWeights =
          reinterpret_cast<float*>(transport.mappedBuffer(outputTopkWeights, destinationRank));
      const size_t outputIdx =
          (static_cast<size_t>(transport.rank_) * maxTokensPerRank + destinationSlot) * nTopk + laneId;
      const bool isLocal = candidateExpert >= 0 && candidateExpert / nLocalExperts == destinationRank;
      destinationTopkIdx[outputIdx] = isLocal ? candidateExpert : invalidTokenExpertId;
      destinationTopkWeights[outputIdx] = isLocal ? candidateWeight : 0.0f;
    }
    leaderMask &= leaderMask - 1;
  }
  __syncwarp();
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void issueRankMajorTokenStore(void* output, const TransportView& transport, int destinationSlot,
                                                    int maxTokensPerRank, void* stagedToken, int destinationRank) {
  if (destinationSlot < 0) return;

  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(Bf16);
  void* destinationBuffer = transport.mappedBuffer(output, destinationRank);
  auto* destinationRow = reinterpret_cast<uint8_t*>(destinationBuffer) +
                         (static_cast<size_t>(transport.rank_) * maxTokensPerRank + destinationSlot) * HiddenBytes;
  mscclpp::bulkStore(destinationRow, stagedToken, static_cast<uint32_t>(HiddenBytes));
  mscclpp::bulkStoreCommit();
}

MSCCLPP_DEVICE_INLINE void completeRankMajorTokenStore(WorkspaceView& workspaceView, int destinationRank) {
  if (destinationRank < 0) return;
  mscclpp::bulkStoreWait();
  (void)mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(
      workspaceView.dispatchRankPayloadCompletions_ + destinationRank, 1, mscclpp::memoryOrderRelease);
}

#if defined(MSCCLPP_USE_GPUNETIO)
// NVLink-only metadata store: identical to sendRankMajorMetadata but skips leaders
// whose destination is cross-domain (their metadata travels with the GPUNetIO send).
MSCCLPP_DEVICE_INLINE void sendRankMajorMetadataNvlink(
    const TransportView& transport, int* outputTopkIdx, float* outputTopkWeights,
    const int64_t* __restrict__ topkIndices, const float* __restrict__ topkWeights, const RankMajorRoute& route,
    int tokenIdx, int nTopk, int nLocalExperts, int maxTokensPerRank, int invalidTokenExpertId, bool skipCrossDomain) {
  const int laneId = get_lane_id();
  const int candidateExpert =
      laneId < nTopk ? static_cast<int>(topkIndices[tokenIdx * nTopk + laneId]) : invalidTokenExpertId;
  const float candidateWeight =
      laneId < nTopk ? (topkWeights == nullptr ? 1.0f : topkWeights[tokenIdx * nTopk + laneId]) : 0.0f;
  unsigned int leaderMask = __ballot_sync(0xffffffff, route.dstRank >= 0 && route.isLeader);
  while (leaderMask != 0) {
    const int leaderLane = __ffs(leaderMask) - 1;
    const int destinationRank = __shfl_sync(0xffffffff, route.dstRank, leaderLane);
    const int destinationSlot = __shfl_sync(0xffffffff, route.destinationSlot, leaderLane);
    leaderMask &= leaderMask - 1;
    if (skipCrossDomain && !transport.isNvlinkPeer(destinationRank)) continue;
    if (laneId < nTopk) {
      auto* destinationTopkIdx = reinterpret_cast<int*>(transport.mappedBuffer(outputTopkIdx, destinationRank));
      auto* destinationTopkWeights =
          reinterpret_cast<float*>(transport.mappedBuffer(outputTopkWeights, destinationRank));
      const size_t outputIdx =
          (static_cast<size_t>(transport.rank_) * maxTokensPerRank + destinationSlot) * nTopk + laneId;
      const bool isLocal = candidateExpert >= 0 && candidateExpert / nLocalExperts == destinationRank;
      destinationTopkIdx[outputIdx] = isLocal ? candidateExpert : invalidTokenExpertId;
      destinationTopkWeights[outputIdx] = isLocal ? candidateWeight : 0.0f;
    }
  }
  __syncwarp();
}

// TMA bulk store of a staged shared-memory span into the registered staging ring (one lane issues it and waits
// for the global writes to complete; the caller's row fence then publishes them to the NIC).
MSCCLPP_DEVICE_INLINE void tmaStageSpan(void* dstGlobal, const void* srcShared, uint32_t bytes) {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  mscclpp::bulkStore(dstGlobal, srcShared, bytes);
  mscclpp::bulkStoreCommit();
  mscclpp::bulkStoreWait();
  asm volatile("fence.proxy.async.global;" ::: "memory");
}

// GPU scope matches the release scope before the doorbell, which already orders the WQEs the NIC reads first.
MSCCLPP_DEVICE_INLINE void fenceRankMajorStagedRow(const TransportView& transport) {
  if (transport.gpuNetIoRowFence_ == GpuNetIoRowFenceSystem)
    __threadfence_system();
  else
    __threadfence();
}

// Cross-domain rank-major send: stages token+metadata in this rank's GPUNetIO
// staging ring, then posts three CQ-signaled writes with one submit. The notify
// block posts ordered per-QP markers after every payload has been submitted. Re-derived for
// the new layout: staging ring slots [nRanks, GpuNetIoStagingSlots) (the reserved
// prefix remains unused; count packets have separate metadata scratch). Remote
// destinations use the symmetric output/outputTopkIdx/outputTopkWeights bases.
template <int Hidden, bool Exclusive = false>
MSCCLPP_DEVICE_INLINE void postRankMajorGpuNetIoRows(const TransportView& transport, void* output, int* outputTopkIdx,
                                                     float* outputTopkWeights, int nRanks, int capacity, int nTopk,
                                                     int destination, int begin, int end, int queue,
                                                     [[maybe_unused]] TimingKind kind, [[maybe_unused]] int detail) {
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(Bf16);
  constexpr int MaxRowsPerWrite = DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE / HiddenBytes;
  static_assert(MaxRowsPerWrite > 0);
  EP_DEVICE_ASSERT(static_cast<size_t>(nTopk) * sizeof(int) <= HiddenBytes);
  const RankMajorBatchView batch(transport.gpuNetIoStagingBuffer_, nRanks, capacity, Hidden, nTopk);
  // Cumulative over rows acquired through chunk completion or the producer join, whatever the submit scope.
  if (transport.gpuNetIoRowFence_ == GpuNetIoRowFenceGpuPoster)
    __threadfence_system();
  else if (transport.gpuNetIoRowFence_ == GpuNetIoRowFenceGpu)
    __threadfence();
  for (; begin < end;) {
    const int rowLimit = transport.gpuNetIoZeroCopy_ ? 8 : MaxRowsPerWrite;
    const int count = end - begin < rowLimit ? end - begin : rowLimit;
    const size_t sourceRow = static_cast<size_t>(destination) * capacity + begin;
    const size_t remoteRow = static_cast<size_t>(transport.rank_) * capacity + begin;
    const size_t metadataBytes = static_cast<size_t>(count) * nTopk * sizeof(int);
    [[maybe_unused]] uint64_t* postTiming = nullptr;
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
    uint64_t postPhases[6] = {};
    uint64_t submitPhases[10] = {};
    uint64_t countValues[8] = {};
    uint64_t* submitCounts = transport.timing_.countSubmit() ? countValues : nullptr;
    uint64_t* submitTiming = transport.timing_.traceSubmit() ? submitPhases : nullptr;
    if (transport.timing_.buffer != nullptr && submitCounts == nullptr) postTiming = postPhases;
#endif
    if (transport.gpuNetIoTmaStage_ >= 3) {  // GPUBURST: block 0 posts a synthetic burst instead
      begin += count;
      continue;
    }
    if (!Exclusive && transport.gpuNetIoZeroCopy_) {
      // 1: 4-SGE (2-WQEBB) writes from x; 2: same writes from the staging rows (copy kept); 3: 2-SGE writes from x.
      const bool fromInput = transport.gpuNetIoZeroCopy_ != 2;
      if (fromInput && transport.gpuNetIoZcBase_ == 0) __trap();
      uint64_t srcAddrs[8];
      for (int r = 0; r < count; ++r)
        srcAddrs[r] = fromInput ? transport.gpuNetIoZcBase_ +
                                      static_cast<uint64_t>(transport.gpuNetIoZcRowSrc_[sourceRow + r]) * HiddenBytes
                                : transport.gpuNetIo_->localBase +
                                      transport.symmetricOffset(batch.tokens_ + (sourceRow + r) * HiddenBytes);
      transport.gpuNetIo_->putGatherRows(
          destination, queue, transport.symmetricOffset(reinterpret_cast<uint8_t*>(output) + remoteRow * HiddenBytes),
          srcAddrs, count, transport.gpuNetIoZeroCopy_ == 3 ? 2 : transport.gpuNetIoZeroCopy_ == 4 ? -2 : 4, static_cast<uint32_t>(HiddenBytes), fromInput ? transport.gpuNetIoZcLkeys_ : nullptr,
          transport.symmetricOffset(outputTopkIdx + remoteRow * nTopk),
          transport.symmetricOffset(batch.ids_ + sourceRow * nTopk), metadataBytes,
          transport.symmetricOffset(outputTopkWeights + remoteRow * nTopk),
          transport.symmetricOffset(batch.weights_ + sourceRow * nTopk), metadataBytes, postTiming
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
          ,
          submitTiming, submitCounts
#endif
      );
    } else if constexpr (Exclusive) {
      transport.gpuNetIo_->putBatched3Exclusive(
          destination, queue, transport.symmetricOffset(reinterpret_cast<uint8_t*>(output) + remoteRow * HiddenBytes),
          transport.symmetricOffset(batch.tokens_ + sourceRow * HiddenBytes), static_cast<size_t>(count) * HiddenBytes,
          transport.symmetricOffset(outputTopkIdx + remoteRow * nTopk),
          transport.symmetricOffset(batch.ids_ + sourceRow * nTopk), metadataBytes,
          transport.symmetricOffset(outputTopkWeights + remoteRow * nTopk),
          transport.symmetricOffset(batch.weights_ + sourceRow * nTopk), metadataBytes, postTiming);
    } else
    transport.gpuNetIo_->putBatched3(
        destination, queue, transport.symmetricOffset(reinterpret_cast<uint8_t*>(output) + remoteRow * HiddenBytes),
        transport.symmetricOffset(batch.tokens_ + sourceRow * HiddenBytes), static_cast<size_t>(count) * HiddenBytes,
        transport.symmetricOffset(outputTopkIdx + remoteRow * nTopk),
        transport.symmetricOffset(batch.ids_ + sourceRow * nTopk), metadataBytes,
        transport.symmetricOffset(outputTopkWeights + remoteRow * nTopk),
        transport.symmetricOffset(batch.weights_ + sourceRow * nTopk), metadataBytes, postTiming
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
        ,
        submitTiming, submitCounts
#endif
    );
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
    transport.timing_.record(kind, postPhases[0], postPhases[4], destination, queue, postPhases[1] - postPhases[0],
                             postPhases[2] - postPhases[1], postPhases[3] - postPhases[2],
                             postPhases[4] - postPhases[3], begin, count,
                             static_cast<size_t>(count) * HiddenBytes + 2 * metadataBytes, postPhases[5]);  // CQTRACE: v7=WQE base
    if (!Exclusive && submitTiming != nullptr)
      transport.timing_.recordSubmit(submitPhases, postPhases[3], postPhases[4], destination, queue);
    if (!Exclusive && submitCounts != nullptr) transport.timing_.recordSubmitCounts(countValues, destination, queue);
#endif
    begin += count;
  }
}

MSCCLPP_DEVICE_INLINE bool completeRankMajorGpuNetIoPeer(GpuNetIoDispatchPeerState& state, int rows, int expected) {
  EP_DEVICE_ASSERT(rows > 0 && rows <= expected);
  const int previous =
      mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(&state.postedRows, rows, mscclpp::memoryOrderRelease);
  EP_DEVICE_ASSERT(previous >= 0 && previous + rows <= expected);
  if (previous + rows != expected) return false;
  (void)mscclpp::atomicLoad<int, mscclpp::scopeDevice>(&state.postedRows, mscclpp::memoryOrderAcquire);
  return true;
}

MSCCLPP_DEVICE_INLINE bool completeRankMajorGpuNetIoChunk(int* completedRows, int chunkRows) {
  const int previous =
      mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(completedRows, 1, mscclpp::memoryOrderRelease);
  EP_DEVICE_ASSERT(previous >= 0 && previous < chunkRows);
  if (previous + 1 != chunkRows) return false;
  (void)mscclpp::atomicLoad<int, mscclpp::scopeDevice>(completedRows, mscclpp::memoryOrderAcquire);
  return true;
}

// Chunk end clipped to the destination's row count once the control CTA has published it (early tail post).
MSCCLPP_DEVICE_INLINE int rankMajorGpuNetIoChunkEnd(const TransportView& transport, int peer, int begin,
                                                    int chunkRows) {
  int end = begin + chunkRows;
  if (transport.gpuNetIoTailRows_ != nullptr) {
    const int rowsPlusOne = mscclpp::atomicLoad<int, mscclpp::scopeDevice>(transport.gpuNetIoTailRows_ + peer,
                                                                           mscclpp::memoryOrderAcquire);
    if (rowsPlusOne > 0 && rowsPlusOne - 1 < end) end = rowsPlusOne - 1;
  }
  return end;
}

// Dedicated-poster mode: one ready flag per (destination, row), consumed by the control-block posters.
MSCCLPP_DEVICE_INLINE void markRankMajorGpuNetIoRowReady(const TransportView& transport, int destination, int slot,
                                                         int capacity) {
  mscclpp::atomicStore<int, mscclpp::scopeDevice>(
      transport.gpuNetIoChunkReady_ + static_cast<size_t>(destination) * capacity + slot, 1,
      mscclpp::memoryOrderRelease);
}

// Arrival marker for (peer, queue): remote atomic add on the peer's per-QP flag.
MSCCLPP_DEVICE_INLINE void postRankMajorGpuNetIoMarker(const TransportView& transport, int peer, int queue) {
  auto* gin = transport.gpuNetIo_;
  auto* flagsSelf = static_cast<uint64_t*>(transport.gpuNetIoFlagsBuffer_) +
                    static_cast<size_t>(transport.rank_) * GpuNetIoMaxQpsPerPeer;
  gin->atomicAdd(peer, transport.symmetricOffset(flagsSelf + queue), 1, queue);
}

MSCCLPP_DEVICE_INLINE void markRankMajorGpuNetIoPeer(const TransportView& transport, int peer) {
  auto* gin = transport.gpuNetIo_;
  for (int queue = 0; queue < gin->numQpsPerPeer; ++queue) {
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
    const uint64_t begin = timingNow();
#endif
    postRankMajorGpuNetIoMarker(transport, peer, queue);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
    transport.timing_.record(TimingKind::DispatchQueue, begin, timingNow(), peer, queue, 0);
#endif
  }
  mscclpp::atomicStore<int, mscclpp::scopeDevice>(&transport.gpuNetIoDispatchPeers_[peer].markersPosted, 1,
                                                  mscclpp::memoryOrderRelease);
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void postRankMajorGpuNetIoAsyncChunk(const TransportView& transport, void* output,
                                                           int* outputTopkIdx, float* outputTopkWeights, int nRanks,
                                                           int capacity, int nTopk, int destination, int chunk,
                                                           int* completed) {
  auto& state = transport.gpuNetIoDispatchPeers_[destination];
  int published;
  do {
    published = mscclpp::atomicLoad<int, mscclpp::scopeDevice>(&state.expectedRowsPlusOne, mscclpp::memoryOrderAcquire);
  } while (published == 0);
  const int expected = published - 1;
  const int chunkRows = transport.gpuNetIoPipelineChunkRows_;
  const int begin = chunk * chunkRows;
  const int rows = expected - begin < chunkRows ? expected - begin : chunkRows;
  EP_DEVICE_ASSERT(expected <= capacity && rows > 0);
  if (!completeRankMajorGpuNetIoChunk(completed, rows)) return;
  const int queues = transport.gpuNetIoBatchAllQps_ ? transport.gpuNetIo_->numQpsPerPeer : transport.gpuNetIo_->numHcas;
  postRankMajorGpuNetIoRows<Hidden>(transport, output, outputTopkIdx, outputTopkWeights, nRanks, capacity, nTopk,
                                    destination, begin, begin + rows, chunk % queues, TimingKind::DispatchChunk,
                                    rows < chunkRows ? 2 : 0);
  if (completeRankMajorGpuNetIoPeer(state, rows, expected)) markRankMajorGpuNetIoPeer(transport, destination);
}

template <int Hidden, bool AsyncDispatch = false>
MSCCLPP_DEVICE_INLINE void sendRankMajorGpuNetIo(const TransportView& transport, void* output, int* outputTopkIdx,
                                                 float* outputTopkWeights, const RankMajorRoute& route,
                                                 const int64_t* __restrict__ topkIndices,
                                                 const float* __restrict__ topkWeights, void* stagedToken, int tokenIdx,
                                                 int nTopk, int nLocalExperts, int maxTokensPerRank, int nRanks,
                                                 int invalidTokenExpertId, int ringSlotBase, uint64_t* usedQpMask,
                                                 [[maybe_unused]] long long* timing) {
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(Bf16);
  const int laneId = get_lane_id();
  auto* gin = transport.gpuNetIo_;
  const size_t slotStride = transport.gpuNetIoSlotStride_;
  auto* stagingBase = reinterpret_cast<uint8_t*>(transport.gpuNetIoStagingBuffer_);

  const int candidateExpert = laneId < nTopk ? static_cast<int>(topkIndices[tokenIdx * nTopk + laneId]) : -1;
  const float candidateWeight =
      laneId < nTopk ? (topkWeights == nullptr ? 1.0f : topkWeights[tokenIdx * nTopk + laneId]) : 0.0f;

  unsigned int leaderMask = __ballot_sync(0xffffffff, route.dstRank >= 0 && route.isLeader);
  int ringSlot = ringSlotBase;
  while (leaderMask != 0) {
    const int leaderLane = __ffs(leaderMask) - 1;
    const int destinationRank = __shfl_sync(0xffffffff, route.dstRank, leaderLane);
    const int destinationSlot = __shfl_sync(0xffffffff, route.destinationSlot, leaderLane);
    const int leaderExpert = __shfl_sync(0xffffffff, candidateExpert, leaderLane);
    leaderMask &= leaderMask - 1;
    if (transport.isNvlinkPeer(destinationRank) || destinationSlot < 0) continue;
    const int qpIndex = (leaderExpert % nLocalExperts) % gin->numQpsPerPeer;
    if (laneId == leaderLane)
      atomicOr(reinterpret_cast<unsigned long long*>(usedQpMask + destinationRank), 1ull << qpIndex);

    auto* slot = stagingBase + static_cast<size_t>(nRanks + ringSlot) * slotStride;
    auto* slotIds = reinterpret_cast<int*>(slot + HiddenBytes);
    auto* slotWeights = reinterpret_cast<float*>(slot + HiddenBytes + static_cast<size_t>(nTopk) * sizeof(int));
    if (transport.gpuNetIoBatchedDispatch_) {
      const RankMajorBatchView batch(stagingBase, nRanks, maxTokensPerRank, Hidden, nTopk);
      const size_t row = static_cast<size_t>(destinationRank) * maxTokensPerRank + destinationSlot;
      slot = batch.tokens_ + row * HiddenBytes;
      slotIds = batch.ids_ + row * nTopk;
      slotWeights = batch.weights_ + row * nTopk;
      if (transport.gpuNetIoZeroCopy_ && laneId == 0) transport.gpuNetIoZcRowSrc_[row] = tokenIdx;
    }

#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
    const long long stageStart = timingNow();
#endif
    {
      const auto* src = reinterpret_cast<const uint4*>(stagedToken);
      auto* dst = reinterpret_cast<uint4*>(slot);
      constexpr int NumVec = static_cast<int>(HiddenBytes / sizeof(uint4));
      if (transport.gpuNetIoTmaStage_ >= 2 || (transport.gpuNetIoZeroCopy_ != 0 && transport.gpuNetIoZeroCopy_ != 2)) {
      } else if (transport.gpuNetIoTmaStage_) {
        if (laneId == 0) tmaStageSpan(dst, src, static_cast<uint32_t>(HiddenBytes));
      } else {
        for (int vector = laneId; vector < NumVec; vector += WARP_SIZE) dst[vector] = src[vector];
      }
    }
    if (laneId < nTopk) {
      const bool isLocal = candidateExpert >= 0 && candidateExpert / nLocalExperts == destinationRank;
      slotIds[laneId] = isLocal ? candidateExpert : invalidTokenExpertId;
      slotWeights[laneId] = isLocal ? candidateWeight : 0.0f;
    }
    __syncwarp();
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
    const long long stageDone = timingNow();
#endif
#if defined(MSCCLPP_EP_GPUNETIO_STAGE_TRACE)
    // Separates local store drain from the extra cost of system scope.
    __threadfence();
    const long long gpuFenceDone = timingNow();
#endif
    // Registered staging must be visible to NIC DMA, not just local GPU threads.
    if (transport.gpuNetIoBatchedDispatch_)
      fenceRankMajorStagedRow(transport);
    else
      __threadfence_system();
    __syncwarp();
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
    const long long fenceDone = timingNow();
#endif

    if (transport.gpuNetIoBatchedDispatch_) {
#if defined(MSCCLPP_EP_GPUNETIO_STAGE_TRACE)
      uint64_t publishNs = 0, posted = AsyncDispatch ? 2 : 0;
#endif
      if (!AsyncDispatch && transport.gpuNetIoDedicatedPoster_) {
        if (laneId == leaderLane)
          markRankMajorGpuNetIoRowReady(transport, destinationRank, destinationSlot, maxTokensPerRank);
      } else if (transport.gpuNetIoPipelineChunkRows_ > 0 && laneId == leaderLane) {
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
        const uint64_t publishBeginNs = timingNow();
#endif
        const int chunkRows = transport.gpuNetIoPipelineChunkRows_;
        const int chunksPerRank = configCellDiv(maxTokensPerRank, chunkRows);
        const int chunk = destinationSlot / chunkRows;
        auto* completed = transport.gpuNetIoChunkReady_ + destinationRank * chunksPerRank + chunk;
        if constexpr (AsyncDispatch) {
          postRankMajorGpuNetIoAsyncChunk<Hidden>(transport, output, outputTopkIdx, outputTopkWeights, nRanks,
                                                  maxTokensPerRank, nTopk, destinationRank, chunk, completed);
        } else if (completeRankMajorGpuNetIoChunk(completed, chunkRows)) {
#if defined(MSCCLPP_EP_GPUNETIO_STAGE_TRACE)
          posted = 1;
#endif
          const int queues = transport.gpuNetIoBatchAllQps_ ? gin->numQpsPerPeer : gin->numHcas;
          const int begin = chunk * chunkRows;
          postRankMajorGpuNetIoRows<Hidden>(transport, output, outputTopkIdx, outputTopkWeights, nRanks,
                                            maxTokensPerRank, nTopk, destinationRank, begin,
                                          rankMajorGpuNetIoChunkEnd(transport, destinationRank, begin, chunkRows),
                                            chunk % queues, TimingKind::DispatchChunk, 0);
        }
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
        timing[2] += timingNow() - publishBeginNs;
#endif
#if defined(MSCCLPP_EP_GPUNETIO_STAGE_TRACE)
        publishNs = timingNow() - publishBeginNs;
#endif
      }
      if (transport.gpuNetIoPipelineChunkRows_ > 0) __syncwarp();
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
      if (laneId == leaderLane) {
        timing[0] += stageDone - stageStart;
        timing[1] += fenceDone - stageDone;
        timing[3] += 1;
#if defined(MSCCLPP_EP_GPUNETIO_STAGE_TRACE)
        transport.timing_.record(TimingKind::DispatchStage, stageStart, fenceDone + publishNs, destinationRank,
                                 ringSlot - ringSlotBase, stageDone - stageStart, gpuFenceDone - stageDone,
                                 fenceDone - gpuFenceDone, publishNs, posted, tokenIdx, 0, destinationSlot);
#endif
      }
#endif
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
      // CHUNKTRACE: row ready (begin=fence done, end=after publish/post). v0=0xC7, v1=slot, v7=path
      if (laneId == leaderLane && transport.timing_.buffer != nullptr)
        transport.timing_.record(TimingKind::DispatchStage, fenceDone, timingNow(), destinationRank, -1, 0xC7,
                                 destinationSlot, 0, 0, 0, 0, 0, 1);
#endif
      ++ringSlot;
      continue;
    }

    if (laneId == leaderLane) {
      const size_t rowBase = static_cast<size_t>(transport.rank_) * maxTokensPerRank + destinationSlot;
      auto* remoteToken = reinterpret_cast<uint8_t*>(output) + rowBase * HiddenBytes;
      auto* remoteIds = reinterpret_cast<uint8_t*>(outputTopkIdx) + rowBase * nTopk * sizeof(int);
      auto* remoteWeights = reinterpret_cast<uint8_t*>(outputTopkWeights) + rowBase * nTopk * sizeof(float);
      [[maybe_unused]] uint64_t* postTiming = nullptr;
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
      uint64_t postPhases[6] = {};
      uint64_t submitPhases[10] = {};
      uint64_t countValues[8] = {};
      uint64_t* submitCounts = transport.timing_.countSubmit() ? countValues : nullptr;
      uint64_t* submitTiming = transport.timing_.traceSubmit() ? submitPhases : nullptr;
      if (transport.timing_.buffer != nullptr && submitCounts == nullptr) postTiming = postPhases;
#endif
      if (transport.gpuNetIoTmaStage_ < 3)
      gin->putBatched3(destinationRank, qpIndex, transport.symmetricOffset(remoteToken),
                       transport.symmetricOffset(slot), HiddenBytes, transport.symmetricOffset(remoteIds),
                       transport.symmetricOffset(slotIds), static_cast<size_t>(nTopk) * sizeof(int),
                       transport.symmetricOffset(remoteWeights), transport.symmetricOffset(slotWeights),
                       static_cast<size_t>(nTopk) * sizeof(float), postTiming
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
                       ,
                       submitTiming, submitCounts
#endif
      );
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
      transport.timing_.record(TimingKind::DispatchPost, postPhases[0], postPhases[4], destinationRank, qpIndex,
                               postPhases[1] - postPhases[0], postPhases[2] - postPhases[1],
                               postPhases[3] - postPhases[2], postPhases[4] - postPhases[3], tokenIdx,
                               HiddenBytes + 2 * static_cast<size_t>(nTopk) * sizeof(int));
      if (submitTiming != nullptr)
        transport.timing_.recordSubmit(submitPhases, postPhases[3], postPhases[4], destinationRank, qpIndex);
      if (submitCounts != nullptr) transport.timing_.recordSubmitCounts(countValues, destinationRank, qpIndex);
#endif
    }
    __syncwarp();
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
    if (laneId == leaderLane) {
      timing[0] += stageDone - stageStart;
      timing[1] += fenceDone - stageDone;
      timing[2] += timingNow() - fenceDone;
      timing[3] += 1;
    }
#endif
    ringSlot += 1;
  }
}

MSCCLPP_DEVICE_INLINE bool useRankMajorParallelStage(const TransportView& transport) {
  return transport.gpuNetIo_ != nullptr && transport.gpuNetIoBatchedDispatch_ && transport.gpuNetIoParallelStage_ > 0;
}

// Arrival counts for rows split across helper warps, cleared by the group leader per token.
MSCCLPP_DEVICE_INLINE int* rankMajorParallelStageParts(int warpGroupId) {
  __shared__ int parts[DispatchMaxNWarpGroups * WARP_SIZE];
  return parts + warpGroupId * WARP_SIZE;
}

#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
MSCCLPP_DEVICE_INLINE unsigned long long* rankMajorParallelStageTiming(int warpGroupId) {
  __shared__ unsigned long long totals[DispatchMaxNWarpGroups * 4];
  return totals + warpGroupId * 4;
}
#endif

// Helper-warp counterpart of the batched sendRankMajorGpuNetIo path: remote destinations of one token
// are staged, fenced and published concurrently instead of serially by the group leader.
template <int Hidden, bool AsyncDispatch>
MSCCLPP_DEVICE_INLINE void stageRankMajorGpuNetIoParallel(
    const TransportView& transport, void* output, int* outputTopkIdx, float* outputTopkWeights,
    const int64_t* __restrict__ topkIndices, const float* __restrict__ topkWeights, const WorkspaceView& workspaceView,
    const void* stagedToken, int tokenIdx, int nTopk, int nLocalExperts, int maxTokensPerRank, int nRanks,
    int invalidTokenExpertId, int helper, int nHelpers, [[maybe_unused]] int warpGroupId, int* parts,
    uint64_t* usedQpMask) {
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(Bf16);
  constexpr int NumVec = static_cast<int>(HiddenBytes / sizeof(uint4));
  const int laneId = get_lane_id();
  auto* gin = transport.gpuNetIo_;
  const int candidateExpert = laneId < nTopk ? static_cast<int>(topkIndices[tokenIdx * nTopk + laneId]) : -1;
  const float candidateWeight =
      laneId < nTopk ? (topkWeights == nullptr ? 1.0f : topkWeights[tokenIdx * nTopk + laneId]) : 0.0f;
  const int dstRank = candidateExpert >= 0 ? candidateExpert / nLocalExperts : -1;
  const int slotIndex = laneId < nTopk ? workspaceView.rankMajorSendIndices_[tokenIdx * nTopk + laneId] : -1;
  const bool isLeader = isFirstLaneForRank(dstRank, laneId);
  const unsigned int remoteMask =
      __ballot_sync(0xffffffff, dstRank >= 0 && isLeader && slotIndex >= 0 && !transport.isNvlinkPeer(dstRank));
  const int count = __popc(remoteMask);
  if (count == 0) return;
  int warpsPerDestination = nHelpers / count;
  if (warpsPerDestination > transport.gpuNetIoParallelStage_) warpsPerDestination = transport.gpuNetIoParallelStage_;
  if (warpsPerDestination < 1) warpsPerDestination = 1;
  const RankMajorBatchView batch(transport.gpuNetIoStagingBuffer_, nRanks, maxTokensPerRank, Hidden, nTopk);

  for (int item = helper; item < count * warpsPerDestination; item += nHelpers) {
    const int order = item / warpsPerDestination;
    const int part = item % warpsPerDestination;
    unsigned int mask = remoteMask;
    for (int skipped = 0; skipped < order; ++skipped) mask &= mask - 1;
    const int leaderLane = __ffs(mask) - 1;
    const int destinationRank = __shfl_sync(0xffffffff, dstRank, leaderLane);
    const int destinationSlot = __shfl_sync(0xffffffff, slotIndex, leaderLane);
    const int leaderExpert = __shfl_sync(0xffffffff, candidateExpert, leaderLane);
    const int qpIndex = (leaderExpert % nLocalExperts) % gin->numQpsPerPeer;
    if (part == 0 && laneId == 0)
      atomicOr(reinterpret_cast<unsigned long long*>(usedQpMask + destinationRank), 1ull << qpIndex);
    const size_t row = static_cast<size_t>(destinationRank) * maxTokensPerRank + destinationSlot;

#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
    const long long stageStart = timingNow();
#endif
    {
      const auto* src = reinterpret_cast<const uint4*>(stagedToken);
      auto* dst = reinterpret_cast<uint4*>(batch.tokens_ + row * HiddenBytes);
      const int vectorBegin = NumVec * part / warpsPerDestination;
      const int vectorEnd = NumVec * (part + 1) / warpsPerDestination;
      if (transport.gpuNetIoTmaStage_ >= 2 || (transport.gpuNetIoZeroCopy_ != 0 && transport.gpuNetIoZeroCopy_ != 2)) {
      } else if (transport.gpuNetIoTmaStage_) {
        if (laneId == 0 && vectorEnd > vectorBegin)
          tmaStageSpan(dst + vectorBegin, src + vectorBegin,
                       static_cast<uint32_t>((vectorEnd - vectorBegin) * sizeof(uint4)));
      } else {
        for (int vector = vectorBegin + laneId; vector < vectorEnd; vector += WARP_SIZE) dst[vector] = src[vector];
      }
    }
    if (transport.gpuNetIoZeroCopy_ && part == 0 && laneId == 0) transport.gpuNetIoZcRowSrc_[row] = tokenIdx;
    if (part == 0 && laneId < nTopk) {
      const bool isLocal = candidateExpert >= 0 && candidateExpert / nLocalExperts == destinationRank;
      batch.ids_[row * nTopk + laneId] = isLocal ? candidateExpert : invalidTokenExpertId;
      batch.weights_[row * nTopk + laneId] = isLocal ? candidateWeight : 0.0f;
    }
    __syncwarp();
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
    const long long stageDone = timingNow();
#endif
#if defined(MSCCLPP_EP_GPUNETIO_STAGE_TRACE)
    __threadfence();
    const long long gpuFenceDone = timingNow();
#endif
    // Every part is fenced before it counts toward the row, so the publisher covers the whole row.
    fenceRankMajorStagedRow(transport);
    __syncwarp();
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
    const long long fenceDone = timingNow();
#endif

    bool publisher = true;
    if (warpsPerDestination > 1) {
      int previous = 0;
      if (laneId == 0)
        previous = mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(parts + order, 1, mscclpp::memoryOrderAcqRel);
      publisher = __shfl_sync(0xffffffff, previous, 0) + 1 == warpsPerDestination;
    }
    [[maybe_unused]] uint64_t publishNs = 0, posted = AsyncDispatch ? 2 : 0;
    if (!AsyncDispatch && transport.gpuNetIoDedicatedPoster_) {
      if (publisher && laneId == 0)
        markRankMajorGpuNetIoRowReady(transport, destinationRank, destinationSlot, maxTokensPerRank);
    } else if (publisher && transport.gpuNetIoPipelineChunkRows_ > 0 && laneId == 0) {
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
      const uint64_t publishBeginNs = timingNow();
#endif
      const int chunkRows = transport.gpuNetIoPipelineChunkRows_;
      const int chunksPerRank = configCellDiv(maxTokensPerRank, chunkRows);
      const int chunk = destinationSlot / chunkRows;
      auto* completed = transport.gpuNetIoChunkReady_ + destinationRank * chunksPerRank + chunk;
      if constexpr (AsyncDispatch) {
        postRankMajorGpuNetIoAsyncChunk<Hidden>(transport, output, outputTopkIdx, outputTopkWeights, nRanks,
                                                maxTokensPerRank, nTopk, destinationRank, chunk, completed);
      } else if (completeRankMajorGpuNetIoChunk(completed, chunkRows)) {
        posted = 1;
        const int queues = transport.gpuNetIoBatchAllQps_ ? gin->numQpsPerPeer : gin->numHcas;
        const int begin = chunk * chunkRows;
        postRankMajorGpuNetIoRows<Hidden>(transport, output, outputTopkIdx, outputTopkWeights, nRanks, maxTokensPerRank,
                                          nTopk, destinationRank, begin,
                                          rankMajorGpuNetIoChunkEnd(transport, destinationRank, begin, chunkRows), chunk % queues,
                                          TimingKind::DispatchChunk, 0);
      }
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
      publishNs = timingNow() - publishBeginNs;
#endif
    }
    __syncwarp();
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
    if (publisher && laneId == 0 && transport.timing_.buffer != nullptr)
      transport.timing_.record(TimingKind::DispatchStage, fenceDone, timingNow(), destinationRank, -1, 0xC7,
                               destinationSlot, posted, 0, 0, 0, 0, 2);  // CHUNKTRACE row ready
#endif
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
    if (laneId == 0) {
      auto* totals = rankMajorParallelStageTiming(warpGroupId);
      atomicAdd(totals + 0, static_cast<unsigned long long>(stageDone - stageStart));
      atomicAdd(totals + 1, static_cast<unsigned long long>(fenceDone - stageDone));
      atomicAdd(totals + 2, static_cast<unsigned long long>(publishNs));
      if (publisher) atomicAdd(totals + 3, 1ull);
#if defined(MSCCLPP_EP_GPUNETIO_STAGE_TRACE)
      if (publisher)
        transport.timing_.record(TimingKind::DispatchStage, stageStart, fenceDone + publishNs, destinationRank, order,
                                 stageDone - stageStart, gpuFenceDone - stageDone, fenceDone - gpuFenceDone, publishNs,
                                 posted, tokenIdx, warpsPerDestination, destinationSlot);
#endif
    }
#endif
  }
}

// Non-leader warps of a producer warp group: wait for the leader to load and route each token, then
// stage its remote destinations. Returns false for warps that must run the leader path instead.
template <int Hidden, bool AsyncDispatch>
MSCCLPP_DEVICE_INLINE bool runRankMajorParallelStageHelper(void* output, int* outputTopkIdx, float* outputTopkWeights,
                                                           int nExperts, int nRanks,
                                                           const int64_t* __restrict__ topkIndices,
                                                           const float* __restrict__ topkWeights, int nTokens,
                                                           int nTopk, int invalidTokenExpertId, int maxTokensPerRank,
                                                           const TransportView& transport, void* workspace,
                                                           int nPayloadBlocks, int* sharedMem, uint64_t* usedQpMask) {
  if (!useRankMajorParallelStage(transport)) return false;
  const int firstProducer = AsyncDispatch ? nRanks : 1;
  if (static_cast<int>(blockIdx.x) < firstProducer || static_cast<int>(blockIdx.x) >= firstProducer + nPayloadBlocks)
    return false;
  const int warpId = static_cast<int>(threadIdx.x) / WARP_SIZE;
  const int nWarpsPerGroup = dispatchNWarpsPerGroup(nTokens, nPayloadBlocks);
  const int subWarpId = warpId % nWarpsPerGroup;
  if (subWarpId == 0) return false;

  const int nWarpGroups = DispatchNWarps / nWarpsPerGroup;
  const int warpGroupId = warpId / nWarpsPerGroup;
  const int senderBlockIdx = static_cast<int>(blockIdx.x) - firstProducer;
  const size_t sharedTokenStride = dispatchPayloadStride<DispatchDataType::BF16>(Hidden, nTopk, 0);
  const auto* stagedToken = reinterpret_cast<uint8_t*>(sharedMem) + dispatchSharedControlBytes(nRanks) +
                            static_cast<size_t>(warpGroupId) * sharedTokenStride;
  const WorkspaceView workspaceView(workspace, nRanks, nExperts);
  int* parts = rankMajorParallelStageParts(warpGroupId);
  const int barrierId = DispatchWarpGroupBarrierBase + warpGroupId;
  const int barrierThreads = nWarpsPerGroup * WARP_SIZE;
  for (int tokenIdx = senderBlockIdx * nWarpGroups + warpGroupId; tokenIdx < nTokens;
       tokenIdx += nPayloadBlocks * nWarpGroups) {
    // The leader waited on the bulk barrier and ran bulkFence() before this barrier.
    syncNamedBarrier(barrierId, barrierThreads);
    stageRankMajorGpuNetIoParallel<Hidden, AsyncDispatch>(
        transport, output, outputTopkIdx, outputTopkWeights, topkIndices, topkWeights, workspaceView, stagedToken,
        tokenIdx, nTopk, nExperts / nRanks, maxTokensPerRank, nRanks, invalidTokenExpertId, subWarpId - 1,
        nWarpsPerGroup - 1, warpGroupId, parts, usedQpMask);
    syncNamedBarrier(barrierId, barrierThreads);
  }
  return true;
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void sendRankMajorGpuNetIoBatches(const TransportView& transport, void* output,
                                                        int* outputTopkIdx, float* outputTopkWeights, int nRanks,
                                                        int capacity, int nTopk, const WorkspaceView& workspace,
                                                        int destination, int stripe) {
  auto* gin = transport.gpuNetIo_;
  const int nStripes = transport.gpuNetIoBatchAllQps_ ? gin->numQpsPerPeer : gin->numHcas;
  if (destination >= nRanks || transport.isNvlinkPeer(destination) || stripe >= nStripes) return;
  const int rows = mscclpp::atomicLoad<int, mscclpp::scopeDevice>(workspace.dispatchRankPayloadSlots_ + destination,
                                                                  mscclpp::memoryOrderAcquire);
  EP_DEVICE_ASSERT(rows >= 0 && rows <= capacity);
  EP_DEVICE_ASSERT(gin->numHcas > 0 && gin->numHcas <= gin->numQpsPerPeer);
  if (transport.gpuNetIoPipelineChunkRows_ > 0) {
    const int chunkRows = transport.gpuNetIoPipelineChunkRows_;
    const int tailRows = rows % chunkRows;
    const int chunk = rows / chunkRows;
    // Early tail post: the producers (or the notify step) already posted the tail chunk.
    if (tailRows > 0 && stripe == chunk % nStripes && transport.gpuNetIoTailRows_ == nullptr) {
      const int chunksPerRank = configCellDiv(capacity, chunkRows);
      const int completed = mscclpp::atomicLoad<int, mscclpp::scopeDevice>(
          transport.gpuNetIoChunkReady_ + destination * chunksPerRank + chunk, mscclpp::memoryOrderAcquire);
      EP_DEVICE_ASSERT(completed == tailRows);
      postRankMajorGpuNetIoRows<Hidden>(transport, output, outputTopkIdx, outputTopkWeights, nRanks, capacity, nTopk,
                                        destination, rows - tailRows, rows, stripe, TimingKind::DispatchChunk, 1);
    }
    return;
  }
  const int end = rows * (stripe + 1) / nStripes;
  postRankMajorGpuNetIoRows<Hidden>(transport, output, outputTopkIdx, outputTopkWeights, nRanks, capacity, nTopk,
                                    destination, rows * stripe / nStripes, end, stripe, TimingKind::DispatchBatch,
                                    rows);
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void sendRankMajorGpuNetIoControlBatches(const TransportView& transport, void* output,
                                                               int* outputTopkIdx, float* outputTopkWeights, int nRanks,
                                                               int capacity, int nTopk,
                                                               const WorkspaceView& workspace) {
  const int stripes =
      transport.gpuNetIoBatchAllQps_ ? transport.gpuNetIo_->numQpsPerPeer : transport.gpuNetIo_->numHcas;
  for (int index = threadIdx.x; index < nRanks * stripes; index += blockDim.x) {
    sendRankMajorGpuNetIoBatches<Hidden>(transport, output, outputTopkIdx, outputTopkWeights, nRanks, capacity, nTopk,
                                         workspace, index / stripes, index % stripes);
  }
}

MSCCLPP_DEVICE_INLINE void joinRankMajorGpuNetIoProducers(int* completed, int nWorkerBlocks) {
  const int block = static_cast<int>(blockIdx.x);
  if (block > 0 && block <= nWorkerBlocks) {
    __syncthreads();
    if (threadIdx.x == 0) {
      const int previous =
          mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(completed, 1, mscclpp::memoryOrderRelease);
      EP_DEVICE_ASSERT(previous >= 0 && previous < nWorkerBlocks);
    }
  } else if (block == nWorkerBlocks + 1) {
    if (threadIdx.x == 0) {
      int count;
      do {
        count = mscclpp::atomicLoad<int, mscclpp::scopeDevice>(completed, mscclpp::memoryOrderAcquire);
        EP_DEVICE_ASSERT(count >= 0 && count <= nWorkerBlocks);
      } while (count != nWorkerBlocks);
    }
    __syncthreads();
  }
}

MSCCLPP_DEVICE_INLINE void flushAllCrossDomain(const TransportView& transport, int nRanks, int qpIndex) {
  for (int peer = 0; peer < nRanks; ++peer) {
    if (!transport.isNvlinkPeer(peer)) transport.gpuNetIo_->flush(peer, qpIndex);
  }
}

MSCCLPP_DEVICE_INLINE void flushAllCrossDomainAllQps(const TransportView& transport, int nRanks) {
  for (int qpIndex = 0; qpIndex < transport.gpuNetIo_->numQpsPerPeer; ++qpIndex)
    flushAllCrossDomain(transport, nRanks, qpIndex);
}
#endif  // defined(MSCCLPP_USE_GPUNETIO)

MSCCLPP_DEVICE_INLINE int* aggregatedDispatchCompletionCounts(int* sharedMem, int nRanks, int warpGroupId) {
  return sharedMem + dispatchSharedSendSlots(nRanks) + static_cast<size_t>(warpGroupId) * nRanks;
}

MSCCLPP_DEVICE_INLINE void initAggregatedDispatchCompletions(int* completionCounts, int nRanks, int laneId) {
  for (int dstRank = laneId; dstRank < nRanks; dstRank += WARP_SIZE) completionCounts[dstRank] = 0;
  __syncwarp();
}

MSCCLPP_DEVICE_INLINE void flushAggregatedDispatchCompletions(WorkspaceView& workspaceView, int* completionCounts,
                                                              int nRanks, int laneId) {
  __syncwarp();
  for (int dstRank = laneId; dstRank < nRanks; dstRank += WARP_SIZE) {
    const int nCompleted = completionCounts[dstRank];
    if (nCompleted > 0) {
      (void)mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(workspaceView.dispatchRankPayloadCompletions_ + dstRank,
                                                               nCompleted, mscclpp::memoryOrderRelease);
    }
  }
}

MSCCLPP_DEVICE_INLINE void writeRankMajorCounts(const TransportView& transport, const int* rankTokenCounts, int nRanks,
                                                void* recvBuffer, uint32_t epoch, bool skipGpuNetIo = false);

MSCCLPP_DEVICE_INLINE void publishDispatchPayloads(const TransportView& transport, const int* rankTokenCounts,
                                                   int nRanks, WorkspaceView workspaceView);

template <int Hidden>
struct RankMajorSendState {
  int laneId_;
  int warpGroupId_;
  int tokenStride_;
  int firstTokenIdx_;
  uint8_t* stagedToken_;
  mscclpp::BulkBarrier* bulkBarrier_;
  uint32_t bulkPhase_;
};

template <int Hidden, bool AsyncDispatch = false>
MSCCLPP_DEVICE_INLINE bool initRankMajorSendState(RankMajorSendState<Hidden>& state, int nTokens, int nTopk, int nRanks,
                                                  int nPayloadBlocks, int* sharedMem) {
  const int firstProducer = AsyncDispatch ? nRanks : 1;
  if (static_cast<int>(blockIdx.x) < firstProducer || static_cast<int>(blockIdx.x) >= firstProducer + nPayloadBlocks)
    return false;

  const int warpId = static_cast<int>(threadIdx.x) / WARP_SIZE;
  const int senderBlockIdx = static_cast<int>(blockIdx.x) - firstProducer;
  const int nWarpsPerGroup = dispatchNWarpsPerGroup(nTokens, nPayloadBlocks);
  const int nWarpGroups = DispatchNWarps / nWarpsPerGroup;
  const int subWarpId = warpId % nWarpsPerGroup;
  if (subWarpId != 0) return false;

  const size_t sharedTokenStride = dispatchPayloadStride<DispatchDataType::BF16>(Hidden, nTopk, 0);
  auto* sharedTokenBase = reinterpret_cast<uint8_t*>(sharedMem) + dispatchSharedControlBytes(nRanks);
  auto* bulkBarriers =
      reinterpret_cast<mscclpp::BulkBarrier*>(sharedTokenBase + DispatchMaxNWarpGroups * sharedTokenStride);

  state.laneId_ = get_lane_id();
  state.warpGroupId_ = warpId / nWarpsPerGroup;
  state.tokenStride_ = nPayloadBlocks * nWarpGroups;
  state.firstTokenIdx_ = senderBlockIdx * nWarpGroups + state.warpGroupId_;
  state.stagedToken_ = sharedTokenBase + static_cast<size_t>(state.warpGroupId_) * sharedTokenStride;
  state.bulkBarrier_ = bulkBarriers + state.warpGroupId_;
  state.bulkPhase_ = 0;
  if (state.firstTokenIdx_ < nTokens && state.laneId_ == 0) state.bulkBarrier_->init();
  return true;
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void stageRankMajorBf16Token(const void* inputTokens, int tokenIdx,
                                                   RankMajorSendState<Hidden>& state) {
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(Bf16);
  constexpr int HiddenVectors = Hidden / mscclpp::bf16x8::Size;
  const auto* inputData =
      reinterpret_cast<const mscclpp::bf16x8*>(inputTokens) + static_cast<size_t>(tokenIdx) * HiddenVectors;
  if (state.laneId_ == 0) {
    state.bulkBarrier_->arriveAndExpect(static_cast<uint32_t>(HiddenBytes));
    mscclpp::bulkLoad(state.stagedToken_, inputData, static_cast<uint32_t>(HiddenBytes), *state.bulkBarrier_);
  }
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void waitRankMajorBf16Token(RankMajorSendState<Hidden>& state) {
  if (state.laneId_ == 0) state.bulkBarrier_->wait(state.bulkPhase_);
  __syncwarp();
  mscclpp::bulkFence();
}

template <int Hidden, bool AsyncDispatch = false>
MSCCLPP_DEVICE_INLINE void dispatchSendRankMajorBf16(
    void* output, int* outputTopkIdx, float* outputTopkWeights, const void* inputTokens, int nExperts, int nRanks,
    const int64_t* __restrict__ topkIndices, const float* __restrict__ topkWeights, int nTokens, int nTopk,
    int invalidTokenExpertId, int maxTokensPerRank, const TransportView& transport, void* workspace, int nPayloadBlocks,
    int* sharedMem, [[maybe_unused]] uint64_t* usedQpMask) {
  RankMajorSendState<Hidden> sendState;
#if defined(MSCCLPP_USE_GPUNETIO)
  if (runRankMajorParallelStageHelper<Hidden, AsyncDispatch>(
          output, outputTopkIdx, outputTopkWeights, nExperts, nRanks, topkIndices, topkWeights, nTokens, nTopk,
          invalidTokenExpertId, maxTokensPerRank, transport, workspace, nPayloadBlocks, sharedMem, usedQpMask))
    return;
#endif
  if (!initRankMajorSendState<Hidden, AsyncDispatch>(sendState, nTokens, nTopk, nRanks, nPayloadBlocks, sharedMem))
    return;

  const int laneId = sendState.laneId_;
  const int nLocalExperts = nExperts / nRanks;
  WorkspaceView workspaceView(workspace, nRanks, nExperts);

  [[maybe_unused]] int tokensSinceFlush = 0;
  [[maybe_unused]] long long timing[4] = {};
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  const uint64_t sendBeginNs = timingNow();
  uint64_t loadNs = 0, reuseFlushNs = 0, tokenCount = 0;
#endif
#if defined(MSCCLPP_USE_GPUNETIO)
  const int senderBlockIdx = static_cast<int>(blockIdx.x) - (AsyncDispatch ? nRanks : 1);
  const int warpGroupId = sendState.warpGroupId_;
  auto* stagedToken = sendState.stagedToken_;
  const int slotsPerGroup = (GpuNetIoStagingSlots - nRanks) / (nPayloadBlocks * DispatchMaxNWarpGroups);
  const int availableTokens = slotsPerGroup / nTopk;
  const int batchTokens = availableTokens < GpuNetIoFlushInterval ? availableTokens : GpuNetIoFlushInterval;
  if (transport.gpuNetIo_ != nullptr && batchTokens < 1) __trap();
  const bool parallelStage = useRankMajorParallelStage(transport);
  const int stageBarrierId = DispatchWarpGroupBarrierBase + warpGroupId;
  const int stageBarrierThreads = dispatchNWarpsPerGroup(nTokens, nPayloadBlocks) * WARP_SIZE;
  int* stageParts = rankMajorParallelStageParts(warpGroupId);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  if (parallelStage && laneId < 4) rankMajorParallelStageTiming(warpGroupId)[laneId] = 0;
#endif
#endif

  for (int tokenIdx = sendState.firstTokenIdx_; tokenIdx < nTokens; tokenIdx += sendState.tokenStride_) {
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
    const uint64_t loadBeginNs = timingNow();
#endif
    stageRankMajorBf16Token(inputTokens, tokenIdx, sendState);
    const RankMajorRoute route =
        prepareRankMajorRoute(workspaceView, topkIndices, tokenIdx, nTopk, nLocalExperts, maxTokensPerRank, laneId);
    waitRankMajorBf16Token(sendState);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
    loadNs += timingNow() - loadBeginNs;
    ++tokenCount;
#endif
    const int completionRank = route.dstRank >= 0 && route.isLeader ? route.dstRank : -1;
#if defined(MSCCLPP_USE_GPUNETIO)
    const bool crossDomain = transport.gpuNetIo_ != nullptr;
    // NVLink token store only for NVLink-mapped destinations; cross-domain
    // destinations travel via the warp-cooperative GPUNetIO send below.
    if (completionRank >= 0 && (!crossDomain || transport.isNvlinkPeer(completionRank))) {
      issueRankMajorTokenStore<Hidden>(output, transport, route.destinationSlot, maxTokensPerRank, stagedToken,
                                       route.dstRank);
    }
    if (crossDomain && parallelStage) {
      stageParts[laneId] = 0;
      syncNamedBarrier(stageBarrierId, stageBarrierThreads);
    } else if (crossDomain) {
      const int ringSlotBase =
          (senderBlockIdx * DispatchMaxNWarpGroups + warpGroupId) * slotsPerGroup + tokensSinceFlush * nTopk;
      sendRankMajorGpuNetIo<Hidden, AsyncDispatch>(
          transport, output, outputTopkIdx, outputTopkWeights, route, topkIndices, topkWeights, stagedToken, tokenIdx,
          nTopk, nLocalExperts, maxTokensPerRank, nRanks, invalidTokenExpertId, ringSlotBase, usedQpMask, timing);
      if (!transport.gpuNetIoBatchedDispatch_ && ++tokensSinceFlush >= batchTokens) {
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
        const uint64_t reuseBeginNs = timingNow();
#endif
        if (laneId == 0) flushAllCrossDomainAllQps(transport, nRanks);
        __syncwarp();
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
        reuseFlushNs += timingNow() - reuseBeginNs;
#endif
        tokensSinceFlush = 0;
      }
    }
    const int nvlinkCompletionRank =
        (completionRank >= 0 && (!crossDomain || transport.isNvlinkPeer(completionRank))) ? completionRank : -1;
    sendRankMajorMetadataNvlink(transport, outputTopkIdx, outputTopkWeights, topkIndices, topkWeights, route, tokenIdx,
                                nTopk, nLocalExperts, maxTokensPerRank, invalidTokenExpertId, crossDomain);
    completeRankMajorTokenStore(workspaceView, nvlinkCompletionRank);
    // Helpers still read the staged token; the next bulk load must not overwrite it.
    if (crossDomain && parallelStage) syncNamedBarrier(stageBarrierId, stageBarrierThreads);
#else
    if (completionRank >= 0) {
      issueRankMajorTokenStore<Hidden>(output, transport, route.destinationSlot, maxTokensPerRank,
                                       sendState.stagedToken_, route.dstRank);
    }
    sendRankMajorMetadata(transport, outputTopkIdx, outputTopkWeights, topkIndices, topkWeights, route, tokenIdx, nTopk,
                          nLocalExperts, maxTokensPerRank, invalidTokenExpertId);
    completeRankMajorTokenStore(workspaceView, completionRank);
#endif  // defined(MSCCLPP_USE_GPUNETIO)
    if (tokenIdx + sendState.tokenStride_ < nTokens) __syncwarp();
  }
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  for (int offset = 16; offset > 0; offset >>= 1) {
    for (int metric = 0; metric < 4; ++metric) {
      timing[metric] += __shfl_down_sync(0xffffffff, timing[metric], offset);
    }
  }
#if defined(MSCCLPP_USE_GPUNETIO)
  if (parallelStage && laneId == 0) {
    for (int metric = 0; metric < 4; ++metric) timing[metric] += rankMajorParallelStageTiming(warpGroupId)[metric];
  }
#endif
  if (laneId == 0)
    transport.timing_.record(TimingKind::DispatchSend, sendBeginNs, timingNow(), -1, -1, timing[0], timing[1],
                             timing[2], timing[3], loadNs, reuseFlushNs, tokenCount, sendState.warpGroupId_);
#endif
}

MSCCLPP_DEVICE_INLINE void countRankMajorTopkExpandedRoutes(int* rankTokenCounts,
                                                            const int64_t* __restrict__ topkIndices, int nTokens,
                                                            int nTopk, int nRanks, int nExperts) {
  const int threadId = static_cast<int>(threadIdx.x);
  const int warpId = threadId / WARP_SIZE;
  const int laneId = get_lane_id();
  const int nLocalExperts = nExperts / nRanks;
  for (int rankIdx = threadId; rankIdx < nRanks; rankIdx += blockDim.x) rankTokenCounts[rankIdx] = 0;
  __syncthreads();
  for (int tokenIdx = warpId; tokenIdx < nTokens; tokenIdx += DispatchNWarps) {
    const int expert = laneId < nTopk ? static_cast<int>(topkIndices[tokenIdx * nTopk + laneId]) : -1;
    const int dstRank = expert >= 0 ? expert / nLocalExperts : -1;
    if (dstRank >= 0) atomicAdd_block(rankTokenCounts + dstRank, 1);
  }
  __syncthreads();
}

MSCCLPP_DEVICE_INLINE void invalidateRankMajorTopkExpandedPadding(const TransportView& transport, int* outputTopkIdx,
                                                                  float* outputTopkWeights, int nRanks, int nTokens,
                                                                  int nTopk, int maxTokensPerRank,
                                                                  int invalidTokenExpertId) {
  const int metadataEntriesPerRank = maxTokensPerRank * nTopk;
  for (int destinationRank = 0; destinationRank < nRanks; ++destinationRank) {
    auto* destinationTopkIdx = reinterpret_cast<int*>(transport.mappedBuffer(outputTopkIdx, destinationRank));
    auto* destinationTopkWeights = reinterpret_cast<float*>(transport.mappedBuffer(outputTopkWeights, destinationRank));
    for (int metadataIdx = nTokens * nTopk + static_cast<int>(threadIdx.x); metadataIdx < metadataEntriesPerRank;
         metadataIdx += static_cast<int>(blockDim.x)) {
      const size_t outputIdx = static_cast<size_t>(transport.rank_) * metadataEntriesPerRank + metadataIdx;
      destinationTopkIdx[outputIdx] = invalidTokenExpertId;
      destinationTopkWeights[outputIdx] = 0.0f;
    }
  }
  __syncthreads();
}

MSCCLPP_DEVICE_INLINE void dispatchRankMajorTopkExpandedNotify(const TransportView& transport, int* outputTopkIdx,
                                                               float* outputTopkWeights, int nExperts, int nRanks,
                                                               const int64_t* __restrict__ topkIndices, int nTokens,
                                                               int nTopk, int maxTokensPerRank,
                                                               int invalidTokenExpertId, void* recvBuffer,
                                                               void* workspace, uint32_t epoch, int* sharedMem) {
  WorkspaceView workspaceView(workspace, nRanks, nExperts);
  auto* rankTokenCounts = sharedMem;
  auto* metadataCompletionCounts = sharedMem + nRanks;
  for (int dstRank = static_cast<int>(threadIdx.x); dstRank < nRanks; dstRank += static_cast<int>(blockDim.x)) {
    // Token payloads are sparse per destination rank, but metadata is dense:
    // each destination receives one [token, top-k] slot for every source route.
    metadataCompletionCounts[dstRank] = nTokens * nTopk;
  }
  countRankMajorTopkExpandedRoutes(rankTokenCounts, topkIndices, nTokens, nTopk, nRanks, nExperts);
  invalidateRankMajorTopkExpandedPadding(transport, outputTopkIdx, outputTopkWeights, nRanks, nTokens, nTopk,
                                         maxTokensPerRank, invalidTokenExpertId);
  writeRankMajorCounts(transport, rankTokenCounts, nRanks, recvBuffer, epoch);
  publishDispatchPayloads(transport, metadataCompletionCounts, nRanks, workspaceView);
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void issueRankMajorTopkExpandedTokenStore(void* output, const TransportView& transport,
                                                                int destinationRank, size_t destIndex,
                                                                void* stagedToken) {
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(Bf16);
  void* destinationBuffer = transport.mappedBuffer(output, destinationRank);
  auto* destinationRow = reinterpret_cast<uint8_t*>(destinationBuffer) + destIndex * HiddenBytes;
  mscclpp::bulkStore(destinationRow, stagedToken, static_cast<uint32_t>(HiddenBytes));
  mscclpp::bulkStoreCommit();
}

MSCCLPP_DEVICE_INLINE void sendRankMajorTopkExpandedMetadata(const TransportView& transport, int* outputTopkIdx,
                                                             float* outputTopkWeights, int destinationRank,
                                                             size_t destIndex, int expert, float weight) {
  auto* destinationTopkIdx = reinterpret_cast<int*>(transport.mappedBuffer(outputTopkIdx, destinationRank));
  auto* destinationTopkWeights = reinterpret_cast<float*>(transport.mappedBuffer(outputTopkWeights, destinationRank));
  destinationTopkIdx[destIndex] = expert;
  destinationTopkWeights[destIndex] = weight;
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void dispatchSendRankMajorTopkExpandedBf16(
    void* output, int* outputTopkIdx, float* outputTopkWeights, const void* inputTokens, int nExperts, int nRanks,
    const int64_t* __restrict__ topkIndices, const float* __restrict__ topkWeights, int nTokens, int nTopk,
    int invalidTokenExpertId, int maxTokensPerRank, bool deduplicateExpandedRoutes, const TransportView& transport,
    void* workspace, int nPayloadBlocks, int* sharedMem) {
  RankMajorSendState<Hidden> sendState;
  if (!initRankMajorSendState(sendState, nTokens, nTopk, nRanks, nPayloadBlocks, sharedMem)) return;

  const int laneId = sendState.laneId_;
  const int nLocalExperts = nExperts / nRanks;
  WorkspaceView workspaceView(workspace, nRanks, nExperts);
  auto* completionCounts = aggregatedDispatchCompletionCounts(sharedMem, nRanks, sendState.warpGroupId_);

  initAggregatedDispatchCompletions(completionCounts, nRanks, laneId);

  for (int tokenIdx = sendState.firstTokenIdx_; tokenIdx < nTokens; tokenIdx += sendState.tokenStride_) {
    stageRankMajorBf16Token(inputTokens, tokenIdx, sendState);
    const int expert = laneId < nTopk ? static_cast<int>(topkIndices[tokenIdx * nTopk + laneId]) : -1;
    const int dstRank = expert >= 0 ? expert / nLocalExperts : -1;
    const float weight =
        laneId < nTopk ? (topkWeights == nullptr ? 1.0f : topkWeights[tokenIdx * nTopk + laneId]) : 0.0f;
    const size_t destIndex = (static_cast<size_t>(transport.rank_) * maxTokensPerRank + tokenIdx) * nTopk + laneId;
    const bool isDestinationLeader = isFirstLaneForRank(dstRank, laneId);

    waitRankMajorBf16Token(sendState);

    if (laneId < nTopk) {
      // Publish invalid metadata to non-destination ranks before the route completion is counted.
      for (int destinationRank = 0; destinationRank < nRanks; ++destinationRank) {
        const bool isLocal = destinationRank == dstRank;
        sendRankMajorTopkExpandedMetadata(transport, outputTopkIdx, outputTopkWeights, destinationRank, destIndex,
                                          isLocal ? expert : invalidTokenExpertId, isLocal ? weight : 0.0f);
      }
    }

    if (dstRank >= 0 && (!deduplicateExpandedRoutes || isDestinationLeader)) {
      issueRankMajorTopkExpandedTokenStore<Hidden>(output, transport, dstRank, destIndex, sendState.stagedToken_);
      mscclpp::bulkStoreWait();
    }
    if (laneId < nTopk) {
      for (int destinationRank = 0; destinationRank < nRanks; ++destinationRank) {
        atomicAdd_block(completionCounts + destinationRank, 1);
      }
    }
    __syncwarp();
  }
  flushAggregatedDispatchCompletions(workspaceView, completionCounts, nRanks, laneId);
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void dispatchSendRankMajorTopkExpanded(
    void* output, int* outputTopkIdx, float* outputTopkWeights, const void* inputTokens, const TransportView& transport,
    int nExperts, int nRanks, const int64_t* __restrict__ topkIndices, const float* __restrict__ topkWeights,
    int nTokens, int nTopk, int invalidTokenExpertId, int maxTokensPerRank, bool deduplicateExpandedRoutes,
    void* recvBuffer, void* workspace, uint32_t epoch, int* sharedMem) {
  const int nWorkerBlocks = static_cast<int>(gridDim.x) - DispatchControlBlocks;
  if (static_cast<int>(blockIdx.x) > 0 && static_cast<int>(blockIdx.x) <= nWorkerBlocks) {
    dispatchSendRankMajorTopkExpandedBf16<Hidden>(output, outputTopkIdx, outputTopkWeights, inputTokens, nExperts,
                                                  nRanks, topkIndices, topkWeights, nTokens, nTopk,
                                                  invalidTokenExpertId, maxTokensPerRank, deduplicateExpandedRoutes,
                                                  transport, workspace, nWorkerBlocks, sharedMem);
  } else if (static_cast<int>(blockIdx.x) == nWorkerBlocks + 1) {
    dispatchRankMajorTopkExpandedNotify(transport, outputTopkIdx, outputTopkWeights, nExperts, nRanks, topkIndices,
                                        nTokens, nTopk, maxTokensPerRank, invalidTokenExpertId, recvBuffer, workspace,
                                        epoch, sharedMem);
  }
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void expandRankMajorTopkDuplicateRoutes(void* output, const int* outputTopkIdx, int tokenIdx,
                                                              int rank, int nExperts, int nRanks, int nTopk,
                                                              int maxTokensPerRank) {
  constexpr int HiddenVectors = Hidden / mscclpp::bf16x8::Size;
  const int nLocalExperts = nExperts / nRanks;
  const int localExpertBegin = nLocalExperts * rank;
  const int localExpertEnd = localExpertBegin + nLocalExperts;
  const int laneId = get_lane_id();
  auto* outputVectors = reinterpret_cast<mscclpp::bf16x8*>(output);

  const size_t routeBase = static_cast<size_t>(tokenIdx) * nTopk;
  int leaderLane = -1;
  for (int topkLane = 0; topkLane < nTopk; ++topkLane) {
    const int expert = outputTopkIdx[routeBase + topkLane];
    if (expert < localExpertBegin || expert >= localExpertEnd) continue;
    if (leaderLane < 0) {
      leaderLane = topkLane;
      continue;
    }

    auto* destination = outputVectors + (routeBase + topkLane) * HiddenVectors;
    const auto* source = outputVectors + (routeBase + leaderLane) * HiddenVectors;
    for (int vectorIdx = laneId; vectorIdx < HiddenVectors; vectorIdx += WARP_SIZE) {
      destination[vectorIdx] = source[vectorIdx];
    }
  }
}

MSCCLPP_DEVICE_INLINE void dispatchRecvRankMajorTopkExpanded(int* outputCount, const TransportView& transport,
                                                             int nExperts, int nRanks, void* recvBuffer,
                                                             void* workspace, uint32_t epoch, int* sharedMem) {
  const int sourceRank = static_cast<int>(blockIdx.x);
  if (sourceRank >= nRanks) return;
  auto* rankTokenCounts = reinterpret_cast<mscclpp::LL8Packet*>(recvBuffer);
  int nStores = 0;
  if (threadIdx.x == 0) {
    nStores = static_cast<int>(rankTokenCounts[sourceRank].read(epoch, -1));
    outputCount[sourceRank] = nStores;
    sharedMem[0] = nStores;
  }
  __syncthreads();
  nStores = sharedMem[0];
  WorkspaceView workspaceView(workspace, nRanks, nExperts);
  if (threadIdx.x == 0) {
    if (transport.isSelf(sourceRank)) {
      workspaceView.dispatchLocalPayloadReady_->acquire();
    } else {
      transport.baseMemoryChannels_[sourceRank].wait(-1);
    }
  }
}

template <DispatchDataType DataType>
MSCCLPP_DEVICE_INLINE void stageDispatchPayloadMetadata(const DispatchPayloadView<DataType>& payloadView,
                                                        void* stagedPayload, int* destinationSlots,
                                                        WorkspaceView& workspaceView,
                                                        const int64_t* __restrict__ topkIndices,
                                                        const float* __restrict__ topkWeights, int tokenIdx, int nTopk,
                                                        int nLocalExperts, int maxTokensPerRank, int rank, int laneId) {
  const int routedExpertIdx = laneId < nTopk ? static_cast<int>(topkIndices[tokenIdx * nTopk + laneId]) : -1;
  const int dstRank = routedExpertIdx >= 0 ? routedExpertIdx / nLocalExperts : -1;
  const bool firstLaneForRank = isFirstLaneForRank(dstRank, laneId);
  if (laneId < nTopk) {
    int destinationSlot = -1;
    if (dstRank >= 0 && firstLaneForRank) {
      destinationSlot = atomicAdd(workspaceView.dispatchRankPayloadSlots_ + dstRank, 1);
      EP_DEVICE_ASSERT(destinationSlot < maxTokensPerRank);
    }
    destinationSlots[laneId] = destinationSlot;
    payloadView.topKIndices(stagedPayload)[laneId] = routedExpertIdx;
    payloadView.topKValues(stagedPayload)[laneId] =
        topkWeights == nullptr ? 1.0f : topkWeights[tokenIdx * nTopk + laneId];
  }
  if (laneId == 0) {
    *payloadView.srcTokenGlobalIdx(stagedPayload) = rank * maxTokensPerRank + tokenIdx;
  }
}

template <DispatchDataType DataType>
MSCCLPP_DEVICE_INLINE void sendStagedDispatchPayload(const DispatchPayloadView<DataType>& payloadView,
                                                     void* stagedPayload, const int* destinationSlots,
                                                     WorkspaceView& workspaceView, int nTopk, int nLocalExperts,
                                                     int maxTokensPerRank, size_t metadataBytes, size_t payloadStride,
                                                     void* recvBuffer, const TransportView& transport, int laneId) {
  const int destinationSlot = laneId < nTopk ? destinationSlots[laneId] : -1;
  if (destinationSlot < 0) return;

  const int dstRank = payloadView.topKIndices(stagedPayload)[laneId] / nLocalExperts;
  void* destinationBuffer = transport.mappedBuffer(recvBuffer, dstRank);
  auto* destinationPayload =
      reinterpret_cast<uint8_t*>(destinationBuffer) + metadataBytes +
      (static_cast<size_t>(transport.rank_) * maxTokensPerRank + destinationSlot) * payloadStride;
  mscclpp::bulkStore(destinationPayload, stagedPayload, static_cast<uint32_t>(payloadView.numBytes_));
  mscclpp::bulkStoreCommit();
  mscclpp::bulkStoreWait();
  (void)mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(workspaceView.dispatchRankPayloadCompletions_ + dstRank, 1,
                                                           mscclpp::memoryOrderRelease);
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void dispatchSendBf16(const void* inputTokens, int nExperts, int rank, int nRanks,
                                            const int64_t* __restrict__ topkIndices,
                                            const float* __restrict__ topkWeights, int nTokens, int nTopk,
                                            int maxTokensPerRank, void* recvBuffer, const TransportView& transport,
                                            void* workspace, int* sharedMem) {
  const int nWorkerBlocks = static_cast<int>(gridDim.x) - DispatchControlBlocks;
  if (blockIdx.x == 0 || static_cast<int>(blockIdx.x) > nWorkerBlocks) return;

  const int warpId = static_cast<int>(threadIdx.x) / WARP_SIZE;
  const int laneId = get_lane_id();
  const int senderBlockIdx = static_cast<int>(blockIdx.x) - 1;
  const int nWarpsPerGroup = dispatchNWarpsPerGroup(nTokens, nWorkerBlocks);
  const int nWarpGroups = DispatchNWarps / nWarpsPerGroup;
  const int warpGroupId = warpId / nWarpsPerGroup;
  const int subWarpId = warpId % nWarpsPerGroup;
  if (subWarpId != 0) return;

  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(Bf16);
  constexpr int HiddenVectors = Hidden / mscclpp::bf16x8::Size;
  const int nLocalExperts = nExperts / nRanks;
  const size_t metadataBytes = dispatchMetadataBytes(nRanks, nExperts);
  const DispatchPayloadView<DispatchDataType::BF16> payloadView(Hidden, nTopk, 0);
  const size_t payloadStride = dispatchPayloadStride<DispatchDataType::BF16>(Hidden, nTopk, 0);
  auto* sharedPayloadBase = reinterpret_cast<uint8_t*>(sharedMem) + dispatchSharedControlBytes(nRanks);
  auto* sendBulkBarriers =
      reinterpret_cast<mscclpp::BulkBarrier*>(sharedPayloadBase + DispatchMaxNWarpGroups * payloadStride);
  auto* stagedPayload = sharedPayloadBase + static_cast<size_t>(warpGroupId) * payloadStride;
  auto* destinationSlots = sharedMem + warpGroupId * WARP_SIZE;
  auto* bulkBarrier = sendBulkBarriers + warpGroupId;
  WorkspaceView workspaceView(workspace, nRanks, nExperts);

  const int tokenStride = nWorkerBlocks * nWarpGroups;
  const int firstTokenIdx = senderBlockIdx * nWarpGroups + warpGroupId;
  uint32_t sendBulkPhase = 0;
  if (firstTokenIdx < nTokens) {
    if (laneId == 0) bulkBarrier->init();
  }

  for (int tokenIdx = firstTokenIdx; tokenIdx < nTokens; tokenIdx += tokenStride) {
    const auto* inputData =
        reinterpret_cast<const mscclpp::bf16x8*>(inputTokens) + static_cast<size_t>(tokenIdx) * HiddenVectors;
    if (laneId == 0) {
      bulkBarrier->arriveAndExpect(static_cast<uint32_t>(HiddenBytes));
      mscclpp::bulkLoad(stagedPayload, inputData, static_cast<uint32_t>(HiddenBytes), *bulkBarrier);
    }
    stageDispatchPayloadMetadata<DispatchDataType::BF16>(payloadView, stagedPayload, destinationSlots, workspaceView,
                                                         topkIndices, topkWeights, tokenIdx, nTopk, nLocalExperts,
                                                         maxTokensPerRank, rank, laneId);
    if (laneId == 0) bulkBarrier->wait(sendBulkPhase);
    __syncwarp();
    mscclpp::bulkFence();
    sendStagedDispatchPayload<DispatchDataType::BF16>(payloadView, stagedPayload, destinationSlots, workspaceView,
                                                      nTopk, nLocalExperts, maxTokensPerRank, metadataBytes,
                                                      payloadStride, recvBuffer, transport, laneId);
    __syncwarp();
  }
}

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
MSCCLPP_DEVICE_INLINE void dispatchSendFp8(const void* inputTokens, int nExperts, int rank, int nRanks,
                                           const int64_t* __restrict__ topkIndices,
                                           const float* __restrict__ topkWeights, int nTokens, int nTopk,
                                           int maxTokensPerRank, void* recvBuffer, const TransportView& transport,
                                           void* workspace, int* sharedMem) {
  static_assert(DataType == DispatchDataType::FP8_E4M3);
  const int nWorkerBlocks = static_cast<int>(gridDim.x) - DispatchControlBlocks;
  if (blockIdx.x == 0 || static_cast<int>(blockIdx.x) > nWorkerBlocks) return;

  const int warpId = static_cast<int>(threadIdx.x) / WARP_SIZE;
  const int laneId = get_lane_id();
  const int senderBlockIdx = static_cast<int>(blockIdx.x) - 1;
  const int nWarpsPerGroup = dispatchNWarpsPerGroup(nTokens, nWorkerBlocks);
  const int nWarpGroups = DispatchNWarps / nWarpsPerGroup;
  const int warpGroupId = warpId / nWarpsPerGroup;
  const int subWarpId = warpId % nWarpsPerGroup;
  const int groupThreadId = subWarpId * WARP_SIZE + laneId;
  const int groupThreadCount = nWarpsPerGroup * WARP_SIZE;
  const int groupBarrierId = DispatchWarpGroupBarrierBase + warpGroupId;

  constexpr int HiddenVectors = Hidden / mscclpp::bf16x8::Size;
  const int nLocalExperts = nExperts / nRanks;
  const size_t metadataBytes = dispatchMetadataBytes(nRanks, nExperts);
  const DispatchPayloadView<DataType> payloadView(Hidden, nTopk, ScaleBlockSize);
  const size_t payloadStride = dispatchPayloadStride<DataType>(Hidden, nTopk, ScaleBlockSize);
  auto* sharedPayloadBase = reinterpret_cast<uint8_t*>(sharedMem) + dispatchSharedControlBytes(nRanks);
  auto* stagedPayload = sharedPayloadBase + static_cast<size_t>(warpGroupId) * payloadStride;
  auto* destinationSlots = sharedMem + warpGroupId * WARP_SIZE;
  auto* outputData = payloadView.template data<mscclpp::f8_e4m3x8>(stagedPayload);
  auto* outputScales = payloadView.scaleFactors(stagedPayload);
  WorkspaceView workspaceView(workspace, nRanks, nExperts);

  const int tokenStride = nWorkerBlocks * nWarpGroups;
  const int firstTokenIdx = senderBlockIdx * nWarpGroups + warpGroupId;
  for (int tokenIdx = firstTokenIdx; tokenIdx < nTokens; tokenIdx += tokenStride) {
    const auto* inputData =
        reinterpret_cast<const mscclpp::bf16x8*>(inputTokens) + static_cast<size_t>(tokenIdx) * HiddenVectors;
    if (subWarpId == 0) {
      stageDispatchPayloadMetadata<DataType>(payloadView, stagedPayload, destinationSlots, workspaceView, topkIndices,
                                             topkWeights, tokenIdx, nTopk, nLocalExperts, maxTokensPerRank, rank,
                                             laneId);
    }
    for (int inputIdx = groupThreadId; inputIdx < HiddenVectors; inputIdx += groupThreadCount) {
      outputData[inputIdx] = quantizeBf16x8ToFp8E4M3<ScaleBlockSize>(
          inputData[inputIdx], outputScales + inputIdx * mscclpp::bf16x8::Size / ScaleBlockSize, laneId);
    }
    syncNamedBarrier(groupBarrierId, groupThreadCount);

    if (subWarpId == 0) {
      mscclpp::bulkFence();
      sendStagedDispatchPayload<DataType>(payloadView, stagedPayload, destinationSlots, workspaceView, nTopk,
                                          nLocalExperts, maxTokensPerRank, metadataBytes, payloadStride, recvBuffer,
                                          transport, laneId);
    }
    syncNamedBarrier(groupBarrierId, groupThreadCount);
  }
}

struct DispatchCountView {
  int* rankTokenCounts_;
  int* expertTokenCounts_;

  MSCCLPP_DEVICE_INLINE DispatchCountView(int* sharedMem, int nRanks)
      : rankTokenCounts_(sharedMem), expertTokenCounts_(sharedMem + nRanks) {}
};

MSCCLPP_DEVICE_INLINE void countDispatchRoutes(DispatchCountView counts, const int64_t* __restrict__ topkIndices,
                                               int nTokens, int nTopk, int nRanks, int nExperts) {
  const int threadId = static_cast<int>(threadIdx.x);
  const int warpId = threadId / WARP_SIZE;
  const int laneId = get_lane_id();
  const int nLocalExperts = nExperts / nRanks;
  for (int rankIdx = threadId; rankIdx < nRanks; rankIdx += blockDim.x) counts.rankTokenCounts_[rankIdx] = 0;
  for (int expertIdx = threadId; expertIdx < nExperts; expertIdx += blockDim.x)
    counts.expertTokenCounts_[expertIdx] = 0;
  __syncthreads();
  for (int tokenIdx = warpId; tokenIdx < nTokens; tokenIdx += DispatchNWarps) {
    const int routedExpertIdx = laneId < nTopk ? static_cast<int>(topkIndices[tokenIdx * nTopk + laneId]) : -1;
    const int dstRank = routedExpertIdx >= 0 ? routedExpertIdx / nLocalExperts : -1;
    if (routedExpertIdx >= 0) atomicAdd_block(counts.expertTokenCounts_ + routedExpertIdx, 1);
    if (isFirstLaneForRank(dstRank, laneId) && dstRank >= 0) {
      atomicAdd_block(counts.rankTokenCounts_ + dstRank, 1);
    }
  }
  __syncthreads();
}

MSCCLPP_DEVICE_INLINE void countRankMajorRoutes(int* rankTokenCounts, const int64_t* __restrict__ topkIndices,
                                                int nTokens, int nTopk, int nRanks, int nExperts) {
  const int threadId = static_cast<int>(threadIdx.x);
  const int warpId = threadId / WARP_SIZE;
  const int laneId = get_lane_id();
  const int nLocalExperts = nExperts / nRanks;
  for (int rankIdx = threadId; rankIdx < nRanks; rankIdx += blockDim.x) rankTokenCounts[rankIdx] = 0;
  __syncthreads();
  for (int tokenIdx = warpId; tokenIdx < nTokens; tokenIdx += DispatchNWarps) {
    const int routedExpertIdx = laneId < nTopk ? static_cast<int>(topkIndices[tokenIdx * nTopk + laneId]) : -1;
    const int dstRank = routedExpertIdx >= 0 ? routedExpertIdx / nLocalExperts : -1;
    if (isFirstLaneForRank(dstRank, laneId) && dstRank >= 0) {
      atomicAdd_block(rankTokenCounts + dstRank, 1);
    }
  }
  __syncthreads();
}

MSCCLPP_DEVICE_INLINE void writeDispatchMetadata(const TransportView& transport, DispatchCountView counts, int nRanks,
                                                 int nExperts, void* recvBuffer, uint32_t epoch) {
  const int threadId = static_cast<int>(threadIdx.x);
  const int nLocalExperts = nExperts / nRanks;
  for (int dstRank = threadId; dstRank < nRanks; dstRank += blockDim.x) {
    auto* destinationPackets = reinterpret_cast<mscclpp::LL8Packet*>(transport.mappedBuffer(recvBuffer, dstRank));
    destinationPackets[transport.rank_].write(static_cast<uint32_t>(counts.rankTokenCounts_[dstRank]), epoch);
  }
  for (int expertIdx = threadId; expertIdx < nExperts; expertIdx += blockDim.x) {
    const int dstRank = expertIdx / nLocalExperts;
    const int localExpertIdx = expertIdx % nLocalExperts;
    auto* destinationPackets = reinterpret_cast<mscclpp::LL8Packet*>(transport.mappedBuffer(recvBuffer, dstRank));
    destinationPackets[nRanks + transport.rank_ * nLocalExperts + localExpertIdx].write(
        static_cast<uint32_t>(counts.expertTokenCounts_[expertIdx]), epoch);
  }
}

MSCCLPP_DEVICE_INLINE void writeRankMajorCounts(const TransportView& transport, const int* rankTokenCounts, int nRanks,
                                                void* recvBuffer, uint32_t epoch, bool skipGpuNetIo) {
  const int threadId = static_cast<int>(threadIdx.x);
  for (int dstRank = threadId; dstRank < nRanks; dstRank += blockDim.x) {
#if defined(MSCCLPP_USE_GPUNETIO)
    // Keep count scratch outside the token/combine staging ring. The rank-major
    // metadata allocation reserves [R,2R) for received counts and [2R,3R) for
    // these registered send packets, including small expert counts.
    if (transport.gpuNetIo_ != nullptr && !transport.isNvlinkPeer(dstRank)) {
      if (skipGpuNetIo) continue;  // Dedicated posters send the count on their exclusively owned QP.
      auto* gin = transport.gpuNetIo_;
      auto* scratch = reinterpret_cast<mscclpp::LL8Packet*>(recvBuffer) + 2 * nRanks + dstRank;
      scratch->write(static_cast<uint32_t>(rankTokenCounts[dstRank]), epoch);
      __threadfence_system();
      auto* remotePacket = reinterpret_cast<mscclpp::LL8Packet*>(recvBuffer) + nRanks + transport.rank_;
      gin->put(dstRank, transport.symmetricOffset(remotePacket), transport.symmetricOffset(scratch),
               sizeof(mscclpp::LL8Packet), static_cast<int>(blockIdx.x) % gin->numQpsPerPeer);
      continue;
    }
#endif  // defined(MSCCLPP_USE_GPUNETIO)
    auto* destinationPackets = reinterpret_cast<mscclpp::LL8Packet*>(transport.mappedBuffer(recvBuffer, dstRank));
    destinationPackets[nRanks + transport.rank_].write(static_cast<uint32_t>(rankTokenCounts[dstRank]), epoch);
  }
}

MSCCLPP_DEVICE_INLINE void publishDispatchPayloads(const TransportView& transport, const int* rankTokenCounts,
                                                   int nRanks, WorkspaceView workspaceView) {
  const int threadId = static_cast<int>(threadIdx.x);
  for (int dstRank = threadId; dstRank < nRanks; dstRank += blockDim.x) {
    const int expectedPayloadCount = rankTokenCounts[dstRank];
#if defined(MSCCLPP_USE_GPUNETIO)
    // Cross-domain producers can still be allocating slots while notify runs.
    // Their fused signals bypass the local completion counter; reset their slots
    // only in the stream-ordered pre-dispatch memset, never concurrently here.
    if (transport.gpuNetIo_ != nullptr && !transport.isNvlinkPeer(dstRank)) {
      continue;
    }
#endif  // defined(MSCCLPP_USE_GPUNETIO)
    if (expectedPayloadCount > 0) {
      while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(workspaceView.dispatchRankPayloadCompletions_ + dstRank,
                                                            mscclpp::memoryOrderAcquire) != expectedPayloadCount);
    }
    workspaceView.dispatchRankPayloadSlots_[dstRank] = 0;
    workspaceView.dispatchRankPayloadCompletions_[dstRank] = 0;
    if (expectedPayloadCount == 0) continue;
    if (transport.isSelf(dstRank)) {
      workspaceView.dispatchLocalPayloadReady_->release();
    } else {
      transport.baseMemoryChannels_[dstRank].signal();
    }
  }
}

MSCCLPP_DEVICE_INLINE void dispatchNotify(const TransportView& transport, int nExperts, int nRanks,
                                          const int64_t* __restrict__ topkIndices, int nTokens, int nTopk,
                                          void* recvBuffer, void* workspace, uint32_t epoch, int* sharedMem) {
  WorkspaceView workspaceView(workspace, nRanks, nExperts);
  DispatchCountView counts(sharedMem, nRanks);
  countDispatchRoutes(counts, topkIndices, nTokens, nTopk, nRanks, nExperts);
  writeDispatchMetadata(transport, counts, nRanks, nExperts, recvBuffer, epoch);
  publishDispatchPayloads(transport, counts.rankTokenCounts_, nRanks, workspaceView);
}

#if defined(MSCCLPP_USE_GPUNETIO)
// Early tail post (control CTA, once per-destination row counts are known): publish each count for chunk-end
// clipping and pad the partial tail chunk's completion counter, so the producer staging its last row posts the tail
// in the same wave as the full chunks. If every tail row is already staged, post it here.
template <int Hidden>
MSCCLPP_DEVICE_INLINE void releaseRankMajorGpuNetIoTails(const TransportView& transport, void* output,
                                                         int* outputTopkIdx, float* outputTopkWeights,
                                                         const int* rankTokenCounts, int nRanks, int maxTokensPerRank,
                                                         int nTopk) {
  const int chunkRows = transport.gpuNetIoPipelineChunkRows_;
  const int chunksPerRank = configCellDiv(maxTokensPerRank, chunkRows);
  const int queues = transport.gpuNetIoBatchAllQps_ ? transport.gpuNetIo_->numQpsPerPeer : transport.gpuNetIo_->numHcas;
  for (int peer = threadIdx.x; peer < nRanks; peer += blockDim.x) {
    if (transport.isNvlinkPeer(peer)) continue;
    const int rows = rankTokenCounts[peer];
    mscclpp::atomicStore<int, mscclpp::scopeDevice>(transport.gpuNetIoTailRows_ + peer, rows + 1,
                                                    mscclpp::memoryOrderRelease);
    const int tail = rows % chunkRows;
    if (tail == 0) continue;
    const int chunk = rows / chunkRows;
    const int pad = chunkRows - tail;
    const int previous = mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(
        transport.gpuNetIoChunkReady_ + peer * chunksPerRank + chunk, pad, mscclpp::memoryOrderAcqRel);
    if (previous + pad == chunkRows)
      postRankMajorGpuNetIoRows<Hidden>(transport, output, outputTopkIdx, outputTopkWeights, nRanks, maxTokensPerRank,
                                        nTopk, peer, chunk * chunkRows, rows, chunk % queues, TimingKind::DispatchChunk,
                                        1);
  }
}
#endif

template <bool AsyncDispatch = false, int Hidden = 0>
MSCCLPP_DEVICE_INLINE void dispatchRankMajorNotify(const TransportView& transport, int nExperts, int nRanks,
                                                   const int64_t* __restrict__ topkIndices, int nTokens, int nTopk,
                                                   void* recvBuffer, void* workspace, uint32_t epoch, int* sharedMem,
                                                   void* output = nullptr, int* outputTopkIdx = nullptr,
                                                   float* outputTopkWeights = nullptr, int maxTokensPerRank = 0) {
  WorkspaceView workspaceView(workspace, nRanks, nExperts);
  auto* rankTokenCounts = sharedMem;
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  const long long countStart = timingNow();
#endif
  countRankMajorRoutes(rankTokenCounts, topkIndices, nTokens, nTopk, nRanks, nExperts);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  const long long countDone = timingNow();
#endif
#if defined(MSCCLPP_USE_GPUNETIO)
  if constexpr (AsyncDispatch) {
    for (int peer = threadIdx.x; peer < nRanks; peer += blockDim.x) {
      if (transport.isNvlinkPeer(peer)) continue;
      mscclpp::atomicStore<int, mscclpp::scopeDevice>(&transport.gpuNetIoDispatchPeers_[peer].expectedRowsPlusOne,
                                                      rankTokenCounts[peer] + 1, mscclpp::memoryOrderRelease);
      if (rankTokenCounts[peer] == 0) markRankMajorGpuNetIoPeer(transport, peer);
    }
  }
  if constexpr (!AsyncDispatch && Hidden > 0) {
    if (transport.gpuNetIoTailRows_ != nullptr) {
      __syncthreads();
      releaseRankMajorGpuNetIoTails<Hidden>(transport, output, outputTopkIdx, outputTopkWeights, rankTokenCounts,
                                            nRanks, maxTokensPerRank, nTopk);
    }
  }
#endif
  writeRankMajorCounts(transport, rankTokenCounts, nRanks, recvBuffer, epoch);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  const long long writeDone = timingNow();
#endif
  publishDispatchPayloads(transport, rankTokenCounts, nRanks, workspaceView);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  if (threadIdx.x < static_cast<unsigned int>(nRanks))
    transport.timing_.record(TimingKind::DispatchNotify, countStart, timingNow(), threadIdx.x, -1,
                             countDone - countStart, writeDone - countDone);
#endif
}

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
MSCCLPP_DEVICE_INLINE void dispatchSend(const void* inputTokens, const TransportView& transport, int nExperts,
                                        int nRanks, const int64_t* __restrict__ topkIndices,
                                        const float* __restrict__ topkWeights, int nTokens, int nTopk,
                                        int maxTokensPerRank, void* recvBuffer, void* workspace, uint32_t epoch,
                                        int* sharedMem) {
  const int nWorkerBlocks = static_cast<int>(gridDim.x) - DispatchControlBlocks;
  if (static_cast<int>(blockIdx.x) > 0 && static_cast<int>(blockIdx.x) <= nWorkerBlocks) {
    if constexpr (DataType == DispatchDataType::BF16) {
      dispatchSendBf16<Hidden>(inputTokens, nExperts, transport.rank_, nRanks, topkIndices, topkWeights, nTokens, nTopk,
                               maxTokensPerRank, recvBuffer, transport, workspace, sharedMem);
    } else {
      dispatchSendFp8<Hidden, DataType, ScaleBlockSize>(inputTokens, nExperts, transport.rank_, nRanks, topkIndices,
                                                        topkWeights, nTokens, nTopk, maxTokensPerRank, recvBuffer,
                                                        transport, workspace, sharedMem);
    }
  } else if (static_cast<int>(blockIdx.x) == nWorkerBlocks + 1) {
    dispatchNotify(transport, nExperts, nRanks, topkIndices, nTokens, nTopk, recvBuffer, workspace, epoch, sharedMem);
  }
}

#if defined(MSCCLPP_USE_GPUNETIO)
// Dedicated posters: lane 0 of each poster warp owns a set of (remote peer, QP) pairs and is the only thread
// posting to those QPs until the producer join, so it can use single-poster submits. Each pair covers a
// contiguous stripe of the peer's rows and posts every contiguous run of ready rows (up to the chunk cap)
// as soon as it appears, so the first post no longer waits for a full chunk.
template <int Hidden>
MSCCLPP_DEVICE_INLINE void runRankMajorGpuNetIoPosters(const TransportView& transport, void* output,
                                                       int* outputTopkIdx, float* outputTopkWeights,
                                                       const int* rankTokenCounts, int nRanks, int capacity, int nTopk,
                                                       void* recvBuffer, uint32_t epoch, int posterId, int nPosters) {
  constexpr int MaxItems = 32;
  constexpr int MaxRowsPerWrite =
      DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE / (static_cast<size_t>(Hidden) * sizeof(Bf16));
  auto* gin = transport.gpuNetIo_;
  const int nQp = gin->numQpsPerPeer;
  int maxRows = transport.gpuNetIoPipelineChunkRows_;
  if (maxRows > MaxRowsPerWrite) maxRows = MaxRowsPerWrite;
  int peers[MaxItems], queues[MaxItems], frontiers[MaxItems], ends[MaxItems];
  int nItems = 0;
  for (int item = posterId; item < nRanks * nQp; item += nPosters) {
    const int peer = item / nQp, queue = item % nQp;
    if (transport.isNvlinkPeer(peer)) continue;
    if (nItems == MaxItems) __trap();
    const int rows = rankTokenCounts[peer];
    peers[nItems] = peer;
    queues[nItems] = queue;
    frontiers[nItems] = static_cast<int>(static_cast<int64_t>(rows) * queue / nQp);
    ends[nItems] = static_cast<int>(static_cast<int64_t>(rows) * (queue + 1) / nQp);
    ++nItems;
    if (queue == 0) {
      auto* scratch = reinterpret_cast<mscclpp::LL8Packet*>(recvBuffer) + 2 * nRanks + peer;
      scratch->write(static_cast<uint32_t>(rows), epoch);
      __threadfence_system();
      auto* remotePacket = reinterpret_cast<mscclpp::LL8Packet*>(recvBuffer) + nRanks + transport.rank_;
      gin->putExclusive(peer, queue, transport.symmetricOffset(remotePacket), transport.symmetricOffset(scratch),
                        sizeof(mscclpp::LL8Packet));
    }
  }
  for (bool pending = true; pending;) {
    pending = false;
    for (int item = 0; item < nItems; ++item) {
      int& frontier = frontiers[item];
      const int end = ends[item];
      if (frontier >= end) continue;
      int* ready = transport.gpuNetIoChunkReady_ + static_cast<size_t>(peers[item]) * capacity;
      int rows = 0;
      while (rows < maxRows && frontier + rows < end &&
             mscclpp::atomicLoad<int, mscclpp::scopeDevice>(ready + frontier + rows, mscclpp::memoryOrderAcquire) != 0)
        ++rows;
      if (rows > 0) {
        postRankMajorGpuNetIoRows<Hidden, true>(transport, output, outputTopkIdx, outputTopkWeights, nRanks, capacity,
                                                nTopk, peers[item], frontier, frontier + rows, queues[item],
                                                TimingKind::DispatchChunk, rows);
        frontier += rows;
      }
      if (frontier < end) pending = true;
    }
  }
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void dispatchRankMajorNotifyWithPosters(const TransportView& transport, void* output,
                                                              int* outputTopkIdx, float* outputTopkWeights,
                                                              int nExperts, int nRanks,
                                                              const int64_t* __restrict__ topkIndices, int nTokens,
                                                              int nTopk, int maxTokensPerRank, void* recvBuffer,
                                                              void* workspace, uint32_t epoch, int* sharedMem) {
  WorkspaceView workspaceView(workspace, nRanks, nExperts);
  auto* rankTokenCounts = sharedMem;
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  const long long countStart = timingNow();
#endif
  countRankMajorRoutes(rankTokenCounts, topkIndices, nTokens, nTopk, nRanks, nExperts);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  const long long countDone = timingNow();
#endif
  const int warpId = static_cast<int>(threadIdx.x) / WARP_SIZE;
  const int nPublishWarps = (nRanks + WARP_SIZE - 1) / WARP_SIZE;
  const int nWarps = static_cast<int>(blockDim.x) / WARP_SIZE;
  // Publish warps spin on NVLink completions, so posters get warps of their own.
  if (nPublishWarps >= nWarps) __trap();
  if (warpId < nPublishWarps) {
    writeRankMajorCounts(transport, rankTokenCounts, nRanks, recvBuffer, epoch, true);
    publishDispatchPayloads(transport, rankTokenCounts, nRanks, workspaceView);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
    if (threadIdx.x < static_cast<unsigned int>(nRanks))
      transport.timing_.record(TimingKind::DispatchNotify, countStart, timingNow(), threadIdx.x, -1,
                               countDone - countStart, 0);
#endif
  } else if (get_lane_id() == 0) {
    runRankMajorGpuNetIoPosters<Hidden>(transport, output, outputTopkIdx, outputTopkWeights, rankTokenCounts, nRanks,
                                        maxTokensPerRank, nTopk, recvBuffer, epoch, warpId - nPublishWarps,
                                        nWarps - nPublishWarps);
  }
}
#endif  // defined(MSCCLPP_USE_GPUNETIO)

template <int Hidden, bool AsyncDispatch = false>
MSCCLPP_DEVICE_INLINE void dispatchSendRankMajor(void* output, int* outputTopkIdx, float* outputTopkWeights,
                                                 const void* inputTokens, const TransportView& transport, int nExperts,
                                                 int nRanks, const int64_t* __restrict__ topkIndices,
                                                 const float* __restrict__ topkWeights, int nTokens, int nTopk,
                                                 int invalidTokenExpertId, int maxTokensPerRank, void* recvBuffer,
                                                 void* workspace, uint32_t epoch, int* sharedMem,
                                                 uint64_t* usedQpMask) {
  const int firstProducer = AsyncDispatch ? nRanks : 1;
  const int controlBlock = static_cast<int>(gridDim.x) - 1;
  const int nWorkerBlocks = controlBlock - firstProducer;
#if defined(MSCCLPP_EP_GPUNETIO_TIMING) && defined(MSCCLPP_USE_GPUNETIO)
  // CQTRACE: block 0 is idle during the send phase; one thread per remote (peer, QP) records the GPU time at which
  // each WQE's CQE becomes visible (non-consuming poll). Record: DispatchQueue, v0=3, v1=WQE index, v2=status.
  if (!AsyncDispatch && blockIdx.x == 0 && transport.gpuNetIo_ != nullptr && transport.timing_.buffer != nullptr) {
    auto* gin = transport.gpuNetIo_;
    const int nQp = gin->numQpsPerPeer;
    const int index = static_cast<int>(threadIdx.x);
    const bool isPoller = index < nRanks * nQp && !transport.isNvlinkPeer(index / nQp);
    uint64_t next = 0;
    if (isPoller)
      next = doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
          &mscclpp::detail::ginQp(gin->qps, index)->sq_rsvd_index);
    __syncthreads();
    // GPUBURST (TMAST=3): rank 0 only; thread 64 + p*nQp + q posts EP-shaped chunks (57600 B data + 2 x 128 B meta,
    // 3 signaled WQEs, one doorbell per chunk) for rack-1 peer 4+p, chunks c = q, q+nQp, ... (chunk c -> QP c % nQp).
    if (transport.gpuNetIoTmaStage_ >= 3 && transport.rank_ == 0 && nRanks == 8 && index >= 64 && index < 64 + 4 * nQp) {
      constexpr int kChunks[4] = {20, 24, 22, 21};
      const int p = (index - 64) / nQp, q = (index - 64) % nQp, peer = 4 + p;
      if (transport.gpuNetIoTmaStage_ == 7) {
        // TMAST=7: latency probe, one signaled 128 B RDMA write per QP, one doorbell.
        doca_gpu_dev_verbs_qp* qp = mscclpp::detail::ginQp(gin->qps, peer * nQp + q);
        const uint32_t rkey = mscclpp::detail::ginRemoteKey(*gin, peer, q);
        const __be32 lkey = mscclpp::detail::ginHtobe32(mscclpp::detail::ginLocalKey(*gin, q));
        const uint64_t off = transport.symmetricOffset(reinterpret_cast<uint8_t*>(outputTopkIdx)) + static_cast<uint64_t>(q) * 128;
        const uint64_t t0 = timingNow();
        const uint64_t base = doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
            qp, 1, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT);
        doca_gpu_dev_verbs_wqe_prepare_write(qp, doca_gpu_dev_verbs_get_wqe_ptr(qp, base), base,
                                             DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                                             DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0, gin->peerBase[peer] + off, rkey,
                                             gin->localBase + off, lkey, 128);
        doca_gpu_dev_verbs_mark_wqes_ready<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(qp, base, base);
        auto* submitState = gin->submitStates == nullptr ? nullptr : gin->submitStates + static_cast<size_t>(peer) * nQp + q;
        mscclpp::detail::ginSubmitBatched(qp, base + 1, submitState);
        const uint64_t t1 = timingNow();
        transport.timing_.record(TimingKind::DispatchChunk, t0, t1, peer, q, 0, 0, 0, 0, q, 4, 128, base - 2);
      } else if (transport.gpuNetIoTmaStage_ >= 4) {
        // TMAST=5: the same chain posted twice back to back (2x data, same addresses) in one doorbell per QP.
        const int reps = transport.gpuNetIoTmaStage_ >= 5 ? transport.gpuNetIoTmaStage_ - 3 : 1;  // 5 -> 2x, 6 -> 3x
        // TMAST=4: whole per-QP chain in one reservation and one doorbell (like ibv_post_send of a WR chain).
        doca_gpu_dev_verbs_qp* qp = mscclpp::detail::ginQp(gin->qps, peer * nQp + q);
        const int n1 = (kChunks[p] - q + nQp - 1) / nQp, n = n1 * reps;
        const uint32_t rkey = mscclpp::detail::ginRemoteKey(*gin, peer, q);
        const __be32 lkey = mscclpp::detail::ginHtobe32(mscclpp::detail::ginLocalKey(*gin, q));
        const uint64_t t0 = timingNow();
        const uint64_t base = doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
            qp, 3 * n, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT);
        for (int k = 0; k < n; ++k) {
          const int c = q + (k % n1) * nQp;
          const uint64_t off[3] = {transport.symmetricOffset(reinterpret_cast<uint8_t*>(output)) + static_cast<uint64_t>(c) * 57344,
                                   transport.symmetricOffset(reinterpret_cast<uint8_t*>(outputTopkIdx)) + static_cast<uint64_t>(c) * 128,
                                   transport.symmetricOffset(reinterpret_cast<uint8_t*>(outputTopkWeights)) + static_cast<uint64_t>(c) * 128};
          const uint32_t sz[3] = {57344, 128, 128};
          for (int j = 0; j < 3; ++j) {
            const uint64_t w = base + 3 * k + j;
            doca_gpu_dev_verbs_wqe_prepare_write(qp, doca_gpu_dev_verbs_get_wqe_ptr(qp, w), w,
                                                 DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                                                 DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0, gin->peerBase[peer] + off[j],
                                                 rkey, gin->localBase + off[j], lkey, sz[j]);
          }
        }
        doca_gpu_dev_verbs_mark_wqes_ready<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(qp, base, base + 3 * n - 1);
        auto* submitState = gin->submitStates == nullptr ? nullptr : gin->submitStates + static_cast<size_t>(peer) * nQp + q;
        mscclpp::detail::ginSubmitBatched(qp, base + 3 * n, submitState);
        const uint64_t t1 = timingNow();
        for (int k = 0; k < n; ++k)
          transport.timing_.record(TimingKind::DispatchChunk, t0, t1, peer, q, 0, 0, 0, 0, q + (k % n1) * nQp, 4, 57600, base + 3 * k);
      } else
      for (int c = q; c < kChunks[p]; c += nQp) {
        const uint64_t dataOff = transport.symmetricOffset(reinterpret_cast<uint8_t*>(output)) + static_cast<uint64_t>(c) * 57344;
        const uint64_t metaOff = transport.symmetricOffset(reinterpret_cast<uint8_t*>(outputTopkIdx)) + static_cast<uint64_t>(c) * 128;
        const uint64_t wOff = transport.symmetricOffset(reinterpret_cast<uint8_t*>(outputTopkWeights)) + static_cast<uint64_t>(c) * 128;
        uint64_t ph[6] = {};
        gin->putBatched3(peer, q, dataOff, dataOff, 57344, metaOff, metaOff, 128, wOff, wOff, 128, ph, nullptr, nullptr);
        transport.timing_.record(TimingKind::DispatchChunk, ph[0], ph[4], peer, q, ph[1] - ph[0], ph[2] - ph[1],
                                 ph[3] - ph[2], ph[4] - ph[3], c, 4, 57600, ph[5]);
      }
    }
    if (isPoller) {
      const int peer = index / nQp, queue = index % nQp;
      doca_gpu_dev_verbs_qp* qp = mscclpp::detail::ginQp(gin->qps, peer * nQp + queue);
      doca_gpu_dev_verbs_cq* cq = doca_gpu_dev_verbs_qp_get_cq_sq(qp);
      const uint64_t startNs = timingNow();
      uint64_t lastNs = startNs;
      bool seen = false;
      for (;;) {
        const uint64_t reserved = doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
            &qp->sq_rsvd_index);
        const uint64_t now = timingNow();
        if (next < reserved) {
          const int status = doca_priv_gpu_dev_verbs_poll_one_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(cq, next);
          if (status != EBUSY) {
            transport.timing_.record(TimingKind::DispatchQueue, now, now, peer, queue, 3, next,
                                     static_cast<uint64_t>(status), reserved);
            ++next;
            lastNs = now;
            seen = true;
          }
        } else if ((seen && now - lastNs > 10000) || now - startNs > 2000000) {
          break;
        }
      }
    }
  }
#endif
  if (static_cast<int>(blockIdx.x) >= firstProducer && static_cast<int>(blockIdx.x) < controlBlock) {
    dispatchSendRankMajorBf16<Hidden, AsyncDispatch>(
        output, outputTopkIdx, outputTopkWeights, inputTokens, nExperts, nRanks, topkIndices, topkWeights, nTokens,
        nTopk, invalidTokenExpertId, maxTokensPerRank, transport, workspace, nWorkerBlocks, sharedMem, usedQpMask);
  } else if (static_cast<int>(blockIdx.x) == controlBlock) {
#if defined(MSCCLPP_USE_GPUNETIO)
    if (!AsyncDispatch && transport.gpuNetIo_ != nullptr && transport.gpuNetIoDedicatedPoster_) {
      dispatchRankMajorNotifyWithPosters<Hidden>(transport, output, outputTopkIdx, outputTopkWeights, nExperts, nRanks,
                                                 topkIndices, nTokens, nTopk, maxTokensPerRank, recvBuffer, workspace,
                                                 epoch, sharedMem);
      return;
    }
#endif
    dispatchRankMajorNotify<AsyncDispatch, Hidden>(transport, nExperts, nRanks, topkIndices, nTokens, nTopk,
                                                   recvBuffer, workspace, epoch, sharedMem, output, outputTopkIdx,
                                                   outputTopkWeights, maxTokensPerRank);
  }
}

MSCCLPP_DEVICE_INLINE int proportionalTaskBoundary(int nTokens, int nTasks, int nTotalTokens) {
  return nTotalTokens == 0 ? 0 : static_cast<int>(static_cast<int64_t>(nTokens) * nTasks / nTotalTokens);
}

MSCCLPP_DEVICE_INLINE void dispatchRecvScheduler(int64_t* outputLayout, int* outputCount,
                                                 const TransportView& transport, int nExperts, int nRanks,
                                                 void* recvBuffer, void* workspace, uint32_t epoch, int* sharedMem) {
  const int threadId = static_cast<int>(threadIdx.x);
  const int warpId = threadId / WARP_SIZE;
  const int laneId = get_lane_id();
  const int nWorkerBlocks = static_cast<int>(gridDim.x) - DispatchControlBlocks;
  auto* rankTokenCounts = reinterpret_cast<mscclpp::LL8Packet*>(recvBuffer);
  const int nLocalExperts = nExperts / nRanks;
  WorkspaceView workspaceView(workspace, nRanks, nExperts);

  const int nRankWarps = (nRanks + WARP_SIZE - 1) / WARP_SIZE;
  const int requestedNLayoutWarps = (nLocalExperts + WARP_SIZE - 1) / WARP_SIZE;
  const int maxNLayoutWarps = DispatchNWarps - nRankWarps;
  const int nLayoutWarps = requestedNLayoutWarps < maxNLayoutWarps ? requestedNLayoutWarps : maxNLayoutWarps;

  if (warpId < nRankWarps) {
    const int sourceRank = threadId;
    const int nRankTokens = sourceRank < nRanks ? static_cast<int>(rankTokenCounts[sourceRank].read(epoch, -1)) : 0;
    const int activeRank = nRankTokens > 0 ? 1 : 0;
    int rankTokenPrefix = warpInclusiveSum(nRankTokens, laneId);
    int activeRankPrefix = warpInclusiveSum(activeRank, laneId);
    if (laneId == WARP_SIZE - 1) {
      sharedMem[warpId] = rankTokenPrefix;
      sharedMem[nRankWarps + warpId] = activeRankPrefix;
    }
    syncNamedBarrier(DispatchSchedulerPrefixBarrier, nRankWarps * WARP_SIZE);

    if (warpId == 0) {
      const int tokenTotal = laneId < nRankWarps ? sharedMem[laneId] : 0;
      const int activeTotal = laneId < nRankWarps ? sharedMem[nRankWarps + laneId] : 0;
      const int tokenPrefix = warpInclusiveSum(tokenTotal, laneId);
      const int activePrefix = warpInclusiveSum(activeTotal, laneId);
      if (laneId < nRankWarps) {
        sharedMem[laneId] = tokenPrefix - tokenTotal;
        sharedMem[nRankWarps + laneId] = activePrefix - activeTotal;
      }
      if (laneId == nRankWarps - 1) {
        sharedMem[2 * nRankWarps] = tokenPrefix;
        sharedMem[2 * nRankWarps + 1] = activePrefix;
      }
    }
    syncNamedBarrier(DispatchSchedulerPrefixBarrier, nRankWarps * WARP_SIZE);

    rankTokenPrefix += sharedMem[warpId];
    activeRankPrefix += sharedMem[nRankWarps + warpId];
    const int nTotalTokens = sharedMem[2 * nRankWarps];
    const int nActiveRanks = sharedMem[2 * nRankWarps + 1];
    const int nTasks = nTotalTokens < nWorkerBlocks ? nTotalTokens : nWorkerBlocks;

    // Reserve one task for every active rank. Distribute the remaining tasks
    // proportionally after removing one token per active rank from the pool.
    const int nReservedTasks = nActiveRanks;
    const int nProportionalTasks = nTasks - nReservedTasks;
    const int nProportionalTokens = nTotalTokens - nReservedTasks;
    const int tokensBeforeRank = rankTokenPrefix - nRankTokens;
    const int reservedTasksBeforeRank = activeRankPrefix - activeRank;
    const int proportionalTokensBeforeRank = tokensBeforeRank - reservedTasksBeforeRank;
    const int proportionalTokensThroughRank = rankTokenPrefix - activeRankPrefix;
    const int proportionalTaskBegin =
        proportionalTaskBoundary(proportionalTokensBeforeRank, nProportionalTasks, nProportionalTokens);
    const int proportionalTaskEnd =
        proportionalTaskBoundary(proportionalTokensThroughRank, nProportionalTasks, nProportionalTokens);
    const int rankTaskBegin = reservedTasksBeforeRank + proportionalTaskBegin;
    const int nRankTasks = activeRank + proportionalTaskEnd - proportionalTaskBegin;
    if (sourceRank < nRanks && nRankTasks > 0) {
      for (int rankTaskIdx = 0; rankTaskIdx < nRankTasks; ++rankTaskIdx) {
        workspaceView.dispatchRecvTasks_[rankTaskBegin + rankTaskIdx] = {
            sourceRank, nRankTokens * rankTaskIdx / nRankTasks, nRankTokens * (rankTaskIdx + 1) / nRankTasks};
      }
    }
    if (threadId == 0) *workspaceView.dispatchNumRecvTasks_ = nTasks;

    syncNamedBarrier(DispatchSchedulerReadyBarrier, (nRankWarps + nLayoutWarps) * WARP_SIZE);
    if (threadId == 0) {
      mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(workspaceView.dispatchTasksReadyEpoch_, epoch,
                                                           mscclpp::memoryOrderRelease);
    }

    if (sourceRank < nRanks && nRankTokens > 0) {
      if (transport.isSelf(sourceRank)) {
        workspaceView.dispatchLocalPayloadReady_->acquire();
      } else {
        transport.baseMemoryChannels_[sourceRank].wait(-1);
      }
      mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(workspaceView.dispatchRankReadyEpochs_ + sourceRank, epoch,
                                                           mscclpp::memoryOrderRelease);
    }
  } else if (warpId < nRankWarps + nLayoutWarps) {
    auto* expertTokenCounts = reinterpret_cast<mscclpp::LL8Packet*>(recvBuffer) + nRanks;
    const int layoutThreadId = (warpId - nRankWarps) * WARP_SIZE + laneId;
    const int nLayoutThreads = nLayoutWarps * WARP_SIZE;
    for (int localExpertIdx = layoutThreadId; localExpertIdx < nLocalExperts; localExpertIdx += nLayoutThreads) {
      int outputOffset = 0;
      for (int sourceRank = 0; sourceRank < nRanks; ++sourceRank) {
        const int nExpertTokens =
            static_cast<int>(expertTokenCounts[sourceRank * nLocalExperts + localExpertIdx].read(epoch, -1));
        outputLayout[localExpertIdx * nRanks + sourceRank] = pack2<int, int64_t>(nExpertTokens, outputOffset);
        outputOffset += nExpertTokens;
      }
      outputCount[localExpertIdx] = outputOffset;
    }
    syncNamedBarrier(DispatchSchedulerReadyBarrier, (nRankWarps + nLayoutWarps) * WARP_SIZE);
  }
}

MSCCLPP_DEVICE_INLINE bool acquireRecvTask(RecvTask& task, WorkspaceView& workspaceView, uint32_t epoch,
                                           int* sharedMem) {
  auto* sharedTask = reinterpret_cast<RecvTask*>(sharedMem);
  const int taskIdx = static_cast<int>(blockIdx.x) - 1;
  if (threadIdx.x == 0) {
    while (mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(workspaceView.dispatchTasksReadyEpoch_,
                                                               mscclpp::memoryOrderAcquire) != epoch);
    if (taskIdx < *workspaceView.dispatchNumRecvTasks_) {
      task = workspaceView.dispatchRecvTasks_[taskIdx];
      while (mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(
                 workspaceView.dispatchRankReadyEpochs_ + task.sourceRank_, mscclpp::memoryOrderAcquire) != epoch);
      *sharedTask = task;
    } else {
      *sharedTask = {-1, 0, 0};
    }
  }
  __syncthreads();
  task = *sharedTask;
  return task.sourceRank_ >= 0;
}

MSCCLPP_DEVICE_INLINE void dispatchRecvRankMajor(int* outputTopkIdx, float* outputTopkWeights, int* outputCount,
                                                 const TransportView& transport, int nExperts, int nRanks, int nTopk,
                                                 int maxTokensPerRank, int invalidTokenExpertId, void* recvBuffer,
                                                 void* workspace, uint32_t epoch, int* sharedMem) {
  const int sourceRank = static_cast<int>(blockIdx.x);
  if (sourceRank >= nRanks) return;
  auto* rankTokenCounts = reinterpret_cast<mscclpp::LL8Packet*>(recvBuffer) + nRanks;
  if (threadIdx.x == 0) {
#if defined(MSCCLPP_EP_GPUNETIO_DIAG)
    uint32_t rawCount = 0;
    for (uint64_t spin = 0; rankTokenCounts[sourceRank].readOnce(epoch, rawCount); ++spin) {
      if (spin > 50000000ULL) {
        for (int source = 0; source < nRanks; ++source) {
          const uint64_t raw = *reinterpret_cast<volatile uint64_t*>(&rankTokenCounts[source]);
          printf("[GINDIAG] rank=%d DISPATCH-COUNT-TIMEOUT src=%d epoch=%u slot=%d flag=%u data=%u nvlink=%d\n",
                 transport.rank_, sourceRank, epoch, source, static_cast<uint32_t>(raw >> 32),
                 static_cast<uint32_t>(raw), static_cast<int>(transport.isNvlinkPeer(source)));
        }
        __trap();
      }
    }
    const int nRankTokens = static_cast<int>(rawCount);
#else
    const int nRankTokens = static_cast<int>(rankTokenCounts[sourceRank].read(epoch, -1));
#endif
    outputCount[sourceRank] = nRankTokens;
    sharedMem[0] = nRankTokens;
  }
  __syncthreads();

  const int nRankTokens = sharedMem[0];
  const int nMetadataEntries = maxTokensPerRank * nTopk;
  for (int metadataIdx = nRankTokens * nTopk + static_cast<int>(threadIdx.x); metadataIdx < nMetadataEntries;
       metadataIdx += static_cast<int>(blockDim.x)) {
    const size_t outputIdx = static_cast<size_t>(sourceRank) * nMetadataEntries + metadataIdx;
    outputTopkIdx[outputIdx] = invalidTokenExpertId;
    outputTopkWeights[outputIdx] = 0.0f;
  }

  WorkspaceView workspaceView(workspace, nRanks, nExperts);
  if (threadIdx.x == 0) {
    // Persist even zero counts: combine must not reuse a previous source's rows.
    workspaceView.dispatchRecvCounts_[sourceRank] = nRankTokens;
  }
#if defined(MSCCLPP_USE_GPUNETIO)
  if (transport.gpuNetIo_ != nullptr && !transport.isNvlinkPeer(sourceRank)) {
    const int nQp = transport.gpuNetIo_->numQpsPerPeer;
    const uint64_t target = workspaceView.dispatchArrivedBaseline_[sourceRank] + 1;
    auto* flags = reinterpret_cast<volatile uint64_t*>(transport.gpuNetIoFlagsBuffer_) +
                  static_cast<size_t>(sourceRank) * GpuNetIoMaxQpsPerPeer;
    for (int qpIndex = static_cast<int>(threadIdx.x); qpIndex < nQp; qpIndex += static_cast<int>(blockDim.x)) {
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
      const uint64_t waitBeginNs = timingNow();
#endif
      while (flags[qpIndex] < target) {
      }
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
      transport.timing_.record(TimingKind::DispatchQueue, waitBeginNs, timingNow(), sourceRank, qpIndex, 2, target);
#endif
    }
    __syncthreads();
    if (threadIdx.x == 0) workspaceView.dispatchArrivedBaseline_[sourceRank] = target;
    return;
  }
#endif
  if (threadIdx.x == 0 && nRankTokens > 0) {
    if (transport.isSelf(sourceRank)) {
      workspaceView.dispatchLocalPayloadReady_->acquire();
    } else {
      transport.baseMemoryChannels_[sourceRank].wait(-1);
    }
  }
}

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
MSCCLPP_DEVICE_INLINE bool dispatchRecvExpertMajorOutput(
    void* output, void* outputScales, int* outputSrcInfo, int64_t* outputLayout,
    const DispatchPayloadView<DataType>& payloadView, void* sourcePayload, int localExpertIdx, int sourceRank,
    int sourceTokenIdx, int nLocalExperts, int nRanks, int nTopk, int maxTokensPerRank, WorkspaceView& workspaceView,
    uint8_t* sharedTile, mscclpp::BulkBarrier* bulkBarrier, uint32_t& recvBulkPhase) {
  using OutputType = DispatchElementType<DataType>;
  constexpr size_t OutputBytes = static_cast<size_t>(Hidden) * sizeof(OutputType);
  constexpr int NumScales = DataType == DispatchDataType::BF16 ? 0 : Hidden / ScaleBlockSize;
  const int laneId = get_lane_id();
  const int nOutputSlotsPerExpert = nRanks * maxTokensPerRank;
  int outputTokenIdx = -1;
  int combineInputOffset = -1;

  if (localExpertIdx >= 0) {
    int expertTokenCount;
    int outputOffset;
    unpack2(outputLayout[localExpertIdx * nRanks + sourceRank], expertTokenCount, outputOffset);
    const int copiedTokenIdx =
        atomicAdd(workspaceView.dispatchExpertCopiedCounts_ + sourceRank * nLocalExperts + localExpertIdx, 1);
    EP_DEVICE_ASSERT(copiedTokenIdx < expertTokenCount);
    if (copiedTokenIdx == expertTokenCount - 1) {
      workspaceView.dispatchExpertCopiedCounts_[sourceRank * nLocalExperts + localExpertIdx] = 0;
    }
    outputTokenIdx = outputOffset + copiedTokenIdx;
    outputSrcInfo[static_cast<size_t>(localExpertIdx) * nOutputSlotsPerExpert + outputTokenIdx] = sourceTokenIdx;
    combineInputOffset = localExpertIdx * nOutputSlotsPerExpert + outputTokenIdx;
  }

  if constexpr (DataType != DispatchDataType::BF16) {
    using ScaleType = DispatchScaleType<DataType>;
    const auto* sourceScales = payloadView.scaleFactors(sourcePayload);
    auto* typedOutputScales = reinterpret_cast<ScaleType*>(outputScales);
    // Each top-k lane may create a row for a different local expert. All lanes
    // cooperate to copy the shared payload's scale vector to every such row.
    for (int topkLane = 0; topkLane < nTopk; ++topkLane) {
      const int scaleLocalExpertIdx = warpBroadcast(localExpertIdx, topkLane);
      const int scaleOutputTokenIdx = warpBroadcast(outputTokenIdx, topkLane);
      if (scaleLocalExpertIdx < 0) continue;
      for (int scaleIdx = laneId; scaleIdx < NumScales; scaleIdx += WARP_SIZE) {
        typedOutputScales[(static_cast<size_t>(scaleLocalExpertIdx) * NumScales + scaleIdx) * nOutputSlotsPerExpert +
                          scaleOutputTokenIdx] = sourceScales[scaleIdx];
      }
    }
  }
  if (laneId < nTopk) payloadView.topKIndices(sourcePayload)[laneId] = combineInputOffset;

  if (laneId == 0) bulkBarrier->wait(recvBulkPhase);
  __syncwarp();
  mscclpp::bulkFence();

  if (localExpertIdx < 0) return false;
  auto* outputData = reinterpret_cast<uint8_t*>(output) +
                     (static_cast<size_t>(localExpertIdx) * nOutputSlotsPerExpert + outputTokenIdx) * OutputBytes;
  mscclpp::bulkStore(outputData, sharedTile, static_cast<uint32_t>(OutputBytes));
  mscclpp::bulkStoreCommit();
  return true;
}

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
MSCCLPP_DEVICE_INLINE void dispatchRecvWorker(void* output, void* outputScales, int* outputSrcInfo,
                                              int64_t* outputLayout, int nExperts, int rank, int nRanks, int nTopk,
                                              int maxTokensPerRank, void* recvBuffer, void* workspace, uint32_t epoch,
                                              int* sharedMem) {
#if defined(__CUDA_ARCH__)
  static_assert(__CUDA_ARCH__ >= 900, "TMA recv requires SM90 or newer");
#endif
  const int threadId = static_cast<int>(threadIdx.x);
  const int warpId = threadId / WARP_SIZE;
  const int laneId = get_lane_id();
  WorkspaceView workspaceView(workspace, nRanks, nExperts);
  RecvTask task;
  if (!acquireRecvTask(task, workspaceView, epoch, sharedMem)) return;
  using OutputType = DispatchElementType<DataType>;
  constexpr int NRecvTmaWorkers = tmaWorkerCount<Hidden, OutputType, DispatchMaxNRecvTmaWorkers>();
  if (warpId >= NRecvTmaWorkers) return;

  const int nLocalExperts = nExperts / nRanks;
  const int sourceRank = task.sourceRank_;
  const int globalExpertBase = rank * nLocalExperts;
  const int globalExpertEnd = globalExpertBase + nLocalExperts;
  const DispatchPayloadView<DataType> payloadView(Hidden, nTopk, ScaleBlockSize);
  const size_t payloadStride = dispatchPayloadStride<DataType>(Hidden, nTopk, ScaleBlockSize);
  constexpr size_t OutputBytes = static_cast<size_t>(Hidden) * sizeof(OutputType);
  constexpr size_t TileBytes = OutputBytes;
  auto* sourcePayloadBase = reinterpret_cast<uint8_t*>(recvBuffer) + dispatchMetadataBytes(nRanks, nExperts) +
                            static_cast<size_t>(sourceRank) * maxTokensPerRank * payloadStride;
  auto* tmaTiles = reinterpret_cast<uint8_t*>(sharedMem) + dispatchSharedControlBytes(nRanks);
  auto* sharedTile = tmaTiles + static_cast<size_t>(warpId) * TileBytes;
  auto* bulkBarriers =
      reinterpret_cast<mscclpp::BulkBarrier*>(tmaTiles + static_cast<size_t>(NRecvTmaWorkers) * TileBytes);
  auto* bulkBarrier = bulkBarriers + warpId;
  bool hasPendingStore = false;
  uint32_t recvBulkPhase = 0;
  if (laneId == 0) bulkBarrier->init();

  for (int sourceTokenSlot = task.tokenBegin_ + warpId; sourceTokenSlot < task.tokenEnd_;
       sourceTokenSlot += NRecvTmaWorkers) {
    if (hasPendingStore) {
      mscclpp::bulkStoreWaitSource();
    }
    __syncwarp();

    auto* sourcePayload = sourcePayloadBase + static_cast<size_t>(sourceTokenSlot) * payloadStride;
    if (laneId == 0) {
      bulkBarrier->arriveAndExpect(static_cast<uint32_t>(OutputBytes));
      mscclpp::bulkLoad(sharedTile, payloadView.template data<OutputType>(sourcePayload),
                        static_cast<uint32_t>(OutputBytes), *bulkBarrier);
    }

    const int routedExpertIdx = laneId < nTopk ? payloadView.topKIndices(sourcePayload)[laneId] : -1;
    const int localExpertIdx = routedExpertIdx >= globalExpertBase && routedExpertIdx < globalExpertEnd
                                   ? routedExpertIdx - globalExpertBase
                                   : -1;
    const int sourceTokenIdx = warpBroadcast(
        laneId == 0 ? *payloadView.srcTokenGlobalIdx(sourcePayload) - sourceRank * maxTokensPerRank : 0, 0);
    hasPendingStore = dispatchRecvExpertMajorOutput<Hidden, DataType, ScaleBlockSize>(
        output, outputScales, outputSrcInfo, outputLayout, payloadView, sourcePayload, localExpertIdx, sourceRank,
        sourceTokenIdx, nLocalExperts, nRanks, nTopk, maxTokensPerRank, workspaceView, sharedTile, bulkBarrier,
        recvBulkPhase);
  }

  if (hasPendingStore) mscclpp::bulkStoreWait();
}

#endif  // MSCCLPP_BULK_AVAILABLE

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize, DispatchLayout Layout, bool AsyncDispatch = false>
MSCCLPP_DEVICE_INLINE void dispatchBody(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                                        float* outputTopkWeights, int64_t* outputLayout, int* outputCount,
                                        const int64_t* __restrict__ topkIndices, const float* __restrict__ topkWeights,
                                        const void* inputTokens, Workload workload, void* recvBuffer,
                                        const DeviceContext* context) {
#if MSCCLPP_BULK_AVAILABLE
  extern __shared__ __align__(128) uint8_t sharedMemory[];
  auto* sharedMem = reinterpret_cast<int*>(sharedMemory);
  const int nWorkerBlocks = static_cast<int>(gridDim.x) - DispatchControlBlocks;
  const int nExperts = workload.numExperts_;
  const int nRanks = context->numRanks_;
  const int nTokens = workload.numTokens_;
  const int nTopk = workload.numTopk_;
  const int invalidTokenExpertId = workload.invalidTokenExpertId_;
  const int maxTokensPerRank = workload.maxTokensPerRank_;
  const TransportView transport(context, 0, workload.epoch_);
  const uint32_t epoch = workload.epoch_;
  [[maybe_unused]] uint64_t* usedQpMask = nullptr;
#if defined(MSCCLPP_USE_GPUNETIO)
  __shared__ uint64_t dispatchUsedQps[64];
  WorkspaceView workspaceView(context->workspace_, nRanks, nExperts);
  usedQpMask = dispatchUsedQps;
  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    if (transport.gpuNetIo_ != nullptr) {
      if (transport.gpuNetIo_->numQpsPerPeer < 1 || transport.gpuNetIo_->numQpsPerPeer > GpuNetIoMaxQpsPerPeer)
        __trap();
      for (int peer = threadIdx.x; peer < nRanks; peer += blockDim.x) usedQpMask[peer] = 0;
      if (transport.gpuNetIoDeferFlush_) {
        // The previous dispatch skipped its send drain: its WQEs may still read the staging rows, count
        // scratch and marker counters this dispatch is about to rewrite. Its tickets are normally long done.
        auto* gin = transport.gpuNetIo_;
        const int nQp = gin->numQpsPerPeer;
        for (int index = threadIdx.x; index < nRanks * nQp; index += blockDim.x) {
          const int peer = index / nQp, qpIndex = index % nQp;
          if (transport.isNvlinkPeer(peer)) continue;
          gin->waitTicket(peer, transport.gpuNetIoDispatchTickets_[peer * GpuNetIoMaxQpsPerPeer + qpIndex], qpIndex);
        }
      }
      __syncthreads();
    }
  }
#endif
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  const long long sendStart = timingNow();
  long long flushNs = 0, barrierNs = 0, recvNs = 0;
  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && transport.timing_.buffer != nullptr &&
        !transport.timing_.countSubmit()) {
      const int firstProducer = AsyncDispatch ? nRanks : 1;
      const int producerBlocks = static_cast<int>(gridDim.x) - firstProducer - 1;
      uint64_t remoteMask = 0;
      int queueCount = 0;
#if defined(MSCCLPP_USE_GPUNETIO)
      if (transport.gpuNetIo_ != nullptr) queueCount = transport.gpuNetIo_->numQpsPerPeer;
#endif
      for (int peer = 0; peer < nRanks && peer < 64; ++peer) {
        if (transport.gpuNetIo_ != nullptr && !transport.isNvlinkPeer(peer)) remoteMask |= uint64_t{1} << peer;
      }
      transport.timing_.record(TimingKind::DispatchTopology, sendStart, sendStart, -1, -1, nRanks, firstProducer,
                               producerBlocks, DispatchNWarps / dispatchNWarpsPerGroup(nTokens, producerBlocks),
                               gridDim.x - 1, queueCount, remoteMask, AsyncDispatch);
    }
  }
#endif
  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    static_assert(DataType == DispatchDataType::BF16);
    dispatchSendRankMajor<Hidden, AsyncDispatch>(output, outputTopkIdx, outputTopkWeights, inputTokens, transport,
                                                 nExperts, nRanks, topkIndices, topkWeights, nTokens, nTopk,
                                                 invalidTokenExpertId, maxTokensPerRank, recvBuffer,
                                                 context->workspace_, epoch, sharedMem, usedQpMask);
  } else if constexpr (Layout == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED) {
    static_assert(DataType == DispatchDataType::BF16);
    dispatchSendRankMajorTopkExpanded<Hidden>(
        output, outputTopkIdx, outputTopkWeights, inputTokens, transport, nExperts, nRanks, topkIndices, topkWeights,
        nTokens, nTopk, invalidTokenExpertId, maxTokensPerRank, workload.deduplicateExpandedRoutes_, recvBuffer,
        context->workspace_, epoch, sharedMem);
  } else {
    dispatchSend<Hidden, DataType, ScaleBlockSize>(inputTokens, transport, nExperts, nRanks, topkIndices, topkWeights,
                                                   nTokens, nTopk, maxTokensPerRank, recvBuffer, context->workspace_,
                                                   epoch, sharedMem);
  }
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  const long long sendNs = timingNow() - sendStart;
#endif
#if defined(MSCCLPP_USE_GPUNETIO)
  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    if (transport.gpuNetIo_ != nullptr) {
      auto* gin = transport.gpuNetIo_;
      const int nQp = gin->numQpsPerPeer;
      if constexpr (AsyncDispatch) {
        if (static_cast<int>(blockIdx.x) == nWorkerBlocks + 1) {
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
          const uint64_t asyncDrainStart = timingNow();
#endif
          for (int index = threadIdx.x; index < nRanks * nQp; index += blockDim.x) {
            const int peer = index / nQp, queue = index % nQp;
            if (transport.isNvlinkPeer(peer)) continue;
            while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(&transport.gpuNetIoDispatchPeers_[peer].markersPosted,
                                                                  mscclpp::memoryOrderAcquire) == 0) {
            }
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
            const uint64_t drainBeginNs = timingNow();
#endif
            gin->flush(peer, queue);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
            transport.timing_.record(TimingKind::DispatchQueue, drainBeginNs, timingNow(), peer, queue, 1);
#endif
          }
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
          flushNs = timingNow() - asyncDrainStart;
#endif
        }
      } else {
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
        const long long barrierStart = timingNow();
#endif
        joinRankMajorGpuNetIoProducers(context->gpuNetIoDispatchDone_, nWorkerBlocks);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
        barrierNs = timingNow() - barrierStart;
#endif
        if (static_cast<int>(blockIdx.x) == nWorkerBlocks + 1) {
          if (transport.gpuNetIoBatchedDispatch_ && !transport.gpuNetIoDedicatedPoster_) {
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
            const uint64_t tailBeginNs = timingNow();
#endif
            sendRankMajorGpuNetIoControlBatches<Hidden>(transport, output, outputTopkIdx, outputTopkWeights, nRanks,
                                                        maxTokensPerRank, nTopk, workspaceView);
            __syncthreads();
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
            barrierNs += timingNow() - tailBeginNs;
#endif
          }
          auto* flagsSelf = static_cast<uint64_t*>(transport.gpuNetIoFlagsBuffer_) +
                            static_cast<size_t>(transport.rank_) * GpuNetIoMaxQpsPerPeer;
          for (int index = threadIdx.x; index < nRanks * nQp; index += blockDim.x) {
            const int peer = index / nQp, qpIndex = index % nQp;
            if (!transport.isNvlinkPeer(peer)) {
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
              const uint64_t markerBeginNs = timingNow();
#endif
              postRankMajorGpuNetIoMarker(transport, peer, qpIndex);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
              transport.timing_.record(TimingKind::DispatchQueue, markerBeginNs, timingNow(), peer, qpIndex, 0);
#endif
            }
          }
          __syncthreads();
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
          const long long flushStart = timingNow();
#endif
          for (int index = threadIdx.x; index < nRanks * nQp; index += blockDim.x) {
            const int peer = index / nQp, qpIndex = index % nQp;
            if (!transport.isNvlinkPeer(peer)) {
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
              const uint64_t drainBeginNs = timingNow();
#endif
              if (transport.gpuNetIoDeferFlush_)
                transport.gpuNetIoDispatchTickets_[peer * GpuNetIoMaxQpsPerPeer + qpIndex] =
                    gin->reservedTicket(peer, qpIndex);
              else
                gin->flush(peer, qpIndex);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
              transport.timing_.record(TimingKind::DispatchQueue, drainBeginNs, timingNow(), peer, qpIndex, 1);
#endif
            }
          }
          __syncthreads();
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
          flushNs = timingNow() - flushStart;
#endif
        }
      }
    }
  }
#endif

  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    if (static_cast<int>(blockIdx.x) < nRanks) {
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
      const long long recvStart = timingNow();
#endif
      dispatchRecvRankMajor(outputTopkIdx, outputTopkWeights, outputCount, transport, nExperts, nRanks, nTopk,
                            maxTokensPerRank, invalidTokenExpertId, recvBuffer, context->workspace_, epoch, sharedMem);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
      const long long recvEnd = timingNow();
      recvNs = recvEnd - recvStart;
      if (threadIdx.x == 0) {
        const int sourceRank = static_cast<int>(blockIdx.x);
        transport.timing_.record(TimingKind::DispatchRecv, recvStart, recvEnd, sourceRank, -1,
                                 transport.isSelf(sourceRank)         ? 0
                                 : transport.isNvlinkPeer(sourceRank) ? 1
                                                                      : 2);
      }
#endif
    }
  } else if constexpr (Layout == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED) {
    if (static_cast<int>(blockIdx.x) < nRanks) {
      dispatchRecvRankMajorTopkExpanded(outputCount, transport, nExperts, nRanks, recvBuffer, context->workspace_,
                                        epoch, sharedMem);
    }
  } else {
    if (static_cast<int>(blockIdx.x) == 0) {
      dispatchRecvScheduler(outputLayout, outputCount, transport, nExperts, nRanks, recvBuffer, context->workspace_,
                            epoch, sharedMem);
    } else if (static_cast<int>(blockIdx.x) <= nWorkerBlocks) {
      dispatchRecvWorker<Hidden, DataType, ScaleBlockSize>(output, outputScales, outputSrcInfo, outputLayout, nExperts,
                                                           context->rank_, nRanks, nTopk, maxTokensPerRank, recvBuffer,
                                                           context->workspace_, epoch, sharedMem);
    }
  }
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    if (threadIdx.x == 0)
      transport.timing_.record(TimingKind::DispatchBlock, sendStart, timingNow(), -1, -1, sendNs, flushNs, barrierNs,
                               recvNs, nTokens, nTopk, gridDim.x, Hidden);
  }
#endif
#endif  // MSCCLPP_BULK_AVAILABLE
}

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize, DispatchLayout Layout, typename KernelSelector>
inline void dispatchHiddenMode(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                               float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                               const int64_t* topkIdx, const float* topkWeights, const Workload& workload,
                               void* recvBuffer, const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  static_assert(Hidden == 2048 || Hidden == 4096 || Hidden == 4352 || Hidden == 5120 || Hidden == 6656 ||
                Hidden == 7168 || Hidden == 8192 || Hidden == 8704 || Hidden == 9216);
  using OutputType = DispatchElementType<DataType>;
  constexpr int NRecvTmaWorkers = tmaWorkerCount<Hidden, OutputType, DispatchMaxNRecvTmaWorkers>();
  static_assert(NRecvTmaWorkers > 0);
  const int nExperts = workload.numExperts_;
  const int nRanks = context.numRanks_;
  const int nTopk = workload.numTopk_;

  const size_t dynamicSharedBytes = dispatchSharedBytes<Hidden, DataType, ScaleBlockSize>(nRanks, nExperts, nTopk);
  static thread_local KernelConfigCache kernelConfig;
  auto kernel = KernelSelector::template get<Hidden, DataType, ScaleBlockSize>();
  const int residentBlocks = configureKernel(kernel, DispatchNThreads, dynamicSharedBytes, context, kernelConfig);
  EP_HOST_ASSERT(residentBlocks >= numBlocks);
  kernel<<<dim3(numBlocks), dim3(DispatchNThreads), dynamicSharedBytes, stream>>>(
      output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, topkIdx,
      topkWeights, input, workload, recvBuffer, context.devicePtr_);
  CUDA_CHECK(cudaGetLastError());
}

template <int Hidden, DispatchLayout Layout, typename KernelSelector>
inline void dispatchHidden(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                           float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                           const int64_t* topkIdx, const float* topkWeights, const Workload& workload, void* recvBuffer,
                           const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    EP_HOST_ASSERT(workload.dispatchDataType_ == DispatchDataType::BF16);
    return dispatchHiddenMode<Hidden, DispatchDataType::BF16, 0, Layout, KernelSelector>(
        output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
        topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
  } else if constexpr (Layout == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED) {
    EP_HOST_ASSERT(workload.dispatchDataType_ == DispatchDataType::BF16);
    return dispatchHiddenMode<Hidden, DispatchDataType::BF16, 0, Layout, KernelSelector>(
        output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
        topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
  } else {
    switch (workload.dispatchDataType_) {
      case DispatchDataType::BF16:
        return dispatchHiddenMode<Hidden, DispatchDataType::BF16, 0, Layout, KernelSelector>(
            output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
            topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
      case DispatchDataType::FP8_E4M3:
        return dispatchHiddenMode<Hidden, DispatchDataType::FP8_E4M3, 128, Layout, KernelSelector>(
            output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
            topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    }
    EP_HOST_ASSERT(false && "unsupported dispatch data type");
  }
}

template <int Hidden>
inline void dispatchLayout(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                           float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                           const int64_t* topkIdx, const float* topkWeights, const Workload& workload, void* recvBuffer,
                           const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  if (workload.outputLayout_ == DispatchLayout::EXPERT_MAJOR) {
    return dispatchHidden<Hidden, DispatchLayout::EXPERT_MAJOR>(
        output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
        topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
  }
  if (workload.outputLayout_ == DispatchLayout::RANK_MAJOR) {
    return dispatchHidden<Hidden, DispatchLayout::RANK_MAJOR>(
        output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
        topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
  }
  if (workload.outputLayout_ == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED) {
    return dispatchHidden<Hidden, DispatchLayout::RANK_MAJOR_TOPK_EXPANDED>(
        output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
        topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
  }
  EP_HOST_ASSERT(false && "unsupported dispatch layout");
}

template <DispatchLayout Layout, typename KernelSelector>
inline void dispatchAlgorithm(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                              float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                              const int64_t* topkIdx, const float* topkWeights, const Workload& workload,
                              void* recvBuffer, const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  const int nExperts = workload.numExperts_;
  const int rank = context.rank_;
  const int nRanks = context.numRanks_;
  const int numWorkerBlocks = numBlocks - DispatchControlBlocks;

  EP_HOST_ASSERT(nRanks > 0);
  EP_HOST_ASSERT(nExperts > 0);
  EP_HOST_ASSERT(nExperts % nRanks == 0);
  EP_HOST_ASSERT(rank >= 0 && rank < nRanks);
  EP_HOST_ASSERT(context.channels_ != nullptr);
  EP_HOST_ASSERT(workload.numTokens_ >= 0);
  EP_HOST_ASSERT(workload.numTopk_ > 0 && workload.numTopk_ <= WARP_SIZE);
  EP_HOST_ASSERT(nRanks <= 2 * WARP_SIZE);
  EP_HOST_ASSERT(numWorkerBlocks >= nRanks && numWorkerBlocks <= MaxWorkerBlocks);
  EP_HOST_ASSERT(output != nullptr);
  EP_HOST_ASSERT(workload.outputLayout_ == Layout);
  EP_HOST_ASSERT(isSupportedDispatchDataType(workload.dispatchDataType_));
  EP_HOST_ASSERT(workload.dispatchDataType_ == DispatchDataType::BF16 || outputScales != nullptr);
  EP_HOST_ASSERT(outputSrcInfo != nullptr || workload.outputLayout_ == DispatchLayout::RANK_MAJOR ||
                 workload.outputLayout_ == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED);
  EP_HOST_ASSERT(outputCount != nullptr);
  EP_HOST_ASSERT(outputLayout != nullptr || workload.outputLayout_ == DispatchLayout::RANK_MAJOR ||
                 workload.outputLayout_ == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED);
  if constexpr (Layout == DispatchLayout::RANK_MAJOR || Layout == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED) {
    EP_HOST_ASSERT(outputTopkIdx != nullptr);
    EP_HOST_ASSERT(outputTopkWeights != nullptr);
  }
  if constexpr (Layout == DispatchLayout::RANK_MAJOR || Layout == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED) {
    EP_HOST_ASSERT(workload.dispatchDataType_ == DispatchDataType::BF16);
  }
  EP_HOST_ASSERT(workload.numTokens_ == 0 || input != nullptr);
  EP_HOST_ASSERT(workload.numTokens_ == 0 || topkIdx != nullptr);
  EP_HOST_ASSERT(recvBuffer != nullptr);
  EP_HOST_ASSERT(context.localBufferBase_ != nullptr);
  EP_HOST_ASSERT(context.peerBufferBases_ != nullptr);
  EP_HOST_ASSERT(context.workspace_ != nullptr);
  EP_HOST_ASSERT(context.devicePtr_ != nullptr);

  switch (workload.hidden_) {
    case 2048:
      return dispatchHidden<2048, Layout, KernelSelector>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 4096:
      return dispatchHidden<4096, Layout, KernelSelector>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 4352:
      return dispatchHidden<4352, Layout, KernelSelector>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 5120:
      return dispatchHidden<5120, Layout, KernelSelector>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 6656:
      return dispatchHidden<6656, Layout, KernelSelector>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 7168:
      return dispatchHidden<7168, Layout, KernelSelector>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 8192:
      return dispatchHidden<8192, Layout, KernelSelector>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 8704:
      return dispatchHidden<8704, Layout, KernelSelector>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 9216:
      return dispatchHidden<9216, Layout, KernelSelector>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    default:
      EP_HOST_ASSERT(false && "unsupported latency dispatch hidden size");
  }
}

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_DISPATCH_COMMON_CUH_
