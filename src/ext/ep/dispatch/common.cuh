// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_DISPATCH_COMMON_CUH_
#define MSCCLPP_EP_DISPATCH_COMMON_CUH_
#include <mscclpp/bulk_device.hpp>
#include <mscclpp/memory_channel_device.hpp>

#include "common/device_helpers.cuh"
#include "common/latency.cuh"
#include "exception.hpp"
#include "kernels.hpp"
#if defined(MSCCLPP_USE_GPUNETIO)
#include "gpu_net_io.hpp"
#endif

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

template <int Hidden, DispatchDataType DataType>
MSCCLPP_DEVICE_INLINE void issueRankMajorTokenStore(void* output, const TransportView& transport, int destinationSlot,
                                                    int maxTokensPerRank, void* stagedToken, int destinationRank,
                                                    void* outputScales = nullptr) {
  if (destinationSlot < 0) return;

  using ElementType = DispatchElementType<DataType>;
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(ElementType);
  const size_t row = static_cast<size_t>(transport.rank_) * maxTokensPerRank + destinationSlot;
  void* destinationBuffer = transport.mappedBuffer(output, destinationRank);
  auto* destinationRow = reinterpret_cast<uint8_t*>(destinationBuffer) + row * HiddenBytes;
  mscclpp::bulkStore(destinationRow, stagedToken, static_cast<uint32_t>(HiddenBytes));
  if constexpr (DataType == DispatchDataType::MXFP8_E4M3) {
    constexpr int ScaleBytes = Hidden / 32;
    auto* destinationScales =
        static_cast<uint8_t*>(transport.mappedBuffer(outputScales, destinationRank)) + row * ScaleBytes;
    auto* stagedScales = static_cast<uint8_t*>(stagedToken) + HiddenBytes;
    if constexpr (ScaleBytes % 16 == 0) {
      mscclpp::bulkStore(destinationScales, stagedScales, ScaleBytes);
    } else {
      // Preserve contiguous scales for widths such as 4352, whose rows are not TMA-aligned.
      static_assert(ScaleBytes % sizeof(uint64_t) == 0);
      for (int i = 0; i < ScaleBytes / sizeof(uint64_t); ++i) {
        reinterpret_cast<uint64_t*>(destinationScales)[i] = reinterpret_cast<const uint64_t*>(stagedScales)[i];
      }
    }
  }
  mscclpp::bulkStoreCommit();
}

MSCCLPP_DEVICE_INLINE void completeRankMajorTokenStore(WorkspaceView& workspaceView, int destinationRank) {
  if (destinationRank < 0) return;
  mscclpp::bulkStoreWait();
  (void)mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(
      workspaceView.dispatchRankPayloadCompletions_ + destinationRank, 1, mscclpp::memoryOrderRelease);
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

#if defined(MSCCLPP_USE_GPUNETIO)
// Cross-domain metadata travels with the network payload, not the mapped stores.
MSCCLPP_DEVICE_INLINE void sendRankMajorMetadataNvlink(
    const TransportView& transport, int* outputTopkIdx, float* outputTopkWeights,
    const int64_t* __restrict__ topkIndices, const float* __restrict__ topkWeights, const RankMajorRoute& route,
    int tokenIdx, int nTopk, int nLocalExperts, int maxTokensPerRank, int invalidTokenExpertId, bool skipCrossDomain) {
  const int laneId = getLaneId();
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
template <int Hidden>
MSCCLPP_DEVICE_INLINE void postRankMajorGpuNetIoRows(const TransportView& transport, void* output, int* outputTopkIdx,
                                                     float* outputTopkWeights, int nRanks, int capacity, int nTopk,
                                                     int destination, int begin, int end, int queue) {
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(Bf16);
  constexpr int MaxRowsPerWrite = DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE / HiddenBytes;
  static_assert(MaxRowsPerWrite > 0);
  EP_DEVICE_ASSERT(static_cast<size_t>(nTopk) * sizeof(int) <= HiddenBytes);
  const RankMajorBatchView batch(transport.gpuNetIoStagingBuffer_, nRanks, capacity, Hidden, nTopk);
  // DOCA supplies GPU-scoped queue publication; gpu+poster additionally publishes
  // the acquired producer rows to system scope.
  if (transport.gpuNetIoRowFence_ == GpuNetIoRowFenceGpuPoster) __threadfence_system();
  for (; begin < end;) {
    const int count = end - begin < MaxRowsPerWrite ? end - begin : MaxRowsPerWrite;
    const size_t sourceRow = static_cast<size_t>(destination) * capacity + begin;
    const size_t remoteRow = static_cast<size_t>(transport.rank_) * capacity + begin;
    const size_t metadataBytes = static_cast<size_t>(count) * nTopk * sizeof(int);
    transport.gpuNetIo_->putBatched3(
        destination, queue, transport.symmetricOffset(reinterpret_cast<uint8_t*>(output) + remoteRow * HiddenBytes),
        transport.symmetricOffset(batch.tokens_ + sourceRow * HiddenBytes), static_cast<size_t>(count) * HiddenBytes,
        transport.symmetricOffset(outputTopkIdx + remoteRow * nTopk),
        transport.symmetricOffset(batch.ids_ + sourceRow * nTopk), metadataBytes,
        transport.symmetricOffset(outputTopkWeights + remoteRow * nTopk),
        transport.symmetricOffset(batch.weights_ + sourceRow * nTopk), metadataBytes);
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

// Arrival marker for (peer, queue): remote atomic add on the peer's per-QP flag.
MSCCLPP_DEVICE_INLINE void postRankMajorGpuNetIoMarker(const TransportView& transport, int peer, int queue) {
  auto* network = transport.gpuNetIo_;
  auto* flagsSelf = static_cast<uint64_t*>(transport.gpuNetIoFlagsBuffer_) +
                    static_cast<size_t>(transport.rank_) * GpuNetIoMaxQpsPerPeer;
  network->atomicAdd(peer, transport.symmetricOffset(flagsSelf + queue), 1, queue);
}

MSCCLPP_DEVICE_INLINE void markRankMajorGpuNetIoPeer(const TransportView& transport, int peer) {
  auto* network = transport.gpuNetIo_;
  for (int queue = 0; queue < network->numQpsPerPeer; ++queue) {
    postRankMajorGpuNetIoMarker(transport, peer, queue);
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
  const int queues = transport.gpuNetIo_->numQpsPerPeer;
  postRankMajorGpuNetIoRows<Hidden>(transport, output, outputTopkIdx, outputTopkWeights, nRanks, capacity, nTopk,
                                    destination, begin, begin + rows, chunk % queues);
  if (completeRankMajorGpuNetIoPeer(state, rows, expected)) markRankMajorGpuNetIoPeer(transport, destination);
}

template <int Hidden, bool AsyncDispatch = false>
MSCCLPP_DEVICE_INLINE void sendRankMajorGpuNetIo(const TransportView& transport, void* output, int* outputTopkIdx,
                                                 float* outputTopkWeights, const RankMajorRoute& route,
                                                 const int64_t* __restrict__ topkIndices,
                                                 const float* __restrict__ topkWeights, void* stagedToken, int tokenIdx,
                                                 int nTopk, int nLocalExperts, int maxTokensPerRank, int nRanks,
                                                 int invalidTokenExpertId, int ringSlotBase,
                                                 [[maybe_unused]] long long* timing) {
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(Bf16);
  const int laneId = getLaneId();
  auto* network = transport.gpuNetIo_;
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
    const int qpIndex = (leaderExpert % nLocalExperts) % network->numQpsPerPeer;

    auto* slot = stagingBase + static_cast<size_t>(nRanks + ringSlot) * slotStride;
    auto* slotIds = reinterpret_cast<int*>(slot + HiddenBytes);
    auto* slotWeights = reinterpret_cast<float*>(slot + HiddenBytes + static_cast<size_t>(nTopk) * sizeof(int));
    if (transport.gpuNetIoBatchedDispatch_) {
      const RankMajorBatchView batch(stagingBase, nRanks, maxTokensPerRank, Hidden, nTopk);
      const size_t row = static_cast<size_t>(destinationRank) * maxTokensPerRank + destinationSlot;
      slot = batch.tokens_ + row * HiddenBytes;
      slotIds = batch.ids_ + row * nTopk;
      slotWeights = batch.weights_ + row * nTopk;
    }

#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
    const long long stageStart = clock64();
#endif
    {
      const auto* src = reinterpret_cast<const uint4*>(stagedToken);
      auto* dst = reinterpret_cast<uint4*>(slot);
      constexpr int NumVec = static_cast<int>(HiddenBytes / sizeof(uint4));
      if (transport.gpuNetIoTmaStage_ >= 2) {
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
    const long long stageDone = clock64();
#endif
    // Registered staging must be visible to NIC DMA, not just local GPU threads.
    fenceRankMajorStagedRow(transport);
    __syncwarp();
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
    const long long fenceDone = clock64();
#endif

    if (transport.gpuNetIoBatchedDispatch_) {
      if (transport.gpuNetIoPipelineChunkRows_ > 0 && laneId == leaderLane) {
        const int chunkRows = transport.gpuNetIoPipelineChunkRows_;
        const int chunksPerRank = configCellDiv(maxTokensPerRank, chunkRows);
        const int chunk = destinationSlot / chunkRows;
        auto* completed = transport.gpuNetIoChunkReady_ + destinationRank * chunksPerRank + chunk;
        if constexpr (AsyncDispatch) {
          postRankMajorGpuNetIoAsyncChunk<Hidden>(transport, output, outputTopkIdx, outputTopkWeights, nRanks,
                                                  maxTokensPerRank, nTopk, destinationRank, chunk, completed);
        }
      }
      if (transport.gpuNetIoPipelineChunkRows_ > 0) __syncwarp();
      ++ringSlot;
      continue;
    }

    if (laneId == leaderLane) {
      const size_t rowBase = static_cast<size_t>(transport.rank_) * maxTokensPerRank + destinationSlot;
      auto* remoteToken = reinterpret_cast<uint8_t*>(output) + rowBase * HiddenBytes;
      auto* remoteIds = reinterpret_cast<uint8_t*>(outputTopkIdx) + rowBase * nTopk * sizeof(int);
      auto* remoteWeights = reinterpret_cast<uint8_t*>(outputTopkWeights) + rowBase * nTopk * sizeof(float);
      network->putBatched3(destinationRank, qpIndex, transport.symmetricOffset(remoteToken),
                           transport.symmetricOffset(slot), HiddenBytes, transport.symmetricOffset(remoteIds),
                           transport.symmetricOffset(slotIds), static_cast<size_t>(nTopk) * sizeof(int),
                           transport.symmetricOffset(remoteWeights), transport.symmetricOffset(slotWeights),
                           static_cast<size_t>(nTopk) * sizeof(float));
    }
    __syncwarp();
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
    if (laneId == leaderLane) {
      timing[0] += stageDone - stageStart;
      timing[1] += fenceDone - stageDone;
      timing[2] += clock64() - fenceDone;
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

// Helper-warp counterpart of the batched sendRankMajorGpuNetIo path: remote destinations of one token
// are staged, fenced and published concurrently instead of serially by the group leader.
template <int Hidden, bool AsyncDispatch>
MSCCLPP_DEVICE_INLINE void stageRankMajorGpuNetIoParallel(
    const TransportView& transport, void* output, int* outputTopkIdx, float* outputTopkWeights,
    const int64_t* __restrict__ topkIndices, const float* __restrict__ topkWeights, const WorkspaceView& workspaceView,
    const void* stagedToken, int tokenIdx, int nTopk, int nLocalExperts, int maxTokensPerRank, int nRanks,
    int invalidTokenExpertId, int helper, int nHelpers, int* parts) {
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(Bf16);
  constexpr int NumVec = static_cast<int>(HiddenBytes / sizeof(uint4));
  const int laneId = getLaneId();
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
    const size_t row = static_cast<size_t>(destinationRank) * maxTokensPerRank + destinationSlot;

    {
      const auto* src = reinterpret_cast<const uint4*>(stagedToken);
      auto* dst = reinterpret_cast<uint4*>(batch.tokens_ + row * HiddenBytes);
      const int vectorBegin = NumVec * part / warpsPerDestination;
      const int vectorEnd = NumVec * (part + 1) / warpsPerDestination;
      if (transport.gpuNetIoTmaStage_ >= 2) {
      } else if (transport.gpuNetIoTmaStage_) {
        if (laneId == 0 && vectorEnd > vectorBegin)
          tmaStageSpan(dst + vectorBegin, src + vectorBegin,
                       static_cast<uint32_t>((vectorEnd - vectorBegin) * sizeof(uint4)));
      } else {
        for (int vector = vectorBegin + laneId; vector < vectorEnd; vector += WARP_SIZE) dst[vector] = src[vector];
      }
    }
    if (part == 0 && laneId < nTopk) {
      const bool isLocal = candidateExpert >= 0 && candidateExpert / nLocalExperts == destinationRank;
      batch.ids_[row * nTopk + laneId] = isLocal ? candidateExpert : invalidTokenExpertId;
      batch.weights_[row * nTopk + laneId] = isLocal ? candidateWeight : 0.0f;
    }
    __syncwarp();
    // Every part is fenced before it counts toward the row, so the publisher covers the whole row.
    fenceRankMajorStagedRow(transport);
    __syncwarp();

    bool publisher = true;
    if (warpsPerDestination > 1) {
      int previous = 0;
      if (laneId == 0)
        previous = mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(parts + order, 1, mscclpp::memoryOrderAcqRel);
      publisher = __shfl_sync(0xffffffff, previous, 0) + 1 == warpsPerDestination;
    }
    if (publisher && transport.gpuNetIoPipelineChunkRows_ > 0 && laneId == 0) {
      const int chunkRows = transport.gpuNetIoPipelineChunkRows_;
      const int chunksPerRank = configCellDiv(maxTokensPerRank, chunkRows);
      const int chunk = destinationSlot / chunkRows;
      auto* completed = transport.gpuNetIoChunkReady_ + destinationRank * chunksPerRank + chunk;
      if constexpr (AsyncDispatch) {
        postRankMajorGpuNetIoAsyncChunk<Hidden>(transport, output, outputTopkIdx, outputTopkWeights, nRanks,
                                                maxTokensPerRank, nTopk, destinationRank, chunk, completed);
      }
    }
    __syncwarp();
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
                                                           int nPayloadBlocks, int* sharedMem) {
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
  const auto* stagedToken = reinterpret_cast<uint8_t*>(sharedMem) + dispatchSharedControlBytes(nRanks, true) +
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
        nWarpsPerGroup - 1, parts);
    syncNamedBarrier(barrierId, barrierThreads);
  }
  return true;
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

MSCCLPP_DEVICE_INLINE void writeRankMajorCounts(const TransportView& transport, const int* rankTokenCounts, int nRanks,
                                                void* recvBuffer, uint32_t epoch);

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
  auto* sharedTokenBase = reinterpret_cast<uint8_t*>(sharedMem) + dispatchSharedControlBytes(nRanks, true);
  auto* bulkBarriers =
      reinterpret_cast<mscclpp::BulkBarrier*>(sharedTokenBase + DispatchMaxNWarpGroups * sharedTokenStride);

  state.laneId_ = getLaneId();
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
MSCCLPP_DEVICE_INLINE void dispatchSendRankMajorBf16(void* output, int* outputTopkIdx, float* outputTopkWeights,
                                                     const void* inputTokens, int nExperts, int nRanks,
                                                     const int64_t* __restrict__ topkIndices,
                                                     const float* __restrict__ topkWeights, int nTokens, int nTopk,
                                                     int invalidTokenExpertId, int maxTokensPerRank,
                                                     const TransportView& transport, void* workspace,
                                                     int nPayloadBlocks, int* sharedMem) {
  RankMajorSendState<Hidden> sendState;
#if defined(MSCCLPP_USE_GPUNETIO)
  if (runRankMajorParallelStageHelper<Hidden, AsyncDispatch>(
          output, outputTopkIdx, outputTopkWeights, nExperts, nRanks, topkIndices, topkWeights, nTokens, nTopk,
          invalidTokenExpertId, maxTokensPerRank, transport, workspace, nPayloadBlocks, sharedMem))
    return;
#endif
  if (!initRankMajorSendState<Hidden, AsyncDispatch>(sendState, nTokens, nTopk, nRanks, nPayloadBlocks, sharedMem))
    return;

  const int laneId = sendState.laneId_;
  const int nLocalExperts = nExperts / nRanks;
  WorkspaceView workspaceView(workspace, nRanks, nExperts);

  [[maybe_unused]] int tokensSinceFlush = 0;
  [[maybe_unused]] long long timing[4] = {};
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
#endif

  for (int tokenIdx = sendState.firstTokenIdx_; tokenIdx < nTokens; tokenIdx += sendState.tokenStride_) {
    stageRankMajorBf16Token(inputTokens, tokenIdx, sendState);
    const RankMajorRoute route =
        prepareRankMajorRoute(workspaceView, topkIndices, tokenIdx, nTopk, nLocalExperts, maxTokensPerRank, laneId);
    waitRankMajorBf16Token(sendState);
    const int completionRank = route.dstRank >= 0 && route.isLeader ? route.dstRank : -1;
#if defined(MSCCLPP_USE_GPUNETIO)
    const bool crossDomain = transport.gpuNetIo_ != nullptr;
    // NVLink token store only for NVLink-mapped destinations; cross-domain
    // destinations travel via the warp-cooperative GPUNetIO send below.
    if (completionRank >= 0 && (!crossDomain || transport.isNvlinkPeer(completionRank))) {
      issueRankMajorTokenStore<Hidden, DispatchDataType::BF16>(output, transport, route.destinationSlot,
                                                               maxTokensPerRank, stagedToken, route.dstRank);
    }
    if (crossDomain && parallelStage) {
      stageParts[laneId] = 0;
      syncNamedBarrier(stageBarrierId, stageBarrierThreads);
    } else if (crossDomain) {
      const int ringSlotBase =
          (senderBlockIdx * DispatchMaxNWarpGroups + warpGroupId) * slotsPerGroup + tokensSinceFlush * nTopk;
      sendRankMajorGpuNetIo<Hidden, AsyncDispatch>(
          transport, output, outputTopkIdx, outputTopkWeights, route, topkIndices, topkWeights, stagedToken, tokenIdx,
          nTopk, nLocalExperts, maxTokensPerRank, nRanks, invalidTokenExpertId, ringSlotBase, timing);
      if (!transport.gpuNetIoBatchedDispatch_ && ++tokensSinceFlush >= batchTokens) {
        if (laneId == 0) flushAllCrossDomainAllQps(transport, nRanks);
        __syncwarp();
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
#endif  // defined(MSCCLPP_USE_GPUNETIO)
    if (tokenIdx + sendState.tokenStride_ < nTokens) __syncwarp();
  }
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  for (int offset = 16; offset > 0; offset >>= 1) {
    for (int metric = 0; metric < 4; ++metric) {
      timing[metric] += __shfl_down_sync(0xffffffff, timing[metric], offset);
    }
  }
  if (laneId == 0 && sendState.warpGroupId_ == 0 && blockIdx.x == 1 && timing[3] > 0)
    printf(
        "[GINTIME-SEND] r=%d nSends=%lld stage_cyc=%lld fence_cyc=%lld put_cyc=%lld "
        "per-send[stage=%lld fence=%lld put=%lld]\n",
        transport.rank_, timing[3], timing[0], timing[1], timing[2], timing[0] / timing[3], timing[1] / timing[3],
        timing[2] / timing[3]);
#endif
}
MSCCLPP_DEVICE_INLINE void countRankMajorRoutes(int* rankTokenCounts, const int64_t* __restrict__ topkIndices,
                                                int nTokens, int nTopk, int nRanks, int nExperts) {
  const int threadId = static_cast<int>(threadIdx.x);
  const int warpId = threadId / WARP_SIZE;
  const int laneId = getLaneId();
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
MSCCLPP_DEVICE_INLINE void writeRankMajorCounts(const TransportView& transport, const int* rankTokenCounts, int nRanks,
                                                void* recvBuffer, uint32_t epoch) {
  const int threadId = static_cast<int>(threadIdx.x);
  for (int dstRank = threadId; dstRank < nRanks; dstRank += blockDim.x) {
#if defined(MSCCLPP_USE_GPUNETIO)
    // Keep count scratch outside the token/combine staging ring. The rank-major
    // metadata allocation reserves [R,2R) for received counts and [2R,3R) for
    // these registered send packets, including small expert counts.
    if (transport.gpuNetIo_ != nullptr && !transport.isNvlinkPeer(dstRank)) {
      auto* network = transport.gpuNetIo_;
      auto* scratch = reinterpret_cast<mscclpp::LL8Packet*>(recvBuffer) + 2 * nRanks + dstRank;
      scratch->write(static_cast<uint32_t>(rankTokenCounts[dstRank]), epoch);
      auto* remotePacket = reinterpret_cast<mscclpp::LL8Packet*>(recvBuffer) + nRanks + transport.rank_;
      network->put(dstRank, transport.symmetricOffset(remotePacket), transport.symmetricOffset(scratch),
                   sizeof(mscclpp::LL8Packet), static_cast<int>(blockIdx.x) % network->numQpsPerPeer);
      continue;
    }
#endif  // defined(MSCCLPP_USE_GPUNETIO)
    auto* destinationPackets = reinterpret_cast<mscclpp::LL8Packet*>(transport.mappedBuffer(recvBuffer, dstRank));
    destinationPackets[nRanks + transport.rank_].write(static_cast<uint32_t>(rankTokenCounts[dstRank]), epoch);
  }
}
template <bool AsyncDispatch = false>
MSCCLPP_DEVICE_INLINE void dispatchRankMajorNotify(const TransportView& transport, int nExperts, int nRanks,
                                                   const int64_t* __restrict__ topkIndices, int nTokens, int nTopk,
                                                   void* recvBuffer, void* workspace, uint32_t epoch, int* sharedMem) {
  WorkspaceView workspaceView(workspace, nRanks, nExperts);
  auto* rankTokenCounts = sharedMem;
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  const long long countStart = clock64();
#endif
  countRankMajorRoutes(rankTokenCounts, topkIndices, nTokens, nTopk, nRanks, nExperts);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  const long long countDone = clock64();
#endif
#if defined(MSCCLPP_USE_GPUNETIO)
  if constexpr (AsyncDispatch) {
    for (int peer = threadIdx.x; peer < nRanks; peer += blockDim.x) {
      if (transport.isNvlinkPeer(peer)) continue;
      workspaceView.dispatchRankPayloadCompletions_[peer] = rankTokenCounts[peer];
      mscclpp::atomicStore<int, mscclpp::scopeDevice>(&transport.gpuNetIoDispatchPeers_[peer].expectedRowsPlusOne,
                                                      rankTokenCounts[peer] + 1, mscclpp::memoryOrderRelease);
      if (rankTokenCounts[peer] == 0) markRankMajorGpuNetIoPeer(transport, peer);
    }
  }
#endif
  writeRankMajorCounts(transport, rankTokenCounts, nRanks, recvBuffer, epoch);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  const long long writeDone = clock64();
#endif
  publishDispatchPayloads(transport, rankTokenCounts, nRanks, workspaceView);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  if (threadIdx.x == 0)
    printf("[GINTIME-NOTIFY] r=%d ep=%u count_cyc=%lld write_cyc=%lld publish_cyc=%lld\n", transport.rank_, epoch,
           countDone - countStart, writeDone - countDone, clock64() - writeDone);
#endif
}
template <int Hidden, bool AsyncDispatch = false>
MSCCLPP_DEVICE_INLINE void dispatchSendRankMajor(void* output, int* outputTopkIdx, float* outputTopkWeights,
                                                 const void* inputTokens, const TransportView& transport, int nExperts,
                                                 int nRanks, const int64_t* __restrict__ topkIndices,
                                                 const float* __restrict__ topkWeights, int nTokens, int nTopk,
                                                 int invalidTokenExpertId, int maxTokensPerRank, void* recvBuffer,
                                                 void* workspace, uint32_t epoch, int* sharedMem) {
  const int firstProducer = AsyncDispatch ? nRanks : 1;
  const int controlBlock = static_cast<int>(gridDim.x) - 1;
  const int nWorkerBlocks = controlBlock - firstProducer;
  if (static_cast<int>(blockIdx.x) >= firstProducer && static_cast<int>(blockIdx.x) < controlBlock) {
    dispatchSendRankMajorBf16<Hidden, AsyncDispatch>(
        output, outputTopkIdx, outputTopkWeights, inputTokens, nExperts, nRanks, topkIndices, topkWeights, nTokens,
        nTopk, invalidTokenExpertId, maxTokensPerRank, transport, workspace, nWorkerBlocks, sharedMem);
  } else if (static_cast<int>(blockIdx.x) == controlBlock) {
    dispatchRankMajorNotify<AsyncDispatch>(transport, nExperts, nRanks, topkIndices, nTokens, nTopk, recvBuffer,
                                           workspace, epoch, sharedMem);
  }
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
      while (flags[qpIndex] < target) {
      }
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

#endif  // MSCCLPP_BULK_AVAILABLE

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize, DispatchLayout Layout, typename KernelSelector,
          bool Network = false>
inline void dispatchHiddenMode(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                               float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                               const int64_t* topkIdx, const float* topkWeights, const Workload& workload,
                               void* recvBuffer, const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  static_assert(Hidden == 1024 || Hidden == 2048 || Hidden == 4096 || Hidden == 4352 || Hidden == 5120 ||
                Hidden == 6656 || Hidden == 7168 || Hidden == 8192 || Hidden == 8704 || Hidden == 9216);
  using OutputType = DispatchElementType<DataType>;
  constexpr int NRecvTmaWorkers = tmaWorkerCount<Hidden, OutputType, DispatchMaxNRecvTmaWorkers>();
  static_assert(NRecvTmaWorkers > 0);
  const int nExperts = workload.numExperts_;
  const int nRanks = context.numRanks_;
  const int nTopk = workload.numTopk_;

  size_t dynamicSharedBytes = dispatchSharedBytes<Hidden, DataType, ScaleBlockSize, Network>(nRanks, nExperts, nTopk);
  if constexpr (Network) {
    if (context.gpuNetIoSendDistribute_) {
      const size_t forwardSharedBytes =
          dispatchSharedControlBytes(nRanks, true) +
          GpuNetIoDistributeForwardWarps *
              (dispatchPayloadStride<DispatchDataType::BF16>(Hidden, nTopk, 0) + sizeof(mscclpp::BulkBarrier));
      dynamicSharedBytes = std::max(dynamicSharedBytes, forwardSharedBytes);
    }
  }
  static thread_local KernelConfigCache kernelConfig;
  auto kernel = KernelSelector::template get<Hidden, DataType, ScaleBlockSize>();
  const int residentBlocks = configureKernel(kernel, DispatchNThreads, dynamicSharedBytes, context, kernelConfig);
  EP_HOST_ASSERT(residentBlocks >= numBlocks);
  kernel<<<dim3(numBlocks), dim3(DispatchNThreads), dynamicSharedBytes, stream>>>(
      output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, topkIdx,
      topkWeights, input, workload, recvBuffer, context.devicePtr_);
  MSCCLPP_CUDATHROW(cudaGetLastError());
}

template <int Hidden, DispatchLayout Layout, typename KernelSelector, bool Network = false>
inline void dispatchHidden(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                           float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                           const int64_t* topkIdx, const float* topkWeights, const Workload& workload, void* recvBuffer,
                           const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  if constexpr (Network) {
    static_assert(Layout == DispatchLayout::RANK_MAJOR);
    EP_HOST_ASSERT(workload.dispatchDataType_ == DispatchDataType::BF16);
    return dispatchHiddenMode<Hidden, DispatchDataType::BF16, 0, Layout, KernelSelector, true>(
        output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
        topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
  } else if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    switch (workload.dispatchDataType_) {
      case DispatchDataType::BF16:
        return dispatchHiddenMode<Hidden, DispatchDataType::BF16, 0, Layout, KernelSelector>(
            output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
            topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
      case DispatchDataType::FP8_E4M3:
        break;
      case DispatchDataType::FP8_E4M3_STATIC:
        return dispatchHiddenMode<Hidden, DispatchDataType::FP8_E4M3_STATIC, 0, Layout, KernelSelector>(
            output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
            topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
      case DispatchDataType::MXFP8_E4M3:
        return dispatchHiddenMode<Hidden, DispatchDataType::MXFP8_E4M3, 32, Layout, KernelSelector>(
            output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
            topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    }
    EP_HOST_ASSERT(false && "unsupported rank-major dispatch data type");
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
      case DispatchDataType::MXFP8_E4M3:
      case DispatchDataType::FP8_E4M3_STATIC:
        break;
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
  EP_HOST_ASSERT(false && "unsupported dispatch layout");
}

template <DispatchLayout Layout, typename KernelSelector, bool Network = false>
inline void dispatchAlgorithm(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                              float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                              const int64_t* topkIdx, const float* topkWeights, const Workload& workload,
                              void* recvBuffer, const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  const int nExperts = workload.numExperts_;
  const int rank = context.rank_;
  const int nRanks = context.numRanks_;
  const int numWorkerBlocks = numBlocks - DispatchControlBlocks;

  EP_HOST_ASSERT(isSupportedRanks(nRanks));
  EP_HOST_ASSERT(nExperts > 0);
  EP_HOST_ASSERT(nExperts % nRanks == 0);
  EP_HOST_ASSERT(rank >= 0 && rank < nRanks);
  EP_HOST_ASSERT(context.channels_ != nullptr);
  EP_HOST_ASSERT(workload.numTokens_ >= 0);
  EP_HOST_ASSERT(workload.numTopk_ > 0 && workload.numTopk_ <= MaxNumTopk);
  EP_HOST_ASSERT(numWorkerBlocks >= nRanks && numWorkerBlocks <= MaxWorkerBlocks);
  EP_HOST_ASSERT(output != nullptr);
  EP_HOST_ASSERT(workload.outputLayout_ == Layout);
  EP_HOST_ASSERT(isSupportedDispatchDataType(workload.dispatchDataType_));
  if constexpr (Layout == DispatchLayout::EXPERT_MAJOR) {
    EP_HOST_ASSERT(workload.dispatchDataType_ == DispatchDataType::BF16 || outputScales != nullptr);
  } else if (workload.dispatchDataType_ == DispatchDataType::FP8_E4M3_STATIC) {
    EP_HOST_ASSERT(outputScales == nullptr);
  } else if (workload.dispatchDataType_ == DispatchDataType::MXFP8_E4M3) {
    EP_HOST_ASSERT(outputScales != nullptr);
  }
  EP_HOST_ASSERT(outputSrcInfo != nullptr || workload.outputLayout_ == DispatchLayout::RANK_MAJOR);
  EP_HOST_ASSERT(outputCount != nullptr);
  EP_HOST_ASSERT(outputLayout != nullptr || workload.outputLayout_ == DispatchLayout::RANK_MAJOR);
  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    EP_HOST_ASSERT(outputTopkIdx != nullptr);
    EP_HOST_ASSERT(outputTopkWeights != nullptr);
  }
  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    EP_HOST_ASSERT(workload.dispatchDataType_ == DispatchDataType::BF16 ||
                   workload.dispatchDataType_ == DispatchDataType::FP8_E4M3_STATIC ||
                   workload.dispatchDataType_ == DispatchDataType::MXFP8_E4M3);
  }
  EP_HOST_ASSERT(workload.numTokens_ == 0 || input != nullptr);
  EP_HOST_ASSERT(workload.numTokens_ == 0 || topkIdx != nullptr);
  EP_HOST_ASSERT(recvBuffer != nullptr);
  EP_HOST_ASSERT(context.localBufferBase_ != nullptr);
  EP_HOST_ASSERT(context.peerBufferBases_ != nullptr);
  EP_HOST_ASSERT(context.workspace_ != nullptr);
  EP_HOST_ASSERT(context.devicePtr_ != nullptr);

  switch (workload.hidden_) {
    case 1024:
      return dispatchHidden<1024, Layout, KernelSelector, Network>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 2048:
      return dispatchHidden<2048, Layout, KernelSelector, Network>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 4096:
      return dispatchHidden<4096, Layout, KernelSelector, Network>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 4352:
      return dispatchHidden<4352, Layout, KernelSelector, Network>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 5120:
      if constexpr (Network) {
        return dispatchHidden<5120, Layout, KernelSelector, true>(
            output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
            topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
      } else {
        EP_HOST_ASSERT(false && "unsupported latency dispatch hidden size");
        return;
      }
    case 6656:
      return dispatchHidden<6656, Layout, KernelSelector, Network>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 7168:
      return dispatchHidden<7168, Layout, KernelSelector, Network>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 8192:
      return dispatchHidden<8192, Layout, KernelSelector, Network>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 8704:
      return dispatchHidden<8704, Layout, KernelSelector, Network>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 9216:
      return dispatchHidden<9216, Layout, KernelSelector, Network>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    default:
      EP_HOST_ASSERT(false && "unsupported latency dispatch hidden size");
  }
}

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_DISPATCH_COMMON_CUH_
