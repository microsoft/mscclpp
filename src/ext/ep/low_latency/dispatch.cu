// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include <mscclpp/bulk_device.hpp>
#include <mscclpp/memory_channel_device.hpp>

#include "api.cuh"
#include "config.cuh"
#include "device_helpers.cuh"
#include "exception.cuh"
#include "quantization.cuh"

#if defined(MSCCLPP_USE_GPUNETIO)
// Pull in the full GPUNetIO device context so the inter-domain send path can
// issue kernel-initiated RDMA. Only compiled into this TU when the backend is
// enabled; otherwise TransportView::gpuNetIo_ stays a forward-declared nullptr.
#include <mscclpp/port_channel_gpunetio_device.hpp>
#endif  // defined(MSCCLPP_USE_GPUNETIO)


namespace mscclpp {
namespace ep {
namespace low_latency {
namespace detail {

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

#if defined(MSCCLPP_USE_GPUNETIO)
// Same as sendRankMajorMetadata but, when skipCrossDomain is set, writes only for
// destinations in this rank's NVLink/IPC domain. Cross-domain destinations carry
// their metadata inside the GPUNetIO send (sendRankMajorGpuNetIo), so writing via
// mappedBuffer here would dereference a null mapped base.
MSCCLPP_DEVICE_INLINE void sendRankMajorMetadataNvlink(const TransportView& transport, int* outputTopkIdx,
                                                       float* outputTopkWeights,
                                                       const int64_t* __restrict__ topkIndices,
                                                       const float* __restrict__ topkWeights,
                                                       const RankMajorRoute& route, int tokenIdx, int nTopk,
                                                       int nLocalExperts, int maxTokensPerRank,
                                                       int invalidTokenExpertId, bool skipCrossDomain) {
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
#endif  // defined(MSCCLPP_USE_GPUNETIO)

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
// Inter-domain rank-major send over GPUNetIO (approach 2). Warp-cooperative:
// mirrors sendRankMajorMetadata's leader loop but targets peers outside this
// rank's NVLink/IPC domain, where transport.mappedBuffer() is unusable. For each
// such destination the leader stages the token + top-k metadata into one
// symmetric staging-ring slot (a GPUNetIO put must source from registered
// memory), then RDMA-writes the token to the peer rank-major token slot and the
// metadata to the peer id/weight buffers as UNSIGNALED puts. The completion flag
// is bumped once per destination by a batched atomic-add(count) posted from the
// dispatch kernel after a grid barrier, not per token.
//
// Called by all lanes of the warp; only lanes with laneId < nTopk carry metadata.
template <int Hidden>
MSCCLPP_DEVICE_INLINE void sendRankMajorGpuNetIo(const TransportView& transport, const RankMajorRoute& route,
                                                 const int64_t* __restrict__ topkIndices,
                                                 const float* __restrict__ topkWeights, void* stagedToken, int tokenIdx,
                                                 int nTopk, int nLocalExperts, int maxTokensPerRank, int nRanks,
                                                 int nExperts, int invalidTokenExpertId, int ringSlotBase,
                                                 uint64_t* usedQpMask, [[maybe_unused]] long long* tcyc) {
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(Bf16);
  const int laneId = get_lane_id();
  auto* gin = transport.gpuNetIo_;
  // rankMajor=true: this path only runs for RANK_MAJOR dispatch, and the GPUNetIO
  // staging/flags offsets depend on it -- must match the symmetric buffer allocation.
  Layout layout(transport.symmetricBufferBase_, maxTokensPerRank, Hidden, nRanks, nExperts, nTopk,
                /*rankMajor=*/true);

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
    // NVLink-domain peers are handled by the direct TMA path; skip them here.
    if (transport.isNvlinkPeer(destinationRank) || destinationSlot < 0) continue;
    // Per-destination-expert QP (mirrors NCCL/DeepEP dst_expert_local_idx): route
    // each cross-domain token onto the QP owned by its destination local expert so
    // the NIC drains different experts' streams in parallel with no head-of-line
    // blocking. The block records every (peer, qp) it touches in usedQpMask so a
    // single block-level flush-only-used pass drains just those QPs before the barrier.
    const int qpIndex = (leaderExpert >= 0 ? leaderExpert % nLocalExperts : 0) % gin->numQpsPerPeer;
    if (laneId == leaderLane)
      atomicOr(reinterpret_cast<unsigned long long*>(&usedQpMask[destinationRank]), 1ull << qpIndex);

    // Reserve staging slots [0, nRanks) for the cross-domain count packets
    // (writeRankMajorCounts) and confine the payload ring to [nRanks, slots) so a
    // concurrent token store can never clobber a count packet before the NIC
    // reads it for the count RDMA put.
    const int payloadSlots = GpuNetIoStagingSlots - nRanks;
    auto* slot = reinterpret_cast<uint8_t*>(layout.gpuNetIoStagingBuffer_) +
                 static_cast<size_t>(nRanks + ringSlot % payloadSlots) * layout.gpuNetIoSlotStride_;
    auto* slotIds = reinterpret_cast<int*>(slot + HiddenBytes);
    auto* slotWeights = reinterpret_cast<float*>(slot + HiddenBytes + static_cast<size_t>(nTopk) * sizeof(int));

    // Stage token + metadata into the symmetric slot. `slot`/`stagedToken` are
    // uniform across the warp for this destination, so the token copy is warp-
    // cooperative (all lanes stride the uint4 vectors) -- ~32x faster than the old
    // single-leader-lane loop.
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
    const long long _t0 = clock64();
#endif
    {
      const auto* src = reinterpret_cast<const uint4*>(stagedToken);
      auto* dst = reinterpret_cast<uint4*>(slot);
      constexpr int NumVec = static_cast<int>(HiddenBytes / sizeof(uint4));
      for (int i = laneId; i < NumVec; i += WARP_SIZE) dst[i] = src[i];
    }
    if (laneId < nTopk) {
      const bool isLocal = candidateExpert >= 0 && candidateExpert / nLocalExperts == destinationRank;
      slotIds[laneId] = isLocal ? candidateExpert : invalidTokenExpertId;
      slotWeights[laneId] = isLocal ? candidateWeight : 0.0f;
    }
    __syncwarp();
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
    const long long _t1 = clock64();
#endif
    // Make all lanes' slot metadata visible to the NIC (system scope) before the
    // leader posts the RDMA reads from it; device-scope would not order the NIC DMA.
    __threadfence_system();
    __syncwarp();
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
    const long long _t2 = clock64();
#endif

    if (laneId == leaderLane) {
      const size_t tokenRowOffset =
          (static_cast<size_t>(transport.rank_) * maxTokensPerRank + destinationSlot) * HiddenBytes;
      const size_t idRowOffset =
          (static_cast<size_t>(transport.rank_) * maxTokensPerRank + destinationSlot) * nTopk * sizeof(int);
      const size_t weightRowOffset =
          (static_cast<size_t>(transport.rank_) * maxTokensPerRank + destinationSlot) * nTopk * sizeof(float);
      auto* remoteToken = reinterpret_cast<uint8_t*>(layout.rankMajorTokenBuffer_) + tokenRowOffset;
      auto* remoteIds = reinterpret_cast<uint8_t*>(layout.rankMajorTopkIdsBuffer_) + idRowOffset;
      auto* remoteWeights = reinterpret_cast<uint8_t*>(layout.rankMajorTopkWeightsBuffer_) + weightRowOffset;

      gin->putBatched3(destinationRank, qpIndex, transport.symmetricOffset(remoteToken), transport.symmetricOffset(slot),
                       HiddenBytes, transport.symmetricOffset(remoteIds), transport.symmetricOffset(slotIds),
                       static_cast<size_t>(nTopk) * sizeof(int), transport.symmetricOffset(remoteWeights),
                       transport.symmetricOffset(slotWeights), static_cast<size_t>(nTopk) * sizeof(float));
    }
    __syncwarp();
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
    if (laneId == leaderLane) {
      tcyc[0] += _t1 - _t0;        // stage: 14KB single-lane copy + metadata
      tcyc[1] += _t2 - _t1;        // fence: __threadfence_system
      tcyc[2] += clock64() - _t2;  // puts: 3x gin->put
      tcyc[3] += 1;                // sends
    }
#endif
    ringSlot += 1;
  }
}

// Drain all cross-domain send queues on `qpIndex` (one flush per peer). Idle
// (peer, qpIndex) QPs no-op (flush returns immediately when the QP has no reserved
// ticket), so this is safe to call unconditionally; a used QP's flush waits for its
// latest completion, covering every warp's puts to that (peer, qpIndex) queue.
MSCCLPP_DEVICE_INLINE void flushAllCrossDomain(const TransportView& transport, int nRanks, int qpIndex = 0) {
  auto* gin = transport.gpuNetIo_;
  for (int peer = 0; peer < nRanks; ++peer) {
    if (!transport.isNvlinkPeer(peer)) gin->flush(peer, qpIndex);
  }
}

// Drain EVERY QP of every cross-domain peer. Needed when a single block spreads its
// payload puts across many QPs (per-destination-expert QP indexing): the grid
// barrier's "all payloads landed" guarantee requires each used QP to be flushed, and
// the block cannot know a-priori which QPs it touched. Idle QPs no-op.
MSCCLPP_DEVICE_INLINE void flushAllCrossDomainAllQps(const TransportView& transport, int nRanks) {
  auto* gin = transport.gpuNetIo_;
  const int nQp = gin->numQpsPerPeer;
  for (int peer = 0; peer < nRanks; ++peer) {
    if (transport.isNvlinkPeer(peer)) continue;
    for (int q = 0; q < nQp; ++q) gin->flush(peer, q);
  }
}
#endif  // defined(MSCCLPP_USE_GPUNETIO)

template <int Hidden>
MSCCLPP_DEVICE_INLINE void dispatchSendRankMajorBf16(void* output, int* outputTopkIdx, float* outputTopkWeights,
                                                     const void* inputTokens, int nExperts, int nRanks,
                                                     const int64_t* __restrict__ topkIndices,
                                                     const float* __restrict__ topkWeights, int nTokens, int nTopk,
                                                     int invalidTokenExpertId, int maxTokensPerRank,
                                                     const TransportView& transport, void* workspace,
                                                     int nPayloadBlocks, int* sharedMem,
                                                     [[maybe_unused]] uint64_t* usedQpMask) {
  if (blockIdx.x == 0 || static_cast<int>(blockIdx.x) > nPayloadBlocks) return;

  const int warpId = static_cast<int>(threadIdx.x) / WARP_SIZE;
  const int laneId = get_lane_id();
  const int senderBlockIdx = static_cast<int>(blockIdx.x) - 1;
  const int nWarpsPerGroup = dispatchNWarpsPerGroup(nTokens, nPayloadBlocks);
  const int nWarpGroups = DispatchNWarps / nWarpsPerGroup;
  const int warpGroupId = warpId / nWarpsPerGroup;
  const int subWarpId = warpId % nWarpsPerGroup;
  if (subWarpId != 0) return;

  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(Bf16);
  constexpr int HiddenVectors = Hidden / mscclpp::bf16x8::Size;
  const int nLocalExperts = nExperts / nRanks;
  const size_t sharedTokenStride = dispatchPayloadStride<DispatchDataType::BF16>(Hidden, nTopk, 0);
  auto* sharedTokenBase = reinterpret_cast<uint8_t*>(sharedMem) + dispatchSharedControlBytes(nRanks);
  auto* sendBulkBarriers =
      reinterpret_cast<mscclpp::BulkBarrier*>(sharedTokenBase + DispatchMaxNWarpGroups * sharedTokenStride);
  WorkspaceView workspaceView(workspace, nRanks, nExperts);

  auto* stagedToken = sharedTokenBase + static_cast<size_t>(warpGroupId) * sharedTokenStride;
  auto* bulkBarrier = sendBulkBarriers + warpGroupId;
  const int tokenStride = nPayloadBlocks * nWarpGroups;
  const int firstTokenIdx = senderBlockIdx * nWarpGroups + warpGroupId;
  uint32_t sendBulkPhase = 0;
  if (firstTokenIdx < nTokens && laneId == 0) bulkBarrier->init();
  [[maybe_unused]] int gpuNetIoTokensSinceFlush = 0;
  [[maybe_unused]] long long tcyc[4] = {0, 0, 0, 0};

  for (int tokenIdx = firstTokenIdx; tokenIdx < nTokens; tokenIdx += tokenStride) {
    const auto* inputData =
        reinterpret_cast<const mscclpp::bf16x8*>(inputTokens) + static_cast<size_t>(tokenIdx) * HiddenVectors;
    if (laneId == 0) {
      bulkBarrier->arriveAndExpect(static_cast<uint32_t>(HiddenBytes));
      mscclpp::bulkLoad(stagedToken, inputData, static_cast<uint32_t>(HiddenBytes), *bulkBarrier);
    }
    const RankMajorRoute route =
        prepareRankMajorRoute(workspaceView, topkIndices, tokenIdx, nTopk, nLocalExperts, maxTokensPerRank, laneId);
    if (laneId == 0) bulkBarrier->wait(sendBulkPhase);
    __syncwarp();
    mscclpp::bulkFence();
    const int completionRank = route.dstRank >= 0 && route.isLeader ? route.dstRank : -1;
#if defined(MSCCLPP_USE_GPUNETIO)
    const bool crossDomain = transport.gpuNetIo_ != nullptr;
    // NVLink token store only for leaders whose destination is NVLink-mapped;
    // cross-domain destinations are handled by the warp-cooperative GPUNetIO send.
    if (completionRank >= 0 && (!crossDomain || transport.isNvlinkPeer(completionRank))) {
      issueRankMajorTokenStore<Hidden>(output, transport, route.destinationSlot, maxTokensPerRank, stagedToken,
                                       route.dstRank);
    }
    if (crossDomain) {
      // Ring base within the payload region [nRanks, GpuNetIoStagingSlots); the
      // first nRanks slots are reserved for the cross-domain count packets.
      const int ringSlotBase =
          ((static_cast<int>(blockIdx.x) * DispatchMaxNWarpGroups + warpGroupId) * nTopk) % (GpuNetIoStagingSlots - nRanks);
      sendRankMajorGpuNetIo<Hidden>(transport, route, topkIndices, topkWeights, stagedToken, tokenIdx, nTopk,
                                    nLocalExperts, maxTokensPerRank, nRanks, nExperts, invalidTokenExpertId,
                                    ringSlotBase, usedQpMask, tcyc);
      // Batched drain: keep puts pipelined, flushing every GpuNetIoFlushInterval
      // tokens (reserve_wq_slots backpressures the send queue in between) instead
      // of once per token, which serialized every put into a round-trip.
      if (++gpuNetIoTokensSinceFlush >= GpuNetIoFlushInterval) {
        __syncwarp();
        if (laneId == 0) flushAllCrossDomainAllQps(transport, nRanks);
        gpuNetIoTokensSinceFlush = 0;
      }
    }
    // NVLink metadata + completion for NVLink leaders only (cross-domain metadata
    // and completion travel with the GPUNetIO send above).
    const int nvlinkCompletionRank =
        (completionRank >= 0 && (!crossDomain || transport.isNvlinkPeer(completionRank))) ? completionRank : -1;
    sendRankMajorMetadataNvlink(transport, outputTopkIdx, outputTopkWeights, topkIndices, topkWeights, route, tokenIdx,
                                nTopk, nLocalExperts, maxTokensPerRank, invalidTokenExpertId, crossDomain);
    completeRankMajorTokenStore(workspaceView, nvlinkCompletionRank);
#else
    if (completionRank >= 0) {
      issueRankMajorTokenStore<Hidden>(output, transport, route.destinationSlot, maxTokensPerRank, stagedToken,
                                       route.dstRank);
    }
    sendRankMajorMetadata(transport, outputTopkIdx, outputTopkWeights, topkIndices, topkWeights, route, tokenIdx, nTopk,
                          nLocalExperts, maxTokensPerRank, invalidTokenExpertId);
    completeRankMajorTokenStore(workspaceView, completionRank);
#endif  // defined(MSCCLPP_USE_GPUNETIO)
    if (tokenIdx + tokenStride < nTokens) __syncwarp();
  }
  // Cross-domain payload puts stay in flight; a single block-level flush-only-used
  // pass in the kernel drains just the touched QPs (usedQpMask) before the grid barrier.
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  {
    long long s = tcyc[0], f = tcyc[1], p = tcyc[2], n = tcyc[3];
    for (int o = 16; o > 0; o >>= 1) {
      s += __shfl_down_sync(0xffffffff, s, o);
      f += __shfl_down_sync(0xffffffff, f, o);
      p += __shfl_down_sync(0xffffffff, p, o);
      n += __shfl_down_sync(0xffffffff, n, o);
    }
    if (laneId == 0 && warpGroupId == 0 && static_cast<int>(blockIdx.x) == 1 && n > 0)
      printf("[GINTIME-SEND] r=%d nSends=%lld stage_cyc=%lld fence_cyc=%lld put_cyc=%lld per-send[stage=%lld fence=%lld put=%lld]\n",
             transport.rank_, n, s, f, p, s / n, f / n, p / n);
  }
#endif
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
                                                void* recvBuffer, uint32_t epoch) {
  const int threadId = static_cast<int>(threadIdx.x);
  for (int dstRank = threadId; dstRank < nRanks; dstRank += blockDim.x) {
#if defined(MSCCLPP_USE_GPUNETIO)
    // Cross-domain: the count packet can't be written through a mapped base, so
    // build the LL8Packet in a symmetric staging slot and RDMA-put it into the
    // peer's recvBuffer count slot for this source rank.
    if (transport.gpuNetIo_ != nullptr && !transport.isNvlinkPeer(dstRank)) {
      auto* gin = transport.gpuNetIo_;
      // Stage the count packet in a dedicated recvBuffer metadata slot (the
      // rank-major-unused expert-count region at [2*nRanks + dstRank]), NOT the
      // shared token staging ring, so a concurrent token/combine store cannot
      // clobber it before the NIC reads it.
      auto* scratch = reinterpret_cast<mscclpp::LL8Packet*>(recvBuffer) + (2 * nRanks + dstRank);
      scratch->write(static_cast<uint32_t>(rankTokenCounts[dstRank]), epoch);
      // System scope: the staged count must be visible to the NIC before its RDMA
      // read; device-scope __threadfence() does not order the NIC DMA, so the NIC
      // could ship a stale (previous-epoch) count from this reused fixed slot.
      __threadfence_system();
      auto* remotePacket = reinterpret_cast<uint8_t*>(recvBuffer) + static_cast<size_t>(nRanks + transport.rank_) * sizeof(mscclpp::LL8Packet);
      gin->put(dstRank, transport.symmetricOffset(remotePacket), transport.symmetricOffset(scratch),
               sizeof(mscclpp::LL8Packet), static_cast<int>(blockIdx.x) % gin->numQpsPerPeer);
      // Intentionally NOT flushed here. This count put rides the same per-block QP
      // (blockIdx % numQpsPerPeer) that the notify block flushes AFTER the grid
      // barrier in dispatchKernel, which drains it and frees the staging slot well
      // before the next dispatch's writeRankMajorCounts overwrites it. Delivery is
      // self-synchronizing (the receiver spins on the LL8Packet epoch flag), so it
      // does not need our flush. Flushing here would drain that QP -- shared with
      // the workers' per-expert payload puts -- forcing the notify block to wait on
      // the workers' payloads and making it the grid-barrier straggler.
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
    // Cross-domain destinations are served entirely by the GPUNetIO send path,
    // which carries its own per-token completion signal (fused put+signal). They
    // never touch the NVLink completion counter, so skip the wait/signal here.
    if (transport.gpuNetIo_ != nullptr && !transport.isNvlinkPeer(dstRank)) continue;
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

MSCCLPP_DEVICE_INLINE void dispatchRankMajorNotify(const TransportView& transport, int nExperts, int nRanks,
                                                   const int64_t* __restrict__ topkIndices, int nTokens, int nTopk,
                                                   void* recvBuffer, void* workspace, uint32_t epoch, int* sharedMem) {
  WorkspaceView workspaceView(workspace, nRanks, nExperts);
  auto* rankTokenCounts = sharedMem;
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  const long long _n0 = clock64();
#endif
  countRankMajorRoutes(rankTokenCounts, topkIndices, nTokens, nTopk, nRanks, nExperts);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  const long long _n1 = clock64();
#endif
  writeRankMajorCounts(transport, rankTokenCounts, nRanks, recvBuffer, epoch);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  const long long _n2 = clock64();
#endif
  publishDispatchPayloads(transport, rankTokenCounts, nRanks, workspaceView);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  // Split the notify straggler: count routes / write+flush count packets /
  // publishDispatchPayloads (spins on NVLink intra-node payload completions).
  if (static_cast<int>(threadIdx.x) == 0)
    printf("[GINTIME-NOTIFY] r=%d ep=%u count_cyc=%lld write_cyc=%lld publish_cyc=%lld\n", transport.rank_, epoch,
           _n1 - _n0, _n2 - _n1, clock64() - _n2);
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

template <int Hidden>
MSCCLPP_DEVICE_INLINE void dispatchSendRankMajor(void* output, int* outputTopkIdx, float* outputTopkWeights,
                                                 const void* inputTokens, const TransportView& transport, int nExperts,
                                                 int nRanks, const int64_t* __restrict__ topkIndices,
                                                 const float* __restrict__ topkWeights, int nTokens, int nTopk,
                                                 int invalidTokenExpertId, int maxTokensPerRank, void* recvBuffer,
                                                 void* workspace, uint32_t epoch, int* sharedMem, uint64_t* usedQpMask) {
  const int nWorkerBlocks = static_cast<int>(gridDim.x) - DispatchControlBlocks;
  if (static_cast<int>(blockIdx.x) > 0 && static_cast<int>(blockIdx.x) <= nWorkerBlocks) {
    dispatchSendRankMajorBf16<Hidden>(output, outputTopkIdx, outputTopkWeights, inputTokens, nExperts, nRanks,
                                      topkIndices, topkWeights, nTokens, nTopk, invalidTokenExpertId, maxTokensPerRank,
                                      transport, workspace, nWorkerBlocks, sharedMem, usedQpMask);
  } else if (static_cast<int>(blockIdx.x) == nWorkerBlocks + 1) {
    dispatchRankMajorNotify(transport, nExperts, nRanks, topkIndices, nTokens, nTopk, recvBuffer, workspace, epoch,
                            sharedMem);
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
  const int nWorkerBlocks = static_cast<int>(gridDim.x) - low_latency::DispatchControlBlocks;
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
  // EXPERIMENT: relocate the rank-major count slots off symmetric base+0 (which
  // is being clobbered by a fixed-address per-block accumulator) into the
  // expert-count region, which the rank-major recv path never reads.
  auto* rankTokenCounts = reinterpret_cast<mscclpp::LL8Packet*>(recvBuffer) + nRanks;
  if (threadIdx.x == 0) {
#if defined(MSCCLPP_EP_GPUNETIO_DIAG)
    uint32_t rawCount = 0;
    for (uint64_t spin = 0; rankTokenCounts[sourceRank].readOnce(epoch, rawCount); ++spin) {
      if (spin > 50000000ULL) {
        // Snapshot the slot with a single volatile 8-byte load (same path as
        // readOnce) so the printed flag/data are the true atomic memory state
        // and not a stale L2-cached view of the plain struct members.
        uint64_t raw = *reinterpret_cast<volatile uint64_t*>(&rankTokenCounts[sourceRank]);
        printf("[GINDIAG] rank=%d DISPATCH-COUNT-TIMEOUT src=%d epoch=%u pktflag=%u pktdata=%u gridDim=%d\n", transport.rank_,
               sourceRank, epoch, static_cast<uint32_t>(raw >> 32), static_cast<uint32_t>(raw), static_cast<int>(gridDim.x));
        // Dump every incoming count slot so we can tell whether ONLY this source
        // is missing (producer stuck) vs. a broader clear/desync, and classify
        // each source as NVLink (atomic write) vs. cross-domain (RDMA put).
        for (int s = 0; s < nRanks; ++s) {
          uint64_t rs = *reinterpret_cast<volatile uint64_t*>(&rankTokenCounts[s]);
          printf("[GINDIAG]   rank=%d slot[src=%d] flag=%u data=%u nvlink=%d self=%d\n", transport.rank_, s,
                 static_cast<uint32_t>(rs >> 32), static_cast<uint32_t>(rs), static_cast<int>(transport.isNvlinkPeer(s)),
                 static_cast<int>(transport.isSelf(s)));
        }
        // Byte offsets (from the symmetric base) of the count slot vs. the two
        // GPUNetIO flag arrays, plus the flag values, to test whether the count
        // slot's data is being clobbered by a per-token completion-signal
        // atomic-add (pktdata that scales with epoch would indicate this).
        auto* base = reinterpret_cast<uint8_t*>(transport.symmetricBufferBase_);
        const long recvOff = reinterpret_cast<uint8_t*>(&rankTokenCounts[sourceRank]) - base;
        auto* flagsP = reinterpret_cast<uint8_t*>(transport.gpuNetIoFlagsBuffer_) + sourceRank * sizeof(uint64_t);
        auto* cflagsP = reinterpret_cast<uint8_t*>(transport.gpuNetIoCombineFlagsBuffer_) + sourceRank * sizeof(uint64_t);
        printf("[GINDIAG]   rank=%d recvOff=%ld flagsOff=%ld flags[src]=%llu cflagsOff=%ld cflags[src]=%llu\n",
               transport.rank_, recvOff, static_cast<long>(flagsP - base),
               static_cast<unsigned long long>(*reinterpret_cast<volatile uint64_t*>(flagsP)),
               static_cast<long>(cflagsP - base),
               static_cast<unsigned long long>(*reinterpret_cast<volatile uint64_t*>(cflagsP)));
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
    // Persist the per-source recv count for the combine PUSH sender; the LL8 count
    // packet in recvBuffer is cleared before combine runs, the workspace is not.
    workspaceView.dispatchRecvCounts_[sourceRank] = nRankTokens;
  }
  if (threadIdx.x == 0 && nRankTokens > 0) {
#if defined(MSCCLPP_USE_GPUNETIO)
    // Cross-domain source: the payload arrived via GPUNetIO, which bumps this
    // rank's per-source completion flag once per token. The flag is a monotonic
    // cumulative counter that is never reset -- resetting it would race with the
    // remote NIC atomic-add of a later epoch and drop a signal, hanging a future
    // iteration. Instead each receiver tracks its own cumulative baseline (touched
    // by a single thread: one block per source rank), so the wait target grows by
    // this epoch's token count and no writer ever races the flag.
    if (transport.gpuNetIo_ != nullptr && !transport.isNvlinkPeer(sourceRank)) {
      auto* flags = reinterpret_cast<volatile uint64_t*>(transport.gpuNetIoFlagsBuffer_);
      const uint64_t target = workspaceView.dispatchArrivedBaseline_[sourceRank] + static_cast<uint64_t>(nRankTokens);
      while (flags[sourceRank] < target) {
      }
      workspaceView.dispatchArrivedBaseline_[sourceRank] = target;
      return;
    }
#endif  // defined(MSCCLPP_USE_GPUNETIO)
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

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize, DispatchLayout Layout>
__global__ __launch_bounds__(DispatchNThreads,
                             1) void dispatchKernel(void* output, void* outputScales, int* outputSrcInfo,
                                                    int* outputTopkIdx, float* outputTopkWeights, int64_t* outputLayout,
                                                    int* outputCount, const int64_t* __restrict__ topkIndices,
                                                    const float* __restrict__ topkWeights, const void* inputTokens,
                                                    Workload workload, void* recvBuffer, CommContext comm,
                                                    void* workspace) {
#if MSCCLPP_BULK_AVAILABLE
  extern __shared__ __align__(128) uint8_t sharedMemory[];
  auto* sharedMem = reinterpret_cast<int*>(sharedMemory);
  const int nWorkerBlocks = static_cast<int>(gridDim.x) - DispatchControlBlocks;
  const int nExperts = workload.numExperts_;
  const int nRanks = comm.numRanks_;
  const int nTokens = workload.numTokens_;
  const int nTopk = workload.numTopk_;
  const int invalidTokenExpertId = workload.invalidTokenExpertId_;
  const int maxTokensPerRank = workload.maxTokensPerRank_;
  const TransportView transport(comm);
  WorkspaceView workspaceView(workspace, nRanks, nExperts);
  const uint32_t epoch = workload.epoch_;
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  [[maybe_unused]] long long tSendCyc = 0, tFlushCyc = 0, tBarCyc = 0, tRecvCyc = 0;
#endif
#if defined(MSCCLPP_USE_GPUNETIO)
  // Per-block bitmask of cross-domain QPs this block posts to (one uint64 per peer,
  // bit q => QP q used). The block-level flush-only-used pass below drains just these.
  __shared__ uint64_t sDispatchUsedQpMask[64];
  uint64_t* usedQpMask = sDispatchUsedQpMask;
  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    if (transport.gpuNetIo_ != nullptr) {
      for (int i = static_cast<int>(threadIdx.x); i < nRanks; i += static_cast<int>(blockDim.x)) usedQpMask[i] = 0;
      __syncthreads();
    }
  }
#else
  uint64_t* usedQpMask = nullptr;
#endif  // defined(MSCCLPP_USE_GPUNETIO)
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  const long long _tSend0 = clock64();
#endif
  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    static_assert(DataType == DispatchDataType::BF16);
    dispatchSendRankMajor<Hidden>(output, outputTopkIdx, outputTopkWeights, inputTokens, transport, nExperts, nRanks,
                                  topkIndices, topkWeights, nTokens, nTopk, invalidTokenExpertId, maxTokensPerRank,
                                  recvBuffer, workspace, epoch, sharedMem, usedQpMask);
  } else {
    dispatchSend<Hidden, DataType, ScaleBlockSize>(inputTokens, transport, nExperts, nRanks, topkIndices, topkWeights,
                                                   nTokens, nTopk, maxTokensPerRank, recvBuffer, workspace, epoch,
                                                   sharedMem);
  }
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  tSendCyc = clock64() - _tSend0;
#endif

#if defined(MSCCLPP_USE_GPUNETIO)
  // Batched cross-domain completion signal. The rank-major send posts payloads as
  // UNSIGNALED writes spread across each peer's QPs, and every worker flushes its
  // qpIndex (payloads complete) before this grid barrier. The barrier therefore
  // guarantees all payloads have LANDED, so the notify block posts ONE
  // atomic-add(count) per cross-domain destination (replacing one signaled put per
  // token) and the receiver's flag advances only after every token has arrived --
  // no cross-QP RC ordering is needed because the payloads are already complete.
  // All dispatch blocks are co-resident (EP_HOST_ASSERT residentBlocks >=
  // numBlocks), so the grid barrier cannot deadlock.
  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    if (transport.gpuNetIo_ != nullptr) {
      {
        // Flush-only-used: drain just the (peer, qp) pairs this block posted to,
        // once each, cooperatively across the block -- not every warp flushing all
        // nRanks*numQpsPerPeer. Removes the idle-QP flush tax that capped useful QP
        // parallelism at scale. flush(peer,qp) drains all warps' tickets on that QP.
        auto* ginDrain = transport.gpuNetIo_;
        const int nQp = ginDrain->numQpsPerPeer;
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
        const long long _tf0 = clock64();
#endif
        __syncthreads();
        for (int idx = static_cast<int>(threadIdx.x); idx < nRanks * nQp; idx += static_cast<int>(blockDim.x)) {
          const int peer = idx / nQp;
          const int q = idx % nQp;
          if (transport.isNvlinkPeer(peer)) continue;
          if (usedQpMask[peer] & (1ull << q)) ginDrain->flush(peer, q);
        }
        __syncthreads();
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
        tFlushCyc = clock64() - _tf0;
#endif
      }
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
      const long long _tb0 = clock64();
#endif
      workspaceView.combineSyncer_->sync(gridDim.x);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
      tBarCyc = clock64() - _tb0;
#endif
      if (static_cast<int>(blockIdx.x) == nWorkerBlocks + 1) {
        auto* gin = transport.gpuNetIo_;
        const int qpIndex = static_cast<int>(blockIdx.x) % gin->numQpsPerPeer;
        auto* flagsSelf = reinterpret_cast<uint8_t*>(transport.gpuNetIoFlagsBuffer_) +
                          static_cast<size_t>(transport.rank_) * sizeof(uint64_t);
        const uint64_t flagOffset = transport.symmetricOffset(flagsSelf);
        // sharedMem[dst] still holds rankTokenCounts (per-destination send count)
        // published by dispatchRankMajorNotify in this same block; the grid
        // barrier only issues __syncthreads + global spins and never clobbers it.
        for (int dst = static_cast<int>(threadIdx.x); dst < nRanks; dst += static_cast<int>(blockDim.x)) {
          if (transport.isNvlinkPeer(dst)) continue;
          const int count = sharedMem[dst];
          if (count > 0) gin->atomicAdd(dst, flagOffset, count, qpIndex);
        }
        __syncthreads();
        if (threadIdx.x == 0) flushAllCrossDomain(transport, nRanks, qpIndex);
      }
    }
  }
#endif  // defined(MSCCLPP_USE_GPUNETIO)

  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    if (static_cast<int>(blockIdx.x) < nRanks) {
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
      const long long _tr0 = clock64();
#endif
      dispatchRecvRankMajor(outputTopkIdx, outputTopkWeights, outputCount, transport, nExperts, nRanks, nTopk,
                            maxTokensPerRank, invalidTokenExpertId, recvBuffer, workspace, epoch, sharedMem);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
      tRecvCyc = clock64() - _tr0;
#endif
    }
  } else {
    if (static_cast<int>(blockIdx.x) == 0) {
      dispatchRecvScheduler(outputLayout, outputCount, transport, nExperts, nRanks, recvBuffer, workspace, epoch,
                            sharedMem);
    } else if (static_cast<int>(blockIdx.x) <= nWorkerBlocks) {
      dispatchRecvWorker<Hidden, DataType, ScaleBlockSize>(output, outputScales, outputSrcInfo, outputLayout, nExperts,
                                                           comm.rank_, nRanks, nTopk, maxTokensPerRank, recvBuffer,
                                                           workspace, epoch, sharedMem);
    }
  }
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    const int _blk = static_cast<int>(blockIdx.x);
    // Sample a recv-only block (0), a spread of worker blocks (their arrival =
    // send+flush exposes the barrier imbalance) and the notify straggler
    // (nWorkerBlocks+1). `ep` lets the parser drop warm-up iterations so the
    // barrier-wait / notify / recv split is measured in steady state.
    const bool _rep = _blk == 0 || _blk == 1 || _blk == nWorkerBlocks / 2 || _blk == nWorkerBlocks ||
                      _blk == nWorkerBlocks + 1;
    if (threadIdx.x == 0 && _rep)
      printf("[GINTIME-BLK] r=%d ep=%u blk=%d send_cyc=%lld flush_cyc=%lld bar_cyc=%lld recv_cyc=%lld\n",
             transport.rank_, epoch, _blk, tSendCyc, tFlushCyc, tBarCyc, tRecvCyc);
  }
#endif
#endif  // MSCCLPP_BULK_AVAILABLE
}

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize, DispatchLayout Layout>
inline void dispatchHiddenMode(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                               float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                               const int64_t* topkIdx, const float* topkWeights, const low_latency::Workload& workload,
                               void* recvBuffer, const low_latency::CommContext& comm, void* workspace, int numBlocks,
                               cudaStream_t stream) {
  static_assert(Hidden == 2048 || Hidden == 4096 || Hidden == 4352 || Hidden == 6656 || Hidden == 7168 ||
                Hidden == 8192 || Hidden == 8704 || Hidden == 9216);
  using OutputType = DispatchElementType<DataType>;
  constexpr int NRecvTmaWorkers = tmaWorkerCount<Hidden, OutputType, DispatchMaxNRecvTmaWorkers>();
  static_assert(NRecvTmaWorkers > 0);
  const int nExperts = workload.numExperts_;
  const int nRanks = comm.numRanks_;
  const int nTopk = workload.numTopk_;

  const size_t dynamicSharedBytes = dispatchSharedBytes<Hidden, DataType, ScaleBlockSize>(nRanks, nExperts, nTopk);
  static thread_local KernelConfigCache kernelConfig;
  const int residentBlocks = configureKernel(dispatchKernel<Hidden, DataType, ScaleBlockSize, Layout>, DispatchNThreads,
                                             dynamicSharedBytes, comm, kernelConfig);
  EP_HOST_ASSERT(residentBlocks >= numBlocks);
  dispatchKernel<Hidden, DataType, ScaleBlockSize, Layout>
      <<<dim3(numBlocks), dim3(DispatchNThreads), dynamicSharedBytes, stream>>>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, topkIdx,
          topkWeights, input, workload, recvBuffer, comm, workspace);
  CUDA_CHECK(cudaGetLastError());
}

template <int Hidden, DispatchLayout Layout>
inline void dispatchHidden(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                           float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                           const int64_t* topkIdx, const float* topkWeights, const low_latency::Workload& workload,
                           void* recvBuffer, const low_latency::CommContext& comm, void* workspace, int numBlocks,
                           cudaStream_t stream) {
  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    EP_HOST_ASSERT(workload.dispatchDataType_ == DispatchDataType::BF16);
    return dispatchHiddenMode<Hidden, DispatchDataType::BF16, 0, Layout>(
        output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
        topkIdx, topkWeights, workload, recvBuffer, comm, workspace, numBlocks, stream);
  } else {
    switch (workload.dispatchDataType_) {
      case DispatchDataType::BF16:
        return dispatchHiddenMode<Hidden, DispatchDataType::BF16, 0, Layout>(
            output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
            topkIdx, topkWeights, workload, recvBuffer, comm, workspace, numBlocks, stream);
      case DispatchDataType::FP8_E4M3:
        return dispatchHiddenMode<Hidden, DispatchDataType::FP8_E4M3, 128, Layout>(
            output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
            topkIdx, topkWeights, workload, recvBuffer, comm, workspace, numBlocks, stream);
    }
    EP_HOST_ASSERT(false && "unsupported dispatch data type");
  }
}

template <int Hidden>
inline void dispatchLayout(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                           float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                           const int64_t* topkIdx, const float* topkWeights, const low_latency::Workload& workload,
                           void* recvBuffer, const low_latency::CommContext& comm, void* workspace, int numBlocks,
                           cudaStream_t stream) {
  if (workload.outputLayout_ == DispatchLayout::EXPERT_MAJOR) {
    return dispatchHidden<Hidden, DispatchLayout::EXPERT_MAJOR>(
        output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
        topkIdx, topkWeights, workload, recvBuffer, comm, workspace, numBlocks, stream);
  }
  if (workload.outputLayout_ == DispatchLayout::RANK_MAJOR) {
    return dispatchHidden<Hidden, DispatchLayout::RANK_MAJOR>(
        output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
        topkIdx, topkWeights, workload, recvBuffer, comm, workspace, numBlocks, stream);
  }
  EP_HOST_ASSERT(false && "unsupported dispatch layout");
}

inline void dispatch(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx, float* outputTopkWeights,
                     int64_t* outputLayout, int* outputCount, const void* input, const int64_t* topkIdx,
                     const float* topkWeights, const low_latency::Workload& workload, void* recvBuffer,
                     const low_latency::CommContext& comm, void* workspace, int numBlocks, cudaStream_t stream) {
  const int nExperts = workload.numExperts_;
  const int rank = comm.rank_;
  const int nRanks = comm.numRanks_;
  const int numWorkerBlocks = numBlocks - DispatchControlBlocks;

  EP_HOST_ASSERT(nRanks > 0);
  EP_HOST_ASSERT(nExperts > 0);
  EP_HOST_ASSERT(nExperts % nRanks == 0);
  EP_HOST_ASSERT(rank >= 0 && rank < nRanks);
  EP_HOST_ASSERT(comm.baseMemoryChannels_ != nullptr);
  EP_HOST_ASSERT(workload.numTokens_ >= 0);
  EP_HOST_ASSERT(workload.numTopk_ > 0 && workload.numTopk_ <= WARP_SIZE);
  EP_HOST_ASSERT(nRanks <= 2 * WARP_SIZE);
  EP_HOST_ASSERT(numWorkerBlocks >= nRanks && numWorkerBlocks <= MaxWorkerBlocks);
  EP_HOST_ASSERT(output != nullptr);
  EP_HOST_ASSERT(workload.outputLayout_ == DispatchLayout::EXPERT_MAJOR ||
                 workload.outputLayout_ == DispatchLayout::RANK_MAJOR);
  EP_HOST_ASSERT(isSupportedDispatchDataType(workload.dispatchDataType_));
  EP_HOST_ASSERT(workload.dispatchDataType_ == DispatchDataType::BF16 || outputScales != nullptr);
  EP_HOST_ASSERT(outputSrcInfo != nullptr || workload.outputLayout_ == DispatchLayout::RANK_MAJOR);
  EP_HOST_ASSERT(outputCount != nullptr);
  EP_HOST_ASSERT(outputLayout != nullptr || workload.outputLayout_ == DispatchLayout::RANK_MAJOR);
  if (workload.outputLayout_ == DispatchLayout::RANK_MAJOR) {
    EP_HOST_ASSERT(outputTopkIdx != nullptr);
    EP_HOST_ASSERT(outputTopkWeights != nullptr);
  }
  if (workload.outputLayout_ == DispatchLayout::RANK_MAJOR) {
    EP_HOST_ASSERT(workload.dispatchDataType_ == DispatchDataType::BF16);
  }
  EP_HOST_ASSERT(workload.numTokens_ == 0 || input != nullptr);
  EP_HOST_ASSERT(workload.numTokens_ == 0 || topkIdx != nullptr);
  EP_HOST_ASSERT(recvBuffer != nullptr);
  EP_HOST_ASSERT(comm.symmetricBufferBase_ != nullptr);
  EP_HOST_ASSERT(comm.peerMappedBufferBases_ != nullptr);
  EP_HOST_ASSERT(workspace != nullptr);

  switch (workload.hidden_) {
    case 2048:
      return dispatchLayout<2048>(output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout,
                                  outputCount, input, topkIdx, topkWeights, workload, recvBuffer, comm, workspace,
                                  numBlocks, stream);
    case 4096:
      return dispatchLayout<4096>(output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout,
                                  outputCount, input, topkIdx, topkWeights, workload, recvBuffer, comm, workspace,
                                  numBlocks, stream);
    case 4352:
      return dispatchLayout<4352>(output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout,
                                  outputCount, input, topkIdx, topkWeights, workload, recvBuffer, comm, workspace,
                                  numBlocks, stream);
    case 6656:
      return dispatchLayout<6656>(output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout,
                                  outputCount, input, topkIdx, topkWeights, workload, recvBuffer, comm, workspace,
                                  numBlocks, stream);
    case 7168:
      return dispatchLayout<7168>(output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout,
                                  outputCount, input, topkIdx, topkWeights, workload, recvBuffer, comm, workspace,
                                  numBlocks, stream);
    case 8192:
      return dispatchLayout<8192>(output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout,
                                  outputCount, input, topkIdx, topkWeights, workload, recvBuffer, comm, workspace,
                                  numBlocks, stream);
    case 8704:
      return dispatchLayout<8704>(output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout,
                                  outputCount, input, topkIdx, topkWeights, workload, recvBuffer, comm, workspace,
                                  numBlocks, stream);
    case 9216:
      return dispatchLayout<9216>(output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout,
                                  outputCount, input, topkIdx, topkWeights, workload, recvBuffer, comm, workspace,
                                  numBlocks, stream);
    default:
      EP_HOST_ASSERT(false && "unsupported optimized low-latency hidden size");
  }
}

}  // namespace detail

size_t workspaceSize(int numRanks, int numExperts, int maxTokensPerRank, int numTopk) {
  return detail::workspaceBytes(numRanks, numExperts, maxTokensPerRank, numTopk);
}

void dispatch(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx, float* outputTopkWeights,
              int64_t* outputLayout, int* outputCount, const void* input, const int64_t* topkIdx,
              const float* topkWeights, const Workload& workload, void* recvBuffer, const CommContext& comm,
              void* workspace, int numBlocks, cudaStream_t stream) {
  detail::dispatch(output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount,
                   input, topkIdx, topkWeights, workload, recvBuffer, comm, workspace, numBlocks, stream);
}

}  // namespace low_latency
}  // namespace ep
}  // namespace mscclpp
