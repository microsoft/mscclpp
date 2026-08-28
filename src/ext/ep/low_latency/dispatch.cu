// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include <mscclpp/bulk_device.hpp>
#include <mscclpp/memory_channel_device.hpp>

#include "api.cuh"
#include "config.cuh"
#include "device_helpers.cuh"
#include "exception.cuh"
#include "quantization.cuh"

namespace mscclpp {
namespace ep {
namespace low_latency {
namespace detail {

MSCCLPP_DEVICE_INLINE void debugProgress(Workload workload, uint64_t stage, uint64_t a = 0, uint64_t b = 0,
                                         uint64_t c = 0) {
  if (workload.debugProgress_ == nullptr) return;
  if (threadIdx.x != 0) return;
  workload.debugProgress_[1] = static_cast<uint64_t>(blockIdx.x);
  workload.debugProgress_[2] = a;
  workload.debugProgress_[3] = b;
  workload.debugProgress_[4] = c;
  memory_fence();
  workload.debugProgress_[0] = stage;
}

MSCCLPP_DEVICE_INLINE void debugProgressAnyThread(Workload workload, uint64_t stage, uint64_t a = 0, uint64_t b = 0,
                                                  uint64_t c = 0) {
  if (workload.debugProgress_ == nullptr) return;
  workload.debugProgress_[1] = static_cast<uint64_t>(blockIdx.x);
  workload.debugProgress_[2] = a;
  workload.debugProgress_[3] = b;
  workload.debugProgress_[4] = c;
  memory_fence();
  workload.debugProgress_[0] = stage;
}

MSCCLPP_DEVICE_INLINE void debugProgressWord(Workload workload, int word, uint64_t value) {
  if (workload.debugProgress_ == nullptr) return;
  if (word < 0) return;
  workload.debugProgress_[word] = value;
}

MSCCLPP_DEVICE_INLINE void fenceProxyAsyncSharedCta() {
#if MSCCLPP_BULK_AVAILABLE
  mscclpp::bulkFence();
#endif
}
MSCCLPP_DEVICE_INLINE void initTmaLoadBarrier(uint64_t* barrier) {
#if MSCCLPP_BULK_AVAILABLE
  reinterpret_cast<mscclpp::BulkBarrier*>(barrier)->init();
#endif
}
MSCCLPP_DEVICE_INLINE void issueTmaLoadAndExpect(const void* source, void* destination, uint64_t* barrier,
                                                 uint32_t bytes) {
#if MSCCLPP_BULK_AVAILABLE
  auto& bulkBarrier = *reinterpret_cast<mscclpp::BulkBarrier*>(barrier);
  bulkBarrier.arriveAndExpect(bytes);
  mscclpp::bulkLoad(destination, source, bytes, bulkBarrier);
#endif
}
template <typename Phase>
MSCCLPP_DEVICE_INLINE void waitTmaLoad(uint64_t* barrier, Phase& phase) {
#if MSCCLPP_BULK_AVAILABLE
  uint32_t bulkPhase = static_cast<uint32_t>(phase);
  reinterpret_cast<mscclpp::BulkBarrier*>(barrier)->wait(bulkPhase);
  phase = static_cast<Phase>(bulkPhase);
#endif
}
MSCCLPP_DEVICE_INLINE void issueTmaStore(void* destination, const void* source, uint32_t bytes) {
#if MSCCLPP_BULK_AVAILABLE
  mscclpp::bulkStore(destination, source, bytes);
  mscclpp::bulkStoreCommit();
#endif
}
template <uint32_t PendingGroups = 0>
MSCCLPP_DEVICE_INLINE void waitBulkGroupRead() {
#if MSCCLPP_BULK_AVAILABLE
  mscclpp::bulkStoreWaitSource<PendingGroups>();
#endif
}
MSCCLPP_DEVICE_INLINE void waitBulkGroup() {
#if MSCCLPP_BULK_AVAILABLE
  mscclpp::bulkStoreWait();
#endif
}

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
  issueTmaStore(destinationRow, stagedToken, static_cast<uint32_t>(HiddenBytes));
}

MSCCLPP_DEVICE_INLINE void completeRankMajorTokenStore(WorkspaceView& workspaceView, int destinationRank) {
  if (destinationRank < 0) return;
  waitBulkGroup();
  (void)mscclpp::atomicFetchAdd<int, mscclpp::scopeSystem>(
      workspaceView.dispatchRankPayloadCompletions_ + destinationRank, 1, mscclpp::memoryOrderRelease);
}

MSCCLPP_DEVICE_INLINE int dispatchSharedSendSlots(int nRanks) {
  constexpr int NSendSlots = DispatchMaxNWarpGroups * WARP_SIZE;
  return nRanks > NSendSlots ? nRanks : NSendSlots;
}

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
      (void)mscclpp::atomicFetchAdd<int, mscclpp::scopeSystem>(
          workspaceView.dispatchRankPayloadCompletions_ + dstRank, nCompleted, mscclpp::memoryOrderRelease);
    }
  }
}

MSCCLPP_DEVICE_INLINE bool useSendProducedRouteCounts(Workload workload) {
  return workload.dispatchProfileMode_ == DispatchProfileMode::SEND_COUNTS_FROM_SEND;
}

MSCCLPP_DEVICE_INLINE void resetSendProducedRouteCounts(WorkspaceView& workspaceView, int nRanks, int nExperts,
                                                        uint32_t dispatchEpoch) {
  const int threadId = static_cast<int>(threadIdx.x);
  for (int countIdx = threadId; countIdx < nRanks + nExperts; countIdx += static_cast<int>(blockDim.x)) {
    workspaceView.dispatchRouteCounts_[countIdx] = 0;
  }
  __syncthreads();
  if (threadId == 0) {
    mscclpp::atomicStore<int, mscclpp::scopeDevice>(workspaceView.dispatchRouteCountsDone_, 0,
                                                    mscclpp::memoryOrderRelaxed);
    memory_fence();
    mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(workspaceView.dispatchRouteCountsReadyEpoch_, dispatchEpoch,
                                                         mscclpp::memoryOrderRelease);
  }
}

MSCCLPP_DEVICE_INLINE void waitForSendProducedRouteCountReset(WorkspaceView& workspaceView, uint32_t dispatchEpoch) {
  while (mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(workspaceView.dispatchRouteCountsReadyEpoch_,
                                                             mscclpp::memoryOrderAcquire) != dispatchEpoch);
}

MSCCLPP_DEVICE_INLINE void addSendProducedRouteCount(WorkspaceView& workspaceView, int nRanks, int routedExpertIdx,
                                                     int dstRank, bool firstLaneForRank) {
  if (routedExpertIdx < 0) return;
  atomicAdd(workspaceView.dispatchRouteCounts_ + nRanks + routedExpertIdx, 1);
  if (firstLaneForRank && dstRank >= 0) {
    atomicAdd(workspaceView.dispatchRouteCounts_ + dstRank, 1);
  }
}

MSCCLPP_DEVICE_INLINE void markSendProducedRouteCountsDone(WorkspaceView& workspaceView) {
  memory_fence();
  (void)mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(workspaceView.dispatchRouteCountsDone_, 1,
                                                           mscclpp::memoryOrderRelease);
}

MSCCLPP_DEVICE_INLINE void waitForSendProducedRouteCounts(WorkspaceView& workspaceView, int nSenderGroups) {
  while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(workspaceView.dispatchRouteCountsDone_,
                                                        mscclpp::memoryOrderAcquire) < nSenderGroups);
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void dispatchSendRankMajorBf16(void* output, int* outputTopkIdx, float* outputTopkWeights,
                                                     const void* inputTokens, int nExperts, int nRanks,
                                                     const int64_t* __restrict__ topkIndices,
                                                     const float* __restrict__ topkWeights, int nTokens, int nTopk,
                                                     int invalidTokenExpertId, int maxTokensPerRank,
                                                     const TransportView& transport, void* workspace,
                                                     int nPayloadBlocks, int* sharedMem) {
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
  auto* sendTmaBarriers = reinterpret_cast<uint64_t*>(sharedTokenBase + DispatchMaxNWarpGroups * sharedTokenStride);
  WorkspaceView workspaceView(workspace, nRanks, nExperts);
  auto* completionCounts = aggregatedDispatchCompletionCounts(sharedMem, nRanks, warpGroupId);

  auto* stagedToken = sharedTokenBase + static_cast<size_t>(warpGroupId) * sharedTokenStride;
  auto* tmaBarrier = sendTmaBarriers + warpGroupId;
  const int tokenStride = nPayloadBlocks * nWarpGroups;
  const int firstTokenIdx = senderBlockIdx * nWarpGroups + warpGroupId;
  uint32_t sendTmaPhase = 0;
  if (firstTokenIdx < nTokens && laneId == 0) initTmaLoadBarrier(tmaBarrier);
  initAggregatedDispatchCompletions(completionCounts, nRanks, laneId);

  for (int tokenIdx = firstTokenIdx; tokenIdx < nTokens; tokenIdx += tokenStride) {
    const auto* inputData =
        reinterpret_cast<const mscclpp::bf16x8*>(inputTokens) + static_cast<size_t>(tokenIdx) * HiddenVectors;
    if (laneId == 0) {
      issueTmaLoadAndExpect(inputData, stagedToken, tmaBarrier, static_cast<uint32_t>(HiddenBytes));
    }
    const RankMajorRoute route =
        prepareRankMajorRoute(workspaceView, topkIndices, tokenIdx, nTopk, nLocalExperts, maxTokensPerRank, laneId);
    if (laneId == 0) waitTmaLoad(tmaBarrier, sendTmaPhase);
    __syncwarp();
    fenceProxyAsyncSharedCta();
    const int completionRank = route.dstRank >= 0 && route.isLeader ? route.dstRank : -1;
    if (completionRank >= 0) {
      issueRankMajorTokenStore<Hidden>(output, transport, route.destinationSlot, maxTokensPerRank, stagedToken,
                                       route.dstRank);
    }
    sendRankMajorMetadata(transport, outputTopkIdx, outputTopkWeights, topkIndices, topkWeights, route, tokenIdx, nTopk,
                          nLocalExperts, maxTokensPerRank, invalidTokenExpertId);
    completeRankMajorTokenStore(workspaceView, completionRank);
    if (tokenIdx + tokenStride < nTokens) __syncwarp();
  }
}

template <DispatchDataType DataType>
MSCCLPP_DEVICE_INLINE void stageDispatchPayloadMetadata(const DispatchPayloadView<DataType>& payloadView,
                                                        void* stagedPayload, int* destinationSlots,
                                                        WorkspaceView& workspaceView,
                                                        const int64_t* __restrict__ topkIndices,
                                                        const float* __restrict__ topkWeights, int tokenIdx, int nTopk,
                                                        int nLocalExperts, int nRanks, int maxTokensPerRank, int rank,
                                                        int laneId, Workload workload) {
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
    if (tokenIdx == 0 && laneId < 8) {
      debugProgressWord(workload, 120 + laneId, dstRank >= 0 ? static_cast<uint64_t>(dstRank) : 0xffffffffull);
      debugProgressWord(workload, 128 + laneId,
                        destinationSlot >= 0 ? static_cast<uint64_t>(destinationSlot) : 0xffffffffull);
    }
    payloadView.topKIndices(stagedPayload)[laneId] = routedExpertIdx;
    payloadView.topKValues(stagedPayload)[laneId] =
        topkWeights == nullptr ? 1.0f : topkWeights[tokenIdx * nTopk + laneId];
    *payloadView.srcTokenGlobalIdx(stagedPayload) = rank * maxTokensPerRank + tokenIdx;
  }
  if (useSendProducedRouteCounts(workload)) {
    addSendProducedRouteCount(workspaceView, nRanks, routedExpertIdx, dstRank, firstLaneForRank);
  }
}

template <DispatchDataType DataType, bool SignalPayloadReady>
MSCCLPP_DEVICE_INLINE void sendStagedDispatchPayload(const DispatchPayloadView<DataType>& payloadView,
                                                     void* stagedPayload, const int* destinationSlots,
                                                     WorkspaceView& workspaceView, int* aggregateCompletionCounts,
                                                     int nTopk, int nLocalExperts, int maxTokensPerRank,
                                                     size_t metadataBytes, size_t payloadStride, void* recvBuffer,
                                                     const TransportView& transport, int laneId,
                                                     int sourceTokenGlobalIdx, size_t sourceInfoOffset,
                                                     uint32_t dispatchEpoch, Workload workload) {
  const int destinationSlot = laneId < nTopk ? destinationSlots[laneId] : -1;
  unsigned int sendMask = __ballot_sync(0xffffffff, destinationSlot >= 0);
  while (sendMask != 0) {
    const int sendLane = __ffs(sendMask) - 1;
    if (laneId == sendLane) {
      const int dstRank = payloadView.topKIndices(stagedPayload)[laneId] / nLocalExperts;
      debugProgressWord(workload, 80 + dstRank, 10);
      debugProgressAnyThread(workload, 5100, dstRank, destinationSlot, sourceTokenGlobalIdx);
      void* destinationBuffer = transport.mappedBuffer(recvBuffer, dstRank);
      auto* destinationPayload =
          reinterpret_cast<uint8_t*>(destinationBuffer) + metadataBytes +
          (static_cast<size_t>(transport.rank_) * maxTokensPerRank + destinationSlot) * payloadStride;
      issueTmaStore(destinationPayload, stagedPayload, static_cast<uint32_t>(payloadView.numBytes_));
      debugProgressWord(workload, 80 + dstRank, 11);
      debugProgressAnyThread(workload, 5101, dstRank, destinationSlot, sourceTokenGlobalIdx);
      waitBulkGroup();
      debugProgressWord(workload, 80 + dstRank, 12);
      debugProgressAnyThread(workload, 5102, dstRank, destinationSlot, sourceTokenGlobalIdx);
      auto* sourceInfoPackets =
          reinterpret_cast<mscclpp::LL8Packet*>(reinterpret_cast<uint8_t*>(destinationBuffer) + sourceInfoOffset);
      sourceInfoPackets[static_cast<size_t>(transport.rank_) * maxTokensPerRank + destinationSlot].write(
          static_cast<uint32_t>(sourceTokenGlobalIdx), dispatchEpoch);
      memory_fence();
      if constexpr (SignalPayloadReady) {
        if (transport.isSelf(dstRank)) {
          workspaceView.dispatchLocalPayloadReady_->release();
        } else {
          transport.baseMemoryChannels_[dstRank].signal();
        }
      } else {
        aggregateCompletionCounts[dstRank] += 1;
      }
      debugProgressAnyThread(workload, 5103, dstRank, destinationSlot, sourceTokenGlobalIdx);
      if constexpr (SignalPayloadReady) {
        (void)mscclpp::atomicFetchAdd<int, mscclpp::scopeSystem>(
            workspaceView.dispatchRankPayloadCompletions_ + dstRank, 1, mscclpp::memoryOrderRelease);
      }
      debugProgressWord(workload, 80 + dstRank, 14);
      debugProgressWord(workload, 48 + dstRank, 2);
      debugProgressAnyThread(workload, 5104, dstRank, destinationSlot, sourceTokenGlobalIdx);
    }
    sendMask &= sendMask - 1;
    __syncwarp();
  }
}

template <int Hidden, DispatchDataType DataType, bool SignalPayloadReady>
MSCCLPP_DEVICE_INLINE void dispatchSendUnquantized(const void* inputTokens, int nExperts, int rank, int nRanks,
                                                   const int64_t* __restrict__ topkIndices,
                                                   const float* __restrict__ topkWeights, int nTokens, int nTopk,
                                                   int maxTokensPerRank, void* recvBuffer,
                                                   const TransportView& transport, void* workspace,
                                                   uint32_t dispatchEpoch, int* sharedMem, Workload workload) {
  static_assert(DataType == DispatchDataType::BF16 || DataType == DispatchDataType::FP16);
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

  using ElementType = DispatchElementType<DataType>;
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(ElementType);
  const int nLocalExperts = nExperts / nRanks;
  const size_t sourceInfoOffset = dispatchSourceInfoOffset(nRanks, nExperts);
  const size_t metadataBytes = dispatchMetadataBytes(nRanks, nExperts, maxTokensPerRank);
  const DispatchPayloadView<DataType> payloadView(Hidden, nTopk, 0);
  const size_t payloadStride = dispatchPayloadStride<DataType>(Hidden, nTopk, 0);
  auto* sharedPayloadBase = reinterpret_cast<uint8_t*>(sharedMem) + dispatchSharedControlBytes(nRanks);
  auto* sendTmaBarriers = reinterpret_cast<uint64_t*>(sharedPayloadBase + DispatchMaxNWarpGroups * payloadStride);
  auto* stagedPayload = sharedPayloadBase + static_cast<size_t>(warpGroupId) * payloadStride;
  auto* destinationSlots = sharedMem + warpGroupId * WARP_SIZE;
  auto* aggregateCompletionCounts = aggregatedDispatchCompletionCounts(sharedMem, nRanks, warpGroupId);
  auto* tmaBarrier = sendTmaBarriers + warpGroupId;
  WorkspaceView workspaceView(workspace, nRanks, nExperts);

  const int tokenStride = nWorkerBlocks * nWarpGroups;
  const int firstTokenIdx = senderBlockIdx * nWarpGroups + warpGroupId;
  uint32_t sendTmaPhase = 0;
  if (firstTokenIdx < nTokens) {
    if (laneId == 0) initTmaLoadBarrier(tmaBarrier);
  }
  initAggregatedDispatchCompletions(aggregateCompletionCounts, nRanks, laneId);

  for (int tokenIdx = firstTokenIdx; tokenIdx < nTokens; tokenIdx += tokenStride) {
    debugProgressWord(workload, 112, 1);
    debugProgressWord(workload, 113, static_cast<uint64_t>(blockIdx.x));
    debugProgressWord(workload, 114, tokenIdx);
    debugProgressWord(workload, 115, firstTokenIdx);
    debugProgressWord(workload, 116, tokenStride);
    debugProgress(workload, 5000, tokenIdx, firstTokenIdx, tokenStride);
    const auto* inputData = reinterpret_cast<const uint8_t*>(inputTokens) + static_cast<size_t>(tokenIdx) * HiddenBytes;
    if (laneId == 0) {
      issueTmaLoadAndExpect(inputData, stagedPayload, tmaBarrier, static_cast<uint32_t>(HiddenBytes));
    }
    stageDispatchPayloadMetadata<DataType>(payloadView, stagedPayload, destinationSlots, workspaceView, topkIndices,
                                           topkWeights, tokenIdx, nTopk, nLocalExperts, nRanks, maxTokensPerRank, rank,
                                           laneId, workload);
    debugProgressWord(workload, 117, 1);
    debugProgress(workload, 5001, tokenIdx, firstTokenIdx, tokenStride);
    if (laneId == 0) waitTmaLoad(tmaBarrier, sendTmaPhase);
    debugProgressWord(workload, 118, 1);
    debugProgress(workload, 5002, tokenIdx, firstTokenIdx, tokenStride);
    __syncwarp();
    fenceProxyAsyncSharedCta();
    debugProgressWord(workload, 119, 1);
    debugProgress(workload, 5003, tokenIdx, firstTokenIdx, tokenStride);
    sendStagedDispatchPayload<DataType, SignalPayloadReady>(
        payloadView, stagedPayload, destinationSlots, workspaceView, aggregateCompletionCounts, nTopk, nLocalExperts,
        maxTokensPerRank, metadataBytes, payloadStride, recvBuffer, transport, laneId,
        rank * maxTokensPerRank + tokenIdx, sourceInfoOffset, dispatchEpoch, workload);
    debugProgress(workload, 5004, tokenIdx, firstTokenIdx, tokenStride);
    __syncwarp();
  }
  if constexpr (!SignalPayloadReady) {
    flushAggregatedDispatchCompletions(workspaceView, aggregateCompletionCounts, nRanks, laneId);
  }
  if (useSendProducedRouteCounts(workload) && subWarpId == 0 && laneId == 0) {
    markSendProducedRouteCountsDone(workspaceView);
  }
}

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize, bool SignalPayloadReady>
MSCCLPP_DEVICE_INLINE void dispatchSendFp8(const void* inputTokens, int nExperts, int rank, int nRanks,
                                           const int64_t* __restrict__ topkIndices,
                                           const float* __restrict__ topkWeights, int nTokens, int nTopk,
                                           int maxTokensPerRank, void* recvBuffer, const TransportView& transport,
                                           void* workspace, uint32_t dispatchEpoch, int* sharedMem) {
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
  const size_t sourceInfoOffset = dispatchSourceInfoOffset(nRanks, nExperts);
  const size_t metadataBytes = dispatchMetadataBytes(nRanks, nExperts, maxTokensPerRank);
  const DispatchPayloadView<DataType> payloadView(Hidden, nTopk, ScaleBlockSize);
  const size_t payloadStride = dispatchPayloadStride<DataType>(Hidden, nTopk, ScaleBlockSize);
  auto* sharedPayloadBase = reinterpret_cast<uint8_t*>(sharedMem) + dispatchSharedControlBytes(nRanks);
  auto* stagedPayload = sharedPayloadBase + static_cast<size_t>(warpGroupId) * payloadStride;
  auto* destinationSlots = sharedMem + warpGroupId * WARP_SIZE;
  auto* aggregateCompletionCounts = aggregatedDispatchCompletionCounts(sharedMem, nRanks, warpGroupId);
  auto* outputData = payloadView.template data<mscclpp::f8_e4m3x8>(stagedPayload);
  auto* outputScales = payloadView.scaleFactors(stagedPayload);
  WorkspaceView workspaceView(workspace, nRanks, nExperts);

  const int tokenStride = nWorkerBlocks * nWarpGroups;
  const int firstTokenIdx = senderBlockIdx * nWarpGroups + warpGroupId;
  if (subWarpId == 0) initAggregatedDispatchCompletions(aggregateCompletionCounts, nRanks, laneId);
  for (int tokenIdx = firstTokenIdx; tokenIdx < nTokens; tokenIdx += tokenStride) {
    const auto* inputData =
        reinterpret_cast<const mscclpp::bf16x8*>(inputTokens) + static_cast<size_t>(tokenIdx) * HiddenVectors;
    if (subWarpId == 0) {
      stageDispatchPayloadMetadata<DataType>(payloadView, stagedPayload, destinationSlots, workspaceView, topkIndices,
                                             topkWeights, tokenIdx, nTopk, nLocalExperts, nRanks, maxTokensPerRank,
                                             rank, laneId, Workload{});
    }
    for (int inputIdx = groupThreadId; inputIdx < HiddenVectors; inputIdx += groupThreadCount) {
      outputData[inputIdx] = quantizeBf16x8ToFp8E4M3<ScaleBlockSize>(
          inputData[inputIdx], outputScales + inputIdx * mscclpp::bf16x8::Size / ScaleBlockSize, laneId);
    }
    syncNamedBarrier(groupBarrierId, groupThreadCount);

    if (subWarpId == 0) {
      fenceProxyAsyncSharedCta();
      sendStagedDispatchPayload<DataType, SignalPayloadReady>(
          payloadView, stagedPayload, destinationSlots, workspaceView, aggregateCompletionCounts, nTopk, nLocalExperts,
          maxTokensPerRank, metadataBytes, payloadStride, recvBuffer, transport, laneId,
          rank * maxTokensPerRank + tokenIdx, sourceInfoOffset, dispatchEpoch, Workload{});
    }
    syncNamedBarrier(groupBarrierId, groupThreadCount);
  }
  if (subWarpId == 0) {
    if constexpr (!SignalPayloadReady) {
      flushAggregatedDispatchCompletions(workspaceView, aggregateCompletionCounts, nRanks, laneId);
    }
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
                                                 int nExperts, void* recvBuffer, uint32_t dispatchEpoch) {
  const int threadId = static_cast<int>(threadIdx.x);
  const int nLocalExperts = nExperts / nRanks;
  for (int dstRank = threadId; dstRank < nRanks; dstRank += blockDim.x) {
    auto* destinationPackets = reinterpret_cast<mscclpp::LL8Packet*>(transport.mappedBuffer(recvBuffer, dstRank));
    destinationPackets[transport.rank_].write(static_cast<uint32_t>(counts.rankTokenCounts_[dstRank]), dispatchEpoch);
  }
  for (int expertIdx = threadId; expertIdx < nExperts; expertIdx += blockDim.x) {
    const int dstRank = expertIdx / nLocalExperts;
    const int localExpertIdx = expertIdx % nLocalExperts;
    auto* destinationPackets = reinterpret_cast<mscclpp::LL8Packet*>(transport.mappedBuffer(recvBuffer, dstRank));
    destinationPackets[nRanks + transport.rank_ * nLocalExperts + localExpertIdx].write(
        static_cast<uint32_t>(counts.expertTokenCounts_[expertIdx]), dispatchEpoch);
  }
}

MSCCLPP_DEVICE_INLINE void writeRankMajorCounts(const TransportView& transport, const int* rankTokenCounts, int nRanks,
                                                void* recvBuffer, uint32_t dispatchEpoch) {
  const int threadId = static_cast<int>(threadIdx.x);
  for (int dstRank = threadId; dstRank < nRanks; dstRank += blockDim.x) {
    auto* destinationPackets = reinterpret_cast<mscclpp::LL8Packet*>(transport.mappedBuffer(recvBuffer, dstRank));
    destinationPackets[transport.rank_].write(static_cast<uint32_t>(rankTokenCounts[dstRank]), dispatchEpoch);
  }
}

template <bool SignalPayloadReady>
MSCCLPP_DEVICE_INLINE void publishDispatchPayloads(const TransportView& transport, const int* rankTokenCounts,
                                                   int nRanks, WorkspaceView workspaceView, Workload workload) {
  const int threadId = static_cast<int>(threadIdx.x);
  for (int dstRank = threadId; dstRank < nRanks; dstRank += blockDim.x) {
    const int expectedPayloadCount = rankTokenCounts[dstRank];
    if (expectedPayloadCount > 0) {
      debugProgressWord(workload, 48 + dstRank, 1);
      debugProgress(workload, 1000, dstRank, expectedPayloadCount,
                    mscclpp::atomicLoad<int, mscclpp::scopeSystem>(
                        workspaceView.dispatchRankPayloadCompletions_ + dstRank, mscclpp::memoryOrderAcquire));
      while (mscclpp::atomicLoad<int, mscclpp::scopeSystem>(workspaceView.dispatchRankPayloadCompletions_ + dstRank,
                                                            mscclpp::memoryOrderAcquire) < expectedPayloadCount);
      debugProgressWord(workload, 48 + dstRank, 2);
      debugProgress(workload, 1001, dstRank, expectedPayloadCount,
                    mscclpp::atomicLoad<int, mscclpp::scopeSystem>(
                        workspaceView.dispatchRankPayloadCompletions_ + dstRank, mscclpp::memoryOrderAcquire));
    }
    workspaceView.dispatchRankPayloadSlots_[dstRank] = 0;
    workspaceView.dispatchRankPayloadCompletions_[dstRank] = 0;
    if (expectedPayloadCount == 0) continue;
    if constexpr (!SignalPayloadReady) {
      if (transport.isSelf(dstRank)) {
        workspaceView.dispatchLocalPayloadReady_->release();
      } else {
        transport.baseMemoryChannels_[dstRank].signal();
      }
      debugProgressWord(workload, 48 + dstRank, 3);
    }
  }
}

template <bool SignalPayloadReady>
MSCCLPP_DEVICE_INLINE void dispatchNotify(const TransportView& transport, int nExperts, int nRanks,
                                          const int64_t* __restrict__ topkIndices, int nTokens, int nTopk,
                                          void* recvBuffer, void* workspace, uint32_t dispatchEpoch, int* sharedMem,
                                          Workload workload) {
  debugProgress(workload, 900, nTokens, nTopk, nRanks);
  WorkspaceView workspaceView(workspace, nRanks, nExperts);
  DispatchCountView counts(sharedMem, nRanks);
  if (useSendProducedRouteCounts(workload)) {
    const int nWorkerBlocks = static_cast<int>(gridDim.x) - DispatchControlBlocks;
    const int nWarpsPerGroup = dispatchNWarpsPerGroup(nTokens, nWorkerBlocks);
    const int nWarpGroups = DispatchNWarps / nWarpsPerGroup;
    waitForSendProducedRouteCountReset(workspaceView, dispatchEpoch);
    waitForSendProducedRouteCounts(workspaceView, nWorkerBlocks * nWarpGroups);
    counts.rankTokenCounts_ = workspaceView.dispatchRouteCounts_;
    counts.expertTokenCounts_ = workspaceView.dispatchRouteCounts_ + nRanks;
  } else {
    countDispatchRoutes(counts, topkIndices, nTokens, nTopk, nRanks, nExperts);
  }
  debugProgress(workload, 901, counts.rankTokenCounts_[0], nExperts, nRanks);
  writeDispatchMetadata(transport, counts, nRanks, nExperts, recvBuffer, dispatchEpoch);
  debugProgress(workload, 902, counts.rankTokenCounts_[0], nExperts, nRanks);
  publishDispatchPayloads<SignalPayloadReady>(transport, counts.rankTokenCounts_, nRanks, workspaceView, workload);
  debugProgress(workload, 903, counts.rankTokenCounts_[0], nExperts, nRanks);
}

MSCCLPP_DEVICE_INLINE void dispatchRankMajorNotify(const TransportView& transport, int nExperts, int nRanks,
                                                   const int64_t* __restrict__ topkIndices, int nTokens, int nTopk,
                                                   void* recvBuffer, void* workspace, uint32_t dispatchEpoch,
                                                   int* sharedMem) {
  WorkspaceView workspaceView(workspace, nRanks, nExperts);
  auto* rankTokenCounts = sharedMem;
  countRankMajorRoutes(rankTokenCounts, topkIndices, nTokens, nTopk, nRanks, nExperts);
  writeRankMajorCounts(transport, rankTokenCounts, nRanks, recvBuffer, dispatchEpoch);
  Workload noDebug{};
  publishDispatchPayloads<false>(transport, rankTokenCounts, nRanks, workspaceView, noDebug);
}

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize, DispatchLayout Layout>
MSCCLPP_DEVICE_INLINE void dispatchSend(const void* inputTokens, const TransportView& transport, int nExperts,
                                        int nRanks, const int64_t* __restrict__ topkIndices,
                                        const float* __restrict__ topkWeights, int nTokens, int nTopk,
                                        int maxTokensPerRank, void* recvBuffer, void* workspace, uint32_t dispatchEpoch,
                                        int* sharedMem, Workload workload) {
  const int nWorkerBlocks = static_cast<int>(gridDim.x) - DispatchControlBlocks;
  if (static_cast<int>(blockIdx.x) == 0) {
    if (useSendProducedRouteCounts(workload)) {
      WorkspaceView workspaceView(workspace, nRanks, nExperts);
      resetSendProducedRouteCounts(workspaceView, nRanks, nExperts, dispatchEpoch);
    }
  } else if (static_cast<int>(blockIdx.x) > 0 && static_cast<int>(blockIdx.x) <= nWorkerBlocks) {
    if (useSendProducedRouteCounts(workload)) {
      WorkspaceView workspaceView(workspace, nRanks, nExperts);
      waitForSendProducedRouteCountReset(workspaceView, dispatchEpoch);
    }
    constexpr bool SignalPayloadReady = false;
    if constexpr (DataType == DispatchDataType::BF16 || DataType == DispatchDataType::FP16) {
      dispatchSendUnquantized<Hidden, DataType, SignalPayloadReady>(
          inputTokens, nExperts, transport.rank_, nRanks, topkIndices, topkWeights, nTokens, nTopk, maxTokensPerRank,
          recvBuffer, transport, workspace, dispatchEpoch, sharedMem, workload);
    } else {
      dispatchSendFp8<Hidden, DataType, ScaleBlockSize, SignalPayloadReady>(
          inputTokens, nExperts, transport.rank_, nRanks, topkIndices, topkWeights, nTokens, nTopk, maxTokensPerRank,
          recvBuffer, transport, workspace, dispatchEpoch, sharedMem);
    }
  } else if (static_cast<int>(blockIdx.x) == nWorkerBlocks + 1) {
    if (workload.dispatchProfileMode_ == DispatchProfileMode::SEND_NOTIFY_RANK_COUNTS) {
      dispatchRankMajorNotify(transport, nExperts, nRanks, topkIndices, nTokens, nTopk, recvBuffer, workspace,
                              dispatchEpoch, sharedMem);
    } else {
      dispatchNotify<false>(transport, nExperts, nRanks, topkIndices, nTokens, nTopk, recvBuffer, workspace,
                            dispatchEpoch, sharedMem, workload);
    }
  }
}

// ===========================================================================
// Token-major dispatch: OpenAI-style [num_tokens, num_topk, hidden] layout.
//
// Unlike rank-major (which deduplicates a token's top-k destinations that land
// on the same rank into ONE payload row and dynamically allocates a slot), token
// major keeps every (source token, top-k slot) as its OWN row at the fixed index
//   ((sourceRank * maxTokensPerRank + sourceToken) * numTopk + slot)
// on the destination rank of that slot's expert. No atomic slot allocation and no
// dedup are needed -- the index is deterministic from (sourceToken, slot). Each
// valid slot issues its own TMA store of the (shared, already-loaded) token row
// plus its top-k id/weight metadata. The per-rank completion count therefore
// equals the number of valid (token, slot) pairs routed to that rank (NOT the
// deduplicated rank count).
//
// NOTE: this is dispatch-only. Padding rows (slots routed to another rank, or
// tokens beyond numTokens) are left untouched here; the token-major combine
// (added separately) drives its reduction from the top-k id metadata, so padding
// content does not affect a correctly-written valid row.
// ===========================================================================
MSCCLPP_DEVICE_INLINE void countTokenMajorRoutes(int* rankTokenCounts, const int64_t* __restrict__ topkIndices,
                                                 int nTokens, int nTopk, int nRanks, int nExperts) {
  const int threadId = static_cast<int>(threadIdx.x);
  const int warpId = threadId / WARP_SIZE;
  const int laneId = get_lane_id();
  const int nLocalExperts = nExperts / nRanks;
  for (int rankIdx = threadId; rankIdx < nRanks; rankIdx += blockDim.x) rankTokenCounts[rankIdx] = 0;
  __syncthreads();
  for (int tokenIdx = warpId; tokenIdx < nTokens; tokenIdx += DispatchNWarps) {
    const int expert = laneId < nTopk ? static_cast<int>(topkIndices[tokenIdx * nTopk + laneId]) : -1;
    const int dstRank = expert >= 0 ? expert / nLocalExperts : -1;
    // No dedup: every valid (token, slot) pair is a distinct destination row.
    if (dstRank >= 0) atomicAdd_block(rankTokenCounts + dstRank, 1);
  }
  __syncthreads();
}

MSCCLPP_DEVICE_INLINE void dispatchTokenMajorNotify(const TransportView& transport, int nExperts, int nRanks,
                                                    const int64_t* __restrict__ topkIndices, int nTokens, int nTopk,
                                                    void* recvBuffer, void* workspace, uint32_t dispatchEpoch,
                                                    int* sharedMem) {
  WorkspaceView workspaceView(workspace, nRanks, nExperts);
  auto* rankTokenCounts = sharedMem;
  countTokenMajorRoutes(rankTokenCounts, topkIndices, nTokens, nTopk, nRanks, nExperts);
  writeRankMajorCounts(transport, rankTokenCounts, nRanks, recvBuffer, dispatchEpoch);
  Workload noDebug{};
  publishDispatchPayloads<false>(transport, rankTokenCounts, nRanks, workspaceView, noDebug);
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void issueTokenMajorTokenStore(void* output, const TransportView& transport, int destinationRank,
                                                     size_t destIndex, void* stagedToken) {
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(Bf16);
  void* destinationBuffer = transport.mappedBuffer(output, destinationRank);
  auto* destinationRow = reinterpret_cast<uint8_t*>(destinationBuffer) + destIndex * HiddenBytes;
  issueTmaStore(destinationRow, stagedToken, static_cast<uint32_t>(HiddenBytes));
}

MSCCLPP_DEVICE_INLINE void sendTokenMajorMetadata(const TransportView& transport, int* outputTopkIdx,
                                                  float* outputTopkWeights, int destinationRank, size_t destIndex,
                                                  int expert, float weight) {
  auto* destinationTopkIdx = reinterpret_cast<int*>(transport.mappedBuffer(outputTopkIdx, destinationRank));
  auto* destinationTopkWeights = reinterpret_cast<float*>(transport.mappedBuffer(outputTopkWeights, destinationRank));
  destinationTopkIdx[destIndex] = expert;
  destinationTopkWeights[destIndex] = weight;
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void dispatchSendTokenMajorBf16(void* output, int* outputTopkIdx, float* outputTopkWeights,
                                                      const void* inputTokens, int nExperts, int nRanks,
                                                      const int64_t* __restrict__ topkIndices,
                                                      const float* __restrict__ topkWeights, int nTokens, int nTopk,
                                                      int maxTokensPerRank, const TransportView& transport,
                                                      void* workspace, int nPayloadBlocks, int* sharedMem) {
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
  auto* sendTmaBarriers = reinterpret_cast<uint64_t*>(sharedTokenBase + DispatchMaxNWarpGroups * sharedTokenStride);
  WorkspaceView workspaceView(workspace, nRanks, nExperts);
  auto* completionCounts = aggregatedDispatchCompletionCounts(sharedMem, nRanks, warpGroupId);

  auto* stagedToken = sharedTokenBase + static_cast<size_t>(warpGroupId) * sharedTokenStride;
  auto* tmaBarrier = sendTmaBarriers + warpGroupId;
  const int tokenStride = nPayloadBlocks * nWarpGroups;
  const int firstTokenIdx = senderBlockIdx * nWarpGroups + warpGroupId;
  uint32_t sendTmaPhase = 0;
  if (firstTokenIdx < nTokens && laneId == 0) initTmaLoadBarrier(tmaBarrier);
  initAggregatedDispatchCompletions(completionCounts, nRanks, laneId);

  for (int tokenIdx = firstTokenIdx; tokenIdx < nTokens; tokenIdx += tokenStride) {
    const auto* inputData =
        reinterpret_cast<const mscclpp::bf16x8*>(inputTokens) + static_cast<size_t>(tokenIdx) * HiddenVectors;
    if (laneId == 0) {
      issueTmaLoadAndExpect(inputData, stagedToken, tmaBarrier, static_cast<uint32_t>(HiddenBytes));
    }

    // Each lane owns one top-k slot; its own row lands on the slot's expert rank.
    const int expert = laneId < nTopk ? static_cast<int>(topkIndices[tokenIdx * nTopk + laneId]) : -1;
    const int dstRank = expert >= 0 ? expert / nLocalExperts : -1;
    const float weight =
        laneId < nTopk ? (topkWeights == nullptr ? 1.0f : topkWeights[tokenIdx * nTopk + laneId]) : 0.0f;
    const size_t destIndex = (static_cast<size_t>(transport.rank_) * maxTokensPerRank + tokenIdx) * nTopk + laneId;

    if (laneId == 0) waitTmaLoad(tmaBarrier, sendTmaPhase);
    __syncwarp();
    fenceProxyAsyncSharedCta();

    if (dstRank >= 0) {
      issueTokenMajorTokenStore<Hidden>(output, transport, dstRank, destIndex, stagedToken);
      sendTokenMajorMetadata(transport, outputTopkIdx, outputTopkWeights, dstRank, destIndex, expert, weight);
      waitBulkGroup();
      atomicAdd_block(completionCounts + dstRank, 1);
    }
    __syncwarp();
  }
  flushAggregatedDispatchCompletions(workspaceView, completionCounts, nRanks, laneId);
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void dispatchSendTokenMajor(void* output, int* outputTopkIdx, float* outputTopkWeights,
                                                  const void* inputTokens, const TransportView& transport, int nExperts,
                                                  int nRanks, const int64_t* __restrict__ topkIndices,
                                                  const float* __restrict__ topkWeights, int nTokens, int nTopk,
                                                  int maxTokensPerRank, void* recvBuffer, void* workspace,
                                                  uint32_t dispatchEpoch, int* sharedMem) {
  const int nWorkerBlocks = static_cast<int>(gridDim.x) - DispatchControlBlocks;
  if (static_cast<int>(blockIdx.x) > 0 && static_cast<int>(blockIdx.x) <= nWorkerBlocks) {
    dispatchSendTokenMajorBf16<Hidden>(output, outputTopkIdx, outputTopkWeights, inputTokens, nExperts, nRanks,
                                       topkIndices, topkWeights, nTokens, nTopk, maxTokensPerRank, transport, workspace,
                                       nWorkerBlocks, sharedMem);
  } else if (static_cast<int>(blockIdx.x) == nWorkerBlocks + 1) {
    dispatchTokenMajorNotify(transport, nExperts, nRanks, topkIndices, nTokens, nTopk, recvBuffer, workspace,
                             dispatchEpoch, sharedMem);
  }
}

MSCCLPP_DEVICE_INLINE void dispatchRecvTokenMajor(int* outputCount, const TransportView& transport, int nExperts,
                                                  int nRanks, void* recvBuffer, void* workspace, uint32_t dispatchEpoch,
                                                  int* sharedMem) {
  const int sourceRank = static_cast<int>(blockIdx.x);
  if (sourceRank >= nRanks) return;
  auto* rankTokenCounts = reinterpret_cast<mscclpp::LL8Packet*>(recvBuffer);
  int nStores = 0;
  if (threadIdx.x == 0) {
    nStores = static_cast<int>(rankTokenCounts[sourceRank].read(dispatchEpoch, -1));
    outputCount[sourceRank] = nStores;
    sharedMem[0] = nStores;
  }
  __syncthreads();
  nStores = sharedMem[0];
  WorkspaceView workspaceView(workspace, nRanks, nExperts);
  if (threadIdx.x == 0 && nStores > 0) {
    if (transport.isSelf(sourceRank)) {
      workspaceView.dispatchLocalPayloadReady_->acquire();
    } else {
      transport.baseMemoryChannels_[sourceRank].wait(-1);
    }
  }
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void dispatchSendRankMajor(void* output, int* outputTopkIdx, float* outputTopkWeights,
                                                 const void* inputTokens, const TransportView& transport, int nExperts,
                                                 int nRanks, const int64_t* __restrict__ topkIndices,
                                                 const float* __restrict__ topkWeights, int nTokens, int nTopk,
                                                 int invalidTokenExpertId, int maxTokensPerRank, void* recvBuffer,
                                                 void* workspace, uint32_t dispatchEpoch, int* sharedMem) {
  const int nWorkerBlocks = static_cast<int>(gridDim.x) - DispatchControlBlocks;
  if (static_cast<int>(blockIdx.x) > 0 && static_cast<int>(blockIdx.x) <= nWorkerBlocks) {
    dispatchSendRankMajorBf16<Hidden>(output, outputTopkIdx, outputTopkWeights, inputTokens, nExperts, nRanks,
                                      topkIndices, topkWeights, nTokens, nTopk, invalidTokenExpertId, maxTokensPerRank,
                                      transport, workspace, nWorkerBlocks, sharedMem);
  } else if (static_cast<int>(blockIdx.x) == nWorkerBlocks + 1) {
    dispatchRankMajorNotify(transport, nExperts, nRanks, topkIndices, nTokens, nTopk, recvBuffer, workspace,
                            dispatchEpoch, sharedMem);
  }
}

MSCCLPP_DEVICE_INLINE int proportionalTaskBoundary(int nTokens, int nTasks, int nTotalTokens) {
  return nTotalTokens == 0 ? 0 : static_cast<int>(static_cast<int64_t>(nTokens) * nTasks / nTotalTokens);
}

template <DispatchLayout Layout>
MSCCLPP_DEVICE_INLINE void dispatchRecvScheduler(int64_t* outputLayout, int* outputCount,
                                                 const TransportView& transport, int nExperts, int nRanks,
                                                 void* recvBuffer, void* workspace, uint32_t dispatchEpoch,
                                                 int* sharedMem, bool buildExpertLayout, Workload workload) {
  const int threadId = static_cast<int>(threadIdx.x);
  const int warpId = threadId / WARP_SIZE;
  const int laneId = get_lane_id();
  const int nWorkerBlocks = static_cast<int>(gridDim.x) - low_latency::DispatchControlBlocks;
  auto* rankTokenCounts = reinterpret_cast<mscclpp::LL8Packet*>(recvBuffer);
  const int nLocalExperts = nExperts / nRanks;
  WorkspaceView workspaceView(workspace, nRanks, nExperts);
  debugProgress(workload, 2000, dispatchEpoch, nRanks, nExperts);

  for (int expertIdx = threadId; expertIdx < nExperts; expertIdx += static_cast<int>(blockDim.x)) {
    workspaceView.dispatchExpertCopiedCounts_[expertIdx] = 0;
  }
  __syncthreads();

  const int nRankWarps = (nRanks + WARP_SIZE - 1) / WARP_SIZE;
  const int requestedNLayoutWarps = buildExpertLayout ? (nLocalExperts + WARP_SIZE - 1) / WARP_SIZE : 0;
  const int maxNLayoutWarps = DispatchNWarps - nRankWarps;
  const int nLayoutWarps = requestedNLayoutWarps < maxNLayoutWarps ? requestedNLayoutWarps : maxNLayoutWarps;

  if (warpId < nRankWarps) {
    const int sourceRank = threadId;
    const int nRankTokens =
        sourceRank < nRanks ? static_cast<int>(rankTokenCounts[sourceRank].read(dispatchEpoch, -1)) : 0;
    if (sourceRank < nRanks) debugProgress(workload, 2001, sourceRank, nRankTokens, dispatchEpoch);
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
    if (threadId == 0) debugProgress(workload, 2002, nTotalTokens, nActiveRanks, nTasks);

    syncNamedBarrier(DispatchSchedulerReadyBarrier, (nRankWarps + nLayoutWarps) * WARP_SIZE);
    if (threadId == 0) {
      memory_fence();
      mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(workspaceView.dispatchTasksReadyEpoch_, dispatchEpoch,
                                                           mscclpp::memoryOrderRelease);
      debugProgress(workload, 2003, nTasks, dispatchEpoch, 0);
    }

    if (sourceRank < nRanks && nRankTokens > 0) {
      const int nReadySignals = 1;
      debugProgress(workload, 2004, sourceRank, nRankTokens, nReadySignals);
      debugProgressWord(workload, 16 + sourceRank, 1);
      if (transport.isSelf(sourceRank)) {
        for (int signalIdx = 0; signalIdx < nReadySignals; ++signalIdx) {
          workspaceView.dispatchLocalPayloadReady_->acquire();
        }
      } else {
        for (int signalIdx = 0; signalIdx < nReadySignals; ++signalIdx) {
          transport.baseMemoryChannels_[sourceRank].wait(-1);
        }
      }
      mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(workspaceView.dispatchRankReadyEpochs_ + sourceRank,
                                                           dispatchEpoch, mscclpp::memoryOrderRelease);
      debugProgressWord(workload, 16 + sourceRank, 2);
      debugProgress(workload, 2005, sourceRank, nRankTokens, nReadySignals);
    }
  } else if (warpId < nRankWarps + nLayoutWarps) {
    auto* expertTokenCounts = reinterpret_cast<mscclpp::LL8Packet*>(recvBuffer) + nRanks;
    const int layoutThreadId = (warpId - nRankWarps) * WARP_SIZE + laneId;
    const int nLayoutThreads = nLayoutWarps * WARP_SIZE;
    for (int localExpertIdx = layoutThreadId; localExpertIdx < nLocalExperts; localExpertIdx += nLayoutThreads) {
      int outputOffset = 0;
      for (int sourceRank = 0; sourceRank < nRanks; ++sourceRank) {
        const int nExpertTokens =
            static_cast<int>(expertTokenCounts[sourceRank * nLocalExperts + localExpertIdx].read(dispatchEpoch, -1));
        outputLayout[localExpertIdx * nRanks + sourceRank] = pack2<int, int64_t>(nExpertTokens, outputOffset);
        outputOffset += nExpertTokens;
      }
      outputCount[localExpertIdx] = outputOffset;
    }
    syncNamedBarrier(DispatchSchedulerReadyBarrier, (nRankWarps + nLayoutWarps) * WARP_SIZE);
  }
}

MSCCLPP_DEVICE_INLINE bool acquireRecvTask(RecvTask& task, WorkspaceView& workspaceView, uint32_t dispatchEpoch,
                                           int* sharedMem, Workload workload) {
  auto* sharedTask = reinterpret_cast<RecvTask*>(sharedMem);
  const int taskIdx = static_cast<int>(blockIdx.x) - 1;
  if (threadIdx.x == 0) {
    debugProgress(workload, 3000, taskIdx, dispatchEpoch,
                  mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(workspaceView.dispatchTasksReadyEpoch_,
                                                                      mscclpp::memoryOrderAcquire));
    while (mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(workspaceView.dispatchTasksReadyEpoch_,
                                                               mscclpp::memoryOrderAcquire) != dispatchEpoch);
    debugProgress(workload, 3001, taskIdx, *workspaceView.dispatchNumRecvTasks_, dispatchEpoch);
    if (taskIdx < *workspaceView.dispatchNumRecvTasks_) {
      task = workspaceView.dispatchRecvTasks_[taskIdx];
      debugProgress(workload, 3002, taskIdx, task.sourceRank_,
                    mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(
                        workspaceView.dispatchRankReadyEpochs_ + task.sourceRank_, mscclpp::memoryOrderAcquire));
      debugProgressWord(workload, 144, taskIdx);
      debugProgressWord(workload, 145, task.sourceRank_);
      debugProgressWord(workload, 146,
                        mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(
                            workspaceView.dispatchRankReadyEpochs_ + task.sourceRank_, mscclpp::memoryOrderAcquire));
      while (mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(
                 workspaceView.dispatchRankReadyEpochs_ + task.sourceRank_, mscclpp::memoryOrderAcquire) !=
             dispatchEpoch);
      debugProgressWord(workload, 146,
                        mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(
                            workspaceView.dispatchRankReadyEpochs_ + task.sourceRank_, mscclpp::memoryOrderAcquire));
      debugProgress(workload, 3003, taskIdx, task.sourceRank_, dispatchEpoch);
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
                                                 void* workspace, uint32_t dispatchEpoch, int* sharedMem) {
  const int sourceRank = static_cast<int>(blockIdx.x);
  if (sourceRank >= nRanks) return;
  auto* rankTokenCounts = reinterpret_cast<mscclpp::LL8Packet*>(recvBuffer);
  if (threadIdx.x == 0) {
    const int nRankTokens = static_cast<int>(rankTokenCounts[sourceRank].read(dispatchEpoch, -1));
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
  if (threadIdx.x == 0 && nRankTokens > 0) {
    if (transport.isSelf(sourceRank)) {
      workspaceView.dispatchLocalPayloadReady_->acquire();
    } else {
      transport.baseMemoryChannels_[sourceRank].wait(-1);
    }
  }
}

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize, DispatchLayout Layout>
MSCCLPP_DEVICE_INLINE bool dispatchRecvExpertMajorOutput(
    void* output, void* outputScales, int* outputSrcInfo, int64_t* outputLayout,
    const DispatchPayloadView<DataType>& payloadView, void* stagedPayload, void* sourcePayload, int localExpertIdx,
    int sourceRank, int sourceTokenIdx, int nLocalExperts, int nRanks, int nTopk, int maxTokensPerRank,
    WorkspaceView& workspaceView, uint8_t* sharedTile, bool skipOutputStore) {
  using OutputType = DispatchElementType<DataType>;
  constexpr size_t OutputBytes = static_cast<size_t>(Hidden) * sizeof(OutputType);
  constexpr int NumScales = DataType == DispatchDataType::FP8_E4M3 ? Hidden / ScaleBlockSize : 0;
  const int laneId = get_lane_id();
  const bool compactExpertMajor = Layout == DispatchLayout::KI_RAGGED;
  const int nOutputSlotsPerExpert = nRanks * maxTokensPerRank;
  int outputTokenIdx = -1;
  int outputRowIdx = -1;
  int combineInputOffset = -1;

  if (localExpertIdx >= 0) {
    int expertTokenCount;
    int outputOffset;
    unpack2(outputLayout[localExpertIdx * nRanks + sourceRank], expertTokenCount, outputOffset);
    const int copiedTokenIdx =
        atomicAdd(workspaceView.dispatchExpertCopiedCounts_ + sourceRank * nLocalExperts + localExpertIdx, 1);
    EP_DEVICE_ASSERT(copiedTokenIdx < expertTokenCount);
    if constexpr (Layout != DispatchLayout::KI_RAGGED) {
      // Existing expert-major callers depend on per-source counters being reset
      // before the next dispatch. KI_RAGGED resets after all recv workers finish;
      // doing it here races when one source rank is split across several tasks.
      if (copiedTokenIdx == expertTokenCount - 1) {
      workspaceView.dispatchExpertCopiedCounts_[sourceRank * nLocalExperts + localExpertIdx] = 0;
      }
    }
    outputTokenIdx = outputOffset + copiedTokenIdx;
    if (compactExpertMajor) {
      int expertBase = 0;
      for (int expertIdx = 0; expertIdx < localExpertIdx; ++expertIdx) {
        int nLastRankTokens;
        int lastRankOffset;
        unpack2(outputLayout[expertIdx * nRanks + nRanks - 1], nLastRankTokens, lastRankOffset);
        expertBase += lastRankOffset + nLastRankTokens;
      }
      outputRowIdx = expertBase + outputTokenIdx;
    } else {
      outputRowIdx = localExpertIdx * nOutputSlotsPerExpert + outputTokenIdx;
    }
    if constexpr (Layout == DispatchLayout::KI_RAGGED) {
      EP_DEVICE_ASSERT(outputRowIdx >= 0 && outputRowIdx < nRanks * maxTokensPerRank * nTopk);
    }
    outputSrcInfo[outputRowIdx] = sourceTokenIdx;
    combineInputOffset = outputRowIdx;
  }

  if constexpr (DataType == DispatchDataType::FP8_E4M3) {
    using ScaleType = DispatchScaleType<DataType>;
    const auto* sourceScales = payloadView.scaleFactors(stagedPayload);
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
  if constexpr (Layout != DispatchLayout::KI_RAGGED) {
    if (laneId < nTopk) payloadView.topKIndices(sourcePayload)[laneId] = combineInputOffset;
  }

  fenceProxyAsyncSharedCta();

  if (localExpertIdx < 0 || skipOutputStore) return false;
  const size_t outputRow = static_cast<size_t>(outputRowIdx);
  auto* outputData = reinterpret_cast<uint8_t*>(output) + outputRow * OutputBytes;
  issueTmaStore(outputData, sharedTile, static_cast<uint32_t>(OutputBytes));
  return true;
}

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize, DispatchLayout Layout>
MSCCLPP_DEVICE_INLINE void dispatchRecvWorker(void* output, void* outputScales, int* outputSrcInfo,
                                              int64_t* outputLayout, int nExperts, int rank, int nRanks, int nTopk,
                                              int maxTokensPerRank, void* recvBuffer, void* workspace,
                                              uint32_t dispatchEpoch, int* sharedMem, bool skipOutputStore,
                                              Workload workload) {
#if defined(__CUDA_ARCH__)
  static_assert(__CUDA_ARCH__ >= 900, "TMA recv requires SM90 or newer");
#endif
  const int threadId = static_cast<int>(threadIdx.x);
  const int warpId = threadId / WARP_SIZE;
  const int laneId = get_lane_id();
  WorkspaceView workspaceView(workspace, nRanks, nExperts);
  RecvTask task;
  if (!acquireRecvTask(task, workspaceView, dispatchEpoch, sharedMem, workload)) return;
  debugProgress(workload, 3100, task.sourceRank_, task.tokenBegin_, task.tokenEnd_);
  debugProgressWord(workload, 136, 1);
  debugProgressWord(workload, 137, task.sourceRank_);
  debugProgressWord(workload, 138, task.tokenBegin_);
  debugProgressWord(workload, 139, task.tokenEnd_);

  const int nLocalExperts = nExperts / nRanks;
  const int sourceRank = task.sourceRank_;
  const int globalExpertBase = rank * nLocalExperts;
  const int globalExpertEnd = globalExpertBase + nLocalExperts;
  const DispatchPayloadView<DataType> payloadView(Hidden, nTopk, ScaleBlockSize);
  const size_t payloadStride = dispatchPayloadStride<DataType>(Hidden, nTopk, ScaleBlockSize);
  const int nRecvTmaWorkers = tmaWorkerCountForTileBytes(payloadStride, DispatchMaxNRecvTmaWorkers);
  if (warpId >= nRecvTmaWorkers) return;
  const size_t tileBytes = payloadStride;
  auto* sourceInfoPackets =
      reinterpret_cast<mscclpp::LL8Packet*>(reinterpret_cast<uint8_t*>(recvBuffer) +
                                            dispatchSourceInfoOffset(nRanks, nExperts));
  auto* sourcePayloadBase = reinterpret_cast<uint8_t*>(recvBuffer) +
                            dispatchMetadataBytes(nRanks, nExperts, maxTokensPerRank) +
                            static_cast<size_t>(sourceRank) * maxTokensPerRank * payloadStride;
  auto* tmaTiles = reinterpret_cast<uint8_t*>(sharedMem) + dispatchSharedControlBytes(nRanks);
  auto* sharedTile = tmaTiles + static_cast<size_t>(warpId) * tileBytes;
  auto* tmaBarriers = reinterpret_cast<uint64_t*>(tmaTiles + static_cast<size_t>(nRecvTmaWorkers) * tileBytes);
  auto* tmaBarrier = tmaBarriers + warpId;
  bool hasPendingStore = false;
  uint32_t recvTmaPhase = 0;
  if (laneId == 0) initTmaLoadBarrier(tmaBarrier);
  if (laneId == 0) debugProgressWord(workload, 136, 2);

  for (int sourceTokenSlot = task.tokenBegin_ + warpId; sourceTokenSlot < task.tokenEnd_;
       sourceTokenSlot += nRecvTmaWorkers) {
    if (hasPendingStore) {
      if (laneId == 0) debugProgressWord(workload, 136, 3);
      waitBulkGroupRead();
      if (laneId == 0) debugProgressWord(workload, 136, 4);
    }
    __syncwarp();
    if (laneId == 0) {
      debugProgressWord(workload, 136, 5);
      debugProgressWord(workload, 140, sourceTokenSlot);
      debugProgressWord(workload, 141, warpId);
    }

    auto* sourcePayload = sourcePayloadBase + static_cast<size_t>(sourceTokenSlot) * payloadStride;
    if (laneId == 0) {
      issueTmaLoadAndExpect(sourcePayload, sharedTile, tmaBarrier, static_cast<uint32_t>(payloadView.numBytes_));
      debugProgressWord(workload, 136, 6);
    }
    if (laneId == 0) {
      waitTmaLoad(tmaBarrier, recvTmaPhase);
      debugProgressWord(workload, 136, 7);
    }
    __syncwarp();
    if (laneId == 0) debugProgressWord(workload, 136, 8);

    auto* stagedPayload = sharedTile;
    const int routedExpertIdx = laneId < nTopk ? payloadView.topKIndices(stagedPayload)[laneId] : -1;
    const int localExpertIdx = routedExpertIdx >= globalExpertBase && routedExpertIdx < globalExpertEnd
                                   ? routedExpertIdx - globalExpertBase
                                   : -1;
    int sourceTokenGlobalIdx = 0;
    int sidebandSourceTokenGlobalIdx = 0;
    int payloadSourceTokenGlobalIdx = 0;
    if (laneId == 0) {
      if constexpr (Layout == DispatchLayout::KI_RAGGED) {
        sidebandSourceTokenGlobalIdx = static_cast<int>(
            sourceInfoPackets[static_cast<size_t>(sourceRank) * maxTokensPerRank + sourceTokenSlot].read(
                dispatchEpoch, -1));
        sourceTokenGlobalIdx = sidebandSourceTokenGlobalIdx;
        if (sourceTokenGlobalIdx < sourceRank * maxTokensPerRank ||
            sourceTokenGlobalIdx >= (sourceRank + 1) * maxTokensPerRank) {
          payloadSourceTokenGlobalIdx = *payloadView.srcTokenGlobalIdx(stagedPayload);
          if (payloadSourceTokenGlobalIdx >= sourceRank * maxTokensPerRank &&
              payloadSourceTokenGlobalIdx < (sourceRank + 1) * maxTokensPerRank) {
            sourceTokenGlobalIdx = payloadSourceTokenGlobalIdx;
          }
        }
      } else {
        sourceTokenGlobalIdx = *payloadView.srcTokenGlobalIdx(stagedPayload);
      }
      debugProgressWord(workload, 136, 9);
    }
    const int sourceTokenIdx = warpBroadcast(sourceTokenGlobalIdx - sourceRank * maxTokensPerRank, 0);
    if (laneId == 0) {
      debugProgressWord(workload, 142, sourceTokenIdx);
    }
    if (laneId < nTopk) {
      debugProgressWord(workload, 143, localExpertIdx >= 0 ? static_cast<uint64_t>(localExpertIdx) : 0xffffffffull);
    }
    hasPendingStore = dispatchRecvExpertMajorOutput<Hidden, DataType, ScaleBlockSize, Layout>(
        output, outputScales, outputSrcInfo, outputLayout, payloadView, stagedPayload, sourcePayload, localExpertIdx,
        sourceRank, sourceTokenIdx, nLocalExperts, nRanks, nTopk, maxTokensPerRank, workspaceView, sharedTile,
        skipOutputStore);
    if (laneId == 0) debugProgressWord(workload, 136, hasPendingStore ? 10 : 11);
  }

  if (hasPendingStore) {
    if (laneId == 0) debugProgressWord(workload, 136, 12);
    waitBulkGroup();
    if (laneId == 0) debugProgressWord(workload, 136, 13);
  }
}

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize, DispatchLayout Layout>
__global__ __launch_bounds__(DispatchNThreads,
                             1) void dispatchKernel(void* output, void* outputScales, int* outputSrcInfo,
                                                    int* outputTopkIdx, float* outputTopkWeights, int64_t* outputLayout,
                                                    int* outputCount, const int64_t* __restrict__ topkIndices,
                                                    const float* __restrict__ topkWeights, const void* inputTokens,
                                                    Workload workload, void* recvBuffer, CommContext comm,
                                                    void* workspace) {
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
  const uint32_t dispatchEpoch = workload.epoch_;
  debugProgress(workload, 100, dispatchEpoch, static_cast<uint64_t>(blockIdx.x), static_cast<uint64_t>(Layout));
  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    static_assert(DataType == DispatchDataType::BF16);
    dispatchSendRankMajor<Hidden>(output, outputTopkIdx, outputTopkWeights, inputTokens, transport, nExperts, nRanks,
                                  topkIndices, topkWeights, nTokens, nTopk, invalidTokenExpertId, maxTokensPerRank,
                                  recvBuffer, workspace, dispatchEpoch, sharedMem);
  } else if constexpr (Layout == DispatchLayout::TOKEN_MAJOR) {
    static_assert(DataType == DispatchDataType::BF16);
    dispatchSendTokenMajor<Hidden>(output, outputTopkIdx, outputTopkWeights, inputTokens, transport, nExperts, nRanks,
                                   topkIndices, topkWeights, nTokens, nTopk, maxTokensPerRank, recvBuffer, workspace,
                                   dispatchEpoch, sharedMem);
  } else {
    debugProgress(workload, 110, dispatchEpoch, static_cast<uint64_t>(blockIdx.x), 0);
    dispatchSend<Hidden, DataType, ScaleBlockSize, Layout>(inputTokens, transport, nExperts, nRanks, topkIndices,
                                                           topkWeights, nTokens, nTopk, maxTokensPerRank, recvBuffer,
                                                           workspace, dispatchEpoch, sharedMem, workload);
    debugProgress(workload, 111, dispatchEpoch, static_cast<uint64_t>(blockIdx.x), 0);
  }

  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    if (static_cast<int>(blockIdx.x) < nRanks) {
      dispatchRecvRankMajor(outputTopkIdx, outputTopkWeights, outputCount, transport, nExperts, nRanks, nTopk,
                            maxTokensPerRank, invalidTokenExpertId, recvBuffer, workspace, dispatchEpoch, sharedMem);
    }
  } else if constexpr (Layout == DispatchLayout::TOKEN_MAJOR) {
    if (static_cast<int>(blockIdx.x) < nRanks) {
      dispatchRecvTokenMajor(outputCount, transport, nExperts, nRanks, recvBuffer, workspace, dispatchEpoch, sharedMem);
    }
  } else {
    const DispatchProfileMode profileMode = workload.dispatchProfileMode_;
    const bool buildExpertLayout = profileMode != DispatchProfileMode::SEND_NOTIFY_RANK_COUNTS &&
                                   profileMode != DispatchProfileMode::SEND_NOTIFY_RANK_WAIT;
    const bool runRecvWorkers = profileMode != DispatchProfileMode::SEND_NOTIFY_RANK_COUNTS &&
                                profileMode != DispatchProfileMode::SEND_NOTIFY_RANK_WAIT &&
                                profileMode != DispatchProfileMode::SEND_NOTIFY_LAYOUT;
    const bool skipOutputStore = profileMode == DispatchProfileMode::SKIP_OUTPUT_STORE;
    if (static_cast<int>(blockIdx.x) == 0) {
      dispatchRecvScheduler<Layout>(outputLayout, outputCount, transport, nExperts, nRanks, recvBuffer, workspace,
                                    dispatchEpoch, sharedMem, buildExpertLayout, workload);
    } else if (runRecvWorkers && static_cast<int>(blockIdx.x) <= nWorkerBlocks) {
      dispatchRecvWorker<Hidden, DataType, ScaleBlockSize, Layout>(output, outputScales, outputSrcInfo, outputLayout,
                                                                   nExperts, comm.rank_, nRanks, nTopk,
                                                                   maxTokensPerRank, recvBuffer, workspace,
                                                                   dispatchEpoch, sharedMem, skipOutputStore, workload);
    }
  }

  if constexpr (Layout == DispatchLayout::KI_RAGGED) {
    // KI_RAGGED hands compact rows directly to following non-mscclpp kernels.
    // Do not publish the dispatch epoch until all recv workers finish writing.
    debugProgress(workload, 4000, dispatchEpoch, static_cast<uint64_t>(blockIdx.x), 0);
    workspaceView.combineSyncer_->sync(gridDim.x);
    debugProgress(workload, 4001, dispatchEpoch, static_cast<uint64_t>(blockIdx.x), 0);
    if (blockIdx.x == 0) {
      for (int counterIdx = static_cast<int>(threadIdx.x); counterIdx < nExperts;
           counterIdx += static_cast<int>(blockDim.x)) {
        workspaceView.dispatchExpertCopiedCounts_[counterIdx] = 0;
      }
      __syncthreads();
    }
  }

}

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize, DispatchLayout Layout>
inline void dispatchHiddenMode(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                               float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                               const int64_t* topkIdx, const float* topkWeights, const low_latency::Workload& workload,
                               void* recvBuffer, const low_latency::CommContext& comm, void* workspace, int numBlocks,
                               cudaStream_t stream) {
  static_assert(Hidden == 2048 || Hidden == 4096 || Hidden == 4352 || Hidden == 5120 || Hidden == 6656 || Hidden == 7168 ||
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
  } else if constexpr (Layout == DispatchLayout::KI_RAGGED) {
    switch (workload.dispatchDataType_) {
      case DispatchDataType::BF16:
        return dispatchHiddenMode<Hidden, DispatchDataType::BF16, 0, Layout>(
            output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
            topkIdx, topkWeights, workload, recvBuffer, comm, workspace, numBlocks, stream);
      case DispatchDataType::FP16:
        return dispatchHiddenMode<Hidden, DispatchDataType::FP16, 0, Layout>(
            output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
            topkIdx, topkWeights, workload, recvBuffer, comm, workspace, numBlocks, stream);
      case DispatchDataType::FP8_E4M3:
        break;
    }
    EP_HOST_ASSERT(false && "unsupported KI_RAGGED dispatch data type");
  } else if constexpr (Layout == DispatchLayout::TOKEN_MAJOR) {
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
  if (workload.outputLayout_ == DispatchLayout::KI_RAGGED) {
    return dispatchHidden<Hidden, DispatchLayout::KI_RAGGED>(
        output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
        topkIdx, topkWeights, workload, recvBuffer, comm, workspace, numBlocks, stream);
  }
  if (workload.outputLayout_ == DispatchLayout::RANK_MAJOR) {
    return dispatchHidden<Hidden, DispatchLayout::RANK_MAJOR>(
        output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
        topkIdx, topkWeights, workload, recvBuffer, comm, workspace, numBlocks, stream);
  }
  if (workload.outputLayout_ == DispatchLayout::TOKEN_MAJOR) {
    return dispatchHidden<Hidden, DispatchLayout::TOKEN_MAJOR>(
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
                 workload.outputLayout_ == DispatchLayout::KI_RAGGED ||
                 workload.outputLayout_ == DispatchLayout::RANK_MAJOR ||
                 workload.outputLayout_ == DispatchLayout::TOKEN_MAJOR);
  EP_HOST_ASSERT(isSupportedDispatchDataType(workload.dispatchDataType_));
  EP_HOST_ASSERT(workload.dispatchDataType_ != DispatchDataType::FP8_E4M3 || outputScales != nullptr);
  EP_HOST_ASSERT(outputSrcInfo != nullptr || workload.outputLayout_ == DispatchLayout::RANK_MAJOR ||
                 workload.outputLayout_ == DispatchLayout::TOKEN_MAJOR);
  EP_HOST_ASSERT(outputCount != nullptr);
  EP_HOST_ASSERT(outputLayout != nullptr || workload.outputLayout_ == DispatchLayout::RANK_MAJOR ||
                 workload.outputLayout_ == DispatchLayout::TOKEN_MAJOR);
  if (workload.outputLayout_ == DispatchLayout::RANK_MAJOR || workload.outputLayout_ == DispatchLayout::TOKEN_MAJOR) {
    EP_HOST_ASSERT(outputTopkIdx != nullptr);
    EP_HOST_ASSERT(outputTopkWeights != nullptr);
  }
  if (workload.outputLayout_ == DispatchLayout::RANK_MAJOR || workload.outputLayout_ == DispatchLayout::TOKEN_MAJOR) {
    EP_HOST_ASSERT(workload.dispatchDataType_ == DispatchDataType::BF16);
  }
  if (workload.outputLayout_ == DispatchLayout::KI_RAGGED) {
    EP_HOST_ASSERT(workload.dispatchDataType_ == DispatchDataType::BF16 ||
                   workload.dispatchDataType_ == DispatchDataType::FP16);
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
    case 5120:
      return dispatchLayout<5120>(output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout,
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
