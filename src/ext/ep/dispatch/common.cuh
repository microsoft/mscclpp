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
                                                           int nTopk, const ExpertMap& map, int maxTokensPerRank,
                                                           int laneId) {
  const int routedExpertIdx = laneId < nTopk ? static_cast<int>(topkIndices[tokenIdx * nTopk + laneId]) : -1;
  const int dstRank = map.leaderRank(routedExpertIdx);
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
                                                 int tokenIdx, int nTopk, const ExpertMap& map, int maxTokensPerRank,
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
      const bool isLocal = map.rankOwnsExpert(destinationRank, candidateExpert);
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

template <int Hidden>
MSCCLPP_DEVICE_INLINE void dispatchSendRankMajorBf16(void* output, int* outputTopkIdx, float* outputTopkWeights,
                                                     const void* inputTokens, const ExpertMap& map,
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
  const int nRanks = map.numRanks();
  const size_t sharedTokenStride = dispatchPayloadStride<DispatchDataType::BF16>(Hidden, nTopk, 0);
  auto* sharedTokenBase = reinterpret_cast<uint8_t*>(sharedMem) + dispatchSharedControlBytes(nRanks);
  auto* sendBulkBarriers =
      reinterpret_cast<mscclpp::BulkBarrier*>(sharedTokenBase + DispatchMaxNWarpGroups * sharedTokenStride);
  WorkspaceView workspaceView(workspace, map);

  auto* stagedToken = sharedTokenBase + static_cast<size_t>(warpGroupId) * sharedTokenStride;
  auto* bulkBarrier = sendBulkBarriers + warpGroupId;
  const int tokenStride = nPayloadBlocks * nWarpGroups;
  const int firstTokenIdx = senderBlockIdx * nWarpGroups + warpGroupId;
  uint32_t sendBulkPhase = 0;
  if (firstTokenIdx < nTokens && laneId == 0) bulkBarrier->init();

  for (int tokenIdx = firstTokenIdx; tokenIdx < nTokens; tokenIdx += tokenStride) {
    const auto* inputData =
        reinterpret_cast<const mscclpp::bf16x8*>(inputTokens) + static_cast<size_t>(tokenIdx) * HiddenVectors;
    if (laneId == 0) {
      bulkBarrier->arriveAndExpect(static_cast<uint32_t>(HiddenBytes));
      mscclpp::bulkLoad(stagedToken, inputData, static_cast<uint32_t>(HiddenBytes), *bulkBarrier);
    }
    const RankMajorRoute route =
        prepareRankMajorRoute(workspaceView, topkIndices, tokenIdx, nTopk, map, maxTokensPerRank, laneId);
    if (laneId == 0) bulkBarrier->wait(sendBulkPhase);
    __syncwarp();
    mscclpp::bulkFence();
    const int completionRank = route.dstRank >= 0 && route.isLeader ? route.dstRank : -1;
    if (completionRank >= 0) {
      issueRankMajorTokenStore<Hidden>(output, transport, route.destinationSlot, maxTokensPerRank, stagedToken,
                                       route.dstRank);
    }
    sendRankMajorMetadata(transport, outputTopkIdx, outputTopkWeights, topkIndices, topkWeights, route, tokenIdx, nTopk,
                          map, maxTokensPerRank, invalidTokenExpertId);
    completeRankMajorTokenStore(workspaceView, completionRank);
    if (tokenIdx + tokenStride < nTokens) __syncwarp();
  }
}

/// Fill the staged payload's routing metadata. Independent of the destination,
/// so it is done once even when the token is replicated across an EP group.
template <DispatchDataType DataType>
MSCCLPP_DEVICE_INLINE void stageDispatchPayloadMetadata(const DispatchPayloadView<DataType>& payloadView,
                                                        void* stagedPayload,
                                                        const int64_t* __restrict__ topkIndices,
                                                        const float* __restrict__ topkWeights, int tokenIdx, int nTopk,
                                                        int maxTokensPerRank, int rank, int laneId) {
  const int routedExpertIdx = laneId < nTopk ? static_cast<int>(topkIndices[tokenIdx * nTopk + laneId]) : -1;
  if (laneId < nTopk) {
    payloadView.topKIndices(stagedPayload)[laneId] = routedExpertIdx;
    payloadView.topKValues(stagedPayload)[laneId] =
        topkWeights == nullptr ? 1.0f : topkWeights[tokenIdx * nTopk + laneId];
  }
  if (laneId == 0) {
    *payloadView.srcTokenGlobalIdx(stagedPayload) = rank * maxTokensPerRank + tokenIdx;
  }
}

/// Reserve one destination slot per distinct destination rank of send copy
/// @p copyIdx. Slots are rank-deduplicated inside the warp, so a token routed
/// to two experts of the same group is sent once.
template <DispatchDataType DataType>
MSCCLPP_DEVICE_INLINE void reserveDispatchPayloadSlots(const DispatchPayloadView<DataType>& payloadView,
                                                       const void* stagedPayload, int* destinationSlots,
                                                       WorkspaceView& workspaceView, int nTopk, const ExpertMap& map,
                                                       int copyIdx, int maxTokensPerRank, int laneId) {
  const int routedExpertIdx = laneId < nTopk ? payloadView.topKIndices(stagedPayload)[laneId] : -1;
  const int dstRank = map.destinationRank(routedExpertIdx, copyIdx);
  const bool firstLaneForRank = isFirstLaneForRank(dstRank, laneId);
  if (laneId < nTopk) {
    int destinationSlot = -1;
    if (dstRank >= 0 && firstLaneForRank) {
      destinationSlot = atomicAdd(workspaceView.dispatchRankPayloadSlots_ + dstRank, 1);
      EP_DEVICE_ASSERT(destinationSlot < maxTokensPerRank);
    }
    destinationSlots[laneId] = destinationSlot;
  }
}

template <DispatchDataType DataType>
MSCCLPP_DEVICE_INLINE void sendStagedDispatchPayload(const DispatchPayloadView<DataType>& payloadView,
                                                     void* stagedPayload, const int* destinationSlots,
                                                     WorkspaceView& workspaceView, int nTopk, const ExpertMap& map,
                                                     int copyIdx, int maxTokensPerRank, size_t metadataBytes,
                                                     size_t payloadStride, void* recvBuffer,
                                                     const TransportView& transport, int laneId) {
  const int destinationSlot = laneId < nTopk ? destinationSlots[laneId] : -1;
  if (destinationSlot < 0) return;

  const int dstRank = map.destinationRank(payloadView.topKIndices(stagedPayload)[laneId], copyIdx);
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
MSCCLPP_DEVICE_INLINE void dispatchSendBf16(const void* inputTokens, const ExpertMap& map,
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
  const int rank = map.rank_;
  const int nRanks = map.numRanks();
  const int nSendCopies = map.numSendCopies();
  const size_t metadataBytes = dispatchMetadataBytes(map);
  const DispatchPayloadView<DispatchDataType::BF16> payloadView(Hidden, nTopk, 0);
  const size_t payloadStride = dispatchPayloadStride<DispatchDataType::BF16>(Hidden, nTopk, 0);
  auto* sharedPayloadBase = reinterpret_cast<uint8_t*>(sharedMem) + dispatchSharedControlBytes(nRanks);
  auto* sendBulkBarriers =
      reinterpret_cast<mscclpp::BulkBarrier*>(sharedPayloadBase + DispatchMaxNWarpGroups * payloadStride);
  auto* stagedPayload = sharedPayloadBase + static_cast<size_t>(warpGroupId) * payloadStride;
  auto* destinationSlots = sharedMem + warpGroupId * WARP_SIZE;
  auto* bulkBarrier = sendBulkBarriers + warpGroupId;
  WorkspaceView workspaceView(workspace, map);

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
    stageDispatchPayloadMetadata<DispatchDataType::BF16>(payloadView, stagedPayload, topkIndices, topkWeights, tokenIdx,
                                                         nTopk, maxTokensPerRank, rank, laneId);
    if (laneId == 0) bulkBarrier->wait(sendBulkPhase);
    __syncwarp();
    mscclpp::bulkFence();
    for (int copyIdx = 0; copyIdx < nSendCopies; ++copyIdx) {
      reserveDispatchPayloadSlots<DispatchDataType::BF16>(payloadView, stagedPayload, destinationSlots, workspaceView,
                                                          nTopk, map, copyIdx, maxTokensPerRank, laneId);
      __syncwarp();
      sendStagedDispatchPayload<DispatchDataType::BF16>(payloadView, stagedPayload, destinationSlots, workspaceView,
                                                        nTopk, map, copyIdx, maxTokensPerRank, metadataBytes,
                                                        payloadStride, recvBuffer, transport, laneId);
      __syncwarp();
    }
  }
}

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
MSCCLPP_DEVICE_INLINE void dispatchSendFp8(const void* inputTokens, const ExpertMap& map,
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
  const int rank = map.rank_;
  const int nRanks = map.numRanks();
  const int nSendCopies = map.numSendCopies();
  const size_t metadataBytes = dispatchMetadataBytes(map);
  const DispatchPayloadView<DataType> payloadView(Hidden, nTopk, ScaleBlockSize);
  const size_t payloadStride = dispatchPayloadStride<DataType>(Hidden, nTopk, ScaleBlockSize);
  auto* sharedPayloadBase = reinterpret_cast<uint8_t*>(sharedMem) + dispatchSharedControlBytes(nRanks);
  auto* stagedPayload = sharedPayloadBase + static_cast<size_t>(warpGroupId) * payloadStride;
  auto* destinationSlots = sharedMem + warpGroupId * WARP_SIZE;
  auto* outputData = payloadView.template data<mscclpp::f8_e4m3x8>(stagedPayload);
  auto* outputScales = payloadView.scaleFactors(stagedPayload);
  WorkspaceView workspaceView(workspace, map);

  const int tokenStride = nWorkerBlocks * nWarpGroups;
  const int firstTokenIdx = senderBlockIdx * nWarpGroups + warpGroupId;
  for (int tokenIdx = firstTokenIdx; tokenIdx < nTokens; tokenIdx += tokenStride) {
    const auto* inputData =
        reinterpret_cast<const mscclpp::bf16x8*>(inputTokens) + static_cast<size_t>(tokenIdx) * HiddenVectors;
    if (subWarpId == 0) {
      stageDispatchPayloadMetadata<DataType>(payloadView, stagedPayload, topkIndices, topkWeights, tokenIdx, nTopk,
                                             maxTokensPerRank, rank, laneId);
    }
    for (int inputIdx = groupThreadId; inputIdx < HiddenVectors; inputIdx += groupThreadCount) {
      outputData[inputIdx] = quantizeBf16x8ToFp8E4M3<ScaleBlockSize>(
          inputData[inputIdx], outputScales + inputIdx * mscclpp::bf16x8::Size / ScaleBlockSize, laneId);
    }
    syncNamedBarrier(groupBarrierId, groupThreadCount);

    if (subWarpId == 0) {
      mscclpp::bulkFence();
      for (int copyIdx = 0; copyIdx < nSendCopies; ++copyIdx) {
        reserveDispatchPayloadSlots<DataType>(payloadView, stagedPayload, destinationSlots, workspaceView, nTopk, map,
                                              copyIdx, maxTokensPerRank, laneId);
        __syncwarp();
        sendStagedDispatchPayload<DataType>(payloadView, stagedPayload, destinationSlots, workspaceView, nTopk, map,
                                            copyIdx, maxTokensPerRank, metadataBytes, payloadStride, recvBuffer,
                                            transport, laneId);
        __syncwarp();
      }
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
                                               int nTokens, int nTopk, const ExpertMap& map) {
  const int threadId = static_cast<int>(threadIdx.x);
  const int warpId = threadId / WARP_SIZE;
  const int laneId = get_lane_id();
  const int nRanks = map.numRanks();
  const int nExperts = map.numExperts_;
  const int nSendCopies = map.numSendCopies();
  for (int rankIdx = threadId; rankIdx < nRanks; rankIdx += blockDim.x) counts.rankTokenCounts_[rankIdx] = 0;
  for (int expertIdx = threadId; expertIdx < nExperts; expertIdx += blockDim.x)
    counts.expertTokenCounts_[expertIdx] = 0;
  __syncthreads();
  for (int tokenIdx = warpId; tokenIdx < nTokens; tokenIdx += DispatchNWarps) {
    const int routedExpertIdx = laneId < nTopk ? static_cast<int>(topkIndices[tokenIdx * nTopk + laneId]) : -1;
    if (routedExpertIdx >= 0) atomicAdd_block(counts.expertTokenCounts_ + routedExpertIdx, 1);
    for (int copyIdx = 0; copyIdx < nSendCopies; ++copyIdx) {
      const int dstRank = map.destinationRank(routedExpertIdx, copyIdx);
      if (isFirstLaneForRank(dstRank, laneId) && dstRank >= 0) {
        atomicAdd_block(counts.rankTokenCounts_ + dstRank, 1);
      }
    }
  }
  __syncthreads();
}

MSCCLPP_DEVICE_INLINE void countRankMajorRoutes(int* rankTokenCounts, const int64_t* __restrict__ topkIndices,
                                                int nTokens, int nTopk, const ExpertMap& map) {
  const int threadId = static_cast<int>(threadIdx.x);
  const int warpId = threadId / WARP_SIZE;
  const int laneId = get_lane_id();
  const int nRanks = map.numRanks();
  for (int rankIdx = threadId; rankIdx < nRanks; rankIdx += blockDim.x) rankTokenCounts[rankIdx] = 0;
  __syncthreads();
  for (int tokenIdx = warpId; tokenIdx < nTokens; tokenIdx += DispatchNWarps) {
    const int routedExpertIdx = laneId < nTopk ? static_cast<int>(topkIndices[tokenIdx * nTopk + laneId]) : -1;
    const int dstRank = map.leaderRank(routedExpertIdx);
    if (isFirstLaneForRank(dstRank, laneId) && dstRank >= 0) {
      atomicAdd_block(rankTokenCounts + dstRank, 1);
    }
  }
  __syncthreads();
}

MSCCLPP_DEVICE_INLINE void writeDispatchMetadata(const TransportView& transport, DispatchCountView counts,
                                                 const ExpertMap& map, void* recvBuffer, uint32_t epoch) {
  const int threadId = static_cast<int>(threadIdx.x);
  const int nRanks = map.numRanks();
  const int nExperts = map.numExperts_;
  const int nSendCopies = map.numSendCopies();
  // Counts are written only into the receive buffers this rank actually sends
  // payloads to. Non-leader ETP peers read them from their group leader.
  for (int dstRank = threadId; dstRank < nRanks; dstRank += blockDim.x) {
    if (map.leaderInGroupOf(dstRank) != dstRank) continue;
    auto* destinationPackets = reinterpret_cast<mscclpp::LL8Packet*>(transport.mappedBuffer(recvBuffer, dstRank));
    destinationPackets[transport.rank_].write(static_cast<uint32_t>(counts.rankTokenCounts_[dstRank]), epoch);
  }
  for (int expertIdx = threadId; expertIdx < nExperts; expertIdx += blockDim.x) {
    const int localExpertIdx = map.localExpert(expertIdx);
    const int metadataSlot = nRanks + map.metadataExpertSlot(transport.rank_, localExpertIdx);
    for (int copyIdx = 0; copyIdx < nSendCopies; ++copyIdx) {
      const int dstRank = map.destinationRank(expertIdx, copyIdx);
      auto* destinationPackets = reinterpret_cast<mscclpp::LL8Packet*>(transport.mappedBuffer(recvBuffer, dstRank));
      destinationPackets[metadataSlot].write(static_cast<uint32_t>(counts.expertTokenCounts_[expertIdx]), epoch);
    }
  }
}

MSCCLPP_DEVICE_INLINE void writeRankMajorCounts(const TransportView& transport, const int* rankTokenCounts, int nRanks,
                                                void* recvBuffer, uint32_t epoch) {
  const int threadId = static_cast<int>(threadIdx.x);
  for (int dstRank = threadId; dstRank < nRanks; dstRank += blockDim.x) {
    auto* destinationPackets = reinterpret_cast<mscclpp::LL8Packet*>(transport.mappedBuffer(recvBuffer, dstRank));
    destinationPackets[transport.rank_].write(static_cast<uint32_t>(rankTokenCounts[dstRank]), epoch);
  }
}

MSCCLPP_DEVICE_INLINE void publishDispatchPayloads(const TransportView& transport, const int* rankTokenCounts,
                                                   const ExpertMap& map, WorkspaceView workspaceView) {
  const int threadId = static_cast<int>(threadIdx.x);
  const int nRanks = map.numRanks();
  // Phase 1: drain the payload completions of every rank this GPU wrote to.
  for (int dstRank = threadId; dstRank < nRanks; dstRank += blockDim.x) {
    if (map.leaderInGroupOf(dstRank) != dstRank) continue;
    const int expectedPayloadCount = rankTokenCounts[dstRank];
    if (expectedPayloadCount > 0) {
      while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(workspaceView.dispatchRankPayloadCompletions_ + dstRank,
                                                            mscclpp::memoryOrderAcquire) != expectedPayloadCount);
    }
    workspaceView.dispatchRankPayloadSlots_[dstRank] = 0;
    workspaceView.dispatchRankPayloadCompletions_[dstRank] = 0;
  }
  // Phase 2: release every rank of every destination group. Under
  // LEADER_SINGLE_SEND the ETP peers of a leader pull the rows over NVLink, so
  // they must be released by the sender too.
  __syncthreads();
  for (int dstRank = threadId; dstRank < nRanks; dstRank += blockDim.x) {
    if (rankTokenCounts[map.leaderInGroupOf(dstRank)] == 0) continue;
    if (transport.isSelf(dstRank)) {
      workspaceView.dispatchLocalPayloadReady_->release();
    } else {
      transport.baseMemoryChannels_[dstRank].signal();
    }
  }
}

MSCCLPP_DEVICE_INLINE void dispatchNotify(const TransportView& transport, const ExpertMap& map,
                                          const int64_t* __restrict__ topkIndices, int nTokens, int nTopk,
                                          void* recvBuffer, void* workspace, uint32_t epoch, int* sharedMem) {
  WorkspaceView workspaceView(workspace, map);
  DispatchCountView counts(sharedMem, map.numRanks());
  countDispatchRoutes(counts, topkIndices, nTokens, nTopk, map);
  writeDispatchMetadata(transport, counts, map, recvBuffer, epoch);
  publishDispatchPayloads(transport, counts.rankTokenCounts_, map, workspaceView);
}

MSCCLPP_DEVICE_INLINE void dispatchRankMajorNotify(const TransportView& transport, const ExpertMap& map,
                                                   const int64_t* __restrict__ topkIndices, int nTokens, int nTopk,
                                                   void* recvBuffer, void* workspace, uint32_t epoch, int* sharedMem) {
  WorkspaceView workspaceView(workspace, map);
  auto* rankTokenCounts = sharedMem;
  countRankMajorRoutes(rankTokenCounts, topkIndices, nTokens, nTopk, map);
  writeRankMajorCounts(transport, rankTokenCounts, map.numRanks(), recvBuffer, epoch);
  publishDispatchPayloads(transport, rankTokenCounts, map, workspaceView);
}

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
MSCCLPP_DEVICE_INLINE void dispatchSend(const void* inputTokens, const TransportView& transport, const ExpertMap& map,
                                        const int64_t* __restrict__ topkIndices,
                                        const float* __restrict__ topkWeights, int nTokens, int nTopk,
                                        int maxTokensPerRank, void* recvBuffer, void* workspace, uint32_t epoch,
                                        int* sharedMem) {
  const int nWorkerBlocks = static_cast<int>(gridDim.x) - DispatchControlBlocks;
  if (static_cast<int>(blockIdx.x) > 0 && static_cast<int>(blockIdx.x) <= nWorkerBlocks) {
    if constexpr (DataType == DispatchDataType::BF16) {
      dispatchSendBf16<Hidden>(inputTokens, map, topkIndices, topkWeights, nTokens, nTopk, maxTokensPerRank, recvBuffer,
                               transport, workspace, sharedMem);
    } else {
      dispatchSendFp8<Hidden, DataType, ScaleBlockSize>(inputTokens, map, topkIndices, topkWeights, nTokens, nTopk,
                                                        maxTokensPerRank, recvBuffer, transport, workspace, sharedMem);
    }
  } else if (static_cast<int>(blockIdx.x) == nWorkerBlocks + 1) {
    dispatchNotify(transport, map, topkIndices, nTokens, nTopk, recvBuffer, workspace, epoch, sharedMem);
  }
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void dispatchSendRankMajor(void* output, int* outputTopkIdx, float* outputTopkWeights,
                                                 const void* inputTokens, const TransportView& transport,
                                                 const ExpertMap& map, const int64_t* __restrict__ topkIndices,
                                                 const float* __restrict__ topkWeights, int nTokens, int nTopk,
                                                 int invalidTokenExpertId, int maxTokensPerRank, void* recvBuffer,
                                                 void* workspace, uint32_t epoch, int* sharedMem) {
  const int nWorkerBlocks = static_cast<int>(gridDim.x) - DispatchControlBlocks;
  if (static_cast<int>(blockIdx.x) > 0 && static_cast<int>(blockIdx.x) <= nWorkerBlocks) {
    dispatchSendRankMajorBf16<Hidden>(output, outputTopkIdx, outputTopkWeights, inputTokens, map, topkIndices,
                                      topkWeights, nTokens, nTopk, invalidTokenExpertId, maxTokensPerRank, transport,
                                      workspace, nWorkerBlocks, sharedMem);
  } else if (static_cast<int>(blockIdx.x) == nWorkerBlocks + 1) {
    dispatchRankMajorNotify(transport, map, topkIndices, nTokens, nTopk, recvBuffer, workspace, epoch, sharedMem);
  }
}

MSCCLPP_DEVICE_INLINE int proportionalTaskBoundary(int nTokens, int nTasks, int nTotalTokens) {
  return nTotalTokens == 0 ? 0 : static_cast<int>(static_cast<int64_t>(nTokens) * nTasks / nTotalTokens);
}

MSCCLPP_DEVICE_INLINE void dispatchRecvScheduler(int64_t* outputLayout, int* outputCount,
                                                 const TransportView& transport, const ExpertMap& map,
                                                 void* recvBuffer, void* workspace, uint32_t epoch, int* sharedMem) {
  const int threadId = static_cast<int>(threadIdx.x);
  const int warpId = threadId / WARP_SIZE;
  const int laneId = get_lane_id();
  const int nWorkerBlocks = static_cast<int>(gridDim.x) - DispatchControlBlocks;
  const int nRanks = map.numRanks();
  auto* rankTokenCounts = reinterpret_cast<mscclpp::LL8Packet*>(recvBuffer);
  const int nLocalExperts = map.numLocalExperts();
  WorkspaceView workspaceView(workspace, map);

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
        const int nExpertTokens = static_cast<int>(
            expertTokenCounts[map.metadataExpertSlot(sourceRank, localExpertIdx)].read(epoch, -1));
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
                                                 const TransportView& transport, const ExpertMap& map, int nTopk,
                                                 int maxTokensPerRank, int invalidTokenExpertId, void* recvBuffer,
                                                 void* workspace, uint32_t epoch, int* sharedMem) {
  const int sourceRank = static_cast<int>(blockIdx.x);
  if (sourceRank >= map.numRanks()) return;
  auto* rankTokenCounts = reinterpret_cast<mscclpp::LL8Packet*>(recvBuffer);
  if (threadIdx.x == 0) {
    const int nRankTokens = static_cast<int>(rankTokenCounts[sourceRank].read(epoch, -1));
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

  WorkspaceView workspaceView(workspace, map);
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
    int sourceTokenIdx, const ExpertMap& map, int nTopk, int maxTokensPerRank, WorkspaceView& workspaceView,
    uint8_t* sharedTile, mscclpp::BulkBarrier* bulkBarrier, uint32_t& recvBulkPhase) {
  using OutputType = DispatchElementType<DataType>;
  constexpr size_t OutputBytes = static_cast<size_t>(Hidden) * sizeof(OutputType);
  constexpr int NumScales = DataType == DispatchDataType::BF16 ? 0 : Hidden / ScaleBlockSize;
  const int laneId = get_lane_id();
  const int nRanks = map.numRanks();
  const int nLocalExperts = map.numLocalExperts();
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
                                              int64_t* outputLayout, const ExpertMap& map, int nTopk,
                                              int maxTokensPerRank, void* recvBuffer, void* workspace, uint32_t epoch,
                                              int* sharedMem) {
#if defined(__CUDA_ARCH__)
  static_assert(__CUDA_ARCH__ >= 900, "TMA recv requires SM90 or newer");
#endif
  const int threadId = static_cast<int>(threadIdx.x);
  const int warpId = threadId / WARP_SIZE;
  const int laneId = get_lane_id();
  const int nRanks = map.numRanks();
  WorkspaceView workspaceView(workspace, map);
  RecvTask task;
  if (!acquireRecvTask(task, workspaceView, epoch, sharedMem)) return;
  using OutputType = DispatchElementType<DataType>;
  constexpr int NRecvTmaWorkers = tmaWorkerCount<Hidden, OutputType, DispatchMaxNRecvTmaWorkers>();
  if (warpId >= NRecvTmaWorkers) return;

  const int sourceRank = task.sourceRank_;
  const int globalExpertBase = map.globalExpertBase();
  const int globalExpertEnd = globalExpertBase + map.numLocalExperts();
  const DispatchPayloadView<DataType> payloadView(Hidden, nTopk, ScaleBlockSize);
  const size_t payloadStride = dispatchPayloadStride<DataType>(Hidden, nTopk, ScaleBlockSize);
  constexpr size_t OutputBytes = static_cast<size_t>(Hidden) * sizeof(OutputType);
  constexpr size_t TileBytes = OutputBytes;
  auto* sourcePayloadBase = reinterpret_cast<uint8_t*>(recvBuffer) + dispatchMetadataBytes(map) +
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
        sourceTokenIdx, map, nTopk, maxTokensPerRank, workspaceView, sharedTile, bulkBarrier, recvBulkPhase);
  }

  if (hasPendingStore) mscclpp::bulkStoreWait();
}

#endif  // MSCCLPP_BULK_AVAILABLE

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize, DispatchLayout Layout>
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
  const ExpertMap map(context, nExperts);
  const int nTokens = workload.numTokens_;
  const int nTopk = workload.numTopk_;
  const int invalidTokenExpertId = workload.invalidTokenExpertId_;
  const int maxTokensPerRank = workload.maxTokensPerRank_;
  const TransportView transport(context);
  const uint32_t epoch = workload.epoch_;
  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    static_assert(DataType == DispatchDataType::BF16);
    dispatchSendRankMajor<Hidden>(output, outputTopkIdx, outputTopkWeights, inputTokens, transport, map, topkIndices,
                                  topkWeights, nTokens, nTopk, invalidTokenExpertId, maxTokensPerRank, recvBuffer,
                                  context->workspace_, epoch, sharedMem);
  } else {
    dispatchSend<Hidden, DataType, ScaleBlockSize>(inputTokens, transport, map, topkIndices, topkWeights, nTokens,
                                                   nTopk, maxTokensPerRank, recvBuffer, context->workspace_, epoch,
                                                   sharedMem);
  }

  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    if (static_cast<int>(blockIdx.x) < nRanks) {
      dispatchRecvRankMajor(outputTopkIdx, outputTopkWeights, outputCount, transport, map, nTopk, maxTokensPerRank,
                            invalidTokenExpertId, recvBuffer, context->workspace_, epoch, sharedMem);
    }
  } else {
    if (static_cast<int>(blockIdx.x) == 0) {
      dispatchRecvScheduler(outputLayout, outputCount, transport, map, recvBuffer, context->workspace_, epoch,
                            sharedMem);
    } else if (static_cast<int>(blockIdx.x) <= nWorkerBlocks) {
      dispatchRecvWorker<Hidden, DataType, ScaleBlockSize>(output, outputScales, outputSrcInfo, outputLayout, map,
                                                           nTopk, maxTokensPerRank, recvBuffer, context->workspace_,
                                                           epoch, sharedMem);
    }
  }
#endif  // MSCCLPP_BULK_AVAILABLE
}

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize, DispatchLayout Layout, typename KernelSelector>
inline void dispatchHiddenMode(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                               float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                               const int64_t* topkIdx, const float* topkWeights, const Workload& workload,
                               void* recvBuffer, const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  static_assert(Hidden == 2048 || Hidden == 4096 || Hidden == 4352 || Hidden == 6656 || Hidden == 7168 ||
                Hidden == 8192 || Hidden == 8704 || Hidden == 9216);
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
  EP_HOST_ASSERT(context.topology_.numRanks == nRanks);
  EP_HOST_ASSERT(context.topology_.epSize > 0 && nExperts % context.topology_.epSize == 0);
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
  EP_HOST_ASSERT(outputSrcInfo != nullptr || workload.outputLayout_ == DispatchLayout::RANK_MAJOR);
  EP_HOST_ASSERT(outputCount != nullptr);
  EP_HOST_ASSERT(outputLayout != nullptr || workload.outputLayout_ == DispatchLayout::RANK_MAJOR);
  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    EP_HOST_ASSERT(outputTopkIdx != nullptr);
    EP_HOST_ASSERT(outputTopkWeights != nullptr);
  }
  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
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
