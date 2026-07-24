// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
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
  __syncwarp();
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
  (void)mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(
      workspaceView.dispatchRankPayloadCompletions_ + destinationRank, 1, mscclpp::memoryOrderRelease);
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void dispatchSendRankMajorBf16(void* output, int* outputTopkIdx, float* outputTopkWeights,
                                                     const void* inputTokens, int nExperts, int nRanks,
                                                     const int64_t* __restrict__ topkIndices,
                                                     const float* __restrict__ topkWeights, int nTokens, int nTopk,
                                                     int invalidTokenExpertId, int maxTokensPerRank,
                                                     const TransportView& transport, void* workspace,
                                                     bool enableTmaPipeline, int nPayloadBlocks, int* sharedMem) {
  const int nWorkerBlocks = static_cast<int>(gridDim.x) - DispatchControlBlocks;
  if (blockIdx.x == 0 || static_cast<int>(blockIdx.x) > nPayloadBlocks) return;

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
  const size_t sharedTokenStride = dispatchPayloadStride<DispatchDataType::BF16>(Hidden, nTopk, 0);
  auto* sharedTokenBase = reinterpret_cast<uint8_t*>(sharedMem) + dispatchSharedControlBytes(nRanks);
  auto* sendTmaBarriers = reinterpret_cast<uint64_t*>(sharedTokenBase + DispatchMaxNWarpGroups * sharedTokenStride);
  WorkspaceView workspaceView(workspace, nRanks, nExperts);

  const bool usePipeline =
      enableTmaPipeline && nWarpGroups == 1 && nTokens > nPayloadBlocks && nTokens <= 2 * nPayloadBlocks;
  if (usePipeline) {
    uint8_t* stagedTokens[2] = {sharedTokenBase, sharedTokenBase + sharedTokenStride};
    uint64_t* tmaBarriers[2] = {sendTmaBarriers, sendTmaBarriers + 1};
    uint32_t sendTmaPhases[2] = {};
    if (laneId == 0) {
      initTmaLoadBarrier(tmaBarriers[0]);
      initTmaLoadBarrier(tmaBarriers[1]);
    }
    __syncwarp();

    const int tokenIndices[2] = {senderBlockIdx, senderBlockIdx + nPayloadBlocks};
    const auto issueLoad = [&](int stage) {
      const auto* inputData = reinterpret_cast<const mscclpp::bf16x8*>(inputTokens) +
                              static_cast<size_t>(tokenIndices[stage]) * HiddenVectors;
      if (laneId == 0) {
        issueTmaLoadAndExpect(inputData, stagedTokens[stage], tmaBarriers[stage], static_cast<uint32_t>(HiddenBytes));
      }
    };
    issueLoad(0);
    const RankMajorRoute route0 = prepareRankMajorRoute(workspaceView, topkIndices, tokenIndices[0], nTopk,
                                                        nLocalExperts, maxTokensPerRank, laneId);
    if (laneId == 0) waitTmaLoad(tmaBarriers[0], sendTmaPhases[0]);
    __syncwarp();
    fenceProxyAsyncSharedCta();

    const int completionRank0 = route0.dstRank >= 0 && route0.isLeader ? route0.dstRank : -1;
    if (completionRank0 >= 0) {
      issueRankMajorTokenStore<Hidden>(output, transport, route0.destinationSlot, maxTokensPerRank, stagedTokens[0],
                                       route0.dstRank);
    }

    const bool hasSecondToken = tokenIndices[1] < nTokens;
    RankMajorRoute route1{-1, -1, false};
    if (hasSecondToken) {
      issueLoad(1);
      route1 = prepareRankMajorRoute(workspaceView, topkIndices, tokenIndices[1], nTopk, nLocalExperts,
                                     maxTokensPerRank, laneId);
    }
    sendRankMajorMetadata(transport, outputTopkIdx, outputTopkWeights, topkIndices, topkWeights, route0,
                          tokenIndices[0], nTopk, nLocalExperts, maxTokensPerRank, invalidTokenExpertId);
    completeRankMajorTokenStore(workspaceView, completionRank0);
    __syncwarp();

    if (hasSecondToken) {
      if (laneId == 0) waitTmaLoad(tmaBarriers[1], sendTmaPhases[1]);
      __syncwarp();
      fenceProxyAsyncSharedCta();
      const int completionRank1 = route1.dstRank >= 0 && route1.isLeader ? route1.dstRank : -1;
      if (completionRank1 >= 0) {
        issueRankMajorTokenStore<Hidden>(output, transport, route1.destinationSlot, maxTokensPerRank, stagedTokens[1],
                                         route1.dstRank);
      }
      sendRankMajorMetadata(transport, outputTopkIdx, outputTopkWeights, topkIndices, topkWeights, route1,
                            tokenIndices[1], nTopk, nLocalExperts, maxTokensPerRank, invalidTokenExpertId);
      completeRankMajorTokenStore(workspaceView, completionRank1);
      __syncwarp();
    }
    return;
  }

  auto* stagedToken = sharedTokenBase + static_cast<size_t>(warpGroupId) * sharedTokenStride;
  auto* tmaBarrier = sendTmaBarriers + warpGroupId;
  const int tokenStride = nPayloadBlocks * nWarpGroups;
  const int firstTokenIdx = senderBlockIdx * nWarpGroups + warpGroupId;
  uint32_t sendTmaPhase = 0;
  if (firstTokenIdx < nTokens && laneId == 0) initTmaLoadBarrier(tmaBarrier);

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
    __syncwarp();
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
  issueTmaStore(destinationPayload, stagedPayload, static_cast<uint32_t>(payloadView.numBytes_));
  waitBulkGroup();
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
  auto* sendTmaBarriers = reinterpret_cast<uint64_t*>(sharedPayloadBase + DispatchMaxNWarpGroups * payloadStride);
  auto* stagedPayload = sharedPayloadBase + static_cast<size_t>(warpGroupId) * payloadStride;
  auto* destinationSlots = sharedMem + warpGroupId * WARP_SIZE;
  auto* tmaBarrier = sendTmaBarriers + warpGroupId;
  WorkspaceView workspaceView(workspace, nRanks, nExperts);

  const int tokenStride = nWorkerBlocks * nWarpGroups;
  const int firstTokenIdx = senderBlockIdx * nWarpGroups + warpGroupId;
  uint32_t sendTmaPhase = 0;
  if (firstTokenIdx < nTokens) {
    if (laneId == 0) initTmaLoadBarrier(tmaBarrier);
  }

  for (int tokenIdx = firstTokenIdx; tokenIdx < nTokens; tokenIdx += tokenStride) {
    const auto* inputData =
        reinterpret_cast<const mscclpp::bf16x8*>(inputTokens) + static_cast<size_t>(tokenIdx) * HiddenVectors;
    if (laneId == 0) {
      issueTmaLoadAndExpect(inputData, stagedPayload, tmaBarrier, static_cast<uint32_t>(HiddenBytes));
    }
    stageDispatchPayloadMetadata<DispatchDataType::BF16>(payloadView, stagedPayload, destinationSlots, workspaceView,
                                                         topkIndices, topkWeights, tokenIdx, nTopk, nLocalExperts,
                                                         maxTokensPerRank, rank, laneId);
    if (laneId == 0) waitTmaLoad(tmaBarrier, sendTmaPhase);
    __syncwarp();
    fenceProxyAsyncSharedCta();
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
  static_assert(DataType == DispatchDataType::FP8_E4M3 || DataType == DispatchDataType::MXFP8_E4M3);
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
      if constexpr (DataType == DispatchDataType::MXFP8_E4M3) {
        outputData[inputIdx] = quantizeBf16x8ToMxFp8E4M3<ScaleBlockSize>(
            inputData[inputIdx], outputScales + inputIdx * mscclpp::bf16x8::Size / ScaleBlockSize, laneId);
      } else {
        outputData[inputIdx] = quantizeBf16x8ToFp8E4M3<ScaleBlockSize>(
            inputData[inputIdx], outputScales + inputIdx * mscclpp::bf16x8::Size / ScaleBlockSize, laneId);
      }
    }
    syncNamedBarrier(groupBarrierId, groupThreadCount);

    if (subWarpId == 0) {
      fenceProxyAsyncSharedCta();
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

MSCCLPP_DEVICE_INLINE void publishDispatchPayloads(const TransportView& transport, const int* rankTokenCounts,
                                                   int nRanks, WorkspaceView workspaceView) {
  const int threadId = static_cast<int>(threadIdx.x);
  for (int dstRank = threadId; dstRank < nRanks; dstRank += blockDim.x) {
    const int expectedPayloadCount = rankTokenCounts[dstRank];
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
                                          void* recvBuffer, void* workspace, uint32_t dispatchEpoch, int* sharedMem) {
  WorkspaceView workspaceView(workspace, nRanks, nExperts);
  DispatchCountView counts(sharedMem, nRanks);
  countDispatchRoutes(counts, topkIndices, nTokens, nTopk, nRanks, nExperts);
  writeDispatchMetadata(transport, counts, nRanks, nExperts, recvBuffer, dispatchEpoch);
  publishDispatchPayloads(transport, counts.rankTokenCounts_, nRanks, workspaceView);
}

MSCCLPP_DEVICE_INLINE void dispatchRankMajorNotify(const TransportView& transport, int nExperts, int nRanks,
                                                   const int64_t* __restrict__ topkIndices, int nTokens, int nTopk,
                                                   void* recvBuffer, void* workspace, uint32_t dispatchEpoch,
                                                   int* sharedMem) {
  WorkspaceView workspaceView(workspace, nRanks, nExperts);
  auto* rankTokenCounts = sharedMem;
  countRankMajorRoutes(rankTokenCounts, topkIndices, nTokens, nTopk, nRanks, nExperts);
  writeRankMajorCounts(transport, rankTokenCounts, nRanks, recvBuffer, dispatchEpoch);
  publishDispatchPayloads(transport, rankTokenCounts, nRanks, workspaceView);
}

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
MSCCLPP_DEVICE_INLINE void dispatchSend(const void* inputTokens, const TransportView& transport, int nExperts,
                                        int nRanks, const int64_t* __restrict__ topkIndices,
                                        const float* __restrict__ topkWeights, int nTokens, int nTopk,
                                        int maxTokensPerRank, void* recvBuffer, void* workspace, uint32_t dispatchEpoch,
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
    dispatchNotify(transport, nExperts, nRanks, topkIndices, nTokens, nTopk, recvBuffer, workspace, dispatchEpoch,
                   sharedMem);
  }
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void dispatchSendRankMajor(void* output, int* outputTopkIdx, float* outputTopkWeights,
                                                 const void* inputTokens, const TransportView& transport, int nExperts,
                                                 int nRanks, const int64_t* __restrict__ topkIndices,
                                                 const float* __restrict__ topkWeights, int nTokens, int nTopk,
                                                 int invalidTokenExpertId, int maxTokensPerRank, void* recvBuffer,
                                                 void* workspace, uint32_t dispatchEpoch, bool enableTmaPipeline,
                                                 int* sharedMem) {
  const int nWorkerBlocks = static_cast<int>(gridDim.x) - DispatchControlBlocks;
  if (static_cast<int>(blockIdx.x) > 0 && static_cast<int>(blockIdx.x) <= nWorkerBlocks) {
    dispatchSendRankMajorBf16<Hidden>(output, outputTopkIdx, outputTopkWeights, inputTokens, nExperts, nRanks,
                                      topkIndices, topkWeights, nTokens, nTopk, invalidTokenExpertId, maxTokensPerRank,
                                      transport, workspace, enableTmaPipeline, nWorkerBlocks, sharedMem);
  } else if (static_cast<int>(blockIdx.x) == nWorkerBlocks + 1) {
    dispatchRankMajorNotify(transport, nExperts, nRanks, topkIndices, nTokens, nTopk, recvBuffer, workspace,
                            dispatchEpoch, sharedMem);
  }
}

MSCCLPP_DEVICE_INLINE int proportionalTaskBoundary(int nTokens, int nTasks, int nTotalTokens) {
  return nTotalTokens == 0 ? 0 : static_cast<int>(static_cast<int64_t>(nTokens) * nTasks / nTotalTokens);
}

MSCCLPP_DEVICE_INLINE void dispatchRecvScheduler(int64_t* outputLayout, int* outputCount,
                                                 const TransportView& transport, int nExperts, int nRanks,
                                                 void* recvBuffer, void* workspace, uint32_t dispatchEpoch,
                                                 int* sharedMem) {
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
    const int nRankTokens =
        sourceRank < nRanks ? static_cast<int>(rankTokenCounts[sourceRank].read(dispatchEpoch, -1)) : 0;
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
      mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(workspaceView.dispatchTasksReadyEpoch_, dispatchEpoch,
                                                           mscclpp::memoryOrderRelease);
    }

    if (sourceRank < nRanks && nRankTokens > 0) {
      if (transport.isSelf(sourceRank)) {
        workspaceView.dispatchLocalPayloadReady_->acquire();
      } else {
        transport.baseMemoryChannels_[sourceRank].wait(-1);
      }
      mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(workspaceView.dispatchRankReadyEpochs_ + sourceRank,
                                                           dispatchEpoch, mscclpp::memoryOrderRelease);
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
                                           int* sharedMem) {
  auto* sharedTask = reinterpret_cast<RecvTask*>(sharedMem);
  const int taskIdx = static_cast<int>(blockIdx.x) - 1;
  if (threadIdx.x == 0) {
    while (mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(workspaceView.dispatchTasksReadyEpoch_,
                                                               mscclpp::memoryOrderAcquire) != dispatchEpoch);
    if (taskIdx < *workspaceView.dispatchNumRecvTasks_) {
      task = workspaceView.dispatchRecvTasks_[taskIdx];
      while (mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(
                 workspaceView.dispatchRankReadyEpochs_ + task.sourceRank_, mscclpp::memoryOrderAcquire) !=
             dispatchEpoch);
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

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
MSCCLPP_DEVICE_INLINE bool dispatchRecvExpertMajorOutput(
    void* output, void* outputScales, int* outputSrcInfo, int64_t* outputLayout,
    const DispatchPayloadView<DataType>& payloadView, void* sourcePayload, int localExpertIdx, int sourceRank,
    int sourceTokenIdx, int nLocalExperts, int nRanks, int nTopk, int maxTokensPerRank, WorkspaceView& workspaceView,
    uint8_t* sharedTile, uint64_t* tmaBarrier, uint32_t& recvTmaPhase) {
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

  if (laneId == 0) waitTmaLoad(tmaBarrier, recvTmaPhase);
  __syncwarp();
  fenceProxyAsyncSharedCta();

  if (localExpertIdx < 0) return false;
  auto* outputData = reinterpret_cast<uint8_t*>(output) +
                     (static_cast<size_t>(localExpertIdx) * nOutputSlotsPerExpert + outputTokenIdx) * OutputBytes;
  issueTmaStore(outputData, sharedTile, static_cast<uint32_t>(OutputBytes));
  return true;
}

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
MSCCLPP_DEVICE_INLINE void dispatchRecvWorker(void* output, void* outputScales, int* outputSrcInfo,
                                              int64_t* outputLayout, int nExperts, int rank, int nRanks, int nTopk,
                                              int maxTokensPerRank, void* recvBuffer, void* workspace,
                                              uint32_t dispatchEpoch, int* sharedMem) {
#if defined(__CUDA_ARCH__)
  static_assert(__CUDA_ARCH__ >= 900, "TMA recv requires SM90 or newer");
#endif
  const int threadId = static_cast<int>(threadIdx.x);
  const int warpId = threadId / WARP_SIZE;
  const int laneId = get_lane_id();
  WorkspaceView workspaceView(workspace, nRanks, nExperts);
  RecvTask task;
  if (!acquireRecvTask(task, workspaceView, dispatchEpoch, sharedMem)) return;
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
  auto* tmaBarriers = reinterpret_cast<uint64_t*>(tmaTiles + static_cast<size_t>(NRecvTmaWorkers) * TileBytes);
  auto* tmaBarrier = tmaBarriers + warpId;
  bool hasPendingStore = false;
  uint32_t recvTmaPhase = 0;
  if (laneId == 0) initTmaLoadBarrier(tmaBarrier);

  for (int sourceTokenSlot = task.tokenBegin_ + warpId; sourceTokenSlot < task.tokenEnd_;
       sourceTokenSlot += NRecvTmaWorkers) {
    if (hasPendingStore) {
      waitBulkGroupRead();
    }
    __syncwarp();

    auto* sourcePayload = sourcePayloadBase + static_cast<size_t>(sourceTokenSlot) * payloadStride;
    if (laneId == 0) {
      issueTmaLoadAndExpect(payloadView.template data<OutputType>(sourcePayload), sharedTile, tmaBarrier,
                            static_cast<uint32_t>(OutputBytes));
    }

    const int routedExpertIdx = laneId < nTopk ? payloadView.topKIndices(sourcePayload)[laneId] : -1;
    const int localExpertIdx = routedExpertIdx >= globalExpertBase && routedExpertIdx < globalExpertEnd
                                   ? routedExpertIdx - globalExpertBase
                                   : -1;
    const int sourceTokenIdx = warpBroadcast(
        laneId == 0 ? *payloadView.srcTokenGlobalIdx(sourcePayload) - sourceRank * maxTokensPerRank : 0, 0);
    hasPendingStore = dispatchRecvExpertMajorOutput<Hidden, DataType, ScaleBlockSize>(
        output, outputScales, outputSrcInfo, outputLayout, payloadView, sourcePayload, localExpertIdx, sourceRank,
        sourceTokenIdx, nLocalExperts, nRanks, nTopk, maxTokensPerRank, workspaceView, sharedTile, tmaBarrier,
        recvTmaPhase);
  }

  if (hasPendingStore) waitBulkGroup();
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
  const uint32_t dispatchEpoch = *workspaceView.dispatchEpoch_ + 1;
  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    static_assert(DataType == DispatchDataType::BF16);
    dispatchSendRankMajor<Hidden>(output, outputTopkIdx, outputTopkWeights, inputTokens, transport, nExperts, nRanks,
                                  topkIndices, topkWeights, nTokens, nTopk, invalidTokenExpertId, maxTokensPerRank,
                                  recvBuffer, workspace, dispatchEpoch, workload.enableRankMajorTmaPipeline_,
                                  sharedMem);
  } else {
    dispatchSend<Hidden, DataType, ScaleBlockSize>(inputTokens, transport, nExperts, nRanks, topkIndices, topkWeights,
                                                   nTokens, nTopk, maxTokensPerRank, recvBuffer, workspace,
                                                   dispatchEpoch, sharedMem);
  }

  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    if (static_cast<int>(blockIdx.x) < nRanks) {
      dispatchRecvRankMajor(outputTopkIdx, outputTopkWeights, outputCount, transport, nExperts, nRanks, nTopk,
                            maxTokensPerRank, invalidTokenExpertId, recvBuffer, workspace, dispatchEpoch, sharedMem);
    }
  } else {
    if (static_cast<int>(blockIdx.x) == 0) {
      dispatchRecvScheduler(outputLayout, outputCount, transport, nExperts, nRanks, recvBuffer, workspace,
                            dispatchEpoch, sharedMem);
    } else if (static_cast<int>(blockIdx.x) <= nWorkerBlocks) {
      dispatchRecvWorker<Hidden, DataType, ScaleBlockSize>(output, outputScales, outputSrcInfo, outputLayout, nExperts,
                                                           comm.rank_, nRanks, nTopk, maxTokensPerRank, recvBuffer,
                                                           workspace, dispatchEpoch, sharedMem);
    }
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *workspaceView.dispatchEpoch_ = dispatchEpoch;
  }
}

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize, DispatchLayout Layout>
inline void dispatchHiddenMode(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                               float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                               const int64_t* topkIdx, const float* topkWeights, const low_latency::Workload& workload,
                               void* recvBuffer, const low_latency::CommContext& comm, void* workspace, int numBlocks,
                               cudaStream_t stream) {
  static_assert(Hidden == 4096 || Hidden == 6656 || Hidden == 7168 || Hidden == 8192 || Hidden == 9216);
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
      case DispatchDataType::MXFP8_E4M3:
        return dispatchHiddenMode<Hidden, DispatchDataType::MXFP8_E4M3, 32, Layout>(
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
    case 4096:
      return dispatchLayout<4096>(output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout,
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
