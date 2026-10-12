// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "common.cuh"
#include "common/quantization.cuh"

namespace mscclpp {
namespace ep {
namespace {
#if MSCCLPP_BULK_AVAILABLE

struct Route {
  int dstRank;
  int destinationSlot;
  bool isLeader;
};

MSCCLPP_DEVICE_INLINE Route prepareRoute(WorkspaceView& workspaceView, const int64_t* __restrict__ topkIndices,
                                         int tokenIdx, int nTopk, int nLocalExperts, int maxTokensPerRank, int laneId) {
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

MSCCLPP_DEVICE_INLINE void sendMetadata(const TransportView& transport, int* outputTopkIdx, float* outputTopkWeights,
                                        const int64_t* __restrict__ topkIndices, const float* __restrict__ topkWeights,
                                        const Route& route, int tokenIdx, int nTopk, int nLocalExperts,
                                        int maxTokensPerRank, int invalidTokenExpertId) {
  const int laneId = getLaneId();
  const int candidateExpert =
      laneId < nTopk ? static_cast<int>(topkIndices[tokenIdx * nTopk + laneId]) : invalidTokenExpertId;
  const float candidateWeight =
      laneId < nTopk ? (topkWeights == nullptr ? 1.0f : topkWeights[tokenIdx * nTopk + laneId]) : 0.0f;
  unsigned int leaderMask = warpLaneMask(route.dstRank >= 0 && route.isLeader);
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

template <int Hidden, DispatchDataType DataType = DispatchDataType::BF16>
MSCCLPP_DEVICE_INLINE void issueTokenStore(void* output, const TransportView& transport, int destinationSlot,
                                           int maxTokensPerRank, void* stagedToken, int destinationRank,
                                           void* outputScales = nullptr) {
  if (destinationSlot < 0) return;

  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(DispatchElementType<DataType>);
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

MSCCLPP_DEVICE_INLINE void completeTokenStore(WorkspaceView& workspaceView, int destinationRank) {
  if (destinationRank < 0) return;
  mscclpp::bulkStoreWait();
  (void)mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(
      workspaceView.dispatchRankPayloadCompletions_ + destinationRank, 1, mscclpp::memoryOrderRelease);
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void dispatchSendBf16(void* output, int* outputTopkIdx, float* outputTopkWeights,
                                            const void* inputTokens, int nExperts, int nRanks,
                                            const int64_t* __restrict__ topkIndices,
                                            const float* __restrict__ topkWeights, int nTokens, int nTopk,
                                            int invalidTokenExpertId, int maxTokensPerRank,
                                            const TransportView& transport, void* workspace, int nPayloadBlocks,
                                            int* sharedMem) {
  if (blockIdx.x == 0 || static_cast<int>(blockIdx.x) > nPayloadBlocks) return;

  const int warpId = static_cast<int>(threadIdx.x) / WARP_SIZE;
  const int laneId = getLaneId();
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

  for (int tokenIdx = firstTokenIdx; tokenIdx < nTokens; tokenIdx += tokenStride) {
    const auto* inputData =
        reinterpret_cast<const mscclpp::bf16x8*>(inputTokens) + static_cast<size_t>(tokenIdx) * HiddenVectors;
    if (laneId == 0) {
      bulkBarrier->arriveAndExpect(static_cast<uint32_t>(HiddenBytes));
      mscclpp::bulkLoad(stagedToken, inputData, static_cast<uint32_t>(HiddenBytes), *bulkBarrier);
    }
    const Route route =
        prepareRoute(workspaceView, topkIndices, tokenIdx, nTopk, nLocalExperts, maxTokensPerRank, laneId);
    if (laneId == 0) bulkBarrier->wait(sendBulkPhase);
    __syncwarp();
    mscclpp::bulkFence();
    const int dstRank = route.dstRank >= 0 && route.isLeader ? route.dstRank : -1;
    if (dstRank >= 0) {
      issueTokenStore<Hidden>(output, transport, route.destinationSlot, maxTokensPerRank, stagedToken, dstRank);
    }
    sendMetadata(transport, outputTopkIdx, outputTopkWeights, topkIndices, topkWeights, route, tokenIdx, nTopk,
                 nLocalExperts, maxTokensPerRank, invalidTokenExpertId);
    completeTokenStore(workspaceView, dstRank);
    if (tokenIdx + tokenStride < nTokens) __syncwarp();
  }
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void dispatchSendMxfp8(void* output, void* outputScales, int* outputTopkIdx,
                                             float* outputTopkWeights, const void* inputTokens, int nExperts,
                                             int nRanks, const int64_t* __restrict__ topkIndices,
                                             const float* __restrict__ topkWeights, int nTokens, int nTopk,
                                             int invalidTokenExpertId, int maxTokensPerRank,
                                             const TransportView& transport, void* workspace, int nPayloadBlocks,
                                             int* sharedMem) {
  const int warpId = static_cast<int>(threadIdx.x) / WARP_SIZE;
  const int laneId = getLaneId();
  const int nWarpsPerGroup = dispatchNWarpsPerGroup(nTokens, nPayloadBlocks);
  const int nWarpGroups = DispatchNWarps / nWarpsPerGroup;
  const int warpGroupId = warpId / nWarpsPerGroup;
  const int subWarpId = warpId % nWarpsPerGroup;
  const int groupThreadId = subWarpId * WARP_SIZE + laneId;
  const int groupThreadCount = nWarpsPerGroup * WARP_SIZE;
  const int groupBarrierId = DispatchWarpGroupBarrierBase + warpGroupId;
  constexpr int HiddenVectors = Hidden / mscclpp::bf16x8::Size;
  static_assert(DispatchMinNWarpsPerGroup > 1);
  constexpr int MinQuantizeThreads = (DispatchMinNWarpsPerGroup - 1) * WARP_SIZE;
  constexpr int MaxVectorsPerThread = (HiddenVectors + MinQuantizeThreads - 1) / MinQuantizeThreads;
  const int quantizeThreadId = groupThreadId - WARP_SIZE;
  const int quantizeThreadCount = groupThreadCount - WARP_SIZE;
  const int nLocalExperts = nExperts / nRanks;
  const size_t sharedTokenStride = dispatchPayloadStride<DispatchDataType::MXFP8_E4M3>(Hidden, nTopk, 32);
  auto* stagedToken = reinterpret_cast<uint8_t*>(sharedMem) + dispatchSharedControlBytes(nRanks) +
                      static_cast<size_t>(warpGroupId) * sharedTokenStride;
  auto* stagedData = reinterpret_cast<mscclpp::f8_e4m3x8*>(stagedToken);
  auto* stagedScales = stagedToken + Hidden;
  WorkspaceView workspaceView(workspace, nRanks, nExperts);
  const int tokenStride = nPayloadBlocks * nWarpGroups;
  const int firstTokenIdx = (static_cast<int>(blockIdx.x) - 1) * nWarpGroups + warpGroupId;

  for (int tokenIdx = firstTokenIdx; tokenIdx < nTokens; tokenIdx += tokenStride) {
    Route route{-1, -1, false};
    // Warp 0 reserves destination slots while the other warps quantize, so the two latencies overlap.
    if (subWarpId == 0) {
      route = prepareRoute(workspaceView, topkIndices, tokenIdx, nTopk, nLocalExperts, maxTokensPerRank, laneId);
    } else {
      const auto* inputData =
          static_cast<const mscclpp::bf16x8*>(inputTokens) + static_cast<size_t>(tokenIdx) * HiddenVectors;
      mscclpp::bf16x8 values[MaxVectorsPerThread];
#pragma unroll
      for (int i = 0; i < MaxVectorsPerThread; ++i) {
        const int inputIdx = quantizeThreadId + i * quantizeThreadCount;
        if (inputIdx < HiddenVectors) values[i] = inputData[inputIdx];
      }
#pragma unroll
      for (int i = 0; i < MaxVectorsPerThread; ++i) {
        const int inputIdx = quantizeThreadId + i * quantizeThreadCount;
        if (inputIdx < HiddenVectors) {
          uint8_t scale;
          stagedData[inputIdx] = quantizeBf16x8ToMxfp8(values[i], scale, laneId);
          if (laneId % 4 == 0) stagedScales[inputIdx / 4] = scale;
        }
      }
    }
    mscclpp::bulkFence();
    syncNamedBarrier(groupBarrierId, groupThreadCount);

    if (subWarpId == 0) {
      const int dstRank = route.dstRank >= 0 && route.isLeader ? route.dstRank : -1;
      if (dstRank >= 0) {
        issueTokenStore<Hidden, DispatchDataType::MXFP8_E4M3>(output, transport, route.destinationSlot,
                                                              maxTokensPerRank, stagedToken, dstRank, outputScales);
      }
      sendMetadata(transport, outputTopkIdx, outputTopkWeights, topkIndices, topkWeights, route, tokenIdx, nTopk,
                   nLocalExperts, maxTokensPerRank, invalidTokenExpertId);
      completeTokenStore(workspaceView, dstRank);
    }
    syncNamedBarrier(groupBarrierId, groupThreadCount);
  }
}

MSCCLPP_DEVICE_INLINE void countRoutes(int* rankTokenCounts, const int64_t* __restrict__ topkIndices, int nTokens,
                                       int nTopk, int nRanks, int nExperts) {
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

MSCCLPP_DEVICE_INLINE void writeCounts(const TransportView& transport, const int* rankTokenCounts, int nRanks,
                                       void* recvBuffer, uint32_t epoch) {
  const int threadId = static_cast<int>(threadIdx.x);
  for (int dstRank = threadId; dstRank < nRanks; dstRank += blockDim.x) {
    auto* destinationPackets = reinterpret_cast<mscclpp::LL8Packet*>(transport.mappedBuffer(recvBuffer, dstRank));
    destinationPackets[transport.rank_].write(static_cast<uint32_t>(rankTokenCounts[dstRank]), epoch);
  }
}

MSCCLPP_DEVICE_INLINE void dispatchNotify(const TransportView& transport, int nExperts, int nRanks,
                                          const int64_t* __restrict__ topkIndices, int nTokens, int nTopk,
                                          void* recvBuffer, void* workspace, uint32_t epoch, int* sharedMem) {
  WorkspaceView workspaceView(workspace, nRanks, nExperts);
  auto* rankTokenCounts = sharedMem;
  countRoutes(rankTokenCounts, topkIndices, nTokens, nTopk, nRanks, nExperts);
  writeCounts(transport, rankTokenCounts, nRanks, recvBuffer, epoch);
  publishDispatchPayloads(transport, rankTokenCounts, nRanks, workspaceView);
}

template <int Hidden, DispatchDataType DataType>
MSCCLPP_DEVICE_INLINE void dispatchSend(void* output, void* outputScales, int* outputTopkIdx, float* outputTopkWeights,
                                        const void* inputTokens, const TransportView& transport, int nExperts,
                                        int nRanks, const int64_t* __restrict__ topkIndices,
                                        const float* __restrict__ topkWeights, int nTokens, int nTopk,
                                        int invalidTokenExpertId, int maxTokensPerRank, void* recvBuffer,
                                        void* workspace, uint32_t epoch, int* sharedMem) {
  const int nWorkerBlocks = static_cast<int>(gridDim.x) - DispatchControlBlocks;
  if (static_cast<int>(blockIdx.x) > 0 && static_cast<int>(blockIdx.x) <= nWorkerBlocks) {
    if constexpr (DataType == DispatchDataType::BF16) {
      dispatchSendBf16<Hidden>(output, outputTopkIdx, outputTopkWeights, inputTokens, nExperts, nRanks, topkIndices,
                               topkWeights, nTokens, nTopk, invalidTokenExpertId, maxTokensPerRank, transport,
                               workspace, nWorkerBlocks, sharedMem);
    } else {
      dispatchSendMxfp8<Hidden>(output, outputScales, outputTopkIdx, outputTopkWeights, inputTokens, nExperts, nRanks,
                                topkIndices, topkWeights, nTokens, nTopk, invalidTokenExpertId, maxTokensPerRank,
                                transport, workspace, nWorkerBlocks, sharedMem);
    }
  } else if (static_cast<int>(blockIdx.x) == nWorkerBlocks + 1) {
    dispatchNotify(transport, nExperts, nRanks, topkIndices, nTokens, nTopk, recvBuffer, workspace, epoch, sharedMem);
  }
}

MSCCLPP_DEVICE_INLINE void dispatchRecv(int* outputTopkIdx, float* outputTopkWeights, int* outputCount,
                                        const TransportView& transport, int nExperts, int nRanks, int nTopk,
                                        int maxTokensPerRank, int invalidTokenExpertId, void* recvBuffer,
                                        void* workspace, uint32_t epoch, int* sharedMem) {
  const int sourceRank = static_cast<int>(blockIdx.x);
  if (sourceRank >= nRanks) return;
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

  WorkspaceView workspaceView(workspace, nRanks, nExperts);
  if (threadIdx.x == 0 && nRankTokens > 0) {
    if (transport.isSelf(sourceRank)) {
      workspaceView.dispatchLocalPayloadReady_->acquire();
    } else {
      transport.baseMemoryChannels_[sourceRank].wait(-1);
    }
  }
}

#endif  // MSCCLPP_BULK_AVAILABLE

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
__global__ __launch_bounds__(DispatchNThreads,
                             1) void dispatchKernel(void* output, void* outputScales, int* outputSrcInfo,
                                                    int* outputTopkIdx, float* outputTopkWeights, int64_t* outputLayout,
                                                    int* outputCount, const int64_t* topkIndices,
                                                    const float* topkWeights, const void* inputTokens,
                                                    Workload workload, void* recvBuffer, const DeviceContext* context) {
#if MSCCLPP_BULK_AVAILABLE
  static_assert(DataType == DispatchDataType::BF16 || DataType == DispatchDataType::MXFP8_E4M3);
  extern __shared__ __align__(128) uint8_t sharedMemory[];
  auto* sharedMem = reinterpret_cast<int*>(sharedMemory);
  const int nExperts = workload.numExperts_;
  const int nRanks = context->numRanks_;
  const int nTokens = workload.numTokens_;
  const int nTopk = workload.numTopk_;
  const int invalidTokenExpertId = workload.invalidTokenExpertId_;
  const int maxTokensPerRank = workload.maxTokensPerRank_;
  const TransportView transport(context);
  const uint32_t epoch = workload.epoch_;

  dispatchSend<Hidden, DataType>(output, outputScales, outputTopkIdx, outputTopkWeights, inputTokens, transport,
                                 nExperts, nRanks, topkIndices, topkWeights, nTokens, nTopk, invalidTokenExpertId,
                                 maxTokensPerRank, recvBuffer, context->workspace_, epoch, sharedMem);

  if (static_cast<int>(blockIdx.x) < nRanks) {
    dispatchRecv(outputTopkIdx, outputTopkWeights, outputCount, transport, nExperts, nRanks, nTopk, maxTokensPerRank,
                 invalidTokenExpertId, recvBuffer, context->workspace_, epoch, sharedMem);
  }
#endif  // MSCCLPP_BULK_AVAILABLE
}

struct KernelSelector {
  template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
  static auto get() {
    return dispatchKernel<Hidden, DataType, ScaleBlockSize>;
  }
};

}  // namespace

void rankMajorDispatch(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                       float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                       const int64_t* topkIdx, const float* topkWeights, const Workload& workload, void* recvBuffer,
                       const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  dispatchAlgorithm<DispatchLayout::RANK_MAJOR, KernelSelector>(
      output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input, topkIdx,
      topkWeights, workload, recvBuffer, context, numBlocks, stream);
}

}  // namespace ep
}  // namespace mscclpp
