// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "common.cuh"
#include "common/quantization.cuh"

namespace mscclpp {
namespace ep {
namespace {

#if MSCCLPP_BULK_AVAILABLE

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
  const int laneId = getLaneId();
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
  const int laneId = getLaneId();
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
  const int laneId = getLaneId();
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

MSCCLPP_DEVICE_INLINE void dispatchNotify(const TransportView& transport, int nExperts, int nRanks,
                                          const int64_t* __restrict__ topkIndices, int nTokens, int nTopk,
                                          void* recvBuffer, void* workspace, uint32_t epoch, int* sharedMem) {
  WorkspaceView workspaceView(workspace, nRanks, nExperts);
  DispatchCountView counts(sharedMem, nRanks);
  countDispatchRoutes(counts, topkIndices, nTokens, nTopk, nRanks, nExperts);
  writeDispatchMetadata(transport, counts, nRanks, nExperts, recvBuffer, epoch);
  publishDispatchPayloads(transport, counts.rankTokenCounts_, nRanks, workspaceView);
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

MSCCLPP_DEVICE_INLINE int proportionalTaskBoundary(int nTokens, int nTasks, int nTotalTokens) {
  return nTotalTokens == 0 ? 0 : static_cast<int>(static_cast<int64_t>(nTokens) * nTasks / nTotalTokens);
}

MSCCLPP_DEVICE_INLINE void dispatchRecvScheduler(int64_t* outputLayout, int* outputCount,
                                                 const TransportView& transport, int nExperts, int nRanks,
                                                 void* recvBuffer, void* workspace, uint32_t epoch, int* sharedMem) {
  const int threadId = static_cast<int>(threadIdx.x);
  const int warpId = threadId / WARP_SIZE;
  const int laneId = getLaneId();
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

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
MSCCLPP_DEVICE_INLINE bool dispatchRecvOutput(void* output, void* outputScales, int* outputSrcInfo,
                                              int64_t* outputLayout, const DispatchPayloadView<DataType>& payloadView,
                                              void* sourcePayload, int localExpertIdx, int sourceRank,
                                              int sourceTokenIdx, int nLocalExperts, int nRanks, int nTopk,
                                              int maxTokensPerRank, WorkspaceView& workspaceView, uint8_t* sharedTile,
                                              mscclpp::BulkBarrier* bulkBarrier, uint32_t& recvBulkPhase) {
  using OutputType = DispatchElementType<DataType>;
  constexpr size_t OutputBytes = static_cast<size_t>(Hidden) * sizeof(OutputType);
  constexpr int NumScales = DataType == DispatchDataType::BF16 ? 0 : Hidden / ScaleBlockSize;
  const int laneId = getLaneId();
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
  const int laneId = getLaneId();
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
    hasPendingStore = dispatchRecvOutput<Hidden, DataType, ScaleBlockSize>(
        output, outputScales, outputSrcInfo, outputLayout, payloadView, sourcePayload, localExpertIdx, sourceRank,
        sourceTokenIdx, nLocalExperts, nRanks, nTopk, maxTokensPerRank, workspaceView, sharedTile, bulkBarrier,
        recvBulkPhase);
  }

  if (hasPendingStore) mscclpp::bulkStoreWait();
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
  extern __shared__ __align__(128) uint8_t sharedMemory[];
  auto* sharedMem = reinterpret_cast<int*>(sharedMemory);
  const int nWorkerBlocks = static_cast<int>(gridDim.x) - DispatchControlBlocks;
  const int nExperts = workload.numExperts_;
  const int nRanks = context->numRanks_;
  const int nTokens = workload.numTokens_;
  const int nTopk = workload.numTopk_;
  const int maxTokensPerRank = workload.maxTokensPerRank_;
  const TransportView transport(context);
  const uint32_t epoch = workload.epoch_;

  dispatchSend<Hidden, DataType, ScaleBlockSize>(inputTokens, transport, nExperts, nRanks, topkIndices, topkWeights,
                                                 nTokens, nTopk, maxTokensPerRank, recvBuffer, context->workspace_,
                                                 epoch, sharedMem);

  if (static_cast<int>(blockIdx.x) == 0) {
    dispatchRecvScheduler(outputLayout, outputCount, transport, nExperts, nRanks, recvBuffer, context->workspace_,
                          epoch, sharedMem);
  } else if (static_cast<int>(blockIdx.x) <= nWorkerBlocks) {
    dispatchRecvWorker<Hidden, DataType, ScaleBlockSize>(output, outputScales, outputSrcInfo, outputLayout, nExperts,
                                                         context->rank_, nRanks, nTopk, maxTokensPerRank, recvBuffer,
                                                         context->workspace_, epoch, sharedMem);
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

void expertMajorDispatch(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                         float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                         const int64_t* topkIdx, const float* topkWeights, const Workload& workload, void* recvBuffer,
                         const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  dispatchAlgorithm<DispatchLayout::EXPERT_MAJOR, KernelSelector>(
      output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input, topkIdx,
      topkWeights, workload, recvBuffer, context, numBlocks, stream);
}

size_t workspaceSize(int numRanks, int numExperts, int maxTokensPerRank, int numTopk) {
  return workspaceBytes(numRanks, numExperts, maxTokensPerRank, numTopk);
}

}  // namespace ep
}  // namespace mscclpp
