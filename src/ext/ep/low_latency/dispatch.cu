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
      destinationSlot = atomicAdd(workspaceView.rankPayloadSlots_ + dstRank, 1);
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
                                                     int maxTokensPerRank, int rank, size_t metadataBytes,
                                                     size_t payloadStride, void* recvBuffer, void* rdmaBufferBase,
                                                     void* const* peerRecvBuffers, int laneId) {
  const int destinationSlot = laneId < nTopk ? destinationSlots[laneId] : -1;
  if (destinationSlot < 0) return;

  const int dstRank = payloadView.topKIndices(stagedPayload)[laneId] / nLocalExperts;
  void* destinationBuffer =
      dstRank == rank ? recvBuffer : peerBufferPtr(recvBuffer, rdmaBufferBase, peerRecvBuffers[dstRank]);
  auto* destinationPayload = reinterpret_cast<uint8_t*>(destinationBuffer) + metadataBytes +
                             (static_cast<size_t>(rank) * maxTokensPerRank + destinationSlot) * payloadStride;
  issueTmaStore(destinationPayload, stagedPayload, static_cast<uint32_t>(payloadView.numBytes_));
  waitBulkGroup();
  (void)mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(workspaceView.rankPayloadCompletions_ + dstRank, 1,
                                                           mscclpp::memoryOrderRelease);
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void dispatchSendBf16(const void* inputTokens, int nExperts, int rank, int nRanks,
                                            const int64_t* __restrict__ topkIndices,
                                            const float* __restrict__ topkWeights, int nTokens, int nTopk,
                                            int maxTokensPerRank, void* recvBuffer, void* const* peerRecvBuffers,
                                            void* rdmaBufferBase, void* workspace, int* sharedMem) {
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
    __syncwarp();
  }

  for (int tokenIdx = firstTokenIdx; tokenIdx < nTokens; tokenIdx += tokenStride) {
    const auto* inputData =
        reinterpret_cast<const mscclpp::bf16x8*>(inputTokens) + static_cast<size_t>(tokenIdx) * HiddenVectors;
    if (laneId == 0) {
      issueTmaLoad(inputData, stagedPayload, tmaBarrier, static_cast<uint32_t>(HiddenBytes));
    }
    stageDispatchPayloadMetadata<DispatchDataType::BF16>(payloadView, stagedPayload, destinationSlots, workspaceView,
                                                         topkIndices, topkWeights, tokenIdx, nTopk, nLocalExperts,
                                                         maxTokensPerRank, rank, laneId);
    if (laneId == 0) waitTmaLoad(tmaBarrier, sendTmaPhase);
    __syncwarp();
    fenceProxyAsyncSharedCta();
    sendStagedDispatchPayload<DispatchDataType::BF16>(
        payloadView, stagedPayload, destinationSlots, workspaceView, nTopk, nLocalExperts, maxTokensPerRank, rank,
        metadataBytes, payloadStride, recvBuffer, rdmaBufferBase, peerRecvBuffers, laneId);
    __syncwarp();
  }
}

template <int Hidden, int ScaleBlockSize>
MSCCLPP_DEVICE_INLINE void dispatchSendFp8(const void* inputTokens, int nExperts, int rank, int nRanks,
                                           const int64_t* __restrict__ topkIndices,
                                           const float* __restrict__ topkWeights, int nTokens, int nTopk,
                                           int maxTokensPerRank, void* recvBuffer, void* const* peerRecvBuffers,
                                           void* rdmaBufferBase, void* workspace, int* sharedMem) {
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
  const DispatchPayloadView<DispatchDataType::FP8_E4M3> payloadView(Hidden, nTopk, ScaleBlockSize);
  const size_t payloadStride = dispatchPayloadStride<DispatchDataType::FP8_E4M3>(Hidden, nTopk, ScaleBlockSize);
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
      stageDispatchPayloadMetadata<DispatchDataType::FP8_E4M3>(payloadView, stagedPayload, destinationSlots,
                                                               workspaceView, topkIndices, topkWeights, tokenIdx, nTopk,
                                                               nLocalExperts, maxTokensPerRank, rank, laneId);
    }
    for (int inputIdx = groupThreadId; inputIdx < HiddenVectors; inputIdx += groupThreadCount) {
      outputData[inputIdx] = quantizeBf16x8ToFp8E4M3<ScaleBlockSize>(
          inputData[inputIdx], outputScales + inputIdx * mscclpp::bf16x8::Size / ScaleBlockSize, laneId);
    }
    syncNamedBarrier(groupBarrierId, groupThreadCount);

    if (subWarpId == 0) {
      fenceProxyAsyncSharedCta();
      sendStagedDispatchPayload<DispatchDataType::FP8_E4M3>(
          payloadView, stagedPayload, destinationSlots, workspaceView, nTopk, nLocalExperts, maxTokensPerRank, rank,
          metadataBytes, payloadStride, recvBuffer, rdmaBufferBase, peerRecvBuffers, laneId);
    }
    syncNamedBarrier(groupBarrierId, groupThreadCount);
  }
}

MSCCLPP_DEVICE_INLINE void dispatchNotify(mscclpp::BaseMemoryChannelDeviceHandle* baseMemoryChannels, int nExperts,
                                          int rank, int nRanks, const int64_t* __restrict__ topkIndices, int nTokens,
                                          int nTopk, void* recvBuffer, void* const* peerRecvBuffers,
                                          void* rdmaBufferBase, void* workspace, uint32_t metadataFlag,
                                          int* sharedMem) {
  const int threadId = static_cast<int>(threadIdx.x);
  const int warpId = threadId / WARP_SIZE;
  const int laneId = get_lane_id();
  const int nLocalExperts = nExperts / nRanks;
  WorkspaceView workspaceView(workspace, nRanks, nExperts);
  int* sharedRankTokenCounts = sharedMem;
  int* sharedExpertTokenCounts = sharedRankTokenCounts + nRanks;
  for (int idx = threadId; idx < nRanks + nExperts; idx += blockDim.x) {
    sharedRankTokenCounts[idx] = 0;
  }
  __syncthreads();
  for (int tokenIdx = warpId; tokenIdx < nTokens; tokenIdx += DispatchNWarps) {
    const int routedExpertIdx = laneId < nTopk ? static_cast<int>(topkIndices[tokenIdx * nTopk + laneId]) : -1;
    const int dstRank = routedExpertIdx >= 0 ? routedExpertIdx / nLocalExperts : -1;
    if (routedExpertIdx >= 0) {
      atomicAdd_block(sharedExpertTokenCounts + routedExpertIdx, 1);
    }
    if (isFirstLaneForRank(dstRank, laneId) && dstRank >= 0) {
      atomicAdd_block(sharedRankTokenCounts + dstRank, 1);
    }
  }
  __syncthreads();

  for (int dstRank = threadId; dstRank < nRanks; dstRank += blockDim.x) {
    void* destinationBuffer =
        dstRank == rank ? recvBuffer : peerBufferPtr(recvBuffer, rdmaBufferBase, peerRecvBuffers[dstRank]);
    reinterpret_cast<mscclpp::LL8Packet*>(destinationBuffer)[rank].write(
        static_cast<uint32_t>(sharedRankTokenCounts[dstRank]), metadataFlag);
  }
  for (int expertIdx = threadId; expertIdx < nExperts; expertIdx += blockDim.x) {
    const int dstRank = expertIdx / nLocalExperts;
    const int localExpertIdx = expertIdx % nLocalExperts;
    void* destinationBuffer =
        dstRank == rank ? recvBuffer : peerBufferPtr(recvBuffer, rdmaBufferBase, peerRecvBuffers[dstRank]);
    reinterpret_cast<mscclpp::LL8Packet*>(destinationBuffer)[nRanks + rank * nLocalExperts + localExpertIdx].write(
        static_cast<uint32_t>(sharedExpertTokenCounts[expertIdx]), metadataFlag);
  }

  for (int dstRank = threadId; dstRank < nRanks; dstRank += blockDim.x) {
    const int expectedPayloadCount = sharedRankTokenCounts[dstRank];
    if (expectedPayloadCount > 0) {
      while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(workspaceView.rankPayloadCompletions_ + dstRank,
                                                            mscclpp::memoryOrderAcquire) != expectedPayloadCount);
    }
    workspaceView.rankPayloadSlots_[dstRank] = 0;
    workspaceView.rankPayloadCompletions_[dstRank] = 0;
    if (expectedPayloadCount == 0) continue;
    if (dstRank == rank) {
      workspaceView.localPayloadReady_->release();
    } else {
      baseMemoryChannels[dstRank].signal();
    }
  }
  __syncthreads();
}

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
MSCCLPP_DEVICE_INLINE void dispatchSend(const void* inputTokens,
                                        mscclpp::BaseMemoryChannelDeviceHandle* baseMemoryChannels, int nExperts,
                                        int rank, int nRanks, const int64_t* __restrict__ topkIndices,
                                        const float* __restrict__ topkWeights, int nTokens, int nTopk,
                                        int maxTokensPerRank, void* recvBuffer, void* const* peerRecvBuffers,
                                        void* rdmaBufferBase, void* workspace, uint32_t metadataFlag, int* sharedMem) {
  const int nWorkerBlocks = static_cast<int>(gridDim.x) - DispatchControlBlocks;
  if (static_cast<int>(blockIdx.x) > 0 && static_cast<int>(blockIdx.x) <= nWorkerBlocks) {
    if constexpr (DataType == DispatchDataType::BF16) {
      dispatchSendBf16<Hidden>(inputTokens, nExperts, rank, nRanks, topkIndices, topkWeights, nTokens, nTopk,
                               maxTokensPerRank, recvBuffer, peerRecvBuffers, rdmaBufferBase, workspace, sharedMem);
    } else {
      dispatchSendFp8<Hidden, ScaleBlockSize>(inputTokens, nExperts, rank, nRanks, topkIndices, topkWeights, nTokens,
                                              nTopk, maxTokensPerRank, recvBuffer, peerRecvBuffers, rdmaBufferBase,
                                              workspace, sharedMem);
    }
  } else if (static_cast<int>(blockIdx.x) == nWorkerBlocks + 1) {
    dispatchNotify(baseMemoryChannels, nExperts, rank, nRanks, topkIndices, nTokens, nTopk, recvBuffer, peerRecvBuffers,
                   rdmaBufferBase, workspace, metadataFlag, sharedMem);
  }
}

MSCCLPP_DEVICE_INLINE int proportionalTaskBoundary(int nTokens, int nTasks, int nTotalTokens) {
  return nTotalTokens == 0 ? 0 : static_cast<int>(static_cast<int64_t>(nTokens) * nTasks / nTotalTokens);
}

MSCCLPP_DEVICE_INLINE void dispatchRecvScheduler(int64_t* outputLayout, int* outputCount,
                                                 mscclpp::BaseMemoryChannelDeviceHandle* baseMemoryChannels,
                                                 int nExperts, int rank, int nRanks, void* recvBuffer, void* workspace,
                                                 uint32_t metadataFlag, int* sharedMem) {
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
        sourceRank < nRanks ? static_cast<int>(rankTokenCounts[sourceRank].read(metadataFlag, -1)) : 0;
    const int activeRank = nRankTokens > 0 ? 1 : 0;
    int rankTokenPrefix = warpInclusiveSum(nRankTokens, laneId);
    int activeRankPrefix = warpInclusiveSum(activeRank, laneId);
    if (laneId == WARP_SIZE - 1) {
      sharedMem[warpId] = rankTokenPrefix;
      sharedMem[nRankWarps + warpId] = activeRankPrefix;
    }
    asm volatile("bar.sync %0, %1;" ::"r"(DispatchSchedulerPrefixBarrier), "r"(nRankWarps * WARP_SIZE) : "memory");

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
    asm volatile("bar.sync %0, %1;" ::"r"(DispatchSchedulerPrefixBarrier), "r"(nRankWarps * WARP_SIZE) : "memory");

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
        workspaceView.recvTasks_[rankTaskBegin + rankTaskIdx] = {sourceRank, nRankTokens * rankTaskIdx / nRankTasks,
                                                                 nRankTokens * (rankTaskIdx + 1) / nRankTasks};
      }
    }
    if (threadId == 0) *workspaceView.nRecvTasks_ = nTasks;

    asm volatile("bar.sync %0, %1;" ::"r"(DispatchSchedulerReadyBarrier), "r"((nRankWarps + nLayoutWarps) * WARP_SIZE)
                 : "memory");
    if (threadId == 0) {
      mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(workspaceView.tasksAssignedEpoch_, metadataFlag,
                                                           mscclpp::memoryOrderRelease);
    }

    if (sourceRank < nRanks && nRankTokens > 0) {
      if (sourceRank == rank) {
        workspaceView.localPayloadReady_->acquire();
      } else {
        baseMemoryChannels[sourceRank].wait(-1);
      }
      mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(workspaceView.rankReadyEpochs_ + sourceRank, metadataFlag,
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
            static_cast<int>(expertTokenCounts[sourceRank * nLocalExperts + localExpertIdx].read(metadataFlag, -1));
        outputLayout[localExpertIdx * nRanks + sourceRank] = pack2<int, int64_t>(nExpertTokens, outputOffset);
        outputOffset += nExpertTokens;
      }
      outputCount[localExpertIdx] = outputOffset;
    }
    asm volatile("bar.sync %0, %1;" ::"r"(DispatchSchedulerReadyBarrier), "r"((nRankWarps + nLayoutWarps) * WARP_SIZE)
                 : "memory");
  }
}

MSCCLPP_DEVICE_INLINE bool acquireRecvTask(RecvTask& task, WorkspaceView& workspaceView, uint32_t metadataFlag,
                                           int* sharedMem) {
  auto* sharedTask = reinterpret_cast<RecvTask*>(sharedMem);
  const int taskIdx = static_cast<int>(blockIdx.x) - 1;
  if (threadIdx.x == 0) {
    while (mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(workspaceView.tasksAssignedEpoch_,
                                                               mscclpp::memoryOrderAcquire) != metadataFlag);
    if (taskIdx < *workspaceView.nRecvTasks_) {
      task = workspaceView.recvTasks_[taskIdx];
      while (mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(workspaceView.rankReadyEpochs_ + task.sourceRank_,
                                                                 mscclpp::memoryOrderAcquire) != metadataFlag);
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
MSCCLPP_DEVICE_INLINE void dispatchRecvWorker(void* output, float* outputScales, int* outputSrcInfo,
                                              int64_t* outputLayout, int nExperts, int rank, int nRanks, int nTopk,
                                              int maxTokensPerRank, void* recvBuffer, void* workspace,
                                              uint32_t metadataFlag, int* sharedMem) {
#if defined(__CUDA_ARCH__)
  static_assert(__CUDA_ARCH__ >= 900, "TMA recv requires SM90 or newer");
#endif
  const int threadId = static_cast<int>(threadIdx.x);
  const int warpId = threadId / WARP_SIZE;
  const int laneId = get_lane_id();
  WorkspaceView workspaceView(workspace, nRanks, nExperts);
  RecvTask task;
  if (!acquireRecvTask(task, workspaceView, metadataFlag, sharedMem)) return;
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
  constexpr int NumScales = DataType == DispatchDataType::BF16 ? 0 : Hidden / ScaleBlockSize;
  const int nOutputSlotsPerExpert = nRanks * maxTokensPerRank;
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
      issueTmaLoad(payloadView.template data<OutputType>(sourcePayload), sharedTile, tmaBarrier,
                   static_cast<uint32_t>(OutputBytes));
    }
    __syncwarp();

    const int routedExpertIdx = laneId < nTopk ? payloadView.topKIndices(sourcePayload)[laneId] : -1;
    const int localExpertIdx = routedExpertIdx >= globalExpertBase && routedExpertIdx < globalExpertEnd
                                   ? routedExpertIdx - globalExpertBase
                                   : -1;
    const int sourceTokenIdx = warpBroadcast(
        laneId == 0 ? *payloadView.srcTokenGlobalIdx(sourcePayload) - sourceRank * maxTokensPerRank : 0, 0);
    int outputTokenIdx = -1;
    int combineInputOffset = -1;
    if (localExpertIdx >= 0) {
      int expertTokenCount;
      int outputOffset;
      unpack2(outputLayout[localExpertIdx * nRanks + sourceRank], expertTokenCount, outputOffset);
      const int copiedTokenIdx =
          atomicAdd(workspaceView.recvExpertCopiedCounts_ + sourceRank * nLocalExperts + localExpertIdx, 1);
      EP_DEVICE_ASSERT(copiedTokenIdx < expertTokenCount);
      if (copiedTokenIdx == expertTokenCount - 1) {
        workspaceView.recvExpertCopiedCounts_[sourceRank * nLocalExperts + localExpertIdx] = 0;
      }
      outputTokenIdx = outputOffset + copiedTokenIdx;
      outputSrcInfo[static_cast<size_t>(localExpertIdx) * nOutputSlotsPerExpert + outputTokenIdx] = sourceTokenIdx;
      combineInputOffset = localExpertIdx * nOutputSlotsPerExpert + outputTokenIdx;
    }
    if constexpr (DataType != DispatchDataType::BF16) {
      const auto* sourceScales = payloadView.scaleFactors(sourcePayload);
      // Each top-k lane may create a row for a different local expert. All lanes
      // cooperate to copy the shared payload's scale vector to every such row.
      for (int topkLane = 0; topkLane < nTopk; ++topkLane) {
        const int scaleLocalExpertIdx = warpBroadcast(localExpertIdx, topkLane);
        const int scaleOutputTokenIdx = warpBroadcast(outputTokenIdx, topkLane);
        if (scaleLocalExpertIdx < 0) continue;
        for (int scaleIdx = laneId; scaleIdx < NumScales; scaleIdx += WARP_SIZE) {
          outputScales[(static_cast<size_t>(scaleLocalExpertIdx) * NumScales + scaleIdx) * nOutputSlotsPerExpert +
                       scaleOutputTokenIdx] = sourceScales[scaleIdx];
        }
      }
    }
    if (laneId < nTopk) payloadView.topKIndices(sourcePayload)[laneId] = combineInputOffset;

    if (laneId == 0) waitTmaLoad(tmaBarrier, recvTmaPhase);
    __syncwarp();
    fenceProxyAsyncSharedCta();

    if (localExpertIdx >= 0) {
      auto* outputData = reinterpret_cast<uint8_t*>(output) +
                         (static_cast<size_t>(localExpertIdx) * nOutputSlotsPerExpert + outputTokenIdx) * OutputBytes;
      issueTmaStore(outputData, sharedTile, static_cast<uint32_t>(OutputBytes));
      hasPendingStore = true;
    } else {
      hasPendingStore = false;
    }
    __syncwarp();
  }

  if (hasPendingStore) waitBulkGroup();
}

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
__global__ __launch_bounds__(DispatchNThreads, 1) void dispatchKernel(
    void* output, float* outputScales, int* outputSrcInfo, int64_t* outputLayout, int* outputCount,
    mscclpp::BaseMemoryChannelDeviceHandle* baseMemoryChannels, int nExperts, int rank, int nRanks,
    const int64_t* __restrict__ topkIndices, const float* __restrict__ topkWeights, const void* inputTokens,
    int nTokens, int nTopk, int maxTokensPerRank, void* recvBuffer, void* rdmaBufferBase, void* const* peerRecvBuffers,
    void* workspace) {
  extern __shared__ __align__(128) uint8_t sharedMemory[];
  auto* sharedMem = reinterpret_cast<int*>(sharedMemory);
  const int nWorkerBlocks = static_cast<int>(gridDim.x) - DispatchControlBlocks;
  WorkspaceView workspaceView(workspace, nRanks, nExperts);
  const uint32_t metadataFlag = *workspaceView.metadataEpoch_ + 1;

  dispatchSend<Hidden, DataType, ScaleBlockSize>(inputTokens, baseMemoryChannels, nExperts, rank, nRanks, topkIndices,
                                                 topkWeights, nTokens, nTopk, maxTokensPerRank, recvBuffer,
                                                 peerRecvBuffers, rdmaBufferBase, workspace, metadataFlag, sharedMem);

  if (static_cast<int>(blockIdx.x) == 0) {
    dispatchRecvScheduler(outputLayout, outputCount, baseMemoryChannels, nExperts, rank, nRanks, recvBuffer, workspace,
                          metadataFlag, sharedMem);
  } else if (static_cast<int>(blockIdx.x) <= nWorkerBlocks) {
    dispatchRecvWorker<Hidden, DataType, ScaleBlockSize>(output, outputScales, outputSrcInfo, outputLayout, nExperts,
                                                         rank, nRanks, nTopk, maxTokensPerRank, recvBuffer, workspace,
                                                         metadataFlag, sharedMem);
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *workspaceView.metadataEpoch_ = metadataFlag;
  }
}

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
inline void dispatchHiddenMode(void* output, float* outputScales, int* outputSrcInfo, int64_t* outputLayout,
                               int* outputCount, const void* input, const int64_t* topkIdx, const float* topkWeights,
                               const low_latency::Workload& workload, void* recvBuffer,
                               const low_latency::CommContext& comm, void* workspace, int numBlocks,
                               cudaStream_t stream) {
  static_assert(Hidden == 4096 || Hidden == 7168 || Hidden == 8192 || Hidden == 9216);
  using OutputType = DispatchElementType<DataType>;
  constexpr int NRecvTmaWorkers = tmaWorkerCount<Hidden, OutputType, DispatchMaxNRecvTmaWorkers>();
  static_assert(NRecvTmaWorkers > 0);
  const int nExperts = workload.numExperts_;
  const int rank = comm.rank_;
  const int nRanks = comm.numRanks_;
  const int nTokens = workload.numTokens_;
  const int nTopk = workload.numTopk_;

  const size_t dynamicSharedBytes = dispatchSharedBytes<Hidden, DataType, ScaleBlockSize>(nRanks, nExperts, nTopk);
  static thread_local KernelConfigCache kernelConfig;
  const int residentBlocks = configureKernel(dispatchKernel<Hidden, DataType, ScaleBlockSize>, DispatchNThreads,
                                             dynamicSharedBytes, comm, kernelConfig);
  EP_HOST_ASSERT(residentBlocks >= numBlocks);
  dispatchKernel<Hidden, DataType, ScaleBlockSize>
      <<<dim3(numBlocks), dim3(DispatchNThreads), dynamicSharedBytes, stream>>>(
          output, outputScales, outputSrcInfo, outputLayout, outputCount, comm.baseMemoryChannels_, nExperts, rank,
          nRanks, topkIdx, topkWeights, input, nTokens, nTopk, workload.maxTokensPerRank_, recvBuffer,
          comm.rdmaBufferBase_, comm.peerBases_, workspace);
  CUDA_CHECK(cudaGetLastError());
}

template <int Hidden>
inline void dispatchHidden(void* output, float* outputScales, int* outputSrcInfo, int64_t* outputLayout,
                           int* outputCount, const void* input, const int64_t* topkIdx, const float* topkWeights,
                           const low_latency::Workload& workload, void* recvBuffer,
                           const low_latency::CommContext& comm, void* workspace, int numBlocks, cudaStream_t stream) {
  switch (workload.dispatchDataType_) {
    case DispatchDataType::BF16:
      return dispatchHiddenMode<Hidden, DispatchDataType::BF16, 0>(output, outputScales, outputSrcInfo, outputLayout,
                                                                   outputCount, input, topkIdx, topkWeights, workload,
                                                                   recvBuffer, comm, workspace, numBlocks, stream);
    case DispatchDataType::FP8_E4M3:
      return dispatchHiddenMode<Hidden, DispatchDataType::FP8_E4M3, 128>(
          output, outputScales, outputSrcInfo, outputLayout, outputCount, input, topkIdx, topkWeights, workload,
          recvBuffer, comm, workspace, numBlocks, stream);
    case DispatchDataType::MXFP8_E4M3:
      EP_HOST_ASSERT(false && "MXFP8 dispatch is not implemented");
  }
  EP_HOST_ASSERT(false && "unsupported dispatch data type");
}

inline void dispatch(void* output, float* outputScales, int* outputSrcInfo, int64_t* outputLayout, int* outputCount,
                     const void* input, const int64_t* topkIdx, const float* topkWeights,
                     const low_latency::Workload& workload, void* recvBuffer, const low_latency::CommContext& comm,
                     void* workspace, int numBlocks, cudaStream_t stream) {
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
  EP_HOST_ASSERT(isSupportedDispatchDataType(workload.dispatchDataType_));
  EP_HOST_ASSERT(workload.dispatchDataType_ == DispatchDataType::BF16 || outputScales != nullptr);
  EP_HOST_ASSERT(outputSrcInfo != nullptr);
  EP_HOST_ASSERT(outputLayout != nullptr);
  EP_HOST_ASSERT(outputCount != nullptr);
  EP_HOST_ASSERT(input != nullptr);
  EP_HOST_ASSERT(topkIdx != nullptr);
  EP_HOST_ASSERT(recvBuffer != nullptr);
  EP_HOST_ASSERT(comm.rdmaBufferBase_ != nullptr);
  EP_HOST_ASSERT(comm.peerBases_ != nullptr);
  EP_HOST_ASSERT(workspace != nullptr);

  switch (workload.hidden_) {
    case 4096:
      return dispatchHidden<4096>(output, outputScales, outputSrcInfo, outputLayout, outputCount, input, topkIdx,
                                  topkWeights, workload, recvBuffer, comm, workspace, numBlocks, stream);
    case 7168:
      return dispatchHidden<7168>(output, outputScales, outputSrcInfo, outputLayout, outputCount, input, topkIdx,
                                  topkWeights, workload, recvBuffer, comm, workspace, numBlocks, stream);
    case 8192:
      return dispatchHidden<8192>(output, outputScales, outputSrcInfo, outputLayout, outputCount, input, topkIdx,
                                  topkWeights, workload, recvBuffer, comm, workspace, numBlocks, stream);
    case 9216:
      return dispatchHidden<9216>(output, outputScales, outputSrcInfo, outputLayout, outputCount, input, topkIdx,
                                  topkWeights, workload, recvBuffer, comm, workspace, numBlocks, stream);
    default:
      EP_HOST_ASSERT(false && "unsupported optimized low-latency hidden size");
  }
}

}  // namespace detail

size_t workspaceSize(int numRanks, int numExperts) { return detail::workspaceBytes(numRanks, numExperts); }

void dispatch(void* output, float* outputScales, int* outputSrcInfo, int64_t* outputLayout, int* outputCount,
              const void* input, const int64_t* topkIdx, const float* topkWeights, const Workload& workload,
              void* recvBuffer, const CommContext& comm, void* workspace, int numBlocks, cudaStream_t stream) {
  detail::dispatch(output, outputScales, outputSrcInfo, outputLayout, outputCount, input, topkIdx, topkWeights,
                   workload, recvBuffer, comm, workspace, numBlocks, stream);
}

}  // namespace low_latency
}  // namespace ep
}  // namespace mscclpp
