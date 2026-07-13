// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include <mscclpp/memory_channel_device.hpp>

#include "api.cuh"
#include "config.cuh"
#include "device_helpers.cuh"
#include "exception.cuh"

namespace mscclpp {
namespace ep {
namespace low_latency {
namespace detail {

template <int Hidden>
MSCCLPP_DEVICE_INLINE void dispatchSend(const void* inputTokens,
                                        mscclpp::BaseMemoryChannelDeviceHandle* baseMemoryChannels, int nExperts,
                                        int rank, int nRanks, const int64_t* __restrict__ topkIndices,
                                        const float* __restrict__ topkWeights, int nTokens, int nTopk,
                                        int maxTokensPerRank, void* recvBuffer, void* const* peerRecvBuffers,
                                        void* rdmaBufferBase, void* workspace, uint32_t metadataFlag, int* sharedMem) {
  const int threadId = static_cast<int>(threadIdx.x);
  const int warpId = threadId / WARP_SIZE;
  const int laneId = get_lane_id();
  const int nWorkerBlocks = static_cast<int>(gridDim.x) - DispatchControlBlocks;
  const int notifyBlockIdx = nWorkerBlocks + 1;
  const int nLocalExperts = nExperts / nRanks;
  const size_t metadataBytes = dispatchMetadataBytes(nRanks, nExperts);
  WorkspaceView workspaceView(workspace, nRanks, nExperts);
  if (static_cast<int>(blockIdx.x) > 0 && static_cast<int>(blockIdx.x) <= nWorkerBlocks) {
    const int senderBlockIdx = static_cast<int>(blockIdx.x) - 1;
    const int nWarpsPerGroup = dispatchNWarpsPerGroup(nTokens, nWorkerBlocks);
    const int nWarpGroups = DispatchNWarps / nWarpsPerGroup;
    const int warpGroupId = warpId / nWarpsPerGroup;
    const int subWarpId = warpId % nWarpsPerGroup;
    const low_latency::PayloadView<__bfloat16> payloadView(Hidden, nTopk);
    const size_t payloadStride = dispatchPayloadStride(Hidden, nTopk);
    constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(__bfloat16);
    constexpr int HiddenInt4Count = HiddenBytes / sizeof(int4);
    auto* sharedPayloadBase = reinterpret_cast<uint8_t*>(sharedMem) + dispatchSharedControlBytes(nRanks);
    auto* sendTmaBarriers = reinterpret_cast<uint64_t*>(sharedPayloadBase + DispatchMaxNWarpGroups * payloadStride);

    if (subWarpId == 0) {
      const int tokenStride = nWorkerBlocks * nWarpGroups;
      const int firstTokenIdx = senderBlockIdx * nWarpGroups + warpGroupId;
      auto* stagedPayload = sharedPayloadBase + static_cast<size_t>(warpGroupId) * payloadStride;
      auto* destinationSlots = sharedMem + warpGroupId * WARP_SIZE;
      auto* tmaBarrier = sendTmaBarriers + warpGroupId;
      uint32_t sendTmaPhase = 0;
      if (firstTokenIdx < nTokens) {
        if (laneId == 0) initTmaLoadBarrier(tmaBarrier);
        __syncwarp();
      }

      for (int tokenIdx = firstTokenIdx; tokenIdx < nTokens; tokenIdx += tokenStride) {
        const auto* inputData =
            reinterpret_cast<const int4*>(inputTokens) + static_cast<size_t>(tokenIdx) * HiddenInt4Count;
        if (laneId == 0) {
          issueTmaLoad(inputData, stagedPayload, tmaBarrier, static_cast<uint32_t>(HiddenBytes));
        }
        const int routedExpertIdx = laneId < nTopk ? static_cast<int>(topkIndices[tokenIdx * nTopk + laneId]) : -1;
        const int dstRank = routedExpertIdx >= 0 ? routedExpertIdx / nLocalExperts : -1;
        const bool firstLaneForRank = isFirstLaneForRank(dstRank, laneId);
        const bool shouldSend = dstRank >= 0 && firstLaneForRank;
        if (laneId < nTopk) {
          int destinationSlot = -1;
          if (shouldSend) {
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
          waitTmaLoad(tmaBarrier, sendTmaPhase);
        }
        __syncwarp();
        fenceProxyAsyncSharedCta();
        bool hasPendingStore = false;
        const int destinationSlot = laneId < nTopk ? destinationSlots[laneId] : -1;
        if (destinationSlot >= 0) {
          const int dstRank = payloadView.topKIndices(stagedPayload)[laneId] / nLocalExperts;
          void* destinationBuffer =
              dstRank == rank ? recvBuffer : peerBufferPtr(recvBuffer, rdmaBufferBase, peerRecvBuffers[dstRank]);
          auto* destinationPayload = reinterpret_cast<uint8_t*>(destinationBuffer) + metadataBytes +
                                     (static_cast<size_t>(rank) * maxTokensPerRank + destinationSlot) * payloadStride;
          issueTmaStore(destinationPayload, stagedPayload, static_cast<uint32_t>(payloadView.numBytes_));
          hasPendingStore = true;
        }
        if (hasPendingStore) {
          waitBulkGroup();
        }
        __syncwarp();
        if (destinationSlot >= 0) {
          const int dstRank = payloadView.topKIndices(stagedPayload)[laneId] / nLocalExperts;
          (void)mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(workspaceView.rankPayloadCompletions_ + dstRank, 1,
                                                                   mscclpp::memoryOrderRelease);
        }
      }
      asm volatile("bar.sync %0, %1;" ::"r"(DispatchMaxNWarpGroups + 1), "r"(nWarpGroups * WARP_SIZE) : "memory");
    }
  } else if (static_cast<int>(blockIdx.x) == notifyBlockIdx) {
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
    // send metadata via packet format
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
    asm volatile("bar.sync %0, %1;" ::"r"(DispatchMaxNWarpGroups + 1), "r"(nRankWarps * WARP_SIZE) : "memory");

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
    asm volatile("bar.sync %0, %1;" ::"r"(DispatchMaxNWarpGroups + 1), "r"(nRankWarps * WARP_SIZE) : "memory");

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

    asm volatile("bar.sync %0, %1;" ::"r"(DispatchMaxNWarpGroups + 2), "r"((nRankWarps + nLayoutWarps) * WARP_SIZE)
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
    asm volatile("bar.sync %0, %1;" ::"r"(DispatchMaxNWarpGroups + 2), "r"((nRankWarps + nLayoutWarps) * WARP_SIZE)
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

template <int Hidden>
MSCCLPP_DEVICE_INLINE void dispatchRecvWorker(void* output, int* outputSrcInfo, int64_t* outputLayout, int nExperts,
                                              int rank, int nRanks, int nTopk, int maxTokensPerRank, void* recvBuffer,
                                              void* workspace, uint32_t metadataFlag, int* sharedMem) {
#if defined(__CUDA_ARCH__)
  static_assert(__CUDA_ARCH__ >= 900, "TMA recv requires SM90 or newer");
#endif
  const int threadId = static_cast<int>(threadIdx.x);
  const int warpId = threadId / WARP_SIZE;
  const int laneId = get_lane_id();
  WorkspaceView workspaceView(workspace, nRanks, nExperts);
  RecvTask task;
  if (!acquireRecvTask(task, workspaceView, metadataFlag, sharedMem)) return;
  constexpr int NRecvTmaWorkers = tmaWorkerCount<Hidden, DispatchMaxNRecvTmaWorkers>();
  if (warpId >= NRecvTmaWorkers) return;

  const int nLocalExperts = nExperts / nRanks;
  const int sourceRank = task.sourceRank_;
  const int globalExpertBase = rank * nLocalExperts;
  const int globalExpertEnd = globalExpertBase + nLocalExperts;
  const low_latency::PayloadView<__bfloat16> payloadView(Hidden, nTopk);
  const size_t payloadStride = dispatchPayloadStride(Hidden, nTopk);
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(__bfloat16);
  constexpr size_t TileBytes = HiddenBytes;
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
      issueTmaLoad(payloadView.template data<int4>(sourcePayload), sharedTile, tmaBarrier,
                   static_cast<uint32_t>(HiddenBytes));
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
    if (laneId < nTopk) payloadView.topKIndices(sourcePayload)[laneId] = combineInputOffset;

    if (laneId == 0) waitTmaLoad(tmaBarrier, recvTmaPhase);
    __syncwarp();
    fenceProxyAsyncSharedCta();

    if (localExpertIdx >= 0) {
      auto* outputData = reinterpret_cast<uint8_t*>(output) +
                         (static_cast<size_t>(localExpertIdx) * nOutputSlotsPerExpert + outputTokenIdx) * HiddenBytes;
      issueTmaStore(outputData, sharedTile, static_cast<uint32_t>(HiddenBytes));
      hasPendingStore = true;
    } else {
      hasPendingStore = false;
    }
    __syncwarp();
  }

  if (hasPendingStore) waitBulkGroup();
}

template <int Hidden>
__global__ __launch_bounds__(DispatchNThreads, 1) void dispatchKernel(
    void* output, int* outputSrcInfo, int64_t* outputLayout, int* outputCount,
    mscclpp::BaseMemoryChannelDeviceHandle* baseMemoryChannels, int nExperts, int rank, int nRanks,
    const int64_t* __restrict__ topkIndices, const float* __restrict__ topkWeights, const void* inputTokens,
    int nTokens, int nTopk, int maxTokensPerRank, void* recvBuffer, void* rdmaBufferBase, void* const* peerRecvBuffers,
    void* workspace) {
  extern __shared__ __align__(128) uint8_t sharedMemory[];
  auto* sharedMem = reinterpret_cast<int*>(sharedMemory);
  const int nWorkerBlocks = static_cast<int>(gridDim.x) - DispatchControlBlocks;
  WorkspaceView workspaceView(workspace, nRanks, nExperts);
  const uint32_t metadataFlag = *workspaceView.metadataEpoch_ + 1;

  dispatchSend<Hidden>(inputTokens, baseMemoryChannels, nExperts, rank, nRanks, topkIndices, topkWeights, nTokens,
                       nTopk, maxTokensPerRank, recvBuffer, peerRecvBuffers, rdmaBufferBase, workspace, metadataFlag,
                       sharedMem);

  if (static_cast<int>(blockIdx.x) == 0) {
    dispatchRecvScheduler(outputLayout, outputCount, baseMemoryChannels, nExperts, rank, nRanks, recvBuffer, workspace,
                          metadataFlag, sharedMem);
  } else if (static_cast<int>(blockIdx.x) <= nWorkerBlocks) {
    dispatchRecvWorker<Hidden>(output, outputSrcInfo, outputLayout, nExperts, rank, nRanks, nTopk, maxTokensPerRank,
                               recvBuffer, workspace, metadataFlag, sharedMem);
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *workspaceView.metadataEpoch_ = metadataFlag;
  }
}

template <int Hidden>
inline void dispatchHidden(void* output, int* outputSrcInfo, int64_t* outputLayout, int* outputCount, const void* input,
                           const int64_t* topkIdx, const float* topkWeights, const low_latency::Workload& workload,
                           void* recvBuffer, const low_latency::CommContext& comm, void* workspace, int numBlocks,
                           cudaStream_t stream) {
  static_assert(Hidden == 4096 || Hidden == 7168 || Hidden == 8192 || Hidden == 9216);
  constexpr int NRecvTmaWorkers = tmaWorkerCount<Hidden, DispatchMaxNRecvTmaWorkers>();
  static_assert(NRecvTmaWorkers > 0);
  const int nExperts = workload.numExperts_;
  const int rank = comm.rank_;
  const int nRanks = comm.numRanks_;
  const int nTokens = workload.numTokens_;
  const int nTopk = workload.numTopk_;

  const size_t dynamicSharedBytes = dispatchSharedBytes<Hidden>(nRanks, nExperts, nTopk);
  static thread_local KernelConfigCache kernelConfig;
  const int residentBlocks =
      configureKernel(dispatchKernel<Hidden>, DispatchNThreads, dynamicSharedBytes, comm, kernelConfig);
  EP_HOST_ASSERT(residentBlocks >= numBlocks);
  dispatchKernel<Hidden><<<dim3(numBlocks), dim3(DispatchNThreads), dynamicSharedBytes, stream>>>(
      output, outputSrcInfo, outputLayout, outputCount, comm.baseMemoryChannels_, nExperts, rank, nRanks, topkIdx,
      topkWeights, input, nTokens, nTopk, workload.maxTokensPerRank_, recvBuffer, comm.rdmaBufferBase_, comm.peerBases_,
      workspace);
  CUDA_CHECK(cudaGetLastError());
}

inline void dispatch(void* output, int* outputSrcInfo, int64_t* outputLayout, int* outputCount, const void* input,
                     const int64_t* topkIdx, const float* topkWeights, const low_latency::Workload& workload,
                     void* recvBuffer, const low_latency::CommContext& comm, void* workspace, int numBlocks,
                     cudaStream_t stream) {
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
      return dispatchHidden<4096>(output, outputSrcInfo, outputLayout, outputCount, input, topkIdx, topkWeights,
                                  workload, recvBuffer, comm, workspace, numBlocks, stream);
    case 7168:
      return dispatchHidden<7168>(output, outputSrcInfo, outputLayout, outputCount, input, topkIdx, topkWeights,
                                  workload, recvBuffer, comm, workspace, numBlocks, stream);
    case 8192:
      return dispatchHidden<8192>(output, outputSrcInfo, outputLayout, outputCount, input, topkIdx, topkWeights,
                                  workload, recvBuffer, comm, workspace, numBlocks, stream);
    case 9216:
      return dispatchHidden<9216>(output, outputSrcInfo, outputLayout, outputCount, input, topkIdx, topkWeights,
                                  workload, recvBuffer, comm, workspace, numBlocks, stream);
    default:
      EP_HOST_ASSERT(false && "unsupported optimized low-latency hidden size");
  }
}

}  // namespace detail

size_t workspaceSize(int numRanks, int numExperts) { return detail::workspaceBytes(numRanks, numExperts); }

void dispatch(void* output, int* outputSrcInfo, int64_t* outputLayout, int* outputCount, const void* input,
              const int64_t* topkIdx, const float* topkWeights, const Workload& workload, void* recvBuffer,
              const CommContext& comm, void* workspace, int numBlocks, cudaStream_t stream) {
  detail::dispatch(output, outputSrcInfo, outputLayout, outputCount, input, topkIdx, topkWeights, workload, recvBuffer,
                   comm, workspace, numBlocks, stream);
}

}  // namespace low_latency
}  // namespace ep
}  // namespace mscclpp
