// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include <mscclpp/memory_channel_device.hpp>

#include "../kernels/api.cuh"
#include "../kernels/exception.cuh"
#include "../kernels/utils.cuh"
#include "config.cuh"

namespace mscclpp {
namespace ep {
namespace low_latency_opt {

template <int kHidden>
MSCCLPP_DEVICE_INLINE void dispatchSend(const void* inputTokens, mscclpp::BaseMemoryChannelDeviceHandle* signalChannels,
                                        int nExperts, int rank, int nRanks, int signalChannelStride,
                                        const int64_t* topkIndices, const float* topkWeights, int nTokens, int nTopk,
                                        int maxTokensPerRank, void* recvBuffer, void* const* peerRecvBuffers,
                                        void* rdmaBufferBase, void* workspace, uint32_t metadataFlag, int* sharedMem) {
  const int threadId = static_cast<int>(threadIdx.x);
  const int warpId = threadId / WARP_SIZE;
  const int laneId = get_lane_id();
  const int nSms = static_cast<int>(gridDim.x) - 2;
  const int notifyBlockIdx = nSms + 1;
  const int nLocalExperts = nExperts / nRanks;
  const size_t metadataBytes = dispatchMetadataBytes(nRanks, nExperts);
  DispatchWorkspaceView workspaceView(workspace, nRanks, nExperts, nSms);
  if (static_cast<int>(blockIdx.x) > 0 && static_cast<int>(blockIdx.x) <= nSms) {
    const int senderBlockIdx = static_cast<int>(blockIdx.x) - 1;
    const int nWarpsPerGroup = dispatchNWarpsPerGroup(nTokens, nSms);
    const int nWarpGroups = kDispatchNWarps / nWarpsPerGroup;
    const int warpGroupId = warpId / nWarpsPerGroup;
    const int subWarpId = warpId % nWarpsPerGroup;
    const LowLatencyPayloadView<nv_bfloat16> payloadView(kHidden, nTopk);
    const size_t payloadStride = dispatchPayloadStride(kHidden, nTopk);
    constexpr size_t kHiddenBytes = static_cast<size_t>(kHidden) * sizeof(nv_bfloat16);
    constexpr int kHiddenInt4Count = kHiddenBytes / sizeof(int4);
    auto* sharedPayloadBase = reinterpret_cast<uint8_t*>(sharedMem) + dispatchSharedControlBytes(nRanks);
    auto* sendTmaBarriers = reinterpret_cast<uint64_t*>(sharedPayloadBase + kDispatchMaxNWarpGroups * payloadStride);

    if (subWarpId == 0) {
      const int tokenStride = nSms * nWarpGroups;
      const int firstTokenIdx = senderBlockIdx * nWarpGroups + warpGroupId;
      auto* stagedPayload = sharedPayloadBase + static_cast<size_t>(warpGroupId) * payloadStride;
      auto* destinationSlots = sharedMem + warpGroupId * WARP_SIZE;
      auto* tmaBarrier = sendTmaBarriers + warpGroupId;
      uint32_t sendTmaPhase = 0;
      if (firstTokenIdx < nTokens) {
        if (laneId == 0) initTmaBarrier(tmaBarrier);
        __syncwarp();
      }

      for (int tokenIdx = firstTokenIdx; tokenIdx < nTokens; tokenIdx += tokenStride) {
        const auto* inputData =
            reinterpret_cast<const int4*>(inputTokens) + static_cast<size_t>(tokenIdx) * kHiddenInt4Count;
        if (laneId == 0) {
          issueTmaG2S(inputData, stagedPayload, tmaBarrier, static_cast<uint32_t>(kHiddenBytes));
        }
        const int routedExpertIdx =
            laneId < nTopk ? static_cast<int>(__ldg(topkIndices + tokenIdx * nTopk + laneId)) : -1;
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
              topkWeights == nullptr ? 1.0f : __ldg(topkWeights + tokenIdx * nTopk + laneId);
        }
        if (laneId == 0) {
          *payloadView.srcTokenGlobalIdx(stagedPayload) = rank * maxTokensPerRank + tokenIdx;
          waitTmaG2S(tmaBarrier, sendTmaPhase);
        }
        __syncwarp();
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        bool hasPendingStore = false;
        const int destinationSlot = laneId < nTopk ? destinationSlots[laneId] : -1;
        if (destinationSlot >= 0) {
          const int dstRank = payloadView.topKIndices(stagedPayload)[laneId] / nLocalExperts;
          void* destinationBuffer =
              dstRank == rank ? recvBuffer : peerBufferPtr(recvBuffer, rdmaBufferBase, peerRecvBuffers[dstRank]);
          auto* destinationPayload = reinterpret_cast<uint8_t*>(destinationBuffer) + metadataBytes +
                                     (static_cast<size_t>(rank) * maxTokensPerRank + destinationSlot) * payloadStride;
          issueTmaS2G(destinationPayload, stagedPayload, static_cast<uint32_t>(payloadView.numBytes_));
          hasPendingStore = true;
        }
        if (hasPendingStore) {
          waitTmaS2G();
        }
        __syncwarp();
        if (destinationSlot >= 0) {
          const int dstRank = payloadView.topKIndices(stagedPayload)[laneId] / nLocalExperts;
          (void)mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(workspaceView.rankPayloadCompletions_ + dstRank, 1,
                                                                   mscclpp::memoryOrderRelease);
        }
      }
      asm volatile("bar.sync %0, %1;" ::"r"(kDispatchMaxNWarpGroups + 1), "r"(nWarpGroups * WARP_SIZE) : "memory");
    }
  } else if (static_cast<int>(blockIdx.x) == notifyBlockIdx) {
    int* sharedRankTokenCounts = sharedMem;
    int* sharedExpertTokenCounts = sharedRankTokenCounts + nRanks;
    for (int idx = threadId; idx < nRanks + nExperts; idx += blockDim.x) {
      sharedRankTokenCounts[idx] = 0;
    }
    __syncthreads();
    for (int tokenIdx = warpId; tokenIdx < nTokens; tokenIdx += kDispatchNWarps) {
      const int routedExpertIdx =
          laneId < nTopk ? static_cast<int>(__ldg(topkIndices + tokenIdx * nTopk + laneId)) : -1;
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
        signalChannels[rankSignalChannelIndex(dstRank, signalChannelStride)].signal();
      }
    }
    __syncthreads();
  }
}

MSCCLPP_DEVICE_INLINE int proportionalTaskBoundary(int nTokens, int nTasks, int nTotalTokens) {
  return nTotalTokens == 0 ? 0 : static_cast<int>(static_cast<int64_t>(nTokens) * nTasks / nTotalTokens);
}

MSCCLPP_DEVICE_INLINE void dispatchRecvScheduler(int64_t* outputLayout, int* outputCount,
                                                 mscclpp::BaseMemoryChannelDeviceHandle* signalChannels, int nExperts,
                                                 int rank, int nRanks, int signalChannelStride, void* recvBuffer,
                                                 void* workspace, uint32_t metadataFlag, int* sharedMem) {
  const int threadId = static_cast<int>(threadIdx.x);
  const int warpId = threadId / WARP_SIZE;
  const int laneId = get_lane_id();
  const int nSms = static_cast<int>(gridDim.x) - 2;
  auto* rankTokenCounts = reinterpret_cast<mscclpp::LL8Packet*>(recvBuffer);
  const int nLocalExperts = nExperts / nRanks;
  DispatchWorkspaceView workspaceView(workspace, nRanks, nExperts, nSms);

  const int nRankWarps = (nRanks + WARP_SIZE - 1) / WARP_SIZE;
  const int requestedNLayoutWarps = (nLocalExperts + WARP_SIZE - 1) / WARP_SIZE;
  const int maxNLayoutWarps = kDispatchNWarps - nRankWarps;
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
    asm volatile("bar.sync %0, %1;" ::"r"(kDispatchMaxNWarpGroups + 1), "r"(nRankWarps * WARP_SIZE) : "memory");

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
    asm volatile("bar.sync %0, %1;" ::"r"(kDispatchMaxNWarpGroups + 1), "r"(nRankWarps * WARP_SIZE) : "memory");

    rankTokenPrefix += sharedMem[warpId];
    activeRankPrefix += sharedMem[nRankWarps + warpId];
    const int nTotalTokens = sharedMem[2 * nRankWarps];
    const int nActiveRanks = sharedMem[2 * nRankWarps + 1];
    const int nTasks = nTotalTokens < nSms ? nTotalTokens : nSms;

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

    asm volatile("bar.sync %0, %1;" ::"r"(kDispatchMaxNWarpGroups + 2), "r"((nRankWarps + nLayoutWarps) * WARP_SIZE)
                 : "memory");
    if (threadId == 0) {
      mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(workspaceView.tasksAssignedEpoch_, metadataFlag,
                                                           mscclpp::memoryOrderRelease);
    }

    if (sourceRank < nRanks && nRankTokens > 0) {
      if (sourceRank == rank) {
        workspaceView.localPayloadReady_->acquire();
      } else {
        signalChannels[rankSignalChannelIndex(sourceRank, signalChannelStride)].wait(-1);
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
    asm volatile("bar.sync %0, %1;" ::"r"(kDispatchMaxNWarpGroups + 2), "r"((nRankWarps + nLayoutWarps) * WARP_SIZE)
                 : "memory");
  }
}

MSCCLPP_DEVICE_INLINE bool acquireRecvTask(RecvTask& task, DispatchWorkspaceView& workspaceView, uint32_t metadataFlag,
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

template <int kHidden>
MSCCLPP_DEVICE_INLINE void dispatchRecvWorker(void* output, int* outputSrcInfo, int64_t* outputLayout, int nExperts,
                                              int rank, int nRanks, int nTopk, int maxTokensPerRank, void* recvBuffer,
                                              void* workspace, uint32_t metadataFlag, int* sharedMem) {
#if defined(__CUDA_ARCH__)
  static_assert(__CUDA_ARCH__ >= 900, "TMA recv requires SM90 or newer");
#endif
  const int threadId = static_cast<int>(threadIdx.x);
  const int warpId = threadId / WARP_SIZE;
  const int laneId = get_lane_id();
  const int nSms = static_cast<int>(gridDim.x) - 2;
  DispatchWorkspaceView workspaceView(workspace, nRanks, nExperts, nSms);
  RecvTask task;
  if (!acquireRecvTask(task, workspaceView, metadataFlag, sharedMem)) return;
  constexpr int kNRecvTmaWorkers = tmaWorkerCount<kHidden, kDispatchMaxNRecvTmaWorkers>();
  if (warpId >= kNRecvTmaWorkers) return;

  const int nLocalExperts = nExperts / nRanks;
  const int sourceRank = task.sourceRank_;
  const int globalExpertBase = rank * nLocalExperts;
  const int globalExpertEnd = globalExpertBase + nLocalExperts;
  const LowLatencyPayloadView<nv_bfloat16> payloadView(kHidden, nTopk);
  const size_t payloadStride = dispatchPayloadStride(kHidden, nTopk);
  constexpr size_t kHiddenBytes = static_cast<size_t>(kHidden) * sizeof(nv_bfloat16);
  constexpr size_t kTileBytes = kHiddenBytes;
  const int nOutputSlotsPerExpert = nRanks * maxTokensPerRank;
  auto* sourcePayloadBase = reinterpret_cast<uint8_t*>(recvBuffer) + dispatchMetadataBytes(nRanks, nExperts) +
                            static_cast<size_t>(sourceRank) * maxTokensPerRank * payloadStride;
  auto* tmaTiles = reinterpret_cast<uint8_t*>(sharedMem) + dispatchSharedControlBytes(nRanks);
  auto* sharedTile = tmaTiles + static_cast<size_t>(warpId) * kTileBytes;
  auto* tmaBarriers = reinterpret_cast<uint64_t*>(tmaTiles + static_cast<size_t>(kNRecvTmaWorkers) * kTileBytes);
  auto* tmaBarrier = tmaBarriers + warpId;
  bool hasPendingStore = false;
  uint32_t recvTmaPhase = 0;
  if (laneId == 0) initTmaBarrier(tmaBarrier);

  for (int sourceTokenSlot = task.tokenBegin_ + warpId; sourceTokenSlot < task.tokenEnd_;
       sourceTokenSlot += kNRecvTmaWorkers) {
    if (hasPendingStore) {
      waitTmaS2GRead();
    }
    __syncwarp();

    auto* sourcePayload = sourcePayloadBase + static_cast<size_t>(sourceTokenSlot) * payloadStride;
    if (laneId == 0) {
      issueTmaG2S(payloadView.template data<int4>(sourcePayload), sharedTile, tmaBarrier,
                  static_cast<uint32_t>(kHiddenBytes));
    }
    __syncwarp();

    const int routedExpertIdx = laneId < nTopk ? ld_nc_global(payloadView.topKIndices(sourcePayload) + laneId) : -1;
    const int localExpertIdx = routedExpertIdx >= globalExpertBase && routedExpertIdx < globalExpertEnd
                                   ? routedExpertIdx - globalExpertBase
                                   : -1;
    const int sourceTokenIdx = __shfl_sync(
        0xffffffff,
        laneId == 0 ? ld_nc_global(payloadView.srcTokenGlobalIdx(sourcePayload)) - sourceRank * maxTokensPerRank : 0,
        0);
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

    if (laneId == 0) waitTmaG2S(tmaBarrier, recvTmaPhase);
    __syncwarp();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    if (localExpertIdx >= 0) {
      auto* outputData = reinterpret_cast<uint8_t*>(output) +
                         (static_cast<size_t>(localExpertIdx) * nOutputSlotsPerExpert + outputTokenIdx) * kHiddenBytes;
      issueTmaS2G(outputData, sharedTile, static_cast<uint32_t>(kHiddenBytes));
      hasPendingStore = true;
    } else {
      hasPendingStore = false;
    }
    __syncwarp();
  }

  if (hasPendingStore) {
    waitTmaS2G();
  }
}

template <int kHidden>
__global__ __launch_bounds__(kDispatchNThreads, 1) void dispatchKernel(
    void* output, int* outputSrcInfo, int64_t* outputLayout, int* outputCount,
    mscclpp::BaseMemoryChannelDeviceHandle* signalChannels, int nExperts, int rank, int nRanks, int signalChannelStride,
    const int64_t* topkIndices, const float* topkWeights, const void* inputTokens, int nTokens, int nTopk,
    int maxTokensPerRank, void* recvBuffer, void* rdmaBufferBase, void* const* peerRecvBuffers, void* workspace) {
  extern __shared__ __align__(128) uint8_t sharedMemory[];
  auto* sharedMem = reinterpret_cast<int*>(sharedMemory);
  const int nSms = static_cast<int>(gridDim.x) - 2;
  DispatchWorkspaceView workspaceView(workspace, nRanks, nExperts, nSms);
  const uint32_t metadataFlag = *workspaceView.metadataEpoch_ + 1;

  dispatchSend<kHidden>(inputTokens, signalChannels, nExperts, rank, nRanks, signalChannelStride, topkIndices,
                        topkWeights, nTokens, nTopk, maxTokensPerRank, recvBuffer, peerRecvBuffers, rdmaBufferBase,
                        workspace, metadataFlag, sharedMem);

  if (static_cast<int>(blockIdx.x) == 0) {
    dispatchRecvScheduler(outputLayout, outputCount, signalChannels, nExperts, rank, nRanks, signalChannelStride,
                          recvBuffer, workspace, metadataFlag, sharedMem);
  } else if (static_cast<int>(blockIdx.x) <= nSms) {
    dispatchRecvWorker<kHidden>(output, outputSrcInfo, outputLayout, nExperts, rank, nRanks, nTopk, maxTokensPerRank,
                                recvBuffer, workspace, metadataFlag, sharedMem);
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *workspaceView.metadataEpoch_ = metadataFlag;
  }
}

template <int kHidden>
inline void dispatchHidden(void* output, int* outputSrcInfo, int64_t* outputLayout, int* outputCount, const void* input,
                           const int64_t* topkIdx, const float* topkWeights, const low_latency::DispatchConfig& config,
                           const low_latency::BufferSet& currentBuffer, const low_latency::TransportContext& transport,
                           void* workspace, int maxSms, cudaStream_t stream) {
  static_assert(kHidden == 4096 || kHidden == 7168 || kHidden == 8192 || kHidden == 9216);
  constexpr int kNRecvTmaWorkers = tmaWorkerCount<kHidden, kDispatchMaxNRecvTmaWorkers>();
  static_assert(kNRecvTmaWorkers > 0);
  const int nExperts = config.numExperts_;
  const int rank = transport.rank_;
  const int nRanks = transport.numRanks_;
  const int signalChannelStride = transport.memoryChannelStride_;
  const int nTokens = config.numTokens_;
  const int nTopk = config.numTopk_;

  static thread_local int sharedMemoryLimitDevice = -1;
  static thread_local size_t sharedMemoryLimit = 0;
  if (sharedMemoryLimitDevice != transport.deviceId_) {
    int sharedMemoryLimitInt;
    cudaFuncAttributes attributes;
    CUDA_CHECK(
        cudaDeviceGetAttribute(&sharedMemoryLimitInt, cudaDevAttrMaxSharedMemoryPerBlockOptin, transport.deviceId_));
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, dispatchKernel<kHidden>));
    EP_HOST_ASSERT(sharedMemoryLimitInt > static_cast<int>(attributes.sharedSizeBytes));
    sharedMemoryLimitDevice = transport.deviceId_;
    sharedMemoryLimit = static_cast<size_t>(sharedMemoryLimitInt) - attributes.sharedSizeBytes;
  }

  const size_t dynamicSharedBytes = dispatchSharedBytes<kHidden>(nRanks, nExperts, nTopk);
  EP_HOST_ASSERT(dynamicSharedBytes <= sharedMemoryLimit);

  cudaLaunchConfig_t cfg = {dim3(maxSms + 2), dim3(kDispatchNThreads), dynamicSharedBytes, stream, nullptr, 0};
  static thread_local int configuredDevice = -1;
  static thread_local size_t configuredDynamicSharedBytes = 0;
  if (configuredDevice != transport.deviceId_ || configuredDynamicSharedBytes < dynamicSharedBytes) {
    CUDA_CHECK(cudaFuncSetAttribute(dispatchKernel<kHidden>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(dynamicSharedBytes)));
    configuredDevice = transport.deviceId_;
    configuredDynamicSharedBytes = dynamicSharedBytes;
  }
  CUDA_CHECK(cudaLaunchKernelEx(&cfg, dispatchKernel<kHidden>, output, outputSrcInfo, outputLayout, outputCount,
                                transport.memoryChannels_, nExperts, rank, nRanks, signalChannelStride, topkIdx,
                                topkWeights, input, nTokens, nTopk, config.numMaxTokensPerRank_,
                                currentBuffer.recvDataBuffer_, transport.rdmaBufferBase_, transport.peerBases_,
                                workspace));
}

inline void dispatch(void* output, int* outputSrcInfo, int64_t* outputLayout, int* outputCount, const void* input,
                     const int64_t* topkIdx, const float* topkWeights, const low_latency::DispatchConfig& config,
                     const low_latency::BufferSet& currentBuffer, const low_latency::TransportContext& transport,
                     void* workspace, int maxSms, cudaStream_t stream) {
  const int nExperts = config.numExperts_;
  const int rank = transport.rank_;
  const int nRanks = transport.numRanks_;

  EP_HOST_ASSERT(nRanks > 0);
  EP_HOST_ASSERT(nExperts > 0);
  EP_HOST_ASSERT(nExperts % nRanks == 0);
  EP_HOST_ASSERT(rank >= 0 && rank < nRanks);
  EP_HOST_ASSERT(transport.memoryChannels_ != nullptr);
  EP_HOST_ASSERT(transport.memoryChannelStride_ >= 1);
  EP_HOST_ASSERT(config.numTokens_ >= 0);
  EP_HOST_ASSERT(config.numTopk_ > 0 && config.numTopk_ <= WARP_SIZE);
  EP_HOST_ASSERT(nRanks <= 2 * WARP_SIZE);
  EP_HOST_ASSERT(maxSms >= nRanks && maxSms <= kDispatchMaxNSms);
  EP_HOST_ASSERT(output != nullptr);
  EP_HOST_ASSERT(outputSrcInfo != nullptr);
  EP_HOST_ASSERT(outputLayout != nullptr);
  EP_HOST_ASSERT(outputCount != nullptr);
  EP_HOST_ASSERT(input != nullptr);
  EP_HOST_ASSERT(topkIdx != nullptr);
  EP_HOST_ASSERT(currentBuffer.recvDataBuffer_ != nullptr);
  EP_HOST_ASSERT(transport.peerBases_ != nullptr);
  EP_HOST_ASSERT(workspace != nullptr);
  EP_HOST_ASSERT(config.outputDType_ == low_latency::DType::BF16);

  switch (config.hidden_) {
    case 4096:
      return dispatchHidden<4096>(output, outputSrcInfo, outputLayout, outputCount, input, topkIdx, topkWeights, config,
                                  currentBuffer, transport, workspace, maxSms, stream);
    case 7168:
      return dispatchHidden<7168>(output, outputSrcInfo, outputLayout, outputCount, input, topkIdx, topkWeights, config,
                                  currentBuffer, transport, workspace, maxSms, stream);
    case 8192:
      return dispatchHidden<8192>(output, outputSrcInfo, outputLayout, outputCount, input, topkIdx, topkWeights, config,
                                  currentBuffer, transport, workspace, maxSms, stream);
    case 9216:
      return dispatchHidden<9216>(output, outputSrcInfo, outputLayout, outputCount, input, topkIdx, topkWeights, config,
                                  currentBuffer, transport, workspace, maxSms, stream);
    default:
      EP_HOST_ASSERT(false && "unsupported optimized low-latency hidden size");
  }
}

}  // namespace low_latency_opt

namespace low_latency {

void dispatchOptimized(void* output, int* outputSrcInfo, int64_t* outputLayout, int* outputCount, const void* input,
                       const int64_t* topkIdx, const float* topkWeights, const DispatchConfig& config,
                       const BufferSet& currentBuffer, const TransportContext& transport, void* workspace, int maxSms,
                       cudaStream_t stream) {
  low_latency_opt::dispatch(output, outputSrcInfo, outputLayout, outputCount, input, topkIdx, topkWeights, config,
                            currentBuffer, transport, workspace, maxSms, stream);
}

}  // namespace low_latency
}  // namespace ep
}  // namespace mscclpp
