// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_COMBINE_COMMON_CUH_
#define MSCCLPP_EP_COMBINE_COMMON_CUH_

#include <mscclpp/bulk_device.hpp>
#include <mscclpp/gpu_data_types.hpp>

#include "common/device_helpers.cuh"
#include "common/latency.cuh"
#include "exception.hpp"
#include "kernels.hpp"

#if defined(MSCCLPP_USE_GPUNETIO)
#include <mscclpp/port_channel_gpunetio_device.hpp>
#endif  // defined(MSCCLPP_USE_GPUNETIO)

namespace mscclpp {
namespace ep {

constexpr int CombineNWarps = 32;
constexpr int CombineNThreads = CombineNWarps * WARP_SIZE;
constexpr int CombineNStages = 8;
constexpr int DirectSendMaxNWorkers = WARP_SIZE;
constexpr int CombineMaxNTopk = 9;
constexpr int RankMajorTmaMaxNTopk = 8;

MSCCLPP_HOST_DEVICE_INLINE size_t directSendControlBytes(int nLocalExperts) {
  return configAlign<size_t>(static_cast<size_t>(nLocalExperts + 1) * sizeof(int), 128);
}

template <int Hidden>
MSCCLPP_HOST_DEVICE_INLINE constexpr size_t directSendWorkerBytes() {
  return static_cast<size_t>(Hidden) * sizeof(Bf16) + sizeof(mscclpp::BulkBarrier);
}

template <int Hidden>
MSCCLPP_HOST_DEVICE_INLINE int directSendWorkerCount(int nLocalExperts) {
  const size_t controlBytes = directSendControlBytes(nLocalExperts);
  if (controlBytes >= OptimizedDynamicSharedMemoryBytes) return 0;
  const int availableWorkers =
      static_cast<int>((OptimizedDynamicSharedMemoryBytes - controlBytes) / directSendWorkerBytes<Hidden>());
  return availableWorkers < DirectSendMaxNWorkers ? availableWorkers : DirectSendMaxNWorkers;
}

template <int Hidden, CombineMode Mode, DispatchLayout Layout>
MSCCLPP_HOST_DEVICE_INLINE size_t combineSharedBytes(int nLocalExperts, int nTopk) {
  if constexpr (Layout == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED) {
    if (nTopk <= RankMajorTmaMaxNTopk) {
      constexpr size_t RowsBytes = static_cast<size_t>(RankMajorTmaMaxNTopk) * Hidden * sizeof(Bf16);
      constexpr size_t BarrierBytes = static_cast<size_t>(RankMajorTmaMaxNTopk) * sizeof(mscclpp::BulkBarrier);
      constexpr size_t ValidBytes = static_cast<size_t>(RankMajorTmaMaxNTopk) * sizeof(int);
      constexpr size_t WeightBytes = static_cast<size_t>(RankMajorTmaMaxNTopk) * sizeof(float);
      constexpr size_t SharedBytes = configAlign<size_t>(RowsBytes + BarrierBytes + ValidBytes + WeightBytes, 128);
      static_assert(SharedBytes <= OptimizedDynamicSharedMemoryBytes);
      return SharedBytes;
    }
    return 0;
  }
  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    if (nTopk <= RankMajorTmaMaxNTopk) {
      constexpr size_t RowsBytes = static_cast<size_t>(RankMajorTmaMaxNTopk) * Hidden * sizeof(Bf16);
      constexpr size_t BarrierBytes = static_cast<size_t>(RankMajorTmaMaxNTopk) * sizeof(mscclpp::BulkBarrier);
      constexpr size_t ValidBytes = static_cast<size_t>(RankMajorTmaMaxNTopk) * sizeof(int);
      constexpr size_t SharedBytes = configAlign<size_t>(RowsBytes + BarrierBytes + ValidBytes, 128);
      static_assert(SharedBytes <= OptimizedDynamicSharedMemoryBytes);
      return SharedBytes;
    }
    return 0;
  }
  constexpr size_t TileBytes = static_cast<size_t>(Hidden) * sizeof(Bf16);
  if constexpr (Mode == CombineMode::DIRECT_SEND) {
    return directSendControlBytes(nLocalExperts) +
           static_cast<size_t>(directSendWorkerCount<Hidden>(nLocalExperts)) * directSendWorkerBytes<Hidden>();
  }
  return CombineNStages * TileBytes;
}

#if MSCCLPP_BULK_AVAILABLE

MSCCLPP_DEVICE_INLINE RecvTask loadRecvTask(const RecvTask* tasks, int taskIdx) {
  const int laneId = get_lane_id();
  RecvTask task{};
  if (laneId == 0) task = tasks[taskIdx];
  task.sourceRank_ = warpBroadcast(task.sourceRank_, 0);
  task.tokenBegin_ = warpBroadcast(task.tokenBegin_, 0);
  task.tokenEnd_ = warpBroadcast(task.tokenEnd_, 0);
  return task;
}

template <int HiddenInt4>
MSCCLPP_DEVICE_INLINE int4 reduceWeightedBf16x8(const void* expertOutput, int rowOffset, float weight, int nTopk,
                                                int hiddenIdx) {
  constexpr int Bf16PairsPerInt4 = sizeof(int4) / sizeof(mscclpp::bf16x2);
  float2 reduced[Bf16PairsPerInt4] = {};
  for (int topkLane = 0; topkLane < nTopk; ++topkLane) {
    const int sourceRowOffset = warpBroadcast(rowOffset, topkLane);
    if (sourceRowOffset < 0) continue;
    const float sourceWeight = warpBroadcast(weight, topkLane);
    const int4 packed =
        reinterpret_cast<const int4*>(expertOutput)[static_cast<size_t>(sourceRowOffset) * HiddenInt4 + hiddenIdx];
    const auto* values = reinterpret_cast<const mscclpp::bf16x2*>(&packed);
#pragma unroll
    for (int pairIdx = 0; pairIdx < Bf16PairsPerInt4; ++pairIdx) {
      const mscclpp::f32x2 value = mscclpp::to<mscclpp::f32x2>(values[pairIdx]);
      reduced[pairIdx].x = fmaf(value.data[0], sourceWeight, reduced[pairIdx].x);
      reduced[pairIdx].y = fmaf(value.data[1], sourceWeight, reduced[pairIdx].y);
    }
  }

  int4 packedOutput;
  auto* outputValues = reinterpret_cast<mscclpp::bf16x2*>(&packedOutput);
#pragma unroll
  for (int pairIdx = 0; pairIdx < Bf16PairsPerInt4; ++pairIdx) {
    outputValues[pairIdx] = mscclpp::to<mscclpp::bf16x2>(mscclpp::f32x2(reduced[pairIdx]));
  }
  return packedOutput;
}

template <int HiddenInt4>
MSCCLPP_DEVICE_INLINE int4 reduceRankPartialsBf16x8(const void* combineRecvBuffer, int partialRankCandidate, int nTopk,
                                                    int maxTokensPerRank, int tokenIdx, int hiddenIdx) {
  constexpr int Bf16PairsPerInt4 = sizeof(int4) / sizeof(mscclpp::bf16x2);
  float2 reduced[Bf16PairsPerInt4] = {};
  for (int topkLane = 0; topkLane < nTopk; ++topkLane) {
    const int partialRank = warpBroadcast(partialRankCandidate, topkLane);
    if (partialRank < 0) continue;
    const int4 packed = reinterpret_cast<const int4*>(
        combineRecvBuffer)[(static_cast<size_t>(partialRank) * maxTokensPerRank + tokenIdx) * HiddenInt4 + hiddenIdx];
    const auto* values = reinterpret_cast<const mscclpp::bf16x2*>(&packed);
#pragma unroll
    for (int pairIdx = 0; pairIdx < Bf16PairsPerInt4; ++pairIdx) {
      const mscclpp::f32x2 value = mscclpp::to<mscclpp::f32x2>(values[pairIdx]);
      reduced[pairIdx].x += value.data[0];
      reduced[pairIdx].y += value.data[1];
    }
  }

  int4 packedOutput;
  auto* outputValues = reinterpret_cast<mscclpp::bf16x2*>(&packedOutput);
#pragma unroll
  for (int pairIdx = 0; pairIdx < Bf16PairsPerInt4; ++pairIdx) {
    outputValues[pairIdx] = mscclpp::to<mscclpp::bf16x2>(mscclpp::f32x2(reduced[pairIdx]));
  }
  return packedOutput;
}

template <int Hidden, DispatchDataType DispatchType, int ScaleBlockSize>
MSCCLPP_DEVICE_INLINE void sendRankReducedPartials(const void* expertOutput, int nExperts, int nRanks, int nTopk,
                                                   int maxTokensPerRank, void* combineRecvBuffer,
                                                   const void* dispatchRecvBuffer, const TransportView& transport,
                                                   WorkspaceView& workspaceView, uint8_t* sharedMemory) {
#if defined(__CUDA_ARCH__)
  static_assert(__CUDA_ARCH__ >= 900, "TMA combine send requires SM90 or newer");
#endif
  const int threadId = static_cast<int>(threadIdx.x);
  const int laneId = get_lane_id();
  const int nLocalExperts = nExperts / nRanks;
  [[maybe_unused]] const int nExpertOutputRows = nLocalExperts * nRanks * maxTokensPerRank;
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(Bf16);
  constexpr int HiddenInt4 = HiddenBytes / sizeof(int4);
  constexpr int ChunksPerThread = (HiddenInt4 + CombineNThreads - 1) / CombineNThreads;
  static_assert(HiddenInt4 % WARP_SIZE == 0);
  const size_t dispatchMetadataSize = dispatchMetadataBytes(nRanks, nExperts);
  const size_t payloadStride = dispatchPayloadStride<DispatchType>(Hidden, nTopk, ScaleBlockSize);
  const DispatchPayloadView<DispatchType> payloadView(Hidden, nTopk, ScaleBlockSize);
  auto* outputTiles = sharedMemory;

  int tokenIteration = 0;
  for (int taskIdx = static_cast<int>(blockIdx.x); taskIdx < *workspaceView.dispatchNumRecvTasks_;
       taskIdx += static_cast<int>(gridDim.x)) {
    const RecvTask recvTask = loadRecvTask(workspaceView.dispatchRecvTasks_, taskIdx);
    const int sourceRank = recvTask.sourceRank_;

    for (int sourceTokenSlot = recvTask.tokenBegin_; sourceTokenSlot < recvTask.tokenEnd_;
         ++sourceTokenSlot, ++tokenIteration) {
      const int stage = tokenIteration % CombineNStages;
      auto* outputTile = reinterpret_cast<int4*>(outputTiles + static_cast<size_t>(stage) * HiddenBytes);
      const auto* sourcePayload =
          reinterpret_cast<const uint8_t*>(dispatchRecvBuffer) + dispatchMetadataSize +
          (static_cast<size_t>(sourceRank) * maxTokensPerRank + sourceTokenSlot) * payloadStride;
      const int rowOffset = laneId < nTopk ? payloadView.topKIndices(sourcePayload)[laneId] : -1;
      const float weight = laneId < nTopk ? payloadView.topKValues(sourcePayload)[laneId] : 0.0f;
      if (rowOffset >= 0) EP_DEVICE_ASSERT(rowOffset < nExpertOutputRows);

      int4 reduced[ChunksPerThread] = {};
#pragma unroll
      for (int chunkIdx = 0; chunkIdx < ChunksPerThread; ++chunkIdx) {
        const int hiddenIdx = threadId + chunkIdx * CombineNThreads;
        if (hiddenIdx < HiddenInt4) {
          reduced[chunkIdx] = reduceWeightedBf16x8<HiddenInt4>(expertOutput, rowOffset, weight, nTopk, hiddenIdx);
        }
      }

      if (tokenIteration >= CombineNStages && threadId == 0) {
        mscclpp::bulkStoreWaitSource<CombineNStages - 1>();
      }
      if (tokenIteration >= CombineNStages) __syncthreads();
#pragma unroll
      for (int chunkIdx = 0; chunkIdx < ChunksPerThread; ++chunkIdx) {
        const int hiddenIdx = threadId + chunkIdx * CombineNThreads;
        if (hiddenIdx < HiddenInt4) outputTile[hiddenIdx] = reduced[chunkIdx];
      }
      __syncthreads();

      if (threadId == 0) {
        mscclpp::bulkFence();
        const int sourceTokenIdx = *payloadView.srcTokenGlobalIdx(sourcePayload) - sourceRank * maxTokensPerRank;
        EP_DEVICE_ASSERT(sourceTokenIdx >= 0 && sourceTokenIdx < maxTokensPerRank);
        void* destinationBuffer = transport.mappedBuffer(combineRecvBuffer, sourceRank);
        auto* destinationRow = reinterpret_cast<uint8_t*>(destinationBuffer) +
                               (static_cast<size_t>(transport.rank_) * maxTokensPerRank + sourceTokenIdx) * HiddenBytes;
        mscclpp::bulkStore(destinationRow, outputTile, static_cast<uint32_t>(HiddenBytes));
        mscclpp::bulkStoreCommit();
      }
    }
  }
  if (tokenIteration > 0 && threadId == 0) mscclpp::bulkStoreWait();
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void sendExpertRowsDirect(const void* expertOutput, const int* srcInfo,
                                                const int64_t* layoutRange, int nExperts, int nRanks,
                                                int maxTokensPerRank, void* combineRecvBuffer,
                                                const TransportView& transport, uint8_t* sharedMemory) {
  if (threadIdx.x >= WARP_SIZE) return;
  const int laneId = get_lane_id();
  const int nLocalExperts = nExperts / nRanks;
  const int nOutputSlotsPerExpert = nRanks * maxTokensPerRank;
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(Bf16);
  const int nWorkers = directSendWorkerCount<Hidden>(nLocalExperts);
  auto* expertTokenPrefix = reinterpret_cast<int*>(sharedMemory);
  auto* outputTiles = sharedMemory + directSendControlBytes(nLocalExperts);

  if (laneId == 0) {
    expertTokenPrefix[0] = 0;
    for (int localExpertIdx = 0; localExpertIdx < nLocalExperts; ++localExpertIdx) {
      int nLastRankTokens;
      int lastRankOffset;
      unpack2(layoutRange[localExpertIdx * nRanks + nRanks - 1], nLastRankTokens, lastRankOffset);
      expertTokenPrefix[localExpertIdx + 1] = expertTokenPrefix[localExpertIdx] + lastRankOffset + nLastRankTokens;
    }
  }
  __syncwarp();

  const int nTotalRows = expertTokenPrefix[nLocalExperts];
  const int blockRowBegin = static_cast<int>(static_cast<int64_t>(nTotalRows) * blockIdx.x / gridDim.x);
  const int blockRowEnd = static_cast<int>(static_cast<int64_t>(nTotalRows) * (blockIdx.x + 1) / gridDim.x);
  auto* bulkBarriers =
      reinterpret_cast<mscclpp::BulkBarrier*>(outputTiles + static_cast<size_t>(nWorkers) * HiddenBytes);
  if (laneId < nWorkers) {
    auto* outputTile = outputTiles + static_cast<size_t>(laneId) * HiddenBytes;
    auto* bulkBarrier = bulkBarriers + laneId;
    uint32_t bulkPhase = 0;
    const bool hasRows = blockRowBegin + laneId < blockRowEnd;
    if (hasRows) bulkBarrier->init();

    bool hasPendingStore = false;
    for (int flatRowIdx = blockRowBegin + laneId; flatRowIdx < blockRowEnd; flatRowIdx += nWorkers) {
      if (hasPendingStore) mscclpp::bulkStoreWaitSource();
      int localExpertIdx = 0;
      while (flatRowIdx >= expertTokenPrefix[localExpertIdx + 1]) ++localExpertIdx;
      const int expertTokenIdx = flatRowIdx - expertTokenPrefix[localExpertIdx];
      int sourceRank = 0;
      for (; sourceRank < nRanks; ++sourceRank) {
        int nRankTokens;
        int rankOffset;
        unpack2(layoutRange[localExpertIdx * nRanks + sourceRank], nRankTokens, rankOffset);
        if (expertTokenIdx >= rankOffset && expertTokenIdx < rankOffset + nRankTokens) break;
      }
      EP_DEVICE_ASSERT(sourceRank < nRanks);
      const int inputRowOffset = localExpertIdx * nOutputSlotsPerExpert + expertTokenIdx;
      const int sourceTokenIdx = srcInfo[inputRowOffset];
      EP_DEVICE_ASSERT(sourceTokenIdx >= 0 && sourceTokenIdx < maxTokensPerRank);
      const auto* inputRow =
          reinterpret_cast<const uint8_t*>(expertOutput) + static_cast<size_t>(inputRowOffset) * HiddenBytes;
      bulkBarrier->arriveAndExpect(static_cast<uint32_t>(HiddenBytes));
      mscclpp::bulkLoad(outputTile, inputRow, static_cast<uint32_t>(HiddenBytes), *bulkBarrier);
      bulkBarrier->wait(bulkPhase);
      mscclpp::bulkFence();
      const int globalExpertIdx = transport.rank_ * nLocalExperts + localExpertIdx;
      void* destinationBuffer = transport.mappedBuffer(combineRecvBuffer, sourceRank);
      auto* destinationRow = reinterpret_cast<uint8_t*>(destinationBuffer) +
                             (static_cast<size_t>(globalExpertIdx) * maxTokensPerRank + sourceTokenIdx) * HiddenBytes;
      mscclpp::bulkStore(destinationRow, outputTile, static_cast<uint32_t>(HiddenBytes));
      mscclpp::bulkStoreCommit();
      hasPendingStore = true;
    }

    if (hasPendingStore) mscclpp::bulkStoreWait();
  }
}

MSCCLPP_DEVICE_INLINE void exchangeCombineReady(const TransportView& transport, int nRanks) {
  const int threadId = static_cast<int>(threadIdx.x);
  if (blockIdx.x == 0 && threadId < nRanks) {
    const int peerRank = threadId;
    if (transport.isSelf(peerRank)) return;
    transport.baseMemoryChannels_[peerRank].signal();
    transport.baseMemoryChannels_[peerRank].wait(-1);
  }
}

MSCCLPP_DEVICE_INLINE void synchronizeRankMajorCombine(const TransportView& transport, int nRanks, uint32_t epoch,
                                                       WorkspaceView& workspaceView) {
  const int threadId = static_cast<int>(threadIdx.x);
  // Port the signal-all / wait-all order from b31afb9. Cross-domain peers use
  // PUSH completion flags rather than uninitialized IPC channel handles.
  if (blockIdx.x == 0 && threadId == 0) {
    for (int peerRank = 0; peerRank < nRanks; ++peerRank) {
      if (transport.isSelf(peerRank) || !transport.isNvlinkPeer(peerRank)) continue;
      transport.baseMemoryChannels_[peerRank].relaxedSignal();
    }
    for (int peerRank = 0; peerRank < nRanks; ++peerRank) {
      if (transport.isSelf(peerRank) || !transport.isNvlinkPeer(peerRank)) continue;
      transport.baseMemoryChannels_[peerRank].relaxedWait(-1);
    }
  }
  if (blockIdx.x == 0) {
    __syncthreads();
    if (threadIdx.x == 0) {
      mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(workspaceView.combineReadyEpoch_, epoch,
                                                           mscclpp::memoryOrderRelaxed);
    }
  } else {
    if (threadIdx.x == 0) {
      while (mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(workspaceView.combineReadyEpoch_,
                                                                 mscclpp::memoryOrderRelaxed) != epoch) {
      }
    }
    __syncthreads();
  }
}

MSCCLPP_DEVICE_INLINE void publishRankMajorCombineReady(const TransportView& transport, int nRanks, uint32_t epoch,
                                                        WorkspaceView& workspaceView) {
  if (blockIdx.x != 0) return;
  const int threadId = static_cast<int>(threadIdx.x);
  if (threadId < nRanks) {
    if (!transport.isSelf(threadId)) {
      transport.baseMemoryChannels_[threadId].relaxedSignal();
      transport.baseMemoryChannels_[threadId].relaxedWait(-1);
    }
    mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(workspaceView.combineRankReadyEpochs_ + threadId, epoch,
                                                         mscclpp::memoryOrderRelaxed);
  }
}

template <int HiddenInt4, CombineMode Mode>
MSCCLPP_DEVICE_INLINE int4 reduceRemoteRankPartialsBf16x8(const void* expertOutput, const TransportView& transport,
                                                          int destinationRankCandidate, int destinationSlotCandidate,
                                                          int nTopk, int maxTokensPerRank, int tokenIdx,
                                                          int hiddenIdx) {
  constexpr bool IsDirectSend = Mode == CombineMode::DIRECT_SEND;
  constexpr int Bf16PairsPerInt4 = sizeof(int4) / sizeof(mscclpp::bf16x2);
  float2 reduced[Bf16PairsPerInt4] = {};
  for (int topkLane = 0; topkLane < nTopk; ++topkLane) {
    const int destinationRank = warpBroadcast(destinationRankCandidate, topkLane);
    if (destinationRank < 0) continue;
    const int destinationSlot = warpBroadcast(destinationSlotCandidate, topkLane);
    EP_DEVICE_ASSERT(destinationSlot >= 0 && destinationSlot < maxTokensPerRank);
    const auto* remoteExpertOutput =
        reinterpret_cast<const int4*>(transport.mappedBuffer(const_cast<void*>(expertOutput), destinationRank));
    const size_t sourceRow = static_cast<size_t>(transport.rank_) * maxTokensPerRank + destinationSlot;
    const size_t inputRow = IsDirectSend ? sourceRow * nTopk + topkLane : sourceRow;
    const int4 packed = remoteExpertOutput[inputRow * HiddenInt4 + hiddenIdx];
    const auto* values = reinterpret_cast<const mscclpp::bf16x2*>(&packed);
#pragma unroll
    for (int pairIdx = 0; pairIdx < Bf16PairsPerInt4; ++pairIdx) {
      const mscclpp::f32x2 value = mscclpp::to<mscclpp::f32x2>(values[pairIdx]);
      reduced[pairIdx].x += value.data[0];
      reduced[pairIdx].y += value.data[1];
    }
  }

  int4 packedOutput;
  auto* outputValues = reinterpret_cast<mscclpp::bf16x2*>(&packedOutput);
#pragma unroll
  for (int pairIdx = 0; pairIdx < Bf16PairsPerInt4; ++pairIdx) {
    outputValues[pairIdx] = mscclpp::to<mscclpp::bf16x2>(mscclpp::f32x2(reduced[pairIdx]));
  }
  return packedOutput;
}

template <int Hidden, CombineMode Mode>
MSCCLPP_DEVICE_INLINE void recvRankMajorRemotePartialsTma(void* output, const void* expertOutput,
                                                          const int64_t* __restrict__ topkIndices, int nTokens,
                                                          int nTopk, int nExperts, int nRanks, int maxTokensPerRank,
                                                          uint32_t epoch, const TransportView& transport,
                                                          WorkspaceView& workspaceView, uint8_t* sharedMemory) {
#if defined(__CUDA_ARCH__)
  static_assert(__CUDA_ARCH__ >= 900, "TMA rank-major combine requires SM90 or newer");
#endif
  static_assert(Hidden % (sizeof(int4) / sizeof(Bf16)) == 0);
  constexpr bool IsDirectSend = Mode == CombineMode::DIRECT_SEND;
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(Bf16);
  constexpr int HiddenInt4 = HiddenBytes / sizeof(int4);
  constexpr int Bf16PairsPerInt4 = sizeof(int4) / sizeof(mscclpp::bf16x2);
  constexpr size_t RowsBytes = static_cast<size_t>(RankMajorTmaMaxNTopk) * HiddenBytes;
  const int threadId = static_cast<int>(threadIdx.x);
  const int warpId = threadId / WARP_SIZE;
  const int laneId = get_lane_id();
  const int nLocalExperts = nExperts / nRanks;
  auto* sharedRows = reinterpret_cast<int4*>(sharedMemory);
  auto* bulkBarriers = reinterpret_cast<mscclpp::BulkBarrier*>(sharedMemory + RowsBytes);
  auto* validRows = reinterpret_cast<int*>(bulkBarriers + RankMajorTmaMaxNTopk);

  const int nWorkerBlocks = static_cast<int>(gridDim.x) - 1;
  // Reuse each mbarrier by advancing its parity instead of reinitializing it.
  uint32_t bulkPhase = 0;
  if (warpId == 0) {
    if (laneId < RankMajorTmaMaxNTopk) {
      bulkBarriers[laneId].init();
    }
    __syncwarp();
  }

  for (int tokenIdx = static_cast<int>(blockIdx.x) - 1; tokenIdx < nTokens; tokenIdx += nWorkerBlocks) {
    if (warpId == 0) {
      const int globalExpertIdx = laneId < nTopk ? static_cast<int>(topkIndices[tokenIdx * nTopk + laneId]) : -1;
      const int destinationRank = globalExpertIdx >= 0 ? globalExpertIdx / nLocalExperts : -1;
      const bool firstLaneForRank = isFirstLaneForRank(destinationRank, laneId);
      const bool validRow = laneId < RankMajorTmaMaxNTopk && destinationRank >= 0 && (IsDirectSend || firstLaneForRank);
      int destinationSlot = -1;
      if (validRow) {
        destinationSlot = workspaceView.rankMajorSendIndices_[tokenIdx * nTopk + laneId];
        EP_DEVICE_ASSERT(destinationSlot >= 0 && destinationSlot < maxTokensPerRank);
      }
      if (laneId < RankMajorTmaMaxNTopk) {
        validRows[laneId] = validRow;
      }
      __syncwarp();

      bool pending = validRow;
      while (__any_sync(0xffffffff, pending)) {
        const bool ready = pending && mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(
                                          workspaceView.combineRankReadyEpochs_ + destinationRank,
                                          mscclpp::memoryOrderRelaxed) == epoch;
        if (pending && ready) {
          const uint8_t* source;
          if (transport.gpuNetIo_ == nullptr || transport.isNvlinkPeer(destinationRank)) {
            const auto* remoteExpertOutput = reinterpret_cast<const uint8_t*>(
                transport.mappedBuffer(const_cast<void*>(expertOutput), destinationRank));
            const size_t sourceRow = static_cast<size_t>(transport.rank_) * maxTokensPerRank + destinationSlot;
            const size_t inputRow = IsDirectSend ? sourceRow * nTopk + laneId : sourceRow;
            source = remoteExpertOutput + inputRow * HiddenBytes;
          } else {
            const auto* landingBase = reinterpret_cast<const uint8_t*>(transport.gpuNetIoCombineLandingBuffer_);
            source =
                landingBase + (static_cast<size_t>(destinationRank) * maxTokensPerRank + destinationSlot) * HiddenBytes;
          }
          auto* sharedRow = reinterpret_cast<uint8_t*>(sharedRows) + static_cast<size_t>(laneId) * HiddenBytes;
          bulkBarriers[laneId].arriveAndExpect(static_cast<uint32_t>(HiddenBytes));
          mscclpp::bulkLoad(sharedRow, source, static_cast<uint32_t>(HiddenBytes), bulkBarriers[laneId]);
          pending = false;
        }
      }
      if (validRow) bulkBarriers[laneId].wait(bulkPhase);
      __syncwarp();
      if (laneId == 0) mscclpp::bulkFence();
    }
    __syncthreads();

    for (int hiddenIdx = threadId; hiddenIdx < HiddenInt4; hiddenIdx += CombineNThreads) {
      float2 reduced[Bf16PairsPerInt4] = {};
#pragma unroll
      for (int stage = 0; stage < RankMajorTmaMaxNTopk; ++stage) {
        if (validRows[stage] == 0) continue;
        const int4 packed = sharedRows[stage * HiddenInt4 + hiddenIdx];
        const auto* values = reinterpret_cast<const mscclpp::bf16x2*>(&packed);
#pragma unroll
        for (int pairIdx = 0; pairIdx < Bf16PairsPerInt4; ++pairIdx) {
          const mscclpp::f32x2 value = mscclpp::to<mscclpp::f32x2>(values[pairIdx]);
          reduced[pairIdx].x += value.data[0];
          reduced[pairIdx].y += value.data[1];
        }
      }

      int4 packedOutput;
      auto* outputValues = reinterpret_cast<mscclpp::bf16x2*>(&packedOutput);
#pragma unroll
      for (int pairIdx = 0; pairIdx < Bf16PairsPerInt4; ++pairIdx) {
        outputValues[pairIdx] = mscclpp::to<mscclpp::bf16x2>(mscclpp::f32x2(reduced[pairIdx]));
      }
      auto* outputRow = reinterpret_cast<int4*>(output) + static_cast<size_t>(tokenIdx) * HiddenInt4;
      outputRow[hiddenIdx] = packedOutput;
    }
    __syncthreads();
  }
}

template <int Hidden, CombineMode Mode>
MSCCLPP_DEVICE_INLINE void recvRankMajorRemotePartials(void* output, const void* expertOutput,
                                                       const int64_t* __restrict__ topkIndices, int nTokens, int nTopk,
                                                       int nExperts, int nRanks, int maxTokensPerRank,
                                                       const TransportView& transport, WorkspaceView& workspaceView) {
  constexpr bool IsDirectSend = Mode == CombineMode::DIRECT_SEND;
  constexpr int Bf16PerInt4 = sizeof(int4) / sizeof(Bf16);
  constexpr int HiddenInt4 = Hidden / Bf16PerInt4;
  const int threadId = static_cast<int>(threadIdx.x);
  const int laneId = get_lane_id();
  const int nLocalExperts = nExperts / nRanks;

  for (int tokenIdx = static_cast<int>(blockIdx.x); tokenIdx < nTokens; tokenIdx += static_cast<int>(gridDim.x)) {
    const int globalExpertIdx = laneId < nTopk ? static_cast<int>(topkIndices[tokenIdx * nTopk + laneId]) : -1;
    const int destinationRank = globalExpertIdx >= 0 ? globalExpertIdx / nLocalExperts : -1;
    const bool firstLaneForRank = isFirstLaneForRank(destinationRank, laneId);
    const int partialRank = destinationRank >= 0 && (IsDirectSend || firstLaneForRank) ? destinationRank : -1;
    const int partialSlot = partialRank >= 0 ? workspaceView.rankMajorSendIndices_[tokenIdx * nTopk + laneId] : -1;

    for (int hiddenIdx = threadId; hiddenIdx < HiddenInt4; hiddenIdx += CombineNThreads) {
      const int4 packed = reduceRemoteRankPartialsBf16x8<HiddenInt4, Mode>(
          expertOutput, transport, partialRank, partialSlot, nTopk, maxTokensPerRank, tokenIdx, hiddenIdx);
      auto* outputRow = reinterpret_cast<int4*>(output) + static_cast<size_t>(tokenIdx) * HiddenInt4;
      outputRow[hiddenIdx] = packed;
    }
  }
}

template <int HiddenInt4>
MSCCLPP_DEVICE_INLINE int4 reduceRemoteTokenPartialsBf16x8(const void* expertOutput, const TransportView& transport,
                                                           int destinationRankCandidate, float weightCandidate,
                                                           int nTopk, int maxTokensPerRank, int tokenIdx,
                                                           int hiddenIdx) {
  constexpr int Bf16PairsPerInt4 = sizeof(int4) / sizeof(mscclpp::bf16x2);
  float2 reduced[Bf16PairsPerInt4] = {};
  for (int topkLane = 0; topkLane < nTopk; ++topkLane) {
    const int destinationRank = warpBroadcast(destinationRankCandidate, topkLane);
    if (destinationRank < 0) continue;
    const float weight = warpBroadcast(weightCandidate, topkLane);
    const auto* remoteExpertOutput =
        reinterpret_cast<const int4*>(transport.mappedBuffer(const_cast<void*>(expertOutput), destinationRank));
    const size_t rowIndex = (static_cast<size_t>(transport.rank_) * maxTokensPerRank + tokenIdx) * nTopk + topkLane;
    const int4 packed = remoteExpertOutput[rowIndex * HiddenInt4 + hiddenIdx];
    const auto* values = reinterpret_cast<const mscclpp::bf16x2*>(&packed);
#pragma unroll
    for (int pairIdx = 0; pairIdx < Bf16PairsPerInt4; ++pairIdx) {
      const mscclpp::f32x2 value = mscclpp::to<mscclpp::f32x2>(values[pairIdx]);
      reduced[pairIdx].x = fmaf(value.data[0], weight, reduced[pairIdx].x);
      reduced[pairIdx].y = fmaf(value.data[1], weight, reduced[pairIdx].y);
    }
  }

  int4 packedOutput;
  auto* outputValues = reinterpret_cast<mscclpp::bf16x2*>(&packedOutput);
#pragma unroll
  for (int pairIdx = 0; pairIdx < Bf16PairsPerInt4; ++pairIdx) {
    outputValues[pairIdx] = mscclpp::to<mscclpp::bf16x2>(mscclpp::f32x2(reduced[pairIdx]));
  }
  return packedOutput;
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void recvRankMajorTopkExpandedRemotePartials(void* output, const void* expertOutput,
                                                                   const int64_t* __restrict__ topkIndices,
                                                                   const float* __restrict__ topkWeights, int nTokens,
                                                                   int nTopk, int nExperts, int nRanks,
                                                                   int maxTokensPerRank,
                                                                   const TransportView& transport) {
  constexpr int Bf16PerInt4 = sizeof(int4) / sizeof(Bf16);
  constexpr int HiddenInt4 = Hidden / Bf16PerInt4;
  const int threadId = static_cast<int>(threadIdx.x);
  const int laneId = get_lane_id();
  const int nLocalExperts = nExperts / nRanks;

  for (int tokenIdx = static_cast<int>(blockIdx.x); tokenIdx < nTokens; tokenIdx += static_cast<int>(gridDim.x)) {
    const int globalExpertIdx = laneId < nTopk ? static_cast<int>(topkIndices[tokenIdx * nTopk + laneId]) : -1;
    const float weight =
        laneId < nTopk ? (topkWeights == nullptr ? 1.0f : topkWeights[tokenIdx * nTopk + laneId]) : 0.0f;
    const bool validExpert = globalExpertIdx >= 0 && globalExpertIdx < nExperts;
    const int destinationRank = validExpert && weight != 0.0f ? globalExpertIdx / nLocalExperts : -1;

    for (int hiddenIdx = threadId; hiddenIdx < HiddenInt4; hiddenIdx += CombineNThreads) {
      const int4 packed = reduceRemoteTokenPartialsBf16x8<HiddenInt4>(expertOutput, transport, destinationRank, weight,
                                                                      nTopk, maxTokensPerRank, tokenIdx, hiddenIdx);
      auto* outputRow = reinterpret_cast<int4*>(output) + static_cast<size_t>(tokenIdx) * HiddenInt4;
      outputRow[hiddenIdx] = packed;
    }
  }
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void recvRankMajorTopkExpandedRemotePartialsTma(
    void* output, const void* expertOutput, const int64_t* __restrict__ topkIndices,
    const float* __restrict__ topkWeights, int nTokens, int nTopk, int nExperts, int nRanks, int maxTokensPerRank,
    uint32_t epoch, const TransportView& transport, WorkspaceView& workspaceView, uint8_t* sharedMemory) {
#if defined(__CUDA_ARCH__)
  static_assert(__CUDA_ARCH__ >= 900, "TMA rank-major-topk-expanded combine requires SM90 or newer");
#endif
  static_assert(Hidden % (sizeof(int4) / sizeof(Bf16)) == 0);
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(Bf16);
  constexpr int HiddenInt4 = HiddenBytes / sizeof(int4);
  constexpr int Bf16PairsPerInt4 = sizeof(int4) / sizeof(mscclpp::bf16x2);
  constexpr size_t RowsBytes = static_cast<size_t>(RankMajorTmaMaxNTopk) * HiddenBytes;
  constexpr size_t BarrierBytes = static_cast<size_t>(RankMajorTmaMaxNTopk) * sizeof(mscclpp::BulkBarrier);
  const int threadId = static_cast<int>(threadIdx.x);
  const int warpId = threadId / WARP_SIZE;
  const int laneId = get_lane_id();
  const int nLocalExperts = nExperts / nRanks;
  auto* sharedRows = reinterpret_cast<int4*>(sharedMemory);
  auto* bulkBarriers = reinterpret_cast<mscclpp::BulkBarrier*>(sharedMemory + RowsBytes);
  auto* validRows = reinterpret_cast<int*>(bulkBarriers + RankMajorTmaMaxNTopk);
  auto* slotWeights = reinterpret_cast<float*>(validRows + RankMajorTmaMaxNTopk);
  const int nWorkerBlocks = static_cast<int>(gridDim.x) - 1;
  uint32_t bulkPhase = 0;

  if (warpId == 0) {
    if (laneId < RankMajorTmaMaxNTopk) bulkBarriers[laneId].init();
    __syncwarp();
  }

  for (int tokenIdx = static_cast<int>(blockIdx.x) - 1; tokenIdx < nTokens; tokenIdx += nWorkerBlocks) {
    if (warpId == 0) {
      const int globalExpertIdx = laneId < nTopk ? static_cast<int>(topkIndices[tokenIdx * nTopk + laneId]) : -1;
      const bool validExpert = globalExpertIdx >= 0 && globalExpertIdx < nExperts;
      const float weight =
          validExpert ? (topkWeights == nullptr ? 1.0f : topkWeights[tokenIdx * nTopk + laneId]) : 0.0f;
      const int destinationRank = validExpert && weight != 0.0f ? globalExpertIdx / nLocalExperts : -1;
      const bool validRow = laneId < RankMajorTmaMaxNTopk && destinationRank >= 0;
      if (laneId < RankMajorTmaMaxNTopk) {
        validRows[laneId] = validRow;
        slotWeights[laneId] = validRow ? weight : 0.0f;
      }
      __syncwarp();

      bool pending = validRow;
      while (__any_sync(0xffffffff, pending)) {
        const bool ready = pending && mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(
                                          workspaceView.combineRankReadyEpochs_ + destinationRank,
                                          mscclpp::memoryOrderRelaxed) == epoch;
        if (pending && ready) {
          const auto* remoteExpertOutput = reinterpret_cast<const uint8_t*>(
              transport.mappedBuffer(const_cast<void*>(expertOutput), destinationRank));
          const auto* source =
              remoteExpertOutput +
              ((static_cast<size_t>(transport.rank_) * maxTokensPerRank + tokenIdx) * nTopk + laneId) * HiddenBytes;
          auto* sharedRow = reinterpret_cast<uint8_t*>(sharedRows) + static_cast<size_t>(laneId) * HiddenBytes;
          bulkBarriers[laneId].arriveAndExpect(static_cast<uint32_t>(HiddenBytes));
          mscclpp::bulkLoad(sharedRow, source, static_cast<uint32_t>(HiddenBytes), bulkBarriers[laneId]);
          pending = false;
        }
      }
      if (validRow) bulkBarriers[laneId].wait(bulkPhase);
      __syncwarp();
      if (laneId == 0) mscclpp::bulkFence();
    }
    __syncthreads();

    for (int hiddenIdx = threadId; hiddenIdx < HiddenInt4; hiddenIdx += CombineNThreads) {
      float2 reduced[Bf16PairsPerInt4] = {};
#pragma unroll
      for (int stage = 0; stage < RankMajorTmaMaxNTopk; ++stage) {
        if (validRows[stage] == 0) continue;
        const float weight = slotWeights[stage];
        const int4 packed = sharedRows[stage * HiddenInt4 + hiddenIdx];
        const auto* values = reinterpret_cast<const mscclpp::bf16x2*>(&packed);
#pragma unroll
        for (int pairIdx = 0; pairIdx < Bf16PairsPerInt4; ++pairIdx) {
          const mscclpp::f32x2 value = mscclpp::to<mscclpp::f32x2>(values[pairIdx]);
          reduced[pairIdx].x = fmaf(value.data[0], weight, reduced[pairIdx].x);
          reduced[pairIdx].y = fmaf(value.data[1], weight, reduced[pairIdx].y);
        }
      }

      int4 packedOutput;
      auto* outputValues = reinterpret_cast<mscclpp::bf16x2*>(&packedOutput);
#pragma unroll
      for (int pairIdx = 0; pairIdx < Bf16PairsPerInt4; ++pairIdx) {
        outputValues[pairIdx] = mscclpp::to<mscclpp::bf16x2>(mscclpp::f32x2(reduced[pairIdx]));
      }
      auto* outputRow = reinterpret_cast<int4*>(output) + static_cast<size_t>(tokenIdx) * HiddenInt4;
      outputRow[hiddenIdx] = packedOutput;
    }
    __syncthreads();
  }
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void recvRankLocalPartials(void* output, const int64_t* __restrict__ topkIndices, int nTokens,
                                                 int nTopk, int nExperts, int nRanks, int maxTokensPerRank,
                                                 const void* combineRecvBuffer, uint8_t* sharedMemory) {
  const int threadId = static_cast<int>(threadIdx.x);
  const int laneId = get_lane_id();
  const int nLocalExperts = nExperts / nRanks;
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(Bf16);
  constexpr int HiddenInt4 = HiddenBytes / sizeof(int4);
  constexpr int ChunksPerThread = (HiddenInt4 + CombineNThreads - 1) / CombineNThreads;
  static_assert(HiddenInt4 % WARP_SIZE == 0);
  auto* outputTiles = sharedMemory;

  int tokenIteration = 0;
  for (int tokenIdx = static_cast<int>(blockIdx.x); tokenIdx < nTokens;
       tokenIdx += static_cast<int>(gridDim.x), ++tokenIteration) {
    const int stage = tokenIteration % CombineNStages;
    auto* outputTile = reinterpret_cast<int4*>(outputTiles + static_cast<size_t>(stage) * HiddenBytes);
    const int globalExpertIdx = laneId < nTopk ? static_cast<int>(topkIndices[tokenIdx * nTopk + laneId]) : -1;
    const int destinationRank = globalExpertIdx >= 0 ? globalExpertIdx / nLocalExperts : -1;
    const bool firstLaneForRank = isFirstLaneForRank(destinationRank, laneId);
    const int partialRank = destinationRank >= 0 && firstLaneForRank ? destinationRank : -1;

    int4 reduced[ChunksPerThread] = {};
#pragma unroll
    for (int chunkIdx = 0; chunkIdx < ChunksPerThread; ++chunkIdx) {
      const int hiddenIdx = threadId + chunkIdx * CombineNThreads;
      if (hiddenIdx < HiddenInt4) {
        reduced[chunkIdx] = reduceRankPartialsBf16x8<HiddenInt4>(combineRecvBuffer, partialRank, nTopk,
                                                                 maxTokensPerRank, tokenIdx, hiddenIdx);
      }
    }
    if (tokenIteration >= CombineNStages && threadId == 0) {
      mscclpp::bulkStoreWaitSource<CombineNStages - 1>();
    }
    if (tokenIteration >= CombineNStages) __syncthreads();
#pragma unroll
    for (int chunkIdx = 0; chunkIdx < ChunksPerThread; ++chunkIdx) {
      const int hiddenIdx = threadId + chunkIdx * CombineNThreads;
      if (hiddenIdx < HiddenInt4) outputTile[hiddenIdx] = reduced[chunkIdx];
    }
    __syncthreads();

    if (threadId == 0) {
      mscclpp::bulkFence();
      auto* outputRow = reinterpret_cast<uint8_t*>(output) + static_cast<size_t>(tokenIdx) * HiddenBytes;
      mscclpp::bulkStore(outputRow, outputTile, static_cast<uint32_t>(HiddenBytes));
      mscclpp::bulkStoreCommit();
    }
  }
  if (tokenIteration > 0 && threadId == 0) mscclpp::bulkStoreWait();
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void recvExpertRowsDirect(void* output, const int64_t* __restrict__ topkIndices,
                                                const float* __restrict__ topkWeights, int nTokens, int nTopk,
                                                int maxTokensPerRank, const void* combineRecvBuffer) {
  constexpr int Bf16PerInt4 = sizeof(int4) / sizeof(Bf16);
  constexpr int HiddenInt4 = Hidden / Bf16PerInt4;
  const int threadId = static_cast<int>(threadIdx.x);

  for (int tokenIdx = static_cast<int>(blockIdx.x); tokenIdx < nTokens; tokenIdx += static_cast<int>(gridDim.x)) {
    int regTopkIndices[CombineMaxNTopk];
    float regTopkWeights[CombineMaxNTopk];
    for (int topkIdx = 0; topkIdx < nTopk; ++topkIdx) {
      regTopkIndices[topkIdx] = static_cast<int>(topkIndices[tokenIdx * nTopk + topkIdx]);
      regTopkWeights[topkIdx] = topkWeights == nullptr ? 1.0f : topkWeights[tokenIdx * nTopk + topkIdx];
    }

#pragma unroll
    for (int hiddenIdx = threadId; hiddenIdx < HiddenInt4; hiddenIdx += CombineNThreads) {
      float reduced[Bf16PerInt4] = {0.0f};
      for (int topkIdx = 0; topkIdx < nTopk; ++topkIdx) {
        const int expertIdx = regTopkIndices[topkIdx];
        if (expertIdx < 0) continue;
        const auto* expertRow = reinterpret_cast<const int4*>(combineRecvBuffer) +
                                (static_cast<size_t>(expertIdx) * maxTokensPerRank + tokenIdx) * HiddenInt4;
        const int4 packed = expertRow[hiddenIdx];
        const auto* values = reinterpret_cast<const Bf16*>(&packed);
#pragma unroll
        for (int elemIdx = 0; elemIdx < Bf16PerInt4; ++elemIdx) {
          reduced[elemIdx] += static_cast<float>(values[elemIdx]) * regTopkWeights[topkIdx];
        }
      }

      int4 packedOutput;
      auto* outputValues = reinterpret_cast<Bf16*>(&packedOutput);
#pragma unroll
      for (int elemIdx = 0; elemIdx < Bf16PerInt4; ++elemIdx) {
        outputValues[elemIdx] = static_cast<Bf16>(reduced[elemIdx]);
      }
      auto* outputRow = reinterpret_cast<int4*>(output) + static_cast<size_t>(tokenIdx) * HiddenInt4;
      outputRow[hiddenIdx] = packedOutput;
    }
  }
}

#endif  // MSCCLPP_BULK_AVAILABLE

// Slot this token routed to destinationRank (or -1); mirrors the rank-major send index.
MSCCLPP_DEVICE_INLINE int rankMajorSlotForDestination(const int64_t* __restrict__ topkIndices,
                                                      const WorkspaceView& workspaceView, int tokenIdx, int nTopk,
                                                      int nLocalExperts, int destinationRank) {
  for (int k = 0; k < nTopk; ++k) {
    const int expertIdx = static_cast<int>(topkIndices[tokenIdx * nTopk + k]);
    if (expertIdx >= 0 && expertIdx / nLocalExperts == destinationRank) {
      return workspaceView.rankMajorSendIndices_[tokenIdx * nTopk + k];
    }
  }
  return -1;
}

#if defined(MSCCLPP_USE_GPUNETIO)
MSCCLPP_DEVICE_INLINE void signalRankMajorCombineLocalStart(const TransportView& transport, int nRanks) {
  if (blockIdx.x != 0) return;
  const int peerRank = static_cast<int>(threadIdx.x);
  if (peerRank < nRanks && !transport.isSelf(peerRank) && transport.isNvlinkPeer(peerRank)) {
    transport.baseMemoryChannels_[peerRank].relaxedSignal();
  }
}

MSCCLPP_DEVICE_INLINE int rankMajorCombineStripeQp(const mscclpp::GpuNetIoDeviceContext* gin, int owner, int stripe) {
  EP_DEVICE_ASSERT(gin->numHcas > 0 && gin->numHcas <= gin->numQpsPerPeer);
  return (owner % gin->numQpsPerPeer + stripe) % gin->numQpsPerPeer;
}

MSCCLPP_DEVICE_INLINE void publishRankMajorCombinePushReady(const int64_t* __restrict__ topkIndices, int nTokens,
                                                            int nTopk, int nLocalExperts, int nRanks, uint32_t epoch,
                                                            const TransportView& transport,
                                                            WorkspaceView& workspaceView) {
  if (blockIdx.x != 0) return;
  const int destinationRank = static_cast<int>(threadIdx.x);
  if (destinationRank >= nRanks) return;

  if (!transport.isSelf(destinationRank) && transport.isNvlinkPeer(destinationRank)) {
    transport.baseMemoryChannels_[destinationRank].relaxedWait(-1);
  } else if (!transport.isSelf(destinationRank)) {
    bool sendsToRank = false;
    for (int tokenIdx = 0; tokenIdx < nTokens && !sendsToRank; ++tokenIdx) {
      sendsToRank =
          rankMajorSlotForDestination(topkIndices, workspaceView, tokenIdx, nTopk, nLocalExperts, destinationRank) >= 0;
    }
    if (sendsToRank) {
      auto* gin = transport.gpuNetIo_;
      auto* flags = reinterpret_cast<volatile uint64_t*>(transport.gpuNetIoCombineFlagsBuffer_);
      const uint64_t target = workspaceView.combineArrivedBaseline_[destinationRank] + 1;
      for (int stripe = 0; stripe < gin->numHcas; ++stripe) {
        const int qpIndex = rankMajorCombineStripeQp(gin, transport.rank_, stripe);
        const size_t flagIndex = static_cast<size_t>(destinationRank) * GpuNetIoMaxQpsPerPeer + qpIndex;
        while (flags[flagIndex] < target) {
        }
      }
      workspaceView.combineArrivedBaseline_[destinationRank] = target;
    }
  }
  mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(workspaceView.combineRankReadyEpochs_ + destinationRank, epoch,
                                                       mscclpp::memoryOrderRelaxed);
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void sendRankMajorCombinePush(const void* expertOutput, int nRanks, int maxTokensPerRank,
                                                    const TransportView& transport, WorkspaceView& workspaceView,
                                                    [[maybe_unused]] uint32_t epoch) {
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(Bf16);
  auto* gin = transport.gpuNetIo_;
  auto* landingBase = reinterpret_cast<uint8_t*>(transport.gpuNetIoCombineLandingBuffer_);
  const int nQp = gin->numQpsPerPeer;
  if (nQp < 1 || nQp > GpuNetIoMaxQpsPerPeer) __trap();
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  const long long postStart = clock64();
#endif
  const int owner = static_cast<int>(blockIdx.x);
  if (owner < nRanks && !transport.isSelf(owner) && !transport.isNvlinkPeer(owner) &&
      workspaceView.dispatchRecvCounts_[owner] > 0 && static_cast<int>(threadIdx.x) < gin->numHcas) {
    const int nRowsToOwner = workspaceView.dispatchRecvCounts_[owner];
    const int stripe = static_cast<int>(threadIdx.x);
    const int rowBegin = nRowsToOwner * stripe / gin->numHcas;
    const int rowEnd = nRowsToOwner * (stripe + 1) / gin->numHcas;
    const int qpIndex = rankMajorCombineStripeQp(gin, owner, stripe);
    const uint64_t srcRowOffset = transport.symmetricOffset(const_cast<void*>(expertOutput)) +
                                  (static_cast<size_t>(owner) * maxTokensPerRank + rowBegin) * HiddenBytes;
    auto* landingSlot =
        landingBase + (static_cast<size_t>(transport.rank_) * maxTokensPerRank + rowBegin) * HiddenBytes;
    auto* remoteFlag = static_cast<uint64_t*>(transport.gpuNetIoCombineFlagsBuffer_) +
                       static_cast<size_t>(transport.rank_) * GpuNetIoMaxQpsPerPeer + qpIndex;
    gin->putWithSignal(owner, transport.symmetricOffset(landingSlot), srcRowOffset,
                       static_cast<uint64_t>(rowEnd - rowBegin) * HiddenBytes, transport.symmetricOffset(remoteFlag), 1,
                       qpIndex);
  }
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  const int stripe = static_cast<int>(threadIdx.x);
  if (stripe < gin->numHcas && owner < nRanks && !transport.isNvlinkPeer(owner))
    printf("[GINTIME-CMBSEND-STRIPE] r=%d owner=%d ep=%u stripe=%d qp=%d rows=%d submit_cyc=%lld\n", transport.rank_,
           owner, epoch, stripe, rankMajorCombineStripeQp(gin, owner, stripe), workspaceView.dispatchRecvCounts_[owner],
           clock64() - postStart);
#endif
}

MSCCLPP_DEVICE_INLINE void drainRankMajorCombinePush(int nRanks, const TransportView& transport,
                                                     WorkspaceView& workspaceView, [[maybe_unused]] uint32_t epoch) {
  auto* gin = transport.gpuNetIo_;
  const int owner = blockIdx.x;
  if (owner >= nRanks || transport.isNvlinkPeer(owner) || workspaceView.dispatchRecvCounts_[owner] <= 0 ||
      static_cast<int>(threadIdx.x) >= gin->numHcas)
    return;
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  const long long drainStart = clock64();
#endif
  gin->flush(owner, rankMajorCombineStripeQp(gin, owner, static_cast<int>(threadIdx.x)));
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  const int stripe = static_cast<int>(threadIdx.x);
  printf("[GINTIME-CMBTAIL] r=%d owner=%d ep=%u stripe=%d qp=%d rows=%d drain_cyc=%lld\n", transport.rank_, owner,
         epoch, stripe, rankMajorCombineStripeQp(gin, owner, stripe), workspaceView.dispatchRecvCounts_[owner],
         clock64() - drainStart);
#endif
}

// Cross-domain rank-major combine receive + reduce (PUSH model): wait for each
// cross-domain expert host's completion flag, then reduce NVLink rows (direct
// mapped read) and cross-domain rows (from the landing region) into the output.
template <int Hidden>
MSCCLPP_DEVICE_INLINE void recvRankMajorCombinePush(void* output, const void* expertOutput,
                                                    const int64_t* __restrict__ topkIndices, int nTokens, int nTopk,
                                                    int nExperts, int nRanks, int maxTokensPerRank,
                                                    const TransportView& transport, WorkspaceView& workspaceView) {
  constexpr int Bf16PerInt4 = sizeof(int4) / sizeof(Bf16);
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(Bf16);
  constexpr int HiddenInt4 = Hidden / Bf16PerInt4;
  constexpr int Bf16PairsPerInt4 = sizeof(int4) / sizeof(mscclpp::bf16x2);
  const int threadId = static_cast<int>(threadIdx.x);
  const int nLocalExperts = nExperts / nRanks;
  auto* landingBase = reinterpret_cast<uint8_t*>(transport.gpuNetIoCombineLandingBuffer_);
  auto* gin = transport.gpuNetIo_;
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  const long long waitStart = clock64();
#endif

  if (blockIdx.x == 0 && threadId == 0) {
    auto* flags = reinterpret_cast<volatile uint64_t*>(transport.gpuNetIoCombineFlagsBuffer_);
    for (int destinationRank = 0; destinationRank < nRanks; ++destinationRank) {
      if (transport.isSelf(destinationRank) || transport.isNvlinkPeer(destinationRank)) continue;
      bool sendsToRank = false;
      for (int tokenIdx = 0; tokenIdx < nTokens && !sendsToRank; ++tokenIdx) {
        if (rankMajorSlotForDestination(topkIndices, workspaceView, tokenIdx, nTopk, nLocalExperts, destinationRank) >=
            0) {
          sendsToRank = true;
        }
      }
      if (!sendsToRank) continue;
      const uint64_t target = workspaceView.combineArrivedBaseline_[destinationRank] + 1;
      for (int stripe = 0; stripe < gin->numHcas; ++stripe) {
        const int qpIndex = rankMajorCombineStripeQp(gin, transport.rank_, stripe);
        const size_t flagIndex = static_cast<size_t>(destinationRank) * GpuNetIoMaxQpsPerPeer + qpIndex;
        while (flags[flagIndex] < target) {
        }
      }
      workspaceView.combineArrivedBaseline_[destinationRank] = target;
    }
  }
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  const long long waitDone = clock64();
#endif
  workspaceView.combineSyncer_->sync(gridDim.x);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  const long long barrierDone = clock64();
#endif

  for (int tokenIdx = static_cast<int>(blockIdx.x); tokenIdx < nTokens; tokenIdx += static_cast<int>(gridDim.x)) {
    for (int hiddenIdx = threadId; hiddenIdx < HiddenInt4; hiddenIdx += CombineNThreads) {
      float2 reduced[Bf16PairsPerInt4] = {};
      for (int destinationRank = 0; destinationRank < nRanks; ++destinationRank) {
        const int destinationSlot =
            rankMajorSlotForDestination(topkIndices, workspaceView, tokenIdx, nTopk, nLocalExperts, destinationRank);
        if (destinationSlot < 0) continue;
        int4 packed;
        if (transport.isNvlinkPeer(destinationRank)) {
          const auto* source =
              reinterpret_cast<const int4*>(transport.mappedBuffer(const_cast<void*>(expertOutput), destinationRank)) +
              (static_cast<size_t>(transport.rank_) * maxTokensPerRank + destinationSlot) * HiddenInt4;
          packed = source[hiddenIdx];
        } else {
          auto* landingSlot =
              landingBase + (static_cast<size_t>(destinationRank) * maxTokensPerRank + destinationSlot) * HiddenBytes;
          packed = reinterpret_cast<const int4*>(landingSlot)[hiddenIdx];
        }
        const auto* values = reinterpret_cast<const mscclpp::bf16x2*>(&packed);
#pragma unroll
        for (int pairIdx = 0; pairIdx < Bf16PairsPerInt4; ++pairIdx) {
          const mscclpp::f32x2 value = mscclpp::to<mscclpp::f32x2>(values[pairIdx]);
          reduced[pairIdx].x += value.data[0];
          reduced[pairIdx].y += value.data[1];
        }
      }
      int4 packedOutput;
      auto* outputValues = reinterpret_cast<mscclpp::bf16x2*>(&packedOutput);
#pragma unroll
      for (int pairIdx = 0; pairIdx < Bf16PairsPerInt4; ++pairIdx) {
        outputValues[pairIdx] = mscclpp::to<mscclpp::bf16x2>(mscclpp::f32x2(reduced[pairIdx]));
      }
      auto* outputRow = reinterpret_cast<int4*>(output) + static_cast<size_t>(tokenIdx) * HiddenInt4;
      outputRow[hiddenIdx] = packedOutput;
    }
  }
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  const int block = blockIdx.x;
  if (threadIdx.x == 0 && (block == 0 || block == 1 || block == gridDim.x / 2 || block == gridDim.x - 1))
    printf("[GINTIME-CMBRECV] r=%d blk=%d wait_cyc=%lld bar_cyc=%lld reduce_cyc=%lld\n", transport.rank_, block,
           waitDone - waitStart, barrierDone - waitDone, clock64() - barrierDone);
#endif
}
#endif  // defined(MSCCLPP_USE_GPUNETIO)

template <CombineMode Mode, int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout>
MSCCLPP_DEVICE_INLINE void combineBody(void* output, const void* expertOutput, const int64_t* __restrict__ topkIndices,
                                       const float* __restrict__ topkWeights, const int* srcInfo,
                                       const int64_t* layoutRange, Workload workload, void* combineRecvBuffer,
                                       const void* dispatchRecvBuffer, const DeviceContext* context) {
#if MSCCLPP_BULK_AVAILABLE
  extern __shared__ __align__(128) uint8_t sharedMemory[];
  const int nTokens = workload.numTokens_;
  const int nExperts = workload.numExperts_;
  const int nRanks = context->numRanks_;
  const int nTopk = workload.numTopk_;
  const int maxTokensPerRank = workload.maxTokensPerRank_;
  const TransportView transport(context);
  WorkspaceView workspaceView(context->workspace_, nRanks, nExperts);

  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    static_assert(Mode == CombineMode::RANK_LOCAL_REDUCE || Mode == CombineMode::DIRECT_SEND);
    static_assert(DispatchType == DispatchDataType::BF16);
#if defined(MSCCLPP_USE_GPUNETIO)
    // Cross-domain active: use the GPUNetIO PUSH combine (NVLink peers via mapped
    // read, cross-domain peers via landing region) instead of the pull/gather.
    if (transport.gpuNetIo_ != nullptr) {
      const uint32_t epoch = workload.epoch_ * 2;
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
      const long long syncStart = clock64();
#endif
      if (nTopk <= RankMajorTmaMaxNTopk) {
        signalRankMajorCombineLocalStart(transport, nRanks);
      } else {
        synchronizeRankMajorCombine(transport, nRanks, epoch, workspaceView);
      }
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
      const long long pushStart = clock64();
#endif
      sendRankMajorCombinePush<Hidden>(expertOutput, nRanks, maxTokensPerRank, transport, workspaceView,
                                       workload.epoch_);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
      const long long recvStart = clock64();
#endif
      if (nTopk <= RankMajorTmaMaxNTopk) {
        if (blockIdx.x == 0) {
          publishRankMajorCombinePushReady(topkIndices, nTokens, nTopk, nExperts / nRanks, nRanks, workload.epoch_,
                                           transport, workspaceView);
        } else {
          recvRankMajorRemotePartialsTma<Hidden, Mode>(output, expertOutput, topkIndices, nTokens, nTopk, nExperts,
                                                       nRanks, maxTokensPerRank, workload.epoch_, transport,
                                                       workspaceView, sharedMemory);
        }
      } else {
        recvRankMajorCombinePush<Hidden>(output, expertOutput, topkIndices, nTokens, nTopk, nExperts, nRanks,
                                         maxTokensPerRank, transport, workspaceView);
      }
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
      const long long drainStart = clock64();
#endif
      drainRankMajorCombinePush(nRanks, transport, workspaceView, workload.epoch_);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
      const long long syncEndStart = clock64();
#endif
      workspaceView.combineSyncer_->sync(gridDim.x);
      synchronizeRankMajorCombine(transport, nRanks, epoch + 1, workspaceView);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
      const int block = static_cast<int>(blockIdx.x);
      const int blocks = static_cast<int>(gridDim.x);
      if (threadIdx.x == 0 && (block == 0 || block == 1 || block == blocks / 2 || block == blocks - 1))
        printf(
            "[GINTIME-CMB] r=%d ep=%u blk=%d start_cyc=%lld push_cyc=%lld recv_cyc=%lld "
            "drain_cyc=%lld sync1_cyc=%lld\n",
            transport.rank_, workload.epoch_, block, pushStart - syncStart, recvStart - pushStart,
            drainStart - recvStart, syncEndStart - drainStart, clock64() - syncEndStart);
#endif
      return;
    }
#endif  // defined(MSCCLPP_USE_GPUNETIO)
    if (nTopk <= RankMajorTmaMaxNTopk) {
      const uint32_t epoch = workload.epoch_;
      if (blockIdx.x == 0) {
        publishRankMajorCombineReady(transport, nRanks, epoch, workspaceView);
      } else {
        recvRankMajorRemotePartialsTma<Hidden, Mode>(output, expertOutput, topkIndices, nTokens, nTopk, nExperts,
                                                     nRanks, maxTokensPerRank, epoch, transport, workspaceView,
                                                     sharedMemory);
      }
      return;
    }
    synchronizeRankMajorCombine(transport, nRanks, workload.epoch_, workspaceView);
    recvRankMajorRemotePartials<Hidden, Mode>(output, expertOutput, topkIndices, nTokens, nTopk, nExperts, nRanks,
                                              maxTokensPerRank, transport, workspaceView);
    return;
  } else if constexpr (Layout == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED) {
    static_assert(Mode == CombineMode::RANK_LOCAL_REDUCE);
    static_assert(DispatchType == DispatchDataType::BF16);
    if (nTopk <= RankMajorTmaMaxNTopk) {
      const uint32_t epoch = workload.epoch_;
      if (blockIdx.x == 0) {
        publishRankMajorCombineReady(transport, nRanks, epoch, workspaceView);
      } else {
        recvRankMajorTopkExpandedRemotePartialsTma<Hidden>(output, expertOutput, topkIndices, topkWeights, nTokens,
                                                           nTopk, nExperts, nRanks, maxTokensPerRank, epoch, transport,
                                                           workspaceView, sharedMemory);
      }
      return;
    }
    synchronizeRankMajorCombine(transport, nRanks, workload.epoch_, workspaceView);
    recvRankMajorTopkExpandedRemotePartials<Hidden>(output, expertOutput, topkIndices, topkWeights, nTokens, nTopk,
                                                    nExperts, nRanks, maxTokensPerRank, transport);
    return;
  } else if constexpr (Mode == CombineMode::RANK_LOCAL_REDUCE) {
    sendRankReducedPartials<Hidden, DispatchType, ScaleBlockSize>(
        expertOutput, nExperts, nRanks, nTopk, maxTokensPerRank, combineRecvBuffer, dispatchRecvBuffer, transport,
        workspaceView, sharedMemory);
  } else {
    sendExpertRowsDirect<Hidden>(expertOutput, srcInfo, layoutRange, nExperts, nRanks, maxTokensPerRank,
                                 combineRecvBuffer, transport, sharedMemory);
  }

  workspaceView.combineSyncer_->sync(gridDim.x);
  exchangeCombineReady(transport, nRanks);
  workspaceView.combineSyncer_->sync(gridDim.x);

  if constexpr (Mode == CombineMode::RANK_LOCAL_REDUCE) {
    recvRankLocalPartials<Hidden>(output, topkIndices, nTokens, nTopk, nExperts, nRanks, maxTokensPerRank,
                                  combineRecvBuffer, sharedMemory);
  } else {
    recvExpertRowsDirect<Hidden>(output, topkIndices, topkWeights, nTokens, nTopk, maxTokensPerRank, combineRecvBuffer);
  }
#endif  // MSCCLPP_BULK_AVAILABLE
}

template <CombineMode Mode, int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout,
          typename KernelSelector>
inline void combineHiddenMode(void* output, const void* expertOutput, const int64_t* topkIndices,
                              const float* topkWeights, const int* srcInfo, const int64_t* layoutRange,
                              const Workload& workload, void* recvBuffer, void* dispatchRecvBuffer,
                              const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  static_assert(Hidden == 2048 || Hidden == 4096 || Hidden == 4352 || Hidden == 5120 || Hidden == 6656 ||
                Hidden == 7168 || Hidden == 8192 || Hidden == 8704 || Hidden == 9216);
  const int nExperts = workload.numExperts_;
  const int nRanks = context.numRanks_;
  const int nLocalExperts = nExperts / nRanks;
  if constexpr (Mode == CombineMode::DIRECT_SEND && Layout == DispatchLayout::EXPERT_MAJOR) {
    EP_HOST_ASSERT(directSendWorkerCount<Hidden>(nLocalExperts) > 0);
  }

  auto combineFunc = KernelSelector::template get<Hidden, DispatchType, ScaleBlockSize, Layout>();
  const size_t sharedBytes = combineSharedBytes<Hidden, Mode, Layout>(nLocalExperts, workload.numTopk_);
  const bool useRankMajorTma = Layout == DispatchLayout::RANK_MAJOR && workload.numTopk_ <= RankMajorTmaMaxNTopk;
  const bool useRankMajorTopkExpandedTma =
      Layout == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED && workload.numTopk_ <= RankMajorTmaMaxNTopk;
  const bool useTmaControlBlock = useRankMajorTma || useRankMajorTopkExpandedTma;
  EP_HOST_ASSERT(!useTmaControlBlock || numBlocks > 1);
  static thread_local KernelConfigCache kernelConfig;
  const int residentBlocks = configureKernel(combineFunc, CombineNThreads, sharedBytes, context, kernelConfig);
  EP_HOST_ASSERT(residentBlocks >= numBlocks);

  combineFunc<<<dim3(numBlocks), dim3(CombineNThreads), sharedBytes, stream>>>(
      output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload, recvBuffer, dispatchRecvBuffer,
      context.devicePtr_);
  CUDA_CHECK(cudaGetLastError());
}

template <CombineMode Mode, int Hidden, typename KernelSelector>
inline void combineHidden(void* output, const void* expertOutput, const int64_t* topkIndices, const float* topkWeights,
                          const int* srcInfo, const int64_t* layoutRange, const Workload& workload, void* recvBuffer,
                          void* dispatchRecvBuffer, const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  if (workload.outputLayout_ == DispatchLayout::RANK_MAJOR) {
    EP_HOST_ASSERT(workload.dispatchDataType_ == DispatchDataType::BF16);
    return combineHiddenMode<Mode, Hidden, DispatchDataType::BF16, 0, DispatchLayout::RANK_MAJOR, KernelSelector>(
        output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload, recvBuffer, dispatchRecvBuffer,
        context, numBlocks, stream);
  }
  if constexpr (Mode == CombineMode::RANK_LOCAL_REDUCE) {
    if (workload.outputLayout_ == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED) {
      EP_HOST_ASSERT(workload.dispatchDataType_ == DispatchDataType::BF16);
      return topk_expanded::combine(output, expertOutput, topkIndices, topkWeights, workload, context, numBlocks,
                                    stream);
    }
  }
  if constexpr (Mode == CombineMode::RANK_LOCAL_REDUCE) {
    switch (workload.dispatchDataType_) {
      case DispatchDataType::BF16:
        return combineHiddenMode<CombineMode::RANK_LOCAL_REDUCE, Hidden, DispatchDataType::BF16, 0,
                                 DispatchLayout::EXPERT_MAJOR, KernelSelector>(
            output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload, recvBuffer,
            dispatchRecvBuffer, context, numBlocks, stream);
      case DispatchDataType::FP8_E4M3:
        return combineHiddenMode<CombineMode::RANK_LOCAL_REDUCE, Hidden, DispatchDataType::FP8_E4M3, 128,
                                 DispatchLayout::EXPERT_MAJOR, KernelSelector>(
            output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload, recvBuffer,
            dispatchRecvBuffer, context, numBlocks, stream);
    }
  } else {
    switch (workload.dispatchDataType_) {
      case DispatchDataType::BF16:
        return combineHiddenMode<CombineMode::DIRECT_SEND, Hidden, DispatchDataType::BF16, 0,
                                 DispatchLayout::EXPERT_MAJOR, KernelSelector>(
            output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload, recvBuffer,
            dispatchRecvBuffer, context, numBlocks, stream);
      case DispatchDataType::FP8_E4M3:
        return combineHiddenMode<CombineMode::DIRECT_SEND, Hidden, DispatchDataType::FP8_E4M3, 128,
                                 DispatchLayout::EXPERT_MAJOR, KernelSelector>(
            output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload, recvBuffer,
            dispatchRecvBuffer, context, numBlocks, stream);
    }
  }
  EP_HOST_ASSERT(false && "unsupported dispatch data type");
}

template <CombineMode Mode, typename KernelSelector>
inline void combineAlgorithm(void* output, const void* expertOutput, const int64_t* topkIndices,
                             const float* topkWeights, const int* srcInfo, const int64_t* layoutRange,
                             const Workload& workload, void* recvBuffer, void* dispatchRecvBuffer,
                             const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  const int nExperts = workload.numExperts_;
  const int rank = context.rank_;
  const int nRanks = context.numRanks_;

  EP_HOST_ASSERT(workload.numTokens_ == 0 || output != nullptr);
  EP_HOST_ASSERT(expertOutput != nullptr);
  EP_HOST_ASSERT(workload.numTokens_ == 0 || topkIndices != nullptr);
  EP_HOST_ASSERT(recvBuffer != nullptr);
  EP_HOST_ASSERT(dispatchRecvBuffer != nullptr);
  EP_HOST_ASSERT(context.localBufferBase_ != nullptr);
  EP_HOST_ASSERT(context.peerBufferBases_ != nullptr);
  EP_HOST_ASSERT(context.channels_ != nullptr);
  EP_HOST_ASSERT(context.workspace_ != nullptr);
  EP_HOST_ASSERT(context.devicePtr_ != nullptr);
  EP_HOST_ASSERT(nRanks > 0 && nRanks <= 2 * WARP_SIZE);
  EP_HOST_ASSERT(nExperts > 0 && nExperts % nRanks == 0);
  EP_HOST_ASSERT(rank >= 0 && rank < nRanks);
  EP_HOST_ASSERT(workload.numTokens_ >= 0 && workload.numTokens_ <= workload.maxTokensPerRank_);
  EP_HOST_ASSERT(workload.numTopk_ > 0 && workload.numTopk_ <= CombineMaxNTopk);
  EP_HOST_ASSERT(numBlocks > 0 && numBlocks <= MaxWorkerBlocks);
  static_assert(Mode == CombineMode::RANK_LOCAL_REDUCE || Mode == CombineMode::DIRECT_SEND);
  EP_HOST_ASSERT(workload.outputLayout_ == DispatchLayout::EXPERT_MAJOR ||
                 workload.outputLayout_ == DispatchLayout::RANK_MAJOR ||
                 workload.outputLayout_ == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED);
  EP_HOST_ASSERT(isSupportedDispatchDataType(workload.dispatchDataType_));
  if constexpr (Mode == CombineMode::RANK_LOCAL_REDUCE) {
    if (workload.outputLayout_ == DispatchLayout::RANK_MAJOR ||
        workload.outputLayout_ == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED) {
      EP_HOST_ASSERT(workload.dispatchDataType_ == DispatchDataType::BF16);
    }
  } else {
    if (workload.outputLayout_ == DispatchLayout::EXPERT_MAJOR) {
      EP_HOST_ASSERT(srcInfo != nullptr);
      EP_HOST_ASSERT(layoutRange != nullptr);
    } else {
      EP_HOST_ASSERT(workload.dispatchDataType_ == DispatchDataType::BF16);
    }
  }

  switch (workload.hidden_) {
    case 4096:
      return combineHidden<Mode, 4096, KernelSelector>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                       layoutRange, workload, recvBuffer, dispatchRecvBuffer, context,
                                                       numBlocks, stream);
    case 4352:
      return combineHidden<Mode, 4352, KernelSelector>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                       layoutRange, workload, recvBuffer, dispatchRecvBuffer, context,
                                                       numBlocks, stream);
    case 5120:
      return combineHidden<Mode, 5120, KernelSelector>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                       layoutRange, workload, recvBuffer, dispatchRecvBuffer, context,
                                                       numBlocks, stream);
    case 6656:
      return combineHidden<Mode, 6656, KernelSelector>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                       layoutRange, workload, recvBuffer, dispatchRecvBuffer, context,
                                                       numBlocks, stream);
    case 7168:
      return combineHidden<Mode, 7168, KernelSelector>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                       layoutRange, workload, recvBuffer, dispatchRecvBuffer, context,
                                                       numBlocks, stream);
    case 8192:
      return combineHidden<Mode, 8192, KernelSelector>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                       layoutRange, workload, recvBuffer, dispatchRecvBuffer, context,
                                                       numBlocks, stream);
    case 8704:
      return combineHidden<Mode, 8704, KernelSelector>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                       layoutRange, workload, recvBuffer, dispatchRecvBuffer, context,
                                                       numBlocks, stream);
    case 9216:
      return combineHidden<Mode, 9216, KernelSelector>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                       layoutRange, workload, recvBuffer, dispatchRecvBuffer, context,
                                                       numBlocks, stream);
    default:
      EP_HOST_ASSERT(false && "unsupported latency combine hidden size");
  }
}

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_COMBINE_COMMON_CUH_
