// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <mscclpp/bulk_device.hpp>
#include <mscclpp/gpu_data_types.hpp>

#include "api.cuh"
#include "config.cuh"
#include "device_helpers.cuh"
#include "exception.cuh"

#if defined(MSCCLPP_USE_GPUNETIO)
#include <mscclpp/port_channel_gpunetio_device.hpp>
#endif  // defined(MSCCLPP_USE_GPUNETIO)

namespace mscclpp {
namespace ep {
namespace low_latency {
namespace detail {

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

template <int Hidden, low_latency::CombineMode Mode, DispatchLayout Layout>
MSCCLPP_HOST_DEVICE_INLINE size_t combineSharedBytes(int nLocalExperts, int nTopk) {
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
  if constexpr (Mode == low_latency::CombineMode::DIRECT_SEND) {
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
  if (blockIdx.x == 0 && threadId < nRanks) {
    const int peerRank = threadId;
    if (!transport.isSelf(peerRank)) {
#if defined(MSCCLPP_USE_GPUNETIO)
      if (transport.gpuNetIo_ != nullptr && !transport.isNvlinkPeer(peerRank)) {
        auto* remoteFlag = reinterpret_cast<uint8_t*>(transport.gpuNetIoFlagsBuffer_) +
                           static_cast<size_t>(transport.rank_) * sizeof(uint64_t);
        transport.gpuNetIo_->atomicAdd(peerRank, transport.symmetricOffset(remoteFlag), 1);
        transport.gpuNetIo_->flush(peerRank);
        auto* flags = reinterpret_cast<volatile uint64_t*>(transport.gpuNetIoFlagsBuffer_);
        while (flags[peerRank] < 1) {
        }
        flags[peerRank] = 0;
      } else
#endif  // defined(MSCCLPP_USE_GPUNETIO)
      {
        transport.baseMemoryChannels_[peerRank].relaxedSignal();
        transport.baseMemoryChannels_[peerRank].relaxedWait(-1);
      }
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

template <int HiddenInt4>
MSCCLPP_DEVICE_INLINE int4 reduceRemoteRankPartialsBf16x8(const void* expertOutput, const TransportView& transport,
                                                          int destinationRankCandidate, int destinationSlotCandidate,
                                                          int nTopk, int maxTokensPerRank, int tokenIdx,
                                                          int hiddenIdx) {
  constexpr int Bf16PairsPerInt4 = sizeof(int4) / sizeof(mscclpp::bf16x2);
  float2 reduced[Bf16PairsPerInt4] = {};
  for (int topkLane = 0; topkLane < nTopk; ++topkLane) {
    const int destinationRank = warpBroadcast(destinationRankCandidate, topkLane);
    if (destinationRank < 0) continue;
    const int destinationSlot = warpBroadcast(destinationSlotCandidate, topkLane);
    EP_DEVICE_ASSERT(destinationSlot >= 0 && destinationSlot < maxTokensPerRank);
    const auto* remoteExpertOutput =
        reinterpret_cast<const int4*>(transport.mappedBuffer(const_cast<void*>(expertOutput), destinationRank));
    const int4 packed =
        remoteExpertOutput[(static_cast<size_t>(transport.rank_) * maxTokensPerRank + destinationSlot) * HiddenInt4 +
                           hiddenIdx];
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

template <int Hidden>
MSCCLPP_DEVICE_INLINE void recvRankMajorRemotePartialsTma(void* output, const void* expertOutput,
                                                          const int64_t* __restrict__ topkIndices, int nTokens,
                                                          int nTopk, int nExperts, int nRanks, int maxTokensPerRank,
                                                          uint32_t epoch, const TransportView& transport,
                                                          WorkspaceView& workspaceView, uint8_t* sharedMemory) {
#if defined(__CUDA_ARCH__)
  static_assert(__CUDA_ARCH__ >= 900, "TMA rank-major combine requires SM90 or newer");
#endif
  static_assert(Hidden % (sizeof(int4) / sizeof(Bf16)) == 0);
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
      const bool validRow = laneId < RankMajorTmaMaxNTopk && destinationRank >= 0 && firstLaneForRank;
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
          const auto* remoteExpertOutput = reinterpret_cast<const uint8_t*>(
              transport.mappedBuffer(const_cast<void*>(expertOutput), destinationRank));
          const auto* source =
              remoteExpertOutput +
              (static_cast<size_t>(transport.rank_) * maxTokensPerRank + destinationSlot) * HiddenBytes;
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

#if defined(MSCCLPP_USE_GPUNETIO)
MSCCLPP_DEVICE_INLINE int rankMajorSlotForDestination(const int64_t* __restrict__ topkIndices,
                                                      WorkspaceView& workspaceView, int tokenIdx, int nTopk,
                                                      int nLocalExperts, int destinationRank) {
  int destinationSlot = -1;
#pragma unroll
  for (int topkLane = 0; topkLane < WARP_SIZE; ++topkLane) {
    if (topkLane < nTopk) {
      const int globalExpertIdx = static_cast<int>(topkIndices[tokenIdx * nTopk + topkLane]);
      const int rank = globalExpertIdx >= 0 ? globalExpertIdx / nLocalExperts : -1;
      if (destinationSlot < 0 && rank == destinationRank) {
        destinationSlot = workspaceView.rankMajorSendIndices_[tokenIdx * nTopk + topkLane];
      }
    }
  }
  return destinationSlot;
}

// Cross-domain rank-major combine (approach 2). The reduce is a remote *read*
// (pull) over peers outside this rank's NVLink/IPC domain, which one-sided put
// cannot express, so each needed remote expert-output row is first RDMA-read
// (gin->get) into the symmetric staging ring, then reduced from there.
template <int Hidden>
MSCCLPP_DEVICE_INLINE void recvRankMajorRemotePartialsGpuNetIo(void* output, const void* expertOutput,
                                                              const int64_t* __restrict__ topkIndices, int nTokens,
                                                              int nTopk, int nExperts, int nRanks, int maxTokensPerRank,
                                                              const TransportView& transport,
                                                              WorkspaceView& workspaceView) {
  constexpr int Bf16PerInt4 = sizeof(int4) / sizeof(Bf16);
  constexpr int HiddenInt4 = Hidden / Bf16PerInt4;
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(Bf16);
  constexpr int Bf16PairsPerInt4 = sizeof(int4) / sizeof(mscclpp::bf16x2);
  const int threadId = static_cast<int>(threadIdx.x);
  const int nLocalExperts = nExperts / nRanks;
  auto* gin = transport.gpuNetIo_;
  auto* stagingBase = reinterpret_cast<uint8_t*>(transport.gpuNetIoStagingBuffer_);

  for (int tokenIdx = static_cast<int>(blockIdx.x); tokenIdx < nTokens; tokenIdx += static_cast<int>(gridDim.x)) {
    if (threadId == 0) {
      for (int destinationRank = 0; destinationRank < nRanks; ++destinationRank) {
        const int destinationSlot =
            rankMajorSlotForDestination(topkIndices, workspaceView, tokenIdx, nTopk, nLocalExperts, destinationRank);
        if (destinationSlot < 0 || transport.isNvlinkPeer(destinationRank)) continue;
        auto* slot = stagingBase +
                     static_cast<size_t>((static_cast<int>(blockIdx.x) * nRanks + destinationRank) %
                                         GpuNetIoStagingSlots) *
                         transport.gpuNetIoSlotStride_;
        const uint64_t remoteRowOffset =
            transport.symmetricOffset(const_cast<void*>(expertOutput)) +
            (static_cast<size_t>(transport.rank_) * maxTokensPerRank + destinationSlot) * HiddenBytes;
        gin->get(destinationRank, remoteRowOffset, transport.symmetricOffset(slot), HiddenBytes);
      }
    }
    __syncthreads();

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
          auto* slot = stagingBase +
                       static_cast<size_t>((static_cast<int>(blockIdx.x) * nRanks + destinationRank) %
                                           GpuNetIoStagingSlots) *
                           transport.gpuNetIoSlotStride_;
          packed = reinterpret_cast<const int4*>(slot)[hiddenIdx];
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
    __syncthreads();
  }
}
#endif  // defined(MSCCLPP_USE_GPUNETIO)

template <int Hidden>
MSCCLPP_DEVICE_INLINE void recvRankMajorRemotePartials(void* output, const void* expertOutput,
                                                       const int64_t* __restrict__ topkIndices, int nTokens, int nTopk,
                                                       int nExperts, int nRanks, int maxTokensPerRank,
                                                       const TransportView& transport, WorkspaceView& workspaceView) {
  constexpr int Bf16PerInt4 = sizeof(int4) / sizeof(Bf16);
  constexpr int HiddenInt4 = Hidden / Bf16PerInt4;
  const int threadId = static_cast<int>(threadIdx.x);
  const int laneId = get_lane_id();
  const int nLocalExperts = nExperts / nRanks;

  for (int tokenIdx = static_cast<int>(blockIdx.x); tokenIdx < nTokens; tokenIdx += static_cast<int>(gridDim.x)) {
    const int globalExpertIdx = laneId < nTopk ? static_cast<int>(topkIndices[tokenIdx * nTopk + laneId]) : -1;
    const int destinationRank = globalExpertIdx >= 0 ? globalExpertIdx / nLocalExperts : -1;
    const bool firstLaneForRank = isFirstLaneForRank(destinationRank, laneId);
    const int partialRank = destinationRank >= 0 && firstLaneForRank ? destinationRank : -1;
    const int partialSlot = partialRank >= 0 ? workspaceView.rankMajorSendIndices_[tokenIdx * nTopk + laneId] : -1;

    for (int hiddenIdx = threadId; hiddenIdx < HiddenInt4; hiddenIdx += CombineNThreads) {
      const int4 packed = reduceRemoteRankPartialsBf16x8<HiddenInt4>(expertOutput, transport, partialRank, partialSlot,
                                                                     nTopk, maxTokensPerRank, tokenIdx, hiddenIdx);
      auto* outputRow = reinterpret_cast<int4*>(output) + static_cast<size_t>(tokenIdx) * HiddenInt4;
      outputRow[hiddenIdx] = packed;
    }
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

template <low_latency::CombineMode Mode, int Hidden, DispatchDataType DispatchType, int ScaleBlockSize,
          DispatchLayout Layout>
__global__ __launch_bounds__(CombineNThreads, 1) void combineKernel(
    void* output, const void* expertOutput, const int64_t* __restrict__ topkIndices,
    const float* __restrict__ topkWeights, const int* srcInfo, const int64_t* layoutRange, Workload workload,
    void* combineRecvBuffer, const void* dispatchRecvBuffer, CommContext comm, void* workspace) {
#if MSCCLPP_BULK_AVAILABLE
  extern __shared__ __align__(128) uint8_t sharedMemory[];
  const int nTokens = workload.numTokens_;
  const int nExperts = workload.numExperts_;
  const int nRanks = comm.numRanks_;
  const int nTopk = workload.numTopk_;
  const int maxTokensPerRank = workload.maxTokensPerRank_;
  const TransportView transport(comm);
  WorkspaceView workspaceView(workspace, nRanks, nExperts);

  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    static_assert(Mode == low_latency::CombineMode::RANK_LOCAL_REDUCE);
    static_assert(DispatchType == DispatchDataType::BF16);
#if defined(MSCCLPP_USE_GPUNETIO)
    // Cross-domain rank-major combine: reduce remote expert outputs by RDMA-read
    // (gin->get) rather than the NVLink pull. The NVLink barrier still orders
    // intra-domain producers; cross-domain ordering relies on the dispatch-epoch
    // handshake and is pending hardware validation.
    if (transport.gpuNetIo_ != nullptr) {
      synchronizeRankMajorCombine(transport, nRanks, workload.epoch_, workspaceView);
      recvRankMajorRemotePartialsGpuNetIo<Hidden>(output, expertOutput, topkIndices, nTokens, nTopk, nExperts, nRanks,
                                                  maxTokensPerRank, transport, workspaceView);
      return;
    }
#endif  // defined(MSCCLPP_USE_GPUNETIO)
    if (nTopk <= RankMajorTmaMaxNTopk) {
      const uint32_t epoch = workload.epoch_;
      if (blockIdx.x == 0) {
        publishRankMajorCombineReady(transport, nRanks, epoch, workspaceView);
      } else {
        recvRankMajorRemotePartialsTma<Hidden>(output, expertOutput, topkIndices, nTokens, nTopk, nExperts, nRanks,
                                               maxTokensPerRank, epoch, transport, workspaceView, sharedMemory);
      }
      return;
    }
    synchronizeRankMajorCombine(transport, nRanks, workload.epoch_, workspaceView);
    recvRankMajorRemotePartials<Hidden>(output, expertOutput, topkIndices, nTokens, nTopk, nExperts, nRanks,
                                        maxTokensPerRank, transport, workspaceView);
    return;
  } else if constexpr (Mode == low_latency::CombineMode::RANK_LOCAL_REDUCE) {
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

  if constexpr (Mode == low_latency::CombineMode::RANK_LOCAL_REDUCE) {
    recvRankLocalPartials<Hidden>(output, topkIndices, nTokens, nTopk, nExperts, nRanks, maxTokensPerRank,
                                  combineRecvBuffer, sharedMemory);
  } else {
    recvExpertRowsDirect<Hidden>(output, topkIndices, topkWeights, nTokens, nTopk, maxTokensPerRank, combineRecvBuffer);
  }
#endif  // MSCCLPP_BULK_AVAILABLE
}

template <low_latency::CombineMode Mode, int Hidden, DispatchDataType DispatchType, int ScaleBlockSize,
          DispatchLayout Layout>
inline void combineHiddenMode(void* output, const void* expertOutput, const int64_t* topkIndices,
                              const float* topkWeights, const int* srcInfo, const int64_t* layoutRange,
                              const low_latency::Workload& workload, void* recvBuffer, void* dispatchRecvBuffer,
                              const low_latency::CommContext& comm, void* workspace, int numBlocks,
                              cudaStream_t stream) {
  static_assert(Hidden == 2048 || Hidden == 4096 || Hidden == 4352 || Hidden == 6656 || Hidden == 7168 ||
                Hidden == 8192 || Hidden == 8704 || Hidden == 9216);
  const int nExperts = workload.numExperts_;
  const int nRanks = comm.numRanks_;
  const int nLocalExperts = nExperts / nRanks;
  if constexpr (Mode == low_latency::CombineMode::DIRECT_SEND) {
    static_assert(Layout == DispatchLayout::EXPERT_MAJOR);
    EP_HOST_ASSERT(directSendWorkerCount<Hidden>(nLocalExperts) > 0);
  }

  auto combineFunc = combineKernel<Mode, Hidden, DispatchType, ScaleBlockSize, Layout>;
  const size_t sharedBytes = combineSharedBytes<Hidden, Mode, Layout>(nLocalExperts, workload.numTopk_);
  const bool useRankMajorTma = Layout == DispatchLayout::RANK_MAJOR &&
                               Mode == low_latency::CombineMode::RANK_LOCAL_REDUCE &&
                               workload.numTopk_ <= RankMajorTmaMaxNTopk;
  const int launchBlocks = numBlocks + (useRankMajorTma ? 1 : 0);
  static thread_local KernelConfigCache kernelConfig;
  const int residentBlocks = configureKernel(combineFunc, CombineNThreads, sharedBytes, comm, kernelConfig);
  EP_HOST_ASSERT(residentBlocks >= launchBlocks);

  combineKernel<Mode, Hidden, DispatchType, ScaleBlockSize, Layout>
      <<<dim3(launchBlocks), dim3(CombineNThreads), sharedBytes, stream>>>(
          output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload, recvBuffer,
          dispatchRecvBuffer, comm, workspace);
  CUDA_CHECK(cudaGetLastError());
}

template <int Hidden>
inline void combineHidden(void* output, const void* expertOutput, const int64_t* topkIndices, const float* topkWeights,
                          const int* srcInfo, const int64_t* layoutRange, const low_latency::Workload& workload,
                          void* recvBuffer, void* dispatchRecvBuffer, const low_latency::CommContext& comm,
                          void* workspace, int numBlocks, low_latency::CombineMode mode, cudaStream_t stream) {
  if (workload.outputLayout_ == DispatchLayout::RANK_MAJOR) {
    EP_HOST_ASSERT(workload.dispatchDataType_ == DispatchDataType::BF16);
    EP_HOST_ASSERT(mode == low_latency::CombineMode::RANK_LOCAL_REDUCE);
    return combineHiddenMode<low_latency::CombineMode::RANK_LOCAL_REDUCE, Hidden, DispatchDataType::BF16, 0,
                             DispatchLayout::RANK_MAJOR>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                         layoutRange, workload, recvBuffer, dispatchRecvBuffer, comm,
                                                         workspace, numBlocks, stream);
  }
  if (mode == low_latency::CombineMode::RANK_LOCAL_REDUCE) {
    switch (workload.dispatchDataType_) {
      case DispatchDataType::BF16:
        return combineHiddenMode<low_latency::CombineMode::RANK_LOCAL_REDUCE, Hidden, DispatchDataType::BF16, 0,
                                 DispatchLayout::EXPERT_MAJOR>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                               layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                               comm, workspace, numBlocks, stream);
      case DispatchDataType::FP8_E4M3:
        return combineHiddenMode<low_latency::CombineMode::RANK_LOCAL_REDUCE, Hidden, DispatchDataType::FP8_E4M3, 128,
                                 DispatchLayout::EXPERT_MAJOR>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                               layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                               comm, workspace, numBlocks, stream);
    }
  }
  switch (workload.dispatchDataType_) {
    case DispatchDataType::BF16:
      return combineHiddenMode<low_latency::CombineMode::DIRECT_SEND, Hidden, DispatchDataType::BF16, 0,
                               DispatchLayout::EXPERT_MAJOR>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                             layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                             comm, workspace, numBlocks, stream);
    case DispatchDataType::FP8_E4M3:
      return combineHiddenMode<low_latency::CombineMode::DIRECT_SEND, Hidden, DispatchDataType::FP8_E4M3, 128,
                               DispatchLayout::EXPERT_MAJOR>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                             layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                             comm, workspace, numBlocks, stream);
  }
  EP_HOST_ASSERT(false && "unsupported dispatch data type");
}

inline void combine(void* output, const void* expertOutput, const int64_t* topkIndices, const float* topkWeights,
                    const int* srcInfo, const int64_t* layoutRange, const low_latency::Workload& workload,
                    void* recvBuffer, void* dispatchRecvBuffer, const low_latency::CommContext& comm, void* workspace,
                    int numBlocks, low_latency::CombineMode mode, cudaStream_t stream) {
  const int nExperts = workload.numExperts_;
  const int rank = comm.rank_;
  const int nRanks = comm.numRanks_;

  EP_HOST_ASSERT(workload.numTokens_ == 0 || output != nullptr);
  EP_HOST_ASSERT(expertOutput != nullptr);
  EP_HOST_ASSERT(workload.numTokens_ == 0 || topkIndices != nullptr);
  EP_HOST_ASSERT(recvBuffer != nullptr);
  EP_HOST_ASSERT(dispatchRecvBuffer != nullptr);
  EP_HOST_ASSERT(comm.symmetricBufferBase_ != nullptr);
  EP_HOST_ASSERT(comm.peerMappedBufferBases_ != nullptr);
  EP_HOST_ASSERT(comm.baseMemoryChannels_ != nullptr);
  EP_HOST_ASSERT(workspace != nullptr);
  EP_HOST_ASSERT(nRanks > 0 && nRanks <= 2 * WARP_SIZE);
  EP_HOST_ASSERT(nExperts > 0 && nExperts % nRanks == 0);
  EP_HOST_ASSERT(rank >= 0 && rank < nRanks);
  EP_HOST_ASSERT(workload.numTokens_ >= 0 && workload.numTokens_ <= workload.maxTokensPerRank_);
  EP_HOST_ASSERT(workload.numTopk_ > 0 && workload.numTopk_ <= CombineMaxNTopk);
  EP_HOST_ASSERT(numBlocks > 0 && numBlocks <= low_latency::MaxWorkerBlocks);
  EP_HOST_ASSERT(mode == low_latency::CombineMode::RANK_LOCAL_REDUCE || mode == low_latency::CombineMode::DIRECT_SEND);
  EP_HOST_ASSERT(workload.outputLayout_ == DispatchLayout::EXPERT_MAJOR ||
                 workload.outputLayout_ == DispatchLayout::RANK_MAJOR);
  EP_HOST_ASSERT(isSupportedDispatchDataType(workload.dispatchDataType_));
  if (workload.outputLayout_ == DispatchLayout::RANK_MAJOR) {
    EP_HOST_ASSERT(mode == low_latency::CombineMode::RANK_LOCAL_REDUCE);
    EP_HOST_ASSERT(workload.dispatchDataType_ == DispatchDataType::BF16);
  } else if (mode == low_latency::CombineMode::DIRECT_SEND) {
    EP_HOST_ASSERT(srcInfo != nullptr);
    EP_HOST_ASSERT(layoutRange != nullptr);
  }

  switch (workload.hidden_) {
    case 4096:
      return combineHidden<4096>(output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload,
                                 recvBuffer, dispatchRecvBuffer, comm, workspace, numBlocks, mode, stream);
    case 4352:
      return combineHidden<4352>(output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload,
                                 recvBuffer, dispatchRecvBuffer, comm, workspace, numBlocks, mode, stream);
    case 6656:
      return combineHidden<6656>(output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload,
                                 recvBuffer, dispatchRecvBuffer, comm, workspace, numBlocks, mode, stream);
    case 7168:
      return combineHidden<7168>(output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload,
                                 recvBuffer, dispatchRecvBuffer, comm, workspace, numBlocks, mode, stream);
    case 8192:
      return combineHidden<8192>(output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload,
                                 recvBuffer, dispatchRecvBuffer, comm, workspace, numBlocks, mode, stream);
    case 8704:
      return combineHidden<8704>(output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload,
                                 recvBuffer, dispatchRecvBuffer, comm, workspace, numBlocks, mode, stream);
    case 9216:
      return combineHidden<9216>(output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload,
                                 recvBuffer, dispatchRecvBuffer, comm, workspace, numBlocks, mode, stream);
    default:
      EP_HOST_ASSERT(false && "unsupported optimized low-latency hidden size");
  }
}

}  // namespace detail

void combine(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights, const int* srcInfo,
             const int64_t* layoutRange, const Workload& workload, void* recvBuffer, void* dispatchRecvBuffer,
             const CommContext& comm, void* workspace, int numBlocks, CombineMode mode, cudaStream_t stream) {
  detail::combine(output, input, topkIdx, topkWeights, srcInfo, layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                  comm, workspace, numBlocks, mode, stream);
}

}  // namespace low_latency
}  // namespace ep
}  // namespace mscclpp
