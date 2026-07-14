// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <mscclpp/gpu_data_types.hpp>

#include "api.cuh"
#include "config.cuh"
#include "device_helpers.cuh"
#include "exception.cuh"

namespace mscclpp {
namespace ep {
namespace low_latency {
namespace detail {

constexpr int CombineNWarps = 32;
constexpr int CombineNThreads = CombineNWarps * WARP_SIZE;
constexpr int CombineNStages = 8;
constexpr int DirectSendMaxNWorkers = WARP_SIZE;
constexpr int CombineMaxNTopk = 9;

MSCCLPP_DEVICE_INLINE RecvTask loadRecvTask(const RecvTask* tasks, int taskIdx) {
  const int laneId = get_lane_id();
  RecvTask task{};
  if (laneId == 0) task = tasks[taskIdx];
  task.sourceRank_ = warpBroadcast(task.sourceRank_, 0);
  task.tokenBegin_ = warpBroadcast(task.tokenBegin_, 0);
  task.tokenEnd_ = warpBroadcast(task.tokenEnd_, 0);
  return task;
}

MSCCLPP_HOST_DEVICE_INLINE size_t directSendControlBytes(int nLocalExperts) {
  return configAlign<size_t>(static_cast<size_t>(nLocalExperts + 1) * sizeof(int), 128);
}

template <int Hidden>
MSCCLPP_HOST_DEVICE_INLINE constexpr size_t directSendWorkerBytes() {
  return static_cast<size_t>(Hidden) * sizeof(Bf16) + sizeof(uint64_t);
}

template <int Hidden>
MSCCLPP_HOST_DEVICE_INLINE int directSendWorkerCount(int nLocalExperts) {
  const size_t controlBytes = directSendControlBytes(nLocalExperts);
  if (controlBytes >= OptimizedDynamicSharedMemoryBytes) return 0;
  const int availableWorkers =
      static_cast<int>((OptimizedDynamicSharedMemoryBytes - controlBytes) / directSendWorkerBytes<Hidden>());
  return availableWorkers < DirectSendMaxNWorkers ? availableWorkers : DirectSendMaxNWorkers;
}

template <int Hidden, low_latency::CombineMode Mode>
MSCCLPP_HOST_DEVICE_INLINE size_t combineSharedBytes(int nLocalExperts) {
  constexpr size_t TileBytes = static_cast<size_t>(Hidden) * sizeof(Bf16);
  if constexpr (Mode == low_latency::CombineMode::DIRECT_SEND) {
    return directSendControlBytes(nLocalExperts) +
           static_cast<size_t>(directSendWorkerCount<Hidden>(nLocalExperts)) * directSendWorkerBytes<Hidden>();
  }
  return CombineNStages * TileBytes;
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
        waitBulkGroupRead<CombineNStages - 1>();
      }
      if (tokenIteration >= CombineNStages) __syncthreads();
#pragma unroll
      for (int chunkIdx = 0; chunkIdx < ChunksPerThread; ++chunkIdx) {
        const int hiddenIdx = threadId + chunkIdx * CombineNThreads;
        if (hiddenIdx < HiddenInt4) outputTile[hiddenIdx] = reduced[chunkIdx];
      }
      __syncthreads();

      if (threadId == 0) {
        fenceProxyAsyncSharedCta();
        const int sourceTokenIdx = *payloadView.srcTokenGlobalIdx(sourcePayload) - sourceRank * maxTokensPerRank;
        EP_DEVICE_ASSERT(sourceTokenIdx >= 0 && sourceTokenIdx < maxTokensPerRank);
        void* destinationBuffer = transport.mappedBuffer(combineRecvBuffer, sourceRank);
        auto* destinationRow = reinterpret_cast<uint8_t*>(destinationBuffer) +
                               (static_cast<size_t>(transport.rank_) * maxTokensPerRank + sourceTokenIdx) * HiddenBytes;
        issueTmaStore(destinationRow, outputTile, static_cast<uint32_t>(HiddenBytes));
      }
    }
  }

  if (tokenIteration > 0 && threadId == 0) waitBulkGroup();
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
  auto* tmaBarriers = reinterpret_cast<uint64_t*>(outputTiles + static_cast<size_t>(nWorkers) * HiddenBytes);
  if (laneId < nWorkers) {
    auto* outputTile = outputTiles + static_cast<size_t>(laneId) * HiddenBytes;
    auto* tmaBarrier = tmaBarriers + laneId;
    uint32_t tmaPhase = 0;
    if (blockRowBegin + laneId < blockRowEnd) initTmaLoadBarrier(tmaBarrier);

    bool hasPendingStore = false;
    for (int flatRowIdx = blockRowBegin + laneId; flatRowIdx < blockRowEnd; flatRowIdx += nWorkers) {
      if (hasPendingStore) waitBulkGroupRead();
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
      issueTmaLoad(inputRow, outputTile, tmaBarrier, static_cast<uint32_t>(HiddenBytes));
      waitTmaLoad(tmaBarrier, tmaPhase);
      fenceProxyAsyncSharedCta();
      const int globalExpertIdx = transport.rank_ * nLocalExperts + localExpertIdx;
      void* destinationBuffer = transport.mappedBuffer(combineRecvBuffer, sourceRank);
      auto* destinationRow = reinterpret_cast<uint8_t*>(destinationBuffer) +
                             (static_cast<size_t>(globalExpertIdx) * maxTokensPerRank + sourceTokenIdx) * HiddenBytes;
      issueTmaStore(destinationRow, outputTile, static_cast<uint32_t>(HiddenBytes));
      hasPendingStore = true;
    }

    if (hasPendingStore) waitBulkGroup();
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
      waitBulkGroupRead<CombineNStages - 1>();
    }
    if (tokenIteration >= CombineNStages) __syncthreads();
#pragma unroll
    for (int chunkIdx = 0; chunkIdx < ChunksPerThread; ++chunkIdx) {
      const int hiddenIdx = threadId + chunkIdx * CombineNThreads;
      if (hiddenIdx < HiddenInt4) outputTile[hiddenIdx] = reduced[chunkIdx];
    }
    __syncthreads();

    if (threadId == 0) {
      fenceProxyAsyncSharedCta();
      auto* outputRow = reinterpret_cast<uint8_t*>(output) + static_cast<size_t>(tokenIdx) * HiddenBytes;
      issueTmaStore(outputRow, outputTile, static_cast<uint32_t>(HiddenBytes));
    }
  }
  if (tokenIteration > 0 && threadId == 0) waitBulkGroup();
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

template <low_latency::CombineMode Mode, int Hidden, DispatchDataType DispatchType, int ScaleBlockSize>
__global__ __launch_bounds__(CombineNThreads, 1) void combineKernel(
    void* output, const void* expertOutput, const int64_t* __restrict__ topkIndices,
    const float* __restrict__ topkWeights, const int* srcInfo, const int64_t* layoutRange, Workload workload,
    void* combineRecvBuffer, const void* dispatchRecvBuffer, CommContext comm, void* workspace) {
  extern __shared__ __align__(128) uint8_t sharedMemory[];
  const int nTokens = workload.numTokens_;
  const int nExperts = workload.numExperts_;
  const int nRanks = comm.numRanks_;
  const int nTopk = workload.numTopk_;
  const int maxTokensPerRank = workload.maxTokensPerRank_;
  const TransportView transport(comm);
  WorkspaceView workspaceView(workspace, nRanks, nExperts);

  if constexpr (Mode == low_latency::CombineMode::RANK_LOCAL_REDUCE) {
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
}

template <low_latency::CombineMode Mode, int Hidden, DispatchDataType DispatchType, int ScaleBlockSize>
inline void combineHiddenMode(void* output, const void* expertOutput, const int64_t* topkIndices,
                              const float* topkWeights, const int* srcInfo, const int64_t* layoutRange,
                              const low_latency::Workload& workload, void* recvBuffer, void* dispatchRecvBuffer,
                              const low_latency::CommContext& comm, void* workspace, int numBlocks,
                              cudaStream_t stream) {
  static_assert(Hidden == 4096 || Hidden == 7168 || Hidden == 8192 || Hidden == 9216);
  const int nExperts = workload.numExperts_;
  const int nRanks = comm.numRanks_;
  const int nLocalExperts = nExperts / nRanks;
  if constexpr (Mode == low_latency::CombineMode::DIRECT_SEND) {
    EP_HOST_ASSERT(directSendWorkerCount<Hidden>(nLocalExperts) > 0);
  }

  auto combineFunc = combineKernel<Mode, Hidden, DispatchType, ScaleBlockSize>;
  const size_t sharedBytes = combineSharedBytes<Hidden, Mode>(nLocalExperts);
  static thread_local KernelConfigCache kernelConfig;
  const int residentBlocks = configureKernel(combineFunc, CombineNThreads, sharedBytes, comm, kernelConfig);
  EP_HOST_ASSERT(residentBlocks >= numBlocks);

  combineKernel<Mode, Hidden, DispatchType, ScaleBlockSize>
      <<<dim3(numBlocks), dim3(CombineNThreads), sharedBytes, stream>>>(output, expertOutput, topkIndices, topkWeights,
                                                                        srcInfo, layoutRange, workload, recvBuffer,
                                                                        dispatchRecvBuffer, comm, workspace);
  CUDA_CHECK(cudaGetLastError());
}

template <int Hidden>
inline void combineHidden(void* output, const void* expertOutput, const int64_t* topkIndices, const float* topkWeights,
                          const int* srcInfo, const int64_t* layoutRange, const low_latency::Workload& workload,
                          void* recvBuffer, void* dispatchRecvBuffer, const low_latency::CommContext& comm,
                          void* workspace, int numBlocks, low_latency::CombineMode mode, cudaStream_t stream) {
  if (mode == low_latency::CombineMode::RANK_LOCAL_REDUCE) {
    switch (workload.dispatchDataType_) {
      case DispatchDataType::BF16:
        return combineHiddenMode<low_latency::CombineMode::RANK_LOCAL_REDUCE, Hidden, DispatchDataType::BF16, 0>(
            output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload, recvBuffer,
            dispatchRecvBuffer, comm, workspace, numBlocks, stream);
      case DispatchDataType::FP8_E4M3:
        return combineHiddenMode<low_latency::CombineMode::RANK_LOCAL_REDUCE, Hidden, DispatchDataType::FP8_E4M3, 128>(
            output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload, recvBuffer,
            dispatchRecvBuffer, comm, workspace, numBlocks, stream);
      case DispatchDataType::MXFP8_E4M3:
        EP_HOST_ASSERT(false && "MXFP8 dispatch metadata is not implemented");
    }
  }
  switch (workload.dispatchDataType_) {
    case DispatchDataType::BF16:
      return combineHiddenMode<low_latency::CombineMode::DIRECT_SEND, Hidden, DispatchDataType::BF16, 0>(
          output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload, recvBuffer,
          dispatchRecvBuffer, comm, workspace, numBlocks, stream);
    case DispatchDataType::FP8_E4M3:
      return combineHiddenMode<low_latency::CombineMode::DIRECT_SEND, Hidden, DispatchDataType::FP8_E4M3, 128>(
          output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload, recvBuffer,
          dispatchRecvBuffer, comm, workspace, numBlocks, stream);
    case DispatchDataType::MXFP8_E4M3:
      EP_HOST_ASSERT(false && "MXFP8 dispatch metadata is not implemented");
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

  EP_HOST_ASSERT(output != nullptr);
  EP_HOST_ASSERT(expertOutput != nullptr);
  EP_HOST_ASSERT(topkIndices != nullptr);
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
  EP_HOST_ASSERT(isSupportedDispatchDataType(workload.dispatchDataType_));
  if (mode == low_latency::CombineMode::DIRECT_SEND) {
    EP_HOST_ASSERT(srcInfo != nullptr);
    EP_HOST_ASSERT(layoutRange != nullptr);
  }

  switch (workload.hidden_) {
    case 4096:
      return combineHidden<4096>(output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload,
                                 recvBuffer, dispatchRecvBuffer, comm, workspace, numBlocks, mode, stream);
    case 7168:
      return combineHidden<7168>(output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload,
                                 recvBuffer, dispatchRecvBuffer, comm, workspace, numBlocks, mode, stream);
    case 8192:
      return combineHidden<8192>(output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload,
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
