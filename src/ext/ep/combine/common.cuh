// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_COMBINE_COMMON_CUH_
#define MSCCLPP_EP_COMBINE_COMMON_CUH_

#include <mscclpp/bulk_device.hpp>
#include <mscclpp/gpu_data_types.hpp>

#include "common/device_helpers.cuh"
#include "common/latency.cuh"
#include "common/quantization.cuh"
#include "exception.hpp"
#include "kernels.hpp"

namespace mscclpp {
namespace ep {

constexpr int CombineNWarps = 32;
constexpr int CombineNThreads = CombineNWarps * WARP_SIZE;
constexpr int CombineNStages = 8;
constexpr int DirectSendMaxNWorkers = WARP_SIZE;
constexpr int HybridCombineNWarpGroups = 2;
constexpr int HybridCombineGroupWarps = CombineNWarps / HybridCombineNWarpGroups;
constexpr int HybridCombineGroupThreads = HybridCombineGroupWarps * WARP_SIZE;
constexpr int HybridCombineGroupBarrierBase = 1;
static_assert(CombineNWarps % HybridCombineNWarpGroups == 0);

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

// Rank-major combine shared-memory control state, placed after the NSlots staged expert-output tiles.
template <int NSlots>
struct RankMajorCombineSharedControl {
  mscclpp::BulkBarrier barriers[NSlots];
  int validRoutes[NSlots];
  float routeWeights[NSlots];
};

// Hybrid combine per-warp-group shared-memory control state, placed after each group's compacted stage tiles.
template <int NSlots>
struct alignas(16) HybridCombineGroupControl {
  mscclpp::BulkBarrier barrier;
  float routeWeights[NSlots];
  uint64_t finalizedMask;
  int nRoundRoutes;
  int nTotalRounds;
  bool remoteReady;
};

template <int Hidden, DispatchDataType DispatchType, int NSlots>
MSCCLPP_HOST_DEVICE_INLINE constexpr size_t rankMajorCombineSharedBytesForSlots() {
  constexpr size_t RowsBytes = static_cast<size_t>(NSlots) * Hidden * sizeof(DispatchElementType<DispatchType>);
  return configAlign<size_t>(RowsBytes + sizeof(RankMajorCombineSharedControl<NSlots>), 128);
}

template <int Hidden, DispatchDataType DispatchType, int NSlots>
MSCCLPP_HOST_DEVICE_INLINE constexpr size_t hybridCombineGroupSharedBytesForSlots() {
  constexpr size_t TokensBytes = static_cast<size_t>(NSlots) * Hidden * sizeof(DispatchElementType<DispatchType>);
  return configAlign<size_t>(TokensBytes + sizeof(HybridCombineGroupControl<NSlots>), BufferAlignmentBytes);
}

template <int Hidden, DispatchDataType DispatchType, int MaxNTopk>
MSCCLPP_HOST_DEVICE_INLINE constexpr int hybridCombineGroupStageSlots() {
  static_assert(MaxNTopk == 8 || MaxNTopk == MaxNumTopk);
  constexpr size_t MaxGroupBytes = OptimizedDynamicSharedMemoryBytes / HybridCombineNWarpGroups;
  int nSlots = MaxNTopk;
  while (nSlots > 1 &&
         configAlign<size_t>(static_cast<size_t>(nSlots) * Hidden * sizeof(DispatchElementType<DispatchType>) +
                                 sizeof(HybridCombineGroupControl<MaxNTopk>),
                             BufferAlignmentBytes) > MaxGroupBytes) {
    --nSlots;
  }
  return nSlots;
}

template <int Hidden, DispatchDataType DispatchType, int MaxNTopk>
MSCCLPP_HOST_DEVICE_INLINE constexpr int rankMajorCombineStageSlots() {
  static_assert(MaxNTopk == 8 || MaxNTopk == MaxNumTopk);
  if constexpr (rankMajorCombineSharedBytesForSlots<Hidden, DispatchType, MaxNTopk>() <=
                OptimizedDynamicSharedMemoryBytes) {
    return MaxNTopk;
  }
  static_assert(rankMajorCombineSharedBytesForSlots<Hidden, DispatchType, 8>() <= OptimizedDynamicSharedMemoryBytes);
  return 8;
}

template <int Hidden, CombineMode Mode, DispatchDataType DispatchType, DispatchLayout Layout, int MaxNTopk,
          bool Hybrid = false>
MSCCLPP_HOST_DEVICE_INLINE size_t combineSharedBytes(int nLocalExperts) {
  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    if constexpr (Hybrid) {
      constexpr int StageSlots = hybridCombineGroupStageSlots<Hidden, DispatchType, MaxNTopk>();
      constexpr size_t SharedBytes =
          HybridCombineNWarpGroups * hybridCombineGroupSharedBytesForSlots<Hidden, DispatchType, StageSlots>();
      static_assert(SharedBytes <= OptimizedDynamicSharedMemoryBytes);
      return SharedBytes;
    }
    constexpr int StageSlots = rankMajorCombineStageSlots<Hidden, DispatchType, MaxNTopk>();
    constexpr size_t SharedBytes = rankMajorCombineSharedBytesForSlots<Hidden, DispatchType, StageSlots>();
    static_assert(SharedBytes <= OptimizedDynamicSharedMemoryBytes);
    return SharedBytes;
  }
  constexpr size_t TileBytes = static_cast<size_t>(Hidden) * sizeof(Bf16);
  if constexpr (Mode == CombineMode::DIRECT_SEND) {
    return directSendControlBytes(nLocalExperts) +
           static_cast<size_t>(directSendWorkerCount<Hidden>(nLocalExperts)) * directSendWorkerBytes<Hidden>();
  }
  return CombineNStages * TileBytes;
}

#if MSCCLPP_BULK_AVAILABLE

MSCCLPP_DEVICE_INLINE void exchangeCombineReady(const TransportView& transport, int nRanks) {
  const int threadId = static_cast<int>(threadIdx.x);
  if (blockIdx.x == 0 && threadId < nRanks) {
    const int peerRank = threadId;
    if (transport.isSelf(peerRank)) return;
    transport.baseMemoryChannels_[peerRank].signal();
    transport.baseMemoryChannels_[peerRank].wait(-1);
  }
}

MSCCLPP_DEVICE_INLINE void publishRankMajorCombineReady(const TransportView& transport, int nRanks, uint32_t flag,
                                                        WorkspaceView& workspaceView) {
  if (blockIdx.x != 0) return;
  const int threadId = static_cast<int>(threadIdx.x);
  if (threadId < nRanks) {
    if (!transport.isSelf(threadId)) {
      transport.baseMemoryChannels_[threadId].relaxedSignal();
      transport.baseMemoryChannels_[threadId].relaxedWait(-1);
    }
    mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(workspaceView.combineRankReadyFlags_ + threadId, flag,
                                                         mscclpp::memoryOrderRelaxed);
  }
}

template <CombineMode Mode, int Hidden, DispatchDataType DispatchType, int StageSlots>
MSCCLPP_DEVICE_INLINE void accumulateRankMajorVector(mscclpp::f32x8& sum, int vectorIdx, const uint8_t* sharedTokens,
                                                     const RankMajorCombineSharedControl<StageSlots>& sharedControl,
                                                     float dequantScale) {
  constexpr bool IsDirectSend = Mode == CombineMode::DIRECT_SEND;
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(DispatchElementType<DispatchType>);
  constexpr int NPairsPerVector = mscclpp::f32x8::Size / mscclpp::f32x2::Size;
  auto* sumPairs = reinterpret_cast<mscclpp::f32x2*>(&sum);
#pragma unroll
  for (int stageIdx = 0; stageIdx < StageSlots; ++stageIdx) {
    if (!sharedControl.validRoutes[stageIdx]) continue;
    const float routeWeight = IsDirectSend ? sharedControl.routeWeights[stageIdx] : 1.0f;
    const auto* routeToken = sharedTokens + static_cast<size_t>(stageIdx) * HiddenBytes;
    if constexpr (DispatchType == DispatchDataType::BF16) {
      const auto values = mscclpp::to<mscclpp::f32x8>(reinterpret_cast<const mscclpp::bf16x8*>(routeToken)[vectorIdx]);
      const auto* pairs = reinterpret_cast<const mscclpp::f32x2*>(&values);
#pragma unroll
      for (int pairIdx = 0; pairIdx < NPairsPerVector; ++pairIdx) {
        if constexpr (IsDirectSend) {
          sumPairs[pairIdx] =
              mscclpp::fma(pairs[pairIdx], mscclpp::f32x2(make_float2(routeWeight, routeWeight)), sumPairs[pairIdx]);
        } else {
          sumPairs[pairIdx] = sumPairs[pairIdx] + pairs[pairIdx];
        }
      }
    } else {
      const auto values = reinterpret_cast<const mscclpp::f8_e4m3x8*>(routeToken)[vectorIdx];
      const float scale = IsDirectSend ? dequantScale * routeWeight : dequantScale;
      if (scale != 0.0f) {
        dequantizeAndAccumulateFp8E4M3(values, scale, sumPairs);
      }
    }
  }
}

// Worker blocks (blockIdx.x > 0): TMA-load rank-major expert outputs from peers and reduce into output.
template <int Hidden, CombineMode Mode, DispatchDataType DispatchType, int MaxNTopk>
MSCCLPP_DEVICE_INLINE void combineRankMajorTokens(void* output, const void* expertOutput,
                                                  const int64_t* __restrict__ topkIndices,
                                                  const float* __restrict__ topkWeights, const Workload& workload,
                                                  const TransportView& transport, WorkspaceView& workspace,
                                                  uint32_t flag, uint8_t* sharedMemory) {
#if defined(__CUDA_ARCH__)
  static_assert(__CUDA_ARCH__ >= 900, "TMA rank-major combine requires SM90 or newer");
#endif
  static_assert(DispatchType == DispatchDataType::BF16 || DispatchType == DispatchDataType::FP8_E4M3_STATIC);
  static_assert(Hidden % mscclpp::bf16x8::Size == 0);
  using ElementType = DispatchElementType<DispatchType>;
  constexpr bool IsDirectSend = Mode == CombineMode::DIRECT_SEND;
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(ElementType);
  constexpr int NHiddenVectors = Hidden / mscclpp::f32x8::Size;
  constexpr int MaxVectorsPerThread = configCellDiv(NHiddenVectors, CombineNThreads);
  constexpr int NPairsPerVector = mscclpp::f32x8::Size / mscclpp::f32x2::Size;
  constexpr int StageSlots = rankMajorCombineStageSlots<Hidden, DispatchType, MaxNTopk>();
  constexpr int NumRounds = configCellDiv(MaxNTopk, StageSlots);
  constexpr size_t TokensBytes = static_cast<size_t>(StageSlots) * HiddenBytes;

  const int nTokens = workload.numTokens_;
  const int nWorkers = static_cast<int>(gridDim.x) - 1;
  const int firstTokenIdx = static_cast<int>(blockIdx.x) - 1;
  if (firstTokenIdx >= nTokens) return;

  const int warpIdx = static_cast<int>(threadIdx.x) / WARP_SIZE;
  const int laneIdx = getLaneId();
  const int nTopk = workload.numTopk_;
  const int nExperts = workload.numExperts_;
  const int nLocalExperts = nExperts / transport.numRanks_;
  const int maxTokensPerRank = workload.maxTokensPerRank_;
  const bool useWeights = workload.applyRouterWeights_ && topkWeights != nullptr;
  const float dequantScale = DispatchType == DispatchDataType::FP8_E4M3_STATIC ? 1.0f / workload.combineScale_ : 1.0f;
  auto& sharedControl = *reinterpret_cast<RankMajorCombineSharedControl<StageSlots>*>(sharedMemory + TokensBytes);
  auto* barriers = sharedControl.barriers;

  // Reuse each mbarrier across tokens and staging rounds by advancing its phase parity.
  uint32_t phase = 0;
  if (warpIdx == 0) {
    if (laneIdx < StageSlots) barriers[laneIdx].init();
    __syncwarp();
  }

  for (int tokenIdx = firstTokenIdx; tokenIdx < nTokens; tokenIdx += nWorkers) {
    int lanePeerIdx = -1;
    uint32_t validRouteMask = 0;
    if (warpIdx == 0) {
      const int64_t rawExpertIdx = laneIdx < nTopk ? topkIndices[tokenIdx * nTopk + laneIdx] : -1;
      const int expertIdx = rawExpertIdx >= 0 && rawExpertIdx < nExperts ? static_cast<int>(rawExpertIdx) : -1;
      lanePeerIdx = expertIdx >= 0 ? expertIdx / nLocalExperts : -1;
      const bool isLeader = isFirstLaneForRank(lanePeerIdx, laneIdx);
      validRouteMask = __ballot_sync(0xffffffff, laneIdx < MaxNTopk && lanePeerIdx >= 0 && (IsDirectSend || isLeader));
    }

    mscclpp::f32x8 tileSum[MaxVectorsPerThread];
#pragma unroll
    for (int stepIdx = 0; stepIdx < MaxVectorsPerThread; ++stepIdx) {
      auto* sumPairs = reinterpret_cast<mscclpp::f32x2*>(&tileSum[stepIdx]);
#pragma unroll
      for (int pairIdx = 0; pairIdx < NPairsPerVector; ++pairIdx) {
        sumPairs[pairIdx].storage = make_float2(0.0f, 0.0f);
      }
    }

#pragma unroll
    for (int roundIdx = 0; roundIdx < NumRounds; ++roundIdx) {
      if (warpIdx == 0) {
        const int routeIdx = roundIdx * StageSlots + laneIdx;
        const bool active = laneIdx < StageSlots && ((validRouteMask >> routeIdx) & 1u) != 0;
        const int peerIdx =
            NumRounds == 1 ? lanePeerIdx : __shfl_sync(0xffffffff, lanePeerIdx, routeIdx < WARP_SIZE ? routeIdx : 0);
        int slotIdx = -1;
        if (active) {
          slotIdx = workspace.rankMajorSendIndices_[tokenIdx * nTopk + routeIdx];
          EP_DEVICE_ASSERT(slotIdx >= 0 && slotIdx < maxTokensPerRank);
        }
        if (laneIdx < StageSlots) {
          sharedControl.validRoutes[laneIdx] = active;
          if constexpr (IsDirectSend) {
            sharedControl.routeWeights[laneIdx] =
                useWeights && routeIdx < nTopk ? topkWeights[tokenIdx * nTopk + routeIdx] : 1.0f;
          }
        }
        __syncwarp();

        bool pending = active;
        while (__any_sync(0xffffffff, pending)) {
          if (pending && mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(workspace.combineRankReadyFlags_ + peerIdx,
                                                                             mscclpp::memoryOrderRelaxed) == flag) {
            const auto* peerOutput =
                static_cast<const uint8_t*>(transport.mappedBuffer(const_cast<void*>(expertOutput), peerIdx));
            const size_t srcTokenIdx = static_cast<size_t>(transport.rank_) * maxTokensPerRank + slotIdx;
            const size_t inputTokenIdx = IsDirectSend ? srcTokenIdx * nTopk + routeIdx : srcTokenIdx;
            barriers[laneIdx].arriveAndExpect(HiddenBytes);
            mscclpp::bulkLoad(sharedMemory + static_cast<size_t>(laneIdx) * HiddenBytes,
                              peerOutput + inputTokenIdx * HiddenBytes, HiddenBytes, barriers[laneIdx]);
            pending = false;
          }
        }
        if (active) barriers[laneIdx].wait(phase);
        __syncwarp();
        if (laneIdx == 0) mscclpp::bulkFence();
      }
      __syncthreads();

#pragma unroll
      for (int stepIdx = 0; stepIdx < MaxVectorsPerThread; ++stepIdx) {
        const int vectorIdx = static_cast<int>(threadIdx.x) + stepIdx * CombineNThreads;
        if (vectorIdx < NHiddenVectors) {
          accumulateRankMajorVector<Mode, Hidden, DispatchType, StageSlots>(tileSum[stepIdx], vectorIdx, sharedMemory,
                                                                            sharedControl, dequantScale);
        }
      }
      __syncthreads();
    }

    auto* tokenOutput = static_cast<mscclpp::bf16x8*>(output) + static_cast<size_t>(tokenIdx) * NHiddenVectors;
#pragma unroll
    for (int stepIdx = 0; stepIdx < MaxVectorsPerThread; ++stepIdx) {
      const int vectorIdx = static_cast<int>(threadIdx.x) + stepIdx * CombineNThreads;
      if (vectorIdx < NHiddenVectors) {
        tokenOutput[vectorIdx] = mscclpp::to<mscclpp::bf16x8>(tileSum[stepIdx]);
      }
    }
  }
  if (warpIdx == 0 && laneIdx < StageSlots) barriers[laneIdx].invalidate();
}

#endif  // MSCCLPP_BULK_AVAILABLE

template <CombineMode Mode, int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout,
          int MaxNTopk, typename KernelSelector, bool Hybrid = false>
inline void combineHiddenMode(void* output, const void* expertOutput, const int64_t* topkIndices,
                              const float* topkWeights, const int* srcInfo, const int64_t* layoutRange,
                              const Workload& workload, void* recvBuffer, void* dispatchRecvBuffer,
                              const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  static_assert(isSupportedHidden(Hidden));
  const int nExperts = workload.numExperts_;
  const int nRanks = context.numRanks_;
  const int nLocalExperts = nExperts / nRanks;
  if constexpr (Mode == CombineMode::DIRECT_SEND && Layout == DispatchLayout::EXPERT_MAJOR) {
    EP_HOST_ASSERT(directSendWorkerCount<Hidden>(nLocalExperts) > 0);
  }

  auto combineFunc = KernelSelector::template get<Hidden, DispatchType, ScaleBlockSize, Layout, MaxNTopk>();
  const size_t sharedBytes = combineSharedBytes<Hidden, Mode, DispatchType, Layout, MaxNTopk, Hybrid>(nLocalExperts);
  EP_HOST_ASSERT(Layout != DispatchLayout::RANK_MAJOR || numBlocks > 1);
  static thread_local KernelConfigCache kernelConfig;
  const int residentBlocks = configureKernel(combineFunc, CombineNThreads, sharedBytes, context, kernelConfig);
  EP_HOST_ASSERT(residentBlocks >= numBlocks);

  combineFunc<<<dim3(numBlocks), dim3(CombineNThreads), sharedBytes, stream>>>(
      output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload, recvBuffer, dispatchRecvBuffer,
      context.devicePtr_);
  MSCCLPP_CUDATHROW(cudaGetLastError());
}

template <CombineMode Mode, int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout,
          typename KernelSelector, bool Hybrid = false>
inline void combineHiddenMode(void* output, const void* expertOutput, const int64_t* topkIndices,
                              const float* topkWeights, const int* srcInfo, const int64_t* layoutRange,
                              const Workload& workload, void* recvBuffer, void* dispatchRecvBuffer,
                              const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  if (workload.numTopk_ <= 8) {
    return combineHiddenMode<Mode, Hidden, DispatchType, ScaleBlockSize, Layout, 8, KernelSelector, Hybrid>(
        output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload, recvBuffer, dispatchRecvBuffer,
        context, numBlocks, stream);
  }
  return combineHiddenMode<Mode, Hidden, DispatchType, ScaleBlockSize, Layout, MaxNumTopk, KernelSelector, Hybrid>(
      output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload, recvBuffer, dispatchRecvBuffer,
      context, numBlocks, stream);
}

template <CombineMode Mode, int Hidden, typename KernelSelector, bool Hybrid = false>
inline void combineHidden(void* output, const void* expertOutput, const int64_t* topkIndices, const float* topkWeights,
                          const int* srcInfo, const int64_t* layoutRange, const Workload& workload, void* recvBuffer,
                          void* dispatchRecvBuffer, const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  if (workload.outputLayout_ == DispatchLayout::RANK_MAJOR) {
    switch (workload.combineDataType_) {
      case DispatchDataType::BF16:
        return combineHiddenMode<Mode, Hidden, DispatchDataType::BF16, 0, DispatchLayout::RANK_MAJOR, KernelSelector,
                                 Hybrid>(output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload,
                                         recvBuffer, dispatchRecvBuffer, context, numBlocks, stream);
      case DispatchDataType::FP8_E4M3:
      case DispatchDataType::MXFP8_E4M3:
        break;
      case DispatchDataType::FP8_E4M3_STATIC:
        return combineHiddenMode<Mode, Hidden, DispatchDataType::FP8_E4M3_STATIC, 0, DispatchLayout::RANK_MAJOR,
                                 KernelSelector, Hybrid>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                         layoutRange, workload, recvBuffer, dispatchRecvBuffer, context,
                                                         numBlocks, stream);
    }
    EP_HOST_ASSERT(false && "unsupported rank-major combine data type");
  }
  if constexpr (!Hybrid) {
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
        case DispatchDataType::MXFP8_E4M3:
        case DispatchDataType::FP8_E4M3_STATIC:
          break;
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
        case DispatchDataType::MXFP8_E4M3:
        case DispatchDataType::FP8_E4M3_STATIC:
          break;
      }
    }
    EP_HOST_ASSERT(false && "unsupported dispatch data type");
  }
}

template <CombineMode Mode, typename KernelSelector, bool Hybrid = false>
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
  EP_HOST_ASSERT(isSupportedRanks(nRanks));
  EP_HOST_ASSERT(nExperts > 0 && nExperts % nRanks == 0);
  EP_HOST_ASSERT(rank >= 0 && rank < nRanks);
  EP_HOST_ASSERT(workload.numTokens_ >= 0 && workload.numTokens_ <= workload.maxTokensPerRank_);
  EP_HOST_ASSERT(workload.numTopk_ > 0 && workload.numTopk_ <= MaxNumTopk);
  const int minCombineBlocks = workload.outputLayout_ == DispatchLayout::RANK_MAJOR ? 2 : 1;
  EP_HOST_ASSERT(numBlocks >= minCombineBlocks && numBlocks <= MaxWorkerBlocks);
  static_assert(Mode == CombineMode::RANK_LOCAL_REDUCE || Mode == CombineMode::DIRECT_SEND);
  EP_HOST_ASSERT(workload.outputLayout_ == DispatchLayout::EXPERT_MAJOR ||
                 workload.outputLayout_ == DispatchLayout::RANK_MAJOR);
  EP_HOST_ASSERT(isSupportedDispatchDataType(workload.dispatchDataType_));
  EP_HOST_ASSERT(workload.outputLayout_ != DispatchLayout::RANK_MAJOR ||
                 isSupportedCombineDataType(workload.combineDataType_));
  if constexpr (Mode == CombineMode::DIRECT_SEND) {
    if (workload.outputLayout_ == DispatchLayout::EXPERT_MAJOR) {
      EP_HOST_ASSERT(srcInfo != nullptr);
      EP_HOST_ASSERT(layoutRange != nullptr);
    }
  }

  switch (workload.hidden_) {
    case 1024:
      return combineHidden<Mode, 1024, KernelSelector, Hybrid>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                               layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                               context, numBlocks, stream);
    case 2048:
      return combineHidden<Mode, 2048, KernelSelector, Hybrid>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                               layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                               context, numBlocks, stream);
    case 4096:
      return combineHidden<Mode, 4096, KernelSelector, Hybrid>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                               layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                               context, numBlocks, stream);
    case 4352:
      return combineHidden<Mode, 4352, KernelSelector, Hybrid>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                               layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                               context, numBlocks, stream);
    case 5120:
      return combineHidden<Mode, 5120, KernelSelector, Hybrid>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                               layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                               context, numBlocks, stream);
    case 5376:
      return combineHidden<Mode, 5376, KernelSelector, Hybrid>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                               layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                               context, numBlocks, stream);
    case 6656:
      return combineHidden<Mode, 6656, KernelSelector, Hybrid>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                               layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                               context, numBlocks, stream);
    case 7168:
      return combineHidden<Mode, 7168, KernelSelector, Hybrid>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                               layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                               context, numBlocks, stream);
    case 8192:
      return combineHidden<Mode, 8192, KernelSelector, Hybrid>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                               layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                               context, numBlocks, stream);
    case 8704:
      return combineHidden<Mode, 8704, KernelSelector, Hybrid>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                               layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                               context, numBlocks, stream);
    case 9216:
      return combineHidden<Mode, 9216, KernelSelector, Hybrid>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                               layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                               context, numBlocks, stream);
    default:
      EP_HOST_ASSERT(false && "unsupported latency combine hidden size");
  }
}

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_COMBINE_COMMON_CUH_
