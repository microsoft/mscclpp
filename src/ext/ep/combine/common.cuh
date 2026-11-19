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

// Shared-memory staging buffer for rank-major combine: token tiles, TMA barriers, and route metadata.
template <int Hidden, DispatchDataType DispatchType, int NSlots>
struct alignas(128) RankMajorCombineStage {
  DispatchElementType<DispatchType> tokens[NSlots][Hidden];
  mscclpp::BulkBarrier barriers[NSlots];
  int validRoutes[NSlots];
  float routeWeights[NSlots];
};

inline constexpr int HybridCombineMaxPendingTokens = 64;

// Hybrid combine per-warp-group shared-memory control state, placed after each group's compacted stage slots.
template <int NSlots>
struct alignas(16) HybridCombineGroupControl {
  mscclpp::BulkBarrier barrier;
  float routeWeights[NSlots];
  int pendingTokens[HybridCombineMaxPendingTokens];
  int nPendingTokens;
  int nRoundRoutes;
  int nTotalRounds;
  bool interDomainTokenReady;
};

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
  if constexpr (sizeof(RankMajorCombineStage<Hidden, DispatchType, MaxNTopk>) <= OptimizedDynamicSharedMemoryBytes) {
    return MaxNTopk;
  }
  static_assert(sizeof(RankMajorCombineStage<Hidden, DispatchType, 8>) <= OptimizedDynamicSharedMemoryBytes);
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
    constexpr size_t SharedBytes = sizeof(RankMajorCombineStage<Hidden, DispatchType, StageSlots>);
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
MSCCLPP_DEVICE_INLINE void accumulateRankMajorFp32x8(
    mscclpp::f32x8& sum, int fp32x8Idx, const RankMajorCombineStage<Hidden, DispatchType, StageSlots>& stage,
    float dequantScale) {
  constexpr bool IsDirectSend = Mode == CombineMode::DIRECT_SEND;
  constexpr int NFp32x2PerFp32x8 = mscclpp::f32x8::Size / mscclpp::f32x2::Size;
  auto* sumFp32x2 = reinterpret_cast<mscclpp::f32x2*>(&sum);
#pragma unroll
  for (int stageIdx = 0; stageIdx < StageSlots; ++stageIdx) {
    if (!stage.validRoutes[stageIdx]) continue;
    const float routeWeight = IsDirectSend ? stage.routeWeights[stageIdx] : 1.0f;
    const auto* routeToken = stage.tokens[stageIdx];
    if constexpr (DispatchType == DispatchDataType::BF16) {
      const auto values = mscclpp::to<mscclpp::f32x8>(reinterpret_cast<const mscclpp::bf16x8*>(routeToken)[fp32x8Idx]);
      const auto* valuesFp32x2 = reinterpret_cast<const mscclpp::f32x2*>(&values);
#pragma unroll
      for (int idx = 0; idx < NFp32x2PerFp32x8; ++idx) {
        if constexpr (IsDirectSend) {
          sumFp32x2[idx] =
              mscclpp::fma(valuesFp32x2[idx], mscclpp::f32x2(make_float2(routeWeight, routeWeight)), sumFp32x2[idx]);
        } else {
          sumFp32x2[idx] = sumFp32x2[idx] + valuesFp32x2[idx];
        }
      }
    } else {
      const auto values = reinterpret_cast<const mscclpp::f8_e4m3x8*>(routeToken)[fp32x8Idx];
      const float scale = IsDirectSend ? dequantScale * routeWeight : dequantScale;
      if (scale != 0.0f) {
        dequantizeAndAccumulateFp8E4M3(values, scale, sumFp32x2);
      }
    }
  }
}

// Worker blocks (blockIdx.x > 0): TMA-load rank-major expert outputs from peers and reduce into output.
template <int Hidden, CombineMode Mode, DispatchDataType DispatchType, int MaxNTopk>
MSCCLPP_DEVICE_INLINE void combineRankMajorTokens(void* output, const void* expertOutput,
                                                  const int64_t* __restrict__ topkIndices,
                                                  const float* __restrict__ topkWeights, const Workload& workload,
                                                  int nRanks, const TransportView& transport, WorkspaceView& workspace,
                                                  uint32_t flag, uint8_t* sharedMemory) {
#if defined(__CUDA_ARCH__)
  static_assert(__CUDA_ARCH__ >= 900, "TMA rank-major combine requires SM90 or newer");
#endif
  static_assert(DispatchType == DispatchDataType::BF16 || DispatchType == DispatchDataType::FP8_E4M3_STATIC);
  static_assert(Hidden % mscclpp::bf16x8::Size == 0);
  using ElementType = DispatchElementType<DispatchType>;
  constexpr bool IsDirectSend = Mode == CombineMode::DIRECT_SEND;
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(ElementType);
  constexpr int NFp32x8 = Hidden / mscclpp::f32x8::Size;
  constexpr int MaxFp32x8PerThread = configCellDiv(NFp32x8, CombineNThreads);
  constexpr int NFp32x2PerFp32x8 = mscclpp::f32x8::Size / mscclpp::f32x2::Size;
  constexpr int StageSlots = rankMajorCombineStageSlots<Hidden, DispatchType, MaxNTopk>();
  constexpr int NumRounds = configCellDiv(MaxNTopk, StageSlots);

  const int nTokens = workload.numTokens_;
  const int nWorkers = static_cast<int>(gridDim.x) - 1;
  const int firstTokenIdx = static_cast<int>(blockIdx.x) - 1;
  if (firstTokenIdx >= nTokens) return;

  const int warpIdx = static_cast<int>(threadIdx.x) / WARP_SIZE;
  const int laneIdx = getLaneId();
  const int nTopk = workload.numTopk_;
  const int nExperts = workload.numExperts_;
  const int nLocalExperts = nExperts / nRanks;
  const int maxTokensPerRank = workload.maxTokensPerRank_;
  const bool useWeights = workload.applyRouterWeights_ && topkWeights != nullptr;
  const float dequantScale = DispatchType == DispatchDataType::FP8_E4M3_STATIC ? 1.0f / workload.combineScale_ : 1.0f;
  auto& stage = *reinterpret_cast<RankMajorCombineStage<Hidden, DispatchType, StageSlots>*>(sharedMemory);
  auto* barriers = stage.barriers;

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

    mscclpp::f32x8 tileSum[MaxFp32x8PerThread];
#pragma unroll
    for (int stepIdx = 0; stepIdx < MaxFp32x8PerThread; ++stepIdx) {
      auto* sumFp32x2 = reinterpret_cast<mscclpp::f32x2*>(&tileSum[stepIdx]);
#pragma unroll
      for (int idx = 0; idx < NFp32x2PerFp32x8; ++idx) {
        sumFp32x2[idx].storage = make_float2(0.0f, 0.0f);
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
          stage.validRoutes[laneIdx] = active;
          if constexpr (IsDirectSend) {
            stage.routeWeights[laneIdx] =
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
            mscclpp::bulkLoad(stage.tokens[laneIdx], peerOutput + inputTokenIdx * HiddenBytes, HiddenBytes,
                              barriers[laneIdx]);
            pending = false;
          }
        }
        if (active) barriers[laneIdx].wait(phase);
        __syncwarp();
        if (laneIdx == 0) mscclpp::bulkFence();
      }
      __syncthreads();

#pragma unroll
      for (int stepIdx = 0; stepIdx < MaxFp32x8PerThread; ++stepIdx) {
        const int fp32x8Idx = static_cast<int>(threadIdx.x) + stepIdx * CombineNThreads;
        if (fp32x8Idx < NFp32x8) {
          accumulateRankMajorFp32x8<Mode, Hidden, DispatchType, StageSlots>(tileSum[stepIdx], fp32x8Idx, stage,
                                                                            dequantScale);
        }
      }
      __syncthreads();
    }

    auto* tokenOutput = static_cast<mscclpp::bf16x8*>(output) + static_cast<size_t>(tokenIdx) * NFp32x8;
#pragma unroll
    for (int stepIdx = 0; stepIdx < MaxFp32x8PerThread; ++stepIdx) {
      const int fp32x8Idx = static_cast<int>(threadIdx.x) + stepIdx * CombineNThreads;
      if (fp32x8Idx < NFp32x8) {
        tokenOutput[fp32x8Idx] = mscclpp::to<mscclpp::bf16x8>(tileSum[stepIdx]);
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
    case 256:
      return combineHidden<Mode, 256, KernelSelector, Hybrid>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                              layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                              context, numBlocks, stream);
    case 512:
      return combineHidden<Mode, 512, KernelSelector, Hybrid>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                              layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                              context, numBlocks, stream);
    case 768:
      return combineHidden<Mode, 768, KernelSelector, Hybrid>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                              layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                              context, numBlocks, stream);
    case 1024:
      return combineHidden<Mode, 1024, KernelSelector, Hybrid>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                               layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                               context, numBlocks, stream);
    case 1280:
      return combineHidden<Mode, 1280, KernelSelector, Hybrid>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                               layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                               context, numBlocks, stream);
    case 1536:
      return combineHidden<Mode, 1536, KernelSelector, Hybrid>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                               layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                               context, numBlocks, stream);
    case 1792:
      return combineHidden<Mode, 1792, KernelSelector, Hybrid>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                               layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                               context, numBlocks, stream);
    case 2048:
      return combineHidden<Mode, 2048, KernelSelector, Hybrid>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                               layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                               context, numBlocks, stream);
    case 2560:
      return combineHidden<Mode, 2560, KernelSelector, Hybrid>(output, expertOutput, topkIndices, topkWeights, srcInfo,
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
