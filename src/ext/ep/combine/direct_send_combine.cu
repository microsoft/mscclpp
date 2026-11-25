// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "common.cuh"

namespace mscclpp::ep {
namespace {

#if MSCCLPP_BULK_AVAILABLE

struct ExpertTokenLocation {
  int localExpertIdx;
  int expertTokenIdx;
  int sourceRank;
};

MSCCLPP_DEVICE_INLINE ExpertTokenLocation locateExpertMajorToken(int flatTokenIdx, const int* expertTokenPrefix,
                                                                 const int64_t* layoutRange, int nRanks) {
  int localExpertIdx = 0;
  while (flatTokenIdx >= expertTokenPrefix[localExpertIdx + 1]) ++localExpertIdx;
  const int expertTokenIdx = flatTokenIdx - expertTokenPrefix[localExpertIdx];
  for (int sourceRank = 0; sourceRank < nRanks; ++sourceRank) {
    int nRankTokens = 0;
    int rankOffset = 0;
    unpack2(layoutRange[localExpertIdx * nRanks + sourceRank], nRankTokens, rankOffset);
    if (expertTokenIdx >= rankOffset && expertTokenIdx < rankOffset + nRankTokens) {
      return {localExpertIdx, expertTokenIdx, sourceRank};
    }
  }
  EP_DEVICE_ASSERT(false);
  return {localExpertIdx, expertTokenIdx, -1};
}

// Warp 0 TMA-copies each local expert's output tokens back to their source ranks' combine buffers.
template <int Hidden>
MSCCLPP_DEVICE_INLINE void sendExpertMajorTokens(const void* expertOutput, const int* srcInfo,
                                                 const int64_t* layoutRange, const Workload& workload, int nRanks,
                                                 void* combineBuffer, const TransportView& transport,
                                                 uint8_t* sharedMemory) {
  if (threadIdx.x >= WARP_SIZE) return;
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(Bf16);
  const int laneIdx = getLaneId();
  const int nLocalExperts = workload.numExperts_ / nRanks;
  const int maxTokensPerRank = workload.maxTokensPerRank_;
  const int nOutputSlotsPerExpert = nRanks * maxTokensPerRank;
  const int nWorkers = directSendWorkerCount<Hidden>(nLocalExperts);
  auto* expertTokenPrefix = reinterpret_cast<int*>(sharedMemory);
  auto* outputTiles = sharedMemory + directSendControlBytes(nLocalExperts);

  if (laneIdx == 0) {
    expertTokenPrefix[0] = 0;
    // Each expert's last rank range ends at the total token count received by that expert.
    for (int localExpertIdx = 0; localExpertIdx < nLocalExperts; ++localExpertIdx) {
      int nLastRankTokens = 0;
      int lastRankOffset = 0;
      unpack2(layoutRange[localExpertIdx * nRanks + nRanks - 1], nLastRankTokens, lastRankOffset);
      expertTokenPrefix[localExpertIdx + 1] = expertTokenPrefix[localExpertIdx] + lastRankOffset + nLastRankTokens;
    }
  }
  __syncwarp();

  const int nTotalTokens = expertTokenPrefix[nLocalExperts];
  const int tokenBegin = static_cast<int>(static_cast<int64_t>(nTotalTokens) * blockIdx.x / gridDim.x);
  const int tokenEnd = static_cast<int>(static_cast<int64_t>(nTotalTokens) * (blockIdx.x + 1) / gridDim.x);
  if (laneIdx >= nWorkers || tokenBegin + laneIdx >= tokenEnd) return;

  auto* tile = outputTiles + static_cast<size_t>(laneIdx) * HiddenBytes;
  auto& barrier =
      reinterpret_cast<mscclpp::BulkBarrier*>(outputTiles + static_cast<size_t>(nWorkers) * HiddenBytes)[laneIdx];
  uint32_t phase = 0;
  barrier.init();

  bool hasPendingStore = false;
  for (int flatTokenIdx = tokenBegin + laneIdx; flatTokenIdx < tokenEnd; flatTokenIdx += nWorkers) {
    if (hasPendingStore) mscclpp::bulkStoreWaitSource();
    const auto loc = locateExpertMajorToken(flatTokenIdx, expertTokenPrefix, layoutRange, nRanks);
    const int inputSlotIdx = loc.localExpertIdx * nOutputSlotsPerExpert + loc.expertTokenIdx;
    const int srcTokenIdx = srcInfo[inputSlotIdx];
    EP_DEVICE_ASSERT(srcTokenIdx >= 0 && srcTokenIdx < maxTokensPerRank);

    const auto* srcToken = static_cast<const uint8_t*>(expertOutput) + static_cast<size_t>(inputSlotIdx) * HiddenBytes;
    barrier.arriveAndExpect(HiddenBytes);
    mscclpp::bulkLoad(tile, srcToken, HiddenBytes, barrier);
    barrier.wait(phase);
    mscclpp::bulkFence();

    const int globalExpertIdx = transport.rank_ * nLocalExperts + loc.localExpertIdx;
    auto* dstBuffer = static_cast<uint8_t*>(transport.mappedBuffer(combineBuffer, loc.sourceRank));
    const size_t dstSlotIdx = static_cast<size_t>(globalExpertIdx) * maxTokensPerRank + srcTokenIdx;
    mscclpp::bulkStore(dstBuffer + dstSlotIdx * HiddenBytes, tile, HiddenBytes);
    mscclpp::bulkStoreCommit();
    hasPendingStore = true;
  }

  mscclpp::bulkStoreWait();
  barrier.invalidate();
}

// Reduce each token's per-expert outputs from combineBuffer with FP32 weighted accumulation.
template <int Hidden, int MaxNTopk>
MSCCLPP_DEVICE_INLINE void reduceExpertMajorTokens(void* output, const int64_t* __restrict__ topkIndices,
                                                   const float* __restrict__ topkWeights, const Workload& workload,
                                                   const void* combineBuffer) {
  constexpr int NFp32x8 = Hidden / mscclpp::f32x8::Size;
  constexpr int NFp32x2PerFp32x8 = mscclpp::f32x8::Size / mscclpp::f32x2::Size;
  const int nTokens = workload.numTokens_;
  const int nTopk = workload.numTopk_;
  const int nExperts = workload.numExperts_;
  const int maxTokensPerRank = workload.maxTokensPerRank_;
  const bool useWeights = workload.applyRouterWeights_ && topkWeights != nullptr;
  const auto* expertTokens = static_cast<const mscclpp::bf16x8*>(combineBuffer);
  auto* outputTokens = static_cast<mscclpp::bf16x8*>(output);

  for (int tokenIdx = static_cast<int>(blockIdx.x); tokenIdx < nTokens; tokenIdx += static_cast<int>(gridDim.x)) {
    int routeExperts[MaxNTopk];
    float routeWeights[MaxNTopk];
    for (int routeIdx = 0; routeIdx < nTopk; ++routeIdx) {
      const int64_t rawExpertIdx = topkIndices[tokenIdx * nTopk + routeIdx];
      routeExperts[routeIdx] = rawExpertIdx >= 0 && rawExpertIdx < nExperts ? static_cast<int>(rawExpertIdx) : -1;
      routeWeights[routeIdx] = useWeights ? topkWeights[tokenIdx * nTopk + routeIdx] : 1.0f;
    }

#pragma unroll
    for (int fp32x8Idx = static_cast<int>(threadIdx.x); fp32x8Idx < NFp32x8; fp32x8Idx += CombineNThreads) {
      mscclpp::f32x8 sum;
      auto* sumFp32x2 = reinterpret_cast<mscclpp::f32x2*>(&sum);
#pragma unroll
      for (int idx = 0; idx < NFp32x2PerFp32x8; ++idx) {
        sumFp32x2[idx].storage = make_float2(0.0f, 0.0f);
      }
      for (int routeIdx = 0; routeIdx < nTopk; ++routeIdx) {
        const int expertIdx = routeExperts[routeIdx];
        if (expertIdx < 0) continue;
        const size_t expertTokenIdx = static_cast<size_t>(expertIdx) * maxTokensPerRank + tokenIdx;
        const auto values = mscclpp::to<mscclpp::f32x8>(expertTokens[expertTokenIdx * NFp32x8 + fp32x8Idx]);
        const auto* valuesFp32x2 = reinterpret_cast<const mscclpp::f32x2*>(&values);
        const mscclpp::f32x2 weightFp32x2(make_float2(routeWeights[routeIdx], routeWeights[routeIdx]));
#pragma unroll
        for (int idx = 0; idx < NFp32x2PerFp32x8; ++idx) {
          sumFp32x2[idx] = mscclpp::fma(valuesFp32x2[idx], weightFp32x2, sumFp32x2[idx]);
        }
      }
      outputTokens[static_cast<size_t>(tokenIdx) * NFp32x8 + fp32x8Idx] = mscclpp::to<mscclpp::bf16x8>(sum);
    }
  }
}

#endif  // MSCCLPP_BULK_AVAILABLE

template <int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout, int MaxNTopk>
__global__ __launch_bounds__(CombineNThreads,
                             1) void combineKernel(void* output, const void* expertOutput, const int64_t* topkIndices,
                                                   const float* topkWeights, const int* srcInfo,
                                                   const int64_t* layoutRange, Workload workload, void* combineBuffer,
                                                   const void* dispatchRecvBuffer, const DeviceContext* context) {
#if MSCCLPP_BULK_AVAILABLE
  extern __shared__ __align__(128) uint8_t sharedMemory[];
  const int nRanks = context->numRanks_;
  const TransportView transport(context);
  WorkspaceView workspace(context->workspace_, nRanks, workload.numExperts_);

  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    const uint32_t flag = getDispatchFlag(workspace);
    if (blockIdx.x == 0) {
      publishRankMajorCombineReady(transport, nRanks, flag, workspace);
    } else {
      combineRankMajorTokens<Hidden, CombineMode::DIRECT_SEND, DispatchType, MaxNTopk>(
          output, expertOutput, topkIndices, topkWeights, workload, nRanks, transport, workspace, flag, sharedMemory);
    }
  } else {
    static_assert(Layout == DispatchLayout::EXPERT_MAJOR);
    sendExpertMajorTokens<Hidden>(expertOutput, srcInfo, layoutRange, workload, nRanks, combineBuffer, transport,
                                  sharedMemory);

    workspace.combineSyncer_->sync(gridDim.x);
    exchangeCombineReady(transport, nRanks);
    workspace.combineSyncer_->sync(gridDim.x);

    reduceExpertMajorTokens<Hidden, MaxNTopk>(output, topkIndices, topkWeights, workload, combineBuffer);
  }
#endif  // MSCCLPP_BULK_AVAILABLE
}

struct KernelSelector {
  template <int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout, int MaxNTopk>
  static auto get() {
    return combineKernel<Hidden, DispatchType, ScaleBlockSize, Layout, MaxNTopk>;
  }
};

}  // namespace

void expertMajorDirectSendCombine(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights,
                                  const int* srcInfo, const int64_t* layoutRange, const Workload& workload,
                                  void* recvBuffer, void* dispatchRecvBuffer, const DeviceContext& context,
                                  int numBlocks, cudaStream_t stream) {
  combineAlgorithm<CombineMode::DIRECT_SEND, KernelSelector>(output, input, topkIdx, topkWeights, srcInfo, layoutRange,
                                                             workload, recvBuffer, dispatchRecvBuffer, context,
                                                             numBlocks, stream);
}

void rankMajorDirectSendCombine(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights,
                                const Workload& workload, void* recvBuffer, void* dispatchRecvBuffer,
                                const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  EP_HOST_ASSERT(workload.outputLayout_ == DispatchLayout::RANK_MAJOR);
  combineAlgorithm<CombineMode::DIRECT_SEND, KernelSelector>(output, input, topkIdx, topkWeights, nullptr, nullptr,
                                                             workload, recvBuffer, dispatchRecvBuffer, context,
                                                             numBlocks, stream);
}

}  // namespace mscclpp::ep
