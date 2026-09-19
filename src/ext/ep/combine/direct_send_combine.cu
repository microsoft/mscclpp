// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "common.cuh"

namespace mscclpp {
namespace ep {
namespace {

#if MSCCLPP_BULK_AVAILABLE

template <int Hidden>
MSCCLPP_DEVICE_INLINE void dispatchSend(const void* expertOutput, const int* srcInfo, const int64_t* layoutRange,
                                        int nExperts, int nRanks, int maxTokensPerRank, void* combineBuffer,
                                        const TransportView& transport, uint8_t* sharedMemory) {
  if (threadIdx.x >= WARP_SIZE) return;
  const int laneId = getLaneId();
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
      void* destinationBuffer = transport.mappedBuffer(combineBuffer, sourceRank);
      auto* destinationRow = reinterpret_cast<uint8_t*>(destinationBuffer) +
                             (static_cast<size_t>(globalExpertIdx) * maxTokensPerRank + sourceTokenIdx) * HiddenBytes;
      mscclpp::bulkStore(destinationRow, outputTile, static_cast<uint32_t>(HiddenBytes));
      mscclpp::bulkStoreCommit();
      hasPendingStore = true;
    }

    if (hasPendingStore) mscclpp::bulkStoreWait();
  }
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void dispatchRecv(void* output, const int64_t* __restrict__ topkIndices,
                                        const float* __restrict__ topkWeights, int nTokens, int nTopk,
                                        int maxTokensPerRank, const void* combineBuffer) {
  constexpr int Bf16PerInt4 = sizeof(int4) / sizeof(Bf16);
  constexpr int HiddenInt4 = Hidden / Bf16PerInt4;
  const int threadId = static_cast<int>(threadIdx.x);

  for (int tokenIdx = static_cast<int>(blockIdx.x); tokenIdx < nTokens; tokenIdx += static_cast<int>(gridDim.x)) {
    int regTopkIndices[MaxNumTopk];
    float regTopkWeights[MaxNumTopk];
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
        const auto* expertRow = reinterpret_cast<const int4*>(combineBuffer) +
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

template <int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout>
__global__ __launch_bounds__(CombineNThreads,
                             1) void combineKernel(void* output, const void* expertOutput, const int64_t* topkIndices,
                                                   const float* topkWeights, const int* srcInfo,
                                                   const int64_t* layoutRange, Workload workload, void* combineBuffer,
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
    static_assert(DispatchType == DispatchDataType::BF16);
    const uint32_t epoch = workload.epoch_;
    if (blockIdx.x == 0) {
      publishRankMajorCombineReady(transport, nRanks, epoch, workspaceView);
    } else {
      recvRankMajorRemotePartialsTma<Hidden, CombineMode::DIRECT_SEND>(output, expertOutput, topkIndices, nTokens,
                                                                       nTopk, nExperts, nRanks, maxTokensPerRank, epoch,
                                                                       transport, workspaceView, sharedMemory);
    }
  } else {
    static_assert(Layout == DispatchLayout::EXPERT_MAJOR);
    dispatchSend<Hidden>(expertOutput, srcInfo, layoutRange, nExperts, nRanks, maxTokensPerRank, combineBuffer,
                         transport, sharedMemory);

    workspaceView.combineSyncer_->sync(gridDim.x);
    exchangeCombineReady(transport, nRanks);
    workspaceView.combineSyncer_->sync(gridDim.x);

    dispatchRecv<Hidden>(output, topkIndices, topkWeights, nTokens, nTopk, maxTokensPerRank, combineBuffer);
  }
#endif  // MSCCLPP_BULK_AVAILABLE
}

struct KernelSelector {
  template <int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout>
  static auto get() {
    return combineKernel<Hidden, DispatchType, ScaleBlockSize, Layout>;
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

void rankMajorDirectSendCombine(void* output, const void* input, const int64_t* topkIdx, const Workload& workload,
                                void* recvBuffer, void* dispatchRecvBuffer, const DeviceContext& context, int numBlocks,
                                cudaStream_t stream) {
  EP_HOST_ASSERT(workload.outputLayout_ == DispatchLayout::RANK_MAJOR);
  combineAlgorithm<CombineMode::DIRECT_SEND, KernelSelector>(output, input, topkIdx, nullptr, nullptr, nullptr,
                                                             workload, recvBuffer, dispatchRecvBuffer, context,
                                                             numBlocks, stream);
}

}  // namespace ep
}  // namespace mscclpp
