// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "common.cuh"

namespace mscclpp {
namespace ep {

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
__global__ __launch_bounds__(DispatchNThreads, 1) void rankMajorDispatchKernel(
    void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx, float* outputTopkWeights,
    int64_t* outputLayout, int* outputCount, const int64_t* topkIndices, const float* topkWeights,
    const void* inputTokens, Workload workload, void* recvBuffer, const DeviceContext* context) {
  dispatchBody<Hidden, DataType, ScaleBlockSize, DispatchLayout::RANK_MAJOR>(
      output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, topkIndices,
      topkWeights, inputTokens, workload, recvBuffer, context);
}

struct RankMajorDispatchKernelSelector {
  template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
  static auto get() {
    return rankMajorDispatchKernel<Hidden, DataType, ScaleBlockSize>;
  }
};

void rankMajorDispatch(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                       float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                       const int64_t* topkIdx, const float* topkWeights, const Workload& workload, void* recvBuffer,
                       const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  // dfa5e151: reset before any producer can allocate a destination slot. Unlike
  // the old workspace, this layout starts with persistent arrival baselines and
  // recv counts; never memset workspace[0..R) here. This node is graph-captured.
  const WorkspaceView workspace(context.workspace_, context.numRanks_, workload.numExperts_);
  CUDA_CHECK(cudaMemsetAsync(workspace.dispatchRankPayloadSlots_, 0,
                             static_cast<size_t>(context.numRanks_) * sizeof(int), stream));
  dispatchAlgorithm<DispatchLayout::RANK_MAJOR, RankMajorDispatchKernelSelector>(
      output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input, topkIdx,
      topkWeights, workload, recvBuffer, context, numBlocks, stream);
}

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
__global__ __launch_bounds__(256, 1) void expandRankMajorTopkDuplicateRoutesKernel(void* output, void* outputScales,
                                                                                   const int* outputTopkIdx, int rank,
                                                                                   int nExperts, int nRanks, int nTopk,
                                                                                   int maxTokensPerRank) {
  constexpr int NWarps = 256 / WARP_SIZE;
  const int tokenIdx = static_cast<int>(blockIdx.x) * NWarps + static_cast<int>(threadIdx.x) / WARP_SIZE;
  if (tokenIdx >= nRanks * maxTokensPerRank) return;
  expandRankMajorTopkDuplicateRoutes<Hidden, DataType, ScaleBlockSize>(output, outputScales, outputTopkIdx, tokenIdx,
                                                                       rank, nExperts, nRanks, nTopk, maxTokensPerRank);
}

void rankMajorTopkExpandedDispatch(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                                   float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                                   const int64_t* topkIdx, const float* topkWeights, const Workload& workload,
                                   void* recvBuffer, const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  static_cast<void>(outputSrcInfo);
  static_cast<void>(outputLayout);
  if (workload.dispatchDataType_ == DispatchDataType::BF16) {
    topk_expanded::dispatch(output, outputTopkIdx, outputTopkWeights, outputCount, input, topkIdx, topkWeights,
                            workload, context, numBlocks, stream);
    return;
  }
  dispatchAlgorithm<DispatchLayout::RANK_MAJOR_TOPK_EXPANDED, RankMajorDispatchKernelSelector>(
      output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input, topkIdx,
      topkWeights, workload, recvBuffer, context, numBlocks, stream);

  if (!workload.deduplicateExpandedRoutes_) return;

  constexpr int NThreads = 256;
  constexpr int NWarps = NThreads / WARP_SIZE;
  const int expansionBlocks = (context.numRanks_ * workload.maxTokensPerRank_ + NWarps - 1) / NWarps;
#define LAUNCH_EXPANSION_TYPED(HIDDEN, TYPE, SCALE)                                                                   \
  expandRankMajorTopkDuplicateRoutesKernel<HIDDEN, TYPE, SCALE><<<expansionBlocks, NThreads, 0, stream>>>(            \
      output, outputScales, outputTopkIdx, context.rank_, workload.numExperts_, context.numRanks_, workload.numTopk_, \
      workload.maxTokensPerRank_)
#define LAUNCH_EXPANSION(HIDDEN)                                                     \
  if (workload.dispatchDataType_ == DispatchDataType::FP8_E4M3) {                    \
    LAUNCH_EXPANSION_TYPED(HIDDEN, DispatchDataType::FP8_E4M3, Mxfp8ScaleBlockSize); \
  } else {                                                                           \
    LAUNCH_EXPANSION_TYPED(HIDDEN, DispatchDataType::BF16, 0);                       \
  }
  switch (workload.hidden_) {
    case 2048:
      LAUNCH_EXPANSION(2048);
      break;
    case 4096:
      LAUNCH_EXPANSION(4096);
      break;
    case 4352:
      LAUNCH_EXPANSION(4352);
      break;
    case 5120:
      LAUNCH_EXPANSION(5120);
      break;
    case 6656:
      LAUNCH_EXPANSION(6656);
      break;
    case 7168:
      LAUNCH_EXPANSION(7168);
      break;
    case 8192:
      LAUNCH_EXPANSION(8192);
      break;
    case 8704:
      LAUNCH_EXPANSION(8704);
      break;
    case 9216:
      LAUNCH_EXPANSION(9216);
      break;
    default:
      EP_HOST_ASSERT(false && "unsupported hidden size");
  }
#undef LAUNCH_EXPANSION
#undef LAUNCH_EXPANSION_TYPED
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace ep
}  // namespace mscclpp
