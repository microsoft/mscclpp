// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "common.cuh"

namespace mscclpp {
namespace ep {
namespace combine {

template <int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout>
__global__ __launch_bounds__(detail::CombineNThreads, 1) void combineRankLocalReduceLatencyKernel(
    void* output, const void* expertOutput, const int64_t* topkIndices, const float* topkWeights, const int* srcInfo,
    const int64_t* layoutRange, Workload workload, void* combineRecvBuffer, const void* dispatchRecvBuffer,
    const DeviceContext* context) {
  detail::combineLatencyBody<CombineMode::RANK_LOCAL_REDUCE, Hidden, DispatchType, ScaleBlockSize, Layout>(
      output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload, combineRecvBuffer,
      dispatchRecvBuffer, context);
}

struct RankLocalReduceLatencyKernelSelector {
  template <int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout>
  static auto get() {
    return combineRankLocalReduceLatencyKernel<Hidden, DispatchType, ScaleBlockSize, Layout>;
  }
};

void rankLocalReduceLatency(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights,
                            const int* srcInfo, const int64_t* layoutRange, const Workload& workload, void* recvBuffer,
                            void* dispatchRecvBuffer, const DeviceContext& context, const DeviceContext* deviceContext,
                            int numBlocks, cudaStream_t stream) {
  detail::combineAlgorithm<CombineMode::RANK_LOCAL_REDUCE, RankLocalReduceLatencyKernelSelector>(
      output, input, topkIdx, topkWeights, srcInfo, layoutRange, workload, recvBuffer, dispatchRecvBuffer, context,
      deviceContext, numBlocks, stream);
}

}  // namespace combine
}  // namespace ep
}  // namespace mscclpp
