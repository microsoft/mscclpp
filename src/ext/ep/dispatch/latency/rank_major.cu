// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "common.cuh"

namespace mscclpp {
namespace ep {
namespace dispatch {

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
__global__ __launch_bounds__(::mscclpp::ep::detail::DispatchNThreads, 1) void dispatchRankMajorLatencyKernel(
    void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx, float* outputTopkWeights,
    int64_t* outputLayout, int* outputCount, const int64_t* topkIndices, const float* topkWeights,
    const void* inputTokens, Workload workload, void* recvBuffer, const DeviceContext* context) {
  detail::dispatchLatencyBody<Hidden, DataType, ScaleBlockSize, DispatchLayout::RANK_MAJOR>(
      output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, topkIndices,
      topkWeights, inputTokens, workload, recvBuffer, context);
}

struct RankMajorLatencyKernelSelector {
  template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
  static auto get() {
    return dispatchRankMajorLatencyKernel<Hidden, DataType, ScaleBlockSize>;
  }
};

void rankMajorLatency(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                      float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                      const int64_t* topkIdx, const float* topkWeights, const Workload& workload, void* recvBuffer,
                      const DeviceContext& context, const DeviceContext* deviceContext, int numBlocks,
                      cudaStream_t stream) {
  detail::dispatchAlgorithm<DispatchLayout::RANK_MAJOR, RankMajorLatencyKernelSelector>(
      output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input, topkIdx,
      topkWeights, workload, recvBuffer, context, deviceContext, numBlocks, stream);
}

}  // namespace dispatch
}  // namespace ep
}  // namespace mscclpp
