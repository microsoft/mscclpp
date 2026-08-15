// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "common.cuh"

namespace mscclpp {
namespace ep {
namespace dispatch {

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
__global__ __launch_bounds__(::mscclpp::ep::detail::DispatchNThreads, 1) void latencyRankMajorKernel(
    void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx, float* outputTopkWeights,
    int64_t* outputLayout, int* outputCount, const int64_t* topkIndices, const float* topkWeights,
    const void* inputTokens, Workload workload, void* recvBuffer, const DeviceContext* context) {
  detail::latencyBody<Hidden, DataType, ScaleBlockSize, DispatchLayout::RANK_MAJOR>(
      output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, topkIndices,
      topkWeights, inputTokens, workload, recvBuffer, context);
}

struct LatencyRankMajorKernelSelector {
  template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
  static auto get() {
    return latencyRankMajorKernel<Hidden, DataType, ScaleBlockSize>;
  }
};

void latencyRankMajor(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                      float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                      const int64_t* topkIdx, const float* topkWeights, const Workload& workload, void* recvBuffer,
                      const DeviceContext& context, const DeviceContext* deviceContext, int numBlocks,
                      cudaStream_t stream) {
  detail::dispatchAlgorithm<DispatchLayout::RANK_MAJOR, LatencyRankMajorKernelSelector>(
      output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input, topkIdx,
      topkWeights, workload, recvBuffer, context, deviceContext, numBlocks, stream);
}

}  // namespace dispatch
}  // namespace ep
}  // namespace mscclpp
