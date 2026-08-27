// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "common.cuh"

namespace mscclpp {
namespace ep {

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
__global__ __launch_bounds__(DispatchNThreads, 1) void expertMajorDispatchKernel(
    void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx, float* outputTopkWeights,
    int64_t* outputLayout, int* outputCount, const int64_t* topkIndices, const float* topkWeights,
    const void* inputTokens, Workload workload, void* recvBuffer, const DeviceContext* context) {
  dispatchBody<Hidden, DataType, ScaleBlockSize, DispatchLayout::EXPERT_MAJOR>(
      output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, topkIndices,
      topkWeights, inputTokens, workload, recvBuffer, context);
}

struct ExpertMajorDispatchKernelSelector {
  template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
  static auto get() {
    return expertMajorDispatchKernel<Hidden, DataType, ScaleBlockSize>;
  }
};

void expertMajorDispatch(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                         float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                         const int64_t* topkIdx, const float* topkWeights, const Workload& workload, void* recvBuffer,
                         const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  dispatchAlgorithm<DispatchLayout::EXPERT_MAJOR, ExpertMajorDispatchKernelSelector>(
      output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input, topkIdx,
      topkWeights, workload, recvBuffer, context, numBlocks, stream);
}

size_t workspaceSize(int numRanks, int numExperts, int maxTokensPerRank, int numTopk, int epSize) {
  return workspaceBytes(numRanks, numExperts, maxTokensPerRank, numTopk, epSize);
}

}  // namespace ep
}  // namespace mscclpp
