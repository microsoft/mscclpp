// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "common.cuh"

namespace mscclpp {
namespace ep {
namespace combine {

template <int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout>
__global__ __launch_bounds__(detail::CombineNThreads, 1) void combineDirectSendLatencyKernel(
    void* output, const void* expertOutput, const int64_t* topkIndices, const float* topkWeights, const int* srcInfo,
    const int64_t* layoutRange, Workload workload, void* combineRecvBuffer, const void* dispatchRecvBuffer,
    const DeviceContext* context) {
  detail::combineLatencyBody<CombineMode::DIRECT_SEND, Hidden, DispatchType, ScaleBlockSize, Layout>(
      output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload, combineRecvBuffer,
      dispatchRecvBuffer, context);
}

struct DirectSendLatencyKernelSelector {
  template <int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout>
  static auto get() {
    return combineDirectSendLatencyKernel<Hidden, DispatchType, ScaleBlockSize, Layout>;
  }
};

void directSendLatency(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights,
                       const int* srcInfo, const int64_t* layoutRange, const Workload& workload, void* recvBuffer,
                       void* dispatchRecvBuffer, const DeviceContext& context, const DeviceContext* deviceContext,
                       int numBlocks, cudaStream_t stream) {
  detail::combineAlgorithm<CombineMode::DIRECT_SEND, DirectSendLatencyKernelSelector>(
      output, input, topkIdx, topkWeights, srcInfo, layoutRange, workload, recvBuffer, dispatchRecvBuffer, context,
      deviceContext, numBlocks, stream);
}

}  // namespace combine
}  // namespace ep
}  // namespace mscclpp
