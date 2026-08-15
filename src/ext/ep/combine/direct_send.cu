// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "common.cuh"

namespace mscclpp {
namespace ep {
namespace combine {

template <int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout>
__global__ __launch_bounds__(detail::CombineNThreads,
                             1) void directSendKernel(void* output, const void* expertOutput,
                                                      const int64_t* topkIndices, const float* topkWeights,
                                                      const int* srcInfo, const int64_t* layoutRange, Workload workload,
                                                      void* combineRecvBuffer, const void* dispatchRecvBuffer,
                                                      const DeviceContext* context) {
  detail::combineBody<CombineMode::DIRECT_SEND, Hidden, DispatchType, ScaleBlockSize, Layout>(
      output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload, combineRecvBuffer,
      dispatchRecvBuffer, context);
}

struct DirectSendKernelSelector {
  template <int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout>
  static auto get() {
    return directSendKernel<Hidden, DispatchType, ScaleBlockSize, Layout>;
  }
};

void directSend(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights, const int* srcInfo,
                const int64_t* layoutRange, const Workload& workload, void* recvBuffer, void* dispatchRecvBuffer,
                const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  detail::combineAlgorithm<CombineMode::DIRECT_SEND, DirectSendKernelSelector>(
      output, input, topkIdx, topkWeights, srcInfo, layoutRange, workload, recvBuffer, dispatchRecvBuffer, context,
      numBlocks, stream);
}

}  // namespace combine
}  // namespace ep
}  // namespace mscclpp
