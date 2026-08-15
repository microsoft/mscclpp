// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "common.cuh"

namespace mscclpp {
namespace ep {
namespace combine {

template <int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout>
__global__ __launch_bounds__(detail::CombineNThreads, 1) void rankLocalReduceKernel(
    void* output, const void* expertOutput, const int64_t* topkIndices, const float* topkWeights, const int* srcInfo,
    const int64_t* layoutRange, Workload workload, void* combineRecvBuffer, const void* dispatchRecvBuffer,
    const DeviceContext* context) {
  detail::combineBody<CombineMode::RANK_LOCAL_REDUCE, Hidden, DispatchType, ScaleBlockSize, Layout>(
      output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload, combineRecvBuffer,
      dispatchRecvBuffer, context);
}

struct RankLocalReduceKernelSelector {
  template <int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout>
  static auto get() {
    return rankLocalReduceKernel<Hidden, DispatchType, ScaleBlockSize, Layout>;
  }
};

namespace {

void runRankLocalReduce(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights,
                        const int* srcInfo, const int64_t* layoutRange, const Workload& workload, void* recvBuffer,
                        void* dispatchRecvBuffer, const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  detail::combineAlgorithm<CombineMode::RANK_LOCAL_REDUCE, RankLocalReduceKernelSelector>(
      output, input, topkIdx, topkWeights, srcInfo, layoutRange, workload, recvBuffer, dispatchRecvBuffer, context,
      numBlocks, stream);
}

}  // namespace

void expertMajorRankLocalReduce(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights,
                                const int* srcInfo, const int64_t* layoutRange, const Workload& workload,
                                void* recvBuffer, void* dispatchRecvBuffer, const DeviceContext& context, int numBlocks,
                                cudaStream_t stream) {
  EP_HOST_ASSERT(workload.outputLayout_ == DispatchLayout::EXPERT_MAJOR);
  runRankLocalReduce(output, input, topkIdx, topkWeights, srcInfo, layoutRange, workload, recvBuffer,
                     dispatchRecvBuffer, context, numBlocks, stream);
}

void rankMajorGatherReduce(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights,
                           const int* srcInfo, const int64_t* layoutRange, const Workload& workload, void* recvBuffer,
                           void* dispatchRecvBuffer, const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  EP_HOST_ASSERT(workload.outputLayout_ == DispatchLayout::RANK_MAJOR);
  runRankLocalReduce(output, input, topkIdx, topkWeights, srcInfo, layoutRange, workload, recvBuffer,
                     dispatchRecvBuffer, context, numBlocks, stream);
}

}  // namespace combine
}  // namespace ep
}  // namespace mscclpp
