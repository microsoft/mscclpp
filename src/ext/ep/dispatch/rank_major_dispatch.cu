// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "common.cuh"

namespace mscclpp {
namespace ep {

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize, bool AsyncDispatch = false>
__global__ __launch_bounds__(DispatchNThreads, 1) void rankMajorDispatchKernel(
    void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx, float* outputTopkWeights,
    int64_t* outputLayout, int* outputCount, const int64_t* topkIndices, const float* topkWeights,
    const void* inputTokens, Workload workload, void* recvBuffer, const DeviceContext* context) {
  dispatchBody<Hidden, DataType, ScaleBlockSize, DispatchLayout::RANK_MAJOR, AsyncDispatch>(
      output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, topkIndices,
      topkWeights, inputTokens, workload, recvBuffer, context);
}

struct RankMajorDispatchKernelSelector {
  template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
  static auto get() {
    return rankMajorDispatchKernel<Hidden, DataType, ScaleBlockSize>;
  }
};

struct RankMajorAsyncDispatchKernelSelector {
  template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
  static auto get() {
    return rankMajorDispatchKernel<Hidden, DataType, ScaleBlockSize, true>;
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
  if (context.gpuNetIoPipelineChunkRows_ > 0) {
    EP_HOST_ASSERT(context.gpuNetIoChunkReady_ != nullptr);
    const size_t counters = static_cast<size_t>(context.numRanks_) *
                            configCellDiv(workload.maxTokensPerRank_, context.gpuNetIoPipelineChunkRows_);
    if (context.gpuNetIoTailRows_ != nullptr)
      CUDA_CHECK(cudaMemsetAsync(context.gpuNetIoTailRows_, 0, (counters + context.numRanks_) * sizeof(int), stream));
    else
      CUDA_CHECK(cudaMemsetAsync(context.gpuNetIoChunkReady_, 0, counters * sizeof(int), stream));
  }
  if (context.gpuNetIoAsyncDispatch_) {
    EP_HOST_ASSERT(context.gpuNetIo_ != nullptr && context.gpuNetIoDispatchPeers_ != nullptr);
    EP_HOST_ASSERT(context.gpuNetIoBatchedDispatch_ && context.gpuNetIoPipelineChunkRows_ > 0);
    EP_HOST_ASSERT(numBlocks > context.numRanks_ + 1);
    CUDA_CHECK(cudaMemsetAsync(context.gpuNetIoDispatchPeers_, 0,
                               static_cast<size_t>(context.numRanks_) * sizeof(GpuNetIoDispatchPeerState), stream));
    return dispatchAlgorithm<DispatchLayout::RANK_MAJOR, RankMajorAsyncDispatchKernelSelector>(
        output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
        topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
  }
  if (context.gpuNetIo_ != nullptr) {
    EP_HOST_ASSERT(context.gpuNetIoDispatchDone_ != nullptr);
    CUDA_CHECK(cudaMemsetAsync(context.gpuNetIoDispatchDone_, 0, sizeof(int), stream));
  }
  dispatchAlgorithm<DispatchLayout::RANK_MAJOR, RankMajorDispatchKernelSelector>(
      output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input, topkIdx,
      topkWeights, workload, recvBuffer, context, numBlocks, stream);
}

void rankMajorTopkExpandedDispatch(void* output, [[maybe_unused]] void* outputScales,
                                   [[maybe_unused]] int* outputSrcInfo, int* outputTopkIdx, float* outputTopkWeights,
                                   [[maybe_unused]] int64_t* outputLayout, int* outputCount, const void* input,
                                   const int64_t* topkIdx, const float* topkWeights, const Workload& workload,
                                   [[maybe_unused]] void* recvBuffer, const DeviceContext& context, int numBlocks,
                                   cudaStream_t stream) {
  topk_expanded::dispatch(output, outputTopkIdx, outputTopkWeights, outputCount, input, topkIdx, topkWeights, workload,
                          context, numBlocks, stream);
}

}  // namespace ep
}  // namespace mscclpp
