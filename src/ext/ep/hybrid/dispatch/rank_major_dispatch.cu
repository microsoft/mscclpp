// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "hybrid/dispatch/common.cuh"

namespace mscclpp {
namespace ep {
namespace hybrid {

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize, bool AsyncDispatch = false>
__global__ __launch_bounds__(DispatchNThreads, 1) void rankMajorDispatchKernel(
    void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx, float* outputTopkWeights,
    int64_t* outputLayout, int* outputCount, const int64_t* topkIndices, const float* topkWeights,
    const void* inputTokens, Workload workload, void* recvBuffer, const DeviceContext* context) {
  workload.epoch_ =
      beginRankMajorDispatchEpoch(WorkspaceView(context->workspace_, context->numRanks_, workload.numExperts_));
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

size_t workspaceSize(int numRanks, int numExperts, int maxTokensPerRank, int numTopk) {
  return workspaceBytes(numRanks, numExperts, maxTokensPerRank, numTopk);
}

void rankMajorDispatch(void* output, int* outputTopkIdx, float* outputTopkWeights, int* outputCount, const void* input,
                       const int64_t* topkIdx, const float* topkWeights, const Workload& workload, void* recvBuffer,
                       const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  void* outputScales = nullptr;
  int* outputSrcInfo = nullptr;
  int64_t* outputLayout = nullptr;
  if (context.gpuNetIoAsyncDispatch_) {
    EP_HOST_ASSERT(context.gpuNetIo_ != nullptr && context.gpuNetIoDispatchPeers_ != nullptr);
    EP_HOST_ASSERT(context.gpuNetIoBatchedDispatch_ && context.gpuNetIoPipelineChunkRows_ > 0);
    EP_HOST_ASSERT(numBlocks > context.numRanks_ + 1);
    const WorkspaceView workspace(context.workspace_, context.numRanks_, workload.numExperts_);
    MSCCLPP_CUDATHROW(cudaMemsetAsync(workspace.dispatchRankPayloadSlots_, 0,
                                      static_cast<size_t>(context.numRanks_) * sizeof(int), stream));
    const size_t counters = static_cast<size_t>(context.numRanks_) *
                            configCellDiv(workload.maxTokensPerRank_, context.gpuNetIoPipelineChunkRows_);
    MSCCLPP_CUDATHROW(cudaMemsetAsync(context.gpuNetIoChunkReady_, 0, counters * sizeof(int), stream));
    MSCCLPP_CUDATHROW(cudaMemsetAsync(context.gpuNetIoDispatchPeers_, 0,
                                      static_cast<size_t>(context.numRanks_) * sizeof(GpuNetIoDispatchPeerState),
                                      stream));
    return dispatchAlgorithm<DispatchLayout::RANK_MAJOR, RankMajorAsyncDispatchKernelSelector>(
        output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
        topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
  }
  dispatchAlgorithm<DispatchLayout::RANK_MAJOR, RankMajorDispatchKernelSelector>(
      output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input, topkIdx,
      topkWeights, workload, recvBuffer, context, numBlocks, stream);
}

}  // namespace hybrid
}  // namespace ep
}  // namespace mscclpp
