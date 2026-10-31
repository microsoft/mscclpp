// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "dispatch/hybrid_dispatch.cuh"

namespace mscclpp::ep {

void hybridDispatch(void* output, int* outputTopkIdx, float* outputTopkWeights, int* outputCount, const void* input,
                    const int64_t* topkIdx, const float* topkWeights, const Workload& workload, void* recvBuffer,
                    const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  EP_HOST_ASSERT(context.gpuNetIo_ != nullptr);
  EP_HOST_ASSERT(numBlocks > context.numRanks_ + GpuNetIoDistributeForwardBlocks + 1);
  if (!context.gpuNetIoAggregateCombine_) {
    const WorkspaceView workspace(context.workspace_, context.numRanks_, workload.numExperts_);
    MSCCLPP_CUDATHROW(cudaMemsetAsync(workspace.dispatchRankPayloadSlots_, 0,
                                      static_cast<size_t>(context.numRanks_) * 2 * sizeof(int), stream));
    MSCCLPP_CUDATHROW(cudaMemsetAsync(
        context.gpuNetIoDistributeScratch_, 0,
        (3 + configCellDiv(context.gpuNetIoDistributeCapacity_, context.gpuNetIoPipelineChunkRows_)) * sizeof(int),
        stream));
  }
  dispatchAlgorithm<DispatchLayout::RANK_MAJOR, HybridSendDistributeDispatchKernelSelector, true>(
      output, nullptr, nullptr, outputTopkIdx, outputTopkWeights, nullptr, outputCount, input, topkIdx, topkWeights,
      workload, recvBuffer, context, numBlocks, stream);
}

}  // namespace mscclpp::ep
