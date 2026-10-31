// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "combine/common.cuh"
#include "combine/hybrid_combine.cuh"

namespace mscclpp {
namespace ep {

template <CombineMode Mode, int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout>
MSCCLPP_DEVICE_INLINE void hybridCombineBody(void* output, const void* expertOutput,
                                             const int64_t* __restrict__ topkIndices,
                                             const float* __restrict__ topkWeights, const int* srcInfo,
                                             const int64_t* layoutRange, Workload workload, void* combineRecvBuffer,
                                             const void* dispatchRecvBuffer, const DeviceContext* context) {
#if defined(MSCCLPP_USE_GPUNETIO) && MSCCLPP_BULK_AVAILABLE
  extern __shared__ __align__(128) uint8_t sharedMemory[];
  const int nTokens = workload.numTokens_;
  const int nExperts = workload.numExperts_;
  const int nRanks = context->numRanks_;
  const int nTopk = workload.numTopk_;
  const int maxTokensPerRank = workload.maxTokensPerRank_;
  const TransportView transport(context);
  WorkspaceView workspaceView(context->workspace_, nRanks, nExperts);

  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    static_assert(Mode == CombineMode::RANK_LOCAL_REDUCE || Mode == CombineMode::DIRECT_SEND);
    static_assert(DispatchType == DispatchDataType::BF16);
#if defined(MSCCLPP_USE_GPUNETIO)
    // Cross-domain active: use the GPUNetIO PUSH combine (NVLink peers via mapped
    // read, cross-domain peers via landing region) instead of the pull/gather.
    if (transport.gpuNetIo_ != nullptr) {
      signalRankMajorCombineLocalStart(transport, nRanks);
      sendRankMajorCombinePush<Hidden>(expertOutput, nRanks, maxTokensPerRank, transport, workspaceView,
                                       workload.epoch_);
      if (blockIdx.x == 0) {
        if (static_cast<int>(threadIdx.x) < WARP_SIZE) {
          publishRankMajorCombinePushReady(topkIndices, nTokens, nTopk, nExperts / nRanks, nRanks, workload.epoch_,
                                           transport, workspaceView);
        }
        __syncthreads();
      }
      recvRankMajorRemotePartialsTma<Hidden, Mode, DispatchDataType::BF16, true>(
          output, expertOutput, topkIndices, nullptr, nTokens, nTopk, nExperts, nRanks, maxTokensPerRank,
          workload.epoch_, transport, workspaceView, sharedMemory, 1.0f);
      drainRankMajorCombinePush(nRanks, transport, workspaceView, workload.epoch_);
      workspaceView.combineSyncer_->sync(gridDim.x);
      if (blockIdx.x == 0) {
        const int peer = static_cast<int>(threadIdx.x);
        if (peer < nRanks && !transport.isSelf(peer)) {
          if (transport.isNvlinkPeer(peer)) {
            transport.baseMemoryChannels_[peer].relaxedSignal();
            transport.baseMemoryChannels_[peer].relaxedWait(-1);
          }
        }
      }
      return;
    }
#endif  // defined(MSCCLPP_USE_GPUNETIO)
  }
#endif  // defined(MSCCLPP_USE_GPUNETIO) && MSCCLPP_BULK_AVAILABLE
}

template <int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout>
__global__ __launch_bounds__(CombineNThreads,
                             1) void hybridCombineKernel(void* output, const void* expertOutput,
                                                         const int64_t* topkIndices, const float* topkWeights,
                                                         const int* srcInfo, const int64_t* layoutRange,
                                                         Workload workload, void* combineRecvBuffer,
                                                         const void* dispatchRecvBuffer, const DeviceContext* context) {
  workload.epoch_ =
      beginRankMajorCombineEpoch(WorkspaceView(context->workspace_, context->numRanks_, workload.numExperts_));
  hybridCombineBody<CombineMode::RANK_LOCAL_REDUCE, Hidden, DispatchType, ScaleBlockSize, Layout>(
      output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload, combineRecvBuffer,
      dispatchRecvBuffer, context);
}

struct HybridCombineKernelSelector {
  template <int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout>
  static auto get() {
    return hybridCombineKernel<Hidden, DispatchType, ScaleBlockSize, Layout>;
  }
};

void hybridCombine(void* output, const void* input, const int64_t* topkIdx, const Workload& workload, void* recvBuffer,
                   void* dispatchRecvBuffer, const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  EP_HOST_ASSERT(workload.outputLayout_ == DispatchLayout::RANK_MAJOR);
  if (context.gpuNetIoAggregateCombine_) {
    EP_HOST_ASSERT(context.gpuNetIo_ != nullptr && workload.numTopk_ <= MaxNumTopk);
    EP_HOST_ASSERT(workload.outputLayout_ == DispatchLayout::RANK_MAJOR &&
                   workload.dispatchDataType_ == DispatchDataType::BF16);
    return combineAlgorithm<CombineMode::RANK_LOCAL_REDUCE, HybridAggregateCombineKernelSelector, true>(
        output, input, topkIdx, nullptr, nullptr, nullptr, workload, recvBuffer, dispatchRecvBuffer, context, numBlocks,
        stream);
  }
  combineAlgorithm<CombineMode::RANK_LOCAL_REDUCE, HybridCombineKernelSelector, true>(
      output, input, topkIdx, nullptr, nullptr, nullptr, workload, recvBuffer, dispatchRecvBuffer, context, numBlocks,
      stream);
}

}  // namespace ep
}  // namespace mscclpp
