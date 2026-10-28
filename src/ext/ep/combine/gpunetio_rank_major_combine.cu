// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "combine/aggregate_combine.cuh"
#include "combine/common.cuh"

namespace mscclpp {
namespace ep {

template <CombineMode Mode, int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout>
MSCCLPP_DEVICE_INLINE void gpuNetIoCombineBody(void* output, const void* expertOutput,
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
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
      const long long syncStart = clock64();
#endif
      signalRankMajorCombineLocalStart(transport, nRanks);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
      const long long pushStart = clock64();
#endif
      sendRankMajorCombinePush<Hidden>(expertOutput, nRanks, maxTokensPerRank, transport, workspaceView,
                                       workload.epoch_);
      if (blockIdx.x == 0) {
        if (static_cast<int>(threadIdx.x) < WARP_SIZE) {
          publishRankMajorCombinePushReady(topkIndices, nTokens, nTopk, nExperts / nRanks, nRanks, workload.epoch_,
                                           transport, workspaceView);
        }
        __syncthreads();
      }
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
      const long long recvStart = clock64();
#endif
      recvRankMajorRemotePartialsTma<Hidden, Mode, DispatchDataType::BF16, true>(
          output, expertOutput, topkIndices, nullptr, nTokens, nTopk, nExperts, nRanks, maxTokensPerRank,
          workload.epoch_, transport, workspaceView, sharedMemory, 1.0f);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
      const long long drainStart = clock64();
#endif
      drainRankMajorCombinePush(nRanks, transport, workspaceView, workload.epoch_);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
      const long long syncEndStart = clock64();
#endif
      workspaceView.combineSyncer_->sync(gridDim.x);
      if (blockIdx.x == 0) {
        const int peer = static_cast<int>(threadIdx.x);
        if (peer < nRanks && !transport.isSelf(peer)) {
          if (transport.isNvlinkPeer(peer)) {
            transport.baseMemoryChannels_[peer].relaxedSignal();
            transport.baseMemoryChannels_[peer].relaxedWait(-1);
          } else if (transport.gpuNetIoCombineChunkRows_ > 0) {
            // Every worker has consumed this launch's chunks (grid sync above); advance the per-QP chunk bases by the
            // chunks each cross-domain peer returned, i.e. the rows this rank dispatched to it.
            const int rows = workspaceView.dispatchRankPayloadCompletions_[peer];
            const int nChunks = (rows + transport.gpuNetIoCombineChunkRows_ - 1) / transport.gpuNetIoCombineChunkRows_;
            const int nQp = transport.gpuNetIo_->numQpsPerPeer;
            for (int stripe = 0; stripe < nQp && stripe < nChunks; ++stripe) {
              transport
                  .gpuNetIoCombineChunkBase_[static_cast<size_t>(peer) * GpuNetIoMaxQpsPerPeer +
                                             rankMajorCombineChunkQp(transport.gpuNetIo_, transport.rank_, stripe)] +=
                  rankMajorCombineChunksOnStripe(nChunks, stripe, nQp);
            }
          }
        }
      }
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
      const int block = static_cast<int>(blockIdx.x);
      const int blocks = static_cast<int>(gridDim.x);
      if (threadIdx.x == 0 && (block == 0 || block == 1 || block == blocks / 2 || block == blocks - 1))
        printf(
            "[GINTIME-CMB] r=%d ep=%u blk=%d start_cyc=%lld push_cyc=%lld recv_cyc=%lld "
            "drain_cyc=%lld sync1_cyc=%lld\n",
            transport.rank_, workload.epoch_, block, pushStart - syncStart, recvStart - pushStart,
            drainStart - recvStart, syncEndStart - drainStart, clock64() - syncEndStart);
#endif
      return;
    }
#endif  // defined(MSCCLPP_USE_GPUNETIO)
  }
#endif  // defined(MSCCLPP_USE_GPUNETIO) && MSCCLPP_BULK_AVAILABLE
}

template <int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout>
__global__ __launch_bounds__(CombineNThreads, 1) void rankLocalReduceCombineKernel(
    void* output, const void* expertOutput, const int64_t* topkIndices, const float* topkWeights, const int* srcInfo,
    const int64_t* layoutRange, Workload workload, void* combineRecvBuffer, const void* dispatchRecvBuffer,
    const DeviceContext* context) {
  workload.epoch_ =
      beginRankMajorCombineEpoch(WorkspaceView(context->workspace_, context->numRanks_, workload.numExperts_));
  gpuNetIoCombineBody<CombineMode::RANK_LOCAL_REDUCE, Hidden, DispatchType, ScaleBlockSize, Layout>(
      output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload, combineRecvBuffer,
      dispatchRecvBuffer, context);
}

struct RankLocalReduceCombineKernelSelector {
  template <int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout>
  static auto get() {
    return rankLocalReduceCombineKernel<Hidden, DispatchType, ScaleBlockSize, Layout>;
  }
};

namespace {

void runRankLocalReduce(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights,
                        const int* srcInfo, const int64_t* layoutRange, const Workload& workload, void* recvBuffer,
                        void* dispatchRecvBuffer, const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  if (context.gpuNetIoAggregateCombine_) {
    EP_HOST_ASSERT(context.gpuNetIoSendDistribute_ && workload.numTopk_ <= MaxNumTopk);
    EP_HOST_ASSERT(workload.outputLayout_ == DispatchLayout::RANK_MAJOR &&
                   workload.dispatchDataType_ == DispatchDataType::BF16);
    return combineAlgorithm<CombineMode::RANK_LOCAL_REDUCE, RankMajorAggregateCombineKernelSelector, true>(
        output, input, topkIdx, topkWeights, srcInfo, layoutRange, workload, recvBuffer, dispatchRecvBuffer, context,
        numBlocks, stream);
  }
  combineAlgorithm<CombineMode::RANK_LOCAL_REDUCE, RankLocalReduceCombineKernelSelector, true>(
      output, input, topkIdx, topkWeights, srcInfo, layoutRange, workload, recvBuffer, dispatchRecvBuffer, context,
      numBlocks, stream);
}

}  // namespace

void gpuNetIoRankMajorCombine(void* output, const void* input, const int64_t* topkIdx, const Workload& workload,
                              void* recvBuffer, void* dispatchRecvBuffer, const DeviceContext& context, int numBlocks,
                              cudaStream_t stream) {
  EP_HOST_ASSERT(workload.outputLayout_ == DispatchLayout::RANK_MAJOR);
  runRankLocalReduce(output, input, topkIdx, nullptr, nullptr, nullptr, workload, recvBuffer, dispatchRecvBuffer,
                     context, numBlocks, stream);
}

}  // namespace ep
}  // namespace mscclpp
