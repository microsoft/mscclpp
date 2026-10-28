// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "dispatch/common.cuh"
#include "dispatch/send_distribute.cuh"

namespace mscclpp {
namespace ep {

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize, DispatchLayout Layout, bool AsyncDispatch = false>
MSCCLPP_DEVICE_INLINE void gpuNetIoDispatchBody(void* output, void* outputScales, int* outputSrcInfo,
                                                int* outputTopkIdx, float* outputTopkWeights, int64_t* outputLayout,
                                                int* outputCount, const int64_t* __restrict__ topkIndices,
                                                const float* __restrict__ topkWeights, const void* inputTokens,
                                                Workload workload, void* recvBuffer, const DeviceContext* context) {
#if defined(MSCCLPP_USE_GPUNETIO) && MSCCLPP_BULK_AVAILABLE
  extern __shared__ __align__(128) uint8_t sharedMemory[];
  auto* sharedMem = reinterpret_cast<int*>(sharedMemory);
  const int nWorkerBlocks = static_cast<int>(gridDim.x) - DispatchControlBlocks;
  const int nExperts = workload.numExperts_;
  const int nRanks = context->numRanks_;
  const int nTokens = workload.numTokens_;
  const int nTopk = workload.numTopk_;
  const int invalidTokenExpertId = workload.invalidTokenExpertId_;
  const int maxTokensPerRank = workload.maxTokensPerRank_;
  const TransportView transport(context);
  const uint32_t epoch = workload.epoch_;
#if defined(MSCCLPP_USE_GPUNETIO)
  WorkspaceView workspaceView(context->workspace_, nRanks, nExperts);
  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    if (transport.gpuNetIo_ != nullptr) {
      if (transport.gpuNetIo_->numQpsPerPeer < 1 || transport.gpuNetIo_->numQpsPerPeer > GpuNetIoMaxQpsPerPeer)
        __trap();
      if (transport.gpuNetIoDeferFlush_) {
        // The previous dispatch skipped its send drain: its WQEs may still read the staging rows, count
        // scratch and marker counters this dispatch is about to rewrite. Its tickets are normally long done.
        auto* network = transport.gpuNetIo_;
        const int nQp = network->numQpsPerPeer;
        for (int index = threadIdx.x; index < nRanks * nQp; index += blockDim.x) {
          const int peer = index / nQp, qpIndex = index % nQp;
          if (transport.isNvlinkPeer(peer)) continue;
          network->waitTicket(peer, transport.gpuNetIoDispatchTickets_[peer * GpuNetIoMaxQpsPerPeer + qpIndex],
                              qpIndex);
        }
      }
      __syncthreads();
    }
  }
#endif
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  const long long sendStart = clock64();
  long long flushCycles = 0, barrierCycles = 0, recvCycles = 0;
#endif
  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    static_assert(DataType == DispatchDataType::BF16);
    dispatchSendRankMajor<Hidden, AsyncDispatch>(
        output, outputTopkIdx, outputTopkWeights, inputTokens, transport, nExperts, nRanks, topkIndices, topkWeights,
        nTokens, nTopk, invalidTokenExpertId, maxTokensPerRank, recvBuffer, context->workspace_, epoch, sharedMem);
  }
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  const long long sendCycles = clock64() - sendStart;
#endif
#if defined(MSCCLPP_USE_GPUNETIO)
  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    if (transport.gpuNetIo_ != nullptr) {
      auto* network = transport.gpuNetIo_;
      const int nQp = network->numQpsPerPeer;
      if constexpr (AsyncDispatch) {
        if (static_cast<int>(blockIdx.x) == nWorkerBlocks + 1) {
          for (int index = threadIdx.x; index < nRanks * nQp; index += blockDim.x) {
            const int peer = index / nQp, queue = index % nQp;
            if (transport.isNvlinkPeer(peer)) continue;
            while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(&transport.gpuNetIoDispatchPeers_[peer].markersPosted,
                                                                  mscclpp::memoryOrderAcquire) == 0) {
            }
            network->flush(peer, queue);
          }
        }
      } else {
        joinRankMajorGpuNetIoProducers(context->gpuNetIoDispatchDone_, nWorkerBlocks);
        if (static_cast<int>(blockIdx.x) == nWorkerBlocks + 1) {
          for (int peer = threadIdx.x; peer < nRanks; peer += blockDim.x) {
            if (!transport.isNvlinkPeer(peer)) {
              workspaceView.dispatchRankPayloadCompletions_[peer] = workspaceView.dispatchRankPayloadSlots_[peer];
              workspaceView.dispatchRankPayloadSlots_[peer] = 0;
            }
          }
          if (threadIdx.x == 0) {
            *context->gpuNetIoDispatchDone_ = 0;
          }
          for (int index = threadIdx.x; index < nRanks * nQp; index += blockDim.x) {
            const int peer = index / nQp, qpIndex = index % nQp;
            if (!transport.isNvlinkPeer(peer)) postRankMajorGpuNetIoMarker(transport, peer, qpIndex);
          }
          __syncthreads();
          for (int index = threadIdx.x; index < nRanks * nQp; index += blockDim.x) {
            const int peer = index / nQp, qpIndex = index % nQp;
            if (!transport.isNvlinkPeer(peer)) {
              if (transport.gpuNetIoDeferFlush_)
                transport.gpuNetIoDispatchTickets_[peer * GpuNetIoMaxQpsPerPeer + qpIndex] =
                    network->reservedTicket(peer, qpIndex);
              else
                network->flush(peer, qpIndex);
            }
          }
          __syncthreads();
        }
      }
    }
  }
#endif

  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    if (static_cast<int>(blockIdx.x) < nRanks) {
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
      const long long recvStart = clock64();
#endif
      dispatchRecvRankMajor(outputTopkIdx, outputTopkWeights, outputCount, transport, nExperts, nRanks, nTopk,
                            maxTokensPerRank, invalidTokenExpertId, recvBuffer, context->workspace_, epoch, sharedMem);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
      recvCycles = clock64() - recvStart;
#endif
    }
  }
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    const int block = blockIdx.x;
    if (threadIdx.x == 0 && (block == 0 || block == 1 || block == nWorkerBlocks / 2 || block >= nWorkerBlocks))
      printf("[GINTIME-BLK] r=%d ep=%u blk=%d send_cyc=%lld flush_cyc=%lld bar_cyc=%lld recv_cyc=%lld\n",
             transport.rank_, epoch, block, sendCycles, flushCycles, barrierCycles, recvCycles);
  }
#endif
#endif  // defined(MSCCLPP_USE_GPUNETIO) && MSCCLPP_BULK_AVAILABLE
}

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize, bool AsyncDispatch = false>
__global__ __launch_bounds__(DispatchNThreads, 1) void rankMajorDispatchKernel(
    void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx, float* outputTopkWeights,
    int64_t* outputLayout, int* outputCount, const int64_t* topkIndices, const float* topkWeights,
    const void* inputTokens, Workload workload, void* recvBuffer, const DeviceContext* context) {
  workload.epoch_ =
      beginRankMajorDispatchEpoch(WorkspaceView(context->workspace_, context->numRanks_, workload.numExperts_));
  gpuNetIoDispatchBody<Hidden, DataType, ScaleBlockSize, DispatchLayout::RANK_MAJOR, AsyncDispatch>(
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

void gpuNetIoRankMajorDispatch(void* output, int* outputTopkIdx, float* outputTopkWeights, int* outputCount,
                               const void* input, const int64_t* topkIdx, const float* topkWeights,
                               const Workload& workload, void* recvBuffer, const DeviceContext& context, int numBlocks,
                               cudaStream_t stream) {
  void* outputScales = nullptr;
  int* outputSrcInfo = nullptr;
  int64_t* outputLayout = nullptr;
  if (context.gpuNetIoSendDistribute_) {
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
    return dispatchAlgorithm<DispatchLayout::RANK_MAJOR, RankMajorSendDistributeKernelSelector, true>(
        output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
        topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
  }
  if (context.gpuNetIoBatchedDispatch_) {
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
    return dispatchAlgorithm<DispatchLayout::RANK_MAJOR, RankMajorAsyncDispatchKernelSelector, true>(
        output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
        topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
  }
  dispatchAlgorithm<DispatchLayout::RANK_MAJOR, RankMajorDispatchKernelSelector, true>(
      output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input, topkIdx,
      topkWeights, workload, recvBuffer, context, numBlocks, stream);
}

}  // namespace ep
}  // namespace mscclpp
