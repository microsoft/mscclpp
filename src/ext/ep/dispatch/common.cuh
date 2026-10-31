// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_DISPATCH_COMMON_CUH_
#define MSCCLPP_EP_DISPATCH_COMMON_CUH_
#include <mscclpp/bulk_device.hpp>
#include <mscclpp/memory_channel_device.hpp>

#include "common/device_helpers.cuh"
#include "common/latency.cuh"
#include "exception.hpp"
#include "kernels.hpp"
#if defined(MSCCLPP_USE_GPUNETIO)
#include "gpu_net_io.hpp"
#endif

namespace mscclpp {
namespace ep {

#if MSCCLPP_BULK_AVAILABLE

struct RankMajorRoute {
  int dstRank;
  int destinationSlot;
  bool isLeader;
};

MSCCLPP_DEVICE_INLINE RankMajorRoute prepareRankMajorRoute(WorkspaceView& workspaceView,
                                                           const int64_t* __restrict__ topkIndices, int tokenIdx,
                                                           int nTopk, int nLocalExperts, int maxTokensPerRank,
                                                           int laneId) {
  const int routedExpertIdx = laneId < nTopk ? static_cast<int>(topkIndices[tokenIdx * nTopk + laneId]) : -1;
  const int dstRank = routedExpertIdx >= 0 ? routedExpertIdx / nLocalExperts : -1;
  const bool firstLaneForRank = isFirstLaneForRank(dstRank, laneId);
  int destinationSlot = -1;
  if (dstRank >= 0 && firstLaneForRank) {
    destinationSlot = atomicAdd(workspaceView.dispatchRankPayloadSlots_ + dstRank, 1);
    EP_DEVICE_ASSERT(destinationSlot < maxTokensPerRank);
  }

  const unsigned matchMask = __match_any_sync(0xffffffff, dstRank);
  const int firstLane = __ffs(matchMask) - 1;
  destinationSlot = __shfl_sync(0xffffffff, destinationSlot, firstLane);
  if (laneId < nTopk) {
    workspaceView.rankMajorSendIndices_[tokenIdx * nTopk + laneId] = dstRank >= 0 ? destinationSlot : -1;
  }
  return {dstRank, destinationSlot, firstLaneForRank};
}

template <int Hidden, DispatchDataType DataType>
MSCCLPP_DEVICE_INLINE void issueRankMajorTokenStore(void* output, const TransportView& transport, int destinationSlot,
                                                    int maxTokensPerRank, void* stagedToken, int destinationRank,
                                                    void* outputScales = nullptr) {
  if (destinationSlot < 0) return;

  using ElementType = DispatchElementType<DataType>;
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(ElementType);
  const size_t row = static_cast<size_t>(transport.rank_) * maxTokensPerRank + destinationSlot;
  void* destinationBuffer = transport.mappedBuffer(output, destinationRank);
  auto* destinationRow = reinterpret_cast<uint8_t*>(destinationBuffer) + row * HiddenBytes;
  mscclpp::bulkStore(destinationRow, stagedToken, static_cast<uint32_t>(HiddenBytes));
  if constexpr (DataType == DispatchDataType::MXFP8_E4M3) {
    constexpr int ScaleBytes = Hidden / 32;
    auto* destinationScales =
        static_cast<uint8_t*>(transport.mappedBuffer(outputScales, destinationRank)) + row * ScaleBytes;
    auto* stagedScales = static_cast<uint8_t*>(stagedToken) + HiddenBytes;
    if constexpr (ScaleBytes % 16 == 0) {
      mscclpp::bulkStore(destinationScales, stagedScales, ScaleBytes);
    } else {
      // Preserve contiguous scales for widths such as 4352, whose rows are not TMA-aligned.
      static_assert(ScaleBytes % sizeof(uint64_t) == 0);
      for (int i = 0; i < ScaleBytes / sizeof(uint64_t); ++i) {
        reinterpret_cast<uint64_t*>(destinationScales)[i] = reinterpret_cast<const uint64_t*>(stagedScales)[i];
      }
    }
  }
  mscclpp::bulkStoreCommit();
}

MSCCLPP_DEVICE_INLINE void completeRankMajorTokenStore(WorkspaceView& workspaceView, int destinationRank) {
  if (destinationRank < 0) return;
  mscclpp::bulkStoreWait();
  (void)mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(
      workspaceView.dispatchRankPayloadCompletions_ + destinationRank, 1, mscclpp::memoryOrderRelease);
}

MSCCLPP_DEVICE_INLINE void publishDispatchPayloads(const TransportView& transport, const int* rankTokenCounts,
                                                   int nRanks, WorkspaceView workspaceView) {
  const int threadId = static_cast<int>(threadIdx.x);
  for (int dstRank = threadId; dstRank < nRanks; dstRank += blockDim.x) {
    const int expectedPayloadCount = rankTokenCounts[dstRank];
    if (expectedPayloadCount > 0) {
      while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(workspaceView.dispatchRankPayloadCompletions_ + dstRank,
                                                            mscclpp::memoryOrderAcquire) != expectedPayloadCount);
    }
    workspaceView.dispatchRankPayloadSlots_[dstRank] = 0;
    workspaceView.dispatchRankPayloadCompletions_[dstRank] = 0;
    if (expectedPayloadCount == 0) continue;
    if (transport.isSelf(dstRank)) {
      workspaceView.dispatchLocalPayloadReady_->release();
    } else {
      transport.baseMemoryChannels_[dstRank].signal();
    }
  }
}

#if defined(MSCCLPP_USE_GPUNETIO)
// Cross-domain metadata travels with the network payload, not the mapped stores.
MSCCLPP_DEVICE_INLINE void sendRankMajorMetadataNvlink(
    const TransportView& transport, int* outputTopkIdx, float* outputTopkWeights,
    const int64_t* __restrict__ topkIndices, const float* __restrict__ topkWeights, const RankMajorRoute& route,
    int tokenIdx, int nTopk, int nLocalExperts, int maxTokensPerRank, int invalidTokenExpertId) {
  const int laneId = getLaneId();
  const int candidateExpert =
      laneId < nTopk ? static_cast<int>(topkIndices[tokenIdx * nTopk + laneId]) : invalidTokenExpertId;
  const float candidateWeight =
      laneId < nTopk ? (topkWeights == nullptr ? 1.0f : topkWeights[tokenIdx * nTopk + laneId]) : 0.0f;
  unsigned int leaderMask = __ballot_sync(0xffffffff, route.dstRank >= 0 && route.isLeader);
  while (leaderMask != 0) {
    const int leaderLane = __ffs(leaderMask) - 1;
    const int destinationRank = __shfl_sync(0xffffffff, route.dstRank, leaderLane);
    const int destinationSlot = __shfl_sync(0xffffffff, route.destinationSlot, leaderLane);
    leaderMask &= leaderMask - 1;
    if (!transport.isNvlinkPeer(destinationRank)) continue;
    if (laneId < nTopk) {
      auto* destinationTopkIdx = reinterpret_cast<int*>(transport.mappedBuffer(outputTopkIdx, destinationRank));
      auto* destinationTopkWeights =
          reinterpret_cast<float*>(transport.mappedBuffer(outputTopkWeights, destinationRank));
      const size_t outputIdx =
          (static_cast<size_t>(transport.rank_) * maxTokensPerRank + destinationSlot) * nTopk + laneId;
      const bool isLocal = candidateExpert >= 0 && candidateExpert / nLocalExperts == destinationRank;
      destinationTopkIdx[outputIdx] = isLocal ? candidateExpert : invalidTokenExpertId;
      destinationTopkWeights[outputIdx] = isLocal ? candidateWeight : 0.0f;
    }
  }
  __syncwarp();
}
#endif

MSCCLPP_DEVICE_INLINE void countRankMajorRoutes(int* rankTokenCounts, const int64_t* __restrict__ topkIndices,
                                                int nTokens, int nTopk, int nRanks, int nExperts) {
  const int threadId = static_cast<int>(threadIdx.x);
  const int warpId = threadId / WARP_SIZE;
  const int laneId = getLaneId();
  const int nLocalExperts = nExperts / nRanks;
  for (int rankIdx = threadId; rankIdx < nRanks; rankIdx += blockDim.x) rankTokenCounts[rankIdx] = 0;
  __syncthreads();
  for (int tokenIdx = warpId; tokenIdx < nTokens; tokenIdx += DispatchNWarps) {
    const int routedExpertIdx = laneId < nTopk ? static_cast<int>(topkIndices[tokenIdx * nTopk + laneId]) : -1;
    const int dstRank = routedExpertIdx >= 0 ? routedExpertIdx / nLocalExperts : -1;
    if (isFirstLaneForRank(dstRank, laneId) && dstRank >= 0) {
      atomicAdd_block(rankTokenCounts + dstRank, 1);
    }
  }
  __syncthreads();
}

#endif  // MSCCLPP_BULK_AVAILABLE

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize, DispatchLayout Layout, typename KernelSelector,
          bool Network = false>
inline void dispatchHiddenMode(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                               float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                               const int64_t* topkIdx, const float* topkWeights, const Workload& workload,
                               void* recvBuffer, const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  static_assert(Hidden == 1024 || Hidden == 2048 || Hidden == 4096 || Hidden == 4352 || Hidden == 5120 ||
                Hidden == 6656 || Hidden == 7168 || Hidden == 8192 || Hidden == 8704 || Hidden == 9216);
  using OutputType = DispatchElementType<DataType>;
  constexpr int NRecvTmaWorkers = tmaWorkerCount<Hidden, OutputType, DispatchMaxNRecvTmaWorkers>();
  static_assert(NRecvTmaWorkers > 0);
  const int nExperts = workload.numExperts_;
  const int nRanks = context.numRanks_;
  const int nTopk = workload.numTopk_;

  size_t dynamicSharedBytes = dispatchSharedBytes<Hidden, DataType, ScaleBlockSize, Network>(nRanks, nExperts, nTopk);
  if constexpr (Network) {
    const size_t forwardSharedBytes =
        dispatchSharedControlBytes(nRanks, true) +
        GpuNetIoDistributeForwardWarps *
            (dispatchPayloadStride<DispatchDataType::BF16>(Hidden, nTopk, 0) + sizeof(mscclpp::BulkBarrier));
    dynamicSharedBytes = std::max(dynamicSharedBytes, forwardSharedBytes);
  }
  static thread_local KernelConfigCache kernelConfig;
  auto kernel = KernelSelector::template get<Hidden, DataType, ScaleBlockSize>();
  const int residentBlocks = configureKernel(kernel, DispatchNThreads, dynamicSharedBytes, context, kernelConfig);
  EP_HOST_ASSERT(residentBlocks >= numBlocks);
  kernel<<<dim3(numBlocks), dim3(DispatchNThreads), dynamicSharedBytes, stream>>>(
      output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, topkIdx,
      topkWeights, input, workload, recvBuffer, context.devicePtr_);
  MSCCLPP_CUDATHROW(cudaGetLastError());
}

template <int Hidden, DispatchLayout Layout, typename KernelSelector, bool Network = false>
inline void dispatchHidden(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                           float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                           const int64_t* topkIdx, const float* topkWeights, const Workload& workload, void* recvBuffer,
                           const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  if constexpr (Network) {
    static_assert(Layout == DispatchLayout::RANK_MAJOR);
    EP_HOST_ASSERT(workload.dispatchDataType_ == DispatchDataType::BF16);
    return dispatchHiddenMode<Hidden, DispatchDataType::BF16, 0, Layout, KernelSelector, true>(
        output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
        topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
  } else if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    switch (workload.dispatchDataType_) {
      case DispatchDataType::BF16:
        return dispatchHiddenMode<Hidden, DispatchDataType::BF16, 0, Layout, KernelSelector>(
            output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
            topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
      case DispatchDataType::FP8_E4M3:
        break;
      case DispatchDataType::FP8_E4M3_STATIC:
        return dispatchHiddenMode<Hidden, DispatchDataType::FP8_E4M3_STATIC, 0, Layout, KernelSelector>(
            output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
            topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
      case DispatchDataType::MXFP8_E4M3:
        return dispatchHiddenMode<Hidden, DispatchDataType::MXFP8_E4M3, 32, Layout, KernelSelector>(
            output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
            topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    }
    EP_HOST_ASSERT(false && "unsupported rank-major dispatch data type");
  } else {
    switch (workload.dispatchDataType_) {
      case DispatchDataType::BF16:
        return dispatchHiddenMode<Hidden, DispatchDataType::BF16, 0, Layout, KernelSelector>(
            output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
            topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
      case DispatchDataType::FP8_E4M3:
        return dispatchHiddenMode<Hidden, DispatchDataType::FP8_E4M3, 128, Layout, KernelSelector>(
            output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
            topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
      case DispatchDataType::MXFP8_E4M3:
      case DispatchDataType::FP8_E4M3_STATIC:
        break;
    }
    EP_HOST_ASSERT(false && "unsupported dispatch data type");
  }
}

template <int Hidden>
inline void dispatchLayout(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                           float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                           const int64_t* topkIdx, const float* topkWeights, const Workload& workload, void* recvBuffer,
                           const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  if (workload.outputLayout_ == DispatchLayout::EXPERT_MAJOR) {
    return dispatchHidden<Hidden, DispatchLayout::EXPERT_MAJOR>(
        output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
        topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
  }
  if (workload.outputLayout_ == DispatchLayout::RANK_MAJOR) {
    return dispatchHidden<Hidden, DispatchLayout::RANK_MAJOR>(
        output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
        topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
  }
  EP_HOST_ASSERT(false && "unsupported dispatch layout");
}

template <DispatchLayout Layout, typename KernelSelector, bool Network = false>
inline void dispatchAlgorithm(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                              float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                              const int64_t* topkIdx, const float* topkWeights, const Workload& workload,
                              void* recvBuffer, const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  const int nExperts = workload.numExperts_;
  const int rank = context.rank_;
  const int nRanks = context.numRanks_;
  const int numWorkerBlocks = numBlocks - DispatchControlBlocks;

  EP_HOST_ASSERT(isSupportedRanks(nRanks));
  EP_HOST_ASSERT(nExperts > 0);
  EP_HOST_ASSERT(nExperts % nRanks == 0);
  EP_HOST_ASSERT(rank >= 0 && rank < nRanks);
  EP_HOST_ASSERT(context.channels_ != nullptr);
  EP_HOST_ASSERT(workload.numTokens_ >= 0);
  EP_HOST_ASSERT(workload.numTopk_ > 0 && workload.numTopk_ <= MaxNumTopk);
  EP_HOST_ASSERT(numWorkerBlocks >= nRanks && numWorkerBlocks <= MaxWorkerBlocks);
  EP_HOST_ASSERT(output != nullptr);
  EP_HOST_ASSERT(workload.outputLayout_ == Layout);
  EP_HOST_ASSERT(isSupportedDispatchDataType(workload.dispatchDataType_));
  if constexpr (Layout == DispatchLayout::EXPERT_MAJOR) {
    EP_HOST_ASSERT(workload.dispatchDataType_ == DispatchDataType::BF16 || outputScales != nullptr);
  } else if (workload.dispatchDataType_ == DispatchDataType::FP8_E4M3_STATIC) {
    EP_HOST_ASSERT(outputScales == nullptr);
  } else if (workload.dispatchDataType_ == DispatchDataType::MXFP8_E4M3) {
    EP_HOST_ASSERT(outputScales != nullptr);
  }
  EP_HOST_ASSERT(outputSrcInfo != nullptr || workload.outputLayout_ == DispatchLayout::RANK_MAJOR);
  EP_HOST_ASSERT(outputCount != nullptr);
  EP_HOST_ASSERT(outputLayout != nullptr || workload.outputLayout_ == DispatchLayout::RANK_MAJOR);
  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    EP_HOST_ASSERT(outputTopkIdx != nullptr);
    EP_HOST_ASSERT(outputTopkWeights != nullptr);
  }
  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    EP_HOST_ASSERT(workload.dispatchDataType_ == DispatchDataType::BF16 ||
                   workload.dispatchDataType_ == DispatchDataType::FP8_E4M3_STATIC ||
                   workload.dispatchDataType_ == DispatchDataType::MXFP8_E4M3);
  }
  EP_HOST_ASSERT(workload.numTokens_ == 0 || input != nullptr);
  EP_HOST_ASSERT(workload.numTokens_ == 0 || topkIdx != nullptr);
  EP_HOST_ASSERT(recvBuffer != nullptr);
  EP_HOST_ASSERT(context.localBufferBase_ != nullptr);
  EP_HOST_ASSERT(context.peerBufferBases_ != nullptr);
  EP_HOST_ASSERT(context.workspace_ != nullptr);
  EP_HOST_ASSERT(context.devicePtr_ != nullptr);

  switch (workload.hidden_) {
    case 1024:
      return dispatchHidden<1024, Layout, KernelSelector, Network>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 2048:
      return dispatchHidden<2048, Layout, KernelSelector, Network>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 4096:
      return dispatchHidden<4096, Layout, KernelSelector, Network>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 4352:
      return dispatchHidden<4352, Layout, KernelSelector, Network>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 5120:
      if constexpr (Network) {
        return dispatchHidden<5120, Layout, KernelSelector, true>(
            output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
            topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
      } else {
        EP_HOST_ASSERT(false && "unsupported latency dispatch hidden size");
        return;
      }
    case 6656:
      return dispatchHidden<6656, Layout, KernelSelector, Network>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 7168:
      return dispatchHidden<7168, Layout, KernelSelector, Network>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 8192:
      return dispatchHidden<8192, Layout, KernelSelector, Network>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 8704:
      return dispatchHidden<8704, Layout, KernelSelector, Network>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 9216:
      return dispatchHidden<9216, Layout, KernelSelector, Network>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    default:
      EP_HOST_ASSERT(false && "unsupported latency dispatch hidden size");
  }
}

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_DISPATCH_COMMON_CUH_
