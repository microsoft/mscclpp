// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_DISPATCH_COMMON_CUH_
#define MSCCLPP_EP_DISPATCH_COMMON_CUH_
#include <mscclpp/bulk_device.hpp>
#include <mscclpp/memory_channel_device.hpp>

#include "common/device_helpers.cuh"
#include "common/gateway.cuh"
#include "common/latency.cuh"
#include "exception.hpp"
#include "kernels.hpp"

namespace mscclpp {
namespace ep {

#if MSCCLPP_BULK_AVAILABLE

struct RankMajorRoute {
  int dstRank;
  int destinationSlot;
  bool isLeader;
  int expertIdx = -1;
};

MSCCLPP_DEVICE_INLINE RankMajorRoute prepareRankMajorRoute(WorkspaceView& workspaceView,
                                                           const int64_t* __restrict__ topkIndices, int tokenIdx,
                                                           int nTopk, int nLocalExperts, int maxTokensPerRank,
                                                           int laneId, int nExperts = (1 << 30)) {
  const int64_t rawExpertIdx = laneId < nTopk ? topkIndices[tokenIdx * nTopk + laneId] : -1;
  const int routedExpertIdx = rawExpertIdx >= 0 && rawExpertIdx < nExperts ? static_cast<int>(rawExpertIdx) : -1;
  const int dstRank = routedExpertIdx >= 0 ? routedExpertIdx / nLocalExperts : -1;
  const int firstLane = __ffs(__match_any_sync(0xffffffff, dstRank)) - 1;
  const bool firstLaneForRank = firstLane == laneId;
  int destinationSlot = -1;
  if (dstRank >= 0 && firstLaneForRank) {
    destinationSlot = atomicAdd(workspaceView.dispatchRankPayloadSlots_ + dstRank, 1);
    EP_DEVICE_ASSERT(destinationSlot < maxTokensPerRank);
  }

  destinationSlot = __shfl_sync(0xffffffff, destinationSlot, firstLane);
  if (laneId < nTopk) {
    workspaceView.rankMajorSendIndices_[tokenIdx * nTopk + laneId] = dstRank >= 0 ? destinationSlot : -1;
  }
  return {dstRank, destinationSlot, firstLaneForRank, routedExpertIdx};
}

MSCCLPP_DEVICE_INLINE void publishDispatchPayloads(const TransportView& transport, const int* rankTokenCounts,
                                                   int nRanks, WorkspaceView workspaceView) {
  const int threadId = static_cast<int>(threadIdx.x);
  for (int dstRank = threadId; dstRank < nRanks; dstRank += blockDim.x) {
    const int expectedPayloadCount = rankTokenCounts[dstRank];
    if (expectedPayloadCount > 0) {
      waitAtLeast<mscclpp::scopeDevice>(workspaceView.dispatchRankPayloadCompletions_ + dstRank, expectedPayloadCount);
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

MSCCLPP_DEVICE_INLINE void dispatchRankMajorRecvSource(int* outputTopkIdx, float* outputTopkWeights, int* outputCount,
                                                       const TransportView& transport, WorkspaceView& workspaceView,
                                                       int sourceRank, int nTopk, int maxTokensPerRank,
                                                       int invalidTokenExpertId, void* recvBuffer, uint32_t flag,
                                                       int* sharedMem, uint32_t* payloadReady = nullptr) {
  if (threadIdx.x == 0) {
    auto* rankTokenCounts = static_cast<mscclpp::LL8Packet*>(recvBuffer);
    const int nRankTokens = static_cast<int>(rankTokenCounts[sourceRank].read(flag, -1));
    outputCount[sourceRank] = nRankTokens;
    sharedMem[0] = nRankTokens;
  }
  __syncthreads();

  const int nRankTokens = sharedMem[0];
  const int nMetadataEntries = maxTokensPerRank * nTopk;
  for (int metadataIdx = nRankTokens * nTopk + static_cast<int>(threadIdx.x); metadataIdx < nMetadataEntries;
       metadataIdx += static_cast<int>(blockDim.x)) {
    const size_t outputIdx = static_cast<size_t>(sourceRank) * nMetadataEntries + metadataIdx;
    outputTopkIdx[outputIdx] = invalidTokenExpertId;
    outputTopkWeights[outputIdx] = 0.0f;
  }

  if (threadIdx.x == 0) {
    if (payloadReady != nullptr && !transport.isIntraDomainPeer(sourceRank)) {
#if defined(MSCCLPP_DEVICE_CUDA) && defined(USE_IBVERBS)
      Gateway::waitIntraDomain(payloadReady, sourceRank, flag);
#endif
    } else if (nRankTokens > 0) {
      if (transport.isSelf(sourceRank)) {
        workspaceView.dispatchLocalPayloadReady_->acquire();
      } else {
        transport.baseMemoryChannels_[sourceRank].wait(-1);
      }
    }
  }
}

#endif  // MSCCLPP_BULK_AVAILABLE

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize, DispatchLayout Layout, typename KernelSelector,
          bool Hierarchical = false>
inline void dispatchHiddenMode(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                               float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                               const int64_t* topkIdx, const float* topkWeights, const Workload& workload,
                               void* recvBuffer, const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  static_assert(isSupportedHidden(Hidden));
  using OutputType = DispatchElementType<DataType>;
  constexpr int NRecvTmaWorkers = tmaWorkerCount<Hidden, OutputType, DispatchMaxNRecvTmaWorkers>();
  static_assert(NRecvTmaWorkers > 0);
  const int nExperts = workload.numExperts_;
  const int nRanks = context.numRanks_;
  const int nTopk = workload.numTopk_;

  const size_t dynamicSharedBytes =
      dispatchSharedBytes<Hidden, DataType, ScaleBlockSize, Hierarchical>(nRanks, nExperts, nTopk);
  static thread_local KernelConfigCache kernelConfig;
  auto kernel = KernelSelector::template get<Hidden, DataType, ScaleBlockSize>();
  const int residentBlocks = configureKernel(kernel, DispatchNThreads, dynamicSharedBytes, context, kernelConfig);
  EP_HOST_ASSERT(residentBlocks >= numBlocks);
  kernel<<<dim3(numBlocks), dim3(DispatchNThreads), dynamicSharedBytes, stream>>>(
      output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, topkIdx,
      topkWeights, input, workload, recvBuffer, context.devicePtr_);
  MSCCLPP_CUDATHROW(cudaGetLastError());
}

template <int Hidden, DispatchLayout Layout, typename KernelSelector, bool Hierarchical = false>
inline void dispatchHidden(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                           float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                           const int64_t* topkIdx, const float* topkWeights, const Workload& workload, void* recvBuffer,
                           const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  if constexpr (Hierarchical) {
    static_assert(Layout == DispatchLayout::RANK_MAJOR);
    switch (workload.dispatchDataType_) {
      case DispatchDataType::BF16:
        return dispatchHiddenMode<Hidden, DispatchDataType::BF16, 0, Layout, KernelSelector, true>(
            output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
            topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
      case DispatchDataType::FP8_E4M3_STATIC:
        return dispatchHiddenMode<Hidden, DispatchDataType::FP8_E4M3_STATIC, 0, Layout, KernelSelector, true>(
            output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
            topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
      case DispatchDataType::MXFP8_E4M3:
        return dispatchHiddenMode<Hidden, DispatchDataType::MXFP8_E4M3, 32, Layout, KernelSelector, true>(
            output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
            topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
      case DispatchDataType::FP8_E4M3:
        break;
    }
    EP_HOST_ASSERT(false && "unsupported hierarchical dispatch data type");
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

template <DispatchLayout Layout, typename KernelSelector, bool Hierarchical = false>
inline void dispatchAlgorithm(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                              float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                              const int64_t* topkIdx, const float* topkWeights, const Workload& workload,
                              void* recvBuffer, const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  const int nExperts = workload.numExperts_;
  const int rank = context.rank_;
  const int nRanks = context.numRanks_;
  const int numWorkerBlocks = numBlocks - DispatchControlBlocks;

  EP_HOST_ASSERT(isSupportedRanks(nRanks, Hierarchical));
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
    case 256:
      return dispatchHidden<256, Layout, KernelSelector, Hierarchical>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 512:
      return dispatchHidden<512, Layout, KernelSelector, Hierarchical>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 768:
      return dispatchHidden<768, Layout, KernelSelector, Hierarchical>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 1024:
      return dispatchHidden<1024, Layout, KernelSelector, Hierarchical>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 1280:
      return dispatchHidden<1280, Layout, KernelSelector, Hierarchical>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 1536:
      return dispatchHidden<1536, Layout, KernelSelector, Hierarchical>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 1792:
      return dispatchHidden<1792, Layout, KernelSelector, Hierarchical>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 2048:
      return dispatchHidden<2048, Layout, KernelSelector, Hierarchical>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 2560:
      return dispatchHidden<2560, Layout, KernelSelector, Hierarchical>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 4096:
      return dispatchHidden<4096, Layout, KernelSelector, Hierarchical>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 4352:
      return dispatchHidden<4352, Layout, KernelSelector, Hierarchical>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 5120:
      return dispatchHidden<5120, Layout, KernelSelector, Hierarchical>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 5376:
      return dispatchHidden<5376, Layout, KernelSelector, Hierarchical>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 6656:
      return dispatchHidden<6656, Layout, KernelSelector, Hierarchical>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 7168:
      return dispatchHidden<7168, Layout, KernelSelector, Hierarchical>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 8192:
      return dispatchHidden<8192, Layout, KernelSelector, Hierarchical>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 8704:
      return dispatchHidden<8704, Layout, KernelSelector, Hierarchical>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    case 9216:
      return dispatchHidden<9216, Layout, KernelSelector, Hierarchical>(
          output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout, outputCount, input,
          topkIdx, topkWeights, workload, recvBuffer, context, numBlocks, stream);
    default:
      EP_HOST_ASSERT(false && "unsupported latency dispatch hidden size");
  }
}

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_DISPATCH_COMMON_CUH_
