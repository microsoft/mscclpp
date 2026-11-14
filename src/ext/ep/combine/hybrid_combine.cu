// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "combine/common.cuh"

namespace mscclpp::ep {

#if defined(MSCCLPP_DEVICE_CUDA) && defined(USE_IBVERBS) && MSCCLPP_BULK_AVAILABLE

MSCCLPP_DEVICE_INLINE void aggregateWaitRemoteToken(int remoteTokenIdx, const DeviceContext* context) {
  const auto& hybridCtx = context->hybridCtx_;
  const int chunkIdx = remoteTokenIdx / hybridCtx.nTokensPerChunk_;
  const int nQps = hybridCtx.portChannels_->numQpsPerPeer;
  const int queueIdx = chunkIdx % nQps;
  const uint64_t target = chunkIdx / nQps + 1;
  while (mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(hybridCtx.aggregateControl_->qpSignalCount + queueIdx,
                                                             mscclpp::memoryOrderAcquire) < target) {
  }
}

template <CombineMode Mode, int Hidden, DispatchDataType DispatchType>
MSCCLPP_DEVICE_INLINE mscclpp::f32x8 aggregateReduceVector(int vectorIdx, const uint8_t* sharedTokens,
                                                           const RankMajorCombineSharedControl& sharedControl,
                                                           float dequantScale) {
  constexpr bool IsDirectSend = Mode == CombineMode::DIRECT_SEND;
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(DispatchElementType<DispatchType>);
  constexpr int NPairsPerVector = mscclpp::f32x8::Size / mscclpp::f32x2::Size;
  mscclpp::f32x8 sum;
  auto* sumPairs = reinterpret_cast<mscclpp::f32x2*>(&sum);
#pragma unroll
  for (int pairIdx = 0; pairIdx < NPairsPerVector; ++pairIdx) {
    sumPairs[pairIdx].storage = make_float2(0.0f, 0.0f);
  }
#pragma unroll
  for (int routeIdx = 0; routeIdx < MaxNumTopk; ++routeIdx) {
    if (!sharedControl.validRoutes[routeIdx]) continue;
    const float routeWeight = IsDirectSend ? sharedControl.routeWeights[routeIdx] : 1.0f;
    const auto* routeToken = sharedTokens + static_cast<size_t>(routeIdx) * HiddenBytes;
    if constexpr (DispatchType == DispatchDataType::BF16) {
      const auto values = mscclpp::to<mscclpp::f32x8>(reinterpret_cast<const mscclpp::bf16x8*>(routeToken)[vectorIdx]);
      const auto* pairs = reinterpret_cast<const mscclpp::f32x2*>(&values);
#pragma unroll
      for (int pairIdx = 0; pairIdx < NPairsPerVector; ++pairIdx) {
        if constexpr (IsDirectSend) {
          sumPairs[pairIdx] =
              mscclpp::fma(pairs[pairIdx], mscclpp::f32x2(make_float2(routeWeight, routeWeight)), sumPairs[pairIdx]);
        } else {
          sumPairs[pairIdx] = sumPairs[pairIdx] + pairs[pairIdx];
        }
      }
    } else {
      const auto values = reinterpret_cast<const mscclpp::f8_e4m3x8*>(routeToken)[vectorIdx];
      const float scale = IsDirectSend ? dequantScale * routeWeight : dequantScale;
      if (scale != 0.0f) {
        dequantizeAndAccumulateFp8E4M3(values, scale, sumPairs);
      }
    }
  }
  return sum;
}

template <int Hidden, DispatchDataType DispatchType>
MSCCLPP_DEVICE_INLINE void aggregateStoreFinalVector(void* output, mscclpp::f32x8 sum, int tokenIdx, int remoteTokenIdx,
                                                     int vectorIdx, const DeviceContext* context) {
  constexpr int NHiddenVectors = Hidden / mscclpp::f32x8::Size;
  constexpr int NPairsPerVector = mscclpp::f32x8::Size / mscclpp::f32x2::Size;
  using WireVector =
      std::conditional_t<DispatchType == DispatchDataType::FP8_E4M3_STATIC, mscclpp::bf16x8, mscclpp::f32x8>;
  const auto& hybridCtx = context->hybridCtx_;
  if (remoteTokenIdx >= 0) {
    const auto* remote = reinterpret_cast<const WireVector*>(hybridCtx.aggregateRecvBuffer_);
    const auto partial =
        mscclpp::to<mscclpp::f32x8>(remote[static_cast<size_t>(remoteTokenIdx) * NHiddenVectors + vectorIdx]);
    auto* sumPairs = reinterpret_cast<mscclpp::f32x2*>(&sum);
    const auto* partialPairs = reinterpret_cast<const mscclpp::f32x2*>(&partial);
#pragma unroll
    for (int pairIdx = 0; pairIdx < NPairsPerVector; ++pairIdx) {
      sumPairs[pairIdx] = sumPairs[pairIdx] + partialPairs[pairIdx];
    }
  }
  static_cast<mscclpp::bf16x8*>(output)[static_cast<size_t>(tokenIdx) * NHiddenVectors + vectorIdx] =
      mscclpp::to<mscclpp::bf16x8>(sum);
}

template <int Hidden, DispatchDataType DispatchType>
MSCCLPP_DEVICE_INLINE void aggregateFinalize(void* output, int tokenIdx, const DeviceContext* context) {
  constexpr int NHiddenVectors = Hidden / mscclpp::f32x8::Size;
  const auto& hybridCtx = context->hybridCtx_;
  const int remoteTokenIdx = hybridCtx.aggregateRemoteTokenIndices_[tokenIdx];
  if (threadIdx.x == 0 && remoteTokenIdx >= 0) aggregateWaitRemoteToken(remoteTokenIdx, context);
  __syncthreads();
  const auto* local =
      reinterpret_cast<const mscclpp::f32x8*>(hybridCtx.aggregateLocalBuffer_ + static_cast<size_t>(tokenIdx) * Hidden);
  for (int vectorIdx = threadIdx.x; vectorIdx < NHiddenVectors; vectorIdx += blockDim.x) {
    aggregateStoreFinalVector<Hidden, DispatchType>(output, local[vectorIdx], tokenIdx, remoteTokenIdx, vectorIdx,
                                                    context);
  }
}

template <CombineMode Mode, int Hidden, DispatchDataType DispatchType, bool Gateway, bool FuseFinal = false>
MSCCLPP_DEVICE_INLINE void aggregateReduceToken(void* destination, const void* expertOutput, const int64_t* topkIndices,
                                                const float* topkWeights, int tokenIdx, const Workload& workload,
                                                const DeviceContext* context, const TransportView& transport,
                                                const WorkspaceView& workspace, uint32_t flag, uint8_t* sharedMemory,
                                                uint32_t& phase, float dequantScale, int localWorkerIdx = 0,
                                                int nLocalWorkers = 1) {
  static_assert(!(Gateway && FuseFinal));
  constexpr bool IsDirectSend = Mode == CombineMode::DIRECT_SEND;
  using ElementType = DispatchElementType<DispatchType>;
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(ElementType);
  constexpr int NHiddenVectors = Hidden / mscclpp::f32x8::Size;
  constexpr int MaxVectorsPerThread = configCellDiv(NHiddenVectors, CombineNThreads);
  constexpr size_t TokensBytes = MaxNumTopk * HiddenBytes;
  const auto& hybridCtx = context->hybridCtx_;
  const int nTokensPerChunk = hybridCtx.nTokensPerChunk_;
  const int laneIdx = getLaneId();
  const int nTopk = workload.numTopk_;
  const int nLocalExperts = workload.numExperts_ / context->numRanks_;
  const int ownerRank = Gateway ? hybridCtx.gatewayRank_ : context->rank_;
  auto& sharedControl = *reinterpret_cast<RankMajorCombineSharedControl*>(sharedMemory + TokensBytes);
  auto* barriers = sharedControl.barriers;
  auto* validRoutes = sharedControl.validRoutes;
  auto* routeWeights = sharedControl.routeWeights;
  if (threadIdx.x < WARP_SIZE) {
    int64_t expertIdx = -1;
    int slotIdx = -1;
    float weight = 1.0f;
    if (laneIdx < nTopk) {
      if constexpr (Gateway) {
        const auto* cached = hybridCtx.aggregateRoutes_ + static_cast<size_t>(tokenIdx) * 3 * nTopk;
        expertIdx = cached[laneIdx];
        slotIdx = cached[nTopk + laneIdx];
        if (IsDirectSend && workload.applyRouterWeights_) {
          weight = reinterpret_cast<const float*>(cached + 2 * nTopk)[laneIdx];
        }
      } else {
        expertIdx = topkIndices[tokenIdx * nTopk + laneIdx];
        slotIdx = workspace.rankMajorSendIndices_[tokenIdx * nTopk + laneIdx];
        if (IsDirectSend && workload.applyRouterWeights_ && topkWeights != nullptr) {
          weight = topkWeights[tokenIdx * nTopk + laneIdx];
        }
      }
    }
    const int peerIdx =
        expertIdx >= 0 && expertIdx < workload.numExperts_ ? static_cast<int>(expertIdx) / nLocalExperts : -1;
    const bool leader = isFirstLaneForRank(peerIdx, laneIdx);
    const bool active = peerIdx >= 0 && (IsDirectSend || leader) && transport.isNvlinkPeer(peerIdx);
    if (laneIdx < MaxNumTopk) {
      validRoutes[laneIdx] = active;
      if constexpr (IsDirectSend) routeWeights[laneIdx] = weight;
    }
    bool pending = active;
    while (__any_sync(0xffffffff, pending)) {
      if (pending && mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(workspace.combineRankReadyFlags_ + peerIdx,
                                                                         mscclpp::memoryOrderAcquire) == flag) {
        EP_DEVICE_ASSERT(slotIdx >= 0 && slotIdx < workload.maxTokensPerRank_);
        const auto* peerOutput =
            static_cast<const uint8_t*>(transport.mappedBuffer(const_cast<void*>(expertOutput), peerIdx));
        const size_t sourceTokenIdx = static_cast<size_t>(ownerRank) * workload.maxTokensPerRank_ + slotIdx;
        const size_t inputTokenIdx = IsDirectSend ? sourceTokenIdx * nTopk + laneIdx : sourceTokenIdx;
        barriers[laneIdx].arriveAndExpect(HiddenBytes);
        mscclpp::bulkLoad(sharedMemory + laneIdx * HiddenBytes, peerOutput + inputTokenIdx * HiddenBytes, HiddenBytes,
                          barriers[laneIdx]);
        pending = false;
      }
    }
    if (active) barriers[laneIdx].wait(phase);
    __syncwarp();
    if (laneIdx == 0) {
      mscclpp::bulkFence();
      if constexpr (Gateway) {
        atomicAddReleaseDevice(hybridCtx.aggregateChunkLoadDone_ + tokenIdx / nTokensPerChunk, 1);
      }
    }
  }
  __syncthreads();
  if constexpr (FuseFinal) {
    mscclpp::f32x8 tileSum[MaxVectorsPerThread];
#pragma unroll
    for (int stepIdx = 0; stepIdx < MaxVectorsPerThread; ++stepIdx) {
      const int vectorIdx = static_cast<int>(threadIdx.x) + stepIdx * CombineNThreads;
      if (vectorIdx < NHiddenVectors) {
        tileSum[stepIdx] =
            aggregateReduceVector<Mode, Hidden, DispatchType>(vectorIdx, sharedMemory, sharedControl, dequantScale);
      }
    }
    // Keep the last token's local sum in registers while finalizing earlier tokens from scratch.
    for (int prevIdx = localWorkerIdx; prevIdx < tokenIdx; prevIdx += nLocalWorkers) {
      aggregateFinalize<Hidden, DispatchType>(destination, prevIdx, context);
    }
    const int remoteTokenIdx = hybridCtx.aggregateRemoteTokenIndices_[tokenIdx];
    if (remoteTokenIdx >= 0) {
      if (threadIdx.x == 0) aggregateWaitRemoteToken(remoteTokenIdx, context);
      __syncthreads();
    }
#pragma unroll
    for (int stepIdx = 0; stepIdx < MaxVectorsPerThread; ++stepIdx) {
      const int vectorIdx = static_cast<int>(threadIdx.x) + stepIdx * CombineNThreads;
      if (vectorIdx < NHiddenVectors) {
        aggregateStoreFinalVector<Hidden, DispatchType>(destination, tileSum[stepIdx], tokenIdx, remoteTokenIdx,
                                                        vectorIdx, context);
      }
    }
  } else {
    for (int vectorIdx = threadIdx.x; vectorIdx < NHiddenVectors; vectorIdx += blockDim.x) {
      const auto sum =
          aggregateReduceVector<Mode, Hidden, DispatchType>(vectorIdx, sharedMemory, sharedControl, dequantScale);
      if constexpr (Gateway && DispatchType == DispatchDataType::FP8_E4M3_STATIC) {
        reinterpret_cast<mscclpp::bf16x8*>(destination)[vectorIdx] = mscclpp::to<mscclpp::bf16x8>(sum);
      } else {
        reinterpret_cast<mscclpp::f32x8*>(destination)[vectorIdx] = sum;
      }
    }
    // The block barrier joins every writer into thread 0's release counter.
    __syncthreads();
    if (Gateway && threadIdx.x == 0) {
      atomicAddReleaseDevice(hybridCtx.aggregateChunkReady_ + tokenIdx / nTokensPerChunk, 1);
    }
  }
}

template <int Hidden, DispatchDataType DispatchType>
MSCCLPP_DEVICE_INLINE void aggregateControl(const DeviceContext* context, const TransportView& transport,
                                            const WorkspaceView& workspace, uint32_t flag) {
  const int peerIdx = threadIdx.x;
  const int nRanks = context->numRanks_;
  const auto& hybridCtx = context->hybridCtx_;
  auto* portChannels = transport.portChannels_;
  const int nQps = portChannels->numQpsPerPeer;
  const int queueIdx = static_cast<int>(threadIdx.x) / WARP_SIZE - configCellDiv(nRanks, WARP_SIZE);
  constexpr auto kExclusive = mscclpp::PortChannelDeviceHandle::Concurrency::Exclusive;
  if (peerIdx < nRanks && transport.isNvlinkPeer(peerIdx)) {
    if (!transport.isSelf(peerIdx)) {
      transport.baseMemoryChannels_[peerIdx].signal();
      transport.baseMemoryChannels_[peerIdx].wait(-1);
    }
    mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(workspace.combineRankReadyFlags_ + peerIdx, flag,
                                                         mscclpp::memoryOrderRelease);
  }
  if (queueIdx >= 0 && queueIdx < nQps && getLaneId() == 0) {
    const int nTokens = *hybridCtx.nIncomingTokens_;
    const int nTokensPerChunk = hybridCtx.nTokensPerChunk_;
    using WireElement = std::conditional_t<DispatchType == DispatchDataType::FP8_E4M3_STATIC, Bf16, float>;
    constexpr size_t TokenBytes = static_cast<size_t>(Hidden) * sizeof(WireElement);
    const int gatewayRank = hybridCtx.gatewayRank_;
    auto handle = portChannels->channel(gatewayRank, queueIdx);
    const uint64_t dstOffset = transport.symmetricOffset(hybridCtx.aggregateRecvBuffer_);
    const uint64_t srcOffset = transport.symmetricOffset(hybridCtx.aggregateSendBuffer_);
    const uint64_t flagOffset = transport.symmetricOffset(hybridCtx.aggregateControl_->qpSignalCount + queueIdx);
    for (int chunkIdx = queueIdx, tokenBegin = queueIdx * nTokensPerChunk; tokenBegin < nTokens;
         chunkIdx += nQps, tokenBegin += nQps * nTokensPerChunk) {
      const int nChunkTokens = min(nTokensPerChunk, nTokens - tokenBegin);
      handle.device_.putWithAccumulateAfter<kExclusive, MaxQpsPerPeer>(
          handle.dstMemory_, dstOffset + tokenBegin * TokenBytes, handle.srcMemory_,
          srcOffset + tokenBegin * TokenBytes, nChunkTokens * TokenBytes, handle.dstMemory_, flagOffset, 1,
          [&] {
            while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(hybridCtx.aggregateChunkReady_ + chunkIdx,
                                                                  mscclpp::memoryOrderAcquire) != nChunkTokens) {
            }
          },
          queueIdx);
    }
  }
  // Combine resets the dispatch counters, then waits for permission to reuse the RDMA landing buffer.
  for (int idx = threadIdx.x; idx < nRanks; idx += blockDim.x) {
    workspace.dispatchRankPayloadSlots_[idx] = 0;
    workspace.dispatchRankPayloadCompletions_[idx] = 0;
  }
  const int nDispatchChunks = configCellDiv(hybridCtx.distributeCapacity_, hybridCtx.nTokensPerChunk_);
  for (int chunkIdx = threadIdx.x; chunkIdx < nDispatchChunks; chunkIdx += blockDim.x) {
    hybridCtx.distributeScratch_->chunkReadySignals[chunkIdx] = 0;
  }
  if (threadIdx.x == 0) {
    hybridCtx.distributeScratch_->nRemoteTokens = 0;
    hybridCtx.distributeScratch_->nCompletedForwarders = 0;
    hybridCtx.distributeScratch_->forwardComplete = 0;
    auto* state = hybridCtx.distributeControl_;
    while (mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(&state->bufferReuseReady, mscclpp::memoryOrderAcquire) ==
           0) {
    }
    state->bufferReuseReady = 0;
  }
}

template <CombineMode Mode, int Hidden, DispatchDataType DispatchType>
MSCCLPP_DEVICE_INLINE void aggregateGatewayTokens(const void* expertOutput, const int64_t* topkIndices,
                                                  const float* topkWeights, const Workload& workload,
                                                  const DeviceContext* context, const TransportView& transport,
                                                  const WorkspaceView& workspace, uint32_t flag, uint8_t* sharedMemory,
                                                  uint32_t& phase, float dequantScale) {
  using ElementType = DispatchElementType<DispatchType>;
  using WireElement = std::conditional_t<DispatchType == DispatchDataType::FP8_E4M3_STATIC, Bf16, float>;
  constexpr int LoadWaveBf16Tokens = 16;
  constexpr int ReadyWaveTokens = 64;
  const auto& hybridCtx = context->hybridCtx_;
  const int nQps = transport.portChannels_->numQpsPerPeer;
  const int workerIdx = static_cast<int>(blockIdx.x) - 1;
  const int nGatewayWorkers = static_cast<int>(gridDim.x) - 1;
  const int nIncomingTokens = *hybridCtx.nIncomingTokens_;
  const int nTokensPerChunk = hybridCtx.nTokensPerChunk_;
  const int nLoadWaveTokens = LoadWaveBf16Tokens * static_cast<int>(sizeof(Bf16) / sizeof(ElementType));
  const int nLoadWaveChunks = max(1, nLoadWaveTokens / nTokensPerChunk);
  const int nReadyWaveChunks = max(nQps, configCellDiv(ReadyWaveTokens, nTokensPerChunk));
  auto* sendBase = reinterpret_cast<WireElement*>(hybridCtx.aggregateSendBuffer_);
  for (int tokenIdx = workerIdx; tokenIdx < nIncomingTokens; tokenIdx += nGatewayWorkers) {
    const int chunkIdx = tokenIdx / nTokensPerChunk;
    if (threadIdx.x == 0) {
      // Bound in-flight NVLink loads by both loaded and reduced chunks, with byte-scaled load waves.
      if (chunkIdx >= nLoadWaveChunks) {
        while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(
                   hybridCtx.aggregateChunkLoadDone_ + (chunkIdx - nLoadWaveChunks), mscclpp::memoryOrderAcquire) !=
               nTokensPerChunk) {
        }
      }
      if (chunkIdx >= nReadyWaveChunks) {
        while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(
                   hybridCtx.aggregateChunkReady_ + (chunkIdx - nReadyWaveChunks), mscclpp::memoryOrderAcquire) !=
               nTokensPerChunk) {
        }
      }
    }
    __syncthreads();
    aggregateReduceToken<Mode, Hidden, DispatchType, true>(
        sendBase + static_cast<size_t>(tokenIdx) * Hidden, expertOutput, topkIndices, topkWeights, tokenIdx, workload,
        context, transport, workspace, flag, sharedMemory, phase, dequantScale);
  }
  // All gateway reads must finish before this block's local reductions compete for NVLink bandwidth.
  if (threadIdx.x == 0) {
    for (int tokenBegin = 0; tokenBegin < nIncomingTokens; tokenBegin += nTokensPerChunk) {
      const int nChunkTokens = min(nTokensPerChunk, nIncomingTokens - tokenBegin);
      while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(
                 hybridCtx.aggregateChunkLoadDone_ + tokenBegin / nTokensPerChunk, mscclpp::memoryOrderAcquire) !=
             nChunkTokens) {
      }
    }
  }
  __syncthreads();
}

#endif

template <CombineMode Mode, int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout>
__global__ __launch_bounds__(CombineNThreads,
                             1) void hybridCombineKernel(void* output, const void* expertOutput,
                                                         const int64_t* topkIndices, const float* topkWeights,
                                                         const int* srcInfo, const int64_t* layoutRange,
                                                         Workload workload, void* combineRecvBuffer,
                                                         const void* dispatchRecvBuffer, const DeviceContext* context) {
#if defined(MSCCLPP_DEVICE_CUDA) && defined(USE_IBVERBS) && MSCCLPP_BULK_AVAILABLE
  if constexpr (Layout == DispatchLayout::RANK_MAJOR &&
                (DispatchType == DispatchDataType::BF16 || DispatchType == DispatchDataType::FP8_E4M3_STATIC)) {
    extern __shared__ __align__(128) uint8_t sharedMemory[];
    const TransportView transport(context);
    const WorkspaceView workspace(context->workspace_, context->numRanks_, workload.numExperts_);
    const uint32_t flag = getDispatchFlag(workspace);
    constexpr size_t TokensBytes = static_cast<size_t>(MaxNumTopk) * Hidden * sizeof(DispatchElementType<DispatchType>);
    const auto& hybridCtx = context->hybridCtx_;
    auto& sharedControl = *reinterpret_cast<RankMajorCombineSharedControl*>(sharedMemory + TokensBytes);
    const float dequantScale = DispatchType == DispatchDataType::FP8_E4M3_STATIC ? 1.0f / workload.combineScale_ : 1.0f;
    uint32_t phase = 0;

    if (blockIdx.x == 0) {
      aggregateControl<Hidden, DispatchType>(context, transport, workspace, flag);
      __syncthreads();
    }
    if (threadIdx.x < MaxNumTopk) sharedControl.barriers[threadIdx.x].init();
    __syncthreads();

    if (blockIdx.x != 0) {
      aggregateGatewayTokens<Mode, Hidden, DispatchType>(expertOutput, topkIndices, topkWeights, workload, context,
                                                         transport, workspace, flag, sharedMemory, phase, dequantScale);
    }

    const int nLocalWorkers = static_cast<int>(gridDim.x);
    const int localWorkerIdx = blockIdx.x == 0 ? nLocalWorkers - 1 : static_cast<int>(blockIdx.x) - 1;
    for (int tokenIdx = localWorkerIdx; tokenIdx < workload.numTokens_; tokenIdx += nLocalWorkers) {
      if (tokenIdx + nLocalWorkers < workload.numTokens_) {
        aggregateReduceToken<Mode, Hidden, DispatchType, false, false>(
            hybridCtx.aggregateLocalBuffer_ + static_cast<size_t>(tokenIdx) * Hidden, expertOutput, topkIndices,
            topkWeights, tokenIdx, workload, context, transport, workspace, flag, sharedMemory, phase, dequantScale);
      } else {
        aggregateReduceToken<Mode, Hidden, DispatchType, false, true>(
            output, expertOutput, topkIndices, topkWeights, tokenIdx, workload, context, transport, workspace, flag,
            sharedMemory, phase, dequantScale, localWorkerIdx, nLocalWorkers);
      }
    }
    __syncthreads();
    if (threadIdx.x < MaxNumTopk) sharedControl.barriers[threadIdx.x].invalidate();
  }
#endif
}

template <CombineMode Mode>
struct HybridCombineKernelSelector {
  template <int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout>
  static auto get() {
    return hybridCombineKernel<Mode, Hidden, DispatchType, ScaleBlockSize, Layout>;
  }
};

void hybridCombine(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights,
                   const Workload& workload, CombineMode combineMode, void* recvBuffer, void* dispatchRecvBuffer,
                   const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  EP_HOST_ASSERT(context.hybridCtx_.portChannels_ != nullptr && workload.numTopk_ <= MaxNumTopk);
  EP_HOST_ASSERT(workload.outputLayout_ == DispatchLayout::RANK_MAJOR);
  EP_HOST_ASSERT(workload.dispatchDataType_ == DispatchDataType::BF16 ||
                 workload.dispatchDataType_ == DispatchDataType::FP8_E4M3_STATIC);
  if (combineMode == CombineMode::DIRECT_SEND) {
    combineAlgorithm<CombineMode::DIRECT_SEND, HybridCombineKernelSelector<CombineMode::DIRECT_SEND>, true>(
        output, input, topkIdx, topkWeights, nullptr, nullptr, workload, recvBuffer, dispatchRecvBuffer, context,
        numBlocks, stream);
  } else {
    EP_HOST_ASSERT(combineMode == CombineMode::RANK_LOCAL_REDUCE);
    combineAlgorithm<CombineMode::RANK_LOCAL_REDUCE, HybridCombineKernelSelector<CombineMode::RANK_LOCAL_REDUCE>, true>(
        output, input, topkIdx, nullptr, nullptr, nullptr, workload, recvBuffer, dispatchRecvBuffer, context, numBlocks,
        stream);
  }
}

}  // namespace mscclpp::ep
