// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "combine/common.cuh"
#include "common/gateway.cuh"

namespace mscclpp::ep {

#if defined(MSCCLPP_DEVICE_CUDA) && defined(USE_IBVERBS) && MSCCLPP_BULK_AVAILABLE

template <int StageSlots>
struct HierarchicalCombineGroup {
  uint8_t* sharedTokens;
  HierarchicalCombineGroupControl<StageSlots>& control;
  uint32_t& phase;
  int warpIdx;
  int threadIdx;
  int barrierId;

  MSCCLPP_DEVICE_INLINE void sync() const { syncNamedBarrier(barrierId, HierarchicalCombineGroupThreads); }
};

struct IntraDomainRoute {
  int peerRank = -1;
  int slotIdx = -1;
  float weight = 1.0f;
};

MSCCLPP_DEVICE_INLINE bool useBf16InterDomainCombine(const HierarchicalDeviceContext& hierarchicalCtx) {
  return hierarchicalCtx.combineInterDomainDtype_ == mscclpp::DataType::BFLOAT16;
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void combineStoreFinalVector(void* output, mscclpp::f32x8 sum, int tokenIdx, int remoteTokenIdx,
                                                   int vectorIdx, const DeviceContext* context) {
  constexpr int NHiddenVectors = Hidden / mscclpp::f32x8::Size;
  constexpr int NPairsPerVector = mscclpp::f32x8::Size / mscclpp::f32x2::Size;
  const auto& hierarchicalCtx = context->hierarchicalCtx_;
  if (remoteTokenIdx >= 0) {
    const size_t remoteVecIdx = static_cast<size_t>(remoteTokenIdx) * NHiddenVectors + vectorIdx;
    const mscclpp::f32x8 partial =
        useBf16InterDomainCombine(hierarchicalCtx)
            ? mscclpp::to<mscclpp::f32x8>(
                  reinterpret_cast<const mscclpp::bf16x8*>(hierarchicalCtx.combineRecvBuffer_)[remoteVecIdx])
            : reinterpret_cast<const mscclpp::f32x8*>(hierarchicalCtx.combineRecvBuffer_)[remoteVecIdx];
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

template <int Hidden, int StageSlots>
MSCCLPP_DEVICE_INLINE void combineFinalize(void* output, int tokenIdx, const DeviceContext* context,
                                           const Gateway& gateway, const HierarchicalCombineGroup<StageSlots>& group) {
  constexpr int NHiddenVectors = Hidden / mscclpp::f32x8::Size;
  const auto& hierarchicalCtx = context->hierarchicalCtx_;
  const int remoteTokenIdx = hierarchicalCtx.combineRemoteTokenIndices_[tokenIdx];
  if (group.threadIdx == 0 && remoteTokenIdx >= 0) gateway.wait(remoteTokenIdx);
  group.sync();
  const auto* local = reinterpret_cast<const mscclpp::f32x8*>(hierarchicalCtx.combineLocalBuffer_ +
                                                              static_cast<size_t>(tokenIdx) * Hidden);
  for (int vectorIdx = group.threadIdx; vectorIdx < NHiddenVectors; vectorIdx += HierarchicalCombineGroupThreads) {
    combineStoreFinalVector<Hidden>(output, local[vectorIdx], tokenIdx, remoteTokenIdx, vectorIdx, context);
  }
}

template <CombineMode Mode>
MSCCLPP_DEVICE_INLINE IntraDomainRoute resolveGatewayRoute(int tokenIdx, int laneIdx, const Workload& workload,
                                                           const DeviceContext* context) {
  constexpr bool IsDirectSend = Mode == CombineMode::DIRECT_SEND;
  const int nTopk = workload.numTopk_;
  if (laneIdx >= nTopk) return {};
  const auto* cached = context->hierarchicalCtx_.combineRoutes_ + static_cast<size_t>(tokenIdx) * 3 * nTopk;
  const int expertIdx = cached[laneIdx];
  const int nLocalExperts = workload.numExperts_ / context->numRanks_;
  const int peerRank = expertIdx >= 0 && expertIdx < workload.numExperts_ ? expertIdx / nLocalExperts : -1;
  const float weight = (IsDirectSend && workload.applyRouterWeights_)
                           ? reinterpret_cast<const float*>(cached + 2 * nTopk)[laneIdx]
                           : 1.0f;
  return {peerRank, cached[nTopk + laneIdx], weight};
}

template <CombineMode Mode>
MSCCLPP_DEVICE_INLINE IntraDomainRoute resolveLocalRoute(int tokenIdx, int laneIdx, const int64_t* topkIndices,
                                                         const float* topkWeights, const Workload& workload,
                                                         const DeviceContext* context, const WorkspaceView& workspace) {
  constexpr bool IsDirectSend = Mode == CombineMode::DIRECT_SEND;
  const int nTopk = workload.numTopk_;
  if (laneIdx >= nTopk) return {};
  const size_t routeOffset = static_cast<size_t>(tokenIdx) * nTopk + laneIdx;
  const int64_t expertIdx = topkIndices[routeOffset];
  const int nLocalExperts = workload.numExperts_ / context->numRanks_;
  const int peerRank =
      expertIdx >= 0 && expertIdx < workload.numExperts_ ? static_cast<int>(expertIdx) / nLocalExperts : -1;
  const float weight =
      (IsDirectSend && workload.applyRouterWeights_ && topkWeights != nullptr) ? topkWeights[routeOffset] : 1.0f;
  return {peerRank, workspace.rankMajorSendIndices_[routeOffset], weight};
}

template <CombineMode Mode, int Hidden, DispatchDataType DispatchType, int StageSlots>
MSCCLPP_DEVICE_INLINE void accumulateHierarchicalVector(mscclpp::f32x8& sum, int vectorIdx, const uint8_t* sharedTokens,
                                                        const float* routeWeights, int nStagedTokens,
                                                        float dequantScale) {
  constexpr bool IsDirectSend = Mode == CombineMode::DIRECT_SEND;
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(DispatchElementType<DispatchType>);
  constexpr int NPairsPerVector = mscclpp::f32x8::Size / mscclpp::f32x2::Size;
  auto* sumPairs = reinterpret_cast<mscclpp::f32x2*>(&sum);
#pragma unroll
  for (int stageIdx = 0; stageIdx < StageSlots; ++stageIdx) {
    if (stageIdx >= nStagedTokens) break;
    const float routeWeight = IsDirectSend ? routeWeights[stageIdx] : 1.0f;
    const auto* routeToken = sharedTokens + static_cast<size_t>(stageIdx) * HiddenBytes;
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
}

template <int NVectors>
MSCCLPP_DEVICE_INLINE void zeroVectors(mscclpp::f32x8 (&vectors)[NVectors]) {
  constexpr int NPairsPerVector = mscclpp::f32x8::Size / mscclpp::f32x2::Size;
#pragma unroll
  for (int stepIdx = 0; stepIdx < NVectors; ++stepIdx) {
    auto* pairs = reinterpret_cast<mscclpp::f32x2*>(&vectors[stepIdx]);
#pragma unroll
    for (int pairIdx = 0; pairIdx < NPairsPerVector; ++pairIdx) {
      pairs[pairIdx].storage = make_float2(0.0f, 0.0f);
    }
  }
}

template <CombineMode Mode, int MaxNTopk>
MSCCLPP_DEVICE_INLINE void compactIntraDomainRoute(const IntraDomainRoute& laneRoute, const TransportView& transport,
                                                   int laneIdx, bool& active, int& compactIdx, int& nRoutes) {
  constexpr bool IsDirectSend = Mode == CombineMode::DIRECT_SEND;
  const bool isLeader = isFirstLaneForRank(laneRoute.peerRank, laneIdx);
  active = laneIdx < MaxNTopk && laneRoute.peerRank >= 0 && (IsDirectSend || isLeader) &&
           transport.isIntraDomainPeer(laneRoute.peerRank);
  const uint32_t validRouteMask = __ballot_sync(0xffffffffu, active);
  nRoutes = __popc(validRouteMask);
  compactIdx = laneIdx < MaxNTopk ? __popc(validRouteMask & ((1u << laneIdx) - 1u)) : 0;
}

template <CombineMode Mode, int Hidden, DispatchDataType DispatchType, int StageSlots>
MSCCLPP_DEVICE_INLINE void loadIntraDomainTokens(int ownerRank, const IntraDomainRoute& laneRoute, int laneIdx,
                                                 bool active, int stageIdx, int nStagedTokens, const void* expertOutput,
                                                 const Workload& workload, const TransportView& transport,
                                                 const WorkspaceView& workspace, uint32_t flag,
                                                 const HierarchicalCombineGroup<StageSlots>& group) {
  constexpr bool IsDirectSend = Mode == CombineMode::DIRECT_SEND;
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(DispatchElementType<DispatchType>);
  const bool inRound = active && stageIdx >= 0 && stageIdx < StageSlots;
  if (inRound) {
    if constexpr (IsDirectSend) group.control.routeWeights[stageIdx] = laneRoute.weight;
  }
  if (laneIdx == 0 && nStagedTokens > 0) {
    group.control.barrier.arriveAndExpect(static_cast<uint32_t>(nStagedTokens * HiddenBytes));
  }
  __syncwarp();
  if (nStagedTokens == 0) return;

  bool pending = inRound;
  while (__any_sync(0xffffffffu, pending)) {
    if (pending && pollEqual<mscclpp::scopeDevice>(workspace.combineRankReadyFlags_ + laneRoute.peerRank, flag)) {
      EP_DEVICE_ASSERT(laneRoute.slotIdx >= 0 && laneRoute.slotIdx < workload.maxTokensPerRank_);
      const auto* peerOutput =
          static_cast<const uint8_t*>(transport.mappedBuffer(const_cast<void*>(expertOutput), laneRoute.peerRank));
      const size_t sourceTokenIdx = static_cast<size_t>(ownerRank) * workload.maxTokensPerRank_ + laneRoute.slotIdx;
      const size_t inputTokenIdx = IsDirectSend ? sourceTokenIdx * workload.numTopk_ + laneIdx : sourceTokenIdx;
      mscclpp::bulkLoad(group.sharedTokens + static_cast<size_t>(stageIdx) * HiddenBytes,
                        peerOutput + inputTokenIdx * HiddenBytes, HiddenBytes, group.control.barrier);
      pending = false;
    }
  }
  if (laneIdx == 0) {
    group.control.barrier.wait(group.phase);
    mscclpp::bulkFence();
  }
  __syncwarp();
}

// Gather intra-domain expert outputs over TMA in StageSlots-sized rounds and accumulate FP32 into tokenSum.
template <CombineMode Mode, int Hidden, DispatchDataType DispatchType, int MaxNTopk, int StageSlots,
          typename OnLastLoadFn>
MSCCLPP_DEVICE_INLINE void reduceIntraDomainToken(
    mscclpp::f32x8 (&tokenSum)[configCellDiv(Hidden / mscclpp::f32x8::Size, HierarchicalCombineGroupThreads)],
    int ownerRank, const IntraDomainRoute& laneRoute, const void* expertOutput, const Workload& workload,
    const TransportView& transport, const WorkspaceView& workspace, uint32_t flag, float dequantScale,
    const HierarchicalCombineGroup<StageSlots>& group, OnLastLoadFn&& onLastLoad) {
  constexpr int NHiddenVectors = Hidden / mscclpp::f32x8::Size;
  constexpr int MaxVectorsPerThread = configCellDiv(NHiddenVectors, HierarchicalCombineGroupThreads);
  constexpr int MaxRounds = configCellDiv(MaxNTopk, StageSlots);
  const int laneIdx = getLaneId();

  bool active = false;
  int nRoutes = 0;
  int compactIdx = 0;
  int nRounds = 1;
  if (group.warpIdx == 0) {
    compactIntraDomainRoute<Mode, MaxNTopk>(laneRoute, transport, laneIdx, active, compactIdx, nRoutes);
    nRounds = max(1, configCellDiv(nRoutes, StageSlots));
    if (laneIdx == 0) group.control.nRoutes = nRoutes;
  }
  zeroVectors(tokenSum);

  for (int roundIdx = 0; roundIdx < MaxRounds; ++roundIdx) {
    const int routeBegin = roundIdx * StageSlots;
    if (group.warpIdx == 0) {
      const int nStagedTokens = min(StageSlots, max(0, nRoutes - routeBegin));
      loadIntraDomainTokens<Mode, Hidden, DispatchType, StageSlots>(
          ownerRank, laneRoute, laneIdx, active, compactIdx - routeBegin, nStagedTokens, expertOutput, workload,
          transport, workspace, flag, group);
      if (laneIdx == 0 && roundIdx + 1 == nRounds) onLastLoad();
    }
    group.sync();

    const int totalRoutes = group.control.nRoutes;
    const int nStagedTokens = min(StageSlots, max(0, totalRoutes - routeBegin));
    if (nStagedTokens > 0) {
#pragma unroll
      for (int stepIdx = 0; stepIdx < MaxVectorsPerThread; ++stepIdx) {
        const int vectorIdx = group.threadIdx + stepIdx * HierarchicalCombineGroupThreads;
        if (vectorIdx < NHiddenVectors) {
          accumulateHierarchicalVector<Mode, Hidden, DispatchType, StageSlots>(
              tokenSum[stepIdx], vectorIdx, group.sharedTokens, group.control.routeWeights, nStagedTokens,
              dequantScale);
        }
      }
    }
    if constexpr (MaxRounds > 1) {
      if (routeBegin + StageSlots >= totalRoutes) break;
      group.sync();
    }
  }
}

// Gateway path: reduce local-domain contributions for a paired remote token and write into combineSendBuffer_.
template <CombineMode Mode, int Hidden, DispatchDataType DispatchType, int MaxNTopk, int StageSlots,
          typename LoadReadyFn>
MSCCLPP_DEVICE_INLINE void combineGatewayToken(void* dst, const void* expertOutput, int tokenIdx,
                                               const Workload& workload, const DeviceContext* context,
                                               const TransportView& transport, const Gateway& gateway,
                                               const WorkspaceView& workspace, uint32_t flag, float dequantScale,
                                               const HierarchicalCombineGroup<StageSlots>& group,
                                               LoadReadyFn&& loadReady) {
  constexpr int NHiddenVectors = Hidden / mscclpp::f32x8::Size;
  constexpr int MaxVectorsPerThread = configCellDiv(NHiddenVectors, HierarchicalCombineGroupThreads);
  const IntraDomainRoute laneRoute =
      group.warpIdx == 0 ? resolveGatewayRoute<Mode>(tokenIdx, getLaneId(), workload, context) : IntraDomainRoute{};
  mscclpp::f32x8 tokenSum[MaxVectorsPerThread];
  reduceIntraDomainToken<Mode, Hidden, DispatchType, MaxNTopk>(tokenSum, gateway.peerRank(), laneRoute, expertOutput,
                                                               workload, transport, workspace, flag, dequantScale,
                                                               group, loadReady);

  const bool bf16InterDomain = useBf16InterDomainCombine(context->hierarchicalCtx_);
#pragma unroll
  for (int stepIdx = 0; stepIdx < MaxVectorsPerThread; ++stepIdx) {
    const int vectorIdx = group.threadIdx + stepIdx * HierarchicalCombineGroupThreads;
    if (vectorIdx < NHiddenVectors) {
      if (bf16InterDomain) {
        static_cast<mscclpp::bf16x8*>(dst)[vectorIdx] = mscclpp::to<mscclpp::bf16x8>(tokenSum[stepIdx]);
      } else {
        static_cast<mscclpp::f32x8*>(dst)[vectorIdx] = tokenSum[stepIdx];
      }
    }
  }
}

// Local path: reduce own intra-domain contributions; overlap in-flight remote RDMA partials across multiple tokens.
template <CombineMode Mode, int Hidden, DispatchDataType DispatchType, int MaxNTopk, int StageSlots>
MSCCLPP_DEVICE_INLINE void combineLocalTokens(void* output, const void* expertOutput, const int64_t* topkIndices,
                                              const float* topkWeights, const Workload& workload,
                                              const DeviceContext* context, const TransportView& transport,
                                              const Gateway& gateway, const WorkspaceView& workspace, uint32_t flag,
                                              float dequantScale, const HierarchicalCombineGroup<StageSlots>& group,
                                              int localWorkerIdx, int nLocalWorkers) {
  constexpr int NHiddenVectors = Hidden / mscclpp::f32x8::Size;
  constexpr int MaxVectorsPerThread = configCellDiv(NHiddenVectors, HierarchicalCombineGroupThreads);
  const auto& hierarchicalCtx = context->hierarchicalCtx_;
  const int laneIdx = getLaneId();

  auto flushPending = [&] {
    const int nPending = group.control.nPendingTokens;
    for (int idx = 0; idx < nPending; ++idx) {
      combineFinalize<Hidden>(output, group.control.pendingTokens[idx], context, gateway, group);
    }
    if (nPending > 0 && group.threadIdx == 0) group.control.nPendingTokens = 0;
  };

  for (int tokenIdx = localWorkerIdx; tokenIdx < workload.numTokens_; tokenIdx += nLocalWorkers) {
    const bool hasNextToken = tokenIdx + nLocalWorkers < workload.numTokens_;
    const bool overlapRemote = hasNextToken || group.control.nPendingTokens > 0;
    const IntraDomainRoute laneRoute =
        group.warpIdx == 0
            ? resolveLocalRoute<Mode>(tokenIdx, laneIdx, topkIndices, topkWeights, workload, context, workspace)
            : IntraDomainRoute{};
    mscclpp::f32x8 tokenSum[MaxVectorsPerThread];
    reduceIntraDomainToken<Mode, Hidden, DispatchType, MaxNTopk>(
        tokenSum, context->rank_, laneRoute, expertOutput, workload, transport, workspace, flag, dequantScale, group,
        [&] {
          if (overlapRemote) {
            const int remoteTokenIdx = hierarchicalCtx.combineRemoteTokenIndices_[tokenIdx];
            group.control.interDomainTokenReady = remoteTokenIdx < 0 || gateway.poll(remoteTokenIdx);
          }
        });

    if (overlapRemote && !group.control.interDomainTokenReady) {
      // Remote RDMA partial is still in flight; stash local FP32 sum so earlier pending tokens or the next local token
      // can proceed without blocking on this RDMA chunk.
      auto* localScratch = reinterpret_cast<mscclpp::f32x8*>(hierarchicalCtx.combineLocalBuffer_ +
                                                             static_cast<size_t>(tokenIdx) * Hidden);
#pragma unroll
      for (int stepIdx = 0; stepIdx < MaxVectorsPerThread; ++stepIdx) {
        const int vectorIdx = group.threadIdx + stepIdx * HierarchicalCombineGroupThreads;
        if (vectorIdx < NHiddenVectors) localScratch[vectorIdx] = tokenSum[stepIdx];
      }
      if (group.threadIdx == 0) group.control.pendingTokens[group.control.nPendingTokens++] = tokenIdx;
      group.sync();
      if (group.control.nPendingTokens == HierarchicalCombineMaxPendingTokens) {
        flushPending();
        group.sync();
      }
      continue;
    }

    const int remoteTokenIdx = hierarchicalCtx.combineRemoteTokenIndices_[tokenIdx];
    if (!overlapRemote && remoteTokenIdx >= 0) {
      if (group.threadIdx == 0) gateway.wait(remoteTokenIdx);
      group.sync();
    }
#pragma unroll
    for (int stepIdx = 0; stepIdx < MaxVectorsPerThread; ++stepIdx) {
      const int vectorIdx = group.threadIdx + stepIdx * HierarchicalCombineGroupThreads;
      if (vectorIdx < NHiddenVectors) {
        combineStoreFinalVector<Hidden>(output, tokenSum[stepIdx], tokenIdx, remoteTokenIdx, vectorIdx, context);
      }
    }
    if (hasNextToken) group.sync();
  }
  flushPending();
}

MSCCLPP_DEVICE_INLINE void combineControl(const DeviceContext* context, const TransportView& transport,
                                          const WorkspaceView& workspace, uint32_t flag) {
  const int peerIdx = threadIdx.x;
  const int nRanks = context->numRanks_;
  const auto& hierarchicalCtx = context->hierarchicalCtx_;
  if (peerIdx < nRanks && transport.isIntraDomainPeer(peerIdx)) {
    if (!transport.isSelf(peerIdx)) {
      transport.baseMemoryChannels_[peerIdx].signal();
      transport.baseMemoryChannels_[peerIdx].wait(-1);
    }
    mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(workspace.combineRankReadyFlags_ + peerIdx, flag,
                                                         mscclpp::memoryOrderRelease);
  }
  // Combine resets the dispatch send counters and waits for the peer to finish forwarding this round.
  for (int idx = threadIdx.x; idx < nRanks; idx += blockDim.x) {
    workspace.dispatchRankPayloadSlots_[idx] = 0;
    workspace.dispatchRankPayloadCompletions_[idx] = 0;
  }
  if (threadIdx.x == 0) {
    hierarchicalCtx.dispatchScratch_->nRemoteTokens = 0;
    hierarchicalCtx.dispatchScratch_->nCompletedForwarders = 0;
    hierarchicalCtx.dispatchScratch_->forwardComplete = 0;
  }
  const Gateway dispatchGateway(
      transport, hierarchicalCtx.gatewayRank_, hierarchicalCtx.nTokensPerChunk_, 0, hierarchicalCtx.dispatchSendBuffer_,
      hierarchicalCtx.dispatchRecvBuffer_, hierarchicalCtx.dispatchControl_->qpSignals,
      hierarchicalCtx.dispatchScratch_->chunkReadySignals, &hierarchicalCtx.dispatchControl_->completionSignal);
  dispatchGateway.reset<Gateway::Scope::ThreadGroup>(hierarchicalCtx.maxTokensPerRank_, static_cast<int>(threadIdx.x),
                                                     static_cast<int>(blockDim.x), /*resetRecv=*/false);
  if (threadIdx.x == 0) dispatchGateway.waitForwardComplete();
}

#endif

template <CombineMode Mode, int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout,
          int MaxNTopk>
__global__ __launch_bounds__(CombineNThreads, 1) void hierarchicalCombineKernel(
    void* output, const void* expertOutput, const int64_t* topkIndices, const float* topkWeights, const int* srcInfo,
    const int64_t* layoutRange, Workload workload, void* combineRecvBuffer, const void* dispatchRecvBuffer,
    const DeviceContext* context) {
#if defined(MSCCLPP_DEVICE_CUDA) && defined(USE_IBVERBS) && MSCCLPP_BULK_AVAILABLE
  if constexpr (Layout == DispatchLayout::RANK_MAJOR &&
                (DispatchType == DispatchDataType::BF16 || DispatchType == DispatchDataType::FP8_E4M3_STATIC)) {
    extern __shared__ __align__(128) uint8_t sharedMemory[];
    const TransportView transport(context);
    const auto& hierarchicalCtx = context->hierarchicalCtx_;
    const size_t wireElemBytes = useBf16InterDomainCombine(hierarchicalCtx) ? sizeof(Bf16) : sizeof(float);
    Gateway gateway(transport, hierarchicalCtx.gatewayRank_, hierarchicalCtx.nTokensPerChunk_,
                    static_cast<size_t>(Hidden) * wireElemBytes, hierarchicalCtx.combineSendBuffer_,
                    hierarchicalCtx.combineRecvBuffer_, hierarchicalCtx.combineControl_->qpSignals,
                    hierarchicalCtx.combineChunkReady_, nullptr, hierarchicalCtx.combineChunkLoadReady_);
    const WorkspaceView workspace(context->workspace_, context->numRanks_, workload.numExperts_);
    const uint32_t flag = getDispatchFlag(workspace);
    constexpr int StageSlots = hierarchicalCombineGroupStageSlots<Hidden, DispatchType, MaxNTopk>();
    constexpr size_t TokensBytes = static_cast<size_t>(StageSlots) * Hidden * sizeof(DispatchElementType<DispatchType>);
    constexpr size_t GroupSharedBytes = hierarchicalCombineGroupSharedBytesForSlots<Hidden, DispatchType, StageSlots>();
    const int warpIdx = static_cast<int>(threadIdx.x) / WARP_SIZE;
    const int groupIdx = warpIdx / HierarchicalCombineGroupWarps;
    const int groupWarpIdx = warpIdx % HierarchicalCombineGroupWarps;
    const int warpGroupThreadIdx = groupWarpIdx * WARP_SIZE + getLaneId();
    // Named barrier 0 is reserved for CTA-wide __syncthreads().
    const int warpGroupBarrierId = groupIdx + 1;
    uint8_t* groupShared = sharedMemory + static_cast<size_t>(groupIdx) * GroupSharedBytes;
    auto& groupControl = *reinterpret_cast<HierarchicalCombineGroupControl<StageSlots>*>(groupShared + TokensBytes);
    const float dequantScale = DispatchType == DispatchDataType::FP8_E4M3_STATIC ? 1.0f / workload.combineScale_ : 1.0f;
    uint32_t phase = 0;
    const HierarchicalCombineGroup<StageSlots> group{groupShared,  groupControl,       phase,
                                                     groupWarpIdx, warpGroupThreadIdx, warpGroupBarrierId};

    if (group.threadIdx == 0) {
      group.control.barrier.init();
      group.control.nPendingTokens = 0;
      group.control.interDomainTokenReady = false;
    }
    group.sync();

    if (blockIdx.x == 0) {
      combineControl(context, transport, workspace, flag);
      const int queueIdx = warpIdx - configCellDiv(context->numRanks_, WARP_SIZE);
      const int nQps = transport.portChannelCtx_->numQpsPerPeer;
      if (queueIdx >= 0 && queueIdx < nQps && getLaneId() == 0) {
        gateway.send(queueIdx, queueIdx * gateway.nTokensPerChunk(), nQps * gateway.nTokensPerChunk(),
                     *hierarchicalCtx.nIncomingTokens_);
      }
      __syncthreads();
    } else {
      // Reserve block 0 for control and QP posting; stripe each warp group across the remaining blocks first.
      const int nGatewayBlocks = static_cast<int>(gridDim.x) - 1;
      const int tokenIdx = groupIdx * nGatewayBlocks + (static_cast<int>(blockIdx.x) - 1);
      const int tokenStride = nGatewayBlocks * HierarchicalCombineNWarpGroups;
      gateway.reduce<Gateway::Scope::ThreadGroup>(
          tokenIdx, tokenStride, *hierarchicalCtx.nIncomingTokens_, group.threadIdx, HierarchicalCombineGroupThreads,
          [&] { group.sync(); },
          [&](uint8_t* dst, int idx, auto&& markLoadReady) {
            combineGatewayToken<Mode, Hidden, DispatchType, MaxNTopk>(dst, expertOutput, idx, workload, context,
                                                                      transport, gateway, workspace, flag, dequantScale,
                                                                      group, markLoadReady);
          });
    }

    // Assign local workers in reverse group/block order so small-T local work lands on idle groups/SMs first and
    // block 0 / gateway tail workers sit at the end of the range.
    const int nLocalBlocks = static_cast<int>(gridDim.x);
    const int nLocalWorkers = nLocalBlocks * HierarchicalCombineNWarpGroups;
    const int revBlockIdx = blockIdx.x == 0 ? nLocalBlocks - 1 : nLocalBlocks - 1 - static_cast<int>(blockIdx.x);
    const int localWorkerIdx = (HierarchicalCombineNWarpGroups - 1 - groupIdx) * nLocalBlocks + revBlockIdx;
    combineLocalTokens<Mode, Hidden, DispatchType, MaxNTopk>(output, expertOutput, topkIndices, topkWeights, workload,
                                                             context, transport, gateway, workspace, flag, dequantScale,
                                                             group, localWorkerIdx, nLocalWorkers);
    __syncthreads();
    if (group.threadIdx == 0) group.control.barrier.invalidate();
  }
#endif
}

template <CombineMode Mode>
struct HierarchicalCombineKernelSelector {
  template <int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout, int MaxNTopk>
  static auto get() {
    return hierarchicalCombineKernel<Mode, Hidden, DispatchType, ScaleBlockSize, Layout, MaxNTopk>;
  }
};

void hierarchicalCombine(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights,
                         const Workload& workload, CombineMode combineMode, void* recvBuffer, void* dispatchRecvBuffer,
                         const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  EP_HOST_ASSERT(context.hierarchicalCtx_.portChannelCtx_ != nullptr && workload.numTopk_ <= MaxNumTopk);
  EP_HOST_ASSERT(workload.outputLayout_ == DispatchLayout::RANK_MAJOR);
  EP_HOST_ASSERT(workload.dispatchDataType_ == DispatchDataType::BF16 ||
                 workload.dispatchDataType_ == DispatchDataType::FP8_E4M3_STATIC ||
                 workload.dispatchDataType_ == DispatchDataType::MXFP8_E4M3);
  EP_HOST_ASSERT(isSupportedCombineDataType(workload.combineDataType_));
  if (combineMode == CombineMode::DIRECT_SEND) {
    combineAlgorithm<CombineMode::DIRECT_SEND, HierarchicalCombineKernelSelector<CombineMode::DIRECT_SEND>, true>(
        output, input, topkIdx, topkWeights, nullptr, nullptr, workload, recvBuffer, dispatchRecvBuffer, context,
        numBlocks, stream);
  } else {
    EP_HOST_ASSERT(combineMode == CombineMode::RANK_LOCAL_REDUCE);
    combineAlgorithm<CombineMode::RANK_LOCAL_REDUCE, HierarchicalCombineKernelSelector<CombineMode::RANK_LOCAL_REDUCE>,
                     true>(output, input, topkIdx, nullptr, nullptr, nullptr, workload, recvBuffer, dispatchRecvBuffer,
                           context, numBlocks, stream);
  }
}

}  // namespace mscclpp::ep
