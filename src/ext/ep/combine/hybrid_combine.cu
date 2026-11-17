// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "combine/common.cuh"
#include "common/gateway.cuh"

namespace mscclpp::ep {

#if defined(MSCCLPP_DEVICE_CUDA) && defined(USE_IBVERBS) && MSCCLPP_BULK_AVAILABLE

template <DispatchDataType DispatchType>
MSCCLPP_DEVICE_INLINE bool useBf16CombinePartialSum(const HybridDeviceContext& hybridCtx) {
  if constexpr (DispatchType == DispatchDataType::FP8_E4M3_STATIC) {
    return true;
  } else {
    return hybridCtx.combineBf16PartialSum_;
  }
}

template <int Hidden, DispatchDataType DispatchType>
MSCCLPP_DEVICE_INLINE void aggregateStoreFinalVector(void* output, mscclpp::f32x8 sum, int tokenIdx, int remoteTokenIdx,
                                                     int vectorIdx, const DeviceContext* context) {
  constexpr int NHiddenVectors = Hidden / mscclpp::f32x8::Size;
  constexpr int NPairsPerVector = mscclpp::f32x8::Size / mscclpp::f32x2::Size;
  const auto& hybridCtx = context->hybridCtx_;
  if (remoteTokenIdx >= 0) {
    const size_t remoteVecIdx = static_cast<size_t>(remoteTokenIdx) * NHiddenVectors + vectorIdx;
    const mscclpp::f32x8 partial =
        useBf16CombinePartialSum<DispatchType>(hybridCtx)
            ? mscclpp::to<mscclpp::f32x8>(
                  reinterpret_cast<const mscclpp::bf16x8*>(hybridCtx.aggregateRecvBuffer_)[remoteVecIdx])
            : reinterpret_cast<const mscclpp::f32x8*>(hybridCtx.aggregateRecvBuffer_)[remoteVecIdx];
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
MSCCLPP_DEVICE_INLINE void aggregateFinalize(void* output, int tokenIdx, const DeviceContext* context,
                                             const Gateway& gateway, int groupThreadIdx, int groupBarrierId) {
  constexpr int NHiddenVectors = Hidden / mscclpp::f32x8::Size;
  const auto& hybridCtx = context->hybridCtx_;
  const int remoteTokenIdx = hybridCtx.aggregateRemoteTokenIndices_[tokenIdx];
  if (groupThreadIdx == 0 && remoteTokenIdx >= 0) gateway.wait(remoteTokenIdx);
  syncNamedBarrier(groupBarrierId, HybridCombineGroupThreads);
  const auto* local =
      reinterpret_cast<const mscclpp::f32x8*>(hybridCtx.aggregateLocalBuffer_ + static_cast<size_t>(tokenIdx) * Hidden);
  for (int vectorIdx = groupThreadIdx; vectorIdx < NHiddenVectors; vectorIdx += HybridCombineGroupThreads) {
    aggregateStoreFinalVector<Hidden, DispatchType>(output, local[vectorIdx], tokenIdx, remoteTokenIdx, vectorIdx,
                                                    context);
  }
}

template <CombineMode Mode, int Hidden, DispatchDataType DispatchType, int StageSlots>
MSCCLPP_DEVICE_INLINE void accumulateHybridVector(mscclpp::f32x8& sum, int vectorIdx, const uint8_t* sharedTokens,
                                                  const float* routeWeights, int nRoundRoutes, float dequantScale) {
  constexpr bool IsDirectSend = Mode == CombineMode::DIRECT_SEND;
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(DispatchElementType<DispatchType>);
  constexpr int NPairsPerVector = mscclpp::f32x8::Size / mscclpp::f32x2::Size;
  auto* sumPairs = reinterpret_cast<mscclpp::f32x2*>(&sum);
#pragma unroll
  for (int stageIdx = 0; stageIdx < StageSlots; ++stageIdx) {
    if (stageIdx >= nRoundRoutes) break;
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

template <CombineMode Mode, int Hidden, DispatchDataType DispatchType, int MaxNTopk, bool IsGateway,
          bool FuseFinal = false, typename LoadDoneFn = void (*)()>
MSCCLPP_DEVICE_INLINE void aggregateReduceToken(void* destination, const void* expertOutput, const int64_t* topkIndices,
                                                const float* topkWeights, int tokenIdx, const Workload& workload,
                                                const DeviceContext* context, const TransportView& transport,
                                                const Gateway& gateway, const WorkspaceView& workspace, uint32_t flag,
                                                uint8_t* groupShared, uint32_t& phase, float dequantScale,
                                                int groupWarpIdx, int groupThreadIdx, int groupBarrierId,
                                                LoadDoneFn&& markLoadDone, int localWorkerIdx = 0,
                                                int nLocalWorkers = 1, int waveIdx = 0) {
  static_assert(!(IsGateway && FuseFinal));
  constexpr bool IsDirectSend = Mode == CombineMode::DIRECT_SEND;
  using ElementType = DispatchElementType<DispatchType>;
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(ElementType);
  constexpr int NHiddenVectors = Hidden / mscclpp::f32x8::Size;
  constexpr int MaxVectorsPerThread = configCellDiv(NHiddenVectors, HybridCombineGroupThreads);
  constexpr int NPairsPerVector = mscclpp::f32x8::Size / mscclpp::f32x2::Size;
  constexpr int StageSlots = hybridCombineGroupStageSlots<Hidden, DispatchType, MaxNTopk>();
  constexpr bool SingleRound = StageSlots >= MaxNTopk;
  constexpr size_t TokensBytes = static_cast<size_t>(StageSlots) * HiddenBytes;
  const auto& hybridCtx = context->hybridCtx_;
  const int laneIdx = getLaneId();
  const int nTopk = workload.numTopk_;
  const int nLocalExperts = workload.numExperts_ / context->numRanks_;
  const int ownerRank = IsGateway ? gateway.peerRank() : context->rank_;
  auto& groupControl = *reinterpret_cast<HybridCombineGroupControl<StageSlots>*>(groupShared + TokensBytes);

  int lanePeerIdx = -1;
  int laneSlotIdx = -1;
  float laneWeight = 1.0f;
  bool active = false;
  uint32_t validRouteMask = 0;
  int nActiveRoutes = 0;
  int compactIdx = 0;
  if (groupWarpIdx == 0) {
    int64_t expertIdx = -1;
    if (laneIdx < nTopk) {
      if constexpr (IsGateway) {
        const auto* cached = hybridCtx.aggregateRoutes_ + static_cast<size_t>(tokenIdx) * 3 * nTopk;
        expertIdx = cached[laneIdx];
        laneSlotIdx = cached[nTopk + laneIdx];
        if (IsDirectSend && workload.applyRouterWeights_) {
          laneWeight = reinterpret_cast<const float*>(cached + 2 * nTopk)[laneIdx];
        }
      } else {
        expertIdx = topkIndices[tokenIdx * nTopk + laneIdx];
        laneSlotIdx = workspace.rankMajorSendIndices_[tokenIdx * nTopk + laneIdx];
        if (IsDirectSend && workload.applyRouterWeights_ && topkWeights != nullptr) {
          laneWeight = topkWeights[tokenIdx * nTopk + laneIdx];
        }
      }
    }
    lanePeerIdx = expertIdx >= 0 && expertIdx < workload.numExperts_ ? static_cast<int>(expertIdx) / nLocalExperts : -1;
    const bool isLeader = isFirstLaneForRank(lanePeerIdx, laneIdx);
    active =
        laneIdx < MaxNTopk && lanePeerIdx >= 0 && (IsDirectSend || isLeader) && transport.isNvlinkPeer(lanePeerIdx);
    validRouteMask = __ballot_sync(0xffffffff, active);
    nActiveRoutes = __popc(validRouteMask);
    compactIdx = laneIdx < MaxNTopk ? __popc(validRouteMask & ((1u << laneIdx) - 1u)) : 0;
  }

  mscclpp::f32x8 tileSum[MaxVectorsPerThread];
#pragma unroll
  for (int stepIdx = 0; stepIdx < MaxVectorsPerThread; ++stepIdx) {
    auto* sumPairs = reinterpret_cast<mscclpp::f32x2*>(&tileSum[stepIdx]);
#pragma unroll
    for (int pairIdx = 0; pairIdx < NPairsPerVector; ++pairIdx) {
      sumPairs[pairIdx].storage = make_float2(0.0f, 0.0f);
    }
  }

  int roundIdx = 0;
  int nTotalRounds = 1;
  do {
    if (groupWarpIdx == 0) {
      const int nRounds = SingleRound ? 1 : max(1, configCellDiv(nActiveRoutes, StageSlots));
      const int routeBegin = roundIdx * StageSlots;
      const int nRoundRoutes = SingleRound ? nActiveRoutes : max(0, min(StageSlots, nActiveRoutes - routeBegin));
      const bool roundActive =
          SingleRound ? active : (active && compactIdx >= routeBegin && compactIdx < routeBegin + StageSlots);
      const int stageIdx = SingleRound ? compactIdx : (compactIdx - routeBegin);
      if (roundActive) {
        if constexpr (IsDirectSend) groupControl.routeWeights[stageIdx] = laneWeight;
      }
      if (laneIdx == 0) {
        groupControl.nRoundRoutes = nRoundRoutes;
        if constexpr (!SingleRound) groupControl.nTotalRounds = nRounds;
        if (nRoundRoutes > 0) {
          groupControl.barrier.arriveAndExpect(static_cast<uint32_t>(nRoundRoutes * HiddenBytes));
        }
      }
      __syncwarp();
      if (nRoundRoutes > 0) {
        bool pending = roundActive;
        while (__any_sync(0xffffffff, pending)) {
          if (pending && mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(
                             workspace.combineRankReadyFlags_ + lanePeerIdx, mscclpp::memoryOrderAcquire) == flag) {
            EP_DEVICE_ASSERT(laneSlotIdx >= 0 && laneSlotIdx < workload.maxTokensPerRank_);
            const auto* peerOutput =
                static_cast<const uint8_t*>(transport.mappedBuffer(const_cast<void*>(expertOutput), lanePeerIdx));
            const size_t sourceTokenIdx = static_cast<size_t>(ownerRank) * workload.maxTokensPerRank_ + laneSlotIdx;
            const size_t inputTokenIdx = IsDirectSend ? sourceTokenIdx * nTopk + laneIdx : sourceTokenIdx;
            mscclpp::bulkLoad(groupShared + static_cast<size_t>(stageIdx) * HiddenBytes,
                              peerOutput + inputTokenIdx * HiddenBytes, HiddenBytes, groupControl.barrier);
            pending = false;
          }
        }
        if (laneIdx == 0) {
          groupControl.barrier.wait(phase);
          mscclpp::bulkFence();
        }
        __syncwarp();
      }
      if (laneIdx == 0 && roundIdx == nRounds - 1) {
        if constexpr (IsGateway) {
          markLoadDone();
        } else if constexpr (!FuseFinal) {
          const int remoteTokenIdx = hybridCtx.aggregateRemoteTokenIndices_[tokenIdx];
          const bool ready = waveIdx < 64 && (remoteTokenIdx < 0 || gateway.poll(remoteTokenIdx));
          groupControl.remoteReady = ready;
          if (ready) groupControl.finalizedMask |= (uint64_t{1} << waveIdx);
        }
      }
    }
    syncNamedBarrier(groupBarrierId, HybridCombineGroupThreads);

    if constexpr (!SingleRound) nTotalRounds = groupControl.nTotalRounds;
    const int nRoundRoutes = groupControl.nRoundRoutes;
    if (nRoundRoutes > 0) {
#pragma unroll
      for (int stepIdx = 0; stepIdx < MaxVectorsPerThread; ++stepIdx) {
        const int vectorIdx = groupThreadIdx + stepIdx * HybridCombineGroupThreads;
        if (vectorIdx < NHiddenVectors) {
          accumulateHybridVector<Mode, Hidden, DispatchType, StageSlots>(
              tileSum[stepIdx], vectorIdx, groupShared, groupControl.routeWeights, nRoundRoutes, dequantScale);
        }
      }
    }
    ++roundIdx;
    if constexpr (!SingleRound) {
      if (roundIdx < nTotalRounds) syncNamedBarrier(groupBarrierId, HybridCombineGroupThreads);
    }
  } while (!SingleRound && roundIdx < nTotalRounds);

  if constexpr (FuseFinal) {
    // Keep the last token's local sum in registers while finalizing any unfinalized earlier tokens from scratch.
    const uint64_t finalizedMask = groupControl.finalizedMask;
    for (int prevIdx = localWorkerIdx, prevWave = 0; prevIdx < tokenIdx; prevIdx += nLocalWorkers, ++prevWave) {
      if (prevWave < 64 && ((finalizedMask >> prevWave) & uint64_t{1}) != 0) continue;
      aggregateFinalize<Hidden, DispatchType>(destination, prevIdx, context, gateway, groupThreadIdx, groupBarrierId);
    }
    const int remoteTokenIdx = hybridCtx.aggregateRemoteTokenIndices_[tokenIdx];
    if (remoteTokenIdx >= 0) {
      if (groupThreadIdx == 0) gateway.wait(remoteTokenIdx);
      syncNamedBarrier(groupBarrierId, HybridCombineGroupThreads);
    }
#pragma unroll
    for (int stepIdx = 0; stepIdx < MaxVectorsPerThread; ++stepIdx) {
      const int vectorIdx = groupThreadIdx + stepIdx * HybridCombineGroupThreads;
      if (vectorIdx < NHiddenVectors) {
        aggregateStoreFinalVector<Hidden, DispatchType>(destination, tileSum[stepIdx], tokenIdx, remoteTokenIdx,
                                                        vectorIdx, context);
      }
    }
  } else if constexpr (IsGateway) {
    const bool bf16PartialSum = useBf16CombinePartialSum<DispatchType>(hybridCtx);
#pragma unroll
    for (int stepIdx = 0; stepIdx < MaxVectorsPerThread; ++stepIdx) {
      const int vectorIdx = groupThreadIdx + stepIdx * HybridCombineGroupThreads;
      if (vectorIdx < NHiddenVectors) {
        if (bf16PartialSum) {
          reinterpret_cast<mscclpp::bf16x8*>(destination)[vectorIdx] = mscclpp::to<mscclpp::bf16x8>(tileSum[stepIdx]);
        } else {
          reinterpret_cast<mscclpp::f32x8*>(destination)[vectorIdx] = tileSum[stepIdx];
        }
      }
    }
    // The group barrier joins every writer into group thread 0's release counter and protects sharedMemory reuse.
    syncNamedBarrier(groupBarrierId, HybridCombineGroupThreads);
  } else {
    if (groupControl.remoteReady) {
      const int remoteTokenIdx = hybridCtx.aggregateRemoteTokenIndices_[tokenIdx];
#pragma unroll
      for (int stepIdx = 0; stepIdx < MaxVectorsPerThread; ++stepIdx) {
        const int vectorIdx = groupThreadIdx + stepIdx * HybridCombineGroupThreads;
        if (vectorIdx < NHiddenVectors) {
          aggregateStoreFinalVector<Hidden, DispatchType>(destination, tileSum[stepIdx], tokenIdx, remoteTokenIdx,
                                                          vectorIdx, context);
        }
      }
    } else {
      auto* localScratch =
          reinterpret_cast<mscclpp::f32x8*>(hybridCtx.aggregateLocalBuffer_ + static_cast<size_t>(tokenIdx) * Hidden);
#pragma unroll
      for (int stepIdx = 0; stepIdx < MaxVectorsPerThread; ++stepIdx) {
        const int vectorIdx = groupThreadIdx + stepIdx * HybridCombineGroupThreads;
        if (vectorIdx < NHiddenVectors) {
          localScratch[vectorIdx] = tileSum[stepIdx];
        }
      }
    }
    syncNamedBarrier(groupBarrierId, HybridCombineGroupThreads);
  }
}

template <int Hidden, DispatchDataType DispatchType>
MSCCLPP_DEVICE_INLINE void aggregateControl(const DeviceContext* context, const TransportView& transport,
                                            const Gateway& gateway, const WorkspaceView& workspace, uint32_t flag) {
  const int peerIdx = threadIdx.x;
  const int nRanks = context->numRanks_;
  const auto& hybridCtx = context->hybridCtx_;
  const int nQps = transport.portChannels_->numQpsPerPeer;
  const int queueIdx = static_cast<int>(threadIdx.x) / WARP_SIZE - configCellDiv(nRanks, WARP_SIZE);
  if (peerIdx < nRanks && transport.isNvlinkPeer(peerIdx)) {
    if (!transport.isSelf(peerIdx)) {
      transport.baseMemoryChannels_[peerIdx].signal();
      transport.baseMemoryChannels_[peerIdx].wait(-1);
    }
    mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(workspace.combineRankReadyFlags_ + peerIdx, flag,
                                                         mscclpp::memoryOrderRelease);
  }
  if (queueIdx >= 0 && queueIdx < nQps && getLaneId() == 0) {
    const size_t wireElemBytes = useBf16CombinePartialSum<DispatchType>(hybridCtx) ? sizeof(Bf16) : sizeof(float);
    gateway.forward(queueIdx, *hybridCtx.nIncomingTokens_, static_cast<size_t>(Hidden) * wireElemBytes);
  }
  // Combine resets the dispatch counters, then waits for permission to reuse the RDMA landing buffer.
  for (int idx = threadIdx.x; idx < nRanks; idx += blockDim.x) {
    workspace.dispatchRankPayloadSlots_[idx] = 0;
    workspace.dispatchRankPayloadCompletions_[idx] = 0;
  }
  if (threadIdx.x == 0) {
    hybridCtx.distributeScratch_->nRemoteTokens = 0;
    hybridCtx.distributeScratch_->nCompletedForwarders = 0;
    hybridCtx.distributeScratch_->forwardComplete = 0;
  }
  const Gateway dispatchGateway(transport, hybridCtx.gatewayRank_, hybridCtx.nTokensPerChunk_,
                                hybridCtx.distributeSendBuffer_, hybridCtx.distributeRecvBuffer_,
                                hybridCtx.distributeControl_->qpSignalCount,
                                hybridCtx.distributeScratch_->chunkReadySignals);
  dispatchGateway.reset(hybridCtx.distributeCapacity_, false, nullptr, &hybridCtx.distributeControl_->bufferReuseReady);
}

#endif

template <CombineMode Mode, int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout,
          int MaxNTopk>
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
    const auto& hybridCtx = context->hybridCtx_;
    const Gateway gateway(transport, hybridCtx.gatewayRank_, hybridCtx.nTokensPerChunk_, hybridCtx.aggregateSendBuffer_,
                          hybridCtx.aggregateRecvBuffer_, hybridCtx.aggregateControl_->qpSignalCount,
                          hybridCtx.aggregateChunkReady_);
    const WorkspaceView workspace(context->workspace_, context->numRanks_, workload.numExperts_);
    const uint32_t flag = getDispatchFlag(workspace);
    constexpr int StageSlots = hybridCombineGroupStageSlots<Hidden, DispatchType, MaxNTopk>();
    constexpr size_t TokensBytes = static_cast<size_t>(StageSlots) * Hidden * sizeof(DispatchElementType<DispatchType>);
    constexpr size_t GroupSharedBytes = hybridCombineGroupSharedBytesForSlots<Hidden, DispatchType, StageSlots>();
    const int warpIdx = static_cast<int>(threadIdx.x) / WARP_SIZE;
    const int groupIdx = warpIdx / HybridCombineGroupWarps;
    const int groupWarpIdx = warpIdx % HybridCombineGroupWarps;
    const int groupThreadIdx = groupWarpIdx * WARP_SIZE + getLaneId();
    const int groupBarrierId = HybridCombineGroupBarrierBase + groupIdx;
    uint8_t* groupShared = sharedMemory + static_cast<size_t>(groupIdx) * GroupSharedBytes;
    auto& groupControl = *reinterpret_cast<HybridCombineGroupControl<StageSlots>*>(groupShared + TokensBytes);
    const float dequantScale = DispatchType == DispatchDataType::FP8_E4M3_STATIC ? 1.0f / workload.combineScale_ : 1.0f;
    uint32_t phase = 0;

    if (blockIdx.x == 0) {
      aggregateControl<Hidden, DispatchType>(context, transport, gateway, workspace, flag);
      __syncthreads();
    }
    if (groupThreadIdx == 0) {
      groupControl.barrier.init();
      groupControl.finalizedMask = 0;
      groupControl.remoteReady = false;
    }
    syncNamedBarrier(groupBarrierId, HybridCombineGroupThreads);

    if (blockIdx.x != 0) {
      const size_t wireElemBytes = useBf16CombinePartialSum<DispatchType>(hybridCtx) ? sizeof(Bf16) : sizeof(float);
      gateway.reduceAndForward<HybridCombineNWarpGroups, HybridCombineGroupThreads>(
          *hybridCtx.nIncomingTokens_, static_cast<size_t>(Hidden) * wireElemBytes, hybridCtx.aggregateChunkLoadDone_,
          groupIdx, groupThreadIdx, groupBarrierId, [&](uint8_t* dst, int tokenIdx, auto&& markLoadDone) {
            aggregateReduceToken<Mode, Hidden, DispatchType, MaxNTopk, true>(
                dst, expertOutput, topkIndices, topkWeights, tokenIdx, workload, context, transport, gateway, workspace,
                flag, groupShared, phase, dequantScale, groupWarpIdx, groupThreadIdx, groupBarrierId, markLoadDone);
          });
    }

    // Assign local workers in reverse group/block order so small-T local work lands on idle groups/SMs first and
    // block 0 / gateway tail workers sit at the end of the range.
    const int nLocalBlocks = static_cast<int>(gridDim.x);
    const int nLocalWorkers = nLocalBlocks * HybridCombineNWarpGroups;
    const int revBlockIdx = blockIdx.x == 0 ? nLocalBlocks - 1 : nLocalBlocks - 1 - static_cast<int>(blockIdx.x);
    const int localWorkerIdx = (HybridCombineNWarpGroups - 1 - groupIdx) * nLocalBlocks + revBlockIdx;
    for (int tokenIdx = localWorkerIdx, waveIdx = 0; tokenIdx < workload.numTokens_;
         tokenIdx += nLocalWorkers, ++waveIdx) {
      if (tokenIdx + nLocalWorkers < workload.numTokens_) {
        aggregateReduceToken<Mode, Hidden, DispatchType, MaxNTopk, false, false>(
            output, expertOutput, topkIndices, topkWeights, tokenIdx, workload, context, transport, gateway, workspace,
            flag, groupShared, phase, dequantScale, groupWarpIdx, groupThreadIdx, groupBarrierId, [] {}, localWorkerIdx,
            nLocalWorkers, waveIdx);
      } else {
        aggregateReduceToken<Mode, Hidden, DispatchType, MaxNTopk, false, true>(
            output, expertOutput, topkIndices, topkWeights, tokenIdx, workload, context, transport, gateway, workspace,
            flag, groupShared, phase, dequantScale, groupWarpIdx, groupThreadIdx, groupBarrierId, [] {}, localWorkerIdx,
            nLocalWorkers, waveIdx);
      }
    }
    __syncthreads();
    if (groupThreadIdx == 0) groupControl.barrier.invalidate();
  }
#endif
}

template <CombineMode Mode>
struct HybridCombineKernelSelector {
  template <int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout, int MaxNTopk>
  static auto get() {
    return hybridCombineKernel<Mode, Hidden, DispatchType, ScaleBlockSize, Layout, MaxNTopk>;
  }
};

void hybridCombine(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights,
                   const Workload& workload, CombineMode combineMode, void* recvBuffer, void* dispatchRecvBuffer,
                   const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  EP_HOST_ASSERT(context.hybridCtx_.portChannels_ != nullptr && workload.numTopk_ <= MaxNumTopk);
  EP_HOST_ASSERT(workload.outputLayout_ == DispatchLayout::RANK_MAJOR);
  EP_HOST_ASSERT(workload.dispatchDataType_ == DispatchDataType::BF16 ||
                 workload.dispatchDataType_ == DispatchDataType::FP8_E4M3_STATIC ||
                 workload.dispatchDataType_ == DispatchDataType::MXFP8_E4M3);
  EP_HOST_ASSERT(isSupportedCombineDataType(workload.combineDataType_));
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
