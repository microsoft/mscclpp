// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "common/gateway.cuh"
#include "common/quantization.cuh"
#include "dispatch/common.cuh"

namespace mscclpp::ep {

#if defined(MSCCLPP_DEVICE_CUDA) && defined(USE_IBVERBS) && MSCCLPP_BULK_AVAILABLE

MSCCLPP_DEVICE_INLINE int hierarchicalNumForwardBlocks(int nGridBlocks) {
  const int nForwardBlocks = (nGridBlocks - 1) / 2;
  return nForwardBlocks < HierarchicalMaxForwardBlocks ? nForwardBlocks : HierarchicalMaxForwardBlocks;
}

MSCCLPP_DEVICE_INLINE int hierarchicalNumProduceGroups(int nTokens, int nProducers) {
  static_assert(HierarchicalMaxProduceGroups == 4);
  return nTokens <= nProducers ? 1 : (nTokens <= 2 * nProducers ? 2 : HierarchicalMaxProduceGroups);
}

MSCCLPP_DEVICE_INLINE void dispatchCountRoutes(const int64_t* routes, const Workload& workload,
                                               const TransportView& transport, int* rankTokenCounts,
                                               int& nRemoteTokens) {
  static_assert(WARP_SIZE % MaxNumTopk == 0);
  constexpr int NTokensPerWarp = WARP_SIZE / MaxNumTopk;
  const int laneIdx = getLaneId();
  const int warpIdx = static_cast<int>(threadIdx.x) / WARP_SIZE;
  const int subWarpIdx = laneIdx / MaxNumTopk;
  const int subLaneIdx = laneIdx % MaxNumTopk;
  const unsigned int subWarpMask = ((1u << MaxNumTopk) - 1u) << (subWarpIdx * MaxNumTopk);
  const int nLocalExperts = workload.numExperts_ / transport.numRanks_;
  for (int tokenBase = warpIdx * NTokensPerWarp; tokenBase < workload.numTokens_;
       tokenBase += DispatchNWarps * NTokensPerWarp) {
    const int tokenIdx = tokenBase + subWarpIdx;
    const int64_t rawExpertIdx = tokenIdx < workload.numTokens_ && subLaneIdx < workload.numTopk_
                                     ? routes[tokenIdx * workload.numTopk_ + subLaneIdx]
                                     : -1;
    const int expertIdx =
        rawExpertIdx >= 0 && rawExpertIdx < workload.numExperts_ ? static_cast<int>(rawExpertIdx) : -1;
    const int dstRank = expertIdx >= 0 ? expertIdx / nLocalExperts : -1;
    const unsigned int sameRankMask = __match_any_sync(0xffffffff, dstRank) & subWarpMask;
    if (dstRank >= 0 && (__ffs(sameRankMask) - 1 == laneIdx)) atomicAdd_block(rankTokenCounts + dstRank, 1);
    const unsigned int remoteMask =
        __ballot_sync(0xffffffff, dstRank >= 0 && !transport.isIntraDomainPeer(dstRank)) & subWarpMask;
    if (subLaneIdx == 0 && remoteMask != 0) atomicAdd_block(&nRemoteTokens, 1);
  }
}

// One warp stores a staged token, optional MXFP8 scales, and masked top-k metadata to intra-domain destinations.
template <int Hidden, DispatchDataType DataType>
MSCCLPP_DEVICE_INLINE void sendTokenIntraDomain(void* output, void* outputScales, int* outputIds, float* outputWeights,
                                                const void* staged, int sourceRank, int dstRank, int slotIdx,
                                                int expertIdx, float weight, bool isLocalLeader,
                                                const Workload& workload, const TransportView& transport) {
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(DispatchElementType<DataType>);
  const int nLocalExperts = workload.numExperts_ / transport.numRanks_;
  if (isLocalLeader) {
    EP_DEVICE_ASSERT(slotIdx >= 0 && slotIdx < workload.maxTokensPerRank_);
    auto* dstBuffer = static_cast<uint8_t*>(transport.mappedBuffer(output, dstRank));
    const size_t dstTokenIdx = static_cast<size_t>(sourceRank) * workload.maxTokensPerRank_ + slotIdx;
    mscclpp::bulkStore(dstBuffer + dstTokenIdx * HiddenBytes, staged, HiddenBytes);
    if constexpr (DataType == DispatchDataType::MXFP8_E4M3) {
      constexpr size_t ScaleBytes = Hidden / 32;
      auto* dstScales = static_cast<uint8_t*>(transport.mappedBuffer(outputScales, dstRank)) + dstTokenIdx * ScaleBytes;
      const auto* stagedScales = static_cast<const uint8_t*>(staged) + HiddenBytes;
      if constexpr (ScaleBytes % 16 == 0) {
        mscclpp::bulkStore(dstScales, stagedScales, ScaleBytes);
      } else {
        // Preserve contiguous scale rows for widths such as 4352, whose 136B scale row is not TMA-aligned.
        static_assert(ScaleBytes % sizeof(uint64_t) == 0);
        for (size_t i = 0; i < ScaleBytes / sizeof(uint64_t); ++i) {
          reinterpret_cast<uint64_t*>(dstScales)[i] = reinterpret_cast<const uint64_t*>(stagedScales)[i];
        }
      }
    }
    mscclpp::bulkStoreCommit();
  }
  const unsigned int leaderMask = __ballot_sync(0xffffffff, isLocalLeader);
  const int outputExpertIdx = workload.emitLocalExpertIds_ && expertIdx >= 0 ? expertIdx % nLocalExperts : expertIdx;
  writeRankMajorMetadataIntraDomain(transport, outputIds, outputWeights, sourceRank, dstRank, slotIdx, outputExpertIdx,
                                    weight, leaderMask, workload.numTopk_, workload.maxTokensPerRank_,
                                    workload.invalidTokenExpertId_);
  if (isLocalLeader) mscclpp::bulkStoreWait();
}

// Producer blocks: quantize or TMA-load input tokens with SM-first token striding, prioritize inter-domain
// enqueuing into dispatchSendBuffer_ so RDMA chunks post early, and then send intra-domain tokens.
template <int Hidden, DispatchDataType DataType>
MSCCLPP_DEVICE_INLINE void dispatchProduce(void* output, void* outputScales, int* outputIds, float* outputWeights,
                                           const void* input, const int64_t* routes, const float* weights,
                                           const Workload& workload, const DeviceContext* context,
                                           const TransportView& transport, const Gateway& gateway,
                                           uint8_t* sharedMemory, int producerIdx, int nProducers) {
  static_assert(DispatchNWarps % HierarchicalMaxProduceGroups == 0);
  const int nGroups = hierarchicalNumProduceGroups(workload.numTokens_, nProducers);
  const int nWarpsPerGroup = DispatchNWarps / nGroups;
  const int nGroupThreads = nWarpsPerGroup * WARP_SIZE;
  const int warpIdx = static_cast<int>(threadIdx.x) / WARP_SIZE;
  const int laneIdx = getLaneId();
  const int groupIdx = warpIdx / nWarpsPerGroup;
  const int subWarpIdx = warpIdx % nWarpsPerGroup;
  const int groupThreadIdx = subWarpIdx * WARP_SIZE + laneIdx;
  const int groupBarrierId = DispatchWarpGroupBarrierBase + groupIdx;

  using ElementType = DispatchElementType<DataType>;
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(ElementType);
  constexpr size_t ScaleBytes = DataType == DispatchDataType::MXFP8_E4M3 ? Hidden / 32 : 0;
  constexpr size_t PayloadBytes = InterDomainDispatchPayload::payloadBytes(Hidden, sizeof(ElementType), ScaleBytes);
  constexpr size_t Stride = hierarchicalDispatchStageStride<Hidden, DataType>();
  const auto& hierarchicalCtx = context->hierarchicalCtx_;
  const int nTopk = workload.numTopk_;
  const int nLocalExperts = workload.numExperts_ / context->numRanks_;
  auto* base = sharedMemory + dispatchSharedControlBytes(context->numRanks_, true);
  auto* staged = base + static_cast<size_t>(groupIdx) * Stride;
  auto& barrier = reinterpret_cast<mscclpp::BulkBarrier*>(base + static_cast<size_t>(HierarchicalMaxProduceGroups) *
                                                                     Stride)[groupIdx];
  __shared__ int groupRemoteTokenIndices[HierarchicalMaxProduceGroups];
  const bool quantizeInput =
      DataType == DispatchDataType::MXFP8_E4M3 ||
      (DataType == DispatchDataType::FP8_E4M3_STATIC && workload.inputMode_ == QuantConfig::InputMode::QUANTIZE);
  // Stripe tokens across producer blocks first to spread small workloads across SMs.
  const int firstTokenIdx = groupIdx * nProducers + producerIdx;
  const int tokenStride = nProducers * nGroups;
  if (firstTokenIdx >= workload.numTokens_) return;
  if (!quantizeInput && groupThreadIdx == 0) barrier.init();
  syncNamedBarrier(groupBarrierId, nGroupThreads);

  uint32_t phase = 0;
  WorkspaceView workspace(context->workspace_, context->numRanks_, workload.numExperts_);
  for (int tokenIdx = firstTokenIdx; tokenIdx < workload.numTokens_; tokenIdx += tokenStride) {
    if (quantizeInput) {
      if constexpr (DataType == DispatchDataType::FP8_E4M3_STATIC || DataType == DispatchDataType::MXFP8_E4M3) {
        constexpr int NHiddenVectors = Hidden / mscclpp::bf16x8::Size;
        const auto* inputTokens =
            static_cast<const mscclpp::bf16x8*>(input) + static_cast<size_t>(tokenIdx) * NHiddenVectors;
        auto* stagedTokens = reinterpret_cast<mscclpp::f8_e4m3x8*>(staged);
        for (int vectorIdx = groupThreadIdx; vectorIdx < NHiddenVectors; vectorIdx += nGroupThreads) {
          if constexpr (DataType == DispatchDataType::MXFP8_E4M3) {
            uint8_t scale = 0;
            stagedTokens[vectorIdx] = quantizeBf16x8ToMxfp8(inputTokens[vectorIdx], scale, laneIdx);
            if (laneIdx % 4 == 0) staged[HiddenBytes + vectorIdx / 4] = scale;
          } else {
            stagedTokens[vectorIdx] = quantizeBf16x8ToStaticFp8(inputTokens[vectorIdx], workload.dispatchScale_);
          }
        }
        if constexpr (DataType == DispatchDataType::MXFP8_E4M3 && ScaleBytes % 16 != 0) {
          if (groupThreadIdx == 0) *reinterpret_cast<uint64_t*>(staged + HiddenBytes + ScaleBytes) = 0;
        }
      }
    } else if (groupThreadIdx == 0) {
      barrier.arriveAndExpect(HiddenBytes);
      mscclpp::bulkLoad(staged, static_cast<const uint8_t*>(input) + static_cast<size_t>(tokenIdx) * HiddenBytes,
                        HiddenBytes, barrier);
    }

    RankMajorRoute route{-1, -1, false, -1};
    float weight = 0.0f;
    bool isRemote = false;
    if (subWarpIdx == 0) {
      route = prepareRankMajorRoute(workspace, routes, tokenIdx, nTopk, nLocalExperts, workload.maxTokensPerRank_,
                                    laneIdx, workload.numExperts_);
      weight = route.expertIdx >= 0 ? (weights == nullptr ? 1.0f : weights[tokenIdx * nTopk + laneIdx]) : 0.0f;
      isRemote = route.dstRank >= 0 && !transport.isIntraDomainPeer(route.dstRank);
      const bool hasRemote = __any_sync(0xffffffff, isRemote);
      int remoteTokenIdx = -1;
      if (laneIdx == 0 && hasRemote) remoteTokenIdx = atomicAdd(&hierarchicalCtx.dispatchScratch_->nRemoteTokens, 1);
      remoteTokenIdx = __shfl_sync(0xffffffff, remoteTokenIdx, 0);
      if (laneIdx == 0) {
        hierarchicalCtx.combineRemoteTokenIndices_[tokenIdx] = remoteTokenIdx;
        groupRemoteTokenIndices[groupIdx] = remoteTokenIdx;
      }
      if (hasRemote && laneIdx < nTopk) {
        EP_DEVICE_ASSERT(remoteTokenIdx < hierarchicalCtx.maxTokensPerRank_);
        auto* sendToken = static_cast<uint8_t*>(hierarchicalCtx.dispatchSendBuffer_) +
                          static_cast<size_t>(remoteTokenIdx) * gateway.tokenBytes();
        const InterDomainDispatchPayload payload(sendToken, Hidden, nTopk, sizeof(ElementType), ScaleBytes);
        payload.expertIds_[laneIdx] = route.expertIdx;
        payload.topkWeights_[laneIdx] = weight;
        payload.dstSlots_[laneIdx] = route.destinationSlot;
      }
    }

    if (!quantizeInput && groupThreadIdx == 0) barrier.wait(phase);
    mscclpp::bulkFence();
    syncNamedBarrier(groupBarrierId, nGroupThreads);

    gateway.enqueue<PayloadBytes>(groupRemoteTokenIndices[groupIdx], staged, groupThreadIdx, nGroupThreads,
                                  [&] { syncNamedBarrier(groupBarrierId, nGroupThreads); });

    if (subWarpIdx == 0) {
      const bool isLocalLeader = route.dstRank >= 0 && route.isLeader && !isRemote;
      sendTokenIntraDomain<Hidden, DataType>(output, outputScales, outputIds, outputWeights, staged, transport.rank_,
                                             route.dstRank, route.destinationSlot, route.expertIdx, weight,
                                             isLocalLeader, workload, transport);
      if (isLocalLeader) {
        (void)mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(
            workspace.dispatchRankPayloadCompletions_ + route.dstRank, 1, mscclpp::memoryOrderRelease);
      }
    }
    if (tokenIdx + tokenStride < workload.numTokens_) {
      syncNamedBarrier(groupBarrierId, nGroupThreads);
    }
  }
  if (!quantizeInput && groupThreadIdx == 0) barrier.invalidate();
}

// Warps after the QP posters publish intra-domain token counts and signal peers once local TMA stores complete.
MSCCLPP_DEVICE_INLINE void dispatchPublishIntraDomain(void* recvBuffer, const int* rankTokenCounts,
                                                      const TransportView& transport, const WorkspaceView& workspace,
                                                      int nQps, uint32_t flag) {
  const int dstRank = static_cast<int>(threadIdx.x) - nQps * WARP_SIZE;
  if (dstRank < 0 || dstRank >= transport.numRanks_ || !transport.isIntraDomainPeer(dstRank)) return;

  const int nTokens = rankTokenCounts[dstRank];
  auto* counts = static_cast<mscclpp::LL8Packet*>(transport.mappedBuffer(recvBuffer, dstRank));
  counts[transport.rank_].write(nTokens, flag);
  if (nTokens == 0) return;

  while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(workspace.dispatchRankPayloadCompletions_ + dstRank,
                                                        mscclpp::memoryOrderAcquire) != nTokens) {
  }
  if (transport.isSelf(dstRank)) {
    workspace.dispatchLocalPayloadReady_->release();
  } else {
    transport.baseMemoryChannels_[dstRank].signal();
  }
}

// Block 0 counts routes, publishes intra-domain readiness, and posts gateway RDMA transfers concurrently.
MSCCLPP_DEVICE_INLINE void dispatchControl(const int64_t* routes, const Workload& workload, void* recvBuffer,
                                           const DeviceContext* context, const TransportView& transport,
                                           Gateway& gateway, uint32_t flag, int* rankTokenCounts) {
  const int nRanks = context->numRanks_;
  const int laneIdx = getLaneId();
  const int warpIdx = static_cast<int>(threadIdx.x) / WARP_SIZE;
  const auto& hierarchicalCtx = context->hierarchicalCtx_;
  auto* state = hierarchicalCtx.dispatchControl_;
  const int nQps = transport.portChannelCtx_->numQpsPerPeer;
  __shared__ int nRemoteTokens;
  WorkspaceView workspace(context->workspace_, nRanks, workload.numExperts_);
  const Gateway combineGateway(transport, hierarchicalCtx.gatewayRank_, hierarchicalCtx.nTokensPerChunk_, 0,
                               hierarchicalCtx.combineSendBuffer_, hierarchicalCtx.combineRecvBuffer_,
                               hierarchicalCtx.combineControl_->qpSignals, hierarchicalCtx.combineChunkReady_);

  for (int peerIdx = threadIdx.x; peerIdx < nRanks; peerIdx += blockDim.x) rankTokenCounts[peerIdx] = 0;
  if (threadIdx.x == 0) nRemoteTokens = 0;
  combineGateway.reset(hierarchicalCtx.maxTokensPerRank_, static_cast<int>(threadIdx.x), static_cast<int>(blockDim.x),
                       true, hierarchicalCtx.combineChunkLoadReady_);
  __syncthreads();
  dispatchCountRoutes(routes, workload, transport, rankTokenCounts, nRemoteTokens);
  __syncthreads();

  for (int peerIdx = threadIdx.x; peerIdx < nRanks; peerIdx += blockDim.x) {
    state->sendHeader.nTokensPerRank[peerIdx].write(rankTokenCounts[peerIdx], flag);
  }
  if (threadIdx.x == 0) {
    state->sendHeader.nRemoteTokens.write(nRemoteTokens, flag);
  }
  __syncthreads();

  dispatchPublishIntraDomain(recvBuffer, rankTokenCounts, transport, workspace, nQps, flag);
  if (warpIdx < nQps && laneIdx == 0) {
    const bool isControlQueue = warpIdx == HierarchicalDispatchControlQueueIdx;
    if (isControlQueue) {
      gateway.sendHeader(warpIdx, &state->recvHeader, &state->sendHeader, InterDomainDispatchHeader::wireBytes(nRanks));
    }
    const int nTokensPerChunk = gateway.nTokensPerChunk();
    gateway.send(warpIdx, warpIdx * nTokensPerChunk, nQps * nTokensPerChunk, nRemoteTokens);
    if (isControlQueue) {
      gateway.complete(warpIdx, &hierarchicalCtx.dispatchScratch_->forwardComplete);
    }
  }
}

// Forwarder blocks: receive header, forward incoming gateway tokens within the domain, and signal intra-domain workers.
template <int Hidden, DispatchDataType DataType>
MSCCLPP_DEVICE_INLINE void dispatchForward(void* output, void* outputScales, int* outputIds, float* outputWeights,
                                           const Workload& workload, void* recvBuffer, const DeviceContext* context,
                                           const TransportView& transport, const Gateway& gateway, uint32_t flag,
                                           uint8_t* sharedMemory, int forwarderIdx, int nForwarders) {
  using ElementType = DispatchElementType<DataType>;
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(ElementType);
  constexpr size_t ScaleBytes = DataType == DispatchDataType::MXFP8_E4M3 ? Hidden / 32 : 0;
  constexpr size_t Stride = hierarchicalDispatchStageStride<Hidden, DataType>();
  const auto& hierarchicalCtx = context->hierarchicalCtx_;
  auto* state = hierarchicalCtx.dispatchControl_;
  const int laneIdx = getLaneId();
  const int nTopk = workload.numTopk_;
  const int nLocalExperts = workload.numExperts_ / context->numRanks_;
  auto* base = sharedMemory + dispatchSharedControlBytes(context->numRanks_, true);

  __shared__ int nIncomingTokens;
  __shared__ bool isLastForwarder;
  if (threadIdx.x == 0) {
    nIncomingTokens = state->recvHeader.nRemoteTokens.read(flag, -1);
    if (forwarderIdx == 0) *hierarchicalCtx.nIncomingTokens_ = nIncomingTokens;
  }
  if (forwarderIdx == 0) {
    for (int peerIdx = threadIdx.x; peerIdx < context->numRanks_; peerIdx += blockDim.x) {
      if (transport.isIntraDomainPeer(peerIdx)) {
        auto* counts = static_cast<mscclpp::LL8Packet*>(transport.mappedBuffer(recvBuffer, peerIdx));
        counts[gateway.peerRank()].write(state->recvHeader.nTokensPerRank[peerIdx].read(flag, -1), flag);
      }
    }
  }
  __syncthreads();

  constexpr int NWarpGroupThreads = HierarchicalForwardWarps * WARP_SIZE;
  static_assert(NWarpGroupThreads <= DispatchNThreads);
  static_assert(NWarpGroupThreads >= MaxNumRanks);
  const int warpGroupThreadIdx = static_cast<int>(threadIdx.x);
  if (warpGroupThreadIdx < NWarpGroupThreads) {
    constexpr int warpGroupBarrierId = DispatchWarpGroupBarrierBase;
    int expertIdx = -1;
    int slotIdx = -1;
    float weight = 0.0f;
    gateway.forward<HiddenBytes, Stride, HierarchicalForwardWarps, ScaleBytes>(
        base, output, outputScales, nIncomingTokens, forwarderIdx, nForwarders, warpGroupThreadIdx, warpGroupBarrierId,
        [&](int tokenIdx, void* landing) {
          const InterDomainDispatchPayload payload(landing, Hidden, nTopk, sizeof(ElementType), ScaleBytes);
          expertIdx = laneIdx < nTopk ? payload.expertIds_[laneIdx] : -1;
          slotIdx = laneIdx < nTopk ? payload.dstSlots_[laneIdx] : -1;
          weight = laneIdx < nTopk ? payload.topkWeights_[laneIdx] : 0.0f;
          if (laneIdx < nTopk) {
            // Preserve routes for combine before releasing recvBuffer for the next dispatch round.
            auto* cached = hierarchicalCtx.combineRoutes_ + static_cast<size_t>(tokenIdx) * 3 * nTopk;
            cached[laneIdx] = expertIdx;
            cached[nTopk + laneIdx] = slotIdx;
            reinterpret_cast<float*>(cached + 2 * nTopk)[laneIdx] = weight;
          }
          const int dstRank = expertIdx >= 0 ? expertIdx / nLocalExperts : -1;
          EP_DEVICE_ASSERT(dstRank < 0 || !transport.isIntraDomainPeer(dstRank) ||
                           (slotIdx >= 0 && slotIdx < workload.maxTokensPerRank_));
          const size_t dstSlotIndex =
              static_cast<size_t>(gateway.peerRank()) * workload.maxTokensPerRank_ + max(0, slotIdx);
          return Gateway::ForwardRoute{dstRank, dstSlotIndex};
        },
        [&](int, const Gateway::ForwardRoute& route, unsigned int leaderMask) {
          const int outputExpertIdx =
              workload.emitLocalExpertIds_ && expertIdx >= 0 ? expertIdx % nLocalExperts : expertIdx;
          writeRankMajorMetadataIntraDomain(transport, outputIds, outputWeights, gateway.peerRank(), route.dstRank,
                                            slotIdx, outputExpertIdx, weight, leaderMask, nTopk,
                                            workload.maxTokensPerRank_, workload.invalidTokenExpertId_);
        });
    gateway.signalIntraDomain(&hierarchicalCtx.dispatchScratch_->nCompletedForwarders, nForwarders,
                              &hierarchicalCtx.dispatchScratch_->forwardComplete, state->payloadReady, flag,
                              &isLastForwarder, warpGroupThreadIdx, NWarpGroupThreads, warpGroupBarrierId);
  }
}

#endif

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
__global__ __launch_bounds__(DispatchNThreads, 1) void hierarchicalDispatchKernel(
    void* output, void* outputScales, int* outputSrcInfo, int* outputIds, float* outputWeights, int64_t* outputLayout,
    int* outputCount, const int64_t* routes, const float* weights, const void* input, Workload workload,
    void* recvBuffer, const DeviceContext* context) {
#if defined(MSCCLPP_DEVICE_CUDA) && defined(USE_IBVERBS) && MSCCLPP_BULK_AVAILABLE
  static_assert(DataType == DispatchDataType::BF16 || DataType == DispatchDataType::FP8_E4M3_STATIC ||
                DataType == DispatchDataType::MXFP8_E4M3);
  constexpr size_t ScaleBytes = DataType == DispatchDataType::MXFP8_E4M3 ? Hidden / 32 : 0;
  extern __shared__ __align__(128) uint8_t sharedMemory[];
  WorkspaceView workspace(context->workspace_, context->numRanks_, workload.numExperts_);
  const uint32_t flag = getDispatchFlag(workspace) + 1;
  const TransportView transport(context);
  const auto& hierarchicalCtx = context->hierarchicalCtx_;
  const size_t stride =
      InterDomainDispatchPayload::stride(Hidden, workload.numTopk_, sizeof(DispatchElementType<DataType>), ScaleBytes);
  Gateway gateway(transport, hierarchicalCtx.gatewayRank_, hierarchicalCtx.nTokensPerChunk_, stride,
                  hierarchicalCtx.dispatchSendBuffer_, hierarchicalCtx.dispatchRecvBuffer_,
                  hierarchicalCtx.dispatchControl_->qpSignals, hierarchicalCtx.dispatchScratch_->chunkReadySignals,
                  &hierarchicalCtx.dispatchControl_->completionSignal);
  if (blockIdx.x == 0) {
    dispatchControl(routes, workload, recvBuffer, context, transport, gateway, flag,
                    reinterpret_cast<int*>(sharedMemory));
  } else {
    const int nGridBlocks = static_cast<int>(gridDim.x);
    const int nWorkers = nGridBlocks - 1;
    const int nForwarders = hierarchicalNumForwardBlocks(nGridBlocks);
    const int workerIdx = static_cast<int>(blockIdx.x) - 1;
    if (workerIdx < nForwarders) {
      dispatchForward<Hidden, DataType>(output, outputScales, outputIds, outputWeights, workload, recvBuffer, context,
                                        transport, gateway, flag, sharedMemory, workerIdx, nForwarders);
    } else {
      const int producerIdx = workerIdx - nForwarders;
      const int nProducers = nWorkers - nForwarders;
      if (producerIdx < workload.numTokens_) {
        dispatchProduce<Hidden, DataType>(output, outputScales, outputIds, outputWeights, input, routes, weights,
                                          workload, context, transport, gateway, sharedMemory, producerIdx, nProducers);
      }
    }
    for (int sourceRank = workerIdx; sourceRank < context->numRanks_; sourceRank += nWorkers) {
      __syncthreads();
      dispatchRankMajorRecvSource(outputIds, outputWeights, outputCount, transport, workspace, sourceRank,
                                  workload.numTopk_, workload.maxTokensPerRank_, workload.invalidTokenExpertId_,
                                  recvBuffer, flag, reinterpret_cast<int*>(sharedMemory),
                                  context->hierarchicalCtx_.dispatchControl_->payloadReady);
    }
  }
  updateDispatchFlag(workspace, flag);
#endif
}

struct HierarchicalDispatchKernelSelector {
  template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
  static auto get() {
    return hierarchicalDispatchKernel<Hidden, DataType, ScaleBlockSize>;
  }
};

void hierarchicalDispatch(void* output, void* outputScales, int* outputTopkIdx, float* outputTopkWeights,
                          int* outputCount, const void* input, const int64_t* topkIdx, const float* topkWeights,
                          const Workload& workload, void* recvBuffer, const DeviceContext& context, int numBlocks,
                          cudaStream_t stream) {
  EP_HOST_ASSERT(context.hierarchicalCtx_.portChannelCtx_ != nullptr);
  dispatchAlgorithm<DispatchLayout::RANK_MAJOR, HierarchicalDispatchKernelSelector, true>(
      output, outputScales, nullptr, outputTopkIdx, outputTopkWeights, nullptr, outputCount, input, topkIdx,
      topkWeights, workload, recvBuffer, context, numBlocks, stream);
}

}  // namespace mscclpp::ep
