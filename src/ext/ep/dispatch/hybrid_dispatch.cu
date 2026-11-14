// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "common/quantization.cuh"
#include "dispatch/common.cuh"

namespace mscclpp::ep {

#if defined(MSCCLPP_DEVICE_CUDA) && defined(USE_IBVERBS) && MSCCLPP_BULK_AVAILABLE

MSCCLPP_DEVICE_INLINE int hybridNumForwardBlocks(int nGridBlocks) {
  const int nForwardBlocks = (nGridBlocks - 1) / 2;
  return nForwardBlocks < HybridMaxForwardBlocks ? nForwardBlocks : HybridMaxForwardBlocks;
}

MSCCLPP_DEVICE_INLINE int hybridNumProduceGroups(int nTokens, int nProducers) {
  static_assert(HybridMaxProduceGroups == 4);
  return nTokens <= nProducers ? 1 : (nTokens <= 2 * nProducers ? 2 : HybridMaxProduceGroups);
}

MSCCLPP_DEVICE_INLINE void distributeCountRoutes(const int64_t* routes, const Workload& workload,
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
        __ballot_sync(0xffffffff, dstRank >= 0 && !transport.isNvlinkPeer(dstRank)) & subWarpMask;
    if (subLaneIdx == 0 && remoteMask != 0) atomicAdd_block(&nRemoteTokens, 1);
  }
}

// One warp stores a staged token and its masked top-k metadata to all NVLink-local destination ranks.
template <int Hidden, DispatchDataType DataType>
MSCCLPP_DEVICE_INLINE void sendNvlinkToken(void* output, int* outputIds, float* outputWeights, const void* staged,
                                           int sourceRank, int dstRank, int slotIdx, int expertIdx, float weight,
                                           bool isLocalLeader, const Workload& workload,
                                           const TransportView& transport) {
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(DispatchElementType<DataType>);
  const int nLocalExperts = workload.numExperts_ / transport.numRanks_;
  if (isLocalLeader) {
    EP_DEVICE_ASSERT(slotIdx >= 0 && slotIdx < workload.maxTokensPerRank_);
    auto* dstBuffer = static_cast<uint8_t*>(transport.mappedBuffer(output, dstRank));
    const size_t dstTokenIdx = static_cast<size_t>(sourceRank) * workload.maxTokensPerRank_ + slotIdx;
    mscclpp::bulkStore(dstBuffer + dstTokenIdx * HiddenBytes, staged, HiddenBytes);
    mscclpp::bulkStoreCommit();
  }
  const unsigned int leaderMask = __ballot_sync(0xffffffff, isLocalLeader);
  const int outputExpertIdx = workload.emitLocalExpertIds_ && expertIdx >= 0 ? expertIdx % nLocalExperts : expertIdx;
  writeRankMajorMetadataNvlink(transport, outputIds, outputWeights, sourceRank, dstRank, slotIdx, outputExpertIdx,
                               weight, leaderMask, workload.numTopk_, workload.maxTokensPerRank_,
                               workload.invalidTokenExpertId_);
  if (isLocalLeader) mscclpp::bulkStoreWait();
}

// Producer blocks: quantize or TMA-load input tokens with SM-first token striding, prioritize cross-domain
// staging into distributeSendBuffer_ so RDMA chunks post early, and then send NVLink-local tokens.
template <int Hidden, DispatchDataType DataType>
MSCCLPP_DEVICE_INLINE void distributeProduce(void* output, int* outputIds, float* outputWeights, const void* input,
                                             const int64_t* routes, const float* weights, const Workload& workload,
                                             const DeviceContext* context, const TransportView& transport,
                                             uint8_t* sharedMemory, int producerIdx, int nProducers) {
  static_assert(DispatchNWarps % HybridMaxProduceGroups == 0);
  const int nGroups = hybridNumProduceGroups(workload.numTokens_, nProducers);
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
  const auto& hybridCtx = context->hybridCtx_;
  const int nTopk = workload.numTopk_;
  const int nTokensPerChunk = hybridCtx.nTokensPerChunk_;
  const int nLocalExperts = workload.numExperts_ / context->numRanks_;
  const size_t stride = InterRackDispatchPayload::stride(Hidden, nTopk, sizeof(ElementType));
  auto* tileBase = sharedMemory + dispatchSharedControlBytes(context->numRanks_, true);
  auto* staged = tileBase + static_cast<size_t>(groupIdx) * HiddenBytes;
  auto& barrier = reinterpret_cast<mscclpp::BulkBarrier*>(tileBase + static_cast<size_t>(HybridMaxProduceGroups) *
                                                                         HiddenBytes)[groupIdx];
  __shared__ int groupRemoteTokenIndices[HybridMaxProduceGroups];
  const bool quantizeInput =
      DataType == DispatchDataType::FP8_E4M3_STATIC && workload.inputMode_ == QuantConfig::InputMode::QUANTIZE;
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
      if constexpr (DataType == DispatchDataType::FP8_E4M3_STATIC) {
        constexpr int NHiddenVectors = Hidden / mscclpp::bf16x8::Size;
        const auto* inputTokens =
            static_cast<const mscclpp::bf16x8*>(input) + static_cast<size_t>(tokenIdx) * NHiddenVectors;
        auto* stagedTokens = reinterpret_cast<mscclpp::f8_e4m3x8*>(staged);
        for (int vectorIdx = groupThreadIdx; vectorIdx < NHiddenVectors; vectorIdx += nGroupThreads) {
          stagedTokens[vectorIdx] = quantizeBf16x8ToStaticFp8(inputTokens[vectorIdx], workload.dispatchScale_);
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
      isRemote = route.dstRank >= 0 && !transport.isNvlinkPeer(route.dstRank);
      const bool hasRemote = __any_sync(0xffffffff, isRemote);
      int remoteTokenIdx = -1;
      if (laneIdx == 0 && hasRemote) remoteTokenIdx = atomicAdd(&hybridCtx.distributeScratch_->nRemoteTokens, 1);
      remoteTokenIdx = __shfl_sync(0xffffffff, remoteTokenIdx, 0);
      if (laneIdx == 0) {
        hybridCtx.aggregateRemoteTokenIndices_[tokenIdx] = remoteTokenIdx;
        groupRemoteTokenIndices[groupIdx] = remoteTokenIdx;
      }
      if (hasRemote && laneIdx < nTopk) {
        EP_DEVICE_ASSERT(remoteTokenIdx < hybridCtx.distributeCapacity_);
        auto* sendToken =
            static_cast<uint8_t*>(hybridCtx.distributeSendBuffer_) + static_cast<size_t>(remoteTokenIdx) * stride;
        const InterRackDispatchPayload payload(sendToken, Hidden, nTopk, sizeof(ElementType));
        payload.expertIds_[laneIdx] = route.expertIdx;
        payload.topkWeights_[laneIdx] = weight;
        payload.dstSlots_[laneIdx] = route.destinationSlot;
      }
    }

    if (!quantizeInput && groupThreadIdx == 0) barrier.wait(phase);
    mscclpp::bulkFence();
    syncNamedBarrier(groupBarrierId, nGroupThreads);

    const int remoteTokenIdx = groupRemoteTokenIndices[groupIdx];
    if (remoteTokenIdx >= 0) {
      auto* sendToken =
          static_cast<uint8_t*>(hybridCtx.distributeSendBuffer_) + static_cast<size_t>(remoteTokenIdx) * stride;
      for (int vectorIdx = groupThreadIdx; vectorIdx < static_cast<int>(HiddenBytes / sizeof(mscclpp::u32x4));
           vectorIdx += nGroupThreads) {
        reinterpret_cast<mscclpp::u32x4*>(sendToken)[vectorIdx] =
            reinterpret_cast<const mscclpp::u32x4*>(staged)[vectorIdx];
      }
      syncNamedBarrier(groupBarrierId, nGroupThreads);
      if (groupThreadIdx == 0) {
        atomicAddReleaseDevice(hybridCtx.distributeScratch_->chunkReadySignals + remoteTokenIdx / nTokensPerChunk, 1);
      }
    }

    if (subWarpIdx == 0) {
      const bool isLocalLeader = route.dstRank >= 0 && route.isLeader && !isRemote;
      sendNvlinkToken<Hidden, DataType>(output, outputIds, outputWeights, staged, transport.rank_, route.dstRank,
                                        route.destinationSlot, route.expertIdx, weight, isLocalLeader, workload,
                                        transport);
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

// One lane per QP posts outgoing chunks; QP 0 also sends the header and the incoming-buffer reuse ACK.
template <int Hidden, DispatchDataType DataType>
MSCCLPP_DEVICE_INLINE void distributePostChunks(const Workload& workload, const DeviceContext* context,
                                                const TransportView& transport, int queueIdx, int nRemoteTokens) {
  constexpr auto kExclusive = mscclpp::PortChannelDeviceHandle::Concurrency::Exclusive;
  const auto& hybridCtx = context->hybridCtx_;
  auto* state = hybridCtx.distributeControl_;
  auto* portChannels = transport.portChannels_;
  const int nQps = portChannels->numQpsPerPeer;
  const int nTokensPerChunk = hybridCtx.nTokensPerChunk_;
  const size_t stride =
      InterRackDispatchPayload::stride(Hidden, workload.numTopk_, sizeof(DispatchElementType<DataType>));
  auto handle = portChannels->channel(hybridCtx.gatewayRank_, queueIdx);
  const uint64_t dstOffset = transport.symmetricOffset(hybridCtx.distributeRecvBuffer_);
  const uint64_t srcOffset = transport.symmetricOffset(hybridCtx.distributeSendBuffer_);
  const uint64_t flagOffset = transport.symmetricOffset(state->qpSignalCount + queueIdx);
  if (queueIdx == 0) {
    const uint64_t recvHeader = transport.symmetricOffset(&state->recvHeader);
    const uint64_t sendHeader = transport.symmetricOffset(&state->sendHeader);
    const size_t headerBytes = InterRackDispatchHeader::wireBytes(context->numRanks_);
    handle.put<kExclusive, MaxQpsPerPeer>(recvHeader, sendHeader, headerBytes, queueIdx);
  }
  for (int chunkIdx = queueIdx, tokenBegin = queueIdx * nTokensPerChunk; tokenBegin < nRemoteTokens;
       chunkIdx += nQps, tokenBegin += nQps * nTokensPerChunk) {
    const int nChunkTokens = min(nTokensPerChunk, nRemoteTokens - tokenBegin);
    handle.device_.putWithAccumulateAfter<kExclusive, MaxQpsPerPeer>(
        handle.dstMemory_, dstOffset + tokenBegin * stride, handle.srcMemory_, srcOffset + tokenBegin * stride,
        nChunkTokens * stride, handle.dstMemory_, flagOffset, 1,
        [&] {
          while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(
                     hybridCtx.distributeScratch_->chunkReadySignals + chunkIdx, mscclpp::memoryOrderAcquire) !=
                 nChunkTokens) {
          }
        },
        queueIdx);
  }
  if (queueIdx == 0) {
    const auto readyOffset = transport.symmetricOffset(&state->bufferReuseReady);
    handle.device_.accumulateAfter<kExclusive, MaxQpsPerPeer>(
        handle.dstMemory_, readyOffset, 1,
        [&] {
          while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(&hybridCtx.distributeScratch_->forwardComplete,
                                                                mscclpp::memoryOrderAcquire) != 1) {
          }
        },
        queueIdx);
  }
}

// Warps after the QP posters publish NVLink token counts and signal peers once local TMA stores complete.
MSCCLPP_DEVICE_INLINE void distributePublishNvlink(void* recvBuffer, const int* rankTokenCounts,
                                                   const TransportView& transport, const WorkspaceView& workspace,
                                                   int nQps, uint32_t flag) {
  const int dstRank = static_cast<int>(threadIdx.x) - nQps * WARP_SIZE;
  if (dstRank < 0 || dstRank >= transport.numRanks_ || !transport.isNvlinkPeer(dstRank)) return;

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

// Block 0 counts routes, publishes NVLink readiness, and posts RDMA transfers concurrently.
template <int Hidden, DispatchDataType DataType>
MSCCLPP_DEVICE_INLINE void distributeControl(const int64_t* routes, const Workload& workload, void* recvBuffer,
                                             const DeviceContext* context, const TransportView& transport,
                                             uint32_t flag, int* rankTokenCounts) {
  const int nRanks = context->numRanks_;
  const int laneIdx = getLaneId();
  const int warpIdx = static_cast<int>(threadIdx.x) / WARP_SIZE;
  const auto& hybridCtx = context->hybridCtx_;
  auto* state = hybridCtx.distributeControl_;
  const int nQps = transport.portChannels_->numQpsPerPeer;
  const int nTokensPerChunk = hybridCtx.nTokensPerChunk_;
  __shared__ int nRemoteTokens;
  WorkspaceView workspace(context->workspace_, nRanks, workload.numExperts_);

  for (int peerIdx = threadIdx.x; peerIdx < nRanks; peerIdx += blockDim.x) rankTokenCounts[peerIdx] = 0;
  if (threadIdx.x == 0) nRemoteTokens = 0;
  // Dispatch initializes the counters consumed by the following combine round.
  for (int queueIdx = threadIdx.x; queueIdx < nQps; queueIdx += blockDim.x) {
    hybridCtx.aggregateControl_->qpSignalCount[queueIdx] = 0;
  }
  const int nCombineCounters = configCellDiv(hybridCtx.distributeCapacity_, nTokensPerChunk);
  for (int idx = threadIdx.x; idx < nCombineCounters; idx += blockDim.x) {
    hybridCtx.aggregateChunkReady_[idx] = 0;
    hybridCtx.aggregateChunkLoadDone_[idx] = 0;
  }
  __syncthreads();
  distributeCountRoutes(routes, workload, transport, rankTokenCounts, nRemoteTokens);
  __syncthreads();

  for (int peerIdx = threadIdx.x; peerIdx < nRanks; peerIdx += blockDim.x) {
    state->sendHeader.nTokensPerRank[peerIdx].write(rankTokenCounts[peerIdx], flag);
  }
  if (threadIdx.x == 0) {
    state->sendHeader.nRemoteTokens.write(nRemoteTokens, flag);
  }
  __syncthreads();

  distributePublishNvlink(recvBuffer, rankTokenCounts, transport, workspace, nQps, flag);
  if (warpIdx < nQps && laneIdx == 0) {
    distributePostChunks<Hidden, DataType>(workload, context, transport, warpIdx, nRemoteTokens);
  }
}

// Forwarder blocks: wait for incoming RDMA chunks from the paired gateway rank and forward them over
// NVLink to their destination ranks in the local domain.
template <int Hidden, DispatchDataType DataType>
MSCCLPP_DEVICE_INLINE void distributeForward(void* output, int* outputIds, float* outputWeights,
                                             const Workload& workload, void* recvBuffer, const DeviceContext* context,
                                             const TransportView& transport, uint32_t flag, uint8_t* sharedMemory,
                                             int forwarderIdx, int nForwarders) {
  constexpr int NForwardWarps = HybridForwardWarps;
  const int laneIdx = getLaneId();
  const int warpIdx = static_cast<int>(threadIdx.x) / WARP_SIZE;
  const auto& hybridCtx = context->hybridCtx_;
  const int sourceRank = hybridCtx.gatewayRank_;
  const int nTopk = workload.numTopk_;
  const int nTokensPerChunk = hybridCtx.nTokensPerChunk_;
  const int nLocalExperts = workload.numExperts_ / context->numRanks_;
  const int nQps = transport.portChannels_->numQpsPerPeer;
  auto* state = hybridCtx.distributeControl_;

  using ElementType = DispatchElementType<DataType>;
  const size_t stride = InterRackDispatchPayload::stride(Hidden, nTopk, sizeof(ElementType));
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(ElementType);
  auto* tileBase = sharedMemory + dispatchSharedControlBytes(context->numRanks_, true);
  auto* staged = tileBase + static_cast<size_t>(warpIdx) * HiddenBytes;
  auto* barriers = reinterpret_cast<mscclpp::BulkBarrier*>(tileBase + static_cast<size_t>(NForwardWarps) * HiddenBytes);
  __shared__ int nIncomingTokens;
  __shared__ bool isLastForwarder;
  if (warpIdx < NForwardWarps && laneIdx == 0) barriers[warpIdx].init();
  if (threadIdx.x == 0) {
    nIncomingTokens = state->recvHeader.nRemoteTokens.read(flag, -1);
    if (forwarderIdx == 0) {
      *hybridCtx.nIncomingTokens_ = nIncomingTokens;
    }
  }
  if (forwarderIdx == 0) {
    for (int peerIdx = threadIdx.x; peerIdx < context->numRanks_; peerIdx += blockDim.x) {
      if (transport.isNvlinkPeer(peerIdx)) {
        auto* counts = static_cast<mscclpp::LL8Packet*>(transport.mappedBuffer(recvBuffer, peerIdx));
        counts[sourceRank].write(state->recvHeader.nTokensPerRank[peerIdx].read(flag, -1), flag);
      }
    }
  }
  __syncthreads();
  const int nTokens = nIncomingTokens;

  uint32_t phase = 0;
  if (warpIdx < NForwardWarps) {
    // Stripe tokens across forwarder blocks first to spread small workloads across SMs.
    for (int tokenIdx = warpIdx * nForwarders + forwarderIdx; tokenIdx < nTokens;
         tokenIdx += nForwarders * NForwardWarps) {
      const int chunkIdx = tokenIdx / nTokensPerChunk;
      if (laneIdx == 0) {
        const int queueIdx = chunkIdx % nQps;
        const uint64_t target = chunkIdx / nQps + 1;
        while (mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(state->qpSignalCount + queueIdx,
                                                                   mscclpp::memoryOrderAcquire) < target) {
        }
      }
      __syncwarp();

      auto* landing = static_cast<uint8_t*>(hybridCtx.distributeRecvBuffer_) + static_cast<size_t>(tokenIdx) * stride;
      const InterRackDispatchPayload payload(landing, Hidden, nTopk, sizeof(ElementType));
      if (laneIdx == 0) {
        barriers[warpIdx].arriveAndExpect(HiddenBytes);
        mscclpp::bulkLoad(staged, landing, HiddenBytes, barriers[warpIdx]);
      }
      const int expertIdx = laneIdx < nTopk ? payload.expertIds_[laneIdx] : -1;
      const int dstRank = expertIdx >= 0 ? expertIdx / nLocalExperts : -1;
      const int slotIdx = laneIdx < nTopk ? payload.dstSlots_[laneIdx] : -1;
      const float weight = laneIdx < nTopk ? payload.topkWeights_[laneIdx] : 0.0f;
      if (laneIdx < nTopk) {
        // Preserve routes for combine before the ACK allows peers to reuse the RDMA landing buffer.
        auto* cached = hybridCtx.aggregateRoutes_ + static_cast<size_t>(tokenIdx) * 3 * nTopk;
        cached[laneIdx] = expertIdx;
        cached[nTopk + laneIdx] = slotIdx;
        reinterpret_cast<float*>(cached + 2 * nTopk)[laneIdx] = weight;
      }
      const bool isLeader = isFirstLaneForRank(dstRank, laneIdx);
      const bool isLocalLeader = dstRank >= 0 && transport.isNvlinkPeer(dstRank) && isLeader;
      if (laneIdx == 0) barriers[warpIdx].wait(phase);
      __syncwarp();
      mscclpp::bulkFence();

      sendNvlinkToken<Hidden, DataType>(output, outputIds, outputWeights, staged, sourceRank, dstRank, slotIdx,
                                        expertIdx, weight, isLocalLeader, workload, transport);
      __syncwarp();
    }
    if (laneIdx == 0) barriers[warpIdx].invalidate();
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    const int nCompletedForwarders = mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(
        &hybridCtx.distributeScratch_->nCompletedForwarders, 1, mscclpp::memoryOrderAcqRel);
    isLastForwarder = nCompletedForwarders == nForwarders - 1;
    if (isLastForwarder) {
      for (int queueIdx = 0; queueIdx < nQps; ++queueIdx) {
        state->qpSignalCount[queueIdx] = 0;
      }
    }
  }
  __syncthreads();
  if (isLastForwarder) {
    // Use the first warp past the peer-signaling lanes so the local reuse-ACK flag is not delayed by system stores.
    const int ackThreadIdx = configAlign(context->numRanks_, WARP_SIZE);
    if (static_cast<int>(threadIdx.x) == ackThreadIdx) {
      mscclpp::atomicStore<int, mscclpp::scopeDevice>(&hybridCtx.distributeScratch_->forwardComplete, 1,
                                                      mscclpp::memoryOrderRelease);
    }
    for (int peerIdx = threadIdx.x; peerIdx < context->numRanks_; peerIdx += blockDim.x) {
      if (transport.isNvlinkPeer(peerIdx)) {
        auto* target = static_cast<HybridDispatchControl*>(transport.mappedBuffer(state, peerIdx));
        mscclpp::atomicStore<uint32_t, mscclpp::scopeSystem>(target->payloadReady + sourceRank, flag,
                                                             mscclpp::memoryOrderRelease);
      }
    }
  }
}

#endif

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
__global__ __launch_bounds__(DispatchNThreads,
                             1) void hybridDispatchKernel(void* output, void* outputScales, int* outputSrcInfo,
                                                          int* outputIds, float* outputWeights, int64_t* outputLayout,
                                                          int* outputCount, const int64_t* routes, const float* weights,
                                                          const void* input, Workload workload, void* recvBuffer,
                                                          const DeviceContext* context) {
#if defined(MSCCLPP_DEVICE_CUDA) && defined(USE_IBVERBS) && MSCCLPP_BULK_AVAILABLE
  static_assert(DataType == DispatchDataType::BF16 || DataType == DispatchDataType::FP8_E4M3_STATIC);
  extern __shared__ __align__(128) uint8_t sharedMemory[];
  WorkspaceView workspace(context->workspace_, context->numRanks_, workload.numExperts_);
  const uint32_t flag = getDispatchFlag(workspace) + 1;
  const TransportView transport(context);
  if (blockIdx.x == 0) {
    distributeControl<Hidden, DataType>(routes, workload, recvBuffer, context, transport, flag,
                                        reinterpret_cast<int*>(sharedMemory));
  } else {
    const int nGridBlocks = static_cast<int>(gridDim.x);
    const int nWorkers = nGridBlocks - 1;
    const int nForwarders = hybridNumForwardBlocks(nGridBlocks);
    const int workerIdx = static_cast<int>(blockIdx.x) - 1;
    if (workerIdx < nForwarders) {
      distributeForward<Hidden, DataType>(output, outputIds, outputWeights, workload, recvBuffer, context, transport,
                                          flag, sharedMemory, workerIdx, nForwarders);
    } else {
      const int producerIdx = workerIdx - nForwarders;
      const int nProducers = nWorkers - nForwarders;
      if (producerIdx < workload.numTokens_) {
        distributeProduce<Hidden, DataType>(output, outputIds, outputWeights, input, routes, weights, workload, context,
                                            transport, sharedMemory, producerIdx, nProducers);
      }
    }
    for (int sourceRank = workerIdx; sourceRank < context->numRanks_; sourceRank += nWorkers) {
      __syncthreads();
      dispatchRankMajorRecvSource(outputIds, outputWeights, outputCount, transport, workspace, sourceRank,
                                  workload.numTopk_, workload.maxTokensPerRank_, workload.invalidTokenExpertId_,
                                  recvBuffer, flag, reinterpret_cast<int*>(sharedMemory),
                                  context->hybridCtx_.distributeControl_->payloadReady);
    }
  }
  updateDispatchFlag(workspace, flag);
#endif
}

struct HybridDispatchKernelSelector {
  template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
  static auto get() {
    return hybridDispatchKernel<Hidden, DataType, ScaleBlockSize>;
  }
};

void hybridDispatch(void* output, int* outputTopkIdx, float* outputTopkWeights, int* outputCount, const void* input,
                    const int64_t* topkIdx, const float* topkWeights, const Workload& workload, void* recvBuffer,
                    const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  EP_HOST_ASSERT(context.hybridCtx_.portChannels_ != nullptr);
  EP_HOST_ASSERT(numBlocks > HybridForwardBlocks + 1);
  dispatchAlgorithm<DispatchLayout::RANK_MAJOR, HybridDispatchKernelSelector, true>(
      output, nullptr, nullptr, outputTopkIdx, outputTopkWeights, nullptr, outputCount, input, topkIdx, topkWeights,
      workload, recvBuffer, context, numBlocks, stream);
}

}  // namespace mscclpp::ep
