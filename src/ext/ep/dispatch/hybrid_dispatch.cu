// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "common/quantization.cuh"
#include "dispatch/common.cuh"

namespace mscclpp::ep {

#if defined(MSCCLPP_DEVICE_CUDA) && defined(USE_IBVERBS) && MSCCLPP_BULK_AVAILABLE

MSCCLPP_DEVICE_INLINE int distributeExpert(const int64_t* routes, int tokenIdx, int laneIdx, const Workload& workload) {
  const int64_t expertIdx = laneIdx < workload.numTopk_ ? routes[tokenIdx * workload.numTopk_ + laneIdx] : -1;
  return expertIdx >= 0 && expertIdx < workload.numExperts_ ? static_cast<int>(expertIdx) : -1;
}

// Producer CTAs: load input tokens via TMA, send NVLink-local destinations directly, and stage
// cross-domain tokens into distributeSendBuffer_ for the control CTA to RDMA-post in chunks.
template <int Hidden, DispatchDataType DataType>
MSCCLPP_DEVICE_INLINE void distributeProduce(void* output, int* outputIds, float* outputWeights, const void* input,
                                             const int64_t* routes, const float* weights, const Workload& workload,
                                             const DeviceContext* context, const TransportView& transport,
                                             int* shared) {
  const int firstProducerIdx = HybridForwardBlocks + 1;
  const int producerIdx = static_cast<int>(blockIdx.x) - firstProducerIdx;
  const int nProducers = static_cast<int>(gridDim.x) - firstProducerIdx;
  const int warpIdx = threadIdx.x / WARP_SIZE;
  const int laneIdx = getLaneId();
  if (warpIdx >= DispatchMaxNWarpGroups) return;

  using ElementType = DispatchElementType<DataType>;
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(ElementType);
  const auto& hybridCtx = context->hybridCtx_;
  const int nTopk = workload.numTopk_;
  const int nTokensPerChunk = hybridCtx.nTokensPerChunk_;
  const int nLocalExperts = workload.numExperts_ / context->numRanks_;
  const size_t stride = InterRackDispatchPayload::stride(Hidden, nTopk, sizeof(ElementType));
  const size_t controlBytes = dispatchSharedControlBytes(context->numRanks_, true);
  const size_t payloadStride = dispatchPayloadStride<DataType>(Hidden, nTopk, 0);
  auto* staged = reinterpret_cast<uint8_t*>(shared) + controlBytes + warpIdx * payloadStride;
  auto* barriers = reinterpret_cast<mscclpp::BulkBarrier*>(reinterpret_cast<uint8_t*>(shared) + controlBytes +
                                                           DispatchMaxNWarpGroups * payloadStride);
  auto& barrier = barriers[warpIdx];
  const bool quantizeInput =
      DataType == DispatchDataType::FP8_E4M3_STATIC && workload.inputMode_ == QuantConfig::InputMode::QUANTIZE;
  if (!quantizeInput && laneIdx == 0) barrier.init();
  __syncwarp();

  uint32_t phase = 0;
  WorkspaceView workspace(context->workspace_, context->numRanks_, workload.numExperts_);
  for (int tokenIdx = producerIdx * DispatchMaxNWarpGroups + warpIdx; tokenIdx < workload.numTokens_;
       tokenIdx += nProducers * DispatchMaxNWarpGroups) {
    if constexpr (DataType == DispatchDataType::FP8_E4M3_STATIC) {
      if (quantizeInput) {
        constexpr int NHiddenVectors = Hidden / mscclpp::bf16x8::Size;
        const auto* inputData = static_cast<const uint4*>(input) + static_cast<size_t>(tokenIdx) * NHiddenVectors;
        auto* stagedData = reinterpret_cast<mscclpp::f8_e4m3x8*>(staged);
        for (int vectorIdx = laneIdx; vectorIdx < NHiddenVectors; vectorIdx += WARP_SIZE) {
          stagedData[vectorIdx] =
              quantizeBf16x8ToStaticFp8(mscclpp::bf16x8(inputData[vectorIdx]), workload.dispatchScale_);
        }
      }
    }
    if (!quantizeInput && laneIdx == 0) {
      barrier.arriveAndExpect(HiddenBytes);
      mscclpp::bulkLoad(staged, static_cast<const uint8_t*>(input) + static_cast<size_t>(tokenIdx) * HiddenBytes,
                        HiddenBytes, barrier);
    }
    const RankMajorRoute route = prepareRankMajorRoute(workspace, routes, tokenIdx, nTopk, nLocalExperts,
                                                       workload.maxTokensPerRank_, laneIdx, workload.numExperts_);
    const int expertIdx = route.expertIdx;
    const float weight = expertIdx >= 0 ? (weights == nullptr ? 1.0f : weights[tokenIdx * nTopk + laneIdx]) : 0.0f;
    const bool remote = route.dstRank >= 0 && !transport.isNvlinkPeer(route.dstRank);
    const bool hasRemote = __any_sync(0xffffffff, remote);
    int remoteTokenIdx = -1;
    if (laneIdx == 0 && hasRemote) remoteTokenIdx = atomicAdd(&hybridCtx.distributeScratch_->nRemoteTokens, 1);
    remoteTokenIdx = __shfl_sync(0xffffffff, remoteTokenIdx, 0);
    if (laneIdx == 0) hybridCtx.aggregateRemoteTokenIndices_[tokenIdx] = remoteTokenIdx;

    if (!quantizeInput && laneIdx == 0) barrier.wait(phase);
    __syncwarp();
    mscclpp::bulkFence();

    if (hasRemote) {
      EP_DEVICE_ASSERT(remoteTokenIdx < hybridCtx.distributeCapacity_);
      auto* send =
          static_cast<uint8_t*>(hybridCtx.distributeSendBuffer_) + static_cast<size_t>(remoteTokenIdx) * stride;
      const InterRackDispatchPayload payload(send, Hidden, nTopk, sizeof(ElementType));
      for (int vectorIdx = laneIdx; vectorIdx < HiddenBytes / sizeof(int4); vectorIdx += WARP_SIZE) {
        reinterpret_cast<int4*>(send)[vectorIdx] = reinterpret_cast<const int4*>(staged)[vectorIdx];
      }
      if (laneIdx < nTopk) {
        payload.expertIds_[laneIdx] = expertIdx;
        payload.topkWeights_[laneIdx] = weight;
        payload.dstSlots_[laneIdx] = route.destinationSlot;
      }
      __syncwarp();
      if (laneIdx == 0) {
        (void)mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(
            hybridCtx.distributeScratch_->chunkReadySignals + remoteTokenIdx / nTokensPerChunk, 1,
            mscclpp::memoryOrderRelease);
      }
    }

    const int localDstRank = (route.dstRank >= 0 && route.isLeader && !remote) ? route.dstRank : -1;
    if (localDstRank >= 0) {
      issueRankMajorTokenStore<Hidden, DataType>(output, transport, route.destinationSlot, workload.maxTokensPerRank_,
                                                 staged, localDstRank);
    }
    const unsigned int localLeaders = __ballot_sync(0xffffffff, localDstRank >= 0);
    const int outputExpertIdx = workload.emitLocalExpertIds_ && expertIdx >= 0 ? expertIdx % nLocalExperts : expertIdx;
    writeRankMajorMetadataNvlink(transport, outputIds, outputWeights, transport.rank_, route.dstRank,
                                 route.destinationSlot, outputExpertIdx, weight, localLeaders, nTopk,
                                 workload.maxTokensPerRank_, workload.invalidTokenExpertId_);
    completeRankMajorTokenStore(workspace, localDstRank);
    __syncwarp();
  }
  if (!quantizeInput && laneIdx == 0) barrier.invalidate();
}

// Control CTA (block 0): count routes, publish NVLink signals, post RDMA header + chunks to the
// paired gateway rank, and exchange the round-completion ACK.
template <int Hidden, DispatchDataType DataType>
MSCCLPP_DEVICE_INLINE void distributeControl(const int64_t* routes, const Workload& workload, void* recvBuffer,
                                             const DeviceContext* context, const TransportView& transport,
                                             uint32_t flag, int* shared) {
  const int rank = context->rank_, nRanks = context->numRanks_;
  const int laneIdx = getLaneId(), warpIdx = threadIdx.x / WARP_SIZE;
  const int nLocalExperts = workload.numExperts_ / nRanks;
  const auto& hybridCtx = context->hybridCtx_;
  auto* state = hybridCtx.distributeControl_;
  auto* network = transport.portChannels_;
  const int gatewayRank = hybridCtx.gatewayRank_;
  const int nQps = network->numQpsPerPeer;
  const int nTokensPerChunk = hybridCtx.nTokensPerChunk_;
  WorkspaceView workspace(context->workspace_, nRanks, workload.numExperts_);

  for (int idx = threadIdx.x; idx <= nRanks; idx += blockDim.x) shared[idx] = 0;
  for (int queueIdx = threadIdx.x; queueIdx < nQps; queueIdx += blockDim.x) {
    hybridCtx.aggregateControl_->qpSignalCount[queueIdx] = 0;
  }
  const int nCombineCounters = configCellDiv(hybridCtx.distributeCapacity_, nTokensPerChunk);
  for (int idx = threadIdx.x; idx < nCombineCounters; idx += blockDim.x) {
    hybridCtx.aggregateChunkReady_[idx] = 0;
  }
  __syncthreads();
  for (int tokenIdx = warpIdx; tokenIdx < workload.numTokens_; tokenIdx += DispatchNWarps) {
    const int expertIdx = distributeExpert(routes, tokenIdx, laneIdx, workload);
    const int dstRank = expertIdx >= 0 ? expertIdx / nLocalExperts : -1;
    const bool leader = isFirstLaneForRank(dstRank, laneIdx);
    if (dstRank >= 0 && leader) atomicAdd_block(shared + dstRank, 1);
    const bool remote = __any_sync(0xffffffff, dstRank >= 0 && !transport.isNvlinkPeer(dstRank));
    if (laneIdx == 0 && remote) atomicAdd_block(shared + nRanks, 1);
  }
  __syncthreads();

  for (int peerIdx = threadIdx.x; peerIdx < nRanks; peerIdx += blockDim.x) {
    state->sendHeader.nTokensPerRank[peerIdx].write(shared[peerIdx], flag);
  }
  const int nRemoteTokens = shared[nRanks];
  if (threadIdx.x == 0) state->sendHeader.nRemoteTokens.write(nRemoteTokens, flag);
  __syncthreads();

  // Warps after the QP posters publish NVLink counts and signals.
  const int firstPublisherIdx = nQps * WARP_SIZE;
  if (threadIdx.x >= firstPublisherIdx && threadIdx.x < firstPublisherIdx + nRanks) {
    const int dstRank = threadIdx.x - firstPublisherIdx;
    const int nTokens = shared[dstRank];
    if (transport.isNvlinkPeer(dstRank)) {
      auto* counts = static_cast<mscclpp::LL8Packet*>(transport.mappedBuffer(recvBuffer, dstRank));
      counts[rank].write(nTokens, flag);
      if (nTokens > 0) {
        while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(workspace.dispatchRankPayloadCompletions_ + dstRank,
                                                              mscclpp::memoryOrderAcquire) != nTokens) {
        }
        if (dstRank == rank)
          workspace.dispatchLocalPayloadReady_->release();
        else
          transport.baseMemoryChannels_[dstRank].signal();
      }
    }
  }

  constexpr auto kExclusive = mscclpp::PortChannelDeviceHandle::Concurrency::Exclusive;
  const size_t stride =
      InterRackDispatchPayload::stride(Hidden, workload.numTopk_, sizeof(DispatchElementType<DataType>));
  if (warpIdx < nQps && laneIdx == 0) {
    const int queueIdx = warpIdx;
    auto handle = network->channel(gatewayRank, queueIdx);
    const uint64_t dstOffset = transport.symmetricOffset(hybridCtx.distributeRecvBuffer_);
    const uint64_t srcOffset = transport.symmetricOffset(hybridCtx.distributeSendBuffer_);
    const uint64_t flagOffset = transport.symmetricOffset(state->qpSignalCount + queueIdx);
    if (queueIdx == 0) {
      const uint64_t recvHeader = transport.symmetricOffset(&state->recvHeader);
      const uint64_t sendHeader = transport.symmetricOffset(&state->sendHeader);
      const size_t headerBytes = InterRackDispatchHeader::wireBytes(nRanks);
      handle.put<kExclusive>(recvHeader, sendHeader, headerBytes, queueIdx);
    }
    for (int chunkIdx = queueIdx, tokenBegin = queueIdx * nTokensPerChunk; tokenBegin < nRemoteTokens;
         chunkIdx += nQps, tokenBegin += nQps * nTokensPerChunk) {
      const int nChunkTokens = min(nTokensPerChunk, nRemoteTokens - tokenBegin);
      handle.device_.putWithAccumulateAfter<kExclusive>(
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
      handle.device_.accumulateAfter<kExclusive>(
          handle.dstMemory_, readyOffset, 1,
          [&] {
            while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(&hybridCtx.distributeScratch_->forwardComplete,
                                                                  mscclpp::memoryOrderAcquire) != 1) {
            }
          },
          queueIdx);
    }
  }
}

// Forward CTAs: wait for incoming RDMA chunks from the paired gateway rank and forward them over
// NVLink to their destination ranks in the local domain.
template <int Hidden, DispatchDataType DataType>
MSCCLPP_DEVICE_INLINE void distributeForward(void* output, int* outputIds, float* outputWeights,
                                             const Workload& workload, void* recvBuffer, const DeviceContext* context,
                                             const TransportView& transport, uint32_t flag, int* shared) {
  const int forwarderIdx = static_cast<int>(blockIdx.x) - 1;
  const int nForwarders = static_cast<int>(gridDim.x) - 1;
  constexpr int NForwardWarps = DispatchMaxNWarpGroups;
  const int laneIdx = getLaneId(), warpIdx = threadIdx.x / WARP_SIZE;
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
  const size_t controlBytes = dispatchSharedControlBytes(context->numRanks_, true);
  const size_t payloadStride = dispatchPayloadStride<DataType>(Hidden, nTopk, 0);
  auto* staged = reinterpret_cast<uint8_t*>(shared) + controlBytes + warpIdx * payloadStride;
  auto* barriers = reinterpret_cast<mscclpp::BulkBarrier*>(reinterpret_cast<uint8_t*>(shared) + controlBytes +
                                                           NForwardWarps * payloadStride);
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
    for (int tokenIdx = forwarderIdx * NForwardWarps + warpIdx; tokenIdx < nTokens;
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
        auto* cached = hybridCtx.aggregateRoutes_ + static_cast<size_t>(tokenIdx) * 3 * nTopk;
        cached[laneIdx] = expertIdx;
        cached[nTopk + laneIdx] = slotIdx;
        reinterpret_cast<float*>(cached + 2 * nTopk)[laneIdx] = weight;
      }
      const bool leader = isFirstLaneForRank(dstRank, laneIdx);
      const bool localLeader = dstRank >= 0 && transport.isNvlinkPeer(dstRank) && leader;
      const unsigned int leaders = __ballot_sync(0xffffffff, localLeader);
      if (laneIdx == 0) barriers[warpIdx].wait(phase);
      __syncwarp();
      mscclpp::bulkFence();

      if (localLeader) {
        EP_DEVICE_ASSERT(slotIdx >= 0 && slotIdx < workload.maxTokensPerRank_);
        auto* destinationBuffer = static_cast<uint8_t*>(transport.mappedBuffer(output, dstRank));
        const size_t dstTokenIdx = static_cast<size_t>(sourceRank) * workload.maxTokensPerRank_ + slotIdx;
        mscclpp::bulkStore(destinationBuffer + dstTokenIdx * HiddenBytes, staged, HiddenBytes);
        mscclpp::bulkStoreCommit();
      }
      const int outputExpertIdx =
          workload.emitLocalExpertIds_ && expertIdx >= 0 ? expertIdx % nLocalExperts : expertIdx;
      writeRankMajorMetadataNvlink(transport, outputIds, outputWeights, sourceRank, dstRank, slotIdx, outputExpertIdx,
                                   weight, leaders, nTopk, workload.maxTokensPerRank_, workload.invalidTokenExpertId_);
      if (localLeader) mscclpp::bulkStoreWait();
      __syncwarp();
    }
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
    if (threadIdx.x == configAlign(context->numRanks_, WARP_SIZE)) {
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
  auto* shared = reinterpret_cast<int*>(sharedMemory);
  WorkspaceView workspace(context->workspace_, context->numRanks_, workload.numExperts_);
  const uint32_t flag = getDispatchFlag(workspace) + 1;
  const TransportView transport(context);
  if (blockIdx.x == 0) {
    distributeControl<Hidden, DataType>(routes, workload, recvBuffer, context, transport, flag, shared);
  } else {
    const int producerIdx = static_cast<int>(blockIdx.x) - (HybridForwardBlocks + 1);
    if (producerIdx >= 0 && producerIdx * DispatchMaxNWarpGroups < workload.numTokens_) {
      distributeProduce<Hidden, DataType>(output, outputIds, outputWeights, input, routes, weights, workload, context,
                                          transport, shared);
      __syncthreads();
    }
    distributeForward<Hidden, DataType>(output, outputIds, outputWeights, workload, recvBuffer, context, transport,
                                        flag, shared);
  }
  for (int sourceRank = static_cast<int>(blockIdx.x); sourceRank < context->numRanks_;
       sourceRank += static_cast<int>(gridDim.x)) {
    __syncthreads();
    dispatchRankMajorRecvSource(outputIds, outputWeights, outputCount, transport, workspace, sourceRank,
                                workload.numTopk_, workload.maxTokensPerRank_, workload.invalidTokenExpertId_,
                                recvBuffer, flag, shared, context->hybridCtx_.distributeControl_->payloadReady);
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
