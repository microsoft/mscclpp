// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "common/quantization.cuh"
#include "dispatch/common.cuh"

namespace mscclpp::ep {

#if defined(MSCCLPP_USE_GPUNETIO) && MSCCLPP_BULK_AVAILABLE

// Scratch layout: [0] compact remote row counter, [1] completed forward CTAs, [2] forward-complete flag,
// [3..] per-chunk staged row counts.
constexpr int DistributeScratchCounters = 3;

MSCCLPP_DEVICE_INLINE int distributeExpert(const int64_t* routes, int token, int lane, const Workload& workload) {
  const int64_t expert = lane < workload.numTopk_ ? routes[token * workload.numTopk_ + lane] : -1;
  return expert >= 0 && expert < workload.numExperts_ ? static_cast<int>(expert) : -1;
}

// Producer CTAs: load input tokens via TMA, send NVLink-local destinations directly, and stage
// cross-domain tokens into distributeSendBuffer_ for the control CTA to RDMA-post in chunks.
template <int Hidden, DispatchDataType DataType>
MSCCLPP_DEVICE_INLINE void distributeProduce(void* output, int* outputIds, float* outputWeights, const void* input,
                                             const int64_t* routes, const float* weights, const Workload& workload,
                                             const DeviceContext* context, const TransportView& transport,
                                             int* shared) {
  const int firstProducer = HybridForwardBlocks + 1;
  const int producer = static_cast<int>(blockIdx.x) - firstProducer;
  const int producers = static_cast<int>(gridDim.x) - firstProducer;
  const int warp = threadIdx.x / WARP_SIZE;
  const int lane = getLaneId();
  if (producer < 0 || warp >= DispatchMaxNWarpGroups) return;

  using ElementType = DispatchElementType<DataType>;
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(ElementType);
  const int topk = workload.numTopk_;
  const int nTokensPerChunk = context->nTokensPerChunk_;
  const int localExperts = workload.numExperts_ / context->numRanks_;
  const size_t stride = HybridDispatchRow::stride(Hidden, topk, sizeof(ElementType));
  const size_t controlBytes = dispatchSharedControlBytes(context->numRanks_, true);
  const size_t payloadStride = dispatchPayloadStride<DataType>(Hidden, topk, 0);
  auto* staged = reinterpret_cast<uint8_t*>(shared) + controlBytes + warp * payloadStride;
  auto* barriers = reinterpret_cast<mscclpp::BulkBarrier*>(reinterpret_cast<uint8_t*>(shared) + controlBytes +
                                                           DispatchMaxNWarpGroups * payloadStride);
  auto& barrier = barriers[warp];
  const bool quantizeInput =
      DataType == DispatchDataType::FP8_E4M3_STATIC && workload.inputMode_ == QuantConfig::InputMode::QUANTIZE;
  if (!quantizeInput && lane == 0) barrier.init();
  __syncwarp();

  uint32_t phase = 0;
  WorkspaceView workspace(context->workspace_, context->numRanks_, workload.numExperts_);
  for (int token = producer * DispatchMaxNWarpGroups + warp; token < workload.numTokens_;
       token += producers * DispatchMaxNWarpGroups) {
    if constexpr (DataType == DispatchDataType::FP8_E4M3_STATIC) {
      if (quantizeInput) {
        constexpr int HiddenVectors = Hidden / mscclpp::bf16x8::Size;
        const auto* inputData = static_cast<const uint4*>(input) + static_cast<size_t>(token) * HiddenVectors;
        auto* stagedData = reinterpret_cast<mscclpp::f8_e4m3x8*>(staged);
        for (int vector = lane; vector < HiddenVectors; vector += WARP_SIZE) {
          stagedData[vector] = quantizeBf16x8ToStaticFp8(mscclpp::bf16x8(inputData[vector]), workload.dispatchScale_);
        }
      }
    }
    if (!quantizeInput && lane == 0) {
      barrier.arriveAndExpect(HiddenBytes);
      mscclpp::bulkLoad(staged, static_cast<const uint8_t*>(input) + static_cast<size_t>(token) * HiddenBytes,
                        HiddenBytes, barrier);
    }
    const RankMajorRoute route = prepareRankMajorRoute(workspace, routes, token, topk, localExperts,
                                                       workload.maxTokensPerRank_, lane, workload.numExperts_);
    const int expert = distributeExpert(routes, token, lane, workload);
    const float weight = expert >= 0 ? (weights == nullptr ? 1.0f : weights[token * topk + lane]) : 0.0f;
    const bool remote = route.dstRank >= 0 && !transport.isNvlinkPeer(route.dstRank);
    const bool hasRemote = __any_sync(0xffffffff, remote);
    int row = -1;
    if (lane == 0 && hasRemote) row = atomicAdd(context->distributeScratch_, 1);
    row = __shfl_sync(0xffffffff, row, 0);
    if (lane == 0) context->aggregateTokenRows_[token] = row;

    if (!quantizeInput && lane == 0) barrier.wait(phase);
    __syncwarp();
    mscclpp::bulkFence();

    if (hasRemote) {
      EP_DEVICE_ASSERT(row < context->distributeCapacity_);
      auto* send = static_cast<uint8_t*>(context->distributeSendBuffer_) + static_cast<size_t>(row) * stride;
      const HybridDispatchRow wire(send, Hidden, topk, sizeof(ElementType));
      for (int vector = lane; vector < HiddenBytes / sizeof(int4); vector += WARP_SIZE) {
        reinterpret_cast<int4*>(send)[vector] = reinterpret_cast<const int4*>(staged)[vector];
      }
      if (lane < topk) {
        wire.ids_[lane] = expert;
        wire.weights_[lane] = weight;
        wire.slots_[lane] = route.destinationSlot;
      }
      __syncwarp();
      if (lane == 0) {
        (void)mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(
            context->distributeScratch_ + DistributeScratchCounters + row / nTokensPerChunk, 1,
            mscclpp::memoryOrderRelease);
      }
    }

    const int localDst = (route.dstRank >= 0 && route.isLeader && !remote) ? route.dstRank : -1;
    if (localDst >= 0) {
      issueRankMajorTokenStore<Hidden, DataType>(output, transport, route.destinationSlot, workload.maxTokensPerRank_,
                                                 staged, localDst);
    }
    const unsigned int localLeaders = __ballot_sync(0xffffffff, localDst >= 0);
    const int outputExpert = workload.emitLocalExpertIds_ && expert >= 0 ? expert % localExperts : expert;
    writeRankMajorMetadataNvlink(transport, outputIds, outputWeights, transport.rank_, route.dstRank,
                                 route.destinationSlot, outputExpert, weight, localLeaders, topk,
                                 workload.maxTokensPerRank_, workload.invalidTokenExpertId_);
    completeRankMajorTokenStore(workspace, localDst);
    __syncwarp();
  }
}

// Control CTA (block 0): count routes, publish NVLink signals, post RDMA header + chunks to the
// paired gateway rank, and exchange the round-completion ACK.
template <int Hidden, DispatchDataType DataType>
MSCCLPP_DEVICE_INLINE void distributeControl(const int64_t* routes, const Workload& workload, void* recvBuffer,
                                             const DeviceContext* context, const TransportView& transport,
                                             uint32_t flag, int* shared) {
  const int rank = context->rank_, ranks = context->numRanks_;
  const int lane = getLaneId(), warp = threadIdx.x / WARP_SIZE;
  const int localExperts = workload.numExperts_ / ranks;
  auto* state = context->distributeControl_;
  auto* network = transport.portChannels_;
  const int gateway = context->gatewayRank_;
  const int qps = network->numQpsPerPeer;
  const int nTokensPerChunk = context->nTokensPerChunk_;
  WorkspaceView workspace(context->workspace_, ranks, workload.numExperts_);

  for (int index = threadIdx.x; index <= ranks; index += blockDim.x) shared[index] = 0;
  __syncthreads();
  for (int token = warp; token < workload.numTokens_; token += DispatchNWarps) {
    const int expert = distributeExpert(routes, token, lane, workload);
    const int destination = expert >= 0 ? expert / localExperts : -1;
    const bool leader = isFirstLaneForRank(destination, lane);
    if (destination >= 0 && leader) atomicAdd_block(shared + destination, 1);
    const bool remote = __any_sync(0xffffffff, destination >= 0 && !transport.isNvlinkPeer(destination));
    if (lane == 0 && remote) atomicAdd_block(shared + ranks, 1);
  }
  __syncthreads();

  for (int peer = threadIdx.x; peer < ranks; peer += blockDim.x) {
    state->sendHeader.counts[peer].write(shared[peer], flag);
  }
  const int rows = shared[ranks];
  if (threadIdx.x == 0) state->sendHeader.rows.write(rows, flag);
  __syncthreads();

  // Warps after the QP posters publish NVLink counts and signals.
  const int firstPublisher = qps * WARP_SIZE;
  if (threadIdx.x >= firstPublisher && threadIdx.x < firstPublisher + ranks) {
    const int destination = threadIdx.x - firstPublisher;
    const int count = shared[destination];
    if (transport.isNvlinkPeer(destination)) {
      auto* counts = static_cast<mscclpp::LL8Packet*>(transport.mappedBuffer(recvBuffer, destination));
      counts[rank].write(count, flag);
      if (count > 0) {
        while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(workspace.dispatchRankPayloadCompletions_ + destination,
                                                              mscclpp::memoryOrderAcquire) != count) {
        }
        if (destination == rank)
          workspace.dispatchLocalPayloadReady_->release();
        else
          transport.baseMemoryChannels_[destination].signal();
      }
    }
  }

  __shared__ mscclpp::GpuNetIoWqeSlot wqeSlots[MaxQpsPerPeer];
  constexpr auto kExclusive = mscclpp::PortChannelSharingMode::Exclusive;
  const size_t stride = HybridDispatchRow::stride(Hidden, workload.numTopk_, sizeof(DispatchElementType<DataType>));
  if (warp < qps && lane == 0) {
    const int queue = warp;
    auto handle = network->channel(gateway, queue);
    handle.setWqeSlot<kExclusive>(&wqeSlots[queue]);
    const uint64_t dstOffset = transport.symmetricOffset(context->distributeRecvBuffer_);
    const uint64_t srcOffset = transport.symmetricOffset(context->distributeSendBuffer_);
    const uint64_t flagOffset = transport.symmetricOffset(state->chunks + queue);
    auto& sequence = network->signalState(gateway, queue);
    if (queue == 0) {
      const uint64_t recvHeader = transport.symmetricOffset(&state->recvHeader);
      const uint64_t sendHeader = transport.symmetricOffset(&state->sendHeader);
      const size_t headerBytes = HybridDispatchHeader::wireBytes(ranks);
      handle.put<kExclusive>(recvHeader, sendHeader, headerBytes);
    }
    for (int chunk = queue, begin = queue * nTokensPerChunk; begin < rows;
         chunk += qps, begin += qps * nTokensPerChunk) {
      const int count = min(nTokensPerChunk, rows - begin);
      handle.device_.putWithSignalAtAfter<kExclusive>(
          handle.dstMemory_, dstOffset + begin * stride, handle.srcMemory_, srcOffset + begin * stride, count * stride,
          handle.dstMemory_, flagOffset, network->nextSignalToken(sequence, flagOffset, 1), [&] {
            while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(
                       context->distributeScratch_ + DistributeScratchCounters + chunk, mscclpp::memoryOrderAcquire) !=
                   count) {
            }
          });
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    const auto offset = transport.symmetricOffset(&state->ack);
    auto channel = network->channel(gateway, 0);
    channel.setWqeSlot<kExclusive>(&wqeSlots[0]);
    auto& sequence = network->signalState(gateway, 0);
    channel.device_.signalAtAfter<kExclusive>(
        channel.dstMemory_, offset, network->nextSignalToken(sequence, offset, 1), [&] {
          while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(context->distributeScratch_ + 2,
                                                                mscclpp::memoryOrderAcquire) != 1) {
          }
        });
    const uint64_t ack = state->ackBase + 1;
    while (mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(&state->ack, mscclpp::memoryOrderAcquire) < ack) {
    }
    state->ackBase = ack;
  }
}

// Forward CTAs: wait for incoming RDMA chunks from the paired gateway rank and forward them over
// NVLink to their destination ranks in the local domain.
template <int Hidden, DispatchDataType DataType>
MSCCLPP_DEVICE_INLINE void distributeForward(void* output, int* outputIds, float* outputWeights,
                                             const Workload& workload, void* recvBuffer, const DeviceContext* context,
                                             const TransportView& transport, uint32_t flag, int* shared) {
  const int forward = static_cast<int>(blockIdx.x) - 1;
  const int forwarders = static_cast<int>(gridDim.x) - 1;
  constexpr int ForwardWarps = DispatchMaxNWarpGroups;
  const int lane = getLaneId(), warp = threadIdx.x / WARP_SIZE;
  const int source = context->gatewayRank_;
  const int topk = workload.numTopk_;
  const int nTokensPerChunk = context->nTokensPerChunk_;
  const int localExperts = workload.numExperts_ / context->numRanks_;
  const int qps = transport.portChannels_->numQpsPerPeer;
  auto* state = context->distributeControl_;

  if (threadIdx.x == 0) {
    shared[0] = state->recvHeader.rows.read(flag, -1);
    if (forward == 0) {
      *context->aggregateIncomingRows_ = shared[0];
    }
  }
  __syncthreads();
  const int rows = shared[0];
  if (forward == 0) {
    for (int peer = threadIdx.x; peer < context->numRanks_; peer += blockDim.x) {
      if (transport.isNvlinkPeer(peer)) {
        auto* counts = static_cast<mscclpp::LL8Packet*>(transport.mappedBuffer(recvBuffer, peer));
        counts[source].write(state->recvHeader.counts[peer].read(flag, -1), flag);
      }
    }
  }

  using ElementType = DispatchElementType<DataType>;
  const int chunks = configCellDiv(rows, nTokensPerChunk);
  const size_t stride = HybridDispatchRow::stride(Hidden, topk, sizeof(ElementType));
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(ElementType);
  const size_t controlBytes = dispatchSharedControlBytes(context->numRanks_, true);
  const size_t payloadStride = dispatchPayloadStride<DataType>(Hidden, topk, 0);
  auto* staged = reinterpret_cast<uint8_t*>(shared) + controlBytes + warp * payloadStride;
  auto* barriers = reinterpret_cast<mscclpp::BulkBarrier*>(reinterpret_cast<uint8_t*>(shared) + controlBytes +
                                                           ForwardWarps * payloadStride);
  if (warp < ForwardWarps && lane == 0) barriers[warp].init();
  __syncthreads();

  uint32_t phase = 0;
  if (warp < ForwardWarps) {
    for (int row = forward * ForwardWarps + warp; row < rows; row += forwarders * ForwardWarps) {
      const int chunk = row / nTokensPerChunk;
      if (lane == 0) {
        const int queue = chunk % qps;
        const uint64_t target = state->chunkBase[queue] + chunk / qps + 1;
        while (mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(state->chunks + queue, mscclpp::memoryOrderAcquire) <
               target) {
        }
      }
      __syncwarp();

      auto* landing = static_cast<uint8_t*>(context->distributeRecvBuffer_) + static_cast<size_t>(row) * stride;
      const HybridDispatchRow wire(landing, Hidden, topk, sizeof(ElementType));
      if (lane == 0) {
        barriers[warp].arriveAndExpect(HiddenBytes);
        mscclpp::bulkLoad(staged, landing, HiddenBytes, barriers[warp]);
      }
      const int expert = lane < topk ? wire.ids_[lane] : -1;
      const int destination = expert >= 0 ? expert / localExperts : -1;
      const int slot = lane < topk ? wire.slots_[lane] : -1;
      const float weight = lane < topk ? wire.weights_[lane] : 0.0f;
      if (lane < topk) {
        auto* cached = context->aggregateRoutes_ + static_cast<size_t>(row) * 3 * topk;
        cached[lane] = expert;
        cached[topk + lane] = slot;
        reinterpret_cast<float*>(cached + 2 * topk)[lane] = weight;
      }
      const bool leader = isFirstLaneForRank(destination, lane);
      const bool localLeader = destination >= 0 && transport.isNvlinkPeer(destination) && leader;
      const unsigned int leaders = __ballot_sync(0xffffffff, localLeader);
      if (lane == 0) barriers[warp].wait(phase);
      __syncwarp();
      mscclpp::bulkFence();

      if (localLeader) {
        EP_DEVICE_ASSERT(slot >= 0 && slot < workload.maxTokensPerRank_);
        auto* destinationBuffer = static_cast<uint8_t*>(transport.mappedBuffer(output, destination));
        const size_t index = static_cast<size_t>(source) * workload.maxTokensPerRank_ + slot;
        mscclpp::bulkStore(destinationBuffer + index * HiddenBytes, staged, HiddenBytes);
        mscclpp::bulkStoreCommit();
      }
      const int outputExpert = workload.emitLocalExpertIds_ && expert >= 0 ? expert % localExperts : expert;
      writeRankMajorMetadataNvlink(transport, outputIds, outputWeights, source, destination, slot, outputExpert, weight,
                                   leaders, topk, workload.maxTokensPerRank_, workload.invalidTokenExpertId_);
      if (localLeader) mscclpp::bulkStoreWait();
      __syncwarp();
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    const int previous = mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(context->distributeScratch_ + 1, 1,
                                                                            mscclpp::memoryOrderAcqRel);
    shared[1] = previous == forwarders - 1;
  }
  __syncthreads();
  if (shared[1]) {
    for (int queue = threadIdx.x; queue < qps; queue += blockDim.x) {
      state->chunkBase[queue] += (chunks + qps - 1 - queue) / qps;
    }
    __syncthreads();
    if (threadIdx.x == configAlign(context->numRanks_, WARP_SIZE)) {
      mscclpp::atomicStore<int, mscclpp::scopeDevice>(context->distributeScratch_ + 2, 1, mscclpp::memoryOrderRelease);
    }
    for (int peer = threadIdx.x; peer < context->numRanks_; peer += blockDim.x) {
      if (transport.isNvlinkPeer(peer)) {
        auto* target = static_cast<HybridDispatchControl*>(transport.mappedBuffer(state, peer));
        mscclpp::atomicStore<uint32_t, mscclpp::scopeSystem>(target->forwardReady + source, flag,
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
#if defined(MSCCLPP_USE_GPUNETIO) && MSCCLPP_BULK_AVAILABLE
  static_assert(DataType == DispatchDataType::BF16 || DataType == DispatchDataType::FP8_E4M3_STATIC);
  extern __shared__ __align__(128) uint8_t sharedMemory[];
  auto* shared = reinterpret_cast<int*>(sharedMemory);
  WorkspaceView workspace(context->workspace_, context->numRanks_, workload.numExperts_);
  const uint32_t flag = getDispatchFlag(workspace) + 1;
  const TransportView transport(context);
  if (blockIdx.x == 0) {
    distributeControl<Hidden, DataType>(routes, workload, recvBuffer, context, transport, flag, shared);
  } else {
    if (blockIdx.x > HybridForwardBlocks) {
      distributeProduce<Hidden, DataType>(output, outputIds, outputWeights, input, routes, weights, workload, context,
                                          transport, shared);
      __syncthreads();
    }
    distributeForward<Hidden, DataType>(output, outputIds, outputWeights, workload, recvBuffer, context, transport,
                                        flag, shared);
  }
  for (int source = static_cast<int>(blockIdx.x); source < context->numRanks_; source += static_cast<int>(gridDim.x)) {
    __syncthreads();
    dispatchRankMajorRecvSource(outputIds, outputWeights, outputCount, transport, workspace, source, workload.numTopk_,
                                workload.maxTokensPerRank_, workload.invalidTokenExpertId_, recvBuffer, flag, shared,
                                context->distributeControl_->forwardReady);
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
  EP_HOST_ASSERT(context.portChannels_ != nullptr);
  EP_HOST_ASSERT(numBlocks > HybridForwardBlocks + 1);
  dispatchAlgorithm<DispatchLayout::RANK_MAJOR, HybridDispatchKernelSelector, true>(
      output, nullptr, nullptr, outputTopkIdx, outputTopkWeights, nullptr, outputCount, input, topkIdx, topkWeights,
      workload, recvBuffer, context, numBlocks, stream);
}

}  // namespace mscclpp::ep
