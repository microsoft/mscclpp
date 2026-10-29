// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_DISPATCH_HYBRID_DISPATCH_CUH_
#define MSCCLPP_EP_DISPATCH_HYBRID_DISPATCH_CUH_

#include "dispatch/common.cuh"

namespace mscclpp::ep {

#if defined(MSCCLPP_USE_GPUNETIO) && MSCCLPP_BULK_AVAILABLE

// Scratch is stream-reset; network flags and their consumption baselines persist.
constexpr int DistributeScratchCounters = 3;
static_assert(GpuNetIoDistributeForwardWarps <= DispatchNWarps);

MSCCLPP_DEVICE_INLINE int distributeExpert(const int64_t* routes, int token, int lane, const Workload& workload) {
  const int64_t expert = lane < workload.numTopk_ ? routes[token * workload.numTopk_ + lane] : -1;
  return expert >= 0 && expert < workload.numExperts_ ? static_cast<int>(expert) : -1;
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void distributeProduce(void* output, int* outputIds, float* outputWeights, const void* input,
                                             const int64_t* routes, const float* weights, const Workload& workload,
                                             const DeviceContext* context, const TransportView& transport,
                                             int* shared) {
  const int firstProducer = context->numRanks_ + GpuNetIoDistributeForwardBlocks + 1;
  const int producer = static_cast<int>(blockIdx.x) - firstProducer;
  if (producer < 0) return;
  const int producers = static_cast<int>(gridDim.x) - firstProducer;
  const int warp = threadIdx.x / WARP_SIZE;
  const int lane = getLaneId();
  // Two independent warp/TMA slots fit the ordinary compact-dispatch shared allocation.
  if (warp >= DispatchMaxNWarpGroups) return;
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(Bf16);
  const int topk = workload.numTopk_;
  const int chunkRows = context->gpuNetIoPipelineChunkRows_;
  const int localExperts = workload.numExperts_ / context->numRanks_;
  const size_t stride = GpuNetIoDistributeRow::stride(Hidden, topk);
  auto* staged = reinterpret_cast<uint8_t*>(shared) + dispatchSharedControlBytes(context->numRanks_, true) +
                 warp * dispatchPayloadStride<DispatchDataType::BF16>(Hidden, topk, 0);
  auto* barriers = reinterpret_cast<mscclpp::BulkBarrier*>(
      reinterpret_cast<uint8_t*>(shared) + dispatchSharedControlBytes(context->numRanks_, true) +
      DispatchMaxNWarpGroups * dispatchPayloadStride<DispatchDataType::BF16>(Hidden, topk, 0));
  auto& barrier = barriers[warp];
  if (lane == 0) barrier.init();
  __syncwarp();
  uint32_t phase = 0;
  WorkspaceView workspace(context->workspace_, context->numRanks_, workload.numExperts_);
  for (int token = producer * DispatchMaxNWarpGroups + warp; token < workload.numTokens_;
       token += producers * DispatchMaxNWarpGroups) {
    if (lane == 0) {
      barrier.arriveAndExpect(HiddenBytes);
      mscclpp::bulkLoad(staged, static_cast<const uint8_t*>(input) + static_cast<size_t>(token) * HiddenBytes,
                        HiddenBytes, barrier);
    }
    const int expert = distributeExpert(routes, token, lane, workload);
    const int destination = expert < 0 ? -1 : expert / localExperts;
    const bool leader = isFirstLaneForRank(destination, lane);
    int slot = -1;
    if (destination >= 0 && leader) {
      slot = atomicAdd(workspace.dispatchRankPayloadSlots_ + destination, 1);
      EP_DEVICE_ASSERT(slot < workload.maxTokensPerRank_);
    }
    const unsigned match = __match_any_sync(0xffffffff, destination);
    slot = __shfl_sync(0xffffffff, slot, __ffs(match) - 1);
    if (lane < topk) workspace.rankMajorSendIndices_[token * topk + lane] = slot;
    const bool remote = destination >= 0 && !transport.isNvlinkPeer(destination);
    const bool hasRemote = __any_sync(0xffffffff, remote);
    int row = -1;
    if (lane == 0 && hasRemote) row = atomicAdd(context->gpuNetIoDistributeScratch_, 1);
    row = __shfl_sync(0xffffffff, row, 0);
    if (context->gpuNetIoAggregateCombine_ && lane == 0) context->gpuNetIoAggregateTokenRows_[token] = row;
    if (lane == 0) barrier.wait(phase);
    __syncwarp();
    mscclpp::bulkFence();
    if (hasRemote) {
      EP_DEVICE_ASSERT(row < context->gpuNetIoDistributeCapacity_);
      auto* send = static_cast<uint8_t*>(context->gpuNetIoDistributeSend_) + static_cast<size_t>(row) * stride;
      const GpuNetIoDistributeRow wire(send, Hidden, topk);
      for (int vector = lane; vector < HiddenBytes / sizeof(int4); vector += WARP_SIZE) {
        reinterpret_cast<int4*>(send)[vector] = reinterpret_cast<const int4*>(staged)[vector];
      }
      if (lane < topk) {
        wire.ids_[lane] = expert;
        wire.weights_[lane] = expert >= 0 ? (weights == nullptr ? 1.0f : weights[token * topk + lane]) : 0.0f;
        wire.slots_[lane] = slot;
      }
      // Warp synchronization joins every writer into lane 0's release RMW.
      // GPU-fenced rows get their system publication from the acquiring QP poster.
      fenceRankMajorStagedRow(transport);
      __syncwarp();
      if (lane == 0) {
        (void)mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(
            context->gpuNetIoDistributeScratch_ + DistributeScratchCounters + row / chunkRows, 1,
            mscclpp::memoryOrderRelease);
      }
    }
    if (destination >= 0 && leader && !remote) {
      issueRankMajorTokenStore<Hidden, DispatchDataType::BF16>(output, transport, slot, workload.maxTokensPerRank_,
                                                               staged, destination);
    }
    const RankMajorRoute route{destination, slot, leader};
    sendRankMajorMetadataNvlink(transport, outputIds, outputWeights, routes, weights, route, token, topk, localExperts,
                                workload.maxTokensPerRank_, workload.invalidTokenExpertId_);
    completeRankMajorTokenStore(workspace, destination >= 0 && leader && !remote ? destination : -1);
    __syncwarp();
  }
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void distributeControl(const int64_t* routes, const Workload& workload, void* recvBuffer,
                                             const DeviceContext* context, const TransportView& transport,
                                             int* shared) {
  const int rank = context->rank_, ranks = context->numRanks_;
  const int lane = getLaneId(), warp = threadIdx.x / WARP_SIZE;
  const int localExperts = workload.numExperts_ / ranks;
  auto* state = context->gpuNetIoDistributeControl_;
  auto* network = transport.gpuNetIo_;
  const int gateway = context->gpuNetIoGateway_;
  const int qps = network->numQpsPerPeer;
  const int chunkRows = context->gpuNetIoPipelineChunkRows_;
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
    state->sendHeader.counts[peer].write(shared[peer], workload.epoch_);
  }
  const int rows = shared[ranks];
  if (threadIdx.x == 0) state->sendHeader.rows.write(rows, workload.epoch_);
  if (threadIdx.x < ranks) __threadfence();
  __syncthreads();
  // Keep local publishers separate from the independent per-QP posting warps.
  const int firstPublisher = qps * WARP_SIZE;
  if (threadIdx.x >= firstPublisher && threadIdx.x < firstPublisher + ranks) {
    const int destination = threadIdx.x - firstPublisher;
    const int count = shared[destination];
    if (transport.isNvlinkPeer(destination)) {
      auto* counts = static_cast<mscclpp::LL8Packet*>(transport.mappedBuffer(recvBuffer, destination)) + ranks;
      // The packet validates its own epoch; nonempty payloads have a separate ready signal.
      counts[rank].write(count, workload.epoch_);
    }
    if (!transport.isNvlinkPeer(destination)) {
      workspace.dispatchRankPayloadCompletions_[destination] = count;
    } else if (count > 0) {
      while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(workspace.dispatchRankPayloadCompletions_ + destination,
                                                            mscclpp::memoryOrderAcquire) != count) {
      }
      if (destination == rank)
        workspace.dispatchLocalPayloadReady_->release();
      else
        transport.baseMemoryChannels_[destination].signal();
    }
  }
  const size_t stride = GpuNetIoDistributeRow::stride(Hidden, workload.numTopk_);
  if (warp < qps && lane == 0) {
    const int queue = warp;
    auto handle = network->channel(gateway, queue);
    const uint64_t dstOffset = transport.symmetricOffset(context->gpuNetIoDistributeRecv_);
    const uint64_t srcOffset = transport.symmetricOffset(context->gpuNetIoDistributeSend_);
    const uint64_t flagOffset = transport.symmetricOffset(state->chunks + queue);
    const auto dst = handle.gpuNetIo_.memory(handle.dst_, false, dstOffset, rows * stride);
    const auto src = handle.gpuNetIo_.memory(handle.src_, true, srcOffset, rows * stride);
    const auto sig = handle.gpuNetIo_.memory(handle.dst_, false, flagOffset, sizeof(uint64_t));
    auto& sequence = network->signalState(gateway, queue);
    if (queue == 0) {
      const uint64_t recvHeader = transport.symmetricOffset(&state->recvHeader);
      const uint64_t sendHeader = transport.symmetricOffset(&state->sendHeader);
      const size_t headerBytes = GpuNetIoDistributeHeader::wireBytes(ranks);
      const auto headerDst = handle.gpuNetIo_.memory(handle.dst_, false, recvHeader, headerBytes);
      const auto headerSrc = handle.gpuNetIo_.memory(handle.src_, true, sendHeader, headerBytes);
      // Inline puts copy the header into the WQE after the CTA join; larger
      // headers remain DMA sources and need system publication.
      if (headerBytes > DOCA_GPUNETIO_VERBS_MAX_INLINE_SIZE) __threadfence_system();
      // Each LL word validates its own epoch; no separate metadata-ready WQE
      // is needed. This says nothing about the independently signaled payload.
      handle.gpuNetIo_.put(headerDst, recvHeader, headerSrc, sendHeader, headerBytes);
    }
    // Each QP has one ordered chunk stream. Other QPs may overtake the header;
    // the gateway validates the LL row count before consuming an arriving chunk.
    for (int chunk = queue, begin = queue * chunkRows; begin < rows; chunk += qps, begin += qps * chunkRows) {
      const int count = rows - begin < chunkRows ? rows - begin : chunkRows;
      while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(
                 context->gpuNetIoDistributeScratch_ + DistributeScratchCounters + chunk,
                 mscclpp::memoryOrderAcquire) != count) {
      }
      // System-fenced writers already published the entire acquired chunk.
      if (transport.gpuNetIoRowFence_ != GpuNetIoRowFenceSystem) __threadfence_system();
      handle.gpuNetIo_.putWithSignalAt(dst, dstOffset + begin * stride, src, srcOffset + begin * stride, count * stride,
                                       sig, flagOffset, network->nextSignalToken(sequence, flagOffset, 1));
    }
  }
  // All QP posting warps finish before QP 0's owner sends the forwarding credit.
  __syncthreads();
  // Combine need not involve the gateway. Its explicit credit is what protects
  // both registered staging regions against the next dispatch (including empty rounds).
  if (threadIdx.x == 0) {
    while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(context->gpuNetIoDistributeScratch_ + 2,
                                                          mscclpp::memoryOrderAcquire) != 1) {
    }
    // Forwarding writers system-fenced before the release/acquire completion join.
    // Send our inline ACK before waiting for the peer's ACK, including empty rounds.
    const auto offset = transport.symmetricOffset(&state->ack);
    auto channel = network->channel(gateway, 0);
    const auto destination = channel.gpuNetIo_.memory(channel.dst_, false, offset, sizeof(uint64_t));
    auto& sequence = network->signalState(gateway, 0);
    channel.gpuNetIo_.signalAt(destination, offset, network->nextSignalToken(sequence, offset, 1));
    const uint64_t ack = state->ackBase + 1;
    while (mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(&state->ack, mscclpp::memoryOrderAcquire) < ack) {
    }
    state->ackBase = ack;
  }
  __syncthreads();
  if (warp < qps && lane == 0) network->channel(gateway, warp).flush(-1);
  __syncthreads();
  if (threadIdx.x != 0) return;
  if (context->gpuNetIoDistributeStats_) {
    int directRows = 0;
    for (int peer = 0; peer < ranks; ++peer) {
      if (!transport.isNvlinkPeer(peer)) directRows += shared[peer];
    }
    printf(
        "[EP-SEND-DISTRIBUTE] rank=%d epoch=%u uniqueRows=%d directRemoteRows=%d payloadBytes=%llu "
        "rowWireBytes=%llu headerBytes=%llu\n",
        rank, workload.epoch_, rows, directRows, static_cast<unsigned long long>(rows) * Hidden * sizeof(Bf16),
        static_cast<unsigned long long>(rows) * stride,
        static_cast<unsigned long long>(GpuNetIoDistributeHeader::wireBytes(ranks)));
  }
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void distributeForward(void* output, int* outputIds, float* outputWeights,
                                             const Workload& workload, void* recvBuffer, const DeviceContext* context,
                                             const TransportView& transport, int* shared) {
  const int forward = static_cast<int>(blockIdx.x) - context->numRanks_ - 1;
  const int forwarders = static_cast<int>(gridDim.x) - context->numRanks_ - 1;
  constexpr int ForwardWarps = DispatchMaxNWarpGroups;
  const int lane = getLaneId(), warp = threadIdx.x / WARP_SIZE;
  const int source = context->gpuNetIoGateway_;
  const int topk = workload.numTopk_;
  const int chunkRows = context->gpuNetIoPipelineChunkRows_;
  const int localExperts = workload.numExperts_ / context->numRanks_;
  const int qps = transport.gpuNetIo_->numQpsPerPeer;
  auto* state = context->gpuNetIoDistributeControl_;
  if (threadIdx.x == 0) {
    shared[0] = state->recvHeader.rows.read(workload.epoch_, -1);
    if (context->gpuNetIoAggregateCombine_ && forward == 0) {
      *context->gpuNetIoAggregateIncomingRows_ = shared[0];
    }
  }
  __syncthreads();
  const int rows = shared[0];
  if (forward == 0) {
    // Publish counts from the gateway, not directly from the original source.
    // The gateway's next Combine handshake includes every local consumer even
    // for zero routes, preventing a future round from overwriting unread counts.
    for (int peer = threadIdx.x; peer < context->numRanks_; peer += blockDim.x) {
      if (transport.isNvlinkPeer(peer)) {
        auto* counts = static_cast<mscclpp::LL8Packet*>(transport.mappedBuffer(recvBuffer, peer)) + context->numRanks_;
        // The row-count word does not order other words in a relaxed-ordering MR.
        counts[source].write(state->recvHeader.counts[peer].read(workload.epoch_, -1), workload.epoch_);
      }
    }
  }
  const int chunks = configCellDiv(rows, chunkRows);
  const size_t stride = GpuNetIoDistributeRow::stride(Hidden, topk);
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(Bf16);
  auto* staged = reinterpret_cast<uint8_t*>(shared) + dispatchSharedControlBytes(context->numRanks_, true) +
                 warp * dispatchPayloadStride<DispatchDataType::BF16>(Hidden, topk, 0);
  auto* barriers = reinterpret_cast<mscclpp::BulkBarrier*>(
      reinterpret_cast<uint8_t*>(shared) + dispatchSharedControlBytes(context->numRanks_, true) +
      ForwardWarps * dispatchPayloadStride<DispatchDataType::BF16>(Hidden, topk, 0));
  if (warp < ForwardWarps && lane == 0) barriers[warp].init();
  __syncthreads();
  uint32_t phase = 0;
  if (warp < ForwardWarps) {
    for (int row = forward * ForwardWarps + warp; row < rows; row += forwarders * ForwardWarps) {
      const int chunk = row / chunkRows;
      if (lane == 0) {
        const int queue = chunk % qps;
        const uint64_t target = state->chunkBase[queue] + chunk / qps + 1;
        while (mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(state->chunks + queue, mscclpp::memoryOrderAcquire) <
               target) {
        }
      }
      __syncwarp();
      auto* landing = static_cast<uint8_t*>(context->gpuNetIoDistributeRecv_) + static_cast<size_t>(row) * stride;
      const GpuNetIoDistributeRow wire(landing, Hidden, topk);
      if (lane == 0) {
        barriers[warp].arriveAndExpect(HiddenBytes);
        mscclpp::bulkLoad(staged, landing, HiddenBytes, barriers[warp]);
      }
      const int expert = lane < topk ? wire.ids_[lane] : -1;
      const int destination = expert >= 0 ? expert / localExperts : -1;
      const int slot = lane < topk ? wire.slots_[lane] : -1;
      if (context->gpuNetIoAggregateCombine_ && lane < topk) {
        auto* cached = context->gpuNetIoAggregateRoutes_ + static_cast<size_t>(row) * 2 * topk;
        cached[lane] = expert;
        cached[topk + lane] = slot;
      }
      const float weight = lane < topk ? wire.weights_[lane] : 0.0f;
      const bool leader = isFirstLaneForRank(destination, lane);
      const bool local = destination >= 0 && transport.isNvlinkPeer(destination);
      unsigned leaders = __ballot_sync(0xffffffff, local && leader);
      if (lane == 0) barriers[warp].wait(phase);
      __syncwarp();
      mscclpp::bulkFence();
      if (local && leader) {
        EP_DEVICE_ASSERT(slot >= 0 && slot < workload.maxTokensPerRank_);
        auto* destinationBuffer = static_cast<uint8_t*>(transport.mappedBuffer(output, destination));
        const size_t index = static_cast<size_t>(source) * workload.maxTokensPerRank_ + slot;
        mscclpp::bulkStore(destinationBuffer + index * HiddenBytes, staged, HiddenBytes);
        mscclpp::bulkStoreCommit();
      }
      while (leaders != 0) {
        const int first = __ffs(leaders) - 1;
        const int peer = __shfl_sync(0xffffffff, destination, first);
        const int peerSlot = __shfl_sync(0xffffffff, slot, first);
        if (lane < topk) {
          const size_t index = (static_cast<size_t>(source) * workload.maxTokensPerRank_ + peerSlot) * topk + lane;
          static_cast<int*>(transport.mappedBuffer(outputIds, peer))[index] =
              destination == peer ? expert : workload.invalidTokenExpertId_;
          static_cast<float*>(transport.mappedBuffer(outputWeights, peer))[index] = destination == peer ? weight : 0.0f;
        }
        leaders &= leaders - 1;
      }
      // Wait for the actual NVLink stores, not merely shared-source consumption.
      if (local && leader) mscclpp::bulkStoreWait();
      __syncwarp();
    }
  }
  // Only row forwarders and count publishers wrote peer-visible memory.
  if ((warp < ForwardWarps && forward * ForwardWarps + warp < rows) ||
      (forward == 0 && threadIdx.x < context->numRanks_)) {
    __threadfence_system();
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const int previous = mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(context->gpuNetIoDistributeScratch_ + 1, 1,
                                                                            mscclpp::memoryOrderAcqRel);
    shared[1] = previous == forwarders - 1;
  }
  __syncthreads();
  if (!shared[1]) return;
  // Only the last CTA updates the baselines. Even empty forward CTAs joined, so
  // no CTA can subsequently read a next-round baseline for this round.
  for (int queue = threadIdx.x; queue < qps; queue += blockDim.x) {
    state->chunkBase[queue] += (chunks + qps - 1 - queue) / qps;
  }
  __syncthreads();
  // Hand forwarding completion to the sole QP 0 poster. Local readiness can
  // publish independently because both depend on the same completed writers.
  if (threadIdx.x == configAlign(context->numRanks_, WARP_SIZE)) {
    mscclpp::atomicStore<int, mscclpp::scopeDevice>(context->gpuNetIoDistributeScratch_ + 2, 1,
                                                    mscclpp::memoryOrderRelease);
  }
  for (int peer = threadIdx.x; peer < context->numRanks_; peer += blockDim.x) {
    if (transport.isNvlinkPeer(peer)) {
      auto* target = static_cast<GpuNetIoDistributeControl*>(transport.mappedBuffer(state, peer));
      mscclpp::atomicStore<uint32_t, mscclpp::scopeSystem>(target->forwardReady + source, workload.epoch_,
                                                           mscclpp::memoryOrderRelease);
    }
  }
}

MSCCLPP_DEVICE_INLINE void distributeReceive(int* outputIds, float* outputWeights, int* outputCount,
                                             const Workload& workload, void* recvBuffer, const DeviceContext* context,
                                             const TransportView& transport, int* shared) {
  const int source = static_cast<int>(blockIdx.x) - 1;
  if (threadIdx.x == 0) {
    const auto* counts = static_cast<mscclpp::LL8Packet*>(recvBuffer) + context->numRanks_;
    shared[0] = counts[source].read(workload.epoch_, -1);
    outputCount[source] = shared[0];
    WorkspaceView workspace(context->workspace_, context->numRanks_, workload.numExperts_);
    workspace.dispatchRecvCounts_[source] = shared[0];
    if (!transport.isNvlinkPeer(source)) {
      while (mscclpp::atomicLoad<uint32_t, mscclpp::scopeSystem>(
                 context->gpuNetIoDistributeControl_->forwardReady + source, mscclpp::memoryOrderAcquire) !=
             workload.epoch_) {
      }
    } else if (shared[0] > 0) {
      if (source == context->rank_)
        workspace.dispatchLocalPayloadReady_->acquire();
      else
        transport.baseMemoryChannels_[source].wait(-1);
    }
  }
  __syncthreads();
  for (int entry = shared[0] * workload.numTopk_ + threadIdx.x; entry < workload.maxTokensPerRank_ * workload.numTopk_;
       entry += blockDim.x) {
    const size_t index = static_cast<size_t>(source) * workload.maxTokensPerRank_ * workload.numTopk_ + entry;
    outputIds[index] = workload.invalidTokenExpertId_;
    outputWeights[index] = 0.0f;
  }
}

#endif

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
__global__ __launch_bounds__(DispatchNThreads, 1) void hybridSendDistributeDispatchKernel(
    void* output, void* outputScales, int* outputSrcInfo, int* outputIds, float* outputWeights, int64_t* outputLayout,
    int* outputCount, const int64_t* routes, const float* weights, const void* input, Workload workload,
    void* recvBuffer, const DeviceContext* context) {
#if defined(MSCCLPP_USE_GPUNETIO) && MSCCLPP_BULK_AVAILABLE
  static_assert(DataType == DispatchDataType::BF16);
  extern __shared__ __align__(128) uint8_t sharedMemory[];
  auto* shared = reinterpret_cast<int*>(sharedMemory);
  workload.epoch_ =
      beginRankMajorDispatchEpoch(WorkspaceView(context->workspace_, context->numRanks_, workload.numExperts_));
  const TransportView transport(context);
  if (blockIdx.x == 0) {
    distributeControl<Hidden>(routes, workload, recvBuffer, context, transport, shared);
  } else if (blockIdx.x <= context->numRanks_) {
    distributeReceive(outputIds, outputWeights, outputCount, workload, recvBuffer, context, transport, shared);
  } else {
    if (blockIdx.x > context->numRanks_ + GpuNetIoDistributeForwardBlocks) {
      distributeProduce<Hidden>(output, outputIds, outputWeights, input, routes, weights, workload, context, transport,
                                shared);
      __syncthreads();
    }
    // Completed producer CTAs reuse their two warp/TMA slots to spread arriving
    // rows across more SMs; every forwarder joins before ACK or buffer reuse.
    distributeForward<Hidden>(output, outputIds, outputWeights, workload, recvBuffer, context, transport, shared);
  }
#endif
}

struct HybridSendDistributeDispatchKernelSelector {
  template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
  static auto get() {
    return hybridSendDistributeDispatchKernel<Hidden, DataType, ScaleBlockSize>;
  }
};

}  // namespace mscclpp::ep

#endif
