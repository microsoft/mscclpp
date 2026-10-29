// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_COMBINE_HYBRID_COMBINE_CUH_
#define MSCCLPP_EP_COMBINE_HYBRID_COMBINE_CUH_

#include "combine/common.cuh"

namespace mscclpp::ep {

#if defined(MSCCLPP_USE_GPUNETIO) && MSCCLPP_BULK_AVAILABLE

template <int Hidden>
MSCCLPP_DEVICE_INLINE void aggregateReduceRow(float* destination, const void* expertOutput, const int64_t* topkIndices,
                                              int row, bool gateway, const Workload& workload,
                                              const DeviceContext* context, const TransportView& transport,
                                              const WorkspaceView& workspace, uint8_t* sharedMemory, uint32_t& phase) {
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(Bf16);
  constexpr int HiddenVectors = HiddenBytes / sizeof(int4);
  constexpr int PairsPerVector = sizeof(int4) / sizeof(mscclpp::bf16x2);
  constexpr size_t RowsBytes = MaxNumTopk * HiddenBytes;
  const int lane = getLaneId();
  const int topk = workload.numTopk_;
  const int localExperts = workload.numExperts_ / context->numRanks_;
  const int owner = gateway ? context->gpuNetIoGateway_ : context->rank_;
  auto* tiles = reinterpret_cast<int4*>(sharedMemory);
  auto* barriers = reinterpret_cast<mscclpp::BulkBarrier*>(sharedMemory + RowsBytes);
  auto* valid = reinterpret_cast<int*>(barriers + MaxNumTopk);
  if (threadIdx.x < WARP_SIZE) {
    int64_t expert = -1;
    int slot = -1;
    if (lane < topk) {
      if (gateway) {
        const auto* cached = context->gpuNetIoAggregateRoutes_ + static_cast<size_t>(row) * 2 * topk;
        expert = cached[lane];
        slot = cached[topk + lane];
      } else {
        expert = topkIndices[row * topk + lane];
        slot = workspace.rankMajorSendIndices_[row * topk + lane];
      }
    }
    const int peer = expert >= 0 && expert < workload.numExperts_ ? static_cast<int>(expert) / localExperts : -1;
    const bool leader = isFirstLaneForRank(peer, lane);
    const bool active = peer >= 0 && leader && transport.isNvlinkPeer(peer);
    if (lane < MaxNumTopk) valid[lane] = active;
    bool pending = active;
    while (__any_sync(0xffffffff, pending)) {
      if (pending && mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(
                         workspace.combineRankReadyEpochs_ + peer, mscclpp::memoryOrderAcquire) == workload.epoch_) {
        EP_DEVICE_ASSERT(slot >= 0 && slot < workload.maxTokensPerRank_);
        const auto* peerOutput =
            static_cast<const uint8_t*>(transport.mappedBuffer(const_cast<void*>(expertOutput), peer));
        const size_t sourceRow = static_cast<size_t>(owner) * workload.maxTokensPerRank_ + slot;
        barriers[lane].arriveAndExpect(HiddenBytes);
        mscclpp::bulkLoad(sharedMemory + lane * HiddenBytes, peerOutput + sourceRow * HiddenBytes, HiddenBytes,
                          barriers[lane]);
        pending = false;
      }
    }
    if (active) barriers[lane].wait(phase);
    __syncwarp();
    if (lane == 0) mscclpp::bulkFence();
  }
  __syncthreads();
  for (int vector = threadIdx.x; vector < HiddenVectors; vector += blockDim.x) {
    mscclpp::f32x2 sum[PairsPerVector];
#pragma unroll
    for (int pair = 0; pair < PairsPerVector; ++pair) {
      sum[pair].storage = make_float2(0.0f, 0.0f);
    }
#pragma unroll
    for (int route = 0; route < MaxNumTopk; ++route) {
      if (!valid[route]) continue;
      const int4 packed = tiles[route * HiddenVectors + vector];
      const auto* pairs = reinterpret_cast<const mscclpp::bf16x2*>(&packed);
#pragma unroll
      for (int pair = 0; pair < PairsPerVector; ++pair) {
        const auto value = mscclpp::to<mscclpp::f32x2>(pairs[pair]);
        sum[pair].data[0] += value.data[0];
        sum[pair].data[1] += value.data[1];
      }
    }
#pragma unroll
    for (int pair = 0; pair < PairsPerVector; ++pair) {
      reinterpret_cast<mscclpp::f32x2*>(destination)[vector * PairsPerVector + pair] = sum[pair];
    }
  }
  // The CTA barrier joins every writer into thread 0's release counter. The
  // acquiring poster system-fences the complete chunk before NIC DMA.
  __syncthreads();
  if (gateway && threadIdx.x == 0) {
    (void)mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(
        context->gpuNetIoAggregateChunkReady_ + row / context->gpuNetIoCombineChunkRows_, 1,
        mscclpp::memoryOrderRelease);
  }
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void aggregateControl(const Workload& workload, const DeviceContext* context,
                                            const TransportView& transport, const WorkspaceView& workspace) {
  const int peer = threadIdx.x;
  const int ranks = context->numRanks_;
  auto* network = transport.gpuNetIo_;
  const int queue = static_cast<int>(threadIdx.x) / WARP_SIZE - configCellDiv(ranks, WARP_SIZE);
  if (peer < ranks && transport.isNvlinkPeer(peer)) {
    if (!transport.isSelf(peer)) {
      transport.baseMemoryChannels_[peer].signal();
      transport.baseMemoryChannels_[peer].wait(-1);
    }
    mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(workspace.combineRankReadyEpochs_ + peer, workload.epoch_,
                                                         mscclpp::memoryOrderRelease);
    // Dispatch leaves its forward-complete word at one. During Combine it counts
    // worker CTAs that have finished every NVLink read; FP32 IB consumption is
    // independent, so the end handshake can overlap that remaining work.
    while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(context->gpuNetIoDistributeScratch_ + 2,
                                                          mscclpp::memoryOrderAcquire) != gridDim.x) {
    }
    if (!transport.isSelf(peer)) {
      transport.baseMemoryChannels_[peer].signal();
      transport.baseMemoryChannels_[peer].wait(-1);
    }
  }
  if (queue < 0 || queue >= network->numQpsPerPeer || getLaneId() != 0) return;
  const int rows = *context->gpuNetIoAggregateIncomingRows_;
  const int chunkRows = context->gpuNetIoCombineChunkRows_;
  constexpr size_t RowBytes = static_cast<size_t>(Hidden) * sizeof(float);
  const int gateway = context->gpuNetIoGateway_;
  auto handle = network->channel(gateway, queue);
  const uint64_t dstOffset = transport.symmetricOffset(context->gpuNetIoAggregateRecv_);
  const uint64_t srcOffset = transport.symmetricOffset(context->gpuNetIoAggregateSend_);
  const uint64_t flagOffset = transport.symmetricOffset(context->gpuNetIoAggregateControl_->chunks + queue);
  const auto dst = handle.gpuNetIo_.memory(handle.dst_, false, dstOffset, rows * RowBytes);
  const auto src = handle.gpuNetIo_.memory(handle.src_, true, srcOffset, rows * RowBytes);
  const auto sig = handle.gpuNetIo_.memory(handle.dst_, false, flagOffset, sizeof(uint64_t));
  // Dispatch has drained its QPs, and this warp is the only Combine submitter for this QP.
  auto& state = network->signalState(gateway, queue);
  for (int chunk = queue, begin = queue * chunkRows; begin < rows;
       chunk += network->numQpsPerPeer, begin += network->numQpsPerPeer * chunkRows) {
    const int count = rows - begin < chunkRows ? rows - begin : chunkRows;
    while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(context->gpuNetIoAggregateChunkReady_ + chunk,
                                                          mscclpp::memoryOrderAcquire) != count) {
    }
    // These DMA sources were written by other CTAs; DOCA's queue-release fence is GPU-scoped.
    __threadfence_system();
    handle.gpuNetIo_.putWithSignalAt(dst, dstOffset + begin * RowBytes, src, srcOffset + begin * RowBytes,
                                     count * RowBytes, sig, flagOffset, network->nextSignalToken(state, flagOffset, 1));
  }
  // The posting warp owns this QP until its final completion drain.
  handle.flush(-1);
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void aggregateFinalize(void* output, int token, const DeviceContext* context) {
  constexpr int HiddenVectors = Hidden * sizeof(Bf16) / sizeof(int4);
  constexpr int PairsPerVector = sizeof(int4) / sizeof(mscclpp::bf16x2);
  const int row = context->gpuNetIoAggregateTokenRows_[token];
  if (threadIdx.x == 0 && row >= 0) {
    const int chunk = row / context->gpuNetIoCombineChunkRows_;
    const int qps = context->gpuNetIo_->numQpsPerPeer;
    const int queue = chunk % qps;
    auto* control = context->gpuNetIoAggregateControl_;
    const uint64_t target = control->chunkBase[queue] + chunk / qps + 1;
    while (mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(control->chunks + queue, mscclpp::memoryOrderAcquire) <
           target) {
    }
  }
  __syncthreads();
  const auto* local =
      reinterpret_cast<const mscclpp::f32x2*>(context->gpuNetIoAggregateLocal_ + static_cast<size_t>(token) * Hidden);
  for (int vector = threadIdx.x; vector < HiddenVectors; vector += blockDim.x) {
    int4 packed;
    auto* pairs = reinterpret_cast<mscclpp::bf16x2*>(&packed);
#pragma unroll
    for (int pair = 0; pair < PairsPerVector; ++pair) {
      auto value = local[vector * PairsPerVector + pair];
      if (row >= 0) {
        const auto* remote = reinterpret_cast<const mscclpp::f32x2*>(context->gpuNetIoAggregateRecv_ +
                                                                     static_cast<size_t>(row) * Hidden);
        const auto partial = remote[vector * PairsPerVector + pair];
        value.data[0] += partial.data[0];
        value.data[1] += partial.data[1];
      }
      pairs[pair] = mscclpp::to<mscclpp::bf16x2>(value);
    }
    static_cast<int4*>(output)[static_cast<size_t>(token) * HiddenVectors + vector] = packed;
  }
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void aggregateCombineBody(void* output, const void* expertOutput, const int64_t* topkIndices,
                                                Workload workload, const DeviceContext* context) {
  extern __shared__ __align__(128) uint8_t sharedMemory[];
  const TransportView transport(context);
  const WorkspaceView workspace(context->workspace_, context->numRanks_, workload.numExperts_);
  if (blockIdx.x == 0) {
    aggregateControl<Hidden>(workload, context, transport, workspace);
  } else {
    constexpr size_t RowsBytes = static_cast<size_t>(MaxNumTopk) * Hidden * sizeof(Bf16);
    auto* barriers = reinterpret_cast<mscclpp::BulkBarrier*>(sharedMemory + RowsBytes);
    if (threadIdx.x < MaxNumTopk) barriers[threadIdx.x].init();
    __syncthreads();
    uint32_t phase = 0;
    const int worker = static_cast<int>(blockIdx.x) - 1;
    const int workers = static_cast<int>(gridDim.x) - 1;
    const int incoming = *context->gpuNetIoAggregateIncomingRows_;
    const int chunkRows = context->gpuNetIoCombineChunkRows_;
    const int qps = transport.gpuNetIo_->numQpsPerPeer;
    // Publish gateway partials first so their IB transfer overlaps the independent
    // local-domain reduction. Each partial retains its original FP32 route order.
    for (int row = worker; row < incoming; row += workers) {
      // Prioritize each QP's first chunk rather than contending with every later
      // row for NVLink. Dependencies point strictly to an earlier chunk.
      if (threadIdx.x == 0 && row >= qps * chunkRows) {
        const int previous = row / chunkRows - qps;
        while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(context->gpuNetIoAggregateChunkReady_ + previous,
                                                              mscclpp::memoryOrderAcquire) != chunkRows) {
        }
      }
      __syncthreads();
      aggregateReduceRow<Hidden>(context->gpuNetIoAggregateSend_ + static_cast<size_t>(row) * Hidden, expertOutput,
                                 topkIndices, row, true, workload, context, transport, workspace, sharedMemory, phase);
    }
    // Keep local-only reads off the critical gateway production path. All CTAs
    // finish their own gateway rows before entering this wait, avoiding cycles.
    if (threadIdx.x == 0) {
      for (int begin = 0; begin < incoming; begin += chunkRows) {
        const int count = min(chunkRows, incoming - begin);
        while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(context->gpuNetIoAggregateChunkReady_ + begin / chunkRows,
                                                              mscclpp::memoryOrderAcquire) != count) {
        }
      }
    }
    __syncthreads();
    for (int row = worker; row < workload.numTokens_; row += workers) {
      aggregateReduceRow<Hidden>(context->gpuNetIoAggregateLocal_ + static_cast<size_t>(row) * Hidden, expertOutput,
                                 topkIndices, row, false, workload, context, transport, workspace, sharedMemory, phase);
    }
    if (threadIdx.x == 0) {
      (void)mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(context->gpuNetIoDistributeScratch_ + 2, 1,
                                                               mscclpp::memoryOrderRelease);
    }
    // A CTA must produce ALL of its gateway rows before waiting for any remote
    // result, including when it owns zero input tokens. Otherwise peers can cycle.
    for (int token = worker; token < workload.numTokens_; token += workers) {
      aggregateFinalize<Hidden>(output, token, context);
    }
  }
  workspace.combineSyncer_->sync(gridDim.x);
  if (blockIdx.x == 0) {
    const int chunks = configCellDiv(context->gpuNetIoDistributeScratch_[0], context->gpuNetIoCombineChunkRows_);
    const int qps = transport.gpuNetIo_->numQpsPerPeer;
    for (int queue = threadIdx.x; queue < qps; queue += blockDim.x) {
      context->gpuNetIoAggregateControl_->chunkBase[queue] += (chunks + qps - 1 - queue) / qps;
    }
    // No additional IB ACK offset: the gateway's NEXT Dispatch cannot finish
    // without our next-Dispatch ACK, issued only after this Combine consumed its
    // landing rows. Thus its next Combine cannot overwrite them early, even for
    // empty dispatch rounds. Separate FP32 send storage is drained above.
    if (threadIdx.x == 0 && context->gpuNetIoDistributeStats_) {
      const int rows = *context->gpuNetIoAggregateIncomingRows_;
      printf("[EP-AGGREGATE-COMBINE] rank=%d epoch=%u rows=%d payloadBytes=%llu chunkRows=%d qps=%d\n", context->rank_,
             workload.epoch_, rows, static_cast<unsigned long long>(rows) * Hidden * sizeof(float),
             context->gpuNetIoCombineChunkRows_, qps);
    }
    // All workers/posters have joined, and this CTA must finish reading the
    // dispatch row count before retiring it. Allocations start zeroed; stream
    // order makes these resets visible to the next paired Dispatch/Combine.
    __syncthreads();
    for (int peer = threadIdx.x; peer < context->numRanks_; peer += blockDim.x) {
      workspace.dispatchRankPayloadSlots_[peer] = 0;
      workspace.dispatchRankPayloadCompletions_[peer] = 0;
      workspace.combineRankReadyEpochs_[peer] = 0;
    }
    const int dispatchCounters =
        3 + configCellDiv(context->gpuNetIoDistributeCapacity_, context->gpuNetIoPipelineChunkRows_);
    for (int index = threadIdx.x; index < dispatchCounters; index += blockDim.x) {
      context->gpuNetIoDistributeScratch_[index] = 0;
    }
    const int combineCounters = configCellDiv(context->gpuNetIoDistributeCapacity_, context->gpuNetIoCombineChunkRows_);
    for (int index = threadIdx.x; index < combineCounters; index += blockDim.x) {
      context->gpuNetIoAggregateChunkReady_[index] = 0;
    }
  }
}

#endif

template <int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout>
__global__ __launch_bounds__(CombineNThreads, 1) void hybridAggregateCombineKernel(
    void* output, const void* expertOutput, const int64_t* topkIndices, const float* topkWeights, const int* srcInfo,
    const int64_t* layoutRange, Workload workload, void* combineRecvBuffer, const void* dispatchRecvBuffer,
    const DeviceContext* context) {
#if defined(MSCCLPP_USE_GPUNETIO) && MSCCLPP_BULK_AVAILABLE
  if constexpr (Layout == DispatchLayout::RANK_MAJOR && DispatchType == DispatchDataType::BF16) {
    workload.epoch_ =
        beginRankMajorCombineEpoch(WorkspaceView(context->workspace_, context->numRanks_, workload.numExperts_));
    if (blockIdx.x == 0) {
      if (threadIdx.x == 0) {
        auto* control = context->gpuNetIoAggregateControl_;
        if (workload.epoch_ == 0 || control->lastCombineEpoch == workload.epoch_) {
          printf(
              "[EPGPUNETIO] aggregate Combine requires one fresh Dispatch per Combine, including graph replays "
              "(rank=%d epoch=%u last=%u)\n",
              context->rank_, workload.epoch_, control->lastCombineEpoch);
          __trap();
        }
        control->lastCombineEpoch = workload.epoch_;
      }
      // No posting warp may publish another round before validation succeeds.
      __syncthreads();
    }
    aggregateCombineBody<Hidden>(output, expertOutput, topkIndices, workload, context);
  }
#endif
}

struct HybridAggregateCombineKernelSelector {
  template <int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout>
  static auto get() {
    return hybridAggregateCombineKernel<Hidden, DispatchType, ScaleBlockSize, Layout>;
  }
};

}  // namespace mscclpp::ep

#endif
