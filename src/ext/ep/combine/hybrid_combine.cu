// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "combine/common.cuh"

namespace mscclpp {
namespace ep {

#if defined(MSCCLPP_USE_GPUNETIO) && MSCCLPP_BULK_AVAILABLE

template <int Hidden>
MSCCLPP_DEVICE_INLINE void aggregateReduceRow(float* destination, const void* expertOutput, const int64_t* topkIndices,
                                              int row, bool gateway, const Workload& workload,
                                              const DeviceContext* context, const TransportView& transport,
                                              const WorkspaceView& workspace, uint32_t flag, uint8_t* sharedMemory,
                                              uint32_t& phase) {
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(Bf16);
  constexpr int HiddenVectors = HiddenBytes / sizeof(int4);
  constexpr int PairsPerVector = sizeof(int4) / sizeof(mscclpp::bf16x2);
  constexpr size_t RowsBytes = MaxNumTopk * HiddenBytes;
  const int lane = getLaneId();
  const int topk = workload.numTopk_;
  const int localExperts = workload.numExperts_ / context->numRanks_;
  const int owner = gateway ? context->gatewayRank_ : context->rank_;
  auto* tiles = reinterpret_cast<int4*>(sharedMemory);
  auto* barriers = reinterpret_cast<mscclpp::BulkBarrier*>(sharedMemory + RowsBytes);
  auto* valid = reinterpret_cast<int*>(barriers + MaxNumTopk);
  if (threadIdx.x < WARP_SIZE) {
    int64_t expert = -1;
    int slot = -1;
    if (lane < topk) {
      if (gateway) {
        const auto* cached = context->aggregateRoutes_ + static_cast<size_t>(row) * 2 * topk;
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
      if (pending && mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(workspace.combineRankReadyFlags_ + peer,
                                                                         mscclpp::memoryOrderAcquire) == flag) {
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
  // The CTA barrier joins every writer into thread 0's release counter.
  __syncthreads();
  if (gateway && threadIdx.x == 0) {
    (void)mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(
        context->aggregateChunkReady_ + row / context->nTokensPerChunk_, 1, mscclpp::memoryOrderRelease);
  }
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void aggregateControl(const DeviceContext* context, const TransportView& transport,
                                            const WorkspaceView& workspace, uint32_t flag) {
  const int peer = threadIdx.x;
  const int ranks = context->numRanks_;
  auto* network = transport.portChannels_;
  const int queue = static_cast<int>(threadIdx.x) / WARP_SIZE - configCellDiv(ranks, WARP_SIZE);
  __shared__ mscclpp::GpuNetIoWqeSlot wqeSlots[MaxQpsPerPeer];
  constexpr auto kExclusive = mscclpp::PortChannelSharingMode::Exclusive;
  if (peer < ranks && transport.isNvlinkPeer(peer)) {
    if (!transport.isSelf(peer)) {
      transport.baseMemoryChannels_[peer].signal();
      transport.baseMemoryChannels_[peer].wait(-1);
    }
    mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(workspace.combineRankReadyFlags_ + peer, flag,
                                                         mscclpp::memoryOrderRelease);
    // Dispatch leaves its forward-complete word at one. During Combine it counts
    // worker CTAs that have finished every NVLink read; FP32 IB consumption is
    // independent, so the end handshake can overlap that remaining work.
    while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(context->distributeScratch_ + 2,
                                                          mscclpp::memoryOrderAcquire) != gridDim.x) {
    }
    if (!transport.isSelf(peer)) {
      transport.baseMemoryChannels_[peer].signal();
      transport.baseMemoryChannels_[peer].wait(-1);
    }
  }
  if (queue < 0 || queue >= network->numQpsPerPeer || getLaneId() != 0) return;
  const int rows = *context->aggregateIncomingRows_;
  const int nTokensPerChunk = context->nTokensPerChunk_;
  constexpr size_t RowBytes = static_cast<size_t>(Hidden) * sizeof(float);
  const int gateway = context->gatewayRank_;
  auto handle = network->channel(gateway, queue);
  handle.setWqeSlot<kExclusive>(&wqeSlots[queue]);
  const uint64_t dstOffset = transport.symmetricOffset(context->aggregateRecvBuffer_);
  const uint64_t srcOffset = transport.symmetricOffset(context->aggregateSendBuffer_);
  const uint64_t flagOffset = transport.symmetricOffset(context->aggregateControl_->chunks + queue);
  // Dispatch has drained its QPs, and this warp is the only Combine submitter for this QP.
  auto& state = network->signalState(gateway, queue);
  for (int chunk = queue, begin = queue * nTokensPerChunk; begin < rows;
       chunk += network->numQpsPerPeer, begin += network->numQpsPerPeer * nTokensPerChunk) {
    const int count = rows - begin < nTokensPerChunk ? rows - begin : nTokensPerChunk;
    while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(context->aggregateChunkReady_ + chunk,
                                                          mscclpp::memoryOrderAcquire) != count) {
    }
    handle.device_.putWithSignalAt<kExclusive>(handle.dstMemory_, dstOffset + begin * RowBytes, handle.srcMemory_,
                                               srcOffset + begin * RowBytes, count * RowBytes, handle.dstMemory_,
                                               flagOffset, network->nextSignalToken(state, flagOffset, 1));
  }
  // The posting warp owns this QP until its final completion drain.
  handle.flush<kExclusive>(-1);
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void aggregateFinalize(void* output, int token, const DeviceContext* context) {
  constexpr int HiddenVectors = Hidden * sizeof(Bf16) / sizeof(int4);
  constexpr int PairsPerVector = sizeof(int4) / sizeof(mscclpp::bf16x2);
  const int row = context->aggregateTokenRows_[token];
  if (threadIdx.x == 0 && row >= 0) {
    const int chunk = row / context->nTokensPerChunk_;
    const int qps = context->portChannels_->numQpsPerPeer;
    const int queue = chunk % qps;
    auto* control = context->aggregateControl_;
    const uint64_t target = control->chunkBase[queue] + chunk / qps + 1;
    while (mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(control->chunks + queue, mscclpp::memoryOrderAcquire) <
           target) {
    }
  }
  __syncthreads();
  const auto* local =
      reinterpret_cast<const mscclpp::f32x2*>(context->aggregateLocalBuffer_ + static_cast<size_t>(token) * Hidden);
  for (int vector = threadIdx.x; vector < HiddenVectors; vector += blockDim.x) {
    int4 packed;
    auto* pairs = reinterpret_cast<mscclpp::bf16x2*>(&packed);
#pragma unroll
    for (int pair = 0; pair < PairsPerVector; ++pair) {
      auto value = local[vector * PairsPerVector + pair];
      if (row >= 0) {
        const auto* remote =
            reinterpret_cast<const mscclpp::f32x2*>(context->aggregateRecvBuffer_ + static_cast<size_t>(row) * Hidden);
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
                                                Workload workload, const DeviceContext* context, uint32_t flag) {
  extern __shared__ __align__(128) uint8_t sharedMemory[];
  const TransportView transport(context);
  const WorkspaceView workspace(context->workspace_, context->numRanks_, workload.numExperts_);
  if (blockIdx.x == 0) {
    aggregateControl<Hidden>(context, transport, workspace, flag);
  } else {
    constexpr size_t RowsBytes = static_cast<size_t>(MaxNumTopk) * Hidden * sizeof(Bf16);
    auto* barriers = reinterpret_cast<mscclpp::BulkBarrier*>(sharedMemory + RowsBytes);
    if (threadIdx.x < MaxNumTopk) barriers[threadIdx.x].init();
    __syncthreads();
    uint32_t phase = 0;
    const int worker = static_cast<int>(blockIdx.x) - 1;
    const int workers = static_cast<int>(gridDim.x) - 1;
    const int incoming = *context->aggregateIncomingRows_;
    const int nTokensPerChunk = context->nTokensPerChunk_;
    const int qps = transport.portChannels_->numQpsPerPeer;
    // Publish gateway partials first so their IB transfer overlaps the independent
    // local-domain reduction. Each partial retains its original FP32 route order.
    for (int row = worker; row < incoming; row += workers) {
      // Prioritize each QP's first chunk rather than contending with every later
      // row for NVLink. Dependencies point strictly to an earlier chunk.
      if (threadIdx.x == 0 && row >= qps * nTokensPerChunk) {
        const int previous = row / nTokensPerChunk - qps;
        while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(context->aggregateChunkReady_ + previous,
                                                              mscclpp::memoryOrderAcquire) != nTokensPerChunk) {
        }
      }
      __syncthreads();
      aggregateReduceRow<Hidden>(context->aggregateSendBuffer_ + static_cast<size_t>(row) * Hidden, expertOutput,
                                 topkIndices, row, true, workload, context, transport, workspace, flag, sharedMemory,
                                 phase);
    }
    // Keep local-only reads off the critical gateway production path. All CTAs
    // finish their own gateway rows before entering this wait, avoiding cycles.
    if (threadIdx.x == 0) {
      for (int begin = 0; begin < incoming; begin += nTokensPerChunk) {
        const int count = min(nTokensPerChunk, incoming - begin);
        while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(context->aggregateChunkReady_ + begin / nTokensPerChunk,
                                                              mscclpp::memoryOrderAcquire) != count) {
        }
      }
    }
    __syncthreads();
    for (int row = worker; row < workload.numTokens_; row += workers) {
      aggregateReduceRow<Hidden>(context->aggregateLocalBuffer_ + static_cast<size_t>(row) * Hidden, expertOutput,
                                 topkIndices, row, false, workload, context, transport, workspace, flag, sharedMemory,
                                 phase);
    }
    if (threadIdx.x == 0) {
      (void)mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(context->distributeScratch_ + 2, 1,
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
    const int chunks = configCellDiv(context->distributeScratch_[0], context->nTokensPerChunk_);
    const int qps = transport.portChannels_->numQpsPerPeer;
    for (int queue = threadIdx.x; queue < qps; queue += blockDim.x) {
      context->aggregateControl_->chunkBase[queue] += (chunks + qps - 1 - queue) / qps;
    }
    // No additional IB ACK offset: the gateway's NEXT Dispatch cannot finish
    // without our next-Dispatch ACK, issued only after this Combine consumed its
    // landing rows. Thus its next Combine cannot overwrite them early, even for
    // empty dispatch rounds. Separate FP32 send storage is drained above.
    // All workers/posters have joined, and this CTA must finish reading the
    // dispatch row count before retiring it. Allocations start zeroed; stream
    // order makes these resets visible to the next paired Dispatch/Combine.
    __syncthreads();
    for (int peer = threadIdx.x; peer < context->numRanks_; peer += blockDim.x) {
      workspace.dispatchRankPayloadSlots_[peer] = 0;
      workspace.dispatchRankPayloadCompletions_[peer] = 0;
      workspace.combineRankReadyFlags_[peer] = 0;
    }
    const int dispatchCounters = 3 + configCellDiv(context->distributeCapacity_, context->nTokensPerChunk_);
    for (int index = threadIdx.x; index < dispatchCounters; index += blockDim.x) {
      context->distributeScratch_[index] = 0;
    }
    const int combineCounters = configCellDiv(context->distributeCapacity_, context->nTokensPerChunk_);
    for (int index = threadIdx.x; index < combineCounters; index += blockDim.x) {
      context->aggregateChunkReady_[index] = 0;
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
    WorkspaceView workspace(context->workspace_, context->numRanks_, workload.numExperts_);
    const uint32_t flag = getDispatchFlag(workspace);
    if (blockIdx.x == 0) {
      if (threadIdx.x == 0) {
        auto* control = context->aggregateControl_;
        if (flag == 0 || control->lastCombineEpoch == flag) {
          printf(
              "[EP] aggregate Combine requires one fresh Dispatch per Combine, including graph replays "
              "(rank=%d epoch=%u last=%u)\n",
              context->rank_, flag, control->lastCombineEpoch);
          __trap();
        }
        control->lastCombineEpoch = flag;
      }
      // No posting warp may publish another round before validation succeeds.
      __syncthreads();
    }
    aggregateCombineBody<Hidden>(output, expertOutput, topkIndices, workload, context, flag);
  }
#endif
}

struct HybridAggregateCombineKernelSelector {
  template <int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout>
  static auto get() {
    return hybridAggregateCombineKernel<Hidden, DispatchType, ScaleBlockSize, Layout>;
  }
};

template <CombineMode Mode, int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout>
MSCCLPP_DEVICE_INLINE void hybridCombineBody(void* output, const void* expertOutput,
                                             const int64_t* __restrict__ topkIndices,
                                             const float* __restrict__ topkWeights, const int* srcInfo,
                                             const int64_t* layoutRange, Workload workload, void* combineRecvBuffer,
                                             const void* dispatchRecvBuffer, const DeviceContext* context,
                                             uint32_t flag) {
#if defined(MSCCLPP_USE_GPUNETIO) && MSCCLPP_BULK_AVAILABLE
  extern __shared__ __align__(128) uint8_t sharedMemory[];
  const int nTokens = workload.numTokens_;
  const int nExperts = workload.numExperts_;
  const int nRanks = context->numRanks_;
  const int nTopk = workload.numTopk_;
  const int maxTokensPerRank = workload.maxTokensPerRank_;
  const TransportView transport(context);
  WorkspaceView workspaceView(context->workspace_, nRanks, nExperts);

  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    static_assert(Mode == CombineMode::RANK_LOCAL_REDUCE || Mode == CombineMode::DIRECT_SEND);
    static_assert(DispatchType == DispatchDataType::BF16);
#if defined(MSCCLPP_USE_GPUNETIO)
    if (transport.portChannels_ != nullptr) {
      signalRankMajorCombineLocalStart(transport, nRanks);
      sendRankMajorCombinePush<Hidden>(expertOutput, nRanks, maxTokensPerRank, transport, workspaceView, flag);
      if (blockIdx.x == 0) {
        if (static_cast<int>(threadIdx.x) < WARP_SIZE) {
          publishRankMajorCombinePushReady(topkIndices, nTokens, nTopk, nExperts / nRanks, nRanks, flag, transport,
                                           workspaceView);
        }
        __syncthreads();
      }
      recvRankMajorRemotePartialsTma<Hidden, Mode, DispatchDataType::BF16, true>(
          output, expertOutput, topkIndices, nullptr, nTokens, nTopk, nExperts, nRanks, maxTokensPerRank, flag,
          transport, workspaceView, sharedMemory, 1.0f);
      drainRankMajorCombinePush(nRanks, transport, workspaceView, flag);
      workspaceView.combineSyncer_->sync(gridDim.x);
      if (blockIdx.x == 0) {
        const int peer = static_cast<int>(threadIdx.x);
        if (peer < nRanks && !transport.isSelf(peer)) {
          if (transport.isNvlinkPeer(peer)) {
            transport.baseMemoryChannels_[peer].relaxedSignal();
            transport.baseMemoryChannels_[peer].relaxedWait(-1);
          }
        }
      }
      return;
    }
#endif  // defined(MSCCLPP_USE_GPUNETIO)
  }
#endif  // defined(MSCCLPP_USE_GPUNETIO) && MSCCLPP_BULK_AVAILABLE
}

template <int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout>
__global__ __launch_bounds__(CombineNThreads,
                             1) void hybridCombineKernel(void* output, const void* expertOutput,
                                                         const int64_t* topkIndices, const float* topkWeights,
                                                         const int* srcInfo, const int64_t* layoutRange,
                                                         Workload workload, void* combineRecvBuffer,
                                                         const void* dispatchRecvBuffer, const DeviceContext* context) {
  WorkspaceView workspace(context->workspace_, context->numRanks_, workload.numExperts_);
  const uint32_t flag = getDispatchFlag(workspace);
  hybridCombineBody<CombineMode::RANK_LOCAL_REDUCE, Hidden, DispatchType, ScaleBlockSize, Layout>(
      output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload, combineRecvBuffer,
      dispatchRecvBuffer, context, flag);
}

struct HybridCombineKernelSelector {
  template <int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout>
  static auto get() {
    return hybridCombineKernel<Hidden, DispatchType, ScaleBlockSize, Layout>;
  }
};

void hybridCombine(void* output, const void* input, const int64_t* topkIdx, const Workload& workload, void* recvBuffer,
                   void* dispatchRecvBuffer, const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  EP_HOST_ASSERT(workload.outputLayout_ == DispatchLayout::RANK_MAJOR);
  if (context.aggregateCombine_) {
    EP_HOST_ASSERT(context.portChannels_ != nullptr && workload.numTopk_ <= MaxNumTopk);
    EP_HOST_ASSERT(workload.outputLayout_ == DispatchLayout::RANK_MAJOR &&
                   workload.dispatchDataType_ == DispatchDataType::BF16);
    return combineAlgorithm<CombineMode::RANK_LOCAL_REDUCE, HybridAggregateCombineKernelSelector, true>(
        output, input, topkIdx, nullptr, nullptr, nullptr, workload, recvBuffer, dispatchRecvBuffer, context, numBlocks,
        stream);
  }
  combineAlgorithm<CombineMode::RANK_LOCAL_REDUCE, HybridCombineKernelSelector, true>(
      output, input, topkIdx, nullptr, nullptr, nullptr, workload, recvBuffer, dispatchRecvBuffer, context, numBlocks,
      stream);
}

}  // namespace ep
}  // namespace mscclpp
