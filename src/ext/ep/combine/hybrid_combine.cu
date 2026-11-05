// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "combine/common.cuh"

namespace mscclpp {
namespace ep {

#if defined(MSCCLPP_USE_GPUNETIO) && MSCCLPP_BULK_AVAILABLE

template <CombineMode Mode, int Hidden, DispatchDataType DispatchType, bool Gateway>
MSCCLPP_DEVICE_INLINE void aggregateReduceRow(void* destination, const void* expertOutput, const int64_t* topkIndices,
                                              const float* topkWeights, int row, const Workload& workload,
                                              const DeviceContext* context, const TransportView& transport,
                                              const WorkspaceView& workspace, uint32_t flag, uint8_t* sharedMemory,
                                              uint32_t& phase, float dequantScale) {
  constexpr bool IsDirectSend = Mode == CombineMode::DIRECT_SEND;
  using ElementType = DispatchElementType<DispatchType>;
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(ElementType);
  constexpr int HiddenVectors = Hidden * sizeof(Bf16) / sizeof(int4);
  constexpr int PairsPerVector = sizeof(int4) / sizeof(mscclpp::bf16x2);
  constexpr size_t RowsBytes = MaxNumTopk * HiddenBytes;
  const int lane = getLaneId();
  const int topk = workload.numTopk_;
  const int localExperts = workload.numExperts_ / context->numRanks_;
  const int owner = Gateway ? context->gatewayRank_ : context->rank_;
  auto* barriers = reinterpret_cast<mscclpp::BulkBarrier*>(sharedMemory + RowsBytes);
  auto* valid = reinterpret_cast<int*>(barriers + MaxNumTopk);
  auto* weights = reinterpret_cast<float*>(valid + MaxNumTopk);
  if (threadIdx.x < WARP_SIZE) {
    int64_t expert = -1;
    int slot = -1;
    float weight = 1.0f;
    if (lane < topk) {
      if constexpr (Gateway) {
        const auto* cached = context->aggregateRoutes_ + static_cast<size_t>(row) * 3 * topk;
        expert = cached[lane];
        slot = cached[topk + lane];
        if (IsDirectSend && workload.applyRouterWeights_) {
          weight = reinterpret_cast<const float*>(cached + 2 * topk)[lane];
        }
      } else {
        expert = topkIndices[row * topk + lane];
        slot = workspace.rankMajorSendIndices_[row * topk + lane];
        if (IsDirectSend && workload.applyRouterWeights_ && topkWeights != nullptr) {
          weight = topkWeights[row * topk + lane];
        }
      }
    }
    const int peer = expert >= 0 && expert < workload.numExperts_ ? static_cast<int>(expert) / localExperts : -1;
    const bool leader = isFirstLaneForRank(peer, lane);
    const bool active = peer >= 0 && (IsDirectSend || leader) && transport.isNvlinkPeer(peer);
    if (lane < MaxNumTopk) {
      valid[lane] = active;
      if constexpr (IsDirectSend) weights[lane] = weight;
    }
    bool pending = active;
    while (__any_sync(0xffffffff, pending)) {
      if (pending && mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(workspace.combineRankReadyFlags_ + peer,
                                                                         mscclpp::memoryOrderAcquire) == flag) {
        EP_DEVICE_ASSERT(slot >= 0 && slot < workload.maxTokensPerRank_);
        const auto* peerOutput =
            static_cast<const uint8_t*>(transport.mappedBuffer(const_cast<void*>(expertOutput), peer));
        const size_t sourceRow = static_cast<size_t>(owner) * workload.maxTokensPerRank_ + slot;
        const size_t inputRow = IsDirectSend ? sourceRow * topk + lane : sourceRow;
        barriers[lane].arriveAndExpect(HiddenBytes);
        mscclpp::bulkLoad(sharedMemory + lane * HiddenBytes, peerOutput + inputRow * HiddenBytes, HiddenBytes,
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
      const float routeWeight = IsDirectSend ? weights[route] : 1.0f;
      const auto* routeRow = sharedMemory + static_cast<size_t>(route) * HiddenBytes;
      if constexpr (DispatchType == DispatchDataType::BF16) {
        const int4 packed = reinterpret_cast<const int4*>(routeRow)[vector];
        const auto* pairs = reinterpret_cast<const mscclpp::bf16x2*>(&packed);
#pragma unroll
        for (int pair = 0; pair < PairsPerVector; ++pair) {
          const auto value = mscclpp::to<mscclpp::f32x2>(pairs[pair]);
          if constexpr (IsDirectSend) {
            sum[pair].data[0] = fmaf(value.data[0], routeWeight, sum[pair].data[0]);
            sum[pair].data[1] = fmaf(value.data[1], routeWeight, sum[pair].data[1]);
          } else {
            sum[pair].data[0] += value.data[0];
            sum[pair].data[1] += value.data[1];
          }
        }
      } else {
        const auto values = reinterpret_cast<const mscclpp::f8_e4m3x8*>(routeRow)[vector];
        const float scale = IsDirectSend ? dequantScale * routeWeight : dequantScale;
        if (scale != 0.0f) {
          dequantizeAndAccumulateFp8E4M3(values, scale, sum);
        }
      }
    }
    if constexpr (Gateway && DispatchType == DispatchDataType::FP8_E4M3_STATIC) {
      int4 packedOut;
      auto* outPairs = reinterpret_cast<mscclpp::bf16x2*>(&packedOut);
#pragma unroll
      for (int pair = 0; pair < PairsPerVector; ++pair) {
        outPairs[pair] = mscclpp::to<mscclpp::bf16x2>(sum[pair]);
      }
      reinterpret_cast<int4*>(destination)[vector] = packedOut;
    } else {
#pragma unroll
      for (int pair = 0; pair < PairsPerVector; ++pair) {
        reinterpret_cast<mscclpp::f32x2*>(destination)[vector * PairsPerVector + pair] = sum[pair];
      }
    }
  }
  // The CTA barrier joins every writer into thread 0's release counter.
  __syncthreads();
  if (Gateway && threadIdx.x == 0) {
    (void)mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(
        context->aggregateChunkReady_ + row / context->nTokensPerChunk_, 1, mscclpp::memoryOrderRelease);
  }
}

template <int Hidden, DispatchDataType DispatchType>
MSCCLPP_DEVICE_INLINE void aggregateControl(const DeviceContext* context, const TransportView& transport,
                                            const WorkspaceView& workspace, uint32_t flag) {
  const int peer = threadIdx.x;
  const int nRanks = context->numRanks_;
  auto* network = transport.portChannels_;
  const int queue = static_cast<int>(threadIdx.x) / WARP_SIZE - configCellDiv(nRanks, WARP_SIZE);
  __shared__ mscclpp::GpuNetIoWqeSlot wqeSlots[MaxQpsPerPeer];
  constexpr auto kExclusive = mscclpp::PortChannelSharingMode::Exclusive;
  if (peer < nRanks && transport.isNvlinkPeer(peer)) {
    if (!transport.isSelf(peer)) {
      transport.baseMemoryChannels_[peer].signal();
      transport.baseMemoryChannels_[peer].wait(-1);
    }
    mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(workspace.combineRankReadyFlags_ + peer, flag,
                                                         mscclpp::memoryOrderRelease);
  }
  if (queue < 0 || queue >= network->numQpsPerPeer || getLaneId() != 0) return;
  const int rows = *context->aggregateIncomingRows_;
  const int nTokensPerChunk = context->nTokensPerChunk_;
  using WireElement = std::conditional_t<DispatchType == DispatchDataType::FP8_E4M3_STATIC, Bf16, float>;
  constexpr size_t RowBytes = static_cast<size_t>(Hidden) * sizeof(WireElement);
  const int gateway = context->gatewayRank_;
  auto handle = network->channel(gateway, queue);
  handle.setWqeSlot(&wqeSlots[queue]);
  const uint64_t dstOffset = transport.symmetricOffset(context->aggregateRecvBuffer_);
  const uint64_t srcOffset = transport.symmetricOffset(context->aggregateSendBuffer_);
  const uint64_t flagOffset = transport.symmetricOffset(context->aggregateControl_->chunks + queue);
  for (int chunk = queue, begin = queue * nTokensPerChunk; begin < rows;
       chunk += network->numQpsPerPeer, begin += network->numQpsPerPeer * nTokensPerChunk) {
    const int count = rows - begin < nTokensPerChunk ? rows - begin : nTokensPerChunk;
    handle.device_.putWithAccumulateAfter<kExclusive>(
        handle.dstMemory_, dstOffset + begin * RowBytes, handle.srcMemory_, srcOffset + begin * RowBytes,
        count * RowBytes, handle.dstMemory_, flagOffset, 1, [&] {
          while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(context->aggregateChunkReady_ + chunk,
                                                                mscclpp::memoryOrderAcquire) != count) {
          }
        });
  }
}

template <int Hidden, DispatchDataType DispatchType>
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
    int4 remotePacked{};
    if constexpr (DispatchType == DispatchDataType::FP8_E4M3_STATIC) {
      if (row >= 0) {
        remotePacked = reinterpret_cast<const int4*>(reinterpret_cast<const Bf16*>(context->aggregateRecvBuffer_) +
                                                     static_cast<size_t>(row) * Hidden)[vector];
      }
    }
#pragma unroll
    for (int pair = 0; pair < PairsPerVector; ++pair) {
      auto value = local[vector * PairsPerVector + pair];
      if (row >= 0) {
        if constexpr (DispatchType == DispatchDataType::FP8_E4M3_STATIC) {
          const auto partial =
              mscclpp::to<mscclpp::f32x2>(reinterpret_cast<const mscclpp::bf16x2*>(&remotePacked)[pair]);
          value.data[0] += partial.data[0];
          value.data[1] += partial.data[1];
        } else {
          const auto* remote = reinterpret_cast<const mscclpp::f32x2*>(context->aggregateRecvBuffer_ +
                                                                       static_cast<size_t>(row) * Hidden);
          const auto partial = remote[vector * PairsPerVector + pair];
          value.data[0] += partial.data[0];
          value.data[1] += partial.data[1];
        }
      }
      pairs[pair] = mscclpp::to<mscclpp::bf16x2>(value);
    }
    static_cast<int4*>(output)[static_cast<size_t>(token) * HiddenVectors + vector] = packed;
  }
}

template <CombineMode Mode, int Hidden, DispatchDataType DispatchType>
MSCCLPP_DEVICE_INLINE void combineBody(void* output, const void* expertOutput, const int64_t* topkIndices,
                                       const float* topkWeights, Workload workload, const DeviceContext* context,
                                       uint32_t flag) {
  extern __shared__ __align__(128) uint8_t sharedMemory[];
  const TransportView transport(context);
  const WorkspaceView workspace(context->workspace_, context->numRanks_, workload.numExperts_);
  if (blockIdx.x == 0) {
    aggregateControl<Hidden, DispatchType>(context, transport, workspace, flag);
  } else {
    using ElementType = DispatchElementType<DispatchType>;
    using WireElement = std::conditional_t<DispatchType == DispatchDataType::FP8_E4M3_STATIC, Bf16, float>;
    constexpr size_t RowsBytes = static_cast<size_t>(MaxNumTopk) * Hidden * sizeof(ElementType);
    auto* barriers = reinterpret_cast<mscclpp::BulkBarrier*>(sharedMemory + RowsBytes);
    if (threadIdx.x < MaxNumTopk) barriers[threadIdx.x].init();
    __syncthreads();
    uint32_t phase = 0;
    const int worker = static_cast<int>(blockIdx.x) - 1;
    const int workers = static_cast<int>(gridDim.x) - 1;
    const int incoming = *context->aggregateIncomingRows_;
    const int nTokensPerChunk = context->nTokensPerChunk_;
    const int qps = transport.portChannels_->numQpsPerPeer;
    const float dequantScale =
        DispatchType == DispatchDataType::FP8_E4M3_STATIC ? 1.0f / workload.dispatchScale_ : 1.0f;
    auto* sendBase = reinterpret_cast<WireElement*>(context->aggregateSendBuffer_);
    for (int row = worker; row < incoming; row += workers) {
      if (threadIdx.x == 0 && row >= qps * nTokensPerChunk) {
        const int previous = row / nTokensPerChunk - qps;
        while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(context->aggregateChunkReady_ + previous,
                                                              mscclpp::memoryOrderAcquire) != nTokensPerChunk) {
        }
      }
      __syncthreads();
      aggregateReduceRow<Mode, Hidden, DispatchType, true>(sendBase + static_cast<size_t>(row) * Hidden, expertOutput,
                                                           topkIndices, topkWeights, row, workload, context, transport,
                                                           workspace, flag, sharedMemory, phase, dequantScale);
    }
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
      aggregateReduceRow<Mode, Hidden, DispatchType, false>(
          context->aggregateLocalBuffer_ + static_cast<size_t>(row) * Hidden, expertOutput, topkIndices, topkWeights,
          row, workload, context, transport, workspace, flag, sharedMemory, phase, dequantScale);
    }
    for (int token = worker; token < workload.numTokens_; token += workers) {
      aggregateFinalize<Hidden, DispatchType>(output, token, context);
    }
  }
  workspace.combineSyncer_->sync(gridDim.x);
  if (blockIdx.x == 0) {
    const int chunks = configCellDiv(context->distributeScratch_[0], context->nTokensPerChunk_);
    const int qps = transport.portChannels_->numQpsPerPeer;
    for (int queue = threadIdx.x; queue < qps; queue += blockDim.x) {
      context->aggregateControl_->chunkBase[queue] += (chunks + qps - 1 - queue) / qps;
    }
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

template <CombineMode Mode, int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout>
__global__ __launch_bounds__(CombineNThreads,
                             1) void hybridCombineKernel(void* output, const void* expertOutput,
                                                         const int64_t* topkIndices, const float* topkWeights,
                                                         const int* srcInfo, const int64_t* layoutRange,
                                                         Workload workload, void* combineRecvBuffer,
                                                         const void* dispatchRecvBuffer, const DeviceContext* context) {
#if defined(MSCCLPP_USE_GPUNETIO) && MSCCLPP_BULK_AVAILABLE
  if constexpr (Layout == DispatchLayout::RANK_MAJOR &&
                (DispatchType == DispatchDataType::BF16 || DispatchType == DispatchDataType::FP8_E4M3_STATIC)) {
    WorkspaceView workspace(context->workspace_, context->numRanks_, workload.numExperts_);
    const uint32_t flag = getDispatchFlag(workspace);
    if (blockIdx.x == 0) {
      if (threadIdx.x == 0) {
        auto* control = context->aggregateControl_;
        if (flag == 0 || control->lastCombineEpoch == flag) {
          printf(
              "[EP] hybrid Combine requires one fresh Dispatch per Combine, including graph replays "
              "(rank=%d epoch=%u last=%u)\n",
              context->rank_, flag, control->lastCombineEpoch);
          __trap();
        }
        control->lastCombineEpoch = flag;
      }
      __syncthreads();
    }
    combineBody<Mode, Hidden, DispatchType>(output, expertOutput, topkIndices, topkWeights, workload, context, flag);
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
  EP_HOST_ASSERT(context.portChannels_ != nullptr && workload.numTopk_ <= MaxNumTopk);
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

}  // namespace ep
}  // namespace mscclpp
