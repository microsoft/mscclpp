// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "combine/common.cuh"

namespace mscclpp {
namespace ep {

#if defined(MSCCLPP_DEVICE_CUDA) && defined(USE_IBVERBS) && MSCCLPP_BULK_AVAILABLE

template <int Hidden, DispatchDataType DispatchType>
MSCCLPP_DEVICE_INLINE void aggregateFinalize(void* output, int tokenIdx, const DeviceContext* context) {
  constexpr int NHiddenVectors = Hidden * sizeof(Bf16) / sizeof(int4);
  constexpr int NPairsPerVector = sizeof(int4) / sizeof(mscclpp::bf16x2);
  const auto& hybridCtx = context->hybridCtx_;
  const int remoteTokenIdx = hybridCtx.aggregateRemoteTokenIndices_[tokenIdx];
  if (threadIdx.x == 0 && remoteTokenIdx >= 0) {
    const int chunkIdx = remoteTokenIdx / hybridCtx.nTokensPerChunk_;
    const int nQps = hybridCtx.portChannels_->numQpsPerPeer;
    const int queueIdx = chunkIdx % nQps;
    auto* control = hybridCtx.aggregateControl_;
    const uint64_t target = chunkIdx / nQps + 1;
    while (mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(control->qpSignalCount + queueIdx,
                                                               mscclpp::memoryOrderAcquire) < target) {
    }
  }
  __syncthreads();
  const auto* local =
      reinterpret_cast<const mscclpp::f32x2*>(hybridCtx.aggregateLocalBuffer_ + static_cast<size_t>(tokenIdx) * Hidden);
  for (int vectorIdx = threadIdx.x; vectorIdx < NHiddenVectors; vectorIdx += blockDim.x) {
    int4 packed;
    auto* pairs = reinterpret_cast<mscclpp::bf16x2*>(&packed);
    int4 remotePacked{};
    if constexpr (DispatchType == DispatchDataType::FP8_E4M3_STATIC) {
      if (remoteTokenIdx >= 0) {
        remotePacked = reinterpret_cast<const int4*>(reinterpret_cast<const Bf16*>(hybridCtx.aggregateRecvBuffer_) +
                                                     static_cast<size_t>(remoteTokenIdx) * Hidden)[vectorIdx];
      }
    }
#pragma unroll
    for (int pairIdx = 0; pairIdx < NPairsPerVector; ++pairIdx) {
      auto value = local[vectorIdx * NPairsPerVector + pairIdx];
      if (remoteTokenIdx >= 0) {
        if constexpr (DispatchType == DispatchDataType::FP8_E4M3_STATIC) {
          const auto partial =
              mscclpp::to<mscclpp::f32x2>(reinterpret_cast<const mscclpp::bf16x2*>(&remotePacked)[pairIdx]);
          value = value + partial;
        } else {
          const auto* remote = reinterpret_cast<const mscclpp::f32x2*>(hybridCtx.aggregateRecvBuffer_ +
                                                                       static_cast<size_t>(remoteTokenIdx) * Hidden);
          value = value + remote[vectorIdx * NPairsPerVector + pairIdx];
        }
      }
      pairs[pairIdx] = mscclpp::to<mscclpp::bf16x2>(value);
    }
    static_cast<int4*>(output)[static_cast<size_t>(tokenIdx) * NHiddenVectors + vectorIdx] = packed;
  }
}

template <CombineMode Mode, int Hidden, DispatchDataType DispatchType, bool Gateway, bool FuseFinal = false>
MSCCLPP_DEVICE_INLINE void aggregateReduceToken(void* destination, const void* expertOutput, const int64_t* topkIndices,
                                                const float* topkWeights, int tokenIdx, const Workload& workload,
                                                const DeviceContext* context, const TransportView& transport,
                                                const WorkspaceView& workspace, uint32_t flag, uint8_t* sharedMemory,
                                                uint32_t& phase, float dequantScale, int localWorkerIdx = 0,
                                                int nLocalWorkers = 1) {
  static_assert(!(Gateway && FuseFinal));
  constexpr bool IsDirectSend = Mode == CombineMode::DIRECT_SEND;
  using ElementType = DispatchElementType<DispatchType>;
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(ElementType);
  constexpr int NHiddenVectors = Hidden * sizeof(Bf16) / sizeof(int4);
  constexpr int NPairsPerVector = sizeof(int4) / sizeof(mscclpp::bf16x2);
  constexpr int MaxVectorsPerThread = configCellDiv(NHiddenVectors, CombineNThreads);
  constexpr size_t TokensBytes = MaxNumTopk * HiddenBytes;
  const auto& hybridCtx = context->hybridCtx_;
  const int nTokensPerChunk = hybridCtx.nTokensPerChunk_;
  const int laneIdx = getLaneId();
  const int nTopk = workload.numTopk_;
  const int nLocalExperts = workload.numExperts_ / context->numRanks_;
  const int ownerRank = Gateway ? hybridCtx.gatewayRank_ : context->rank_;
  auto& sharedControl = *reinterpret_cast<RankMajorCombineSharedControl*>(sharedMemory + TokensBytes);
  auto* barriers = sharedControl.barriers;
  auto* validRoutes = sharedControl.validRoutes;
  auto* routeWeights = sharedControl.routeWeights;
  if (threadIdx.x < WARP_SIZE) {
    int64_t expertIdx = -1;
    int slotIdx = -1;
    float weight = 1.0f;
    if (laneIdx < nTopk) {
      if constexpr (Gateway) {
        const auto* cached = hybridCtx.aggregateRoutes_ + static_cast<size_t>(tokenIdx) * 3 * nTopk;
        expertIdx = cached[laneIdx];
        slotIdx = cached[nTopk + laneIdx];
        if (IsDirectSend && workload.applyRouterWeights_) {
          weight = reinterpret_cast<const float*>(cached + 2 * nTopk)[laneIdx];
        }
      } else {
        expertIdx = topkIndices[tokenIdx * nTopk + laneIdx];
        slotIdx = workspace.rankMajorSendIndices_[tokenIdx * nTopk + laneIdx];
        if (IsDirectSend && workload.applyRouterWeights_ && topkWeights != nullptr) {
          weight = topkWeights[tokenIdx * nTopk + laneIdx];
        }
      }
    }
    const int peerIdx =
        expertIdx >= 0 && expertIdx < workload.numExperts_ ? static_cast<int>(expertIdx) / nLocalExperts : -1;
    const bool leader = isFirstLaneForRank(peerIdx, laneIdx);
    const bool active = peerIdx >= 0 && (IsDirectSend || leader) && transport.isNvlinkPeer(peerIdx);
    if (laneIdx < MaxNumTopk) {
      validRoutes[laneIdx] = active;
      if constexpr (IsDirectSend) routeWeights[laneIdx] = weight;
    }
    bool pending = active;
    while (__any_sync(0xffffffff, pending)) {
      if (pending && mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(workspace.combineRankReadyFlags_ + peerIdx,
                                                                         mscclpp::memoryOrderAcquire) == flag) {
        EP_DEVICE_ASSERT(slotIdx >= 0 && slotIdx < workload.maxTokensPerRank_);
        const auto* peerOutput =
            static_cast<const uint8_t*>(transport.mappedBuffer(const_cast<void*>(expertOutput), peerIdx));
        const size_t sourceTokenIdx = static_cast<size_t>(ownerRank) * workload.maxTokensPerRank_ + slotIdx;
        const size_t inputTokenIdx = IsDirectSend ? sourceTokenIdx * nTopk + laneIdx : sourceTokenIdx;
        barriers[laneIdx].arriveAndExpect(HiddenBytes);
        mscclpp::bulkLoad(sharedMemory + laneIdx * HiddenBytes, peerOutput + inputTokenIdx * HiddenBytes, HiddenBytes,
                          barriers[laneIdx]);
        pending = false;
      }
    }
    if (active) barriers[laneIdx].wait(phase);
    __syncwarp();
    if (laneIdx == 0) {
      mscclpp::bulkFence();
      if constexpr (Gateway) {
        atomicAddReleaseDevice(hybridCtx.aggregateChunkLoadDone_ + tokenIdx / nTokensPerChunk, 1);
      }
    }
  }
  __syncthreads();
  if constexpr (FuseFinal) {
    mscclpp::f32x2 tileSum[MaxVectorsPerThread][NPairsPerVector];
#pragma unroll
    for (int stepIdx = 0; stepIdx < MaxVectorsPerThread; ++stepIdx) {
      const int vectorIdx = static_cast<int>(threadIdx.x) + stepIdx * CombineNThreads;
      if (vectorIdx < NHiddenVectors) {
#pragma unroll
        for (int pairIdx = 0; pairIdx < NPairsPerVector; ++pairIdx) {
          tileSum[stepIdx][pairIdx].storage = make_float2(0.0f, 0.0f);
        }
#pragma unroll
        for (int routeIdx = 0; routeIdx < MaxNumTopk; ++routeIdx) {
          if (!validRoutes[routeIdx]) continue;
          const float routeWeight = IsDirectSend ? routeWeights[routeIdx] : 1.0f;
          const auto* routeToken = sharedMemory + static_cast<size_t>(routeIdx) * HiddenBytes;
          if constexpr (DispatchType == DispatchDataType::BF16) {
            const int4 packed = reinterpret_cast<const int4*>(routeToken)[vectorIdx];
            const auto* pairs = reinterpret_cast<const mscclpp::bf16x2*>(&packed);
#pragma unroll
            for (int pairIdx = 0; pairIdx < NPairsPerVector; ++pairIdx) {
              const auto value = mscclpp::to<mscclpp::f32x2>(pairs[pairIdx]);
              if constexpr (IsDirectSend) {
                tileSum[stepIdx][pairIdx] = mscclpp::fma(value, mscclpp::f32x2(make_float2(routeWeight, routeWeight)),
                                                         tileSum[stepIdx][pairIdx]);
              } else {
                tileSum[stepIdx][pairIdx] = tileSum[stepIdx][pairIdx] + value;
              }
            }
          } else {
            const auto values = reinterpret_cast<const mscclpp::f8_e4m3x8*>(routeToken)[vectorIdx];
            const float scale = IsDirectSend ? dequantScale * routeWeight : dequantScale;
            if (scale != 0.0f) {
              dequantizeAndAccumulateFp8E4M3(values, scale, tileSum[stepIdx]);
            }
          }
        }
      }
    }
    for (int prevIdx = localWorkerIdx; prevIdx < tokenIdx; prevIdx += nLocalWorkers) {
      aggregateFinalize<Hidden, DispatchType>(destination, prevIdx, context);
    }
    const int remoteTokenIdx = hybridCtx.aggregateRemoteTokenIndices_[tokenIdx];
    if (remoteTokenIdx >= 0) {
      if (threadIdx.x == 0) {
        const int chunkIdx = remoteTokenIdx / nTokensPerChunk;
        const int nQps = hybridCtx.portChannels_->numQpsPerPeer;
        const int queueIdx = chunkIdx % nQps;
        const uint64_t target = chunkIdx / nQps + 1;
        while (mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(
                   hybridCtx.aggregateControl_->qpSignalCount + queueIdx, mscclpp::memoryOrderAcquire) < target) {
        }
      }
      __syncthreads();
    }
#pragma unroll
    for (int stepIdx = 0; stepIdx < MaxVectorsPerThread; ++stepIdx) {
      const int vectorIdx = static_cast<int>(threadIdx.x) + stepIdx * CombineNThreads;
      if (vectorIdx < NHiddenVectors) {
        int4 packedOut;
        auto* outPairs = reinterpret_cast<mscclpp::bf16x2*>(&packedOut);
        int4 remotePacked{};
        if constexpr (DispatchType == DispatchDataType::FP8_E4M3_STATIC) {
          if (remoteTokenIdx >= 0) {
            remotePacked = reinterpret_cast<const int4*>(reinterpret_cast<const Bf16*>(hybridCtx.aggregateRecvBuffer_) +
                                                         static_cast<size_t>(remoteTokenIdx) * Hidden)[vectorIdx];
          }
        }
#pragma unroll
        for (int pairIdx = 0; pairIdx < NPairsPerVector; ++pairIdx) {
          auto value = tileSum[stepIdx][pairIdx];
          if (remoteTokenIdx >= 0) {
            if constexpr (DispatchType == DispatchDataType::FP8_E4M3_STATIC) {
              value =
                  value + mscclpp::to<mscclpp::f32x2>(reinterpret_cast<const mscclpp::bf16x2*>(&remotePacked)[pairIdx]);
            } else {
              const auto* remote = reinterpret_cast<const mscclpp::f32x2*>(
                  hybridCtx.aggregateRecvBuffer_ + static_cast<size_t>(remoteTokenIdx) * Hidden);
              value = value + remote[vectorIdx * NPairsPerVector + pairIdx];
            }
          }
          outPairs[pairIdx] = mscclpp::to<mscclpp::bf16x2>(value);
        }
        static_cast<int4*>(destination)[static_cast<size_t>(tokenIdx) * NHiddenVectors + vectorIdx] = packedOut;
      }
    }
  } else {
    for (int vectorIdx = threadIdx.x; vectorIdx < NHiddenVectors; vectorIdx += blockDim.x) {
      mscclpp::f32x2 sum[NPairsPerVector];
#pragma unroll
      for (int pairIdx = 0; pairIdx < NPairsPerVector; ++pairIdx) {
        sum[pairIdx].storage = make_float2(0.0f, 0.0f);
      }
#pragma unroll
      for (int routeIdx = 0; routeIdx < MaxNumTopk; ++routeIdx) {
        if (!validRoutes[routeIdx]) continue;
        const float routeWeight = IsDirectSend ? routeWeights[routeIdx] : 1.0f;
        const auto* routeToken = sharedMemory + static_cast<size_t>(routeIdx) * HiddenBytes;
        if constexpr (DispatchType == DispatchDataType::BF16) {
          const int4 packed = reinterpret_cast<const int4*>(routeToken)[vectorIdx];
          const auto* pairs = reinterpret_cast<const mscclpp::bf16x2*>(&packed);
#pragma unroll
          for (int pairIdx = 0; pairIdx < NPairsPerVector; ++pairIdx) {
            const auto value = mscclpp::to<mscclpp::f32x2>(pairs[pairIdx]);
            if constexpr (IsDirectSend) {
              sum[pairIdx] = mscclpp::fma(value, mscclpp::f32x2(make_float2(routeWeight, routeWeight)), sum[pairIdx]);
            } else {
              sum[pairIdx] = sum[pairIdx] + value;
            }
          }
        } else {
          const auto values = reinterpret_cast<const mscclpp::f8_e4m3x8*>(routeToken)[vectorIdx];
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
        for (int pairIdx = 0; pairIdx < NPairsPerVector; ++pairIdx) {
          outPairs[pairIdx] = mscclpp::to<mscclpp::bf16x2>(sum[pairIdx]);
        }
        reinterpret_cast<int4*>(destination)[vectorIdx] = packedOut;
      } else {
#pragma unroll
        for (int pairIdx = 0; pairIdx < NPairsPerVector; ++pairIdx) {
          reinterpret_cast<mscclpp::f32x2*>(destination)[vectorIdx * NPairsPerVector + pairIdx] = sum[pairIdx];
        }
      }
    }
    // The CTA barrier joins every writer into thread 0's release counter.
    __syncthreads();
    if (Gateway && threadIdx.x == 0) {
      atomicAddReleaseDevice(hybridCtx.aggregateChunkReady_ + tokenIdx / nTokensPerChunk, 1);
    }
  }
}

template <int Hidden, DispatchDataType DispatchType>
MSCCLPP_DEVICE_INLINE void aggregateControl(const DeviceContext* context, const TransportView& transport,
                                            const WorkspaceView& workspace, uint32_t flag) {
  const int peerIdx = threadIdx.x;
  const int nRanks = context->numRanks_;
  const auto& hybridCtx = context->hybridCtx_;
  auto* network = transport.portChannels_;
  const int queueIdx = static_cast<int>(threadIdx.x) / WARP_SIZE - configCellDiv(nRanks, WARP_SIZE);
  constexpr auto kExclusive = mscclpp::PortChannelDeviceHandle::Concurrency::Exclusive;
  if (peerIdx < nRanks && transport.isNvlinkPeer(peerIdx)) {
    if (!transport.isSelf(peerIdx)) {
      transport.baseMemoryChannels_[peerIdx].signal();
      transport.baseMemoryChannels_[peerIdx].wait(-1);
    }
    mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(workspace.combineRankReadyFlags_ + peerIdx, flag,
                                                         mscclpp::memoryOrderRelease);
  }
  if (queueIdx >= 0 && queueIdx < network->numQpsPerPeer && getLaneId() == 0) {
    const int nTokens = *hybridCtx.nIncomingTokens_;
    const int nTokensPerChunk = hybridCtx.nTokensPerChunk_;
    using WireElement = std::conditional_t<DispatchType == DispatchDataType::FP8_E4M3_STATIC, Bf16, float>;
    constexpr size_t TokenBytes = static_cast<size_t>(Hidden) * sizeof(WireElement);
    const int gatewayRank = hybridCtx.gatewayRank_;
    auto handle = network->channel(gatewayRank, queueIdx);
    const uint64_t dstOffset = transport.symmetricOffset(hybridCtx.aggregateRecvBuffer_);
    const uint64_t srcOffset = transport.symmetricOffset(hybridCtx.aggregateSendBuffer_);
    const uint64_t flagOffset = transport.symmetricOffset(hybridCtx.aggregateControl_->qpSignalCount + queueIdx);
    for (int chunkIdx = queueIdx, tokenBegin = queueIdx * nTokensPerChunk; tokenBegin < nTokens;
         chunkIdx += network->numQpsPerPeer, tokenBegin += network->numQpsPerPeer * nTokensPerChunk) {
      const int nChunkTokens = nTokens - tokenBegin < nTokensPerChunk ? nTokens - tokenBegin : nTokensPerChunk;
      handle.device_.putWithAccumulateAfter<kExclusive, MaxQpsPerPeer>(
          handle.dstMemory_, dstOffset + tokenBegin * TokenBytes, handle.srcMemory_,
          srcOffset + tokenBegin * TokenBytes, nChunkTokens * TokenBytes, handle.dstMemory_, flagOffset, 1,
          [&] {
            while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(hybridCtx.aggregateChunkReady_ + chunkIdx,
                                                                  mscclpp::memoryOrderAcquire) != nChunkTokens) {
            }
          },
          queueIdx);
    }
  }
  for (int idx = threadIdx.x; idx < nRanks; idx += blockDim.x) {
    workspace.dispatchRankPayloadSlots_[idx] = 0;
    workspace.dispatchRankPayloadCompletions_[idx] = 0;
  }
  const int nDispatchChunks = configCellDiv(hybridCtx.distributeCapacity_, hybridCtx.nTokensPerChunk_);
  for (int chunkIdx = threadIdx.x; chunkIdx < nDispatchChunks; chunkIdx += blockDim.x) {
    hybridCtx.distributeScratch_->chunkReadySignals[chunkIdx] = 0;
  }
  if (threadIdx.x == 0) {
    hybridCtx.distributeScratch_->nRemoteTokens = 0;
    hybridCtx.distributeScratch_->nCompletedForwarders = 0;
    hybridCtx.distributeScratch_->forwardComplete = 0;
    auto* state = hybridCtx.distributeControl_;
    while (mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(&state->bufferReuseReady, mscclpp::memoryOrderAcquire) ==
           0) {
    }
    state->bufferReuseReady = 0;
  }
}

template <CombineMode Mode, int Hidden, DispatchDataType DispatchType>
MSCCLPP_DEVICE_INLINE void combineBody(void* output, const void* expertOutput, const int64_t* topkIndices,
                                       const float* topkWeights, Workload workload, const DeviceContext* context,
                                       uint32_t flag) {
  extern __shared__ __align__(128) uint8_t sharedMemory[];
  const TransportView transport(context);
  const WorkspaceView workspace(context->workspace_, context->numRanks_, workload.numExperts_);
  using ElementType = DispatchElementType<DispatchType>;
  using WireElement = std::conditional_t<DispatchType == DispatchDataType::FP8_E4M3_STATIC, Bf16, float>;
  constexpr size_t TokensBytes = static_cast<size_t>(MaxNumTopk) * Hidden * sizeof(ElementType);
  const auto& hybridCtx = context->hybridCtx_;
  auto& sharedControl = *reinterpret_cast<RankMajorCombineSharedControl*>(sharedMemory + TokensBytes);
  const int nQps = transport.portChannels_->numQpsPerPeer;
  const float dequantScale = DispatchType == DispatchDataType::FP8_E4M3_STATIC ? 1.0f / workload.combineScale_ : 1.0f;
  uint32_t phase = 0;

  if (blockIdx.x == 0) {
    aggregateControl<Hidden, DispatchType>(context, transport, workspace, flag);
    __syncthreads();
    if (threadIdx.x < MaxNumTopk) sharedControl.barriers[threadIdx.x].init();
    __syncthreads();
  } else {
    if (threadIdx.x < MaxNumTopk) sharedControl.barriers[threadIdx.x].init();
    __syncthreads();
    const int workerIdx = static_cast<int>(blockIdx.x) - 1;
    const int nGatewayWorkers = static_cast<int>(gridDim.x) - 1;
    const int nIncomingTokens = *hybridCtx.nIncomingTokens_;
    const int nTokensPerChunk = hybridCtx.nTokensPerChunk_;
    const int loadWaveTokens = 16 * static_cast<int>(sizeof(Bf16) / sizeof(ElementType));
    const int loadWaveChunks = max(1, loadWaveTokens / nTokensPerChunk);
    const int readyWaveChunks = max(nQps, configCellDiv(64, nTokensPerChunk));
    auto* sendBase = reinterpret_cast<WireElement*>(hybridCtx.aggregateSendBuffer_);
    for (int tokenIdx = workerIdx; tokenIdx < nIncomingTokens; tokenIdx += nGatewayWorkers) {
      const int chunkIdx = tokenIdx / nTokensPerChunk;
      if (threadIdx.x == 0) {
        // Pace gateway NVLink TMA loads in constant-byte waves (~1.15 MiB) without starving small chunks.
        if (chunkIdx >= loadWaveChunks) {
          while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(
                     hybridCtx.aggregateChunkLoadDone_ + (chunkIdx - loadWaveChunks), mscclpp::memoryOrderAcquire) !=
                 nTokensPerChunk) {
          }
        }
        if (chunkIdx >= readyWaveChunks) {
          while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(
                     hybridCtx.aggregateChunkReady_ + (chunkIdx - readyWaveChunks), mscclpp::memoryOrderAcquire) !=
                 nTokensPerChunk) {
          }
        }
      }
      __syncthreads();
      aggregateReduceToken<Mode, Hidden, DispatchType, true>(
          sendBase + static_cast<size_t>(tokenIdx) * Hidden, expertOutput, topkIndices, topkWeights, tokenIdx, workload,
          context, transport, workspace, flag, sharedMemory, phase, dequantScale);
    }
    if (threadIdx.x == 0) {
      for (int tokenBegin = 0; tokenBegin < nIncomingTokens; tokenBegin += nTokensPerChunk) {
        const int nChunkTokens = min(nTokensPerChunk, nIncomingTokens - tokenBegin);
        while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(
                   hybridCtx.aggregateChunkLoadDone_ + tokenBegin / nTokensPerChunk, mscclpp::memoryOrderAcquire) !=
               nChunkTokens) {
        }
      }
    }
    __syncthreads();
  }

  const int nLocalWorkers = static_cast<int>(gridDim.x);
  const int localWorkerIdx = blockIdx.x == 0 ? nLocalWorkers - 1 : static_cast<int>(blockIdx.x) - 1;
  for (int tokenIdx = localWorkerIdx; tokenIdx < workload.numTokens_; tokenIdx += nLocalWorkers) {
    if (tokenIdx + nLocalWorkers < workload.numTokens_) {
      aggregateReduceToken<Mode, Hidden, DispatchType, false, false>(
          hybridCtx.aggregateLocalBuffer_ + static_cast<size_t>(tokenIdx) * Hidden, expertOutput, topkIndices,
          topkWeights, tokenIdx, workload, context, transport, workspace, flag, sharedMemory, phase, dequantScale);
    } else {
      aggregateReduceToken<Mode, Hidden, DispatchType, false, true>(
          output, expertOutput, topkIndices, topkWeights, tokenIdx, workload, context, transport, workspace, flag,
          sharedMemory, phase, dequantScale, localWorkerIdx, nLocalWorkers);
    }
  }
  __syncthreads();
  if (threadIdx.x < MaxNumTopk) sharedControl.barriers[threadIdx.x].invalidate();
}

#endif

template <CombineMode Mode, int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout>
__global__ __launch_bounds__(CombineNThreads,
                             1) void hybridCombineKernel(void* output, const void* expertOutput,
                                                         const int64_t* topkIndices, const float* topkWeights,
                                                         const int* srcInfo, const int64_t* layoutRange,
                                                         Workload workload, void* combineRecvBuffer,
                                                         const void* dispatchRecvBuffer, const DeviceContext* context) {
#if defined(MSCCLPP_DEVICE_CUDA) && defined(USE_IBVERBS) && MSCCLPP_BULK_AVAILABLE
  if constexpr (Layout == DispatchLayout::RANK_MAJOR &&
                (DispatchType == DispatchDataType::BF16 || DispatchType == DispatchDataType::FP8_E4M3_STATIC)) {
    WorkspaceView workspace(context->workspace_, context->numRanks_, workload.numExperts_);
    const uint32_t flag = getDispatchFlag(workspace);
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
  EP_HOST_ASSERT(context.hybridCtx_.portChannels_ != nullptr && workload.numTopk_ <= MaxNumTopk);
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
