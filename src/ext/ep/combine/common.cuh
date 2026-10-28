// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_COMBINE_COMMON_CUH_
#define MSCCLPP_EP_COMBINE_COMMON_CUH_

#include <mscclpp/bulk_device.hpp>
#include <mscclpp/gpu_data_types.hpp>

#include "common/device_helpers.cuh"
#include "common/latency.cuh"
#include "common/quantization.cuh"
#include "exception.hpp"
#include "kernels.hpp"
#if defined(MSCCLPP_USE_GPUNETIO)
#include "gpu_net_io.hpp"
#endif

namespace mscclpp {
namespace ep {

constexpr int CombineNWarps = 32;
constexpr int CombineNThreads = CombineNWarps * WARP_SIZE;
constexpr int CombineNStages = 8;
constexpr int DirectSendMaxNWorkers = WARP_SIZE;

MSCCLPP_HOST_DEVICE_INLINE size_t directSendControlBytes(int nLocalExperts) {
  return configAlign<size_t>(static_cast<size_t>(nLocalExperts + 1) * sizeof(int), 128);
}

template <int Hidden>
MSCCLPP_HOST_DEVICE_INLINE constexpr size_t directSendWorkerBytes() {
  return static_cast<size_t>(Hidden) * sizeof(Bf16) + sizeof(mscclpp::BulkBarrier);
}

template <int Hidden>
MSCCLPP_HOST_DEVICE_INLINE int directSendWorkerCount(int nLocalExperts) {
  const size_t controlBytes = directSendControlBytes(nLocalExperts);
  if (controlBytes >= OptimizedDynamicSharedMemoryBytes) return 0;
  const int availableWorkers =
      static_cast<int>((OptimizedDynamicSharedMemoryBytes - controlBytes) / directSendWorkerBytes<Hidden>());
  return availableWorkers < DirectSendMaxNWorkers ? availableWorkers : DirectSendMaxNWorkers;
}

template <int Hidden, CombineMode Mode, DispatchDataType DispatchType, DispatchLayout Layout>
MSCCLPP_HOST_DEVICE_INLINE size_t combineSharedBytes(int nLocalExperts) {
  if constexpr (Layout == DispatchLayout::RANK_MAJOR) {
    constexpr size_t RowsBytes = static_cast<size_t>(MaxNumTopk) * Hidden * sizeof(DispatchElementType<DispatchType>);
    constexpr size_t BarrierBytes = static_cast<size_t>(MaxNumTopk) * sizeof(mscclpp::BulkBarrier);
    constexpr size_t ValidBytes = static_cast<size_t>(MaxNumTopk) * sizeof(int);
    constexpr size_t SharedBytes = configAlign<size_t>(RowsBytes + BarrierBytes + ValidBytes, 128);
    static_assert(SharedBytes <= OptimizedDynamicSharedMemoryBytes);
    return SharedBytes;
  }
  constexpr size_t TileBytes = static_cast<size_t>(Hidden) * sizeof(Bf16);
  if constexpr (Mode == CombineMode::DIRECT_SEND) {
    return directSendControlBytes(nLocalExperts) +
           static_cast<size_t>(directSendWorkerCount<Hidden>(nLocalExperts)) * directSendWorkerBytes<Hidden>();
  }
  return CombineNStages * TileBytes;
}

#if defined(MSCCLPP_USE_GPUNETIO)
// Pipelined GPUNetIO combine: chunk k of a peer's rows travels on stripe k % numQpsPerPeer, mapped to the QP
// (owner + stripe) % numQpsPerPeer, and bumps that QP's combine flag by one after its payload.
MSCCLPP_DEVICE_INLINE int rankMajorCombineChunkQp(const EpGpuNetIoDeviceContext* network, int owner, int chunk) {
  const int nQp = network->numQpsPerPeer;
  return (owner % nQp + chunk % nQp) % nQp;
}

MSCCLPP_DEVICE_INLINE int rankMajorCombineChunksOnStripe(int nChunks, int stripe, int nQp) {
  return stripe < nChunks ? (nChunks - stripe + nQp - 1) / nQp : 0;
}

// True once the chunk holding `slot` from cross-domain `sourceRank` has landed in this rank's landing region.
MSCCLPP_DEVICE_INLINE bool rankMajorCombineChunkLanded(const TransportView& transport, int sourceRank, int slot) {
  auto* network = transport.gpuNetIo_;
  const int chunk = slot / transport.gpuNetIoCombineChunkRows_;
  const size_t index = static_cast<size_t>(sourceRank) * GpuNetIoMaxQpsPerPeer +
                       rankMajorCombineChunkQp(network, transport.rank_, chunk);
  const uint64_t target = transport.gpuNetIoCombineChunkBase_[index] + chunk / network->numQpsPerPeer + 1;
  auto* flags = static_cast<uint64_t*>(transport.gpuNetIoCombineFlagsBuffer_);
  return mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(flags + index, mscclpp::memoryOrderAcquire) >= target;
}
#endif  // defined(MSCCLPP_USE_GPUNETIO)

#if MSCCLPP_BULK_AVAILABLE

MSCCLPP_DEVICE_INLINE void exchangeCombineReady(const TransportView& transport, int nRanks) {
  const int threadId = static_cast<int>(threadIdx.x);
  if (blockIdx.x == 0 && threadId < nRanks) {
    const int peerRank = threadId;
    if (transport.isSelf(peerRank)) return;
    transport.baseMemoryChannels_[peerRank].signal();
    transport.baseMemoryChannels_[peerRank].wait(-1);
  }
}

MSCCLPP_DEVICE_INLINE void publishRankMajorCombineReady(const TransportView& transport, int nRanks, uint32_t epoch,
                                                        WorkspaceView& workspaceView) {
  if (blockIdx.x != 0) return;
  const int threadId = static_cast<int>(threadIdx.x);
  if (threadId < nRanks) {
    if (!transport.isSelf(threadId)) {
      transport.baseMemoryChannels_[threadId].relaxedSignal();
      transport.baseMemoryChannels_[threadId].relaxedWait(-1);
    }
    mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(workspaceView.combineRankReadyEpochs_ + threadId, epoch,
                                                         mscclpp::memoryOrderRelaxed);
  }
}

template <int Hidden, CombineMode Mode, DispatchDataType DispatchType, bool Network = false>
MSCCLPP_DEVICE_INLINE void recvRankMajorRemotePartialsTma(void* output, const void* expertOutput,
                                                          const int64_t* __restrict__ topkIndices,
                                                          const float* __restrict__ topkWeights, int nTokens, int nTopk,
                                                          int nExperts, int nRanks, int maxTokensPerRank,
                                                          uint32_t epoch, const TransportView& transport,
                                                          WorkspaceView& workspaceView, uint8_t* sharedMemory,
                                                          float dequantScale) {
#if defined(__CUDA_ARCH__)
  static_assert(__CUDA_ARCH__ >= 900, "TMA rank-major combine requires SM90 or newer");
#endif
  static_assert(DispatchType == DispatchDataType::BF16 || DispatchType == DispatchDataType::FP8_E4M3_STATIC);
  static_assert(Hidden % (sizeof(int4) / sizeof(Bf16)) == 0);
  using ElementType = DispatchElementType<DispatchType>;
  constexpr bool IsDirectSend = Mode == CombineMode::DIRECT_SEND;
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(ElementType);
  constexpr int OutputInt4 = static_cast<size_t>(Hidden) * sizeof(Bf16) / sizeof(int4);
  constexpr int Bf16PairsPerInt4 = sizeof(int4) / sizeof(mscclpp::bf16x2);
  constexpr size_t RowsBytes = static_cast<size_t>(MaxNumTopk) * HiddenBytes;
  const int threadId = static_cast<int>(threadIdx.x);
  const int warpId = threadId / WARP_SIZE;
  const int laneId = getLaneId();
  const int nLocalExperts = nExperts / nRanks;
  auto* sharedRows = sharedMemory;
  auto* bulkBarriers = reinterpret_cast<mscclpp::BulkBarrier*>(sharedMemory + RowsBytes);
  auto* validRows = reinterpret_cast<int*>(bulkBarriers + MaxNumTopk);

  const int nWorkerBlocks = static_cast<int>(gridDim.x) - (Network ? 0 : 1);
  const int firstTokenIdx = Network && blockIdx.x == 0 ? nWorkerBlocks - 1 : static_cast<int>(blockIdx.x) - 1;
  // Reuse each mbarrier by advancing its parity instead of reinitializing it.
  uint32_t bulkPhase = 0;
  if (warpId == 0) {
    if (laneId < MaxNumTopk) {
      bulkBarriers[laneId].init();
    }
    __syncwarp();
  }

  for (int tokenIdx = firstTokenIdx; tokenIdx < nTokens; tokenIdx += nWorkerBlocks) {
    if (warpId == 0) {
      const int globalExpertIdx = laneId < nTopk ? static_cast<int>(topkIndices[tokenIdx * nTopk + laneId]) : -1;
      const int destinationRank = globalExpertIdx >= 0 ? globalExpertIdx / nLocalExperts : -1;
      const bool firstLaneForRank = isFirstLaneForRank(destinationRank, laneId);
      const bool validRow = laneId < MaxNumTopk && destinationRank >= 0 && (IsDirectSend || firstLaneForRank);
      int destinationSlot = -1;
      if (validRow) {
        destinationSlot = workspaceView.rankMajorSendIndices_[tokenIdx * nTopk + laneId];
        EP_DEVICE_ASSERT(destinationSlot >= 0 && destinationSlot < maxTokensPerRank);
      }
      if (laneId < MaxNumTopk) {
        validRows[laneId] = validRow;
      }
      __syncwarp();

      bool pending = validRow;
      while (__any_sync(0xffffffff, pending)) {
        bool ready = false;
        if (pending) {
#if defined(MSCCLPP_USE_GPUNETIO)
          if constexpr (Network) {
            if (!transport.isNvlinkPeer(destinationRank) && transport.gpuNetIoCombineChunkRows_ > 0)
              ready = rankMajorCombineChunkLanded(transport, destinationRank, destinationSlot);
            else
              ready =
                  mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(
                      workspaceView.combineRankReadyEpochs_ + destinationRank, mscclpp::memoryOrderAcquire) == epoch;
          } else
#endif
            ready = mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(
                        workspaceView.combineRankReadyEpochs_ + destinationRank, mscclpp::memoryOrderRelaxed) == epoch;
        }
        if (pending && ready) {
          const uint8_t* source;
          if (Network && !transport.isNvlinkPeer(destinationRank)) {
            const auto* landing = static_cast<const uint8_t*>(transport.gpuNetIoCombineLandingBuffer_);
            source =
                landing + (static_cast<size_t>(destinationRank) * maxTokensPerRank + destinationSlot) * HiddenBytes;
          } else {
            const auto* remoteExpertOutput = reinterpret_cast<const uint8_t*>(
                transport.mappedBuffer(const_cast<void*>(expertOutput), destinationRank));
            const size_t sourceRow = static_cast<size_t>(transport.rank_) * maxTokensPerRank + destinationSlot;
            const size_t inputRow = IsDirectSend ? sourceRow * nTopk + laneId : sourceRow;
            source = remoteExpertOutput + inputRow * HiddenBytes;
          }
          auto* sharedRow = sharedRows + static_cast<size_t>(laneId) * HiddenBytes;
          bulkBarriers[laneId].arriveAndExpect(static_cast<uint32_t>(HiddenBytes));
          mscclpp::bulkLoad(sharedRow, source, static_cast<uint32_t>(HiddenBytes), bulkBarriers[laneId]);
          pending = false;
        }
      }
      if (validRow) bulkBarriers[laneId].wait(bulkPhase);
      __syncwarp();
      if (laneId == 0) mscclpp::bulkFence();
    }
    __syncthreads();

    for (int hiddenIdx = threadId; hiddenIdx < OutputInt4; hiddenIdx += CombineNThreads) {
      int4 packedOutput;
      auto* outputValues = reinterpret_cast<mscclpp::bf16x2*>(&packedOutput);
      if constexpr (DispatchType == DispatchDataType::BF16) {
        float2 reduced[Bf16PairsPerInt4] = {};
#pragma unroll
        for (int stage = 0; stage < MaxNumTopk; ++stage) {
          if (validRows[stage] == 0) continue;
          const auto* row = sharedRows + static_cast<size_t>(stage) * HiddenBytes;
          const int4 packed = reinterpret_cast<const int4*>(row)[hiddenIdx];
          const auto* values = reinterpret_cast<const mscclpp::bf16x2*>(&packed);
          const float sourceWeight =
              IsDirectSend && topkWeights != nullptr ? topkWeights[tokenIdx * nTopk + stage] : 1.0f;
#pragma unroll
          for (int pairIdx = 0; pairIdx < Bf16PairsPerInt4; ++pairIdx) {
            const mscclpp::f32x2 value = mscclpp::to<mscclpp::f32x2>(values[pairIdx]);
            reduced[pairIdx].x = fmaf(value.data[0], sourceWeight, reduced[pairIdx].x);
            reduced[pairIdx].y = fmaf(value.data[1], sourceWeight, reduced[pairIdx].y);
          }
        }
#pragma unroll
        for (int pairIdx = 0; pairIdx < Bf16PairsPerInt4; ++pairIdx) {
          outputValues[pairIdx] = mscclpp::to<mscclpp::bf16x2>(mscclpp::f32x2(reduced[pairIdx]));
        }
      } else {
        mscclpp::f32x2 reduced[Bf16PairsPerInt4];
#pragma unroll
        for (int pairIdx = 0; pairIdx < Bf16PairsPerInt4; ++pairIdx) {
          reduced[pairIdx].storage = make_float2(0.0f, 0.0f);
        }
#pragma unroll
        for (int stage = 0; stage < MaxNumTopk; ++stage) {
          if (validRows[stage] == 0) continue;
          const auto* row = sharedRows + static_cast<size_t>(stage) * HiddenBytes;
          const auto values = reinterpret_cast<const mscclpp::f8_e4m3x8*>(row)[hiddenIdx];
          const float sourceWeight =
              IsDirectSend && topkWeights != nullptr ? topkWeights[tokenIdx * nTopk + stage] : 1.0f;
          if (sourceWeight == 0.0f) continue;
          const float accumulationScale = IsDirectSend ? dequantScale * sourceWeight : dequantScale;
          if (!IsDirectSend || isfinite(accumulationScale)) {
            dequantizeAndAccumulateFp8E4M3(values, accumulationScale, reduced);
          } else {
            // The fused scale overflowed, so |sourceWeight| > 1 (or it is not finite). Scaling the payload
            // first is then exact up to rounding: if payload * dequantScale overflows, so does the true result.
            dequantizeWeightAndAccumulateFp8E4M3(values, dequantScale, sourceWeight, reduced);
          }
        }
#pragma unroll
        for (int pairIdx = 0; pairIdx < Bf16PairsPerInt4; ++pairIdx) {
          outputValues[pairIdx] = mscclpp::to<mscclpp::bf16x2>(reduced[pairIdx]);
        }
      }
      auto* outputRow = reinterpret_cast<int4*>(output) + static_cast<size_t>(tokenIdx) * OutputInt4;
      outputRow[hiddenIdx] = packedOutput;
    }
    __syncthreads();
  }
}

#if defined(MSCCLPP_USE_GPUNETIO)
MSCCLPP_DEVICE_INLINE void signalRankMajorCombineLocalStart(const TransportView& transport, int nRanks) {
  if (blockIdx.x != 0) return;
  const int peerRank = static_cast<int>(threadIdx.x);
  if (peerRank < nRanks && !transport.isSelf(peerRank) && transport.isNvlinkPeer(peerRank)) {
    transport.baseMemoryChannels_[peerRank].relaxedSignal();
  }
}

MSCCLPP_DEVICE_INLINE int rankMajorCombineStripeQp(const EpGpuNetIoDeviceContext* network, int owner, int stripe) {
  EP_DEVICE_ASSERT(network->numHcas > 0 && network->numHcas <= network->numQpsPerPeer);
  return (owner % network->numQpsPerPeer + stripe) % network->numQpsPerPeer;
}

MSCCLPP_DEVICE_INLINE void publishRankMajorCombinePushReady(const int64_t* __restrict__ /*topkIndices*/,
                                                            int /*nTokens*/, int /*nTopk*/, int /*nLocalExperts*/,
                                                            int nRanks, uint32_t epoch, const TransportView& transport,
                                                            WorkspaceView& workspaceView) {
  const int laneId = getLaneId();
  for (int destinationRank = laneId; destinationRank < nRanks; destinationRank += WARP_SIZE) {
    if (transport.isSelf(destinationRank)) {
      mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(workspaceView.combineRankReadyEpochs_ + destinationRank,
                                                           epoch, mscclpp::memoryOrderRelease);
    } else if (transport.isNvlinkPeer(destinationRank)) {
      transport.baseMemoryChannels_[destinationRank].relaxedWait(-1);
      mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(workspaceView.combineRankReadyEpochs_ + destinationRank,
                                                           epoch, mscclpp::memoryOrderRelease);
    } else if (transport.gpuNetIoCombineChunkRows_ > 0) {
      mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(workspaceView.combineRankReadyEpochs_ + destinationRank,
                                                           epoch, mscclpp::memoryOrderRelease);
    }
  }
  if (transport.gpuNetIoCombineChunkRows_ > 0) return;

  auto* network = transport.gpuNetIo_;
  auto* flags = reinterpret_cast<volatile uint64_t*>(transport.gpuNetIoCombineFlagsBuffer_);
  for (int baseRank = 0; baseRank < nRanks; baseRank += WARP_SIZE) {
    const int destinationRank = baseRank + laneId;
    bool pending = false;
    uint64_t target = 0;
    if (destinationRank < nRanks && !transport.isSelf(destinationRank) && !transport.isNvlinkPeer(destinationRank)) {
      const int sentRows = workspaceView.dispatchRankPayloadCompletions_[destinationRank];
      if (sentRows > 0) {
        pending = true;
        target = workspaceView.combineArrivedBaseline_[destinationRank] + 1;
      } else {
        mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(workspaceView.combineRankReadyEpochs_ + destinationRank,
                                                             epoch, mscclpp::memoryOrderRelease);
      }
    }
    while (__any_sync(0xffffffff, pending)) {
      if (pending) {
        bool allLanded = true;
        for (int stripe = 0; stripe < network->numHcas; ++stripe) {
          const int qpIndex = rankMajorCombineStripeQp(network, transport.rank_, stripe);
          const size_t flagIndex = static_cast<size_t>(destinationRank) * GpuNetIoMaxQpsPerPeer + qpIndex;
          if (flags[flagIndex] < target) {
            allLanded = false;
            break;
          }
        }
        if (allLanded) {
          workspaceView.combineArrivedBaseline_[destinationRank] = target;
          mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(workspaceView.combineRankReadyEpochs_ + destinationRank,
                                                               epoch, mscclpp::memoryOrderRelease);
          pending = false;
        }
      }
    }
  }
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void sendRankMajorCombinePushOwner(const void* expertOutput, int nRanks, int maxTokensPerRank,
                                                         const TransportView& transport, WorkspaceView& workspaceView,
                                                         int owner, int stripe) {
  constexpr size_t HiddenBytes = static_cast<size_t>(Hidden) * sizeof(Bf16);
  auto* network = transport.gpuNetIo_;
  auto* landingBase = reinterpret_cast<uint8_t*>(transport.gpuNetIoCombineLandingBuffer_);
  const int nQp = network->numQpsPerPeer;
  if (nQp < 1 || nQp > GpuNetIoMaxQpsPerPeer) __trap();
  if (transport.gpuNetIoCombineChunkRows_ > 0) {
    if (owner < nRanks && !transport.isSelf(owner) && !transport.isNvlinkPeer(owner) &&
        workspaceView.dispatchRecvCounts_[owner] > 0 && stripe < nQp) {
      const int nRowsToOwner = workspaceView.dispatchRecvCounts_[owner];
      const int chunkRows = transport.gpuNetIoCombineChunkRows_;
      const int nChunks = (nRowsToOwner + chunkRows - 1) / chunkRows;
      const size_t rowBytes = HiddenBytes;
      auto* remoteFlag = static_cast<uint64_t*>(transport.gpuNetIoCombineFlagsBuffer_) +
                         static_cast<size_t>(transport.rank_) * GpuNetIoMaxQpsPerPeer;
      for (int chunk = stripe; chunk < nChunks; chunk += nQp) {
        const int rowBegin = chunk * chunkRows;
        const int rowEnd = rowBegin + chunkRows < nRowsToOwner ? rowBegin + chunkRows : nRowsToOwner;
        const int qpIndex = rankMajorCombineChunkQp(network, owner, chunk);
        const uint64_t srcRowOffset = transport.symmetricOffset(const_cast<void*>(expertOutput)) +
                                      (static_cast<size_t>(owner) * maxTokensPerRank + rowBegin) * rowBytes;
        auto* landingSlot =
            landingBase + (static_cast<size_t>(transport.rank_) * maxTokensPerRank + rowBegin) * rowBytes;
        network->putWithSignal(owner, transport.symmetricOffset(landingSlot), srcRowOffset,
                               static_cast<uint64_t>(rowEnd - rowBegin) * rowBytes,
                               transport.symmetricOffset(remoteFlag + qpIndex), 1, qpIndex);
      }
    }
    return;
  }
  if (owner < nRanks && !transport.isSelf(owner) && !transport.isNvlinkPeer(owner) &&
      workspaceView.dispatchRecvCounts_[owner] > 0 && stripe < network->numHcas) {
    const int nRowsToOwner = workspaceView.dispatchRecvCounts_[owner];
    const int rowBegin = nRowsToOwner * stripe / network->numHcas;
    const int rowEnd = nRowsToOwner * (stripe + 1) / network->numHcas;
    const int qpIndex = rankMajorCombineStripeQp(network, owner, stripe);
    const uint64_t srcRowOffset = transport.symmetricOffset(const_cast<void*>(expertOutput)) +
                                  (static_cast<size_t>(owner) * maxTokensPerRank + rowBegin) * HiddenBytes;
    auto* landingSlot =
        landingBase + (static_cast<size_t>(transport.rank_) * maxTokensPerRank + rowBegin) * HiddenBytes;
    auto* remoteFlag = static_cast<uint64_t*>(transport.gpuNetIoCombineFlagsBuffer_) +
                       static_cast<size_t>(transport.rank_) * GpuNetIoMaxQpsPerPeer + qpIndex;
    network->putWithSignal(owner, transport.symmetricOffset(landingSlot), srcRowOffset,
                           static_cast<uint64_t>(rowEnd - rowBegin) * HiddenBytes,
                           transport.symmetricOffset(remoteFlag), 1, qpIndex);
  }
}

template <int Hidden>
MSCCLPP_DEVICE_INLINE void sendRankMajorCombinePush(const void* expertOutput, int nRanks, int maxTokensPerRank,
                                                    const TransportView& transport, WorkspaceView& workspaceView,
                                                    [[maybe_unused]] uint32_t epoch) {
  const int owner = static_cast<int>(gridDim.x) - 1 - static_cast<int>(blockIdx.x);
  if (owner < 0 || owner >= nRanks) return;
  sendRankMajorCombinePushOwner<Hidden>(expertOutput, nRanks, maxTokensPerRank, transport, workspaceView, owner,
                                        static_cast<int>(threadIdx.x));
}

MSCCLPP_DEVICE_INLINE void drainRankMajorCombinePushOwner(int nRanks, const TransportView& transport,
                                                          WorkspaceView& workspaceView, int owner, int stripe) {
  auto* network = transport.gpuNetIo_;
  if (transport.gpuNetIoCombineChunkRows_ > 0) {
    if (owner >= nRanks || transport.isNvlinkPeer(owner) || workspaceView.dispatchRecvCounts_[owner] <= 0) return;
    const int nChunks = (workspaceView.dispatchRecvCounts_[owner] + transport.gpuNetIoCombineChunkRows_ - 1) /
                        transport.gpuNetIoCombineChunkRows_;
    if (stripe < network->numQpsPerPeer && stripe < nChunks)
      network->flush(owner, rankMajorCombineChunkQp(network, owner, stripe));
    return;
  }
  if (owner >= nRanks || transport.isNvlinkPeer(owner) || workspaceView.dispatchRecvCounts_[owner] <= 0 ||
      stripe >= network->numHcas)
    return;
  network->flush(owner, rankMajorCombineStripeQp(network, owner, stripe));
}

MSCCLPP_DEVICE_INLINE void drainRankMajorCombinePush(int nRanks, const TransportView& transport,
                                                     WorkspaceView& workspaceView, [[maybe_unused]] uint32_t epoch) {
  const int owner = static_cast<int>(gridDim.x) - 1 - static_cast<int>(blockIdx.x);
  if (owner < 0 || owner >= nRanks) return;
  drainRankMajorCombinePushOwner(nRanks, transport, workspaceView, owner, static_cast<int>(threadIdx.x));
}

#endif

#endif  // MSCCLPP_BULK_AVAILABLE

template <CombineMode Mode, int Hidden, DispatchDataType DispatchType, int ScaleBlockSize, DispatchLayout Layout,
          typename KernelSelector, bool Network = false>
inline void combineHiddenMode(void* output, const void* expertOutput, const int64_t* topkIndices,
                              const float* topkWeights, const int* srcInfo, const int64_t* layoutRange,
                              const Workload& workload, void* recvBuffer, void* dispatchRecvBuffer,
                              const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  static_assert(Hidden == 1024 || Hidden == 2048 || Hidden == 4096 || Hidden == 4352 || Hidden == 5120 ||
                Hidden == 6656 || Hidden == 7168 || Hidden == 8192 || Hidden == 8704 || Hidden == 9216);
  const int nExperts = workload.numExperts_;
  const int nRanks = context.numRanks_;
  const int nLocalExperts = nExperts / nRanks;
  if constexpr (Mode == CombineMode::DIRECT_SEND && Layout == DispatchLayout::EXPERT_MAJOR) {
    EP_HOST_ASSERT(directSendWorkerCount<Hidden>(nLocalExperts) > 0);
  }

  auto combineFunc = KernelSelector::template get<Hidden, DispatchType, ScaleBlockSize, Layout>();
  const size_t sharedBytes = combineSharedBytes<Hidden, Mode, DispatchType, Layout>(nLocalExperts);
  EP_HOST_ASSERT(Layout != DispatchLayout::RANK_MAJOR || numBlocks > 1);
  static thread_local KernelConfigCache kernelConfig;
  const int residentBlocks = configureKernel(combineFunc, CombineNThreads, sharedBytes, context, kernelConfig);
  EP_HOST_ASSERT(residentBlocks >= numBlocks);

  if constexpr (Network) {
    if (!context.gpuNetIoAggregateCombine_) {
      const WorkspaceView workspace(context.workspace_, nRanks, nExperts);
      MSCCLPP_CUDATHROW(cudaMemsetAsync(workspace.combineRankReadyEpochs_, 0,
                                        static_cast<size_t>(nRanks) * sizeof(uint32_t), stream));
    }
  }

  combineFunc<<<dim3(numBlocks), dim3(CombineNThreads), sharedBytes, stream>>>(
      output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload, recvBuffer, dispatchRecvBuffer,
      context.devicePtr_);
  MSCCLPP_CUDATHROW(cudaGetLastError());
}

template <CombineMode Mode, int Hidden, typename KernelSelector, bool Network = false>
inline void combineHidden(void* output, const void* expertOutput, const int64_t* topkIndices, const float* topkWeights,
                          const int* srcInfo, const int64_t* layoutRange, const Workload& workload, void* recvBuffer,
                          void* dispatchRecvBuffer, const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  if constexpr (Network) {
    EP_HOST_ASSERT(workload.outputLayout_ == DispatchLayout::RANK_MAJOR);
    EP_HOST_ASSERT(workload.dispatchDataType_ == DispatchDataType::BF16 &&
                   workload.combineDataType_ == DispatchDataType::BF16);
    return combineHiddenMode<Mode, Hidden, DispatchDataType::BF16, 0, DispatchLayout::RANK_MAJOR, KernelSelector, true>(
        output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload, recvBuffer, dispatchRecvBuffer,
        context, numBlocks, stream);
  } else {
    if (workload.outputLayout_ == DispatchLayout::RANK_MAJOR) {
      switch (workload.combineDataType_) {
        case DispatchDataType::BF16:
          return combineHiddenMode<Mode, Hidden, DispatchDataType::BF16, 0, DispatchLayout::RANK_MAJOR, KernelSelector>(
              output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload, recvBuffer,
              dispatchRecvBuffer, context, numBlocks, stream);
        case DispatchDataType::FP8_E4M3:
        case DispatchDataType::MXFP8_E4M3:
          break;
        case DispatchDataType::FP8_E4M3_STATIC:
          return combineHiddenMode<Mode, Hidden, DispatchDataType::FP8_E4M3_STATIC, 0, DispatchLayout::RANK_MAJOR,
                                   KernelSelector>(output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange,
                                                   workload, recvBuffer, dispatchRecvBuffer, context, numBlocks,
                                                   stream);
      }
      EP_HOST_ASSERT(false && "unsupported rank-major combine data type");
    }
    if constexpr (Mode == CombineMode::RANK_LOCAL_REDUCE) {
      switch (workload.dispatchDataType_) {
        case DispatchDataType::BF16:
          return combineHiddenMode<CombineMode::RANK_LOCAL_REDUCE, Hidden, DispatchDataType::BF16, 0,
                                   DispatchLayout::EXPERT_MAJOR, KernelSelector>(
              output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload, recvBuffer,
              dispatchRecvBuffer, context, numBlocks, stream);
        case DispatchDataType::FP8_E4M3:
          return combineHiddenMode<CombineMode::RANK_LOCAL_REDUCE, Hidden, DispatchDataType::FP8_E4M3, 128,
                                   DispatchLayout::EXPERT_MAJOR, KernelSelector>(
              output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload, recvBuffer,
              dispatchRecvBuffer, context, numBlocks, stream);
        case DispatchDataType::MXFP8_E4M3:
        case DispatchDataType::FP8_E4M3_STATIC:
          break;
      }
    } else {
      switch (workload.dispatchDataType_) {
        case DispatchDataType::BF16:
          return combineHiddenMode<CombineMode::DIRECT_SEND, Hidden, DispatchDataType::BF16, 0,
                                   DispatchLayout::EXPERT_MAJOR, KernelSelector>(
              output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload, recvBuffer,
              dispatchRecvBuffer, context, numBlocks, stream);
        case DispatchDataType::FP8_E4M3:
          return combineHiddenMode<CombineMode::DIRECT_SEND, Hidden, DispatchDataType::FP8_E4M3, 128,
                                   DispatchLayout::EXPERT_MAJOR, KernelSelector>(
              output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, workload, recvBuffer,
              dispatchRecvBuffer, context, numBlocks, stream);
        case DispatchDataType::MXFP8_E4M3:
        case DispatchDataType::FP8_E4M3_STATIC:
          break;
      }
    }
    EP_HOST_ASSERT(false && "unsupported dispatch data type");
  }
}

template <CombineMode Mode, typename KernelSelector, bool Network = false>
inline void combineAlgorithm(void* output, const void* expertOutput, const int64_t* topkIndices,
                             const float* topkWeights, const int* srcInfo, const int64_t* layoutRange,
                             const Workload& workload, void* recvBuffer, void* dispatchRecvBuffer,
                             const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  const int nExperts = workload.numExperts_;
  const int rank = context.rank_;
  const int nRanks = context.numRanks_;

  EP_HOST_ASSERT(workload.numTokens_ == 0 || output != nullptr);
  EP_HOST_ASSERT(expertOutput != nullptr);
  EP_HOST_ASSERT(workload.numTokens_ == 0 || topkIndices != nullptr);
  EP_HOST_ASSERT(recvBuffer != nullptr);
  EP_HOST_ASSERT(dispatchRecvBuffer != nullptr);
  EP_HOST_ASSERT(context.localBufferBase_ != nullptr);
  EP_HOST_ASSERT(context.peerBufferBases_ != nullptr);
  EP_HOST_ASSERT(context.channels_ != nullptr);
  EP_HOST_ASSERT(context.workspace_ != nullptr);
  EP_HOST_ASSERT(context.devicePtr_ != nullptr);
  EP_HOST_ASSERT(isSupportedRanks(nRanks));
  EP_HOST_ASSERT(nExperts > 0 && nExperts % nRanks == 0);
  EP_HOST_ASSERT(rank >= 0 && rank < nRanks);
  EP_HOST_ASSERT(workload.numTokens_ >= 0 && workload.numTokens_ <= workload.maxTokensPerRank_);
  EP_HOST_ASSERT(workload.numTopk_ > 0 && workload.numTopk_ <= MaxNumTopk);
  EP_HOST_ASSERT(numBlocks > 0 && numBlocks <= MaxWorkerBlocks);
  static_assert(Mode == CombineMode::RANK_LOCAL_REDUCE || Mode == CombineMode::DIRECT_SEND);
  EP_HOST_ASSERT(workload.outputLayout_ == DispatchLayout::EXPERT_MAJOR ||
                 workload.outputLayout_ == DispatchLayout::RANK_MAJOR);
  EP_HOST_ASSERT(isSupportedDispatchDataType(workload.dispatchDataType_));
  EP_HOST_ASSERT(workload.outputLayout_ != DispatchLayout::RANK_MAJOR ||
                 isSupportedCombineDataType(workload.combineDataType_));
  if constexpr (Mode == CombineMode::DIRECT_SEND) {
    if (workload.outputLayout_ == DispatchLayout::EXPERT_MAJOR) {
      EP_HOST_ASSERT(srcInfo != nullptr);
      EP_HOST_ASSERT(layoutRange != nullptr);
    }
  }

  switch (workload.hidden_) {
    case 1024:
      return combineHidden<Mode, 1024, KernelSelector, Network>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                                layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                                context, numBlocks, stream);
    case 2048:
      return combineHidden<Mode, 2048, KernelSelector, Network>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                                layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                                context, numBlocks, stream);
    case 4096:
      return combineHidden<Mode, 4096, KernelSelector, Network>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                                layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                                context, numBlocks, stream);
    case 4352:
      return combineHidden<Mode, 4352, KernelSelector, Network>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                                layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                                context, numBlocks, stream);
    case 5120:
      if constexpr (Network) {
        return combineHidden<Mode, 5120, KernelSelector, true>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                               layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                               context, numBlocks, stream);
      } else {
        EP_HOST_ASSERT(false && "unsupported latency combine hidden size");
        return;
      }
    case 6656:
      return combineHidden<Mode, 6656, KernelSelector, Network>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                                layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                                context, numBlocks, stream);
    case 7168:
      return combineHidden<Mode, 7168, KernelSelector, Network>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                                layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                                context, numBlocks, stream);
    case 8192:
      return combineHidden<Mode, 8192, KernelSelector, Network>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                                layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                                context, numBlocks, stream);
    case 8704:
      return combineHidden<Mode, 8704, KernelSelector, Network>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                                layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                                context, numBlocks, stream);
    case 9216:
      return combineHidden<Mode, 9216, KernelSelector, Network>(output, expertOutput, topkIndices, topkWeights, srcInfo,
                                                                layoutRange, workload, recvBuffer, dispatchRecvBuffer,
                                                                context, numBlocks, stream);
    default:
      EP_HOST_ASSERT(false && "unsupported latency combine hidden size");
  }
}

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_COMBINE_COMMON_CUH_
