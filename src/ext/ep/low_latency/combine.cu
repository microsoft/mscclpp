// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "../kernels/api.cuh"
#include "../kernels/exception.cuh"
#include "../kernels/utils.cuh"
#include "config.cuh"

namespace mscclpp {
namespace ep {
namespace low_latency_opt {

constexpr int kCombineNWarps = 32;
constexpr int kCombineNThreads = kCombineNWarps * WARP_SIZE;
constexpr int kCombineNStages = 8;
constexpr int kDirectSendMaxNWorkers = WARP_SIZE;
constexpr int kCombineMaxNBlocks = 128;
constexpr int kCombineMaxNTopk = 9;

MSCCLPP_HOST_DEVICE_INLINE size_t combineControlBytes(int nLocalExperts) {
  const size_t directControlBytes = static_cast<size_t>(nLocalExperts + 1) * sizeof(int);
  return configAlign<size_t>(directControlBytes > sizeof(RecvTask) ? directControlBytes : sizeof(RecvTask), 128);
}

template <int kHidden, low_latency::OptimizedCombineMode kMode>
MSCCLPP_HOST_DEVICE_INLINE size_t combineSharedBytes(int nLocalExperts) {
  constexpr size_t kTileBytes = static_cast<size_t>(kHidden) * sizeof(nv_bfloat16);
  if constexpr (kMode == low_latency::OptimizedCombineMode::DIRECT_SEND) {
    constexpr int kNWorkers = tmaWorkerCount<kHidden, kDirectSendMaxNWorkers>();
    return combineControlBytes(nLocalExperts) + static_cast<size_t>(kNWorkers) * (kTileBytes + sizeof(uint64_t));
  }
  return combineControlBytes(nLocalExperts) + kCombineNStages * kTileBytes;
}

template <int kHiddenInt4>
MSCCLPP_DEVICE_INLINE int4 reduceWeightedBf16x8(const void* expertOutput, int rowOffset, float weight, int nTopk,
                                                int hiddenIdx) {
  constexpr int kBf16PairsPerInt4 = sizeof(int4) / sizeof(nv_bfloat162);
  float2 reduced[kBf16PairsPerInt4] = {};
  for (int topkLane = 0; topkLane < nTopk; ++topkLane) {
    const int sourceRowOffset = __shfl_sync(0xffffffff, rowOffset, topkLane);
    if (sourceRowOffset < 0) continue;
    const float sourceWeight = __shfl_sync(0xffffffff, weight, topkLane);
    const int4 packed = ld_nc_global(reinterpret_cast<const int4*>(expertOutput) +
                                     static_cast<size_t>(sourceRowOffset) * kHiddenInt4 + hiddenIdx);
    const auto* values = reinterpret_cast<const nv_bfloat162*>(&packed);
#pragma unroll
    for (int pairIdx = 0; pairIdx < kBf16PairsPerInt4; ++pairIdx) {
      const float2 value = __bfloat1622float2(values[pairIdx]);
      reduced[pairIdx].x = fmaf(value.x, sourceWeight, reduced[pairIdx].x);
      reduced[pairIdx].y = fmaf(value.y, sourceWeight, reduced[pairIdx].y);
    }
  }

  int4 packedOutput;
  auto* outputValues = reinterpret_cast<nv_bfloat162*>(&packedOutput);
#pragma unroll
  for (int pairIdx = 0; pairIdx < kBf16PairsPerInt4; ++pairIdx) {
    outputValues[pairIdx] = __float22bfloat162_rn(reduced[pairIdx]);
  }
  return packedOutput;
}

template <int kHiddenInt4>
MSCCLPP_DEVICE_INLINE int4 reduceRankPartialsBf16x8(const void* combineRecvBuffer, int partialRankCandidate, int nTopk,
                                                    int maxTokensPerRank, int tokenIdx, int hiddenIdx) {
  constexpr int kBf16PairsPerInt4 = sizeof(int4) / sizeof(nv_bfloat162);
  float2 reduced[kBf16PairsPerInt4] = {};
  for (int topkLane = 0; topkLane < nTopk; ++topkLane) {
    const int partialRank = __shfl_sync(0xffffffff, partialRankCandidate, topkLane);
    if (partialRank < 0) continue;
    const int4 packed =
        ld_nc_global(reinterpret_cast<const int4*>(combineRecvBuffer) +
                     (static_cast<size_t>(partialRank) * maxTokensPerRank + tokenIdx) * kHiddenInt4 + hiddenIdx);
    const auto* values = reinterpret_cast<const nv_bfloat162*>(&packed);
#pragma unroll
    for (int pairIdx = 0; pairIdx < kBf16PairsPerInt4; ++pairIdx) {
      const float2 value = __bfloat1622float2(values[pairIdx]);
      reduced[pairIdx].x += value.x;
      reduced[pairIdx].y += value.y;
    }
  }

  int4 packedOutput;
  auto* outputValues = reinterpret_cast<nv_bfloat162*>(&packedOutput);
#pragma unroll
  for (int pairIdx = 0; pairIdx < kBf16PairsPerInt4; ++pairIdx) {
    outputValues[pairIdx] = __float22bfloat162_rn(reduced[pairIdx]);
  }
  return packedOutput;
}

template <int kHidden>
MSCCLPP_DEVICE_INLINE void sendRankLocalPartials(const void* expertOutput, int nExperts, int rank, int nRanks,
                                                 int nTopk, int maxTokensPerRank, void* combineRecvBuffer,
                                                 const void* dispatchRecvBuffer, void* rdmaBufferBase,
                                                 void* const* peerRecvBuffers, DispatchWorkspaceView& workspaceView,
                                                 uint8_t* sharedMemory) {
#if defined(__CUDA_ARCH__)
  static_assert(__CUDA_ARCH__ >= 900, "TMA combine send requires SM90 or newer");
#endif
  const int threadId = static_cast<int>(threadIdx.x);
  const int laneId = get_lane_id();
  const int nLocalExperts = nExperts / nRanks;
  [[maybe_unused]] const int nExpertOutputRows = nLocalExperts * nRanks * maxTokensPerRank;
  constexpr size_t kHiddenBytes = static_cast<size_t>(kHidden) * sizeof(nv_bfloat16);
  constexpr int kHiddenInt4 = kHiddenBytes / sizeof(int4);
  constexpr int kChunksPerThread = (kHiddenInt4 + kCombineNThreads - 1) / kCombineNThreads;
  static_assert(kHiddenInt4 % WARP_SIZE == 0);
  const size_t dispatchMetadataSize = dispatchMetadataBytes(nRanks, nExperts);
  const size_t payloadStride = dispatchPayloadStride(kHidden, nTopk);
  const LowLatencyPayloadView<nv_bfloat16> payloadView(kHidden, nTopk);
  auto* recvTask = reinterpret_cast<RecvTask*>(sharedMemory);
  auto* outputTiles = sharedMemory + combineControlBytes(nLocalExperts);

  int tokenIteration = 0;
  for (int taskIdx = static_cast<int>(blockIdx.x); taskIdx < *workspaceView.nRecvTasks_;
       taskIdx += static_cast<int>(gridDim.x)) {
    if (threadId == 0) *recvTask = workspaceView.recvTasks_[taskIdx];
    __syncthreads();
    const int sourceRank = recvTask->sourceRank_;

    for (int sourceTokenSlot = recvTask->tokenBegin_; sourceTokenSlot < recvTask->tokenEnd_;
         ++sourceTokenSlot, ++tokenIteration) {
      const int stage = tokenIteration % kCombineNStages;
      auto* outputTile = reinterpret_cast<int4*>(outputTiles + static_cast<size_t>(stage) * kHiddenBytes);
      const auto* sourcePayload =
          reinterpret_cast<const uint8_t*>(dispatchRecvBuffer) + dispatchMetadataSize +
          (static_cast<size_t>(sourceRank) * maxTokensPerRank + sourceTokenSlot) * payloadStride;
      const int rowOffset = laneId < nTopk ? ld_nc_global(payloadView.topKIndices(sourcePayload) + laneId) : -1;
      const float weight = laneId < nTopk ? ld_nc_global(payloadView.topKValues(sourcePayload) + laneId) : 0.0f;
      if (rowOffset >= 0) EP_DEVICE_ASSERT(rowOffset < nExpertOutputRows);

      int4 reduced[kChunksPerThread] = {};
#pragma unroll
      for (int chunkIdx = 0; chunkIdx < kChunksPerThread; ++chunkIdx) {
        const int hiddenIdx = threadId + chunkIdx * kCombineNThreads;
        if (hiddenIdx < kHiddenInt4) {
          reduced[chunkIdx] = reduceWeightedBf16x8<kHiddenInt4>(expertOutput, rowOffset, weight, nTopk, hiddenIdx);
        }
      }

      if (tokenIteration >= kCombineNStages && threadId == 0) {
        waitTmaS2GRead<kCombineNStages - 1>();
      }
      if (tokenIteration >= kCombineNStages) __syncthreads();
#pragma unroll
      for (int chunkIdx = 0; chunkIdx < kChunksPerThread; ++chunkIdx) {
        const int hiddenIdx = threadId + chunkIdx * kCombineNThreads;
        if (hiddenIdx < kHiddenInt4) outputTile[hiddenIdx] = reduced[chunkIdx];
      }
      __syncthreads();

      if (threadId == 0) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        const int sourceTokenIdx =
            ld_nc_global(payloadView.srcTokenGlobalIdx(sourcePayload)) - sourceRank * maxTokensPerRank;
        EP_DEVICE_ASSERT(sourceTokenIdx >= 0 && sourceTokenIdx < maxTokensPerRank);
        void* destinationBuffer = sourceRank == rank
                                      ? combineRecvBuffer
                                      : peerBufferPtr(combineRecvBuffer, rdmaBufferBase, peerRecvBuffers[sourceRank]);
        auto* destinationRow = reinterpret_cast<uint8_t*>(destinationBuffer) +
                               (static_cast<size_t>(rank) * maxTokensPerRank + sourceTokenIdx) * kHiddenBytes;
        issueTmaS2G(destinationRow, outputTile, static_cast<uint32_t>(kHiddenBytes));
      }
    }
    __syncthreads();
  }

  if (tokenIteration > 0 && threadId == 0) waitTmaS2G();
}

template <int kHidden>
MSCCLPP_DEVICE_INLINE void sendExpertRowsDirect(const void* expertOutput, const int* srcInfo,
                                                const int64_t* layoutRange, int nExperts, int rank, int nRanks,
                                                int maxTokensPerRank, void* combineRecvBuffer, void* rdmaBufferBase,
                                                void* const* peerRecvBuffers, uint8_t* sharedMemory) {
  const int threadId = static_cast<int>(threadIdx.x);
  const int warpId = threadId / WARP_SIZE;
  const int laneId = get_lane_id();
  const int nLocalExperts = nExperts / nRanks;
  const int nOutputSlotsPerExpert = nRanks * maxTokensPerRank;
  constexpr int kNWorkers = tmaWorkerCount<kHidden, kDirectSendMaxNWorkers>();
  constexpr size_t kHiddenBytes = static_cast<size_t>(kHidden) * sizeof(nv_bfloat16);
  auto* expertTokenPrefix = reinterpret_cast<int*>(sharedMemory);
  auto* outputTiles = sharedMemory + combineControlBytes(nLocalExperts);

  if (threadId == 0) {
    expertTokenPrefix[0] = 0;
    for (int localExpertIdx = 0; localExpertIdx < nLocalExperts; ++localExpertIdx) {
      int nLastRankTokens;
      int lastRankOffset;
      unpack2(layoutRange[localExpertIdx * nRanks + nRanks - 1], nLastRankTokens, lastRankOffset);
      expertTokenPrefix[localExpertIdx + 1] = expertTokenPrefix[localExpertIdx] + lastRankOffset + nLastRankTokens;
    }
  }
  __syncthreads();

  const int nTotalRows = expertTokenPrefix[nLocalExperts];
  const int blockRowBegin = static_cast<int>(static_cast<int64_t>(nTotalRows) * blockIdx.x / gridDim.x);
  const int blockRowEnd = static_cast<int>(static_cast<int64_t>(nTotalRows) * (blockIdx.x + 1) / gridDim.x);
  auto* tmaBarriers = reinterpret_cast<uint64_t*>(outputTiles + static_cast<size_t>(kNWorkers) * kHiddenBytes);
  if (warpId == 0 && laneId < kNWorkers) {
    auto* outputTile = outputTiles + static_cast<size_t>(laneId) * kHiddenBytes;
    auto* tmaBarrier = tmaBarriers + laneId;
    uint32_t tmaPhase = 0;
    if (blockRowBegin + laneId < blockRowEnd) initTmaBarrier(tmaBarrier);

    bool hasPendingStore = false;
    for (int flatRowIdx = blockRowBegin + laneId; flatRowIdx < blockRowEnd; flatRowIdx += kNWorkers) {
      if (hasPendingStore) waitTmaS2GRead();
      int localExpertIdx = 0;
      while (flatRowIdx >= expertTokenPrefix[localExpertIdx + 1]) ++localExpertIdx;
      const int expertTokenIdx = flatRowIdx - expertTokenPrefix[localExpertIdx];
      int sourceRank = 0;
      for (; sourceRank < nRanks; ++sourceRank) {
        int nRankTokens;
        int rankOffset;
        unpack2(layoutRange[localExpertIdx * nRanks + sourceRank], nRankTokens, rankOffset);
        if (expertTokenIdx >= rankOffset && expertTokenIdx < rankOffset + nRankTokens) break;
      }
      EP_DEVICE_ASSERT(sourceRank < nRanks);
      const int inputRowOffset = localExpertIdx * nOutputSlotsPerExpert + expertTokenIdx;
      const int sourceTokenIdx = ld_nc_global(srcInfo + inputRowOffset);
      EP_DEVICE_ASSERT(sourceTokenIdx >= 0 && sourceTokenIdx < maxTokensPerRank);
      const auto* inputRow =
          reinterpret_cast<const uint8_t*>(expertOutput) + static_cast<size_t>(inputRowOffset) * kHiddenBytes;
      issueTmaG2S(inputRow, outputTile, tmaBarrier, static_cast<uint32_t>(kHiddenBytes));
      waitTmaG2S(tmaBarrier, tmaPhase);
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      const int globalExpertIdx = rank * nLocalExperts + localExpertIdx;
      void* destinationBuffer = sourceRank == rank
                                    ? combineRecvBuffer
                                    : peerBufferPtr(combineRecvBuffer, rdmaBufferBase, peerRecvBuffers[sourceRank]);
      auto* destinationRow = reinterpret_cast<uint8_t*>(destinationBuffer) +
                             (static_cast<size_t>(globalExpertIdx) * maxTokensPerRank + sourceTokenIdx) * kHiddenBytes;
      issueTmaS2G(destinationRow, outputTile, static_cast<uint32_t>(kHiddenBytes));
      hasPendingStore = true;
    }

    if (hasPendingStore) waitTmaS2G();
  }
}

MSCCLPP_DEVICE_INLINE void combineSynchronize(mscclpp::BaseMemoryChannelDeviceHandle* signalChannels,
                                              mscclpp::DeviceSemaphore* localReady, int rank, int nRanks,
                                              int signalChannelStride) {
  const int threadId = static_cast<int>(threadIdx.x);
  if (blockIdx.x == 0 && threadId < nRanks) {
    const int peerRank = threadId;
    if (peerRank == rank) {
      localReady->release();
      localReady->acquire();
    } else {
      auto& signalChannel = signalChannels[rankSignalChannelIndex(peerRank, signalChannelStride)];
      signalChannel.signal();
      signalChannel.wait(-1);
    }
  }
}

template <int kHidden>
MSCCLPP_DEVICE_INLINE void recvRankLocalPartials(void* output, const int64_t* topkIndices, int nTokens, int nTopk,
                                                 int nExperts, int nRanks, int maxTokensPerRank,
                                                 const void* combineRecvBuffer, uint8_t* sharedMemory) {
  const int threadId = static_cast<int>(threadIdx.x);
  const int laneId = get_lane_id();
  const int nLocalExperts = nExperts / nRanks;
  constexpr size_t kHiddenBytes = static_cast<size_t>(kHidden) * sizeof(nv_bfloat16);
  constexpr int kHiddenInt4 = kHiddenBytes / sizeof(int4);
  constexpr int kChunksPerThread = (kHiddenInt4 + kCombineNThreads - 1) / kCombineNThreads;
  static_assert(kHiddenInt4 % WARP_SIZE == 0);
  auto* outputTiles = sharedMemory + combineControlBytes(nLocalExperts);

  int tokenIteration = 0;
  for (int tokenIdx = static_cast<int>(blockIdx.x); tokenIdx < nTokens;
       tokenIdx += static_cast<int>(gridDim.x), ++tokenIteration) {
    const int stage = tokenIteration % kCombineNStages;
    auto* outputTile = reinterpret_cast<int4*>(outputTiles + static_cast<size_t>(stage) * kHiddenBytes);
    const int globalExpertIdx = laneId < nTopk ? static_cast<int>(__ldg(topkIndices + tokenIdx * nTopk + laneId)) : -1;
    const int destinationRank = globalExpertIdx >= 0 ? globalExpertIdx / nLocalExperts : -1;
    const bool firstLaneForRank = isFirstLaneForRank(destinationRank, laneId);
    const int partialRank = destinationRank >= 0 && firstLaneForRank ? destinationRank : -1;

    int4 reduced[kChunksPerThread] = {};
#pragma unroll
    for (int chunkIdx = 0; chunkIdx < kChunksPerThread; ++chunkIdx) {
      const int hiddenIdx = threadId + chunkIdx * kCombineNThreads;
      if (hiddenIdx < kHiddenInt4) {
        reduced[chunkIdx] = reduceRankPartialsBf16x8<kHiddenInt4>(combineRecvBuffer, partialRank, nTopk,
                                                                  maxTokensPerRank, tokenIdx, hiddenIdx);
      }
    }
    if (tokenIteration >= kCombineNStages && threadId == 0) {
      waitTmaS2GRead<kCombineNStages - 1>();
    }
    if (tokenIteration >= kCombineNStages) __syncthreads();
#pragma unroll
    for (int chunkIdx = 0; chunkIdx < kChunksPerThread; ++chunkIdx) {
      const int hiddenIdx = threadId + chunkIdx * kCombineNThreads;
      if (hiddenIdx < kHiddenInt4) outputTile[hiddenIdx] = reduced[chunkIdx];
    }
    __syncthreads();

    if (threadId == 0) {
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      auto* outputRow = reinterpret_cast<uint8_t*>(output) + static_cast<size_t>(tokenIdx) * kHiddenBytes;
      issueTmaS2G(outputRow, outputTile, static_cast<uint32_t>(kHiddenBytes));
    }
  }
  if (tokenIteration > 0 && threadId == 0) waitTmaS2G();
}

template <int kHidden>
MSCCLPP_DEVICE_INLINE void recvExpertRowsDirect(void* output, const int64_t* topkIndices, const float* topkWeights,
                                                int nTokens, int nTopk, int maxTokensPerRank,
                                                const void* combineRecvBuffer) {
  constexpr int kBf16PerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
  constexpr int kHiddenInt4 = kHidden / kBf16PerInt4;
  const int threadId = static_cast<int>(threadIdx.x);

  for (int tokenIdx = static_cast<int>(blockIdx.x); tokenIdx < nTokens; tokenIdx += static_cast<int>(gridDim.x)) {
    int regTopkIndices[kCombineMaxNTopk];
    float regTopkWeights[kCombineMaxNTopk];
    for (int topkIdx = 0; topkIdx < nTopk; ++topkIdx) {
      regTopkIndices[topkIdx] = static_cast<int>(__ldg(topkIndices + tokenIdx * nTopk + topkIdx));
      regTopkWeights[topkIdx] = topkWeights == nullptr ? 1.0f : __ldg(topkWeights + tokenIdx * nTopk + topkIdx);
    }

#pragma unroll
    for (int hiddenIdx = threadId; hiddenIdx < kHiddenInt4; hiddenIdx += kCombineNThreads) {
      float reduced[kBf16PerInt4] = {0.0f};
      for (int topkIdx = 0; topkIdx < nTopk; ++topkIdx) {
        const int expertIdx = regTopkIndices[topkIdx];
        if (expertIdx < 0) continue;
        const auto* expertRow = reinterpret_cast<const int4*>(combineRecvBuffer) +
                                (static_cast<size_t>(expertIdx) * maxTokensPerRank + tokenIdx) * kHiddenInt4;
        const int4 packed = ld_nc_global(expertRow + hiddenIdx);
        const auto* values = reinterpret_cast<const nv_bfloat16*>(&packed);
#pragma unroll
        for (int elemIdx = 0; elemIdx < kBf16PerInt4; ++elemIdx) {
          reduced[elemIdx] += static_cast<float>(values[elemIdx]) * regTopkWeights[topkIdx];
        }
      }

      int4 packedOutput;
      auto* outputValues = reinterpret_cast<nv_bfloat16*>(&packedOutput);
#pragma unroll
      for (int elemIdx = 0; elemIdx < kBf16PerInt4; ++elemIdx) {
        outputValues[elemIdx] = static_cast<nv_bfloat16>(reduced[elemIdx]);
      }
      auto* outputRow = reinterpret_cast<int4*>(output) + static_cast<size_t>(tokenIdx) * kHiddenInt4;
      outputRow[hiddenIdx] = packedOutput;
    }
  }
}

template <low_latency::OptimizedCombineMode kMode, int kHidden>
__global__ __launch_bounds__(kCombineNThreads, 1) void combineKernel(
    void* output, const void* expertOutput, const int64_t* topkIndices, const float* topkWeights, const int* srcInfo,
    const int64_t* layoutRange, int nTokens, int nExperts, int rank, int nRanks, int nTopk, int maxTokensPerRank,
    void* combineRecvBuffer, const void* dispatchRecvBuffer, void* rdmaBufferBase, void* const* peerRecvBuffers,
    mscclpp::BaseMemoryChannelDeviceHandle* signalChannels, void* workspace, int dispatchMaxSms,
    int signalChannelStride) {
  extern __shared__ __align__(128) uint8_t sharedMemory[];
  DispatchWorkspaceView workspaceView(workspace, nRanks, nExperts, dispatchMaxSms);

  if constexpr (kMode == low_latency::OptimizedCombineMode::RANK_LOCAL_REDUCE) {
    sendRankLocalPartials<kHidden>(expertOutput, nExperts, rank, nRanks, nTopk, maxTokensPerRank, combineRecvBuffer,
                                   dispatchRecvBuffer, rdmaBufferBase, peerRecvBuffers, workspaceView, sharedMemory);
  } else {
    sendExpertRowsDirect<kHidden>(expertOutput, srcInfo, layoutRange, nExperts, rank, nRanks, maxTokensPerRank,
                                  combineRecvBuffer, rdmaBufferBase, peerRecvBuffers, sharedMemory);
  }

  workspaceView.combineSyncer_->sync(gridDim.x);
  combineSynchronize(signalChannels, workspaceView.localPayloadReady_, rank, nRanks, signalChannelStride);
  workspaceView.combineSyncer_->sync(gridDim.x);

  if constexpr (kMode == low_latency::OptimizedCombineMode::RANK_LOCAL_REDUCE) {
    recvRankLocalPartials<kHidden>(output, topkIndices, nTokens, nTopk, nExperts, nRanks, maxTokensPerRank,
                                   combineRecvBuffer, sharedMemory);
  } else {
    recvExpertRowsDirect<kHidden>(output, topkIndices, topkWeights, nTokens, nTopk, maxTokensPerRank,
                                  combineRecvBuffer);
  }
}

template <low_latency::OptimizedCombineMode kMode, int kHidden>
inline void combineHiddenMode(void* output, const void* expertOutput, const int64_t* topkIndices,
                              const float* topkWeights, const int* srcInfo, const int64_t* layoutRange,
                              const low_latency::CombineConfig& config, const low_latency::BufferSet& currentBuffer,
                              void* dispatchRecvBuffer, const low_latency::TransportContext& transport, void* workspace,
                              int numBlocks, cudaStream_t stream) {
  static_assert(kHidden == 4096 || kHidden == 7168 || kHidden == 8192 || kHidden == 9216);
  if constexpr (kMode == low_latency::OptimizedCombineMode::DIRECT_SEND) {
    static_assert(tmaWorkerCount<kHidden, kDirectSendMaxNWorkers>() > 0);
  }
  const int nExperts = config.numExperts_;
  const int rank = transport.rank_;
  const int nRanks = transport.numRanks_;
  const int nTokens = config.numCombinedTokens_;
  const int nTopk = config.numTopk_;
  const int nLocalExperts = nExperts / nRanks;
  const int maxTokensPerRank = config.numMaxTokensPerRank_;
  const int signalChannelStride = transport.memoryChannelStride_;

  auto combineFunc = combineKernel<kMode, kHidden>;
  static thread_local int sharedMemoryLimitDevice = -1;
  static thread_local size_t sharedMemoryLimit = 0;
  if (sharedMemoryLimitDevice != transport.deviceId_) {
    int sharedMemoryLimitInt;
    cudaFuncAttributes attributes;
    CUDA_CHECK(
        cudaDeviceGetAttribute(&sharedMemoryLimitInt, cudaDevAttrMaxSharedMemoryPerBlockOptin, transport.deviceId_));
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, combineFunc));
    EP_HOST_ASSERT(sharedMemoryLimitInt > static_cast<int>(attributes.sharedSizeBytes));
    sharedMemoryLimitDevice = transport.deviceId_;
    sharedMemoryLimit = static_cast<size_t>(sharedMemoryLimitInt) - attributes.sharedSizeBytes;
  }

  const size_t sharedBytes = combineSharedBytes<kHidden, kMode>(nLocalExperts);
  EP_HOST_ASSERT(sharedBytes <= sharedMemoryLimit);
  static thread_local int configuredDevice = -1;
  static thread_local size_t configuredSharedBytes = 0;
  static thread_local int residentBlocks = 0;
  if (configuredDevice != transport.deviceId_ || configuredSharedBytes < sharedBytes) {
    CUDA_CHECK(
        cudaFuncSetAttribute(combineFunc, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sharedBytes)));
    int blocksPerSm;
    int numSms;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSm, combineFunc, kCombineNThreads, sharedBytes));
    CUDA_CHECK(cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, transport.deviceId_));
    configuredDevice = transport.deviceId_;
    configuredSharedBytes = sharedBytes;
    residentBlocks = blocksPerSm * numSms;
  }
  EP_HOST_ASSERT(residentBlocks >= numBlocks);

  cudaLaunchConfig_t launchCfg = {dim3(numBlocks), dim3(kCombineNThreads), sharedBytes, stream, nullptr, 0};
  CUDA_CHECK(cudaLaunchKernelEx(&launchCfg, combineFunc, output, expertOutput, topkIndices, topkWeights, srcInfo,
                                layoutRange, nTokens, nExperts, rank, nRanks, nTopk, maxTokensPerRank,
                                currentBuffer.recvDataBuffer_, dispatchRecvBuffer, transport.rdmaBufferBase_,
                                transport.peerBases_, transport.memoryChannels_, workspace, numBlocks,
                                signalChannelStride));
}

template <int kHidden>
inline void combineHidden(void* output, const void* expertOutput, const int64_t* topkIndices, const float* topkWeights,
                          const int* srcInfo, const int64_t* layoutRange, const low_latency::CombineConfig& config,
                          const low_latency::BufferSet& currentBuffer, void* dispatchRecvBuffer,
                          const low_latency::TransportContext& transport, void* workspace, int numBlocks,
                          low_latency::OptimizedCombineMode mode, cudaStream_t stream) {
  if (mode == low_latency::OptimizedCombineMode::RANK_LOCAL_REDUCE) {
    return combineHiddenMode<low_latency::OptimizedCombineMode::RANK_LOCAL_REDUCE, kHidden>(
        output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, config, currentBuffer, dispatchRecvBuffer,
        transport, workspace, numBlocks, stream);
  }
  return combineHiddenMode<low_latency::OptimizedCombineMode::DIRECT_SEND, kHidden>(
      output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, config, currentBuffer, dispatchRecvBuffer,
      transport, workspace, numBlocks, stream);
}

inline void combine(void* output, const void* expertOutput, const int64_t* topkIndices, const float* topkWeights,
                    const int* srcInfo, const int64_t* layoutRange, const low_latency::CombineConfig& config,
                    const low_latency::BufferSet& currentBuffer, void* dispatchRecvBuffer,
                    const low_latency::TransportContext& transport, void* workspace, int numBlocks,
                    low_latency::OptimizedCombineMode mode, cudaStream_t stream) {
  const int nExperts = config.numExperts_;
  const int rank = transport.rank_;
  const int nRanks = transport.numRanks_;

  EP_HOST_ASSERT(output != nullptr);
  EP_HOST_ASSERT(expertOutput != nullptr);
  EP_HOST_ASSERT(topkIndices != nullptr);
  EP_HOST_ASSERT(currentBuffer.recvDataBuffer_ != nullptr);
  EP_HOST_ASSERT(dispatchRecvBuffer != nullptr);
  EP_HOST_ASSERT(transport.rdmaBufferBase_ != nullptr);
  EP_HOST_ASSERT(transport.peerBases_ != nullptr);
  EP_HOST_ASSERT(transport.memoryChannels_ != nullptr);
  EP_HOST_ASSERT(transport.ipcReady_);
  EP_HOST_ASSERT(workspace != nullptr);
  EP_HOST_ASSERT(nRanks > 0 && nRanks <= 2 * WARP_SIZE);
  EP_HOST_ASSERT(nExperts > 0 && nExperts % nRanks == 0);
  EP_HOST_ASSERT(rank >= 0 && rank < nRanks);
  EP_HOST_ASSERT(config.numCombinedTokens_ >= 0 && config.numCombinedTokens_ <= config.numMaxTokensPerRank_);
  EP_HOST_ASSERT(config.numTopk_ > 0 && config.numTopk_ <= kCombineMaxNTopk);
  EP_HOST_ASSERT(transport.memoryChannelStride_ >= 1);
  EP_HOST_ASSERT(config.inputDType_ == low_latency::DType::BF16);
  EP_HOST_ASSERT(config.outputDType_ == low_latency::DType::BF16);
  EP_HOST_ASSERT(numBlocks > 0 && numBlocks <= kCombineMaxNBlocks);
  EP_HOST_ASSERT(mode == low_latency::OptimizedCombineMode::RANK_LOCAL_REDUCE ||
                 mode == low_latency::OptimizedCombineMode::DIRECT_SEND);
  if (mode == low_latency::OptimizedCombineMode::DIRECT_SEND) {
    EP_HOST_ASSERT(srcInfo != nullptr);
    EP_HOST_ASSERT(layoutRange != nullptr);
  }

  switch (config.hidden_) {
    case 4096:
      return combineHidden<4096>(output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, config,
                                 currentBuffer, dispatchRecvBuffer, transport, workspace, numBlocks, mode, stream);
    case 7168:
      return combineHidden<7168>(output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, config,
                                 currentBuffer, dispatchRecvBuffer, transport, workspace, numBlocks, mode, stream);
    case 8192:
      return combineHidden<8192>(output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, config,
                                 currentBuffer, dispatchRecvBuffer, transport, workspace, numBlocks, mode, stream);
    case 9216:
      return combineHidden<9216>(output, expertOutput, topkIndices, topkWeights, srcInfo, layoutRange, config,
                                 currentBuffer, dispatchRecvBuffer, transport, workspace, numBlocks, mode, stream);
    default:
      EP_HOST_ASSERT(false && "unsupported optimized low-latency hidden size");
  }
}

}  // namespace low_latency_opt

namespace low_latency {

void combineOptimized(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights,
                      const int* srcInfo, const int64_t* layoutRange, const CombineConfig& config,
                      const BufferSet& currentBuffer, void* dispatchRecvBuffer, const TransportContext& transport,
                      void* workspace, int numBlocks, OptimizedCombineMode mode, cudaStream_t stream) {
  low_latency_opt::combine(output, input, topkIdx, topkWeights, srcInfo, layoutRange, config, currentBuffer,
                           dispatchRecvBuffer, transport, workspace, numBlocks, mode, stream);
}

}  // namespace low_latency
}  // namespace ep
}  // namespace mscclpp
