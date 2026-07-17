// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP),
// branch `chhwang/dev-atomic-add-cleanup`. Licensed under the MIT License.

#include <cooperative_groups.h>

#include "api.cuh"
#include "constants.cuh"
#include "device_helpers.cuh"
#include "exception.cuh"
#include "launch.cuh"

namespace mscclpp {
namespace ep {
namespace high_throughput {
namespace detail {

template <int NumRanks>
__global__ void notifyDispatchKernel(const int* numTokensPerRank, int* mappedRecvCounter, const int* numTokensPerExpert,
                                     int* mappedRecvExpertCounters, int numExperts, int numTokens, int numChannels,
                                     const bool* isTokenInRank, int* channelPrefixMatrix, int* rankPrefixMatrix,
                                     int expertAlignment, void** bufferPtrs, int** taskFifoPtrs, int head, int rank) {
  const int blockId = static_cast<int>(blockIdx.x);
  const int threadId = static_cast<int>(threadIdx.x);
  const int numThreads = static_cast<int>(blockDim.x);
  const int laneId = threadId % WARP_SIZE;
  const int warpId = threadId / WARP_SIZE;
  const int numWarps = numThreads / WARP_SIZE;

  if (blockId == 0) {
    barrier_device<NumRanks>(taskFifoPtrs, head, rank);
    move_fifo_slots<NumRanks>(head);

    const int numExpertsPerRank = numExperts / NumRanks;
    if (threadId < NumRanks) {
      auto* peerRankCounts = reinterpret_cast<int*>(bufferPtrs[threadId]);
      auto* peerExpertCounts = peerRankCounts + NumRanks * NumRanks;
#pragma unroll
      for (int dstRank = 0; dstRank < NumRanks; ++dstRank) {
        peerRankCounts[rank * NumRanks + dstRank] = numTokensPerRank[dstRank];
      }
#pragma unroll
      for (int localExpert = 0; localExpert < numExpertsPerRank; ++localExpert) {
        peerExpertCounts[rank * numExpertsPerRank + localExpert] =
            numTokensPerExpert[threadId * numExpertsPerRank + localExpert];
      }
    }
    __syncthreads();

    barrier_device<NumRanks>(taskFifoPtrs, head, rank);
    move_fifo_slots<NumRanks>(head);

    auto* localRankCounts = reinterpret_cast<int*>(bufferPtrs[rank]);
    if (threadId < NumRanks) {
#pragma unroll
      for (int srcRank = 1; srcRank < NumRanks; ++srcRank) {
        localRankCounts[srcRank * NumRanks + threadId] += localRankCounts[(srcRank - 1) * NumRanks + threadId];
      }
      if (threadId == rank) *mappedRecvCounter = localRankCounts[(NumRanks - 1) * NumRanks + rank];
    }

    auto* localExpertCounts = localRankCounts + NumRanks * NumRanks;
    if (threadId < numExpertsPerRank) {
      int count = 0;
#pragma unroll
      for (int srcRank = 0; srcRank < NumRanks; ++srcRank) {
        count += localExpertCounts[srcRank * numExpertsPerRank + threadId];
      }
      mappedRecvExpertCounters[threadId] = (count + expertAlignment - 1) / expertAlignment * expertAlignment;
    }
    __syncthreads();

#pragma unroll
    for (int index = threadId; index < NumRanks * NumRanks; index += numThreads) {
      rankPrefixMatrix[index] = localRankCounts[index];
    }
    memory_fence();
    barrier_device<NumRanks>(taskFifoPtrs, head, rank);
    return;
  }

  const int dstRank = blockId - 1;
  for (int channel = warpId; channel < numChannels; channel += numWarps) {
    int tokenBegin;
    int tokenEnd;
    get_channel_task_range(numTokens, numChannels, channel, tokenBegin, tokenEnd);

    int count = 0;
    for (int token = tokenBegin + laneId; token < tokenEnd; token += WARP_SIZE) {
      count += isTokenInRank[token * NumRanks + dstRank];
    }
    count = warp_reduce_sum(count);
    if (laneId == 0) channelPrefixMatrix[dstRank * numChannels + channel] = count;
  }
  __syncthreads();

  if (threadId == 0) {
#pragma unroll
    for (int channel = 1; channel < numChannels; ++channel) {
      channelPrefixMatrix[dstRank * numChannels + channel] += channelPrefixMatrix[dstRank * numChannels + channel - 1];
    }
  }
}

void notifyDispatch(const int* numTokensPerRank, int* mappedRecvCounter, int numRanks, const int* numTokensPerExpert,
                    int* mappedRecvExpertCounters, int numExperts, int numTokens, const bool* isTokenInRank,
                    int* channelPrefixMatrix, int* rankPrefixMatrix, int expertAlignment, void** bufferPtrs,
                    int** taskFifoPtrs, int head, int rank, cudaStream_t stream, int numChannels) {
#define NOTIFY_DISPATCH_LAUNCH_CASE(ranks)                                                                        \
  LAUNCH_KERNEL(&cfg, notifyDispatchKernel<ranks>, numTokensPerRank, mappedRecvCounter, numTokensPerExpert,       \
                mappedRecvExpertCounters, numExperts, numTokens, numChannels, isTokenInRank, channelPrefixMatrix, \
                rankPrefixMatrix, expertAlignment, bufferPtrs, taskFifoPtrs, head, rank);                         \
  break

  constexpr int NumThreads = 128;
  EP_HOST_ASSERT(numExperts % numRanks == 0);
  EP_HOST_ASSERT(numExperts / numRanks <= NumThreads && numRanks <= NumThreads);
  EP_HOST_ASSERT(numChannels > 0);

  SETUP_LAUNCH_CONFIG(1 + numRanks, NumThreads, stream);
  SWITCH_RANKS(numRanks, NOTIFY_DISPATCH_LAUNCH_CASE);
#undef NOTIFY_DISPATCH_LAUNCH_CASE
}

template <int NumRanks>
__global__ void cachedNotifyDispatchKernel(const int* rankPrefixMatrix, void** bufferPtrs, int** taskFifoPtrs, int head,
                                           int rank) {
  barrier_device<NumRanks>(taskFifoPtrs, head, rank);
  move_fifo_slots<NumRanks>(head);

  const int threadId = static_cast<int>(threadIdx.x);
  const int numThreads = static_cast<int>(blockDim.x);
  auto* localRankCounts = reinterpret_cast<int*>(bufferPtrs[rank]);
#pragma unroll
  for (int index = threadId; index < NumRanks * NumRanks; index += numThreads) {
    localRankCounts[index] = rankPrefixMatrix[index];
  }
  memory_fence();
  __syncthreads();
  barrier_device<NumRanks>(taskFifoPtrs, head, rank);
}

void cachedNotifyDispatch(const int* rankPrefixMatrix, void** bufferPtrs, int** taskFifoPtrs, int head, int rank,
                          int numRanks, cudaStream_t stream) {
#define CACHED_NOTIFY_DISPATCH_LAUNCH_CASE(ranks)                                                                 \
  LAUNCH_KERNEL(&cfg, cachedNotifyDispatchKernel<ranks>, rankPrefixMatrix, bufferPtrs, taskFifoPtrs, head, rank); \
  break

  SETUP_LAUNCH_CONFIG(1, 128, stream);
  SWITCH_RANKS(numRanks, CACHED_NOTIFY_DISPATCH_LAUNCH_CASE);
#undef CACHED_NOTIFY_DISPATCH_LAUNCH_CASE
}

template <int NumRanks, int NumThreads>
__global__ void __launch_bounds__(NumThreads, 1)
    dispatchKernel(int* sendHead, const int4* input, const int64_t* topkIdx, const float* topkWeights,
                   const float* inputScales, const bool* isTokenInRank, const int* channelPrefixMatrix, int numTokens,
                   int numRecvTokens, int hiddenInt4, int numTopk, int numExperts, int numScales, int64_t* recvTopkIdx,
                   float* recvTopkWeights, float* recvXScales, void** bufferPtrs, int** taskFifoPtrs, int head,
                   int rank, void** recvPoolPtrs, int64_t recvPoolHeaderBytes, int64_t recvPoolMetadataOffset,
                   int64_t metadataSlotBytes, int* combineRecvIdx) {
  const int numChannels = static_cast<int>(gridDim.x);
  const int channel = static_cast<int>(blockIdx.x);
  const int threadId = static_cast<int>(threadIdx.x);
  const int threadsPerRank = NumThreads / NumRanks;
  const int dstRank = threadId / threadsPerRank;
  const int rankThreadId = threadId % threadsPerRank;
  const int laneId = threadId % WARP_SIZE;
  const int expertsPerRank = numExperts / NumRanks;
  EP_DEVICE_ASSERT(NumRanks <= WARP_SIZE);
  EP_DEVICE_ASSERT(NumThreads % NumRanks == 0);

  const int* dstRankPrefix = reinterpret_cast<const int*>(bufferPtrs[dstRank]);
  const int rankOffset = rank > 0 ? dstRankPrefix[(rank - 1) * NumRanks + dstRank] : 0;
  const int channelOffset = channel > 0 ? channelPrefixMatrix[dstRank * numChannels + channel - 1] : 0;
  const int64_t outputBase = static_cast<int64_t>(rankOffset + channelOffset);
  auto* dstPool = reinterpret_cast<uint8_t*>(recvPoolPtrs[dstRank]);
  auto* dstTokens = reinterpret_cast<int4*>(dstPool + recvPoolHeaderBytes);
  auto* dstMetadata = dstPool + recvPoolMetadataOffset;

  int tokenBegin;
  int tokenEnd;
  get_channel_task_range(numTokens, numChannels, channel, tokenBegin, tokenEnd);

  int outputOffset = 0;
  for (int token = tokenBegin; token < tokenEnd; ++token) {
    const bool selected = isTokenInRank[token * NumRanks + dstRank];
    const int64_t outputIndex = outputBase + outputOffset;
    if (rankThreadId == 0) sendHead[token * NumRanks + dstRank] = selected ? static_cast<int>(outputIndex) : -1;
    if (!selected) continue;

    const int4* srcRow = input + static_cast<int64_t>(token) * hiddenInt4;
    int4* dstRow = dstTokens + outputIndex * hiddenInt4;
    for (int hiddenIndex = rankThreadId; hiddenIndex < hiddenInt4; hiddenIndex += threadsPerRank) {
      st_na_global(dstRow + hiddenIndex, __ldg(srcRow + hiddenIndex));
    }

    auto* metadata = dstMetadata + outputIndex * metadataSlotBytes;
    if (rankThreadId == 0 && combineRecvIdx != nullptr) {
      combineRecvIdx[token * NumRanks + dstRank] = static_cast<int>(outputIndex);
    }
    if (topkIdx != nullptr && laneId < numTopk && rankThreadId < WARP_SIZE) {
      auto* metadataTopkIdx = reinterpret_cast<int*>(metadata);
      auto* metadataTopkWeights = reinterpret_cast<float*>(metadata + static_cast<size_t>(numTopk) * sizeof(int));
      const int expertBegin = dstRank * expertsPerRank;
      const int expertEnd = expertBegin + expertsPerRank;
      const int64_t expert = __ldg(topkIdx + static_cast<int64_t>(token) * numTopk + laneId);
      const int localExpert = expert >= expertBegin && expert < expertEnd ? static_cast<int>(expert) - expertBegin : -1;
      metadataTopkIdx[laneId] = localExpert;
      metadataTopkWeights[laneId] =
          localExpert >= 0 ? __ldg(topkWeights + static_cast<int64_t>(token) * numTopk + laneId) : 0.0f;
    }
    if (inputScales != nullptr && rankThreadId < WARP_SIZE) {
      auto* metadataScales =
          reinterpret_cast<float*>(metadata + static_cast<size_t>(numTopk) * (sizeof(int) + sizeof(float)));
      for (int scale = laneId; scale < numScales; scale += WARP_SIZE) {
        metadataScales[scale] = __ldg(inputScales + static_cast<int64_t>(token) * numScales + scale);
      }
    }
    ++outputOffset;
  }

  memory_fence();
  cooperative_groups::this_grid().sync();
  if (blockIdx.x == 0 && threadIdx.x < WARP_SIZE) barrier_device<NumRanks>(taskFifoPtrs, head, rank);
  cooperative_groups::this_grid().sync();

  const auto* localPool = reinterpret_cast<const uint8_t*>(recvPoolPtrs[rank]);
  const int globalThreadId = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int gridThreads = static_cast<int>(gridDim.x * blockDim.x);
  for (int token = globalThreadId; token < numRecvTokens; token += gridThreads) {
    const auto* metadata = localPool + recvPoolMetadataOffset + static_cast<int64_t>(token) * metadataSlotBytes;
    if (recvTopkIdx != nullptr) {
      const auto* metadataTopkIdx = reinterpret_cast<const int*>(metadata);
      const auto* metadataTopkWeights =
          reinterpret_cast<const float*>(metadata + static_cast<size_t>(numTopk) * sizeof(int));
      for (int topk = 0; topk < numTopk; ++topk) {
        recvTopkIdx[static_cast<int64_t>(token) * numTopk + topk] = metadataTopkIdx[topk];
        recvTopkWeights[static_cast<int64_t>(token) * numTopk + topk] = metadataTopkWeights[topk];
      }
    }
    if (recvXScales != nullptr) {
      const auto* metadataScales =
          reinterpret_cast<const float*>(metadata + static_cast<size_t>(numTopk) * (sizeof(int) + sizeof(float)));
      for (int scale = 0; scale < numScales; ++scale) {
        recvXScales[static_cast<int64_t>(token) * numScales + scale] = metadataScales[scale];
      }
    }
  }
}

template <int NumRanks, int NumThreads>
int maxCooperativeDispatchBlocks() {
  static int cachedDevice = -1;
  static int cachedMaxBlocks = 0;

  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  if (device != cachedDevice) {
    int blocksPerSm;
    int numSms;
    auto kernel = dispatchKernel<NumRanks, NumThreads>;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSm, kernel, NumThreads, 0));
    CUDA_CHECK(cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, device));
    cachedDevice = device;
    cachedMaxBlocks = blocksPerSm * numSms;
  }
  return cachedMaxBlocks;
}

void dispatch(int* sendHead, const void* input, const int64_t* topkIdx, const float* topkWeights,
              const float* inputScales, const bool* isTokenInRank, const int* channelPrefixMatrix, int numTokens,
              int numRecvTokens, int hiddenInt4, int numTopk, int numExperts, int numScales, int64_t* recvTopkIdx,
              float* recvTopkWeights, float* recvXScales, void** bufferPtrs, int** taskFifoPtrs, int head, int rank,
              int numRanks, cudaStream_t stream, int numBlocks, void** recvPoolPtrs, int64_t recvPoolHeaderBytes,
              int64_t recvPoolMetadataOffset, int64_t metadataSlotBytes, int* combineRecvIdx) {
  constexpr int NumThreads = 512;
  EP_HOST_ASSERT(recvPoolPtrs != nullptr);
  EP_HOST_ASSERT(numBlocks > 0);
  EP_HOST_ASSERT(static_cast<int64_t>(numTopk) * static_cast<int64_t>(sizeof(int) + sizeof(float)) +
                     static_cast<int64_t>(numScales) * static_cast<int64_t>(sizeof(float)) <=
                 metadataSlotBytes);

#define DISPATCH_LAUNCH_CASE(ranks)                                                                                  \
  EP_HOST_ASSERT((numBlocks <= maxCooperativeDispatchBlocks<ranks, NumThreads>()));                                  \
  LAUNCH_KERNEL(&cfg, (dispatchKernel<ranks, NumThreads>), sendHead, reinterpret_cast<const int4*>(input), topkIdx,  \
                topkWeights, inputScales, isTokenInRank, channelPrefixMatrix, numTokens, numRecvTokens, hiddenInt4,  \
                numTopk, numExperts, numScales, recvTopkIdx, recvTopkWeights, recvXScales, bufferPtrs, taskFifoPtrs, \
                head, rank, recvPoolPtrs, recvPoolHeaderBytes, recvPoolMetadataOffset, metadataSlotBytes,            \
                combineRecvIdx);                                                                                     \
  break

  SETUP_LAUNCH_CONFIG(numBlocks, NumThreads, stream);
  SWITCH_RANKS(numRanks, DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

}  // namespace detail

void notifyDispatch(const int* numTokensPerRank, int* mappedRecvCounter, int numRanks, const int* numTokensPerExpert,
                    int* mappedRecvExpertCounters, int numExperts, int numTokens, const bool* isTokenInRank,
                    int* channelPrefixMatrix, int* rankPrefixMatrix, int expertAlignment, void** bufferPtrs,
                    int** taskFifoPtrs, int head, int rank, cudaStream_t stream, int numChannels) {
  detail::notifyDispatch(numTokensPerRank, mappedRecvCounter, numRanks, numTokensPerExpert, mappedRecvExpertCounters,
                         numExperts, numTokens, isTokenInRank, channelPrefixMatrix, rankPrefixMatrix, expertAlignment,
                         bufferPtrs, taskFifoPtrs, head, rank, stream, numChannels);
}

void cachedNotifyDispatch(const int* rankPrefixMatrix, void** bufferPtrs, int** taskFifoPtrs, int head, int rank,
                          int numRanks, cudaStream_t stream) {
  detail::cachedNotifyDispatch(rankPrefixMatrix, bufferPtrs, taskFifoPtrs, head, rank, numRanks, stream);
}

void dispatch(int* sendHead, const void* input, const int64_t* topkIdx, const float* topkWeights,
              const float* inputScales, const bool* isTokenInRank, const int* channelPrefixMatrix, int numTokens,
              int numRecvTokens, int hiddenInt4, int numTopk, int numExperts, int numScales, int64_t* recvTopkIdx,
              float* recvTopkWeights, float* recvXScales, void** bufferPtrs, int** taskFifoPtrs, int head, int rank,
              int numRanks, cudaStream_t stream, int numBlocks, void** recvPoolPtrs, int64_t recvPoolHeaderBytes,
              int64_t recvPoolMetadataOffset, int64_t metadataSlotBytes, int* combineRecvIdx) {
  detail::dispatch(sendHead, input, topkIdx, topkWeights, inputScales, isTokenInRank, channelPrefixMatrix, numTokens,
                   numRecvTokens, hiddenInt4, numTopk, numExperts, numScales, recvTopkIdx, recvTopkWeights, recvXScales,
                   bufferPtrs, taskFifoPtrs, head, rank, numRanks, stream, numBlocks, recvPoolPtrs, recvPoolHeaderBytes,
                   recvPoolMetadataOffset, metadataSlotBytes, combineRecvIdx);
}

}  // namespace high_throughput
}  // namespace ep
}  // namespace mscclpp
