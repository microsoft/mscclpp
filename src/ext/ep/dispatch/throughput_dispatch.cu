// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP),
// branch `chhwang/dev-atomic-add-cleanup`. Licensed under the MIT License.

#include <cooperative_groups.h>

#include "common/device_helpers.cuh"
#include "exception.hpp"
#include "kernels.hpp"

namespace mscclpp {
namespace ep {

MSCCLPP_HOST_DEVICE_INLINE void getChannelTaskRange(int numTokens, int numChannels, int channel, int& tokenBegin,
                                                    int& tokenEnd) {
  tokenBegin = static_cast<int>(static_cast<int64_t>(numTokens) * channel / numChannels);
  tokenEnd = static_cast<int>(static_cast<int64_t>(numTokens) * (channel + 1) / numChannels);
}

__global__ void exchangeThroughputCountsKernel(const int* numTokensPerRank, const int* numTokensPerExpert,
                                               int numExperts, int numTokens, int numChannels,
                                               const bool* isTokenInRank, int* channelPrefixMatrix,
                                               int* rankPrefixMatrix, int expertAlignment,
                                               const DeviceContext* context) {
  const int numRanks = context->numRanks_;
  const int blockId = static_cast<int>(blockIdx.x);
  const int threadId = static_cast<int>(threadIdx.x);
  const int numThreads = static_cast<int>(blockDim.x);
  const int laneId = threadId % WARP_SIZE;
  const int warpId = threadId / WARP_SIZE;
  const int numWarps = numThreads / WARP_SIZE;

  if (blockId == 0) {
    if (threadId < WARP_SIZE) barrier(context->channels_, context->rank_, numRanks);
    __syncthreads();

    const int numExpertsPerRank = numExperts / numRanks;
    if (threadId < numRanks) {
      auto* peerRankCounts = reinterpret_cast<int*>(context->peerBufferBases_[threadId]);
      auto* peerExpertCounts = peerRankCounts + numRanks * numRanks;
      for (int dstRank = 0; dstRank < numRanks; ++dstRank) {
        peerRankCounts[context->rank_ * numRanks + dstRank] = numTokensPerRank[dstRank];
      }
#pragma unroll
      for (int localExpert = 0; localExpert < numExpertsPerRank; ++localExpert) {
        peerExpertCounts[context->rank_ * numExpertsPerRank + localExpert] =
            numTokensPerExpert[threadId * numExpertsPerRank + localExpert];
      }
    }
    __syncthreads();

    if (threadId < WARP_SIZE) barrier(context->channels_, context->rank_, numRanks);
    __syncthreads();

    auto* localRankCounts = reinterpret_cast<int*>(context->peerBufferBases_[context->rank_]);
    if (threadId < numRanks) {
      for (int srcRank = 1; srcRank < numRanks; ++srcRank) {
        localRankCounts[srcRank * numRanks + threadId] += localRankCounts[(srcRank - 1) * numRanks + threadId];
      }
      if (threadId == context->rank_)
        *context->mappedRecvCounter_ = localRankCounts[(numRanks - 1) * numRanks + context->rank_];
    }

    auto* localExpertCounts = localRankCounts + numRanks * numRanks;
    if (threadId < numExpertsPerRank) {
      int count = 0;
      for (int srcRank = 0; srcRank < numRanks; ++srcRank) {
        count += localExpertCounts[srcRank * numExpertsPerRank + threadId];
      }
      context->mappedRecvExpertCounters_[threadId] = (count + expertAlignment - 1) / expertAlignment * expertAlignment;
    }
    __syncthreads();

    for (int index = threadId; index < numRanks * numRanks; index += numThreads) {
      rankPrefixMatrix[index] = localRankCounts[index];
    }
    __threadfence_system();
    if (threadId < WARP_SIZE) barrier(context->channels_, context->rank_, numRanks);
    __syncthreads();
    return;
  }

  const int dstRank = blockId - 1;
  for (int channel = warpId; channel < numChannels; channel += numWarps) {
    int tokenBegin;
    int tokenEnd;
    getChannelTaskRange(numTokens, numChannels, channel, tokenBegin, tokenEnd);

    int count = 0;
    for (int token = tokenBegin + laneId; token < tokenEnd; token += WARP_SIZE) {
      count += isTokenInRank[token * numRanks + dstRank];
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) count += __shfl_down_sync(0xffffffffu, count, offset);
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

void throughputExchangeCounts(const int* numTokensPerRank, const int* numTokensPerExpert, int numExperts, int numTokens,
                              const bool* isTokenInRank, int* channelPrefixMatrix, int* rankPrefixMatrix,
                              int expertAlignment, const DeviceContext& context, cudaStream_t stream, int numChannels) {
  constexpr int NumThreads = 128;
  EP_HOST_ASSERT(numExperts % context.numRanks_ == 0);
  EP_HOST_ASSERT(numExperts / context.numRanks_ <= NumThreads && context.numRanks_ <= NumThreads);
  EP_HOST_ASSERT(numChannels > 0);

  exchangeThroughputCountsKernel<<<1 + context.numRanks_, NumThreads, 0, stream>>>(
      numTokensPerRank, numTokensPerExpert, numExperts, numTokens, numChannels, isTokenInRank, channelPrefixMatrix,
      rankPrefixMatrix, expertAlignment, context.devicePtr_);
  MSCCLPP_CUDATHROW(cudaGetLastError());
}

__global__ void publishCachedThroughputPrefixKernel(const int* rankPrefixMatrix, const DeviceContext* context) {
  const int numRanks = context->numRanks_;
  if (threadIdx.x < WARP_SIZE) barrier(context->channels_, context->rank_, numRanks);
  __syncthreads();

  const int threadId = static_cast<int>(threadIdx.x);
  const int numThreads = static_cast<int>(blockDim.x);
  auto* localRankCounts = reinterpret_cast<int*>(context->peerBufferBases_[context->rank_]);
  for (int index = threadId; index < numRanks * numRanks; index += numThreads) {
    localRankCounts[index] = rankPrefixMatrix[index];
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x < WARP_SIZE) barrier(context->channels_, context->rank_, numRanks);
}

void throughputPublishCachedPrefix(const int* rankPrefixMatrix, const DeviceContext& context, cudaStream_t stream) {
  publishCachedThroughputPrefixKernel<<<1, 128, 0, stream>>>(rankPrefixMatrix, context.devicePtr_);
  MSCCLPP_CUDATHROW(cudaGetLastError());
}

template <int NumThreads, DispatchLayout Layout>
__global__ void __launch_bounds__(NumThreads, 1)
    throughputDispatchKernel(int* sendHead, const int4* input, const int64_t* topkIdx, const float* topkWeights,
                             const float* inputScales, const bool* isTokenInRank, const int* channelPrefixMatrix,
                             int numTokens, int numRecvTokens, int hiddenInt4, int numTopk, int numExperts,
                             int numScales, int* recvTopkIdx, float* recvTopkWeights, float* recvXScales,
                             int64_t recvPoolHeaderBytes, int64_t recvPoolMetadataOffset, int64_t metadataSlotBytes,
                             int maxTokensPerRank, const DeviceContext* context) {
  const int numRanks = context->numRanks_;
  const int numChannels = static_cast<int>(gridDim.x);
  const int channel = static_cast<int>(blockIdx.x);
  const int threadId = static_cast<int>(threadIdx.x);
  const int threadsPerRank = NumThreads / numRanks;
  const int dstRank = threadId / threadsPerRank;
  const int rankThreadId = threadId % threadsPerRank;
  const int laneId = threadId % WARP_SIZE;
  const int expertsPerRank = numExperts / numRanks;
  EP_DEVICE_ASSERT(numRanks <= WARP_SIZE);
  EP_DEVICE_ASSERT(NumThreads % numRanks == 0);

  const int* dstRankPrefix = reinterpret_cast<const int*>(context->peerBufferBases_[dstRank]);
  const int rankOffset = context->rank_ > 0 ? dstRankPrefix[(context->rank_ - 1) * numRanks + dstRank] : 0;
  const int channelOffset = channel > 0 ? channelPrefixMatrix[dstRank * numChannels + channel - 1] : 0;
  const int64_t rankBase =
      Layout == DispatchLayout::RANK_MAJOR ? static_cast<int64_t>(context->rank_) * maxTokensPerRank : rankOffset;
  const int64_t outputBase = rankBase + channelOffset;
  auto* dstPool = reinterpret_cast<uint8_t*>(context->peerPayloadBases_[dstRank]);
  auto* dstTokens = reinterpret_cast<int4*>(dstPool + recvPoolHeaderBytes);
  auto* dstMetadata = dstPool + recvPoolMetadataOffset;

  int tokenBegin;
  int tokenEnd;
  getChannelTaskRange(numTokens, numChannels, channel, tokenBegin, tokenEnd);

  int outputOffset = 0;
  for (int token = tokenBegin; token < tokenEnd; ++token) {
    const bool selected = isTokenInRank[token * numRanks + dstRank];
    const int64_t outputIndex = outputBase + outputOffset;
    if (rankThreadId == 0) sendHead[token * numRanks + dstRank] = selected ? static_cast<int>(outputIndex) : -1;
    if (!selected) continue;

    const int4* srcRow = input + static_cast<int64_t>(token) * hiddenInt4;
    int4* dstRow = dstTokens + outputIndex * hiddenInt4;
    for (int hiddenIndex = rankThreadId; hiddenIndex < hiddenInt4; hiddenIndex += threadsPerRank) {
      dstRow[hiddenIndex] = __ldg(srcRow + hiddenIndex);
    }

    auto* metadata = dstMetadata + outputIndex * metadataSlotBytes;
    if (rankThreadId == 0 && context->combineRecvIdx_ != nullptr) {
      context->combineRecvIdx_[token * numRanks + dstRank] = static_cast<int>(outputIndex);
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
          localExpert >= 0
              ? (topkWeights == nullptr ? 1.0f : __ldg(topkWeights + static_cast<int64_t>(token) * numTopk + laneId))
              : 0.0f;
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

  __threadfence_system();
  cooperative_groups::this_grid().sync();
  if (blockIdx.x == 0 && threadIdx.x < WARP_SIZE) barrier(context->channels_, context->rank_, numRanks);
  cooperative_groups::this_grid().sync();

  const auto* localPool = reinterpret_cast<const uint8_t*>(context->peerPayloadBases_[context->rank_]);
  const int globalThreadId = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int gridThreads = static_cast<int>(gridDim.x * blockDim.x);
  for (int token = globalThreadId; token < numRecvTokens; token += gridThreads) {
    const auto* metadata = localPool + recvPoolMetadataOffset + static_cast<int64_t>(token) * metadataSlotBytes;
    if (recvTopkIdx != nullptr || recvTopkWeights != nullptr) {
      const auto* metadataTopkIdx = reinterpret_cast<const int*>(metadata);
      const auto* metadataTopkWeights =
          reinterpret_cast<const float*>(metadata + static_cast<size_t>(numTopk) * sizeof(int));
      for (int topk = 0; topk < numTopk; ++topk) {
        if (recvTopkIdx != nullptr) recvTopkIdx[static_cast<int64_t>(token) * numTopk + topk] = metadataTopkIdx[topk];
        if (recvTopkWeights != nullptr) {
          recvTopkWeights[static_cast<int64_t>(token) * numTopk + topk] = metadataTopkWeights[topk];
        }
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

template <int NumThreads, DispatchLayout Layout>
int maxCooperativeThroughputDispatchBlocks() {
  static int cachedDevice = -1;
  static int cachedMaxBlocks = 0;

  int device;
  MSCCLPP_CUDATHROW(cudaGetDevice(&device));
  if (device != cachedDevice) {
    int blocksPerSm;
    int numSms;
    auto kernel = throughputDispatchKernel<NumThreads, Layout>;
    MSCCLPP_CUDATHROW(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSm, kernel, NumThreads, 0));
    MSCCLPP_CUDATHROW(cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, device));
    cachedDevice = device;
    cachedMaxBlocks = blocksPerSm * numSms;
  }
  return cachedMaxBlocks;
}

void throughputDispatch(int* sendHead, const void* input, const int64_t* topkIdx, const float* topkWeights,
                        const float* inputScales, const bool* isTokenInRank, const int* channelPrefixMatrix,
                        int numTokens, int numRecvTokens, int hiddenInt4, int numTopk, int numExperts, int numScales,
                        int* recvTopkIdx, float* recvTopkWeights, float* recvXScales, int numBlocks,
                        int64_t recvPoolHeaderBytes, int64_t recvPoolMetadataOffset, int64_t metadataSlotBytes,
                        DispatchLayout layout, int maxTokensPerRank, const DeviceContext& context,
                        cudaStream_t stream) {
  constexpr int NumThreads = 512;
  EP_HOST_ASSERT(context.peerPayloadBases_ != nullptr);
  EP_HOST_ASSERT(numBlocks > 0);
  EP_HOST_ASSERT(static_cast<int64_t>(numTopk) * static_cast<int64_t>(sizeof(int) + sizeof(float)) +
                     static_cast<int64_t>(numScales) * static_cast<int64_t>(sizeof(float)) <=
                 metadataSlotBytes);

  const bool rankMajor = layout == DispatchLayout::RANK_MAJOR;
  const int maxBlocks = rankMajor ? maxCooperativeThroughputDispatchBlocks<NumThreads, DispatchLayout::RANK_MAJOR>()
                                  : maxCooperativeThroughputDispatchBlocks<NumThreads, DispatchLayout::TOKEN_MAJOR>();
  EP_HOST_ASSERT(numBlocks <= maxBlocks);
  auto kernel = rankMajor ? throughputDispatchKernel<NumThreads, DispatchLayout::RANK_MAJOR>
                          : throughputDispatchKernel<NumThreads, DispatchLayout::TOKEN_MAJOR>;
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeCooperative;
  attribute.val.cooperative = 1;
  cudaLaunchConfig_t config{dim3(numBlocks), dim3(NumThreads), 0, stream, &attribute, 1};
  MSCCLPP_CUDATHROW(cudaLaunchKernelEx(&config, kernel, sendHead, reinterpret_cast<const int4*>(input), topkIdx,
                                       topkWeights, inputScales, isTokenInRank, channelPrefixMatrix, numTokens,
                                       numRecvTokens, hiddenInt4, numTopk, numExperts, numScales, recvTopkIdx,
                                       recvTopkWeights, recvXScales, recvPoolHeaderBytes, recvPoolMetadataOffset,
                                       metadataSlotBytes, maxTokensPerRank, context.devicePtr_));
}

}  // namespace ep
}  // namespace mscclpp
