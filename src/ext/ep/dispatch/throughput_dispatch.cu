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
namespace {

constexpr int ThroughputDispatchThreads = 512;

}  // namespace

MSCCLPP_HOST_DEVICE_INLINE void getChannelTaskRange(int numTokens, int numChannels, int channel, int& tokenBegin,
                                                    int& tokenEnd) {
  tokenBegin = static_cast<int>(static_cast<int64_t>(numTokens) * channel / numChannels);
  tokenEnd = static_cast<int>(static_cast<int64_t>(numTokens) * (channel + 1) / numChannels);
}

__global__ void exchangeThroughputCountsKernel(ThroughputWorkspaceLayout workspace, Workload workload, int numChannels,
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

    const int numExpertsPerRank = workload.numExperts_ / numRanks;
    if (threadId < numRanks) {
      auto* peerRankCounts = reinterpret_cast<int*>(context->peerBufferBases_[threadId]);
      auto* peerExpertCounts = peerRankCounts + numRanks * numRanks;
      for (int dstRank = 0; dstRank < numRanks; ++dstRank) {
        peerRankCounts[context->rank_ * numRanks + dstRank] = workspace.numTokensPerRank_[dstRank];
      }
#pragma unroll
      for (int localExpert = 0; localExpert < numExpertsPerRank; ++localExpert) {
        peerExpertCounts[context->rank_ * numExpertsPerRank + localExpert] =
            workspace.numTokensPerExpert_[threadId * numExpertsPerRank + localExpert];
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
        *workspace.numRecvTokens_ = localRankCounts[(numRanks - 1) * numRanks + context->rank_];
    }

    auto* localExpertCounts = localRankCounts + numRanks * numRanks;
    if (threadId < numExpertsPerRank) {
      int count = 0;
      for (int srcRank = 0; srcRank < numRanks; ++srcRank) {
        count += localExpertCounts[srcRank * numExpertsPerRank + threadId];
      }
      if (workload.outputLayout_ == DispatchLayout::TOKEN_MAJOR) workspace.recvCounts_[threadId] = count;
    }
    __syncthreads();

    if (workload.outputLayout_ == DispatchLayout::RANK_MAJOR && threadId < numRanks) {
      const int prefix = localRankCounts[threadId * numRanks + context->rank_];
      const int previous = threadId == 0 ? 0 : localRankCounts[(threadId - 1) * numRanks + context->rank_];
      workspace.recvCounts_[threadId] = prefix - previous;
    }
    for (int index = threadId; index < numRanks * numRanks; index += numThreads) {
      workspace.rankPrefixMatrix_[index] = localRankCounts[index];
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
    getChannelTaskRange(workload.numTokens_, numChannels, channel, tokenBegin, tokenEnd);

    int count = 0;
    for (int token = tokenBegin + laneId; token < tokenEnd; token += WARP_SIZE) {
      count += workspace.tokenRankMask_[token * numRanks + dstRank];
    }
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) count += __shfl_down_sync(0xffffffffu, count, offset);
    if (laneId == 0) workspace.channelPrefixMatrix_[dstRank * numChannels + channel] = count;
  }
  __syncthreads();

  if (threadId == 0) {
#pragma unroll
    for (int channel = 1; channel < numChannels; ++channel) {
      workspace.channelPrefixMatrix_[dstRank * numChannels + channel] +=
          workspace.channelPrefixMatrix_[dstRank * numChannels + channel - 1];
    }
  }
}

void throughputExchangeCounts(const ThroughputWorkspaceLayout& workspace, const Workload& workload,
                              const DeviceContext& context, int numChannels, cudaStream_t stream) {
  constexpr int NumThreads = ThroughputCountThreads;
  EP_HOST_ASSERT(workload.numExperts_ % context.numRanks_ == 0);
  EP_HOST_ASSERT(workload.numExperts_ / context.numRanks_ <= NumThreads && context.numRanks_ <= NumThreads);
  EP_HOST_ASSERT(numChannels > 0);

  exchangeThroughputCountsKernel<<<1 + context.numRanks_, NumThreads, 0, stream>>>(workspace, workload, numChannels,
                                                                                   context.devicePtr_);
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

void throughputPublishCachedPrefix(const ThroughputWorkspaceLayout& workspace, const DeviceContext& context,
                                   cudaStream_t stream) {
  publishCachedThroughputPrefixKernel<<<1, ThroughputCountThreads, 0, stream>>>(workspace.rankPrefixMatrix_,
                                                                                context.devicePtr_);
  MSCCLPP_CUDATHROW(cudaGetLastError());
}

template <int NumThreads, DispatchLayout Layout>
__global__ void __launch_bounds__(NumThreads, 1)
    throughputDispatchKernel(void* output, int* outputTopkIdx, float* outputTopkWeights, float* outputScales,
                             const int4* input, const int64_t* topkIdx, const float* topkWeights,
                             const float* inputScales, Workload workload, ThroughputWorkspaceLayout workspace,
                             void* recvBuffer, const DeviceContext* context) {
  const int numTopk = workload.numTopk_;
  const int maxTokensPerRank = workload.maxTokensPerRank_;
  const int hiddenInt4 = workload.hidden_ * dispatchElementBytes(workload.dispatchDataType_) / sizeof(int4);
  const int numScales = dispatchNumScales(workload.dispatchDataType_, workload.hidden_);
  const ThroughputPayloadView payload(numTopk);
  const TransportView transport(context);
  const int numRanks = context->numRanks_;
  const int receivedTokens = *workspace.numRecvTokens_;
  EP_DEVICE_ASSERT(receivedTokens >= 0 && receivedTokens <= numRanks * maxTokensPerRank);
  EP_DEVICE_ASSERT(output != nullptr || receivedTokens == 0);
  const int numChannels = static_cast<int>(gridDim.x);
  const int channel = static_cast<int>(blockIdx.x);
  const int threadId = static_cast<int>(threadIdx.x);
  const int threadsPerRank = NumThreads / numRanks;
  const int dstRank = threadId / threadsPerRank;
  const int rankThreadId = threadId % threadsPerRank;
  const int laneId = threadId % WARP_SIZE;
  const int expertsPerRank = workload.numExperts_ / numRanks;
  EP_DEVICE_ASSERT(numRanks <= WARP_SIZE);
  EP_DEVICE_ASSERT(NumThreads % numRanks == 0);

  const int* dstRankPrefix = reinterpret_cast<const int*>(context->peerBufferBases_[dstRank]);
  const int rankOffset = context->rank_ > 0 ? dstRankPrefix[(context->rank_ - 1) * numRanks + dstRank] : 0;
  const int channelOffset = channel > 0 ? workspace.channelPrefixMatrix_[dstRank * numChannels + channel - 1] : 0;
  const int64_t rankBase =
      Layout == DispatchLayout::RANK_MAJOR ? static_cast<int64_t>(context->rank_) * maxTokensPerRank : rankOffset;
  const int64_t outputBase = rankBase + channelOffset;
  void* dstBuffer = transport.mappedBuffer(recvBuffer, dstRank);
  auto* dstTokens = payload.data<int4>(dstBuffer);

  int tokenBegin;
  int tokenEnd;
  getChannelTaskRange(workload.numTokens_, numChannels, channel, tokenBegin, tokenEnd);

  int outputOffset = 0;
  for (int token = tokenBegin; token < tokenEnd; ++token) {
    const bool selected = workspace.tokenRankMask_[token * numRanks + dstRank];
    const int64_t outputIndex = outputBase + outputOffset;
    if (rankThreadId == 0)
      workspace.recvTokenIndices_[token * numRanks + dstRank] = selected ? static_cast<int>(outputIndex) : -1;
    if (!selected) continue;

    const int4* srcRow = input + static_cast<int64_t>(token) * hiddenInt4;
    int4* dstRow = dstTokens + outputIndex * hiddenInt4;
    for (int hiddenIndex = rankThreadId; hiddenIndex < hiddenInt4; hiddenIndex += threadsPerRank) {
      dstRow[hiddenIndex] = __ldg(srcRow + hiddenIndex);
    }

    if (topkIdx != nullptr && laneId < numTopk && rankThreadId < WARP_SIZE) {
      auto* metadataTopkIdx = payload.topKIndices(dstBuffer, outputIndex);
      auto* metadataTopkWeights = payload.topKValues(dstBuffer, outputIndex);
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
      auto* metadataScales = payload.scaleFactors(dstBuffer, outputIndex);
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

  const void* localTokens = payload.data<int4>(recvBuffer);
  const int globalThreadId = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int gridThreads = static_cast<int>(gridDim.x * blockDim.x);
  const int outputRows = Layout == DispatchLayout::RANK_MAJOR ? numRanks * maxTokensPerRank : receivedTokens;
  if (output != nullptr && output != localTokens) {
    copyThroughputRows(output, localTokens, outputRows, hiddenInt4 * sizeof(int4), workspace.recvCounts_,
                       maxTokensPerRank, Layout == DispatchLayout::RANK_MAJOR, globalThreadId, gridThreads);
  }
  for (int token = globalThreadId; token < outputRows; token += gridThreads) {
    if (!isActiveThroughputRow(token, workspace.recvCounts_, maxTokensPerRank, Layout == DispatchLayout::RANK_MAJOR))
      continue;
    if (outputTopkIdx != nullptr || outputTopkWeights != nullptr) {
      const auto* metadataTopkIdx = payload.topKIndices(recvBuffer, token);
      const auto* metadataTopkWeights = payload.topKValues(recvBuffer, token);
      for (int topk = 0; topk < numTopk; ++topk) {
        if (outputTopkIdx != nullptr)
          outputTopkIdx[static_cast<int64_t>(token) * numTopk + topk] = metadataTopkIdx[topk];
        if (outputTopkWeights != nullptr) {
          outputTopkWeights[static_cast<int64_t>(token) * numTopk + topk] = metadataTopkWeights[topk];
        }
      }
    }
    if (outputScales != nullptr) {
      const auto* metadataScales = payload.scaleFactors(recvBuffer, token);
      for (int scale = 0; scale < numScales; ++scale) {
        outputScales[static_cast<int64_t>(token) * numScales + scale] = metadataScales[scale];
      }
    }
  }
}

template <int NumThreads, DispatchLayout Layout>
int maxCooperativeThroughputDispatchBlocks(const DeviceContext& context) {
  static thread_local KernelConfigCache kernelConfig;
  return configureKernel(throughputDispatchKernel<NumThreads, Layout>, NumThreads, 0, context, kernelConfig);
}

int maxCooperativeThroughputDispatchBlocks(DispatchLayout layout, const DeviceContext& context) {
  return layout == DispatchLayout::RANK_MAJOR
             ? maxCooperativeThroughputDispatchBlocks<ThroughputDispatchThreads, DispatchLayout::RANK_MAJOR>(context)
             : maxCooperativeThroughputDispatchBlocks<ThroughputDispatchThreads, DispatchLayout::TOKEN_MAJOR>(context);
}

void throughputDispatch(void* output, int* outputTopkIdx, float* outputTopkWeights, float* outputScales,
                        const void* input, const int64_t* topkIdx, const float* topkWeights, const float* inputScales,
                        const Workload& workload, const ThroughputWorkspaceLayout& workspace, void* recvBuffer,
                        const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  constexpr int NumThreads = ThroughputDispatchThreads;
  EP_HOST_ASSERT(recvBuffer != nullptr && context.peerBufferBases_ != nullptr);
  EP_HOST_ASSERT(workspace.numRecvTokens_ != nullptr && workspace.recvCounts_ != nullptr);
  EP_HOST_ASSERT(numBlocks > 0);
  EP_HOST_ASSERT(isSupportedDispatchDataType(workload.dispatchDataType_));
  const ThroughputPayloadView payload(workload.numTopk_);
  EP_HOST_ASSERT(payload.metadataBytes(dispatchNumScales(workload.dispatchDataType_, workload.hidden_)) <=
                 ThroughputPayloadView::MetadataSlotBytes);

  const bool rankMajor = workload.outputLayout_ == DispatchLayout::RANK_MAJOR;
  EP_HOST_ASSERT(numBlocks <= maxCooperativeThroughputDispatchBlocks(workload.outputLayout_, context));
  auto kernel = rankMajor ? throughputDispatchKernel<NumThreads, DispatchLayout::RANK_MAJOR>
                          : throughputDispatchKernel<NumThreads, DispatchLayout::TOKEN_MAJOR>;
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeCooperative;
  attribute.val.cooperative = 1;
  cudaLaunchConfig_t config{dim3(numBlocks), dim3(NumThreads), 0, stream, &attribute, 1};
  MSCCLPP_CUDATHROW(cudaLaunchKernelEx(&config, kernel, output, outputTopkIdx, outputTopkWeights, outputScales,
                                       reinterpret_cast<const int4*>(input), topkIdx, topkWeights, inputScales,
                                       workload, workspace, recvBuffer, context.devicePtr_));
}

}  // namespace ep
}  // namespace mscclpp
