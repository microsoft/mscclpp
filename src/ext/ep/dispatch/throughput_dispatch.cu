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

constexpr int ThroughputDispatchThreads = DispatchNWarps * WARP_SIZE;

}  // namespace

MSCCLPP_HOST_DEVICE_INLINE constexpr int throughputWarpsPerGroup(int numTokens, int numBlocks, int hiddenInt4) {
  constexpr int MaxVectorsPerThread = 16;
  int groups = 1;
  while (groups < DispatchNWarps && numTokens > numBlocks * groups) groups *= 2;
  // Balance independent tokens without making wide rows a long per-lane copy loop.
  int rowWarps = 1;
  while (rowWarps < DispatchNWarps && hiddenInt4 > rowWarps * WARP_SIZE * MaxVectorsPerThread) rowWarps *= 2;
  const int tokenWarps = DispatchNWarps / groups;
  return tokenWarps > rowWarps ? tokenWarps : rowWarps;
}

__global__ void exchangeThroughputCountsKernel(ThroughputWorkspaceLayout workspace, Workload workload,
                                               const DeviceContext* context) {
  const int numRanks = context->numRanks_;
  const int threadId = static_cast<int>(threadIdx.x);
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
    const int rankPrefix = context->rank_ > 0 ? localRankCounts[(context->rank_ - 1) * numRanks + threadId] : 0;
    workspace.rankOffsets_[threadId] =
        workload.outputLayout_ == DispatchLayout::RANK_MAJOR ? context->rank_ * workload.maxTokensPerRank_ : rankPrefix;
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
  // Dispatch and combine use local rank offsets; peers no longer read these prefixes.
}

void throughputExchangeCounts(const ThroughputWorkspaceLayout& workspace, const Workload& workload,
                              const DeviceContext& context, cudaStream_t stream) {
  constexpr int NumThreads = ThroughputCountThreads;
  EP_HOST_ASSERT(workload.numExperts_ % context.numRanks_ == 0);
  EP_HOST_ASSERT(workload.numExperts_ / context.numRanks_ <= NumThreads && context.numRanks_ <= NumThreads);

  exchangeThroughputCountsKernel<<<1, NumThreads, 0, stream>>>(workspace, workload, context.devicePtr_);
  MSCCLPP_CUDATHROW(cudaGetLastError());
}

__global__ void synchronizeThroughputPeersKernel(const DeviceContext* context) {
  barrier(context->channels_, context->rank_, context->numRanks_);
}

void throughputSynchronizePeers(const DeviceContext& context, cudaStream_t stream) {
  synchronizeThroughputPeersKernel<<<1, WARP_SIZE, 0, stream>>>(context.devicePtr_);
  MSCCLPP_CUDATHROW(cudaGetLastError());
}

template <int NumThreads, DispatchLayout Layout, DispatchDataType DataType>
__global__ void __launch_bounds__(NumThreads, 1)
    throughputDispatchKernel(void* output, int* outputTopkIdx, float* outputTopkWeights, float* outputScales,
                             const int4* input, const int64_t* topkIdx, const float* topkWeights,
                             const float* inputScales, Workload workload, ThroughputWorkspaceLayout workspace,
                             ThroughputPayloadView payload, void* recvBuffer, const DeviceContext* context) {
  static_assert(NumThreads == DispatchNWarps * WARP_SIZE);
  const int numTopk = workload.numTopk_;
  const int maxTokensPerRank = workload.maxTokensPerRank_;
  const int hiddenInt4 = workload.hidden_ * dispatchElementBytes(DataType) / sizeof(int4);
  const int numScales = dispatchNumScales(DataType, workload.hidden_);
  const TransportView transport(context);
  const int numRanks = context->numRanks_;
  const int receivedTokens = *workspace.numRecvTokens_;
  EP_DEVICE_ASSERT(receivedTokens >= 0 && receivedTokens <= numRanks * maxTokensPerRank);
  EP_DEVICE_ASSERT(output != nullptr || receivedTokens == 0);
  const int numBlocks = static_cast<int>(gridDim.x);
  const int threadId = static_cast<int>(threadIdx.x);
  const int laneId = getLaneId();
  const int warpsPerGroup = throughputWarpsPerGroup(workload.numTokens_, numBlocks, hiddenInt4);
  const int threadsPerGroup = warpsPerGroup * WARP_SIZE;
  const int groupsPerBlock = NumThreads / threadsPerGroup;
  const int groupId = threadId / threadsPerGroup;
  const int groupThreadId = threadId % threadsPerGroup;
  const int tokenStride = numBlocks * groupsPerBlock;
  const int firstToken = static_cast<int>(blockIdx.x) * groupsPerBlock + groupId;
  const int expertsPerRank = workload.numExperts_ / numRanks;
  EP_DEVICE_ASSERT(numRanks <= WARP_SIZE);
  void* laneBuffer = laneId < numRanks ? transport.mappedBuffer(recvBuffer, laneId) : nullptr;

  for (int token = firstToken; token < workload.numTokens_; token += tokenStride) {
    const int laneSlot = laneId < numRanks ? workspace.recvTokenIndex(token, laneId, numRanks) : -1;
    const unsigned destinationMask = __ballot_sync(0xffffffffu, laneSlot >= 0);
    if (destinationMask == 0) continue;
    int4* laneRow =
        laneSlot >= 0 ? payload.data<int4>(laneBuffer) + static_cast<int64_t>(laneSlot) * hiddenInt4 : nullptr;

    // Load each input vector once, then fan it out to the token's distinct destinations.
    const int4* srcRow = input + static_cast<int64_t>(token) * hiddenInt4;
    // Keep tail iterations warp-uniform because destination pointers use shuffles.
    for (int hiddenBase = groupThreadId / WARP_SIZE * WARP_SIZE; hiddenBase < hiddenInt4;
         hiddenBase += threadsPerGroup) {
      const int hiddenIndex = hiddenBase + laneId;
      const bool valid = hiddenIndex < hiddenInt4;
      int4 value{};
      if (valid) value = __ldg(srcRow + hiddenIndex);
      unsigned destinations = destinationMask;
      while (destinations != 0) {
        const int dstRank = __ffs(static_cast<int>(destinations)) - 1;
        auto* dstRow = warpBroadcast(laneRow, dstRank);
        if (valid) dstRow[hiddenIndex] = value;
        destinations &= destinations - 1u;
      }
    }

    // A complete warp owns metadata, regardless of the rank count.
    if (groupThreadId < WARP_SIZE) {
      const int64_t expert = laneId < numTopk ? __ldg(topkIdx + static_cast<int64_t>(token) * numTopk + laneId) : -1;
      const float weight = laneId < numTopk && topkWeights != nullptr
                               ? __ldg(topkWeights + static_cast<int64_t>(token) * numTopk + laneId)
                               : 1.0f;
      const bool cacheScales = DataType == DispatchDataType::FP8_E4M3 && numScales <= 2 * WARP_SIZE;
      float firstScale = 0.0f;
      float secondScale = 0.0f;
      if (cacheScales) {
        if (laneId < numScales) firstScale = __ldg(inputScales + static_cast<int64_t>(token) * numScales + laneId);
        if (laneId + WARP_SIZE < numScales)
          secondScale = __ldg(inputScales + static_cast<int64_t>(token) * numScales + laneId + WARP_SIZE);
      }
      unsigned destinations = destinationMask;
      while (destinations != 0) {
        const int dstRank = __ffs(static_cast<int>(destinations)) - 1;
        const int outputIndex = __shfl_sync(0xffffffffu, laneSlot, dstRank);
        void* dstBuffer = transport.mappedBuffer(recvBuffer, dstRank);
        if (laneId < numTopk) {
          const int expertBegin = dstRank * expertsPerRank;
          const int expertEnd = expertBegin + expertsPerRank;
          const int localExpert =
              expert >= expertBegin && expert < expertEnd ? static_cast<int>(expert) - expertBegin : -1;
          payload.topKIndices(dstBuffer, outputIndex)[laneId] = localExpert;
          payload.topKValues(dstBuffer, outputIndex)[laneId] = localExpert >= 0 ? weight : 0.0f;
        }
        if constexpr (DataType == DispatchDataType::FP8_E4M3) {
          auto* metadataScales = payload.scaleFactors(dstBuffer, outputIndex);
          if (cacheScales) {
            if (laneId < numScales) metadataScales[laneId] = firstScale;
            if (laneId + WARP_SIZE < numScales) metadataScales[laneId + WARP_SIZE] = secondScale;
          } else {
            for (int scale = laneId; scale < numScales; scale += WARP_SIZE) {
              metadataScales[scale] = __ldg(inputScales + static_cast<int64_t>(token) * numScales + scale);
            }
          }
        }
        destinations &= destinations - 1u;
      }
    }
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

template <int NumThreads, DispatchLayout Layout, DispatchDataType DataType>
int maxCooperativeThroughputDispatchBlocks(const DeviceContext& context) {
  static thread_local KernelConfigCache kernelConfig;
  return configureKernel(throughputDispatchKernel<NumThreads, Layout, DataType>, NumThreads, 0, context, kernelConfig);
}

int maxCooperativeThroughputDispatchBlocks(DispatchLayout layout, const DeviceContext& context) {
  if (layout == DispatchLayout::RANK_MAJOR) {
    return std::min(maxCooperativeThroughputDispatchBlocks<ThroughputDispatchThreads, DispatchLayout::RANK_MAJOR,
                                                           DispatchDataType::BF16>(context),
                    maxCooperativeThroughputDispatchBlocks<ThroughputDispatchThreads, DispatchLayout::RANK_MAJOR,
                                                           DispatchDataType::FP8_E4M3>(context));
  }
  return std::min(maxCooperativeThroughputDispatchBlocks<ThroughputDispatchThreads, DispatchLayout::TOKEN_MAJOR,
                                                         DispatchDataType::BF16>(context),
                  maxCooperativeThroughputDispatchBlocks<ThroughputDispatchThreads, DispatchLayout::TOKEN_MAJOR,
                                                         DispatchDataType::FP8_E4M3>(context));
}

void throughputDispatch(void* output, int* outputTopkIdx, float* outputTopkWeights, float* outputScales,
                        const void* input, const int64_t* topkIdx, const float* topkWeights, const float* inputScales,
                        const Workload& workload, const ThroughputWorkspaceLayout& workspace,
                        const ThroughputPayloadView& payload, void* recvBuffer, const DeviceContext& context,
                        int numBlocks, cudaStream_t stream) {
  constexpr int NumThreads = ThroughputDispatchThreads;
  EP_HOST_ASSERT(recvBuffer != nullptr && context.peerBufferBases_ != nullptr);
  EP_HOST_ASSERT(workspace.numRecvTokens_ != nullptr && workspace.recvCounts_ != nullptr);
  EP_HOST_ASSERT(numBlocks > 0);
  EP_HOST_ASSERT(isSupportedDispatchDataType(workload.dispatchDataType_));
  EP_HOST_ASSERT(payload.metadataBytes(dispatchNumScales(workload.dispatchDataType_, workload.hidden_)) <=
                 payload.metadataSlotBytes_);

  const bool rankMajor = workload.outputLayout_ == DispatchLayout::RANK_MAJOR;
  const bool fp8 = workload.dispatchDataType_ == DispatchDataType::FP8_E4M3;
  EP_HOST_ASSERT(numBlocks <= maxCooperativeThroughputDispatchBlocks(workload.outputLayout_, context));
  auto kernel =
      rankMajor ? (fp8 ? throughputDispatchKernel<NumThreads, DispatchLayout::RANK_MAJOR, DispatchDataType::FP8_E4M3>
                       : throughputDispatchKernel<NumThreads, DispatchLayout::RANK_MAJOR, DispatchDataType::BF16>)
                : (fp8 ? throughputDispatchKernel<NumThreads, DispatchLayout::TOKEN_MAJOR, DispatchDataType::FP8_E4M3>
                       : throughputDispatchKernel<NumThreads, DispatchLayout::TOKEN_MAJOR, DispatchDataType::BF16>);
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeCooperative;
  attribute.val.cooperative = 1;
  cudaLaunchConfig_t config{dim3(numBlocks), dim3(NumThreads), 0, stream, &attribute, 1};
  MSCCLPP_CUDATHROW(cudaLaunchKernelEx(&config, kernel, output, outputTopkIdx, outputTopkWeights, outputScales,
                                       reinterpret_cast<const int4*>(input), topkIdx, topkWeights, inputScales,
                                       workload, workspace, payload, recvBuffer, context.devicePtr_));
}

}  // namespace ep
}  // namespace mscclpp
