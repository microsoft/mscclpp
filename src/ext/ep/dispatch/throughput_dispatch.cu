// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cooperative_groups.h>

#include "common/device_helpers.cuh"
#include "exception.hpp"
#include "kernels.hpp"

namespace mscclpp {
namespace ep {
namespace {

constexpr int ThroughputDispatchWarps = 16;
constexpr int ThroughputDispatchThreads = ThroughputDispatchWarps * WARP_SIZE;

}  // namespace

MSCCLPP_HOST_DEVICE_INLINE constexpr int throughputWarpsPerGroup(int numTokens, int numBlocks, int hiddenInt4) {
  constexpr int MaxVectorsPerThread = 16;
  int groups = 1;
  while (groups < ThroughputDispatchWarps && numTokens > numBlocks * groups) groups *= 2;
  // Balance independent tokens without making wide rows a long per-lane copy loop.
  int rowWarps = 1;
  while (rowWarps < ThroughputDispatchWarps && hiddenInt4 > rowWarps * WARP_SIZE * MaxVectorsPerThread) rowWarps *= 2;
  const int tokenWarps = ThroughputDispatchWarps / groups;
  return tokenWarps > rowWarps ? tokenWarps : rowWarps;
}

__global__ void synchronizeThroughputPeersKernel(const DeviceContext* context) {
  blockPeerBarrier(context->channels_, context->rank_, context->numRanks_);
}

void throughputSynchronizePeers(const DeviceContext& context, cudaStream_t stream) {
  const int numThreads = std::max(context.numRanks_, WARP_SIZE);
  synchronizeThroughputPeersKernel<<<1, numThreads, 0, stream>>>(context.devicePtr_);
  MSCCLPP_CUDATHROW(cudaGetLastError());
}

template <int NumThreads, DispatchLayout Layout, DispatchDataType DataType>
__global__ void __launch_bounds__(NumThreads, 1)
    throughputDispatchKernel(void* output, int* outputTopkIdx, float* outputTopkWeights, float* outputScales,
                             const int4* input, const int64_t* topkIdx, const float* topkWeights,
                             const float* inputScales, Workload workload, ThroughputWorkspaceLayout workspace,
                             ThroughputPayloadView payload, void* recvBuffer, const DeviceContext* context) {
  static_assert(NumThreads == ThroughputDispatchThreads);
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
  EP_DEVICE_ASSERT(numTopk > 0 && numTopk <= WARP_SIZE);

  for (int token = firstToken; token < workload.numTokens_; token += tokenStride) {
    const ThroughputTokenRoute route = laneId < numTopk
                                           ? workspace.tokenRoutes_[static_cast<size_t>(token) * numTopk + laneId]
                                           : ThroughputTokenRoute{-1, -1};
    const int laneRank = route.rank_;
    const int laneSlot = workspace.recvTokenIndex(route);
    void* laneBuffer = laneSlot >= 0 ? transport.mappedBuffer(recvBuffer, laneRank) : nullptr;
    // Valid routes form a contiguous prefix, independent of global rank IDs.
    const int numDestinations = __popc(warpLaneMask(laneSlot >= 0));
    if (numDestinations == 0) continue;
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
      for (int destinationLane = 0; destinationLane < numDestinations; ++destinationLane) {
        auto* dstRow = warpBroadcast(laneRow, destinationLane);
        if (valid) dstRow[hiddenIndex] = value;
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
        if (laneId < numScales) firstScale = inputScales[static_cast<int64_t>(token) * numScales + laneId];
        if (laneId + WARP_SIZE < numScales)
          secondScale = inputScales[static_cast<int64_t>(token) * numScales + laneId + WARP_SIZE];
      }
      for (int destinationLane = 0; destinationLane < numDestinations; ++destinationLane) {
        const int dstRank = warpBroadcast(laneRank, destinationLane);
        const int outputIndex = warpBroadcast(laneSlot, destinationLane);
        void* dstBuffer = warpBroadcast(laneBuffer, destinationLane);
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
              metadataScales[scale] = inputScales[static_cast<int64_t>(token) * numScales + scale];
            }
          }
        }
      }
    }
  }

  // Join all writers before block 0 publishes their stores with system-release signals.
  cooperative_groups::this_grid().sync();
  if (blockIdx.x == 0) blockPeerBarrier(context->channels_, context->rank_, numRanks);
  // All receiving blocks must wait for block 0's peer acquires.
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
  auto kernel = throughputDispatchKernel<NumThreads, Layout, DataType>;
  return configureKernel(kernel, NumThreads, 0, context, kernelConfig);
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
  EP_HOST_ASSERT(workspace.tokenRoutes_ != nullptr && workspace.rankOffsets_ != nullptr);
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
