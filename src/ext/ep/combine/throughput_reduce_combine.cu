// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP)

#include <cooperative_groups.h>

#include <algorithm>
#include <mscclpp/bulk_device.hpp>
#include <mscclpp/gpu_data_types.hpp>

#include "common/device_helpers.cuh"
#include "exception.hpp"
#include "kernels.hpp"

namespace mscclpp {
namespace ep {

constexpr int COMBINE_TMA_CHUNK_INT4 = 64;
constexpr int COMBINE_TMA_STAGES = 2;
constexpr int COMBINE_TMA_WARPS = 16;
constexpr int COMBINE_TMA_WARPS_WIDE = 14;
constexpr int COMBINE_TMA_WARPS_NARROW = 12;
constexpr int COMBINE_TMA_WIDE_MAX_BLOCKS = 24;

template <int MaxContributors, int NumWarps>
__global__ void __launch_bounds__(NumWarps* WARP_SIZE, 1)
    throughputReduceCombineKernel(int4* output, float* outputTopkWeights, Workload workload,
                                  ThroughputWorkspaceLayout workspace, ThroughputPayloadView payload,
                                  void* dispatchRecvBuffer, void* combineBuffer, const DeviceContext* context) {
#if MSCCLPP_BULK_AVAILABLE
  const int numTopk = workload.numTopk_;
  const TransportView transport(context);
  const int numRanks = context->numRanks_;
  // MaxContributors is min(numTopk, numRanks) rounded up to 2, 4, or 8 slots.
  EP_DEVICE_ASSERT(MaxContributors >= (numTopk < numRanks ? numTopk : numRanks));
  constexpr int ChunkInt4 = COMBINE_TMA_CHUNK_INT4;
  constexpr int NumStages = COMBINE_TMA_STAGES;
  constexpr int ChunkBytes = ChunkInt4 * static_cast<int>(sizeof(int4));
  constexpr int Bf16PerInt4 = mscclpp::bf16x8::Size;
  static_assert(sizeof(mscclpp::bf16x8) == sizeof(int4));

  const int laneId = getLaneId();
  const int warpId = static_cast<int>(threadIdx.x) / WARP_SIZE;
  const int hiddenInt4 = workload.hidden_ / Bf16PerInt4;

  extern __shared__ uint8_t sharedMemory[];
  const size_t warpStageBytes = static_cast<size_t>(NumStages) * MaxContributors * ChunkBytes;
  auto* warpStages = sharedMemory + warpId * warpStageBytes;
  auto* barriers =
      reinterpret_cast<mscclpp::BulkBarrier*>(sharedMemory + static_cast<size_t>(NumWarps) * warpStageBytes) +
      warpId * NumStages;
  auto stage = [&](int stageIdx, int contributorIdx) -> uint8_t* {
    return warpStages + (static_cast<size_t>(stageIdx) * MaxContributors + contributorIdx) * ChunkBytes;
  };
  uint32_t barrierPhases[NumStages] = {};

  if (blockIdx.x == 0) blockPeerBarrier(context->channels_, context->rank_, numRanks);
  cooperative_groups::this_grid().sync();
  if (laneId == 0) {
#pragma unroll
    for (int stageIdx = 0; stageIdx < NumStages; ++stageIdx) barriers[stageIdx].relaxedInit();
    mscclpp::bulkFence();
  }
  __syncwarp();

  const int globalWarp = static_cast<int>(blockIdx.x) * NumWarps + warpId;
  const int totalWarps = static_cast<int>(gridDim.x) * NumWarps;
  const int numChunks = (hiddenInt4 + ChunkInt4 - 1) / ChunkInt4;

  for (int token = globalWarp; token < workload.numTokens_; token += totalWarps) {
    int contributorRanks[MaxContributors];
    int contributorSlots[MaxContributors];
    const ThroughputTokenRoute route = laneId < numTopk
                                           ? workspace.tokenRoutes_[static_cast<size_t>(token) * numTopk + laneId]
                                           : ThroughputTokenRoute{-1, -1};
    const int slot = workspace.recvTokenIndex(route);
    const int numContributors = __popc(warpLaneMask(slot >= 0));
    EP_DEVICE_ASSERT(numContributors <= MaxContributors);
    for (int contributor = 0; contributor < numContributors; ++contributor) {
      contributorRanks[contributor] = warpBroadcast(route.rank_, contributor);
      contributorSlots[contributor] = warpBroadcast(slot, contributor);
    }

    auto* outputRow = output + static_cast<int64_t>(token) * hiddenInt4;
    auto issueLoads = [&](int stageIdx, int chunkOffset, int chunkSize) {
      if (laneId != 0) return;

      const uint32_t chunkBytes = static_cast<uint32_t>(chunkSize * static_cast<int>(sizeof(int4)));
      barriers[stageIdx].arriveAndExpect(chunkBytes * numContributors);
      for (int contributor = 0; contributor < numContributors; ++contributor) {
        const void* peerBuffer = transport.mappedBuffer(combineBuffer, contributorRanks[contributor]);
        const auto* source = payload.data<int4>(peerBuffer) +
                             static_cast<int64_t>(contributorSlots[contributor]) * hiddenInt4 + chunkOffset;
        mscclpp::bulkLoad(stage(stageIdx, contributor), source, chunkBytes, barriers[stageIdx]);
      }
    };

    auto waitStage = [&](int stageIdx) {
      if (laneId == 0) barriers[stageIdx].wait(barrierPhases[stageIdx]);
      __syncwarp();
      mscclpp::bulkFence();
    };

    auto reduceStore = [&](int stageIdx, int chunkOffset, int chunkSize) {
      for (int index = laneId; index < chunkSize; index += WARP_SIZE) {
        mscclpp::f32x8 values;
#pragma unroll
        for (int element = 0; element < Bf16PerInt4; ++element) values.data[element] = 0.0f;
#pragma unroll
        for (int contributor = 0; contributor < MaxContributors; ++contributor) {
          if (contributor >= numContributors) break;
          const auto* peerValues = reinterpret_cast<const mscclpp::bf16x8*>(stage(stageIdx, contributor));
          const auto packed = peerValues[index];
          const auto inputValues = mscclpp::to<mscclpp::f32x8>(packed);
#pragma unroll
          for (int element = 0; element < Bf16PerInt4; ++element) {
            values.data[element] += inputValues.data[element];
          }
        }

        outputRow[chunkOffset + index] = mscclpp::bit_cast<int4>(mscclpp::to<mscclpp::bf16x8>(values));
      }
    };

#pragma unroll
    for (int stageIdx = 0; stageIdx < NumStages - 1; ++stageIdx) {
      if (stageIdx < numChunks) {
        const int chunkOffset = stageIdx * ChunkInt4;
        const int chunkSize = hiddenInt4 - chunkOffset < ChunkInt4 ? hiddenInt4 - chunkOffset : ChunkInt4;
        issueLoads(stageIdx, chunkOffset, chunkSize);
      }
    }
    for (int chunk = 0; chunk < numChunks; ++chunk) {
      const int stageIdx = chunk % NumStages;
      const int chunkOffset = chunk * ChunkInt4;
      const int chunkSize = hiddenInt4 - chunkOffset < ChunkInt4 ? hiddenInt4 - chunkOffset : ChunkInt4;
      const int nextChunk = chunk + NumStages - 1;
      if (nextChunk < numChunks) {
        const int nextStage = nextChunk % NumStages;
        const int nextOffset = nextChunk * ChunkInt4;
        const int nextSize = hiddenInt4 - nextOffset < ChunkInt4 ? hiddenInt4 - nextOffset : ChunkInt4;
        issueLoads(nextStage, nextOffset, nextSize);
      }
      waitStage(stageIdx);
      reduceStore(stageIdx, chunkOffset, chunkSize);
      __syncwarp();
    }

    if (outputTopkWeights != nullptr && laneId < numTopk) {
      float weight = 0.0f;
#pragma unroll
      for (int contributor = 0; contributor < MaxContributors; ++contributor) {
        if (contributor >= numContributors) break;
        const void* peerBuffer = transport.mappedBuffer(dispatchRecvBuffer, contributorRanks[contributor]);
        const auto* weights = payload.topKValues(peerBuffer, contributorSlots[contributor]);
        weight += __ldg(weights + laneId);
      }
      outputTopkWeights[static_cast<int64_t>(token) * numTopk + laneId] = weight;
    }
  }
#endif  // MSCCLPP_BULK_AVAILABLE
}

template <int MaxContributors, int NumWarps>
void launchThroughputCombine(void* output, float* outputTopkWeights, const Workload& workload,
                             const ThroughputWorkspaceLayout& workspace, const ThroughputPayloadView& payload,
                             void* dispatchRecvBuffer, void* combineBuffer, const DeviceContext& context, int numBlocks,
                             cudaStream_t stream) {
  constexpr int NumStages = COMBINE_TMA_STAGES;
  constexpr int ChunkInt4 = COMBINE_TMA_CHUNK_INT4;
  constexpr int NumThreads = NumWarps * WARP_SIZE;
  constexpr size_t SharedBytes =
      static_cast<size_t>(NumWarps) * NumStages * MaxContributors * ChunkInt4 * sizeof(int4) +
      static_cast<size_t>(NumWarps) * NumStages * sizeof(mscclpp::BulkBarrier);
  auto kernel = throughputReduceCombineKernel<MaxContributors, NumWarps>;
  static thread_local KernelConfigCache kernelConfig;
  const int residentBlocks = configureKernel(kernel, NumThreads, SharedBytes, context, kernelConfig);
  EP_HOST_ASSERT(numBlocks <= residentBlocks);
  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeCooperative;
  attribute.val.cooperative = 1;
  cudaLaunchConfig_t config{dim3(numBlocks), dim3(NumThreads), SharedBytes, stream, &attribute, 1};
  MSCCLPP_CUDATHROW(cudaLaunchKernelEx(&config, kernel, static_cast<int4*>(output), outputTopkWeights, workload,
                                       workspace, payload, dispatchRecvBuffer, combineBuffer, context.devicePtr_));
}

void throughputReduceCombine(void* output, float* outputTopkWeights, const Workload& workload,
                             const ThroughputWorkspaceLayout& workspace, const ThroughputPayloadView& payload,
                             void* dispatchRecvBuffer, void* combineBuffer, const DeviceContext& context, int numBlocks,
                             cudaStream_t stream) {
  EP_HOST_ASSERT(output != nullptr || workload.numTokens_ == 0);
  EP_HOST_ASSERT(workspace.tokenRoutes_ != nullptr && workspace.rankOffsets_ != nullptr);
  EP_HOST_ASSERT(workspace.numRecvTokens_ != nullptr && workspace.recvCounts_ != nullptr);
  EP_HOST_ASSERT(dispatchRecvBuffer != nullptr && combineBuffer != nullptr && context.peerBufferBases_ != nullptr);
  EP_HOST_ASSERT(context.channels_ != nullptr);
  EP_HOST_ASSERT(numBlocks > 0);
  EP_HOST_ASSERT(isSupportedRanks(context.numRanks_));

  const int numTopk = workload.numTopk_;
  EP_HOST_ASSERT(numTopk > 0 && numTopk <= MaxNumTopk);
  // Rank-deduplicated routing needs at most top-k contributors, independent of rank count.
  const int maxContributors = std::min(numTopk, context.numRanks_);
  const bool useWideKernel = numBlocks <= COMBINE_TMA_WIDE_MAX_BLOCKS;
  auto launch = launchThroughputCombine<2, COMBINE_TMA_WARPS>;
  if (maxContributors > 4) {
    launch = useWideKernel ? launchThroughputCombine<8, COMBINE_TMA_WARPS_WIDE>
                           : launchThroughputCombine<8, COMBINE_TMA_WARPS_NARROW>;
  } else if (maxContributors > 2) {
    launch = launchThroughputCombine<4, COMBINE_TMA_WARPS>;
  }
  launch(output, outputTopkWeights, workload, workspace, payload, dispatchRecvBuffer, combineBuffer, context, numBlocks,
         stream);
}

}  // namespace ep
}  // namespace mscclpp
