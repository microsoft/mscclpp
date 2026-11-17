// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP)

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
constexpr int COMBINE_TMA_MAX_STAGE_CONTRIBUTORS = 8;
constexpr int COMBINE_TMA_WARPS = 16;
constexpr int COMBINE_TMA_WARPS_WIDE = 14;
constexpr int COMBINE_TMA_WARPS_NARROW = 12;
constexpr int COMBINE_TMA_WIDE_MAX_BLOCKS = 24;

template <int MaxContributors, int NumWarps>
__global__ void __launch_bounds__(NumWarps* WARP_SIZE, 1)
    throughputReduceCombineKernel(int4* output, Workload workload, ThroughputWorkspaceLayout workspace,
                                  ThroughputPayloadView payload, void* combineBuffer, const DeviceContext* context) {
#if MSCCLPP_BULK_AVAILABLE
  const int numTopk = workload.numTopk_;
  const TransportView transport(context);
  const int numRanks = context->numRanks_;
  // MaxContributors is min(numTopk, numRanks) rounded up to 2, 4, 8, or 16; shared memory caps stage slots at 8.
  EP_DEVICE_ASSERT(MaxContributors >= (numTopk < numRanks ? numTopk : numRanks));
  constexpr int ChunkInt4 = COMBINE_TMA_CHUNK_INT4;
  constexpr int NumStages = COMBINE_TMA_STAGES;
  constexpr int StageContributors =
      MaxContributors < COMBINE_TMA_MAX_STAGE_CONTRIBUTORS ? MaxContributors : COMBINE_TMA_MAX_STAGE_CONTRIBUTORS;
  constexpr int MaxRounds = (MaxContributors + StageContributors - 1) / StageContributors;
  constexpr int VectorsPerLane = (ChunkInt4 + WARP_SIZE - 1) / WARP_SIZE;
  constexpr int ChunkBytes = ChunkInt4 * static_cast<int>(sizeof(int4));
  constexpr int Bf16PerInt4 = mscclpp::bf16x8::Size;
  static_assert(sizeof(mscclpp::bf16x8) == sizeof(int4));

  const int laneId = getLaneId();
  const int warpId = static_cast<int>(threadIdx.x) / WARP_SIZE;
  const int hiddenInt4 = workload.hidden_ / Bf16PerInt4;

  extern __shared__ __align__(128) uint8_t sharedMemory[];
  const size_t warpStageBytes = static_cast<size_t>(NumStages) * StageContributors * ChunkBytes;
  auto* warpStages = sharedMemory + warpId * warpStageBytes;
  auto* barriers =
      reinterpret_cast<mscclpp::BulkBarrier*>(sharedMemory + static_cast<size_t>(NumWarps) * warpStageBytes) +
      warpId * NumStages;
  auto stage = [&](int stageIdx, int slotIdx) -> uint8_t* {
    return warpStages + (static_cast<size_t>(stageIdx) * StageContributors + slotIdx) * ChunkBytes;
  };
  uint32_t barrierPhases[NumStages] = {};

  if (blockIdx.x == 0) blockPeerBarrier(context->channels_, context->rank_, numRanks);
  workspace.syncer_->sync(gridDim.x);
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

    const int numRounds =
        MaxRounds == 1
            ? 1
            : (numContributors > StageContributors ? (numContributors + StageContributors - 1) / StageContributors : 1);
    const int totalSteps = numChunks * numRounds;
    auto* outputRow = output + static_cast<int64_t>(token) * hiddenInt4;

    auto stepChunkSize = [&](int chunkIdx) {
      const int chunkOffset = chunkIdx * ChunkInt4;
      return hiddenInt4 - chunkOffset < ChunkInt4 ? hiddenInt4 - chunkOffset : ChunkInt4;
    };

    auto roundContributors = [&](int roundIdx) {
      const int begin = roundIdx * StageContributors;
      if (begin >= numContributors) return 0;
      const int remaining = numContributors - begin;
      return remaining < StageContributors ? remaining : StageContributors;
    };

    auto issueLoads = [&](int stageIdx, int chunkIdx, int roundIdx) {
      if (laneId != 0) return;

      const int chunkOffset = chunkIdx * ChunkInt4;
      const int chunkSize = stepChunkSize(chunkIdx);
      const int begin = roundIdx * StageContributors;
      const int count = roundContributors(roundIdx);
      const uint32_t chunkBytes = static_cast<uint32_t>(chunkSize * static_cast<int>(sizeof(int4)));
      barriers[stageIdx].arriveAndExpect(chunkBytes * count);
      for (int slotIdx = 0; slotIdx < count; ++slotIdx) {
        const int contributor = begin + slotIdx;
        const void* peerBuffer = transport.mappedBuffer(combineBuffer, contributorRanks[contributor]);
        const auto* source = payload.data<int4>(peerBuffer) +
                             static_cast<int64_t>(contributorSlots[contributor]) * hiddenInt4 + chunkOffset;
        mscclpp::bulkLoad(stage(stageIdx, slotIdx), source, chunkBytes, barriers[stageIdx]);
      }
    };

    auto waitStage = [&](int stageIdx) {
      if (laneId == 0) barriers[stageIdx].wait(barrierPhases[stageIdx]);
      __syncwarp();
      mscclpp::bulkFence();
    };

    mscclpp::f32x8 acc[VectorsPerLane];
    auto accumulateStage = [&](int stageIdx, int chunkIdx, int roundIdx) {
      const int chunkOffset = chunkIdx * ChunkInt4;
      const int chunkSize = stepChunkSize(chunkIdx);
      const int count = roundContributors(roundIdx);
#pragma unroll
      for (int vecIdx = 0; vecIdx < VectorsPerLane; ++vecIdx) {
        const int index = laneId + vecIdx * WARP_SIZE;
        if (index >= chunkSize) continue;
        if (roundIdx == 0) {
#pragma unroll
          for (int element = 0; element < Bf16PerInt4; ++element) acc[vecIdx].data[element] = 0.0f;
        }
#pragma unroll
        for (int slotIdx = 0; slotIdx < StageContributors; ++slotIdx) {
          if (slotIdx >= count) break;
          const auto* peerValues = reinterpret_cast<const mscclpp::bf16x8*>(stage(stageIdx, slotIdx));
          const auto inputValues = mscclpp::to<mscclpp::f32x8>(peerValues[index]);
#pragma unroll
          for (int element = 0; element < Bf16PerInt4; ++element) {
            acc[vecIdx].data[element] += inputValues.data[element];
          }
        }
        if (roundIdx == numRounds - 1) {
          outputRow[chunkOffset + index] = mscclpp::bit_cast<int4>(mscclpp::to<mscclpp::bf16x8>(acc[vecIdx]));
        }
      }
    };

#pragma unroll
    for (int step = 0; step < NumStages - 1; ++step) {
      if (step < totalSteps) {
        issueLoads(step, step / numRounds, step % numRounds);
      }
    }
    for (int step = 0; step < totalSteps; ++step) {
      const int stageIdx = step % NumStages;
      const int nextStep = step + NumStages - 1;
      if (nextStep < totalSteps) {
        issueLoads(nextStep % NumStages, nextStep / numRounds, nextStep % numRounds);
      }
      waitStage(stageIdx);
      accumulateStage(stageIdx, step / numRounds, step % numRounds);
      __syncwarp();
    }
  }
#endif  // MSCCLPP_BULK_AVAILABLE
}

template <int MaxContributors, int NumWarps>
void launchThroughputCombine(void* output, const Workload& workload, const ThroughputWorkspaceLayout& workspace,
                             const ThroughputPayloadView& payload, void* combineBuffer, const DeviceContext& context,
                             int numBlocks, cudaStream_t stream) {
  constexpr int NumStages = COMBINE_TMA_STAGES;
  constexpr int StageContributors =
      MaxContributors < COMBINE_TMA_MAX_STAGE_CONTRIBUTORS ? MaxContributors : COMBINE_TMA_MAX_STAGE_CONTRIBUTORS;
  constexpr int ChunkInt4 = COMBINE_TMA_CHUNK_INT4;
  constexpr int NumThreads = NumWarps * WARP_SIZE;
  constexpr size_t SharedBytes =
      static_cast<size_t>(NumWarps) * NumStages * StageContributors * ChunkInt4 * sizeof(int4) +
      static_cast<size_t>(NumWarps) * NumStages * sizeof(mscclpp::BulkBarrier);
  auto kernel = throughputReduceCombineKernel<MaxContributors, NumWarps>;
  static thread_local KernelConfigCache kernelConfig;
  const int residentBlocks = configureKernel(kernel, NumThreads, SharedBytes, context, kernelConfig);
  EP_HOST_ASSERT(numBlocks <= residentBlocks);
  kernel<<<dim3(numBlocks), dim3(NumThreads), SharedBytes, stream>>>(static_cast<int4*>(output), workload, workspace,
                                                                     payload, combineBuffer, context.devicePtr_);
  MSCCLPP_CUDATHROW(cudaGetLastError());
}

void throughputReduceCombine(void* output, const Workload& workload, const ThroughputWorkspaceLayout& workspace,
                             const ThroughputPayloadView& payload, void* combineBuffer, const DeviceContext& context,
                             int numBlocks, cudaStream_t stream) {
  EP_HOST_ASSERT(output != nullptr || workload.numTokens_ == 0);
  EP_HOST_ASSERT(workspace.tokenRoutes_ != nullptr && workspace.rankOffsets_ != nullptr);
  EP_HOST_ASSERT(workspace.numRecvTokens_ != nullptr && workspace.recvCounts_ != nullptr);
  EP_HOST_ASSERT(workspace.syncer_ != nullptr);
  EP_HOST_ASSERT(combineBuffer != nullptr && context.peerBufferBases_ != nullptr);
  EP_HOST_ASSERT(context.channels_ != nullptr);
  EP_HOST_ASSERT(numBlocks > 0);
  EP_HOST_ASSERT(isSupportedRanks(context.numRanks_));

  const int numTopk = workload.numTopk_;
  EP_HOST_ASSERT(numTopk > 0 && numTopk <= MaxNumTopk);
  // Rank-deduplicated routing needs at most top-k contributors, independent of rank count.
  const int maxContributors = std::min(numTopk, context.numRanks_);
  const bool useWideKernel = numBlocks <= COMBINE_TMA_WIDE_MAX_BLOCKS;
  auto launch = launchThroughputCombine<2, COMBINE_TMA_WARPS>;
  if (maxContributors > 8) {
    launch = useWideKernel ? launchThroughputCombine<16, COMBINE_TMA_WARPS_WIDE>
                           : launchThroughputCombine<16, COMBINE_TMA_WARPS_NARROW>;
  } else if (maxContributors > 4) {
    launch = useWideKernel ? launchThroughputCombine<8, COMBINE_TMA_WARPS_WIDE>
                           : launchThroughputCombine<8, COMBINE_TMA_WARPS_NARROW>;
  } else if (maxContributors > 2) {
    launch = launchThroughputCombine<4, COMBINE_TMA_WARPS>;
  }
  launch(output, workload, workspace, payload, combineBuffer, context, numBlocks, stream);
}

}  // namespace ep
}  // namespace mscclpp
