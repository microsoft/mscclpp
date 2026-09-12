// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cooperative_groups.h>

#include <algorithm>
#include <mscclpp/bulk_device.hpp>

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
    throughputReduceCombineKernel(int4* output, float* outputTopkWeights, const void* input, Workload workload,
                                  ThroughputWorkspaceLayout workspace, ThroughputPayloadView payload, void* recvBuffer,
                                  const DeviceContext* context) {
#if MSCCLPP_BULK_AVAILABLE
  const int numTopk = workload.numTopk_;
  const int maxTokensPerRank = workload.maxTokensPerRank_;
  const bool rankMajor = workload.outputLayout_ == DispatchLayout::RANK_MAJOR;
  const TransportView transport(context);
  const int numRanks = context->numRanks_;
  EP_DEVICE_ASSERT(MaxContributors <= numRanks);
  constexpr int ChunkInt4 = COMBINE_TMA_CHUNK_INT4;
  constexpr int NumStages = COMBINE_TMA_STAGES;
  constexpr int ChunkBytes = ChunkInt4 * static_cast<int>(sizeof(int4));
  constexpr int Bf16PerInt4 = sizeof(int4) / sizeof(nv_bfloat16);

  const int laneId = getLaneId();
  const int warpId = static_cast<int>(threadIdx.x) / WARP_SIZE;
  const int hiddenInt4 = workload.hidden_ / Bf16PerInt4;
  const int receivedTokens = *workspace.numRecvTokens_;
  EP_DEVICE_ASSERT(receivedTokens >= 0 && receivedTokens <= numRanks * maxTokensPerRank);
  EP_DEVICE_ASSERT(input != nullptr || receivedTokens == 0);

  auto* localTokens = payload.data<int4>(recvBuffer);
  if (receivedTokens > 0 && input != localTokens) {
    const int rows = rankMajor ? numRanks * maxTokensPerRank : receivedTokens;
    copyThroughputRows(localTokens, input, rows, hiddenInt4 * sizeof(int4), workspace.recvCounts_, maxTokensPerRank,
                       rankMajor, blockIdx.x * blockDim.x + threadIdx.x, gridDim.x * blockDim.x);
    // Every local staging copy must finish before the peer barrier publishes it.
    __threadfence_system();
    cooperative_groups::this_grid().sync();
  }

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

  if (blockIdx.x == 0 && threadIdx.x < WARP_SIZE) barrier(context->channels_, context->rank_, numRanks);
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
    int numContributors = 0;
    for (int rankBase = 0; rankBase < numRanks; rankBase += WARP_SIZE) {
      const int peerRank = rankBase + laneId;
      const int slot = peerRank < numRanks ? workspace.recvTokenIndex(token, peerRank, numRanks) : -1;
      unsigned contributors = __ballot_sync(0xffffffffu, slot >= 0);
      while (contributors != 0u) {
        const int sourceLane = __ffs(static_cast<int>(contributors)) - 1;
        if (numContributors < MaxContributors) {
          contributorRanks[numContributors] = rankBase + sourceLane;
          contributorSlots[numContributors] = __shfl_sync(0xffffffffu, slot, sourceLane);
          ++numContributors;
        }
        contributors &= contributors - 1u;
      }
    }

    auto* outputRow = output + static_cast<int64_t>(token) * hiddenInt4;
    auto issueLoads = [&](int stageIdx, int chunkOffset, int chunkSize) {
      if (laneId != 0) return;

      const uint32_t chunkBytes = static_cast<uint32_t>(chunkSize * static_cast<int>(sizeof(int4)));
      barriers[stageIdx].arriveAndExpect(chunkBytes * numContributors);
      for (int contributor = 0; contributor < numContributors; ++contributor) {
        const void* peerBuffer = transport.mappedBuffer(recvBuffer, contributorRanks[contributor]);
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
        float values[Bf16PerInt4] = {};
#pragma unroll
        for (int contributor = 0; contributor < MaxContributors; ++contributor) {
          if (contributor >= numContributors) break;
          const int4 packed =
              *reinterpret_cast<const int4*>(stage(stageIdx, contributor) + index * static_cast<int>(sizeof(int4)));
          const auto* inputValues = reinterpret_cast<const nv_bfloat16*>(&packed);
#pragma unroll
          for (int element = 0; element < Bf16PerInt4; ++element) {
            values[element] += static_cast<float>(inputValues[element]);
          }
        }

        int4 packedOutput;
        auto* outputValues = reinterpret_cast<nv_bfloat16*>(&packedOutput);
#pragma unroll
        for (int element = 0; element < Bf16PerInt4; ++element) {
          outputValues[element] = static_cast<nv_bfloat16>(values[element]);
        }
        outputRow[chunkOffset + index] = packedOutput;
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
        const void* peerBuffer = transport.mappedBuffer(recvBuffer, contributorRanks[contributor]);
        const auto* weights = payload.topKValues(peerBuffer, contributorSlots[contributor]);
        weight += __ldg(weights + laneId);
      }
      outputTopkWeights[static_cast<int64_t>(token) * numTopk + laneId] = weight;
    }
  }
#endif  // MSCCLPP_BULK_AVAILABLE
}

template <int MaxContributors, int NumWarps>
void launchThroughputCombine(void* output, float* outputTopkWeights, const void* input, const Workload& workload,
                             const ThroughputWorkspaceLayout& workspace, const ThroughputPayloadView& payload,
                             void* recvBuffer, const DeviceContext& context, int numBlocks, cudaStream_t stream) {
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
  MSCCLPP_CUDATHROW(cudaLaunchKernelEx(&config, kernel, static_cast<int4*>(output), outputTopkWeights, input, workload,
                                       workspace, payload, recvBuffer, context.devicePtr_));
}

void throughputReduceCombine(void* output, float* outputTopkWeights, const void* input, const Workload& workload,
                             const ThroughputWorkspaceLayout& workspace, const ThroughputPayloadView& payload,
                             void* recvBuffer, const DeviceContext& context, int numBlocks, cudaStream_t stream) {
  EP_HOST_ASSERT(output != nullptr || workload.numTokens_ == 0);
  EP_HOST_ASSERT(workspace.recvTokenOffsets_ != nullptr && workspace.rankOffsets_ != nullptr);
  EP_HOST_ASSERT(workspace.numRecvTokens_ != nullptr && workspace.recvCounts_ != nullptr);
  EP_HOST_ASSERT(recvBuffer != nullptr && context.peerBufferBases_ != nullptr);
  EP_HOST_ASSERT(context.channels_ != nullptr);
  EP_HOST_ASSERT(numBlocks > 0);

  const int numTopk = workload.numTopk_;
  const bool useWideKernel = numBlocks <= COMBINE_TMA_WIDE_MAX_BLOCKS;
  auto launch = launchThroughputCombine<2, COMBINE_TMA_WARPS>;

  switch (context.numRanks_) {
    case 2:
      break;
    case 4:
      if (numTopk > 2) launch = launchThroughputCombine<4, COMBINE_TMA_WARPS>;
      break;
    case 8:
      if (numTopk <= 4) {
        launch = launchThroughputCombine<4, COMBINE_TMA_WARPS>;
      } else {
        launch = useWideKernel ? launchThroughputCombine<8, COMBINE_TMA_WARPS_WIDE>
                               : launchThroughputCombine<8, COMBINE_TMA_WARPS_NARROW>;
      }
      break;
    case 16:
      if (numTopk <= 4) {
        launch = launchThroughputCombine<4, COMBINE_TMA_WARPS>;
      } else if (numTopk <= 8) {
        launch = useWideKernel ? launchThroughputCombine<8, COMBINE_TMA_WARPS_WIDE>
                               : launchThroughputCombine<8, COMBINE_TMA_WARPS_NARROW>;
      } else if (numTopk <= 12) {
        launch = launchThroughputCombine<12, 9>;
      } else {
        launch = launchThroughputCombine<16, 7>;
      }
      break;
    default:
      EP_HOST_ASSERT(false && "Unsupported ranks");
  }
  launch(output, outputTopkWeights, input, workload, workspace, payload, recvBuffer, context, numBlocks, stream);
}

}  // namespace ep
}  // namespace mscclpp
