// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cooperative_groups.h>

#include <algorithm>

#include "api.cuh"
#include "config.cuh"
#include "device_helpers.cuh"
#include "exception.cuh"
#include "launch.cuh"

namespace mscclpp {
namespace ep {
namespace high_throughput {
namespace detail {

#ifndef EP_HT_COMBINE_TMA_CHUNK_INT4
#define EP_HT_COMBINE_TMA_CHUNK_INT4 64
#endif
#ifndef EP_HT_COMBINE_TMA_STAGES
#define EP_HT_COMBINE_TMA_STAGES 2
#endif
#ifndef EP_HT_COMBINE_TMA_WARPS
#define EP_HT_COMBINE_TMA_WARPS 16
#endif
#ifndef EP_HT_COMBINE_TMA_WARPS_WIDE
#define EP_HT_COMBINE_TMA_WARPS_WIDE 14
#endif
#ifndef EP_HT_COMBINE_TMA_WARPS_NARROW
#define EP_HT_COMBINE_TMA_WARPS_NARROW 12
#endif
#ifndef EP_HT_COMBINE_TMA_WIDE_MAX_BLOCKS
#define EP_HT_COMBINE_TMA_WIDE_MAX_BLOCKS 24
#endif

template <int NumRanks, int MaxContributors, int NumWarps>
__global__ void __launch_bounds__(NumWarps* WARP_SIZE, 1)
    combineKernel(int4* output, float* outputTopkWeights, const int* sendHead, int numOutputTokens, int hidden,
                  int numTopk, void** recvPoolPtrs, const int* combineRecvIdx, int** taskFifoPtrs, int head, int rank,
                  int64_t recvPoolHeaderBytes, int64_t recvPoolMetadataOffset, int64_t metadataSlotBytes) {
  static_assert(MaxContributors <= NumRanks);
  constexpr int ChunkInt4 = EP_HT_COMBINE_TMA_CHUNK_INT4;
  constexpr int NumStages = EP_HT_COMBINE_TMA_STAGES;
  constexpr int ChunkBytes = ChunkInt4 * static_cast<int>(sizeof(int4));
  constexpr int Bf16PerInt4 = sizeof(int4) / sizeof(nv_bfloat16);

  const int laneId = get_lane_id();
  const int warpId = static_cast<int>(threadIdx.x) / WARP_SIZE;
  const int hiddenInt4 = hidden / Bf16PerInt4;

  extern __shared__ uint8_t sharedMemory[];
  const size_t warpStageBytes = static_cast<size_t>(NumStages) * MaxContributors * ChunkBytes;
  auto* warpStages = sharedMemory + warpId * warpStageBytes;
  auto* barriers =
      reinterpret_cast<uint64_t*>(sharedMemory + static_cast<size_t>(NumWarps) * warpStageBytes) + warpId * NumStages;
  auto stage = [&](int stageIdx, int contributorIdx) -> uint8_t* {
    return warpStages + (static_cast<size_t>(stageIdx) * MaxContributors + contributorIdx) * ChunkBytes;
  };

  if (blockIdx.x == 0 && threadIdx.x < WARP_SIZE) barrier_device<NumRanks>(taskFifoPtrs, head, rank);
  cooperative_groups::this_grid().sync();

  const int globalWarp = static_cast<int>(blockIdx.x) * NumWarps + warpId;
  const int totalWarps = static_cast<int>(gridDim.x) * NumWarps;
  const int numChunks = (hiddenInt4 + ChunkInt4 - 1) / ChunkInt4;

  for (int token = globalWarp; token < numOutputTokens; token += totalWarps) {
    int contributorRanks[MaxContributors];
    int contributorSlots[MaxContributors];
    int numContributors = 0;
    for (int rankBase = 0; rankBase < NumRanks; rankBase += WARP_SIZE) {
      const int peerRank = rankBase + laneId;
      const bool contributes = peerRank < NumRanks && sendHead[static_cast<int64_t>(token) * NumRanks + peerRank] >= 0;
      const int slot = contributes ? combineRecvIdx[static_cast<int64_t>(token) * NumRanks + peerRank] : 0;
      unsigned contributors = __ballot_sync(0xffffffffu, contributes);
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

      initTmaLoadBarrier(&barriers[stageIdx]);
      const uint32_t chunkBytes = static_cast<uint32_t>(chunkSize * static_cast<int>(sizeof(int4)));
      for (int contributor = 0; contributor < numContributors; ++contributor) {
        const auto* source =
            reinterpret_cast<const uint8_t*>(recvPoolPtrs[contributorRanks[contributor]]) + recvPoolHeaderBytes +
            static_cast<int64_t>(contributorSlots[contributor]) * hiddenInt4 * static_cast<int64_t>(sizeof(int4)) +
            static_cast<int64_t>(chunkOffset) * sizeof(int4);
        issueTmaLoadCopy(source, stage(stageIdx, contributor), &barriers[stageIdx], chunkBytes);
      }
      expectTmaLoad(&barriers[stageIdx], chunkBytes * numContributors);
    };

    auto waitStage = [&](int stageIdx) {
      if (laneId == 0) {
        uint32_t phase = 0;
        waitTmaLoad(&barriers[stageIdx], phase);
      }
      __syncwarp();
      fenceProxyAsyncSharedCta();
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
        st_na_global(outputRow + chunkOffset + index, packedOutput);
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
        const auto* metadata = reinterpret_cast<const uint8_t*>(recvPoolPtrs[contributorRanks[contributor]]) +
                               recvPoolMetadataOffset +
                               static_cast<int64_t>(contributorSlots[contributor]) * metadataSlotBytes;
        const auto* weights = reinterpret_cast<const float*>(metadata + static_cast<size_t>(numTopk) * sizeof(int));
        weight += ld_nc_global(weights + laneId);
      }
      st_na_global(outputTopkWeights + static_cast<int64_t>(token) * numTopk + laneId, weight);
    }
  }
}

template <int NumRanks, int MaxContributors, int NumWarps>
int maxCooperativeBlocks(size_t dynamicSharedBytes) {
  static int cachedDevice = -1;
  static size_t cachedSharedBytes = 0;
  static int cachedMaxBlocks = 0;

  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  if (device != cachedDevice || dynamicSharedBytes != cachedSharedBytes) {
    int blocksPerSm;
    int numSms;
    auto kernel = combineKernel<NumRanks, MaxContributors, NumWarps>;
    CUDA_CHECK(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSm, kernel, NumWarps * WARP_SIZE, dynamicSharedBytes));
    CUDA_CHECK(cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, device));
    cachedDevice = device;
    cachedSharedBytes = dynamicSharedBytes;
    cachedMaxBlocks = blocksPerSm * numSms;
  }
  return cachedMaxBlocks;
}

void combine(void* output, float* outputTopkWeights, const int* sendHead, int numOutputTokens, int hidden, int numTopk,
             int numRanks, void** recvPoolPtrs, const int* combineRecvIdx, int** taskFifoPtrs, int head, int rank,
             int64_t recvPoolHeaderBytes, int64_t recvPoolMetadataOffset, int64_t metadataSlotBytes, int numBlocks,
             cudaStream_t stream) {
  EP_HOST_ASSERT(output != nullptr || numOutputTokens == 0);
  EP_HOST_ASSERT(sendHead != nullptr);
  EP_HOST_ASSERT(recvPoolPtrs != nullptr);
  EP_HOST_ASSERT(combineRecvIdx != nullptr);
  EP_HOST_ASSERT(taskFifoPtrs != nullptr);
  EP_HOST_ASSERT(numBlocks > 0);

  constexpr int NumStages = EP_HT_COMBINE_TMA_STAGES;
  constexpr int ChunkInt4 = EP_HT_COMBINE_TMA_CHUNK_INT4;
  const bool useWideKernel = numBlocks <= EP_HT_COMBINE_TMA_WIDE_MAX_BLOCKS;

#define COMBINE_LAUNCH(ranks, maxContributors, numWarps)                                                           \
  {                                                                                                                \
    auto kernel = combineKernel<ranks, maxContributors, numWarps>;                                                 \
    const size_t sharedBytes =                                                                                     \
        static_cast<size_t>(numWarps) * NumStages * maxContributors * ChunkInt4 * sizeof(int4) +                   \
        static_cast<size_t>(numWarps) * NumStages * sizeof(uint64_t);                                              \
    CUDA_CHECK(                                                                                                    \
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(sharedBytes))); \
    EP_HOST_ASSERT((numBlocks <= maxCooperativeBlocks<ranks, maxContributors, numWarps>(sharedBytes)));            \
    cudaLaunchAttribute attribute{};                                                                               \
    attribute.id = cudaLaunchAttributeCooperative;                                                                 \
    attribute.val.cooperative = 1;                                                                                 \
    cudaLaunchConfig_t config = {static_cast<unsigned>(numBlocks),                                                 \
                                 static_cast<unsigned>(numWarps * WARP_SIZE),                                      \
                                 sharedBytes,                                                                      \
                                 stream,                                                                           \
                                 &attribute,                                                                       \
                                 1};                                                                               \
    LAUNCH_KERNEL(&config, kernel, reinterpret_cast<int4*>(output), outputTopkWeights, sendHead, numOutputTokens,  \
                  hidden, numTopk, recvPoolPtrs, combineRecvIdx, taskFifoPtrs, head, rank, recvPoolHeaderBytes,    \
                  recvPoolMetadataOffset, metadataSlotBytes);                                                      \
  }

  switch (numRanks) {
    case 2:
      COMBINE_LAUNCH(2, 2, EP_HT_COMBINE_TMA_WARPS);
      break;
    case 4:
      if (numTopk <= 2)
        COMBINE_LAUNCH(4, 2, EP_HT_COMBINE_TMA_WARPS)
      else
        COMBINE_LAUNCH(4, 4, EP_HT_COMBINE_TMA_WARPS)
      break;
    case 8:
      if (numTopk <= 4) {
        COMBINE_LAUNCH(8, 4, EP_HT_COMBINE_TMA_WARPS)
      } else if (useWideKernel) {
        COMBINE_LAUNCH(8, 8, EP_HT_COMBINE_TMA_WARPS_WIDE)
      } else {
        COMBINE_LAUNCH(8, 8, EP_HT_COMBINE_TMA_WARPS_NARROW)
      }
      break;
    case 16:
      if (numTopk <= 4) {
        COMBINE_LAUNCH(16, 4, EP_HT_COMBINE_TMA_WARPS)
      } else if (numTopk <= 8) {
        if (useWideKernel)
          COMBINE_LAUNCH(16, 8, EP_HT_COMBINE_TMA_WARPS_WIDE)
        else
          COMBINE_LAUNCH(16, 8, EP_HT_COMBINE_TMA_WARPS_NARROW)
      } else if (numTopk <= 12) {
        COMBINE_LAUNCH(16, 12, 9);
      } else {
        COMBINE_LAUNCH(16, 16, 7);
      }
      break;
    default:
      EP_HOST_ASSERT(false && "Unsupported ranks");
  }
#undef COMBINE_LAUNCH
}

}  // namespace detail

void combine(void* output, float* outputTopkWeights, const int* sendHead, int numOutputTokens, int hidden, int numTopk,
             int numRanks, void** recvPoolPtrs, const int* combineRecvIdx, int** taskFifoPtrs, int head, int rank,
             int64_t recvPoolHeaderBytes, int64_t recvPoolMetadataOffset, int64_t metadataSlotBytes, int numBlocks,
             cudaStream_t stream) {
  detail::combine(output, outputTopkWeights, sendHead, numOutputTokens, hidden, numTopk, numRanks, recvPoolPtrs,
                  combineRecvIdx, taskFifoPtrs, head, rank, recvPoolHeaderBytes, recvPoolMetadataOffset,
                  metadataSlotBytes, numBlocks, stream);
}

}  // namespace high_throughput
}  // namespace ep
}  // namespace mscclpp
