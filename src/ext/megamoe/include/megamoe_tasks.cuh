// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_TASKS_CUH_
#define MSCCLPP_EXT_MEGAMOE_TASKS_CUH_

#include "megamoe_device.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

template <class P>
__device__ __forceinline__ int fc1TaskTiles(int intermediate) {
  return 2 * intermediate / P::Tiles::M;
}

template <class P>
__device__ __forceinline__ int fc2TaskTiles(int hidden) {
  return (hidden + P::Tiles::M - 1) / P::Tiles::M;
}

template <class P>
__device__ __forceinline__ int fc1CompletionCount(int intermediate) {
  return P::Tiles::ClusterSize * fc1TaskTiles<P>(intermediate);
}

template <class P>
__device__ __forceinline__ int taskTokenBlocks(const P& p, int tokens) {
  if constexpr (P::LocalExpert) return (tokens + P::Tiles::N - 1) / P::Tiles::N;
  return p.workspace.control->tokenBlocks;
}

template <class P>
__device__ __forceinline__ TokenBlock resolveTokenBlock(const P& p, int& block, int& m, int tiles, int tokens) {
  if constexpr (P::LocalExpert) {
    return TokenBlock::make(0, min(int(P::Tiles::N), tokens - block * P::Tiles::N));
  }
  TokenBlock tokenBlock = p.workspace.blocks[block];
  int shift = tokenBlock.groupShift();
  if (shift > 0) {
    int inGroup = int(tokenBlock.blockIdxInGroup) * tiles + m;
    block += (inGroup & ((1 << shift) - 1)) - int(tokenBlock.blockIdxInGroup);
    m = inGroup >> shift;
  }
  return tokenBlock;
}

template <class P>
constexpr bool AllowLag = !P::LocalExpert && P::WeightMxfp4 && (P::Collective::DispatchWarps >= 4);

constexpr int InterleaveBlockLag = 32;

template <class P>
__device__ __forceinline__ bool useInterleavedTasks(int tokenBlocks, int localExperts) {
  return AllowLag<P> && (tokenBlocks > max(2 * localExperts, 2 * InterleaveBlockLag));
}

template <class P>
__device__ __forceinline__ Task taskAtFast(const P& p, int ordinal, int tokenBlocks, int localExperts, int fc1Tiles,
                                           int fc2Tiles, int tokens = 0) {
  bool fc1;
  int block, m;
  if constexpr (AllowLag<P>) {
    if (useInterleavedTasks<P>(tokenBlocks, localExperts)) {
      constexpr int Lag = InterleaveBlockLag;
      const int prologue = Lag * fc1Tiles;
      const int stepTasks = fc1Tiles + fc2Tiles;
      const int steadySteps = tokenBlocks - Lag;
      const int steadyEnd = prologue + steadySteps * stepTasks;
      if (ordinal < prologue) {
        fc1 = true;
        p.fc1TaskDivisor(block, m, ordinal);
      } else if (ordinal < steadyEnd) {
        int rel = ordinal - prologue;
        int s = rel / stepTasks;
        int rem = rel - s * stepTasks;
        if (rem < fc1Tiles) {
          fc1 = true;
          block = Lag + s;
          m = rem;
        } else {
          fc1 = false;
          block = s;
          m = rem - fc1Tiles;
        }
      } else {
        fc1 = false;
        int rel = ordinal - steadyEnd;
        int tailBlock;
        p.fc2TaskDivisor(tailBlock, m, rel);
        block = steadySteps + tailBlock;
      }
      int tiles = fc1 ? fc1Tiles : fc2Tiles;
      TokenBlock tokenBlock = resolveTokenBlock(p, block, m, tiles, tokens);
      return Task{fc1, block, m, tokenBlock};
    }
  }
  int fc1Tasks = tokenBlocks * fc1Tiles;
  fc1 = ordinal < fc1Tasks;
  int local = fc1 ? ordinal : ordinal - fc1Tasks;
  int tiles = fc1 ? fc1Tiles : fc2Tiles;
  (fc1 ? p.fc1TaskDivisor : p.fc2TaskDivisor)(block, m, local);
  TokenBlock tokenBlock = resolveTokenBlock(p, block, m, tiles, tokens);
  return Task{fc1, block, m, tokenBlock};
}

template <class P>
__device__ Task taskAt(const P& p, int ordinal, int hidden, int intermediate, int tokens = 0) {
  int fc1Tiles = fc1TaskTiles<P>(intermediate);
  int fc2Tiles = fc2TaskTiles<P>(hidden);
  int tokenBlocks = taskTokenBlocks(p, tokens);
  int localExperts = p.config.numExperts / p.config.worldSize;
  return taskAtFast(p, ordinal, tokenBlocks, localExperts, fc1Tiles, fc2Tiles, tokens);
}

template <class P>
__device__ __forceinline__ int taskKTiles(const P&, const Task& task, int hidden, int intermediate) {
  constexpr int KernelTileK = P::Tiles::K;
  return ((task.fc1 ? hidden : intermediate) + KernelTileK - 1) / KernelTileK;
}

template <class P>
__device__ __forceinline__ auto taskCoord(const P&, const Task& task, int cta) {
  constexpr int BlockStride = P::Tiles::TokenStride / P::Tiles::N;
  return cute::make_coord(task.m * P::Tiles::ClusterSize + cta, task.block * BlockStride, 0, task.tokens.expertIdx);
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_TASKS_CUH_
