// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_TASKS_CUH_
#define MSCCLPP_EXT_MEGAMOE_TASKS_CUH_

#include "megamoe_device.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

struct TaskRange {
  int begin;
  int end;
  int localBegin;
  // Optional cluster subset: ordinals go round-robin over clusters [clusterBegin, clusterBegin + clusterCount),
  // starting at clusterBegin + clusterOffset. clusterCount == 0 uses every cluster.
  int clusterBegin = 0;
  int clusterCount = 0;
  int clusterOffset = 0;
};

__device__ __forceinline__ int firstTaskOrdinal(const TaskRange& range, int cluster, int clusters) {
  if (range.clusterCount > 0) {
    if (cluster < range.clusterBegin || cluster >= range.clusterBegin + range.clusterCount) return range.end;
    int delta = (cluster - range.clusterBegin - range.clusterOffset) % range.clusterCount;
    if (delta < 0) delta += range.clusterCount;
    return range.begin + delta;
  }
  int delta = (cluster - range.begin) % clusters;
  if (delta < 0) delta += clusters;
  return range.begin + delta;
}

__device__ __forceinline__ int taskOrdinalStride(const TaskRange& range, int clusters) {
  return range.clusterCount > 0 ? range.clusterCount : clusters;
}

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
    return TokenBlock{0, int16_t(min(int(P::Tiles::N), tokens - block * P::Tiles::N)), 0, 0};
  }
  TokenBlock tokenBlock = p.workspace.blocks[block];
  if (tokenBlock.groupShift > 0) {
    int inGroup = int(tokenBlock.groupOffset) * tiles + m;
    block += (inGroup & ((1 << tokenBlock.groupShift) - 1)) - int(tokenBlock.groupOffset);
    m = inGroup >> tokenBlock.groupShift;
  }
  return tokenBlock;
}

template <class P>
__device__ Task taskAt(const P& p, int ordinal, int hidden, int intermediate, int tokens = 0) {
  int fc1Tiles = fc1TaskTiles<P>(intermediate);
  int fc2Tiles = fc2TaskTiles<P>(hidden);
  int fc1Tasks = taskTokenBlocks(p, tokens) * fc1Tiles;
  bool fc1 = ordinal < fc1Tasks;
  int local = fc1 ? ordinal : ordinal - fc1Tasks;
  int tiles = fc1 ? fc1Tiles : fc2Tiles;
  int block, m;
  (fc1 ? p.fc1TaskDivisor : p.fc2TaskDivisor)(block, m, local);
  TokenBlock tokenBlock = resolveTokenBlock(p, block, m, tiles, tokens);
  return Task{fc1, block, m, tokenBlock};
}

template <class P>
__device__ Task taskAt(const P& p, const TaskRange& range, int ordinal, int hidden, int intermediate, int tokens = 0) {
  return taskAt(p, range.localBegin + ordinal - range.begin, hidden, intermediate, tokens);
}

template <class P>
__device__ __forceinline__ int taskKTiles(const P&, const Task& task, int hidden, int intermediate) {
  constexpr int KernelTileK = P::Tiles::K;
  return ((task.fc1 ? hidden : intermediate) + KernelTileK - 1) / KernelTileK;
}

template <class P>
__device__ __forceinline__ auto taskCoord(const P&, const Task& task, int cta) {
  constexpr int BlockStride = P::Tiles::TokenStride / P::Tiles::N;
  return cute::make_coord(task.m * P::Tiles::ClusterSize + cta, task.block * BlockStride, 0, task.tokens.expert);
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_TASKS_CUH_
