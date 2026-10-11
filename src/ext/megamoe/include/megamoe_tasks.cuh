// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_TASKS_CUH_
#define MSCCLPP_EXT_MEGAMOE_TASKS_CUH_

#include "megamoe_device.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

template <bool Local, bool Mxfp4 = false>
__device__ __forceinline__ int fc1TaskTiles(int intermediate) {
  return 2 * intermediate / TilePolicy<Local, Mxfp4>::M;
}

template <bool Local, bool Mxfp4 = false>
__device__ __forceinline__ int fc2TaskTiles(int hidden) {
  return (hidden + TilePolicy<Local, Mxfp4>::M - 1) / TilePolicy<Local, Mxfp4>::M;
}

template <bool Local, bool Mxfp4 = false>
__device__ __forceinline__ int fc1CompletionCount(int intermediate) {
  return TilePolicy<Local, Mxfp4>::ClusterSize * fc1TaskTiles<Local, Mxfp4>(intermediate);
}

template <class P>
__device__ Task taskAt(const P& p, int ordinal, int tokens, int hidden, int intermediate) {
  constexpr bool Local = P::LocalExpert;
  constexpr int KernelTileN = P::Tiles::N;
  int fc1Tiles = fc1TaskTiles<Local, P::WeightMxfp4>(intermediate);
  int fc2Tiles = fc2TaskTiles<Local, P::WeightMxfp4>(hidden);
  int tokenBlocks = Local ? (tokens + KernelTileN - 1) / KernelTileN : p.workspace.control->tokenBlocks;
  int fc1Tasks = tokenBlocks * fc1Tiles;
  bool fc1 = ordinal < fc1Tasks;
  int local = fc1 ? ordinal : ordinal - fc1Tasks;
  int tiles = fc1 ? fc1Tiles : fc2Tiles;
  int block, m;
  if constexpr (P::WeightMxfp4) {
    (fc1 ? p.fc1TaskDivisor : p.fc2TaskDivisor)(block, m, local);
    if constexpr (W4ExpertMajorTasks) {
      int expert = p.workspace.blocks[block].expert;
      int first = p.workspace.starts[expert] / KernelTileN;
      int offset;
      p.workspace.expertBlockDivisors[expert](m, offset, local - first * tiles);
      block = first + offset;
    }
  } else {
    block = local / tiles;
    m = local % tiles;
  }
  TokenBlock tokenBlock;
  if constexpr (Local) {
    tokenBlock = TokenBlock{0, min(int(KernelTileN), tokens - block * KernelTileN)};
  } else {
    tokenBlock = p.workspace.blocks[block];
  }
  return Task{fc1, block, m, tokenBlock};
}

template <class P>
__device__ __forceinline__ int taskKTiles(const P&, const Task& task, int hidden, int intermediate) {
  constexpr int KernelTileK = P::Tiles::K;
  return ((task.fc1 ? hidden : intermediate) + (P::WeightMxfp4 ? KernelTileK - 1 : 0)) / KernelTileK;
}

template <class P>
__device__ __forceinline__ auto taskCoord(const P&, const Task& task, int cta) {
  constexpr int BlockStride = P::WeightMxfp4 ? P::Tiles::TokenStride / P::Tiles::N : 1;
  return cute::make_coord(task.m * P::Tiles::ClusterSize + cta, task.block * BlockStride, 0, task.tokens.expert);
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_TASKS_CUH_
