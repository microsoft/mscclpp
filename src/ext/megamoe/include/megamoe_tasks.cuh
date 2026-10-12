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
__device__ Task taskAt(const P& p, int ordinal, int hidden, int intermediate) {
  constexpr int KernelTileN = P::Tiles::N;
  int fc1Tiles = fc1TaskTiles<P>(intermediate);
  int fc2Tiles = fc2TaskTiles<P>(hidden);
  int fc1Tasks = p.workspace.control->tokenBlocks * fc1Tiles;
  bool fc1 = ordinal < fc1Tasks;
  int local = fc1 ? ordinal : ordinal - fc1Tasks;
  int tiles = fc1 ? fc1Tiles : fc2Tiles;
  int block, m;
  (fc1 ? p.fc1TaskDivisor : p.fc2TaskDivisor)(block, m, local);
  if constexpr (W4ExpertMajorTasks) {
    int expert = p.workspace.blocks[block].expert;
    int first = p.workspace.starts[expert] / KernelTileN;
    int offset;
    p.workspace.expertBlockDivisors[expert](m, offset, local - first * tiles);
    block = first + offset;
  }
  return Task{fc1, block, m, p.workspace.blocks[block]};
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
