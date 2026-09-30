// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_ROLES_CUH_
#define MSCCLPP_EXT_MEGAMOE_ROLES_CUH_

#include "megamoe_epilogue.cuh"
#include "megamoe_mma.cuh"
#include "megamoe_routing.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

template <bool Local>
__device__ __forceinline__ int fc1TaskTiles(int intermediate) {
  return 2 * intermediate / TilePolicy<Local>::M;
}

template <bool Local>
__device__ __forceinline__ int fc2TaskTiles(int hidden) {
  return (hidden + TilePolicy<Local>::M - 1) / TilePolicy<Local>::M;
}

template <bool Local>
__device__ __forceinline__ int fc1CompletionCount(int intermediate) {
  return ClusterM * fc1TaskTiles<Local>(intermediate);
}

template <class P>
__device__ Task taskAt(const P& p, int ordinal, int tokens, int hidden, int intermediate) {
  constexpr bool Local = P::LocalExpert;
  constexpr int KernelTileN = P::Tiles::N;
  int fc1Tiles = fc1TaskTiles<Local>(intermediate);
  int fc2Tiles = fc2TaskTiles<Local>(hidden);
  int tokenBlocks = Local ? (tokens + KernelTileN - 1) / KernelTileN : p.workspace.control->tokenBlocks;
  int fc1Tasks = tokenBlocks * fc1Tiles;
  bool fc1 = ordinal < fc1Tasks;
  int local = fc1 ? ordinal : ordinal - fc1Tasks;
  int tiles = fc1 ? fc1Tiles : fc2Tiles;
  int block, m;
  if constexpr (P::WeightMxfp4) {
    (fc1 ? p.fc1TaskDivisor : p.fc2TaskDivisor)(block, m, local);
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
  constexpr int BlockStride = P::WeightMxfp4 ? W4TokenStride / W4TileN : 1;
  return cute::make_coord(task.m * ClusterM + cta, task.block * BlockStride, 0, task.tokens.expert);
}

template <int LocalMode, class Schedule, class P, class Storage, class Accumulate, class Accumulators>
__device__ __forceinline__ void runEpilogueRole(const P& p, int tokens, __bfloat16* output, Storage& s,
                                                Accumulate& pipeline, Accumulators accumulators, int hidden,
                                                int intermediate, int cluster, int tasks) {
  constexpr bool Local = LocalMode != 0;
  static_assert(P::LocalExpert == Local);
  constexpr int RoleRegisters = Local ? LocalComputeRegisters : ComputeRegisters;
  constexpr int RestoreRegisters = Local ? LocalEntryRegisters : EntryRegisters;
  cutlass::arch::warpgroup_reg_alloc<RoleRegisters>();
  typename Accumulate::PipelineState state;
  for (int i = cluster; i < tasks; i += gridDim.x / ClusterM)
    epilogue<LocalMode, Schedule>(p, taskAt(p, i, tokens, hidden, intermediate), s, pipeline, state, accumulators,
                                  output, hidden, intermediate);
#if MSCCLPP_BULK_AVAILABLE
  if constexpr (Local) {
    if (threadIdx.x % 32 == 0) bulkStoreWait();
  }
#endif
  cutlass::arch::warpgroup_reg_dealloc<RestoreRegisters>();
}

template <bool E5M2, bool Local, class LoadA, class Transform, class Mainloop, class Accumulators>
__device__ __forceinline__ void runTransformRole(const Parameters<E5M2, Local>& p, int tokens,
                                                 SharedStorage<E5M2, Local>& s, LoadA& loadPipeline,
                                                 Transform& transformPipeline, Mainloop& fc1,
                                                 const ProblemShape& shape1, Accumulators accumulators, int hidden,
                                                 int intermediate, int cluster, int tasks) {
  constexpr int RoleRegisters = Local ? LocalComputeRegisters : ComputeRegisters;
  constexpr int RestoreRegisters = Local ? LocalEntryRegisters : EntryRegisters;
  cutlass::arch::warpgroup_reg_alloc<RoleRegisters>();
  typename LoadA::PipelineState loadState;
  auto transformState = cutlass::make_producer_start_state<Transform>();
  auto inputs = fc1.transform_init(p.fc1, shape1, accumulators, s.tensors);
  for (int i = cluster; i < tasks; i += gridDim.x / ClusterM) {
    Task task = taskAt(p, i, tokens, hidden, intermediate);
    transformWeights<E5M2, Local>(loadPipeline, loadState, transformPipeline, transformState, inputs,
                                  taskKTiles(p, task, hidden, intermediate));
  }
  transformPipeline.producer_tail(transformState);
  cutlass::arch::warpgroup_reg_dealloc<RestoreRegisters>();
}

template <bool E5M2, bool Local, class Schedule, class LoadA, class LoadB, class Transform, class Accumulate,
          class Mainloop, class Accumulators>
__device__ __forceinline__ void runTransferRole(const Parameters<E5M2, Local>& p, int tokens,
                                                SharedStorage<E5M2, Local>& s, LoadA& aPipeline, LoadB& bPipeline,
                                                Transform& tPipeline, Accumulate& cPipeline, Mainloop& fc1,
                                                Mainloop& fc2, const ProblemShape& shape1, const ProblemShape& shape2,
                                                Accumulators accumulators, int hidden, int intermediate, int warp,
                                                int lane, int cta, int cluster, int tasks) {
  using namespace cute;
  static_assert(Schedule::HasDispatch == !Local);
  cutlass::arch::warpgroup_reg_dealloc<TransferRegisters>();
  if constexpr (Schedule::HasDispatch) {
    static_assert(Schedule::DispatchEnd - Schedule::DispatchBegin == DispatchWarpCount);
    if (warp >= Schedule::DispatchBegin && warp < Schedule::DispatchEnd)
      dispatchTokens(p, s, warp - Schedule::DispatchBegin);
  }
  if (warp == Schedule::LoadAWarp && lane == 0) {
    auto state = cutlass::make_producer_start_state<LoadA>();
    auto load1 = fc1.load_init(shape1, p.fc1, s.tensors);
    auto load2 = fc2.load_init(shape2, p.fc2, s.tensors);
    for (int i = cluster; i < tasks; i += gridDim.x / ClusterM) {
      Task task = taskAt(p, i, tokens, hidden, intermediate);
      auto coord = taskCoord(p, task, cta);
      int kTiles = taskKTiles(p, task, hidden, intermediate);
      auto iterator = cute::make_coord_iterator(0, kTiles);
      auto result = task.fc1 ? fc1.load_A(p.fc1, aPipeline, state, load1, coord, iterator, kTiles)
                             : fc2.load_A(p.fc2, aPipeline, state, load2, coord, iterator, kTiles);
      state = get<0>(result);
    }
    aPipeline.producer_tail(state);
  } else if (warp == Schedule::LoadBWarp && lane == 0) {
    auto state = cutlass::make_producer_start_state<LoadB>();
    auto load1 = fc1.load_init(shape1, p.fc1, s.tensors);
    auto load2 = fc2.load_init(shape2, p.fc2, s.tensors);
    for (int i = cluster; i < tasks; i += gridDim.x / ClusterM) {
      Task task = taskAt(p, i, tokens, hidden, intermediate);
      if constexpr (Local) {
        if (!task.fc1)
          waitAtLeast<int, scopeDevice>(p.workspace.hiddenReady + task.block, fc1CompletionCount<Local>(intermediate));
      } else {
        waitAtLeast<int, scopeDevice>((task.fc1 ? p.workspace.inputReady : p.workspace.hiddenReady) + task.block,
                                      task.fc1 ? task.tokens.rows : fc1CompletionCount<Local>(intermediate));
      }
      auto coord = taskCoord(p, task, cta);
      int kTiles = taskKTiles(p, task, hidden, intermediate);
      auto iterator = cute::make_coord_iterator(0, kTiles);
      auto result = task.fc1 ? fc1.load_B(p.fc1, bPipeline, state, load1, coord, iterator, kTiles)
                             : fc2.load_B(p.fc2, bPipeline, state, load2, coord, iterator, kTiles);
      state = get<0>(result);
    }
    bPipeline.producer_tail(state);
  } else if (warp == Schedule::MmaWarp && cta == 0) {
    typename LoadB::PipelineState bState;
    typename Transform::PipelineState tState;
    auto cState = cutlass::make_producer_start_state<Accumulate>();
    auto inputs = fc1.mma_init(accumulators, s.tensors);
    for (int i = cluster; i < tasks; i += gridDim.x / ClusterM) {
      Task task = taskAt(p, i, tokens, hidden, intermediate);
      auto result = mmaTiles<E5M2, Local>(bPipeline, bState, tPipeline, tState, cPipeline, cState, accumulators, inputs,
                                          taskKTiles(p, task, hidden, intermediate));
      bState = get<0>(result);
      tState = get<1>(result);
      cState = get<2>(result);
    }
    cPipeline.producer_tail(cState);
  }
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_ROLES_CUH_
