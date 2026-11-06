// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_W8A16_ROLES_CUH_
#define MSCCLPP_EXT_MEGAMOE_W8A16_ROLES_CUH_

#include "megamoe_collective.cuh"
#include "megamoe_dispatch.cuh"
#include "megamoe_epilogue.cuh"
#include "megamoe_mma.cuh"
#include "megamoe_tasks.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

template <bool Local, class P, class Storage, class LoadA, class Transform, class Mainloop, class Accumulators>
__device__ __forceinline__ void runW8A16TransformRole(const P& p, int tokens, Storage& s, LoadA& loadPipeline,
                                                      Transform& transformPipeline, Mainloop& fc1,
                                                      const ProblemShape& shape1, Accumulators accumulators, int hidden,
                                                      int intermediate, int cluster, int tasks) {
  cutlass::arch::warpgroup_reg_alloc<P::Collective::EpilogueRegisters>();
  typename LoadA::PipelineState loadState;
  auto transformState = cutlass::make_producer_start_state<Transform>();
  auto inputs = fc1.transform_init(p.fc1, shape1, accumulators, s.tensors);
  for (int i = cluster; i < tasks; i += gridDim.x / P::Tiles::ClusterSize) {
    Task task = taskAt(p, i, hidden, intermediate, tokens);
    auto phase = task.fc1 ? KernelTracePhase::TransformFc1 : KernelTracePhase::TransformFc2;
    traceKernel(phase, true, i);
    transformWeights<Local>(loadPipeline, loadState, transformPipeline, transformState, inputs,
                            taskKTiles(p, task, hidden, intermediate));
    traceKernel(phase, false, i);
  }
  transformPipeline.producer_tail(transformState);
}

template <bool Local, class P, class LoadTiles>
__device__ __forceinline__ void loadW8A16ActivationTiles(const P& p, const Task& task, int hidden, int intermediate,
                                                         int kTiles, LoadTiles&& loadTiles) {
  using Types = typename P::Collective;
  if constexpr (Local) {
    if (!task.fc1)
      waitAtLeast<int, scopeDevice>(p.workspace.hiddenReady + size_t(task.block) * W4ReadyCounterStride,
                                    fc1CompletionCount<P>(intermediate));
    loadTiles(0, kTiles);
    return;
  }
  if (task.fc1) {
    constexpr int TilesPerChunk = Types::DispatchChunk / (P::Tiles::K * sizeof(__bfloat16));
    for (int first = 0; first < kTiles; first += TilesPerChunk) {
      traceKernel(KernelTracePhase::TokenReady, true, first / TilesPerChunk);
      waitAtLeast<int, scopeDevice>(w4InputChunkCounter<Types>(p.workspace, hidden, task.block, first / TilesPerChunk),
                                    task.tokens.rows);
      traceKernel(KernelTracePhase::TokenReady, false, first / TilesPerChunk);
      loadTiles(first, min(TilesPerChunk, kTiles - first));
    }
  } else {
    traceKernel(KernelTracePhase::TokenReady, true, int(task.fc1));
    waitAtLeast<int, scopeDevice>(p.workspace.hiddenReady + size_t(task.block) * W4ReadyCounterStride,
                                  fc1CompletionCount<P>(intermediate));
    traceKernel(KernelTracePhase::TokenReady, false, int(task.fc1));
    loadTiles(0, kTiles);
  }
}

template <bool Local, class Schedule, class P, class Storage, class LoadA, class LoadB, class Transform,
          class Accumulate, class Mainloop, class Accumulators>
__device__ __forceinline__ void runW8A16TransferRole(const P& p, int tokens, Storage& s, LoadA& aPipeline,
                                                     LoadB& bPipeline, Transform& tPipeline, Accumulate& cPipeline,
                                                     Mainloop& fc1, Mainloop& fc2, const ProblemShape& shape1,
                                                     const ProblemShape& shape2, Accumulators accumulators, int hidden,
                                                     int intermediate, int warp, int lane, int cta, int cluster,
                                                     int tasks) {
  using namespace cute;
  using Types = typename P::Collective;
  cutlass::arch::warpgroup_reg_dealloc<TransferRegisters>();
  if constexpr (!Local) {
    runDispatchRole<0, Types::DispatchChunk, Types::DispatchStages>(p, s, warp);
  }
  if (warp == Schedule::LoadAWarp && lane == 0) {
    auto state = cutlass::make_producer_start_state<LoadA>();
    auto load1 = fc1.load_init(shape1, p.fc1, s.tensors);
    auto load2 = fc2.load_init(shape2, p.fc2, s.tensors);
    for (int i = cluster; i < tasks; i += gridDim.x / P::Tiles::ClusterSize) {
      Task task = taskAt(p, i, hidden, intermediate, tokens);
      auto coord = taskCoord(p, task, cta);
      int kTiles = taskKTiles(p, task, hidden, intermediate);
      auto iterator = cute::make_coord_iterator(0, kTiles);
      auto phase = task.fc1 ? KernelTracePhase::LoadFc1 : KernelTracePhase::LoadFc2;
      traceKernel(phase, true, i);
      auto result = task.fc1 ? fc1.load_A(p.fc1, aPipeline, state, load1, coord, iterator, kTiles)
                             : fc2.load_A(p.fc2, aPipeline, state, load2, coord, iterator, kTiles);
      traceKernel(phase, false, i);
      state = get<0>(result);
    }
    aPipeline.producer_tail(state);
  } else if (warp == Schedule::LoadBWarp && lane == 0) {
    auto state = cutlass::make_producer_start_state<LoadB>();
    auto load1 = fc1.load_init(shape1, p.fc1, s.tensors);
    auto load2 = fc2.load_init(shape2, p.fc2, s.tensors);
    for (int i = cluster; i < tasks; i += gridDim.x / P::Tiles::ClusterSize) {
      Task task = taskAt(p, i, hidden, intermediate, tokens);
      auto coord = taskCoord(p, task, cta);
      int kTiles = taskKTiles(p, task, hidden, intermediate);
      int mmaRows = (int(task.tokens.rows) + 15) / 16 * 16;
      auto shiftLoadB = [&](const auto& baseInputs, const auto& params) {
        auto rowLayout = params.tma_load_b.get_tma_tensor(make_shape(1, 1, 1)).layout();
        auto rowOffset = rowLayout(make_coord(cta * (mmaRows - P::Tiles::N) / P::Tiles::ClusterSize, 0, 0));
        auto shiftedB = make_tensor(get<3>(baseInputs).data() + rowOffset, get<3>(baseInputs).layout());
        return cute::make_tuple(get<0>(baseInputs), get<1>(baseInputs), get<2>(baseInputs), shiftedB,
                                get<4>(baseInputs), get<5>(baseInputs), get<6>(baseInputs), get<7>(baseInputs),
                                get<8>(baseInputs));
      };
      auto phase = task.fc1 ? KernelTracePhase::LoadFc1 : KernelTracePhase::LoadFc2;
      traceKernel(phase, true, i);
      loadW8A16ActivationTiles<Local>(p, task, hidden, intermediate, kTiles, [&](int first, int count) {
        auto iterator = cute::make_coord_iterator(first, kTiles);
        auto result = task.fc1 ? fc1.load_B(p.fc1, bPipeline, state, shiftLoadB(load1, p.fc1), coord, iterator, count)
                               : fc2.load_B(p.fc2, bPipeline, state, shiftLoadB(load2, p.fc2), coord, iterator, count);
        state = get<0>(result);
      });
      traceKernel(phase, false, i);
    }
    bPipeline.producer_tail(state);
  } else {
    runW8A16MmaRole<Local, Schedule>(p, tokens, s, bPipeline, tPipeline, cPipeline, fc1, accumulators, hidden,
                                     intermediate, warp, cta, cluster, tasks);
  }
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_W8A16_ROLES_CUH_
