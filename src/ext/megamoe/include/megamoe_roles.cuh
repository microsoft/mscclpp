// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_ROLES_CUH_
#define MSCCLPP_EXT_MEGAMOE_ROLES_CUH_

#include "megamoe_dispatch.cuh"
#include "megamoe_epilogue.cuh"
#include "megamoe_mma.cuh"
#include "megamoe_tasks.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

template <class Pipeline, class Schedule, class KernelClusterShape = ClusterShape>
__device__ __forceinline__ Pipeline makeAccumulatorPipeline(typename Pipeline::SharedStorage& storage, int warp) {
  typename Pipeline::Params params{};
  params.role = warp == Schedule::MmaWarp ? Pipeline::ThreadCategory::Producer
                                          : (warp >= Schedule::EpilogueBegin && warp < Schedule::EpilogueEnd
                                                 ? Pipeline::ThreadCategory::Consumer
                                                 : Pipeline::ThreadCategory::NonParticipant);
  params.producer_arv_count = 1;
  params.consumer_arv_count = cute::size(KernelClusterShape{}) * (Schedule::EpilogueEnd - Schedule::EpilogueBegin) * 32;
  params.initializing_warp = Schedule::EpilogueBegin;
  return Pipeline(storage, params, KernelClusterShape{}, cute::true_type{}, cute::false_type{});
}

template <bool E5M2, bool Local, class LoadA, class Transform, class Mainloop, class Accumulators>
__device__ __forceinline__ void runTransformRole(const Parameters<E5M2, Local>& p, int tokens,
                                                 SharedStorage<E5M2, Local>& s, LoadA& loadPipeline,
                                                 Transform& transformPipeline, Mainloop& fc1,
                                                 const ProblemShape& shape1, Accumulators accumulators, int hidden,
                                                 int intermediate, int cluster, int tasks) {
  constexpr int RoleRegisters = Local ? LocalComputeRegisters : ComputeRegisters;
  cutlass::arch::warpgroup_reg_alloc<RoleRegisters>();
  typename LoadA::PipelineState loadState;
  auto transformState = cutlass::make_producer_start_state<Transform>();
  auto inputs = fc1.transform_init(p.fc1, shape1, accumulators, s.tensors);
  for (int i = cluster; i < tasks; i += gridDim.x / ClusterM) {
    Task task = taskAt(p, i, tokens, hidden, intermediate);
    auto phase = task.fc1 ? KernelTracePhase::TransformFc1 : KernelTracePhase::TransformFc2;
    traceKernel(phase, true, i);
    transformWeights<E5M2, Local>(loadPipeline, loadState, transformPipeline, transformState, inputs,
                                  taskKTiles(p, task, hidden, intermediate));
    traceKernel(phase, false, i);
  }
  transformPipeline.producer_tail(transformState);
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
  cutlass::arch::warpgroup_reg_dealloc<TransferRegisters>();
  runDispatchRole<Local, Schedule>(p, s, warp);
  if (warp == Schedule::LoadAWarp && lane == 0) {
    auto state = cutlass::make_producer_start_state<LoadA>();
    auto load1 = fc1.load_init(shape1, p.fc1, s.tensors);
    auto load2 = fc2.load_init(shape2, p.fc2, s.tensors);
    for (int i = cluster; i < tasks; i += gridDim.x / ClusterM) {
      Task task = taskAt(p, i, tokens, hidden, intermediate);
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
      auto phase = task.fc1 ? KernelTracePhase::LoadFc1 : KernelTracePhase::LoadFc2;
      traceKernel(phase, true, i);
      auto result = task.fc1 ? fc1.load_B(p.fc1, bPipeline, state, load1, coord, iterator, kTiles)
                             : fc2.load_B(p.fc2, bPipeline, state, load2, coord, iterator, kTiles);
      traceKernel(phase, false, i);
      state = get<0>(result);
    }
    bPipeline.producer_tail(state);
  } else {
    runMmaRole<E5M2, Local, Schedule>(p, tokens, s, bPipeline, tPipeline, cPipeline, fc1, accumulators, hidden,
                                      intermediate, warp, cta, cluster, tasks);
  }
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_ROLES_CUH_
