// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_W8A16_ROLES_CUH_
#define MSCCLPP_EXT_MEGAMOE_W8A16_ROLES_CUH_

#include "megamoe_collective.cuh"
#include "megamoe_dispatch.cuh"
#include "megamoe_epilogue.cuh"
#include "megamoe_mma.cuh"
#include "megamoe_routing.cuh"
#include "megamoe_tasks.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

template <bool Local>
struct W8A16Pipelines {
  using Types = W8A16CollectiveTypesT<Local>;
  typename Types::LoadA loadA;
  typename Types::LoadB loadB;
  typename Types::Transform transform;
  typename Types::Accumulate accumulate;
  cute::TMEM::Allocator2Sm allocator;
};

template <bool Local, class Storage>
MSCCLPP_DEVICE_INLINE W8A16Pipelines<Local> initW8A16PipelinesAndStorage(Storage& sharedStorage,
                                                                         const KernelThreadLocation& threadLoc) {
  using namespace cute;
  using Types = W8A16CollectiveTypesT<Local>;
  using Mainloop = typename Types::Mainloop;
  using LoadA = typename Types::LoadA;
  using LoadB = typename Types::LoadB;
  using Transform = typename Types::Transform;
  using Accumulate = typename Types::Accumulate;
  using ClusterShape = typename Types::ClusterShape;
  using Schedule = typename Types::WarpSchedule;

  typename LoadA::Params aParams{};
  aParams.role = threadLoc.warpIdx == Schedule::LoadAWarp
                     ? LoadA::ThreadCategory::Producer
                     : (threadLoc.warpIdx >= Schedule::TransformBegin && threadLoc.warpIdx < Schedule::TransformEnd
                            ? LoadA::ThreadCategory::Consumer
                            : LoadA::ThreadCategory::NonParticipant);
  aParams.is_leader = threadLoc.laneIdx == 0;
  aParams.num_consumers = 128;
  aParams.transaction_bytes = Mainloop::TmaTransactionBytes_A;
  aParams.initializing_warp = Schedule::LoadAWarp;
  LoadA aPipeline(sharedStorage.mainloopA, aParams, ClusterShape{}, cutlass::McastDirection::kRow, true_type{},
                  false_type{});

  typename LoadB::Params bParams{};
  bParams.role = threadLoc.warpIdx == Schedule::LoadBWarp
                     ? LoadB::ThreadCategory::Producer
                     : (threadLoc.warpIdx == Schedule::MmaWarp ? LoadB::ThreadCategory::Consumer
                                                               : LoadB::ThreadCategory::NonParticipant);
  bParams.is_leader = threadLoc.laneIdx == 0 && threadLoc.ctaIdx == 0 && threadLoc.warpIdx == Schedule::LoadBWarp;
  bParams.num_consumers = 32;
  bParams.transaction_bytes = Mainloop::TmaTransactionBytes_B;
  bParams.initializing_warp = Schedule::LoadBWarp;
  LoadB bPipeline(sharedStorage.mainloopB, bParams, ClusterShape{}, cutlass::McastDirection::kCol, true_type{},
                  false_type{});

  typename Transform::Params tParams{};
  tParams.role = threadLoc.warpIdx >= Schedule::TransformBegin && threadLoc.warpIdx < Schedule::TransformEnd
                     ? Transform::ThreadCategory::Producer
                     : (threadLoc.warpIdx == Schedule::MmaWarp ? Transform::ThreadCategory::Consumer
                                                               : Transform::ThreadCategory::NonParticipant);
  tParams.consumer_arv_count = 1;
  tParams.producer_arv_count = 256;
  tParams.initializing_warp = Schedule::TransformBegin;
  Transform tPipeline(sharedStorage.transformed, tParams, ClusterShape{}, true_type{}, false_type{});

  auto cPipeline =
      makeAccumulatorPipeline<Accumulate, Schedule, ClusterShape>(sharedStorage.accumulated, threadLoc.warpIdx);
  aPipeline.init_masks(ClusterShape{}, cute::block_id_in_cluster(), cutlass::McastDirection::kRow);
  bPipeline.init_masks(ClusterShape{}, cutlass::McastDirection::kCol);
  tPipeline.init_masks(ClusterShape{});
  cPipeline.init_masks(ClusterShape{});
  cute::TMEM::Allocator2Sm allocator{};
  if (threadLoc.warpIdx == Schedule::MmaWarp) allocator.allocate(512, &sharedStorage.tmem);
  if constexpr (!Local)
    initDispatchStorage<Types::DispatchStages, Schedule>(sharedStorage, threadLoc.warpIdx, threadLoc.laneIdx);
  cutlass::arch::fence_barrier_init();
  return W8A16Pipelines<Local>{aPipeline, bPipeline, tPipeline, cPipeline, allocator};
}

template <bool Local, class KernelParams>
MSCCLPP_DEVICE_INLINE int runW8A16RoutingStage(const KernelParams& params, int nTokens, const int32_t* ids,
                                               const float* scores) {
  if constexpr (Local) {
    __syncthreads();
    cute::cluster_sync();
    return (nTokens + KernelParams::Tiles::N - 1) / KernelParams::Tiles::N;
  } else {
    traceKernel(KernelTracePhase::Routing, true);
    if (params.config.computeOnly) {
      prepareLocalRoutes(params, nTokens, ids);
    } else {
      prepareRoutes<false>(params, nTokens, ids, scores);
    }
    traceKernel(KernelTracePhase::Routing, false);
    return params.workspace.control->nTokenTiles;
  }
}

template <bool Local, class KernelParams, class Storage, class LoadA, class Transform, class Mainloop,
          class Accumulators>
MSCCLPP_DEVICE_INLINE void runW8A16TransformRole(const KernelParams& params, int nTokens, Storage& sharedStorage,
                                                 LoadA& loadPipeline, Transform& transformPipeline, Mainloop& fc1,
                                                 const ProblemShape& shape1, Accumulators accumulators, int hidden,
                                                 int intermediate, int clusterIdx, int nTasks) {
  cutlass::arch::warpgroup_reg_alloc<KernelParams::Collective::EpilogueRegisters>();
  typename LoadA::PipelineState loadState;
  auto transformState = cutlass::make_producer_start_state<Transform>();
  auto inputs = fc1.transform_init(params.fc1, shape1, accumulators, sharedStorage.tensors);
  for (int i = clusterIdx; i < nTasks; i += gridDim.x / KernelParams::Tiles::ClusterSize) {
    Task task = taskAt(params, i, hidden, intermediate, nTokens);
    auto phase = task.fc1 ? KernelTracePhase::TransformFc1 : KernelTracePhase::TransformFc2;
    traceKernel(phase, true, i);
    transformWeights<Local>(loadPipeline, loadState, transformPipeline, transformState, inputs,
                            taskKTiles(params, task, hidden, intermediate));
    traceKernel(phase, false, i);
  }
  transformPipeline.producer_tail(transformState);
}

template <bool Local, class KernelParams, class LoadTiles>
MSCCLPP_DEVICE_INLINE void loadW8A16ActivationTiles(const KernelParams& params, const Task& task, int hidden,
                                                    int intermediate, int kTiles, LoadTiles&& loadTiles) {
  using Types = typename KernelParams::Collective;
  if constexpr (Local) {
    if (!task.fc1)
      waitAtLeast<int, scopeDevice>(params.workspace.hiddenReady + size_t(task.block) * W4ReadyCounterStride,
                                    fc1CompletionCount<KernelParams>(intermediate));
    loadTiles(0, kTiles);
    return;
  }
  if (task.fc1) {
    constexpr int TilesPerChunk = Types::DispatchChunk / (KernelParams::Tiles::K * sizeof(__bfloat16));
    for (int first = 0; first < kTiles; first += TilesPerChunk) {
      traceKernel(KernelTracePhase::TokenReady, true, first / TilesPerChunk);
      waitAtLeast<int, scopeDevice>(
          w4InputChunkCounter<Types>(params.workspace, hidden, task.block, first / TilesPerChunk), task.tokens.nTokens);
      traceKernel(KernelTracePhase::TokenReady, false, first / TilesPerChunk);
      loadTiles(first, min(TilesPerChunk, kTiles - first));
    }
  } else {
    traceKernel(KernelTracePhase::TokenReady, true, int(task.fc1));
    waitAtLeast<int, scopeDevice>(params.workspace.hiddenReady + size_t(task.block) * W4ReadyCounterStride,
                                  fc1CompletionCount<KernelParams>(intermediate));
    traceKernel(KernelTracePhase::TokenReady, false, int(task.fc1));
    loadTiles(0, kTiles);
  }
}

template <class KernelParams>
struct W8A16ComputeContext {
  using Types = typename KernelParams::Collective;
  using Mainloop = typename Types::Mainloop;
  using Accumulators = decltype(Mainloop::TiledMma::make_fragment_C(
      cute::append(std::declval<Mainloop>().partition_accumulator_shape(), cute::_2{})));
  Mainloop fc1;
  Mainloop fc2;
  Accumulators accumulators;
  ProblemShape shape1;
  ProblemShape shape2;
  int nTasks;
};

template <class KernelParams, class Storage>
MSCCLPP_DEVICE_INLINE W8A16ComputeContext<KernelParams> makeW8A16ComputeContext(const KernelParams& params,
                                                                                Storage& sharedStorage, int ctaIdx,
                                                                                int nTokenTiles) {
  using namespace cute;
  using Types = typename KernelParams::Collective;
  using Mainloop = typename Types::Mainloop;
  using ClusterShape = typename Types::ClusterShape;
  const int hidden = params.config.hidden;
  const int intermediate = params.config.intermediate;
  Mainloop fc1(params.fc1, ClusterShape{}, ctaIdx);
  Mainloop fc2(params.fc2, ClusterShape{}, ctaIdx);
  auto acc = Mainloop::TiledMma::make_fragment_C(append(fc1.partition_accumulator_shape(), _2{}));
  acc.data() = sharedStorage.tmem;
  const int nTasks = nTokenTiles * (fc1TaskTiles<KernelParams>(intermediate) + fc2TaskTiles<KernelParams>(hidden));
  const int nLocalExperts = params.config.numExperts / params.config.worldSize;
  return W8A16ComputeContext<KernelParams>{
      fc1,
      fc2,
      acc,
      ProblemShape{2 * intermediate, params.workspace.poolRows, hidden, nLocalExperts},
      ProblemShape{hidden, params.workspace.poolRows, intermediate, nLocalExperts},
      nTasks};
}

template <class KernelParams, class Storage, class LoadA, class ComputeCtx>
MSCCLPP_DEVICE_INLINE void runW8A16LoadAStage(const KernelParams& params, Storage& sharedStorage, LoadA& aPipeline,
                                              ComputeCtx& computeCtx, const KernelThreadLocation& threadLoc,
                                              int nTokens) {
  using namespace cute;
  if (threadLoc.laneIdx != 0) return;
  const int hidden = params.config.hidden;
  const int intermediate = params.config.intermediate;
  auto state = cutlass::make_producer_start_state<LoadA>();
  auto load1 = computeCtx.fc1.load_init(computeCtx.shape1, params.fc1, sharedStorage.tensors);
  auto load2 = computeCtx.fc2.load_init(computeCtx.shape2, params.fc2, sharedStorage.tensors);
  for (int i = threadLoc.clusterIdx; i < computeCtx.nTasks; i += gridDim.x / KernelParams::Tiles::ClusterSize) {
    Task task = taskAt(params, i, hidden, intermediate, nTokens);
    auto coord = taskCoord(params, task, threadLoc.ctaIdx);
    int kTiles = taskKTiles(params, task, hidden, intermediate);
    auto iterator = cute::make_coord_iterator(0, kTiles);
    auto phase = task.fc1 ? KernelTracePhase::LoadFc1 : KernelTracePhase::LoadFc2;
    traceKernel(phase, true, i);
    auto result = task.fc1 ? computeCtx.fc1.load_A(params.fc1, aPipeline, state, load1, coord, iterator, kTiles)
                           : computeCtx.fc2.load_A(params.fc2, aPipeline, state, load2, coord, iterator, kTiles);
    traceKernel(phase, false, i);
    state = get<0>(result);
  }
  aPipeline.producer_tail(state);
}

template <bool Local, class KernelParams, class Storage, class LoadB, class ComputeCtx>
MSCCLPP_DEVICE_INLINE void runW8A16LoadBStage(const KernelParams& params, Storage& sharedStorage, LoadB& bPipeline,
                                              ComputeCtx& computeCtx, const KernelThreadLocation& threadLoc,
                                              int nTokens) {
  using namespace cute;
  if (threadLoc.laneIdx != 0) return;
  const int hidden = params.config.hidden;
  const int intermediate = params.config.intermediate;
  auto state = cutlass::make_producer_start_state<LoadB>();
  auto load1 = computeCtx.fc1.load_init(computeCtx.shape1, params.fc1, sharedStorage.tensors);
  auto load2 = computeCtx.fc2.load_init(computeCtx.shape2, params.fc2, sharedStorage.tensors);
  for (int i = threadLoc.clusterIdx; i < computeCtx.nTasks; i += gridDim.x / KernelParams::Tiles::ClusterSize) {
    Task task = taskAt(params, i, hidden, intermediate, nTokens);
    auto coord = taskCoord(params, task, threadLoc.ctaIdx);
    int kTiles = taskKTiles(params, task, hidden, intermediate);
    int mmaRows = (int(task.tokens.nTokens) + 15) / 16 * 16;
    auto shiftLoadB = [&](const auto& baseInputs, const auto& mainloopParams) {
      auto rowLayout = mainloopParams.tma_load_b.get_tma_tensor(make_shape(1, 1, 1)).layout();
      auto rowOffset = rowLayout(
          make_coord(threadLoc.ctaIdx * (mmaRows - KernelParams::Tiles::N) / KernelParams::Tiles::ClusterSize, 0, 0));
      auto shiftedB = make_tensor(get<3>(baseInputs).data() + rowOffset, get<3>(baseInputs).layout());
      return cute::make_tuple(get<0>(baseInputs), get<1>(baseInputs), get<2>(baseInputs), shiftedB, get<4>(baseInputs),
                              get<5>(baseInputs), get<6>(baseInputs), get<7>(baseInputs), get<8>(baseInputs));
    };
    auto phase = task.fc1 ? KernelTracePhase::LoadFc1 : KernelTracePhase::LoadFc2;
    traceKernel(phase, true, i);
    loadW8A16ActivationTiles<Local>(params, task, hidden, intermediate, kTiles, [&](int first, int count) {
      auto iterator = cute::make_coord_iterator(first, kTiles);
      auto result = task.fc1 ? computeCtx.fc1.load_B(params.fc1, bPipeline, state, shiftLoadB(load1, params.fc1), coord,
                                                     iterator, count)
                             : computeCtx.fc2.load_B(params.fc2, bPipeline, state, shiftLoadB(load2, params.fc2), coord,
                                                     iterator, count);
      state = get<0>(result);
    });
    traceKernel(phase, false, i);
  }
  bPipeline.producer_tail(state);
}

template <bool Local, class KernelParams, class Storage, class LoadA, class Transform, class ComputeCtx>
MSCCLPP_DEVICE_INLINE void runW8A16TransformStage(const KernelParams& params, Storage& sharedStorage, LoadA& aPipeline,
                                                  Transform& tPipeline, ComputeCtx& computeCtx,
                                                  const KernelThreadLocation& threadLoc, int nTokens) {
  runW8A16TransformRole<Local>(params, nTokens, sharedStorage, aPipeline, tPipeline, computeCtx.fc1, computeCtx.shape1,
                               computeCtx.accumulators, params.config.hidden, params.config.intermediate,
                               threadLoc.clusterIdx, computeCtx.nTasks);
}

template <bool Local, class Schedule, class KernelParams, class Storage, class Pipelines, class ComputeCtx>
MSCCLPP_DEVICE_INLINE void runW8A16MmaStage(const KernelParams& params, Storage& sharedStorage, Pipelines& pipelines,
                                            ComputeCtx& computeCtx, const KernelThreadLocation& threadLoc,
                                            int nTokens) {
  runW8A16MmaRole<Local, Schedule>(params, nTokens, sharedStorage, pipelines.loadB, pipelines.transform,
                                   pipelines.accumulate, computeCtx.fc1, computeCtx.accumulators, params.config.hidden,
                                   params.config.intermediate, threadLoc.warpIdx, threadLoc.ctaIdx,
                                   threadLoc.clusterIdx, computeCtx.nTasks);
}

template <class Schedule, int LocalMode, class KernelParams, class Storage, class Accumulate, class ComputeCtx>
MSCCLPP_DEVICE_INLINE void runW8A16EpilogueStage(const KernelParams& params, Storage& sharedStorage,
                                                 Accumulate& cPipeline, ComputeCtx& computeCtx,
                                                 const KernelThreadLocation& threadLoc, int nTokens,
                                                 __bfloat16* output) {
  runEpilogueRole<Schedule, LocalMode>(params, sharedStorage, cPipeline, computeCtx.accumulators, params.config.hidden,
                                       params.config.intermediate, threadLoc.clusterIdx, computeCtx.nTasks, output,
                                       nTokens);
}

template <bool Local, class Schedule, class RoleFn>
MSCCLPP_DEVICE_INLINE void dispatchW8A16WarpRole(int warpIdx, RoleFn&& runRole) {
  if (warpIdx >= Schedule::EpilogueBegin && warpIdx < Schedule::EpilogueEnd) {
    runRole.template operator()<WarpRole::Epilogue>();
  } else if (warpIdx >= Schedule::TransformBegin && warpIdx < Schedule::TransformEnd) {
    runRole.template operator()<WarpRole::Transform>();
  } else {
    cutlass::arch::warpgroup_reg_dealloc<TransferRegisters>();
    if constexpr (!Local) {
      if (warpIdx >= Schedule::DispatchBegin && warpIdx < Schedule::DispatchEnd) {
        runRole.template operator()<WarpRole::Dispatch>();
        return;
      }
    }
    if (warpIdx == Schedule::LoadAWarp) {
      runRole.template operator()<WarpRole::LoadA>();
    } else if (warpIdx == Schedule::LoadBWarp) {
      runRole.template operator()<WarpRole::LoadB>();
    } else if (warpIdx == Schedule::MmaWarp) {
      runRole.template operator()<WarpRole::Mma>();
    }
  }
}

template <int Hidden, bool Local, class KernelParams, class Storage, class Allocator>
MSCCLPP_DEVICE_INLINE void finishW8A16KernelStage(const KernelParams& params, Storage& sharedStorage,
                                                  Allocator& allocator, int warpIdx, int nTokens, __bfloat16* output) {
  using Schedule = typename W8A16CollectiveTypesT<Local>::WarpSchedule;
  constexpr int RestoreRegisters = Local ? LocalEntryRegisters : EntryRegisters;
  __syncthreads();
  if ((warpIdx >= Schedule::EpilogueBegin && warpIdx < Schedule::EpilogueEnd) ||
      (warpIdx >= Schedule::TransformBegin && warpIdx < Schedule::TransformEnd)) {
    cutlass::arch::warpgroup_reg_dealloc<RestoreRegisters>();
  } else {
    cutlass::arch::warpgroup_reg_alloc<RestoreRegisters>();
  }
  cute::cluster_sync();
  if (warpIdx == Schedule::MmaWarp) {
    allocator.release_allocation_lock();
    allocator.free(sharedStorage.tmem, 512);
  }
  if constexpr (!Local) finishAndCombine<Hidden>(params, nTokens, output);
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_W8A16_ROLES_CUH_
