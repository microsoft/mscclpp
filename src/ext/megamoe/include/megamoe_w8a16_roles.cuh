// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_W8A16_ROLES_CUH_
#define MSCCLPP_EXT_MEGAMOE_W8A16_ROLES_CUH_

#include "megamoe_collective.cuh"
#include "megamoe_combine.cuh"
#include "megamoe_dispatch.cuh"
#include "megamoe_epilogue.cuh"
#include "megamoe_mma.cuh"
#include "megamoe_quantization.cuh"
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
  // W8A16Mainloop::load_B sets exact sub-tile transaction bytes per K-tile.
  bParams.is_leader = false;
  bParams.num_consumers = WARP_SIZE;
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

template <int BlockSize, class ScaleLayout>
constexpr bool hasUniformScaleBlocks(ScaleLayout layout) {
  constexpr int Total = decltype(cute::size(layout))::value;
  if (Total % BlockSize != 0) return false;
  for (int block = 0; block < Total / BlockSize; ++block) {
    const auto baseOffset = layout(block * BlockSize);
    for (int i = 1; i < BlockSize; ++i) {
      if (layout(block * BlockSize + i) != baseOffset) return false;
    }
  }
  return true;
}

template <bool Local = false, class Inputs>
MSCCLPP_DEVICE_INLINE void transformWeights(
    typename W8A16CollectiveTypesT<Local>::LoadA& load,
    typename W8A16CollectiveTypesT<Local>::LoadA::PipelineState& loadState,
    typename W8A16CollectiveTypesT<Local>::Transform& transformed,
    typename W8A16CollectiveTypesT<Local>::Transform::PipelineState& transformState, Inputs& inputs, int kTiles) {
  using namespace cute;
  using Types = W8A16CollectiveTypesT<Local>;
  auto copyOp = get<1>(inputs);
  auto& raw = get<2>(inputs);
  auto& converted = get<3>(inputs);
  auto& scaleInputs = get<4>(inputs);
  auto& smemScales = get<2>(scaleInputs);
  auto rRaw = make_tensor<typename Types::Weight>(shape(raw(_, _, _, _, 0)));
  auto rConverted = make_tensor<typename Types::Activation>(shape(rRaw));
  auto packedRaw = recast<uint16_t>(rRaw);
  auto packedConverted = recast<uint32_t>(rConverted);
  constexpr int NumElements = decltype(size(rRaw))::value;
  constexpr int NumScaleBlocks = NumElements / MxfpScaleBlockSize;
  constexpr int PairsPerScaleBlock = MxfpScaleBlockSize / 2;
  static_assert(NumElements == NumScaleBlocks * MxfpScaleBlockSize);
  using ScaleStageLayout = decltype(smemScales(_, _, _, _, _0{}).layout());
  static_assert(hasUniformScaleBlocks<MxfpScaleBlockSize>(ScaleStageLayout{}));
  cutlass::arch::NamedBarrier copyBarrier(128, cutlass::arch::ReservedNamedBarriers::TransformBarrier);

  CUTLASS_PRAGMA_NO_UNROLL
  for (int k = 0; k < kTiles; ++k) {
    load.consumer_wait(loadState);
    transformed.producer_acquire(transformState);
    copy(AutoVectorizingCopy{}, raw(_, _, _, _, loadState.index()), rRaw);
    auto stageScales = smemScales(_, _, _, _, loadState.index());
    uint8_t blockScales[NumScaleBlocks];
#pragma unroll
    for (int b = 0; b < NumScaleBlocks; ++b) {
      blockScales[b] = mscclpp::bit_cast<uint8_t>(stageScales(b * MxfpScaleBlockSize));
    }
    copyBarrier.sync();
    load.consumer_release(loadState);
    ++loadState;

    // Each K32 block shares one E8M0 scale byte; expand once per block and reuse across all 16 BF16 pairs.
#pragma unroll
    for (int b = 0; b < NumScaleBlocks; ++b) {
      const uint32_t expandedScale = expandE8M0ScalePair(blockScales[b]);
#pragma unroll
      for (int j = 0; j < PairsPerScaleBlock; ++j) {
        const int pairIdx = b * PairsPerScaleBlock + j;
        packedConverted(pairIdx) = decodeScaledPairWithExpandedScale(packedRaw(pairIdx), expandedScale);
      }
    }
    copy(copyOp, rConverted, converted(_, _, _, _, transformState.index()));
    cutlass::arch::fence_view_async_tmem_store();
    transformed.producer_commit(transformState);
    ++transformState;
  }
}

template <bool Local, class KernelParams, class Storage, class LoadA, class Transform, class Mainloop,
          class Accumulators>
MSCCLPP_DEVICE_INLINE void runW8A16TransformRole(const KernelParams& params, int nTokens, Storage& sharedStorage,
                                                 LoadA& loadPipeline, Transform& transformPipeline, Mainloop& fc1,
                                                 const ProblemShape& shape1, Accumulators accumulators, int hidden,
                                                 int intermediate, int clusterIdx, int nTasks, int nTokenTiles,
                                                 int nLocalExperts) {
  cutlass::arch::warpgroup_reg_alloc<KernelParams::Collective::TransformRegisters>();
  typename LoadA::PipelineState loadState;
  auto transformState = cutlass::make_producer_start_state<Transform>();
  auto inputs = fc1.transform_init(params.fc1, shape1, accumulators, sharedStorage.tensors);
  const auto scheduler = makeTaskScheduler<KernelParams>(nTokenTiles, nLocalExperts, hidden, intermediate);
  const int numClusters = gridDim.x / KernelParams::Tiles::ClusterSize;
  if (clusterIdx < nTasks) {
    Task nextTask = scheduler.taskAt(params, clusterIdx, nTokens);
    for (int i = clusterIdx; i < nTasks; i += numClusters) {
      Task task = nextTask;
      const bool hasNext = (i + numClusters) < nTasks;
      auto phase = task.fc1 ? KernelTracePhase::TransformFc1 : KernelTracePhase::TransformFc2;
      traceKernel(phase, true, i);
      transformWeights<Local>(loadPipeline, loadState, transformPipeline, transformState, inputs,
                              taskKTiles(params, task, hidden, intermediate));
      if (hasNext) nextTask = scheduler.taskAt(params, i + numClusters, nTokens);
      traceKernel(phase, false, i);
    }
  }
  transformPipeline.producer_tail(transformState);
}

template <bool Local, class KernelParams, class LoadTiles>
MSCCLPP_DEVICE_INLINE void loadW8A16ActivationTiles(const KernelParams& params, const Task& task, int hidden,
                                                    int intermediate, int kTiles, LoadTiles&& loadTiles) {
  using Types = typename KernelParams::Collective;
  if constexpr (Local) {
    if (!task.fc1)
      waitReadyCounter(params.workspace.hiddenReady + size_t(task.block) * W4ReadyCounterStride,
                       fc1CompletionCount<KernelParams>(intermediate));
    loadTiles(0, kTiles);
    return;
  }
  if (task.fc1) {
    constexpr int TilesPerChunk = Types::DispatchChunk / (KernelParams::Tiles::K * sizeof(__bfloat16));
    for (int first = 0; first < kTiles; first += TilesPerChunk) {
      traceKernel(KernelTracePhase::TokenReady, true, first / TilesPerChunk);
      waitReadyCounter(w4InputChunkCounter<Types>(params.workspace, hidden, task.block, first / TilesPerChunk),
                       task.tokens.nTokens);
      traceKernel(KernelTracePhase::TokenReady, false, first / TilesPerChunk);
      loadTiles(first, min(TilesPerChunk, kTiles - first));
    }
  } else {
    traceKernel(KernelTracePhase::TokenReady, true, int(task.fc1));
    waitReadyCounter(params.workspace.hiddenReady + size_t(task.block) * W4ReadyCounterStride,
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
      cute::append(std::declval<Mainloop>().partition_accumulator_shape(), cute::Int<Types::Accumulate::Stages>{})));
  Mainloop fc1;
  Mainloop fc2;
  Accumulators accumulators;
  ProblemShape shape1;
  ProblemShape shape2;
  int nTokenTiles;
  int nTasks;
  int nLocalExperts;
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
  auto acc =
      Mainloop::TiledMma::make_fragment_C(append(fc1.partition_accumulator_shape(), Int<Types::Accumulate::Stages>{}));
  acc.data() = sharedStorage.tmem;
  const int nTasks = nTokenTiles * (fc1TaskTiles<KernelParams>(intermediate) + fc2TaskTiles<KernelParams>(hidden));
  const int nLocalExperts = params.config.numExperts / params.config.worldSize;
  return W8A16ComputeContext<KernelParams>{
      fc1,
      fc2,
      acc,
      ProblemShape{2 * intermediate, params.workspace.poolRows, hidden, nLocalExperts},
      ProblemShape{hidden, params.workspace.poolRows, intermediate, nLocalExperts},
      nTokenTiles,
      nTasks,
      nLocalExperts};
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
  const auto scheduler =
      makeTaskScheduler<KernelParams>(computeCtx.nTokenTiles, computeCtx.nLocalExperts, hidden, intermediate);
  const int numClusters = gridDim.x / KernelParams::Tiles::ClusterSize;
  if (threadLoc.clusterIdx < computeCtx.nTasks) {
    Task nextTask = scheduler.taskAt(params, threadLoc.clusterIdx, nTokens);
    for (int i = threadLoc.clusterIdx; i < computeCtx.nTasks; i += numClusters) {
      Task task = nextTask;
      const bool hasNext = (i + numClusters) < computeCtx.nTasks;
      auto coord = taskCoord(params, task, threadLoc.ctaIdx);
      int kTiles = taskKTiles(params, task, hidden, intermediate);
      const bool weightEvictNormal =
          KernelParams::LocalExpert ? (nTokens > KernelParams::Tiles::N) : task.tokens.weightEvictNormal();
      auto iterator = cute::make_coord_iterator(0, kTiles);
      auto phase = task.fc1 ? KernelTracePhase::LoadFc1 : KernelTracePhase::LoadFc2;
      traceKernel(phase, true, i);
      auto result =
          task.fc1
              ? computeCtx.fc1.load_A(params.fc1, aPipeline, state, load1, coord, iterator, kTiles, weightEvictNormal)
              : computeCtx.fc2.load_A(params.fc2, aPipeline, state, load2, coord, iterator, kTiles, weightEvictNormal);
      if (hasNext) nextTask = scheduler.taskAt(params, i + numClusters, nTokens);
      traceKernel(phase, false, i);
      state = get<0>(result);
    }
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
  const auto scheduler =
      makeTaskScheduler<KernelParams>(computeCtx.nTokenTiles, computeCtx.nLocalExperts, hidden, intermediate);
  const int numClusters = gridDim.x / KernelParams::Tiles::ClusterSize;
  if (threadLoc.clusterIdx < computeCtx.nTasks) {
    Task nextTask = scheduler.taskAt(params, threadLoc.clusterIdx, nTokens);
    for (int i = threadLoc.clusterIdx; i < computeCtx.nTasks; i += numClusters) {
      Task task = nextTask;
      const bool hasNext = (i + numClusters) < computeCtx.nTasks;
      auto coord = taskCoord(params, task, threadLoc.ctaIdx);
      int kTiles = taskKTiles(params, task, hidden, intermediate);
      auto phase = task.fc1 ? KernelTracePhase::LoadFc1 : KernelTracePhase::LoadFc2;
      traceKernel(phase, true, i);
      loadW8A16ActivationTiles<Local>(params, task, hidden, intermediate, kTiles, [&](int first, int count) {
        auto iterator = cute::make_coord_iterator(first, kTiles);
        auto result = task.fc1 ? computeCtx.fc1.load_B(params.fc1, bPipeline, state, load1, coord, iterator, count,
                                                       task.tokens.nTokens)
                               : computeCtx.fc2.load_B(params.fc2, bPipeline, state, load2, coord, iterator, count,
                                                       task.tokens.nTokens);
        state = get<0>(result);
      });
      if (hasNext) nextTask = scheduler.taskAt(params, i + numClusters, nTokens);
      traceKernel(phase, false, i);
    }
  }
  bPipeline.producer_tail(state);
}

template <bool Local, class KernelParams, class Storage, class LoadA, class Transform, class ComputeCtx>
MSCCLPP_DEVICE_INLINE void runW8A16TransformStage(const KernelParams& params, Storage& sharedStorage, LoadA& aPipeline,
                                                  Transform& tPipeline, ComputeCtx& computeCtx,
                                                  const KernelThreadLocation& threadLoc, int nTokens) {
  runW8A16TransformRole<Local>(params, nTokens, sharedStorage, aPipeline, tPipeline, computeCtx.fc1, computeCtx.shape1,
                               computeCtx.accumulators, params.config.hidden, params.config.intermediate,
                               threadLoc.clusterIdx, computeCtx.nTasks, computeCtx.nTokenTiles,
                               computeCtx.nLocalExperts);
}

template <bool Local, class Schedule, class KernelParams, class Storage, class Pipelines, class ComputeCtx>
MSCCLPP_DEVICE_INLINE void runW8A16MmaStage(const KernelParams& params, Storage& sharedStorage, Pipelines& pipelines,
                                            ComputeCtx& computeCtx, const KernelThreadLocation& threadLoc,
                                            int nTokens) {
  runW8A16MmaRole<Local, Schedule>(
      params, nTokens, sharedStorage, pipelines.loadB, pipelines.transform, pipelines.accumulate, computeCtx.fc1,
      computeCtx.accumulators, params.config.hidden, params.config.intermediate, threadLoc.warpIdx, threadLoc.ctaIdx,
      threadLoc.clusterIdx, computeCtx.nTasks, computeCtx.nTokenTiles, computeCtx.nLocalExperts);
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
  using Types = W8A16CollectiveTypesT<Local>;
  if constexpr (!Local) {
    static_assert(4 * Types::EpilogueRegisters + 4 * Types::TransformRegisters + 4 * Types::DispatchRegisters +
                      4 * TransferRegisters <=
                  Types::NumWarps * EntryRegisters);
  }
  if (warpIdx >= Schedule::EpilogueBegin && warpIdx < Schedule::EpilogueEnd) {
    runRole.template operator()<WarpRole::Epilogue>();
  } else if (warpIdx >= Schedule::TransformBegin && warpIdx < Schedule::TransformEnd) {
    runRole.template operator()<WarpRole::Transform>();
  } else if constexpr (!Local) {
    if (warpIdx >= Schedule::DispatchBegin && warpIdx < Schedule::DispatchEnd) {
      cutlass::arch::warpgroup_reg_dealloc<Types::DispatchRegisters>();
      runRole.template operator()<WarpRole::Dispatch>();
      return;
    }
    cutlass::arch::warpgroup_reg_dealloc<TransferRegisters>();
    if (warpIdx == Schedule::LoadAWarp) {
      runRole.template operator()<WarpRole::LoadA>();
    } else if (warpIdx == Schedule::LoadBWarp) {
      runRole.template operator()<WarpRole::LoadB>();
    } else if (warpIdx == Schedule::MmaWarp) {
      runRole.template operator()<WarpRole::Mma>();
    }
  } else {
    cutlass::arch::warpgroup_reg_dealloc<TransferRegisters>();
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
  (void)output;
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
  if constexpr (!Local) {
    if (!params.config.computeOnly) completeStandaloneCombineEpoch(params, nTokens);
  }
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_W8A16_ROLES_CUH_
