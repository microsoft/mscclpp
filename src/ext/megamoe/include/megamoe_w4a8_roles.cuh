// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_W4A8_ROLES_CUH_
#define MSCCLPP_EXT_MEGAMOE_W4A8_ROLES_CUH_

#include "megamoe_dispatch.cuh"
#include "megamoe_epilogue.cuh"
#include "megamoe_mma.cuh"
#include "megamoe_routing.cuh"
#include "megamoe_tasks.cuh"
#include "megamoe_trace.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

template <bool Activation, class Types, class Storage>
MSCCLPP_DEVICE_INLINE auto makeW4A8LoadPipeline(Storage& storage, int warpIdx, int laneIdx, int ctaIdx) {
  using namespace cute;
  using Load = typename Types::Load;
  using Mainloop = typename Types::Mainloop;
  using Schedule = typename Types::WarpSchedule;
  using ClusterShape = typename Types::ClusterShape;
  constexpr int InitializingWarp = Activation ? Schedule::ActivationLoadWarp : Schedule::LoadWarp;
  constexpr bool SharedProducers = !Activation && !Types::SplitPipelines && Types::LoadWarps == 2;
  bool producer = warpIdx == InitializingWarp || (SharedProducers && warpIdx == Schedule::ActivationLoadWarp);
  typename Load::Params pipelineParams{};
  pipelineParams.role =
      producer ? Load::ThreadCategory::Producer
               : (warpIdx == Schedule::MmaWarp ? Load::ThreadCategory::Consumer : Load::ThreadCategory::NonParticipant);
  pipelineParams.is_leader = false;
  pipelineParams.transaction_bytes =
      Activation ? Mainloop::ActivationTransactionBytes
                 : (Types::LoadWarps == 1 ? Mainloop::TmaTransactionBytes
                                          : (warpIdx == Schedule::LoadWarp ? Mainloop::WeightTransactionBytes
                                                                           : Mainloop::ActivationTransactionBytes));
  if constexpr (Types::SplitPipelines && !Activation)
    pipelineParams.transaction_bytes = Mainloop::WeightTransactionBytes;
  pipelineParams.initializing_warp = InitializingWarp;
  Load pipeline(storage, pipelineParams, ClusterShape{}, false_type{}, false_type{});
  if constexpr (SharedProducers) {
    if (warpIdx == InitializingWarp) {
      // Both loaders arrive before the shared stage can become ready.
      cutlass::arch::detail::initialize_barrier_array_pair_aligned<decltype(storage.full_barrier_),
                                                                   decltype(storage.empty_barrier_), Load::Stages>(
          storage.full_barrier_, storage.empty_barrier_, 2, 1);
    }
  } else {
    Load::init_barriers(storage, pipelineParams, ClusterShape{});
  }
  return pipeline;
}

template <class KernelParams, class Storage>
MSCCLPP_DEVICE_INLINE void runW4A8LoadRole(const KernelParams& params, Storage& sharedStorage,
                                           typename KernelParams::Collective::Load& loadPipeline,
                                           typename KernelParams::Collective::Load& activationPipeline,
                                           typename KernelParams::Collective::Mainloop& fc1,
                                           typename KernelParams::Collective::Mainloop& fc2, const ProblemShape& shape1,
                                           const ProblemShape& shape2, int hidden, int intermediate, int warpIdx,
                                           int laneIdx, int ctaIdx, int clusterIdx, int nTasks, int nTokenTiles,
                                           int nLocalExperts) {
  using namespace cute;
  using Types = typename KernelParams::Collective;
  using Load = typename KernelParams::Collective::Load;
  using Schedule = typename Types::WarpSchedule;
  (void)ctaIdx;
  auto& producerPipeline =
      Types::SplitPipelines && warpIdx == Schedule::ActivationLoadWarp ? activationPipeline : loadPipeline;
  auto state = cutlass::make_producer_start_state<Load>();
  auto load1 = fc1.load_init(shape1, sharedStorage.tensors);
  auto load2 = fc2.load_init(shape2, sharedStorage.tensors);
  const auto scheduler = makeTaskScheduler<KernelParams>(nTokenTiles, nLocalExperts, hidden, intermediate);
  const int numClusters = gridDim.x / KernelParams::Tiles::ClusterSize;
  if (clusterIdx < nTasks) {
    Task nextTask = scheduler.taskAt(params, clusterIdx);
    for (int i = clusterIdx; i < nTasks; i += numClusters) {
      Task task = nextTask;
      const bool hasNext = (i + numClusters) < nTasks;
      const bool weightEvictNormal =
          (Types::LoadWarps == 1 || warpIdx == Schedule::LoadWarp) && task.tokens.weightEvictNormal();
      if (!task.fc1 && (Types::LoadWarps == 1 || warpIdx == Schedule::ActivationLoadWarp)) {
        traceKernel(KernelTracePhase::TokenReady, true, int(task.fc1));
        if (laneIdx == 0)
          waitAtLeast<int, scopeDevice>(params.workspace.hiddenReady + size_t(task.block) * W4ReadyCounterStride,
                                        fc1CompletionCount<KernelParams>(intermediate));
        __syncwarp();
        traceKernel(KernelTracePhase::TokenReady, false, int(task.fc1));
      }
      auto coord = taskCoord(params, task, ctaIdx);
      int kTiles = taskKTiles(params, task, hidden, intermediate);
      auto phase = task.fc1 ? KernelTracePhase::LoadFc1 : KernelTracePhase::LoadFc2;
      traceKernel(phase, true, i);
      auto loadTiles = [&](int first, int count) {
        auto iterator = cute::make_coord_iterator(first, kTiles);
        constexpr ActivationDtype ActDtype = KernelParams::Activation;
        auto result = [&](auto& firstMainloop, auto& secondMainloop) {
          if constexpr (Types::LoadWarps == 2) {
            if (warpIdx == Schedule::LoadWarp)
              return task.fc1
                         ? firstMainloop.template load<1, ActDtype>(producerPipeline, state, load1, coord, iterator,
                                                                    count, task.tokens.nTokens, weightEvictNormal)
                         : secondMainloop.template load<1, ActDtype>(producerPipeline, state, load2, coord, iterator,
                                                                     count, task.tokens.nTokens, weightEvictNormal);
            return task.fc1 ? firstMainloop.template load<2, ActDtype>(producerPipeline, state, load1, coord, iterator,
                                                                       count, task.tokens.nTokens, weightEvictNormal)
                            : secondMainloop.template load<2, ActDtype>(producerPipeline, state, load2, coord, iterator,
                                                                        count, task.tokens.nTokens, weightEvictNormal);
          } else {
            return task.fc1 ? firstMainloop.template load<0, ActDtype>(loadPipeline, state, load1, coord, iterator,
                                                                       count, task.tokens.nTokens, weightEvictNormal)
                            : secondMainloop.template load<0, ActDtype>(loadPipeline, state, load2, coord, iterator,
                                                                        count, task.tokens.nTokens, weightEvictNormal);
          }
        }(fc1, fc2);
        state = get<0>(result);
      };
      if (task.fc1 && (Types::LoadWarps == 1 || warpIdx == Schedule::ActivationLoadWarp)) {
        constexpr int TilesPerChunk = Types::DispatchChunk / KernelParams::Tiles::K;
        for (int first = 0; first < kTiles; first += TilesPerChunk) {
          traceKernel(KernelTracePhase::TokenReady, true, first / TilesPerChunk);
          if (laneIdx == 0)
            waitAtLeast<int, scopeDevice>(
                w4InputChunkCounter<Types>(params.workspace, hidden, task.block, first / TilesPerChunk),
                task.tokens.nTokens);
          __syncwarp();
          traceKernel(KernelTracePhase::TokenReady, false, first / TilesPerChunk);
          int count = min(TilesPerChunk, kTiles - first);
          if constexpr (Types::DispatchChunk >= 4096) {
            if (first == 0 && hasNext && count > 1) {
              loadTiles(0, 1);
              nextTask = scheduler.taskAt(params, i + numClusters);
              loadTiles(1, count - 1);
            } else {
              loadTiles(first, count);
              if (first == 0 && hasNext) nextTask = scheduler.taskAt(params, i + numClusters);
            }
          } else {
            loadTiles(first, count);
            if (first == 0 && hasNext) nextTask = scheduler.taskAt(params, i + numClusters);
          }
        }
      } else {
        if constexpr (Types::DispatchChunk >= 4096) {
          if (hasNext && kTiles > 1) {
            loadTiles(0, 1);
            nextTask = scheduler.taskAt(params, i + numClusters);
            loadTiles(1, kTiles - 1);
          } else {
            loadTiles(0, kTiles);
            if (hasNext) nextTask = scheduler.taskAt(params, i + numClusters);
          }
        } else {
          loadTiles(0, kTiles);
          if (hasNext) nextTask = scheduler.taskAt(params, i + numClusters);
        }
      }
      traceKernel(phase, false, i);
    }
  }
  fc1.load_tail(producerPipeline, state);
}

template <class Types>
struct W4A8Pipelines {
  using Allocator =
      cute::conditional_t<Types::Tiles::ClusterSize == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm>;
  typename Types::Load load;
  typename Types::Load activation;
  typename Types::Accumulate accumulate;
  Allocator allocator;
};

template <class Types, class Storage>
MSCCLPP_DEVICE_INLINE W4A8Pipelines<Types> initW4A8PipelinesAndStorage(Storage& sharedStorage,
                                                                       const KernelThreadLocation& threadLoc) {
  using Schedule = typename Types::WarpSchedule;
  using ClusterShape = typename Types::ClusterShape;
  using Accumulate = typename Types::Accumulate;
  auto loadPipeline = makeW4A8LoadPipeline<false, Types>(sharedStorage.mainloop, threadLoc.warpIdx, threadLoc.laneIdx,
                                                         threadLoc.ctaIdx);
  auto activationPipeline = [&](auto& activationStorage) {
    if constexpr (Types::SplitPipelines) {
      return makeW4A8LoadPipeline<true, Types>(activationStorage, threadLoc.warpIdx, threadLoc.laneIdx,
                                               threadLoc.ctaIdx);
    } else {
      return loadPipeline;
    }
  }(sharedStorage.activationLoad);
  auto accumulatePipeline =
      makeAccumulatorPipeline<Accumulate, Schedule, ClusterShape>(sharedStorage.accumulated, threadLoc.warpIdx);
  typename W4A8Pipelines<Types>::Allocator allocator{};
  if (threadLoc.warpIdx < Schedule::DispatchBegin) {
    loadPipeline.init_masks(ClusterShape{}, cute::block_id_in_cluster());
    if constexpr (Types::SplitPipelines) activationPipeline.init_masks(ClusterShape{}, cute::block_id_in_cluster());
    accumulatePipeline.init_masks(ClusterShape{});
    if (threadLoc.warpIdx == Schedule::MmaWarp) allocator.allocate(W4TmemColumns, &sharedStorage.tmem);
  } else {
    initDispatchStorage<Types::DispatchStages, Schedule>(sharedStorage, threadLoc.warpIdx, threadLoc.laneIdx);
  }
  cutlass::arch::fence_barrier_init();
  return W4A8Pipelines<Types>{loadPipeline, activationPipeline, accumulatePipeline, allocator};
}

template <bool FixedTokenCount, class KernelParams>
MSCCLPP_DEVICE_INLINE void runW4A8RoutingStage(const KernelParams& params, int nTokens, const int32_t* ids,
                                               const float* scores) {
  traceKernel(KernelTracePhase::Routing, true);
  if (params.config.computeOnly) {
    prepareLocalRoutes(params, nTokens, ids);
  } else {
    // Local routes read source inputs directly; remote routes consume current-epoch packets.
    prepareRoutes<FixedTokenCount>(params, nTokens, ids, scores);
  }
  traceKernel(KernelTracePhase::Routing, false);
}

template <ActivationDtype ActDtype, class Types, class Storage>
MSCCLPP_DEVICE_INLINE void initW4A8StaticScaleStage(Storage& sharedStorage) {
  if constexpr (ActDtype == ActivationDtype::Fp8Static) {
    constexpr int SfbWordsPerStage = sizeof(sharedStorage.tensors.smem_SFB) / (Types::Load::Stages * sizeof(uint32_t));
    auto* sfbStage0 = reinterpret_cast<uint32_t*>(sharedStorage.tensors.smem_SFB.begin());
    for (int i = threadIdx.x; i < SfbWordsPerStage; i += W4Threads) sfbStage0[i] = 0x7f7f7f7fU;
    if constexpr (Types::Tiles::ClusterSize > 1) {
      cute::cluster_sync();
    } else {
      __syncthreads();
    }
    cutlass::arch::fence_view_async_shared();
  }
}

template <class KernelParams>
struct W4A8ComputeContext {
  typename KernelParams::Collective::Mainloop fc1;
  typename KernelParams::Collective::Mainloop fc2;
  ProblemShape shape1;
  ProblemShape shape2;
  int nTokenTiles;
  int nTasks;
  int nLocalExperts;
};

template <class KernelParams>
MSCCLPP_DEVICE_INLINE W4A8ComputeContext<KernelParams> makeW4A8ComputeContext(const KernelParams& params, int ctaIdx) {
  using Types = typename KernelParams::Collective;
  using Mainloop = typename Types::Mainloop;
  using ClusterShape = typename Types::ClusterShape;
  const int hidden = params.config.hidden;
  const int intermediate = params.config.intermediate;
  const int nTokenTiles = params.workspace.control->nTokenTiles;
  const int nTasks = nTokenTiles * (fc1TaskTiles<KernelParams>(intermediate) + fc2TaskTiles<KernelParams>(hidden));
  const int nLocalExperts = params.config.numExperts / params.config.worldSize;
  const int storageRows = w4StorageRow<typename KernelParams::Tiles>(params.workspace.poolRows);
  return W4A8ComputeContext<KernelParams>{Mainloop(params.fc1, ClusterShape{}, ctaIdx),
                                          Mainloop(params.fc2, ClusterShape{}, ctaIdx),
                                          ProblemShape{2 * intermediate, storageRows, hidden, nLocalExperts},
                                          ProblemShape{hidden, storageRows, intermediate, nLocalExperts},
                                          nTokenTiles,
                                          nTasks,
                                          nLocalExperts};
}

template <class KernelParams, class Storage, class Pipelines>
MSCCLPP_DEVICE_INLINE void runW4A8LoadStage(const KernelParams& params, Storage& sharedStorage, Pipelines& pipelines,
                                            const KernelThreadLocation& threadLoc) {
  auto computeCtx = makeW4A8ComputeContext(params, threadLoc.ctaIdx);
  runW4A8LoadRole(params, sharedStorage, pipelines.load, pipelines.activation, computeCtx.fc1, computeCtx.fc2,
                  computeCtx.shape1, computeCtx.shape2, params.config.hidden, params.config.intermediate,
                  threadLoc.warpIdx, threadLoc.laneIdx, threadLoc.ctaIdx, threadLoc.clusterIdx, computeCtx.nTasks,
                  computeCtx.nTokenTiles, computeCtx.nLocalExperts);
}

template <class KernelParams, class Storage, class Pipelines>
MSCCLPP_DEVICE_INLINE void runW4A8MmaStage(const KernelParams& params, Storage& sharedStorage, Pipelines& pipelines,
                                           const KernelThreadLocation& threadLoc) {
  using Mainloop = typename KernelParams::Collective::Mainloop;
  using EpilogueTile = cute::Shape<cute::Int<KernelParams::Tiles::CtaM>, cute::Int<KernelParams::Tiles::N>>;
  auto computeCtx = makeW4A8ComputeContext(params, threadLoc.ctaIdx);
  auto tmemStorage = Mainloop::template init_tmem_tensors<EpilogueTile, false>(EpilogueTile{});
  Mainloop::set_tmem_offsets(tmemStorage, sharedStorage.tmem);
  runMmaRole(params, sharedStorage, pipelines.load, pipelines.activation, pipelines.accumulate, computeCtx.fc1,
             tmemStorage, params.config.hidden, params.config.intermediate, threadLoc.warpIdx, threadLoc.ctaIdx,
             threadLoc.clusterIdx, computeCtx.nTasks, computeCtx.nTokenTiles, computeCtx.nLocalExperts);
}

template <class Schedule, class KernelParams, class Storage, class Pipelines>
MSCCLPP_DEVICE_INLINE void runW4A8EpilogueStage(const KernelParams& params, Storage& sharedStorage,
                                                Pipelines& pipelines, const KernelThreadLocation& threadLoc,
                                                int nTokens, __bfloat16* output) {
  using Mainloop = typename KernelParams::Collective::Mainloop;
  using EpilogueTile = cute::Shape<cute::Int<KernelParams::Tiles::CtaM>, cute::Int<KernelParams::Tiles::N>>;
  auto computeCtx = makeW4A8ComputeContext(params, threadLoc.ctaIdx);
  auto tmemStorage = Mainloop::template init_tmem_tensors<EpilogueTile, false>(EpilogueTile{});
  Mainloop::set_tmem_offsets(tmemStorage, sharedStorage.tmem);
  runEpilogueRole<Schedule>(params, sharedStorage, pipelines.accumulate, tmemStorage.accumulators, params.config.hidden,
                            params.config.intermediate, threadLoc.clusterIdx, computeCtx.nTasks, output, nTokens);
}

template <class Types, class RoleFn>
MSCCLPP_DEVICE_INLINE void dispatchW4A8WarpRole(int warpIdx, RoleFn&& runRole) {
  using Schedule = typename Types::WarpSchedule;
  if (warpIdx < Schedule::DispatchBegin) {
    // Reconfigure the complete transfer warpgroup before its load/MMA roles diverge.
    if (warpIdx >= Schedule::EpilogueEnd) cutlass::arch::warpgroup_reg_dealloc<Types::TransferRegisters>();
    const bool isLoadWarp =
        warpIdx == Schedule::LoadWarp || (Types::LoadWarps == 2 && warpIdx == Schedule::ActivationLoadWarp);
    if (isLoadWarp) {
      runRole.template operator()<WarpRole::Load>();
    } else if (warpIdx >= Schedule::EpilogueBegin && warpIdx < Schedule::EpilogueEnd) {
      runRole.template operator()<WarpRole::Epilogue>();
    } else if (warpIdx == Schedule::MmaWarp) {
      runRole.template operator()<WarpRole::Mma>();
    }
  } else {
    // Donate only the registers needed to fund epilogue growth; retain the remaining CTA budget on dispatch/offload
    // warps.
    constexpr int DispatchThreads = W4Threads - Types::EpilogueWarps * 32 - 128;
    constexpr int AvailDispatchRegs =
        ((W4Threads * EntryRegisters - Types::EpilogueWarps * 32 * Types::EpilogueRegisters -
          128 * Types::TransferRegisters) /
         DispatchThreads) &
        ~7;
    constexpr int DispatchRoleRegisters =
        AvailDispatchRegs < TransferRegisters ? TransferRegisters : (AvailDispatchRegs > 80 ? 80 : AvailDispatchRegs);
    if constexpr (DispatchRoleRegisters < EntryRegisters) cutlass::arch::warpgroup_reg_dealloc<DispatchRoleRegisters>();
    if (warpIdx < Schedule::DispatchEnd) {
      runRole.template operator()<WarpRole::Dispatch>();
    } else {
      runRole.template operator()<WarpRole::OutputOffload>();
    }
  }
}

template <int Hidden, class Types, class KernelParams, class Storage, class Allocator>
MSCCLPP_DEVICE_INLINE void finishW4A8KernelStage(const KernelParams& params, Storage& sharedStorage,
                                                 Allocator& allocator, int warpIdx, int nTokens, __bfloat16* output) {
  using Schedule = typename Types::WarpSchedule;
  // Join even idle roles before the next register reconfiguration.
  __syncthreads();
  if (warpIdx >= Schedule::EpilogueBegin && warpIdx < Schedule::EpilogueEnd)
    cutlass::arch::warpgroup_reg_dealloc<EntryRegisters>();
  else
    cutlass::arch::warpgroup_reg_alloc<EntryRegisters>();
  cute::cluster_sync();
  if (warpIdx == Schedule::MmaWarp) {
    allocator.release_allocation_lock();
    allocator.free(sharedStorage.tmem, W4TmemColumns);
  }
  finishW4A8<Hidden>(params, nTokens, output);
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_W4A8_ROLES_CUH_
