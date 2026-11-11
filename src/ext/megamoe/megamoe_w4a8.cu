// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <array>
#include <stdexcept>

#include "megamoe_dispatch.cuh"
#include "megamoe_epilogue.cuh"
#include "megamoe_mma.cuh"
#include "megamoe_routing.cuh"
#include "megamoe_tasks.cuh"
#include "megamoe_w4a8_roles.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

#if !defined(MSCCLPP_MEGAMOE_JIT_MODULE) || !MSCCLPP_MEGAMOE_JIT_MODULE
#if MSCCLPP_MEGAMOE_TRACE
extern "C" int mscclpp_megamoe_w4_trace_reset() { return int(resetKernelTrace()); }

extern "C" int mscclpp_megamoe_w4_trace_copy(void* events, size_t bytes, uint32_t* counts, size_t countBytes) {
  return int(copyKernelTrace(events, bytes, counts, countBytes));
}
#endif
#endif

template <class P>
struct FinalOutputParameterView : P {
  __bfloat16* finalOutput;
};

template <class TensorStorage, class Epilogue>
struct W4TaskStorageView {
  TensorStorage& tensors;
  Epilogue& epilogue;
};

template <class Types, class TensorStorage>
__device__ __forceinline__ void initializeUnitSfb(TensorStorage& tensors, int warp, int lane) {
  using Schedule = typename Types::WarpSchedule;
  if (warp != Schedule::MmaWarp) return;
  constexpr int Words = sizeof(tensors.smem_SFB) / (Types::Load::Stages * sizeof(uint32_t));
  auto* stage0 = reinterpret_cast<uint32_t*>(tensors.smem_SFB.begin());
  for (int i = lane; i < Words; i += warpSize) stage0[i] = 0x7f7f7f7fU;
  cutlass::arch::fence_view_async_shared();
}

template <class Schedule, int ClusterSize, class Storage>
__device__ __forceinline__ void syncFusedCompute(Storage& s, int& phase) {
  // Warp roles reach this helper through different control-flow paths, so use an mbarrier instead of divergent
  // bar.sync instructions.
  s.computeReady.arrive();
  s.computeReady.wait(phase);
  if (threadIdx.x == 0) {
#pragma unroll
    for (int peerCta = 0; peerCta < ClusterSize; ++peerCta) s.phaseReady.arrive(peerCta);
  }
  s.phaseReady.wait(phase);
  phase ^= 1;
}

// The CTA's routing warps arrive once the final routing grid barrier has passed, so every CTA's routes are visible.
template <class Storage>
__device__ __forceinline__ void waitForRouting(Storage& s) {
  s.routingReady.wait(0);
}

// Fused shared-expert schedule (see docs in native-megamoe-shared-benchmark-instructions.md, "Fused v2"):
//  1. The MMA/load warpgroup streams shared FC1 into TMEM while every other warp (epilogue + dispatch + idle) routes
//     at the entry register budget. The epilogue warps drain shared FC1 once routing is done.
//  2. Routed FC1/FC2 tasks.
//  3. Each CTA arrives at the resident grid barrier as soon as its routed epilogue stores land, so the peer signal
//     and combine never wait for shared FC2.
//  4. Shared FC2 runs afterwards on clusters without combine work, starting with the clusters whose routed work ends
//     first (one routed task fewer).
template <int Hidden, int Intermediate, bool FixedTokenCount>
__device__ __forceinline__ void runFusedW4A8(const W4A8FusedParameters& parameters, int tokens, __bfloat16* output,
                                             __bfloat16* sharedOutput, uint32_t* kernelEntrySignal, const int32_t* ids,
                                             const float* scores, const __bfloat16* input,
                                             const __bfloat16* sharedInput) {
  using namespace cute;
  using Parameters = W4A8Parameters;
  using Types = typename Parameters::Collective;
  using SharedParameters = W8A8LocalParameters;
  using SharedTypes = typename SharedParameters::Collective;
  using Mainloop = Types::Mainloop;
  using SharedMainloop = SharedTypes::Mainloop;
  using Accumulate = Types::Accumulate;
  using Schedule = Types::WarpSchedule;
  using ClusterShape = Types::ClusterShape;
  constexpr int ClusterSize = Parameters::Tiles::ClusterSize;
  static_assert(std::is_same_v<Types::Load, SharedTypes::Load>);
  static_assert(std::is_same_v<Types::Accumulate, SharedTypes::Accumulate>);
  static_assert(std::is_same_v<Types::WarpSchedule, SharedTypes::WarpSchedule>);
  NativeConfig configuration = parameters.routed.config;
  configuration.hidden = Hidden;
  configuration.intermediate = Intermediate;
  auto routedView = w4a8ParameterView<ActivationDtype::Fp8Static>(parameters.routed, configuration);
  auto p = FinalOutputParameterView<decltype(routedView)>{routedView, output};
  const auto& shared = parameters.shared;
  extern __shared__ __align__(1024) char storage[];
  auto& s = *reinterpret_cast<W4A8FusedSharedStorage*>(storage);
  // The routed roles use the routed layout; the shared expert reuses its tensor buffers and epilogue staging.
  auto& r = s.routed;
  auto sharedStorage = W4TaskStorageView<W4A8FusedSharedStorage::SharedTensorStorage, decltype(r.epilogue)>{
      s.sharedTensors(), r.epilogue};
  int warp = threadIdx.x / warpSize;
  int lane = threadIdx.x % warpSize;
  int cta = blockIdx.x % ClusterSize;
  int cluster = blockIdx.x / ClusterSize;
  const int clusters = gridDim.x / ClusterSize;
  const int hidden = p.config.hidden;
  const int intermediate = p.config.intermediate;
  const int sharedHidden = shared.config.hidden;
  const int sharedIntermediate = shared.config.intermediate;
  const int sharedTokenBlocks = taskTokenBlocks(shared, tokens);
  const int sharedFc1Tasks = sharedTokenBlocks * fc1TaskTiles<SharedParameters>(sharedIntermediate);
  const int sharedFc2Tasks = sharedTokenBlocks * fc2TaskTiles<SharedParameters>(sharedHidden);
  // Shared FC1 starts past cluster 0, whose CTA 0 publishes this rank's routes on the routing critical path, and past
  // the combine clusters. Routed tasks follow it, so the shared FC1 clusters get the last (shorter) routed round.
  constexpr int SharedFc1FirstCluster = 2;
  const int sharedFc1Begin = clusters > SharedFc1FirstCluster + sharedFc1Tasks ? SharedFc1FirstCluster : 0;
#if defined(MSCCLPP_MEGAMOE_FUSED_DEBUG_SKIP) && MSCCLPP_MEGAMOE_FUSED_DEBUG_SKIP
  // Debug attribution flags (bits 24+): 2 skips shared FC2, 3 skips both shared GEMMs, 4 (with 3) starts the routed
  // ordinals at 0, 8 spreads the combine over the whole grid.
  const uint32_t debugSkip = parameters.sharedFc1DelayNs >> 24;
  const uint32_t sharedFc1DelayNs = parameters.sharedFc1DelayNs & 0xffffffu;
  const bool skipFc1 = (debugSkip & 3u) == 3u, skipFc2 = debugSkip & 2u;
  const bool routedFromZero = skipFc1 && (debugSkip & 4u), spreadCombine = debugSkip & 8u;
#else
  const uint32_t sharedFc1DelayNs = parameters.sharedFc1DelayNs;
  constexpr bool skipFc1 = false, skipFc2 = false, routedFromZero = false, spreadCombine = false;
#endif
  // Skipped phases keep their ordinals, so the routed placement is unchanged.
  const TaskRange sharedFc1Range{sharedFc1Begin, sharedFc1Begin + (skipFc1 ? 0 : sharedFc1Tasks), 0};
  // Shared FC2 runs on clusters without combine work, beginning after the cluster that holds the last routed task.
  const int combineClusters = (combineCtaCount<Hidden>(p, tokens) + ClusterSize - 1) / ClusterSize;
  const int fc2ClusterCount = clusters - combineClusters;
  auto sharedFc2RangeAfter = [&](int routedEnd) {
    TaskRange range{routedEnd, routedEnd + (skipFc2 ? 0 : sharedFc2Tasks), sharedFc1Tasks};
    if (fc2ClusterCount > 0) {
      int start = routedEnd % clusters;
      range.clusterBegin = combineClusters;
      range.clusterCount = fc2ClusterCount;
      range.clusterOffset = start >= combineClusters ? start - combineClusters : 0;
    }
    return range;
  };
  const bool sharedFc2Cluster = fc2ClusterCount <= 0 || cluster >= combineClusters;
  uint32_t gridPhase = 0;
  if (blockIdx.x == 0 && threadIdx.x == 0 && kernelEntrySignal)
    atomicStore<uint32_t, scopeDevice>(kernelEntrySignal, 1, memoryOrderRelease);
  conditional_t<ClusterSize == 1, TMEM::Allocator1Sm, TMEM::Allocator2Sm> allocator;
  auto route = [&]() {
    traceKernel(KernelTracePhase::Routing, true);
    prepareRoutesDispatch<FixedTokenCount>(p, s.routingOverlay, tokens, ids, scores, input);
    traceKernel(KernelTracePhase::Routing, false);
  };

  if (warp < Schedule::DispatchBegin) {
    // The spare warp of the MMA/load warpgroup initializes the fused barriers, so the epilogue warps can start routing
    // at kernel entry, like the dispatch warps.
    constexpr int InitWarp = Schedule::DispatchBegin - 1;
    static_assert(InitWarp != Schedule::MmaWarp && InitWarp != Schedule::LoadWarp &&
                  InitWarp != Schedule::ActivationLoadWarp && InitWarp >= Schedule::EpilogueEnd);
    const bool epilogueWarp = warp >= Schedule::EpilogueBegin && warp < Schedule::EpilogueEnd;
    auto* sharedQuantizeState = &p.workspace.control->sharedQuantizeSyncState;
    uint32_t sharedQuantizePhase = 0;
    if (warp == InitWarp && lane == 0) {
      s.tmemReady.init(ClusterSize);
      s.computeReady.init(Schedule::DispatchBegin * warpSize);
      s.phaseReady.init(ClusterSize);
      s.routingReady.init(1);
    }
    auto loadPipeline = makeW4A8LoadPipeline<false, Types>(r.mainloop, warp, lane, cta);
    auto activationPipeline = [&](auto& activationStorage) {
      if constexpr (Types::SplitPipelines)
        return makeW4A8LoadPipeline<true, Types>(activationStorage, warp, lane, cta);
      else
        return loadPipeline;
    }(r.activationLoad);
    auto accumulatePipeline =
        makeAccumulatorPipeline<Accumulate, Schedule, ClusterShape>(r.accumulated, warp, InitWarp);
    loadPipeline.init_masks(ClusterShape{}, cute::block_id_in_cluster());
    if constexpr (Types::SplitPipelines) activationPipeline.init_masks(ClusterShape{}, cute::block_id_in_cluster());
    accumulatePipeline.init_masks(ClusterShape{});
    if (epilogueWarp) {
      // The epilogue warps initialize nothing: count them in the init barriers without waiting, route, and only then
      // wait for the cluster barrier, which publishes every barrier initialization.
      cutlass::arch::NamedBarrier::arrive(Schedule::DispatchBegin * warpSize, 0);
      cute::cluster_arrive_relaxed();
      route();
      cute::cluster_wait();
      if (threadIdx.x == 0) s.routingReady.arrive();
    } else {
      // Read the shared quantization phase before this CTA's init warp can arrive (after the barrier below), so the
      // flip cannot be missed.
      if (sharedInput && lane == 0 && (warp == Schedule::LoadWarp || warp == Schedule::ActivationLoadWarp))
        sharedQuantizePhase = atomicLoad<uint32_t, scopeDevice>(sharedQuantizeState, memoryOrderRelaxed) & (1u << 31);
      cutlass::arch::fence_barrier_init();
      cutlass::arch::NamedBarrier::sync(Schedule::DispatchBegin * warpSize, 0);
      cute::cluster_arrive();
      cute::cluster_wait();
    }

    Mainloop routedFc1(p.fc1, ClusterShape{}, cta);
    Mainloop routedFc2(p.fc2, ClusterShape{}, cta);
    SharedMainloop sharedFc1(shared.fc1, ClusterShape{}, cta);
    SharedMainloop sharedFc2(shared.fc2, ClusterShape{}, cta);
    const int routedRows = w4StorageRow<typename Parameters::Tiles>(p.workspace.poolRows);
    const int sharedRows = w4StorageRow<typename SharedParameters::Tiles>(shared.workspace.poolRows);
    ProblemShape routedShape1{2 * intermediate, routedRows, hidden, p.config.numExperts / p.config.worldSize};
    ProblemShape routedShape2{hidden, routedRows, intermediate, p.config.numExperts / p.config.worldSize};
    ProblemShape sharedShape1{2 * sharedIntermediate, sharedRows, sharedHidden, 1};
    ProblemShape sharedShape2{sharedHidden, sharedRows, sharedIntermediate, 1};

    if (warp >= Schedule::EpilogueEnd) cutlass::arch::warpgroup_reg_dealloc<Types::TransferRegisters>();
    bool producerWarp = warp == Schedule::LoadWarp || (Types::LoadWarps == 2 && warp == Schedule::ActivationLoadWarp);
    // In-kernel static-FP8 shared input (MSCCLPP_MEGAMOE_W4_FUSE_STATIC_QUANTIZE; the host passes sharedInput only
    // then): the otherwise idle init warp of every CTA quantizes a grid-strided slice while the shared FC1 loads wait
    // out sharedFc1DelayNs, and CTA 0 also clears the shared FC1 readiness counters. Each init warp then arrives once
    // at a sense-reversing counter, and the shared FC1 activation producer waits for the phase flip before its TMA
    // loads.
    if (sharedInput && warp == InitWarp) {
      constexpr int ValuesPerThread = 8;
      if (blockIdx.x == 0) {
        const int readyInts = (tokens + W4TileN - 1) / W4TileN * W4ReadyCounterStride;
        for (int i = lane; i < readyInts; i += warpSize) shared.workspace.hiddenReady[i] = 0;
      }
      const int vectorsPerToken = sharedHidden / ValuesPerThread;
      const float scale = shared.config.activationScale;
      const float invScale = scale != 0.0f ? 1.0f / scale : 0.0f;
      // One vector per lane at a time: measured gentler on the concurrent routing exchange than batched loads.
#pragma unroll 1
      for (int vector = int(blockIdx.x) * warpSize + lane; vector < tokens * vectorsPerToken;
           vector += int(gridDim.x) * warpSize) {
        const size_t offset =
            size_t(vector / vectorsPerToken) * sharedHidden + vector % vectorsPerToken * ValuesPerThread;
        int4 packed = *reinterpret_cast<const int4*>(sharedInput + offset);
        mscclpp::f32x8 decoded = mscclpp::to<mscclpp::f32x8>(mscclpp::bit_cast<mscclpp::bf16x8>(packed));
        *reinterpret_cast<uint2*>(shared.workspace.quantizedInput + offset) = quantizeE4M3(decoded, invScale).storage;
      }
      asm volatile("fence.proxy.async.global;" ::: "memory");
      __syncwarp();
      if (lane == 0) (void)arriveResidentGrid(sharedQuantizeState, gridDim.x);
    }
    auto routedRangeNow = [&]() {
      const int routedTasks = p.workspace.control->tokenBlocks *
                              (fc1TaskTiles<Parameters>(intermediate) + fc2TaskTiles<Parameters>(hidden));
      const int routedBegin = routedFromZero ? 0 : sharedFc1Begin + sharedFc1Tasks;
      return TaskRange{routedBegin, routedBegin + routedTasks, 0};
    };
    if (producerWarp) {
      using Load = Types::Load;
      auto loadState = cutlass::make_producer_start_state<Load>();
      auto& producerPipeline =
          Types::SplitPipelines && warp == Schedule::ActivationLoadWarp ? activationPipeline : loadPipeline;
      int phase = 0;
      if (sharedFc1DelayNs) {
        // Hold the load warps back briefly (all clusters measured best): the shared FC1 weight stream otherwise slows
        // the latency-bound routing exchange.
        uint64_t start, now;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
        do asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
        while (now - start < sharedFc1DelayNs);
      }
      // Only the activation producer reads the quantized input; the weight producer streams FC1 weights meanwhile.
      if (sharedInput && (Types::LoadWarps == 1 || warp == Schedule::ActivationLoadWarp)) {
        if (lane == 0) waitResidentGrid(sharedQuantizeState, gridDim.x, sharedQuantizePhase);
        __syncwarp();
      }
      runW4A8LoadTasks(shared, sharedStorage, loadPipeline, activationPipeline, sharedFc1, sharedFc2, sharedShape1,
                       sharedShape2, sharedHidden, sharedIntermediate, warp, lane, cta, cluster, sharedFc1Range,
                       loadState, tokens);
      sharedFc1.load_tail(producerPipeline, loadState);
      syncFusedCompute<Schedule, ClusterSize>(s, phase);
      waitForRouting(s);
      const TaskRange routedRange = routedRangeNow();
      runW4A8LoadTasks(p, r, loadPipeline, activationPipeline, routedFc1, routedFc2, routedShape1, routedShape2, hidden,
                       intermediate, warp, lane, cta, cluster, routedRange, loadState);
      routedFc1.load_tail(producerPipeline, loadState);
      if (sharedFc2Cluster) {
        syncFusedCompute<Schedule, ClusterSize>(s, phase);
        runW4A8LoadTasks(shared, sharedStorage, loadPipeline, activationPipeline, sharedFc1, sharedFc2, sharedShape1,
                         sharedShape2, sharedHidden, sharedIntermediate, warp, lane, cta, cluster,
                         sharedFc2RangeAfter(routedRange.end), loadState, tokens);
        sharedFc1.load_tail(producerPipeline, loadState);
      }
    } else {
      if (warp == Schedule::MmaWarp) {
        // The MMA warp publishes the allocation itself, so the routing epilogue warps need not join a named barrier.
        allocator.allocate(W4TmemColumns, &r.tmem);
        __syncwarp();
        if (lane == 0) {
#pragma unroll
          for (int peerCta = 0; peerCta < ClusterSize; ++peerCta) s.tmemReady.arrive(peerCta);
        }
      }
      s.tmemReady.wait(0);
      using EpilogueTile = Shape<Int<Parameters::Tiles::CtaM>, Int<Parameters::Tiles::N>>;
      auto routedTmem = Mainloop::template init_tmem_tensors<EpilogueTile, false>(EpilogueTile{});
      auto sharedTmem = SharedMainloop::template init_tmem_tensors<EpilogueTile, false>(EpilogueTile{});
      Mainloop::set_tmem_offsets(routedTmem, r.tmem);
      SharedMainloop::set_tmem_offsets(sharedTmem, r.tmem);
      int phase = 0;
      if (epilogueWarp) {
        cutlass::arch::warpgroup_reg_alloc<Types::EpilogueRegisters>();
        typename Accumulate::PipelineState accumulateState;
        if constexpr (W4A8SharedStorage::OverlapEpilogueAndDispatch) {
          // The epilogue staging aliases the dispatch buffers until this CTA's dispatch finishes.
          if (lane == 0)
            waitAtLeast<uint32_t, cuda::thread_scope_block>(&r.dispatch.dispatchDone, uint32_t(Types::DispatchWarps));
          __syncwarp();
        }
        runEpilogueTasks<Schedule, 2>(shared, sharedStorage, accumulatePipeline, sharedTmem.accumulators, sharedHidden,
                                      sharedIntermediate, cluster, sharedFc1Range, accumulateState, sharedOutput,
                                      tokens);
#if MSCCLPP_BULK_AVAILABLE
        bulkStoreWait();
#endif
        syncFusedCompute<Schedule, ClusterSize>(s, phase);
        const TaskRange routedRange = routedRangeNow();
        runEpilogueTasks<Schedule>(p, r, accumulatePipeline, routedTmem.accumulators, hidden, intermediate, cluster,
                                   routedRange, accumulateState, output);
#if MSCCLPP_BULK_AVAILABLE
        bulkStoreWait();
#endif
        // This CTA's routed outputs have landed: arrive now so the peer signal never waits for shared FC2.
        cutlass::arch::NamedBarrier::sync(Types::EpilogueWarps * warpSize,
                                          cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        if (threadIdx.x == 0) gridPhase = arriveResidentGrid(&p.workspace.control->residentGridSyncState, gridDim.x);
        if (sharedFc2Cluster) {
          syncFusedCompute<Schedule, ClusterSize>(s, phase);
          runEpilogueTasks<Schedule, 2>(shared, sharedStorage, accumulatePipeline, sharedTmem.accumulators,
                                        sharedHidden, sharedIntermediate, cluster, sharedFc2RangeAfter(routedRange.end),
                                        accumulateState, sharedOutput, tokens);
#if MSCCLPP_BULK_AVAILABLE
          bulkStoreWait();
#endif
        }
      } else if (warp == Schedule::MmaWarp) {
        using Load = Types::Load;
        typename Load::PipelineState loadState;
        auto accumulateState = cutlass::make_producer_start_state<Accumulate>();
        initializeUnitSfb<SharedTypes>(s.sharedTensors(), warp, lane);
        runMmaTasks(shared, sharedStorage, loadPipeline, activationPipeline, accumulatePipeline, sharedFc1, sharedTmem,
                    sharedHidden, sharedIntermediate, warp, cta, cluster, sharedFc1Range, loadState, accumulateState,
                    tokens);
        if (cta == 0) accumulatePipeline.producer_tail(accumulateState);
        syncFusedCompute<Schedule, ClusterSize>(s, phase);
        waitForRouting(s);
        const TaskRange routedRange = routedRangeNow();
        initializeUnitSfb<Types>(r.tensors, warp, lane);
        runMmaTasks(p, r, loadPipeline, activationPipeline, accumulatePipeline, routedFc1, routedTmem, hidden,
                    intermediate, warp, cta, cluster, routedRange, loadState, accumulateState);
        if (cta == 0) accumulatePipeline.producer_tail(accumulateState);
        if (sharedFc2Cluster) {
          syncFusedCompute<Schedule, ClusterSize>(s, phase);
          initializeUnitSfb<SharedTypes>(s.sharedTensors(), warp, lane);
          runMmaTasks(shared, sharedStorage, loadPipeline, activationPipeline, accumulatePipeline, sharedFc1,
                      sharedTmem, sharedHidden, sharedIntermediate, warp, cta, cluster,
                      sharedFc2RangeAfter(routedRange.end), loadState, accumulateState, tokens);
          if (cta == 0) accumulatePipeline.producer_tail(accumulateState);
        }
      } else {
        syncFusedCompute<Schedule, ClusterSize>(s, phase);
        if (sharedFc2Cluster) syncFusedCompute<Schedule, ClusterSize>(s, phase);
      }
    }
  } else {
    initDispatchStorage<Types::DispatchStages, Schedule>(r, warp, lane);
    cutlass::arch::fence_barrier_init();
    cute::cluster_arrive_relaxed();
    // Route at the entry register budget, then shrink to the transfer budget.
    route();
    cutlass::arch::warpgroup_reg_dealloc<TransferRegisters>();
    runDispatchRole<Hidden, Types::DispatchChunk, Types::DispatchStages>(p, r, warp);
  }
  __syncthreads();
  if (warp >= Schedule::EpilogueBegin && warp < Schedule::EpilogueEnd)
    cutlass::arch::warpgroup_reg_dealloc<EntryRegisters>();
  else
    cutlass::arch::warpgroup_reg_alloc<EntryRegisters>();
  cute::cluster_sync();
  if (warp == Schedule::MmaWarp) {
    allocator.release_allocation_lock();
    allocator.free(r.tmem, W4TmemColumns);
  }
  finishAndCombineArrived<Hidden>(p, tokens, output, gridPhase, spreadCombine ? int(gridDim.x) : 0);
}

template <int Hidden, int Intermediate, bool FixedTokenCount = false>
__global__ __launch_bounds__(W4Threads,
                             1) void megaMoeW4A8FusedShared(__grid_constant__ const W4A8FusedParameters parameters,
                                                            int tokens, __bfloat16* output, uint32_t* kernelEntrySignal,
                                                            const int32_t* ids, const float* scores,
                                                            __bfloat16* sharedOutput, const __bfloat16* input,
                                                            const __bfloat16* sharedInput) {
  runFusedW4A8<Hidden, Intermediate, FixedTokenCount>(parameters, tokens, output, sharedOutput, kernelEntrySignal, ids,
                                                      scores, input, sharedInput);
}

template <int Hidden, int Intermediate, bool FixedTokenCount = false, ActivationDtype ActDtype = ActivationDtype::Mxfp8>
__global__ __launch_bounds__(W4Threads, 1) void megaMoeW4A8(__grid_constant__ const W4A8Parameters parameters,
                                                            int tokens, __bfloat16* output, uint32_t* kernelEntrySignal,
                                                            const int32_t* ids, const float* scores,
                                                            const __bfloat16* input = nullptr) {
  using namespace cute;
  using Parameters = W4A8Parameters;
  using Types = typename Parameters::Collective;
  using Mainloop = Types::Mainloop;
  using Accumulate = Types::Accumulate;
  using Schedule = typename Types::WarpSchedule;
  using ClusterShape = typename Types::ClusterShape;
  constexpr int ClusterSize = Parameters::Tiles::ClusterSize;
  static_assert(validW4RegisterBudget(Types::EpilogueWarps, Types::EpilogueRegisters, Types::TransferRegisters));
  NativeConfig configuration = parameters.config;
  configuration.hidden = Hidden;
  configuration.intermediate = Intermediate;
  auto view = w4a8ParameterView<ActDtype>(parameters, configuration);
  auto p = FinalOutputParameterView<decltype(view)>{view, output};
  extern __shared__ __align__(1024) char storage[];
  auto& s = *reinterpret_cast<W4A8SharedStorageT<Types>*>(storage);
  int warp = threadIdx.x / warpSize;
  int lane = threadIdx.x % warpSize;
  int cta = blockIdx.x % ClusterSize;
  int cluster = blockIdx.x / ClusterSize;
  const int hidden = p.config.hidden;
  const int intermediate = p.config.intermediate;
  // Notify a consumer stream of kernel entry, not output readiness.
  if (blockIdx.x == 0 && threadIdx.x == 0 && kernelEntrySignal)
    atomicStore<uint32_t, scopeDevice>(kernelEntrySignal, 1, memoryOrderRelease);
  cute::conditional_t<ClusterSize == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm> allocator;
  auto loadPipeline = makeW4A8LoadPipeline<false, Types>(s.mainloop, warp, lane, cta);
  auto activationPipeline = [&](auto& activationStorage) {
    if constexpr (Types::SplitPipelines) {
      return makeW4A8LoadPipeline<true, Types>(activationStorage, warp, lane, cta);
    } else {
      return loadPipeline;
    }
  }(s.activationLoad);
  auto accumulatePipeline = makeAccumulatorPipeline<Accumulate, Schedule, ClusterShape>(s.accumulated, warp);
  if (warp < Schedule::DispatchBegin) {
    loadPipeline.init_masks(ClusterShape{}, cute::block_id_in_cluster());
    if constexpr (Types::SplitPipelines) activationPipeline.init_masks(ClusterShape{}, cute::block_id_in_cluster());
    accumulatePipeline.init_masks(ClusterShape{});
    if (warp == Schedule::MmaWarp) allocator.allocate(W4TmemColumns, &s.tmem);
  } else {
    initDispatchStorage<Types::DispatchStages, Schedule>(s, warp, lane);
  }
  cutlass::arch::fence_barrier_init();
  traceKernel(KernelTracePhase::Routing, true);
  // Local routes read source inputs directly; remote routes consume current-epoch packets.
  int tokenBlocks = prepareRoutes<FixedTokenCount>(p, tokens, ids, scores, input);
  traceKernel(KernelTracePhase::Routing, false);

  if (warp < Schedule::DispatchBegin) {
    Mainloop fc1(p.fc1, ClusterShape{}, cta);
    Mainloop fc2(p.fc2, ClusterShape{}, cta);
    int tasks = tokenBlocks * (fc1TaskTiles<Parameters>(intermediate) + fc2TaskTiles<Parameters>(hidden));
    int localExperts = p.config.numExperts / p.config.worldSize;
    ProblemShape shape1{2 * intermediate, w4StorageRow<typename Parameters::Tiles>(p.workspace.poolRows), hidden,
                        localExperts};
    ProblemShape shape2{hidden, w4StorageRow<typename Parameters::Tiles>(p.workspace.poolRows), intermediate,
                        localExperts};

    // Reconfigure the complete transfer warpgroup before its load/MMA roles diverge.
    if (warp >= Schedule::EpilogueEnd) cutlass::arch::warpgroup_reg_dealloc<Types::TransferRegisters>();
    bool producerWarp = warp == Schedule::LoadWarp || (Types::LoadWarps == 2 && warp == Schedule::ActivationLoadWarp);
    if (producerWarp) {
      // Producers need initialized pipeline barriers, not the TMEM allocation.
      runW4A8LoadRole(p, s, loadPipeline, activationPipeline, fc1, fc2, shape1, shape2, hidden, intermediate, warp,
                      lane, cta, cluster, tasks);
    } else {
      if constexpr (ActDtype == ActivationDtype::Fp8Static) {
        if (warp == Schedule::MmaWarp) {
          constexpr int SfbWordsPerStage = sizeof(s.tensors.smem_SFB) / (Types::Load::Stages * sizeof(uint32_t));
          auto* sfbStage0 = reinterpret_cast<uint32_t*>(s.tensors.smem_SFB.begin());
          for (int i = lane; i < SfbWordsPerStage; i += warpSize) sfbStage0[i] = 0x7f7f7f7fU;
          cutlass::arch::fence_view_async_shared();
        }
      }
      using EpilogueTile = Shape<Int<Parameters::Tiles::CtaM>, Int<Parameters::Tiles::N>>;
      auto tmemStorage = Mainloop::template init_tmem_tensors<EpilogueTile, false>(EpilogueTile{});
      Mainloop::set_tmem_offsets(tmemStorage, s.tmem);
      if (warp >= Schedule::EpilogueBegin && warp < Schedule::EpilogueEnd) {
        runEpilogueRole<Schedule>(p, s, accumulatePipeline, tmemStorage.accumulators, hidden, intermediate, cluster,
                                  tasks, output);
      } else {
        runMmaRole(p, s, loadPipeline, activationPipeline, accumulatePipeline, fc1, tmemStorage, hidden, intermediate,
                   warp, cta, cluster, tasks);
      }
    }
  } else {
    // Lower dispatch/idle warp groups to 32 registers per thread to fund epilogue growth.
    cutlass::arch::warpgroup_reg_dealloc<TransferRegisters>();
    runDispatchRole<Hidden, Types::DispatchChunk, Types::DispatchStages>(p, s, warp);
  }
  // Join even idle roles before the next register reconfiguration.
  __syncthreads();
  if (warp >= Schedule::EpilogueBegin && warp < Schedule::EpilogueEnd)
    cutlass::arch::warpgroup_reg_dealloc<EntryRegisters>();
  else
    cutlass::arch::warpgroup_reg_alloc<EntryRegisters>();
  traceKernel(KernelTracePhase::OutputJoin, true);
  syncResidentGrid(&p.workspace.control->residentGridSyncState, gridDim.x);
  if (warp == Schedule::MmaWarp) {
    allocator.release_allocation_lock();
    allocator.free(s.tmem, W4TmemColumns);
  }
  finishAndCombine<Hidden, false>(p, tokens, output);
}

struct W4A8KernelRegistration {
  bool (*matches)(const NativeConfig&);
  W4A8KernelEntry entry;
  W4A8KernelEntry captureEntry;
  W4A8KernelEntry staticFp8Entry;
  W4A8KernelEntry staticFp8CaptureEntry;
};

struct W4A8FusedKernelRegistration {
  bool (*matches)(const NativeConfig&);
  W4A8FusedKernelEntry entry;
  W4A8FusedKernelEntry captureEntry;
};

template <int Hidden, int Intermediate>
constexpr W4A8KernelRegistration registerW4A8Kernel() {
  static_assert(Hidden >= 4096);
  return {[](const NativeConfig& config) { return config.hidden == Hidden && config.intermediate == Intermediate; },
          megaMoeW4A8<Hidden, Intermediate, false, ActivationDtype::Mxfp8>,
          megaMoeW4A8<Hidden, Intermediate, W4FixedTokenCount, ActivationDtype::Mxfp8>,
          megaMoeW4A8<Hidden, Intermediate, false, ActivationDtype::Fp8Static>,
          megaMoeW4A8<Hidden, Intermediate, W4FixedTokenCount, ActivationDtype::Fp8Static>};
}

W4A8KernelEntry w4a8KernelEntry(const NativeConfig& config, bool capturing) {
  static constexpr std::array kernels{registerW4A8Kernel<4096, 4352>(), registerW4A8Kernel<4096, 4608>(),
                                      registerW4A8Kernel<4096, 5120>(), registerW4A8Kernel<4096, 6656>(),
                                      registerW4A8Kernel<8192, 4096>(), registerW4A8Kernel<9216, 4096>(),
                                      registerW4A8Kernel<9216, 4608>()};
  const bool staticFp8 = config.activationDtype == ActivationDtype::Fp8Static;
  for (const auto& kernel : kernels) {
    if (kernel.matches(config))
      return staticFp8 ? (capturing ? kernel.staticFp8CaptureEntry : kernel.staticFp8Entry)
                       : (capturing ? kernel.captureEntry : kernel.entry);
  }
  throw std::invalid_argument("MegaMoE W4A8 H/I has no compiled specialization");
}

template <int Hidden, int Intermediate>
constexpr W4A8FusedKernelRegistration registerW4A8FusedKernel() {
  static_assert(Hidden >= 4096);
  return {[](const NativeConfig& config) { return config.hidden == Hidden && config.intermediate == Intermediate; },
          megaMoeW4A8FusedShared<Hidden, Intermediate, false>,
          megaMoeW4A8FusedShared<Hidden, Intermediate, W4FixedTokenCount>};
}

W4A8FusedKernelEntry w4a8FusedKernelEntry(const NativeConfig& config, bool capturing) {
  static constexpr std::array kernels{registerW4A8FusedKernel<4096, 4352>(), registerW4A8FusedKernel<4096, 4608>(),
                                      registerW4A8FusedKernel<4096, 5120>(), registerW4A8FusedKernel<4096, 6656>(),
                                      registerW4A8FusedKernel<8192, 4096>(), registerW4A8FusedKernel<9216, 4096>(),
                                      registerW4A8FusedKernel<9216, 4608>()};
  for (const auto& kernel : kernels)
    if (kernel.matches(config)) return capturing ? kernel.captureEntry : kernel.entry;
  throw std::invalid_argument("MegaMoE fused W4A8 H/I has no compiled specialization");
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail
