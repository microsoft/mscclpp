// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <array>
#include <stdexcept>

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

template <int Hidden, int Intermediate, bool CachedRoutes = false, bool FixedTokenCount = false>
__global__ __launch_bounds__(W4Threads, 1) void megaMoeW4A8(__grid_constant__ const W4A8Parameters parameters,
                                                            int tokens, __bfloat16* output, uint32_t* kernelEntrySignal,
                                                            const int32_t* ids, const float* scores) {
  using namespace cute;
  using Parameters = W4A8Parameters;
  using Types = typename Parameters::Collective;
  using Mainloop = Types::Mainloop;
  using Accumulate = Types::Accumulate;
  using Schedule = typename Types::WarpSchedule;
  static_assert(validW4RegisterBudget(Types::EpilogueWarps, Types::EpilogueRegisters, Types::TransferRegisters));
  NativeConfig configuration = parameters.config;
  configuration.hidden = Hidden;
  configuration.intermediate = Intermediate;
  auto p = w4a8ParameterView(parameters, configuration);
  extern __shared__ __align__(1024) char storage[];
  auto& s = *reinterpret_cast<W4A8SharedStorageT<Types>*>(storage);
  int warp = threadIdx.x / warpSize;
  int lane = threadIdx.x % warpSize;
  int cta = blockIdx.x % ClusterM;
  int cluster = blockIdx.x / ClusterM;
  const int hidden = p.config.hidden;
  const int intermediate = p.config.intermediate;
  // Notify a consumer stream of kernel entry, not output readiness.
  if (blockIdx.x == 0 && threadIdx.x == 0 && kernelEntrySignal)
    atomicStore<uint32_t, scopeDevice>(kernelEntrySignal, 1, memoryOrderRelease);
  traceKernel(KernelTracePhase::Routing, true);
  // Routing publishes a local ready epoch before GEMM reuses the tensor staging buffers.
  prepareRoutes<CachedRoutes, FixedTokenCount>(p, tokens, ids, scores);
  traceKernel(KernelTracePhase::Routing, false);
  cute::TMEM::Allocator2Sm allocator;

  if (warp < Schedule::DispatchBegin) {
    if (threadIdx.x == 0) s.tmemReady.init(ClusterM);
    auto loadPipeline = makeW4A8LoadPipeline<false, Types>(s.mainloop, warp, lane, cta);
    auto activationPipeline = [&](auto& activationStorage) {
      if constexpr (Types::SplitPipelines) {
        return makeW4A8LoadPipeline<true, Types>(activationStorage, warp, lane, cta);
      } else {
        return loadPipeline;
      }
    }(s.activationLoad);
    auto accumulatePipeline = makeAccumulatorPipeline<Accumulate, Schedule>(s.accumulated, warp);
    cutlass::arch::fence_barrier_init();
    loadPipeline.init_masks(ClusterShape{}, cute::block_id_in_cluster());
    if constexpr (Types::SplitPipelines) activationPipeline.init_masks(ClusterShape{}, cute::block_id_in_cluster());
    accumulatePipeline.init_masks(ClusterShape{});
    cutlass::arch::NamedBarrier::sync(Schedule::DispatchBegin * warpSize, 0);
    cute::cluster_arrive();
    cute::cluster_wait();

    Mainloop fc1(p.fc1, ClusterShape{}, cta);
    Mainloop fc2(p.fc2, ClusterShape{}, cta);
    int tokenBlocks = p.workspace.control->tokenBlocks;
    int tasks = tokenBlocks * (fc1TaskTiles<false>(intermediate) + fc2TaskTiles<false>(hidden));
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
      if (warp == Schedule::MmaWarp) allocator.allocate(W4TmemColumns, &s.tmem);
      cutlass::arch::NamedBarrier::sync((Schedule::DispatchBegin - Types::LoadWarps) * warpSize, 2);
      if (threadIdx.x == 0) {
        CUTE_UNROLL
        for (int peerCta = 0; peerCta < ClusterM; ++peerCta) s.tmemReady.arrive(peerCta);
      }
      s.tmemReady.wait(0);
      using EpilogueTile = Shape<Int<TileM / ClusterM>, Int<Parameters::Tiles::N>>;
      auto tmemStorage = Mainloop::template init_tmem_tensors<EpilogueTile, false>(EpilogueTile{});
      Mainloop::set_tmem_offsets(tmemStorage, s.tmem);
      if (warp >= Schedule::EpilogueBegin && warp < Schedule::EpilogueEnd) {
        runEpilogueRole<0, Schedule>(p, tokens, output, s, accumulatePipeline, tmemStorage.accumulators, hidden,
                                     intermediate, cluster, tasks);
      } else {
        runW4A8MmaRole(p, s, loadPipeline, activationPipeline, accumulatePipeline, fc1, tmemStorage, hidden,
                       intermediate, warp, cta, cluster, tasks);
      }
    }
  } else {
    // Lower dispatch/idle warp groups to 32 registers per thread to fund epilogue growth.
    cutlass::arch::warpgroup_reg_dealloc<TransferRegisters>();
    // Arrive without waiting so copies overlap the compute-side initialization.
    cute::cluster_arrive_relaxed();
    if (warp < Schedule::DispatchEnd) {
      traceKernel(KernelTracePhase::Dispatch, true);
      dispatchW4A8Tokens<Hidden, Types::DispatchChunk, Types::DispatchStages>(p, s, warp - Schedule::DispatchBegin);
      traceKernel(KernelTracePhase::Dispatch, false);
    }
  }
  // Join even idle roles before the next register reconfiguration.
  __syncthreads();
  if (warp >= Schedule::EpilogueBegin && warp < Schedule::EpilogueEnd)
    cutlass::arch::warpgroup_reg_dealloc<EntryRegisters>();
  else
    cutlass::arch::warpgroup_reg_alloc<EntryRegisters>();
  cute::cluster_sync();
  if (warp == Schedule::MmaWarp) {
    allocator.release_allocation_lock();
    allocator.free(s.tmem, W4TmemColumns);
  }
  traceKernel(KernelTracePhase::OutputJoin, true);
  p.workspace.control->gridBarrier.sync(gridDim.x, SpinLimit);
  if (blockIdx.x == 0 && threadIdx.x == 0) *at<uint64_t>(p.local, p.symmetric.epoch) = p.workspace.control->epoch;
  if (threadIdx.x < p.config.worldSize) {
    // Only CTA0 consumes the peer notification; followers observe the same epoch.
    if (blockIdx.x == 0)
      signalAndWait(p, threadIdx.x);
    else
      waitAtLeast<uint64_t, scopeSystem>(at<uint64_t>(p.local, p.symmetric.peerSignals) + threadIdx.x,
                                         p.workspace.control->epoch);
  }
  __syncthreads();
  traceKernel(KernelTracePhase::OutputJoin, false);
  traceKernel(KernelTracePhase::Combine, true);
  combineResults(p, tokens, output);
  traceKernel(KernelTracePhase::Combine, false);
}

struct W4A8KernelRegistration {
  bool (*matches)(const NativeConfig&);
  W4A8KernelEntry entry;
  W4A8KernelEntry captureEntry;
};

template <int Hidden, int Intermediate>
constexpr W4A8KernelRegistration registerW4A8Kernel() {
  static_assert(Hidden >= 4096);
  return {[](const NativeConfig& config) { return config.hidden == Hidden && config.intermediate == Intermediate; },
          megaMoeW4A8<Hidden, Intermediate>, megaMoeW4A8<Hidden, Intermediate, true, W4FixedTokenCount>};
}

W4A8KernelEntry w4a8KernelEntry(const NativeConfig& config, bool capturing) {
  static constexpr std::array kernels{registerW4A8Kernel<4096, 6656>(), registerW4A8Kernel<8192, 4096>(),
                                      registerW4A8Kernel<9216, 4096>(), registerW4A8Kernel<9216, 4608>()};
  for (const auto& kernel : kernels) {
    if (kernel.matches(config)) return capturing ? kernel.captureEntry : kernel.entry;
  }
  throw std::invalid_argument("MegaMoE W4A8 H/I has no compiled specialization");
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail
