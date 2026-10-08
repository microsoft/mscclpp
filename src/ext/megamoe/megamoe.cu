// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <array>

#include "megamoe_roles.cuh"

#if MSCCLPP_MEGAMOE_COMPILE_W4A8
#include <stdexcept>

#include "megamoe_w4a8_roles.cuh"
#endif

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

template <class Pipeline, class Schedule>
__device__ __forceinline__ Pipeline makeAccumulatorPipeline(typename Pipeline::SharedStorage& storage, int warp) {
  typename Pipeline::Params params{};
  params.role = warp == Schedule::MmaWarp ? Pipeline::ThreadCategory::Producer
                                          : (warp >= Schedule::EpilogueBegin && warp < Schedule::EpilogueEnd
                                                 ? Pipeline::ThreadCategory::Consumer
                                                 : Pipeline::ThreadCategory::NonParticipant);
  params.producer_arv_count = 1;
  params.consumer_arv_count = ClusterM * (Schedule::EpilogueEnd - Schedule::EpilogueBegin) * 32;
  params.initializing_warp = Schedule::EpilogueBegin;
  return Pipeline(storage, params, ClusterShape{}, cute::true_type{}, cute::false_type{});
}

template <bool E5M2, int LocalMode = 0>
__global__ __launch_bounds__((LocalMode ? LocalThreads : Threads),
                             1) void megaMoe(__grid_constant__ const Parameters<E5M2, (LocalMode != 0)> p, int tokens,
                                             __bfloat16* output, uint32_t* startSignal) {
  using namespace cute;
  using Types = CollectiveTypes<E5M2, (LocalMode != 0)>;
  using Mainloop = typename Types::Mainloop;
  using LoadA = typename Types::LoadA;
  using LoadB = typename Types::LoadB;
  using Transform = typename Types::Transform;
  using Accumulate = typename Types::Accumulate;
  using Schedule = WarpSchedule<(LocalMode != 0)>;
  extern __shared__ __align__(1024) char storage[];
  constexpr bool Local = LocalMode != 0;
  constexpr int KernelTileN = Local ? LocalTileN : TileN;
  constexpr int RestoreRegisters = Local ? LocalEntryRegisters : EntryRegisters;
  auto& s = *reinterpret_cast<SharedStorage<E5M2, Local>*>(storage);
  int warp = threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  int cta = blockIdx.x % ClusterM;
  int cluster = blockIdx.x / ClusterM;
  const int hidden = p.config.hidden;
  const int intermediate = p.config.intermediate;
  if (blockIdx.x == 0 && threadIdx.x == 0 && startSignal)
    atomicStore<uint32_t, scopeDevice>(startSignal, 1, memoryOrderRelease);
  if constexpr (Local) {
    if (tokens == 0) return;
  } else {
    prepareRoutes(p, tokens);
  }

  typename LoadA::Params aParams{};
  aParams.role = warp == Schedule::LoadAWarp ? LoadA::ThreadCategory::Producer
                                             : (warp >= Schedule::TransformBegin && warp < Schedule::TransformEnd
                                                    ? LoadA::ThreadCategory::Consumer
                                                    : LoadA::ThreadCategory::NonParticipant);
  aParams.is_leader = lane == 0;
  aParams.num_consumers = 128;
  aParams.transaction_bytes = Mainloop::TmaTransactionBytes_A;
  aParams.initializing_warp = Schedule::LoadAWarp;
  LoadA aPipeline(s.loadA, aParams, ClusterShape{}, cutlass::McastDirection::kRow, true_type{}, false_type{});
  typename LoadB::Params bParams{};
  bParams.role = warp == Schedule::LoadBWarp ? LoadB::ThreadCategory::Producer
                                             : (warp == Schedule::MmaWarp ? LoadB::ThreadCategory::Consumer
                                                                          : LoadB::ThreadCategory::NonParticipant);
  bParams.is_leader = lane == 0 && cta == 0 && warp == Schedule::LoadBWarp;
  bParams.num_consumers = 32;
  bParams.transaction_bytes = Mainloop::TmaTransactionBytes_B;
  bParams.initializing_warp = Schedule::LoadBWarp;
  LoadB bPipeline(s.loadB, bParams, ClusterShape{}, cutlass::McastDirection::kCol, true_type{}, false_type{});
  typename Transform::Params tParams{};
  tParams.role = warp >= Schedule::TransformBegin && warp < Schedule::TransformEnd
                     ? Transform::ThreadCategory::Producer
                     : (warp == Schedule::MmaWarp ? Transform::ThreadCategory::Consumer
                                                  : Transform::ThreadCategory::NonParticipant);
  tParams.consumer_arv_count = 1;
  tParams.producer_arv_count = 256;
  tParams.initializing_warp = Schedule::TransformBegin;
  Transform tPipeline(s.transformed, tParams, ClusterShape{}, true_type{}, false_type{});
  auto cPipeline = makeAccumulatorPipeline<Accumulate, Schedule>(s.accumulated, warp);
  cutlass::arch::fence_barrier_init();
  cute::cluster_sync();
  aPipeline.init_masks(ClusterShape{}, cute::block_id_in_cluster(), cutlass::McastDirection::kRow);
  bPipeline.init_masks(ClusterShape{}, cutlass::McastDirection::kCol);
  tPipeline.init_masks(ClusterShape{});
  cPipeline.init_masks(ClusterShape{});
  cute::TMEM::Allocator2Sm allocator;
  if (warp == Schedule::MmaWarp) allocator.allocate(512, &s.tmem);
  __syncthreads();
  cute::cluster_sync();

  Mainloop fc1(p.fc1, ClusterShape{}, cta);
  Mainloop fc2(p.fc2, ClusterShape{}, cta);
  auto acc = Mainloop::TiledMma::make_fragment_C(append(fc1.partition_accumulator_shape(), _2{}));
  acc.data() = s.tmem;
  int tokenBlocks = Local ? (tokens + KernelTileN - 1) / KernelTileN : p.workspace.control->tokenBlocks;
  int tasks = tokenBlocks * (fc1TaskTiles<Local>(intermediate) + fc2TaskTiles<Local>(hidden));
  int localExperts = p.config.numExperts / p.config.worldSize;
  ProblemShape shape1{2 * intermediate, p.workspace.poolRows, hidden, localExperts};
  ProblemShape shape2{hidden, p.workspace.poolRows, intermediate, localExperts};

  // Keep register reconfiguration inside each disjoint role branch. Merging
  // before dispatch makes ptxas constrain the compute roles to the smaller budget.
  if (warp >= Schedule::EpilogueBegin && warp < Schedule::EpilogueEnd) {
    runEpilogueRole<LocalMode, Schedule>(p, tokens, output, s, cPipeline, acc, hidden, intermediate, cluster, tasks);
  } else if (warp >= Schedule::TransformBegin && warp < Schedule::TransformEnd) {
    runTransformRole<E5M2, Local>(p, tokens, s, aPipeline, tPipeline, fc1, shape1, acc, hidden, intermediate, cluster,
                                  tasks);
  } else {
    runTransferRole<E5M2, Local, Schedule>(p, tokens, s, aPipeline, bPipeline, tPipeline, cPipeline, fc1, fc2, shape1,
                                           shape2, acc, hidden, intermediate, warp, lane, cta, cluster, tasks);
  }
  // An idle transfer warp must not reclaim registers before every compute warp
  // has acquired its budget and finished; that can deadlock the initial allocation.
  __syncthreads();
  if (!(warp >= Schedule::EpilogueBegin && warp < Schedule::EpilogueEnd) &&
      !(warp >= Schedule::TransformBegin && warp < Schedule::TransformEnd))
    cutlass::arch::warpgroup_reg_alloc<RestoreRegisters>();
  cute::cluster_sync();
  if (warp == Schedule::MmaWarp) {
    allocator.release_allocation_lock();
    allocator.free(s.tmem, 512);
  }
  if constexpr (!Local) {
    p.workspace.control->gridBarrier.sync(gridDim.x, SpinLimit);
    // Bulk stores have landed; CTA/grid joins transfer their completion to these publishers.
    if (blockIdx.x == 0 && threadIdx.x < p.config.worldSize) {
      if (threadIdx.x == 0) *at<uint64_t>(p.local, p.symmetric.epoch) = p.workspace.control->epoch;
      signalAndWait(p, threadIdx.x);
    }
    p.workspace.control->gridBarrier.sync(gridDim.x, SpinLimit);
    combineResults(p, tokens, output);
  }
}

template <bool E5M2, int LocalMode>
KernelEntry<E5M2, LocalMode> kernelEntry() {
  return megaMoe<E5M2, LocalMode>;
}

template KernelEntry<false, 0> kernelEntry<false, 0>();
template KernelEntry<true, 0> kernelEntry<true, 0>();
template KernelEntry<false, 1> kernelEntry<false, 1>();
template KernelEntry<true, 1> kernelEntry<true, 1>();
template KernelEntry<false, 2> kernelEntry<false, 2>();
template KernelEntry<true, 2> kernelEntry<true, 2>();

#if !defined(MSCCLPP_MEGAMOE_JIT_MODULE) || !MSCCLPP_MEGAMOE_JIT_MODULE
#if defined(MSCCLPP_MEGAMOE_W4_TRACE) && MSCCLPP_MEGAMOE_W4_TRACE
extern "C" int mscclpp_megamoe_w4_trace_reset() {
  void* counts = nullptr;
  auto result = cudaGetSymbolAddress(&counts, w4TraceCounts);
  if (result != cudaSuccess) return int(result);
  result = cudaMemset(counts, 0, sizeof(w4TraceCounts));
  if (result != cudaSuccess) return int(result);
  bool enabled = true;
  return int(cudaMemcpyToSymbol(w4TraceEnabled, &enabled, sizeof(enabled)));
}

extern "C" int mscclpp_megamoe_w4_trace_copy(void* events, size_t bytes, uint32_t* counts, size_t countBytes) {
  if (!events || !counts || bytes < sizeof(w4TraceEvents) || countBytes < sizeof(w4TraceCounts))
    return int(cudaErrorInvalidValue);
  bool enabled = false;
  auto result = cudaMemcpyToSymbol(w4TraceEnabled, &enabled, sizeof(enabled));
  if (result != cudaSuccess) return int(result);
  result = cudaMemcpyFromSymbol(counts, w4TraceCounts, sizeof(w4TraceCounts));
  if (result != cudaSuccess) return int(result);
  return int(cudaMemcpyFromSymbol(events, w4TraceEvents, sizeof(w4TraceEvents)));
}
#endif
#endif

#if MSCCLPP_MEGAMOE_COMPILE_W4A8
template <int Hidden, int Intermediate = 0, int WorldSize = 0>
__global__ __launch_bounds__(W4Threads, 1) void megaMoeW4A8(__grid_constant__ const W4A8Parameters parameters,
                                                            int tokens, __bfloat16* output, uint32_t* startSignal,
                                                            const int32_t* ids, const float* scores) {
  using namespace cute;
  using Parameters = W4A8Parameters;
  using Types = typename Parameters::Collective;
  using Mainloop = Types::Mainloop;
  using Accumulate = Types::Accumulate;
  using Schedule = typename Types::WarpSchedule;
  static_assert(validW4RegisterBudget(Types::EpilogueWarps, Types::EpilogueRegisters, Types::TransferRegisters));
  NativeConfig configuration = parameters.config;
  if constexpr (WorldSize != 0) {
    static_assert(WorldSize == 4 || WorldSize == 32);
    configuration.worldSize = WorldSize;
    configuration.hidden = Hidden;
    configuration.intermediate = Intermediate;
    configuration.numExperts = 16 * WorldSize;
    configuration.topK = 8;
    configuration.gateUpClamp = -1.0f;
  }
  auto p = w4a8ParameterView(parameters, configuration);
  extern __shared__ __align__(1024) char storage[];
  auto& s = *reinterpret_cast<W4A8SharedStorageT<Types>*>(storage);
  int warp = threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  int cta = blockIdx.x % ClusterM;
  int cluster = blockIdx.x / ClusterM;
  const int hidden = p.config.hidden;
  const int intermediate = p.config.intermediate;
  // signalStart notifies a consumer stream of kernel entry, not output readiness.
  if (blockIdx.x == 0 && threadIdx.x == 0 && startSignal)
    atomicStore<uint32_t, scopeDevice>(startSignal, 1, memoryOrderRelease);
  traceW4(W4TracePhase::Routing, true);
  // Routing publishes a local ready epoch before GEMM reuses the tensor staging buffers.
  prepareRoutes<WorldSize != 0>(p, tokens, ids, scores);
  traceW4(W4TracePhase::Routing, false);
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
    cutlass::arch::NamedBarrier::sync(Schedule::DispatchBegin * 32, 0);
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
      cutlass::arch::NamedBarrier::sync((Schedule::DispatchBegin - Types::LoadWarps) * 32, 2);
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
    // Keep the loader's register budget independent of dispatch's 32-register role.
    cutlass::arch::warpgroup_reg_dealloc<TransferRegisters>();
    // Arrive without waiting so copies overlap the compute-side initialization.
    cute::cluster_arrive_relaxed();
    if (warp < Schedule::DispatchEnd) {
      traceW4(W4TracePhase::Dispatch, true);
      dispatchW4A8Tokens<Hidden, Types::DispatchChunk, Types::DispatchStages>(p, s, warp - Schedule::DispatchBegin);
      traceW4(W4TracePhase::Dispatch, false);
    }
  }
  // Compute has waited for initialization; this CTA join lets dispatch safely
  // reuse the completed cluster phase without a separate initialization wait.
  __syncthreads();
  if (!(warp >= Schedule::EpilogueBegin && warp < Schedule::EpilogueEnd))
    cutlass::arch::warpgroup_reg_alloc<EntryRegisters>();
  cute::cluster_sync();
  if (warp == Schedule::MmaWarp) {
    allocator.release_allocation_lock();
    allocator.free(s.tmem, W4TmemColumns);
  }
  traceW4(W4TracePhase::OutputJoin, true);
  p.workspace.control->gridBarrier.sync(gridDim.x, SpinLimit);
  if (blockIdx.x == 0 && threadIdx.x == 0) *at<uint64_t>(p.local, p.symmetric.epoch) = p.workspace.control->epoch;
  if (blockIdx.x == 0 && threadIdx.x < p.config.worldSize) {
    MemoryDevice2DeviceSemaphoreDeviceHandle channel{
        at<uint64_t>(p.local, p.symmetric.peerSignals) + threadIdx.x,
        peerAt<uint64_t>(p, threadIdx.x, p.symmetric.peerSignals) + p.config.rank,
        at<uint64_t>(p.local, p.symmetric.expectedPeerSignals) + threadIdx.x};
    channel.incExpectedInbound();
    channel.signal();
  }
  if (threadIdx.x < p.config.worldSize)
    waitAtLeast<uint64_t, scopeSystem>(at<uint64_t>(p.local, p.symmetric.peerSignals) + threadIdx.x,
                                       p.workspace.control->epoch);
  __syncthreads();
  traceW4(W4TracePhase::OutputJoin, false);
  traceW4(W4TracePhase::Combine, true);
  combineResults(p, tokens, output);
  traceW4(W4TracePhase::Combine, false);
}

struct W4A8KernelRegistration {
  int hidden;
  int intermediate;
  int worldSize;
  W4A8KernelEntry entry;
};

template <int Hidden, int Intermediate = 0, int WorldSize = 0>
constexpr W4A8KernelRegistration registerW4A8Kernel() {
  return {Hidden, Intermediate, WorldSize, megaMoeW4A8<Hidden, Intermediate, WorldSize>};
}

W4A8KernelEntry w4a8KernelEntry(const NativeConfig& config) {
  static constexpr std::array kernels{registerW4A8Kernel<128>(),
                                      registerW4A8Kernel<384>(),
                                      registerW4A8Kernel<2176>(),
                                      registerW4A8Kernel<4096>(),
                                      registerW4A8Kernel<8192>(),
                                      registerW4A8Kernel<8704>(),
                                      registerW4A8Kernel<9216>(),
                                      registerW4A8Kernel<8192, 4096, 4>(),
                                      registerW4A8Kernel<8192, 4096, 32>(),
                                      registerW4A8Kernel<9216, 4096, 4>(),
                                      registerW4A8Kernel<9216, 4096, 32>(),
                                      registerW4A8Kernel<9216, 4608, 4>(),
                                      registerW4A8Kernel<9216, 4608, 32>()};
  bool specialized = mscclpp::megamoe::detail::useSpecializedW4A8Kernel(config);
  int intermediate = specialized ? config.intermediate : 0;
  int worldSize = specialized ? config.worldSize : 0;
  for (const auto& kernel : kernels) {
    if (kernel.hidden == config.hidden && kernel.intermediate == intermediate && kernel.worldSize == worldSize)
      return kernel.entry;
  }
  throw std::invalid_argument("MegaMoE W4A8 hidden size has no compiled specialization");
}
#endif

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail
