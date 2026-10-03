// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "megamoe_roles.cuh"

#if !defined(MSCCLPP_MEGAMOE_JIT_MODULE) || !MSCCLPP_MEGAMOE_JIT_MODULE
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
  params.consumer_arv_count = 256;
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

template <int Hidden, int Intermediate = 0, int WorldSize = 0>
__global__ __launch_bounds__(W4Threads, 1) void megaMoeW4A8(__grid_constant__ const W4A8Parameters parameters,
                                                            int tokens, __bfloat16* output, uint32_t* startSignal,
                                                            const int32_t* ids, const float* scores) {
  using namespace cute;
  using Types = W4A8CollectiveTypes;
  using Mainloop = Types::Mainloop;
  using Load = Types::Load;
  using Accumulate = Types::Accumulate;
  using Schedule = W4A8WarpSchedule;
  NativeConfig configuration = parameters.config;
  if constexpr (WorldSize != 0) {
    static_assert(WorldSize == 4 || WorldSize == 32);
    configuration.worldSize = WorldSize;
    configuration.maxTokens = 64;
    configuration.hidden = Hidden;
    configuration.intermediate = Intermediate;
    configuration.numExperts = 16 * WorldSize;
    configuration.topK = 8;
    configuration.gateUpClamp = -1.0f;
  }
  auto p = w4a8ParameterView(parameters, configuration);
  extern __shared__ __align__(1024) char storage[];
  auto& s = *reinterpret_cast<W4A8SharedStorage*>(storage);
  int warp = threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  int cta = blockIdx.x % ClusterM;
  int cluster = blockIdx.x / ClusterM;
  const int hidden = p.config.hidden;
  const int intermediate = p.config.intermediate;
  if (blockIdx.x == 0 && threadIdx.x == 0 && startSignal)
    atomicStore<uint32_t, scopeDevice>(startSignal, 1, memoryOrderRelease);
  traceW4(W4TracePhase::Routing, true);
  // Routing publishes a local ready epoch before GEMM reuses the tensor staging buffers.
  prepareRoutes<WorldSize != 0>(p, tokens, ids, scores);
  traceW4(W4TracePhase::Routing, false);

  typename Load::Params loadParams{};
  bool loadWarp =
      warp == Schedule::LoadWarp || (!W4SplitPipelines && W4LoadWarps == 2 && warp == Schedule::ActivationLoadWarp);
  loadParams.role =
      loadWarp ? Load::ThreadCategory::Producer
               : (warp == Schedule::MmaWarp ? Load::ThreadCategory::Consumer : Load::ThreadCategory::NonParticipant);
  loadParams.is_leader = lane == 0 && cta == 0 && loadWarp;
  loadParams.transaction_bytes = W4LoadWarps == 1 ? Mainloop::TmaTransactionBytes
                                                  : (warp == Schedule::LoadWarp ? Mainloop::WeightTransactionBytes
                                                                                : Mainloop::ActivationTransactionBytes);
  if constexpr (W4SplitPipelines) loadParams.transaction_bytes = Mainloop::WeightTransactionBytes;
  loadParams.initializing_warp = Schedule::LoadWarp;
  Load loadPipeline(s.mainloop, loadParams, ClusterShape{}, false_type{}, false_type{});
  if constexpr (W4LoadWarps == 2 && !W4SplitPipelines) {
    if (warp == Schedule::LoadWarp) {
      // Both producers arrive with their own byte count; the stage becomes
      // ready only after both arrivals and all four TMA copies have completed.
      cutlass::arch::detail::initialize_barrier_array_pair_aligned<decltype(s.mainloop.full_barrier_),
                                                                   decltype(s.mainloop.empty_barrier_), W4LoadStages>(
          s.mainloop.full_barrier_, s.mainloop.empty_barrier_, 2, 1);
    }
  } else {
    Load::init_barriers(s.mainloop, loadParams, ClusterShape{});
  }
  auto activationPipeline = [&](auto& activationStorage) {
    if constexpr (W4SplitPipelines) {
      auto activationParams = loadParams;
      activationParams.role =
          warp == Schedule::ActivationLoadWarp
              ? Load::ThreadCategory::Producer
              : (warp == Schedule::MmaWarp ? Load::ThreadCategory::Consumer : Load::ThreadCategory::NonParticipant);
      activationParams.is_leader = lane == 0 && cta == 0 && warp == Schedule::ActivationLoadWarp;
      activationParams.transaction_bytes = Mainloop::ActivationTransactionBytes;
      activationParams.initializing_warp = Schedule::ActivationLoadWarp;
      return Load(activationStorage, activationParams, ClusterShape{}, true_type{}, false_type{});
    } else {
      return loadPipeline;
    }
  }(s.activationLoad);
  auto accumulatePipeline = makeAccumulatorPipeline<Accumulate, Schedule>(s.accumulated, warp);
  cutlass::arch::fence_barrier_init();
  cute::cluster_sync();
  loadPipeline.init_masks(ClusterShape{}, cute::block_id_in_cluster());
  if constexpr (W4SplitPipelines) activationPipeline.init_masks(ClusterShape{}, cute::block_id_in_cluster());
  accumulatePipeline.init_masks(ClusterShape{});
  cute::TMEM::Allocator2Sm allocator;
  if (warp == Schedule::MmaWarp) allocator.allocate(512, &s.tmem);
  __syncthreads();
  cute::cluster_sync();

  Mainloop fc1(p.fc1, ClusterShape{}, cta);
  Mainloop fc2(p.fc2, ClusterShape{}, cta);
  using EpilogueTile = Shape<Int<TileM / ClusterM>, Int<W4TileN>>;
  auto tmemStorage = Mainloop::template init_tmem_tensors<EpilogueTile, false>(EpilogueTile{});
  Mainloop::set_tmem_offsets(tmemStorage, s.tmem);
  int tokenBlocks = p.workspace.control->tokenBlocks;
  int tasks = tokenBlocks * (fc1TaskTiles<false>(intermediate) + fc2TaskTiles<false>(hidden));
  int localExperts = p.config.numExperts / p.config.worldSize;
  ProblemShape shape1{2 * intermediate, w4StorageRow(p.workspace.poolRows), hidden, localExperts};
  ProblemShape shape2{hidden, w4StorageRow(p.workspace.poolRows), intermediate, localExperts};

  if (warp >= Schedule::EpilogueBegin && warp < Schedule::EpilogueEnd) {
    runEpilogueRole<0, Schedule>(p, tokens, output, s, accumulatePipeline, tmemStorage.accumulators, hidden,
                                 intermediate, cluster, tasks);
  } else if (warp < Schedule::DispatchBegin) {
    // Keep mainloop and dispatch reconfiguration in disjoint branches so ptxas
    // does not constrain the four-descriptor loader to dispatch's 32 registers.
    runW4A8MainloopRole(p, s, loadPipeline, activationPipeline, accumulatePipeline, fc1, fc2, shape1, shape2,
                        tmemStorage, hidden, intermediate, warp, lane, cta, cluster, tasks);
  } else {
    cutlass::arch::warpgroup_reg_dealloc<TransferRegisters>();
    if (warp < Schedule::DispatchEnd) {
      traceW4(W4TracePhase::Dispatch, true);
      dispatchW4A8Tokens<Hidden>(p, s, warp - Schedule::DispatchBegin);
      traceW4(W4TracePhase::Dispatch, false);
    }
  }
  __syncthreads();
  if (!(warp >= Schedule::EpilogueBegin && warp < Schedule::EpilogueEnd))
    cutlass::arch::warpgroup_reg_alloc<EntryRegisters>();
  cute::cluster_sync();
  if (warp == Schedule::MmaWarp) {
    allocator.release_allocation_lock();
    allocator.free(s.tmem, 512);
  }
  traceW4(W4TracePhase::OutputJoin, true);
  if (threadIdx.x == 0) {
    // The final arrival acquires every CTA's completed writes before the system release.
    s.epilogue.routing.outputPublisher =
        atomicFetchAdd<int, scopeDevice>(&p.workspace.control->completedCtas, 1, memoryOrderAcqRel) == gridDim.x - 1;
  }
  __syncthreads();
  if (s.epilogue.routing.outputPublisher && threadIdx.x == 0)
    *at<uint64_t>(p.local, p.symmetric.epoch) = p.workspace.control->epoch;
  if (s.epilogue.routing.outputPublisher && threadIdx.x < p.config.worldSize) {
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

W4A8KernelEntry w4a8KernelEntry(const NativeConfig& config) {
  if (useSpecializedW4A8Kernel(config)) {
    if (config.intermediate == 4096)
      return config.worldSize == 4 ? megaMoeW4A8<9216, 4096, 4> : megaMoeW4A8<9216, 4096, 32>;
    if (config.intermediate == 4608)
      return config.worldSize == 4 ? megaMoeW4A8<9216, 4608, 4> : megaMoeW4A8<9216, 4608, 32>;
  }
  switch (config.hidden) {
    case 128:
      return megaMoeW4A8<128>;
    case 384:
      return megaMoeW4A8<384>;
    case 2176:
      return megaMoeW4A8<2176>;
    case 4096:
      return megaMoeW4A8<4096>;
    case 8704:
      return megaMoeW4A8<8704>;
    case 9216:
      return megaMoeW4A8<9216>;
    default:
      throw std::invalid_argument("MegaMoE W4A8 hidden size has no compiled specialization");
  }
}
#endif

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail
