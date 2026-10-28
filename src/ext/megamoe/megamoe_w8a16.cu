// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "megamoe_routing.cuh"
#include "megamoe_w8a16_roles.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

#if (!defined(MSCCLPP_MEGAMOE_JIT_MODULE) || !MSCCLPP_MEGAMOE_JIT_MODULE) && MSCCLPP_MEGAMOE_TRACE
extern "C" int mscclpp_megamoe_w8_trace_reset() { return int(resetKernelTrace()); }

extern "C" int mscclpp_megamoe_w8_trace_copy(void* events, size_t bytes, uint32_t* counts, size_t countBytes) {
  return int(copyKernelTrace(events, bytes, counts, countBytes));
}
#endif

template <bool Local, class Parameters>
__device__ __forceinline__ int prepareW8A16TokenBlocks(const Parameters& p, int tokens, const int32_t* ids,
                                                       const float* scores) {
  if constexpr (Local) {
    return (tokens + Parameters::Tiles::N - 1) / Parameters::Tiles::N;
  } else {
    traceKernel(KernelTracePhase::Routing, true);
    prepareRoutes<false>(p, tokens, ids, scores);
    traceKernel(KernelTracePhase::Routing, false);
    return p.workspace.control->tokenBlocks;
  }
}

template <bool Local, class Parameters>
__device__ __forceinline__ void finalizeW8A16Output(const Parameters& p, int tokens, __bfloat16* output) {
  if constexpr (!Local) finishAndCombine<0>(p, tokens, output);
}

template <int LocalMode = 0,
          class Parameters = std::conditional_t<LocalMode != 0, W8A16LocalParameters, W8A16Parameters>>
__global__ __launch_bounds__((LocalMode ? LocalThreads : W4Threads),
                             1) void megaMoeW8A16(__grid_constant__ const Parameters p, int tokens, __bfloat16* output,
                                                  uint32_t* kernelEntrySignal, const int32_t* ids,
                                                  const float* scores) {
  using namespace cute;
  constexpr bool Local = LocalMode != 0;
  using Types = W8A16CollectiveTypesT<Local>;
  using Mainloop = typename Types::Mainloop;
  using LoadA = typename Types::LoadA;
  using LoadB = typename Types::LoadB;
  using Transform = typename Types::Transform;
  using Accumulate = typename Types::Accumulate;
  using ClusterShape = typename Types::ClusterShape;
  using Schedule = typename Types::WarpSchedule;
  using SharedStorage = typename Parameters::SharedStorage;
  extern __shared__ __align__(1024) char storage[];
  constexpr int ClusterSize = Parameters::Tiles::ClusterSize;
  constexpr int RestoreRegisters = Local ? LocalEntryRegisters : EntryRegisters;
  auto& s = *reinterpret_cast<SharedStorage*>(storage);
  int warp = threadIdx.x / warpSize;
  int lane = threadIdx.x % warpSize;
  int cta = blockIdx.x % ClusterSize;
  int cluster = blockIdx.x / ClusterSize;
  const int hidden = p.config.hidden;
  const int intermediate = p.config.intermediate;
  if (blockIdx.x == 0 && threadIdx.x == 0 && kernelEntrySignal)
    atomicStore<uint32_t, scopeDevice>(kernelEntrySignal, 1, memoryOrderRelease);
  if (Local && tokens == 0) return;
  int tokenBlocks = prepareW8A16TokenBlocks<Local>(p, tokens, ids, scores);

  typename LoadA::Params aParams{};
  aParams.role = warp == Schedule::LoadAWarp ? LoadA::ThreadCategory::Producer
                                             : (warp >= Schedule::TransformBegin && warp < Schedule::TransformEnd
                                                    ? LoadA::ThreadCategory::Consumer
                                                    : LoadA::ThreadCategory::NonParticipant);
  aParams.is_leader = lane == 0;
  aParams.num_consumers = 128;
  aParams.transaction_bytes = Mainloop::TmaTransactionBytes_A;
  aParams.initializing_warp = Schedule::LoadAWarp;
  LoadA aPipeline(s.mainloopA, aParams, ClusterShape{}, cutlass::McastDirection::kRow, true_type{}, false_type{});
  typename LoadB::Params bParams{};
  bParams.role = warp == Schedule::LoadBWarp ? LoadB::ThreadCategory::Producer
                                             : (warp == Schedule::MmaWarp ? LoadB::ThreadCategory::Consumer
                                                                          : LoadB::ThreadCategory::NonParticipant);
  bParams.is_leader = lane == 0 && cta == 0 && warp == Schedule::LoadBWarp;
  bParams.num_consumers = 32;
  bParams.transaction_bytes = Mainloop::TmaTransactionBytes_B;
  bParams.initializing_warp = Schedule::LoadBWarp;
  LoadB bPipeline(s.mainloopB, bParams, ClusterShape{}, cutlass::McastDirection::kCol, true_type{}, false_type{});
  typename Transform::Params tParams{};
  tParams.role = warp >= Schedule::TransformBegin && warp < Schedule::TransformEnd
                     ? Transform::ThreadCategory::Producer
                     : (warp == Schedule::MmaWarp ? Transform::ThreadCategory::Consumer
                                                  : Transform::ThreadCategory::NonParticipant);
  tParams.consumer_arv_count = 1;
  tParams.producer_arv_count = 256;
  tParams.initializing_warp = Schedule::TransformBegin;
  Transform tPipeline(s.transformed, tParams, ClusterShape{}, true_type{}, false_type{});
  auto cPipeline = makeAccumulatorPipeline<Accumulate, Schedule, ClusterShape>(s.accumulated, warp);
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
  int tasks = tokenBlocks * (fc1TaskTiles<Parameters>(intermediate) + fc2TaskTiles<Parameters>(hidden));
  int localExperts = p.config.numExperts / p.config.worldSize;
  ProblemShape shape1{2 * intermediate, p.workspace.poolRows, hidden, localExperts};
  ProblemShape shape2{hidden, p.workspace.poolRows, intermediate, localExperts};

  if (warp >= Schedule::EpilogueBegin && warp < Schedule::EpilogueEnd) {
    runEpilogueRole<Schedule, LocalMode>(p, s, cPipeline, acc, hidden, intermediate, cluster, tasks, output, tokens);
  } else if (warp >= Schedule::TransformBegin && warp < Schedule::TransformEnd) {
    runW8A16TransformRole<Local>(p, tokens, s, aPipeline, tPipeline, fc1, shape1, acc, hidden, intermediate, cluster,
                                 tasks);
  } else {
    runW8A16TransferRole<Local, Schedule>(p, tokens, s, aPipeline, bPipeline, tPipeline, cPipeline, fc1, fc2, shape1,
                                          shape2, acc, hidden, intermediate, warp, lane, cta, cluster, tasks);
  }
  __syncthreads();
  if ((warp >= Schedule::EpilogueBegin && warp < Schedule::EpilogueEnd) ||
      (warp >= Schedule::TransformBegin && warp < Schedule::TransformEnd))
    cutlass::arch::warpgroup_reg_dealloc<RestoreRegisters>();
  else
    cutlass::arch::warpgroup_reg_alloc<RestoreRegisters>();
  cute::cluster_sync();
  if (warp == Schedule::MmaWarp) {
    allocator.release_allocation_lock();
    allocator.free(s.tmem, 512);
  }
  finalizeW8A16Output<Local>(p, tokens, output);
}

template <int LocalMode>
W8A16KernelEntry<LocalMode> w8a16KernelEntry() {
  return megaMoeW8A16<LocalMode>;
}

template W8A16KernelEntry<0> w8a16KernelEntry<0>();
template W8A16KernelEntry<1> w8a16KernelEntry<1>();
template W8A16KernelEntry<2> w8a16KernelEntry<2>();

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail
