// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "megamoe_roles.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

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
  int cta = blockIdx.x % 2;
  int cluster = blockIdx.x / 2;
  const int hidden = p.config.hidden;
  const int intermediate = p.config.intermediate;
  if (blockIdx.x == 0 && threadIdx.x == 0 && startSignal)
    atomicStore<uint32_t, scopeDevice>(startSignal, 1, memoryOrderRelease);
  if constexpr (Local) {
    if (tokens == 0) return;
  } else {
    prepareRoutes(p, tokens, s.epilogue.routing);
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
  typename Accumulate::Params cParams{};
  cParams.role = warp == Schedule::MmaWarp ? Accumulate::ThreadCategory::Producer
                                           : (warp >= Schedule::EpilogueBegin && warp < Schedule::EpilogueEnd
                                                  ? Accumulate::ThreadCategory::Consumer
                                                  : Accumulate::ThreadCategory::NonParticipant);
  cParams.producer_arv_count = 1;
  cParams.consumer_arv_count = 256;
  cParams.initializing_warp = Schedule::EpilogueBegin;
  Accumulate cPipeline(s.accumulated, cParams, ClusterShape{}, true_type{}, false_type{});
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
  int tasks = tokenBlocks * (intermediate / 128 + (hidden + 255) / 256);
  int localExperts = p.config.numExperts / p.config.worldSize;
  ProblemShape shape1{2 * intermediate, p.workspace.poolRows, hidden, localExperts};
  ProblemShape shape2{hidden, p.workspace.poolRows, intermediate, localExperts};

  // Keep register reconfiguration inside each disjoint role branch. Merging
  // before dispatch makes ptxas constrain the compute roles to the smaller budget.
  if (warp >= Schedule::EpilogueBegin && warp < Schedule::EpilogueEnd) {
    runEpilogueRole<E5M2, LocalMode>(p, tokens, output, s, cPipeline, acc, hidden, intermediate, cluster, tasks);
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

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail
