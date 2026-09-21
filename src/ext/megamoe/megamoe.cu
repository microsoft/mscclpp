// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "megamoe_epilogue.cuh"
#include "megamoe_mma.cuh"
#include "megamoe_routing.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

template <bool E5M2, bool Local>
__device__ Task taskAt(const Parameters<E5M2, Local>& p, int ordinal, int tokens, int hidden, int intermediate) {
  constexpr int KernelTileN = Local ? LocalTileN : TileN;
  int fc1Tiles = intermediate / 128;
  int fc2Tiles = (hidden + 255) / 256;
  int tokenBlocks = Local ? (tokens + KernelTileN - 1) / KernelTileN : p.workspace.control->tokenBlocks;
  int fc1Tasks = tokenBlocks * fc1Tiles;
  bool fc1 = ordinal < fc1Tasks;
  int local = fc1 ? ordinal : ordinal - fc1Tasks;
  int tiles = fc1 ? fc1Tiles : fc2Tiles;
  int block = local / tiles;
  TokenBlock tokenBlock;
  if constexpr (Local) {
    tokenBlock = TokenBlock{0, min(int(KernelTileN), tokens - block * KernelTileN)};
  } else {
    tokenBlock = p.workspace.blocks[block];
  }
  return Task{fc1, block, local % tiles, tokenBlock};
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
  extern __shared__ __align__(1024) char storage[];
  constexpr bool Local = LocalMode != 0;
  constexpr int KernelTileN = Local ? LocalTileN : TileN;
  constexpr int TransformWarp = Local ? 8 : 12;
  constexpr int RoleRegisters = Local ? LocalComputeRegisters : ComputeRegisters;
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
  aParams.role =
      warp == 5 ? LoadA::ThreadCategory::Producer
                : (warp >= TransformWarp ? LoadA::ThreadCategory::Consumer : LoadA::ThreadCategory::NonParticipant);
  aParams.is_leader = lane == 0;
  aParams.num_consumers = 128;
  aParams.transaction_bytes = Mainloop::TmaTransactionBytes_A;
  aParams.initializing_warp = 5;
  LoadA aPipeline(s.loadA, aParams, ClusterShape{}, cutlass::McastDirection::kRow, true_type{}, false_type{});
  typename LoadB::Params bParams{};
  bParams.role = warp == 6 ? LoadB::ThreadCategory::Producer
                           : (warp == 4 ? LoadB::ThreadCategory::Consumer : LoadB::ThreadCategory::NonParticipant);
  bParams.is_leader = lane == 0 && cta == 0 && warp == 6;
  bParams.num_consumers = 32;
  bParams.transaction_bytes = Mainloop::TmaTransactionBytes_B;
  bParams.initializing_warp = 6;
  LoadB bPipeline(s.loadB, bParams, ClusterShape{}, cutlass::McastDirection::kCol, true_type{}, false_type{});
  typename Transform::Params tParams{};
  tParams.role = warp >= TransformWarp
                     ? Transform::ThreadCategory::Producer
                     : (warp == 4 ? Transform::ThreadCategory::Consumer : Transform::ThreadCategory::NonParticipant);
  tParams.consumer_arv_count = 1;
  tParams.producer_arv_count = 256;
  tParams.initializing_warp = TransformWarp;
  Transform tPipeline(s.transformed, tParams, ClusterShape{}, true_type{}, false_type{});
  typename Accumulate::Params cParams{};
  cParams.role = warp == 4
                     ? Accumulate::ThreadCategory::Producer
                     : (warp < 4 ? Accumulate::ThreadCategory::Consumer : Accumulate::ThreadCategory::NonParticipant);
  cParams.producer_arv_count = 1;
  cParams.consumer_arv_count = 256;
  cParams.initializing_warp = 0;
  Accumulate cPipeline(s.accumulated, cParams, ClusterShape{}, true_type{}, false_type{});
  cutlass::arch::fence_barrier_init();
  cute::cluster_sync();
  aPipeline.init_masks(ClusterShape{}, cute::block_id_in_cluster(), cutlass::McastDirection::kRow);
  bPipeline.init_masks(ClusterShape{}, cutlass::McastDirection::kCol);
  tPipeline.init_masks(ClusterShape{});
  cPipeline.init_masks(ClusterShape{});
  cute::TMEM::Allocator2Sm allocator;
  if (warp == 4) allocator.allocate(512, &s.tmem);
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
  if (warp < 4) {
    cutlass::arch::warpgroup_reg_alloc<RoleRegisters>();
    typename Accumulate::PipelineState state;
    for (int i = cluster; i < tasks; i += gridDim.x / 2)
      epilogue<E5M2, LocalMode>(p, taskAt<E5M2, Local>(p, i, tokens, hidden, intermediate), s, cPipeline, state, acc,
                                output, hidden, intermediate);
#if MSCCLPP_BULK_AVAILABLE
    if constexpr (Local) {
      if (threadIdx.x % 32 == 0) bulkStoreWait();
    }
#endif
    cutlass::arch::warpgroup_reg_dealloc<RestoreRegisters>();
  } else if (warp >= TransformWarp) {
    cutlass::arch::warpgroup_reg_alloc<RoleRegisters>();
    typename LoadA::PipelineState aState;
    auto tState = cutlass::make_producer_start_state<Transform>();
    auto inputs = fc1.transform_init(p.fc1, shape1, acc, s.tensors);
    for (int i = cluster; i < tasks; i += gridDim.x / 2) {
      Task task = taskAt<E5M2, Local>(p, i, tokens, hidden, intermediate);
      transformWeights<E5M2, Local>(aPipeline, aState, tPipeline, tState, inputs,
                                    task.fc1 ? hidden / 128 : intermediate / 128);
    }
    tPipeline.producer_tail(tState);
    cutlass::arch::warpgroup_reg_dealloc<RestoreRegisters>();
  } else {
    cutlass::arch::warpgroup_reg_dealloc<TransferRegisters>();
    if constexpr (!Local) {
      if (warp >= 8) {
        dispatchTokens(p, s);
      }
    }
    if (warp == 5 && lane == 0) {
      auto state = cutlass::make_producer_start_state<LoadA>();
      auto load1 = fc1.load_init(shape1, p.fc1, s.tensors);
      auto load2 = fc2.load_init(shape2, p.fc2, s.tensors);
      for (int i = cluster; i < tasks; i += gridDim.x / 2) {
        Task task = taskAt<E5M2, Local>(p, i, tokens, hidden, intermediate);
        auto coord = make_coord(task.m * 2 + cta, task.block, 0, task.tokens.expert);
        auto iterator = cute::make_coord_iterator(0, task.fc1 ? hidden / 128 : intermediate / 128);
        auto result = task.fc1 ? fc1.load_A(p.fc1, aPipeline, state, load1, coord, iterator, hidden / 128)
                               : fc2.load_A(p.fc2, aPipeline, state, load2, coord, iterator, intermediate / 128);
        state = get<0>(result);
      }
      aPipeline.producer_tail(state);
    } else if (warp == 6 && lane == 0) {
      auto state = cutlass::make_producer_start_state<LoadB>();
      auto load1 = fc1.load_init(shape1, p.fc1, s.tensors);
      auto load2 = fc2.load_init(shape2, p.fc2, s.tensors);
      for (int i = cluster; i < tasks; i += gridDim.x / 2) {
        Task task = taskAt<E5M2, Local>(p, i, tokens, hidden, intermediate);
        if constexpr (Local) {
          if (!task.fc1) waitAtLeast<int, scopeDevice>(p.workspace.hiddenReady + task.block, 2 * (intermediate / 128));
        } else {
          waitAtLeast<int, scopeDevice>((task.fc1 ? p.workspace.inputReady : p.workspace.hiddenReady) + task.block,
                                        task.fc1 ? task.tokens.rows : 2 * (intermediate / 128));
        }
        auto coord = make_coord(task.m * 2 + cta, task.block, 0, task.tokens.expert);
        auto iterator = cute::make_coord_iterator(0, task.fc1 ? hidden / 128 : intermediate / 128);
        auto result = task.fc1 ? fc1.load_B(p.fc1, bPipeline, state, load1, coord, iterator, hidden / 128)
                               : fc2.load_B(p.fc2, bPipeline, state, load2, coord, iterator, intermediate / 128);
        state = get<0>(result);
      }
      bPipeline.producer_tail(state);
    } else if (warp == 4 && cta == 0) {
      typename LoadB::PipelineState bState;
      typename Transform::PipelineState tState;
      auto cState = cutlass::make_producer_start_state<Accumulate>();
      auto inputs = fc1.mma_init(acc, s.tensors);
      for (int i = cluster; i < tasks; i += gridDim.x / 2) {
        Task task = taskAt<E5M2, Local>(p, i, tokens, hidden, intermediate);
        auto result = mmaTiles<E5M2, Local>(bPipeline, bState, tPipeline, tState, cPipeline, cState, acc, inputs,
                                            task.fc1 ? hidden / 128 : intermediate / 128);
        bState = get<0>(result);
        tState = get<1>(result);
        cState = get<2>(result);
      }
      cPipeline.producer_tail(cState);
    }
  }
  // An idle transfer warp must not reclaim registers before every compute warp
  // has acquired its budget and finished; that can deadlock the initial allocation.
  __syncthreads();
  if (warp >= 4 && warp < TransformWarp) cutlass::arch::warpgroup_reg_alloc<RestoreRegisters>();
  cute::cluster_sync();
  if (warp == 4) {
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
