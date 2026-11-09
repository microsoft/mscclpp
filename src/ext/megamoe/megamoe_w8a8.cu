// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Local W8A8 shared expert: FP8 E4M3 weights x static-FP8 activations on the block-scaled W4A8 pipeline.
// It is the W4A8 counterpart of megaMoeW8A16<2>: one unweighted local expert (route weight 1.0) with no
// routing, dispatch, or combine. FC1 reads the input quantized by the launcher, FC2 waits on per-block
// hiddenReady counters, and the FC2 epilogue writes BF16 output rows directly. MXFP8 weights carry E8M0 K32
// block scales; fp8_nv weights are packed with unit block scales and their per-matrix FP32 scales
// (p.fc1WeightScale/p.fc2WeightScale) are applied in the epilogue together with the static activation scale.

#include "megamoe_epilogue.cuh"
#include "megamoe_mma.cuh"
#include "megamoe_tasks.cuh"
#include "megamoe_w4a8_roles.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

__global__ __launch_bounds__(W4Threads, 1) void megaMoeW8A8Local(__grid_constant__ const W8A8LocalParameters p,
                                                                 int tokens, __bfloat16* output) {
  using namespace cute;
  using Parameters = W8A8LocalParameters;
  using Types = typename Parameters::Collective;
  using Mainloop = typename Types::Mainloop;
  using Accumulate = typename Types::Accumulate;
  using Schedule = typename Types::WarpSchedule;
  using ClusterShape = typename Types::ClusterShape;
  constexpr int ClusterSize = Parameters::Tiles::ClusterSize;
  static_assert(Parameters::LocalExpert && Parameters::Activation == ActivationDtype::Fp8Static);
  static_assert(validW4RegisterBudget(Types::EpilogueWarps, Types::EpilogueRegisters, Types::TransferRegisters));
  if (tokens == 0) return;
  extern __shared__ __align__(1024) char storage[];
  auto& s = *reinterpret_cast<typename Parameters::SharedStorage*>(storage);
  int warp = threadIdx.x / warpSize;
  int lane = threadIdx.x % warpSize;
  int cta = blockIdx.x % ClusterSize;
  int cluster = blockIdx.x / ClusterSize;
  const int hidden = p.config.hidden;
  const int intermediate = p.config.intermediate;
  cute::conditional_t<ClusterSize == 1, cute::TMEM::Allocator1Sm, cute::TMEM::Allocator2Sm> allocator;

  if (warp < Schedule::DispatchBegin) {
    if (threadIdx.x == 0) s.tmemReady.init(ClusterSize);
    auto loadPipeline = makeW4A8LoadPipeline<false, Types>(s.mainloop, warp, lane, cta);
    auto activationPipeline = [&](auto& activationStorage) {
      if constexpr (Types::SplitPipelines) {
        return makeW4A8LoadPipeline<true, Types>(activationStorage, warp, lane, cta);
      } else {
        return loadPipeline;
      }
    }(s.activationLoad);
    auto accumulatePipeline = makeAccumulatorPipeline<Accumulate, Schedule, ClusterShape>(s.accumulated, warp);
    cutlass::arch::fence_barrier_init();
    loadPipeline.init_masks(ClusterShape{}, cute::block_id_in_cluster());
    if constexpr (Types::SplitPipelines) activationPipeline.init_masks(ClusterShape{}, cute::block_id_in_cluster());
    accumulatePipeline.init_masks(ClusterShape{});
    cutlass::arch::NamedBarrier::sync(Schedule::DispatchBegin * warpSize, 0);
    cute::cluster_arrive();
    cute::cluster_wait();

    Mainloop fc1(p.fc1, ClusterShape{}, cta);
    Mainloop fc2(p.fc2, ClusterShape{}, cta);
    int tokenBlocks = taskTokenBlocks(p, tokens);
    int tasks = tokenBlocks * (fc1TaskTiles<Parameters>(intermediate) + fc2TaskTiles<Parameters>(hidden));
    const int activationRows = w4StorageRow<typename Parameters::Tiles>(p.workspace.poolRows);
    ProblemShape shape1{2 * intermediate, activationRows, hidden, 1};
    ProblemShape shape2{hidden, activationRows, intermediate, 1};

    // Reconfigure the complete transfer warpgroup before its load/MMA roles diverge.
    if (warp >= Schedule::EpilogueEnd) cutlass::arch::warpgroup_reg_dealloc<Types::TransferRegisters>();
    bool producerWarp = warp == Schedule::LoadWarp || (Types::LoadWarps == 2 && warp == Schedule::ActivationLoadWarp);
    if (producerWarp) {
      runW4A8LoadRole(p, s, loadPipeline, activationPipeline, fc1, fc2, shape1, shape2, hidden, intermediate, warp,
                      lane, cta, cluster, tasks, tokens);
    } else {
      if (warp == Schedule::MmaWarp) {
        allocator.allocate(W4TmemColumns, &s.tmem);
        // Static FP8 activations carry no block scales: MMA reads unit (E8M0 127) activation scales.
        constexpr int SfbWordsPerStage = sizeof(s.tensors.smem_SFB) / (Types::Load::Stages * sizeof(uint32_t));
        auto* sfbStage0 = reinterpret_cast<uint32_t*>(s.tensors.smem_SFB.begin());
        for (int i = lane; i < SfbWordsPerStage; i += warpSize) sfbStage0[i] = 0x7f7f7f7fU;
        cutlass::arch::fence_view_async_shared();
      }
      cutlass::arch::NamedBarrier::sync((Schedule::DispatchBegin - Types::LoadWarps) * warpSize, 2);
      if (threadIdx.x == 0) {
#pragma unroll
        for (int peerCta = 0; peerCta < ClusterSize; ++peerCta) s.tmemReady.arrive(peerCta);
      }
      s.tmemReady.wait(0);
      using EpilogueTile = Shape<Int<Parameters::Tiles::CtaM>, Int<Parameters::Tiles::N>>;
      auto tmemStorage = Mainloop::template init_tmem_tensors<EpilogueTile, false>(EpilogueTile{});
      Mainloop::set_tmem_offsets(tmemStorage, s.tmem);
      if (warp >= Schedule::EpilogueBegin && warp < Schedule::EpilogueEnd) {
        runEpilogueRole<Schedule, 2>(p, s, accumulatePipeline, tmemStorage.accumulators, hidden, intermediate, cluster,
                                     tasks, output, tokens);
      } else {
        runMmaRole(p, s, loadPipeline, activationPipeline, accumulatePipeline, fc1, tmemStorage, hidden, intermediate,
                   warp, cta, cluster, tasks, tokens);
      }
    }
  } else {
    // No dispatch role: these warps only release registers and join the cluster barriers.
    cutlass::arch::warpgroup_reg_dealloc<TransferRegisters>();
    cute::cluster_arrive_relaxed();
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
}

W8A8LocalKernelEntry w8a8LocalKernelEntry() { return megaMoeW8A8Local; }

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail
