// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_MMA_CUH_
#define MSCCLPP_EXT_MEGAMOE_MMA_CUH_

#include "megamoe_collective.cuh"
#include "megamoe_tasks.cuh"
#include "megamoe_trace.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

template <class Pipeline, class Schedule, class ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>>
MSCCLPP_DEVICE_INLINE Pipeline makeAccumulatorPipeline(typename Pipeline::SharedStorage& storage, int warp) {
  typename Pipeline::Params params{};
  params.role = warp == Schedule::MmaWarp ? Pipeline::ThreadCategory::Producer
                                          : (warp >= Schedule::EpilogueBegin && warp < Schedule::EpilogueEnd
                                                 ? Pipeline::ThreadCategory::Consumer
                                                 : Pipeline::ThreadCategory::NonParticipant);
  params.producer_arv_count = 1;
  params.consumer_arv_count =
      cute::size(ClusterShape{}) * (Schedule::EpilogueEnd - Schedule::EpilogueBegin) * WARP_SIZE;
  params.initializing_warp = Schedule::EpilogueBegin;
  return Pipeline(storage, params, ClusterShape{}, cute::true_type{}, cute::false_type{});
}

MSCCLPP_DEVICE_INLINE void issueW8A16Mma(uint32_t destination, uint32_t weightAddress, uint64_t activationDescriptor,
                                         uint32_t instruction, uint32_t accumulate) {
  asm volatile(
      "{ .reg .pred p; setp.ne.b32 p, %4, 0;"
      "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3,"
      "{%5,%5,%5,%5,%5,%5,%5,%5}, p; }"
      :
      : "r"(destination), "r"(weightAddress), "l"(activationDescriptor), "r"(instruction), "r"(accumulate),
        "r"(uint32_t(0)));
}

template <int ClusterSize>
MSCCLPP_DEVICE_INLINE void issueW4A8BlockScaledMma(uint32_t destination, uint64_t descriptorA, uint64_t descriptorB,
                                                   uint32_t instruction, uint32_t scaleA, uint32_t scaleB,
                                                   uint32_t accumulate) {
  static_assert(ClusterSize == 1 || ClusterSize == 2);
  if constexpr (ClusterSize == 1) {
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %6, 0;"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%4], [%5], p; }"
        :
        : "r"(destination), "l"(descriptorA), "l"(descriptorB), "r"(instruction), "r"(scaleA), "r"(scaleB),
          "r"(accumulate));
  } else {
    asm volatile(
        "{ .reg .pred p; setp.ne.b32 p, %6, 0;"
        "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%4], [%5], p; }"
        :
        : "r"(destination), "l"(descriptorA), "l"(descriptorB), "r"(instruction), "r"(scaleA), "r"(scaleB),
          "r"(accumulate));
  }
}

template <bool Local = false, class FrgEngine, class FrgLayout, class TensorA, class TensorB>
MSCCLPP_DEVICE_INLINE auto mmaW8A16Tiles(
    typename W8A16CollectiveTypesT<Local>::LoadB loadPipeline,
    typename W8A16CollectiveTypesT<Local>::LoadB::PipelineState loadState,
    typename W8A16CollectiveTypesT<Local>::Transform transformPipeline,
    typename W8A16CollectiveTypesT<Local>::Transform::PipelineState transformState,
    typename W8A16CollectiveTypesT<Local>::Accumulate accumulatePipeline,
    typename W8A16CollectiveTypesT<Local>::Accumulate::PipelineState accumulateState,
    const cute::Tensor<FrgEngine, FrgLayout>& accumulators, const cute::tuple<TensorA, TensorB>& operands, int kTiles,
    int validRows) {
  using namespace cute;
  using TiledMma = typename W8A16CollectiveTypesT<Local>::Mainloop::TiledMma;
  TiledMma tiledMma;
  tiledMma.idesc_.n_dim_ = ((validRows + 15) / 16 * 16) >> 3;
  auto currentLoad = loadState;
  auto nextLoad = loadState;
  auto currentTransform = transformState;
  auto nextTransform = transformState;
  uint32_t skip = kTiles <= 0;
  auto transformReady = transformPipeline.consumer_try_wait(nextTransform, skip);
  auto loadReady = loadPipeline.consumer_try_wait(nextLoad, skip);
  ++nextTransform;
  ++nextLoad;
  const auto [weights, activations] = operands;
  accumulatePipeline.producer_acquire(accumulateState);
  auto accumulator = accumulators(_, _, _, accumulateState.index());
  auto currentAccumulate = accumulateState;
  ++accumulateState;
  tiledMma.accumulate_ = UMMA::ScaleOut::Zero;
  bool issuer = elect_one_sync();
  uint32_t destination = raw_pointer_cast(accumulator.data());
  uint32_t instruction = uint32_t(UMMA::make_runtime_instr_desc<>(tiledMma.idesc_) >> 32);
  CUTLASS_PRAGMA_NO_UNROLL
  for (; kTiles > 0; --kTiles) {
    loadPipeline.consumer_wait(currentLoad, loadReady);
    transformPipeline.consumer_wait(currentTransform, transformReady);
    auto a = weights(_, _, _, currentTransform.index());
    auto b = activations(_, _, _, currentLoad.index());

    // Elect once for the K128 bundle, not once for each K16 MMA. Pipeline
    // waits/releases below still require the complete issuing warp.
    if (issuer) {
#pragma unroll
      for (int k = 0; k < size<2>(weights); ++k) {
        auto weight = a(_, _, k);
        auto activation = b(_, _, k);
        issueW8A16Mma(destination, raw_pointer_cast(weight.data()), activation(0), instruction,
                      k == 0 ? uint32_t(tiledMma.accumulate_) : 1u);
      }
    }
    tiledMma.accumulate_ = UMMA::ScaleOut::One;
    loadPipeline.consumer_release(currentLoad);
    transformPipeline.consumer_release(currentTransform);
    skip = kTiles <= 1;
    loadReady = loadPipeline.consumer_try_wait(nextLoad, skip);
    transformReady = transformPipeline.consumer_try_wait(nextTransform, skip);
    currentLoad = nextLoad;
    currentTransform = nextTransform;
    ++nextLoad;
    ++nextTransform;
  }
  accumulatePipeline.producer_commit(currentAccumulate);
  return make_tuple(currentLoad, currentTransform, accumulateState);
}

template <int ClusterSize, bool SplitPipelines, ActivationDtype ActDtype, class LoadPipeline, class LoadState,
          class AccumulatePipeline, class AccumulateState, class Accumulators, class MmaInputs>
MSCCLPP_DEVICE_INLINE LoadState mmaW4A8Tiles(LoadPipeline& loadPipe, LoadPipeline& activationPipe, LoadState loadState,
                                             AccumulatePipeline& accPipe, AccumulateState accState,
                                             Accumulators accumulators, const MmaInputs& inputs, int kTiles,
                                             int validRows) {
  using namespace cute;
  auto [tiled, a, b, sfa, sfb, copyA, sourceA, targetA, copyB, sourceB, targetB] = inputs;
  tiled.idesc_.n_dim_ = ((validRows + 15) / 16 * 16) >> 3;
  traceKernel(KernelTracePhase::AccumulatorFree, true, accState.index());
  accPipe.producer_acquire(accState);
  traceKernel(KernelTracePhase::AccumulatorFree, false, accState.index());
  tiled.accumulate_ = UMMA::ScaleOut::Zero;
  uint64_t waitNs = 0;
  uint64_t activationWaitNs = 0;
  uint64_t issueNs = 0;
  uint64_t releaseNs = 0;
  uint32_t weightReady = 0;
  uint32_t activationReady = 0;
  const int iterations = kTiles;
  // Elect once for the entire K-tile loop while keeping pipeline waits/releases warp-wide.
  const bool issuer = elect_one_sync();
  const uint32_t destination = raw_pointer_cast(get<0>(accumulators).data());
  for (; kTiles > 0; --kTiles) {
#if MSCCLPP_MEGAMOE_TRACE == 1
    // Nonblocking probes sample both operands before either blocking wait.
    weightReady +=
        cutlass::arch::ClusterBarrier::test_wait(loadPipe.producer_get_barrier(loadState), loadState.phase(), 1);
    if constexpr (SplitPipelines)
      activationReady += cutlass::arch::ClusterBarrier::test_wait(activationPipe.producer_get_barrier(loadState),
                                                                  loadState.phase(), 1);
#endif
    auto start = traceKernelClock();
    loadPipe.consumer_wait(loadState);
    waitNs += traceKernelClock() - start;
    if constexpr (SplitPipelines) {
      start = traceKernelClock();
      activationPipe.consumer_wait(loadState);
      activationWaitNs += traceKernelClock() - start;
    }
    int stage = loadState.index();
    start = traceKernelClock();
    if (issuer) {
      copy(copyA, sourceA(_, _, _, _, stage), targetA);
      if constexpr (ActDtype == ActivationDtype::Mxfp8) copy(copyB, sourceB(_, _, _, _, stage), targetB);
#pragma unroll
      for (int k = 0; k < size<2>(a); ++k) {
        uint32_t scaleA = raw_pointer_cast(sfa(_, _, k).data());
        uint32_t scaleB = raw_pointer_cast(sfb(_, _, k).data());
        uint32_t instruction =
            uint32_t(UMMA::make_runtime_instr_desc_block_scaled<>(tiled.idesc_, scaleA, scaleB) >> 32);
        issueW4A8BlockScaledMma<ClusterSize>(destination, a(_, _, k, stage)(0), b(_, _, k, stage)(0), instruction,
                                             scaleA, scaleB, k == 0 ? uint32_t(tiled.accumulate_) : 1u);
      }
    }
    tiled.accumulate_ = UMMA::ScaleOut::One;
    issueNs += traceKernelClock() - start;
    start = traceKernelClock();
    loadPipe.consumer_release(loadState);
    if constexpr (SplitPipelines) activationPipe.consumer_release(loadState);
    releaseNs += traceKernelClock() - start;
    ++loadState;
  }
  if constexpr (SplitPipelines) {
    traceKernelTotal(KernelTracePhase::MmaWeightWait, waitNs, iterations);
    traceKernelTotal(KernelTracePhase::MmaActivationWait, activationWaitNs, iterations);
    traceKernelCount(KernelTracePhase::WeightReadyAtEntry, weightReady, iterations);
    traceKernelCount(KernelTracePhase::ActivationReadyAtEntry, activationReady, iterations);
  } else {
    traceKernelTotal(KernelTracePhase::MmaInputWait, waitNs, iterations);
    traceKernelCount(KernelTracePhase::InputReadyAtEntry, weightReady, iterations);
  }
  traceKernelTotal(KernelTracePhase::MmaIssue, issueNs, iterations);
  traceKernelTotal(KernelTracePhase::StageRelease, releaseNs, iterations);
  return loadState;
}

template <bool Local, class Schedule, class KernelParams, class Storage, class LoadB, class Transform, class Accumulate,
          class Mainloop, class Accumulators>
MSCCLPP_DEVICE_INLINE void runW8A16MmaRole(const KernelParams& params, int nTokens, Storage& sharedStorage,
                                           LoadB& bPipeline, Transform& tPipeline, Accumulate& cPipeline, Mainloop& fc1,
                                           Accumulators accumulators, int hidden, int intermediate, int warpIdx,
                                           int ctaIdx, int clusterIdx, int nTasks, int nTokenTiles, int nLocalExperts) {
  if (warpIdx == Schedule::MmaWarp && ctaIdx == 0) {
    typename LoadB::PipelineState bState;
    typename Transform::PipelineState tState;
    auto cState = cutlass::make_producer_start_state<Accumulate>();
    auto inputs = fc1.mma_init(accumulators, sharedStorage.tensors);
    const auto scheduler = makeTaskScheduler<KernelParams>(nTokenTiles, nLocalExperts, hidden, intermediate);
    const int numClusters = gridDim.x / KernelParams::Tiles::ClusterSize;
    if (clusterIdx < nTasks) {
      Task nextTask = scheduler.taskAt(params, clusterIdx, nTokens);
      for (int i = clusterIdx; i < nTasks; i += numClusters) {
        Task task = nextTask;
        const bool hasNext = (i + numClusters) < nTasks;
        auto phase = task.fc1 ? KernelTracePhase::MmaFc1 : KernelTracePhase::MmaFc2;
        traceKernel(phase, true, i);
        auto result = mmaW8A16Tiles<Local>(bPipeline, bState, tPipeline, tState, cPipeline, cState, accumulators,
                                           inputs, taskKTiles(params, task, hidden, intermediate), task.tokens.nTokens);
        if (hasNext) nextTask = scheduler.taskAt(params, i + numClusters, nTokens);
        traceKernel(phase, false, i);
        bState = cute::get<0>(result);
        tState = cute::get<1>(result);
        cState = cute::get<2>(result);
      }
    }
    cPipeline.producer_tail(cState);
  }
}

template <class KernelParams, class Storage, class TmemStorage>
MSCCLPP_DEVICE_INLINE void runMmaRole(const KernelParams& params, Storage& sharedStorage,
                                      typename KernelParams::Collective::Load& loadPipeline,
                                      typename KernelParams::Collective::Load& activationPipeline,
                                      typename KernelParams::Collective::Accumulate& accumulatePipeline,
                                      typename KernelParams::Collective::Mainloop& fc1, TmemStorage tmemStorage,
                                      int hidden, int intermediate, int warpIdx, int ctaIdx, int clusterIdx, int nTasks,
                                      int nTokenTiles, int nLocalExperts) {
  using namespace cute;
  using Types = typename KernelParams::Collective;
  using Mainloop = typename Types::Mainloop;
  using Load = typename Types::Load;
  using Accumulate = typename Types::Accumulate;
  using Schedule = typename Types::WarpSchedule;
  if (warpIdx == Schedule::MmaWarp && ctaIdx == 0) {
    (void)ctaIdx;
    typename Load::PipelineState loadState;
    auto accumulateState = cutlass::make_producer_start_state<Accumulate>();
    auto mmaInputs = fc1.mma_init(tmemStorage, sharedStorage.tensors);
    if constexpr (KernelParams::Activation == ActivationDtype::Fp8Static) {
      if (elect_one_sync()) {
        copy(mmaInputs.tiled_copy_s2t_SFB, mmaInputs.thr_tCsSFB_s2t(_, _, _, _, 0), mmaInputs.thr_tCtSFB_s2t);
      }
    }
    const auto scheduler = makeTaskScheduler<KernelParams>(nTokenTiles, nLocalExperts, hidden, intermediate);
    const int numClusters = gridDim.x / KernelParams::Tiles::ClusterSize;
    if (clusterIdx < nTasks) {
      Task nextTask = scheduler.taskAt(params, clusterIdx);
      for (int i = clusterIdx; i < nTasks; i += numClusters) {
        Task task = nextTask;
        const bool hasNext = (i + numClusters) < nTasks;
        int kTiles = taskKTiles(params, task, hidden, intermediate);
        auto accumulator = Mainloop::slice_accumulator(tmemStorage, accumulateState.index());
        auto phase = task.fc1 ? KernelTracePhase::MmaFc1 : KernelTracePhase::MmaFc2;
        traceKernel(phase, true, i);
        loadState = mmaW4A8Tiles<KernelParams::Tiles::ClusterSize, Types::SplitPipelines, KernelParams::Activation>(
            loadPipeline, activationPipeline, loadState, accumulatePipeline, accumulateState, accumulator, mmaInputs,
            kTiles, task.tokens.nTokens);
        if (hasNext) nextTask = scheduler.taskAt(params, i + numClusters);
        traceKernel(phase, false, i);
        accumulatePipeline.producer_commit(accumulateState);
        ++accumulateState;
      }
    }
    accumulatePipeline.producer_tail(accumulateState);
  }
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_MMA_CUH_
