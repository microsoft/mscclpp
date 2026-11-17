// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_MMA_CUH_
#define MSCCLPP_EXT_MEGAMOE_MMA_CUH_

#include "megamoe_collective.cuh"
#include "megamoe_tasks.cuh"
#include "megamoe_trace.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

template <class Pipeline, class Schedule, class ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>>
__device__ __forceinline__ Pipeline makeAccumulatorPipeline(typename Pipeline::SharedStorage& storage, int warp) {
  typename Pipeline::Params params{};
  params.role = warp == Schedule::MmaWarp ? Pipeline::ThreadCategory::Producer
                                          : (warp >= Schedule::EpilogueBegin && warp < Schedule::EpilogueEnd
                                                 ? Pipeline::ThreadCategory::Consumer
                                                 : Pipeline::ThreadCategory::NonParticipant);
  params.producer_arv_count = 1;
  params.consumer_arv_count = cute::size(ClusterShape{}) * (Schedule::EpilogueEnd - Schedule::EpilogueBegin) * 32;
  params.initializing_warp = Schedule::EpilogueBegin;
  return Pipeline(storage, params, ClusterShape{}, cute::true_type{}, cute::false_type{});
}

template <bool Local = false, class FrgEngine, class FrgLayout, class TensorA, class TensorB>
__device__ __forceinline__ auto mmaTiles(
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
        uint32_t weightAddress = raw_pointer_cast(weight.data());
        uint64_t activationDescriptor = activation(0);
        uint32_t accumulate = k == 0 ? uint32_t(tiledMma.accumulate_) : 1;
        asm volatile(
            "{ .reg .pred p; setp.ne.b32 p, %4, 0;"
            "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3,"
            "{%5,%5,%5,%5,%5,%5,%5,%5}, p; }"
            :
            : "r"(destination), "r"(weightAddress), "l"(activationDescriptor), "r"(instruction), "r"(accumulate),
              "r"(uint32_t(0)));
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

template <bool Local, class Schedule, class P, class Storage, class LoadB, class Transform, class Accumulate,
          class Mainloop, class Accumulators>
__device__ __forceinline__ void runW8A16MmaRole(const P& p, int tokens, Storage& s, LoadB& bPipeline,
                                                Transform& tPipeline, Accumulate& cPipeline, Mainloop& fc1,
                                                Accumulators accumulators, int hidden, int intermediate, int warp,
                                                int cta, int cluster, int tasks) {
  if (warp == Schedule::MmaWarp && cta == 0) {
    typename LoadB::PipelineState bState;
    typename Transform::PipelineState tState;
    auto cState = cutlass::make_producer_start_state<Accumulate>();
    auto inputs = fc1.mma_init(accumulators, s.tensors);
    for (int i = cluster; i < tasks; i += gridDim.x / P::Tiles::ClusterSize) {
      Task task = taskAt(p, i, hidden, intermediate, tokens);
      auto phase = task.fc1 ? KernelTracePhase::MmaFc1 : KernelTracePhase::MmaFc2;
      traceKernel(phase, true, i);
      auto result = mmaTiles<Local>(bPipeline, bState, tPipeline, tState, cPipeline, cState, accumulators, inputs,
                                    taskKTiles(p, task, hidden, intermediate), task.tokens.rows);
      traceKernel(phase, false, i);
      bState = cute::get<0>(result);
      tState = cute::get<1>(result);
      cState = cute::get<2>(result);
    }
    cPipeline.producer_tail(cState);
  }
}

template <class P, class Storage, class TmemStorage>
__device__ __forceinline__ void runMmaRole(const P& p, Storage& s, typename P::Collective::Load& loadPipeline,
                                           typename P::Collective::Load& activationPipeline,
                                           typename P::Collective::Accumulate& accumulatePipeline,
                                           typename P::Collective::Mainloop& fc1, TmemStorage tmemStorage, int hidden,
                                           int intermediate, int warp, int cta, int cluster, int tasks) {
  using namespace cute;
  using Mainloop = typename P::Collective::Mainloop;
  using Load = typename P::Collective::Load;
  using Accumulate = typename P::Collective::Accumulate;
  using Schedule = typename P::Collective::WarpSchedule;
  if (warp == Schedule::MmaWarp && cta == 0) {
    typename Load::PipelineState loadState;
    auto accumulateState = cutlass::make_producer_start_state<Accumulate>();
    auto mmaInputs = fc1.mma_init(tmemStorage, s.tensors);
    if constexpr (P::Activation == ActivationDtype::Fp8Static) {
      if (elect_one_sync()) copy(get<8>(mmaInputs), get<9>(mmaInputs)(_, _, _, _, 0), get<10>(mmaInputs));
    }
    for (int i = cluster; i < tasks; i += gridDim.x / P::Tiles::ClusterSize) {
      Task task = taskAt(p, i, hidden, intermediate);
      auto coord = taskCoord(p, task, cta);
      int kTiles = taskKTiles(p, task, hidden, intermediate);
      auto accumulator = Mainloop::slice_accumulator(tmemStorage, accumulateState.index());
      auto phase = task.fc1 ? KernelTracePhase::MmaFc1 : KernelTracePhase::MmaFc2;
      traceKernel(phase, true, i);
      loadState = fc1.template mma<P::Activation>(make_tuple(loadPipeline, activationPipeline, accumulatePipeline),
                                                  make_tuple(loadState, accumulateState), accumulator, mmaInputs, coord,
                                                  kTiles, task.tokens.rows);
      traceKernel(phase, false, i);
      accumulatePipeline.producer_commit(accumulateState);
      ++accumulateState;
    }
    accumulatePipeline.producer_tail(accumulateState);
  }
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_MMA_CUH_
