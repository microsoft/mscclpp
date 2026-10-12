// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_MMA_CUH_
#define MSCCLPP_EXT_MEGAMOE_MMA_CUH_

#include "megamoe_collective.cuh"
#include "megamoe_tasks.cuh"
#include "megamoe_trace.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

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
    for (int i = cluster; i < tasks; i += gridDim.x / P::Tiles::ClusterSize) {
      Task task = taskAt(p, i, hidden, intermediate);
      auto coord = taskCoord(p, task, cta);
      int kTiles = taskKTiles(p, task, hidden, intermediate);
      auto accumulator = Mainloop::slice_accumulator(tmemStorage, accumulateState.index());
      auto phase = task.fc1 ? KernelTracePhase::MmaFc1 : KernelTracePhase::MmaFc2;
      traceKernel(phase, true, i);
      loadState =
          fc1.mma(make_tuple(loadPipeline, activationPipeline, accumulatePipeline),
                  make_tuple(loadState, accumulateState), accumulator, mmaInputs, coord, kTiles, task.tokens.rows);
      traceKernel(phase, false, i);
      accumulatePipeline.producer_commit(accumulateState);
      ++accumulateState;
    }
    accumulatePipeline.producer_tail(accumulateState);
  }
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_MMA_CUH_
