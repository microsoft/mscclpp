// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// The pipeline loop is adapted from CUTLASS sm100_mma_warpspecialized_mixed_input.hpp:
// Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef MSCCLPP_EXT_MEGAMOE_MMA_CUH_
#define MSCCLPP_EXT_MEGAMOE_MMA_CUH_

#include "megamoe_collective.cuh"
#include "megamoe_tasks.cuh"
#include "megamoe_trace.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

template <bool E5M2, bool Local = false, class FrgEngine, class FrgLayout, class TensorA, class TensorB>
__device__ __forceinline__ auto mmaTiles(
    typename CollectiveTypes<E5M2, Local>::LoadB loadPipeline,
    typename CollectiveTypes<E5M2, Local>::LoadB::PipelineState loadState,
    typename CollectiveTypes<E5M2, Local>::Transform transformPipeline,
    typename CollectiveTypes<E5M2, Local>::Transform::PipelineState transformState,
    typename CollectiveTypes<E5M2, Local>::Accumulate accumulatePipeline,
    typename CollectiveTypes<E5M2, Local>::Accumulate::PipelineState accumulateState,
    const cute::Tensor<FrgEngine, FrgLayout>& accumulators, const cute::tuple<TensorA, TensorB>& operands, int kTiles) {
  using namespace cute;
  using TiledMma = typename CollectiveTypes<E5M2, Local>::Mainloop::TiledMma;
  TiledMma tiledMma;
  auto currentLoad = loadState;
  auto nextLoad = loadState;
  auto currentTransform = transformState;
  auto nextTransform = transformState;
  uint32_t skip = kTiles <= 0;
  auto transformReady = transformPipeline.consumer_try_wait(nextTransform, skip || Local);
  auto loadReady = loadPipeline.consumer_try_wait(nextLoad, skip || Local);
  ++nextTransform;
  ++nextLoad;
  const auto [weights, activations] = operands;
  accumulatePipeline.producer_acquire(accumulateState);
  auto accumulator = accumulators(_, _, _, accumulateState.index());
  auto currentAccumulate = accumulateState;
  ++accumulateState;
  tiledMma.accumulate_ = UMMA::ScaleOut::Zero;
  bool localIssuer = false;
  uint32_t localDestination = 0, localInstruction = 0;
  bool routedIssuer = false;
  uint32_t routedDestination = 0, routedInstruction = 0;
  if constexpr (Local) {
    localIssuer = elect_one_sync();
    localDestination = raw_pointer_cast(accumulator.data());
    localInstruction = uint32_t(UMMA::make_runtime_instr_desc<>(tiledMma.idesc_) >> 32);
  } else {
    routedIssuer = elect_one_sync();
    routedDestination = raw_pointer_cast(accumulator.data());
    routedInstruction = uint32_t(UMMA::make_runtime_instr_desc<>(tiledMma.idesc_) >> 32);
  }
  CUTLASS_PRAGMA_NO_UNROLL
  for (; kTiles > 0; --kTiles) {
    if constexpr (Local) {
      loadPipeline.consumer_wait(currentLoad);
      transformPipeline.consumer_wait(currentTransform);
    } else {
      loadPipeline.consumer_wait(currentLoad, loadReady);
      transformPipeline.consumer_wait(currentTransform, transformReady);
    }
    auto a = weights(_, _, _, currentTransform.index());
    auto b = activations(_, _, _, currentLoad.index());

    // Elect once for the K128 bundle, not once for each K16 MMA. Pipeline
    // waits/releases below still require the complete issuing warp.
    if (Local ? localIssuer : routedIssuer) {
      uint32_t destination = Local ? localDestination : routedDestination;
      uint32_t instruction = Local ? localInstruction : routedInstruction;
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
    if constexpr (Local) {
      ++currentLoad;
      ++currentTransform;
    } else {
      skip = kTiles <= 1;
      loadReady = loadPipeline.consumer_try_wait(nextLoad, skip);
      transformReady = transformPipeline.consumer_try_wait(nextTransform, skip);
      currentLoad = nextLoad;
      currentTransform = nextTransform;
      ++nextLoad;
      ++nextTransform;
    }
  }
  accumulatePipeline.producer_commit(currentAccumulate);
  return make_tuple(currentLoad, currentTransform, accumulateState);
}

template <bool E5M2, bool Local, class Schedule, class P, class Storage, class LoadB, class Transform, class Accumulate,
          class Mainloop, class Accumulators>
__device__ __forceinline__ void runMmaRole(const P& p, int tokens, Storage& s, LoadB& bPipeline, Transform& tPipeline,
                                           Accumulate& cPipeline, Mainloop& fc1, Accumulators accumulators, int hidden,
                                           int intermediate, int warp, int cta, int cluster, int tasks) {
  if (warp == Schedule::MmaWarp && cta == 0) {
    typename LoadB::PipelineState bState;
    typename Transform::PipelineState tState;
    auto cState = cutlass::make_producer_start_state<Accumulate>();
    auto inputs = fc1.mma_init(accumulators, s.tensors);
    for (int i = cluster; i < tasks; i += gridDim.x / ClusterM) {
      Task task = taskAt(p, i, tokens, hidden, intermediate);
      auto phase = task.fc1 ? KernelTracePhase::MmaFc1 : KernelTracePhase::MmaFc2;
      traceKernel(phase, true, i);
      auto result = mmaTiles<E5M2, Local>(bPipeline, bState, tPipeline, tState, cPipeline, cState, accumulators, inputs,
                                          taskKTiles(p, task, hidden, intermediate));
      traceKernel(phase, false, i);
      bState = cute::get<0>(result);
      tState = cute::get<1>(result);
      cState = cute::get<2>(result);
    }
    cPipeline.producer_tail(cState);
  }
}

#if MSCCLPP_MEGAMOE_COMPILE_W4A8

template <class P, class Storage, class TmemStorage>
__device__ __forceinline__ void runW4A8MmaRole(const P& p, Storage& s, typename P::Collective::Load& loadPipeline,
                                               typename P::Collective::Load& activationPipeline,
                                               typename P::Collective::Accumulate& accumulatePipeline,
                                               typename P::Collective::Mainloop& fc1, TmemStorage tmemStorage,
                                               int hidden, int intermediate, int warp, int cta, int cluster,
                                               int tasks) {
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
      Task task = taskAt(p, i, 0, hidden, intermediate);
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

#endif  // MSCCLPP_MEGAMOE_COMPILE_W4A8

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_MMA_CUH_
