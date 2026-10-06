// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// The scale-layout and pipeline operations adapt CUTLASS's SM100 collective:
// Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
// GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
// OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef MSCCLPP_EXT_MEGAMOE_W4A8_COLLECTIVE_CUH_
#define MSCCLPP_EXT_MEGAMOE_W4A8_COLLECTIVE_CUH_

#include <cute/tensor.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>

#include "megamoe_trace.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

// Reuse CUTLASS A/SF layouts with an explicit token-tile pipeline. N32 needs
// quarter-tile SFB selection, which the stock collective does not implement.
template <class Base>
struct W4A8Mainloop {
  using ElementA = typename Base::ElementA;
  using ElementB = typename Base::ElementB;
  using ArrayElementA = typename Base::ArrayElementA;
  using ArrayElementB = typename Base::ArrayElementB;
  using ElementSF = typename Base::ElementSF;
  using LayoutSFA = typename Base::LayoutSFA;
  using LayoutSFB = typename Base::LayoutSFB;
  using Sm1xxBlkScaledConfig = typename Base::Sm1xxBlkScaledConfig;
  using Arguments = typename Base::Arguments;
  using AtomThrShapeMNK = typename Base::AtomThrShapeMNK;
  using MainloopPipeline = typename Base::MainloopPipeline;
  using MainloopPipelineState = typename MainloopPipeline::PipelineState;
  using KernelTile = cute::Shape<cute::Int<256>, cute::Int<W4TileN>, cute::Int<W4TileK>>;
  using TiledMma = decltype(cute::make_tiled_mma(
      cute::SM100_MMA_MXF8F6F4_2x1SM_SS<typename Base::ElementAMma, typename Base::ElementBMma, float, ElementSF, 256,
                                        W4TileN, cute::UMMA::Major::K, cute::UMMA::Major::K>{}));
  using ScaleMma = typename Base::TiledMMA_SF;
  using SmemLayoutA = typename Base::SmemLayoutA;
  using SmemLayoutSFA = typename Base::SmemLayoutSFA;
  using SmemLayoutSFB = typename Base::SmemLayoutSFB;
  using SmemLayoutB = decltype(cute::UMMA::tile_to_mma_shape(
      typename Base::SmemLayoutAtomB{},
      cute::append(cute::partition_shape_B(TiledMma{}, cute::make_shape(cute::Int<W4TileN>{}, cute::Int<W4TileK>{})),
                   cute::Int<W4LoadStages>{}),
      cute::Step<cute::_1, cute::_2, cute::_3>{}));
  using ClusterLayout = typename Base::Params::ClusterLayout_VMNK;
  using TmaB = decltype(cute::make_tma_atom_B_sm100<typename Base::TmaInternalElementB>(
      typename Base::GmemTiledCopyB{},
      cute::make_tensor(cute::recast_ptr<typename Base::TmaInternalElementB>(nullptr),
                        cute::repeat_like(typename Base::StrideB{}, int32_t(0)), typename Base::StrideB{}),
      SmemLayoutB{}(cute::_, cute::_, cute::_, cute::_0{}), KernelTile{}, TiledMma{}, ClusterLayout{}));

  struct TensorStorage : cute::aligned_struct<128, cute::_0> {
    cute::ArrayEngine<typename Base::SmemAllocTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
    cute::ArrayEngine<typename Base::SmemAllocTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
    cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFA>> smem_SFA;
    cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFB>> smem_SFB;
  };
  struct Params {
    typename Base::Params::TMA_A tma_load_a;
    TmaB tma_load_b;
    typename Base::Params::TMA_SFA tma_load_sfa;
    typename Base::Params::TMA_SFB tma_load_sfb;
    LayoutSFA layout_SFA;
    LayoutSFB layout_SFB;
  };
  static constexpr uint32_t TmaTransactionBytes =
      Base::TmaTransactionBytes - 2 *
                                      (cute::cosize(cute::take<0, 3>(typename Base::SmemLayoutB{})) -
                                       cute::cosize(cute::take<0, 3>(SmemLayoutB{}))) *
                                      cute::sizeof_bits_v<ElementB> / 8;
  static constexpr uint32_t WeightTransactionBytes =
      2 *
      (cute::cosize(cute::take<0, 3>(SmemLayoutA{})) * cute::sizeof_bits_v<ElementA> +
       cute::cosize(cute::take<0, 3>(SmemLayoutSFA{})) * cute::sizeof_bits_v<ElementSF>) /
      8;
  static constexpr uint32_t ActivationTransactionBytes = TmaTransactionBytes - WeightTransactionBytes;

  const Params& params;
  int cta;

  __device__ W4A8Mainloop(const Params& p, ClusterShape, uint32_t rank) : params(p), cta(rank) {}

  __host__ __device__ static constexpr int mmaRows(int validRows) { return (validRows + 15) / 16 * 16; }

  static bool can_implement(const ProblemShape& shape, const Arguments& args) {
    return Base::can_implement(shape, args);
  }
  static Params to_underlying_arguments(const ProblemShape& shape, const Arguments& args, void* workspace) {
    using namespace cute;
    auto base = Base::to_underlying_arguments(shape, args, workspace);
    auto [m, n, k, l] = shape;
    auto tensor = make_tensor(recast_ptr<typename Base::TmaInternalElementB>(args.ptr_B),
                              make_layout(make_shape(n, k, l), args.dB));
    auto cluster = tiled_divide(make_layout(ClusterShape{}), make_tile(typename TiledMma::AtomThrID{}));
    auto tmaB = make_tma_atom_B_sm100<typename Base::TmaInternalElementB>(
        typename Base::GmemTiledCopyB{}, tensor, SmemLayoutB{}(_, _, _, _0{}), KernelTile{}, TiledMma{}, cluster);
    return {base.tma_load_a, tmaB, base.tma_load_sfa, base.tma_load_sfb, args.layout_SFA, args.layout_SFB};
  }
  template <class EpilogueTile, bool Overlap = false>
  __device__ static auto init_tmem_tensors(EpilogueTile) {
    using namespace cute;
    static_assert(!Overlap);
    auto base = Base::template init_tmem_tensors<EpilogueTile, false>(EpilogueTile{});
    auto acc = TiledMma::make_fragment_C(append(partition_shape_C(TiledMma{}, take<0, 2>(KernelTile{})), _2{}));
    return typename Base::template TmemStorage<decltype(acc), decltype(base.tCtSFA), decltype(base.tCtSFB)>{
        acc, base.tCtSFA, base.tCtSFB};
  }
  template <class Tmem>
  __device__ static void set_tmem_offsets(Tmem& tensors, uint32_t base) {
    Base::set_tmem_offsets(tensors, base);
  }
  template <class Tmem>
  __device__ static auto slice_accumulator(Tmem tensors, int stage) {
    return Base::slice_accumulator(tensors, stage);
  }

  __device__ auto load_init(const ProblemShape& shape, TensorStorage& s) const {
    using namespace cute;
    using X = Underscore;
    auto [m, n, k, l] = shape;
    auto a = params.tma_load_a.get_tma_tensor(make_shape(m, k, l));
    auto b = params.tma_load_b.get_tma_tensor(make_shape(n, k, l));
    auto sfa = params.tma_load_sfa.get_tma_tensor(cute::shape(params.layout_SFA));
    auto originalSfb = params.tma_load_sfb.get_tma_tensor(cute::shape(params.layout_SFB));
    auto sfbShape = make_shape(
        make_shape(cute::shape<0, 0>(originalSfb), make_shape(Int<128 / W4TileN>{}, cute::shape<0, 1>(originalSfb))),
        cute::shape<1>(originalSfb), cute::shape<2>(originalSfb));
    auto sfbStride = make_stride(make_stride(stride<0, 0>(originalSfb), make_stride(_0{}, stride<0, 1>(originalSfb))),
                                 stride<1>(originalSfb), stride<2>(originalSfb));
    auto sfb = make_tensor(originalSfb.data(), make_layout(sfbShape, sfbStride));
    auto mma = TiledMma{}.get_slice(cta);
    auto sfMma = ScaleMma{}.get_slice(cta % size(typename ScaleMma::AtomThrID{}));
    auto ga = mma.partition_A(local_tile(a, KernelTile{}, make_coord(_, _, _), Step<_1, X, _1>{}));
    auto gb = mma.partition_B(local_tile(b, KernelTile{}, make_coord(_, _, _), Step<X, _1, _1>{}));
    auto gsa = mma.partition_A(local_tile(sfa, KernelTile{}, make_coord(_, _, _), Step<_1, X, _1>{}));
    auto gsb =
        sfMma.partition_B(local_tile(sfb, typename Base::TileShape_SF{}, make_coord(_, _, _), Step<X, _1, _1>{}));
    auto sa = make_tensor(make_smem_ptr(s.smem_A.begin()), SmemLayoutA{});
    auto sb = make_tensor(make_smem_ptr(s.smem_B.begin()), SmemLayoutB{});
    auto ssa = make_tensor(make_smem_ptr(s.smem_SFA.begin()), SmemLayoutSFA{});
    auto ssb = make_tensor(make_smem_ptr(s.smem_SFB.begin()), SmemLayoutSFB{});
    auto cluster = tiled_divide(make_layout(ClusterShape{}), make_tile(typename TiledMma::AtomThrID{}));
    auto sfCluster = tiled_divide(make_layout(ClusterShape{}), make_tile(typename ScaleMma::AtomThrID{}));
    auto coord = cluster.get_flat_coord(cta);
    auto sfCoord = sfCluster.get_flat_coord(cta);
    auto [tga, tsa] = tma_partition(params.tma_load_a, get<2>(coord), make_layout(size<2>(cluster)),
                                    group_modes<0, 3>(sa), group_modes<0, 3>(ga));
    auto [tgb, tsb] = tma_partition(params.tma_load_b, get<1>(coord), make_layout(size<1>(cluster)),
                                    group_modes<0, 3>(sb), group_modes<0, 3>(gb));
    auto [tgsa, tssa] = tma_partition(params.tma_load_sfa, get<2>(coord), make_layout(size<2>(cluster)),
                                      group_modes<0, 3>(ssa), group_modes<0, 3>(gsa));
    auto [tgsb, tssb] = tma_partition(params.tma_load_sfb, get<1>(sfCoord), make_layout(size<1>(sfCluster)),
                                      group_modes<0, 3>(ssb), group_modes<0, 3>(gsb));
    return typename Base::template LoadParams<decltype(size<4>(ga)), decltype(tga), decltype(tgb), decltype(tsa),
                                              decltype(tsb), decltype(tgsa), decltype(tgsb), decltype(tssa),
                                              decltype(tssb)>{size<4>(ga),
                                                              tga,
                                                              tgb,
                                                              tsa,
                                                              tsb,
                                                              tgsa,
                                                              tgsb,
                                                              tssa,
                                                              tssb,
                                                              create_tma_multicast_mask<2>(cluster, coord),
                                                              create_tma_multicast_mask<1>(cluster, coord),
                                                              create_tma_multicast_mask<2>(cluster, coord),
                                                              create_tma_multicast_mask<1>(sfCluster, sfCoord)};
  }

  template <class Tmem>
  __device__ auto mma_init(Tmem tmem, TensorStorage& s) const {
    using namespace cute;
    auto a = TiledMma::make_fragment_A(make_tensor(make_smem_ptr(s.smem_A.begin()), SmemLayoutA{}));
    auto b = TiledMma::make_fragment_B(make_tensor(make_smem_ptr(s.smem_B.begin()), SmemLayoutB{}));
    auto sa = make_tensor(make_smem_ptr(s.smem_SFA.begin()), SmemLayoutSFA{});
    auto sb = make_tensor(make_smem_ptr(s.smem_SFB.begin()), SmemLayoutSFB{});
    auto ta = make_tensor(tmem.tCtSFA.data(), filter_zeros(tmem.tCtSFA.layout()));
    auto tb = make_tensor(tmem.tCtSFB.data(), filter_zeros(tmem.tCtSFB.layout()));
    using Utccp = SM100_UTCCP_4x32dp128bit_2cta;
    auto ca = make_utccp_copy(Utccp{}, ta);
    auto cb = make_utccp_copy(Utccp{}, tb);
    auto sfa = get_utccp_smem_desc_tensor<Utccp>(
        ca.get_slice(0).partition_S(make_tensor(sa.data(), filter_zeros(sa.layout()))));
    auto sfb = get_utccp_smem_desc_tensor<Utccp>(
        cb.get_slice(0).partition_S(make_tensor(sb.data(), filter_zeros(sb.layout()))));
    return make_tuple(TiledMma{}, a, b, tmem.tCtSFA, tmem.tCtSFB, ca, sfa, ca.get_slice(0).partition_D(ta), cb, sfb,
                      cb.get_slice(0).partition_D(tb));
  }

  template <int Operands = 0, class Inputs, class Coord, class Iterator>
  __device__ auto load(MainloopPipeline pipe, MainloopPipelineState state, const Inputs& inputs, Coord coord,
                       Iterator iterator, int kTiles, int validRows) const {
    using namespace cute;
    int m = get<0>(coord) / 2, n = get<1>(coord), expert = get<3>(coord);
    auto a = inputs.tAgA_mkl(_, m, _, expert);
    auto originalB = inputs.tBgB_nkl(_, n, _, expert);
    // A two-CTA MMA splits activation rows at its runtime N/2, not the allocated tile's N/2.
    auto rowLayout = params.tma_load_b.get_tma_tensor(make_shape(1, 1, 1)).layout();
    auto rowOffset = rowLayout(make_coord(cta * (mmaRows(validRows) - W4TileN) / 2, 0, 0));
    auto b = make_tensor(originalB.data() + rowOffset, originalB.layout());
    auto sfa = inputs.tAgSFA_mkl(_, m, _, expert);
    auto sfb = inputs.tBgSFB_nkl(_, n, _, expert);
    uint64_t waitNs = 0, issueNs = 0;
    const int iterations = kTiles;
    for (; kTiles > 0; --kTiles, ++iterator) {
      auto start = traceW4Clock();
      pipe.producer_acquire(state);
      waitNs += traceW4Clock() - start;
      auto* barrier = pipe.producer_get_barrier(state);
      int stage = state.index();
      start = traceW4Clock();
      if (elect_one_sync()) {
        if constexpr (Operands != 2) {
          copy(params.tma_load_a.with(*barrier, inputs.mcast_mask_a, TMA::CacheHintSm100::EVICT_FIRST), a(_, *iterator),
               inputs.tAsA(_, stage));
          copy(params.tma_load_sfa.with(*barrier, inputs.mcast_mask_sfa, TMA::CacheHintSm100::EVICT_FIRST),
               sfa(_, *iterator), inputs.tAsSFA(_, stage));
        }
        if constexpr (Operands != 1) {
          copy(params.tma_load_b.with(*barrier, inputs.mcast_mask_b, TMA::CacheHintSm100::EVICT_LAST), b(_, *iterator),
               inputs.tBsB(_, stage));
          copy(params.tma_load_sfb.with(*barrier, inputs.mcast_mask_sfb, TMA::CacheHintSm100::EVICT_LAST),
               sfb(_, *iterator), inputs.tBsSFB(_, stage));
        }
      }
      issueNs += traceW4Clock() - start;
      ++state;
    }
    traceW4Total(W4TracePhase::LoadAcquire, waitNs, iterations);
    traceW4Total(W4TracePhase::TmaIssue, issueNs, iterations);
    return make_tuple(state, iterator);
  }
  __device__ void load_tail(MainloopPipeline pipe, MainloopPipelineState state) const { pipe.producer_tail(state); }
  template <class Pipelines, class States, class Accumulators, class Inputs, class Coord>
  __device__ auto mma(Pipelines pipes, States states, Accumulators accumulators, const Inputs& inputs, Coord coord,
                      int kTiles, int validRows) const {
    using namespace cute;
    auto [loadPipe, activationPipe, accPipe] = pipes;
    auto [loadState, accState] = states;
    auto [tiled, a, b, sfa, sfb, copyA, sourceA, targetA, copyB, sourceB, targetB] = inputs;
    tiled.idesc_.n_dim_ = mmaRows(validRows) >> 3;
    sfb.data() = sfb.data().get() + (get<1>(coord) % (128 / W4TileN)) * (W4TileN / 32);
    traceW4(W4TracePhase::AccumulatorFree, true, accState.index());
    accPipe.producer_acquire(accState);
    traceW4(W4TracePhase::AccumulatorFree, false, accState.index());
    tiled.accumulate_ = UMMA::ScaleOut::Zero;
    uint64_t waitNs = 0, activationWaitNs = 0, issueNs = 0, releaseNs = 0;
    uint32_t weightReady = 0, activationReady = 0;
    const int iterations = kTiles;
    const bool issuer = elect_one_sync();
    const uint32_t destination = raw_pointer_cast(get<0>(accumulators).data());
    for (; kTiles > 0; --kTiles) {
#if defined(MSCCLPP_MEGAMOE_W4_TRACE) && MSCCLPP_MEGAMOE_W4_TRACE == 1
      // Nonblocking probes sample both operands before either blocking wait.
      weightReady +=
          cutlass::arch::ClusterBarrier::test_wait(loadPipe.producer_get_barrier(loadState), loadState.phase(), 1);
      if constexpr (W4SplitPipelines)
        activationReady += cutlass::arch::ClusterBarrier::test_wait(activationPipe.producer_get_barrier(loadState),
                                                                    loadState.phase(), 1);
#endif
      auto start = traceW4Clock();
      loadPipe.consumer_wait(loadState);
      waitNs += traceW4Clock() - start;
      if constexpr (W4SplitPipelines) {
        start = traceW4Clock();
        activationPipe.consumer_wait(loadState);
        activationWaitNs += traceW4Clock() - start;
      }
      int stage = loadState.index();
      start = traceW4Clock();
      if (issuer) {
        copy(copyA, sourceA(_, _, _, _, stage), targetA);
        copy(copyB, sourceB(_, _, _, _, stage), targetB);
        // Pipeline waits/releases remain warp-wide.
        CUTE_UNROLL
        for (int k = 0; k < size<2>(a); ++k) {
          uint32_t scaleA = raw_pointer_cast(sfa(_, _, k).data());
          uint32_t scaleB = raw_pointer_cast(sfb(_, _, k).data());
          uint32_t instruction =
              uint32_t(UMMA::make_runtime_instr_desc_block_scaled<>(tiled.idesc_, scaleA, scaleB) >> 32);
          uint64_t descriptorA = a(_, _, k, stage)(0);
          uint64_t descriptorB = b(_, _, k, stage)(0);
          uint32_t accumulate = k == 0 ? uint32_t(tiled.accumulate_) : 1;
          asm volatile(
              "{ .reg .pred p; setp.ne.b32 p, %6, 0;"
              "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale"
              " [%0], %1, %2, %3, [%4], [%5], p; }"
              :
              : "r"(destination), "l"(descriptorA), "l"(descriptorB), "r"(instruction), "r"(scaleA), "r"(scaleB),
                "r"(accumulate));
        }
      }
      tiled.accumulate_ = UMMA::ScaleOut::One;
      issueNs += traceW4Clock() - start;
      start = traceW4Clock();
      loadPipe.consumer_release(loadState);
      if constexpr (W4SplitPipelines) activationPipe.consumer_release(loadState);
      releaseNs += traceW4Clock() - start;
      ++loadState;
    }
    if constexpr (W4SplitPipelines) {
      traceW4Total(W4TracePhase::MmaWeightWait, waitNs, iterations);
      traceW4Total(W4TracePhase::MmaActivationWait, activationWaitNs, iterations);
      traceW4Count(W4TracePhase::WeightReadyAtEntry, weightReady, iterations);
      traceW4Count(W4TracePhase::ActivationReadyAtEntry, activationReady, iterations);
    } else {
      traceW4Total(W4TracePhase::MmaInputWait, waitNs, iterations);
      traceW4Count(W4TracePhase::InputReadyAtEntry, weightReady, iterations);
    }
    traceW4Total(W4TracePhase::MmaIssue, issueNs, iterations);
    traceW4Total(W4TracePhase::StageRelease, releaseNs, iterations);
    return loadState;
  }
};

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_W4A8_COLLECTIVE_CUH_
