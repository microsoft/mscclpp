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

// Extends CUTLASS's SM100 block-scaled CollectiveMma (`Base`) with sub-tile activation TMA descriptors (N=64, N=32),
// split weight/activation load warps (`Operands`), and static-FP8 scale bypass.
template <class Base, int TileN, int TileK, int LoadStages, bool SplitPipelines, int ClusterSize>
struct W4A8Mainloop : Base {
  static_assert(ClusterSize == 1 || ClusterSize == 2);
  static_assert(TileN == 128);

  using ClusterShape = cute::Shape<cute::Int<ClusterSize>, cute::_1, cute::_1>;
  using CacheHint = cute::conditional_t<ClusterSize == 1, cute::TMA::CacheHintSm90, cute::TMA::CacheHintSm100>;
  using typename Base::Arguments;
  using typename Base::ElementA;
  using typename Base::ElementB;
  using typename Base::ElementSF;
  using typename Base::MainloopPipeline;
  using typename Base::MainloopPipelineState;
  using typename Base::SmemLayoutA;
  using typename Base::SmemLayoutB;
  using typename Base::SmemLayoutSFA;
  using typename Base::SmemLayoutSFB;
  using typename Base::TensorStorage;
  using typename Base::TiledMma;

  static constexpr uint32_t WeightTransactionBytes =
      ClusterSize *
      (cute::cosize(cute::take<0, 3>(SmemLayoutA{})) * cute::sizeof_bits_v<ElementA> +
       cute::cosize(cute::take<0, 3>(SmemLayoutSFA{})) * cute::sizeof_bits_v<ElementSF>) /
      8;
  static constexpr uint32_t SfbTransactionBytes =
      ClusterSize * cute::cosize(cute::take<0, 3>(SmemLayoutSFB{})) * cute::sizeof_bits_v<ElementSF> / 8;
  static constexpr uint32_t ActivationTransactionBytes = Base::TmaTransactionBytes - WeightTransactionBytes;

  template <int SubTileN>
  struct SubtileActivationTma {
    using SubTileShape = cute::Shape<cute::Int<128 * ClusterSize>, cute::Int<SubTileN>, cute::Int<TileK>>;
    using SubMmaOp = cute::conditional_t<
        ClusterSize == 1,
        cute::SM100_MMA_MXF8F6F4_SS<typename Base::ElementAMma, typename Base::ElementBMma, float, ElementSF, 128,
                                    SubTileN, cute::UMMA::Major::K, cute::UMMA::Major::K>,
        cute::SM100_MMA_MXF8F6F4_2x1SM_SS<typename Base::ElementAMma, typename Base::ElementBMma, float, ElementSF, 256,
                                          SubTileN, cute::UMMA::Major::K, cute::UMMA::Major::K>>;
    using SubTiledMma = decltype(cute::make_tiled_mma(SubMmaOp{}));
    using SubSmemLayoutB = decltype(cute::UMMA::tile_to_mma_shape(
        typename Base::SmemLayoutAtomB{},
        cute::append(
            cute::partition_shape_B(SubTiledMma{}, cute::make_shape(cute::Int<SubTileN>{}, cute::Int<TileK>{})),
            cute::Int<LoadStages>{}),
        cute::Step<cute::_1, cute::_2, cute::_3>{}));
    static constexpr uint32_t TransactionBytes =
        ClusterSize * cute::cosize(cute::take<0, 3>(SubSmemLayoutB{})) * cute::sizeof_bits_v<ElementB> / 8 +
        SfbTransactionBytes;

    static cute::TmaDescriptor makeDescriptor(const ProblemShape& shape, const Arguments& args) {
      using namespace cute;
      auto [m, n, k, l] = shape;
      (void)m;
      auto tensorB = make_tensor(recast_ptr<typename Base::TmaInternalElementB>(args.ptr_B),
                                 make_layout(make_shape(n, k, l), args.dB));
      auto cluster = tiled_divide(make_layout(ClusterShape{}), make_tile(typename TiledMma::AtomThrID{}));
      auto tmaB = make_tma_atom_B_sm100<typename Base::TmaInternalElementB>(typename Base::GmemTiledCopyB{}, tensorB,
                                                                            SubSmemLayoutB{}(_, _, _, _0{}),
                                                                            SubTileShape{}, SubTiledMma{}, cluster);
      return *tmaB.get_tma_descriptor();
    }
  };

  struct Params : Base::Params {
    cute::TmaDescriptor tma_desc_b_64;
    cute::TmaDescriptor tma_desc_b_32;
  };

  const Params& params_;

  MSCCLPP_DEVICE_INLINE W4A8Mainloop(const Params& params, ClusterShape cluster, uint32_t ctaRank)
      : Base(params, cluster, ctaRank), params_(params) {}

  __host__ __device__ static constexpr int mmaRows(int validRows) { return (validRows + 15) / 16 * 16; }

  static Params to_underlying_arguments(const ProblemShape& shape, const Arguments& args, void* workspace) {
    return Params{Base::to_underlying_arguments(shape, args, workspace),
                  SubtileActivationTma<64>::makeDescriptor(shape, args),
                  SubtileActivationTma<32>::makeDescriptor(shape, args)};
  }

  MSCCLPP_DEVICE_INLINE auto load_init(const ProblemShape& shape, TensorStorage& sharedTensors) {
    if (cute::elect_one_sync()) {
      this->prefetch_tma_descriptors();
      cute::prefetch_tma_descriptor(&params_.tma_desc_b_64);
      cute::prefetch_tma_descriptor(&params_.tma_desc_b_32);
    }
    return Base::load_init(shape, sharedTensors);
  }

  template <int Operands = 0, ActivationDtype ActDtype = ActivationDtype::Mxfp8, class Inputs, class Coord,
            class Iterator>
  MSCCLPP_DEVICE_INLINE auto load(MainloopPipeline pipe, MainloopPipelineState state, const Inputs& inputs, Coord coord,
                                  Iterator iterator, int kTiles, int validRows, bool weightEvictNormal) const {
    using namespace cute;
    const int m = get<0>(coord) / ClusterSize;
    const int n = get<1>(coord);
    const int expert = get<3>(coord);
    const int ctaIdx = int(this->block_rank_in_cluster_);
    auto a = inputs.tAgA_mkl(_, m, _, expert);
    auto originalB = inputs.tBgB_nkl(_, n, _, expert);
    // A two-CTA MMA splits activation rows at its runtime N/2, not the allocated tile's N/2.
    auto rowLayout = this->observed_tma_load_b_->get_tma_tensor(make_shape(1, 1, 1)).layout();
    auto rowOffset = rowLayout(make_coord(ctaIdx * (mmaRows(validRows) - TileN) / ClusterSize, 0, 0));
    auto b = make_tensor(originalB.data() + rowOffset, originalB.layout());
    auto sfa = inputs.tAgSFA_mkl(_, m, _, expert);
    auto sfb = inputs.tBgSFB_nkl(_, n, _, expert);

    const auto weightHint = weightEvictNormal ? CacheHint::EVICT_NORMAL : CacheHint::EVICT_FIRST;
    const cute::TmaDescriptor* bDesc = this->observed_tma_load_b_->get_tma_descriptor();
    uint32_t actTxBytes = ActivationTransactionBytes;
    if (validRows <= 32) {
      bDesc = &params_.tma_desc_b_32;
      actTxBytes = SubtileActivationTma<32>::TransactionBytes;
    } else if (validRows <= 64) {
      bDesc = &params_.tma_desc_b_64;
      actTxBytes = SubtileActivationTma<64>::TransactionBytes;
    }
    if constexpr (ActDtype == ActivationDtype::Fp8Static) actTxBytes -= SfbTransactionBytes;
    const uint32_t txBytes =
        Operands == 1 ? WeightTransactionBytes : (Operands == 2 ? actTxBytes : WeightTransactionBytes + actTxBytes);

    uint64_t waitNs = 0;
    uint64_t issueNs = 0;
    const int iterations = kTiles;
    for (; kTiles > 0; --kTiles, ++iterator) {
      auto start = traceKernelClock();
      pipe.producer_acquire(state);
      waitNs += traceKernelClock() - start;
      auto* barrier = pipe.producer_get_barrier(state);
      int stage = state.index();
      start = traceKernelClock();
      if (elect_one_sync()) {
        if (ctaIdx == 0) cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(barrier, txBytes);
        if constexpr (Operands != 2) {
          copy(this->observed_tma_load_a_->with(*barrier, inputs.mcast_mask_a, weightHint), a(_, *iterator),
               inputs.tAsA(_, stage));
          copy(this->observed_tma_load_sfa_->with(*barrier, inputs.mcast_mask_sfa, weightHint), sfa(_, *iterator),
               inputs.tAsSFA(_, stage));
        }
        if constexpr (Operands != 1) {
          copy(this->observed_tma_load_b_->with(bDesc, *barrier, inputs.mcast_mask_b, CacheHint::EVICT_LAST),
               b(_, *iterator), inputs.tBsB(_, stage));
          if constexpr (ActDtype == ActivationDtype::Mxfp8)
            copy(this->observed_tma_load_sfb_->with(*barrier, inputs.mcast_mask_sfb, CacheHint::EVICT_LAST),
                 sfb(_, *iterator), inputs.tBsSFB(_, stage));
        }
      }
      issueNs += traceKernelClock() - start;
      ++state;
    }
    traceKernelTotal(KernelTracePhase::LoadAcquire, waitNs, iterations);
    traceKernelTotal(KernelTracePhase::TmaIssue, issueNs, iterations);
    return make_tuple(state, iterator);
  }
};

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_W4A8_COLLECTIVE_CUH_
