// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// The sub-tile TMA and pipeline extensions adapt CUTLASS's SM100 mixed-input collective:
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
//    may be used to endorse or promote products derived from the distribution
//    without specific prior written permission.
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

#ifndef MSCCLPP_EXT_MEGAMOE_W8A16_COLLECTIVE_CUH_
#define MSCCLPP_EXT_MEGAMOE_W8A16_COLLECTIVE_CUH_

#include <cute/tensor.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

// Extends CUTLASS's SM100 mixed-input CollectiveMma (`Base`) with sub-tile activation TMA descriptors (N=64, N=32),
// 2-CTA runtime row shifting, and adaptive L2 cache hints for weights and activations.
template <class Base, int TileN, int TileK, int LoadStages>
struct W8A16Mainloop : Base {
  static_assert(TileN == 128);

  using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
  using typename Base::Arguments;
  using typename Base::AtomThrShapeMNK;
  using typename Base::ElementA;
  using typename Base::ElementAMma;
  using typename Base::ElementB;
  using typename Base::GmemTiledCopyB;
  using typename Base::Load2MmaPipeline;
  using typename Base::Load2MmaPipelineState;
  using typename Base::Load2TransformPipeline;
  using typename Base::Load2TransformPipelineState;
  using typename Base::SmemLayoutAtomB;
  using typename Base::TensorStorage;
  using typename Base::TiledMma;

  template <int SubTileN>
  struct SubtileActivationTma {
    using SubTileShape = cute::Shape<cute::_256, cute::Int<SubTileN>, cute::Int<TileK>>;
    using SubTiledMma = decltype(cutlass::gemm::collective::detail::sm100_make_trivial_mixed_input_tiled_mma<
                                 ElementAMma, ElementB, float, SubTileShape, ClusterShape, cute::UMMA::Major::K,
                                 cute::UMMA::Major::K, cutlass::gemm::KernelTmaWarpSpecialized2SmMixedInputSm100>());
    using SubSmemLayoutB = decltype(cute::UMMA::tile_to_mma_shape(
        SmemLayoutAtomB{},
        cute::append(
            cute::partition_shape_B(SubTiledMma{}, cute::make_shape(cute::Int<SubTileN>{}, cute::Int<TileK>{})),
            cute::Int<LoadStages>{}),
        cute::Step<cute::_1, cute::_2, cute::_3>{}));
    static constexpr uint32_t TransactionBytes =
        cutlass::bits_to_bytes(cute::size(AtomThrShapeMNK{}) * cute::cosize(cute::take<0, 3>(SubSmemLayoutB{})) *
                               cute::sizeof_bits_v<ElementB>);

    static cute::TmaDescriptor makeDescriptor(const ProblemShape& shape, const Arguments& args) {
      using namespace cute;
      auto [m, n, k, l] = shape;
      (void)m;
      auto tensorB = make_tensor(args.ptr_B, make_layout(make_shape(n, k, l), args.dB));
      auto cluster = tiled_divide(make_layout(ClusterShape{}), make_tile(typename TiledMma::AtomThrID{}));
      auto tmaB = make_tma_atom_B_sm100<ElementB>(GmemTiledCopyB{}, tensorB, SubSmemLayoutB{}(_, _, _, _0{}),
                                                  SubTileShape{}, SubTiledMma{}, cluster);
      return *tmaB.get_tma_descriptor();
    }
  };

  struct Params : Base::Params {
    cute::TmaDescriptor tma_desc_b_64;
    cute::TmaDescriptor tma_desc_b_32;
  };

  const Params& params_;
  uint32_t ctaRank_;

  MSCCLPP_DEVICE_INLINE W8A16Mainloop(const Params& params, ClusterShape cluster, uint32_t ctaRank)
      : Base(params, cluster, ctaRank), params_(params), ctaRank_(ctaRank) {}

  __host__ __device__ static constexpr int mmaRows(int validRows) { return (validRows + 15) / 16 * 16; }

  static Params to_underlying_arguments(const ProblemShape& shape, const Arguments& args, void* workspace) {
    return Params{Base::to_underlying_arguments(shape, args, workspace),
                  SubtileActivationTma<64>::makeDescriptor(shape, args),
                  SubtileActivationTma<32>::makeDescriptor(shape, args)};
  }

  MSCCLPP_DEVICE_INLINE auto load_init(const ProblemShape& shape, const Params& params,
                                       TensorStorage& sharedTensors) const {
    if (threadIdx.x % WARP_SIZE == 0) {
      Base::prefetch_tma_descriptors(params);
      cute::prefetch_tma_descriptor(&params.tma_desc_b_64);
      cute::prefetch_tma_descriptor(&params.tma_desc_b_32);
    }
    return Base::load_init(shape, params, sharedTensors);
  }

  template <class LoadInputs, class Coord, class Iterator>
  MSCCLPP_DEVICE_INLINE auto load_A(const Params& params, Load2TransformPipeline pipe,
                                    Load2TransformPipelineState state, const LoadInputs& inputs, const Coord& coord,
                                    Iterator kTileIter, int kTileCount, bool weightEvictNormal) const {
    using namespace cute;
    auto& tAgA_mkl = get<2>(inputs);
    auto& tAsA = get<4>(inputs);
    const uint16_t mcastMaskA = get<6>(inputs);
    auto& extraPartitions = get<8>(inputs);
    auto& tSgS_mkl = get<0>(extraPartitions);
    auto& tSsS = get<1>(extraPartitions);

    const int mTile = get<0>(coord) / size(typename TiledMma::AtomThrID{});
    const int expert = get<3>(coord);
    auto tAgA = tAgA_mkl(_, mTile, _, expert);
    auto tSgS = tSgS_mkl(_, mTile, _, expert);

    const auto weightHint =
        weightEvictNormal ? cute::TMA::CacheHintSm90::EVICT_NORMAL : cute::TMA::CacheHintSm90::EVICT_FIRST;
    uint32_t skipWait = (kTileCount <= 0);
    auto pipeFlag = pipe.producer_try_acquire(state, skipWait);

    CUTLASS_PRAGMA_NO_UNROLL
    for (; kTileCount > 0; --kTileCount) {
      pipe.producer_acquire(state, pipeFlag);
      const int writeStage = state.index();
      auto* barrier = pipe.producer_get_barrier(state);
      ++state;
      skipWait = (kTileCount <= 1);
      pipeFlag = pipe.producer_try_acquire(state, skipWait);

      copy(params.tma_load_a.with(*barrier, mcastMaskA, weightHint), tAgA(_, *kTileIter), tAsA(_, writeStage));
      copy(params.tma_load_scale.with(*barrier, mcastMaskA, weightHint), tSgS(_, *kTileIter), tSsS(_, writeStage));
      ++kTileIter;
    }
    return cute::make_tuple(state, kTileIter);
  }

  template <class LoadInputs, class Coord, class Iterator>
  MSCCLPP_DEVICE_INLINE auto load_B(const Params& params, Load2MmaPipeline pipe, Load2MmaPipelineState state,
                                    const LoadInputs& inputs, const Coord& coord, Iterator kTileIter, int kTileCount,
                                    int validRows) const {
    using namespace cute;
    auto& tBgB_nkl = get<3>(inputs);
    auto& tBsB = get<5>(inputs);
    const uint16_t mcastMaskB = get<7>(inputs);

    const int ctaIdx = int(ctaRank_);
    auto originalB = tBgB_nkl(_, get<1>(coord), _, get<3>(coord));
    // A two-CTA MMA splits activation rows at its runtime N/2, not the allocated tile's N/2.
    auto rowLayout = params.tma_load_b.get_tma_tensor(make_shape(1, 1, 1)).layout();
    auto rowOffset = rowLayout(make_coord(ctaIdx * (mmaRows(validRows) - TileN) / 2, 0, 0));
    auto tBgB = make_tensor(originalB.data() + rowOffset, originalB.layout());

    const cute::TmaDescriptor* bDesc = params.tma_load_b.get_tma_descriptor();
    uint32_t actTxBytes = Base::TmaTransactionBytes_B;
    if (validRows <= 32) {
      bDesc = &params.tma_desc_b_32;
      actTxBytes = SubtileActivationTma<32>::TransactionBytes;
    } else if (validRows <= 64) {
      bDesc = &params.tma_desc_b_64;
      actTxBytes = SubtileActivationTma<64>::TransactionBytes;
    }

    uint32_t skipWait = (kTileCount <= 0);
    auto pipeFlag = pipe.producer_try_acquire(state, skipWait);

    CUTLASS_PRAGMA_NO_UNROLL
    for (; kTileCount > 0; --kTileCount) {
      pipe.producer_acquire(state, pipeFlag);
      const int writeStage = state.index();
      auto* barrier = pipe.producer_get_barrier(state);
      if (ctaIdx == 0) cutlass::arch::ClusterTransactionBarrier::arrive_and_expect_tx(barrier, actTxBytes);
      ++state;
      skipWait = (kTileCount <= 1);
      pipeFlag = pipe.producer_try_acquire(state, skipWait);

      copy(params.tma_load_b.with(bDesc, *barrier, mcastMaskB), tBgB(_, *kTileIter), tBsB(_, writeStage));
      ++kTileIter;
    }
    return cute::make_tuple(state, kTileIter);
  }
};

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_W8A16_COLLECTIVE_CUH_
