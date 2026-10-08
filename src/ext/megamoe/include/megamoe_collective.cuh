// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_COLLECTIVE_CUH_
#define MSCCLPP_EXT_MEGAMOE_COLLECTIVE_CUH_

#include <cutlass/arch/reg_reconfig.h>

#include <cute/tensor.hpp>
#include <cutlass/detail/sm100_blockscaled_layout.hpp>
#include <cutlass/detail/sm100_mixed_dtype_blockwise_layout.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp>
#include <cutlass/gemm/collective/sm100_mma_warpspecialized_mixed_input.hpp>

#include "megamoe_specialization.hpp"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {
using ClusterShape = cute::Shape<cute::Int<ClusterM>, cute::_1, cute::_1>;
using ProblemShape = cute::Shape<int, int, int, int>;
}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail
#if MSCCLPP_MEGAMOE_COMPILE_W4A8
#include "megamoe_w4a8_collective.cuh"
#endif

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

using TileShape = cute::Shape<cute::Int<TileM>, cute::Int<TileN>, cute::Int<TileK>>;
using ScaleConfig = cutlass::detail::Sm100MixedInputBlockwiseScaleConfig<1, 32>;

template <bool E5M2, bool Local = false>
struct CollectiveTypes {
  using Tiles = TilePolicy<Local>;
  using KernelTile =
      cute::conditional_t<Local, cute::Shape<cute::Int<LocalTileM>, cute::Int<LocalTileN>, cute::Int<LocalTileK>>,
                          TileShape>;
  using Weight = cute::conditional_t<E5M2, cutlass::float_e5m2_t, cutlass::float_e4m3_t>;
  using Scale = cutlass::float_ue8m0_t;
  using Activation = cutlass::bfloat16_t;
  using Builder = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, cute::tuple<Weight, Scale>,
      cute::tuple<cutlass::layout::RowMajor, ScaleConfig::LayoutScale>, 16, Activation, cutlass::layout::ColumnMajor, 8,
      float, KernelTile, ClusterShape, cutlass::gemm::collective::StageCountAutoCarveout<65536>,
      cutlass::gemm::KernelTmaWarpSpecialized2SmMixedInputSm100>::CollectiveOp;
  using Policy =
      cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedMixedInput<Local ? 4 : LoadStages, Local ? 4 : TransformStages,
                                                                   2, 2, ClusterShape, cutlass::arch::Sm100>;
  using Mainloop = cutlass::gemm::collective::CollectiveMma<
      Policy, KernelTile, typename Builder::ElementAOptionalTuple,
      cute::tuple<typename Builder::StrideA, typename Builder::LayoutScale>, typename Builder::ElementBOptionalTuple,
      typename Builder::StrideB, typename Builder::TiledMma, typename Builder::GmemTiledCopyA,
      typename Builder::SmemLayoutAtomsA, typename Builder::CopyAtomsA, typename Builder::TransformA,
      typename Builder::GmemTiledCopyB, typename Builder::SmemLayoutAtomsB, typename Builder::CopyAtomsB,
      typename Builder::TransformB>;
  using LoadA = typename Mainloop::Load2TransformPipeline;
  using LoadB = typename Mainloop::Load2MmaPipeline;
  using Transform = typename Mainloop::Transform2MmaPipeline;
  using Accumulate = typename Mainloop::Mma2AccumPipeline;
};

#if MSCCLPP_MEGAMOE_COMPILE_W4A8
template <int TileN_, int TileK_, int LoadStages_, int NumWarps_, int TransferRegisters_, int LoadWarps_,
          bool SplitPipelines_, int EpilogueTokens_, int EpilogueWarps_, int EpilogueRegisters_, int DispatchChunk_,
          int DispatchWarps_, int DispatchStages_>
struct W4A8CollectiveTypesT {
  static constexpr int NumWarps = NumWarps_;
  static constexpr int EpilogueWarps = EpilogueWarps_;
  static constexpr int EpilogueRegisters = EpilogueRegisters_;
  static constexpr int TransferRegisters = TransferRegisters_;
  static constexpr int LoadWarps = LoadWarps_;
  static constexpr bool SplitPipelines = SplitPipelines_;
  static constexpr int EpilogueTokens = EpilogueTokens_;
  static constexpr int DispatchChunk = DispatchChunk_;
  static constexpr int DispatchWarps = DispatchWarps_;
  static constexpr int DispatchStages = DispatchStages_;
  using WarpSchedule = W4A8WarpScheduleT<NumWarps, EpilogueWarps, DispatchWarps>;
  using Tiles = TilePolicy<false, true, TileN_, TileK_>;
  using W4TileShape = cute::Shape<cute::Int<TileM>, cute::Int<(TileN_ < 64 ? 64 : TileN_)>, cute::Int<TileK_>>;
  using Weight = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
  using Activation = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  using Scale = cutlass::float_ue8m0_t;
  using Builder = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp, Weight, cutlass::layout::RowMajor, 128,
      Activation, cutlass::layout::ColumnMajor, 16, float, W4TileShape, ClusterShape,
      cutlass::gemm::collective::StageCount<LoadStages_>,
      cutlass::gemm::KernelTmaWarpSpecialized2SmMxf8f6f4Sm100>::CollectiveOp;
  using Mainloop = W4A8Mainloop<Builder, TileN_, TileK_, LoadStages_, SplitPipelines>;
  using Load = typename Mainloop::MainloopPipeline;
  using Accumulate = cutlass::PipelineUmmaAsync<2, typename Mainloop::AtomThrShapeMNK>;
  using ScaleConfig = typename Mainloop::Sm1xxBlkScaledConfig;
  static_assert(NumWarps == 12 || NumWarps == 16);
  static_assert(LoadWarps == 1 || LoadWarps == 2);
  static_assert(!SplitPipelines || LoadWarps == 2);
  static_assert(EpilogueTokens == 16 || EpilogueTokens == 32);
  static_assert(TileN_ % EpilogueTokens == 0);
  static_assert(EpilogueWarps == 4 || EpilogueWarps == 8);
  static_assert(EpilogueRegisters >= 128 && EpilogueRegisters <= 224 && EpilogueRegisters % 8 == 0);
  static_assert(TransferRegisters >= 32 && TransferRegisters <= 128 && TransferRegisters % 32 == 0);
  static_assert(DispatchChunk % TileK_ == 0);
  static_assert(DispatchWarps >= 2 && DispatchWarps <= 4);
  static_assert(DispatchStages == 1 || DispatchStages == 2);
  static_assert(WarpSchedule::DispatchEnd <= WarpSchedule::NumWarps);
};
using W4A8CollectiveTypes =
    W4A8CollectiveTypesT<W4TileN, W4TileK, W4LoadStages, W4NumWarps, W4TransferRegisters, W4LoadWarps, W4SplitPipelines,
                         W4EpilogueTokens, W4EpilogueWarps, W4EpilogueRegisters, W4DispatchChunk, W4DispatchWarps,
                         W4DispatchStages>;
static_assert(W4A8CollectiveTypes::Load::Stages == W4LoadStages);
#endif

#if __CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 3)
template <bool E5M2>
__device__ __forceinline__ uint32_t decodeScaledPair(uint16_t weights, uint16_t scales) {
  uint32_t decoded, expanded, result;
  if constexpr (E5M2) {
    asm("cvt.rn.bf16x2.e5m2x2 %0, %1;" : "=r"(decoded) : "h"(weights));
  } else {
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(decoded) : "h"(weights));
  }
  asm("cvt.rn.bf16x2.ue8m0x2 %0, %1;" : "=r"(expanded) : "h"(scales));
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(result) : "r"(decoded), "r"(expanded));
  return result;
}
#else
__device__ __forceinline__ uint16_t expandE8M0ToBf16(uint8_t scale) {
  return uint16_t(uint16_t(scale) << 7) | uint16_t(scale == 0) << 6 | uint16_t(scale == 255) * 0x7f;
}

template <bool E5M2>
__device__ __forceinline__ uint32_t decodeScaledPair(uint16_t weights, uint16_t scales) {
  using Weight = std::conditional_t<E5M2, cutlass::float_e5m2_t, cutlass::float_e4m3_t>;
  using PackedWeights = cutlass::Array<Weight, 2>;
  auto decoded =
      cutlass::NumericArrayConverter<cutlass::bfloat16_t, Weight, 2>{}(mscclpp::bit_cast<PackedWeights>(weights));
  uint32_t expanded = uint32_t(expandE8M0ToBf16(uint8_t(scales))) | uint32_t(expandE8M0ToBf16(uint8_t(scales >> 8)))
                                                                        << 16;
  uint32_t result;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(result) : "r"(mscclpp::bit_cast<uint32_t>(decoded)), "r"(expanded));
  return result;
}
#endif

template <bool E5M2, bool Local = false, class Inputs>
__device__ __forceinline__ void transformWeights(
    typename CollectiveTypes<E5M2, Local>::LoadA& load,
    typename CollectiveTypes<E5M2, Local>::LoadA::PipelineState& loadState,
    typename CollectiveTypes<E5M2, Local>::Transform& transformed,
    typename CollectiveTypes<E5M2, Local>::Transform::PipelineState& transformState, Inputs& inputs, int kTiles) {
  using namespace cute;
  using Types = CollectiveTypes<E5M2, Local>;
  using Mainloop = typename Types::Mainloop;
  using Utils = cutlass::gemm::collective::detail::MixedInputUtils<Mainloop>;
  auto copyOp = get<1>(inputs);
  auto& raw = get<2>(inputs);
  auto& converted = get<3>(inputs);
  auto& scaleInputs = get<4>(inputs);
  auto rRaw = make_tensor<typename Types::Weight>(shape(raw(_, _, _, _, 0)));
  auto rConverted = make_tensor<typename Types::Activation>(shape(rRaw));
  auto packedRaw = recast<uint16_t>(rRaw);
  auto packedConverted = recast<uint32_t>(rConverted);
  auto scaleBytes = recast<uint8_t>(get<1>(scaleInputs));
  static_assert(size(rRaw) == size(scaleBytes));
  cutlass::arch::NamedBarrier copyBarrier(128, cutlass::arch::ReservedNamedBarriers::TransformBarrier);

  for (int k = 0; k < kTiles; ++k) {
    load.consumer_wait(loadState);
    transformed.producer_acquire(transformState);
    copy(AutoVectorizingCopy{}, raw(_, _, _, _, loadState.index()), rRaw);
    Utils::copy_scale_zeros_for_transform(scaleInputs, loadState.index());
    copyBarrier.sync();
    load.consumer_release(loadState);
    ++loadState;

    // Keep E8M0 in its byte representation: the stock mixed-input helper only
    // multiplies scales when their element type already matches the MMA type.
    CUTE_UNROLL
    for (int i = 0; i < size(packedConverted); ++i) {
      uint16_t scales = uint16_t(scaleBytes(2 * i)) | (uint16_t(scaleBytes(2 * i + 1)) << 8);
      packedConverted(i) = decodeScaledPair<E5M2>(packedRaw(i), scales);
    }
    copy(copyOp, rConverted, converted(_, _, _, _, transformState.index()));
    cutlass::arch::fence_view_async_tmem_store();
    transformed.producer_commit(transformState);
    ++transformState;
  }
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_COLLECTIVE_CUH_
