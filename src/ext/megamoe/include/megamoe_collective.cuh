// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_COLLECTIVE_CUH_
#define MSCCLPP_EXT_MEGAMOE_COLLECTIVE_CUH_

#include <cutlass/arch/reg_reconfig.h>

#include <cute/tensor.hpp>
#include <cutlass/detail/sm100_mixed_dtype_blockwise_layout.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/sm100_mma_warpspecialized_mixed_input.hpp>

namespace mscclpp::megamoe::detail {

using TileShape = cute::Shape<cute::_256, cute::_128, cute::_128>;
using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
using ProblemShape = cute::Shape<int, int, int, int>;
using ScaleConfig = cutlass::detail::Sm100MixedInputBlockwiseScaleConfig<1, 32>;

template <bool E5M2>
struct CollectiveTypes {
  using Weight = cute::conditional_t<E5M2, cutlass::float_e5m2_t, cutlass::float_e4m3_t>;
  using Scale = cutlass::float_ue8m0_t;
  using Activation = cutlass::bfloat16_t;
  using Builder = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, cute::tuple<Weight, Scale>,
      cute::tuple<cutlass::layout::RowMajor, ScaleConfig::LayoutScale>, 16, Activation, cutlass::layout::ColumnMajor, 8,
      float, TileShape, ClusterShape, cutlass::gemm::collective::StageCountAutoCarveout<65536>,
      cutlass::gemm::KernelTmaWarpSpecialized2SmMixedInputSm100>::CollectiveOp;
  using Policy =
      cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedMixedInput<4, 4, 2, 2, ClusterShape, cutlass::arch::Sm100>;
  using Mainloop = cutlass::gemm::collective::CollectiveMma<
      Policy, TileShape, typename Builder::ElementAOptionalTuple,
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

template <bool E5M2, class Inputs>
__device__ __forceinline__ void transformWeights(
    typename CollectiveTypes<E5M2>::LoadA& load, typename CollectiveTypes<E5M2>::LoadA::PipelineState& loadState,
    typename CollectiveTypes<E5M2>::Transform& transformed,
    typename CollectiveTypes<E5M2>::Transform::PipelineState& transformState, Inputs& inputs, int kTiles) {
  using namespace cute;
  using Types = CollectiveTypes<E5M2>;
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

}  // namespace mscclpp::megamoe::detail

#endif  // MSCCLPP_EXT_MEGAMOE_COLLECTIVE_CUH_
