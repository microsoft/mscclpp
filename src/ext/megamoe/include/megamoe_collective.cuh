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
#include <mscclpp/device.hpp>

#include "megamoe_kernel.hpp"
#include "megamoe_specialization.hpp"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {
using ProblemShape = cute::Shape<int, int, int, int>;
}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#include "megamoe_w4a8_collective.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

using W8ScaleConfig = cutlass::detail::Sm100MixedInputBlockwiseScaleConfig<1, 32>;

template <bool Local = false>
struct W8A16CollectiveTypesT {
  static constexpr int NumWarps = Local ? 12 : 16;
  static constexpr int EpilogueWarps = 4;
  static constexpr int EpilogueRegisters = Local ? 232 : 224;
  static constexpr int TransferRegisters = 32;
  static constexpr int LoadWarps = 2;
  static constexpr bool SplitPipelines = false;
  static constexpr int EpilogueTokens = 32;
  static constexpr int DispatchChunk = W4DispatchChunk;
  static constexpr int DispatchWarps = 4;
  static constexpr int DispatchStages = 2;
  static constexpr bool WeightMxfp4 = false;
  static constexpr bool LocalExpert = Local;
  using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
  using WarpSchedule = W8A16WarpSchedule<Local>;
  using Tiles = TilePolicy<128, 128>;
  using KernelTile = cute::Shape<cute::_256, cute::_128, cute::_128>;
  using Weight = cutlass::float_e4m3_t;
  using Scale = cutlass::float_ue8m0_t;
  using Activation = cutlass::bfloat16_t;
  using Builder = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, cute::tuple<Weight, Scale>,
      cute::tuple<cutlass::layout::RowMajor, W8ScaleConfig::LayoutScale>, 16, Activation, cutlass::layout::ColumnMajor,
      8, float, KernelTile, ClusterShape, cutlass::gemm::collective::StageCountAutoCarveout<65536>,
      cutlass::gemm::KernelTmaWarpSpecialized2SmMixedInputSm100>::CollectiveOp;
  using Policy = cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedMixedInput<W8LoadStages, W8TransformStages, 2, 2,
                                                                              ClusterShape, cutlass::arch::Sm100>;
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
  static constexpr bool WeightMxfp4 = true;
  static constexpr bool LocalExpert = false;
  using ClusterShape = cute::Shape<cute::Int<W4ClusterSize>, cute::_1, cute::_1>;
  using WarpSchedule = W4A8WarpScheduleT<NumWarps, EpilogueWarps, DispatchWarps>;
  using Tiles = TilePolicy<TileN_, TileK_>;
  using W4TileShape = cute::Shape<cute::Int<W4TileM>, cute::Int<(TileN_ < 64 ? 64 : TileN_)>, cute::Int<TileK_>>;
  using Weight = cutlass::mx_float4_t<cutlass::float_e2m1_t>;
  using Activation = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
  using Scale = cutlass::float_ue8m0_t;
  using KernelSchedule =
      cute::conditional_t<W4ClusterSize == 1, cutlass::gemm::KernelTmaWarpSpecialized1SmMxf8f6f4Sm100,
                          cutlass::gemm::KernelTmaWarpSpecialized2SmMxf8f6f4Sm100>;
  using Builder = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp, Weight, cutlass::layout::RowMajor, 128,
      Activation, cutlass::layout::ColumnMajor, 16, float, W4TileShape, ClusterShape,
      cutlass::gemm::collective::StageCount<LoadStages_>, KernelSchedule>::CollectiveOp;
  using Policy = cutlass::gemm::MainloopSm100TmaUmmaWarpSpecializedBlockScaled<LoadStages_, 2, 3, ClusterShape,
                                                                               cutlass::arch::Sm100>;
  using CollectiveBase = cutlass::gemm::collective::CollectiveMma<
      Policy, W4TileShape, typename Builder::ElementPairA, typename Builder::StridePairA,
      typename Builder::ElementPairB, typename Builder::StridePairB, typename Builder::TiledMma,
      typename Builder::GmemTiledCopyPairA, typename Builder::SmemLayoutAtomPairA, typename Builder::SmemCopyAtomA,
      typename Builder::TransformA, typename Builder::GmemTiledCopyPairB, typename Builder::SmemLayoutAtomPairB,
      typename Builder::SmemCopyAtomB, typename Builder::TransformB>;
  using Mainloop = W4A8Mainloop<CollectiveBase, TileN_, TileK_, LoadStages_, SplitPipelines, W4ClusterSize>;
  using Load = typename Mainloop::MainloopPipeline;
  using Accumulate = cutlass::PipelineUmmaAsync<3, typename Mainloop::AtomThrShapeMNK>;
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
  static_assert(DispatchWarps == 2 || DispatchWarps == 4 || DispatchWarps == 8);
  static_assert(DispatchStages == 1 || DispatchStages == 2);
  static_assert(WarpSchedule::DispatchEnd <= WarpSchedule::NumWarps);
};

using W4A8CollectiveTypes =
    W4A8CollectiveTypesT<W4TileN, W4TileK, W4LoadStages, W4NumWarps, W4TransferRegisters, W4LoadWarps, W4SplitPipelines,
                         W4EpilogueTokens, W4EpilogueWarps, W4EpilogueRegisters, W4DispatchChunk, W4DispatchWarps,
                         W4DispatchStages>;

static_assert(W4A8CollectiveTypes::Load::Stages == W4LoadStages);

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_COLLECTIVE_CUH_
