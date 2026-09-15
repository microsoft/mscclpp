// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_SPECIALIZATION_HPP_
#define MSCCLPP_EXT_MEGAMOE_SPECIALIZATION_HPP_

#ifndef MSCCLPP_MEGAMOE_TILE_N
#define MSCCLPP_MEGAMOE_TILE_N 32
#endif
#ifndef MSCCLPP_MEGAMOE_LOAD_STAGES
#define MSCCLPP_MEGAMOE_LOAD_STAGES 8
#endif
#ifndef MSCCLPP_MEGAMOE_TRANSFORM_STAGES
#define MSCCLPP_MEGAMOE_TRANSFORM_STAGES 7
#endif

#if defined(MSCCLPP_MEGAMOE_JIT_MODULE) && MSCCLPP_MEGAMOE_JIT_MODULE
#define MSCCLPP_MEGAMOE_KERNEL_NAMESPACE mscclpp::megamoe::jit
#else
#define MSCCLPP_MEGAMOE_KERNEL_NAMESPACE mscclpp::megamoe
#endif

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

constexpr int TileN = MSCCLPP_MEGAMOE_TILE_N;
constexpr int LoadStages = MSCCLPP_MEGAMOE_LOAD_STAGES;
constexpr int TransformStages = MSCCLPP_MEGAMOE_TRANSFORM_STAGES;
static_assert(TileN > 0 && TileN % 32 == 0, "MegaMoE routed N must be a positive multiple of 32");
static_assert(LoadStages > 0 && TransformStages > 0, "MegaMoE pipeline depths must be positive");
static_assert(2 * TileN + 64 * TransformStages <= 512, "MegaMoE specialization exceeds 512 TMEM columns");
static_assert((TileN == 32 && LoadStages == 8 && TransformStages == 7) ||
                  (TileN == 32 && LoadStages == 6 && TransformStages == 7) ||
                  (TileN == 64 && LoadStages == 6 && TransformStages == 6) ||
                  (TileN == 128 && LoadStages == 4 && TransformStages == 4),
              "Unsupported MegaMoE routed specialization");

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_SPECIALIZATION_HPP_
