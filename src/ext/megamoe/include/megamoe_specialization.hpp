// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_SPECIALIZATION_HPP_
#define MSCCLPP_EXT_MEGAMOE_SPECIALIZATION_HPP_

#ifndef MSCCLPP_MEGAMOE_TILE_N
#define MSCCLPP_MEGAMOE_TILE_N 32
#endif
#ifndef MSCCLPP_MEGAMOE_TILE_K
#define MSCCLPP_MEGAMOE_TILE_K 128
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

// Performance-tunable tile and pipeline policy.
constexpr int TileM = 256;
constexpr int TileN = MSCCLPP_MEGAMOE_TILE_N;
constexpr int TileK = MSCCLPP_MEGAMOE_TILE_K;
constexpr int LoadStages = MSCCLPP_MEGAMOE_LOAD_STAGES;
constexpr int TransformStages = MSCCLPP_MEGAMOE_TRANSFORM_STAGES;
constexpr int LocalTileM = 256;
constexpr int LocalTileN = 128;
constexpr int LocalTileK = 128;
static_assert(TileN > 0 && TileN % 32 == 0, "MegaMoE routed N must be a positive multiple of 32");
static_assert(TileK == 32 || TileK == 64 || TileK == 128, "MegaMoE routed K must be 32, 64, or 128");
static_assert(LoadStages > 0 && TransformStages > 0, "MegaMoE pipeline depths must be positive");
static_assert(2 * TileN + TileK / 2 * TransformStages <= 512, "MegaMoE specialization exceeds 512 TMEM columns");
static_assert((TileN == 32 && LoadStages == 8 && TransformStages == 7) ||
                  (TileN == 32 && LoadStages == 6 && TransformStages == 7) ||
                  (TileN == 64 && LoadStages == 6 && TransformStages == 6) ||
                  (TileN == 128 && LoadStages == 4 && TransformStages == 4),
              "Unsupported MegaMoE routed specialization");

// Performance-tunable warp assignments. The role implementations and pipeline
// types are independent of these physical warp IDs.
struct RoutedWarpSchedule {
  static constexpr int NumWarps = 16;
  static constexpr int EpilogueBegin = 0;
  static constexpr int EpilogueEnd = 4;
  static constexpr int MmaWarp = 4;
  static constexpr int LoadAWarp = 5;
  static constexpr int LoadBWarp = 6;
  static constexpr int DispatchBegin = 8;
  static constexpr int DispatchEnd = 12;
  static constexpr int TransformBegin = 12;
  static constexpr int TransformEnd = 16;
  static constexpr bool HasDispatch = true;
};

struct LocalWarpSchedule {
  static constexpr int NumWarps = 12;
  static constexpr int EpilogueBegin = 0;
  static constexpr int EpilogueEnd = 4;
  static constexpr int MmaWarp = 4;
  static constexpr int LoadAWarp = 5;
  static constexpr int LoadBWarp = 6;
  static constexpr int DispatchBegin = 0;
  static constexpr int DispatchEnd = 0;
  static constexpr int TransformBegin = 8;
  static constexpr int TransformEnd = 12;
  static constexpr bool HasDispatch = false;
};

template <bool Local>
struct WarpSchedule;

template <>
struct WarpSchedule<false> : RoutedWarpSchedule {};

template <>
struct WarpSchedule<true> : LocalWarpSchedule {};

constexpr bool warpInRange(int warp, int begin, int end) { return warp >= begin && warp < end; }

constexpr bool warpRangesOverlap(int firstBegin, int firstEnd, int secondBegin, int secondEnd) {
  return firstBegin < secondEnd && secondBegin < firstEnd;
}

template <class Schedule>
constexpr bool validWarpSchedule() {
  constexpr bool fixedGroups =
      Schedule::NumWarps > 0 && Schedule::NumWarps % 4 == 0 && Schedule::EpilogueBegin == 0 &&
      Schedule::EpilogueEnd - Schedule::EpilogueBegin == 4 && Schedule::EpilogueEnd <= Schedule::NumWarps &&
      Schedule::TransformBegin >= 0 && Schedule::TransformBegin % 4 == 0 &&
      Schedule::TransformEnd - Schedule::TransformBegin == 4 && Schedule::TransformEnd <= Schedule::NumWarps &&
      !warpRangesOverlap(Schedule::EpilogueBegin, Schedule::EpilogueEnd, Schedule::TransformBegin,
                         Schedule::TransformEnd);
  constexpr bool singletonRoles =
      warpInRange(Schedule::MmaWarp, 0, Schedule::NumWarps) &&
      warpInRange(Schedule::LoadAWarp, 0, Schedule::NumWarps) &&
      warpInRange(Schedule::LoadBWarp, 0, Schedule::NumWarps) && Schedule::MmaWarp != Schedule::LoadAWarp &&
      Schedule::MmaWarp != Schedule::LoadBWarp && Schedule::LoadAWarp != Schedule::LoadBWarp;
  constexpr bool singletonRolesAreTransfer =
      !warpInRange(Schedule::MmaWarp, Schedule::EpilogueBegin, Schedule::EpilogueEnd) &&
      !warpInRange(Schedule::MmaWarp, Schedule::TransformBegin, Schedule::TransformEnd) &&
      !warpInRange(Schedule::LoadAWarp, Schedule::EpilogueBegin, Schedule::EpilogueEnd) &&
      !warpInRange(Schedule::LoadAWarp, Schedule::TransformBegin, Schedule::TransformEnd) &&
      !warpInRange(Schedule::LoadBWarp, Schedule::EpilogueBegin, Schedule::EpilogueEnd) &&
      !warpInRange(Schedule::LoadBWarp, Schedule::TransformBegin, Schedule::TransformEnd);
  constexpr bool dispatch =
      Schedule::HasDispatch
          ? Schedule::DispatchBegin % 4 == 0 && Schedule::DispatchEnd - Schedule::DispatchBegin == 4 &&
                Schedule::DispatchBegin >= 0 && Schedule::DispatchEnd <= Schedule::NumWarps &&
                !warpRangesOverlap(Schedule::DispatchBegin, Schedule::DispatchEnd, Schedule::EpilogueBegin,
                                   Schedule::EpilogueEnd) &&
                !warpRangesOverlap(Schedule::DispatchBegin, Schedule::DispatchEnd, Schedule::TransformBegin,
                                   Schedule::TransformEnd) &&
                !warpInRange(Schedule::MmaWarp, Schedule::DispatchBegin, Schedule::DispatchEnd) &&
                !warpInRange(Schedule::LoadAWarp, Schedule::DispatchBegin, Schedule::DispatchEnd) &&
                !warpInRange(Schedule::LoadBWarp, Schedule::DispatchBegin, Schedule::DispatchEnd)
          : Schedule::DispatchBegin == Schedule::DispatchEnd;
  return fixedGroups && singletonRoles && singletonRolesAreTransfer && dispatch;
}

static_assert(validWarpSchedule<RoutedWarpSchedule>(), "Invalid routed MegaMoE warp schedule");
static_assert(validWarpSchedule<LocalWarpSchedule>(), "Invalid local MegaMoE warp schedule");

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_SPECIALIZATION_HPP_
