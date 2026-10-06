// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_SPECIALIZATION_HPP_
#define MSCCLPP_EXT_MEGAMOE_SPECIALIZATION_HPP_

#ifndef MSCCLPP_MEGAMOE_TILE_N
#define MSCCLPP_MEGAMOE_TILE_N 32
#endif
#ifndef MSCCLPP_MEGAMOE_TILE_M
#define MSCCLPP_MEGAMOE_TILE_M 256
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
#ifndef MSCCLPP_MEGAMOE_W4_TILE_N
#define MSCCLPP_MEGAMOE_W4_TILE_N 32
#endif
#ifndef MSCCLPP_MEGAMOE_W4_TILE_K
#define MSCCLPP_MEGAMOE_W4_TILE_K 256
#endif
#ifndef MSCCLPP_MEGAMOE_W4_LOAD_STAGES
#define MSCCLPP_MEGAMOE_W4_LOAD_STAGES 5
#endif
#ifndef MSCCLPP_MEGAMOE_W4_NUM_WARPS
#define MSCCLPP_MEGAMOE_W4_NUM_WARPS 16
#endif
#ifndef MSCCLPP_MEGAMOE_W4_TRANSFER_REGISTERS
#define MSCCLPP_MEGAMOE_W4_TRANSFER_REGISTERS 128
#endif
#ifndef MSCCLPP_MEGAMOE_W4_LOAD_WARPS
#define MSCCLPP_MEGAMOE_W4_LOAD_WARPS 2
#endif
#ifndef MSCCLPP_MEGAMOE_W4_SPLIT_PIPELINES
#define MSCCLPP_MEGAMOE_W4_SPLIT_PIPELINES 0
#endif
#ifndef MSCCLPP_MEGAMOE_W4_EPILOGUE_TOKENS
#define MSCCLPP_MEGAMOE_W4_EPILOGUE_TOKENS 32
#endif
#ifndef MSCCLPP_MEGAMOE_W4_DISPATCH_CHUNK
#define MSCCLPP_MEGAMOE_W4_DISPATCH_CHUNK 8192
#endif
#ifndef MSCCLPP_MEGAMOE_W4_DISPATCH_WARPS
#define MSCCLPP_MEGAMOE_W4_DISPATCH_WARPS 1
#endif

#if defined(MSCCLPP_MEGAMOE_JIT_MODULE) && MSCCLPP_MEGAMOE_JIT_MODULE
#define MSCCLPP_MEGAMOE_KERNEL_NAMESPACE mscclpp::megamoe::jit
#else
#define MSCCLPP_MEGAMOE_KERNEL_NAMESPACE mscclpp::megamoe
#endif

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

// Performance-tunable tile and pipeline policy.
constexpr int ClusterM = 2;
constexpr int TileM = MSCCLPP_MEGAMOE_TILE_M;
constexpr int TileN = MSCCLPP_MEGAMOE_TILE_N;
constexpr int TileK = MSCCLPP_MEGAMOE_TILE_K;
constexpr int LoadStages = MSCCLPP_MEGAMOE_LOAD_STAGES;
constexpr int TransformStages = MSCCLPP_MEGAMOE_TRANSFORM_STAGES;
constexpr int W4TileN = MSCCLPP_MEGAMOE_W4_TILE_N;
constexpr int W4TileK = MSCCLPP_MEGAMOE_W4_TILE_K;
constexpr int W4LoadStages = MSCCLPP_MEGAMOE_W4_LOAD_STAGES;
constexpr int W4NumWarps = MSCCLPP_MEGAMOE_W4_NUM_WARPS;
constexpr int W4TransferRegisters = MSCCLPP_MEGAMOE_W4_TRANSFER_REGISTERS;
constexpr int W4LoadWarps = MSCCLPP_MEGAMOE_W4_LOAD_WARPS;
constexpr bool W4SplitPipelines = MSCCLPP_MEGAMOE_W4_SPLIT_PIPELINES != 0;
constexpr int W4EpilogueTokens = MSCCLPP_MEGAMOE_W4_EPILOGUE_TOKENS;
constexpr int W4DispatchChunk = MSCCLPP_MEGAMOE_W4_DISPATCH_CHUNK;
constexpr int W4DispatchWarps = MSCCLPP_MEGAMOE_W4_DISPATCH_WARPS;
constexpr int W4TokenStride = W4TileN < 64 ? 64 : W4TileN;
static_assert(W4TileN == 32 || W4TileN == 64 || W4TileN == 128);
static_assert(W4TileK == 128 || W4TileK == 256 || W4TileK == 512);
static_assert(W4LoadStages >= 2 && W4LoadStages <= 10);
static_assert(W4NumWarps == 12 || W4NumWarps == 16);
static_assert(W4TransferRegisters == 32 || W4TransferRegisters == 64 || W4TransferRegisters == 96 ||
              W4TransferRegisters == 128);
static_assert(W4LoadWarps == 1 || W4LoadWarps == 2);
static_assert(!W4SplitPipelines || W4LoadWarps == 2);
static_assert(W4EpilogueTokens == 16 || W4EpilogueTokens == 32);
static_assert(W4DispatchChunk == 512 || W4DispatchChunk == 1024 || W4DispatchChunk == 2048 ||
              W4DispatchChunk == 4096 || W4DispatchChunk == 8192);
static_assert(W4DispatchChunk % W4TileK == 0);
static_assert(W4DispatchWarps >= 1 && W4DispatchWarps <= 4);
constexpr int LocalTileM = 256;
constexpr int LocalTileN = 128;
constexpr int LocalTileK = 128;
static_assert(TileM == 128 || TileM == 256, "MegaMoE routed M must be 128 or 256");
static_assert(TileN > 0 && TileN % 32 == 0, "MegaMoE routed N must be a positive multiple of 32");
static_assert(TileK == 32 || TileK == 64 || TileK == 128, "MegaMoE routed K must be 32, 64, or 128");
static_assert(TileM == 256 || TileK >= 64, "MegaMoE M128 requires K64 or K128 for 128B scale transactions");
static_assert(TileM % (ClusterM * 32) == 0, "MegaMoE per-CTA M must preserve gate/up K32 packing");
static_assert(LoadStages > 0 && TransformStages > 0, "MegaMoE pipeline depths must be positive");
static_assert(2 * TileN + TileK / 2 * TransformStages <= 512, "MegaMoE specialization exceeds 512 TMEM columns");
static_assert((TileN == 32 && LoadStages == 8 && TransformStages == 7) ||
                  (TileN == 32 && LoadStages == 6 && TransformStages == 7) ||
                  (TileN == 64 && LoadStages == 6 && TransformStages == 6) ||
                  (TileN == 128 && LoadStages == 4 && TransformStages == 4),
              "Unsupported MegaMoE routed specialization");

template <bool Local, bool Mxfp4 = false>
struct TilePolicy {
  static constexpr int M = Local ? LocalTileM : TileM;
  static constexpr int N = Local ? LocalTileN : (Mxfp4 ? W4TileN : TileN);
  static constexpr int K = Local ? LocalTileK : (Mxfp4 ? W4TileK : TileK);
  static constexpr int CtaM = M / ClusterM;
  static constexpr int Fc1M = M / 2;
  static constexpr int CtaFc1M = CtaM / 2;
};

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

#if !defined(MSCCLPP_MEGAMOE_JIT_MODULE) || !MSCCLPP_MEGAMOE_JIT_MODULE
struct W4A8WarpSchedule {
  static constexpr int NumWarps = W4NumWarps;
  static constexpr int EpilogueBegin = 0;
  static constexpr int EpilogueEnd = 4;
  static constexpr int MmaWarp = 4;
  static constexpr int LoadWarp = 5;
  static constexpr int ActivationLoadWarp = 6;
  static constexpr int DispatchBegin = 8;
  static constexpr int DispatchEnd = DispatchBegin + W4DispatchWarps;
  static constexpr bool HasDispatch = true;
};
#endif

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
