// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_SPECIALIZATION_HPP_
#define MSCCLPP_EXT_MEGAMOE_SPECIALIZATION_HPP_

#ifndef MSCCLPP_MEGAMOE_W4_TILE_N
#define MSCCLPP_MEGAMOE_W4_TILE_N 128
#endif
#ifndef MSCCLPP_MEGAMOE_W4_TILE_K
#define MSCCLPP_MEGAMOE_W4_TILE_K 128
#endif
#ifndef MSCCLPP_MEGAMOE_W4_LOAD_STAGES
#define MSCCLPP_MEGAMOE_W4_LOAD_STAGES 8
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
#ifndef MSCCLPP_MEGAMOE_W4_EPILOGUE_WARPS
#define MSCCLPP_MEGAMOE_W4_EPILOGUE_WARPS 4
#endif
#ifndef MSCCLPP_MEGAMOE_W4_EPILOGUE_REGISTERS
#define MSCCLPP_MEGAMOE_W4_EPILOGUE_REGISTERS 224
#endif
#ifndef MSCCLPP_MEGAMOE_W4_DISPATCH_CHUNK
#define MSCCLPP_MEGAMOE_W4_DISPATCH_CHUNK 4096
#endif
#ifndef MSCCLPP_MEGAMOE_W4_DISPATCH_WARPS
#define MSCCLPP_MEGAMOE_W4_DISPATCH_WARPS 4
#endif
#ifndef MSCCLPP_MEGAMOE_W4_DISPATCH_STAGES
#define MSCCLPP_MEGAMOE_W4_DISPATCH_STAGES 1
#endif
#ifndef MSCCLPP_MEGAMOE_W4_FIXED_TOKEN_COUNT
#define MSCCLPP_MEGAMOE_W4_FIXED_TOKEN_COUNT 0
#endif
#ifndef MSCCLPP_MEGAMOE_W4_EXPERT_MAJOR_TASKS
#define MSCCLPP_MEGAMOE_W4_EXPERT_MAJOR_TASKS 0
#endif
#ifndef MSCCLPP_MEGAMOE_W4_CACHE_WEIGHTS
#define MSCCLPP_MEGAMOE_W4_CACHE_WEIGHTS 0
#endif
#ifndef MSCCLPP_MEGAMOE_W4_ADAPTIVE_CACHE_WEIGHTS
#define MSCCLPP_MEGAMOE_W4_ADAPTIVE_CACHE_WEIGHTS 0
#endif
#ifndef MSCCLPP_MEGAMOE_W4_CLUSTER_SIZE
#define MSCCLPP_MEGAMOE_W4_CLUSTER_SIZE 2
#endif
#if defined(MSCCLPP_MEGAMOE_JIT_MODULE) && MSCCLPP_MEGAMOE_JIT_MODULE
#define MSCCLPP_MEGAMOE_KERNEL_NAMESPACE mscclpp::megamoe::jit
#else
#define MSCCLPP_MEGAMOE_KERNEL_NAMESPACE mscclpp::megamoe
#endif

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

constexpr int W4ClusterSize = MSCCLPP_MEGAMOE_W4_CLUSTER_SIZE;
constexpr int W4TileM = 128 * W4ClusterSize;
constexpr int W4TileN = MSCCLPP_MEGAMOE_W4_TILE_N;
constexpr int W4TileK = MSCCLPP_MEGAMOE_W4_TILE_K;
constexpr int W4LoadStages = MSCCLPP_MEGAMOE_W4_LOAD_STAGES;
constexpr int W4NumWarps = MSCCLPP_MEGAMOE_W4_NUM_WARPS;
constexpr int W4TransferRegisters = MSCCLPP_MEGAMOE_W4_TRANSFER_REGISTERS;
constexpr int W4LoadWarps = MSCCLPP_MEGAMOE_W4_LOAD_WARPS;
constexpr bool W4SplitPipelines = MSCCLPP_MEGAMOE_W4_SPLIT_PIPELINES != 0;
constexpr int W4EpilogueTokens = MSCCLPP_MEGAMOE_W4_EPILOGUE_TOKENS;
constexpr int W4EpilogueWarps = MSCCLPP_MEGAMOE_W4_EPILOGUE_WARPS;
constexpr int W4EpilogueRegisters = MSCCLPP_MEGAMOE_W4_EPILOGUE_REGISTERS;
constexpr int W4DispatchChunk = MSCCLPP_MEGAMOE_W4_DISPATCH_CHUNK;
constexpr int W4DispatchWarps = MSCCLPP_MEGAMOE_W4_DISPATCH_WARPS;
constexpr int W4DispatchStages = MSCCLPP_MEGAMOE_W4_DISPATCH_STAGES;
constexpr bool W4FixedTokenCount = MSCCLPP_MEGAMOE_W4_FIXED_TOKEN_COUNT != 0;
constexpr bool W4ExpertMajorTasks = MSCCLPP_MEGAMOE_W4_EXPERT_MAJOR_TASKS != 0;
constexpr bool W4CacheWeights = MSCCLPP_MEGAMOE_W4_CACHE_WEIGHTS != 0;
constexpr bool W4AdaptiveCacheWeights = MSCCLPP_MEGAMOE_W4_ADAPTIVE_CACHE_WEIGHTS != 0;
constexpr int W4TokenStride = W4TileN < 64 ? 64 : W4TileN;

static_assert(W4ClusterSize == 1 || W4ClusterSize == 2);
static_assert(W4TileN == 128, "MegaMoE uses a fixed N128 upper bound with runtime row sizing");
static_assert(W4TileK == 128 || W4TileK == 256 || W4TileK == 512);
static_assert(W4LoadStages >= 2 && W4LoadStages <= 10);
static_assert(W4NumWarps == 12 || W4NumWarps == 16);
static_assert(W4TransferRegisters == 32 || W4TransferRegisters == 64 || W4TransferRegisters == 96 ||
              W4TransferRegisters == 128);
static_assert(W4LoadWarps == 1 || W4LoadWarps == 2);
static_assert(!W4SplitPipelines || W4LoadWarps == 2);
static_assert(W4EpilogueTokens == 16 || W4EpilogueTokens == 32);
static_assert(W4EpilogueWarps == 4 || W4EpilogueWarps == 8);
static_assert(W4EpilogueRegisters >= 128 && W4EpilogueRegisters <= 224 && W4EpilogueRegisters % 8 == 0);
static_assert(W4DispatchChunk == 512 || W4DispatchChunk == 1024 || W4DispatchChunk == 2048 || W4DispatchChunk == 3072 ||
              W4DispatchChunk == 4096);
static_assert(W4DispatchChunk % W4TileK == 0);
static_assert(W4DispatchWarps == 2 || W4DispatchWarps == 4 || W4DispatchWarps == 8);
static_assert(W4DispatchStages == 1 || W4DispatchStages == 2);
static_assert(W4EpilogueWarps + 4 + W4DispatchWarps <= W4NumWarps);
static_assert(MSCCLPP_MEGAMOE_W4_FIXED_TOKEN_COUNT == 0 || MSCCLPP_MEGAMOE_W4_FIXED_TOKEN_COUNT == 1);
static_assert(MSCCLPP_MEGAMOE_W4_EXPERT_MAJOR_TASKS == 0 || MSCCLPP_MEGAMOE_W4_EXPERT_MAJOR_TASKS == 1);
static_assert(MSCCLPP_MEGAMOE_W4_CACHE_WEIGHTS == 0 || MSCCLPP_MEGAMOE_W4_CACHE_WEIGHTS == 1);
static_assert(MSCCLPP_MEGAMOE_W4_ADAPTIVE_CACHE_WEIGHTS == 0 || MSCCLPP_MEGAMOE_W4_ADAPTIVE_CACHE_WEIGHTS == 1);
static_assert(!(W4CacheWeights && W4AdaptiveCacheWeights), "Static and adaptive weight caching are mutually exclusive");

template <int TileN_ = W4TileN, int TileK_ = W4TileK>
struct TilePolicy {
  static constexpr int ClusterSize = W4ClusterSize;
  static constexpr int M = W4TileM;
  static constexpr int N = TileN_;
  static constexpr int K = TileK_;
  static constexpr int TokenStride = N < 64 ? 64 : N;
  static constexpr int CtaM = M / ClusterSize;
  static constexpr int Fc1M = M / 2;
  static constexpr int CtaFc1M = CtaM / 2;
};

template <int NumWarps_, int EpilogueWarps, int DispatchWarps>
struct W4A8WarpScheduleT {
  static constexpr int NumWarps = NumWarps_;
  static constexpr int EpilogueBegin = 0;
  static constexpr int EpilogueEnd = EpilogueWarps;
  static constexpr int MmaWarp = EpilogueEnd;
  static constexpr int LoadWarp = EpilogueEnd + 1;
  static constexpr int ActivationLoadWarp = EpilogueEnd + 2;
  static constexpr int DispatchBegin = EpilogueEnd + 4;
  static constexpr int DispatchEnd = DispatchBegin + DispatchWarps;
};

using W4A8WarpSchedule = W4A8WarpScheduleT<W4NumWarps, W4EpilogueWarps, W4DispatchWarps>;

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_SPECIALIZATION_HPP_
