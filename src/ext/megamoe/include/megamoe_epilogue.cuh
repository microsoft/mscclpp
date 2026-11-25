// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_EPILOGUE_CUH_
#define MSCCLPP_EXT_MEGAMOE_EPILOGUE_CUH_

#include "megamoe_device.cuh"
#include "megamoe_math.cuh"
#include "megamoe_quantization.cuh"
#include "megamoe_tasks.cuh"
#include "megamoe_trace.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

#if MSCCLPP_BULK_AVAILABLE

// Token-major post-SwiGLU FC1 scratch [ChunkTokens, 64].
template <int CtaFc1M>
MSCCLPP_DEVICE_INLINE int fc1ScratchIndex(int tokenIdx, int featureIdx) {
  static_assert(CtaFc1M == 64);
  int swizzle = ((tokenIdx & 3) << 2) | (((tokenIdx >> 3) & 1) << 4);
  return tokenIdx * CtaFc1M + (featureIdx ^ swizzle);
}

template <int LocalMode, class KernelParams>
MSCCLPP_DEVICE_INLINE float fc1RouteWeight(const KernelParams& params, int storageRow) {
  if constexpr (LocalMode == 2) return 1.0f;
  if constexpr (LocalMode == 1) {
    int id = at<int>(params.local, params.symmetric.topkIds)[storageRow];
    assert(id >= -1 && id <= 0);
    return id == 0 ? at<float>(params.local, params.symmetric.topkWeights)[storageRow] : 0.0f;
  }
  return params.config.combineRouteWeights ? 1.0f : params.workspace.routes[storageRow].weight;
}

template <int EpilogueThreads>
MSCCLPP_DEVICE_INLINE void syncBeforeFc1ScratchWrite(int tokenOffset, bool interleaved) {
  if (tokenOffset == 0 && interleaved) bulkStoreWaitSource();
  if (tokenOffset > 0 || interleaved) {
    cutlass::arch::NamedBarrier::sync(EpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
  }
}

template <int CtaFc1M, int HalfTokens>
MSCCLPP_DEVICE_INLINE void writeSwigluPairsToScratch(float* scratch, int baseToken, int outFeature,
                                                     const float (&gate)[HalfTokens], const float (&up)[HalfTokens]) {
#pragma unroll
  for (int i = 0; i < HalfTokens; i += 2) {
    float2 pair = swigluPair(make_float2(gate[i], gate[i + 1]), make_float2(up[i], up[i + 1]));
    scratch[fc1ScratchIndex<CtaFc1M>(baseToken + i, outFeature)] = pair.x;
    scratch[fc1ScratchIndex<CtaFc1M>(baseToken + i + 1, outFeature)] = pair.y;
  }
}

template <int CtaFc1M, int Features>
MSCCLPP_DEVICE_INLINE mscclpp::VectorType<float, Features> loadFc1ScratchVector(const float* scratch, int tokenInChunk,
                                                                                int featureOffset) {
  mscclpp::VectorType<float, Features> activated;
#pragma unroll
  for (int i = 0; i < Features / 4; ++i) {
    reinterpret_cast<mscclpp::f32x4*>(&activated)[i] = *reinterpret_cast<const mscclpp::f32x4*>(
        scratch + fc1ScratchIndex<CtaFc1M>(tokenInChunk, featureOffset + 4 * i));
  }
  return activated;
}

template <class Tiles>
MSCCLPP_DEVICE_INLINE void computeFc1HiddenCoord(const Task& task, int tokenOffset, int tokenInChunk, int featureOffset,
                                                 int& storageRow, int& column) {
  storageRow = w4StorageRow<Tiles>(task.block * Tiles::N + tokenOffset + tokenInChunk);
  column = task.m * Tiles::Fc1M + (blockIdx.x % Tiles::ClusterSize) * Tiles::CtaFc1M + featureOffset;
}

template <int EpilogueThreads>
MSCCLPP_DEVICE_INLINE void finalizeFc1TaskStores() {
  // FC2 loads hidden through TMA; order every generic store of this task before its completion release.
  fenceAsyncProxyGlobal();
  cutlass::arch::NamedBarrier::sync(EpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
}

template <ActivationDtype ActDtype>
struct Fc1ActivationPolicy;

template <>
struct Fc1ActivationPolicy<ActivationDtype::Bf16> {
  template <int HalfTokens, class KernelParams>
  MSCCLPP_DEVICE_INLINE static void preprocessGateUp(const KernelParams&, float (&)[HalfTokens],
                                                     float (&)[HalfTokens]) {}

  template <int LocalMode, int ThreadsPerScale, class Activations, class KernelParams>
  MSCCLPP_DEVICE_INLINE static void quantizeAndStoreHidden(const KernelParams& params, const Task& task,
                                                           const Activations& activated, int tokenOffset,
                                                           int tokenInChunk, int featureOffset, int validRows,
                                                           int intermediate) {
    if (tokenInChunk >= validRows) return;
    using Tiles = typename KernelParams::Tiles;
    constexpr int Features = Activations::Size;
    int storageRow = 0;
    int column = 0;
    computeFc1HiddenCoord<Tiles>(task, tokenOffset, tokenInChunk, featureOffset, storageRow, column);
    float weight = fc1RouteWeight<LocalMode>(params, storageRow);
    Activations weighted = activated;
#pragma unroll
    for (int i = 0; i < Features; ++i) weighted[i] = __fmul_rn(weighted[i], weight);
    using PackedBf16 = mscclpp::VectorType<__bfloat16, Features>;
    auto* dst =
        reinterpret_cast<__bfloat16*>(params.workspace.quantizedHidden) + size_t(storageRow) * intermediate + column;
    *reinterpret_cast<PackedBf16*>(dst) = mscclpp::to<PackedBf16>(weighted);
  }
};

template <>
struct Fc1ActivationPolicy<ActivationDtype::Fp8Static> {
  template <int HalfTokens, class KernelParams>
  MSCCLPP_DEVICE_INLINE static void preprocessGateUp(const KernelParams& params, float (&gate)[HalfTokens],
                                                     float (&up)[HalfTokens]) {
    const float actScale = params.config.activationScale;
#pragma unroll
    for (int i = 0; i < HalfTokens; ++i) {
      gate[i] *= actScale;
      up[i] *= actScale;
    }
  }

  template <int LocalMode, int ThreadsPerScale, class Activations, class KernelParams>
  MSCCLPP_DEVICE_INLINE static void quantizeAndStoreHidden(const KernelParams& params, const Task& task,
                                                           const Activations& activated, int tokenOffset,
                                                           int tokenInChunk, int featureOffset, int validRows,
                                                           int intermediate) {
    const float actScale = params.config.activationScale;
    const float invActScale = actScale != 0.0f ? (1.0f / actScale) : 0.0f;
    auto quantized = quantizeE4M3(activated, invActScale);
    if (tokenInChunk < validRows) {
      using Tiles = typename KernelParams::Tiles;
      int storageRow = 0;
      int column = 0;
      computeFc1HiddenCoord<Tiles>(task, tokenOffset, tokenInChunk, featureOffset, storageRow, column);
      *reinterpret_cast<decltype(quantized)*>(params.workspace.quantizedHidden + size_t(storageRow) * intermediate +
                                              column) = quantized;
    }
  }
};

template <>
struct Fc1ActivationPolicy<ActivationDtype::Mxfp8> {
  template <int HalfTokens, class KernelParams>
  MSCCLPP_DEVICE_INLINE static void preprocessGateUp(const KernelParams&, float (&)[HalfTokens],
                                                     float (&)[HalfTokens]) {}

  template <int LocalMode, int ThreadsPerScale, class Activations, class KernelParams>
  MSCCLPP_DEVICE_INLINE static void quantizeAndStoreHidden(const KernelParams& params, const Task& task,
                                                           const Activations& activated, int tokenOffset,
                                                           int tokenInChunk, int featureOffset, int validRows,
                                                           int intermediate) {
    const float maxAbs = warpSubgroupMaxAbs<ThreadsPerScale>(activated);
    const uint8_t scale = quantizeE8M0Scale(maxAbs);
    auto quantized = quantizeE4M3(activated, inverseE8M0Scale(scale));
    if (tokenInChunk < validRows) {
      using Tiles = typename KernelParams::Tiles;
      int storageRow = 0;
      int column = 0;
      computeFc1HiddenCoord<Tiles>(task, tokenOffset, tokenInChunk, featureOffset, storageRow, column);
      if (threadIdx.x % ThreadsPerScale == 0) {
        params.workspace.hiddenScale[params.fc2.layout_SFB(cute::make_coord(storageRow, column, 0))] = scale;
      }
      *reinterpret_cast<decltype(quantized)*>(params.workspace.quantizedHidden + size_t(storageRow) * intermediate +
                                              column) = quantized;
    }
  }
};

template <int EpilogueThreads, int ChunkTokens, int LocalMode = 0, class KernelParams, class Storage, class Values>
MSCCLPP_DEVICE_INLINE void activateFc1Chunk(const KernelParams& params, const Task& task, Storage& sharedStorage,
                                            const Values& values, int tokenOffset, int validRows, int intermediate,
                                            bool lastChunk, bool interleaved = false) {
  using Tiles = typename KernelParams::Tiles;
  using Policy = Fc1ActivationPolicy<KernelParams::Activation>;
  constexpr int CtaTileM = Tiles::CtaM;
  constexpr int CtaFc1M = Tiles::CtaFc1M;
  constexpr int ReaderGroups = EpilogueThreads / CtaTileM;
  constexpr int ReadTokens = ChunkTokens / ReaderGroups;
  constexpr int HalfTokens = ReadTokens / 2;
  // Each thread activates consecutive features of one token; ThreadsPerScale lanes share one E8M0 scale.
  constexpr int Features = ChunkTokens * CtaFc1M / EpilogueThreads;
  constexpr int ThreadsPerToken = CtaFc1M / Features;
  constexpr int ThreadsPerScale = MxfpScaleBlockSize / Features;
  static_assert(CtaFc1M == 64 && HalfTokens % 2 == 0 && (Features == 4 || Features == 8 || Features == 16));

  const int laneIdx = threadIdx.x % WARP_SIZE;
  const int warpInGroup = (threadIdx.x % CtaTileM) / WARP_SIZE;
  const int readerGroup = threadIdx.x / CtaTileM;
  const bool isUpHalf = (laneIdx & (WARP_SIZE / 2)) != 0;
  const int outFeature = warpInGroup * (WARP_SIZE / 2) + (laneIdx & (WARP_SIZE / 2 - 1));
  const int baseToken = readerGroup * ReadTokens + (isUpHalf ? HalfTokens : 0);

  // 1. Exchange half of gate/up intra-warp, apply static activation scale if needed, and clamp in registers.
  float gate[HalfTokens];
  float up[HalfTokens];
  if (readerGroup * ReadTokens < validRows) {
    exchangeGateUpHalfIntraWarp<HalfTokens>(values, isUpHalf, gate, up);
    Policy::template preprocessGateUp<HalfTokens>(params, gate, up);
    clampGateUpArray<HalfTokens>(gate, up, params.config.gateUpClamp);
  }

  // 2. Evaluate SwiGLU in registers and stage into swizzled shared scratch.
  syncBeforeFc1ScratchWrite<EpilogueThreads>(tokenOffset, interleaved);
  if (baseToken < validRows) {
    writeSwigluPairsToScratch<CtaFc1M, HalfTokens>(sharedStorage.epilogue.scratch, baseToken, outFeature, gate, up);
  }
  cutlass::arch::NamedBarrier::sync(EpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

  // 3. Reload contiguous token features and quantize/store hidden activations via ActivationDtype policy.
  const int tokenInChunk = threadIdx.x / ThreadsPerToken;
  if ((threadIdx.x / WARP_SIZE) * (WARP_SIZE / ThreadsPerToken) < validRows) {
    const int featureOffset = (threadIdx.x % ThreadsPerToken) * Features;
    const auto activated =
        loadFc1ScratchVector<CtaFc1M, Features>(sharedStorage.epilogue.scratch, tokenInChunk, featureOffset);
    Policy::template quantizeAndStoreHidden<LocalMode, ThreadsPerScale>(
        params, task, activated, tokenOffset, tokenInChunk, featureOffset, validRows, intermediate);
  }

  // 4. Order generic hidden stores before publishing FC1 tile completion.
  if (lastChunk) finalizeFc1TaskStores<EpilogueThreads>();
}

// 16-datapath TMEM loads return stmatrix fragments: for each 8-column group, thread t receives lanes t/4 and
// t/4 + 8 at columns 2(t%4) and 2(t%4) + 1.
template <int Groups>
MSCCLPP_DEVICE_INLINE void loadFc2Accumulators(uint32_t address, uint32_t (&data)[4 * Groups]) {
  using namespace cute;
  if constexpr (Groups == 4) {
    SM100_TMEM_LOAD_16dp256b4x::copy(address, data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
                                     data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15]);
  } else if constexpr (Groups == 2) {
    SM100_TMEM_LOAD_16dp256b2x::copy(address, data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]);
  } else {
    static_assert(Groups == 1);
    SM100_TMEM_LOAD_16dp256b1x::copy(address, data[0], data[1], data[2], data[3]);
  }
}

template <int LocalMode, class KernelParams>
MSCCLPP_DEVICE_INLINE float2 fc2RouteWeights(const KernelParams& params, const Task& task, int tokenOffset,
                                             int validRows, int firstToken, int token) {
  constexpr int KernelTileN = KernelParams::Tiles::N;
  int row = task.block * KernelTileN + tokenOffset + firstToken + token;
  if constexpr (LocalMode == 1) {
    const int* ids = at<int>(params.local, params.symmetric.topkIds);
    return make_float2(firstToken + token < validRows && ids[row] == 0 ? 1.0f : 0.0f,
                       firstToken + token + 1 < validRows && ids[row + 1] == 0 ? 1.0f : 0.0f);
  }
  if constexpr (KernelParams::Activation == ActivationDtype::Bf16) return make_float2(1.0f, 1.0f);
  const float scale = (KernelParams::Activation == ActivationDtype::Fp8Static && params.config.computeOnly)
                          ? params.config.activationScale
                          : 1.0f;
  if (params.config.combineRouteWeights) return make_float2(scale, scale);
  const Route* routes = params.workspace.routes + row;
  return make_float2(routes[0].weight * scale, routes[1].weight * scale);
}

template <int ReadTokens, int LocalMode, class KernelParams, class Storage>
MSCCLPP_DEVICE_INLINE void packFc2ChunkFp8Static(const KernelParams& params, const Task& task, Storage& sharedStorage,
                                                 const uint32_t (&accumulator)[2][ReadTokens / 2], int tokenOffset,
                                                 int validRows, int firstToken, int laneIdx) {
  constexpr int FeaturesPerWarp = KernelParams::Tiles::CtaM / 4;
  const int warpFeatureBase = (threadIdx.x / WARP_SIZE % 4) * FeaturesPerWarp;
#pragma unroll
  for (int group = 0; group < ReadTokens / 8; ++group) {
    int token = group * 8 + laneIdx % 4 * 2;
    float2 weights = fc2RouteWeights<LocalMode>(params, task, tokenOffset, validRows, firstToken, token);
    uint8_t* tokenRows = sharedStorage.epilogue.packedFp8 + (firstToken + token) * PackedFp8RowStride;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t* pair = accumulator[i / 2] + group * 4 + i % 2 * 2;
      mscclpp::f32x2 scaled = __fmul2_rn(make_float2(__uint_as_float(pair[0]), __uint_as_float(pair[1])), weights);
      uint16_t fp8Pair = mscclpp::bit_cast<uint16_t>(mscclpp::to<mscclpp::f8_e4m3x2>(scaled));
      storeTransposedFp8Quad(tokenRows, PackedFp8RowStride, warpFeatureBase + i * 8, laneIdx, fp8Pair);
    }
  }
}

template <int ReadTokens, int LocalMode, class KernelParams, class Storage>
MSCCLPP_DEVICE_INLINE void packFc2ChunkBf16(const KernelParams& params, const Task& task, Storage& sharedStorage,
                                            const uint32_t (&accumulator)[2][ReadTokens / 2], int tokenOffset,
                                            int validRows, int firstToken, int laneIdx) {
  constexpr int FeaturesPerWarp = KernelParams::Tiles::CtaM / 4;
  // Matrix m of each x4 store holds the warp's features [8m, 8m + 8); lane 8m + j addresses token row j.
  auto* rows = sharedStorage.epilogue.packed + (firstToken + laneIdx % 8) * PackedRowStride +
               threadIdx.x / WARP_SIZE % 4 * FeaturesPerWarp + laneIdx / 8 * 8;
#pragma unroll
  for (int group = 0; group < ReadTokens / 8; ++group) {
    int token = group * 8 + laneIdx % 4 * 2;
    float2 weights = fc2RouteWeights<LocalMode>(params, task, tokenOffset, validRows, firstToken, token);
    uint32_t words[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      // Matrices 0..3 are lanes t/4 and t/4 + 8 of the first, then the second 16-lane half.
      const uint32_t* pair = accumulator[i / 2] + group * 4 + i % 2 * 2;
      mscclpp::f32x2 scaled = __fmul2_rn(make_float2(__uint_as_float(pair[0]), __uint_as_float(pair[1])), weights);
      words[i] = mscclpp::bit_cast<uint32_t>(mscclpp::to<mscclpp::bf16x2>(scaled));
    }
    cute::SM90_U16x8_STSM_T::copy(words[0], words[1], words[2], words[3],
                                  *reinterpret_cast<cute::uint128_t*>(rows + group * 8 * PackedRowStride));
  }
}

// Scales a warp's 32 FC2 features x ReadTokens tokens by route weights in pairs and writes token-major BF16/FP8 rows.
// Padded tokens are never stored, so they are not masked.
template <int ReadTokens, int LocalMode = 0, class KernelParams, class Storage>
MSCCLPP_DEVICE_INLINE void packFc2Chunk(const KernelParams& params, const Task& task, Storage& sharedStorage,
                                        const uint32_t (&accumulator)[2][ReadTokens / 2], int tokenOffset,
                                        int validRows, int readerGroup) {
  static_assert(KernelParams::Tiles::CtaM == 128 && ReadTokens % 8 == 0);
  int laneIdx = threadIdx.x % WARP_SIZE;
  int firstToken = readerGroup * ReadTokens;
  if constexpr (KernelParams::Activation == ActivationDtype::Fp8Static) {
    if (!params.config.computeOnly) {
      packFc2ChunkFp8Static<ReadTokens, LocalMode>(params, task, sharedStorage, accumulator, tokenOffset, validRows,
                                                   firstToken, laneIdx);
      return;
    }
  }
  packFc2ChunkBf16<ReadTokens, LocalMode>(params, task, sharedStorage, accumulator, tokenOffset, validRows, firstToken,
                                          laneIdx);
}

template <int EpilogueWarps, class KernelParams, class Storage>
MSCCLPP_DEVICE_INLINE void storeLocalOutputChunk(const Task& task, Storage& sharedStorage, int tokenOffset,
                                                 int validRows, int hidden, int warpIdx, int laneIdx,
                                                 __bfloat16* finalOutput) {
  using Tiles = typename KernelParams::Tiles;
  constexpr int KernelTileN = Tiles::N;
  constexpr int CtaTileM = Tiles::CtaM;
  constexpr int WarpTokens = (KernelParams::Collective::EpilogueTokens + EpilogueWarps - 1) / EpilogueWarps;
  int feature = task.m * CtaTileM * Tiles::ClusterSize + (blockIdx.x % Tiles::ClusterSize) * CtaTileM;
  if (feature >= hidden) return;
  if ((reinterpret_cast<uintptr_t>(finalOutput) & 15) != 0) {
    constexpr int EpilogueThreads = EpilogueWarps * WARP_SIZE;
    for (int tokenIdx = 0; tokenIdx < validRows; ++tokenIdx) {
      size_t offset = size_t(task.block * KernelTileN + tokenOffset + tokenIdx) * hidden + feature;
      mscclpp::detail::copy<__bfloat16>(finalOutput + offset,
                                        sharedStorage.epilogue.packed + tokenIdx * PackedRowStride, CtaTileM,
                                        threadIdx.x, EpilogueThreads);
    }
    return;
  }
  int tokenIdx = warpIdx + laneIdx * EpilogueWarps;
  if (laneIdx < WarpTokens && tokenIdx < validRows) {
    size_t offset = size_t(task.block * KernelTileN + tokenOffset + tokenIdx) * hidden + feature;
    bulkStore(finalOutput + offset, sharedStorage.epilogue.packed + tokenIdx * PackedRowStride,
              CtaTileM * sizeof(__bfloat16));
    bulkStoreCommit();
  }
}

template <int EpilogueWarps, int LocalMode = 0, class KernelParams, class Storage>
MSCCLPP_DEVICE_INLINE void storeOutputChunk(const KernelParams& params, const Task& task, Storage& sharedStorage,
                                            int tokenOffset, int validRows, int hidden, int warpIdx, int laneIdx,
                                            __bfloat16* finalOutput, int runtimeTokens = 0) {
  if constexpr (LocalMode != 0) {
    storeLocalOutputChunk<EpilogueWarps, KernelParams>(task, sharedStorage, tokenOffset, validRows, hidden, warpIdx,
                                                       laneIdx, finalOutput);
    return;
  }
  using Tiles = typename KernelParams::Tiles;
  constexpr int KernelTileN = Tiles::N;
  constexpr int CtaTileM = Tiles::CtaM;
  constexpr int WarpTokens = (KernelParams::Collective::EpilogueTokens + EpilogueWarps - 1) / EpilogueWarps;
  static_assert(WarpTokens <= WARP_SIZE);
  int feature = task.m * CtaTileM * Tiles::ClusterSize + (blockIdx.x % Tiles::ClusterSize) * CtaTileM;
  if (feature >= hidden) return;
  // Lane l of warp w stores token w + l * EpilogueWarps with its own bulk copy, so route and peer lookups run in
  // parallel; each issuing lane owns its bulk group and drains the shared-memory reads before the next chunk.
  int tokenIdx = warpIdx + laneIdx * EpilogueWarps;
  if (laneIdx < WarpTokens && tokenIdx < validRows) {
    Route route = params.workspace.routes[task.block * KernelTileN + tokenOffset + tokenIdx];
    if (params.config.computeOnly) {
      size_t offset =
          ((size_t(route.rank) * runtimeTokens + route.token) * params.config.topK + route.slot) * hidden + feature;
      bulkStore(finalOutput + offset, sharedStorage.epilogue.packed + tokenIdx * PackedRowStride,
                CtaTileM * sizeof(__bfloat16));
      bulkStoreCommit();
      return;
    }
    if constexpr (KernelParams::Activation == ActivationDtype::Fp8Static) {
      auto* output = peerAt<uint8_t>(params, route.rank, params.symmetric.partialOutput);
      size_t offset = (size_t(route.token) * params.config.topK + route.slot) * params.config.hidden + feature;
      bulkStore(output + offset, sharedStorage.epilogue.packedFp8 + tokenIdx * PackedFp8RowStride,
                CtaTileM * sizeof(uint8_t));
      bulkStoreCommit();
      return;
    }
    __bfloat16* output;
    size_t offset;
    if (!params.config.combineRouteWeights && KernelParams::WeightMxfp4 && params.config.worldSize == 1 &&
        params.config.topK == 1) {
      // Delivered also finalizes a unique local contribution by TMA store; BF16 TMA reduction is only needed
      // when another contribution targets the same token.
      output = finalOutput;
      offset = size_t(route.token) * hidden + feature;
    } else {
      output = peerAt<__bfloat16>(params, route.rank, params.symmetric.partialOutput);
      offset = (size_t(route.token) * params.config.topK + route.slot) * params.config.hidden + feature;
    }
    bulkStore(output + offset, sharedStorage.epilogue.packed + tokenIdx * PackedRowStride,
              CtaTileM * sizeof(__bfloat16));
    bulkStoreCommit();
  }
}

#endif  // MSCCLPP_BULK_AVAILABLE

template <class Schedule, int LocalMode, class KernelParams>
constexpr bool UseOutputOffload =
    (LocalMode == 0) && KernelParams::WeightMxfp4 && (KernelParams::Tiles::K >= 256) &&
    (KernelParams::Collective::DispatchWarps >= 4) && (KernelParams::Collective::EpilogueTokens == 32) &&
    ((Schedule::EpilogueEnd - Schedule::EpilogueBegin) == 4) && ((Schedule::NumWarps - Schedule::DispatchEnd) >= 4);

constexpr uint32_t OutputOffloadSlotEmptyBarrier = 0u;
constexpr uint32_t OutputOffloadSlotFullBarrier = 1u;
constexpr int OutputOffloadThreads = 256;

template <class Schedule, class KernelParams, class Storage>
MSCCLPP_DEVICE_INLINE void runOutputStoreOffloadRole(const KernelParams& params, Storage& sharedStorage, int warpIdx,
                                                     int laneIdx, int hidden, __bfloat16* finalOutput = nullptr,
                                                     int runtimeTokens = 0) {
#if MSCCLPP_BULK_AVAILABLE
  if constexpr (UseOutputOffload<Schedule, 0, KernelParams>) {
    constexpr int OffloadBegin = Schedule::NumWarps - 4;
    if (warpIdx < OffloadBegin || warpIdx >= Schedule::NumWarps) return;
    using Tiles = typename KernelParams::Tiles;
    constexpr int CtaTileM = Tiles::CtaM;
    cutlass::arch::NamedBarrier::arrive(OutputOffloadThreads, OutputOffloadSlotEmptyBarrier);
    while (true) {
      cutlass::arch::NamedBarrier::sync(OutputOffloadThreads, OutputOffloadSlotFullBarrier);
      const OutputOffloadTask offloadTask = sharedStorage.outputOffloadTask;
      if (offloadTask.op() == OutputOffloadOp::Stop) break;
      if (offloadTask.op() == OutputOffloadOp::Store) {
        const int validRows = offloadTask.nTokens();
        const int feature =
            offloadTask.mTileIdx() * (CtaTileM * Tiles::ClusterSize) + (blockIdx.x % Tiles::ClusterSize) * CtaTileM;
        const int baseRow = offloadTask.baseRowIdx();
        const int tokenIdx = (warpIdx - OffloadBegin) + laneIdx * 4;
        if (feature < hidden && laneIdx < 8 && tokenIdx < validRows) {
          Route route = params.workspace.routes[baseRow + tokenIdx];
          traceKernel(KernelTracePhase::Fc2Store, true, baseRow);
          if (params.config.computeOnly) {
            size_t offset =
                ((size_t(route.rank) * runtimeTokens + route.token) * params.config.topK + route.slot) * hidden +
                feature;
            bulkStore(finalOutput + offset, sharedStorage.epilogue.packed + tokenIdx * PackedRowStride,
                      CtaTileM * sizeof(__bfloat16));
            bulkStoreCommit();
          } else if constexpr (KernelParams::Activation == ActivationDtype::Fp8Static) {
            auto* out = peerAt<uint8_t>(params, route.rank, params.symmetric.partialOutput);
            size_t offset = (size_t(route.token) * params.config.topK + route.slot) * params.config.hidden + feature;
            bulkStore(out + offset, sharedStorage.epilogue.packedFp8 + tokenIdx * PackedFp8RowStride,
                      CtaTileM * sizeof(uint8_t));
            bulkStoreCommit();
          } else {
            __bfloat16* out;
            size_t offset;
            if (!params.config.combineRouteWeights && params.config.worldSize == 1 && params.config.topK == 1) {
              out = finalOutput;
              offset = size_t(route.token) * hidden + feature;
            } else {
              out = peerAt<__bfloat16>(params, route.rank, params.symmetric.partialOutput);
              offset = (size_t(route.token) * params.config.topK + route.slot) * params.config.hidden + feature;
            }
            bulkStore(out + offset, sharedStorage.epilogue.packed + tokenIdx * PackedRowStride,
                      CtaTileM * sizeof(__bfloat16));
            bulkStoreCommit();
          }
          traceKernel(KernelTracePhase::Fc2Store, false, baseRow);
        }
        bulkStoreWaitSource();
      }
      cutlass::arch::NamedBarrier::arrive(OutputOffloadThreads, OutputOffloadSlotEmptyBarrier);
    }
    bulkStoreWait();
  }
#endif
}

template <class Schedule, int LocalMode = 0, class KernelParams, class Storage, class Pipeline, class Accumulators>
MSCCLPP_DEVICE_INLINE void epilogue(const KernelParams& params, const Task& task, Storage& sharedStorage,
                                    Pipeline& pipeline, typename Pipeline::PipelineState& state,
                                    Accumulators accumulators, int hidden, int intermediate,
                                    __bfloat16* finalOutput = nullptr, int nTokens = 0, bool interleaved = false) {
#if MSCCLPP_BULK_AVAILABLE
  using namespace cute;
  using Tiles = typename KernelParams::Tiles;
  constexpr int KernelTileN = Tiles::N;
  constexpr int CtaTileM = Tiles::CtaM;
  constexpr int ChunkTokens = KernelParams::Collective::EpilogueTokens;
  constexpr int EpilogueWarps = Schedule::EpilogueEnd - Schedule::EpilogueBegin;
  constexpr int EpilogueThreads = EpilogueWarps * WARP_SIZE;
  constexpr int ReaderGroups = EpilogueThreads / CtaTileM;
  constexpr int ReadTokens = ChunkTokens / ReaderGroups;
  constexpr bool OffloadOutput = UseOutputOffload<Schedule, LocalMode, KernelParams>;
  int warpIdx = threadIdx.x / WARP_SIZE;
  int laneIdx = threadIdx.x % WARP_SIZE;
  auto matrix = accumulators(make_coord(_, _), _0{}, _0{}, state.index());
  CUTE_STATIC_ASSERT_V(size<0>(matrix) == Int<CtaTileM>{});
  CUTE_STATIC_ASSERT_V(size<1>(matrix) == Int<KernelTileN>{});
  traceKernel(KernelTracePhase::EpilogueReady, true, int(task.fc1));
  pipeline.consumer_wait(state);
  traceKernel(KernelTracePhase::EpilogueReady, false, int(task.fc1));
  for (int tokenOffset = 0; tokenOffset < task.tokens.nTokens; tokenOffset += ChunkTokens) {
    int validRows = min(int(ChunkTokens), task.tokens.nTokens - tokenOffset);
    bool lastChunk = tokenOffset + ChunkTokens >= task.tokens.nTokens;
    int readerGroup = threadIdx.x / CtaTileM;
    if (task.fc1) {
      int readerThread = threadIdx.x % CtaTileM;
      auto acc = local_tile(matrix, make_shape(Int<CtaTileM>{}, Int<ReadTokens>{}),
                            make_coord(0, tokenOffset / ReadTokens + readerGroup));
      auto copyOp = make_tmem_copy(SM100_TMEM_LOAD_32dp32b8x{}, acc);
      auto threadCopy = copyOp.get_slice(readerThread);
      auto source = threadCopy.partition_S(acc);
      auto identity = make_identity_tensor(make_shape(Int<CtaTileM>{}, Int<ChunkTokens>{}));
      auto readIdentity =
          local_tile(identity, make_shape(Int<CtaTileM>{}, Int<ReadTokens>{}), make_coord(0, readerGroup));
      auto coordinates = threadCopy.partition_D(readIdentity);
      auto values = make_tensor<float>(shape(coordinates));
      if (readerGroup * ReadTokens < validRows) copy(copyOp, source, values);
      cutlass::arch::fence_view_async_tmem_load();
      if (lastChunk) pipeline.consumer_release(state);
      traceKernel(KernelTracePhase::ActivationQuantize, true, tokenOffset);
      activateFc1Chunk<EpilogueThreads, ChunkTokens, LocalMode>(params, task, sharedStorage, values, tokenOffset,
                                                                validRows, intermediate, lastChunk,
                                                                OffloadOutput ? false : interleaved);
      traceKernel(KernelTracePhase::ActivationQuantize, false, tokenOffset);
      continue;
    }
    // FC2 reads both 16-lane halves of the warp's TMEM lane quarter as stmatrix fragments.
    uint32_t accumulator[2][ReadTokens / 2];
    if (readerGroup * ReadTokens < validRows) {
      uint32_t address = raw_pointer_cast(matrix.data()) + tokenOffset + readerGroup * ReadTokens;
#pragma unroll
      for (int half = 0; half < 2; ++half)
        loadFc2Accumulators<ReadTokens / 8>(address + uint32_t((warpIdx % 4 * 32 + half * 16) << 16),
                                            accumulator[half]);
    }
    cutlass::arch::fence_view_async_tmem_load();
    if (lastChunk) pipeline.consumer_release(state);
    if constexpr (OffloadOutput) {
      cutlass::arch::NamedBarrier::sync(OutputOffloadThreads, OutputOffloadSlotEmptyBarrier);
      if (readerGroup * ReadTokens < validRows)
        packFc2Chunk<ReadTokens, LocalMode>(params, task, sharedStorage, accumulator, tokenOffset, validRows,
                                            readerGroup);
      if (threadIdx.x == 0)
        sharedStorage.outputOffloadTask =
            OutputOffloadTask::makeStore(task.block * KernelTileN + tokenOffset, validRows, task.m);
      bulkFence();
      cutlass::arch::NamedBarrier::arrive(OutputOffloadThreads, OutputOffloadSlotFullBarrier);
    } else {
      bulkStoreWaitSource();
      cutlass::arch::NamedBarrier::sync(EpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
      if (readerGroup * ReadTokens < validRows)
        packFc2Chunk<ReadTokens, LocalMode>(params, task, sharedStorage, accumulator, tokenOffset, validRows,
                                            readerGroup);
      bulkFence();
      cutlass::arch::NamedBarrier::sync(EpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
      traceKernel(KernelTracePhase::Fc2Store, true, tokenOffset);
      storeOutputChunk<EpilogueWarps, LocalMode>(params, task, sharedStorage, tokenOffset, validRows, hidden, warpIdx,
                                                 laneIdx, finalOutput, nTokens);
      traceKernel(KernelTracePhase::Fc2Store, false, tokenOffset);
    }
  }
  ++state;
  if (task.fc1 && threadIdx.x == 0) {
    signalReadyCounter(params.workspace.hiddenReady + size_t(task.block) * W4ReadyCounterStride);
  }
#endif
}

template <class Schedule, int LocalMode = 0, class KernelParams, class Storage, class Accumulate, class Accumulators>
MSCCLPP_DEVICE_INLINE void runEpilogueRole(const KernelParams& params, Storage& sharedStorage, Accumulate& pipeline,
                                           Accumulators accumulators, int hidden, int intermediate, int clusterIdx,
                                           int nTasks, __bfloat16* finalOutput = nullptr, int nTokens = 0) {
  cutlass::arch::warpgroup_reg_alloc<KernelParams::Collective::EpilogueRegisters>();
  typename Accumulate::PipelineState state;
  const int nTokenTiles = taskTokenTiles(params, nTokens);
  const int nLocalExperts = params.config.numExperts / params.config.worldSize;
  const auto scheduler = makeTaskScheduler<KernelParams>(nTokenTiles, nLocalExperts, hidden, intermediate);
  const int nClusters = gridDim.x / KernelParams::Tiles::ClusterSize;
  const bool interleaved = scheduler.isInterleaved;
  constexpr bool OffloadOutput = UseOutputOffload<Schedule, LocalMode, KernelParams>;
  bool prevWasFc2 = false;
  if (clusterIdx < nTasks) {
    Task nextTask = scheduler.taskAt(params, clusterIdx, nTokens);
    for (int i = clusterIdx; i < nTasks; i += nClusters) {
      Task task = nextTask;
      if (i + nClusters < nTasks) nextTask = scheduler.taskAt(params, i + nClusters, nTokens);
      if constexpr (OffloadOutput) {
        if (task.fc1 && interleaved && prevWasFc2) {
          cutlass::arch::NamedBarrier::sync(OutputOffloadThreads, OutputOffloadSlotEmptyBarrier);
          if (threadIdx.x == 0) sharedStorage.outputOffloadTask = OutputOffloadTask::makeNoop();
          cutlass::arch::NamedBarrier::arrive(OutputOffloadThreads, OutputOffloadSlotFullBarrier);
        }
      }
      epilogue<Schedule, LocalMode>(params, task, sharedStorage, pipeline, state, accumulators, hidden, intermediate,
                                    finalOutput, nTokens, interleaved);
      prevWasFc2 = !task.fc1;
    }
  }
#if MSCCLPP_BULK_AVAILABLE
  if constexpr (OffloadOutput) {
    cutlass::arch::NamedBarrier::sync(OutputOffloadThreads, OutputOffloadSlotEmptyBarrier);
    if (threadIdx.x == 0) sharedStorage.outputOffloadTask = OutputOffloadTask::makeStop();
    cutlass::arch::NamedBarrier::arrive(OutputOffloadThreads, OutputOffloadSlotFullBarrier);
  } else {
    // Every lane may own FC2 bulk stores; they must complete before the CTA publishes its completion.
    bulkStoreWait();
  }
#endif
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_EPILOGUE_CUH_
