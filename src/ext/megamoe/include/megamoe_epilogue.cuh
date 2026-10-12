// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_EPILOGUE_CUH_
#define MSCCLPP_EXT_MEGAMOE_EPILOGUE_CUH_

#include "megamoe_device.cuh"
#include "megamoe_quantization.cuh"
#include "megamoe_tasks.cuh"
#include "megamoe_trace.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

#if MSCCLPP_BULK_AVAILABLE

MSCCLPP_DEVICE_INLINE float2 swigluPair(float2 gate, float2 up) {
  float2 product = __fmul2_rn(gate, up);
  float2 exponent = __fmul2_rn(gate, make_float2(-1.4426950408889634f, -1.4426950408889634f));
  float2 denominator;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(denominator.x) : "f"(exponent.x));
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(denominator.y) : "f"(exponent.y));
  denominator = __fadd2_rn(denominator, make_float2(1.0f, 1.0f));
  float2 inverse;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inverse.x) : "f"(denominator.x));
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inverse.y) : "f"(denominator.y));
  return __fmul2_rn(product, inverse);
}

// Token-major FC1 scratch. XOR-swizzling 16-byte chunks by token parity and 32-row gate/up group keeps both
// the per-row TMEM spills and the per-token vector activation loads free of shared-memory bank conflicts.
template <int CtaTileM>
MSCCLPP_DEVICE_INLINE int fc1ScratchIndex(int token, int row) {
  static_assert(CtaTileM == 128);
  int group = row / 32;
  int swizzle = ((group & 1) << 2) | ((token & 1) << 1) | (group >> 1);
  return token * CtaTileM + (row ^ (swizzle << 2));
}

template <int EpilogueThreads, int ChunkTokens, class KernelParams, class Storage, class Values, class Coordinates>
MSCCLPP_DEVICE_INLINE void activateFc1Chunk(const KernelParams& params, const Task& task, Storage& s,
                                            const Values& values, const Coordinates& coordinates, int tokenOffset,
                                            int validRows, int intermediate, bool lastChunk) {
  using namespace cute;
  using Tiles = typename KernelParams::Tiles;
  constexpr int KernelTileN = Tiles::N;
  constexpr int CtaTileM = Tiles::CtaM;
  constexpr int CtaFc1M = Tiles::CtaFc1M;
  constexpr int GateUpGroupSize = 16;
  constexpr int ScaleGroupSize = 32;
  // Each thread activates consecutive features of one token; ThreadsPerScale lanes share one E8M0 scale.
  constexpr int Features = ChunkTokens * CtaFc1M / EpilogueThreads;
  constexpr int ThreadsPerToken = CtaFc1M / Features;
  constexpr int ThreadsPerScale = ScaleGroupSize / Features;
  static_assert(CtaFc1M == 64 && (Features == 4 || Features == 8 || Features == 16));
  using Activations = mscclpp::VectorType<float, Features>;

#pragma unroll
  for (int i = 0; i < size(values); ++i) {
    auto coord = coordinates(i);
    s.epilogue.scratch[fc1ScratchIndex<CtaTileM>(int(get<1>(coord)), int(get<0>(coord)))] = values(i);
  }
  cutlass::arch::NamedBarrier::sync(EpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

  int token = threadIdx.x / ThreadsPerToken;
  // Warps holding only padded tokens skip the work; scale shuffles never cross a warp.
  if (threadIdx.x / 32 * (32 / ThreadsPerToken) < validRows) {
    int feature = threadIdx.x % ThreadsPerToken * Features;
    int gateRow = feature / GateUpGroupSize * 2 * GateUpGroupSize + feature % GateUpGroupSize;
    Activations gate, up;
#pragma unroll
    for (int i = 0; i < Features / 4; ++i) {
      reinterpret_cast<mscclpp::f32x4*>(&gate)[i] = *reinterpret_cast<const mscclpp::f32x4*>(
          s.epilogue.scratch + fc1ScratchIndex<CtaTileM>(token, gateRow + 4 * i));
      reinterpret_cast<mscclpp::f32x4*>(&up)[i] = *reinterpret_cast<const mscclpp::f32x4*>(
          s.epilogue.scratch + fc1ScratchIndex<CtaTileM>(token, gateRow + GateUpGroupSize + 4 * i));
    }
    // Clamp in one uniform pass so the activation below stays a single basic block.
    if (float clamp = params.config.gateUpClamp; clamp >= 0) {
#pragma unroll
      for (int i = 0; i < Features; ++i) {
        gate[i] = fminf(gate[i], clamp);
        up[i] = fmaxf(-clamp, fminf(up[i], clamp));
      }
    }
    Activations activated;
#pragma unroll
    for (int i = 0; i < Features; i += 2) {
      float2 pair = swigluPair(make_float2(gate[i], gate[i + 1]), make_float2(up[i], up[i + 1]));
      activated[i] = pair.x;
      activated[i + 1] = pair.y;
    }
    Activations magnitude;
#pragma unroll
    for (int i = 0; i < Features; ++i) magnitude[i] = fabsf(activated[i]);
#pragma unroll
    for (int width = Features / 2; width > 0; width /= 2) {
#pragma unroll
      for (int i = 0; i < width; ++i) magnitude[i] = fmaxf(magnitude[i], magnitude[i + width]);
    }
    float maximum = magnitude[0];
#pragma unroll
    for (int offset = ThreadsPerScale / 2; offset > 0; offset /= 2)
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    uint8_t scale = quantizeE8M0Scale(maximum);
    auto quantized = quantizeE4M3(activated, inverseE8M0Scale(scale));
    if (token < validRows) {
      int row = w4StorageRow<Tiles>(task.block * KernelTileN + tokenOffset + token);
      int column = task.m * Tiles::Fc1M + (blockIdx.x % Tiles::ClusterSize) * CtaFc1M + feature;
      if (threadIdx.x % ThreadsPerScale == 0)
        params.workspace.hiddenScale[params.fc2.layout_SFB(make_coord(row, column, 0))] = scale;
      *reinterpret_cast<decltype(quantized)*>(params.workspace.quantizedHidden + size_t(row) * intermediate + column) =
          quantized;
    }
  }
  // FC2 loads hidden through TMA; order every generic store of this task before its completion release.
  if (lastChunk) asm volatile("fence.proxy.async.global;" ::: "memory");
  cutlass::arch::NamedBarrier::sync(EpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
}

template <class KernelParams, class Storage, class Values, class Coordinates>
MSCCLPP_DEVICE_INLINE void packFc2Chunk(const KernelParams& params, const Task& task, Storage& s, const Values& values,
                                        const Coordinates& coordinates, int tokenOffset, int validRows) {
  using namespace cute;
  constexpr int KernelTileN = KernelParams::Tiles::N;
  constexpr int CtaTileM = KernelParams::Tiles::CtaM;

#pragma unroll
  for (int i = 0; i < size(values); ++i) {
    auto coord = coordinates(i);
    int token = int(get<1>(coord));
    float value = 0.0f;
    if (token < validRows)
      value = values(i) * params.workspace.routes[task.block * KernelTileN + tokenOffset + token].weight;
    s.epilogue.packed[token * CtaTileM + int(get<0>(coord)) % CtaTileM] = __bfloat16(value);
  }
}

template <int EpilogueWarps, class KernelParams, class Storage>
MSCCLPP_DEVICE_INLINE void storeOutputChunk(const KernelParams& params, const Task& task, Storage& s, int tokenOffset,
                                            int validRows, int hidden, int warp, int lane) {
  using Tiles = typename KernelParams::Tiles;
  constexpr int KernelTileN = Tiles::N;
  constexpr int CtaTileM = Tiles::CtaM;
  if (lane != 0) return;
  int feature = task.m * CtaTileM * Tiles::ClusterSize + (blockIdx.x % Tiles::ClusterSize) * CtaTileM;
  if (feature >= hidden) return;
  for (int token = warp; token < validRows; token += EpilogueWarps) {
    Route route = params.workspace.routes[task.block * KernelTileN + tokenOffset + token];
    auto output = peerAt<__bfloat16>(params, route.rank, params.symmetric.partialOutput);
    size_t offset = (size_t(route.token) * params.config.topK + route.slot) * params.config.hidden + feature;
    bulkStore(output + offset, s.epilogue.packed + token * CtaTileM, CtaTileM * sizeof(__bfloat16));
    bulkStoreCommit();
    bulkStoreWaitSource<4>();
  }
  bulkStoreWaitSource();
}

#endif  // MSCCLPP_BULK_AVAILABLE

template <class Schedule, class KernelParams, class Storage, class Pipeline, class Accumulators>
MSCCLPP_DEVICE_INLINE void epilogue(const KernelParams& params, const Task& task, Storage& s, Pipeline& pipeline,
                                    typename Pipeline::PipelineState& state, Accumulators accumulators, int hidden,
                                    int intermediate) {
#if MSCCLPP_BULK_AVAILABLE
  using namespace cute;
  using Tiles = typename KernelParams::Tiles;
  constexpr int KernelTileN = Tiles::N;
  constexpr int CtaTileM = Tiles::CtaM;
  constexpr int ChunkTokens = KernelParams::Collective::EpilogueTokens;
  constexpr int WarpSize = 32;
  constexpr int EpilogueWarps = Schedule::EpilogueEnd - Schedule::EpilogueBegin;
  constexpr int EpilogueThreads = EpilogueWarps * WarpSize;
  constexpr int ReaderGroups = EpilogueThreads / CtaTileM;
  constexpr int ReadTokens = ChunkTokens / ReaderGroups;
  int warp = threadIdx.x / WarpSize;
  int lane = threadIdx.x % WarpSize;
  auto matrix = accumulators(make_coord(_, _), _0{}, _0{}, state.index());
  CUTE_STATIC_ASSERT_V(size<0>(matrix) == Int<CtaTileM>{});
  CUTE_STATIC_ASSERT_V(size<1>(matrix) == Int<KernelTileN>{});
  traceKernel(KernelTracePhase::EpilogueReady, true, int(task.fc1));
  pipeline.consumer_wait(state);
  traceKernel(KernelTracePhase::EpilogueReady, false, int(task.fc1));
  for (int tokenOffset = 0; tokenOffset < task.tokens.rows; tokenOffset += ChunkTokens) {
    int validRows = min(int(ChunkTokens), task.tokens.rows - tokenOffset);
    int readerGroup = threadIdx.x / CtaTileM;
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
    copy(copyOp, source, values);
    cutlass::arch::fence_view_async_tmem_load();
    if (tokenOffset + ChunkTokens >= task.tokens.rows) pipeline.consumer_release(state);
    if (task.fc1) {
      traceKernel(KernelTracePhase::ActivationQuantize, true, tokenOffset);
      activateFc1Chunk<EpilogueThreads, ChunkTokens>(params, task, s, values, coordinates, tokenOffset, validRows,
                                                     intermediate, tokenOffset + ChunkTokens >= task.tokens.rows);
      traceKernel(KernelTracePhase::ActivationQuantize, false, tokenOffset);
      continue;
    }
    packFc2Chunk(params, task, s, values, coordinates, tokenOffset, validRows);
    bulkFence();
    cutlass::arch::NamedBarrier::sync(EpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    traceKernel(KernelTracePhase::Fc2Store, true, tokenOffset);
    storeOutputChunk<EpilogueWarps>(params, task, s, tokenOffset, validRows, hidden, warp, lane);
    traceKernel(KernelTracePhase::Fc2Store, false, tokenOffset);
    cutlass::arch::NamedBarrier::sync(EpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
  }
  ++state;
  if (task.fc1 && threadIdx.x == 0) {
    auto* counter = params.workspace.hiddenReady + size_t(task.block) * W4ReadyCounterStride;
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(counter), "r"(uint32_t(1)) : "memory");
  }
#endif
}

template <class Schedule, class KernelParams, class Storage, class Accumulate, class Accumulators>
MSCCLPP_DEVICE_INLINE void runEpilogueRole(const KernelParams& params, Storage& s, Accumulate& pipeline,
                                           Accumulators accumulators, int hidden, int intermediate, int cluster,
                                           int tasks) {
  cutlass::arch::warpgroup_reg_alloc<KernelParams::Collective::EpilogueRegisters>();
  typename Accumulate::PipelineState state;
  for (int i = cluster; i < tasks; i += gridDim.x / KernelParams::Tiles::ClusterSize)
    epilogue<Schedule>(params, taskAt(params, i, hidden, intermediate), s, pipeline, state, accumulators, hidden,
                       intermediate);
#if MSCCLPP_BULK_AVAILABLE
  if (threadIdx.x % warpSize == 0) bulkStoreWait();
#endif
}

template <int Elements>
struct ResultAccumulator {
  float values[Elements] = {};
};

template <class Element, int Elements>
struct ResultVectorOps {
  MSCCLPP_DEVICE_INLINE static void accumulate(ResultAccumulator<Elements>& accumulator, const Element* source) {
#pragma unroll
    for (int element = 0; element < Elements; ++element) accumulator.values[element] += float(source[element]);
  }

  MSCCLPP_DEVICE_INLINE static void store(Element* destination, const ResultAccumulator<Elements>& accumulator) {
#pragma unroll
    for (int element = 0; element < Elements; ++element) destination[element] = Element(accumulator.values[element]);
  }
};

template <>
struct ResultVectorOps<__bfloat16, 8> {
  MSCCLPP_DEVICE_INLINE static void accumulate(ResultAccumulator<8>& accumulator, const __bfloat16* source) {
    uint32_t word0, word1, word2, word3;
    asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(word0), "=r"(word1), "=r"(word2), "=r"(word3)
                 : "l"(source)
                 : "memory");
    accumulator.values[0] += float(mscclpp::bit_cast<__bfloat16>(uint16_t(word0)));
    accumulator.values[1] += float(mscclpp::bit_cast<__bfloat16>(uint16_t(word0 >> 16)));
    accumulator.values[2] += float(mscclpp::bit_cast<__bfloat16>(uint16_t(word1)));
    accumulator.values[3] += float(mscclpp::bit_cast<__bfloat16>(uint16_t(word1 >> 16)));
    accumulator.values[4] += float(mscclpp::bit_cast<__bfloat16>(uint16_t(word2)));
    accumulator.values[5] += float(mscclpp::bit_cast<__bfloat16>(uint16_t(word2 >> 16)));
    accumulator.values[6] += float(mscclpp::bit_cast<__bfloat16>(uint16_t(word3)));
    accumulator.values[7] += float(mscclpp::bit_cast<__bfloat16>(uint16_t(word3 >> 16)));
  }

  MSCCLPP_DEVICE_INLINE static uint32_t pack(float low, float high) {
    return uint32_t(mscclpp::bit_cast<uint16_t>(__bfloat16(low))) |
           uint32_t(mscclpp::bit_cast<uint16_t>(__bfloat16(high))) << 16;
  }

  MSCCLPP_DEVICE_INLINE static void store(__bfloat16* destination, const ResultAccumulator<8>& accumulator) {
    if ((reinterpret_cast<uintptr_t>(destination) & 15) != 0) {
#pragma unroll
      for (int element = 0; element < 8; ++element) destination[element] = __bfloat16(accumulator.values[element]);
      return;
    }
    uint32_t word0 = pack(accumulator.values[0], accumulator.values[1]);
    uint32_t word1 = pack(accumulator.values[2], accumulator.values[3]);
    uint32_t word2 = pack(accumulator.values[4], accumulator.values[5]);
    uint32_t word3 = pack(accumulator.values[6], accumulator.values[7]);
    asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(destination), "r"(word0), "r"(word1), "r"(word2), "r"(word3)
                 : "memory");
  }
};

template <class Element, int Elements>
MSCCLPP_DEVICE_INLINE void reduceResultVector(const Element* partial, int rowStride, int topK, Element* output,
                                              const int* routeIds = nullptr) {
  ResultAccumulator<Elements> accumulator;
  for (int slot = 0; slot < topK; ++slot) {
    if (routeIds && routeIds[slot] < 0) continue;
    ResultVectorOps<Element, Elements>::accumulate(accumulator, partial + size_t(slot) * rowStride);
  }
  ResultVectorOps<Element, Elements>::store(output, accumulator);
}

template <class KernelParams>
MSCCLPP_DEVICE_INLINE void combineResults(const KernelParams& params, int tokens, __bfloat16* output) {
  constexpr int Elements = 8;
  auto partial = at<__bfloat16>(params.local, params.symmetric.partialOutput);
  auto routeIds = at<int>(params.local, params.symmetric.topkIds);
  for (size_t i = blockIdx.x * KernelParams::ThreadCount + threadIdx.x;
       i < size_t(tokens) * (params.config.hidden / Elements); i += gridDim.x * KernelParams::ThreadCount) {
    size_t token = i / (params.config.hidden / Elements);
    int feature = Elements * (i % (params.config.hidden / Elements));
    auto source = partial + size_t(token) * params.config.topK * params.config.hidden + feature;
    reduceResultVector<__bfloat16, Elements>(source, params.config.hidden, params.config.topK, output + Elements * i,
                                             routeIds + size_t(token) * params.config.topK);
  }
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_EPILOGUE_CUH_
