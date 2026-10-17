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

// Token-major post-SwiGLU FC1 scratch [ChunkTokens, 64].
template <int CtaFc1M>
MSCCLPP_DEVICE_INLINE int fc1ScratchIndex(int token, int feature) {
  static_assert(CtaFc1M == 64);
  int swizzle = ((token & 3) << 2) | (((token >> 3) & 1) << 4);
  return token * CtaFc1M + (feature ^ swizzle);
}

template <int EpilogueThreads, int ChunkTokens, class KernelParams, class Storage, class Values, class Coordinates>
MSCCLPP_DEVICE_INLINE void activateFc1Chunk(const KernelParams& params, const Task& task, Storage& s,
                                            const Values& values, const Coordinates&, int tokenOffset, int validRows,
                                            int intermediate, bool lastChunk) {
  using namespace cute;
  using Tiles = typename KernelParams::Tiles;
  constexpr int KernelTileN = Tiles::N;
  constexpr int CtaTileM = Tiles::CtaM;
  constexpr int CtaFc1M = Tiles::CtaFc1M;
  constexpr int ScaleGroupSize = 32;
  constexpr int ReaderGroups = EpilogueThreads / CtaTileM;
  constexpr int ReadTokens = ChunkTokens / ReaderGroups;
  constexpr int HalfTokens = ReadTokens / 2;
  // Each thread activates consecutive features of one token; ThreadsPerScale lanes share one E8M0 scale.
  constexpr int Features = ChunkTokens * CtaFc1M / EpilogueThreads;
  constexpr int ThreadsPerToken = CtaFc1M / Features;
  constexpr int ThreadsPerScale = ScaleGroupSize / Features;
  static_assert(CtaFc1M == 64 && HalfTokens % 2 == 0 && (Features == 4 || Features == 8 || Features == 16));
  using Activations = mscclpp::VectorType<float, Features>;

  // Lanes 0..15 hold gate and lanes 16..31 hold up for the same 16 features and ReadTokens tokens.
  // Pair them across offset 16 to evaluate SwiGLU in registers before writing half-width scratch.
  int lane = threadIdx.x % 32;
  int warpInGroup = (threadIdx.x % CtaTileM) / 32;
  int readerGroup = threadIdx.x / CtaTileM;
  bool isUp = (lane & 16) != 0;
  int outFeature = warpInGroup * 16 + (lane & 15);
  int baseToken = readerGroup * ReadTokens + (isUp ? HalfTokens : 0);
  float gate[HalfTokens], up[HalfTokens];
#pragma unroll
  for (int i = 0; i < HalfTokens; ++i) {
    float mySend = isUp ? values(i) : values(i + HalfTokens);
    float other = __shfl_xor_sync(0xffffffff, mySend, 16);
    gate[i] = isUp ? other : values(i);
    up[i] = isUp ? values(i + HalfTokens) : other;
  }
  if (float clamp = params.config.gateUpClamp; clamp >= 0) {
#pragma unroll
    for (int i = 0; i < HalfTokens; ++i) {
      gate[i] = fminf(gate[i], clamp);
      up[i] = fmaxf(-clamp, fminf(up[i], clamp));
    }
  }
  if (tokenOffset > 0)
    cutlass::arch::NamedBarrier::sync(EpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
#pragma unroll
  for (int i = 0; i < HalfTokens; i += 2) {
    float2 pair = swigluPair(make_float2(gate[i], gate[i + 1]), make_float2(up[i], up[i + 1]));
    s.epilogue.scratch[fc1ScratchIndex<CtaFc1M>(baseToken + i, outFeature)] = pair.x;
    s.epilogue.scratch[fc1ScratchIndex<CtaFc1M>(baseToken + i + 1, outFeature)] = pair.y;
  }
  cutlass::arch::NamedBarrier::sync(EpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

  int token = threadIdx.x / ThreadsPerToken;
  // Warps holding only padded tokens skip the work; scale shuffles never cross a warp.
  if (threadIdx.x / 32 * (32 / ThreadsPerToken) < validRows) {
    int feature = threadIdx.x % ThreadsPerToken * Features;
    Activations activated;
#pragma unroll
    for (int i = 0; i < Features / 4; ++i) {
      reinterpret_cast<mscclpp::f32x4*>(&activated)[i] = *reinterpret_cast<const mscclpp::f32x4*>(
          s.epilogue.scratch + fc1ScratchIndex<CtaFc1M>(token, feature + 4 * i));
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
  if (lastChunk) {
    asm volatile("fence.proxy.async.global;" ::: "memory");
    cutlass::arch::NamedBarrier::sync(EpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
  }
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

// Scales a warp's 32 FC2 features x ReadTokens tokens by route weights in pairs and writes token-major BF16 rows
// with transposing stmatrix. Padded tokens are never stored, so they are not masked.
template <int ReadTokens, class KernelParams, class Storage>
MSCCLPP_DEVICE_INLINE void packFc2Chunk(const KernelParams& params, const Task& task, Storage& s,
                                        const uint32_t (&accumulator)[2][ReadTokens / 2], int tokenOffset,
                                        int readerGroup) {
  constexpr int KernelTileN = KernelParams::Tiles::N;
  static_assert(KernelParams::Tiles::CtaM == 128 && ReadTokens % 8 == 0);
  int lane = threadIdx.x % 32;
  int firstToken = readerGroup * ReadTokens;
  const Route* routes = params.workspace.routes + task.block * KernelTileN + tokenOffset + firstToken;
  // Matrix m of each x4 store holds the warp's features [8m, 8m + 8); lane 8m + j addresses token row j.
  auto* rows = s.epilogue.packed + (firstToken + lane % 8) * PackedRowStride + threadIdx.x / 32 % 4 * 32 + lane / 8 * 8;
#pragma unroll
  for (int group = 0; group < ReadTokens / 8; ++group) {
    int token = group * 8 + lane % 4 * 2;
    float2 weights = make_float2(routes[token].weight, routes[token + 1].weight);
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

template <int EpilogueWarps, class KernelParams, class Storage>
MSCCLPP_DEVICE_INLINE void storeOutputChunk(const KernelParams& params, const Task& task, Storage& s, int tokenOffset,
                                            int validRows, int hidden, int warp, int lane) {
  using Tiles = typename KernelParams::Tiles;
  constexpr int KernelTileN = Tiles::N;
  constexpr int CtaTileM = Tiles::CtaM;
  constexpr int WarpTokens = (KernelParams::Collective::EpilogueTokens + EpilogueWarps - 1) / EpilogueWarps;
  static_assert(WarpTokens <= 32);
  int feature = task.m * CtaTileM * Tiles::ClusterSize + (blockIdx.x % Tiles::ClusterSize) * CtaTileM;
  if (feature >= hidden) return;
  // Lane l of warp w stores token w + l * EpilogueWarps with its own bulk copy, so route and peer lookups run in
  // parallel; each issuing lane owns its bulk group and drains the shared-memory reads before the next chunk.
  int token = warp + lane * EpilogueWarps;
  if (lane < WarpTokens && token < validRows) {
    Route route = params.workspace.routes[task.block * KernelTileN + tokenOffset + token];
    __bfloat16* output;
    size_t offset;
    if (params.config.worldSize == 1 && params.config.topK == 1) {
      // Delivered also finalizes a unique local contribution by TMA store; BF16 TMA reduction is only needed
      // when another contribution targets the same token.
      output = params.finalOutput;
      offset = size_t(route.token) * hidden + feature;
    } else {
      output = peerAt<__bfloat16>(params, route.rank, params.symmetric.partialOutput);
      offset = (size_t(route.token) * params.config.topK + route.slot) * params.config.hidden + feature;
    }
    bulkStore(output + offset, s.epilogue.packed + token * PackedRowStride, CtaTileM * sizeof(__bfloat16));
    bulkStoreCommit();
  }
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
    bool lastChunk = tokenOffset + ChunkTokens >= task.tokens.rows;
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
      copy(copyOp, source, values);
      cutlass::arch::fence_view_async_tmem_load();
      if (lastChunk) pipeline.consumer_release(state);
      traceKernel(KernelTracePhase::ActivationQuantize, true, tokenOffset);
      activateFc1Chunk<EpilogueThreads, ChunkTokens>(params, task, s, values, coordinates, tokenOffset, validRows,
                                                     intermediate, lastChunk);
      traceKernel(KernelTracePhase::ActivationQuantize, false, tokenOffset);
      continue;
    }
    // FC2 reads both 16-lane halves of the warp's TMEM lane quarter as stmatrix fragments.
    uint32_t accumulator[2][ReadTokens / 2];
    uint32_t address = raw_pointer_cast(matrix.data()) + tokenOffset + readerGroup * ReadTokens;
#pragma unroll
    for (int half = 0; half < 2; ++half)
      loadFc2Accumulators<ReadTokens / 8>(address + uint32_t((warp % 4 * 32 + half * 16) << 16), accumulator[half]);
    cutlass::arch::fence_view_async_tmem_load();
    if (lastChunk) pipeline.consumer_release(state);
    bulkStoreWaitSource();
    cutlass::arch::NamedBarrier::sync(EpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    if (readerGroup * ReadTokens < validRows)
      packFc2Chunk<ReadTokens>(params, task, s, accumulator, tokenOffset, readerGroup);
    bulkFence();
    cutlass::arch::NamedBarrier::sync(EpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    traceKernel(KernelTracePhase::Fc2Store, true, tokenOffset);
    storeOutputChunk<EpilogueWarps>(params, task, s, tokenOffset, validRows, hidden, warp, lane);
    traceKernel(KernelTracePhase::Fc2Store, false, tokenOffset);
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
  // Every lane may own FC2 bulk stores; they must complete before the CTA publishes its completion.
  bulkStoreWait();
#endif
}

// 16-byte partial loads each combine thread keeps in flight; enough to cover memory latency at full bandwidth.
constexpr int CombineLoads = 16;
// Vectors per lane per warp task; more keeps their FP32 conversions live past the 128-register budget.
constexpr int CombineItems = 8;
// Widest slot batch. Only it loops over batches for larger topK, carrying FP32 sums between them.
constexpr int CombineSlots = 8;

MSCCLPP_DEVICE_INLINE void storeCombined(__bfloat16* destination, const mscclpp::f32x8& sums, bool aligned) {
  auto values = mscclpp::to<mscclpp::bf16x8>(sums);
  if (aligned) {
    *reinterpret_cast<uint4*>(destination) = values.storage;
    return;
  }
#pragma unroll
  for (int element = 0; element < values.Size; ++element) destination[element] = values.data[element];
}

// Sums the routed partial rows of every token in FP32 slot order and rounds once to BF16. A warp task covers Items
// consecutive warp-wide vectors of one token row; each lane issues all Slots * Items 16-byte loads before accumulating.
// Narrower batches require topK <= Slots, so their sums never stay live across loads.
template <int Hidden, int Slots, bool ProcessMultipleSlotBatches = false, class KernelParams>
MSCCLPP_DEVICE_INLINE void combineSlots(const KernelParams& params, int tokens, __bfloat16* output) {
  constexpr int WarpSize = 32;
  constexpr int Elements = mscclpp::bf16x8::Size;
  constexpr int Vectors = Hidden / Elements;
  constexpr int Items = CombineLoads / Slots < CombineItems ? CombineLoads / Slots : CombineItems;
  constexpr int TaskVectors = Items * WarpSize;
  constexpr int TokenTasks = (Vectors + TaskVectors - 1) / TaskVectors;
  static_assert(Hidden % Elements == 0 && CombineLoads % Slots == 0 && Slots <= CombineSlots);
  const int topK = params.config.topK;
  auto partial = reinterpret_cast<const uint4*>(at<__bfloat16>(params.local, params.symmetric.partialOutput));
  auto routeIds = at<int>(params.local, params.symmetric.topkIds);
  bool aligned = (reinterpret_cast<uintptr_t>(output) & 15) == 0;
  int lane = threadIdx.x % WarpSize;
  int warps = gridDim.x * (KernelParams::ThreadCount / WarpSize);
  int tasks = tokens * TokenTasks;
  // Route ids of the first slot batch are loaded one task ahead, so partial loads never wait on them.
  auto loadIds = [&](int task, int(&ids)[Slots]) {
#pragma unroll
    for (int slot = 0; slot < Slots; ++slot)
      ids[slot] = task < tasks && slot < topK ? routeIds[task / TokenTasks * topK + slot] : -1;
  };
  int task = blockIdx.x * (KernelParams::ThreadCount / WarpSize) + threadIdx.x / WarpSize;
  int nextIds[Slots];
  loadIds(task, nextIds);
  for (; task < tasks; task += warps) {
    int token = task / TokenTasks;
    int first = task % TokenTasks * TaskVectors + lane;
    int taskIds[Slots];
#pragma unroll
    for (int slot = 0; slot < Slots; ++slot) taskIds[slot] = nextIds[slot];
    loadIds(task + warps, nextIds);
    mscclpp::f32x8 sums[Items];
#pragma unroll
    for (int item = 0; item < Items; ++item) {
#pragma unroll
      for (int element = 0; element < Elements; ++element) sums[item].data[element] = 0.0f;
    }
    for (int base = 0; base < (ProcessMultipleSlotBatches ? topK : 1); base += Slots) {
      bool routed[Slots];
#pragma unroll
      for (int slot = 0; slot < Slots; ++slot)
        routed[slot] = !ProcessMultipleSlotBatches || base == 0
                           ? taskIds[slot] >= 0
                           : base + slot < topK && routeIds[token * topK + base + slot] >= 0;
      // L2-only loads: peers write the partial rows during this kernel. Unrouted slots read as +0, which is the same
      // as skipping them: the sums start at +0, and round-to-nearest addition yields -0 only from two -0 operands.
      const uint4* source = partial + size_t(token * topK + base) * Vectors + first;
      mscclpp::bf16x8 values[Slots][Items];
#pragma unroll
      for (int slot = 0; slot < Slots; ++slot) {
#pragma unroll
        for (int item = 0; item < Items; ++item) {
          values[slot][item].storage = make_uint4(0, 0, 0, 0);
          if (routed[slot] && (Vectors % TaskVectors == 0 || first + item * WarpSize < Vectors))
            values[slot][item].storage = __ldcg(source + slot * Vectors + item * WarpSize);
        }
      }
#pragma unroll
      for (int item = 0; item < Items; ++item) {
#pragma unroll
        for (int slot = 0; slot < Slots; ++slot) {
          auto partialValues = mscclpp::to<mscclpp::f32x8>(values[slot][item]);
#pragma unroll
          for (int element = 0; element < Elements; ++element) sums[item].data[element] += partialValues.data[element];
        }
      }
    }
#pragma unroll
    for (int item = 0; item < Items; ++item) {
      if (Vectors % TaskVectors == 0 || first + item * WarpSize < Vectors)
        storeCombined(output + (size_t(token) * Vectors + first + item * WarpSize) * Elements, sums[item], aligned);
    }
  }
}

template <int Hidden, class KernelParams>
MSCCLPP_DEVICE_INLINE bool hasCombineWork(const KernelParams& params, int tokens) {
  constexpr int WarpSize = 32;
  constexpr int Vectors = Hidden / mscclpp::bf16x8::Size;
  const int topK = params.config.topK;
  const int slots = topK == 1 ? 1 : (topK == 2 ? 2 : (topK <= 4 ? 4 : CombineSlots));
  const int items = CombineLoads / slots < CombineItems ? CombineLoads / slots : CombineItems;
  const int tokenTasks = (Vectors + items * WarpSize - 1) / (items * WarpSize);
  return blockIdx.x * (KernelParams::ThreadCount / WarpSize) < tokens * tokenTasks;
}

template <int Hidden, class KernelParams>
MSCCLPP_DEVICE_INLINE void combineResults(const KernelParams& params, int tokens, __bfloat16* output) {
  const int topK = params.config.topK;
  if (topK == 1)
    combineSlots<Hidden, 1>(params, tokens, output);
  else if (topK == 2)
    combineSlots<Hidden, 2>(params, tokens, output);
  else if (topK <= 4)
    combineSlots<Hidden, 4>(params, tokens, output);
  else if (topK <= CombineSlots)
    combineSlots<Hidden, CombineSlots>(params, tokens, output);
  else
    combineSlots<Hidden, CombineSlots, true>(params, tokens, output);
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_EPILOGUE_CUH_
