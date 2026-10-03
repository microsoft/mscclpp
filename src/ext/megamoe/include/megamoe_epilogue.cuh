// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_EPILOGUE_CUH_
#define MSCCLPP_EXT_MEGAMOE_EPILOGUE_CUH_

#include "megamoe_device.cuh"
#include "megamoe_quantization.cuh"
#include "megamoe_trace.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

__device__ __forceinline__ float swiglu(float gate, float up, float probability, float clamp) {
  if (clamp >= 0) {
    gate = fminf(gate, clamp);
    up = fmaxf(-clamp, fminf(up, clamp));
  }
  float product, exponent, denominator, inverse, activated, weighted;
  // Preserve the reference's (up * gate) * sigmoid(gate) * probability association.
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(product) : "f"(up), "f"(gate));
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(exponent) : "f"(gate), "f"(-1.4426950408889634f));
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(denominator) : "f"(exponent));
  asm("add.rn.f32 %0, %1, 0f3f800000;" : "=f"(denominator) : "f"(denominator));
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inverse) : "f"(denominator));
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(activated) : "f"(product), "f"(inverse));
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(weighted) : "f"(activated), "f"(probability));
  return weighted;
}

#if MSCCLPP_BULK_AVAILABLE

template <int LocalMode, int EpilogueThreads, int ChunkTokens, class P, class Storage, class Values, class Coordinates>
__device__ __forceinline__ void activateFc1Chunk(const P& p, const Task& task, Storage& s, const Values& values,
                                                 const Coordinates& coordinates, int tokenOffset, int validRows,
                                                 int intermediate) {
  using namespace cute;
  constexpr bool Local = LocalMode != 0;
  using Tiles = typename P::Tiles;
  constexpr int KernelTileN = Tiles::N;
  constexpr int CtaTileM = Tiles::CtaM;
  constexpr int CtaFc1M = Tiles::CtaFc1M;
  constexpr int GateUpGroupSize = 16;
  constexpr int GateUpPairSize = 2 * GateUpGroupSize;

  CUTE_UNROLL
  for (int i = 0; i < size(values); ++i) {
    auto coord = coordinates(i);
    s.epilogue.scratch[(int(get<0>(coord)) % CtaTileM) * (ChunkTokens + 1) + int(get<1>(coord))] = values(i);
  }
  cutlass::arch::NamedBarrier::sync(EpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

  if constexpr (P::WeightMxfp4) {
    static_assert(CtaFc1M % 32 == 0);
    constexpr int GroupsPerRow = CtaFc1M / 32;
    constexpr int EpilogueWarps = EpilogueThreads / 32;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    for (int group = warp; group < validRows * GroupsPerRow; group += EpilogueWarps) {
      int token = group / GroupsPerRow;
      int feature = group % GroupsPerRow * 32 + lane;
      int gateRow = feature / GateUpGroupSize * GateUpPairSize + feature % GateUpGroupSize;
      float gate = s.epilogue.scratch[gateRow * (ChunkTokens + 1) + token];
      float up = s.epilogue.scratch[(gateRow + GateUpGroupSize) * (ChunkTokens + 1) + token];
      float activated = swiglu(gate, up, 1.0f, p.config.gateUpClamp);
      float maximum = fabsf(activated);
      CUTE_UNROLL
      for (int offset = 16; offset > 0; offset /= 2)
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
      uint8_t scale = quantizeE8M0Scale(maximum);
      float other = __shfl_xor_sync(0xffffffff, activated, 1);
      int row = w4StorageRow(task.block * KernelTileN + tokenOffset + token);
      int column = task.m * Tiles::Fc1M + (blockIdx.x % ClusterM) * CtaFc1M + feature;
      if (lane == 0) p.workspace.hiddenScale[p.fc2.layout_SFB(make_coord(row, column, 0))] = scale;
      if (lane % 2 == 0) {
        auto destination =
            reinterpret_cast<uint16_t*>(p.workspace.quantizedHidden + size_t(row) * intermediate + column);
        *destination = quantizeE4M3Pair(activated, other, inverseE8M0Scale(scale));
      }
    }
    asm volatile("fence.proxy.async.global;" ::: "memory");
    cutlass::arch::NamedBarrier::sync(EpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
  } else {
    constexpr int FoldedValuesPerThread = ChunkTokens * CtaFc1M / EpilogueThreads;
    static_assert(ChunkTokens * CtaFc1M % EpilogueThreads == 0);
    CUTE_UNROLL
    for (int j = 0; j < FoldedValuesPerThread; ++j) {
      int i = threadIdx.x + j * EpilogueThreads;
      int token = i / CtaFc1M;
      int feature = i % CtaFc1M;
      if (token < validRows) {
        int gateRow = feature / GateUpGroupSize * GateUpPairSize + feature % GateUpGroupSize;
        float gate = s.epilogue.scratch[gateRow * (ChunkTokens + 1) + token];
        float up = s.epilogue.scratch[(gateRow + GateUpGroupSize) * (ChunkTokens + 1) + token];
        int row = task.block * KernelTileN + tokenOffset + token;
        float probability = 1.0f;
        bool active = true;
        if constexpr (LocalMode == 1) {
          int id = at<int>(p.local, p.symmetric.topkIds)[row];
          if (id < -1 || id > 0) {
            printf("MegaMoE local expert: invalid routing ID %d for token %d\n", id, row);
            __trap();
          }
          active = id == 0;
          if (active) probability = at<float>(p.local, p.symmetric.topkWeights)[row];
        } else if constexpr (LocalMode == 0) {
          probability = p.workspace.routes[row].weight;
        }
        __bfloat16 folded = active ? __bfloat16(swiglu(gate, up, probability, p.config.gateUpClamp)) : __bfloat16(0.0f);
        if constexpr (Local) {
          int column = task.m * Tiles::Fc1M + (blockIdx.x % ClusterM) * CtaFc1M + feature;
          p.workspace.hidden[size_t(row) * intermediate + column] = folded;
        } else {
          s.epilogue.packed[i] = folded;
        }
      }
    }

    if constexpr (Local) {
      // Publish ordinary global stores to the async proxy before FC2's TMA load.
      asm volatile("fence.proxy.async.global;" ::: "memory");
      cutlass::arch::NamedBarrier::sync(EpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    }
  }
}

template <int LocalMode, class P, class Storage, class Values, class Coordinates>
__device__ __forceinline__ void packFc2Chunk(const P& p, const Task& task, Storage& s, const Values& values,
                                             const Coordinates& coordinates, int tokenOffset, int validRows) {
  using namespace cute;
  constexpr int KernelTileN = P::Tiles::N;
  constexpr int CtaTileM = P::Tiles::CtaM;

  CUTE_UNROLL
  for (int i = 0; i < size(values); ++i) {
    auto coord = coordinates(i);
    int token = int(get<1>(coord));
    float value = values(i);
    if constexpr (P::WeightMxfp4) {
      value = 0.0f;
      if (token < validRows)
        value = values(i) * p.workspace.routes[task.block * KernelTileN + tokenOffset + token].weight;
    } else if constexpr (LocalMode == 1) {
      int row = task.block * KernelTileN + tokenOffset + token;
      if (token < validRows && at<int>(p.local, p.symmetric.topkIds)[row] != 0) value = 0.0f;
    }
    s.epilogue.packed[token * CtaTileM + int(get<0>(coord)) % CtaTileM] = __bfloat16(value);
  }
}

template <int LocalMode, int EpilogueWarps, class P, class Storage>
__device__ __forceinline__ void storeOutputChunk(const P& p, const Task& task, Storage& s, __bfloat16* directOutput,
                                                 int tokenOffset, int validRows, int hidden, int intermediate, int warp,
                                                 int lane) {
  constexpr bool Local = LocalMode != 0;
  using Tiles = typename P::Tiles;
  constexpr int KernelTileN = Tiles::N;
  constexpr int CtaTileM = Tiles::CtaM;
  constexpr int CtaFc1M = Tiles::CtaFc1M;
  constexpr int WarpSize = 32;

  if constexpr (Local) {
    if (!task.fc1 && (reinterpret_cast<uintptr_t>(directOutput) & 15) != 0) {
      int feature = task.m * Tiles::M + (blockIdx.x % ClusterM) * CtaTileM;
      if (feature < hidden) {
        for (int token = warp; token < validRows; token += EpilogueWarps) {
          size_t offset = size_t(task.block * KernelTileN + tokenOffset + token) * hidden + feature;
          auto* destination = directOutput + offset;
          auto* source = s.epilogue.packed + token * CtaTileM;
          // Preserve support for contiguous BF16 views with unaligned storage offsets.
          mscclpp::detail::copy<__bfloat16>(destination, source, CtaTileM, lane, WarpSize);
        }
      }
      return;
    }
  }

  if (lane != 0) return;
  bool fc1 = !P::WeightMxfp4 && task.fc1;
  int width = fc1 ? CtaFc1M : CtaTileM;
  int feature = task.m * width * ClusterM + (blockIdx.x % ClusterM) * width;
  if constexpr (P::WeightMxfp4) {
    if (feature >= hidden) return;
  }
  for (int token = warp; token < validRows; token += EpilogueWarps) {
    if (fc1) {
      auto output =
          p.workspace.hidden + size_t(task.block * KernelTileN + tokenOffset + token) * intermediate + feature;
      bulkStore(output, s.epilogue.packed + token * width, width * sizeof(__bfloat16));
    } else if (feature < hidden) {
      if constexpr (Local) {
        size_t offset = size_t(task.block * KernelTileN + tokenOffset + token) * hidden + feature;
        bulkStore(directOutput + offset, s.epilogue.packed + token * width, width * sizeof(__bfloat16));
      } else {
        Route route = p.workspace.routes[task.block * KernelTileN + tokenOffset + token];
        auto output = peerAt<__bfloat16>(p, route.rank, p.symmetric.partialOutput);
        size_t offset = (size_t(route.token) * p.config.topK + route.slot) * p.config.hidden + feature;
        bulkStore(output + offset, s.epilogue.packed + token * width, width * sizeof(__bfloat16));
      }
    }
    bulkStoreCommit();
    bulkStoreWaitSource<4>();
  }
  if constexpr (Local || P::WeightMxfp4) {
    bulkStoreWaitSource();
  } else {
    bulkStoreWait();
  }
}

#endif

template <int LocalMode, class Schedule, class P, class Storage, class Pipeline, class Accumulators>
__device__ __forceinline__ void epilogue(const P& p, const Task& task, Storage& s, Pipeline& pipeline,
                                         typename Pipeline::PipelineState& state, Accumulators accumulators,
                                         __bfloat16* directOutput, int hidden, int intermediate) {
#if MSCCLPP_BULK_AVAILABLE
  using namespace cute;
  constexpr bool Local = LocalMode != 0;
  using Tiles = typename P::Tiles;
  constexpr int KernelTileN = Tiles::N;
  constexpr int CtaTileM = Tiles::CtaM;
  constexpr int ChunkTokens = P::WeightMxfp4 ? W4EpilogueTokens : EpilogueTokens;
  constexpr int WarpSize = 32;
  constexpr int EpilogueWarps = Schedule::EpilogueEnd - Schedule::EpilogueBegin;
  constexpr int EpilogueThreads = EpilogueWarps * WarpSize;
  int warp = threadIdx.x / WarpSize;
  int lane = threadIdx.x % WarpSize;
  auto matrix = accumulators(make_coord(_, _), _0{}, _0{}, state.index());
  CUTE_STATIC_ASSERT_V(size<0>(matrix) == Int<CtaTileM>{});
  CUTE_STATIC_ASSERT_V(size<1>(matrix) == Int<KernelTileN>{});
  if constexpr (P::WeightMxfp4) traceW4(W4TracePhase::EpilogueReady, true, int(task.fc1));
  pipeline.consumer_wait(state);
  if constexpr (P::WeightMxfp4) traceW4(W4TracePhase::EpilogueReady, false, int(task.fc1));
  for (int tokenOffset = 0; tokenOffset < task.tokens.rows; tokenOffset += ChunkTokens) {
    int validRows = min(int(ChunkTokens), task.tokens.rows - tokenOffset);
    // Load one token chunk from TMEM into per-thread FP32 registers.
    auto acc =
        local_tile(matrix, make_shape(Int<CtaTileM>{}, Int<ChunkTokens>{}), make_coord(0, tokenOffset / ChunkTokens));
    auto copyOp = make_tmem_copy(SM100_TMEM_LOAD_32dp32b8x{}, acc);
    auto threadCopy = copyOp.get_slice(threadIdx.x);
    auto source = threadCopy.partition_S(acc);
    auto identity = make_identity_tensor(make_shape(Int<CtaTileM>{}, Int<ChunkTokens>{}));
    auto coordinates = threadCopy.partition_D(identity);
    auto values = make_tensor<float>(shape(coordinates));
    copy(copyOp, source, values);
    cutlass::arch::fence_view_async_tmem_load();
    if (tokenOffset + ChunkTokens >= task.tokens.rows) pipeline.consumer_release(state);
    if (task.fc1) {
      if constexpr (P::WeightMxfp4) traceW4(W4TracePhase::ActivationQuantize, true, tokenOffset);
      activateFc1Chunk<LocalMode, EpilogueThreads, ChunkTokens>(p, task, s, values, coordinates, tokenOffset, validRows,
                                                                intermediate);
      if constexpr (P::WeightMxfp4) traceW4(W4TracePhase::ActivationQuantize, false, tokenOffset);
      // Local and quantized FC1 write hidden directly instead of staging a bulk store.
      if constexpr (Local || P::WeightMxfp4) continue;
    } else {
      packFc2Chunk<LocalMode>(p, task, s, values, coordinates, tokenOffset, validRows);
    }
    // Publish the packed BF16 tile before warp leaders launch bulk stores.
    bulkFence();
    cutlass::arch::NamedBarrier::sync(EpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    if constexpr (P::WeightMxfp4) traceW4(W4TracePhase::Fc2Store, true, tokenOffset);
    storeOutputChunk<LocalMode, EpilogueWarps>(p, task, s, directOutput, tokenOffset, validRows, hidden, intermediate,
                                               warp, lane);
    if constexpr (P::WeightMxfp4) traceW4(W4TracePhase::Fc2Store, false, tokenOffset);
    cutlass::arch::NamedBarrier::sync(EpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
  }
  ++state;
  if (task.fc1 && threadIdx.x == 0) {
    constexpr int ReadyStride = P::WeightMxfp4 ? W4ReadyCounterStride : 1;
    auto* counter = p.workspace.hiddenReady + size_t(task.block) * ReadyStride;
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(counter), "r"(uint32_t(1)) : "memory");
  }
#endif
}

template <int Elements>
struct ResultAccumulator {
  float values[Elements] = {};
};

template <class Element, int Elements>
struct ResultVectorOps {
  __device__ __forceinline__ static void accumulate(ResultAccumulator<Elements>& accumulator, const Element* source) {
    CUTE_UNROLL
    for (int element = 0; element < Elements; ++element) accumulator.values[element] += float(source[element]);
  }

  __device__ __forceinline__ static void store(Element* destination, const ResultAccumulator<Elements>& accumulator) {
    CUTE_UNROLL
    for (int element = 0; element < Elements; ++element) destination[element] = Element(accumulator.values[element]);
  }
};

template <>
struct ResultVectorOps<__bfloat16, 8> {
  __device__ __forceinline__ static void accumulate(ResultAccumulator<8>& accumulator, const __bfloat16* source) {
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

  __device__ __forceinline__ static uint32_t pack(float low, float high) {
    return uint32_t(mscclpp::bit_cast<uint16_t>(__bfloat16(low))) |
           uint32_t(mscclpp::bit_cast<uint16_t>(__bfloat16(high))) << 16;
  }

  __device__ __forceinline__ static void store(__bfloat16* destination, const ResultAccumulator<8>& accumulator) {
    if ((reinterpret_cast<uintptr_t>(destination) & 15) != 0) {
      CUTE_UNROLL
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
__device__ __forceinline__ void reduceResultVector(const Element* partial, int rowStride, int topK, Element* output,
                                                   const int* routeIds = nullptr) {
  ResultAccumulator<Elements> accumulator;
  for (int slot = 0; slot < topK; ++slot) {
    if (routeIds && routeIds[slot] < 0) continue;
    ResultVectorOps<Element, Elements>::accumulate(accumulator, partial + size_t(slot) * rowStride);
  }
  ResultVectorOps<Element, Elements>::store(output, accumulator);
}

// All peer output stores must be complete and visible before this local reduction.
template <class P>
__device__ __forceinline__ void combineResults(const P& p, int tokens, __bfloat16* output) {
  constexpr int Elements = 8;
  auto partial = at<__bfloat16>(p.local, p.symmetric.partialOutput);
  auto routeIds = at<int>(p.local, p.symmetric.topkIds);
  for (size_t i = blockIdx.x * P::ThreadCount + threadIdx.x; i < size_t(tokens) * (p.config.hidden / Elements);
       i += gridDim.x * P::ThreadCount) {
    size_t token = i / (p.config.hidden / Elements);
    int feature = Elements * (i % (p.config.hidden / Elements));
    auto source = partial + size_t(token) * p.config.topK * p.config.hidden + feature;
    reduceResultVector<__bfloat16, Elements>(source, p.config.hidden, p.config.topK, output + Elements * i,
                                             routeIds + size_t(token) * p.config.topK);
  }
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_EPILOGUE_CUH_
