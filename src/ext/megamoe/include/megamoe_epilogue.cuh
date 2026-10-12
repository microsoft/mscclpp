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

__device__ __forceinline__ float2 swigluPair(float2 gate, float2 up, float clamp) {
  if (clamp >= 0) {
    gate.x = fminf(gate.x, clamp);
    gate.y = fminf(gate.y, clamp);
    up.x = fmaxf(-clamp, fminf(up.x, clamp));
    up.y = fmaxf(-clamp, fminf(up.y, clamp));
  }
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

template <int EpilogueThreads, int ChunkTokens, class P, class Storage, class Values, class Coordinates>
__device__ __forceinline__ void activateFc1Chunk(const P& p, const Task& task, Storage& s, const Values& values,
                                                 const Coordinates& coordinates, int tokenOffset, int validRows,
                                                 int intermediate) {
  using namespace cute;
  using Tiles = typename P::Tiles;
  constexpr int KernelTileN = Tiles::N;
  constexpr int CtaTileM = Tiles::CtaM;
  constexpr int CtaFc1M = Tiles::CtaFc1M;
  constexpr int GateUpGroupSize = 16;
  constexpr int GateUpPairSize = 2 * GateUpGroupSize;

#pragma unroll
  for (int i = 0; i < size(values); ++i) {
    auto coord = coordinates(i);
    s.epilogue.scratch[int(get<1>(coord)) * CtaTileM + int(get<0>(coord))] = values(i);
  }
  cutlass::arch::NamedBarrier::sync(EpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

  static_assert(CtaFc1M == 64);
  constexpr int EpilogueWarps = EpilogueThreads / 32;
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  for (int token = warp; token < validRows; token += EpilogueWarps) {
    int feature = lane / 16 * 32 + lane % 16 * 2;
    int gateRow = feature / GateUpGroupSize * GateUpPairSize + feature % GateUpGroupSize;
    auto* gateAddress = s.epilogue.scratch + token * CtaTileM + gateRow;
    auto* upAddress = gateAddress + GateUpGroupSize;
    float2 activated = swigluPair(*reinterpret_cast<const float2*>(gateAddress),
                                  *reinterpret_cast<const float2*>(upAddress), p.config.gateUpClamp);
    float maximum = fmaxf(fabsf(activated.x), fabsf(activated.y));
#pragma unroll
    for (int offset = 8; offset > 0; offset /= 2)
      maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset, 16));
    uint8_t scale = quantizeE8M0Scale(maximum);
    int row = w4StorageRow<typename P::Tiles>(task.block * KernelTileN + tokenOffset + token);
    int column = task.m * Tiles::Fc1M + (blockIdx.x % Tiles::ClusterSize) * CtaFc1M + feature;
    if (lane % 16 == 0) p.workspace.hiddenScale[p.fc2.layout_SFB(make_coord(row, column, 0))] = scale;
    auto destination = reinterpret_cast<uint16_t*>(p.workspace.quantizedHidden + size_t(row) * intermediate + column);
    *destination = quantizeE4M3Pair(activated.x, activated.y, inverseE8M0Scale(scale));
  }
  asm volatile("fence.proxy.async.global;" ::: "memory");
  cutlass::arch::NamedBarrier::sync(EpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
}

template <class P, class Storage, class Values, class Coordinates>
__device__ __forceinline__ void packFc2Chunk(const P& p, const Task& task, Storage& s, const Values& values,
                                             const Coordinates& coordinates, int tokenOffset, int validRows) {
  using namespace cute;
  constexpr int KernelTileN = P::Tiles::N;
  constexpr int CtaTileM = P::Tiles::CtaM;

#pragma unroll
  for (int i = 0; i < size(values); ++i) {
    auto coord = coordinates(i);
    int token = int(get<1>(coord));
    float value = 0.0f;
    if (token < validRows)
      value = values(i) * p.workspace.routes[task.block * KernelTileN + tokenOffset + token].weight;
    s.epilogue.packed[token * CtaTileM + int(get<0>(coord)) % CtaTileM] = __bfloat16(value);
  }
}

template <int EpilogueWarps, class P, class Storage>
__device__ __forceinline__ void storeOutputChunk(const P& p, const Task& task, Storage& s, int tokenOffset,
                                                 int validRows, int hidden, int warp, int lane) {
  using Tiles = typename P::Tiles;
  constexpr int KernelTileN = Tiles::N;
  constexpr int CtaTileM = Tiles::CtaM;
  if (lane != 0) return;
  int feature = task.m * CtaTileM * Tiles::ClusterSize + (blockIdx.x % Tiles::ClusterSize) * CtaTileM;
  if (feature >= hidden) return;
  for (int token = warp; token < validRows; token += EpilogueWarps) {
    Route route = p.workspace.routes[task.block * KernelTileN + tokenOffset + token];
    auto output = peerAt<__bfloat16>(p, route.rank, p.symmetric.partialOutput);
    size_t offset = (size_t(route.token) * p.config.topK + route.slot) * p.config.hidden + feature;
    bulkStore(output + offset, s.epilogue.packed + token * CtaTileM, CtaTileM * sizeof(__bfloat16));
    bulkStoreCommit();
    bulkStoreWaitSource<4>();
  }
  bulkStoreWaitSource();
}

#endif  // MSCCLPP_BULK_AVAILABLE

template <class Schedule, class P, class Storage, class Pipeline, class Accumulators>
__device__ __forceinline__ void epilogue(const P& p, const Task& task, Storage& s, Pipeline& pipeline,
                                         typename Pipeline::PipelineState& state, Accumulators accumulators, int hidden,
                                         int intermediate) {
#if MSCCLPP_BULK_AVAILABLE
  using namespace cute;
  using Tiles = typename P::Tiles;
  constexpr int KernelTileN = Tiles::N;
  constexpr int CtaTileM = Tiles::CtaM;
  constexpr int ChunkTokens = P::Collective::EpilogueTokens;
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
      activateFc1Chunk<EpilogueThreads, ChunkTokens>(p, task, s, values, coordinates, tokenOffset, validRows,
                                                     intermediate);
      traceKernel(KernelTracePhase::ActivationQuantize, false, tokenOffset);
      continue;
    }
    packFc2Chunk(p, task, s, values, coordinates, tokenOffset, validRows);
    bulkFence();
    cutlass::arch::NamedBarrier::sync(EpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    traceKernel(KernelTracePhase::Fc2Store, true, tokenOffset);
    storeOutputChunk<EpilogueWarps>(p, task, s, tokenOffset, validRows, hidden, warp, lane);
    traceKernel(KernelTracePhase::Fc2Store, false, tokenOffset);
    cutlass::arch::NamedBarrier::sync(EpilogueThreads, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
  }
  ++state;
  if (task.fc1 && threadIdx.x == 0) {
    auto* counter = p.workspace.hiddenReady + size_t(task.block) * W4ReadyCounterStride;
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(counter), "r"(uint32_t(1)) : "memory");
  }
#endif
}

template <class Schedule, class P, class Storage, class Accumulate, class Accumulators>
__device__ __forceinline__ void runEpilogueRole(const P& p, Storage& s, Accumulate& pipeline, Accumulators accumulators,
                                                int hidden, int intermediate, int cluster, int tasks) {
  cutlass::arch::warpgroup_reg_alloc<P::Collective::EpilogueRegisters>();
  typename Accumulate::PipelineState state;
  for (int i = cluster; i < tasks; i += gridDim.x / P::Tiles::ClusterSize)
    epilogue<Schedule>(p, taskAt(p, i, hidden, intermediate), s, pipeline, state, accumulators, hidden, intermediate);
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
  __device__ __forceinline__ static void accumulate(ResultAccumulator<Elements>& accumulator, const Element* source) {
#pragma unroll
    for (int element = 0; element < Elements; ++element) accumulator.values[element] += float(source[element]);
  }

  __device__ __forceinline__ static void store(Element* destination, const ResultAccumulator<Elements>& accumulator) {
#pragma unroll
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
__device__ __forceinline__ void reduceResultVector(const Element* partial, int rowStride, int topK, Element* output,
                                                   const int* routeIds = nullptr) {
  ResultAccumulator<Elements> accumulator;
  for (int slot = 0; slot < topK; ++slot) {
    if (routeIds && routeIds[slot] < 0) continue;
    ResultVectorOps<Element, Elements>::accumulate(accumulator, partial + size_t(slot) * rowStride);
  }
  ResultVectorOps<Element, Elements>::store(output, accumulator);
}

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
