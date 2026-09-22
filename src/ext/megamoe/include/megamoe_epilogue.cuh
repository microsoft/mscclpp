// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_EPILOGUE_CUH_
#define MSCCLPP_EXT_MEGAMOE_EPILOGUE_CUH_

#include "megamoe_device.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

__device__ __forceinline__ float swiglu(float gate, float up, float probability) {
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

template <bool E5M2, int LocalMode, class Accumulators>
__device__ __forceinline__ void epilogue(
    const Parameters<E5M2, (LocalMode != 0)>& p, const Task& task, SharedStorage<E5M2, (LocalMode != 0)>& s,
    typename CollectiveTypes<E5M2, (LocalMode != 0)>::Accumulate& pipeline,
    typename CollectiveTypes<E5M2, (LocalMode != 0)>::Accumulate::PipelineState& state, Accumulators accumulators,
    __bfloat16* directOutput, int hidden, int intermediate) {
#if MSCCLPP_BULK_AVAILABLE
  using namespace cute;
  constexpr bool Local = LocalMode != 0;
  using Tiles = TilePolicy<Local>;
  constexpr int KernelTileN = Tiles::N;
  constexpr int CtaTileM = Tiles::CtaM;
  constexpr int CtaFc1M = Tiles::CtaFc1M;
  constexpr int FoldedValuesPerThread = EpilogueTokens * CtaFc1M / 128;
  static_assert(EpilogueTokens * CtaFc1M % 128 == 0);
  auto matrix = accumulators(make_coord(_, _), _0{}, _0{}, state.index());
  CUTE_STATIC_ASSERT_V(size<0>(matrix) == Int<CtaTileM>{});
  CUTE_STATIC_ASSERT_V(size<1>(matrix) == Int<KernelTileN>{});
  pipeline.consumer_wait(state);
  for (int tokenOffset = 0; tokenOffset < task.tokens.rows; tokenOffset += EpilogueTokens) {
    int validRows = min(int(EpilogueTokens), task.tokens.rows - tokenOffset);
    auto acc = local_tile(matrix, make_shape(Int<CtaTileM>{}, Int<EpilogueTokens>{}),
                          make_coord(0, tokenOffset / EpilogueTokens));
    auto copyOp = make_tmem_copy(SM100_TMEM_LOAD_32dp32b8x{}, acc);
    auto threadCopy = copyOp.get_slice(threadIdx.x);
    auto source = threadCopy.partition_S(acc);
    auto identity = make_identity_tensor(make_shape(Int<CtaTileM>{}, Int<EpilogueTokens>{}));
    auto coordinates = threadCopy.partition_D(identity);
    auto values = make_tensor<float>(shape(coordinates));
    copy(copyOp, source, values);
    cutlass::arch::fence_view_async_tmem_load();
    if (tokenOffset + EpilogueTokens >= task.tokens.rows) pipeline.consumer_release(state);
    if (task.fc1) {
      CUTE_UNROLL
      for (int i = 0; i < size(values); ++i) {
        auto coord = coordinates(i);
        s.epilogue.scratch[(int(get<0>(coord)) % CtaTileM) * ScratchStride + int(get<1>(coord))] = values(i);
      }
      cutlass::arch::NamedBarrier::sync(128, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
      __bfloat16 folded[FoldedValuesPerThread];
      CUTE_UNROLL
      for (int j = 0; j < FoldedValuesPerThread; ++j) {
        int i = threadIdx.x + j * 128;
        int token = i / CtaFc1M;
        int feature = i % CtaFc1M;
        if (token < validRows) {
          int gateRow = feature / 16 * 32 + feature % 16;
          float gate = s.epilogue.scratch[gateRow * ScratchStride + token];
          float up = s.epilogue.scratch[(gateRow + 16) * ScratchStride + token];
          if (p.config.gateUpClamp >= 0) {
            gate = fminf(gate, p.config.gateUpClamp);
            up = fmaxf(-p.config.gateUpClamp, fminf(up, p.config.gateUpClamp));
          }
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
          folded[j] = active ? __bfloat16(swiglu(gate, up, probability)) : __bfloat16(0.0f);
          if constexpr (LocalMode != 0) {
            int column = task.m * Tiles::Fc1M + (blockIdx.x % ClusterM) * CtaFc1M + feature;
            p.workspace.hidden[size_t(row) * intermediate + column] = folded[j];
          }
        }
      }
      if constexpr (LocalMode != 0) {
        // Publish ordinary global stores to the async proxy before FC2's TMA load.
        asm volatile("fence.proxy.async.global;" ::: "memory");
        cutlass::arch::NamedBarrier::sync(128, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        continue;
      }
      // Every FP32 reader must finish before the same storage becomes a BF16 tile.
      cutlass::arch::NamedBarrier::sync(128, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
      CUTE_UNROLL
      for (int j = 0; j < FoldedValuesPerThread; ++j) {
        int i = threadIdx.x + j * 128;
        if (i < validRows * CtaFc1M) s.epilogue.packed[i] = folded[j];
      }
    } else {
      CUTE_UNROLL
      for (int i = 0; i < size(values); ++i) {
        auto coord = coordinates(i);
        bool active = true;
        if constexpr (LocalMode == 1) {
          int row = task.block * KernelTileN + tokenOffset + int(get<1>(coord));
          if (int(get<1>(coord)) < validRows) active = at<int>(p.local, p.symmetric.topkIds)[row] == 0;
        }
        s.epilogue.packed[int(get<1>(coord)) * CtaTileM + int(get<0>(coord)) % CtaTileM] =
            active ? __bfloat16(values(i)) : __bfloat16(0.0f);
      }
    }
    bulkFence();
    cutlass::arch::NamedBarrier::sync(128, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    if constexpr (LocalMode != 0) {
      if (!task.fc1 && (reinterpret_cast<uintptr_t>(directOutput) & 15) != 0) {
        int feature = task.m * Tiles::M + (blockIdx.x % ClusterM) * CtaTileM;
        if (feature < hidden) {
          for (int token = threadIdx.x / 32; token < validRows; token += 4) {
            size_t offset = size_t(task.block * KernelTileN + tokenOffset + token) * hidden + feature;
            auto* destination = directOutput + offset;
            auto* source = s.epilogue.packed + token * CtaTileM;
            // Preserve support for contiguous BF16 views with unaligned storage offsets.
            mscclpp::detail::copy<__bfloat16>(destination, source, CtaTileM, threadIdx.x % 32, 32);
          }
        }
        cutlass::arch::NamedBarrier::sync(128, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        continue;
      }
    }
    if (threadIdx.x % 32 == 0) {
      int width = task.fc1 ? CtaFc1M : CtaTileM;
      int feature = task.m * width * ClusterM + (blockIdx.x % ClusterM) * width;
      for (int token = threadIdx.x / 32; token < validRows; token += 4) {
        if (task.fc1) {
          auto output =
              p.workspace.hidden + size_t(task.block * KernelTileN + tokenOffset + token) * intermediate + feature;
          bulkStore(output, s.epilogue.packed + token * width, width * sizeof(__bfloat16));
        } else if (feature < hidden) {
          if constexpr (LocalMode != 0) {
            size_t offset = size_t(task.block * KernelTileN + tokenOffset + token) * hidden + feature;
            bulkStore(directOutput + offset, s.epilogue.packed + token * width, width * sizeof(__bfloat16));
          } else {
            Route route = p.workspace.routes[task.block * TileN + tokenOffset + token];
            auto output = peerAt<E5M2, __bfloat16>(p, route.rank, p.symmetric.partialOutput);
            size_t offset = (size_t(route.token) * p.config.topK + route.slot) * p.config.hidden + feature;
            bulkStore(output + offset, s.epilogue.packed + token * width, width * sizeof(__bfloat16));
          }
        }
        bulkStoreCommit();
        bulkStoreWait<4>();
      }
      if constexpr (LocalMode != 0) {
        if (task.fc1)
          bulkStoreWait();
        else
          bulkStoreWaitSource();
      } else {
        bulkStoreWait();
      }
    }
    cutlass::arch::NamedBarrier::sync(128, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
  }
  ++state;
  if (task.fc1 && threadIdx.x == 0)
    atomicFetchAdd<int, scopeDevice>(p.workspace.hiddenReady + task.block, 1, memoryOrderRelease);
#endif
}

// All peer output stores must be complete and visible before this local reduction.
template <bool E5M2>
__device__ __forceinline__ void combineResults(const Parameters<E5M2>& p, int tokens, __bfloat16* output) {
  auto partial = at<uint16_t>(p.local, p.symmetric.partialOutput);
  for (size_t i = blockIdx.x * Threads + threadIdx.x; i < size_t(tokens) * p.config.hidden; i += gridDim.x * Threads) {
    size_t token = i / p.config.hidden;
    int feature = i % p.config.hidden;
    float sum = 0.0f;
    for (int slot = 0; slot < p.config.topK; ++slot) {
      uint16_t bits;
      auto address = partial + (token * p.config.topK + slot) * p.config.hidden + feature;
      asm volatile("ld.global.cg.u16 %0, [%1];" : "=h"(bits) : "l"(address) : "memory");
      sum += float(mscclpp::bit_cast<__bfloat16>(bits));
    }
    output[i] = __bfloat16(sum);
  }
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_EPILOGUE_CUH_
