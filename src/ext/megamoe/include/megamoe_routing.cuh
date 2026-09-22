// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_ROUTING_CUH_
#define MSCCLPP_EXT_MEGAMOE_ROUTING_CUH_

#include "megamoe_device.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

template <bool E5M2>
__device__ int localExpert(const Parameters<E5M2>& p, int id, int rank, int token, int slot) {
  if (id < -1 || id >= p.config.numExperts) {
    printf("MegaMoE rank %d: invalid expert %d from rank %d, token %d, slot %d\n", p.config.rank, id, rank, token,
           slot);
    __trap();
  }
  int localExperts = p.config.numExperts / p.config.worldSize;
  return id >= p.config.rank * localExperts && id < (p.config.rank + 1) * localExperts
             ? id - p.config.rank * localExperts
             : -1;
}

template <bool E5M2>
__device__ int routeExpert(const Parameters<E5M2>& p, int rank, int token, int slot) {
  if (token >= *peerAt<E5M2, int>(p, rank, p.symmetric.tokenCount)) return -1;
  int id = peerAt<E5M2, int>(p, rank, p.symmetric.topkIds)[token * p.config.topK + slot];
  return localExpert(p, id, rank, token, slot);
}

template <bool E5M2>
__device__ void prepareSmallRoutes(const Parameters<E5M2>& p, RoutingStorage& scratch) {
  const auto& c = p.config;
  const auto& w = p.workspace;
  int experts = c.numExperts / c.worldSize;
  int slots = c.worldSize * c.maxTokens * c.topK;
  if (threadIdx.x < experts) scratch.counts[threadIdx.x] = 0;
  if (threadIdx.x < c.worldSize) {
    // Each peer-owning thread has already waited for that peer's publication.
    scratch.peers[threadIdx.x] = p.peers[threadIdx.x];
    scratch.tokenCounts[threadIdx.x] = *peerAt<E5M2, int>(p, threadIdx.x, p.symmetric.tokenCount);
  }
  __syncthreads();

  int assignedExperts[SmallRoutingSlots / Threads];
  int expertRows[SmallRoutingSlots / Threads];
  float weights[SmallRoutingSlots / Threads];
  CUTE_UNROLL
  for (int j = 0; j < SmallRoutingSlots / Threads; ++j) {
    int i = threadIdx.x + j * Threads;
    int expert = -1;
    if (i < slots) {
      int rank = i / (c.maxTokens * c.topK);
      int token = i / c.topK % c.maxTokens;
      int slot = i % c.topK;
      if (token < scratch.tokenCounts[rank]) {
        void* peer = reinterpret_cast<void*>(scratch.peers[rank]);
        int id = at<int>(peer, p.symmetric.topkIds)[token * c.topK + slot];
        weights[j] = at<float>(peer, p.symmetric.topkWeights)[token * c.topK + slot];
        expert = localExpert(p, id, rank, token, slot);
      }
    }
    assignedExperts[j] = expert;
    if (expert >= 0) expertRows[j] = atomicAdd(scratch.counts + expert, 1);
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    int preceding = 0;
    for (int base = 0; base < experts; base += 32) {
      int expert = base + threadIdx.x;
      int count = expert < experts ? scratch.counts[expert] : 0;
      int blocks = (count + TileN - 1) / TileN;
      int scan = blocks;
      CUTE_UNROLL
      for (int distance = 1; distance < 32; distance *= 2) {
        int other = __shfl_up_sync(0xffffffff, scan, distance);
        if (threadIdx.x >= distance) scan += other;
      }
      int firstBlock = preceding + scan - blocks;
      if (expert < experts) {
        scratch.starts[expert] = w.starts[expert] = firstBlock * TileN;
        w.counts[expert] = w.cursors[expert] = count;
        for (int block = 0; block < blocks; ++block)
          w.blocks[firstBlock + block] = TokenBlock{expert, min(int(TileN), count - block * TileN)};
      }
      preceding += __shfl_sync(0xffffffff, scan, 31);
    }
    if (threadIdx.x == 0) w.control->tokenBlocks = preceding;
  }
  __syncthreads();
  CUTE_UNROLL
  for (int j = 0; j < SmallRoutingSlots / Threads; ++j) {
    int expert = assignedExperts[j];
    if (expert >= 0) {
      int i = threadIdx.x + j * Threads;
      int rank = i / (c.maxTokens * c.topK);
      int token = i / c.topK % c.maxTokens;
      int slot = i % c.topK;
      w.routes[scratch.starts[expert] + expertRows[j]] = Route{rank, token, slot, weights[j]};
    }
  }
}

template <bool E5M2>
__device__ void prepareRoutes(const Parameters<E5M2>& p, int tokens, RoutingStorage& scratch) {
  const auto& c = p.config;
  const auto& w = p.workspace;
  int thread = blockIdx.x * Threads + threadIdx.x;
  int stride = gridDim.x * Threads;
  if (thread == 0) {
    w.control->epoch = ++*at<uint64_t>(p.local, p.symmetric.epoch);
    *at<int>(p.local, p.symmetric.tokenCount) = tokens;
  }
  for (int e = thread; e < c.numExperts / c.worldSize; e += stride) {
    w.counts[e] = 0;
    w.cursors[e] = 0;
  }
  for (int r = thread; r < w.poolRows; r += stride) w.routes[r].rank = -1;
  for (int b = thread; b < w.poolRows / TileN; b += stride) {
    w.inputReady[b] = 0;
    w.hiddenReady[b] = 0;
  }
  auto partial = at<__bfloat16>(p.local, p.symmetric.partialOutput);
  for (size_t i = thread; i < size_t(tokens) * c.topK * c.hidden; i += stride) partial[i] = __bfloat16(0.0f);
  w.control->gridBarrier.sync(gridDim.x, SpinLimit);

  // The grid barrier joins all staging/reset writes before this system release.
  if (blockIdx.x == 0 && threadIdx.x < c.worldSize) {
    signalAndWait(p, threadIdx.x);
  }
  int slots = c.worldSize * c.maxTokens * c.topK;
  if (slots <= SmallRoutingSlots && c.numExperts / c.worldSize <= SmallRoutingExperts) {
    // Only the planner CTA needs peer readiness. Publish its entire plan once;
    // the histogram ticket is also the final row offset, so IDs are read once.
    if (blockIdx.x == 0) prepareSmallRoutes(p, scratch);
    w.control->gridBarrier.sync(gridDim.x, SpinLimit);
    return;
  }
  w.control->gridBarrier.sync(gridDim.x, SpinLimit);

  for (int i = thread; i < slots; i += stride) {
    int rank = i / (c.maxTokens * c.topK);
    int token = i / c.topK % c.maxTokens;
    int expert = routeExpert(p, rank, token, i % c.topK);
    if (expert >= 0) atomicFetchAdd<int, scopeDevice>(w.counts + expert, 1, memoryOrderRelaxed);
  }
  w.control->gridBarrier.sync(gridDim.x, SpinLimit);

  if (thread == 0) {
    int block = 0;
    for (int e = 0; e < c.numExperts / c.worldSize; ++e) {
      w.starts[e] = block * TileN;
      for (int row = 0; row < w.counts[e]; row += TileN)
        w.blocks[block++] = TokenBlock{e, min(int(TileN), w.counts[e] - row)};
    }
    w.control->tokenBlocks = block;
  }
  w.control->gridBarrier.sync(gridDim.x, SpinLimit);

  for (int i = thread; i < slots; i += stride) {
    int rank = i / (c.maxTokens * c.topK);
    int token = i / c.topK % c.maxTokens;
    int slot = i % c.topK;
    int expert = routeExpert(p, rank, token, slot);
    if (expert >= 0) {
      int row = w.starts[expert] + atomicFetchAdd<int, scopeDevice>(w.cursors + expert, 1, memoryOrderRelaxed);
      float weight = peerAt<E5M2, float>(p, rank, p.symmetric.topkWeights)[token * c.topK + slot];
      w.routes[row] = Route{rank, token, slot, weight};
    }
  }
  w.control->gridBarrier.sync(gridDim.x, SpinLimit);
}

template <bool E5M2>
__device__ __forceinline__ void dispatchTokens(const Parameters<E5M2>& p, SharedStorage<E5M2>& s, int localWarp) {
#if MSCCLPP_BULK_AVAILABLE
  const auto& w = p.workspace;
  if (threadIdx.x % 32 == 0) {
    auto& barriers = s.dispatch.barriers[localWarp];
    barriers[0].relaxedInit();
    barriers[1].relaxedInit();
    bulkFence();
    uint32_t phases[2] = {0, 0};
    int bytes = p.config.hidden * sizeof(__bfloat16);
    int chunks = 1 + (bytes - 1) / DispatchChunkBytes;
    for (int row = blockIdx.x * DispatchWarpCount + localWarp; row < w.control->tokenBlocks * TileN;
         row += gridDim.x * DispatchWarpCount) {
      Route route = w.routes[row];
      if (route.rank < 0) continue;
      auto src = reinterpret_cast<const uint8_t*>(peerAt<E5M2, __bfloat16>(p, route.rank, p.symmetric.input) +
                                                  size_t(route.token) * p.config.hidden);
      auto dst = reinterpret_cast<uint8_t*>(w.input + size_t(row) * p.config.hidden);
      auto load = [&](int chunk) {
        int stage = chunk % 2;
        int size = min(int(DispatchChunkBytes), bytes - chunk * DispatchChunkBytes);
        barriers[stage].arriveAndExpect(size);
        bulkLoad(s.dispatch.tiles[localWarp][stage], src + chunk * DispatchChunkBytes, size, barriers[stage]);
      };
      load(0);
      if (chunks > 1) load(1);
      for (int chunk = 0; chunk < chunks; ++chunk) {
        int stage = chunk % 2;
        int size = min(int(DispatchChunkBytes), bytes - chunk * DispatchChunkBytes);
        barriers[stage].wait(phases[stage], SpinLimit);
        // No generic shared-memory access between these two async-proxy copies.
        bulkStore(dst + chunk * DispatchChunkBytes, s.dispatch.tiles[localWarp][stage], size);
        bulkStoreCommit();
        if (chunk + 2 < chunks) {
          bulkStoreWaitSource();
          load(chunk + 2);
        }
      }
      bulkStoreWait();
      atomicFetchAdd<int, scopeDevice>(w.inputReady + row / TileN, 1, memoryOrderRelease);
    }
    barriers[0].invalidate();
    barriers[1].invalidate();
  }
#endif
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_ROUTING_CUH_
