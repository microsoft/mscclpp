// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_ROUTING_CUH_
#define MSCCLPP_EXT_MEGAMOE_ROUTING_CUH_

#include "megamoe_device.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

template <class P>
__device__ int localExpert(const P& p, int id, int rank, int token, int slot) {
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

template <class P>
__device__ uint2 readRoutePacket(const P& p, uint64_t epoch, int rank, int token, int slot) {
  auto packets = peerAt<mscclpp::LLPacket>(p, rank, p.symmetric.routingPackets);
  return packets[token * p.config.topK + slot].read(uint32_t(epoch), SpinLimit);
}

template <class P>
__device__ uint2 readRoutingHeader(const P& p, uint64_t epoch, int rank) {
  return peerAt<mscclpp::LLPacket>(p, rank, p.symmetric.routingHeader)->read(uint32_t(epoch), SpinLimit);
}

__device__ __forceinline__ int tokenRankAt(const Workspace& w, int worldSize, int token) {
  int low = 0;
  int high = worldSize;
  while (low + 1 < high) {
    int middle = (low + high) / 2;
    if (token < w.peerTokenOffsets[middle])
      high = middle;
    else
      low = middle;
  }
  return low;
}

template <bool FixedTokenCount, class P>
__device__ int readPlannedRoute(const P& p, uint64_t epoch, int index, int tokens, Route& route) {
  int rank, token, slot;
  if constexpr (FixedTokenCount) {
    int routesPerRank = tokens * p.config.topK;
    rank = index / routesPerRank;
    int local = index - rank * routesPerRank;
    token = local / p.config.topK;
    slot = local % p.config.topK;
  } else {
    int ordinal = index / p.config.topK;
    rank = tokenRankAt(p.workspace, p.config.worldSize, ordinal);
    token = ordinal - p.workspace.peerTokenOffsets[rank];
    slot = index % p.config.topK;
  }
  uint2 packet = readRoutePacket(p, epoch, rank, token, slot);
  route = Route{rank, token, slot, mscclpp::bit_cast<float>(packet.y)};
  return localExpert(p, mscclpp::bit_cast<int>(packet.x), rank, token, slot);
}

__device__ __forceinline__ void countRouteGroup(const Workspace& w, int expert, int lane) {
  unsigned int group = __match_any_sync(0xffffffff, expert);
  if (expert >= 0 && lane == __ffs(group) - 1)
    atomicFetchAdd<int, scopeDevice>(w.counts + expert, __popc(group), memoryOrderRelaxed);
}

__device__ __forceinline__ void fillRouteGroup(const Workspace& w, int expert, const Route& route, int lane) {
  unsigned int group = __match_any_sync(0xffffffff, expert);
  if (expert >= 0) {
    int leader = __ffs(group) - 1;
    int base = 0;
    if (lane == leader) base = atomicFetchAdd<int, scopeDevice>(w.cursors + expert, __popc(group), memoryOrderRelaxed);
    base = __shfl_sync(group, base, leader);
    int row = w.starts[expert] + base + __popc(group & ((1u << lane) - 1));
    w.routes[row] = route;
  }
}

template <class P>
__device__ __forceinline__ int plannerCtaCount(int items) {
  return min(int(gridDim.x), max(1, 1 + (items - 1) / P::ThreadCount));
}

// Publish this rank's {expert ID, router weight} routes as epoch-tagged LL packets, so peers can read
// them without another flag. Publishers are a prefix of the planner CTAs, which orders the local topk
// copies used by the epilogue before routingReadyEpoch.
template <bool CachedRoutes, class P>
__device__ __forceinline__ void publishRoutePackets(const P& p, int tokens, const int32_t* ids, const float* scores,
                                                    uint64_t epoch) {
  int routes = tokens * p.config.topK;
  int publishers = plannerCtaCount<P>(CachedRoutes ? routes : tokens);
  if (blockIdx.x >= publishers) return;
  for (int index = blockIdx.x * P::ThreadCount + threadIdx.x; index < routes; index += publishers * P::ThreadCount) {
    int id = ids ? ids[index] : at<int>(p.local, p.symmetric.topkIds)[index];
    float weight = scores ? scores[index] : at<float>(p.local, p.symmetric.topkWeights)[index];
    if (ids) at<int>(p.local, p.symmetric.topkIds)[index] = id;
    if (scores) at<float>(p.local, p.symmetric.topkWeights)[index] = weight;
    at<mscclpp::LLPacket>(p.local, p.symmetric.routingPackets)[index].write(
        mscclpp::bit_cast<uint32_t>(id), mscclpp::bit_cast<uint32_t>(weight), uint32_t(epoch));
  }
}

// Dynamic row counts (CTA0 only): exchange per-rank token counts and build peer token prefixes.
// The work is O(worldSize): peers are strided over the CTA, so one CTA covers any EP size.
template <bool CachedRoutes, class P>
__device__ void exchangeTokenCounts(const P& p, int tokens, uint64_t epoch) {
  constexpr int WarpSize = 32;
  const auto& c = p.config;
  const auto& w = p.workspace;
  int warp = threadIdx.x / WarpSize;
  int lane = threadIdx.x % WarpSize;
  if (threadIdx.x == 0) {
    w.peerTokenCounts[c.rank] = tokens;
    at<mscclpp::LLPacket>(p.local, p.symmetric.routingHeader)->write(uint32_t(tokens), uint32_t(0), uint32_t(epoch));
  }
  for (int peer = threadIdx.x; peer < c.worldSize; peer += P::ThreadCount) {
    if (peer == c.rank) continue;
    int peerTokens = int(readRoutingHeader(p, epoch, peer).x);
    if (peerTokens < 0 || peerTokens > c.maxTokens) {
      printf("MegaMoE rank %d: invalid token count %d from rank %d\n", c.rank, peerTokens, peer);
      __trap();
    }
    w.peerTokenCounts[peer] = peerTokens;
  }
  __syncthreads();
  if (warp != 0) return;
  int preceding = 0;
  for (int base = 0; base < c.worldSize; base += WarpSize) {
    int peer = base + lane;
    int peerTokens = peer < c.worldSize ? w.peerTokenCounts[peer] : 0;
    int scan = peerTokens;
    CUTE_UNROLL
    for (int distance = 1; distance < WarpSize; distance *= 2) {
      int other = __shfl_up_sync(0xffffffff, scan, distance);
      if (lane >= distance) scan += other;
    }
    if (peer < c.worldSize) w.peerTokenOffsets[peer] = preceding + scan - peerTokens;
    preceding += __shfl_sync(0xffffffff, scan, WarpSize - 1);
  }
  __syncwarp();
  if (lane == 0) {
    w.peerTokenOffsets[c.worldSize] = preceding;
    w.control->routingTokens = preceding;
    w.control->routingPlannerCtas = plannerCtaCount<P>(CachedRoutes ? preceding * c.topK : preceding);
    atomicStore<uint64_t, scopeDevice>(&w.control->routingInitEpoch, epoch, memoryOrderRelease);
  }
}

// Elect the last of `ctas` arrivals and clear the counter for the next launch. Arrivals are acq_rel,
// so the winner observes every CTA's planner writes before publishing the next phase.
__device__ __forceinline__ bool lastArrival(int* arrivals, int ctas) {
  if (ctas == 1) return true;
  if (atomicFetchAdd<int, scopeDevice>(arrivals, 1, memoryOrderAcqRel) != ctas - 1) return false;
  *arrivals = 0;
  return true;
}

// Wait until the last planner publishes a routing phase for this epoch.
__device__ __forceinline__ void waitRoutingPhase(uint64_t* phase, uint64_t epoch) {
  if (threadIdx.x == 0) waitAtLeast<uint64_t, scopeDevice>(phase, epoch);
  __syncthreads();
}

// Count each local expert's routes. Captured routing assigns one route per thread and keeps it in
// registers for the fill phase when the planner covers all routes; eager routing walks tokens.
template <bool CachedRoutes, bool FixedTokenCount, class P>
__device__ __forceinline__ void countRoutes(const P& p, int tokens, uint64_t epoch, int items, int plannerCtas,
                                            int& cachedExpert, Route& cachedRoute) {
  const auto& c = p.config;
  const auto& w = p.workspace;
  int lane = threadIdx.x % 32;
  int plannerThread = blockIdx.x * P::ThreadCount + threadIdx.x;
  int plannerStride = plannerCtas * P::ThreadCount;
  if constexpr (CachedRoutes) {
    if (items <= plannerStride) {
      if (plannerThread < items)
        cachedExpert = readPlannedRoute<FixedTokenCount>(p, epoch, plannerThread, tokens, cachedRoute);
      countRouteGroup(w, cachedExpert, lane);
      return;
    }
    // Keep tail lanes participating in the warp's expert grouping.
    for (int first = blockIdx.x * P::ThreadCount; first < items; first += plannerStride) {
      int index = first + threadIdx.x;
      int expert = -1;
      Route route{};
      if (index < items) expert = readPlannedRoute<FixedTokenCount>(p, epoch, index, tokens, route);
      countRouteGroup(w, expert, lane);
    }
  } else {
    for (int i = plannerThread; i < items; i += plannerStride) {
      int rank = tokenRankAt(w, c.worldSize, i);
      int token = i - w.peerTokenOffsets[rank];
      CUTE_UNROLL
      for (int slot = 0; slot < c.topK; ++slot) {
        uint2 packet = readRoutePacket(p, epoch, rank, token, slot);
        int expert = localExpert(p, mscclpp::bit_cast<int>(packet.x), rank, token, slot);
        if (expert >= 0) atomicFetchAdd<int, scopeDevice>(w.counts + expert, 1, memoryOrderRelaxed);
      }
    }
  }
}

// Last planner CTA, warp 0: convert expert counts into token-tile offsets and blocks.
template <bool CachedRoutes, class P>
__device__ void planTokenBlocks(const P& p, uint64_t epoch) {
  constexpr int WarpSize = 32;
  constexpr int RoutingTileN = P::Tiles::N;
  const auto& w = p.workspace;
  int lane = threadIdx.x % WarpSize;
  int experts = p.config.numExperts / p.config.worldSize;
  int precedingBlocks = 0;
  for (int base = 0; base < experts; base += WarpSize) {
    int expert = base + lane;
    int count = expert < experts ? w.counts[expert] : 0;
    int blocks = (count + RoutingTileN - 1) / RoutingTileN;
    int scan = blocks;
    CUTE_UNROLL
    for (int distance = 1; distance < WarpSize; distance *= 2) {
      int other = __shfl_up_sync(0xffffffff, scan, distance);
      if (lane >= distance) scan += other;
    }
    int firstBlock = precedingBlocks + scan - blocks;
    if (expert < experts) {
      w.starts[expert] = firstBlock * RoutingTileN;
      w.cursors[expert] = 0;
      // Clear after the count join so eager and captured routing can alternate.
      w.counts[expert] = 0;
      if constexpr (!CachedRoutes) {
        for (int localBlock = 0; localBlock < blocks; ++localBlock) {
          int row = localBlock * RoutingTileN;
          w.blocks[firstBlock + localBlock] = TokenBlock{expert, min(int(RoutingTileN), count - row)};
        }
      }
    }
    if constexpr (CachedRoutes) {
      // Warps distribute block construction for experts spanning several token blocks.
      if (__any_sync(0xffffffff, blocks > 1)) {
        int batchBlocks = __shfl_sync(0xffffffff, scan, WarpSize - 1);
        for (int first = 0; first < batchBlocks; first += WarpSize) {
          int block = first + lane;
          int owner = 0;
          CUTE_UNROLL
          for (int step = WarpSize / 2; step > 0; step /= 2) {
            int candidate = owner + step;
            int end = __shfl_sync(0xffffffff, scan, candidate - 1);
            if (end <= block) owner = candidate;
          }
          int expertFirst = __shfl_sync(0xffffffff, scan - blocks, owner);
          int expertCount = __shfl_sync(0xffffffff, count, owner);
          if (block < batchBlocks) {
            int row = (block - expertFirst) * RoutingTileN;
            w.blocks[precedingBlocks + block] = TokenBlock{base + owner, min(int(RoutingTileN), expertCount - row)};
          }
        }
      } else if (blocks) {
        w.blocks[firstBlock] = TokenBlock{expert, count};
      }
    }
    precedingBlocks += __shfl_sync(0xffffffff, scan, WarpSize - 1);
  }
  __syncwarp();
  if (lane == 0) {
    w.control->tokenBlocks = precedingBlocks;
    atomicStore<uint64_t, scopeDevice>(&w.control->routingOffsetsEpoch, epoch, memoryOrderRelease);
  }
}

template <class P>
__device__ __forceinline__ void resetReadyCounters(const P& p, int plannerCtas) {
  const auto& w = p.workspace;
  int plannerStride = plannerCtas * P::ThreadCount;
  for (int block = blockIdx.x * P::ThreadCount + threadIdx.x; block < w.control->tokenBlocks; block += plannerStride) {
    constexpr int ReadyStride = P::WeightMxfp4 ? W4ReadyCounterStride : 1;
    w.inputReady[size_t(block) * ReadyStride] = 0;
    w.hiddenReady[size_t(block) * ReadyStride] = 0;
    if constexpr (P::WeightMxfp4)
      for (int chunk = 0; chunk < w4InputChunks<typename P::Collective>(p.config.hidden) - 1; ++chunk)
        *w4InputChunkCounter<typename P::Collective>(w, p.config.hidden, block, chunk) = 0;
  }
}

// Write each route to its expert-major row, reusing the route cached by countRoutes when present.
template <bool CachedRoutes, bool FixedTokenCount, class P>
__device__ __forceinline__ void fillRoutes(const P& p, int tokens, uint64_t epoch, int items, int plannerCtas,
                                           int cachedExpert, const Route& cachedRoute) {
  const auto& c = p.config;
  const auto& w = p.workspace;
  int lane = threadIdx.x % 32;
  int plannerThread = blockIdx.x * P::ThreadCount + threadIdx.x;
  int plannerStride = plannerCtas * P::ThreadCount;
  if constexpr (CachedRoutes) {
    if (items <= plannerStride) {
      fillRouteGroup(w, cachedExpert, cachedRoute, lane);
      return;
    }
    for (int first = blockIdx.x * P::ThreadCount; first < items; first += plannerStride) {
      int index = first + threadIdx.x;
      int expert = -1;
      Route route{};
      if (index < items) expert = readPlannedRoute<FixedTokenCount>(p, epoch, index, tokens, route);
      fillRouteGroup(w, expert, route, lane);
    }
  } else {
    for (int i = plannerThread; i < items; i += plannerStride) {
      int rank = tokenRankAt(w, c.worldSize, i);
      int token = i - w.peerTokenOffsets[rank];
      CUTE_UNROLL
      for (int slot = 0; slot < c.topK; ++slot) {
        uint2 packet = readRoutePacket(p, epoch, rank, token, slot);
        int expert = localExpert(p, mscclpp::bit_cast<int>(packet.x), rank, token, slot);
        if (expert >= 0) {
          int row = w.starts[expert] + atomicFetchAdd<int, scopeDevice>(w.cursors + expert, 1, memoryOrderRelaxed);
          w.routes[row] = Route{rank, token, slot, mscclpp::bit_cast<float>(packet.y)};
        }
      }
    }
  }
}

// CachedRoutes selects the captured planner (one cached route per thread, warp-aggregated atomics).
// FixedTokenCount additionally assumes every rank submits the same row count, skipping the count exchange.
template <bool CachedRoutes, bool FixedTokenCount, class P>
__device__ void preparePacketRoutes(const P& p, int tokens, const int32_t* ids = nullptr,
                                    const float* scores = nullptr) {
  static_assert(!FixedTokenCount || CachedRoutes);
  const auto& c = p.config;
  const auto& w = p.workspace;
  uint64_t epoch = *at<uint64_t>(p.local, p.symmetric.epoch) + 1;
  if (blockIdx.x == 0 && threadIdx.x == 0) w.control->epoch = epoch;
  publishRoutePackets<CachedRoutes>(p, tokens, ids, scores, epoch);

  int items, plannerCtas;
  if constexpr (FixedTokenCount) {
    items = c.worldSize * tokens * c.topK;
    plannerCtas = plannerCtaCount<P>(items);
  } else {
    if (blockIdx.x == 0) exchangeTokenCounts<CachedRoutes>(p, tokens, epoch);
    waitRoutingPhase(&w.control->routingInitEpoch, epoch);
    items = w.control->routingTokens * (CachedRoutes ? c.topK : 1);
    plannerCtas = w.control->routingPlannerCtas;
  }

  if (blockIdx.x < plannerCtas) {
    Route cachedRoute{};
    int cachedExpert = -1;
    countRoutes<CachedRoutes, FixedTokenCount>(p, tokens, epoch, items, plannerCtas, cachedExpert, cachedRoute);
    __syncthreads();
    if (threadIdx.x < 32) {
      int last = threadIdx.x == 0 && lastArrival(&w.control->routingCountCtas, plannerCtas);
      if (__shfl_sync(0xffffffff, last, 0)) planTokenBlocks<CachedRoutes>(p, epoch);
    }
    waitRoutingPhase(&w.control->routingOffsetsEpoch, epoch);
    resetReadyCounters(p, plannerCtas);
    fillRoutes<CachedRoutes, FixedTokenCount>(p, tokens, epoch, items, plannerCtas, cachedExpert, cachedRoute);
    __syncthreads();
    if (threadIdx.x == 0 && lastArrival(&w.control->routingFillCtas, plannerCtas))
      atomicStore<uint64_t, scopeDevice>(&w.control->routingReadyEpoch, epoch, memoryOrderRelease);
  }
  waitRoutingPhase(&w.control->routingReadyEpoch, epoch);
}

template <bool CachedRoutes = false, bool FixedTokenCount = false, class P>
__device__ void prepareRoutes(const P& p, int tokens, const int32_t* ids = nullptr, const float* scores = nullptr) {
  preparePacketRoutes<CachedRoutes, FixedTokenCount>(p, tokens, ids, scores);
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
      if (row % TileN >= w.blocks[row / TileN].rows) continue;
      Route route = w.routes[row];
      if (route.rank < 0) continue;
      // prepareRoutes consumed this peer's current-epoch route packet, which is
      // published by a kernel launched after the peer's same-stream input staging.
      auto src = reinterpret_cast<const uint8_t*>(peerAt<__bfloat16>(p, route.rank, p.symmetric.input) +
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
      auto* counter = w.inputReady + row / TileN;
      asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(counter), "r"(uint32_t(1)) : "memory");
    }
    barriers[0].invalidate();
    barriers[1].invalidate();
  }
#endif
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_ROUTING_CUH_
