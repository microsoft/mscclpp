// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_ROUTING_CUH_
#define MSCCLPP_EXT_MEGAMOE_ROUTING_CUH_

#include "megamoe_device.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

template <class P>
MSCCLPP_DEVICE_INLINE int localExpert(const P& p, int id, int rank, int token, int slot) {
  assert(id >= -1 && id < p.config.numExperts);
  int localExperts = p.config.numExperts / p.config.worldSize;
  return id >= p.config.rank * localExperts && id < (p.config.rank + 1) * localExperts
             ? id - p.config.rank * localExperts
             : -1;
}

template <class P>
MSCCLPP_DEVICE_INLINE uint2 readRoutingHeader(const P& p, uint64_t epoch, int rank) {
  return peerAt<mscclpp::LLPacket>(p, rank, p.symmetric.routingHeader)->read(uint32_t(epoch), SpinLimit);
}

MSCCLPP_DEVICE_INLINE int tokenRankAt(const Workspace& w, int worldSize, int token) {
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
MSCCLPP_DEVICE_INLINE int readPlannedRoute(const P& p, uint64_t epoch, int index, int tokens, Route& route) {
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
  size_t packetIndex = (size_t(rank) * p.config.maxTokens + token) * p.config.topK + slot;
  auto packets = at<mscclpp::LL8Packet>(p.local, p.symmetric.routingPackets);
  int id = mscclpp::bit_cast<int>(packets[packetIndex].read(uint32_t(epoch), SpinLimit));
  route = Route{rank, token, slot, 0.0f};
  return localExpert(p, id, rank, token, slot);
}

MSCCLPP_DEVICE_INLINE void countRouteGroup(const Workspace& w, int expert, int lane) {
  unsigned int group = __match_any_sync(0xffffffff, expert);
  if (expert >= 0 && lane == __ffs(group) - 1)
    atomicFetchAdd<int, scopeDevice>(w.counts + expert, __popc(group), memoryOrderRelaxed);
}

MSCCLPP_DEVICE_INLINE void fillRouteGroup(const Workspace& w, int expert, const Route& route, int lane) {
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
MSCCLPP_DEVICE_INLINE int plannerCtaCount(int items) {
  return min(int(gridDim.x), max(1, 1 + (items - 1) / P::ThreadCount));
}

template <class P>
MSCCLPP_DEVICE_INLINE float loadRouteWeight(const P& p, const Route& route) {
  auto* address =
      peerAt<float>(p, route.rank, p.symmetric.topkWeights) + size_t(route.token) * p.config.topK + route.slot;
  uint32_t bits;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(bits) : "l"(address) : "memory");
  return mscclpp::bit_cast<float>(bits);
}

// Push every live source route into the same fixed slot on every rank.
// Dynamic routing pushes only the source's live tokens; the existing count
// header bounds the valid token range.
template <class P>
MSCCLPP_DEVICE_INLINE void publishRoutePackets(const P& p, int tokens, const int32_t* ids, const float* scores,
                                               uint64_t epoch) {
  const auto& c = p.config;
  int routes = tokens * p.config.topK;
  int writes = c.worldSize * routes;
  int publishers = plannerCtaCount<P>(writes);
  if (blockIdx.x >= publishers) return;
  for (int write = blockIdx.x * P::ThreadCount + threadIdx.x; write < writes; write += publishers * P::ThreadCount) {
    int destination = write / routes;
    int index = write - destination * routes;
    int id = ids ? ids[index] : at<int>(p.local, p.symmetric.topkIds)[index];
    float weight = scores ? scores[index] : at<float>(p.local, p.symmetric.topkWeights)[index];
    assert(id >= -1 && id < c.numExperts);
    if (destination == 0) {
      if (ids) at<int>(p.local, p.symmetric.topkIds)[index] = id;
      if (scores) at<float>(p.local, p.symmetric.topkWeights)[index] = weight;
    }
    int token = index / c.topK;
    int slot = index % c.topK;
    size_t packetIndex = (size_t(c.rank) * c.maxTokens + token) * c.topK + slot;
    auto packets = peerAt<mscclpp::LL8Packet>(p, destination, p.symmetric.routingPackets);
    packets[packetIndex].write(mscclpp::bit_cast<uint32_t>(id), uint32_t(epoch));
  }
}

// Dynamic routing exchanges per-rank live token counts before scanning pushed packets.
template <class P>
MSCCLPP_DEVICE_INLINE void exchangeTokenCounts(const P& p, int tokens, uint64_t epoch) {
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
    w.control->routingPlannerCtas = plannerCtaCount<P>(preceding * c.topK);
    atomicStore<uint64_t, scopeDevice>(&w.control->routingInitEpoch, epoch, memoryOrderRelease);
  }
}

// Elect the last of `ctas` arrivals and clear the counter for the next launch. Arrivals are acq_rel,
// so the winner observes every CTA's planner writes before publishing the next phase.
MSCCLPP_DEVICE_INLINE bool lastArrival(int* arrivals, int ctas) {
  if (ctas == 1) return true;
  if (atomicFetchAdd<int, scopeDevice>(arrivals, 1, memoryOrderAcqRel) != ctas - 1) return false;
  *arrivals = 0;
  return true;
}

// Wait until the last planner publishes a routing phase for this epoch.
MSCCLPP_DEVICE_INLINE void waitRoutingPhase(uint64_t* phase, uint64_t epoch) {
  if (threadIdx.x == 0) waitAtLeast<uint64_t, scopeDevice>(phase, epoch);
  __syncthreads();
}

template <bool FixedTokenCount, class P>
MSCCLPP_DEVICE_INLINE void countRoutes(const P& p, int tokens, uint64_t epoch, int items, int plannerCtas) {
  const auto& w = p.workspace;
  int lane = threadIdx.x % 32;
  int plannerThread = blockIdx.x * P::ThreadCount + threadIdx.x;
  int plannerStride = plannerCtas * P::ThreadCount;
  for (int index = plannerThread; index < items; index += plannerStride) {
    Route route{};
    int expert = readPlannedRoute<FixedTokenCount>(p, epoch, index, tokens, route);
    countRouteGroup(w, expert, lane);
  }
}

// Last planner CTA, warp 0: convert expert counts into token-tile offsets and blocks.
template <class P>
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
      w.counts[expert] = 0;
    }
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
    precedingBlocks += __shfl_sync(0xffffffff, scan, WarpSize - 1);
  }
  __syncwarp();
  if (lane == 0) {
    w.control->tokenBlocks = precedingBlocks;
    atomicStore<uint64_t, scopeDevice>(&w.control->routingOffsetsEpoch, epoch, memoryOrderRelease);
  }
}

template <class P>
MSCCLPP_DEVICE_INLINE void resetReadyCounters(const P& p, int plannerCtas) {
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

template <bool FixedTokenCount, class P>
MSCCLPP_DEVICE_INLINE void fillRoutes(const P& p, int tokens, uint64_t epoch, int items, int plannerCtas) {
  const auto& w = p.workspace;
  int lane = threadIdx.x % 32;
  int plannerThread = blockIdx.x * P::ThreadCount + threadIdx.x;
  int plannerStride = plannerCtas * P::ThreadCount;
  for (int index = plannerThread; index < items; index += plannerStride) {
    Route route{};
    int expert = readPlannedRoute<FixedTokenCount>(p, epoch, index, tokens, route);
    fillRouteGroup(w, expert, route, lane);
  }
}

// Route packets are pushed into the destination's fixed local inbox. Count and fill are separate local scans.
template <bool FixedTokenCount, class P>
__device__ void preparePacketRoutes(const P& p, int tokens, const int32_t* ids = nullptr,
                                    const float* scores = nullptr) {
  const auto& c = p.config;
  const auto& w = p.workspace;
  uint64_t epoch = *at<uint64_t>(p.local, p.symmetric.epoch) + 1;
  if (blockIdx.x == 0 && threadIdx.x == 0) w.control->epoch = epoch;
  publishRoutePackets(p, tokens, ids, scores, epoch);

  int items, plannerCtas;
  if constexpr (FixedTokenCount) {
    items = c.worldSize * tokens * c.topK;
    plannerCtas = plannerCtaCount<P>(items);
  } else {
    if (blockIdx.x == 0) exchangeTokenCounts(p, tokens, epoch);
    waitRoutingPhase(&w.control->routingInitEpoch, epoch);
    items = w.control->routingTokens * c.topK;
    plannerCtas = w.control->routingPlannerCtas;
  }

  if (blockIdx.x < plannerCtas) {
    countRoutes<FixedTokenCount>(p, tokens, epoch, items, plannerCtas);
    __syncthreads();
    if (threadIdx.x < 32) {
      int last = threadIdx.x == 0 && lastArrival(&w.control->routingCountCtas, plannerCtas);
      if (__shfl_sync(0xffffffff, last, 0)) planTokenBlocks(p, epoch);
    }
    waitRoutingPhase(&w.control->routingOffsetsEpoch, epoch);
    resetReadyCounters(p, plannerCtas);
    fillRoutes<FixedTokenCount>(p, tokens, epoch, items, plannerCtas);
    __syncthreads();
    if (threadIdx.x == 0 && lastArrival(&w.control->routingFillCtas, plannerCtas))
      atomicStore<uint64_t, scopeDevice>(&w.control->routingReadyEpoch, epoch, memoryOrderRelease);
  }
  waitRoutingPhase(&w.control->routingReadyEpoch, epoch);
}

template <bool FixedTokenCount = false, class P>
__device__ void prepareRoutes(const P& p, int tokens, const int32_t* ids = nullptr, const float* scores = nullptr) {
  preparePacketRoutes<FixedTokenCount>(p, tokens, ids, scores);
}

template <bool E5M2>
MSCCLPP_DEVICE_INLINE void dispatchTokens(const Parameters<E5M2>& p, SharedStorage<E5M2>& s, int localWarp) {
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
      float routeWeight = loadRouteWeight(p, route);
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
      w.routes[row].weight = routeWeight;
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
