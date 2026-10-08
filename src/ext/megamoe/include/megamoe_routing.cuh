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

__device__ int tokenRankAt(const Workspace& w, int worldSize, int token) {
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

template <class P>
__device__ int readFixedRoute(const P& p, uint64_t epoch, int index, int routesPerRank, Route& route) {
  int rank = index / routesPerRank;
  int local = index - rank * routesPerRank;
  int token = local / p.config.topK;
  int slot = local % p.config.topK;
  uint2 packet = readRoutePacket(p, epoch, rank, token, slot);
  route = Route{rank, token, slot, mscclpp::bit_cast<float>(packet.y)};
  return localExpert(p, mscclpp::bit_cast<int>(packet.x), rank, token, slot);
}

__device__ void countRouteGroup(const Workspace& w, int expert, int lane) {
  unsigned int group = __match_any_sync(0xffffffff, expert);
  if (expert >= 0 && lane == __ffs(group) - 1)
    atomicFetchAdd<int, scopeDevice>(w.counts + expert, __popc(group), memoryOrderRelaxed);
}

__device__ void fillRouteGroup(const Workspace& w, int expert, const Route& route, int lane) {
  unsigned int group = __match_any_sync(0xffffffff, expert);
  if (expert >= 0) {
    int leader = __ffs(group) - 1;
    int base = 0;
    if (lane == leader) base = atomicFetchAdd<int, scopeDevice>(w.cursors + expert, __popc(group), memoryOrderRelaxed);
    base = __shfl_sync(group, base, leader);
    int offset = __popc(group & ((1u << lane) - 1));
    w.routes[w.starts[expert] + base + offset] = route;
  }
}

template <bool FixedTokenCount, class P>
__device__ void preparePacketRoutes(const P& p, int tokens, const int32_t* ids = nullptr,
                                    const float* scores = nullptr) {
  constexpr int WarpSize = 32;
  constexpr int RoutingTileN = P::Tiles::N;
  const auto& c = p.config;
  const auto& w = p.workspace;
  int warp = threadIdx.x / WarpSize;
  int lane = threadIdx.x % WarpSize;
  int experts = c.numExperts / c.worldSize;
  uint64_t epoch = *at<uint64_t>(p.local, p.symmetric.epoch) + 1;

  // One CTA gathers peer token counts and initializes the route-planning phases.
  if (blockIdx.x == 0) {
    if (threadIdx.x == 0) {
      w.control->epoch = epoch;
      w.control->completedCtas = 0;
      if constexpr (!FixedTokenCount) {
        w.control->routingCountCtas = 0;
        w.control->routingFillCtas = 0;
        w.peerTokenCounts[c.rank] = tokens;
        at<mscclpp::LLPacket>(p.local, p.symmetric.routingHeader)
            ->write(uint32_t(tokens), uint32_t(0), uint32_t(epoch));
      }
    }
    for (int index = threadIdx.x; index < tokens * c.topK; index += P::ThreadCount) {
      int id = ids ? ids[index] : at<int>(p.local, p.symmetric.topkIds)[index];
      float weight = scores ? scores[index] : at<float>(p.local, p.symmetric.topkWeights)[index];
      if (ids) at<int>(p.local, p.symmetric.topkIds)[index] = id;
      if (scores) at<float>(p.local, p.symmetric.topkWeights)[index] = weight;
      at<mscclpp::LLPacket>(p.local, p.symmetric.routingPackets)[index].write(
          mscclpp::bit_cast<uint32_t>(id), mscclpp::bit_cast<uint32_t>(weight), uint32_t(epoch));
    }
    if constexpr (!FixedTokenCount) {
      if (threadIdx.x < c.worldSize && threadIdx.x != c.rank) {
        int peer = threadIdx.x;
        int peerTokens = int(readRoutingHeader(p, epoch, peer).x);
        if (peerTokens < 0 || peerTokens > c.maxTokens) {
          printf("MegaMoE rank %d: invalid token count %d from rank %d\n", c.rank, peerTokens, peer);
          __trap();
        }
        w.peerTokenCounts[peer] = peerTokens;
      }
      for (int expert = threadIdx.x; expert < experts; expert += P::ThreadCount) {
        w.counts[expert] = 0;
        w.cursors[expert] = 0;
      }
      __syncthreads();
      if (warp == 0) {
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
          w.control->routingPlannerCtas =
              min(int(gridDim.x), max(1, (preceding + P::ThreadCount - 1) / P::ThreadCount));
          atomicStore<uint64_t, scopeDevice>(&w.control->routingInitEpoch, epoch, memoryOrderRelease);
        }
      }
    }
  }

  int totalTokens;
  int plannerCtas;
  if constexpr (FixedTokenCount) {
    totalTokens = c.worldSize * tokens * c.topK;
    plannerCtas = min(int(gridDim.x), max(1, (totalTokens + P::ThreadCount - 1) / P::ThreadCount));
  } else {
    waitAtLeast<uint64_t, scopeDevice>(&w.control->routingInitEpoch, epoch);
    totalTokens = w.control->routingTokens;
    plannerCtas = w.control->routingPlannerCtas;
  }
  int plannerThread = blockIdx.x * P::ThreadCount + threadIdx.x;
  int plannerStride = plannerCtas * P::ThreadCount;
  bool cacheRoute = totalTokens <= plannerStride;
  Route cachedRoute{};
  int cachedExpert = -1;
  if (blockIdx.x < plannerCtas) {
    if constexpr (FixedTokenCount) {
      int routesPerRank = tokens * c.topK;
      if (cacheRoute) {
        if (plannerThread < totalTokens)
          cachedExpert = readFixedRoute(p, epoch, plannerThread, routesPerRank, cachedRoute);
        countRouteGroup(w, cachedExpert, lane);
      } else {
        // Keep tail lanes participating in the warp's expert grouping.
        for (int first = blockIdx.x * P::ThreadCount; first < totalTokens; first += plannerStride) {
          int index = first + threadIdx.x;
          int expert = -1;
          Route route{};
          if (index < totalTokens) expert = readFixedRoute(p, epoch, index, routesPerRank, route);
          countRouteGroup(w, expert, lane);
        }
      }
    } else {
      for (int i = plannerThread; i < totalTokens; i += plannerStride) {
        int rank;
        rank = tokenRankAt(w, c.worldSize, i);
        int token = i - w.peerTokenOffsets[rank];
        CUTE_UNROLL
        for (int slot = 0; slot < c.topK; ++slot) {
          uint2 packet = readRoutePacket(p, epoch, rank, token, slot);
          int expert = localExpert(p, mscclpp::bit_cast<int>(packet.x), rank, token, slot);
          if (expert >= 0) atomicFetchAdd<int, scopeDevice>(w.counts + expert, 1, memoryOrderRelaxed);
        }
      }
    }
    __syncthreads();
    if (warp == 0) {
      int last = FixedTokenCount && plannerCtas == 1;
      if (lane == 0 && !last)
        last = atomicFetchAdd<int, scopeDevice>(&w.control->routingCountCtas, 1, memoryOrderAcqRel) == plannerCtas - 1;
      last = __shfl_sync(0xffffffff, last, 0);
      if (last) {
        __syncwarp();
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
            if constexpr (FixedTokenCount) {
              // Counts and arrival counters are cleared for the next ordered forward.
              w.counts[expert] = 0;
            } else {
              for (int localBlock = 0; localBlock < blocks; ++localBlock) {
                int row = localBlock * RoutingTileN;
                w.blocks[firstBlock + localBlock] = TokenBlock{expert, min(int(RoutingTileN), count - row)};
              }
            }
          }
          if constexpr (FixedTokenCount) {
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
                  w.blocks[precedingBlocks + block] =
                      TokenBlock{base + owner, min(int(RoutingTileN), expertCount - row)};
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
          if constexpr (FixedTokenCount) w.control->routingCountCtas = 0;
          w.control->tokenBlocks = precedingBlocks;
          atomicStore<uint64_t, scopeDevice>(&w.control->routingOffsetsEpoch, epoch, memoryOrderRelease);
        }
      }
    }

    if (threadIdx.x == 0) waitAtLeast<uint64_t, scopeDevice>(&w.control->routingOffsetsEpoch, epoch);
    __syncthreads();
    for (int block = plannerThread; block < w.control->tokenBlocks; block += plannerStride) {
      constexpr int ReadyStride = P::WeightMxfp4 ? W4ReadyCounterStride : 1;
      w.inputReady[size_t(block) * ReadyStride] = 0;
      w.hiddenReady[size_t(block) * ReadyStride] = 0;
      if constexpr (P::WeightMxfp4)
        for (int chunk = 0; chunk < w4InputChunks<typename P::Collective>(c.hidden) - 1; ++chunk)
          *w4InputChunkCounter<typename P::Collective>(w, c.hidden, block, chunk) = 0;
    }
    if constexpr (FixedTokenCount) {
      int routesPerRank = tokens * c.topK;
      if (cacheRoute) {
        fillRouteGroup(w, cachedExpert, cachedRoute, lane);
      } else {
        for (int first = blockIdx.x * P::ThreadCount; first < totalTokens; first += plannerStride) {
          int index = first + threadIdx.x;
          int expert = -1;
          Route route{};
          if (index < totalTokens) expert = readFixedRoute(p, epoch, index, routesPerRank, route);
          fillRouteGroup(w, expert, route, lane);
        }
      }
    } else {
      for (int i = plannerThread; i < totalTokens; i += plannerStride) {
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
    __syncthreads();
    if (threadIdx.x == 0) {
      bool last = FixedTokenCount && plannerCtas == 1;
      if (!last)
        last = atomicFetchAdd<int, scopeDevice>(&w.control->routingFillCtas, 1, memoryOrderAcqRel) == plannerCtas - 1;
      if (last) {
        if constexpr (FixedTokenCount) w.control->routingFillCtas = 0;
        atomicStore<uint64_t, scopeDevice>(&w.control->routingReadyEpoch, epoch, memoryOrderRelease);
      }
    }
  }

  if (threadIdx.x == 0) waitAtLeast<uint64_t, scopeDevice>(&w.control->routingReadyEpoch, epoch);
  __syncthreads();
}

template <bool FixedTokenCount = false, class P>
__device__ void prepareRoutes(const P& p, int tokens, const int32_t* ids = nullptr, const float* scores = nullptr) {
  preparePacketRoutes<FixedTokenCount>(p, tokens, ids, scores);
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
