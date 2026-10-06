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
__device__ int routeExpert(const P& p, int rank, int token, int slot) {
  if (token >= *peerAt<int>(p, rank, p.symmetric.tokenCount)) return -1;
  int id = peerAt<int>(p, rank, p.symmetric.topkIds)[token * p.config.topK + slot];
  return localExpert(p, id, rank, token, slot);
}

template <bool Prefetched = false, class P>
__device__ void prepareSmallRoutes(const P& p, RoutingStorage& scratch, const int* stagedIds = nullptr,
                                   const float* stagedWeights = nullptr, int stagedStride = 0) {
  constexpr int RoutingTileN = P::WeightMxfp4 ? W4TileN : TileN;
  constexpr int PlannerThreads = P::ThreadCount;
  constexpr int SlotsPerThread = (SmallRoutingSlots + PlannerThreads - 1) / PlannerThreads;
  const auto& c = p.config;
  const auto& w = p.workspace;
  int experts = c.numExperts / c.worldSize;
  int slots = c.worldSize * c.maxTokens * c.topK;
  if (threadIdx.x < experts) scratch.counts[threadIdx.x] = 0;
  if (!Prefetched && threadIdx.x < c.worldSize) {
    // Each peer-owning thread has already waited for that peer's publication.
    scratch.peers[threadIdx.x] = p.peers[threadIdx.x];
    scratch.tokenCounts[threadIdx.x] = *peerAt<int>(p, threadIdx.x, p.symmetric.tokenCount);
  }
  __syncthreads();

  int assignedExperts[SlotsPerThread];
  int expertRows[SlotsPerThread];
  float weights[SlotsPerThread];
  CUTE_UNROLL
  for (int j = 0; j < SlotsPerThread; ++j) {
    int i = threadIdx.x + j * PlannerThreads;
    int expert = -1;
    if (i < slots) {
      int rank = i / (c.maxTokens * c.topK);
      int token = i / c.topK % c.maxTokens;
      int slot = i % c.topK;
      if (token < scratch.tokenCounts[rank]) {
        void* peer = reinterpret_cast<void*>(scratch.peers[rank]);
        int id = Prefetched ? stagedIds[rank * stagedStride + token * c.topK + slot]
                            : at<int>(peer, p.symmetric.topkIds)[token * c.topK + slot];
        weights[j] = Prefetched ? stagedWeights[rank * stagedStride + token * c.topK + slot]
                                : at<float>(peer, p.symmetric.topkWeights)[token * c.topK + slot];
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
      int blocks = (count + RoutingTileN - 1) / RoutingTileN;
      int scan = blocks;
      CUTE_UNROLL
      for (int distance = 1; distance < 32; distance *= 2) {
        int other = __shfl_up_sync(0xffffffff, scan, distance);
        if (threadIdx.x >= distance) scan += other;
      }
      int firstBlock = preceding + scan - blocks;
      if (expert < experts) {
        scratch.starts[expert] = w.starts[expert] = firstBlock * RoutingTileN;
        w.counts[expert] = w.cursors[expert] = count;
        for (int block = 0; block < blocks; ++block)
          w.blocks[firstBlock + block] = TokenBlock{expert, min(int(RoutingTileN), count - block * RoutingTileN)};
      }
      preceding += __shfl_sync(0xffffffff, scan, 31);
    }
    if (threadIdx.x == 0) w.control->tokenBlocks = preceding;
  }
  __syncthreads();
  CUTE_UNROLL
  for (int j = 0; j < SlotsPerThread; ++j) {
    int expert = assignedExperts[j];
    if (expert >= 0) {
      int i = threadIdx.x + j * PlannerThreads;
      int rank = i / (c.maxTokens * c.topK);
      int token = i / c.topK % c.maxTokens;
      int slot = i % c.topK;
      w.routes[scratch.starts[expert] + expertRows[j]] = Route{rank, token, slot, weights[j]};
    }
  }
}

template <class P>
__device__ void prepareRoutes(const P& p, int tokens, RoutingStorage& scratch, W4DispatchStorage* dispatch = nullptr) {
  constexpr int RoutingTileN = P::WeightMxfp4 ? W4TileN : TileN;
  const auto& c = p.config;
  const auto& w = p.workspace;
  int thread = blockIdx.x * P::ThreadCount + threadIdx.x;
  int stride = gridDim.x * P::ThreadCount;
  if constexpr (P::WeightMxfp4) {
    if (c.worldSize * c.maxTokens * c.topK <= SmallRoutingSlots && c.numExperts / c.worldSize <= SmallRoutingExperts) {
      if (blockIdx.x == 0) {
        if (threadIdx.x == 0) {
          w.control->epoch = ++*at<uint64_t>(p.local, p.symmetric.epoch);
          w.control->completedCtas = 0;
          w.control->planningArrivals = 0;
          w.control->planningReadyEpoch = 0;
          *at<int>(p.local, p.symmetric.tokenCount) = tokens;
        }
        __syncthreads();
        if (threadIdx.x < c.worldSize) signalAndWait(p, threadIdx.x);
        bool prefetched = false;
#if MSCCLPP_BULK_AVAILABLE
        int slotStride = (c.maxTokens * c.topK + 3) / 4 * 4;
        int peerBytes = slotStride * sizeof(int);
        if (dispatch && 2 * c.worldSize * peerBytes <= sizeof(dispatch->tiles)) {
          auto stagedIds = reinterpret_cast<int*>(dispatch->tiles);
          auto stagedWeights = reinterpret_cast<float*>(stagedIds + c.worldSize * slotStride);
          auto& barrier = dispatch->barriers[0][0];
          if (threadIdx.x == 0) {
            barrier.init();
            barrier.arriveAndExpect(2 * c.worldSize * peerBytes);
          }
          __syncthreads();
          if (threadIdx.x < c.worldSize) {
            int peer = threadIdx.x;
            bulkLoad(stagedIds + peer * slotStride, peerAt<int>(p, peer, p.symmetric.topkIds), peerBytes, barrier);
            bulkLoad(stagedWeights + peer * slotStride, peerAt<float>(p, peer, p.symmetric.topkWeights), peerBytes,
                     barrier);
            scratch.peers[peer] = p.peers[peer];
            scratch.tokenCounts[peer] = *peerAt<int>(p, peer, p.symmetric.tokenCount);
          }
          if (threadIdx.x == 0) {
            uint32_t phase = 0;
            barrier.wait(phase, SpinLimit);
            bulkFence();
          }
          __syncthreads();
          prepareSmallRoutes<true>(p, scratch, stagedIds, stagedWeights, slotStride);
          if (threadIdx.x == 0) barrier.invalidate();
          prefetched = true;
        }
#endif
        if (!prefetched) prepareSmallRoutes(p, scratch);
        for (int block = threadIdx.x; block < w.control->tokenBlocks; block += P::ThreadCount) {
          w.inputReady[block] = 0;
          w.hiddenReady[block] = 0;
          for (int chunk = 0; chunk < w4InputChunks(c.hidden) - 1; ++chunk)
            w.inputChunkReady[size_t(block) * (w4InputChunks(c.hidden) - 1) + chunk] = 0;
        }
      }
      w.control->gridBarrier.sync(gridDim.x, SpinLimit);
      return;
    }
  }
  if (thread == 0) {
    w.control->epoch = ++*at<uint64_t>(p.local, p.symmetric.epoch);
    if constexpr (P::WeightMxfp4) {
      w.control->completedCtas = 0;
      w.control->planningArrivals = 0;
      w.control->planningReadyEpoch = 0;
    }
    *at<int>(p.local, p.symmetric.tokenCount) = tokens;
  }
  for (int e = thread; e < c.numExperts / c.worldSize; e += stride) {
    w.counts[e] = 0;
    w.cursors[e] = 0;
  }
  // W4 dispatch checks live rows before loading routes; padding is never consumed.
  if constexpr (!P::WeightMxfp4)
    for (int r = thread; r < w.poolRows; r += stride) w.routes[r].rank = -1;
  for (int b = thread; b < w.poolRows / RoutingTileN; b += stride) {
    w.inputReady[b] = 0;
    w.hiddenReady[b] = 0;
    if constexpr (P::WeightMxfp4)
      for (int chunk = 0; chunk < w4InputChunks(c.hidden) - 1; ++chunk)
        w.inputChunkReady[size_t(b) * (w4InputChunks(c.hidden) - 1) + chunk] = 0;
  }
  if constexpr (P::WeightMxfp4) {
    // FC2 overwrites every active slot completely; combine excludes inactive slots.
    // Staged input is ready on entry. Only CTA 0's epoch/count publication must
    // precede its peer releases; the following grid join finishes private resets
    // and transfers the peer acquisitions to every CTA before histogramming.
    if (blockIdx.x == 0) __syncthreads();
  } else {
    auto partial = at<__bfloat16>(p.local, p.symmetric.partialOutput);
    for (size_t i = thread; i < size_t(tokens) * c.topK * c.hidden; i += stride) partial[i] = __bfloat16(0.0f);
    w.control->gridBarrier.sync(gridDim.x, SpinLimit);
  }

  // All CTA-0 threads join before its peer-owning threads perform system releases.
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

  bool tagged = false;
  if constexpr (P::WeightMxfp4) tagged = useRoutingTags(c);
  for (int i = thread; i < slots; i += stride) {
    int rank = i / (c.maxTokens * c.topK);
    int token = i / c.topK % c.maxTokens;
    // routeExpert guards every ID read with the acquired current peer count.
    int expert = routeExpert(p, rank, token, i % c.topK);
    if (tagged) {
      uint32_t tag = InvalidRoutingTag;
      if (expert >= 0) {
        int ticket = atomicFetchAdd<int, scopeDevice>(w.counts + expert, 1, memoryOrderRelaxed);
        tag = (uint32_t(expert) << RoutingTagTicketBits) | uint32_t(ticket);
      }
      // Unique grid-stride owner overwrites even nonlive/nonlocal/inactive slots
      // on every replay. No primed tag survives the histogram publication join.
      w.routingTags[i] = tag;
    } else if (expert >= 0) {
      atomicFetchAdd<int, scopeDevice>(w.counts + expert, 1, memoryOrderRelaxed);
    }
  }
  bool planLeader = thread == 0;
  if (tagged) {
    // All CTAs, including idle ones, collect their producers before arriving.
    // The acq_rel modification-order chain transfers every count/tag write to
    // the unique last leader. Full-grid residency is recomputed by preflight.
    __syncthreads();
    planLeader = false;
    if (threadIdx.x == 0)
      planLeader = atomicFetchAdd<int, scopeDevice>(&w.control->planningArrivals, 1, memoryOrderAcqRel) ==
                   int(gridDim.x) - 1;
  } else {
    w.control->gridBarrier.sync(gridDim.x, SpinLimit);
  }

  if (planLeader) {
    int block = 0;
    for (int e = 0; e < c.numExperts / c.worldSize; ++e) {
      w.starts[e] = block * RoutingTileN;
      for (int row = 0; row < w.counts[e]; row += RoutingTileN)
        w.blocks[block++] = TokenBlock{e, min(int(RoutingTileN), w.counts[e] - row)};
    }
    w.control->tokenBlocks = block;
    if (tagged)
      atomicStore<uint64_t, scopeDevice>(&w.control->planningReadyEpoch, w.control->epoch, memoryOrderRelease);
  }
  if (tagged) {
    // Even the publishing leader acquires the exact current device epoch.
    // The consumer join transfers prefix/tag visibility to every scatterer.
    if (threadIdx.x == 0) {
      const uint64_t epoch = w.control->epoch;
      POLL_MAYBE_JAILBREAK(
          (atomicLoad<uint64_t, scopeDevice>(&w.control->planningReadyEpoch, memoryOrderAcquire) != epoch), SpinLimit);
    }
    __syncthreads();
  } else {
    w.control->gridBarrier.sync(gridDim.x, SpinLimit);
  }

  for (int i = thread; i < slots; i += stride) {
    int rank = i / (c.maxTokens * c.topK);
    int token = i / c.topK % c.maxTokens;
    int slot = i % c.topK;
    if (tagged) {
      uint32_t tag = w.routingTags[i];
      if (tag != InvalidRoutingTag) {
        // A current accepted tag proves a live token and validated local ID;
        // histogram tickets uniquely cover exactly the expert's live rows.
        int expert = int(tag >> RoutingTagTicketBits);
        int row = w.starts[expert] + int(tag & RoutingTagTicketMask);
        float weight = peerAt<float>(p, rank, p.symmetric.topkWeights)[token * c.topK + slot];
        w.routes[row] = Route{rank, token, slot, weight};
      }
    } else {
      int expert = routeExpert(p, rank, token, slot);
      if (expert >= 0) {
        int row = w.starts[expert] + atomicFetchAdd<int, scopeDevice>(w.cursors + expert, 1, memoryOrderRelaxed);
        float weight = peerAt<float>(p, rank, p.symmetric.topkWeights)[token * c.topK + slot];
        w.routes[row] = Route{rank, token, slot, weight};
      }
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
      atomicFetchAdd<int, scopeDevice>(w.inputReady + row / TileN, 1, memoryOrderRelease);
    }
    barriers[0].invalidate();
    barriers[1].invalidate();
  }
#endif
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_ROUTING_CUH_
