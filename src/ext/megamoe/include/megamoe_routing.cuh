// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_ROUTING_CUH_
#define MSCCLPP_EXT_MEGAMOE_ROUTING_CUH_

#include "megamoe_device.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

// Routing exchange. Each rank splits its own routes into chunks of RouteChunk consecutive routes, and one CTA pass
// handles a chunk with one route per thread. A route whose expert lives on another rank goes only to that owner: the
// chunk packs the owner's routes into LL8 entries at routingPackets[source][chunk * RouteChunk + entry] on the owner,
// then sends every peer one LL16 header {entries, source tokens} at routingHeaders[source][chunk]. Routes of local
// experts never leave the source CTA. Entries and headers carry the launch epoch, so owners poll only their own memory.
// A rank's own routingPackets slice instead holds the LL8 weights of its remote routes, which owners read in dispatch.

// Pack a route entry: the owner's local expert and the route's offset in its source chunk.
MSCCLPP_DEVICE_INLINE uint32_t packRouteEntry(int expert, int offset) {
  return uint32_t(expert) << 16 | uint32_t(offset);
}

// Decode a global route index, source * maxTokens * topK + the source's route index.
template <class P>
MSCCLPP_DEVICE_INLINE Route makeRoute(const P& p, uint32_t route) {
  const int topK = p.config.topK;
  const uint32_t routesPerRank = uint32_t(p.config.maxTokens) * topK;
  int rank = p.config.worldSize > 1 ? int(route / routesPerRank) : 0;
  int index = int(route - uint32_t(rank) * routesPerRank);
  if ((topK & (topK - 1)) == 0) return Route{rank, index >> (__ffs(topK) - 1), index & (topK - 1), 0.0f};
  return Route{rank, index / topK, index % topK, 0.0f};
}

MSCCLPP_DEVICE_INLINE int atomicAddShared(int* counter, int value) {
  int old;
  // An explicit shared address avoids the generic-address atomic instruction.
  uint32_t address = static_cast<uint32_t>(__cvta_generic_to_shared(counter));
  asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(old) : "r"(address), "r"(value) : "memory");
  return old;
}

// Reserve consecutive positions from a shared counter per key, one atomic per group of lanes sharing a key.
// All lanes of the warp must call this; lanes with a negative key receive -1.
MSCCLPP_DEVICE_INLINE int reserveRouteGroup(int* counters, int key, int lane) {
  unsigned int group = __match_any_sync(0xffffffff, key);
  if (key < 0) return -1;
  int leader = __ffs(group) - 1;
  int base = 0;
  if (lane == leader) base = atomicAddShared(counters + key, __popc(group));
  base = __shfl_sync(group, base, leader);
  return base + __popc(group & ((1u << lane) - 1));
}

// Reserve one shared slot per lane in `lanes`, with one atomic per warp. All lanes of the warp must call this.
MSCCLPP_DEVICE_INLINE int reserveWarpSlots(int* counter, unsigned int lanes, int lane) {
  int base = 0;
  if (lane == 0) base = atomicAddShared(counter, __popc(lanes));
  base = __shfl_sync(0xffffffff, base, 0);
  return base + __popc(lanes & ((1u << lane) - 1));
}

// A local-expert route cached in shared memory: its global route index, its local expert, and its position among
// this CTA's routes of that expert.
struct alignas(8) CachedRoute {
  uint32_t route;
  uint16_t expert;
  uint16_t position;
};

static_assert(sizeof(CachedRoute) == sizeof(uint2));

// Per-CTA routing state in the idle GEMM tensor storage. The first three counters are contiguous, so one loop zeroes
// them at launch.
struct RoutingScratch {
  int* nRoutesByExpert;     // this CTA's local-expert routes per expert
  int* nRoutesByOwner;      // routes of the current source chunk per owner rank
  int* nRoutes;             // cached routes; beyond `capacity`, the fill rescans
  int* routeOffsets;        // this CTA's first row per expert, once the counts are reserved
  int* nChunksBySource;     // live chunks per source rank, for dynamic token counts
  int* nFirstChunkEntries;  // this rank's entries in each source's chunk 0, for dynamic token counts
  int* segmentEnds;         // inclusive entry prefix over the current segment batch
  uint32_t* segmentFirsts;  // global route index of each batch segment's first route
  int* nTokenBlocks;
  int* maxSourceChunks;  // most live chunks of any source, for dynamic token counts
  CachedRoute* cachedRoutes;
  int capacity;
};

template <class SharedStorage>
MSCCLPP_DEVICE_INLINE RoutingScratch makeRoutingScratch(SharedStorage& shared, int experts, int worldSize) {
  constexpr size_t ScratchInts = sizeof(SharedStorage::routing) / sizeof(int);
  size_t headerInts = routingScratchHeaderInts(experts, worldSize);
  assert(experts <= 0xffff && headerInts <= ScratchInts);
  RoutingScratch scratch;
  scratch.nRoutesByExpert = shared.routing;
  scratch.nRoutesByOwner = scratch.nRoutesByExpert + experts;
  scratch.nRoutes = scratch.nRoutesByOwner + worldSize;
  scratch.routeOffsets = scratch.nRoutes + 1;
  scratch.nChunksBySource = scratch.routeOffsets + experts;
  scratch.nFirstChunkEntries = scratch.nChunksBySource + worldSize;
  scratch.segmentEnds = scratch.nFirstChunkEntries + worldSize;
  scratch.segmentFirsts = reinterpret_cast<uint32_t*>(scratch.segmentEnds + RouteSegmentBatch);
  scratch.nTokenBlocks = reinterpret_cast<int*>(scratch.segmentFirsts + RouteSegmentBatch);
  scratch.maxSourceChunks = scratch.nTokenBlocks + 1;
  scratch.cachedRoutes = reinterpret_cast<CachedRoute*>(shared.routing + headerInts);
  scratch.capacity = int(min((ScratchInts - headerInts) / 2, size_t(1) << 16));
  return scratch;
}

// Cache one route per lane. All lanes of the warp must call this; lanes without a local route pass expert -1.
MSCCLPP_DEVICE_INLINE void cacheRoute(const RoutingScratch& scratch, int expert, uint32_t route, int lane) {
  unsigned int localLanes = __ballot_sync(0xffffffff, expert >= 0);
  if (!localLanes) return;
  int position = reserveRouteGroup(scratch.nRoutesByExpert, expert, lane);
  int slot = reserveWarpSlots(scratch.nRoutes, localLanes, lane);
  if (expert >= 0 && slot < scratch.capacity)
    scratch.cachedRoutes[slot] = CachedRoute{route, uint16_t(expert), uint16_t(position)};
}

// Load the expert ID of the rank's own route `routeIndex`, or -1 past its live routes.
template <class P>
MSCCLPP_DEVICE_INLINE int loadRouteId(const P& p, const int32_t* ids, int routeIndex, int nRoutes) {
  if (routeIndex >= nRoutes) return -1;
  return ids ? ids[routeIndex] : at<int>(p.local, p.symmetric.topkIds)[routeIndex];
}

// Process this rank's source-route chunks and pass each lane to
// `handleOwnedRoute(localExpert, globalRoute, lane)`. A lane whose route belongs to another rank passes -1 so every
// lane still participates in the handler's warp collectives. With Publish, the processing also stages direct inputs,
// sends remote-expert routes and their weights, and ends every chunk with the peer headers. `firstExpertId` is this
// thread's expert ID in the CTA's first chunk, loaded by the caller.
template <bool Publish, class P, class OwnedRouteHandler>
__device__ void processSourceRouteChunks(const P& p, int nTokens, const int32_t* ids, const float* scores,
                                         uint64_t epoch, int* nRoutesByOwner, int firstExpertId,
                                         OwnedRouteHandler&& handleOwnedRoute) {
  static_assert(P::ThreadCount == RouteChunk);
  const auto& c = p.config;
  const int lane = threadIdx.x % warpSize;
  const int nRoutes = nTokens * c.topK;
  const int nLocalExperts = c.numExperts / c.worldSize;
  const size_t nRoutesPerRank = size_t(c.maxTokens) * c.topK;
  auto* localIds = at<int>(p.local, p.symmetric.topkIds);
  auto* localScores = at<float>(p.local, p.symmetric.topkWeights);
  auto* weightPackets = at<mscclpp::LL8Packet>(p.local, p.symmetric.routingPackets) + c.rank * nRoutesPerRank;
  const bool shouldStageIds = Publish && ids && ids != localIds;
  const bool shouldStageScores = Publish && scores && scores != localScores;
  const bool shouldSendHeaders = Publish && c.worldSize > 1;
  // Thread `peer` sends that peer the headers of this rank's chunks.
  mscclpp::LLPacket* peerHeaders = nullptr;
  if (shouldSendHeaders && threadIdx.x < c.worldSize && threadIdx.x != c.rank)
    peerHeaders = peerAt<mscclpp::LLPacket>(p, threadIdx.x, p.symmetric.routingHeaders) +
                  size_t(c.rank) * routeChunks(c.maxTokens * c.topK);
  for (int chunkIdx = blockIdx.x; chunkIdx < routeChunks(nRoutes); chunkIdx += gridDim.x) {
    // The previous chunk's header writers reset their owner counts before this chunk reserves entries.
    if (shouldSendHeaders && chunkIdx != blockIdx.x) __syncthreads();
    int routeIndex = chunkIdx * RouteChunk + threadIdx.x;
    int expertId = chunkIdx == blockIdx.x ? firstExpertId : loadRouteId(p, ids, routeIndex, nRoutes);
    if (routeIndex < nRoutes) {
      assert(expertId >= -1 && expertId < c.numExperts);
      if (shouldStageIds) localIds[routeIndex] = expertId;
    }
    int ownerRank = -1;
    int localExpert = -1;
    if (expertId >= 0) {
      ownerRank = c.worldSize > 1 ? expertId / nLocalExperts : 0;
      localExpert = expertId - ownerRank * nLocalExperts;
    }
    if constexpr (Publish) {
      const bool isRemote = ownerRank >= 0 && ownerRank != c.rank;
      if (routeIndex < nRoutes && (shouldStageScores || isRemote)) {
        float weight = scores ? scores[routeIndex] : localScores[routeIndex];
        if (shouldStageScores) localScores[routeIndex] = weight;
        if (isRemote) weightPackets[routeIndex].write(mscclpp::bit_cast<uint32_t>(weight), uint32_t(epoch));
      }
      if (shouldSendHeaders) {
        auto* ownerPackets = isRemote ? peerAt<mscclpp::LL8Packet>(p, ownerRank, p.symmetric.routingPackets) : nullptr;
        int packetEntry = reserveRouteGroup(nRoutesByOwner, isRemote ? ownerRank : -1, lane);
        if (isRemote)
          ownerPackets[c.rank * nRoutesPerRank + chunkIdx * RouteChunk + packetEntry].write(
              packRouteEntry(localExpert, threadIdx.x), uint32_t(epoch));
      }
    }
    handleOwnedRoute(ownerRank == c.rank ? localExpert : -1, uint32_t(c.rank * nRoutesPerRank + routeIndex), lane);
    if (shouldSendHeaders) {
      __syncthreads();
      if (peerHeaders) {
        peerHeaders[chunkIdx].write(uint32_t(nRoutesByOwner[threadIdx.x]), uint32_t(nTokens), uint32_t(epoch));
        nRoutesByOwner[threadIdx.x] = 0;
      }
    }
  }
}

// Warp 0 reads every source's chunk-0 header, whose token count bounds that source's live chunks.
template <class P>
MSCCLPP_DEVICE_INLINE void tabulateSourceChunks(const P& p, uint64_t epoch, const RoutingScratch& scratch) {
  const auto& c = p.config;
  if (threadIdx.x >= warpSize) return;
  const size_t chunkCapacity = routeChunks(c.maxTokens * c.topK);
  auto* headers = at<mscclpp::LLPacket>(p.local, p.symmetric.routingHeaders);
  int maxChunks = 0;
  for (int source = threadIdx.x; source < c.worldSize; source += warpSize) {
    int chunks = 0;
    int entries = 0;
    if (source != c.rank) {
      uint2 header = headers[source * chunkCapacity].read(uint32_t(epoch), SpinLimit);
      entries = int(header.x);
      chunks = routeChunks(int(header.y) * c.topK);
    }
    scratch.nChunksBySource[source] = chunks;
    scratch.nFirstChunkEntries[source] = entries;
    maxChunks = max(maxChunks, chunks);
  }
  maxChunks = __reduce_max_sync(0xffffffff, maxChunks);
  if (threadIdx.x == 0) *scratch.maxSourceChunks = maxChunks;
}

// Process route chunks received from other ranks and pass their entries to
// `handleOwnedRoute(localExpert, globalRoute, lane)`. Invalid lanes pass -1 so every lane still participates in the
// handler's warp collectives. CTAs take (source, chunk) segments round-robin in chunk-major order, so small batches
// spread their sources over separate CTAs. Warp 0 reads the counts of up to RouteSegmentBatch segments, and then every
// warp of the CTA takes an even share of their entries.
template <bool FixedTokenCount, class P, class OwnedRouteHandler>
__device__ void processReceivedRouteChunks(const P& p, int tokens, uint64_t epoch, const RoutingScratch& scratch,
                                           OwnedRouteHandler&& handleOwnedRoute) {
  constexpr int WarpSize = 32;
  static_assert(RouteSegmentBatch == WarpSize);
  const auto& c = p.config;
  const int lane = threadIdx.x % WarpSize;
  const size_t routesPerRank = size_t(c.maxTokens) * c.topK;
  const size_t chunkCapacity = routeChunks(c.maxTokens * c.topK);
  // Equal token counts give every source the same chunks; dynamic counts use the chunk-0 headers.
  const int chunks = FixedTokenCount ? routeChunks(tokens * c.topK) : *scratch.maxSourceChunks;
  const int segments = c.worldSize * chunks;
  auto* headers = at<mscclpp::LLPacket>(p.local, p.symmetric.routingHeaders);
  auto* packets = at<mscclpp::LL8Packet>(p.local, p.symmetric.routingPackets);
  for (int batch = blockIdx.x; batch < segments; batch += gridDim.x * RouteSegmentBatch) {
    if (threadIdx.x < WarpSize) {
      int segment = batch + lane * gridDim.x;
      int source = segment % c.worldSize;
      int chunk = segment / c.worldSize;
      int entries = 0;
      if (segment < segments && source != c.rank) {
        if (!FixedTokenCount && chunk == 0)
          entries = scratch.nFirstChunkEntries[source];
        else if (FixedTokenCount || chunk < scratch.nChunksBySource[source])
          entries = int(headers[source * chunkCapacity + chunk].read(uint32_t(epoch), SpinLimit).x);
      }
      assert(entries <= RouteChunk);
      int end = entries;
#pragma unroll
      for (int distance = 1; distance < WarpSize; distance *= 2) {
        int other = __shfl_up_sync(0xffffffff, end, distance);
        if (lane >= distance) end += other;
      }
      scratch.segmentEnds[lane] = end;
      scratch.segmentFirsts[lane] = uint32_t(source * routesPerRank + size_t(chunk) * RouteChunk);
    }
    __syncthreads();
    const int total = scratch.segmentEnds[RouteSegmentBatch - 1];
    for (int first = threadIdx.x - lane; first < total; first += P::ThreadCount) {
      int item = first + lane;
      int expert = -1;
      uint32_t route = 0;
      if (item < total) {
        // The item's segment is the first one whose inclusive end exceeds it.
        int segment = 0;
#pragma unroll
        for (int step = RouteSegmentBatch / 2; step; step /= 2)
          if (scratch.segmentEnds[segment + step - 1] <= item) segment += step;
        int entry = item - (segment ? scratch.segmentEnds[segment - 1] : 0);
        uint32_t base = scratch.segmentFirsts[segment];
        uint32_t data = packets[base + entry].read(uint32_t(epoch), SpinLimit);
        expert = int(data >> 16);
        route = base + (data & 0xffff);
      }
      handleOwnedRoute(expert, route, lane);
    }
    // The next batch reuses the segment table.
    __syncthreads();
  }
}

// Match FastDivmod's ceil(2^p / divisor) without a device-side 64-bit division. One double-precision
// reciprocal refinement puts the estimate within one integer; the integer product checks make it exact.
MSCCLPP_DEVICE_INLINE cutlass::FastDivmod makeExpertBlockDivisor(int divisor) {
  assert(divisor > 0);
  cutlass::FastDivmod result;
  result.divisor = divisor;
  if (divisor > 1) {
    unsigned p = 31 + cutlass::find_log2(unsigned(divisor));
    uint64_t numerator = uint64_t(1) << p;
    double reciprocal = double(__frcp_rn(float(divisor)));
    double residual = __fma_rn(-double(divisor), reciprocal, 1.0);
    reciprocal = __fma_rn(reciprocal, residual, reciprocal);
    uint64_t multiplier = __double2uint_ru(double(numerator) * reciprocal);
    if (multiplier * unsigned(divisor) < numerator)
      ++multiplier;
    else if ((multiplier - 1) * unsigned(divisor) >= numerator)
      --multiplier;
    result.multiplier = uint32_t(multiplier);
    result.shift_right = p - 32;
  }
  return result;
}

// Warp 0 of every CTA computes the expert prefixes; token-block construction is distributed across CTAs.
// A CTA's per-expert route offsets, when given, become absolute route positions. The caller publishes plan completion.
template <class P>
__device__ int buildTokenBlockMetadata(const P& p, int* routeOffsets = nullptr) {
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
#pragma unroll
    for (int distance = 1; distance < WarpSize; distance *= 2) {
      int other = __shfl_up_sync(0xffffffff, scan, distance);
      if (lane >= distance) scan += other;
    }
    int firstBlock = precedingBlocks + scan - blocks;
    if (expert < experts && blockIdx.x == 0) {
      w.starts[expert] = firstBlock * RoutingTileN;
      // Downstream cache policies read each expert's total routed rows.
      w.cursors[expert] = count;
      if constexpr (W4ExpertMajorTasks)
        w.expertBlockDivisors[expert] = makeExpertBlockDivisor(max(1, count / RoutingTileN));
    }
    if (expert < experts && routeOffsets) routeOffsets[expert] += firstBlock * RoutingTileN;
    for (int block = blockIdx.x; block < blocks; block += gridDim.x)
      w.blocks[firstBlock + block] = TokenBlock{expert, min(int(RoutingTileN), count - block * RoutingTileN)};
    precedingBlocks += __shfl_sync(0xffffffff, scan, WarpSize - 1);
  }
  __syncwarp();
  return precedingBlocks;
}

template <class P>
MSCCLPP_DEVICE_INLINE void resetReadyCounters(const P& p, int tokenBlocks) {
  const auto& w = p.workspace;
  int stride = gridDim.x * P::ThreadCount;
  for (int block = blockIdx.x * P::ThreadCount + threadIdx.x; block < tokenBlocks; block += stride) {
    w.inputReady[size_t(block) * W4ReadyCounterStride] = 0;
    w.hiddenReady[size_t(block) * W4ReadyCounterStride] = 0;
    for (int chunk = 0; chunk < w4InputChunks<typename P::Collective>(p.config.hidden) - 1; ++chunk)
      *w4InputChunkCounter<typename P::Collective>(w, p.config.hidden, block, chunk) = 0;
  }
}

// Every CTA sends its chunks of the rank's routes to their owners and caches its local-expert routes and its share of
// the received entries in the idle GEMM tensor storage, then reserves rows with one device atomic per CTA and expert.
// After a grid barrier, every CTA builds the expert prefix and its token blocks and writes its cached routes; a CTA
// whose routes exceed the scratch rescans them instead.
template <bool FixedTokenCount = false, class P>
__device__ void prepareRoutes(const P& p, int tokens, const int32_t* ids = nullptr, const float* scores = nullptr) {
  using SharedStorage = W4A8SharedStorageT<typename P::Collective>;
  const auto& c = p.config;
  const auto& w = p.workspace;
  // Issue this CTA's first route load ahead of the routing setup, which hides its latency.
  const int firstExpertId = loadRouteId(p, ids, blockIdx.x * RouteChunk + threadIdx.x, tokens * c.topK);
  const int nLocalExperts = c.numExperts / c.worldSize;
  const uint64_t epoch = *at<uint64_t>(p.local, p.symmetric.epoch) + 1;
  if (blockIdx.x == 0 && threadIdx.x == 0) w.control->epoch = epoch;
  if ((FixedTokenCount || c.worldSize == 1) && tokens == 0) {
    // Every rank is empty, so nothing is exchanged.
    if (blockIdx.x == 0 && threadIdx.x < warpSize) {
      int blocks = buildTokenBlockMetadata(p);
      if (threadIdx.x == 0) w.control->tokenBlocks = blocks;
    }
    syncResidentGrid(&w.control->residentGridSyncState, gridDim.x);
    return;
  }

  extern __shared__ __align__(1024) char storage[];
  auto& shared = *reinterpret_cast<SharedStorage*>(storage);
  const RoutingScratch scratch = makeRoutingScratch(shared, nLocalExperts, c.worldSize);
  // Each thread clears at most a few counters, so unrolling would only lengthen the setup every warp executes.
#pragma unroll 1
  for (int counter = threadIdx.x; counter <= nLocalExperts + c.worldSize; counter += P::ThreadCount)
    scratch.nRoutesByExpert[counter] = 0;
  __syncthreads();
  auto cacheOwnedRoute = [&](int expert, uint32_t route, int lane) { cacheRoute(scratch, expert, route, lane); };
  processSourceRouteChunks<true>(p, tokens, ids, scores, epoch, scratch.nRoutesByOwner, firstExpertId, cacheOwnedRoute);
  if (c.worldSize > 1) {
    if constexpr (!FixedTokenCount) {
      tabulateSourceChunks(p, epoch, scratch);
      __syncthreads();
    }
    processReceivedRouteChunks<FixedTokenCount>(p, tokens, epoch, scratch, cacheOwnedRoute);
  }
  __syncthreads();
  // One device atomic per CTA and expert reserves the CTA's contiguous range of the expert's rows.
#pragma unroll 1
  for (int expert = threadIdx.x; expert < nLocalExperts; expert += P::ThreadCount) {
    int count = scratch.nRoutesByExpert[expert];
    scratch.routeOffsets[expert] =
        count ? atomicFetchAdd<int, scopeDevice>(w.counts + expert, count, memoryOrderRelaxed) : 0;
  }
  syncResidentGrid(&w.control->residentGridSyncState, gridDim.x);
  if (threadIdx.x < warpSize) {
    int blocks = buildTokenBlockMetadata(p, scratch.routeOffsets);
    if (threadIdx.x == 0) {
      *scratch.nTokenBlocks = blocks;
      if (blockIdx.x == 0) w.control->tokenBlocks = blocks;
    }
  }
  __syncthreads();
  resetReadyCounters(p, *scratch.nTokenBlocks);
  const int nRoutes = *scratch.nRoutes;
  if (nRoutes <= scratch.capacity) {
#pragma unroll 1
    for (int slot = threadIdx.x; slot < nRoutes; slot += P::ThreadCount) {
      CachedRoute entry = scratch.cachedRoutes[slot];
      w.routes[scratch.routeOffsets[entry.expert] + entry.position] = makeRoute(p, entry.route);
    }
  } else {
    // Rescan this CTA's routes and take positions directly from its absolute expert offsets.
    auto placeOwnedRoute = [&](int expert, uint32_t route, int lane) {
      int position = reserveRouteGroup(scratch.routeOffsets, expert, lane);
      if (position >= 0) w.routes[position] = makeRoute(p, route);
    };
    processSourceRouteChunks<false>(p, tokens, ids, scores, epoch, scratch.nRoutesByOwner, firstExpertId,
                                    placeOwnedRoute);
    if (c.worldSize > 1) processReceivedRouteChunks<FixedTokenCount>(p, tokens, epoch, scratch, placeOwnedRoute);
  }
#if MSCCLPP_BULK_AVAILABLE
  // Hand the shared routing scratch back to the asynchronous GEMM tensor loads.
  bulkFence();
#endif
  syncResidentGrid(&w.control->residentGridSyncState, gridDim.x);
  // Every prefix reader has finished. Clear counts for the next launch, which follows this kernel's completion.
#pragma unroll 1
  for (int expert = blockIdx.x * P::ThreadCount + threadIdx.x; expert < nLocalExperts;
       expert += gridDim.x * P::ThreadCount)
    w.counts[expert] = 0;
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_ROUTING_CUH_
