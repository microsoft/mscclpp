// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_ROUTING_CUH_
#define MSCCLPP_EXT_MEGAMOE_ROUTING_CUH_

#include "megamoe_device.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

// Routing exchange. Each rank splits its own routes into chunks of RouteChunk consecutive routes, and one CTA pass
// handles a chunk with one route per thread. A route whose expert lives on another rank goes only to that owner: the
// chunk packs the owner's routes into LL8 entries at routingPackets[source][chunk * RouteChunk + entry] on the owner,
// then sends every peer one LL8 header {entries} at routingHeaders[source][chunk]. With dynamic token counts, a rank
// also writes its token count to every peer's LL8 tokenCount[source] at launch. Routes of local experts never leave
// the source CTA. Entries, headers, and token counts carry the launch epoch, so owners poll only their own memory.
// A rank's own routingPackets slice instead holds the LL8 weights of its remote routes, which owners read in dispatch.

// Pack a route entry: the owner's local expert and the route's offset in its source chunk.
MSCCLPP_DEVICE_INLINE uint32_t packRouteEntry(int expertIdx, int offset) {
  return uint32_t(expertIdx) << 16 | uint32_t(offset);
}

// Decode a global route index, source * tokensPerRank * topK + the source's route index.
template <class KernelParams>
MSCCLPP_DEVICE_INLINE Route makeRoute(const KernelParams& params, uint32_t route, int nTokens = 0) {
  const int topK = params.config.topK;
  const int tokensPerRank = (params.config.computeOnly && nTokens > 0) ? nTokens : params.config.maxTokens;
  const uint32_t nRoutesPerRank = uint32_t(tokensPerRank) * topK;
  int rank = params.config.worldSize > 1 ? int(route / nRoutesPerRank) : 0;
  int index = int(route - uint32_t(rank) * nRoutesPerRank);
  if ((topK & (topK - 1)) == 0) return Route{rank, index >> (__ffs(topK) - 1), index & (topK - 1), 0.0f};
  return Route{rank, index / topK, index % topK, 0.0f};
}

// Reserve consecutive positions from a shared counter per key, one atomic per group of lanes sharing a key.
// All lanes of the warp must call this; lanes with a negative key receive -1.
MSCCLPP_DEVICE_INLINE int reserveRouteGroup(int* counters, int key, int laneIdx) {
  unsigned int group = __match_any_sync(0xffffffff, key);
  if (key < 0) return -1;
  int leader = __ffs(group) - 1;
  int base = 0;
  if (laneIdx == leader) base = atomicAddShared(counters + key, __popc(group));
  base = __shfl_sync(group, base, leader);
  return base + __popc(group & ((1u << laneIdx) - 1));
}

// Reserve one shared slot per lane in `lanes`, with one atomic per warp. All lanes of the warp must call this.
MSCCLPP_DEVICE_INLINE int reserveWarpSlots(int* counter, unsigned int lanes, int laneIdx) {
  int base = 0;
  if (laneIdx == 0) base = atomicAddShared(counter, __popc(lanes));
  base = __shfl_sync(0xffffffff, base, 0);
  return base + __popc(lanes & ((1u << laneIdx) - 1));
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
  int* nTokenTiles;
  int* maxNLiveChunks;  // most live chunks of any source, for dynamic token counts
  CachedRoute* cachedRoutes;
  int capacity;
};

template <class SharedStorage>
MSCCLPP_DEVICE_INLINE RoutingScratch makeRoutingScratch(SharedStorage& sharedStorage, int nLocalExperts,
                                                        int worldSize) {
  constexpr size_t NumScratchInts = sizeof(SharedStorage::routing) / sizeof(int);
  size_t nHeaderInts = routingScratchHeaderInts(nLocalExperts, worldSize);
  assert(nLocalExperts <= 0xffff && nHeaderInts <= NumScratchInts);
  RoutingScratch scratch;
  scratch.nRoutesByExpert = sharedStorage.routing;
  scratch.nRoutesByOwner = scratch.nRoutesByExpert + nLocalExperts;
  scratch.nRoutes = scratch.nRoutesByOwner + worldSize;
  scratch.routeOffsets = scratch.nRoutes + 1;
  scratch.nChunksBySource = scratch.routeOffsets + nLocalExperts;
  scratch.nFirstChunkEntries = scratch.nChunksBySource + worldSize;
  scratch.segmentEnds = scratch.nFirstChunkEntries + worldSize;
  scratch.segmentFirsts = reinterpret_cast<uint32_t*>(scratch.segmentEnds + RouteSegmentBatch);
  scratch.nTokenTiles = reinterpret_cast<int*>(scratch.segmentFirsts + RouteSegmentBatch);
  scratch.maxNLiveChunks = scratch.nTokenTiles + 1;
  scratch.cachedRoutes = reinterpret_cast<CachedRoute*>(sharedStorage.routing + nHeaderInts);
  scratch.capacity = int(min((NumScratchInts - nHeaderInts) / 2, size_t(1) << 16));
  return scratch;
}

// Cache one route per lane. All lanes of the warp must call this; lanes without a local route pass expert -1.
MSCCLPP_DEVICE_INLINE void cacheRoute(const RoutingScratch& scratch, int expertIdx, uint32_t route, int laneIdx) {
  unsigned int localLanes = __ballot_sync(0xffffffff, expertIdx >= 0);
  if (!localLanes) return;
  int position = reserveRouteGroup(scratch.nRoutesByExpert, expertIdx, laneIdx);
  int slot = reserveWarpSlots(scratch.nRoutes, localLanes, laneIdx);
  if (expertIdx >= 0 && slot < scratch.capacity)
    scratch.cachedRoutes[slot] = CachedRoute{route, uint16_t(expertIdx), uint16_t(position)};
}

// Load the expert ID of the rank's own route `routeIndex`, or -1 past its live routes.
template <class KernelParams>
MSCCLPP_DEVICE_INLINE int loadRouteId(const KernelParams& params, const int32_t* ids, int routeIndex, int nRoutes) {
  if (routeIndex >= nRoutes) return -1;
  return ids ? ids[routeIndex] : at<int>(params.local, params.symmetric.topkIds)[routeIndex];
}

// Issue this CTA's first route load ahead of routing setup to hide global-memory latency.
template <class KernelParams>
MSCCLPP_DEVICE_INLINE int prefetchFirstChunkRouteId(const KernelParams& params, const int32_t* ids, int nRoutes) {
  return loadRouteId(params, ids, blockIdx.x * RouteChunk + threadIdx.x, nRoutes);
}

// Process this rank's source-route chunks and pass each lane to
// `handleOwnedRoute(localExpert, globalRoute, laneIdx)`. A lane whose route belongs to another rank passes -1 so every
// lane still participates in the handler's warp collectives. With Publish, the processing also stages direct inputs,
// sends remote-expert routes and their weights, and ends every chunk with the peer headers. `firstChunkExpertId` is
// this thread's expert ID in the CTA's first chunk, prefetched by the caller.
template <bool Publish, class KernelParams, class OwnedRouteHandler>
__device__ void processSourceRouteChunks(const KernelParams& params, int nTokens, const int32_t* ids,
                                         const float* scores, uint64_t epoch, int* nRoutesByOwner,
                                         int firstChunkExpertId, OwnedRouteHandler&& handleOwnedRoute) {
  static_assert(KernelParams::ThreadCount == RouteChunk);
  const auto& config = params.config;
  const int laneIdx = threadIdx.x % WARP_SIZE;
  const int nRoutes = nTokens * config.topK;
  const int nLocalExperts = config.numExperts / config.worldSize;
  const size_t nRoutesPerRank = size_t(config.maxTokens) * config.topK;
  auto* localIds = at<int>(params.local, params.symmetric.topkIds);
  auto* localScores = at<float>(params.local, params.symmetric.topkWeights);
  auto* weightPackets =
      at<mscclpp::LL8Packet>(params.local, params.symmetric.routingPackets) + config.rank * nRoutesPerRank;
  const bool shouldStageIds = Publish && ids && ids != localIds;
  const bool shouldStageScores = Publish && scores && scores != localScores;
  const bool shouldSendHeaders = Publish && config.worldSize > 1;
  const int nLiveChunks = nRoutes ? routeChunks(nRoutes) : 0;
  // Thread `peer` sends that peer the headers of this rank's chunks.
  mscclpp::LL8Packet* peerHeaders = nullptr;
  if (shouldSendHeaders && threadIdx.x < config.worldSize && threadIdx.x != config.rank)
    peerHeaders = peerAt<mscclpp::LL8Packet>(params, threadIdx.x, params.symmetric.routingHeaders) +
                  size_t(config.rank) * routeChunks(config.maxTokens * config.topK);
  for (int chunkIdx = blockIdx.x; chunkIdx < nLiveChunks; chunkIdx += gridDim.x) {
    // The previous chunk's header writers reset their owner counts before this chunk reserves entries.
    if (shouldSendHeaders && chunkIdx != blockIdx.x) __syncthreads();
    int routeIndex = chunkIdx * RouteChunk + threadIdx.x;
    int expertId = chunkIdx == blockIdx.x ? firstChunkExpertId : loadRouteId(params, ids, routeIndex, nRoutes);
    if (routeIndex < nRoutes) {
      assert(expertId >= -1 && expertId < config.numExperts);
      if (shouldStageIds) localIds[routeIndex] = expertId;
    }
    int ownerRank = -1;
    int localExpert = -1;
    if (expertId >= 0) {
      ownerRank = config.worldSize > 1 ? expertId / nLocalExperts : 0;
      localExpert = expertId - ownerRank * nLocalExperts;
    }
    if constexpr (Publish) {
      const bool isRemote = ownerRank >= 0 && ownerRank != config.rank;
      const bool sendRemoteWeights = isRemote && !config.combineRouteWeights;
      if (routeIndex < nRoutes && (shouldStageScores || sendRemoteWeights)) {
        float weight = scores ? scores[routeIndex] : localScores[routeIndex];
        if (shouldStageScores) localScores[routeIndex] = weight;
        if (sendRemoteWeights) weightPackets[routeIndex].write(mscclpp::bit_cast<uint32_t>(weight), uint32_t(epoch));
      }
      if (shouldSendHeaders) {
        auto* ownerPackets =
            isRemote ? peerAt<mscclpp::LL8Packet>(params, ownerRank, params.symmetric.routingPackets) : nullptr;
        int packetEntry = reserveRouteGroup(nRoutesByOwner, isRemote ? ownerRank : -1, laneIdx);
        if (isRemote)
          ownerPackets[config.rank * nRoutesPerRank + chunkIdx * RouteChunk + packetEntry].write(
              packRouteEntry(localExpert, threadIdx.x), uint32_t(epoch));
      }
    }
    handleOwnedRoute(ownerRank == config.rank ? localExpert : -1, uint32_t(config.rank * nRoutesPerRank + routeIndex),
                     laneIdx);
    if (shouldSendHeaders) {
      __syncthreads();
      if (peerHeaders) {
        peerHeaders[chunkIdx].write(uint32_t(nRoutesByOwner[threadIdx.x]), uint32_t(epoch));
        nRoutesByOwner[threadIdx.x] = 0;
      }
    }
  }
}

// Warp 0 reads every source's early token-count packet, which bounds that source's live chunks.
template <class KernelParams>
MSCCLPP_DEVICE_INLINE void tabulateSourceChunks(const KernelParams& params, uint64_t epoch,
                                                const RoutingScratch& scratch) {
  const auto& config = params.config;
  if (threadIdx.x >= WARP_SIZE) return;
  auto* tokenCounts = at<mscclpp::LL8Packet>(params.local, params.symmetric.tokenCount);
  int maxNLiveChunks = 0;
  for (int source = threadIdx.x; source < config.worldSize; source += WARP_SIZE) {
    int nLiveChunks = 0;
    if (source != config.rank) {
      int nSourceTokens = int(tokenCounts[source].read(uint32_t(epoch), SpinLimit));
      nLiveChunks = nSourceTokens ? routeChunks(nSourceTokens * config.topK) : 0;
    }
    scratch.nChunksBySource[source] = nLiveChunks;
    maxNLiveChunks = max(maxNLiveChunks, nLiveChunks);
  }
  maxNLiveChunks = __reduce_max_sync(0xffffffff, maxNLiveChunks);
  if (threadIdx.x == 0) *scratch.maxNLiveChunks = maxNLiveChunks;
}

// Process route chunks received from other ranks and pass their entries to
// `handleOwnedRoute(localExpert, globalRoute, laneIdx)`. Invalid lanes pass -1 so every lane still participates in the
// handler's warp collectives. CTAs take (source, chunk) segments round-robin in reverse CTA order so CTAs without
// local source chunks take the first received segments. Warp 0 reads the counts of up to RouteSegmentBatch segments,
// and then every warp of the CTA takes an even share of their entries.
template <bool FixedTokenCount, class KernelParams, class OwnedRouteHandler>
__device__ void processReceivedRouteChunks(const KernelParams& params, int nTokens, uint64_t epoch,
                                           const RoutingScratch& scratch, OwnedRouteHandler&& handleOwnedRoute) {
  static_assert(RouteSegmentBatch == WARP_SIZE);
  const auto& config = params.config;
  const int laneIdx = threadIdx.x % WARP_SIZE;
  const int recvBlock = gridDim.x - 1 - blockIdx.x;
  const size_t nRoutesPerRank = size_t(config.maxTokens) * config.topK;
  const size_t nChunksPerRank = routeChunks(config.maxTokens * config.topK);
  // Equal token counts give every source the same chunks; dynamic counts use the early tokenCount packets.
  const int nLiveChunks =
      FixedTokenCount ? (nTokens ? routeChunks(nTokens * config.topK) : 0) : *scratch.maxNLiveChunks;
  const int nSegments = config.worldSize * nLiveChunks;
  auto* headers = at<mscclpp::LL8Packet>(params.local, params.symmetric.routingHeaders);
  auto* packets = at<mscclpp::LL8Packet>(params.local, params.symmetric.routingPackets);
  for (int batch = recvBlock; batch < nSegments; batch += gridDim.x * RouteSegmentBatch) {
    if (threadIdx.x < WARP_SIZE) {
      int segment = batch + laneIdx * gridDim.x;
      int source = segment % config.worldSize;
      int chunk = segment / config.worldSize;
      int nEntries = 0;
      if (segment < nSegments && source != config.rank) {
        if (FixedTokenCount || chunk < scratch.nChunksBySource[source])
          nEntries = int(headers[source * nChunksPerRank + chunk].read(uint32_t(epoch), SpinLimit));
      }
      assert(nEntries <= RouteChunk);
      int end = nEntries;
#pragma unroll
      for (int distance = 1; distance < WARP_SIZE; distance *= 2) {
        int other = __shfl_up_sync(0xffffffff, end, distance);
        if (laneIdx >= distance) end += other;
      }
      scratch.segmentEnds[laneIdx] = end;
      scratch.segmentFirsts[laneIdx] = uint32_t(source * nRoutesPerRank + size_t(chunk) * RouteChunk);
    }
    __syncthreads();
    const int nBatchEntries = scratch.segmentEnds[RouteSegmentBatch - 1];
    for (int first = threadIdx.x - laneIdx; first < nBatchEntries; first += KernelParams::ThreadCount) {
      int item = first + laneIdx;
      int expertIdx = -1;
      uint32_t route = 0;
      if (item < nBatchEntries) {
        // The item's segment is the first one whose inclusive end exceeds it.
        int segment = 0;
#pragma unroll
        for (int step = RouteSegmentBatch / 2; step; step /= 2)
          if (scratch.segmentEnds[segment + step - 1] <= item) segment += step;
        int entry = item - (segment ? scratch.segmentEnds[segment - 1] : 0);
        uint32_t base = scratch.segmentFirsts[segment];
        uint32_t data = packets[base + entry].read(uint32_t(epoch), SpinLimit);
        expertIdx = int(data >> 16);
        route = base + (data & 0xffff);
      }
      handleOwnedRoute(expertIdx, route, laneIdx);
    }
    // The next batch reuses the segment table.
    __syncthreads();
  }
}

// Warp 0 of every CTA computes the expert prefixes; token-tile construction is distributed across CTAs.
// A CTA's per-expert route offsets, when given, become absolute route positions. The caller publishes plan completion.
template <class KernelParams>
__device__ int buildTokenTileMetadata(const KernelParams& params, int* routeOffsets = nullptr) {
  constexpr int RoutingTileN = KernelParams::Tiles::N;
  constexpr int MaxGroupBlocks = 512 / RoutingTileN > 1 ? 512 / RoutingTileN : 1;
  constexpr int MaxGroupShift = MaxGroupBlocks == 16 ? 4 : (MaxGroupBlocks == 8 ? 3 : (MaxGroupBlocks == 4 ? 2 : 0));
  const auto& workspace = params.workspace;
  int laneIdx = threadIdx.x % WARP_SIZE;
  int nLocalExperts = params.config.numExperts / params.config.worldSize;
  int nPrecedingBlocks = 0;
  for (int base = 0; base < nLocalExperts; base += WARP_SIZE) {
    int expertIdx = base + laneIdx;
    int nExpertRows = expertIdx < nLocalExperts ? workspace.counts[expertIdx] : 0;
    int nExpertBlocks = (nExpertRows + RoutingTileN - 1) / RoutingTileN;
    int nFullBlocks = nExpertRows / RoutingTileN;
    int scan = nExpertBlocks;
#pragma unroll
    for (int distance = 1; distance < WARP_SIZE; distance *= 2) {
      int other = __shfl_up_sync(0xffffffff, scan, distance);
      if (laneIdx >= distance) scan += other;
    }
    int firstBlock = nPrecedingBlocks + scan - nExpertBlocks;
    int warpBlocks = __shfl_sync(0xffffffff, scan, WARP_SIZE - 1);
    int warpExperts = min(WARP_SIZE, nLocalExperts - base);
    if (expertIdx < nLocalExperts && blockIdx.x == 0) {
      workspace.starts[expertIdx] = firstBlock * RoutingTileN;
      // Downstream cache policies read each expert's total routed rows.
      workspace.cursors[expertIdx] = nExpertRows;
    }
    if (expertIdx < nLocalExperts && routeOffsets) routeOffsets[expertIdx] += firstBlock * RoutingTileN;
    const bool weightEvictNormal = nExpertRows > RoutingTileN;
    for (int block = blockIdx.x; block < nExpertBlocks; block += gridDim.x) {
      int groupShift = 0;
      int tileIdxInGroup = 0;
      if (warpBlocks < 2 * warpExperts) {
        int xorBits = nFullBlocks ^ block;
        groupShift = block < nFullBlocks ? (xorBits >= MaxGroupBlocks ? MaxGroupShift : (31 - __clz(xorBits))) : 0;
        tileIdxInGroup = block & ((1 << groupShift) - 1);
      }
      workspace.tokenTiles[firstBlock + block] =
          TokenTile::make(expertIdx, min(int(RoutingTileN), nExpertRows - block * RoutingTileN), tileIdxInGroup,
                          groupShift, weightEvictNormal);
    }
    nPrecedingBlocks += warpBlocks;
  }
  __syncwarp();
  return nPrecedingBlocks;
}

template <class KernelParams>
MSCCLPP_DEVICE_INLINE void resetReadyCounters(const KernelParams& params, int nTokenTiles) {
  const auto& workspace = params.workspace;
  int stride = gridDim.x * KernelParams::ThreadCount;
  for (int block = blockIdx.x * KernelParams::ThreadCount + threadIdx.x; block < nTokenTiles; block += stride) {
    workspace.inputReady[size_t(block) * W4ReadyCounterStride] = 0;
    workspace.hiddenReady[size_t(block) * W4ReadyCounterStride] = 0;
    for (int chunk = 0; chunk < w4InputChunks<typename KernelParams::Collective>(params.config.hidden) - 1; ++chunk)
      *w4InputChunkCounter<typename KernelParams::Collective>(workspace, params.config.hidden, block, chunk) = 0;
  }
}

template <bool FixedTokenCount, class KernelParams>
MSCCLPP_DEVICE_INLINE uint64_t beginRoutingEpoch(const KernelParams& params, int nTokens) {
  const auto& config = params.config;
  const uint64_t epoch = *at<uint64_t>(params.local, params.symmetric.epoch) + 1;
  if (blockIdx.x == 0 && threadIdx.x == 0) params.workspace.control->epoch = epoch;
  if constexpr (!FixedTokenCount) {
    if (config.worldSize > 1 && blockIdx.x == gridDim.x - 1) {
      for (int peer = threadIdx.x; peer < config.worldSize; peer += KernelParams::ThreadCount) {
        if (peer != config.rank) {
          peerAt<mscclpp::LL8Packet>(params, peer, params.symmetric.tokenCount)[config.rank].write(uint32_t(nTokens),
                                                                                                   uint32_t(epoch));
        }
      }
    }
  }
  return epoch;
}

template <bool FixedTokenCount, class KernelParams>
MSCCLPP_DEVICE_INLINE bool isGloballyEmptyRoutingBatch(const KernelParams& params, int nTokens) {
  return (FixedTokenCount || params.config.worldSize == 1) && nTokens == 0;
}

template <class KernelParams>
MSCCLPP_DEVICE_INLINE void completeEmptyRoutingBatch(const KernelParams& params) {
  if (blockIdx.x == 0 && threadIdx.x < WARP_SIZE) {
    int nTokenTiles = buildTokenTileMetadata(params);
    if (threadIdx.x == 0) params.workspace.control->nTokenTiles = nTokenTiles;
  }
  syncResidentGrid(&params.workspace.control->residentGridSyncState, gridDim.x);
}

template <class KernelParams>
MSCCLPP_DEVICE_INLINE RoutingScratch initRoutingScratch(const KernelParams& params) {
  using SharedStorage = typename KernelParams::SharedStorage;
  extern __shared__ __align__(1024) char rawSharedStorage[];
  auto& sharedStorage = *reinterpret_cast<SharedStorage*>(rawSharedStorage);
  const int nLocalExperts = params.config.numExperts / params.config.worldSize;
  const RoutingScratch scratch = makeRoutingScratch(sharedStorage, nLocalExperts, params.config.worldSize);
  // Each thread clears at most a few counters, so unrolling would only lengthen the setup every warp executes.
#pragma unroll 1
  for (int counter = threadIdx.x; counter <= nLocalExperts + params.config.worldSize;
       counter += KernelParams::ThreadCount) {
    scratch.nRoutesByExpert[counter] = 0;
  }
  __syncthreads();
  return scratch;
}

// Reserves each CTA's per-expert row range, builds token-tile metadata across the resident grid, and zeroes tile
// readiness counters before routes are materialized.
template <class KernelParams>
MSCCLPP_DEVICE_INLINE void buildRoutingTilePlan(const KernelParams& params, const RoutingScratch& scratch) {
  const auto& workspace = params.workspace;
  const int nLocalExperts = params.config.numExperts / params.config.worldSize;
#pragma unroll 1
  for (int expertIdx = threadIdx.x; expertIdx < nLocalExperts; expertIdx += KernelParams::ThreadCount) {
    int nExpertRoutes = scratch.nRoutesByExpert[expertIdx];
    scratch.routeOffsets[expertIdx] =
        nExpertRoutes
            ? atomicFetchAdd<int, scopeDevice>(workspace.counts + expertIdx, nExpertRoutes, memoryOrderRelaxed)
            : 0;
  }
  syncResidentGrid(&workspace.control->residentGridSyncState, gridDim.x);
  if (threadIdx.x < WARP_SIZE) {
    int nTokenTiles = buildTokenTileMetadata(params, scratch.routeOffsets);
    if (threadIdx.x == 0) {
      *scratch.nTokenTiles = nTokenTiles;
      if (blockIdx.x == 0) workspace.control->nTokenTiles = nTokenTiles;
    }
  }
  __syncthreads();
  resetReadyCounters(params, *scratch.nTokenTiles);
}

template <class KernelParams>
MSCCLPP_DEVICE_INLINE void writeCachedRoutes(const KernelParams& params, const RoutingScratch& scratch,
                                             int nCachedRoutes, int nTokens = 0) {
#pragma unroll 1
  for (int slot = threadIdx.x; slot < nCachedRoutes; slot += KernelParams::ThreadCount) {
    CachedRoute entry = scratch.cachedRoutes[slot];
    params.workspace.routes[scratch.routeOffsets[entry.expert] + entry.position] =
        makeRoute(params, entry.route, nTokens);
  }
}

// Publishes global route writes, hands the shared routing scratch back to async GEMM TMA loads, and clears per-expert
// counts for the next launch.
template <class KernelParams>
MSCCLPP_DEVICE_INLINE void finalizeRoutingStage(const KernelParams& params) {
  const auto& workspace = params.workspace;
  const int nLocalExperts = params.config.numExperts / params.config.worldSize;
#if MSCCLPP_BULK_AVAILABLE
  bulkFence();
#endif
  syncResidentGrid(&workspace.control->residentGridSyncState, gridDim.x);
#pragma unroll 1
  for (int expertIdx = blockIdx.x * KernelParams::ThreadCount + threadIdx.x; expertIdx < nLocalExperts;
       expertIdx += gridDim.x * KernelParams::ThreadCount) {
    workspace.counts[expertIdx] = 0;
  }
}

template <class KernelParams, class OwnedRouteHandler>
MSCCLPP_DEVICE_INLINE void scanLocalGatherRoutes(const KernelParams& params, int nRoutes, const int32_t* ids,
                                                 int firstChunkExpertId, OwnedRouteHandler&& handleOwnedRoute) {
  const int laneIdx = threadIdx.x % WARP_SIZE;
  const int nLocalExperts = params.config.numExperts / params.config.worldSize;
  const int firstLocalExpert = params.config.rank * nLocalExperts;
  const int nLiveChunks = routeChunks(nRoutes);
  for (int chunkIdx = blockIdx.x; chunkIdx < nLiveChunks; chunkIdx += gridDim.x) {
    int routeIndex = chunkIdx * RouteChunk + threadIdx.x;
    int expertId = chunkIdx == blockIdx.x ? firstChunkExpertId : loadRouteId(params, ids, routeIndex, nRoutes);
    int localExpert = (expertId >= firstLocalExpert && expertId < firstLocalExpert + nLocalExperts)
                          ? (expertId - firstLocalExpert)
                          : -1;
    handleOwnedRoute(localExpert, uint32_t(routeIndex), laneIdx);
  }
}

template <class KernelParams>
MSCCLPP_DEVICE_INLINE void collectLocalGatherRoutes(const KernelParams& params, const RoutingScratch& scratch,
                                                    int nRoutes, const int32_t* ids, int firstChunkExpertId) {
  scanLocalGatherRoutes(params, nRoutes, ids, firstChunkExpertId, [&](int expertIdx, uint32_t route, int laneIdx) {
    cacheRoute(scratch, expertIdx, route, laneIdx);
  });
  __syncthreads();
}

template <class KernelParams>
MSCCLPP_DEVICE_INLINE void materializeLocalGatherRoutes(const KernelParams& params, const RoutingScratch& scratch,
                                                        int nTokens, int nRoutes, const int32_t* ids,
                                                        int firstChunkExpertId) {
  const int nCachedRoutes = *scratch.nRoutes;
  if (nCachedRoutes <= scratch.capacity) {
    writeCachedRoutes(params, scratch, nCachedRoutes, nTokens);
    return;
  }
  scanLocalGatherRoutes(params, nRoutes, ids, firstChunkExpertId, [&](int expertIdx, uint32_t route, int laneIdx) {
    int position = reserveRouteGroup(scratch.routeOffsets, expertIdx, laneIdx);
    if (position >= 0) params.workspace.routes[position] = makeRoute(params, route, nTokens);
  });
}

template <bool FixedTokenCount, class KernelParams>
MSCCLPP_DEVICE_INLINE void collectDistributedRoutes(const KernelParams& params, const RoutingScratch& scratch,
                                                    int nTokens, const int32_t* ids, const float* scores,
                                                    uint64_t epoch, int firstChunkExpertId) {
  auto cacheOwnedRoute = [&](int expertIdx, uint32_t route, int laneIdx) {
    cacheRoute(scratch, expertIdx, route, laneIdx);
  };
  processSourceRouteChunks<true>(params, nTokens, ids, scores, epoch, scratch.nRoutesByOwner, firstChunkExpertId,
                                 cacheOwnedRoute);
  if (params.config.worldSize > 1) {
    if constexpr (!FixedTokenCount) {
      tabulateSourceChunks(params, epoch, scratch);
      __syncthreads();
    }
    processReceivedRouteChunks<FixedTokenCount>(params, nTokens, epoch, scratch, cacheOwnedRoute);
  }
  __syncthreads();
}

template <bool FixedTokenCount, class KernelParams>
MSCCLPP_DEVICE_INLINE void materializeDistributedRoutes(const KernelParams& params, const RoutingScratch& scratch,
                                                        int nTokens, const int32_t* ids, const float* scores,
                                                        uint64_t epoch, int firstChunkExpertId) {
  const int nCachedRoutes = *scratch.nRoutes;
  if (nCachedRoutes <= scratch.capacity) {
    writeCachedRoutes(params, scratch, nCachedRoutes);
    return;
  }
  // Rescan this CTA's routes and take positions directly from its absolute expert offsets.
  auto placeOwnedRoute = [&](int expertIdx, uint32_t route, int laneIdx) {
    int position = reserveRouteGroup(scratch.routeOffsets, expertIdx, laneIdx);
    if (position >= 0) params.workspace.routes[position] = makeRoute(params, route);
  };
  processSourceRouteChunks<false>(params, nTokens, ids, scores, epoch, scratch.nRoutesByOwner, firstChunkExpertId,
                                  placeOwnedRoute);
  if (params.config.worldSize > 1) {
    processReceivedRouteChunks<FixedTokenCount>(params, nTokens, epoch, scratch, placeOwnedRoute);
  }
}

template <class KernelParams>
__device__ void prepareLocalRoutes(const KernelParams& params, int nTokens, const int32_t* ids) {
  const int nRoutes = params.config.worldSize * nTokens * params.config.topK;
  const int firstChunkExpertId = prefetchFirstChunkRouteId(params, ids, nRoutes);
  if (nTokens == 0) {
    completeEmptyRoutingBatch(params);
    return;
  }

  const RoutingScratch scratch = initRoutingScratch(params);
  collectLocalGatherRoutes(params, scratch, nRoutes, ids, firstChunkExpertId);
  buildRoutingTilePlan(params, scratch);
  materializeLocalGatherRoutes(params, scratch, nTokens, nRoutes, ids, firstChunkExpertId);
  finalizeRoutingStage(params);
}

template <bool FixedTokenCount = false, class KernelParams>
__device__ void prepareRoutes(const KernelParams& params, int nTokens, const int32_t* ids = nullptr,
                              const float* scores = nullptr) {
  const int firstChunkExpertId = prefetchFirstChunkRouteId(params, ids, nTokens * params.config.topK);
  const uint64_t epoch = beginRoutingEpoch<FixedTokenCount>(params, nTokens);
  if (isGloballyEmptyRoutingBatch<FixedTokenCount>(params, nTokens)) {
    completeEmptyRoutingBatch(params);
    return;
  }

  const RoutingScratch scratch = initRoutingScratch(params);
  cudaGridDependencySynchronize();
  collectDistributedRoutes<FixedTokenCount>(params, scratch, nTokens, ids, scores, epoch, firstChunkExpertId);
  buildRoutingTilePlan(params, scratch);
  materializeDistributedRoutes<FixedTokenCount>(params, scratch, nTokens, ids, scores, epoch, firstChunkExpertId);
  finalizeRoutingStage(params);
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_ROUTING_CUH_
