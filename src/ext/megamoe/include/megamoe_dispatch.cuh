// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_DISPATCH_CUH_
#define MSCCLPP_EXT_MEGAMOE_DISPATCH_CUH_

#include "megamoe_routing.cuh"
#include "megamoe_trace.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

template <class P>
MSCCLPP_DEVICE_INLINE float loadRouteWeight(const P& p, const Route& route) {
  size_t index = size_t(route.token) * p.config.topK + route.slot;
  uint32_t bits;
  if (route.rank == p.config.rank) {
    auto* address = at<float>(p.local, p.symmetric.topkWeights) + index;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(bits) : "l"(address) : "memory");
  } else {
    // Each remote score has its own epoch, so it does not rely on the ID packet to publish side data.
    auto* packets = peerAt<mscclpp::LL8Packet>(p, route.rank, p.symmetric.routingPackets);
    size_t packetIndex = size_t(route.rank) * p.config.maxTokens * p.config.topK + index;
    bits = packets[packetIndex].read(uint32_t(p.workspace.control->epoch), SpinLimit);
  }
  return mscclpp::bit_cast<float>(bits);
}

template <class Types, class Storage>
__device__ __forceinline__ void publishDispatchChunk(Storage& storage, int localWarp, int* counter, bool hasRow,
                                                     uint32_t phase) {
#if MSCCLPP_BULK_AVAILABLE
  // Every warp arrives, including padded rows; only live rows contribute to readiness.
  constexpr uint32_t Arrival = 1u << 16;
  uint32_t contribution = Arrival + uint32_t(hasRow);
  uint32_t sharedCounter = static_cast<uint32_t>(__cvta_generic_to_shared(&storage.arrivals));
  asm volatile("red.release.cta.shared.add.u32 [%0], %1;" ::"r"(sharedCounter), "r"(contribution) : "memory");
  if (localWarp == 0) {
    uint32_t ready;
    POLL_MAYBE_JAILBREAK((ready = atomicLoad<uint32_t, cuda::thread_scope_block>(
                              &storage.arrivals, memoryOrderAcquire)) < Types::DispatchWarps * Arrival,
                         SpinLimit);
    uint32_t rows = ready & (Arrival - 1);
    if (rows) asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(counter), "r"(rows) : "memory");
    asm volatile("atom.relaxed.cta.shared.exch.b32 _, [%0], 0;" ::"r"(sharedCounter) : "memory");
    atomicStore<uint32_t, cuda::thread_scope_block>(&storage.publishedPhase, phase, memoryOrderRelease);
  } else {
    // Wait for the leader before reusing the shared counter for the next chunk.
    POLL_MAYBE_JAILBREAK(
        (atomicLoad<uint32_t, cuda::thread_scope_block>(&storage.publishedPhase, memoryOrderAcquire) != phase),
        SpinLimit);
  }
#endif
}

template <int DispatchBytes, int DispatchStages, class P, class Storage, class NextLoad>
__device__ __forceinline__ void storeDispatchChunk(const P& p, Storage& storage, int localWarp, int row, int chunk,
                                                   uint8_t* destination, int bytes, uint32_t& phase, bool issueNext,
                                                   NextLoad&& loadNext) {
#if MSCCLPP_BULK_AVAILABLE
  int lane = threadIdx.x % warpSize;
  int stage = chunk % DispatchStages;
  if (lane == 0) {
    storage.barriers[localWarp][stage].wait(phase, SpinLimit);
    bulkStore(destination + chunk * DispatchBytes, storage.tiles[localWarp][stage], bytes);
    bulkStoreCommit();
    bulkFence();
  }
  __syncwarp();
  if constexpr (P::Activation == ActivationDtype::Mxfp8) {
    int k = 4 * lane;
    if (k < bytes / 32) {
      size_t offset =
          p.fc1.layout_SFB(cute::make_coord(w4StorageRow<typename P::Tiles>(row), chunk * DispatchBytes + k * 32, 0));
      *reinterpret_cast<uint32_t*>(p.workspace.inputScale + offset) =
          *reinterpret_cast<const uint32_t*>(storage.scales[localWarp][stage] + k);
    }
  }
  asm volatile("fence.proxy.async.global;" ::: "memory");
  __syncwarp();
  if (lane == 0) {
    if (issueNext) {
      bulkStoreWaitSource();
      bulkFence();
      loadNext();
    }
    bulkStoreWait();
  }
#endif
}

// A warp's window of up to 32 consecutive local experts. Lane i describes expert firstExpert + i. A group is
// DispatchWarps consecutive rows of a tile, and only groups with live rows are counted.
struct ExpertWindow {
  int firstExpert;       // first local expert of the window
  int endGroup;          // first group after the window
  int expertFirstGroup;  // first group of this lane's expert
  int expertFirstTile;   // first tile of this lane's expert
};

// Load the window of experts starting at firstExpert, whose groups start at firstGroup. All lanes of the warp must call
// this.
template <int TileN, int DispatchWarps>
MSCCLPP_DEVICE_INLINE ExpertWindow loadExpertWindow(const Workspace& w, int nLocalExperts, int firstExpert,
                                                    int firstGroup, int lane) {
  constexpr int nDispatchGroupsPerTile = (TileN + DispatchWarps - 1) / DispatchWarps;
  const int expert = firstExpert + lane;
  const int nExpertRows = expert < nLocalExperts ? w.cursors[expert] : 0;
  const int nFullTiles = nExpertRows / TileN;
  const int nTailRows = nExpertRows - nFullTiles * TileN;
  const int nExpertGroups = nFullTiles * nDispatchGroupsPerTile + (nTailRows + DispatchWarps - 1) / DispatchWarps;
  int scan = nExpertGroups;
#pragma unroll
  for (int distance = 1; distance < warpSize; distance *= 2) {
    int other = __shfl_up_sync(0xffffffff, scan, distance);
    if (lane >= distance) scan += other;
  }
  ExpertWindow window;
  window.firstExpert = firstExpert;
  window.endGroup = firstGroup + __shfl_sync(0xffffffff, scan, warpSize - 1);
  window.expertFirstGroup = firstGroup + scan - nExpertGroups;
  window.expertFirstTile = expert < nLocalExperts ? w.starts[expert] / TileN : 0;
  return window;
}

// Map a live group to its tile and its group within the tile. Groups must arrive in increasing order, so the window
// only moves forward. Returns false past the last live group. All lanes of the warp must call this.
template <int TileN, int DispatchWarps>
MSCCLPP_DEVICE_INLINE bool mapGroupToTile(const Workspace& w, int nLocalExperts, ExpertWindow& window,
                                          int dispatchGroupIdx, int lane, int& tile, int& groupIdxInTile) {
  constexpr int nDispatchGroupsPerTile = (TileN + DispatchWarps - 1) / DispatchWarps;
  while (dispatchGroupIdx >= window.endGroup) {
    const int nextExpert = window.firstExpert + warpSize;
    if (nextExpert >= nLocalExperts) return false;
    window = loadExpertWindow<TileN, DispatchWarps>(w, nLocalExperts, nextExpert, window.endGroup, lane);
  }
  // The group belongs to the last expert whose first group does not follow it, which skips experts without rows.
  const unsigned int startedExperts = __ballot_sync(
      0xffffffff, window.firstExpert + lane < nLocalExperts && dispatchGroupIdx >= window.expertFirstGroup);
  const int expertLane = 31 - __clz(startedExperts);
  const int expertGroup = dispatchGroupIdx - __shfl_sync(0xffffffff, window.expertFirstGroup, expertLane);
  tile = __shfl_sync(0xffffffff, window.expertFirstTile, expertLane) + expertGroup / nDispatchGroupsPerTile;
  groupIdxInTile = expertGroup % nDispatchGroupsPerTile;
  return true;
}

template <int Hidden, int DispatchBytes, int DispatchStages, class P, class Storage>
__device__ __forceinline__ void dispatchTokens(const P& p, Storage& s, int localWarp) {
#if MSCCLPP_BULK_AVAILABLE
  using Types = typename P::Collective;
  static_assert(Hidden >= 0);
  static_assert(DispatchBytes == sizeof(s.dispatch.tiles[0][0]));
  static_assert(DispatchStages == std::extent_v<decltype(s.dispatch.barriers), 1>);
  static_assert((DispatchBytes / 32 + 15) / 16 * 16 <= sizeof(s.dispatch.scales[0][0]));
  const auto& w = p.workspace;
  int lane = threadIdx.x % warpSize;
  auto& barriers = s.dispatch.barriers[localWarp];
  if (lane == 0 && localWarp == 0) {
    s.dispatch.arrivals = 0;
    s.dispatch.publishedPhase = 0;
  }
  // Each warp owns its stage barriers, so separate lanes initialize them concurrently.
  if (lane < DispatchStages) {
    barriers[lane].relaxedInit();
    bulkFence();
  }
  // All dispatch warps must observe the leader's cleared counters and initialized barriers before publishing.
  cutlass::arch::NamedBarrier::sync(Types::DispatchWarps * warpSize, 1);
  uint32_t loadPhases[DispatchStages] = {};
  constexpr int ElemBytes = P::Activation == ActivationDtype::Bf16 ? sizeof(__bfloat16) : sizeof(uint8_t);
  const int hidden = Hidden > 0 ? Hidden : p.config.hidden;
  const int rowBytes = hidden * ElemBytes;
  const int chunks = (rowBytes + DispatchBytes - 1) / DispatchBytes;
  constexpr int TileN = P::Tiles::N;
  constexpr int nDispatchGroupsPerTile = (TileN + Types::DispatchWarps - 1) / Types::DispatchWarps;
  static_assert((nDispatchGroupsPerTile & (nDispatchGroupsPerTile - 1)) == 0,
                "XOR group swizzling needs a power-of-two group count");
  // This parity hands off the shared counter, independently of each TMA stage's parity.
  uint32_t publicationPhase = 0;
  const int nLocalExperts = p.config.numExperts / p.config.worldSize;
  ExpertWindow window = loadExpertWindow<TileN, Types::DispatchWarps>(w, nLocalExperts, 0, 0, lane);
  for (int dispatchGroupIdx = blockIdx.x;; dispatchGroupIdx += gridDim.x) {
    int tile;
    int groupIdxInTile;
    if (!mapGroupToTile<TileN, Types::DispatchWarps>(w, nLocalExperts, window, dispatchGroupIdx, lane, tile,
                                                     groupIdxInTile))
      break;
    int rows = w.blocks[tile].rows;
    // Only full tiles are swizzled, so a tail tile's live groups stay first.
    if (rows == TileN) groupIdxInTile ^= tile & (nDispatchGroupsPerTile - 1);
    int firstRowIndexInGroup = groupIdxInTile * Types::DispatchWarps;
    // A group covers one row per dispatch warp. Skip groups fully past the live rows; padded tail warps in a
    // partially live group still arrive with hasRow=false so the shared publication counter remains convergent.
    if (firstRowIndexInGroup >= rows) continue;
    int rowInTile = firstRowIndexInGroup + localWarp;
    int row = tile * TileN + rowInTile;
    bool hasRow = rowInTile < rows;
    Route route{};
    if (hasRow) route = w.routes[row];
    hasRow = hasRow && route.rank >= 0;
    // Local input staging/quantization precedes this kernel; remote inputs are ordered by
    // the source's current-epoch route packets consumed in prepareRoutes.
    constexpr bool IsBf16 = P::Activation == ActivationDtype::Bf16;
    auto* source = hasRow ? peerAt<uint8_t>(p, route.rank, IsBf16 ? p.symmetric.input : p.symmetric.quantizedInput) +
                                size_t(route.token) * rowBytes
                          : nullptr;
    auto* sourceScale = (!IsBf16 && hasRow) ? peerAt<uint8_t>(p, route.rank, p.symmetric.quantizedInputScale) +
                                                  size_t(route.token) * w4SourceScaleStride(hidden)
                                            : nullptr;
    auto* destination = hasRow ? w.quantizedInput + size_t(w4StorageRow<typename P::Tiles>(row)) * rowBytes : nullptr;
    auto load = [&](int chunk) {
      int stage = chunk % DispatchStages;
      int bytes = min(DispatchBytes, rowBytes - chunk * DispatchBytes);
      if constexpr (P::Activation != ActivationDtype::Mxfp8) {
        barriers[stage].arriveAndExpect(bytes);
        bulkLoad(s.dispatch.tiles[localWarp][stage], source + chunk * DispatchBytes, bytes, barriers[stage]);
      } else {
        int scaleBytes = (bytes / 32 + 15) / 16 * 16;
        barriers[stage].arriveAndExpect(bytes + scaleBytes);
        bulkLoad(s.dispatch.tiles[localWarp][stage], source + chunk * DispatchBytes, bytes, barriers[stage]);
        bulkLoad(s.dispatch.scales[localWarp][stage], sourceScale + chunk * (DispatchBytes / 32), scaleBytes,
                 barriers[stage]);
      }
    };
    float routeWeight = 1.0f;
    if (lane == 0 && hasRow) {
      bulkFence();
#pragma unroll
      for (int stage = 0; stage < DispatchStages; ++stage)
        if (stage < chunks) load(stage);
      if (!p.config.combineRouteWeights) routeWeight = loadRouteWeight(p, route);
    }
    auto process = [&](int chunk) {
      int stage = chunk % DispatchStages;
      int bytes = min(DispatchBytes, rowBytes - chunk * DispatchBytes);
      if (hasRow)
        storeDispatchChunk<DispatchBytes, DispatchStages>(p, s.dispatch, localWarp, row, chunk, destination, bytes,
                                                          loadPhases[stage], chunk + DispatchStages < chunks,
                                                          [&] { load(chunk + DispatchStages); });
      if (lane == 0) {
        auto* counter = w4InputChunkCounter<Types>(w, hidden, tile, chunk);
        if (hasRow && chunk == chunks - 1 && !p.config.combineRouteWeights) w.routes[row].weight = routeWeight;
        publicationPhase ^= 1u;
        publishDispatchChunk<Types>(s.dispatch, localWarp, counter, hasRow, publicationPhase);
      }
      __syncwarp();
    };
    for (int chunk = 0; chunk < chunks; ++chunk) process(chunk);
  }
  if (lane == 0) {
#pragma unroll
    for (int stage = 0; stage < DispatchStages; ++stage) barriers[stage].invalidate();
  }
#endif
}

template <int Hidden, int DispatchBytes, int DispatchStages, class P, class Storage>
__device__ __forceinline__ void runDispatchRole(const P& p, Storage& s, int warp) {
  using Schedule = typename P::Collective::WarpSchedule;
  if (warp >= Schedule::DispatchBegin && warp < Schedule::DispatchEnd) {
    traceKernel(KernelTracePhase::Dispatch, true);
    dispatchTokens<Hidden, DispatchBytes, DispatchStages>(p, s, warp - Schedule::DispatchBegin);
    traceKernel(KernelTracePhase::Dispatch, false);
  }
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_DISPATCH_CUH_
