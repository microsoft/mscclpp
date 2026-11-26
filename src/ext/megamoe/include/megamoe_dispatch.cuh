// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_DISPATCH_CUH_
#define MSCCLPP_EXT_MEGAMOE_DISPATCH_CUH_

#include "megamoe_routing.cuh"
#include "megamoe_trace.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

template <class KernelParams>
MSCCLPP_DEVICE_INLINE float loadRouteWeight(const KernelParams& params, const Route& route) {
  const size_t index = size_t(route.token) * params.config.topK + route.slot;
  uint32_t bits;
  if (route.rank == params.config.rank) {
    auto* address = at<float>(params.local, params.symmetric.topkWeights) + index;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(bits) : "l"(address) : "memory");
  } else {
    // Each remote score has its own epoch, so it does not rely on the ID packet to publish side data.
    auto* packets = peerAt<mscclpp::LL8Packet>(params, route.rank, params.symmetric.routingPackets);
    const size_t packetIndex = size_t(route.rank) * params.config.maxTokens * params.config.topK + index;
    bits = packets[packetIndex].read(uint32_t(params.workspace.control->epoch), SpinLimit);
  }
  return mscclpp::bit_cast<float>(bits);
}

template <class KernelParams>
MSCCLPP_DEVICE_INLINE float loadLocalRouteWeight(const KernelParams& params, const Route& route,
                                                 const float* directScores, int nRuntimeTokens) {
  const float* scores = directScores ? directScores : at<float>(params.local, params.symmetric.topkWeights);
  const size_t index = (size_t(route.rank) * nRuntimeTokens + route.token) * params.config.topK + route.slot;
  return scores[index];
}

template <bool LocalGather, class KernelParams>
MSCCLPP_DEVICE_INLINE float resolveRouteWeight(const KernelParams& params, const Route& route,
                                               const float* directScores, int nRuntimeTokens) {
  if constexpr (LocalGather) {
    return loadLocalRouteWeight(params, route, directScores, nRuntimeTokens);
  } else {
    return loadRouteWeight(params, route);
  }
}

__host__ __device__ constexpr bool requiresLoadScale(ActivationDtype dtype) {
  // Only MXFP8 carries per-block activation scales; Bf16 is unquantized and Fp8Static uses a tensor-wide scalar scale.
  return dtype == ActivationDtype::Mxfp8;
}

template <class KernelParams>
concept RequiresLoadScale = requiresLoadScale(KernelParams::Activation);

template <bool IsBf16, class KernelParams>
MSCCLPP_DEVICE_INLINE void resolvePeerRouteSource(const KernelParams& params, const Route& route, int hidden,
                                                  int rowBytes, const uint8_t*& source, const uint8_t*& sourceScale) {
  source = peerAt<uint8_t>(params, route.rank, IsBf16 ? params.symmetric.input : params.symmetric.quantizedInput) +
           size_t(route.token) * rowBytes;
  if constexpr (RequiresLoadScale<KernelParams>) {
    sourceScale = peerAt<uint8_t>(params, route.rank, params.symmetric.quantizedInputScale) +
                  size_t(route.token) * w4SourceScaleStride(hidden);
  }
}

template <bool IsBf16, class KernelParams>
MSCCLPP_DEVICE_INLINE void resolveLocalGatherSource(const KernelParams& params, const Route& route, int hidden,
                                                    int rowBytes, const uint8_t* directInput,
                                                    const uint8_t* directInputScale, int nRuntimeTokens,
                                                    const uint8_t*& source, const uint8_t*& sourceScale) {
  const size_t sourceRow = size_t(route.rank) * nRuntimeTokens + route.token;
  const uint8_t* inputBase =
      directInput ? directInput
                  : at<uint8_t>(params.local, IsBf16 ? params.symmetric.input : params.symmetric.quantizedInput);
  source = inputBase + sourceRow * rowBytes;
  if constexpr (RequiresLoadScale<KernelParams>) {
    const int scaleStride = directInputScale ? (hidden / MxfpScaleBlockSize) : w4SourceScaleStride(hidden);
    const uint8_t* scaleBase =
        directInputScale ? directInputScale : at<uint8_t>(params.local, params.symmetric.quantizedInputScale);
    sourceScale = scaleBase + sourceRow * scaleStride;
  }
}

template <bool LocalGather, class KernelParams>
MSCCLPP_DEVICE_INLINE void resolveDispatchRouteSource(const KernelParams& params, const Route& route, int hidden,
                                                      int rowBytes, const uint8_t* directInput,
                                                      const uint8_t* directInputScale, int nRuntimeTokens,
                                                      const uint8_t*& source, const uint8_t*& sourceScale) {
  constexpr bool IsBf16 = KernelParams::Activation == ActivationDtype::Bf16;
  if constexpr (LocalGather) {
    resolveLocalGatherSource<IsBf16>(params, route, hidden, rowBytes, directInput, directInputScale, nRuntimeTokens,
                                     source, sourceScale);
  } else {
    resolvePeerRouteSource<IsBf16>(params, route, hidden, rowBytes, source, sourceScale);
  }
}

// A warp's window of up to WARP_SIZE consecutive local experts. Lane i describes expert firstExpert + i. A group is
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
MSCCLPP_DEVICE_INLINE ExpertWindow loadExpertWindow(const Workspace& workspace, int nLocalExperts, int firstExpert,
                                                    int firstGroup, int laneIdx) {
  constexpr int nDispatchGroupsPerTile = (TileN + DispatchWarps - 1) / DispatchWarps;
  const int expertIdx = firstExpert + laneIdx;
  const int nExpertRows = expertIdx < nLocalExperts ? workspace.cursors[expertIdx] : 0;
  const int nFullTiles = nExpertRows / TileN;
  const int nTailRows = nExpertRows - nFullTiles * TileN;
  const int nExpertGroups = nFullTiles * nDispatchGroupsPerTile + (nTailRows + DispatchWarps - 1) / DispatchWarps;
  int scan = nExpertGroups;
#pragma unroll
  for (int distance = 1; distance < WARP_SIZE; distance *= 2) {
    int other = __shfl_up_sync(0xffffffff, scan, distance);
    if (laneIdx >= distance) scan += other;
  }
  ExpertWindow window;
  window.firstExpert = firstExpert;
  window.endGroup = firstGroup + __shfl_sync(0xffffffff, scan, WARP_SIZE - 1);
  window.expertFirstGroup = firstGroup + scan - nExpertGroups;
  window.expertFirstTile = expertIdx < nLocalExperts ? workspace.starts[expertIdx] / TileN : 0;
  return window;
}

// Map a live group to its tile and its group within the tile. Groups must arrive in increasing order, so the window
// only moves forward. Returns false past the last live group. All lanes of the warp must call this.
template <int TileN, int DispatchWarps>
MSCCLPP_DEVICE_INLINE bool mapGroupToTile(const Workspace& workspace, int nLocalExperts, ExpertWindow& window,
                                          int dispatchGroupIdx, int laneIdx, int& tileIdx, int& groupIdxInTile) {
  constexpr int nDispatchGroupsPerTile = (TileN + DispatchWarps - 1) / DispatchWarps;
  while (dispatchGroupIdx >= window.endGroup) {
    const int nextExpert = window.firstExpert + WARP_SIZE;
    if (nextExpert >= nLocalExperts) return false;
    window = loadExpertWindow<TileN, DispatchWarps>(workspace, nLocalExperts, nextExpert, window.endGroup, laneIdx);
  }
  // The group belongs to the last expert whose first group does not follow it, which skips experts without rows.
  const unsigned int startedExperts = __ballot_sync(
      0xffffffff, window.firstExpert + laneIdx < nLocalExperts && dispatchGroupIdx >= window.expertFirstGroup);
  const int expertLane = (WARP_SIZE - 1) - __clz(startedExperts);
  const int expertGroup = dispatchGroupIdx - __shfl_sync(0xffffffff, window.expertFirstGroup, expertLane);
  tileIdx = __shfl_sync(0xffffffff, window.expertFirstTile, expertLane) + expertGroup / nDispatchGroupsPerTile;
  groupIdxInTile = expertGroup % nDispatchGroupsPerTile;
  return true;
}

template <int DispatchStages, class Schedule, class Storage>
MSCCLPP_DEVICE_INLINE void initDispatchStorage(Storage& sharedStorage, int warpIdx, int laneIdx) {
#if MSCCLPP_BULK_AVAILABLE
  if (warpIdx >= Schedule::DispatchBegin && warpIdx < Schedule::DispatchEnd) {
    const int localWarp = warpIdx - Schedule::DispatchBegin;
    if (laneIdx == 0 && localWarp == 0) {
      sharedStorage.dispatch.arrivals = 0;
      sharedStorage.dispatch.publishedPhase = 0;
    }
    constexpr int NumBarriers = std::extent_v<decltype(sharedStorage.dispatch.barriers), 1>;
    static_assert(NumBarriers >= DispatchStages && NumBarriers <= WARP_SIZE);
    if (laneIdx < NumBarriers) sharedStorage.dispatch.barriers[localWarp][laneIdx].relaxedInit();
  }
#endif
}

template <class KernelParams, class Storage>
  requires(!RequiresLoadScale<KernelParams>)
MSCCLPP_DEVICE_INLINE void issueDispatchChunkLoad(Storage& sharedStorage, int localWarp, int stage,
                                                  const uint8_t* source, const uint8_t*, int byteOffset, int bytes,
                                                  BulkBarrier& barrier) {
#if MSCCLPP_BULK_AVAILABLE
  barrier.arriveAndExpect(bytes);
  bulkLoad(sharedStorage.dispatch.tiles[localWarp][stage], source + byteOffset, bytes, barrier);
#endif
}

template <class KernelParams, class Storage>
  requires RequiresLoadScale<KernelParams>
MSCCLPP_DEVICE_INLINE void issueDispatchChunkLoad(Storage& sharedStorage, int localWarp, int stage,
                                                  const uint8_t* source, const uint8_t* sourceScale, int byteOffset,
                                                  int bytes, BulkBarrier& barrier) {
#if MSCCLPP_BULK_AVAILABLE
  const int scaleBytes = alignedMxfpScaleBytes(bytes);
  barrier.arriveAndExpect(bytes + scaleBytes);
  bulkLoad(sharedStorage.dispatch.tiles[localWarp][stage], source + byteOffset, bytes, barrier);
  bulkLoad(sharedStorage.dispatch.scales[localWarp][stage], sourceScale + (byteOffset / MxfpScaleBlockSize), scaleBytes,
           barrier);
#endif
}

template <bool LocalGather, int DispatchBytes, int NumStages, class KernelParams, class Storage>
MSCCLPP_DEVICE_INLINE void issueRouteChunkLoad(const KernelParams& params, Storage& sharedStorage, const Route& route,
                                               int localWarp, int chunkIdx, int hidden, int rowBytes,
                                               int nRuntimeTokens, const uint8_t* directInput,
                                               const uint8_t* directInputScale) {
  const uint8_t* source = nullptr;
  const uint8_t* sourceScale = nullptr;
  resolveDispatchRouteSource<LocalGather>(params, route, hidden, rowBytes, directInput, directInputScale,
                                          nRuntimeTokens, source, sourceScale);
  const int stage = NumStages == 2 ? (chunkIdx & 1) : 0;
  const int byteOffset = chunkIdx * DispatchBytes;
  const int bytes = min(DispatchBytes, rowBytes - byteOffset);
  issueDispatchChunkLoad<KernelParams>(sharedStorage, localWarp, stage, source, sourceScale, byteOffset, bytes,
                                       sharedStorage.dispatch.barriers[localWarp][stage]);
}

template <class KernelParams, class DispatchStorage>
  requires(!RequiresLoadScale<KernelParams>)
MSCCLPP_DEVICE_INLINE void storeDispatchChunkScales(const KernelParams&, const DispatchStorage&, int, int, int, int,
                                                    int, int) {}

template <class KernelParams, class DispatchStorage>
  requires RequiresLoadScale<KernelParams>
MSCCLPP_DEVICE_INLINE void storeDispatchChunkScales(const KernelParams& params, const DispatchStorage& dispatch,
                                                    int localWarp, int stage, int laneIdx, int storageRow,
                                                    int byteOffset, int bytes) {
  __syncwarp();
  const int k = 4 * laneIdx;
  if (k < bytes / MxfpScaleBlockSize) {
    const size_t offset = params.fc1.layout_SFB(cute::make_coord(storageRow, byteOffset + k * MxfpScaleBlockSize, 0));
    *reinterpret_cast<uint32_t*>(params.workspace.inputScale + offset) =
        *reinterpret_cast<const uint32_t*>(dispatch.scales[localWarp][stage] + k);
  }
  fenceAsyncProxyGlobal();
  __syncwarp();
}

// Prefetches up to NumStages initial chunks of `rowIdx` and records its route weight. Must be called convergently by
// all lanes of the warp when `rowInTile < nTileRows`; returns false if the route slot is unrouted (`rank < 0`).
template <bool LocalGather, int DispatchBytes, int NumStages, class KernelParams, class Storage>
MSCCLPP_DEVICE_INLINE bool prefetchDispatchRow(const KernelParams& params, Storage& sharedStorage, int localWarp,
                                               int laneIdx, int rowIdx, int nChunks, int hidden, int rowBytes,
                                               int nRuntimeTokens, const uint8_t* directInput,
                                               const uint8_t* directInputScale, const float* directScores) {
#if MSCCLPP_BULK_AVAILABLE
  const auto& workspace = params.workspace;
  bool hasRow = true;
  if (laneIdx == 0) {
    const Route route = workspace.routes[rowIdx];
    if (route.rank < 0) {
      hasRow = false;
    } else {
      bulkFence();
#pragma unroll
      for (int stage = 0; stage < NumStages; ++stage) {
        if (stage < nChunks) {
          issueRouteChunkLoad<LocalGather, DispatchBytes, NumStages>(params, sharedStorage, route, localWarp, stage,
                                                                     hidden, rowBytes, nRuntimeTokens, directInput,
                                                                     directInputScale);
        }
      }
      if (!params.config.combineRouteWeights) {
        workspace.routes[rowIdx].weight = resolveRouteWeight<LocalGather>(params, route, directScores, nRuntimeTokens);
      }
    }
  }
  return __shfl_sync(0xffffffff, int(hasRow), 0) != 0;
#else
  return false;
#endif
}

// Waits for `chunkIdx` to arrive in shared memory, stores tile data (and MXFP8 scales if required) to global workspace,
// and prefetches `chunkIdx + NumStages` before waiting for the global bulk store to finish.
template <bool LocalGather, int DispatchBytes, int NumStages, class KernelParams, class Storage>
MSCCLPP_DEVICE_INLINE void transferDispatchRowChunk(const KernelParams& params, Storage& sharedStorage, int localWarp,
                                                    int laneIdx, int rowIdx, int chunkIdx, int nChunks, int hidden,
                                                    int rowBytes, uint32_t& loadPhaseBits, int nRuntimeTokens,
                                                    const uint8_t* directInput, const uint8_t* directInputScale) {
#if MSCCLPP_BULK_AVAILABLE
  const auto& workspace = params.workspace;
  auto& dispatch = sharedStorage.dispatch;
  const int stage = NumStages == 2 ? (chunkIdx & 1) : 0;
  const int byteOffset = chunkIdx * DispatchBytes;
  const int bytes = min(DispatchBytes, rowBytes - byteOffset);
  const int storageRow = w4StorageRow<typename KernelParams::Tiles>(rowIdx);

  if (laneIdx == 0) {
    uint32_t phase = (loadPhaseBits >> stage) & 1u;
    dispatch.barriers[localWarp][stage].wait(phase, SpinLimit);
    loadPhaseBits ^= (1u << stage);
    bulkStore(workspace.quantizedInput + size_t(storageRow) * rowBytes + byteOffset, dispatch.tiles[localWarp][stage],
              bytes);
    bulkStoreCommit();
    bulkFence();
  }

  storeDispatchChunkScales(params, dispatch, localWarp, stage, laneIdx, storageRow, byteOffset, bytes);

  if (laneIdx == 0) {
    if (chunkIdx + NumStages < nChunks) {
      const Route route = workspace.routes[rowIdx];
      bulkStoreWaitSource();
      bulkFence();
      issueRouteChunkLoad<LocalGather, DispatchBytes, NumStages>(params, sharedStorage, route, localWarp,
                                                                 chunkIdx + NumStages, hidden, rowBytes, nRuntimeTokens,
                                                                 directInput, directInputScale);
    }
    bulkStoreWait();
  }
#endif
}

template <class Types, class DispatchStorage>
MSCCLPP_DEVICE_INLINE void publishDispatchChunk(DispatchStorage& dispatch, int localWarp, int* counter, bool hasRow,
                                                uint32_t phase) {
#if MSCCLPP_BULK_AVAILABLE
  // Every warp arrives, including padded rows; only live rows contribute to readiness.
  constexpr uint32_t Arrival = 1u << 16;
  const uint32_t contribution = Arrival + uint32_t(hasRow);
  addSharedRelease(&dispatch.arrivals, contribution);
  if (localWarp == 0) {
    uint32_t ready;
    POLL_MAYBE_JAILBREAK((ready = atomicLoad<uint32_t, cuda::thread_scope_block>(
                              &dispatch.arrivals, memoryOrderAcquire)) < Types::DispatchWarps * Arrival,
                         SpinLimit);
    const uint32_t nLiveRows = ready & (Arrival - 1);
    if (nLiveRows) signalReadyCounter(counter, nLiveRows);
    resetSharedRelaxed(&dispatch.arrivals);
    atomicStore<uint32_t, cuda::thread_scope_block>(&dispatch.publishedPhase, phase, memoryOrderRelease);
  } else {
    // Wait for the leader before reusing the shared counter for the next chunk.
    POLL_MAYBE_JAILBREAK(
        (atomicLoad<uint32_t, cuda::thread_scope_block>(&dispatch.publishedPhase, memoryOrderAcquire) != phase),
        SpinLimit);
  }
#endif
}

template <class Policy, class Types, class DispatchStorage>
concept DispatchPublishPolicy =
    requires(DispatchStorage& dispatch, int localWarp, int* counter, bool hasRow, uint32_t& publicationPhase) {
      { Policy::SkipInactiveRowEarly } -> std::convertible_to<bool>;
      Policy::template publishChunk<Types>(dispatch, localWarp, counter, hasRow, publicationPhase);
    };

// Coalesces chunk-readiness arrivals of all CTA dispatch warps in shared memory before one leader updates global state.
struct CoalescedDispatchPublishPolicy {
  static constexpr bool SkipInactiveRowEarly = false;

  template <class Types, class DispatchStorage>
  MSCCLPP_DEVICE_INLINE static void publishChunk(DispatchStorage& dispatch, int localWarp, int* counter, bool hasRow,
                                                 uint32_t& publicationPhase) {
    publicationPhase ^= 1u;
    publishDispatchChunk<Types>(dispatch, localWarp, counter, hasRow, publicationPhase);
  }
};

// Publishes chunk readiness directly from each active dispatch warp without intra-CTA shared-memory coalescing.
struct PerWarpDispatchPublishPolicy {
  static constexpr bool SkipInactiveRowEarly = true;

  template <class Types, class DispatchStorage>
  MSCCLPP_DEVICE_INLINE static void publishChunk(DispatchStorage&, int, int* counter, bool hasRow, uint32_t&) {
    if (hasRow) signalReadyCounter(counter);
  }
};

template <class KernelParams>
using SelectDispatchPublishPolicy =
    std::conditional_t<!KernelParams::WeightMxfp4 || KernelParams::Collective::DispatchWarps >= 4,
                       CoalescedDispatchPublishPolicy, PerWarpDispatchPublishPolicy>;

template <class PublishPolicy, bool LocalGather, int DispatchBytes, class KernelParams, class Storage>
  requires DispatchPublishPolicy<PublishPolicy, typename KernelParams::Collective,
                                 std::remove_reference_t<decltype(std::declval<Storage&>().dispatch)>>
MSCCLPP_DEVICE_INLINE void stageExpertTokensWithPolicy(const KernelParams& params, Storage& sharedStorage,
                                                       int localWarp, int laneIdx, int hidden, int rowBytes,
                                                       int nChunks, int nLocalExperts, ExpertWindow window,
                                                       int nRuntimeTokens, const uint8_t* directInput,
                                                       const uint8_t* directInputScale, const float* directScores) {
#if MSCCLPP_BULK_AVAILABLE
  using Types = typename KernelParams::Collective;
  using DispatchStorage = std::remove_reference_t<decltype(sharedStorage.dispatch)>;
  constexpr int TileN = KernelParams::Tiles::N;
  constexpr int nDispatchGroupsPerTile = (TileN + Types::DispatchWarps - 1) / Types::DispatchWarps;
  constexpr int NumStages = DispatchStorage::NumStages;
  static_assert(NumStages == 1 || NumStages == 2);
  static_assert(NumStages == std::extent_v<decltype(sharedStorage.dispatch.barriers), 1>);

  const auto& workspace = params.workspace;
  uint32_t publicationPhase = 0;
  uint32_t loadPhaseBits = 0;

  for (int dispatchGroupIdx = blockIdx.x;; dispatchGroupIdx += gridDim.x) {
    int tileIdx = 0;
    int groupIdxInTile = 0;
    if (!mapGroupToTile<TileN, Types::DispatchWarps>(workspace, nLocalExperts, window, dispatchGroupIdx, laneIdx,
                                                     tileIdx, groupIdxInTile)) {
      break;
    }
    const int nTileRows = workspace.tokenTiles[tileIdx].nTokens;
    // Only full tiles are swizzled, so a tail tile's live groups stay first.
    if (nTileRows == TileN) groupIdxInTile ^= tileIdx & (nDispatchGroupsPerTile - 1);
    const int firstRowInGroup = groupIdxInTile * Types::DispatchWarps;
    // A group covers one row per dispatch warp. Skip groups fully past the live rows; under coalesced publication,
    // padded tail warps in a partially live group still arrive with hasRow=false.
    if (firstRowInGroup >= nTileRows) continue;

    const int rowInTile = firstRowInGroup + localWarp;
    const int rowIdx = tileIdx * TileN + rowInTile;
    bool hasRow = rowInTile < nTileRows;
    if constexpr (PublishPolicy::SkipInactiveRowEarly) {
      if (!hasRow) continue;
    }
    if (hasRow) {
      hasRow = prefetchDispatchRow<LocalGather, DispatchBytes, NumStages>(
          params, sharedStorage, localWarp, laneIdx, rowIdx, nChunks, hidden, rowBytes, nRuntimeTokens, directInput,
          directInputScale, directScores);
      if constexpr (PublishPolicy::SkipInactiveRowEarly) {
        if (!hasRow) continue;
      }
    }

    for (int chunkIdx = 0; chunkIdx < nChunks; ++chunkIdx) {
      if (hasRow) {
        transferDispatchRowChunk<LocalGather, DispatchBytes, NumStages>(
            params, sharedStorage, localWarp, laneIdx, rowIdx, chunkIdx, nChunks, hidden, rowBytes, loadPhaseBits,
            nRuntimeTokens, directInput, directInputScale);
      }
      if (laneIdx == 0) {
        int* counter = w4InputChunkCounter<Types>(workspace, hidden, tileIdx, chunkIdx);
        PublishPolicy::template publishChunk<Types>(sharedStorage.dispatch, localWarp, counter, hasRow,
                                                    publicationPhase);
      }
      __syncwarp();
    }
  }

  if (laneIdx == 0) {
#pragma unroll
    for (int stage = 0; stage < NumStages; ++stage) {
      sharedStorage.dispatch.barriers[localWarp][stage].invalidate();
    }
  }
#endif
}

template <bool LocalGather, int Hidden, int DispatchBytes, int DispatchStages, class KernelParams, class Storage>
MSCCLPP_DEVICE_INLINE void stageExpertTokens(const KernelParams& params, Storage& sharedStorage, int localWarp,
                                             int nRuntimeTokens = 0, const uint8_t* directInput = nullptr,
                                             const uint8_t* directInputScale = nullptr,
                                             const float* directScores = nullptr) {
#if MSCCLPP_BULK_AVAILABLE
  using Types = typename KernelParams::Collective;
  using DispatchStorage = std::remove_reference_t<decltype(sharedStorage.dispatch)>;
  using PublishPolicy = SelectDispatchPublishPolicy<KernelParams>;
  static_assert(Hidden >= 0);
  static_assert(DispatchStages <= DispatchStorage::NumStages);
  static_assert(DispatchBytes == sizeof(sharedStorage.dispatch.tiles[0][0]));
  static_assert(alignedMxfpScaleBytes(DispatchBytes) <= sizeof(sharedStorage.dispatch.scales[0][0]));
  constexpr int ElemBytes = KernelParams::Activation == ActivationDtype::Bf16 ? sizeof(__bfloat16) : sizeof(uint8_t);
  constexpr int TileN = KernelParams::Tiles::N;
  constexpr int nDispatchGroupsPerTile = (TileN + Types::DispatchWarps - 1) / Types::DispatchWarps;
  static_assert((nDispatchGroupsPerTile & (nDispatchGroupsPerTile - 1)) == 0,
                "XOR group swizzling needs a power-of-two group count");
  const int laneIdx = threadIdx.x % WARP_SIZE;
  const int hidden = Hidden > 0 ? Hidden : params.config.hidden;
  const int rowBytes = hidden * ElemBytes;
  const int nChunks = (rowBytes + DispatchBytes - 1) / DispatchBytes;
  const int nLocalExperts = params.config.numExperts / params.config.worldSize;
  const ExpertWindow window =
      loadExpertWindow<TileN, Types::DispatchWarps>(params.workspace, nLocalExperts, 0, 0, laneIdx);
  stageExpertTokensWithPolicy<PublishPolicy, LocalGather, DispatchBytes>(
      params, sharedStorage, localWarp, laneIdx, hidden, rowBytes, nChunks, nLocalExperts, window, nRuntimeTokens,
      directInput, directInputScale, directScores);
#endif
}

template <int Hidden, int DispatchBytes, int DispatchStages, class KernelParams, class Storage>
MSCCLPP_DEVICE_INLINE void dispatchTokens(const KernelParams& params, Storage& sharedStorage, int localWarp) {
  stageExpertTokens<false, Hidden, DispatchBytes, DispatchStages>(params, sharedStorage, localWarp);
}

template <int Hidden, int DispatchBytes, int DispatchStages, class KernelParams, class Storage>
MSCCLPP_DEVICE_INLINE void gatherLocalTokens(const KernelParams& params, Storage& sharedStorage, int localWarp,
                                             int nRuntimeTokens, const uint8_t* directInput,
                                             const uint8_t* directInputScale, const float* directScores) {
  stageExpertTokens<true, Hidden, DispatchBytes, DispatchStages>(params, sharedStorage, localWarp, nRuntimeTokens,
                                                                 directInput, directInputScale, directScores);
}

template <int Hidden, int DispatchBytes, int DispatchStages, class KernelParams, class Storage>
MSCCLPP_DEVICE_INLINE void runDispatchRole(const KernelParams& params, Storage& sharedStorage, int warpIdx,
                                           int nRuntimeTokens = 0, const void* directInput = nullptr,
                                           const uint8_t* directInputScale = nullptr,
                                           const float* directScores = nullptr) {
  using Schedule = typename KernelParams::Collective::WarpSchedule;
  if (warpIdx >= Schedule::DispatchBegin && warpIdx < Schedule::DispatchEnd) {
    traceKernel(KernelTracePhase::Dispatch, true);
    if (params.config.computeOnly)
      gatherLocalTokens<Hidden, DispatchBytes, DispatchStages>(params, sharedStorage, warpIdx - Schedule::DispatchBegin,
                                                               nRuntimeTokens, static_cast<const uint8_t*>(directInput),
                                                               directInputScale, directScores);
    else
      dispatchTokens<Hidden, DispatchBytes, DispatchStages>(params, sharedStorage, warpIdx - Schedule::DispatchBegin);
    traceKernel(KernelTracePhase::Dispatch, false);
  }
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_DISPATCH_CUH_
