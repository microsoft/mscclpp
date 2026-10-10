// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_W4A8_ROLES_CUH_
#define MSCCLPP_EXT_MEGAMOE_W4A8_ROLES_CUH_

#include "megamoe_roles.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

template <bool Activation, class Types, class Storage>
__device__ __forceinline__ auto makeW4A8LoadPipeline(Storage& storage, int warp, int lane, int cta) {
  using namespace cute;
  using Load = typename Types::Load;
  using Mainloop = typename Types::Mainloop;
  using Schedule = typename Types::WarpSchedule;
  constexpr int InitializingWarp = Activation ? Schedule::ActivationLoadWarp : Schedule::LoadWarp;
  constexpr bool SharedProducers = !Activation && !Types::SplitPipelines && Types::LoadWarps == 2;
  bool producer = warp == InitializingWarp || (SharedProducers && warp == Schedule::ActivationLoadWarp);
  typename Load::Params params{};
  params.role =
      producer ? Load::ThreadCategory::Producer
               : (warp == Schedule::MmaWarp ? Load::ThreadCategory::Consumer : Load::ThreadCategory::NonParticipant);
  params.is_leader = lane == 0 && cta == 0 && producer;
  params.transaction_bytes =
      Activation ? Mainloop::ActivationTransactionBytes
                 : (Types::LoadWarps == 1 ? Mainloop::TmaTransactionBytes
                                          : (warp == Schedule::LoadWarp ? Mainloop::WeightTransactionBytes
                                                                        : Mainloop::ActivationTransactionBytes));
  if constexpr (Types::SplitPipelines && !Activation) params.transaction_bytes = Mainloop::WeightTransactionBytes;
  params.initializing_warp = InitializingWarp;
  Load pipeline(storage, params, ClusterShape{}, false_type{}, false_type{});
  if constexpr (SharedProducers) {
    if (warp == InitializingWarp) {
      // Both loaders arrive before the shared stage can become ready.
      cutlass::arch::detail::initialize_barrier_array_pair_aligned<decltype(storage.full_barrier_),
                                                                   decltype(storage.empty_barrier_), Load::Stages>(
          storage.full_barrier_, storage.empty_barrier_, 2, 1);
    }
  } else {
    Load::init_barriers(storage, params, ClusterShape{});
  }
  return pipeline;
}

template <class Types, class Storage>
__device__ __forceinline__ void publishW4A8Chunk(Storage& storage, int localWarp, int* counter, bool hasRow,
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

template <int DispatchBytes, int DispatchStages, class P, class Storage>
__device__ __forceinline__ void storeW4A8Chunk(const P& p, Storage& storage, int localWarp, int row, int chunk,
                                               uint8_t* destination, int bytes, uint32_t& phase) {
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
  int k = 4 * lane;
  if (k < bytes / 32) {
    size_t offset =
        p.fc1.layout_SFB(cute::make_coord(w4StorageRow<typename P::Tiles>(row), chunk * DispatchBytes + k * 32, 0));
    *reinterpret_cast<uint32_t*>(p.workspace.inputScale + offset) =
        *reinterpret_cast<const uint32_t*>(storage.scales[localWarp][stage] + k);
  }
  asm volatile("fence.proxy.async.global;" ::: "memory");
  __syncwarp();
  if (lane == 0) bulkStoreWait();
#endif
}

template <int Hidden, int DispatchBytes, int DispatchStages, class P, class Storage>
__device__ __forceinline__ void dispatchW4A8Tokens(const P& p, Storage& s, int localWarp) {
#if MSCCLPP_BULK_AVAILABLE
  using Types = typename P::Collective;
  static_assert(Hidden > 0);
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
  constexpr int hidden = Hidden;
  constexpr int chunks = (Hidden + DispatchBytes - 1) / DispatchBytes;
  constexpr int TileN = P::Tiles::N;
  constexpr int GroupsPerBlock = (TileN + Types::DispatchWarps - 1) / Types::DispatchWarps;
  // This parity hands off the shared counter, independently of each TMA stage's parity.
  uint32_t publicationPhase = 0;
  int groups = w.control->tokenBlocks * GroupsPerBlock;
  for (int group = blockIdx.x; group < groups; group += gridDim.x) {
    int block = group / GroupsPerBlock;
    int firstRowIndexInGroup = group % GroupsPerBlock * Types::DispatchWarps;
    int rows = w.blocks[block].rows;
    // A group covers one row per dispatch warp. Skip groups fully past the live rows; padded tail warps in a
    // partially live group still arrive with hasRow=false so the shared publication counter remains convergent.
    if (firstRowIndexInGroup >= rows) continue;
    int rowInBlock = firstRowIndexInGroup + localWarp;
    int row = block * TileN + rowInBlock;
    bool hasRow = rowInBlock < rows;
    Route route{};
    if (hasRow) route = w.routes[row];
    hasRow = hasRow && route.rank >= 0;
    // prepareRoutes consumed this peer's current-epoch route packet, which is
    // published by a kernel launched after the peer's same-stream quantization.
    auto* source =
        hasRow ? peerAt<uint8_t>(p, route.rank, p.symmetric.quantizedInput) + size_t(route.token) * hidden : nullptr;
    auto* sourceScale = hasRow ? peerAt<uint8_t>(p, route.rank, p.symmetric.quantizedInputScale) +
                                     size_t(route.token) * w4SourceScaleStride(hidden)
                               : nullptr;
    auto* destination = hasRow ? w.quantizedInput + size_t(w4StorageRow<typename P::Tiles>(row)) * hidden : nullptr;
    auto load = [&](int chunk) {
      int stage = chunk % DispatchStages;
      int bytes = min(DispatchBytes, hidden - chunk * DispatchBytes);
      int scaleBytes = (bytes / 32 + 15) / 16 * 16;
      barriers[stage].arriveAndExpect(bytes + scaleBytes);
      bulkLoad(s.dispatch.tiles[localWarp][stage], source + chunk * DispatchBytes, bytes, barriers[stage]);
      bulkLoad(s.dispatch.scales[localWarp][stage], sourceScale + chunk * (DispatchBytes / 32), scaleBytes,
               barriers[stage]);
    };
    float routeWeight = 0.0f;
    if (lane == 0 && hasRow) {
      bulkFence();
      CUTE_UNROLL
      for (int stage = 0; stage < DispatchStages; ++stage)
        if (stage < chunks) load(stage);
      routeWeight = loadRouteWeight(p, route);
    }
    auto process = [&](int chunk) {
      int stage = chunk % DispatchStages;
      int bytes = min(DispatchBytes, hidden - chunk * DispatchBytes);
      if (hasRow)
        storeW4A8Chunk<DispatchBytes, DispatchStages>(p, s.dispatch, localWarp, row, chunk, destination, bytes,
                                                      loadPhases[stage]);
      if (lane == 0) {
        auto* counter = w4InputChunkCounter<Types>(w, hidden, block, chunk);
        if (hasRow && chunk == chunks - 1) w.routes[row].weight = routeWeight;
        publicationPhase ^= 1u;
        publishW4A8Chunk<Types>(s.dispatch, localWarp, counter, hasRow, publicationPhase);
        if (hasRow && chunk + DispatchStages < chunks) {
          bulkFence();
          load(chunk + DispatchStages);
        }
      }
      __syncwarp();
    };
    for (int chunk = 0; chunk < chunks; ++chunk) process(chunk);
  }
  if (lane == 0) {
    CUTE_UNROLL
    for (int stage = 0; stage < DispatchStages; ++stage) barriers[stage].invalidate();
  }
#endif
}

template <class P, class Storage>
__device__ __forceinline__ void runW4A8LoadRole(const P& p, Storage& s, typename P::Collective::Load& loadPipeline,
                                                typename P::Collective::Load& activationPipeline,
                                                typename P::Collective::Mainloop& fc1,
                                                typename P::Collective::Mainloop& fc2, const ProblemShape& shape1,
                                                const ProblemShape& shape2, int hidden, int intermediate, int warp,
                                                int lane, int cta, int cluster, int tasks) {
  using namespace cute;
  using Types = typename P::Collective;
  using Load = typename P::Collective::Load;
  using Schedule = typename Types::WarpSchedule;
  auto& producerPipeline =
      Types::SplitPipelines && warp == Schedule::ActivationLoadWarp ? activationPipeline : loadPipeline;
  auto state = cutlass::make_producer_start_state<Load>();
  auto load1 = fc1.load_init(shape1, s.tensors);
  auto load2 = fc2.load_init(shape2, s.tensors);
  for (int i = cluster; i < tasks; i += gridDim.x / ClusterM) {
    Task task = taskAt(p, i, 0, hidden, intermediate);
    if (!task.fc1 && (Types::LoadWarps == 1 || warp == Schedule::ActivationLoadWarp)) {
      traceKernel(KernelTracePhase::TokenReady, true, int(task.fc1));
      if (lane == 0)
        waitAtLeast<int, scopeDevice>(p.workspace.hiddenReady + size_t(task.block) * W4ReadyCounterStride,
                                      fc1CompletionCount<false>(intermediate));
      __syncwarp();
      traceKernel(KernelTracePhase::TokenReady, false, int(task.fc1));
    }
    auto coord = taskCoord(p, task, cta);
    int kTiles = taskKTiles(p, task, hidden, intermediate);
    auto phase = task.fc1 ? KernelTracePhase::LoadFc1 : KernelTracePhase::LoadFc2;
    traceKernel(phase, true, i);
    auto loadTiles = [&](int first, int count) {
      auto iterator = cute::make_coord_iterator(first, kTiles);
      auto result = [&](auto& firstMainloop, auto& secondMainloop) {
        if constexpr (Types::LoadWarps == 2) {
          if (warp == Schedule::LoadWarp)
            return task.fc1 ? firstMainloop.template load<1>(producerPipeline, state, load1, coord, iterator, count,
                                                             task.tokens.rows)
                            : secondMainloop.template load<1>(producerPipeline, state, load2, coord, iterator, count,
                                                              task.tokens.rows);
          return task.fc1 ? firstMainloop.template load<2>(producerPipeline, state, load1, coord, iterator, count,
                                                           task.tokens.rows)
                          : secondMainloop.template load<2>(producerPipeline, state, load2, coord, iterator, count,
                                                            task.tokens.rows);
        } else {
          return task.fc1 ? firstMainloop.load(loadPipeline, state, load1, coord, iterator, count, task.tokens.rows)
                          : secondMainloop.load(loadPipeline, state, load2, coord, iterator, count, task.tokens.rows);
        }
      }(fc1, fc2);
      state = get<0>(result);
    };
    if (task.fc1 && (Types::LoadWarps == 1 || warp == Schedule::ActivationLoadWarp)) {
      constexpr int TilesPerChunk = Types::DispatchChunk / P::Tiles::K;
      for (int first = 0; first < kTiles; first += TilesPerChunk) {
        traceKernel(KernelTracePhase::TokenReady, true, first / TilesPerChunk);
        if (lane == 0)
          waitAtLeast<int, scopeDevice>(
              w4InputChunkCounter<Types>(p.workspace, hidden, task.block, first / TilesPerChunk), task.tokens.rows);
        __syncwarp();
        traceKernel(KernelTracePhase::TokenReady, false, first / TilesPerChunk);
        loadTiles(first, min(TilesPerChunk, kTiles - first));
      }
    } else {
      loadTiles(0, kTiles);
    }
    traceKernel(phase, false, i);
  }
  fc1.load_tail(producerPipeline, state);
}

template <class P, class Storage, class TmemStorage>
__device__ __forceinline__ void runW4A8MmaRole(const P& p, Storage& s, typename P::Collective::Load& loadPipeline,
                                               typename P::Collective::Load& activationPipeline,
                                               typename P::Collective::Accumulate& accumulatePipeline,
                                               typename P::Collective::Mainloop& fc1, TmemStorage tmemStorage,
                                               int hidden, int intermediate, int warp, int cta, int cluster,
                                               int tasks) {
  using namespace cute;
  using Mainloop = typename P::Collective::Mainloop;
  using Load = typename P::Collective::Load;
  using Accumulate = typename P::Collective::Accumulate;
  using Schedule = typename P::Collective::WarpSchedule;
  if (warp == Schedule::MmaWarp && cta == 0) {
    typename Load::PipelineState loadState;
    auto accumulateState = cutlass::make_producer_start_state<Accumulate>();
    auto mmaInputs = fc1.mma_init(tmemStorage, s.tensors);
    for (int i = cluster; i < tasks; i += gridDim.x / ClusterM) {
      Task task = taskAt(p, i, 0, hidden, intermediate);
      auto coord = taskCoord(p, task, cta);
      int kTiles = taskKTiles(p, task, hidden, intermediate);
      auto accumulator = Mainloop::slice_accumulator(tmemStorage, accumulateState.index());
      auto phase = task.fc1 ? KernelTracePhase::MmaFc1 : KernelTracePhase::MmaFc2;
      traceKernel(phase, true, i);
      loadState =
          fc1.mma(make_tuple(loadPipeline, activationPipeline, accumulatePipeline),
                  make_tuple(loadState, accumulateState), accumulator, mmaInputs, coord, kTiles, task.tokens.rows);
      traceKernel(phase, false, i);
      accumulatePipeline.producer_commit(accumulateState);
      ++accumulateState;
    }
    accumulatePipeline.producer_tail(accumulateState);
  }
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_W4A8_ROLES_CUH_
