// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_W4A8_ROLES_CUH_
#define MSCCLPP_EXT_MEGAMOE_W4A8_ROLES_CUH_

#include "megamoe_roles.cuh"

namespace mscclpp::megamoe::detail {

template <int Hidden, class P>
__device__ __forceinline__ void dispatchW4A8Tokens(const P& p, W4A8SharedStorage& s, int localWarp) {
#if MSCCLPP_BULK_AVAILABLE
  static_assert(Hidden > 0);
  static_assert((W4DispatchChunk / 32 + 15) / 16 * 16 <= W4ScaleStageBytes);
  const auto& w = p.workspace;
  int lane = threadIdx.x % 32;
  auto& barriers = s.dispatch.barriers[localWarp];
  uint32_t phases[W4DispatchStages] = {};
  if (lane == 0) {
    CUTE_UNROLL
    for (int stage = 0; stage < W4DispatchStages; ++stage) barriers[stage].relaxedInit();
    bulkFence();
  }
  __syncwarp();
  constexpr int hidden = Hidden;
  constexpr int chunks = (Hidden + W4DispatchChunk - 1) / W4DispatchChunk;
  for (int row = blockIdx.x * W4DispatchWarps + localWarp; row < w.control->tokenBlocks * W4TileN;
       row += gridDim.x * W4DispatchWarps) {
    if (row % W4TileN >= w.blocks[row / W4TileN].rows) continue;
    Route route = w.routes[row];
    if (route.rank < 0) continue;
    // prepareRoutes consumed this peer's current-epoch route packet, which is
    // published by a kernel launched after the peer's same-stream quantization.
    auto* source = peerAt<uint8_t>(p, route.rank, p.symmetric.quantizedInput) + size_t(route.token) * hidden;
    auto* sourceScale = peerAt<uint8_t>(p, route.rank, p.symmetric.quantizedInputScale) +
                        size_t(route.token) * w4SourceScaleStride(hidden);
    auto* destination = w.quantizedInput + size_t(w4StorageRow(row)) * hidden;
    auto load = [&](int chunk) {
      int stage = chunk % W4DispatchStages;
      int bytes = min(int(W4DispatchChunk), hidden - chunk * W4DispatchChunk);
      int scaleBytes = (bytes / 32 + 15) / 16 * 16;
      barriers[stage].arriveAndExpect(bytes + scaleBytes);
      bulkLoad(s.dispatch.tiles[localWarp][stage], source + chunk * W4DispatchChunk, bytes, barriers[stage]);
      bulkLoad(s.dispatch.scales[localWarp][stage], sourceScale + chunk * (W4DispatchChunk / 32), scaleBytes,
               barriers[stage]);
    };
    if (lane == 0) {
      bulkFence();
      CUTE_UNROLL
      for (int stage = 0; stage < W4DispatchStages; ++stage)
        if (stage < chunks) load(stage);
    }
    auto process = [&](int chunk) {
      int stage = chunk % W4DispatchStages;
      int bytes = min(int(W4DispatchChunk), hidden - chunk * W4DispatchChunk);
      int scaleValues = bytes / 32;
      if (lane == 0) {
        barriers[stage].wait(phases[stage], SpinLimit);
        bulkStore(destination + chunk * W4DispatchChunk, s.dispatch.tiles[localWarp][stage], bytes);
        bulkStoreCommit();
        bulkFence();
      }
      __syncwarp();
      int k = 4 * lane;
      if (k < scaleValues) {
        size_t offset = p.fc1.layout_SFB(cute::make_coord(w4StorageRow(row), chunk * W4DispatchChunk + k * 32, 0));
        *reinterpret_cast<uint32_t*>(w.inputScale + offset) =
            *reinterpret_cast<const uint32_t*>(s.dispatch.scales[localWarp][stage] + k);
      }
      asm volatile("fence.proxy.async.global;" ::: "memory");
      __syncwarp();
      if (lane == 0) {
        bulkStoreWait();
        atomicFetchAdd<int, scopeDevice>(w4InputChunkCounter(w, hidden, row / W4TileN, chunk), 1, memoryOrderRelease);
        if (chunk + W4DispatchStages < chunks) {
          bulkFence();
          load(chunk + W4DispatchStages);
        }
      }
      __syncwarp();
    };
    CUTE_UNROLL
    for (int chunk = 0; chunk < chunks; ++chunk) process(chunk);
  }
  if (lane == 0) {
    CUTE_UNROLL
    for (int stage = 0; stage < W4DispatchStages; ++stage) barriers[stage].invalidate();
  }
#endif
}

template <class P, class TmemStorage>
__device__ __forceinline__ void runW4A8MainloopRole(
    const P& p, W4A8SharedStorage& s, W4A8CollectiveTypes::Load& loadPipeline,
    W4A8CollectiveTypes::Load& activationPipeline, W4A8CollectiveTypes::Accumulate& accumulatePipeline,
    W4A8CollectiveTypes::Mainloop& fc1, W4A8CollectiveTypes::Mainloop& fc2, const ProblemShape& shape1,
    const ProblemShape& shape2, TmemStorage tmemStorage, int hidden, int intermediate, int warp, int lane, int cta,
    int cluster, int tasks) {
  using namespace cute;
  using Mainloop = W4A8CollectiveTypes::Mainloop;
  using Load = W4A8CollectiveTypes::Load;
  using Accumulate = W4A8CollectiveTypes::Accumulate;
  using Schedule = W4A8WarpSchedule;
  cutlass::arch::warpgroup_reg_dealloc<W4TransferRegisters>();
  if (warp == Schedule::LoadWarp || (W4LoadWarps == 2 && warp == Schedule::ActivationLoadWarp)) {
    auto& producerPipeline =
        W4SplitPipelines && warp == Schedule::ActivationLoadWarp ? activationPipeline : loadPipeline;
    auto state = cutlass::make_producer_start_state<Load>();
    auto load1 = fc1.load_init(shape1, s.tensors);
    auto load2 = fc2.load_init(shape2, s.tensors);
    for (int i = cluster; i < tasks; i += gridDim.x / ClusterM) {
      Task task = taskAt(p, i, 0, hidden, intermediate);
      if (!task.fc1 && (W4LoadWarps == 1 || warp == Schedule::ActivationLoadWarp)) {
        traceW4(W4TracePhase::TokenReady, true, int(task.fc1));
        if (lane == 0)
          waitAtLeast<int, scopeDevice>(p.workspace.hiddenReady + task.block, fc1CompletionCount<false>(intermediate));
        __syncwarp();
        traceW4(W4TracePhase::TokenReady, false, int(task.fc1));
      }
      auto coord = taskCoord(p, task, cta);
      int kTiles = taskKTiles(p, task, hidden, intermediate);
      auto phase = task.fc1 ? W4TracePhase::LoadFc1 : W4TracePhase::LoadFc2;
      traceW4(phase, true, i);
      auto loadTiles = [&](int first, int count) {
        auto iterator = cute::make_coord_iterator(first, kTiles);
        auto result = [&](auto& firstMainloop, auto& secondMainloop) {
          if constexpr (W4LoadWarps == 2) {
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
      if (task.fc1 && (W4LoadWarps == 1 || warp == Schedule::ActivationLoadWarp)) {
        constexpr int TilesPerChunk = W4DispatchChunk / W4TileK;
        for (int first = 0; first < kTiles; first += TilesPerChunk) {
          traceW4(W4TracePhase::TokenReady, true, first / TilesPerChunk);
          if (lane == 0)
            waitAtLeast<int, scopeDevice>(w4InputChunkCounter(p.workspace, hidden, task.block, first / TilesPerChunk),
                                          task.tokens.rows);
          __syncwarp();
          traceW4(W4TracePhase::TokenReady, false, first / TilesPerChunk);
          loadTiles(first, min(TilesPerChunk, kTiles - first));
        }
      } else {
        loadTiles(0, kTiles);
      }
      traceW4(phase, false, i);
    }
    fc1.load_tail(producerPipeline, state);
  } else if (warp == Schedule::MmaWarp && cta == 0) {
    typename Load::PipelineState loadState;
    auto accumulateState = cutlass::make_producer_start_state<Accumulate>();
    auto mmaInputs = fc1.mma_init(tmemStorage, s.tensors);
    for (int i = cluster; i < tasks; i += gridDim.x / ClusterM) {
      Task task = taskAt(p, i, 0, hidden, intermediate);
      auto coord = taskCoord(p, task, cta);
      int kTiles = taskKTiles(p, task, hidden, intermediate);
      auto accumulator = Mainloop::slice_accumulator(tmemStorage, accumulateState.index());
      auto phase = task.fc1 ? W4TracePhase::MmaFc1 : W4TracePhase::MmaFc2;
      traceW4(phase, true, i);
      loadState =
          fc1.mma(make_tuple(loadPipeline, activationPipeline, accumulatePipeline),
                  make_tuple(loadState, accumulateState), accumulator, mmaInputs, coord, kTiles, task.tokens.rows);
      traceW4(phase, false, i);
      accumulatePipeline.producer_commit(accumulateState);
      ++accumulateState;
    }
    accumulatePipeline.producer_tail(accumulateState);
  }
}

}  // namespace mscclpp::megamoe::detail

#endif  // MSCCLPP_EXT_MEGAMOE_W4A8_ROLES_CUH_
