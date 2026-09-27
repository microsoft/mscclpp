// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_W4A8_ROLES_CUH_
#define MSCCLPP_EXT_MEGAMOE_W4A8_ROLES_CUH_

#include "megamoe_roles.cuh"

namespace mscclpp::megamoe::detail {

__device__ __forceinline__ void dispatchW4A8Tokens(const W4A8Parameters& p, W4A8SharedStorage& s, int localWarp) {
#if MSCCLPP_BULK_AVAILABLE
  constexpr int ScaleStageBytes = W4ScaleChunkBytes / 2;
  static_assert((W4DispatchChunk / 32 + 15) / 16 * 16 <= ScaleStageBytes);
  const auto& w = p.workspace;
  int lane = threadIdx.x % 32;
  auto& barriers = s.dispatch.barriers[localWarp];
  uint32_t phases[2] = {0, 0};
  if (lane == 0) {
    barriers[0].relaxedInit();
    barriers[1].relaxedInit();
    bulkFence();
  }
  __syncwarp();
  const int hidden = p.config.hidden;
  const int chunks = w4InputChunks(hidden);
  for (int row = blockIdx.x * W4DispatchWarps + localWarp; row < w.control->tokenBlocks * W4TileN;
       row += gridDim.x * W4DispatchWarps) {
    if (row % W4TileN >= w.blocks[row / W4TileN].rows) continue;
    Route route = w.routes[row];
    if (route.rank < 0) continue;
    auto* source = peerAt<uint8_t>(p, route.rank, p.symmetric.quantizedInput) + size_t(route.token) * hidden;
    auto* sourceScale = peerAt<uint8_t>(p, route.rank, p.symmetric.quantizedInputScale) +
                        size_t(route.token) * w4SourceScaleStride(hidden);
    auto* destination = w.quantizedInput + size_t(w4StorageRow(row)) * hidden;
    auto load = [&](int chunk) {
      int stage = chunk % 2;
      int bytes = min(int(W4DispatchChunk), hidden - chunk * W4DispatchChunk);
      int scaleBytes = (bytes / 32 + 15) / 16 * 16;
      barriers[stage].arriveAndExpect(bytes + scaleBytes);
      bulkLoad(s.dispatch.tiles[localWarp][stage], source + chunk * W4DispatchChunk, bytes, barriers[stage]);
      bulkLoad(s.dispatch.scales[localWarp] + stage * ScaleStageBytes, sourceScale + chunk * (W4DispatchChunk / 32),
               scaleBytes, barriers[stage]);
    };
    if (lane == 0) {
      bulkFence();
      load(0);
      if (chunks > 1) load(1);
    }
    for (int chunk = 0; chunk < chunks; ++chunk) {
      int stage = chunk % 2;
      int bytes = min(int(W4DispatchChunk), hidden - chunk * W4DispatchChunk);
      if (lane == 0) {
        barriers[stage].wait(phases[stage], SpinLimit);
        bulkStore(destination + chunk * W4DispatchChunk, s.dispatch.tiles[localWarp][stage], bytes);
        bulkStoreCommit();
        bulkFence();
      }
      __syncwarp();
      for (int k = 4 * lane; k < bytes / 32; k += 128) {
        size_t offset = p.fc1.layout_SFB(cute::make_coord(w4StorageRow(row), chunk * W4DispatchChunk + k * 32, 0));
        *reinterpret_cast<uint32_t*>(w.inputScale + offset) =
            *reinterpret_cast<const uint32_t*>(s.dispatch.scales[localWarp] + stage * ScaleStageBytes + k);
      }
      asm volatile("fence.proxy.async.global;" ::: "memory");
      __syncwarp();
      if (lane == 0) {
        bulkStoreWait();
        atomicFetchAdd<int, scopeDevice>(w4InputChunkCounter(w, hidden, row / W4TileN, chunk), 1, memoryOrderRelease);
        if (chunk + 2 < chunks) {
          bulkFence();
          load(chunk + 2);
        }
      }
      __syncwarp();
    }
  }
  if (lane == 0) {
    barriers[0].invalidate();
    barriers[1].invalidate();
  }
#endif
}

template <class TmemStorage>
__device__ __forceinline__ void runW4A8MainloopRole(
    const W4A8Parameters& p, W4A8SharedStorage& s, W4A8CollectiveTypes::Load& loadPipeline,
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
              return task.fc1 ? firstMainloop.template load<1>(producerPipeline, state, load1, coord, iterator, count)
                              : secondMainloop.template load<1>(producerPipeline, state, load2, coord, iterator, count);
            return task.fc1 ? firstMainloop.template load<2>(producerPipeline, state, load1, coord, iterator, count)
                            : secondMainloop.template load<2>(producerPipeline, state, load2, coord, iterator, count);
          } else {
            return task.fc1 ? firstMainloop.load(loadPipeline, state, load1, coord, iterator, count)
                            : secondMainloop.load(loadPipeline, state, load2, coord, iterator, count);
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
      loadState = fc1.mma(make_tuple(loadPipeline, activationPipeline, accumulatePipeline),
                          make_tuple(loadState, accumulateState), accumulator, mmaInputs, coord, kTiles);
      traceW4(phase, false, i);
      accumulatePipeline.producer_commit(accumulateState);
      ++accumulateState;
    }
    accumulatePipeline.producer_tail(accumulateState);
  }
}

}  // namespace mscclpp::megamoe::detail

#endif  // MSCCLPP_EXT_MEGAMOE_W4A8_ROLES_CUH_
