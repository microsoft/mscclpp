// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_W4A8_ROLES_CUH_
#define MSCCLPP_EXT_MEGAMOE_W4A8_ROLES_CUH_

#include "megamoe_tasks.cuh"
#include "megamoe_trace.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

template <bool Activation, class Types, class Storage>
__device__ __forceinline__ auto makeW4A8LoadPipeline(Storage& storage, int warp, int lane, int cta) {
  using namespace cute;
  using Load = typename Types::Load;
  using Mainloop = typename Types::Mainloop;
  using Schedule = typename Types::WarpSchedule;
  using ClusterShape = typename Types::ClusterShape;
  constexpr int InitializingWarp = Activation ? Schedule::ActivationLoadWarp : Schedule::LoadWarp;
  constexpr bool SharedProducers = !Activation && !Types::SplitPipelines && Types::LoadWarps == 2;
  bool producer = warp == InitializingWarp || (SharedProducers && warp == Schedule::ActivationLoadWarp);
  typename Load::Params params{};
  params.role =
      producer ? Load::ThreadCategory::Producer
               : (warp == Schedule::MmaWarp ? Load::ThreadCategory::Consumer : Load::ThreadCategory::NonParticipant);
  params.is_leader = false;
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
  const int stride = gridDim.x / P::Tiles::ClusterSize;
  const bool checkReady = Types::LoadWarps == 1 || warp == Schedule::ActivationLoadWarp;
  int prefetchedBlock = -1;
  int prefetchedReady = -1;
  auto load1 = fc1.load_init(shape1, s.tensors);
  auto load2 = fc2.load_init(shape2, s.tensors);
  int c = cluster;
  for (int base = 0; base < tasks; base += stride) {
    int i = waveTaskOrdinal(base, c, cluster, stride, tasks);
    int nextC = stepClusterOffset(c, stride);
    int nextI = waveTaskOrdinal(base + stride, nextC, cluster, stride, tasks);
    c = nextC;
    if (i >= tasks) break;
    Task task = taskAt(p, i, hidden, intermediate);
    int curPrefBlock = prefetchedBlock;
    int curPrefReady = prefetchedReady;
    auto prefetchNextReady = [&] {
      if (checkReady && nextI < tasks && lane == 0) {
        Task nextTask = taskAt(p, nextI, hidden, intermediate);
        prefetchedBlock = nextTask.block;
        int* counter = (nextTask.fc1 ? p.workspace.inputReady : p.workspace.hiddenReady) +
                       size_t(nextTask.block) * W4ReadyCounterStride;
        asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(prefetchedReady) : "l"(counter));
      }
    };
    const bool cacheWeights = (Types::LoadWarps == 1 || warp == Schedule::LoadWarp) && bool(task.tokens.cacheWeights);
    if (!task.fc1 && checkReady) {
      traceKernel(KernelTracePhase::TokenReady, true, int(task.fc1));
      int needed = fc1CompletionCount<P>(intermediate);
      if (lane == 0 && (task.block != curPrefBlock || curPrefReady < needed))
        waitAtLeast<int, scopeDevice>(p.workspace.hiddenReady + size_t(task.block) * W4ReadyCounterStride, needed);
      __syncwarp();
      traceKernel(KernelTracePhase::TokenReady, false, int(task.fc1));
    }
    auto coord = taskCoord(p, task, cta);
    int kTiles = taskKTiles(p, task, hidden, intermediate);
    auto phase = task.fc1 ? KernelTracePhase::LoadFc1 : KernelTracePhase::LoadFc2;
    traceKernel(phase, true, i);
    auto loadTiles = [&](int first, int count) {
      auto iterator = cute::make_coord_iterator(first, kTiles);
      constexpr ActivationDtype ActDtype = P::Activation;
      auto result = [&](auto& firstMainloop, auto& secondMainloop) {
        if constexpr (Types::LoadWarps == 2) {
          if (warp == Schedule::LoadWarp)
            return task.fc1 ? firstMainloop.template load<1, ActDtype>(producerPipeline, state, load1, coord, iterator,
                                                                       count, task.tokens.rows, cacheWeights)
                            : secondMainloop.template load<1, ActDtype>(producerPipeline, state, load2, coord, iterator,
                                                                        count, task.tokens.rows, cacheWeights);
          return task.fc1 ? firstMainloop.template load<2, ActDtype>(producerPipeline, state, load1, coord, iterator,
                                                                     count, task.tokens.rows, cacheWeights)
                          : secondMainloop.template load<2, ActDtype>(producerPipeline, state, load2, coord, iterator,
                                                                      count, task.tokens.rows, cacheWeights);
        } else {
          return task.fc1 ? firstMainloop.template load<0, ActDtype>(loadPipeline, state, load1, coord, iterator, count,
                                                                     task.tokens.rows, cacheWeights)
                          : secondMainloop.template load<0, ActDtype>(loadPipeline, state, load2, coord, iterator,
                                                                      count, task.tokens.rows, cacheWeights);
        }
      }(fc1, fc2);
      state = get<0>(result);
    };
    if (task.fc1 && checkReady) {
      constexpr int TilesPerChunk = Types::DispatchChunk / P::Tiles::K;
      int needed = task.tokens.rows;
      for (int first = 0; first < kTiles; first += TilesPerChunk) {
        traceKernel(KernelTracePhase::TokenReady, true, first / TilesPerChunk);
        if (lane == 0 && (task.block != curPrefBlock || curPrefReady < needed))
          waitAtLeast<int, scopeDevice>(
              w4InputChunkCounter<Types>(p.workspace, hidden, task.block, first / TilesPerChunk), needed);
        __syncwarp();
        traceKernel(KernelTracePhase::TokenReady, false, first / TilesPerChunk);
        if (first + TilesPerChunk >= kTiles) prefetchNextReady();
        loadTiles(first, min(TilesPerChunk, kTiles - first));
      }
    } else {
      prefetchNextReady();
      loadTiles(0, kTiles);
    }
    traceKernel(phase, false, i);
  }
  fc1.load_tail(producerPipeline, state);
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_W4A8_ROLES_CUH_
