// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_TASKS_CUH_
#define MSCCLPP_EXT_MEGAMOE_TASKS_CUH_

#include "megamoe_device.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

template <class KernelParams>
MSCCLPP_DEVICE_INLINE int fc1TaskTiles(int intermediate) {
  return 2 * intermediate / KernelParams::Tiles::M;
}

template <class KernelParams>
MSCCLPP_DEVICE_INLINE int fc2TaskTiles(int hidden) {
  return (hidden + KernelParams::Tiles::M - 1) / KernelParams::Tiles::M;
}

template <class KernelParams>
MSCCLPP_DEVICE_INLINE int fc1CompletionCount(int intermediate) {
  return KernelParams::Tiles::ClusterSize * fc1TaskTiles<KernelParams>(intermediate);
}

template <class KernelParams>
MSCCLPP_DEVICE_INLINE int taskTokenTiles(const KernelParams& params, int nTokens) {
  if constexpr (KernelParams::LocalExpert) return (nTokens + KernelParams::Tiles::N - 1) / KernelParams::Tiles::N;
  return params.workspace.control->nTokenTiles;
}

template <class KernelParams>
MSCCLPP_DEVICE_INLINE TokenTile resolveTokenTile(const KernelParams& params, int& tileIdx, int& mTileIdx,
                                                 int nLayerTiles, int nTokens) {
  if constexpr (KernelParams::LocalExpert) {
    return TokenTile::make(0, min(int(KernelParams::Tiles::N), nTokens - tileIdx * KernelParams::Tiles::N));
  }
  TokenTile tokenTile = params.workspace.tokenTiles[tileIdx];
  int shift = tokenTile.groupShift();
  if (shift > 0) {
    int inGroup = int(tokenTile.tileIdxInGroup) * nLayerTiles + mTileIdx;
    tileIdx += (inGroup & ((1 << shift) - 1)) - int(tokenTile.tileIdxInGroup);
    mTileIdx = inGroup >> shift;
  }
  return tokenTile;
}

struct ScheduledTaskCoord {
  bool isFc1;
  int tileIdx;
  int mTileIdx;
};

template <class Policy, class KernelParams>
concept TaskSchedulePolicy =
    requires(const KernelParams& params, int taskOrdinal, int nTokenTiles, int nFc1Tiles, int nFc2Tiles) {
      requires std::is_same_v<decltype(Policy::mapOrdinal(params, taskOrdinal, nTokenTiles, nFc1Tiles, nFc2Tiles)),
                              ScheduledTaskCoord>;
    };

// Executes all FC1 tasks across every token tile before starting FC2 tasks.
struct SequentialTaskSchedulePolicy {
  template <class KernelParams>
  MSCCLPP_DEVICE_INLINE static ScheduledTaskCoord mapOrdinal(const KernelParams& params, int taskOrdinal,
                                                             int nTokenTiles, int nFc1Tiles, int /*nFc2Tiles*/) {
    const int nFc1Tasks = nTokenTiles * nFc1Tiles;
    const bool isFc1 = taskOrdinal < nFc1Tasks;
    const int layerOrdinal = isFc1 ? taskOrdinal : (taskOrdinal - nFc1Tasks);
    int tileIdx = 0;
    int mTileIdx = 0;
    (isFc1 ? params.fc1TaskDivisor : params.fc2TaskDivisor)(tileIdx, mTileIdx, layerOrdinal);
    return ScheduledTaskCoord{isFc1, tileIdx, mTileIdx};
  }
};

constexpr int DefaultInterleaveTileLag = 32;

// Overlaps outbound FC2 stores of tile `s` with FC1 compute of tile `s + TileLag` after a `TileLag`-tile FC1 prologue.
template <int TileLag = DefaultInterleaveTileLag>
struct LaggedInterleavedTaskSchedulePolicy {
  static_assert(TileLag > 0);

  template <class KernelParams>
  MSCCLPP_DEVICE_INLINE static ScheduledTaskCoord mapOrdinal(const KernelParams& params, int taskOrdinal,
                                                             int nTokenTiles, int nFc1Tiles, int nFc2Tiles) {
    const int prologueTasks = TileLag * nFc1Tiles;
    const int tasksPerStep = nFc1Tiles + nFc2Tiles;
    const int nSteadySteps = nTokenTiles - TileLag;
    const int steadyEndOrdinal = prologueTasks + nSteadySteps * tasksPerStep;

    if (taskOrdinal < prologueTasks) {
      int tileIdx = 0;
      int mTileIdx = 0;
      params.fc1TaskDivisor(tileIdx, mTileIdx, taskOrdinal);
      return ScheduledTaskCoord{true, tileIdx, mTileIdx};
    }
    if (taskOrdinal < steadyEndOrdinal) {
      const int steadyOffset = taskOrdinal - prologueTasks;
      const int stepIdx = steadyOffset / tasksPerStep;
      const int stepRem = steadyOffset - stepIdx * tasksPerStep;
      if (stepRem < nFc1Tiles) {
        return ScheduledTaskCoord{true, TileLag + stepIdx, stepRem};
      }
      return ScheduledTaskCoord{false, stepIdx, stepRem - nFc1Tiles};
    }
    const int tailOrdinal = taskOrdinal - steadyEndOrdinal;
    int tailTileOffset = 0;
    int mTileIdx = 0;
    params.fc2TaskDivisor(tailTileOffset, mTileIdx, tailOrdinal);
    return ScheduledTaskCoord{false, nSteadySteps + tailTileOffset, mTileIdx};
  }
};

template <class Policy, class KernelParams>
  requires TaskSchedulePolicy<Policy, KernelParams>
MSCCLPP_DEVICE_INLINE Task resolveScheduledTask(const KernelParams& params, int taskOrdinal, int nTokenTiles,
                                                int nFc1Tiles, int nFc2Tiles, int nTokens = 0) {
  ScheduledTaskCoord coord = Policy::mapOrdinal(params, taskOrdinal, nTokenTiles, nFc1Tiles, nFc2Tiles);
  const int nLayerTiles = coord.isFc1 ? nFc1Tiles : nFc2Tiles;
  TokenTile tokenTile = resolveTokenTile(params, coord.tileIdx, coord.mTileIdx, nLayerTiles, nTokens);
  return Task{coord.isFc1, coord.tileIdx, coord.mTileIdx, tokenTile};
}

template <class KernelParams>
constexpr bool SupportsInterleavedSchedule =
    !KernelParams::LocalExpert && KernelParams::WeightMxfp4 && (KernelParams::Collective::DispatchWarps >= 4);

template <class KernelParams>
MSCCLPP_DEVICE_INLINE bool shouldInterleaveTasks(int nTokenTiles, int nLocalExperts) {
  return SupportsInterleavedSchedule<KernelParams> &&
         (nTokenTiles > max(2 * nLocalExperts, 2 * DefaultInterleaveTileLag));
}

struct TaskScheduler {
  int nTokenTiles;
  int nFc1Tiles;
  int nFc2Tiles;
  bool isInterleaved;

  template <class KernelParams>
  MSCCLPP_DEVICE_INLINE Task taskAt(const KernelParams& params, int taskOrdinal, int nTokens = 0) const {
    if constexpr (SupportsInterleavedSchedule<KernelParams>) {
      if (isInterleaved) {
        return resolveScheduledTask<LaggedInterleavedTaskSchedulePolicy<>>(params, taskOrdinal, nTokenTiles, nFc1Tiles,
                                                                           nFc2Tiles, nTokens);
      }
    }
    return resolveScheduledTask<SequentialTaskSchedulePolicy>(params, taskOrdinal, nTokenTiles, nFc1Tiles, nFc2Tiles,
                                                              nTokens);
  }
};

template <class KernelParams>
MSCCLPP_DEVICE_INLINE TaskScheduler makeTaskScheduler(int nTokenTiles, int nLocalExperts, int hidden,
                                                      int intermediate) {
  return TaskScheduler{nTokenTiles, fc1TaskTiles<KernelParams>(intermediate), fc2TaskTiles<KernelParams>(hidden),
                       shouldInterleaveTasks<KernelParams>(nTokenTiles, nLocalExperts)};
}

template <class KernelParams>
__device__ Task taskAt(const KernelParams& params, int taskOrdinal, int hidden, int intermediate, int nTokens = 0) {
  const int nTokenTiles = taskTokenTiles(params, nTokens);
  const int nLocalExperts = params.config.numExperts / params.config.worldSize;
  return makeTaskScheduler<KernelParams>(nTokenTiles, nLocalExperts, hidden, intermediate)
      .taskAt(params, taskOrdinal, nTokens);
}

template <class KernelParams>
MSCCLPP_DEVICE_INLINE int taskKTiles(const KernelParams&, const Task& task, int hidden, int intermediate) {
  constexpr int KernelTileK = KernelParams::Tiles::K;
  return ((task.fc1 ? hidden : intermediate) + KernelTileK - 1) / KernelTileK;
}

template <class KernelParams>
MSCCLPP_DEVICE_INLINE auto taskCoord(const KernelParams&, const Task& task, int ctaIdx) {
  constexpr int BlockStride = KernelParams::Tiles::TokenStride / KernelParams::Tiles::N;
  return cute::make_coord(task.m * KernelParams::Tiles::ClusterSize + ctaIdx, task.block * BlockStride, 0,
                          task.tokens.expertIdx);
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_TASKS_CUH_
