// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <array>
#include <stdexcept>

#include "megamoe_routing.cuh"
#include "megamoe_w8a16_roles.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

#if (!defined(MSCCLPP_MEGAMOE_JIT_MODULE) || !MSCCLPP_MEGAMOE_JIT_MODULE) && MSCCLPP_MEGAMOE_TRACE
extern "C" int mscclpp_megamoe_w8_trace_reset() { return int(resetKernelTrace()); }

extern "C" int mscclpp_megamoe_w8_trace_copy(void* events, size_t bytes, uint32_t* counts, size_t countBytes) {
  return int(copyKernelTrace(events, bytes, counts, countBytes));
}
#endif

template <int Hidden, int Intermediate, int LocalMode = 0,
          class Parameters = std::conditional_t<LocalMode != 0, W8A16LocalParameters, W8A16Parameters>>
__global__ __launch_bounds__((LocalMode ? LocalThreads : W4Threads),
                             1) void megaMoeW8A16(__grid_constant__ const Parameters parameters, int nTokens,
                                                  __bfloat16* output, uint32_t* kernelEntrySignal, const void* input,
                                                  const int32_t* ids, const float* scores) {
  constexpr bool Local = LocalMode != 0;
  using Types = W8A16CollectiveTypesT<Local>;
  using Schedule = typename Types::WarpSchedule;
  using SharedStorage = typename Parameters::SharedStorage;

  // 1. Signal kernel entry and initialize pipelines, TMEM, and dispatch barriers.
  signalKernelEntry(kernelEntrySignal);
  if (Local && nTokens == 0) return;

  extern __shared__ __align__(1024) char rawSharedStorage[];
  auto& sharedStorage = *reinterpret_cast<SharedStorage*>(rawSharedStorage);
  auto params = makeW8A16KernelView<Hidden, Intermediate>(parameters);
  const KernelThreadLocation threadLoc = makeKernelThreadLocation<Parameters::Tiles::ClusterSize>();
  auto pipelines = initW8A16PipelinesAndStorage<Local>(sharedStorage, threadLoc);

  // 2. Exchange routes (or synchronize local-expert tiles).
  const int nTokenTiles = runW8A16RoutingStage<Local>(params, nTokens, ids, scores);
  auto computeCtx = makeW8A16ComputeContext(params, sharedStorage, threadLoc.ctaIdx, nTokenTiles);

  // 3. Execute warp-specialized roles.
  dispatchW8A16WarpRole<Local, Schedule>(threadLoc.warpIdx, [&]<WarpRole Role>() {
    if constexpr (Role == WarpRole::Dispatch) {
      if constexpr (!Local) {
        runDispatchRole<Hidden, Types::DispatchChunk, Types::DispatchStages>(params, sharedStorage, threadLoc.warpIdx,
                                                                             nTokens, input, nullptr, scores);
      }
    } else if constexpr (Role == WarpRole::LoadA) {
      runW8A16LoadAStage(params, sharedStorage, pipelines.loadA, computeCtx, threadLoc, nTokens);
    } else if constexpr (Role == WarpRole::LoadB) {
      runW8A16LoadBStage<Local>(params, sharedStorage, pipelines.loadB, computeCtx, threadLoc, nTokens);
    } else if constexpr (Role == WarpRole::Transform) {
      runW8A16TransformStage<Local>(params, sharedStorage, pipelines.loadA, pipelines.transform, computeCtx, threadLoc,
                                    nTokens);
    } else if constexpr (Role == WarpRole::Mma) {
      runW8A16MmaStage<Local, Schedule>(params, sharedStorage, pipelines, computeCtx, threadLoc, nTokens);
    } else if constexpr (Role == WarpRole::Epilogue) {
      runW8A16EpilogueStage<Schedule, LocalMode>(params, sharedStorage, pipelines.accumulate, computeCtx, threadLoc,
                                                 nTokens, output);
    }
  });

  // 4. Restore warpgroup registers, release TMEM, and combine outputs.
  finishW8A16KernelStage<Hidden, Local>(params, sharedStorage, pipelines.allocator, threadLoc.warpIdx, nTokens, output);
}

struct W8A16KernelRegistration {
  bool (*matches)(const NativeConfig&);
  W8A16KernelEntry<0> routedEntry;
  W8A16KernelEntry<1> localEntry;
  W8A16KernelEntry<2> sharedEntry;
};

template <int Hidden, int Intermediate>
constexpr W8A16KernelRegistration registerW8A16Kernel() {
  static_assert(Hidden >= 4096);
  return {[](const NativeConfig& config) { return config.hidden == Hidden && config.intermediate == Intermediate; },
          megaMoeW8A16<Hidden, Intermediate, 0>, megaMoeW8A16<Hidden, Intermediate, 1>,
          megaMoeW8A16<Hidden, Intermediate, 2>};
}

template <int LocalMode>
W8A16KernelEntry<LocalMode> w8a16KernelEntry(const NativeConfig& config) {
  static constexpr std::array kernels{registerW8A16Kernel<4096, 4352>(), registerW8A16Kernel<4096, 4608>(),
                                      registerW8A16Kernel<4096, 5120>(), registerW8A16Kernel<4096, 6656>(),
                                      registerW8A16Kernel<5120, 2688>(), registerW8A16Kernel<8192, 4096>(),
                                      registerW8A16Kernel<9216, 4096>(), registerW8A16Kernel<9216, 4608>()};
  for (const auto& kernel : kernels) {
    if (kernel.matches(config)) {
      if constexpr (LocalMode == 0)
        return kernel.routedEntry;
      else if constexpr (LocalMode == 1)
        return kernel.localEntry;
      else
        return kernel.sharedEntry;
    }
  }
  throw std::invalid_argument("MegaMoE W8A16 H/I has no compiled specialization");
}

template W8A16KernelEntry<0> w8a16KernelEntry<0>(const NativeConfig&);
template W8A16KernelEntry<1> w8a16KernelEntry<1>(const NativeConfig&);
template W8A16KernelEntry<2> w8a16KernelEntry<2>(const NativeConfig&);

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail
