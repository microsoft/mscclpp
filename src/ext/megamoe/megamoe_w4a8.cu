// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <array>
#include <stdexcept>

#include "megamoe_dispatch.cuh"
#include "megamoe_epilogue.cuh"
#include "megamoe_mma.cuh"
#include "megamoe_routing.cuh"
#include "megamoe_tasks.cuh"
#include "megamoe_w4a8_roles.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

#if !defined(MSCCLPP_MEGAMOE_JIT_MODULE) || !MSCCLPP_MEGAMOE_JIT_MODULE
#if MSCCLPP_MEGAMOE_TRACE
extern "C" int mscclpp_megamoe_w4_trace_reset() { return int(resetKernelTrace()); }

extern "C" int mscclpp_megamoe_w4_trace_copy(void* events, size_t bytes, uint32_t* counts, size_t countBytes) {
  return int(copyKernelTrace(events, bytes, counts, countBytes));
}
#endif
#endif

template <int Hidden, int Intermediate, bool FixedTokenCount = false, ActivationDtype ActDtype = ActivationDtype::Mxfp8>
__global__ __launch_bounds__(W4Threads,
                             1) void megaMoeW4A8(__grid_constant__ const W4A8Parameters parameters, int nTokens,
                                                 __bfloat16* output, uint32_t* kernelEntrySignal, const void* input,
                                                 const uint8_t* inputScales, const int32_t* ids, const float* scores) {
  using Parameters = W4A8Parameters;
  using Types = typename Parameters::Collective;
  using Schedule = typename Types::WarpSchedule;
  static_assert(validW4RegisterBudget(Types::EpilogueWarps, Types::EpilogueRegisters, Types::TransferRegisters));

  extern __shared__ __align__(1024) char rawSharedStorage[];
  auto& sharedStorage = *reinterpret_cast<W4A8SharedStorageT<Types>*>(rawSharedStorage);
  auto params = makeW4A8KernelView<Hidden, Intermediate, ActDtype>(parameters);
  const KernelThreadLocation threadLoc = makeKernelThreadLocation<Parameters::Tiles::ClusterSize>();

  // 1. Signal kernel entry and initialize pipelines, TMEM, and dispatch barriers.
  signalKernelEntry(kernelEntrySignal);
  auto pipelines = initW4A8PipelinesAndStorage<Types>(sharedStorage, threadLoc);

  // 2. Exchange routes and initialize static FP8 activation scales when required.
  runW4A8RoutingStage<FixedTokenCount>(params, nTokens, ids, scores);
  initW4A8StaticScaleStage<ActDtype, Types>(sharedStorage);

  // 3. Execute warp-specialized roles.
  dispatchW4A8WarpRole<Types>(threadLoc.warpIdx, [&]<WarpRole Role>() {
    if constexpr (Role == WarpRole::Dispatch) {
      runDispatchRole<Hidden, Types::DispatchChunk, Types::DispatchStages>(params, sharedStorage, threadLoc.warpIdx,
                                                                           nTokens, input, inputScales, scores);
    } else if constexpr (Role == WarpRole::OutputOffload) {
      runOutputStoreOffloadRole<Schedule>(params, sharedStorage, threadLoc.warpIdx, threadLoc.laneIdx, Hidden, output,
                                          nTokens);
    } else if constexpr (Role == WarpRole::Load) {
      runW4A8LoadStage(params, sharedStorage, pipelines, threadLoc);
    } else if constexpr (Role == WarpRole::Mma) {
      runW4A8MmaStage(params, sharedStorage, pipelines, threadLoc);
    } else if constexpr (Role == WarpRole::Epilogue) {
      runW4A8EpilogueStage<Schedule>(params, sharedStorage, pipelines, threadLoc, nTokens, output);
    }
  });

  // 4. Restore warpgroup registers, release TMEM, and complete output synchronization.
  finishW4A8KernelStage<Hidden, Types>(params, sharedStorage, pipelines.allocator, threadLoc.warpIdx, nTokens, output);
}

struct W4A8KernelRegistration {
  bool (*matches)(const NativeConfig&);
  W4A8KernelEntry entry;
  W4A8KernelEntry captureEntry;
  W4A8KernelEntry staticFp8Entry;
  W4A8KernelEntry staticFp8CaptureEntry;
};

template <int Hidden, int Intermediate>
constexpr W4A8KernelRegistration registerW4A8Kernel() {
  static_assert(Hidden >= 4096);
  return {[](const NativeConfig& config) { return config.hidden == Hidden && config.intermediate == Intermediate; },
          megaMoeW4A8<Hidden, Intermediate, false, ActivationDtype::Mxfp8>,
          megaMoeW4A8<Hidden, Intermediate, W4FixedTokenCount, ActivationDtype::Mxfp8>,
          megaMoeW4A8<Hidden, Intermediate, false, ActivationDtype::Fp8Static>,
          megaMoeW4A8<Hidden, Intermediate, W4FixedTokenCount, ActivationDtype::Fp8Static>};
}

W4A8KernelEntry w4a8KernelEntry(const NativeConfig& config, bool capturing) {
  static constexpr std::array kernels{registerW4A8Kernel<4096, 4352>(), registerW4A8Kernel<4096, 4608>(),
                                      registerW4A8Kernel<4096, 5120>(), registerW4A8Kernel<4096, 6656>(),
                                      registerW4A8Kernel<5120, 2688>(), registerW4A8Kernel<8192, 4096>(),
                                      registerW4A8Kernel<9216, 4096>(), registerW4A8Kernel<9216, 4608>()};
  const bool staticFp8 = config.activationDtype == ActivationDtype::Fp8Static;
  for (const auto& kernel : kernels) {
    if (kernel.matches(config))
      return staticFp8 ? (capturing ? kernel.staticFp8CaptureEntry : kernel.staticFp8Entry)
                       : (capturing ? kernel.captureEntry : kernel.entry);
  }
  throw std::invalid_argument("MegaMoE W4A8 H/I has no compiled specialization");
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail
