// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "megamoe_jit.hpp"
#include "megamoe_kernel.hpp"

#if !defined(MSCCLPP_MEGAMOE_JIT_MODULE) || !MSCCLPP_MEGAMOE_JIT_MODULE
#error "The MegaMoE JIT entrypoint must be built as a JIT module"
#endif
#ifndef MSCCLPP_MEGAMOE_JIT_ID
#error "A MegaMoE JIT module requires MSCCLPP_MEGAMOE_JIT_ID"
#endif

namespace {

using namespace mscclpp::megamoe;
using JitPlan = std::shared_ptr<jit::KernelPlan>;

constexpr char JitId[] = MSCCLPP_MEGAMOE_JIT_ID;
static_assert(sizeof(JitId) == MSCCLPP_MEGAMOE_JIT_ID_CAPACITY, "MegaMoE JIT ID must be a 64-character hex digest");
static_assert(
    [] {
      for (size_t i = 0; i < sizeof(JitId) - 1; ++i)
        if (!((JitId[i] >= '0' && JitId[i] <= '9') || (JitId[i] >= 'a' && JitId[i] <= 'f'))) return false;
      return true;
    }(),
    "MegaMoE JIT ID must be a lowercase hex digest");

template <class Function>
int jitCall(char* error, size_t capacity, Function&& function) noexcept {
  if (error && capacity) error[0] = '\0';
  auto report = [&](const char* message) {
    if (error && capacity) {
      const size_t bytes = std::min(capacity - 1, std::strlen(message));
      std::memcpy(error, message, bytes);
      error[bytes] = '\0';
    }
  };
  try {
    function();
    return 0;
  } catch (const std::invalid_argument& e) {
    report(e.what());
    return 1;
  } catch (const std::exception& e) {
    report(e.what());
    return 2;
  } catch (...) {
    report("Unknown C++ exception in MegaMoE JIT module");
    return 2;
  }
}

NativeConfig nativeConfig(const MegaMoeJitConfigV1* c) {
  if (!c || (c->weightE5M2 != 0 && c->weightE5M2 != 1))
    throw std::invalid_argument("Invalid MegaMoE JIT configuration");
  return NativeConfig{c->rank,       c->worldSize, c->maxTokens, c->hidden,           c->intermediate,
                      c->numExperts, c->topK,      c->smMargin,  bool(c->weightE5M2), c->gateUpClamp};
}

PackedWeights nativeWeights(const MegaMoeJitWeightsV1* weights) {
  if (!weights) throw std::invalid_argument("Null MegaMoE JIT weight descriptor");
  return PackedWeights{weights->fc1, weights->fc1Scale, weights->fc2, weights->fc2Scale};
}

int jitPreflight(const MegaMoeJitConfigV1* config, MegaMoeJitLayoutV1* output, char* error, size_t capacity) noexcept {
  return jitCall(error, capacity, [&] {
    if (!output) throw std::invalid_argument("Null MegaMoE JIT layout descriptor");
    auto c = nativeConfig(config);
    auto layout = jit::getSymmetricLayout(c);
    const size_t workspaceBytes = jit::getPrivateWorkspaceBytes(c);
    auto resources = jit::preflightKernel(c);
    *output = MegaMoeJitLayoutV1{layout.bytes,
                                 layout.input,
                                 layout.topkIds,
                                 layout.topkWeights,
                                 layout.partialOutput,
                                 layout.epoch,
                                 layout.peerSignals,
                                 layout.expectedPeerSignals,
                                 layout.tokenCount,
                                 workspaceBytes,
                                 resources.sharedBytes,
                                 resources.ctas,
                                 0};
  });
}

int jitPack(const MegaMoeJitConfigV1* config, const MegaMoeJitWeightsV1* source, const MegaMoeJitWeightsV1* destination,
            void* stream, char* error, size_t capacity) noexcept {
  return jitCall(error, capacity, [&] {
    jit::packNativeWeights(nativeConfig(config), nativeWeights(source), nativeWeights(destination),
                           reinterpret_cast<cudaStream_t>(stream));
  });
}

int jitCreate(const MegaMoeJitConfigV1* config, void* symmetric, const uint64_t* peers, void* workspace,
              const MegaMoeJitWeightsV1* weights, void** plan, char* error, size_t capacity) noexcept {
  if (plan) *plan = nullptr;
  return jitCall(error, capacity, [&] {
    if (!plan) throw std::invalid_argument("Null MegaMoE JIT plan output");
    auto result = std::make_unique<JitPlan>(
        jit::createKernelPlan(nativeConfig(config), symmetric, peers, workspace, nativeWeights(weights)));
    *plan = result.release();
  });
}

void jitDestroy(void* plan) noexcept { delete static_cast<JitPlan*>(plan); }

int jitLaunch(void* plan, int32_t tokens, void* output, void* stream, uint32_t* startSignal, int32_t shared,
              char* error, size_t capacity) noexcept {
  return jitCall(error, capacity, [&] {
    if (!plan || (shared != 0 && shared != 1)) throw std::invalid_argument("Invalid MegaMoE JIT launch");
    const auto& native = *static_cast<JitPlan*>(plan);
    if (shared) {
      jit::launchNativeSharedExpert(native, tokens, output, reinterpret_cast<cudaStream_t>(stream));
    } else {
      jit::launchNativeMegaMoe(native, tokens, output, reinterpret_cast<cudaStream_t>(stream), startSignal);
    }
  });
}

const MegaMoeJitApiV1 JitApi = [] {
  MegaMoeJitApiV1 api{};
  api.abiVersion = MSCCLPP_MEGAMOE_JIT_ABI_VERSION;
  api.structBytes = sizeof(MegaMoeJitApiV1);
  api.configBytes = sizeof(MegaMoeJitConfigV1);
  api.weightsBytes = sizeof(MegaMoeJitWeightsV1);
  api.layoutBytes = sizeof(MegaMoeJitLayoutV1);
  api.tileM = 256;
  api.tileN = jit::detail::TileN;
  api.tileK = 128;
  api.loadStages = jit::detail::LoadStages;
  api.transformStages = jit::detail::TransformStages;
  api.clusterSize = 2;
  api.accumulatorStages = 2;
  api.architecture = 1000;
  std::memcpy(api.kernelId, JitId, sizeof(JitId));
  api.preflight = jitPreflight;
  api.packWeights = jitPack;
  api.createPlan = jitCreate;
  api.destroyPlan = jitDestroy;
  api.launch = jitLaunch;
  return api;
}();

}  // namespace

extern "C" __attribute__((visibility("default"))) const MegaMoeJitApiV1* mscclpp_megamoe_jit_get_api_v1() {
  return &JitApi;
}
