// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_KERNEL_HPP_
#define MSCCLPP_EXT_MEGAMOE_KERNEL_HPP_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>

#include "megamoe_specialization.hpp"

namespace mscclpp::megamoe {

enum class WeightDtype : int32_t {
  Mxfp4 = 0,
  Mxfp8 = 1,
  // fp8_nv: FP8 E4M3 values with one FP32 scale per weight matrix. Single local W8A8 experts only.
  Fp8Nv = 2,
};

enum class ActivationDtype : int32_t {
  Mxfp8 = 0,
  Fp8Static = 1,
  Bf16 = 2,
};

struct NativeConfig {
  int rank = 0;
  int worldSize = 1;
  int maxTokens = 32;
  int hidden = 4096;
  int intermediate = 6656;
  int numExperts = 512;
  int topK = 7;
  int smMargin = 0;
  // Negative disables clamping; otherwise gate <= clamp and -clamp <= up <= clamp.
  float gateUpClamp = -1.0f;
  WeightDtype weightDtype = WeightDtype::Mxfp4;
  ActivationDtype activationDtype = ActivationDtype::Mxfp8;
  float activationScale = 1.0f;
  bool combineRouteWeights = false;
};

struct SharedExpertConfig {
  // Zero reuses the routed hidden width.
  int hidden = 0;
  int intermediate = 0;
  // Negative disables clamping; otherwise gate <= clamp and -clamp <= up <= clamp.
  float gateUpClamp = -1.0f;
  // Delay before the shared FC1 loads start, in ns; negative selects the default. Runtime-only: it does not change
  // the workspace layout. MSCCLPP_MEGAMOE_FUSED_FC1_DELAY_NS overrides it.
  int fc1DelayNs = -1;
};

struct SymmetricLayout {
  size_t bytes = 0;
  size_t input = 0;
  size_t topkIds = 0;
  size_t topkWeights = 0;
  size_t partialOutput = 0;
  size_t epoch = 0;
  size_t peerSignals = 0;
  size_t expectedPeerSignals = 0;
  size_t tokenCount = 0;
  size_t quantizedInput = 0;
  size_t quantizedInputScale = 0;
  size_t quantizedInputReady = 0;
  size_t routedInputReady = 0;
  size_t routingHeaders = 0;
  size_t routingPackets = 0;
};

struct PackedWeights {
  uint8_t* fc1 = nullptr;
  uint8_t* fc1Scale = nullptr;
  uint8_t* fc2 = nullptr;
  uint8_t* fc2Scale = nullptr;
  // Host copies of the per-matrix fp8_nv weight scales, applied in the W8A8 epilogue; 1 for block-scaled weights.
  float fc1TensorScale = 1.0f;
  float fc2TensorScale = 1.0f;
};

struct KernelResources {
  int ctas = 0;
  int device = -1;
  size_t sharedBytes = 0;
};

}  // namespace mscclpp::megamoe

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE {

// MXFP4 weights are packed uint8 [E_local, 2I, H/2] / [E_local, H, I/2],
// with the lower nibble holding the even K element. Scales use canonical
// row-major E8M0 [E_local, M, K/32].
void packNativeWeights(const NativeConfig& config, const PackedWeights& source, const PackedWeights& destination,
                       cudaStream_t stream);

// Packs the canonical fp8_nv weights of the fused shared expert of `config` (same layout as packNativeWeights for a
// single local W8A8 expert).
void packSharedExpertWeights(const NativeConfig& config, const SharedExpertConfig& sharedConfig,
                             const PackedWeights& source, const PackedWeights& destination, cudaStream_t stream);

SymmetricLayout getSymmetricLayout(const NativeConfig& config);
size_t getPrivateWorkspaceBytes(const NativeConfig& config);
size_t getFusedPrivateWorkspaceBytes(const NativeConfig& config, const SharedExpertConfig& sharedConfig);
void validateNativeConfig(const NativeConfig& config);
void validateFusedConfig(const NativeConfig& config, const SharedExpertConfig& sharedConfig);
KernelResources preflightKernel(const NativeConfig& config);
KernelResources preflightFusedKernel(const NativeConfig& config, const SharedExpertConfig& sharedConfig);

struct KernelPlan;

// All allocations and imported peer mappings must outlive this plan and every
// queued launch. Create outside CUDA Graph capture; initialize workspaces to zero.
std::shared_ptr<KernelPlan> createKernelPlan(const NativeConfig& config, void* symmetricBase,
                                             const uint64_t* devicePeerBases, void* privateWorkspace,
                                             const PackedWeights& weights);
std::shared_ptr<KernelPlan> createFusedKernelPlan(const NativeConfig& config, const SharedExpertConfig& sharedConfig,
                                                  void* symmetricBase, const uint64_t* devicePeerBases,
                                                  void* privateWorkspace, const PackedWeights& weights,
                                                  const PackedWeights& sharedWeights);
int kernelPlanCtaCount(const KernelPlan& plan);
size_t kernelPlanSharedBytes(const KernelPlan& plan);
void launchNativeW4A8(const std::shared_ptr<KernelPlan>& plan, const void* input, const int32_t* ids,
                      const float* scores, int numTokens, void* output, cudaStream_t stream,
                      uint32_t* kernelEntrySignal);
void launchNativeW4A8FusedShared(const std::shared_ptr<KernelPlan>& plan, const void* input, const void* sharedInput,
                                 const int32_t* ids, const float* scores, int numTokens, void* output,
                                 void* sharedOutput, cudaStream_t stream, uint32_t* kernelEntrySignal);
void launchNativeW8A16(const std::shared_ptr<KernelPlan>& plan, const int32_t* ids, const float* scores, int numTokens,
                       void* output, cudaStream_t stream, uint32_t* kernelEntrySignal);
// Runs the single local expert unweighted. A W8A8 plan quantizes `input` (16-byte aligned BF16) or, when null, the
// staged symmetric input; a W8A16 plan always reads the staged symmetric input.
void launchNativeSharedExpert(const std::shared_ptr<KernelPlan>& plan, int numTokens, void* output, cudaStream_t stream,
                              const void* input = nullptr);

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE

#endif  // MSCCLPP_EXT_MEGAMOE_KERNEL_HPP_
