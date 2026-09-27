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

struct NativeConfig {
  int rank = 0;
  int worldSize = 1;
  int maxTokens = 32;
  int hidden = 4096;
  int intermediate = 4352;
  int numExperts = 512;
  int topK = 7;
  int smMargin = 0;
  bool weightE5M2 = false;
  // Negative disables clamping; otherwise gate <= clamp and -clamp <= up <= clamp.
  float gateUpClamp = -1.0f;
  bool weightMxfp4 = false;
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
};

struct PackedWeights {
  uint8_t* fc1 = nullptr;
  uint8_t* fc1Scale = nullptr;
  uint8_t* fc2 = nullptr;
  uint8_t* fc2Scale = nullptr;
};

struct KernelResources {
  int ctas = 0;
  int device = -1;
  size_t sharedBytes = 0;
};

}  // namespace mscclpp::megamoe

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE {

// MXFP8 weights are canonical [E_local, 2I, H] / [E_local, H, I]. MXFP4
// weights are packed uint8 [E_local, 2I, H/2] / [E_local, H, I/2], with
// the lower nibble holding the even K element. Both use canonical row-major
// E8M0 scales [E_local, M, K/32].
void packNativeWeights(const NativeConfig& config, const PackedWeights& source, const PackedWeights& destination,
                       cudaStream_t stream);

SymmetricLayout getSymmetricLayout(const NativeConfig& config);
size_t getPrivateWorkspaceBytes(const NativeConfig& config);
void validateNativeConfig(const NativeConfig& config);
KernelResources preflightKernel(const NativeConfig& config);

struct KernelPlan;

// All allocations and imported peer mappings must outlive this plan and every
// queued launch. Create outside CUDA Graph capture; initialize workspaces to zero.
std::shared_ptr<KernelPlan> createKernelPlan(const NativeConfig& config, void* symmetricBase,
                                             const uint64_t* devicePeerBases, void* privateWorkspace,
                                             const PackedWeights& weights);
int kernelPlanCtaCount(const KernelPlan& plan);
size_t kernelPlanSharedBytes(const KernelPlan& plan);
void launchNativeMegaMoe(const std::shared_ptr<KernelPlan>& plan, int numTokens, void* output, cudaStream_t stream,
                         uint32_t* startSignal = nullptr);
void launchNativeW4A8(const std::shared_ptr<KernelPlan>& plan, const void* input, const int32_t* ids,
                      const float* scores, int numTokens, void* output, cudaStream_t stream, uint32_t* startSignal);
void launchNativeSharedExpert(const std::shared_ptr<KernelPlan>& plan, int numTokens, void* output,
                              cudaStream_t stream);

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE

#endif  // MSCCLPP_EXT_MEGAMOE_KERNEL_HPP_
