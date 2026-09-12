// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_KERNEL_HPP_
#define MSCCLPP_EXT_MEGAMOE_KERNEL_HPP_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>

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
};

struct SymmetricLayout {
  size_t bytes = 0;
  size_t input = 0;
  size_t topkIds = 0;
  size_t topkWeights = 0;
  size_t partialOutput = 0;
  size_t epoch = 0;
  size_t readySignals = 0;
  size_t doneSignals = 0;
  size_t tokenCount = 0;
};

struct PackedWeights {
  uint8_t* fc1 = nullptr;
  uint8_t* fc1Scale = nullptr;
  uint8_t* fc2 = nullptr;
  uint8_t* fc2Scale = nullptr;
};

// Input weights are canonical [E_local, 2I, H] / [E_local, H, I] with
// row-major E8M0 scales [E_local, M, K/32]. Packing preserves byte counts.
void packNativeWeights(const NativeConfig& config, const PackedWeights& source, const PackedWeights& destination,
                       cudaStream_t stream);

SymmetricLayout getSymmetricLayout(const NativeConfig& config);
size_t getPrivateWorkspaceBytes(const NativeConfig& config);
void validateNativeConfig(const NativeConfig& config);

struct KernelPlan;

// All allocations and imported peer mappings must outlive this plan and every
// queued launch. Create outside CUDA Graph capture; initialize workspaces to zero.
std::shared_ptr<KernelPlan> createKernelPlan(const NativeConfig& config, void* symmetricBase,
                                             const uint64_t* devicePeerBases, void* privateWorkspace,
                                             const PackedWeights& weights);
int kernelPlanCtaCount(const KernelPlan& plan);
size_t kernelPlanSharedBytes(const KernelPlan& plan);
void launchNativeMegaMoe(const std::shared_ptr<KernelPlan>& plan, int numTokens, void* output, cudaStream_t stream);

}  // namespace mscclpp::megamoe

#endif  // MSCCLPP_EXT_MEGAMOE_KERNEL_HPP_
