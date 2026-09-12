// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_HPP_
#define MSCCLPP_EXT_MEGAMOE_HPP_

#include <mscclpp/core.hpp>

#include "megamoe_kernel.hpp"

namespace mscclpp::megamoe {

/// Owns a native MegaMoE plan, packed weights, registered workspace, and peer mappings.
///
/// Construction is collective and must take place outside CUDA Graph capture on
/// the rank's current CUDA device. All ranks must keep their contexts alive until
/// all collective launches have completed. A context is not concurrently reusable:
/// launches (including graph replays) must be ordered on a single CUDA stream.
class MegaMoeContext {
 public:
  /// Collectively create a context and pack canonical MXFP8 weights.
  /// @param communicator Communicator spanning one NVLink fabric.
  /// @param config Identical configuration on all ranks, except for rank.
  /// @param weights Canonical local weights and E8M0 K32 scales.
  /// @param stream CUDA stream used to pack weights; synchronized before returning.
  /// @param tag Bootstrap tag reserved for this collective context construction.
  MegaMoeContext(std::shared_ptr<Communicator> communicator, const NativeConfig& config, const PackedWeights& weights,
                 cudaStream_t stream = nullptr, int tag = 17920);
  ~MegaMoeContext();
  MegaMoeContext(const MegaMoeContext&) = delete;
  MegaMoeContext& operator=(const MegaMoeContext&) = delete;

  /// Return the immutable configuration.
  const NativeConfig& config() const;
  /// Return the owning CUDA device ordinal.
  int device() const;
  /// Return the BF16 [maxTokens, hidden] registered input buffer.
  void* input() const;
  /// Return the int32 [maxTokens, topK] registered routing buffer.
  void* topkIds() const;
  /// Return the float32 [maxTokens, topK] registered router-weight buffer.
  void* topkWeights() const;
  /// Return the number of CTAs launched by the persistent kernel.
  int ctaCount() const;
  /// Return the dynamic shared memory bytes per CTA.
  size_t sharedBytes() const;
  /// Return the registered symmetric workspace size in bytes.
  size_t symmetricBytes() const;
  /// Return the private workspace size in bytes, excluding weights.
  size_t privateBytes() const;

  /// Stage inputs and enqueue routed SwiGLU expert computation and combination.
  ///
  /// Pointer arguments reference contiguous CUDA arrays on the owning device.
  /// Output is BF16 [numTokens, hidden], disjoint from the registered workspace.
  /// Copies are skipped for exact aliases of the corresponding registered input.
  /// No allocation, registration, host synchronization, or host epoch update occurs
  /// here. This method is CUDA Graph capturable. Caller-owned arrays and this
  /// context must outlive all queued execution and every graph replay.
  void forward(const void* input, const int32_t* topkIds, const float* topkWeights, void* output, int numTokens,
               cudaStream_t stream);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mscclpp::megamoe

#endif  // MSCCLPP_EXT_MEGAMOE_HPP_
