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
  /// @param signalStart Reset and publish an execution-start signal for waitUntilStarted().
  /// This opt-in adds a reset/event before the kernel; ordinary forwards do not.
  void forward(const void* input, const int32_t* topkIds, const float* topkWeights, void* output, int numTokens,
               cudaStream_t stream, bool signalStart = false);

  /// Order a consumer stream after the latest signalStart forward has entered its kernel.
  /// This does not wait for completion or make the forward's output ready. The
  /// reset event prevents reading a prior replay's signal. The stream memory wait
  /// uses no resident CTA and is CUDA Graph capturable. Join the consumer stream
  /// before reusing the context, including before the next graph replay.
  /// @param stream Consumer stream on the context's device.
  void waitUntilStarted(cudaStream_t stream);

  /// Enqueue one unweighted local expert without routing metadata or communication.
  /// Requires worldSize=1, numExperts=1, topK=1. Inputs/outputs are BF16
  /// [numTokens, hidden]. Shares the context's nonconcurrent workspace lifetime.
  void forwardShared(const void* input, void* output, int numTokens, cudaStream_t stream);

 private:
  size_t validateForward(const void* input, void* output, int numTokens) const;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mscclpp::megamoe

#endif  // MSCCLPP_EXT_MEGAMOE_HPP_
