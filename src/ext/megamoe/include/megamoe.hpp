// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_HPP_
#define MSCCLPP_EXT_MEGAMOE_HPP_

#include <array>
#include <mscclpp/core.hpp>
#include <string>

#include "megamoe_kernel.hpp"

namespace mscclpp::megamoe {

/// Owns a native MegaMoE plan, packed weights, registered workspace, and peer mappings.
///
/// Construction is collective and must take place outside CUDA Graph capture on
/// the rank's current CUDA device. All ranks must keep their contexts alive until
/// all collective launches have completed. A context is not concurrently reusable:
/// launches (including graph replays) must be ordered on a single CUDA stream,
/// or explicitly ordered with CUDA events. Contexts sharing a workspace form one
/// nonconcurrent execution domain and must be used in the same order on all ranks.
class MegaMoeContext {
 public:
  /// Collectively create a context and pack canonical MXFP4 weights.
  /// @param communicator Communicator spanning one NVLink fabric.
  /// @param config Identical configuration on all ranks, except for rank.
  /// @param weights Canonical local weights and E8M0 K32 scales. MXFP4 values
  /// are packed two per byte with the even K element in the lower nibble.
  /// @param stream CUDA stream used to pack weights; synchronized before returning.
  /// @param tag Bootstrap tag reserved for this collective context construction.
  /// @param kernelPath Absolute path to an optional native JIT module; empty selects the builtin kernel.
  /// @param kernelId Expected 64-character specialization digest; empty for the builtin kernel.
  /// @param sharedWorkspace Optional context whose runtime workspace and peer mappings are reused.
  /// Must use the same communicator, device, configuration, and kernel specialization.
  /// Sharing mode and the original workspace owner must agree on every rank.
  /// Packed weights and weight-specific plans remain independently owned. The
  /// original context may be destroyed while aliases remain alive.
  MegaMoeContext(std::shared_ptr<Communicator> communicator, const NativeConfig& config, const PackedWeights& weights,
                 cudaStream_t stream = nullptr, int tag = 17920, const std::string& kernelPath = "",
                 const std::string& kernelId = "", std::shared_ptr<MegaMoeContext> sharedWorkspace = nullptr);
  MegaMoeContext(std::shared_ptr<Communicator> communicator, const NativeConfig& config, const PackedWeights& weights,
                 const SharedExpertConfig& sharedConfig, const PackedWeights& sharedWeights,
                 cudaStream_t stream = nullptr, int tag = 17920,
                 std::shared_ptr<MegaMoeContext> sharedWorkspace = nullptr);
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
  /// Return the private workspace address; equal for live contexts iff their runtime is shared.
  uintptr_t workspaceId() const;
  /// Return this context's distinct packed FC1, FC1-scale, FC2, and FC2-scale addresses.
  std::array<uintptr_t, 4> packedWeightPointers() const;
  /// Return the specialization digest, or "builtin" for the builtin module.
  const std::string& kernelId() const;
  /// Return the routed output-feature tile width.
  int kernelTileM() const;
  /// Return the number of cooperating CTAs per cluster.
  int kernelClusterSize() const;
  /// Return the routed token tile width.
  int kernelTileN() const;
  /// Return the routed reduction tile width.
  int kernelTileK() const;
  /// Return the routed raw-weight/scale/activation load pipeline depth.
  int kernelLoadStages() const;
  /// Return the routed transformed-weight pipeline depth (zero for W4A8).
  int kernelTransformStages() const;
  /// Return the routed kernel warp count.
  int kernelNumWarps() const;
  /// Return the W4A8 transfer-role register budget.
  int kernelTransferRegisters() const;
  /// Return the W4A8 loader-warp count.
  int kernelLoadWarps() const;
  /// Return whether the W4A8 weight and activation loaders use separate pipelines.
  bool kernelSplitPipelines() const;
  /// Return the W4A8 epilogue token chunk.
  int kernelEpilogueTokens() const;
  /// Return the W4A8 epilogue warp count.
  int kernelEpilogueWarps() const;
  /// Return the W4A8 epilogue register budget.
  int kernelEpilogueRegisters() const;
  /// Return the W4A8 dispatch chunk in bytes.
  int kernelDispatchChunk() const;
  /// Return the W4A8 dispatch warp count.
  int kernelDispatchWarps() const;
  /// Return the W4A8 dispatch pipeline depth.
  int kernelDispatchStages() const;
  /// Return whether captured W4A8 decode assumes equal input row counts on all ranks.
  bool kernelFixedTokenCount() const;
  /// Return whether W4A8 tasks reuse an output tile across each expert's token blocks.
  bool kernelExpertMajorTasks() const;
  /// Return whether all W4A8 weights and their scales use normal L2 priority.
  bool kernelCacheWeights() const;
  /// Return whether W4A8 weights and their scales use normal L2 priority only for experts spanning multiple token
  /// tiles.
  bool kernelAdaptiveCacheWeights() const;
  /// Return whether small static-FP8 batches quantize their BF16 inputs inside the W4A8 kernel instead of a separate
  /// launch.
  bool kernelFuseStaticQuantize() const;

  /// Stage inputs and enqueue routed SwiGLU expert computation and combination.
  ///
  /// Pointer arguments reference contiguous CUDA arrays on the owning device.
  /// Output is BF16 [numTokens, hidden], disjoint from the registered workspace.
  /// Copies are skipped for exact aliases of the corresponding registered input.
  /// No allocation, registration, host synchronization, or host epoch update occurs
  /// here. This method is CUDA Graph capturable. Caller-owned arrays and this
  /// context must outlive all queued execution and every graph replay.
  /// W4A8 modules with fixed token counts require equal captured input row counts
  /// across ranks. Expert ID -1 masks unused slots; valid route counts may differ.
  /// @param signalStart Reset and publish an execution-start signal for waitUntilStarted().
  /// This opt-in adds a reset/event before the kernel; ordinary forwards do not.
  void forward(const void* input, const int32_t* topkIds, const float* topkWeights, void* output, int numTokens,
               cudaStream_t stream, bool signalStart = false);

  /// Order a consumer stream after the latest signalStart forward has entered its kernel.
  /// This does not wait for completion or make the forward's output ready. The
  /// reset event prevents reading a prior replay's signal. The stream memory wait
  /// uses no resident CTA and is CUDA Graph capturable. Join the consumer stream
  /// before reusing the context, including before the next graph replay.
  /// Workspace aliases share the latest signal/reset event and this ordering
  /// obligation; any alias's next forward replaces or invalidates that signal.
  /// @param stream Consumer stream on the context's device.
  void waitUntilStarted(cudaStream_t stream);

  /// Enqueue one unweighted local expert without routing metadata or communication.
  /// Requires weightDtype=Mxfp8, worldSize=1, numExperts=1, topK=1. Inputs/outputs
  /// are BF16 [numTokens, hidden]. Shares the context's nonconcurrent workspace lifetime.
  void forwardShared(const void* input, void* output, int numTokens, cudaStream_t stream);

  /// Enqueue routed experts and one fused unweighted local fp8_nv shared expert.
  /// The routed input/output use config.hidden; the shared input/output use
  /// SharedExpertConfig.hidden (or config.hidden when zero). Outputs must not overlap.
  void forwardFusedShared(const void* input, const void* sharedInput, const int32_t* topkIds, const float* topkWeights,
                          void* output, void* sharedOutput, int numTokens, cudaStream_t stream,
                          bool signalStart = false);

 private:
  MegaMoeContext(std::shared_ptr<Communicator> communicator, const NativeConfig& config, const PackedWeights& weights,
                 const SharedExpertConfig* sharedConfig, const PackedWeights* sharedWeights, cudaStream_t stream,
                 int tag, const std::string& kernelPath, const std::string& kernelId,
                 std::shared_ptr<MegaMoeContext> sharedWorkspace);
  void forwardImpl(const void* input, const void* sharedInput, const int32_t* topkIds, const float* topkWeights,
                   void* output, void* sharedOutput, int numTokens, cudaStream_t stream, bool signalStart, bool fused);
  size_t validateForward(const void* input, void* output, int numTokens) const;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mscclpp::megamoe

#endif  // MSCCLPP_EXT_MEGAMOE_HPP_
