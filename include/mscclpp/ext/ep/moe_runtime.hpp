// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EXT_EP_MOE_RUNTIME_HPP_
#define MSCCLPP_EXT_EP_MOE_RUNTIME_HPP_

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/ext/ep/types.hpp>

namespace mscclpp {
namespace ep {
struct LatencyRuntimeContext;
struct ThroughputRuntimeContext;

/// Unified host runtime for expert-parallel dispatch and combine.
///
/// One runtime owns the communication buffers and synchronization state for the
/// selected mode. LATENCY uses fixed-capacity expert-major or rank-major
/// layouts. THROUGHPUT uses a receive pool exposed as compact token-major or
/// fixed-stride rank-major rows.
/// Preparation, dispatch, and combine execute asynchronously on the request's
/// CUDA stream. Routing counts and prefixes stay in device memory; an empty
/// preparation handle makes dispatch enqueue preparation internally.
/// Host calls sharing a runtime must be serialized, and GPU work reusing its
/// buffers must be ordered, including across streams.
class MoERuntime {
 public:
  /// Construct a runtime for the selected mode and topology.
  ///
  /// Only resources required by @p mode are allocated. Mode-specific
  /// communicator buffers are deferred until initialize().
  /// @param communicator Initialized MSCCL++ communicator.
  /// @param mode Runtime algorithm family.
  /// @param maxTokensPerRank Fixed per-rank token capacity.
  /// @param hidden Hidden dimension.
  /// @param numExperts Global expert count.
  /// @param numTopk Number of routed experts per token.
  /// @param outputLayout Dispatch output layout.
  /// @param combineMode Latency-mode combine algorithm.
  /// @throws EPException For an unsupported mode or invalid configuration.
  /// @warning @p communicator must remain alive until initialize() returns.
  MoERuntime(mscclpp::Communicator& communicator, MoEMode mode, int maxTokensPerRank, int hidden, int numExperts,
             int numTopk, DispatchLayout outputLayout = DispatchLayout::EXPERT_MAJOR,
             CombineMode combineMode = CombineMode::RANK_LOCAL_REDUCE);
  /// Release owned resources without explicitly synchronizing user streams.
  /// The caller must ensure all local and peer GPU work using these resources,
  /// including graph replays, has completed before destruction.
  ~MoERuntime() noexcept(false);

  MoERuntime(const MoERuntime&) = delete;
  MoERuntime& operator=(const MoERuntime&) = delete;

  /// Return the configured runtime mode.
  MoEMode mode() const { return mode_; }
  /// Return whether the selected mode supports the detected topology.
  bool isAvailable() const { return available_; }
  /// Collectively initialize deferred runtime resources.
  ///
  /// All ranks must call this method exactly once and in the same order. The
  /// Python API makes repeated initialize() calls idempotent.
  void initialize();

  /// Return the local rank.
  int rank() const { return rank_; }
  /// Return the global rank count.
  int numRanks() const { return numRanks_; }
  /// Return the NVLink-local rank count.
  int numNvlRanks() const { return numNvlRanks_; }
  /// Return the rank count in one CUDA IPC domain.
  int numRanksPerIpcDomain() const { return numRanksPerIpcDomain_; }

  /// Return the runtime-owned rank-major top-k ID buffer.
  void* outputTopkIdsBuffer() const;
  /// Return the runtime-owned rank-major top-k weight buffer.
  void* outputTopkWeightsBuffer() const;
  /// Return the runtime-owned dispatch output buffer.
  void* dispatchOutputBuffer() const;
  /// Return the runtime-owned combine input buffer.
  void* combineInputBuffer() const;

  /// Collectively prepare throughput routing without moving token payloads.
  ///
  /// Computes local routing counts, exchanges peer counts, and constructs rank
  /// and channel prefixes entirely on the GPU, without a host copy or wait.
  /// Dispatch can consume the handle on another stream using a device-side
  /// event dependency. Preparation is supported only in THROUGHPUT mode and
  /// can be captured together with dispatch in a CUDA graph.
  ///
  /// All ranks must prepare and dispatch in the same order. Starting a valid
  /// preparation invalidates earlier preparation and dispatch handles on this
  /// runtime. Rejected inputs do not invalidate handles. Previously enqueued
  /// work must be ordered before preparation, including across streams.
  /// A captured preparation must execute before consumers outside that graph.
  ///
  /// Attach the result to an otherwise unchanged throughput dispatch request:
  /// @code
  /// auto routing = runtime.prepare({topkIdx, numTokens, maxTokensPerRank, numBlocks, stream});
  /// dispatchRequest.prepareHandle = routing;
  /// auto dispatched = runtime.dispatch(DispatchRequest{dispatchRequest});
  /// @endcode
  /// @param request Routing IDs, token counts, dispatch grid size, and CUDA stream.
  /// @return A reusable, non-owning preparation handle with device receive counts.
  /// @throws EPException For invalid inputs or an unsupported mode.
  PrepareHandle prepare(const PrepareRequest& request);

  /// Dispatch tokens using the configured runtime mode.
  ///
  /// @p request must contain the request type matching mode(): a
  /// LatencyDispatchRequest for LATENCY or a ThroughputDispatchRequest for
  /// THROUGHPUT. Output buffers remain owned by the caller unless obtained
  /// through a runtime buffer accessor. Hidden size, expert count, top-k count,
  /// and output layout come from the runtime. Token count and active capacity
  /// may vary between dispatches, with
  /// 0 <= numTokens <= maxTokensPerRank <= the runtime's capacity.
  /// Throughput requests may reuse routing through prepareHandle. An empty
  /// handle requests automatic GPU preparation; a non-empty handle skips count
  /// recomputation. Both paths support CUDA graph capture. Keep routing
  /// unchanged when replaying a graph that does not recompute preparation.
  /// @param request Dispatch inputs, outputs, and CUDA stream.
  /// @return A non-owning handle identifying this dispatch. A successful new
  /// dispatch or throughput preparation invalidates prior dispatch handles;
  /// a rejected request does not.
  /// @throws EPException If @p request is invalid or does not match mode().
  DispatchHandle dispatch(const DispatchRequest& request);

  /// Combine expert outputs using the configured runtime mode.
  ///
  /// The request's handle supplies routing metadata, token count, active
  /// capacity, and epoch. Fixed dimensions, layout, and algorithm come from the
  /// runtime. Handle ownership and freshness are checked when enqueuing or
  /// capturing combine, not when replaying a graph. Device metadata is not
  /// validated on the host.
  /// @param request Expert inputs, outputs, dispatch handle, block count, and CUDA stream.
  /// @throws EPException If @p request is invalid or its handle is empty,
  /// expired, stale, or belongs to another runtime.
  void combine(const CombineRequest& request);

 private:
  void requireMode(MoEMode expected) const;
  DispatchHandle launchLatencyDispatch(const LatencyDispatchRequest& request);
  DispatchHandle launchThroughputDispatch(const ThroughputDispatchRequest& request);
  void launchLatencyCombine(const LatencyCombineRequest& request);
  void launchThroughputCombine(const ThroughputCombineRequest& request);

  std::shared_ptr<mscclpp::Bootstrap> bootstrap_;
  MoEMode mode_;
  int rank_;
  int numRanks_;
  int numNvlRanks_;
  int numRanksPerIpcDomain_;
  bool available_ = false;
  std::shared_ptr<LatencyRuntimeContext> latencyContext_;
  std::shared_ptr<ThroughputRuntimeContext> throughputContext_;
};

/// Create the unified MoE runtime selected by @p mode.
std::shared_ptr<MoERuntime> createMoERuntime(mscclpp::Communicator& communicator, MoEMode mode, int maxTokensPerRank,
                                             int hidden, int numExperts, int numTopk,
                                             DispatchLayout outputLayout = DispatchLayout::EXPERT_MAJOR,
                                             CombineMode combineMode = CombineMode::RANK_LOCAL_REDUCE);

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EXT_EP_MOE_RUNTIME_HPP_
