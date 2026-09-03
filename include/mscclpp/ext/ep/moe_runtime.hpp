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
struct LatencyContext;
struct ThroughputContext;
/// Unified host runtime for expert-parallel dispatch and combine.
///
/// One runtime owns the communication buffers and synchronization state for the
/// selected mode. This library initially implements LATENCY with fixed-capacity
/// expert-major or rank-major layouts. The THROUGHPUT API is reserved for a
/// follow-up implementation and is rejected at runtime.
/// Operations are asynchronous with respect to the host and execute on the
/// CUDA stream supplied by each request.
class MoERuntime {
 public:
  /// Construct a runtime for the selected mode and topology.
  ///
  /// Only resources required by @p mode are allocated. Mode-specific
  /// communicator buffers are deferred until initialize().
  /// @param communicator Initialized MSCCL++ communicator.
  /// @param mode Runtime algorithm family.
  /// @param maxTokensPerRank Fixed per-rank token capacity.
  /// @param hidden Hidden dimension for latency-mode buffers.
  /// @param numExperts Global expert count.
  /// @param numTopk Number of routed experts per token.
  /// @param outputLayout Dispatch output layout.
  /// @param combineMode Latency-mode combine algorithm.
  /// @throws std::invalid_argument If @p mode is not MoEMode::LATENCY.
  /// @warning @p communicator must remain alive until initialize() returns.
  MoERuntime(mscclpp::Communicator& communicator, MoEMode mode, int maxTokensPerRank, int hidden, int numExperts,
             int numTopk, DispatchLayout outputLayout = DispatchLayout::EXPERT_MAJOR,
             CombineMode combineMode = CombineMode::RANK_LOCAL_REDUCE);
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
  /// Return the runtime-owned rank-major combine input buffer.
  void* combineInputBuffer() const;

  /// Dispatch tokens using the configured runtime mode.
  ///
  /// ThroughputDispatchRequest is reserved for a follow-up implementation.
  /// Output buffers remain owned by the caller unless obtained through a runtime buffer accessor.
  /// @param request Dispatch inputs, outputs, dimensions, and CUDA stream.
  /// @throws std::invalid_argument If @p request is not a latency request.
  void dispatch(const DispatchRequest& request);

  /// Combine expert outputs using the configured runtime mode.
  ///
  /// A combine request must follow its matching dispatch so the runtime can
  /// reuse routing metadata and synchronization epochs. ThroughputCombineRequest
  /// is reserved for a follow-up implementation.
  /// @param request Combine inputs, outputs, dimensions, and CUDA stream.
  /// @throws std::invalid_argument If @p request is not a latency request.
  void combine(const CombineRequest& request);

 private:
  void requireMode(MoEMode expected) const;
  void launchLatencyDispatch(const LatencyDispatchRequest& request);
  void launchLatencyCombine(const LatencyCombineRequest& request);

  std::shared_ptr<mscclpp::Bootstrap> bootstrap_;
  MoEMode mode_;
  int rank_;
  int numRanks_;
  int numNvlRanks_;
  int numRanksPerIpcDomain_;
  bool available_ = false;
  std::unique_ptr<LatencyContext> latencyContext_;
  std::unique_ptr<ThroughputContext> throughputContext_;
};

/// Create the unified MoE runtime selected by @p mode.
std::shared_ptr<MoERuntime> createMoERuntime(mscclpp::Communicator& communicator, MoEMode mode, int maxTokensPerRank,
                                             int hidden, int numExperts, int numTopk,
                                             DispatchLayout outputLayout = DispatchLayout::EXPERT_MAJOR,
                                             CombineMode combineMode = CombineMode::RANK_LOCAL_REDUCE);

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EXT_EP_MOE_RUNTIME_HPP_
