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
/// selected mode. LATENCY uses fixed-capacity expert-major or rank-major
/// layouts. THROUGHPUT uses a dynamically sized token-major receive pool.
/// Operations are asynchronous with respect to the host and execute on the
/// CUDA stream supplied by each request.
class MoERuntime {
 public:
  /// Construct a runtime for the selected mode and topology.
  ///
  /// Only resources required by @p mode are allocated.
  /// @param communicator Initialized MSCCL++ communicator.
  /// @param mode Runtime algorithm family.
  /// @param maxTokensPerRank Fixed latency-mode token capacity.
  /// @param hidden Hidden dimension for latency-mode buffers.
  /// @param numExperts Global expert count.
  /// @param numTopk Number of routed experts per token.
  /// @param maxHiddenBytes Maximum throughput-mode bytes per token row.
  /// @param numBlocks Communication block budget.
  /// @param outputLayout Latency-mode dispatch output layout.
  MoERuntime(mscclpp::Communicator& communicator, MoEMode mode, int maxTokensPerRank, int hidden, int numExperts,
             int numTopk, int64_t maxHiddenBytes, int numBlocks,
             DispatchLayout outputLayout = DispatchLayout::EXPERT_MAJOR);
  ~MoERuntime() noexcept(false);

  MoERuntime(const MoERuntime&) = delete;
  MoERuntime& operator=(const MoERuntime&) = delete;

  /// Return the configured runtime mode.
  MoEMode mode() const { return mode_; }
  /// Return whether the selected mode supports the detected topology.
  bool isAvailable() const { return available_; }
  /// Return whether the runtime is available across more than one node.
  bool isInternodeAvailable() const { return available_ && numRanks_ > numNvlRanks_; }

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
  /// @p request must contain the request type matching mode(): a
  /// LatencyDispatchRequest for LATENCY or a ThroughputDispatchRequest for
  /// THROUGHPUT. Output buffers remain owned by the caller unless obtained
  /// through a runtime buffer accessor.
  /// @param request Dispatch inputs, outputs, dimensions, and CUDA stream.
  /// @throws std::invalid_argument If the request type does not match mode().
  void dispatch(const DispatchRequest& request);

  /// Combine expert outputs using the configured runtime mode.
  ///
  /// A combine request must follow its matching dispatch so the runtime can
  /// reuse routing metadata and synchronization epochs. @p request must contain
  /// a LatencyCombineRequest for LATENCY or a ThroughputCombineRequest for
  /// THROUGHPUT.
  /// @param request Combine inputs, outputs, dimensions, and CUDA stream.
  /// @throws std::invalid_argument If the request type does not match mode().
  void combine(const CombineRequest& request);

  /// Build throughput-mode token routing metadata.
  ///
  /// Computes per-rank counts, per-expert counts, and token-to-rank membership
  /// on @p stream without moving token payloads.
  void tokenMajorPrepare(int* numTokensPerRank, int* numTokensPerExpert, bool* isTokenInRank, const int64_t* topkIdx,
                         int numTokens, int numTopk, int numExperts, cudaStream_t stream);
  /// Return the throughput-mode communication channel count.
  int tokenMajorNumChannels(int xElementSize) const;
  /// Resolve the runtime-owned throughput receive buffer.
  void* tokenMajorResolveRecvBuffer(int numTokens, int numRecvTokens, int hidden, int xElementSize) const;
  /// Exchange throughput routing counts and return the receive-token count.
  ///
  /// This host-synchronizing metadata phase must precede throughput dispatch
  /// when cached routing metadata is unavailable.
  int tokenMajorNotify(int* rankPrefixMatrix, int* channelPrefixMatrix, int* numRecvTokensPerExpert,
                       const int* numTokensPerRank, const int* numTokensPerExpert, const bool* isTokenInRank,
                       int numTokens, int numExperts, int xElementSize, int expertAlignment, cudaStream_t stream);

 private:
  void requireMode(MoEMode expected) const;
  void launchLatencyDispatch(const LatencyDispatchRequest& request);
  void launchThroughputDispatch(const ThroughputDispatchRequest& request);
  void launchLatencyCombine(const LatencyCombineRequest& request);
  void launchThroughputCombine(const ThroughputCombineRequest& request);

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
                                             int hidden, int numExperts, int numTopk, int64_t maxHiddenBytes,
                                             int numBlocks, DispatchLayout outputLayout = DispatchLayout::EXPERT_MAJOR);

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EXT_EP_MOE_RUNTIME_HPP_
