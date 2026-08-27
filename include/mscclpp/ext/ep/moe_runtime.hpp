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
/// layouts. THROUGHPUT uses a receive pool exposed as compact token-major or
/// fixed-stride rank-major rows.
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
  /// @param maxHiddenBytes Maximum throughput-mode bytes per token row.
  /// @param numBlocks Total communication block count. LATENCY includes its two
  /// reserved scheduler/control blocks in this value; THROUGHPUT uses all blocks
  /// as communication workers.
  /// @param outputLayout Dispatch output layout.
  /// @param combineMode Latency-mode combine algorithm.
  /// @param etpSize Number of ranks sharing one expert's weights (expert
  /// tensor parallelism). 1 reproduces the plain expert-parallel behavior.
  /// @param etpRankOrder Rank numbering convention of the EP x ETP grid.
  /// @param etpReduceMode Placement of the ETP partial-output reduction.
  /// @param etpDispatchMode Dispatch replication strategy across an EP group.
  MoERuntime(mscclpp::Communicator& communicator, MoEMode mode, int maxTokensPerRank, int hidden, int numExperts,
             int numTopk, int64_t maxHiddenBytes, int numBlocks,
             DispatchLayout outputLayout = DispatchLayout::EXPERT_MAJOR,
             CombineMode combineMode = CombineMode::RANK_LOCAL_REDUCE, int etpSize = 1,
             EtpRankOrder etpRankOrder = EtpRankOrder::EP_MAJOR,
             EtpReduceMode etpReduceMode = EtpReduceMode::GROUP_REDUCE_SCATTER,
             EtpDispatchMode etpDispatchMode = EtpDispatchMode::LEADER_SINGLE_SEND);
  ~MoERuntime() noexcept(false);

  MoERuntime(const MoERuntime&) = delete;
  MoERuntime& operator=(const MoERuntime&) = delete;

  /// Return the configured runtime mode.
  MoEMode mode() const { return mode_; }
  /// Return whether the selected mode supports the detected topology.
  bool isAvailable() const { return available_; }
  /// Return whether the runtime is available across more than one node.
  bool isInternodeAvailable() const { return available_ && numRanks_ > numNvlRanks_; }
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
  /// Return the number of expert-parallel groups.
  int epSize() const { return topology_.epSize; }
  /// Return the number of ranks sharing one expert's weights.
  int etpSize() const { return topology_.etpSize; }
  /// Return this rank's expert-parallel group index.
  int epIndex() const { return topology_.epIndex(rank_); }
  /// Return this rank's tensor-parallel index inside its EP group.
  int tpIndex() const { return topology_.tpIndex(rank_); }
  /// Return the full EP x ETP topology.
  const MoETopology& topology() const { return topology_; }

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

  /// Build token routing metadata.
  ///
  /// Computes per-rank counts, per-expert counts, and token-to-rank membership
  /// on @p stream without moving token payloads.
  void prepare(int* numTokensPerRank, int* numTokensPerExpert, bool* isTokenInRank, const int64_t* topkIdx,
               int numTokens, int numTopk, int numExperts, cudaStream_t stream);
  /// Exchange routing counts and return the receive-token count.
  ///
  /// This host-synchronizing metadata phase must precede throughput dispatch
  /// when cached routing metadata is unavailable.
  int notify(int* rankPrefixMatrix, int* channelPrefixMatrix, int* numRecvTokensPerExpert, const int* numTokensPerRank,
             const int* numTokensPerExpert, const bool* isTokenInRank, int numTokens, int numExperts, int xElementSize,
             int expertAlignment, cudaStream_t stream);

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
  MoETopology topology_;
  bool available_ = false;

  std::unique_ptr<LatencyContext> latencyContext_;
  std::unique_ptr<ThroughputContext> throughputContext_;
};

/// Create the unified MoE runtime selected by @p mode.
std::shared_ptr<MoERuntime> createMoERuntime(mscclpp::Communicator& communicator, MoEMode mode, int maxTokensPerRank,
                                             int hidden, int numExperts, int numTopk, int64_t maxHiddenBytes,
                                             int numBlocks, DispatchLayout outputLayout = DispatchLayout::EXPERT_MAJOR,
                                             CombineMode combineMode = CombineMode::RANK_LOCAL_REDUCE, int etpSize = 1,
                                             EtpRankOrder etpRankOrder = EtpRankOrder::EP_MAJOR,
                                             EtpReduceMode etpReduceMode = EtpReduceMode::GROUP_REDUCE_SCATTER,
                                             EtpDispatchMode etpDispatchMode = EtpDispatchMode::LEADER_SINGLE_SEND);

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EXT_EP_MOE_RUNTIME_HPP_
