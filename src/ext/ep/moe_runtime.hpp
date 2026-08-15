// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_MOE_RUNTIME_HPP_
#define MSCCLPP_EP_MOE_RUNTIME_HPP_

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <mscclpp/core.hpp>

#include "api.cuh"

namespace mscclpp {
namespace ep {
namespace detail {
struct LatencyContext;
struct ThroughputContext;
struct LatencyDispatchRequest;
struct ThroughputDispatchRequest;
struct DispatchRequest;
struct LatencyCombineRequest;
struct ThroughputCombineRequest;
struct CombineRequest;
}  // namespace detail

/// Unified host runtime for all expert-parallel dispatch and combine algorithms.
///
/// The selected mode allocates only its required latency or throughput context.
class MoERuntime {
 public:
  MoERuntime(mscclpp::Communicator& communicator, MoEMode mode, int maxTokensPerRank, int hidden, int numExperts,
             int numTopk, int64_t maxHiddenBytes, int numBlocks,
             DispatchLayout outputLayout = DispatchLayout::EXPERT_MAJOR);
  ~MoERuntime() noexcept(false);

  MoERuntime(const MoERuntime&) = delete;
  MoERuntime& operator=(const MoERuntime&) = delete;

  MoEMode mode() const { return mode_; }
  bool isAvailable() const { return available_; }
  bool isInternodeAvailable() const { return available_ && numRanks_ > numNvlRanks_; }

  int rank() const { return rank_; }
  int numRanks() const { return numRanks_; }
  int numNvlRanks() const { return numNvlRanks_; }
  int numRanksPerIpcDomain() const { return numRanksPerIpcDomain_; }

  void* outputTopkIdsBuffer() const;
  void* outputTopkWeightsBuffer() const;
  void* dispatchOutputBuffer() const;
  void* combineInputBuffer() const;

  void dispatch(const detail::DispatchRequest& request);
  void combine(const detail::CombineRequest& request);

  void tokenMajorPrepare(int* numTokensPerRank, int* numTokensPerExpert, bool* isTokenInRank, const int64_t* topkIdx,
                         int numTokens, int numTopk, int numExperts, cudaStream_t stream);
  int tokenMajorNumChannels(int xElementSize) const;
  void* tokenMajorResolveRecvBuffer(int numTokens, int numRecvTokens, int hidden, int xElementSize) const;
  int tokenMajorNotify(int* rankPrefixMatrix, int* channelPrefixMatrix, int* numRecvTokensPerExpert,
                       const int* numTokensPerRank, const int* numTokensPerExpert, const bool* isTokenInRank,
                       int numTokens, int numExperts, int xElementSize, int expertAlignment, cudaStream_t stream);

 private:
  void requireMode(MoEMode expected) const;
  void launchLatencyDispatch(const detail::LatencyDispatchRequest& request);
  void launchThroughputDispatch(const detail::ThroughputDispatchRequest& request);
  void launchLatencyCombine(const detail::LatencyCombineRequest& request);
  void launchThroughputCombine(const detail::ThroughputCombineRequest& request);

  std::shared_ptr<mscclpp::Bootstrap> bootstrap_;
  MoEMode mode_;
  int rank_;
  int numRanks_;
  int numNvlRanks_;
  int numRanksPerIpcDomain_;
  bool available_ = false;

  std::unique_ptr<detail::LatencyContext> latencyContext_;
  std::unique_ptr<detail::ThroughputContext> throughputContext_;
};

/// Create the unified MoE runtime selected by @p mode.
std::shared_ptr<MoERuntime> createMoERuntime(mscclpp::Communicator& communicator, MoEMode mode, int maxTokensPerRank,
                                             int hidden, int numExperts, int numTopk, int64_t maxHiddenBytes,
                                             int numBlocks, DispatchLayout outputLayout = DispatchLayout::EXPERT_MAJOR);

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_MOE_RUNTIME_HPP_
