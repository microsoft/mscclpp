// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <mscclpp/core.hpp>

#include "api.cuh"

namespace mscclpp {
namespace ep {
namespace detail {
struct FixedBufferResources;
struct RecvPoolResources;
}  // namespace detail

/// Unified host runtime for all expert-parallel dispatch and combine algorithms.
///
/// `MoEMode` remains a compatibility selector for the Python API. The selected
/// mode allocates only its required fixed-buffer or receive-pool resources.
class MoERuntime {
 public:
  MoERuntime(mscclpp::Communicator& communicator, MoEMode mode, int maxTokensPerRank, int hidden, int numExperts,
             int numTopk, int64_t maxHiddenBytes, int numSms,
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

  void latencyDispatch(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                       float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                       const int64_t* topkIdx, const float* topkWeights, int numTokens, int hidden, int numTopk,
                       int maxTokensPerRank, int numExperts, int invalidTokenExpertId, DispatchLayout dispatchLayout,
                       DispatchDataType dispatchDataType, int numBlocks, cudaStream_t stream);

  void latencyCombine(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights,
                      const int* srcInfo, const int64_t* layoutRange, int numTokens, int hidden, int numTopk,
                      int maxTokensPerRank, int numExperts, DispatchLayout dispatchLayout,
                      DispatchDataType dispatchDataType, CombineMode mode, int numBlocks, cudaStream_t stream);

  void tokenMajorPrepare(int* numTokensPerRank, int* numTokensPerExpert, bool* isTokenInRank, const int64_t* topkIdx,
                         int numTokens, int numTopk, int numExperts, cudaStream_t stream);
  int tokenMajorNumChannels(int xElementSize) const;
  void* tokenMajorResolveRecvBuffer(int numTokens, int numRecvTokens, int hidden, int xElementSize) const;
  int tokenMajorNotify(int* rankPrefixMatrix, int* channelPrefixMatrix, int* numRecvTokensPerExpert,
                       const int* numTokensPerRank, const int* numTokensPerExpert, const bool* isTokenInRank,
                       int numTokens, int numExperts, int xElementSize, int expertAlignment, cudaStream_t stream);
  void tokenMajorDispatch(void* recvX, float* recvXScales, int64_t* recvTopkIdx, float* recvTopkWeights, int* sendHead,
                          const void* x, const float* xScales, const int64_t* topkIdx, const float* topkWeights,
                          const bool* isTokenInRank, const int* rankPrefixMatrix, const int* channelPrefixMatrix,
                          int numTokens, int hidden, int numTopk, int numScales, int numExperts, int xElementSize,
                          int numRecvTokens, bool cachedMode, cudaStream_t stream);
  void tokenMajorCombine(void* combinedX, float* combinedTopkWeights, const void* x, const float* topkWeights,
                         const int* sendHead, int numInputTokens, int numOutputTokens, int hidden, int numTopk,
                         int xElementSize, cudaStream_t stream);

 private:
  void requireMode(MoEMode expected) const;

  std::shared_ptr<mscclpp::Bootstrap> bootstrap_;
  MoEMode mode_;
  int rank_;
  int numRanks_;
  int numNvlRanks_;
  int numRanksPerIpcDomain_;
  bool available_ = false;

  std::unique_ptr<detail::FixedBufferResources> fixedBuffer_;
  std::unique_ptr<detail::RecvPoolResources> recvPool_;
};

/// Create the unified MoE runtime selected by @p mode.
std::shared_ptr<MoERuntime> createMoERuntime(mscclpp::Communicator& communicator, MoEMode mode, int maxTokensPerRank,
                                             int hidden, int numExperts, int numTopk, int64_t maxHiddenBytes,
                                             int numSms, DispatchLayout outputLayout = DispatchLayout::EXPERT_MAJOR);

}  // namespace ep
}  // namespace mscclpp
