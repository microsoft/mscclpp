// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP)
// branch `chhwang/dev-atomic-add-cleanup`. Licensed under the MIT License.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/memory_channel.hpp>
#include <vector>

#include "high-throughput/config.cuh"
#include "runtime_base.hpp"

namespace mscclpp {
namespace ep {

class MoEHighThroughputRuntime : public MoERuntime {
 public:
  MoEHighThroughputRuntime(mscclpp::Communicator& communicator, int64_t maxHiddenBytes,
                           const high_throughput::Config& config);
  ~MoEHighThroughputRuntime() noexcept(false);

  MoEMode mode() const override { return MoEMode::HIGH_THROUGHPUT; }

  /// Count tokens per rank and per expert, and record token-to-rank membership.
  /// This is routing metadata for dispatch, unrelated to `DispatchLayout`.
  void computeDispatchCounts(int* numTokensPerRank, int* numTokensPerExpert, bool* isTokenInRank,
                             const int64_t* topkIdx, int numTokens, int numTopk, int numExperts, cudaStream_t stream);

  int getDispatchNumChannels(int xElementSize) const;

  void* resolveRecvXBuffer(int numTokens, int numRecvTokens, int hidden, int xElementSize) const;

  /// Exchange dispatch counts and wait for mapped receive counters so the caller
  /// can expose an exact-size receive view before launching payload dispatch.
  int notifyDispatch(int* rankPrefixMatrix, int* channelPrefixMatrix, int* numRecvTokensPerExpert,
                     const int* numTokensPerRank, const int* numTokensPerExpert, const bool* isTokenInRank,
                     int numTokens, int numExperts, int xElementSize, int expertAlignment, cudaStream_t stream);

  void dispatch(void* recvX, float* recvXScales, int64_t* recvTopkIdx, float* recvTopkWeights, int* sendHead,
                const void* x, const float* xScales, const int64_t* topkIdx, const float* topkWeights,
                const bool* isTokenInRank, const int* rankPrefixMatrix, const int* channelPrefixMatrix, int numTokens,
                int hidden, int numTopk, int numScales, int numExperts, int xElementSize, int numRecvTokens,
                bool cachedMode, cudaStream_t stream);

  void combine(void* combinedX, float* combinedTopkWeights, const void* x, const float* topkWeights,
               const int* sendHead, int numInputTokens, int numOutputTokens, int hidden, int numTopk, int xElementSize,
               cudaStream_t stream);

 private:
  void setup(mscclpp::Communicator& communicator);
  int dispatchBlockCount(int xElementSize) const;
  bool canUseDirectRecvPool(int numTokens, int numRecvTokens, int hidden, int xElementSize) const;

  int64_t maxHiddenBytes_;
  size_t controlBufferBytes_ = 0;
  size_t symmetricBufferBytes_ = 0;
  size_t recvPoolBytes_ = 0;
  bool physicalControlBuffer_ = false;
  bool dispatchReady_ = false;
  bool dispatchMetadataReady_ = false;
  bool collectiveDirectReady_ = false;
  high_throughput::Config config_;

  void* symmetricBuffer_ = nullptr;
  void* recvPool_ = nullptr;
  std::vector<void*> bufferPtrs_;
  std::vector<void*> recvPoolPtrs_;
  std::vector<mscclpp::BaseMemoryChannel> barrierChannels_;
  std::vector<mscclpp::RegisteredMemory> peerMemories_;
  std::vector<mscclpp::RegisteredMemory> recvPoolMemories_;
  void** bufferPtrsGpu_ = nullptr;
  void** recvPoolPtrsGpu_ = nullptr;
  std::shared_ptr<mscclpp::BaseMemoryChannelDeviceHandle> barrierChannelHandles_;
  int* combineRecvIdxGpu_ = nullptr;
  const float* recvTopkWeights_ = nullptr;

  volatile int* moeRecvCounter_ = nullptr;
  int* moeRecvCounterMapped_ = nullptr;
  volatile int* moeRecvExpertCounter_ = nullptr;
  int* moeRecvExpertCounterMapped_ = nullptr;
};

}  // namespace ep
}  // namespace mscclpp
