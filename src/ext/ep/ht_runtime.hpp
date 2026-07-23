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
#include <vector>

#include "high-throughput/config.cuh"

namespace mscclpp {
namespace ep {

class MoEHighThroughputRuntime {
 public:
  MoEHighThroughputRuntime(mscclpp::Communicator& communicator, int64_t maxHiddenBytes,
                           const high_throughput::Config& config);
  ~MoEHighThroughputRuntime() noexcept(false);

  bool isAvailable() const;
  bool isInternodeAvailable() const;

  void layout(int* numTokensPerRank, int* numTokensPerExpert, bool* isTokenInRank, const int64_t* topkIdx,
              int numTokens, int numTopk, int numExperts, cudaStream_t stream);

  int getDispatchNumChannels(int xElementSize) const;

  void* resolveRecvXBuffer(int numTokens, int numRecvTokens, int hidden, int xElementSize) const;

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
  void moveFifoSlots(int numSlots = 1);
  int dispatchBlockCount(int xElementSize) const;
  bool canUseDirectRecvPool(int numTokens, int numRecvTokens, int hidden, int xElementSize) const;

  std::shared_ptr<mscclpp::Bootstrap> bootstrap_;
  int rank_;
  int numRanks_;
  int numNvlRanks_;
  int numRanksPerIpcDomain_;
  int head_ = 0;
  int64_t maxHiddenBytes_;
  size_t controlBufferBytes_ = 0;
  size_t taskFifoOffset_ = 0;
  size_t symmetricBufferBytes_ = 0;
  size_t recvPoolBytes_ = 0;
  bool available_ = false;
  bool physicalControlBuffer_ = false;
  bool dispatchReady_ = false;
  bool dispatchMetadataReady_ = false;
  bool collectiveDirectReady_ = false;
  high_throughput::Config config_;

  void* symmetricBuffer_ = nullptr;
  void* recvPool_ = nullptr;
  std::vector<void*> bufferPtrs_;
  std::vector<int*> taskFifoPtrs_;
  std::vector<void*> recvPoolPtrs_;
  std::vector<mscclpp::RegisteredMemory> peerMemories_;
  std::vector<mscclpp::RegisteredMemory> recvPoolMemories_;
  void** bufferPtrsGpu_ = nullptr;
  int** taskFifoPtrsGpu_ = nullptr;
  void** recvPoolPtrsGpu_ = nullptr;
  int* combineRecvIdxGpu_ = nullptr;
  const float* recvTopkWeights_ = nullptr;

  volatile int* moeRecvCounter_ = nullptr;
  int* moeRecvCounterMapped_ = nullptr;
  volatile int* moeRecvExpertCounter_ = nullptr;
  int* moeRecvExpertCounterMapped_ = nullptr;
};

}  // namespace ep
}  // namespace mscclpp
