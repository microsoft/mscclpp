// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP)
// branch `chhwang/dev-atomic-add-cleanup`. Licensed under the MIT License.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <mscclpp/core.hpp>
#include <vector>

#include "ht/config.hpp"

namespace mscclpp {
namespace ep {

class MoEHighThroughputRuntime {
 public:
  MoEHighThroughputRuntime(mscclpp::Communicator& communicator, int64_t maxHiddenBytes, const Config& config);
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

  void dispatch(void* recvX, float* recvXScales, int64_t* recvTopkIdx, float* recvTopkWeights, int* recvSrcIdx,
                int* sendHead, int* recvChannelPrefixMatrix, const void* x, const float* xScales,
                const int64_t* topkIdx, const float* topkWeights, const bool* isTokenInRank,
                const int* rankPrefixMatrix, const int* channelPrefixMatrix, int numTokens, int hidden, int numTopk,
                int numScales, int numExperts, int xElementSize, int numRecvTokens, bool cachedMode,
                cudaStream_t stream);

  void combine(void* combinedX, float* combinedTopkWeights, const void* x, const float* topkWeights, const int* srcIdx,
               const int* rankPrefixMatrix, const int* channelPrefixMatrix, const int* sendHead, int numTokens,
               int numRecvTokens, int hidden, int numTopk, int xElementSize, int ringNumChannels, cudaStream_t stream);

 private:
  void setup(mscclpp::Communicator& communicator);
  void moveFifoSlots(int numSlots = 1);
  void computeDispatchChannels(int xElementSize, int& dispatchNumSms, bool& allSender, int& numChannels) const;
  bool canUseDirectRecvPool(int numTokens, int numRecvTokens, int hidden, int xElementSize) const;

  int rank_;
  int numRanks_;
  int numNvlRanks_;
  int numRanksPerIpcDomain_;
  int head_ = 0;
  int64_t maxHiddenBytes_;
  size_t ringBufferBytes_ = 0;
  size_t taskFifoOffset_ = 0;
  size_t symmetricBufferBytes_ = 0;
  size_t recvPoolBytes_ = 0;
  bool available_ = false;
  bool physicalRingBuffer_ = false;
  bool directDispatchReady_ = false;
  Config config_;

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

  volatile int* moeRecvCounter_ = nullptr;
  int* moeRecvCounterMapped_ = nullptr;
  volatile int* moeRecvExpertCounter_ = nullptr;
  int* moeRecvExpertCounterMapped_ = nullptr;
};

}  // namespace ep
}  // namespace mscclpp
