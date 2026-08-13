// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/memory_channel.hpp>
#include <vector>

#include "common/recv_pool.cuh"
#include "config.hpp"
#include "include/api.cuh"
#include "include/device_context.cuh"

namespace mscclpp {
namespace ep {
namespace detail {

struct FixedBufferResources {
  FixedBufferResources(mscclpp::Communicator& communicator, int rank, int numRanks, int numNvlRanks,
                       int numRanksPerIpcDomain, int maxTokensPerRank, int hidden, int numExperts, int numTopk,
                       DispatchLayout outputLayout);
  ~FixedBufferResources() noexcept(false);

  void* outputTopkIdsBuffer() const;
  void* outputTopkWeightsBuffer() const;
  void* dispatchOutputBuffer() const;
  void* expertOutputBuffer() const;

  void dispatch(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx, float* outputTopkWeights,
                int64_t* outputLayout, int* outputCount, const void* input, const int64_t* topkIdx,
                const float* topkWeights, int numTokens, int hidden, int numTopk, int maxTokensPerRank, int numExperts,
                int invalidTokenExpertId, DispatchLayout dispatchLayout, DispatchDataType dispatchDataType,
                int numBlocks, cudaStream_t stream);
  void combine(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights, const int* srcInfo,
               const int64_t* layoutRange, int numTokens, int hidden, int numTopk, int maxTokensPerRank, int numExperts,
               DispatchLayout dispatchLayout, DispatchDataType dispatchDataType, CombineMode mode, int numBlocks,
               cudaStream_t stream);

  bool available() const { return available_; }

 private:
  void setup();

  int rank_;
  int numRanks_;
  int numNvlRanks_;
  int numRanksPerIpcDomain_;
  bool available_ = false;
  int deviceId_;
  int maxTokensPerRank_;
  int hidden_;
  int numExperts_;
  int numTopk_;
  DispatchLayout outputLayout_;
  int64_t symmetricBufferBytes_;
  size_t workspaceBytes_;
  void* symmetricBuffer_ = nullptr;
  void* workspace_ = nullptr;
  DeviceContext contextHost_{};
  DeviceContext* contextDevice_ = nullptr;
  mscclpp::Communicator* communicator_ = nullptr;
  std::vector<void*> peerMappedBufferBases_;
  std::vector<mscclpp::RegisteredMemory> peerBufferMemories_;
  void** peerMappedBufferBasesGpu_ = nullptr;
  std::vector<mscclpp::BaseMemoryChannel> baseMemoryChannels_;
  std::shared_ptr<mscclpp::BaseMemoryChannelDeviceHandle> baseMemoryChannelHandles_;
};

struct RecvPoolResources {
  RecvPoolResources(mscclpp::Communicator& communicator, int rank, int numRanks, int numNvlRanks,
                    int numRanksPerIpcDomain, int64_t maxHiddenBytes, const RecvPoolConfig& config);
  ~RecvPoolResources() noexcept(false);

  void prepare(int* numTokensPerRank, int* numTokensPerExpert, bool* isTokenInRank, const int64_t* topkIdx,
               int numTokens, int numTopk, int numExperts, cudaStream_t stream);
  int numChannels(int xElementSize) const;
  void* resolveRecvBuffer(int numTokens, int numRecvTokens, int hidden, int xElementSize) const;
  int notify(int* rankPrefixMatrix, int* channelPrefixMatrix, int* numRecvTokensPerExpert, const int* numTokensPerRank,
             const int* numTokensPerExpert, const bool* isTokenInRank, int numTokens, int numExperts, int xElementSize,
             int expertAlignment, cudaStream_t stream);
  void dispatch(void* recvX, float* recvXScales, int64_t* recvTopkIdx, float* recvTopkWeights, int* sendHead,
                const void* x, const float* xScales, const int64_t* topkIdx, const float* topkWeights,
                const bool* isTokenInRank, const int* rankPrefixMatrix, const int* channelPrefixMatrix, int numTokens,
                int hidden, int numTopk, int numScales, int numExperts, int xElementSize, int numRecvTokens,
                bool cachedMode, cudaStream_t stream);
  void combine(void* combinedX, float* combinedTopkWeights, const void* x, const float* topkWeights,
               const int* sendHead, int numInputTokens, int numOutputTokens, int hidden, int numTopk, int xElementSize,
               cudaStream_t stream);

  bool available() const { return available_; }

 private:
  void setup(mscclpp::Communicator& communicator);
  int dispatchBlockCount(int xElementSize) const;
  bool canUseDirectRecvPool(int numTokens, int numRecvTokens, int hidden, int xElementSize) const;

  int rank_;
  int numRanks_;
  int numNvlRanks_;
  int numRanksPerIpcDomain_;
  bool available_ = false;
  std::shared_ptr<mscclpp::Bootstrap> bootstrap_;
  int64_t maxHiddenBytes_;
  size_t controlBufferBytes_ = 0;
  size_t symmetricBufferBytes_ = 0;
  size_t recvPoolBytes_ = 0;
  bool physicalControlBuffer_ = false;
  bool dispatchReady_ = false;
  bool dispatchMetadataReady_ = false;
  bool collectiveDirectReady_ = false;
  RecvPoolConfig config_;
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
  DeviceContext contextHost_{};
  DeviceContext* contextDevice_ = nullptr;
};

}  // namespace detail
}  // namespace ep
}  // namespace mscclpp
