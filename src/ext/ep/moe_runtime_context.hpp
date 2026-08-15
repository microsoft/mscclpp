// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_MOE_RUNTIME_CONTEXT_HPP_
#define MSCCLPP_EP_MOE_RUNTIME_CONTEXT_HPP_

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/memory_channel.hpp>
#include <utility>
#include <variant>
#include <vector>

#include "common/recv_pool.cuh"
#include "config.hpp"
#include "include/api.cuh"
#include "include/device_context.cuh"

namespace mscclpp {
namespace ep {
class MoERuntime;

namespace detail {

struct LatencyDispatchRequest {
  void* output_;
  void* outputScales_;
  int* outputSrcInfo_;
  int* outputTopkIdx_;
  float* outputTopkWeights_;
  int64_t* outputLayoutRange_;
  int* outputCount_;
  const void* input_;
  const int64_t* topkIdx_;
  const float* topkWeights_;
  int numTokens_;
  int hidden_;
  int numTopk_;
  int maxTokensPerRank_;
  int numExperts_;
  int invalidTokenExpertId_;
  DispatchLayout dispatchLayout_;
  DispatchDataType dispatchDataType_;
  int numBlocks_;
  cudaStream_t stream_;
};

struct ThroughputDispatchRequest {
  void* recvX_;
  float* recvXScales_;
  int64_t* recvTopkIdx_;
  float* recvTopkWeights_;
  int* sendHead_;
  const void* input_;
  const float* inputScales_;
  const int64_t* topkIdx_;
  const float* topkWeights_;
  const bool* isTokenInRank_;
  const int* rankPrefixMatrix_;
  const int* channelPrefixMatrix_;
  int numTokens_;
  int hidden_;
  int numTopk_;
  int numScales_;
  int numExperts_;
  int inputElementSize_;
  int numRecvTokens_;
  bool cachedMode_;
  cudaStream_t stream_;
};

struct DispatchRequest {
  explicit DispatchRequest(LatencyDispatchRequest request) : value_(std::move(request)) {}
  explicit DispatchRequest(ThroughputDispatchRequest request) : value_(std::move(request)) {}

 private:
  friend class ::mscclpp::ep::MoERuntime;
  std::variant<LatencyDispatchRequest, ThroughputDispatchRequest> value_;
};

struct LatencyCombineRequest {
  void* output_;
  const void* input_;
  const int64_t* topkIdx_;
  const float* topkWeights_;
  const int* srcInfo_;
  const int64_t* layoutRange_;
  int numTokens_;
  int hidden_;
  int numTopk_;
  int maxTokensPerRank_;
  int numExperts_;
  DispatchLayout dispatchLayout_;
  DispatchDataType dispatchDataType_;
  CombineMode combineMode_;
  int numBlocks_;
  cudaStream_t stream_;
};

struct ThroughputCombineRequest {
  void* output_;
  float* outputTopkWeights_;
  const void* input_;
  const float* topkWeights_;
  const int* sendHead_;
  int numInputTokens_;
  int numOutputTokens_;
  int hidden_;
  int numTopk_;
  int inputElementSize_;
  cudaStream_t stream_;
};

struct CombineRequest {
  explicit CombineRequest(LatencyCombineRequest request) : value_(std::move(request)) {}
  explicit CombineRequest(ThroughputCombineRequest request) : value_(std::move(request)) {}

 private:
  friend class ::mscclpp::ep::MoERuntime;
  std::variant<LatencyCombineRequest, ThroughputCombineRequest> value_;
};

// Mode-specific contexts owned by MoERuntime.
struct LatencyContext {
  LatencyContext(mscclpp::Communicator& communicator, int rank, int numRanks, int numNvlRanks, int numRanksPerIpcDomain,
                 int maxTokensPerRank, int hidden, int numExperts, int numTopk, DispatchLayout outputLayout);
  ~LatencyContext() noexcept(false);

 private:
  friend class ::mscclpp::ep::MoERuntime;

  void setup();

  int rank_;
  int numRanks_;
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
  uint32_t epoch_ = 0;
  void* symmetricBuffer_ = nullptr;
  void* workspace_ = nullptr;
  DeviceContext deviceContext_{};
  mscclpp::Communicator* communicator_ = nullptr;
  std::vector<void*> peerMappedBufferBases_;
  std::vector<mscclpp::RegisteredMemory> peerBufferMemories_;
  void** peerMappedBufferBasesGpu_ = nullptr;
  std::vector<mscclpp::BaseMemoryChannel> baseMemoryChannels_;
  std::shared_ptr<mscclpp::BaseMemoryChannelDeviceHandle> baseMemoryChannelHandles_;
};

struct ThroughputContext {
  ThroughputContext(mscclpp::Communicator& communicator, int rank, int numRanks, int numNvlRanks,
                    int numRanksPerIpcDomain, int64_t maxHiddenBytes, const RecvPoolConfig& config);
  ~ThroughputContext() noexcept(false);

 private:
  friend class ::mscclpp::ep::MoERuntime;

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
  DeviceContext deviceContext_{};
};

}  // namespace detail
}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_MOE_RUNTIME_CONTEXT_HPP_
