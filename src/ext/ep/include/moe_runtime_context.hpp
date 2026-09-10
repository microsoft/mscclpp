// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_MOE_RUNTIME_CONTEXT_HPP_
#define MSCCLPP_EP_MOE_RUNTIME_CONTEXT_HPP_

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/ext/ep/moe_runtime.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/memory_channel.hpp>
#include <utility>
#include <variant>
#include <vector>

#include "config.hpp"
#include "device_context.hpp"
#include "recv_pool.hpp"

namespace mscclpp {
namespace ep {

struct DispatchHandle::Impl {
  struct LatencyMetadata {
    const int64_t* topkIdx_;
    const float* topkWeights_;
    const int* srcInfo_;
    const int64_t* layoutRange_;
  };

  struct ThroughputMetadata {
    const int* sendHead_;
    int numRecvTokens_;
  };

  Impl(std::weak_ptr<void> owner, uint32_t epoch, const LatencyDispatchRequest& request)
      : owner_(std::move(owner)),
        epoch_(epoch),
        numTokens_(request.numTokens),
        maxTokensPerRank_(request.maxTokensPerRank),
        dispatchDataType_(request.dispatchDataType),
        metadata_(
            LatencyMetadata{request.topkIdx, request.topkWeights, request.outputSrcInfo, request.outputLayoutRange}) {}

  Impl(std::weak_ptr<void> owner, uint32_t epoch, const ThroughputDispatchRequest& request, const int* sendHead,
       int numRecvTokens)
      : owner_(std::move(owner)),
        epoch_(epoch),
        numTokens_(request.numTokens),
        maxTokensPerRank_(request.maxTokensPerRank),
        dispatchDataType_(request.dispatchDataType),
        metadata_(ThroughputMetadata{sendHead, numRecvTokens}) {}

  std::weak_ptr<void> owner_;
  uint32_t epoch_;
  int numTokens_;
  int maxTokensPerRank_;
  DispatchDataType dispatchDataType_;
  std::variant<LatencyMetadata, ThroughputMetadata> metadata_;
};

// Mode-specific contexts owned by MoERuntime.
struct LatencyRuntimeContext {
  LatencyRuntimeContext(mscclpp::Communicator& communicator, int rank, int numRanks, int numNvlRanks,
                        int numRanksPerIpcDomain, int maxTokensPerRank, int hidden, int numExperts, int numTopk,
                        DispatchLayout outputLayout, CombineMode combineMode);
  ~LatencyRuntimeContext() noexcept(false);

 private:
  friend class MoERuntime;

  void initialize();

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
  CombineMode combineMode_;
  int64_t symmetricBufferBytes_;
  size_t workspaceBytes_;
  uint32_t epoch_ = 0;
  void* symmetricBuffer_ = nullptr;
  void* workspace_ = nullptr;
  DeviceContext deviceContext_{};
  mscclpp::Communicator& communicator_;
  std::vector<void*> peerMappedBufferBases_;
  std::vector<mscclpp::RegisteredMemory> peerBufferMemories_;
  void** peerMappedBufferBasesGpu_ = nullptr;
  std::vector<mscclpp::BaseMemoryChannel> baseMemoryChannels_;
  std::shared_ptr<mscclpp::BaseMemoryChannelDeviceHandle> baseMemoryChannelHandles_;
};

struct ThroughputRuntimeContext {
  ThroughputRuntimeContext(mscclpp::Communicator& communicator, int rank, int numRanks, int numNvlRanks,
                           int numRanksPerIpcDomain, int maxTokensPerRank, int hidden, int numExperts, int numTopk,
                           DispatchLayout outputLayout);
  ~ThroughputRuntimeContext() noexcept(false);

 private:
  friend class MoERuntime;

  void initialize();
  bool canUseDirectRecvPool(int maxTokensPerRank) const;

  int rank_;
  int numRanks_;
  int numNvlRanks_;
  int numRanksPerIpcDomain_;
  bool available_ = false;
  std::shared_ptr<mscclpp::Bootstrap> bootstrap_;
  int maxTokensPerRank_;
  int hidden_;
  int numExperts_;
  int numTopk_;
  int64_t maxHiddenBytes_;
  DispatchLayout outputLayout_;
  size_t controlBufferBytes_ = 0;
  size_t symmetricBufferBytes_ = 0;
  size_t recvPoolBytes_ = 0;
  size_t workspaceBytes_ = 0;
  bool physicalControlBuffer_ = false;
  mscclpp::Communicator& communicator_;
  uint32_t epoch_ = 0;
  void* symmetricBuffer_ = nullptr;
  void* recvPool_ = nullptr;
  void* workspace_ = nullptr;
  std::vector<void*> bufferPtrs_;
  std::vector<void*> recvPoolPtrs_;
  std::vector<mscclpp::BaseMemoryChannel> barrierChannels_;
  std::vector<mscclpp::RegisteredMemory> peerMemories_;
  std::vector<mscclpp::RegisteredMemory> recvPoolMemories_;
  void** bufferPtrsGpu_ = nullptr;
  void** recvPoolPtrsGpu_ = nullptr;
  std::shared_ptr<mscclpp::BaseMemoryChannelDeviceHandle> barrierChannelHandles_;
  int* combineRecvIdxGpu_ = nullptr;
  volatile int* moeRecvCounter_ = nullptr;
  int* moeRecvCounterMapped_ = nullptr;
  volatile int* moeRecvExpertCounter_ = nullptr;
  int* moeRecvExpertCounterMapped_ = nullptr;
  DeviceContext deviceContext_{};
};

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_MOE_RUNTIME_CONTEXT_HPP_
