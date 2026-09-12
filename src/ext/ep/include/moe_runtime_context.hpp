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

namespace mscclpp {
namespace ep {

struct Workload;

struct PrepareHandle::Impl {
  Impl(std::weak_ptr<ThroughputRuntimeContext> owner, uint64_t epoch, const PrepareRequest& request)
      : owner_(std::move(owner)),
        epoch_(epoch),
        topkIdx_(request.topkIdx),
        numTokens_(request.numTokens),
        maxTokensPerRank_(request.maxTokensPerRank),
        numBlocks_(request.numBlocks) {}

  std::weak_ptr<ThroughputRuntimeContext> owner_;
  uint64_t epoch_;
  const int64_t* topkIdx_;
  int numTokens_;
  int maxTokensPerRank_;
  int numBlocks_;
};

struct DispatchHandle::Impl {
  struct LatencyMetadata {
    const int64_t* topkIdx_;
    const float* topkWeights_;
    const int* srcInfo_;
    const int64_t* layoutRange_;
  };

  Impl(std::weak_ptr<void> owner, uint32_t epoch, const LatencyDispatchRequest& request)
      : owner_(std::move(owner)),
        epoch_(epoch),
        numTokens_(request.numTokens),
        maxTokensPerRank_(request.maxTokensPerRank),
        dispatchDataType_(request.dispatchDataType),
        metadata_(
            LatencyMetadata{request.topkIdx, request.topkWeights, request.outputSrcInfo, request.outputLayoutRange}) {}

  Impl(std::weak_ptr<void> owner, uint32_t epoch, const ThroughputDispatchRequest& request)
      : owner_(std::move(owner)),
        epoch_(epoch),
        numTokens_(request.numTokens),
        maxTokensPerRank_(request.maxTokensPerRank),
        dispatchDataType_(request.dispatchDataType),
        metadata_(std::monostate{}) {}

  std::weak_ptr<void> owner_;
  uint32_t epoch_;
  int numTokens_;
  int maxTokensPerRank_;
  DispatchDataType dispatchDataType_;
  // Latency borrows caller metadata; throughput's inverse routing lives in its workspace.
  std::variant<LatencyMetadata, std::monostate> metadata_;
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
  ThroughputRuntimeContext(mscclpp::Communicator& communicator, int rank, int numRanks, int numRanksPerIpcDomain,
                           int maxTokensPerRank, int hidden, int numExperts, int numTopk, DispatchLayout outputLayout);
  ~ThroughputRuntimeContext() noexcept(false);

 private:
  friend class MoERuntime;

  void initialize();
  bool fitsReceiveBuffer(int maxTokensPerRank) const;
  ThroughputStorageLayout storageLayout() const;
  void validatePrepareRequest(const PrepareRequest& request) const;
  Workload makeWorkload(int numTokens, int maxTokensPerRank, DispatchDataType dataType = DispatchDataType::BF16) const;

  int rank_;
  int numRanks_;
  int numRanksPerIpcDomain_;
  bool available_ = false;
  std::shared_ptr<mscclpp::Bootstrap> bootstrap_;
  int maxTokensPerRank_;
  int hidden_;
  int numExperts_;
  int numTopk_;
  DispatchLayout outputLayout_;
  size_t symmetricBufferBytes_ = 0;
  size_t workspaceBytes_ = 0;
  mscclpp::Communicator& communicator_;
  uint32_t epoch_ = 0;
  uint64_t prepareEpoch_ = 0;
  cudaEvent_t prepareEvent_ = nullptr;
  void* symmetricBuffer_ = nullptr;
  void* workspace_ = nullptr;
  std::vector<void*> peerMappedBufferBases_;
  std::vector<mscclpp::RegisteredMemory> peerBufferMemories_;
  void** peerMappedBufferBasesGpu_ = nullptr;
  std::vector<mscclpp::BaseMemoryChannel> baseMemoryChannels_;
  std::shared_ptr<mscclpp::BaseMemoryChannelDeviceHandle> baseMemoryChannelHandles_;
  DeviceContext deviceContext_{};
};

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_MOE_RUNTIME_CONTEXT_HPP_
