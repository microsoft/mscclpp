// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/memory_channel.hpp>
#include <string>
#include <vector>

#include "api.cuh"
#include "config.hpp"
#include "runtime_base.hpp"

namespace mscclpp {
namespace ep {

class MoELowLatencyRuntime : public MoERuntime {
 public:
  MoELowLatencyRuntime(mscclpp::Communicator& communicator, int maxTokensPerRank, int hidden, int numExperts,
                       int numTopk, DispatchLayout outputLayout);
  ~MoELowLatencyRuntime() noexcept(false);

  MoEMode mode() const override { return MoEMode::LOW_LATENCY; }

  void* outputTopkIdsBuffer() const;
  void* outputTopkWeightsBuffer() const;
  void* outputTokensBuffer() const;
  void* expertOutputBuffer() const;

  void dispatch(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx, float* outputTopkWeights,
                int64_t* outputLayout, int* outputCount, const void* input, const int64_t* topkIdx,
                const float* topkWeights, int numTokens, int hidden, int numTopk, int maxTokensPerRank, int numExperts,
                int invalidTokenExpertId, DispatchLayout dispatchLayout, low_latency::DispatchDataType dispatchDataType,
                int numBlocks, cudaStream_t stream);

  void combine(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights, const int* srcInfo,
               const int64_t* layoutRange, int numTokens, int hidden, int numTopk, int maxTokensPerRank, int numExperts,
               DispatchLayout dispatchLayout, low_latency::DispatchDataType dispatchDataType,
               low_latency::CombineMode mode, int numBlocks, cudaStream_t stream);

 private:
  int deviceId_;
  int maxTokensPerRank_;
  int hidden_;
  int numExperts_;
  int numTopk_;
  DispatchLayout outputLayout_;
  int64_t symmetricBufferBytes_;
  size_t workspaceBytes_;
  uint32_t dispatchEpoch_ = 0;
  void* symmetricBuffer_ = nullptr;
  void* workspace_ = nullptr;
  low_latency::CommContext commContext_{};

  mscclpp::Communicator* communicator_ = nullptr;

  std::vector<void*> peerMappedBufferBases_;
  std::vector<mscclpp::RegisteredMemory> peerBufferMemories_;
  void** peerMappedBufferBasesGpu_ = nullptr;
  std::vector<mscclpp::BaseMemoryChannel> baseMemoryChannels_;
  std::shared_ptr<mscclpp::BaseMemoryChannelDeviceHandle> baseMemoryChannelHandles_;

  void setup();
};

}  // namespace ep
}  // namespace mscclpp
