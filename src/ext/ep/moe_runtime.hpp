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

namespace mscclpp {
namespace ep {

class MoERuntime {
 public:
  MoERuntime(mscclpp::Communicator& communicator, int maxTokensPerRank, int hidden, int numExperts, int numTopk);
  ~MoERuntime() noexcept(false);

  bool isAvailable() const;
  bool isInternodeAvailable() const;

  void dispatch(void* output, float* outputScales, int* outputSrcInfo, int64_t* outputLayout, int* outputCount,
                const void* input, const int64_t* topkIdx, const float* topkWeights, int numTokens, int hidden,
                int numTopk, int maxTokensPerRank, int numExperts, low_latency::DispatchDataType dispatchDataType,
                int numBlocks, cudaStream_t stream);

  void combine(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights, const int* srcInfo,
               const int64_t* layoutRange, int numTokens, int hidden, int numTopk, int maxTokensPerRank, int numExperts,
               low_latency::DispatchDataType dispatchDataType, low_latency::CombineMode mode, int numBlocks,
               cudaStream_t stream);

 private:
  int rank_;
  int numRanks_;
  int numNvlRanks_;
  int numRanksPerIpcDomain_;
  int deviceId_;
  int64_t symmetricBufferBytes_;
  bool available_ = false;
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
