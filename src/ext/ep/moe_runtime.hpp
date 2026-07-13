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
  MoERuntime(mscclpp::Communicator& communicator, int64_t numNvlBytes, int64_t numRdmaBytes, MoEMode mode);
  ~MoERuntime() noexcept(false);

  bool isAvailable() const;
  bool isInternodeAvailable() const;
  int getNumRdmaRanks() const;
  int getRdmaRank() const;
  int getRootRdmaRank(bool global) const;
  int getLocalDeviceId() const;
  std::string getLocalIpcHandle() const;

  void dispatch(void* output, int* outputSrcInfo, int64_t* outputLayout, int* outputCount, const void* input,
                const int64_t* topkIdx, const float* topkWeights, int numTokens, int hidden, int numTopk,
                int maxTokensPerRank, int numExperts, int numBlocks, cudaStream_t stream);

  void combine(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights, const int* srcInfo,
               const int64_t* layoutRange, int numTokens, int hidden, int numTopk, int maxTokensPerRank, int numExperts,
               low_latency::CombineMode mode, int numBlocks, cudaStream_t stream);

 private:
  int lowLatencyBufferIdx_ = 0;
  int rank_;
  int rdmaRank_;
  int nvlRank_;
  int numRanks_;
  int numRdmaRanks_;
  int numNvlRanks_;
  int deviceId_;
  int64_t numNvlBytes_;
  int64_t numRdmaBytes_;
  MoEMode mode_;
  bool available_ = false;
  void* rdmaBufferPtr_ = nullptr;
  void* workspace_ = nullptr;
  low_latency::CommContext commContext_{};

  mscclpp::Communicator* communicator_ = nullptr;

  std::vector<void*> peerRdmaBases_;
  std::vector<mscclpp::RegisteredMemory> peerRdmaMemories_;
  void** peerRdmaBasesGpu_ = nullptr;
  std::vector<mscclpp::BaseMemoryChannel> baseMemoryChannels_;
  std::shared_ptr<mscclpp::BaseMemoryChannelDeviceHandle> baseMemoryChannelHandles_;

  void setup();
};

}  // namespace ep
}  // namespace mscclpp
