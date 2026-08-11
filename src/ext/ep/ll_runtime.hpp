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
// GPU-initiated networking host service (defined in the vendored GPUNetIO
// module). Forward-declared so this header stays free of the DOCA dependency;
// the LL runtime only holds a unique_ptr whose destructor is out-of-line.
class GpuNetIoService;

namespace ep {

class MoELowLatencyRuntime : public MoERuntime {
 public:
  MoELowLatencyRuntime(mscclpp::Communicator& communicator, int maxTokensPerRank, int hidden, int numExperts,
                       int numTopk);
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
  int64_t symmetricBufferBytes_;
  size_t workspaceBytes_;
  void* symmetricBuffer_ = nullptr;
  void* workspace_ = nullptr;
  low_latency::CommContext commContext_{};

  mscclpp::Communicator* communicator_ = nullptr;

  std::vector<void*> peerMappedBufferBases_;
  std::vector<mscclpp::RegisteredMemory> peerBufferMemories_;
  void** peerMappedBufferBasesGpu_ = nullptr;
  std::vector<mscclpp::BaseMemoryChannel> baseMemoryChannels_;
  std::shared_ptr<mscclpp::BaseMemoryChannelDeviceHandle> baseMemoryChannelHandles_;

  // GPU-initiated networking service for reaching peers outside this rank's
  // NVLink/IPC domain. Non-null only when the GPUNetIO backend is compiled in
  // and deliberately enabled; its device context is published in commContext_.
  // shared_ptr so this header need not see the complete type for destruction.
  std::shared_ptr<mscclpp::GpuNetIoService> gpuNetIoService_;

  void setup();
};

}  // namespace ep
}  // namespace mscclpp
