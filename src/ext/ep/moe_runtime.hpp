// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/port_channel.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "config.hpp"
#include "kernels/api.cuh"

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

  void dispatch(void* output, float* outputScales, int* outputSrcInfo, int64_t* outputLayout, int* outputCount,
                const void* input, const int64_t* topkIdx, int numTokens, int hidden, int numTopk,
                int numMaxDispatchTokensPerRank, int numExperts, bool useFp8, DispatchLayout dispatchLayout,
                cudaStream_t stream);

  void combine(void* output, const void* input, const float* inputScales, const int64_t* topkIdx,
               const float* topkWeights, const int* srcInfo, const int64_t* layoutRange, int numTokens, int hidden,
               int numTopk, int numMaxDispatchTokensPerRank, int numExperts, bool requiresDequantization,
               cudaStream_t stream);

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
  int numProxyServices_ = 1;
  int llRanksPerIpcDomain_ = 0;
  bool llIpcReady_ = false;

  void* rdmaBufferPtr_ = nullptr;
  void* workspace_ = nullptr;
  cudaStream_t commStream_ = nullptr;

  mscclpp::Communicator* communicator_ = nullptr;
  std::vector<std::shared_ptr<mscclpp::ProxyService>> proxyServices_;
  std::vector<mscclpp::PortChannel> portChannels_;
  std::shared_ptr<mscclpp::PortChannelDeviceHandle> portChannelHandlesDevicePtr_;

  std::vector<void*> peerRdmaBases_;
  void** peerRdmaBasesGpu_ = nullptr;
  std::vector<mscclpp::MemoryChannel> llMemoryChannels_;
  std::shared_ptr<mscclpp::BaseMemoryChannelDeviceHandle> llMemoryChannelHandlesDevicePtr_;

  void setup();
};

}  // namespace ep
}  // namespace mscclpp
