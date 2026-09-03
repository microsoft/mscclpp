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
#include <vector>

#include "config.hpp"
#include "device_context.hpp"

namespace mscclpp {
namespace ep {

// Mode-specific contexts owned by MoERuntime.
struct LatencyContext {
  LatencyContext(mscclpp::Communicator& communicator, int rank, int numRanks, int numNvlRanks, int numRanksPerIpcDomain,
                 int maxTokensPerRank, int hidden, int numExperts, int numTopk, DispatchLayout outputLayout,
                 CombineMode combineMode);
  ~LatencyContext() noexcept(false);

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

// Reserve the unified runtime object layout for the follow-up throughput implementation.
struct ThroughputContext {};

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_MOE_RUNTIME_CONTEXT_HPP_
