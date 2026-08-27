// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_DEVICE_CONTEXT_HPP_
#define MSCCLPP_EP_DEVICE_CONTEXT_HPP_

#include <mscclpp/ext/ep/types.hpp>
#include <mscclpp/memory_channel_device.hpp>

namespace mscclpp {
namespace ep {

/// Persistent device resources shared by every dispatch and combine algorithm.
struct DeviceContext {
  /// Local base used to translate local buffer addresses into peer mappings.
  void* localBufferBase_;
  /// Peer-mapped control or symmetric-buffer bases.
  void* const* peerBufferBases_;
  /// Optional peer-mapped payload-pool bases.
  void* const* peerPayloadBases_;
  /// Peer synchronization channels.
  mscclpp::BaseMemoryChannelDeviceHandle* channels_;
  /// Optional algorithm workspace.
  void* workspace_;
  /// Optional token/rank receive indices used by throughput combine.
  int* combineRecvIdx_;
  /// Optional mapped receive count used by throughput preparation.
  int* mappedRecvCounter_;
  /// Optional mapped per-expert receive counts used by throughput preparation.
  int* mappedRecvExpertCounters_;
  /// Maximum dynamic shared memory available to one block.
  int maxSharedMemoryPerBlock_;
  /// Number of SMs on the device.
  int numSms_;
  /// CUDA device ID.
  int deviceId_;
  /// Local rank.
  int rank_;
  /// Number of ranks.
  int numRanks_;
  /// EP x ETP rank grid. etpSize == 1 reproduces plain expert parallelism.
  MoETopology topology_;
  /// Placement of the ETP partial-output reduction.
  EtpReduceMode etpReduceMode_ = EtpReduceMode::GROUP_REDUCE_SCATTER;
  /// Dispatch replication strategy across an EP group.
  EtpDispatchMode etpDispatchMode_ = EtpDispatchMode::LEADER_SINGLE_SEND;
  /// Symmetric staging buffer for the ETP group reduce-scatter; null unless
  /// etpSize > 1 with EtpReduceMode::GROUP_REDUCE_SCATTER.
  void* etpReduceBuffer_ = nullptr;
  /// Persistent device copy used by kernel launches. Host launch code only.
  DeviceContext* devicePtr_ = nullptr;
};

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_DEVICE_CONTEXT_HPP_
