// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

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
  /// Optional token/rank receive indices used by overlap combine.
  int* combineRecvIdx_;
  /// Optional mapped receive count used by overlap preparation.
  int* mappedRecvCounter_;
  /// Optional mapped per-expert receive counts used by overlap preparation.
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
  /// Persistent device copy used by kernel launches. Host launch code only.
  DeviceContext* devicePtr_ = nullptr;
};

}  // namespace ep
}  // namespace mscclpp
