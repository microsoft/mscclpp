// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_DEVICE_CONTEXT_HPP_
#define MSCCLPP_EP_DEVICE_CONTEXT_HPP_

#include <mscclpp/memory_channel_device.hpp>
#include <mscclpp/port_channel_device.hpp>

namespace mscclpp {
namespace ep {

using PortChannelDeviceHandle = mscclpp::PortChannelDeviceHandleT<mscclpp::PortChannelBackend::Device>;

struct PortChannelSignalState {
  uint64_t offsets[4];
  uint64_t tokens[4];
};

struct PortChannelDeviceContext {
  PortChannelDeviceHandle* channels;
  PortChannelSignalState* signalStates;
  int numPeers;
  int numQpsPerPeer;

#if defined(MSCCLPP_DEVICE_COMPILE)
  MSCCLPP_DEVICE_INLINE PortChannelDeviceHandle channel(int peer, int queue) const {
    MSCCLPP_ASSERT_DEVICE(channels != nullptr && peer >= 0 && peer < numPeers && queue >= 0 && queue < numQpsPerPeer,
                          "Invalid PortChannel index");
    return channels[static_cast<size_t>(peer) * numQpsPerPeer + queue];
  }

  MSCCLPP_DEVICE_INLINE PortChannelSignalState& signalState(int peer, int queue) const {
    MSCCLPP_ASSERT_DEVICE(
        signalStates != nullptr && peer >= 0 && peer < numPeers && queue >= 0 && queue < numQpsPerPeer,
        "Invalid PortChannel signal state index");
    return signalStates[static_cast<size_t>(peer) * numQpsPerPeer + queue];
  }

  MSCCLPP_DEVICE_INLINE static uint64_t nextSignalToken(PortChannelSignalState& state, uint64_t signalOffset,
                                                        uint64_t signalValue) {
    for (int i = 0; i < 4; ++i) {
      if (state.tokens[i] == 0 || state.offsets[i] == signalOffset) {
        state.offsets[i] = signalOffset;
        return state.tokens[i] += signalValue;
      }
    }
    MSCCLPP_ASSERT_DEVICE(false, "Too many active PortChannel signal offsets");
    return 0;
  }
#endif
};

struct HybridDispatchControl;
struct HybridAggregateControl;

/// Persistent device resources shared by every dispatch and combine algorithm.
struct DeviceContext {
  /// Local base used to translate local buffer addresses into peer mappings.
  void* localBufferBase_;
  /// Peer mappings of the same symmetric allocation.
  void* const* peerBufferBases_;
  /// Peer synchronization channels.
  mscclpp::BaseMemoryChannelDeviceHandle* channels_;
  /// Optional algorithm workspace.
  void* workspace_;
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
  /// Cross-domain PortChannel device context for peers outside this rank's NVLink/IPC domain,
  /// or nullptr when every peer is NVLink-mapped.
  PortChannelDeviceContext* portChannels_ = nullptr;
  /// Tokens per hybrid dispatch/combine IB chunk.
  int nTokensPerChunk_ = 0;
  int gatewayRank_ = -1;
  int distributeCapacity_ = 0;
  void* distributeSendBuffer_ = nullptr;
  void* distributeRecvBuffer_ = nullptr;
  HybridDispatchControl* distributeControl_ = nullptr;
  /// Local [compact row allocator, completed forward CTAs, forward-complete/Combine credits, per-chunk staged rows].
  int* distributeScratch_ = nullptr;
  int* aggregateTokenRows_ = nullptr;
  int* aggregateIncomingRows_ = nullptr;
  /// Private [compact row, IDs then destination slots then weights] cache, never exposed to remote writes.
  int* aggregateRoutes_ = nullptr;
  float* aggregateLocalBuffer_ = nullptr;
  /// Independent registered FP32 send/landing rows, indexed by compact remote row.
  float* aggregateSendBuffer_ = nullptr;
  float* aggregateRecvBuffer_ = nullptr;
  HybridAggregateControl* aggregateControl_ = nullptr;
  /// Private completed-row counters, reset on the caller stream before each aggregate combine.
  int* aggregateChunkReady_ = nullptr;
};

struct TransportView {
  void* symmetricBufferBase_;
  void* const* peerMappedBufferBases_;
  mscclpp::BaseMemoryChannelDeviceHandle* baseMemoryChannels_;
  int rank_;
  int numRanks_;
  PortChannelDeviceContext* portChannels_;

  MSCCLPP_HOST_DEVICE_INLINE explicit TransportView(const DeviceContext* context)
      : symmetricBufferBase_(context->localBufferBase_),
        peerMappedBufferBases_(context->peerBufferBases_),
        baseMemoryChannels_(context->channels_),
        rank_(context->rank_),
        numRanks_(context->numRanks_),
        portChannels_(context->portChannels_) {}

  MSCCLPP_HOST_DEVICE_INLINE bool isSelf(int peerRank) const { return peerRank == rank_; }

  // Byte offset of a symmetric-buffer pointer from this rank's base; the same
  // offset addresses the identical region on any peer.
  MSCCLPP_HOST_DEVICE_INLINE uint64_t symmetricOffset(const void* ptr) const {
    return static_cast<uint64_t>(reinterpret_cast<const uint8_t*>(ptr) -
                                 reinterpret_cast<const uint8_t*>(symmetricBufferBase_));
  }

  // A peer is NVLink/IPC-reachable if it is self or its symmetric buffer is
  // directly mapped; otherwise it is a cross-domain peer served over PortChannel.
  MSCCLPP_HOST_DEVICE_INLINE bool isNvlinkPeer(int peerRank) const {
    return isSelf(peerRank) || peerMappedBufferBases_[peerRank] != nullptr;
  }

  MSCCLPP_HOST_DEVICE_INLINE void* mappedBuffer(void* localBuffer, int peerRank) const {
    if (isSelf(peerRank)) return localBuffer;
    const auto offset = reinterpret_cast<uint8_t*>(localBuffer) - reinterpret_cast<uint8_t*>(symmetricBufferBase_);
    return reinterpret_cast<uint8_t*>(peerMappedBufferBases_[peerRank]) + offset;
  }
};

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_DEVICE_CONTEXT_HPP_
