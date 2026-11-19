// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_DEVICE_CONTEXT_HPP_
#define MSCCLPP_EP_DEVICE_CONTEXT_HPP_

#include <mscclpp/gpu_data_types.hpp>
#include <mscclpp/memory_channel_device.hpp>
#include <mscclpp/port_channel_device.hpp>

namespace mscclpp {
namespace ep {

using PortChannelDeviceHandle = mscclpp::PortChannelDeviceHandle<mscclpp::PortChannelBackend::Device>;

struct PortChannelDeviceContext {
  PortChannelDeviceHandle* channels;
  int numPeers;
  int numQpsPerPeer;

#if defined(MSCCLPP_DEVICE_COMPILE)
  MSCCLPP_DEVICE_INLINE PortChannelDeviceHandle channel(int peer, int queue) const {
    MSCCLPP_ASSERT_DEVICE(channels != nullptr && peer >= 0 && peer < numPeers && queue >= 0 && queue < numQpsPerPeer,
                          "Invalid PortChannel index");
    return channels[static_cast<size_t>(peer) * numQpsPerPeer + queue];
  }
#endif
};

struct HybridDispatchControl;
struct HybridCombineControl;
struct HybridDispatchScratch;

struct HybridDeviceContext {
  /// Cross-domain PortChannel device context for peers outside this rank's NVLink/IPC domain,
  /// or nullptr when every peer is intra-domain mapped.
  PortChannelDeviceContext* portChannels_ = nullptr;
  /// Tokens per hybrid dispatch/combine RDMA chunk.
  int nTokensPerChunk_ = 0;
  int gatewayRank_ = -1;
  int maxTokensPerRank_ = 0;
  void* dispatchSendBuffer_ = nullptr;
  void* dispatchRecvBuffer_ = nullptr;
  HybridDispatchControl* dispatchControl_ = nullptr;
  HybridDispatchScratch* dispatchScratch_ = nullptr;
  int* combineRemoteTokenIndices_ = nullptr;
  int* nIncomingTokens_ = nullptr;
  /// Private [compact token, IDs then destination slots then weights] cache, never exposed to remote writes.
  int* combineRoutes_ = nullptr;
  float* combineLocalBuffer_ = nullptr;
  /// Independent registered FP32 send/landing tokens, indexed by compact remote token.
  float* combineSendBuffer_ = nullptr;
  float* combineRecvBuffer_ = nullptr;
  HybridCombineControl* combineControl_ = nullptr;
  /// Private completed-token and TMA-load counters, reset during dispatch.
  int* combineChunkReady_ = nullptr;
  int* combineChunkLoadReady_ = nullptr;
  mscclpp::DataType combineInterDomainDtype_ = mscclpp::DataType::FLOAT32;
};

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
  /// Cross-domain resources for hybrid dispatch and combine.
  HybridDeviceContext hybridCtx_{};
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
        portChannels_(context->hybridCtx_.portChannels_) {}

  MSCCLPP_HOST_DEVICE_INLINE bool isSelf(int peerRank) const { return peerRank == rank_; }

  // Byte offset of a symmetric-buffer pointer from this rank's base; the same
  // offset addresses the identical region on any peer.
  MSCCLPP_HOST_DEVICE_INLINE uint64_t symmetricOffset(const void* ptr) const {
    return static_cast<uint64_t>(reinterpret_cast<const uint8_t*>(ptr) -
                                 reinterpret_cast<const uint8_t*>(symmetricBufferBase_));
  }

  // A peer is intra-domain (IPC/NVLink) reachable if it is self or its symmetric
  // buffer is directly mapped; otherwise it is an inter-domain peer served over PortChannel.
  MSCCLPP_HOST_DEVICE_INLINE bool isIntraDomainPeer(int peerRank) const {
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
