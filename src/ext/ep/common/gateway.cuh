// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_COMMON_GATEWAY_CUH_
#define MSCCLPP_EP_COMMON_GATEWAY_CUH_

#include <mscclpp/bulk_device.hpp>
#include <mscclpp/gpu_data_types.hpp>

#include "common/device_helpers.cuh"
#include "config.hpp"
#include "device_context.hpp"

namespace mscclpp {
namespace ep {

#if defined(MSCCLPP_DEVICE_CUDA) && defined(USE_IBVERBS) && MSCCLPP_BULK_AVAILABLE

// Cross-domain two-hop gateway channel over device-initiated PortChannels and local NVLink peers.
class Gateway {
 public:
  MSCCLPP_DEVICE_INLINE Gateway(const TransportView& transport, int peerRank, int nItemsPerChunk, void* sendBuffer,
                                void* recvBuffer, uint64_t* qpSignalCount, int* chunkReadySignals)
      : transport_(transport),
        peerRank_(peerRank),
        nItemsPerChunk_(nItemsPerChunk),
        sendBuffer_(sendBuffer),
        recvBuffer_(recvBuffer),
        qpSignalCount_(qpSignalCount),
        chunkReadySignals_(chunkReadySignals) {}

  MSCCLPP_DEVICE_INLINE int peerRank() const { return peerRank_; }

  // Reset per-round send/receive counters and optionally wait for landing-buffer reuse permission.
  MSCCLPP_DEVICE_INLINE void reset(int maxItems, bool resetRecv = true, int* extraChunkSignals = nullptr,
                                   uint64_t* localReuseFlag = nullptr) const {
    if (resetRecv) resetRecvSignals();
    resetSendSignals(maxItems, extraChunkSignals);
    if (localReuseFlag != nullptr && threadIdx.x == 0) waitReuseAck(localReuseFlag);
  }

  // 1-dst IB producer forward: copy staged payload into sendBuffer_ and mark its chunk ready.
  template <size_t PayloadBytes>
  MSCCLPP_DEVICE_INLINE void forward(int itemIdx, const void* staged, size_t itemStride, int groupThreadIdx,
                                     int nGroupThreads, int groupBarrierId) const {
    if (itemIdx < 0) return;
    auto* sendItem = static_cast<uint8_t*>(sendBuffer_) + static_cast<size_t>(itemIdx) * itemStride;
    for (int vectorIdx = groupThreadIdx; vectorIdx < static_cast<int>(PayloadBytes / sizeof(mscclpp::u32x4));
         vectorIdx += nGroupThreads) {
      reinterpret_cast<mscclpp::u32x4*>(sendItem)[vectorIdx] =
          reinterpret_cast<const mscclpp::u32x4*>(staged)[vectorIdx];
    }
    syncNamedBarrier(groupBarrierId, nGroupThreads);
    if (groupThreadIdx == 0) markReady(itemIdx);
  }

  // 1-dst IB control forward: post optional header, stream ready chunks over IB, and post optional reuse ACK.
  MSCCLPP_DEVICE_INLINE void forward(int queueIdx, int nItems, size_t itemStride, void* dstHeader = nullptr,
                                     const void* srcHeader = nullptr, size_t headerBytes = 0,
                                     uint64_t* dstReuseFlag = nullptr, int* localDoneFlag = nullptr) const {
    auto channel = transport_.portChannels_->channel(peerRank_, queueIdx);
    const bool sendHdr = queueIdx == 0 && headerBytes > 0;
    if (sendHdr) {
      sendHeader(channel, queueIdx, dstHeader, srcHeader, headerBytes);
    }
    sendChunks(channel, queueIdx, nItems, itemStride, !sendHdr);
    if (queueIdx == 0 && dstReuseFlag != nullptr) {
      sendReuseAck(channel, queueIdx, dstReuseFlag, localDoneFlag);
    }
  }

  // Multi-dst NVLink forward: wait for incoming IB chunks, TMA-load each item's payload into shared memory, and invoke
  // forwardItem(itemIdx, landingItem, stagedPayload, waitPayloadReady) to scatter to 1 or multiple local NVLink peers.
  template <size_t PayloadBytes, size_t TileStride, int NForwardWarps, typename ForwardItemFn>
  MSCCLPP_DEVICE_INLINE void forward(uint8_t* tileBase, int nItems, size_t itemStride, int forwarderIdx,
                                     int nForwarders, ForwardItemFn&& forwardItem) const {
    const int laneIdx = getLaneId();
    const int warpIdx = static_cast<int>(threadIdx.x) / WARP_SIZE;
    auto* staged = tileBase + static_cast<size_t>(warpIdx) * TileStride;
    auto* barriers =
        reinterpret_cast<mscclpp::BulkBarrier*>(tileBase + static_cast<size_t>(NForwardWarps) * TileStride);
    if (warpIdx < NForwardWarps) {
      if (laneIdx == 0) barriers[warpIdx].init();
      uint32_t phase = 0;
      // Stripe items across forwarder blocks first to spread small workloads across SMs.
      for (int itemIdx = warpIdx * nForwarders + forwarderIdx; itemIdx < nItems;
           itemIdx += nForwarders * NForwardWarps) {
        if (laneIdx == 0) wait(itemIdx);
        __syncwarp();

        auto* landing = static_cast<uint8_t*>(recvBuffer_) + static_cast<size_t>(itemIdx) * itemStride;
        if (laneIdx == 0) {
          barriers[warpIdx].arriveAndExpect(PayloadBytes);
          mscclpp::bulkLoad(staged, landing, PayloadBytes, barriers[warpIdx]);
        }
        auto waitPayload = [&] {
          if (laneIdx == 0) barriers[warpIdx].wait(phase);
          __syncwarp();
          mscclpp::bulkFence();
        };
        forwardItem(itemIdx, landing, staged, waitPayload);
        __syncwarp();
      }
      if (laneIdx == 0) barriers[warpIdx].invalidate();
    }
    __syncthreads();
  }

  // Wave-paced NVLink gather + reduction into sendBuffer_, marking chunks ready for 1-dst IB forward.
  template <int NWarpGroups, int GroupThreads, typename ReduceItemFn>
  MSCCLPP_DEVICE_INLINE void reduceAndForward(int nItems, size_t itemBytes, int* chunkLoadDoneSignals, int groupIdx,
                                              int groupThreadIdx, int groupBarrierId, ReduceItemFn&& reduceItem) const {
    const int nGatewayBlocks = static_cast<int>(gridDim.x) - 1;
    const int nGatewayWorkers = nGatewayBlocks * NWarpGroups;
    // Stripe SM-first so group 0 handles earlier chunks while group 1 on the same SM overlaps TMA with group 0 compute.
    const int workerIdx = groupIdx * nGatewayBlocks + (static_cast<int>(blockIdx.x) - 1);
    const int nSmWaveChunks = max(1, nGatewayBlocks / nItemsPerChunk_);
    auto* sendBase = static_cast<uint8_t*>(sendBuffer_);
    for (int itemIdx = workerIdx; itemIdx < nItems; itemIdx += nGatewayWorkers) {
      const int chunkIdx = itemIdx / nItemsPerChunk_;
      if (chunkIdx > 0) {
        if (groupThreadIdx == 0) {
          // Launch chunk 0 uncontested to kick off IB RDMA immediately, then pace by one 1-CTA-per-SM grid wave.
          const int prevLoadChunk = max(0, chunkIdx - nSmWaveChunks);
          while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(chunkLoadDoneSignals + prevLoadChunk,
                                                                mscclpp::memoryOrderAcquire) != nItemsPerChunk_) {
          }
        }
        syncNamedBarrier(groupBarrierId, GroupThreads);
      }
      auto markLoadDone = [&] { markChunkCount(chunkLoadDoneSignals, itemIdx); };
      reduceItem(sendBase + static_cast<size_t>(itemIdx) * itemBytes, itemIdx, markLoadDone);
      if (groupThreadIdx == 0) markReady(itemIdx);
    }
    // Wait for priority gateway TMA loads before local reductions compete on NVLink, skipping any partial tail wave.
    const int nPriorityItems = nItems <= nGatewayWorkers ? nItems : (nItems / nGatewayWorkers) * nGatewayWorkers;
    if (groupThreadIdx == 0) {
      for (int itemBegin = 0; itemBegin < nPriorityItems; itemBegin += nItemsPerChunk_) {
        if (nItems > nGatewayWorkers && itemBegin + nItemsPerChunk_ > nPriorityItems) break;
        const int nChunkItems = min(nItemsPerChunk_, nItems - itemBegin);
        while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(chunkLoadDoneSignals + itemBegin / nItemsPerChunk_,
                                                              mscclpp::memoryOrderAcquire) != nChunkItems) {
        }
      }
    }
    syncNamedBarrier(groupBarrierId, GroupThreads);
  }

  // Multi-receiver completion signal: last forwarder resets incoming QP counters, marks local forwarding complete, and
  // notifies local NVLink receivers.
  MSCCLPP_DEVICE_INLINE void forwardSignal(int* nCompletedForwarders, int nForwarders, int* forwardComplete,
                                           uint32_t* peerReadyFlags, uint32_t flag) const {
    const int nRanks = transport_.numRanks_;
    const int nQps = transport_.portChannels_->numQpsPerPeer;
    __shared__ bool isLastForwarder;
    if (threadIdx.x == 0) {
      const int nCompleted =
          mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(nCompletedForwarders, 1, mscclpp::memoryOrderAcqRel);
      isLastForwarder = nCompleted == nForwarders - 1;
      if (isLastForwarder) {
        for (int queueIdx = 0; queueIdx < nQps; ++queueIdx) qpSignalCount_[queueIdx] = 0;
      }
    }
    __syncthreads();
    if (isLastForwarder) {
      // Use the first warp past the peer-signaling lanes so the local completion flag is not delayed by system stores.
      const int ackThreadIdx = configAlign(nRanks, WARP_SIZE);
      if (static_cast<int>(threadIdx.x) == ackThreadIdx) {
        mscclpp::atomicStore<int, mscclpp::scopeDevice>(forwardComplete, 1, mscclpp::memoryOrderRelease);
      }
      for (int peerIdx = threadIdx.x; peerIdx < nRanks; peerIdx += blockDim.x) {
        if (transport_.isNvlinkPeer(peerIdx)) {
          auto* target = static_cast<uint32_t*>(transport_.mappedBuffer(peerReadyFlags, peerIdx));
          mscclpp::atomicStore<uint32_t, mscclpp::scopeSystem>(target + peerRank_, flag, mscclpp::memoryOrderRelease);
        }
      }
    }
  }

  // 1-dst IB receiver poll/wait.
  MSCCLPP_DEVICE_INLINE bool poll(int itemIdx) const {
    const int nQps = transport_.portChannels_->numQpsPerPeer;
    const int chunkIdx = itemIdx / nItemsPerChunk_;
    const int queueIdx = chunkIdx % nQps;
    const uint64_t target = static_cast<uint64_t>(chunkIdx / nQps + 1);
    return mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(qpSignalCount_ + queueIdx,
                                                               mscclpp::memoryOrderAcquire) >= target;
  }

  MSCCLPP_DEVICE_INLINE void wait(int itemIdx) const {
    while (!poll(itemIdx)) {
    }
  }

  // Multi-receiver NVLink poll/wait (paired with forwardSignal).
  MSCCLPP_DEVICE_INLINE static bool forwardPoll(const uint32_t* peerReadyFlags, int sourceRank, uint32_t flag) {
    return mscclpp::atomicLoad<uint32_t, mscclpp::scopeSystem>(const_cast<uint32_t*>(peerReadyFlags + sourceRank),
                                                               mscclpp::memoryOrderAcquire) == flag;
  }

  MSCCLPP_DEVICE_INLINE static void forwardWait(const uint32_t* peerReadyFlags, int sourceRank, uint32_t flag) {
    while (!forwardPoll(peerReadyFlags, sourceRank, flag)) {
    }
  }

 private:
  MSCCLPP_DEVICE_INLINE void resetRecvSignals() const {
    const int nQps = transport_.portChannels_->numQpsPerPeer;
    for (int queueIdx = threadIdx.x; queueIdx < nQps; queueIdx += blockDim.x) {
      qpSignalCount_[queueIdx] = 0;
    }
  }

  MSCCLPP_DEVICE_INLINE void resetSendSignals(int maxItems, int* extraChunkSignals) const {
    const int nChunks = configCellDiv(maxItems, nItemsPerChunk_);
    for (int chunkIdx = threadIdx.x; chunkIdx < nChunks; chunkIdx += blockDim.x) {
      chunkReadySignals_[chunkIdx] = 0;
      if (extraChunkSignals != nullptr) extraChunkSignals[chunkIdx] = 0;
    }
  }

  MSCCLPP_DEVICE_INLINE void markReady(int itemIdx) const {
    atomicAddReleaseDevice(chunkReadySignals_ + itemIdx / nItemsPerChunk_, 1);
  }

  MSCCLPP_DEVICE_INLINE void markChunkCount(int* chunkSignals, int itemIdx) const {
    atomicAddReleaseDevice(chunkSignals + itemIdx / nItemsPerChunk_, 1);
  }

  MSCCLPP_DEVICE_INLINE void sendHeader(PortChannelDeviceHandle& channel, int queueIdx, void* dstHeader,
                                        const void* srcHeader, size_t bytes) const {
    constexpr auto kExclusive = mscclpp::PortChannelDeviceHandle::Concurrency::Exclusive;
    channel.put<kExclusive, MaxQpsPerPeer>(transport_.symmetricOffset(dstHeader), transport_.symmetricOffset(srcHeader),
                                           bytes, queueIdx);
  }

  // Post ready chunks striped round-robin across configured QPs.
  MSCCLPP_DEVICE_INLINE void sendChunks(PortChannelDeviceHandle& channel, int queueIdx, int nItems, size_t itemStride,
                                        bool warmChannel) const {
    constexpr auto kExclusive = mscclpp::PortChannelDeviceHandle::Concurrency::Exclusive;
    const int nQps = transport_.portChannels_->numQpsPerPeer;
    const uint64_t dstOffset = transport_.symmetricOffset(recvBuffer_);
    const uint64_t srcOffset = transport_.symmetricOffset(sendBuffer_);
    const uint64_t flagOffset = transport_.symmetricOffset(qpSignalCount_ + queueIdx);
    if (warmChannel && queueIdx * nItemsPerChunk_ < nItems) {
      // Drain prior-round SQ completions and warm QP/CQ state while producers/reducers prepare the first chunk.
      channel.flush<kExclusive>(-1);
    }
    for (int chunkIdx = queueIdx, itemBegin = queueIdx * nItemsPerChunk_; itemBegin < nItems;
         chunkIdx += nQps, itemBegin += nQps * nItemsPerChunk_) {
      const int nChunkItems = min(nItemsPerChunk_, nItems - itemBegin);
      const uint64_t byteOffset = static_cast<uint64_t>(itemBegin) * itemStride;
      putWithAccumulateAfter<MaxQpsPerPeer>(
          channel, dstOffset + byteOffset, srcOffset + byteOffset, static_cast<uint64_t>(nChunkItems) * itemStride,
          flagOffset, 1, queueIdx, [&] {
            while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(chunkReadySignals_ + chunkIdx,
                                                                  mscclpp::memoryOrderAcquire) != nChunkItems) {
            }
          });
    }
  }

  MSCCLPP_DEVICE_INLINE void sendReuseAck(PortChannelDeviceHandle& channel, int queueIdx, uint64_t* dstReuseFlag,
                                          int* localDoneFlag) const {
    accumulateAfter<MaxQpsPerPeer>(channel, transport_.symmetricOffset(dstReuseFlag), 1, queueIdx, [&] {
      while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(localDoneFlag, mscclpp::memoryOrderAcquire) != 1) {
      }
    });
  }

  MSCCLPP_DEVICE_INLINE static void waitReuseAck(uint64_t* localReuseFlag) {
    while (mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(localReuseFlag, mscclpp::memoryOrderAcquire) == 0) {
    }
    *localReuseFlag = 0;
  }

  const TransportView& transport_;
  int peerRank_;
  int nItemsPerChunk_;
  void* sendBuffer_;
  void* recvBuffer_;
  uint64_t* qpSignalCount_;
  int* chunkReadySignals_;
};

#endif  // defined(MSCCLPP_DEVICE_CUDA) && defined(USE_IBVERBS) && MSCCLPP_BULK_AVAILABLE

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_COMMON_GATEWAY_CUH_
