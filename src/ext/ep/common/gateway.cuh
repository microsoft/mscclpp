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
  struct ForwardRoute {
    int dstRank = -1;
    size_t dstIndex = 0;
  };
  MSCCLPP_DEVICE_INLINE Gateway(const TransportView& transport, int peerRank, int nItemsPerChunk, void* sendBuffer,
                                void* recvBuffer, uint64_t* qpSignals, int* chunkReadySignals)
      : transport_(transport),
        peerRank_(peerRank),
        nItemsPerChunk_(nItemsPerChunk),
        sendBuffer_(sendBuffer),
        recvBuffer_(recvBuffer),
        qpSignals_(qpSignals),
        chunkReadySignals_(chunkReadySignals) {}

  MSCCLPP_DEVICE_INLINE int peerRank() const { return peerRank_; }

  // Reset per-round send/receive counters and optionally wait until the peer's recvBuffer is free.
  MSCCLPP_DEVICE_INLINE void reset(int maxItems, bool resetRecv = true, int* extraChunkSignals = nullptr,
                                   uint64_t* recvBufferFree = nullptr) const {
    if (resetRecv) resetRecvSignals();
    resetSendSignals(maxItems, extraChunkSignals);
    if (recvBufferFree != nullptr && threadIdx.x == 0) waitRecvBufferFree(recvBufferFree);
  }

  // Copy payload into sendBuffer_ slot and enqueue it for RDMA chunk transmission.
  template <size_t PayloadBytes>
  MSCCLPP_DEVICE_INLINE void enqueue(int itemIdx, const void* staged, size_t itemStride, int groupThreadIdx,
                                     int nGroupThreads, int groupBarrierId) const {
    if (itemIdx < 0) return;
    auto* sendItem = static_cast<uint8_t*>(sendBuffer_) + static_cast<size_t>(itemIdx) * itemStride;
    for (int vectorIdx = groupThreadIdx; vectorIdx < static_cast<int>(PayloadBytes / sizeof(mscclpp::u32x4));
         vectorIdx += nGroupThreads) {
      reinterpret_cast<mscclpp::u32x4*>(sendItem)[vectorIdx] =
          reinterpret_cast<const mscclpp::u32x4*>(staged)[vectorIdx];
    }
    syncNamedBarrier(groupBarrierId, nGroupThreads);
    if (groupThreadIdx == 0) ready(itemIdx);
  }

  // 1-dst IB control forward: post optional header, stream ready chunks over IB, and release recvBuffer once consumed.
  MSCCLPP_DEVICE_INLINE void forward(int queueIdx, int nItems, size_t itemStride, void* dstHeader = nullptr,
                                     const void* srcHeader = nullptr, size_t headerBytes = 0,
                                     uint64_t* peerRecvBufferFree = nullptr, int* localForwardDone = nullptr) const {
    auto channel = transport_.portChannels_->channel(peerRank_, queueIdx);
    const bool sendHdr = queueIdx == 0 && headerBytes > 0;
    if (sendHdr) {
      sendHeader(channel, queueIdx, dstHeader, srcHeader, headerBytes);
    }
    sendChunks(channel, queueIdx, nItems, itemStride, !sendHdr);
    if (queueIdx == 0 && peerRecvBufferFree != nullptr) {
      releaseRecvBuffer(channel, queueIdx, peerRecvBufferFree, localForwardDone);
    }
  }

  // Dynamic multi-dst NVLink forward: wait for incoming IB chunks, TMA-load each item into shared memory, resolve
  // per-lane destinations during bulkLoad, TMA-store payload (and optional AuxBytes) to each unique local NVLink peer,
  // and run overlapFn while bulkStore is in flight.
  template <size_t PrimaryBytes, size_t TileStride, int NForwardWarps, size_t AuxBytes = 0, typename ResolveRouteFn,
            typename OverlapFn>
  MSCCLPP_DEVICE_INLINE void forward(uint8_t* tileBase, void* dstBuffer, void* dstAuxBuffer, int nItems,
                                     size_t itemStride, int forwarderIdx, int nForwarders,
                                     ResolveRouteFn&& resolveRoute, OverlapFn&& overlapFn) const {
    constexpr size_t LoadBytes = configAlign<size_t>(PrimaryBytes + AuxBytes, 16);
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
          barriers[warpIdx].arriveAndExpect(LoadBytes);
          mscclpp::bulkLoad(staged, landing, LoadBytes, barriers[warpIdx]);
        }
        const ForwardRoute route = resolveRoute(itemIdx, landing);
        const bool isLeader = isFirstLaneForRank(route.dstRank, laneIdx);
        const bool isLocalLeader = route.dstRank >= 0 && transport_.isNvlinkPeer(route.dstRank) && isLeader;
        if (laneIdx == 0) barriers[warpIdx].wait(phase);
        __syncwarp();
        mscclpp::bulkFence();

        if (isLocalLeader) {
          auto* dstPrimary = static_cast<uint8_t*>(transport_.mappedBuffer(dstBuffer, route.dstRank));
          mscclpp::bulkStore(dstPrimary + route.dstIndex * PrimaryBytes, staged, PrimaryBytes);
          if constexpr (AuxBytes > 0) {
            auto* dstAux =
                static_cast<uint8_t*>(transport_.mappedBuffer(dstAuxBuffer, route.dstRank)) + route.dstIndex * AuxBytes;
            const auto* stagedAux = staged + PrimaryBytes;
            if constexpr (AuxBytes % 16 == 0) {
              mscclpp::bulkStore(dstAux, stagedAux, AuxBytes);
            } else {
              static_assert(AuxBytes % sizeof(uint64_t) == 0);
              for (size_t i = 0; i < AuxBytes / sizeof(uint64_t); ++i) {
                reinterpret_cast<uint64_t*>(dstAux)[i] = reinterpret_cast<const uint64_t*>(stagedAux)[i];
              }
            }
          }
          mscclpp::bulkStoreCommit();
        }
        const unsigned int leaderMask = __ballot_sync(0xffffffff, isLocalLeader);
        overlapFn(itemIdx, route, leaderMask);
        if (isLocalLeader) mscclpp::bulkStoreWait();
        __syncwarp();
      }
      if (laneIdx == 0) barriers[warpIdx].invalidate();
    }
    __syncthreads();
  }

  template <size_t PrimaryBytes, size_t TileStride, int NForwardWarps, typename ResolveRouteFn>
  MSCCLPP_DEVICE_INLINE void forward(uint8_t* tileBase, void* dstBuffer, int nItems, size_t itemStride,
                                     int forwarderIdx, int nForwarders, ResolveRouteFn&& resolveRoute) const {
    forward<PrimaryBytes, TileStride, NForwardWarps, 0>(tileBase, dstBuffer, nullptr, nItems, itemStride, forwarderIdx,
                                                        nForwarders, static_cast<ResolveRouteFn&&>(resolveRoute),
                                                        [](int, const ForwardRoute&, unsigned int) {});
  }

  // Fixed single-dst NVLink forward without callbacks.
  template <size_t PrimaryBytes, size_t TileStride, int NForwardWarps, size_t AuxBytes = 0>
  MSCCLPP_DEVICE_INLINE void forward(uint8_t* tileBase, int dstRank, void* dstBuffer, size_t dstBaseIndex, int nItems,
                                     size_t itemStride, int forwarderIdx, int nForwarders,
                                     void* dstAuxBuffer = nullptr) const {
    const int laneIdx = getLaneId();
    forward<PrimaryBytes, TileStride, NForwardWarps, AuxBytes>(
        tileBase, dstBuffer, dstAuxBuffer, nItems, itemStride, forwarderIdx, nForwarders,
        [&](int itemIdx, void*) {
          return laneIdx == 0 ? ForwardRoute{dstRank, dstBaseIndex + static_cast<size_t>(itemIdx)} : ForwardRoute{};
        },
        [](int, const ForwardRoute&, unsigned int) {});
  }

  // Cooperative gateway gather-reduce + IB forward: block 0 QP lanes forward ready reduced chunks over IB while
  // worker blocks (1..gridDim.x-1) wave-pace NVLink loads, reduce items into sendBuffer_, and mark chunks ready.
  template <int NWarpGroups, int GroupThreads, typename ReduceItemFn>
  MSCCLPP_DEVICE_INLINE void reduceAndForward(int nItems, size_t itemBytes, int* chunkLoadDoneSignals,
                                              int posterQueueIdx, int groupIdx, int groupThreadIdx, int groupBarrierId,
                                              ReduceItemFn&& reduceItem) const {
    if (blockIdx.x == 0) {
      const int nQps = transport_.portChannels_->numQpsPerPeer;
      if (posterQueueIdx >= 0 && posterQueueIdx < nQps && getLaneId() == 0) {
        auto channel = transport_.portChannels_->channel(peerRank_, posterQueueIdx);
        sendChunks(channel, posterQueueIdx, nItems, itemBytes, true);
      }
      return;
    }
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
      if (groupThreadIdx == 0) ready(itemIdx);
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

  // Intra-domain completion signal: last forwarder resets incoming QP counters, marks local forwarding complete, and
  // notifies intra-domain NVLink receivers.
  MSCCLPP_DEVICE_INLINE void signalIntraDomain(int* nCompletedForwarders, int nForwarders, int* forwardComplete,
                                               uint32_t* peerReadyFlags, uint32_t flag) const {
    const int nRanks = transport_.numRanks_;
    const int nQps = transport_.portChannels_->numQpsPerPeer;
    __shared__ bool isLastForwarder;
    if (threadIdx.x == 0) {
      const int nCompleted =
          mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(nCompletedForwarders, 1, mscclpp::memoryOrderAcqRel);
      isLastForwarder = nCompleted == nForwarders - 1;
      if (isLastForwarder) {
        for (int queueIdx = 0; queueIdx < nQps; ++queueIdx) qpSignals_[queueIdx] = 0;
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
    return mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(qpSignals_ + queueIdx, mscclpp::memoryOrderAcquire) >=
           target;
  }

  MSCCLPP_DEVICE_INLINE void wait(int itemIdx) const {
    while (!poll(itemIdx)) {
    }
  }

  // Intra-domain NVLink poll/wait (paired with signalIntraDomain).
  MSCCLPP_DEVICE_INLINE static bool pollIntraDomain(const uint32_t* peerReadyFlags, int sourceRank, uint32_t flag) {
    return mscclpp::atomicLoad<uint32_t, mscclpp::scopeSystem>(const_cast<uint32_t*>(peerReadyFlags + sourceRank),
                                                               mscclpp::memoryOrderAcquire) == flag;
  }

  MSCCLPP_DEVICE_INLINE static void waitIntraDomain(const uint32_t* peerReadyFlags, int sourceRank, uint32_t flag) {
    while (!pollIntraDomain(peerReadyFlags, sourceRank, flag)) {
    }
  }

 private:
  MSCCLPP_DEVICE_INLINE void resetRecvSignals() const {
    const int nQps = transport_.portChannels_->numQpsPerPeer;
    for (int queueIdx = threadIdx.x; queueIdx < nQps; queueIdx += blockDim.x) {
      qpSignals_[queueIdx] = 0;
    }
  }

  MSCCLPP_DEVICE_INLINE void resetSendSignals(int maxItems, int* extraChunkSignals) const {
    const int nChunks = configCellDiv(maxItems, nItemsPerChunk_);
    for (int chunkIdx = threadIdx.x; chunkIdx < nChunks; chunkIdx += blockDim.x) {
      chunkReadySignals_[chunkIdx] = 0;
      if (extraChunkSignals != nullptr) extraChunkSignals[chunkIdx] = 0;
    }
  }

  MSCCLPP_DEVICE_INLINE void ready(int itemIdx) const {
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
    const uint64_t flagOffset = transport_.symmetricOffset(qpSignals_ + queueIdx);
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

  MSCCLPP_DEVICE_INLINE void releaseRecvBuffer(PortChannelDeviceHandle& channel, int queueIdx,
                                               uint64_t* peerRecvBufferFree, int* localForwardDone) const {
    accumulateAfter<MaxQpsPerPeer>(channel, transport_.symmetricOffset(peerRecvBufferFree), 1, queueIdx, [&] {
      while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(localForwardDone, mscclpp::memoryOrderAcquire) != 1) {
      }
    });
  }

  MSCCLPP_DEVICE_INLINE static void waitRecvBufferFree(uint64_t* recvBufferFree) {
    while (mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(recvBufferFree, mscclpp::memoryOrderAcquire) == 0) {
    }
    *recvBufferFree = 0;
  }

  const TransportView& transport_;
  int peerRank_;
  int nItemsPerChunk_;
  void* sendBuffer_;
  void* recvBuffer_;
  uint64_t* qpSignals_;
  int* chunkReadySignals_;
};

#endif  // defined(MSCCLPP_DEVICE_CUDA) && defined(USE_IBVERBS) && MSCCLPP_BULK_AVAILABLE

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_COMMON_GATEWAY_CUH_
