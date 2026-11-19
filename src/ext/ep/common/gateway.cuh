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

// Cross-domain two-hop gateway channel over device-initiated RDMA PortChannels and intra-domain peers.
class Gateway {
 public:
  struct ForwardRoute {
    int dstRank = -1;
    size_t dstSlotIndex = 0;
  };
  // Counts exclude QP posters. Each worker warp-group passes a dense workerIdx in [0, nWorkers) to reduceAndSend().
  struct ReduceConfig {
    int nWorkerBlocks = 0;
    int nWorkers = 0;
  };
  MSCCLPP_DEVICE_INLINE Gateway(const TransportView& transport, int peerRank, int nTokensPerChunk, size_t tokenBytes,
                                void* sendBuffer, void* recvBuffer, uint64_t* qpSignals, int* chunkReadySignals,
                                uint64_t* completionSignal = nullptr)
      : Gateway(transport, peerRank, nTokensPerChunk, tokenBytes, sendBuffer, recvBuffer, qpSignals, chunkReadySignals,
                ReduceConfig{}, completionSignal) {}
  MSCCLPP_DEVICE_INLINE Gateway(const TransportView& transport, int peerRank, int nTokensPerChunk, size_t tokenBytes,
                                void* sendBuffer, void* recvBuffer, uint64_t* qpSignals, int* chunkReadySignals,
                                ReduceConfig reduceConfig, uint64_t* completionSignal = nullptr)
      : transport_(transport),
        peerRank_(peerRank),
        nTokensPerChunk_(nTokensPerChunk),
        tokenBytes_(tokenBytes),
        sendBuffer_(sendBuffer),
        recvBuffer_(recvBuffer),
        qpSignals_(qpSignals),
        chunkReadySignals_(chunkReadySignals),
        completionSignal_(completionSignal),
        reduceConfig_(reduceConfig) {}

  MSCCLPP_DEVICE_INLINE int peerRank() const { return peerRank_; }
  MSCCLPP_DEVICE_INLINE size_t tokenBytes() const { return tokenBytes_; }

  // Reset per-round chunk counters and, when requested, the incoming QP counters.
  MSCCLPP_DEVICE_INLINE void reset(int maxTokens, int tid, int nThreads, bool resetRecv = true,
                                   int* extraChunkSignals = nullptr) const {
    EP_DEVICE_ASSERT(nThreads > 0 && tid >= 0 && tid < nThreads);
    if (resetRecv) resetRecvSignals(tid, nThreads);
    resetSendSignals(maxTokens, tid, nThreads, extraChunkSignals);
  }

  // Copy payload into sendBuffer_ slot and enqueue it for RDMA chunk transmission.
  template <size_t PayloadBytes>
  MSCCLPP_DEVICE_INLINE void enqueue(int tokenIdx, const void* staged, int warpGroupThreadIdx, int nWarpGroupThreads,
                                     int warpGroupBarrierId) const {
    static_assert(PayloadBytes % sizeof(mscclpp::u32x4) == 0);
    if (tokenIdx < 0) return;
    EP_DEVICE_ASSERT(PayloadBytes <= tokenBytes_ && warpGroupBarrierId > 0);
    auto* sendToken = static_cast<uint8_t*>(sendBuffer_) + static_cast<size_t>(tokenIdx) * tokenBytes_;
    for (int vectorIdx = warpGroupThreadIdx; vectorIdx < static_cast<int>(PayloadBytes / sizeof(mscclpp::u32x4));
         vectorIdx += nWarpGroupThreads) {
      reinterpret_cast<mscclpp::u32x4*>(sendToken)[vectorIdx] =
          reinterpret_cast<const mscclpp::u32x4*>(staged)[vectorIdx];
    }
    syncNamedBarrier(warpGroupBarrierId, nWarpGroupThreads);
    if (warpGroupThreadIdx == 0) ready(tokenIdx);
  }

  MSCCLPP_DEVICE_INLINE void sendHeader(int queueIdx, void* dstHeader, const void* srcHeader, size_t headerBytes) {
    auto& channel = getChannel(queueIdx);
    constexpr auto kExclusive = PortChannelDeviceHandle::Concurrency::Exclusive;
    channel.put<kExclusive, MaxQpsPerPeer>(transport_.symmetricOffset(dstHeader), transport_.symmetricOffset(srcHeader),
                                           headerBytes, queueIdx);
    clearQpFlush(queueIdx);
  }

  // Stream ready chunks striped across QPs.
  MSCCLPP_DEVICE_INLINE void send(int queueIdx, int nTokens) {
    auto& channel = getChannel(queueIdx);
    constexpr auto kExclusive = PortChannelDeviceHandle::Concurrency::Exclusive;
    const int nQps = transport_.portChannels_->numQpsPerPeer;
    const uint64_t dstOffset = transport_.symmetricOffset(recvBuffer_);
    const uint64_t srcOffset = transport_.symmetricOffset(sendBuffer_);
    const uint64_t flagOffset = transport_.symmetricOffset(qpSignals_ + queueIdx);
    if (requireQpFlush(queueIdx, nTokens)) {
      // Drain prior-round SQ completions while producers/reducers prepare the first chunk.
      channel.flush<kExclusive>(-1);
    }
    clearQpFlush(queueIdx);
    for (int chunkIdx = queueIdx, tokenBegin = queueIdx * nTokensPerChunk_; tokenBegin < nTokens;
         chunkIdx += nQps, tokenBegin += nQps * nTokensPerChunk_) {
      const int nChunkTokens = min(nTokensPerChunk_, nTokens - tokenBegin);
      const uint64_t byteOffset = static_cast<uint64_t>(tokenBegin) * tokenBytes_;
      putWithAccumulateAfter<MaxQpsPerPeer>(
          channel, dstOffset + byteOffset, srcOffset + byteOffset, static_cast<uint64_t>(nChunkTokens) * tokenBytes_,
          flagOffset, 1, queueIdx, [&] {
            while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(chunkReadySignals_ + chunkIdx,
                                                                  mscclpp::memoryOrderAcquire) != nChunkTokens) {
            }
          });
    }
  }

  // Once local forwarding is complete, notify the peer so the next round can reuse the receive buffers.
  MSCCLPP_DEVICE_INLINE void complete(int queueIdx, int* localDone) {
    EP_DEVICE_ASSERT(completionSignal_ != nullptr);
    auto& channel = getChannel(queueIdx);
    const uint64_t signalOffset = transport_.symmetricOffset(completionSignal_);
    constexpr auto kExclusive = PortChannelDeviceHandle::Concurrency::Exclusive;
    while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(localDone, mscclpp::memoryOrderAcquire) != 1) {
    }
    channel.accumulate<kExclusive, MaxQpsPerPeer>(signalOffset, 1, queueIdx);
  }

  // Wait for and consume the peer's completion signal before starting the next round.
  MSCCLPP_DEVICE_INLINE void waitComplete() const {
    EP_DEVICE_ASSERT(completionSignal_ != nullptr);
    while (mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(completionSignal_, mscclpp::memoryOrderAcquire) == 0) {
    }
    *completionSignal_ = 0;
  }

  // Dynamic multi-dst intra-domain forward: wait for incoming RDMA chunks, TMA-load each token into shared memory,
  // resolve per-lane destinations during bulkLoad, TMA-store payload (and optional AuxBytes) to each unique
  // intra-domain peer, and run overlapFn while bulkStore is in flight.
  template <size_t PrimaryBytes, size_t Stride, int NWarpGroupWarps, size_t AuxBytes = 0, typename ResolveRouteFn,
            typename OverlapFn>
  MSCCLPP_DEVICE_INLINE void forward(uint8_t* base, void* dstBuffer, void* dstAuxBuffer, int nTokens, int forwarderIdx,
                                     int nForwarders, int warpGroupThreadIdx, int warpGroupBarrierId,
                                     ResolveRouteFn&& resolveRoute, OverlapFn&& overlapFn) const {
    static_assert(NWarpGroupWarps > 0);
    constexpr size_t LoadBytes = configAlign<size_t>(PrimaryBytes + AuxBytes, 16);
    constexpr int NWarpGroupThreads = NWarpGroupWarps * WARP_SIZE;
    EP_DEVICE_ASSERT(warpGroupThreadIdx >= 0 && warpGroupThreadIdx < NWarpGroupThreads);
    EP_DEVICE_ASSERT(warpGroupBarrierId > 0);
    const int laneIdx = getLaneId();
    const int warpIdx = warpGroupThreadIdx / WARP_SIZE;
    auto* staged = base + static_cast<size_t>(warpIdx) * Stride;
    auto* barriers = reinterpret_cast<mscclpp::BulkBarrier*>(base + static_cast<size_t>(NWarpGroupWarps) * Stride);
    if (laneIdx == 0) barriers[warpIdx].init();
    uint32_t phase = 0;
    // Stripe tokens across forwarder groups first to spread small workloads across SMs.
    for (int tokenIdx = warpIdx * nForwarders + forwarderIdx; tokenIdx < nTokens;
         tokenIdx += nForwarders * NWarpGroupWarps) {
      if (laneIdx == 0) wait(tokenIdx);
      __syncwarp();

      auto* landing = static_cast<uint8_t*>(recvBuffer_) + static_cast<size_t>(tokenIdx) * tokenBytes_;
      if (laneIdx == 0) {
        barriers[warpIdx].arriveAndExpect(LoadBytes);
        mscclpp::bulkLoad(staged, landing, LoadBytes, barriers[warpIdx]);
      }
      const ForwardRoute route = resolveRoute(tokenIdx, landing);
      const bool isLeader = isFirstLaneForRank(route.dstRank, laneIdx);
      const bool isLocalLeader = route.dstRank >= 0 && transport_.isIntraDomainPeer(route.dstRank) && isLeader;
      if (laneIdx == 0) barriers[warpIdx].wait(phase);
      __syncwarp();
      mscclpp::bulkFence();

      if (isLocalLeader) {
        auto* dstPrimary = static_cast<uint8_t*>(transport_.mappedBuffer(dstBuffer, route.dstRank));
        mscclpp::bulkStore(dstPrimary + route.dstSlotIndex * PrimaryBytes, staged, PrimaryBytes);
        if constexpr (AuxBytes > 0) {
          auto* dstAux = static_cast<uint8_t*>(transport_.mappedBuffer(dstAuxBuffer, route.dstRank)) +
                         route.dstSlotIndex * AuxBytes;
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
      overlapFn(tokenIdx, route, leaderMask);
      if (isLocalLeader) mscclpp::bulkStoreWait();
      __syncwarp();
    }
    if (laneIdx == 0) barriers[warpIdx].invalidate();
    syncNamedBarrier(warpGroupBarrierId, NWarpGroupThreads);
  }

  template <size_t PrimaryBytes, size_t Stride, int NWarpGroupWarps, typename ResolveRouteFn>
  MSCCLPP_DEVICE_INLINE void forward(uint8_t* base, void* dstBuffer, int nTokens, int forwarderIdx, int nForwarders,
                                     int warpGroupThreadIdx, int warpGroupBarrierId,
                                     ResolveRouteFn&& resolveRoute) const {
    forward<PrimaryBytes, Stride, NWarpGroupWarps, 0>(
        base, dstBuffer, nullptr, nTokens, forwarderIdx, nForwarders, warpGroupThreadIdx, warpGroupBarrierId,
        static_cast<ResolveRouteFn&&>(resolveRoute), [](int, const ForwardRoute&, unsigned int) {});
  }

  // Fixed single-dst intra-domain forward without callbacks.
  template <size_t PrimaryBytes, size_t Stride, int NWarpGroupWarps, size_t AuxBytes = 0>
  MSCCLPP_DEVICE_INLINE void forward(uint8_t* base, int dstRank, void* dstBuffer, size_t dstBaseSlotIndex, int nTokens,
                                     int forwarderIdx, int nForwarders, int warpGroupThreadIdx, int warpGroupBarrierId,
                                     void* dstAuxBuffer = nullptr) const {
    const int laneIdx = getLaneId();
    forward<PrimaryBytes, Stride, NWarpGroupWarps, AuxBytes>(
        base, dstBuffer, dstAuxBuffer, nTokens, forwarderIdx, nForwarders, warpGroupThreadIdx, warpGroupBarrierId,
        [&](int tokenIdx, void*) {
          return laneIdx == 0 ? ForwardRoute{dstRank, dstBaseSlotIndex + static_cast<size_t>(tokenIdx)}
                              : ForwardRoute{};
        },
        [](int, const ForwardRoute&, unsigned int) {});
  }

  // QP posters send ready chunks while worker groups [0, nWorkers) wave-pace loads and reduce into sendBuffer_.
  template <int NWarpGroupThreads, typename ReduceTokenFn>
  MSCCLPP_DEVICE_INLINE void reduceAndSend(int nTokens, int* chunkLoadReadySignals, bool isPoster, int workerIdx,
                                           int posterQueueIdx, int warpGroupThreadIdx, int warpGroupBarrierId,
                                           ReduceTokenFn&& reduceToken) {
    if (isPoster) {
      const int nQps = transport_.portChannels_->numQpsPerPeer;
      if (posterQueueIdx >= 0 && posterQueueIdx < nQps && getLaneId() == 0) {
        send(posterQueueIdx, nTokens);
      }
      return;
    }
    EP_DEVICE_ASSERT(reduceConfig_.nWorkerBlocks > 0 && reduceConfig_.nWorkers >= reduceConfig_.nWorkerBlocks);
    EP_DEVICE_ASSERT(workerIdx >= 0 && workerIdx < reduceConfig_.nWorkers);
    const int nGatewayWorkers = reduceConfig_.nWorkers;
    const int nSmWaveChunks = max(1, reduceConfig_.nWorkerBlocks / nTokensPerChunk_);
    auto* sendBase = static_cast<uint8_t*>(sendBuffer_);
    for (int tokenIdx = workerIdx; tokenIdx < nTokens; tokenIdx += nGatewayWorkers) {
      const int chunkIdx = tokenIdx / nTokensPerChunk_;
      if (chunkIdx > 0) {
        if (warpGroupThreadIdx == 0) {
          // Launch chunk 0 uncontested to kick off RDMA immediately, then pace by one 1-CTA-per-SM grid wave.
          const int prevLoadChunk = max(0, chunkIdx - nSmWaveChunks);
          while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(chunkLoadReadySignals + prevLoadChunk,
                                                                mscclpp::memoryOrderAcquire) != nTokensPerChunk_) {
          }
        }
        syncNamedBarrier(warpGroupBarrierId, NWarpGroupThreads);
      }
      auto loadReady = [&] { ready(chunkLoadReadySignals, tokenIdx); };
      reduceToken(sendBase + static_cast<size_t>(tokenIdx) * tokenBytes_, tokenIdx, loadReady);
      // Join all warp-group writers before thread 0 publishes chunk readiness.
      syncNamedBarrier(warpGroupBarrierId, NWarpGroupThreads);
      if (warpGroupThreadIdx == 0) ready(tokenIdx);
    }
    // Wait for priority gateway TMA loads before local reductions compete intra-domain, skipping any partial tail wave.
    const int nPriorityTokens = nTokens <= nGatewayWorkers ? nTokens : (nTokens / nGatewayWorkers) * nGatewayWorkers;
    if (warpGroupThreadIdx == 0) {
      for (int tokenBegin = 0; tokenBegin < nPriorityTokens; tokenBegin += nTokensPerChunk_) {
        if (nTokens > nGatewayWorkers && tokenBegin + nTokensPerChunk_ > nPriorityTokens) break;
        const int nChunkTokens = min(nTokensPerChunk_, nTokens - tokenBegin);
        while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(chunkLoadReadySignals + tokenBegin / nTokensPerChunk_,
                                                              mscclpp::memoryOrderAcquire) != nChunkTokens) {
        }
      }
    }
    syncNamedBarrier(warpGroupBarrierId, NWarpGroupThreads);
  }

  // Warp-group-cooperative intra-domain completion signal: last forwarder resets incoming QP counters, marks local
  // forwarding complete, and notifies intra-domain receivers in parallel.
  MSCCLPP_DEVICE_INLINE void signalIntraDomain(int* nCompletedForwarders, int nForwarders, int* forwardComplete,
                                               uint32_t* peerReadyFlags, uint32_t flag, bool* sharedIsLastForwarder,
                                               int warpGroupThreadIdx, int nWarpGroupThreads,
                                               int warpGroupBarrierId) const {
    EP_DEVICE_ASSERT(sharedIsLastForwarder != nullptr);
    EP_DEVICE_ASSERT(nWarpGroupThreads > 0 && nWarpGroupThreads % WARP_SIZE == 0);
    EP_DEVICE_ASSERT(warpGroupThreadIdx >= 0 && warpGroupThreadIdx < nWarpGroupThreads);
    EP_DEVICE_ASSERT(warpGroupBarrierId > 0);
    const int nRanks = transport_.numRanks_;
    const int nQps = transport_.portChannels_->numQpsPerPeer;
    if (warpGroupThreadIdx == 0) {
      const int nCompleted =
          mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(nCompletedForwarders, 1, mscclpp::memoryOrderAcqRel);
      const bool isLast = nCompleted == nForwarders - 1;
      if (isLast) {
        for (int queueIdx = 0; queueIdx < nQps; ++queueIdx) qpSignals_[queueIdx] = 0;
        mscclpp::atomicStore<int, mscclpp::scopeDevice>(forwardComplete, 1, mscclpp::memoryOrderRelease);
      }
      *sharedIsLastForwarder = isLast;
    }
    syncNamedBarrier(warpGroupBarrierId, nWarpGroupThreads);
    if (*sharedIsLastForwarder) {
      for (int peerIdx = warpGroupThreadIdx; peerIdx < nRanks; peerIdx += nWarpGroupThreads) {
        if (transport_.isIntraDomainPeer(peerIdx)) {
          auto* target = static_cast<uint32_t*>(transport_.mappedBuffer(peerReadyFlags, peerIdx));
          mscclpp::atomicStore<uint32_t, mscclpp::scopeSystem>(target + peerRank_, flag, mscclpp::memoryOrderRelease);
        }
      }
    }
  }

  // 1-dst RDMA receiver poll/wait.
  MSCCLPP_DEVICE_INLINE bool poll(int tokenIdx) const {
    const int nQps = transport_.portChannels_->numQpsPerPeer;
    const int chunkIdx = tokenIdx / nTokensPerChunk_;
    const int queueIdx = chunkIdx % nQps;
    const uint64_t target = static_cast<uint64_t>(chunkIdx / nQps + 1);
    return mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(qpSignals_ + queueIdx, mscclpp::memoryOrderAcquire) >=
           target;
  }

  MSCCLPP_DEVICE_INLINE void wait(int tokenIdx) const {
    while (!poll(tokenIdx)) {
    }
  }

  // Intra-domain poll/wait (paired with signalIntraDomain).
  MSCCLPP_DEVICE_INLINE static bool pollIntraDomain(const uint32_t* peerReadyFlags, int sourceRank, uint32_t flag) {
    return mscclpp::atomicLoad<uint32_t, mscclpp::scopeSystem>(const_cast<uint32_t*>(peerReadyFlags + sourceRank),
                                                               mscclpp::memoryOrderAcquire) == flag;
  }

  MSCCLPP_DEVICE_INLINE static void waitIntraDomain(const uint32_t* peerReadyFlags, int sourceRank, uint32_t flag) {
    while (!pollIntraDomain(peerReadyFlags, sourceRank, flag)) {
    }
  }

 private:
  MSCCLPP_DEVICE_INLINE PortChannelDeviceHandle& getChannel(int queueIdx) {
    if (cachedQueueIdx_ != queueIdx) {
      cachedChannel_ = transport_.portChannels_->channel(peerRank_, queueIdx);
      cachedQueueIdx_ = queueIdx;
    }
    return cachedChannel_;
  }

  MSCCLPP_DEVICE_INLINE bool requireQpFlush(int queueIdx, int nTokens) const {
    return (needQpFlushMask_ & (1u << queueIdx)) != 0 && queueIdx * nTokensPerChunk_ < nTokens;
  }

  MSCCLPP_DEVICE_INLINE void clearQpFlush(int queueIdx) { needQpFlushMask_ &= ~(1u << queueIdx); }

  MSCCLPP_DEVICE_INLINE void resetRecvSignals(int tid, int nThreads) const {
    const int nQps = transport_.portChannels_->numQpsPerPeer;
    for (int queueIdx = tid; queueIdx < nQps; queueIdx += nThreads) {
      qpSignals_[queueIdx] = 0;
    }
  }

  MSCCLPP_DEVICE_INLINE void resetSendSignals(int maxTokens, int tid, int nThreads, int* extraChunkSignals) const {
    const int nChunks = configCellDiv(maxTokens, nTokensPerChunk_);
    for (int chunkIdx = tid; chunkIdx < nChunks; chunkIdx += nThreads) {
      chunkReadySignals_[chunkIdx] = 0;
      if (extraChunkSignals != nullptr) extraChunkSignals[chunkIdx] = 0;
    }
  }

  MSCCLPP_DEVICE_INLINE void ready(int* chunkSignals, int tokenIdx) const {
    atomicAddReleaseDevice(chunkSignals + tokenIdx / nTokensPerChunk_, 1);
  }

  MSCCLPP_DEVICE_INLINE void ready(int tokenIdx) const { ready(chunkReadySignals_, tokenIdx); }

  const TransportView& transport_;
  int peerRank_;
  int nTokensPerChunk_;
  size_t tokenBytes_;
  void* sendBuffer_;
  void* recvBuffer_;
  uint64_t* qpSignals_;
  int* chunkReadySignals_;
  uint64_t* completionSignal_;
  ReduceConfig reduceConfig_;
  PortChannelDeviceHandle cachedChannel_{};
  int cachedQueueIdx_ = -1;
  uint32_t needQpFlushMask_ = ~0u;
};

#endif  // defined(MSCCLPP_DEVICE_CUDA) && defined(USE_IBVERBS) && MSCCLPP_BULK_AVAILABLE

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_COMMON_GATEWAY_CUH_
