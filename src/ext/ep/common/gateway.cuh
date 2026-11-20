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

// Per-warp shared-memory staging workspace (128B-aligned payload + mbarrier) for Gateway::forward.
template <size_t PrimaryBytes, size_t AuxBytes = 0>
struct alignas(BufferAlignmentBytes) WarpForwardWorkspace {
  static constexpr size_t LoadBytes = configAlign<size_t>(PrimaryBytes + AuxBytes, 16);
  uint8_t staged[LoadBytes];
  mscclpp::BulkBarrier barrier;
};

#if defined(MSCCLPP_DEVICE_CUDA) && defined(USE_IBVERBS) && MSCCLPP_BULK_AVAILABLE

// Cross-domain two-hop gateway channel over device-initiated RDMA PortChannels and intra-domain peers.
class Gateway {
 public:
  enum class Scope : uint8_t {
    Thread,
    Warp,
    ThreadGroup,
  };

  struct ForwardRoute {
    int dstRank = -1;
    size_t dstSlotIndex = 0;
  };

  MSCCLPP_DEVICE_INLINE Gateway(const TransportView& transport, int peerRank, int nTokensPerChunk, size_t tokenBytes,
                                void* sendBuffer, void* recvBuffer, uint64_t* qpSignals, int* chunkReadySignals,
                                uint64_t* completionSignal = nullptr, int* chunkLoadReadySignals = nullptr)
      : transport_(transport),
        peerRank_(peerRank),
        nTokensPerChunk_(nTokensPerChunk),
        tokenBytes_(tokenBytes),
        sendBuffer_(sendBuffer),
        recvBuffer_(recvBuffer),
        qpSignals_(qpSignals),
        chunkReadySignals_(chunkReadySignals),
        completionSignal_(completionSignal),
        chunkLoadReadySignals_(chunkLoadReadySignals) {}

  MSCCLPP_DEVICE_INLINE int peerRank() const { return peerRank_; }
  MSCCLPP_DEVICE_INLINE int nTokensPerChunk() const { return nTokensPerChunk_; }
  MSCCLPP_DEVICE_INLINE size_t tokenBytes() const { return tokenBytes_; }

  // ---------------------------------------------------------------------------
  // Single-thread operations (Scope::Thread)
  // ---------------------------------------------------------------------------

  template <Scope Mode = Scope::Thread>
  MSCCLPP_DEVICE_INLINE void sendHeader(int queueIdx, void* dstHeader, const void* srcHeader, size_t headerBytes) {
    static_assert(Mode == Scope::Thread);
    auto& channel = getChannel(queueIdx);
    constexpr auto kExclusive = PortChannelDeviceHandle::Concurrency::Exclusive;
    channel.put<kExclusive, MaxQpsPerPeer>(transport_.symmetricOffset(dstHeader), transport_.symmetricOffset(srcHeader),
                                           headerBytes, queueIdx);
    resetFlushQueueFlag(queueIdx);
  }

  // Stream ready chunk-aligned tokens starting at tokenIdx with step tokenStride up to nTokens on queueIdx.
  template <Scope Mode = Scope::Thread>
  MSCCLPP_DEVICE_INLINE void send(int queueIdx, int tokenIdx, int tokenStride, int nTokens) {
    static_assert(Mode == Scope::Thread);
    EP_DEVICE_ASSERT(queueIdx >= 0 && tokenIdx >= 0 && tokenStride > 0 && nTokens >= 0);
    EP_DEVICE_ASSERT(tokenIdx % nTokensPerChunk_ == 0 && tokenStride % nTokensPerChunk_ == 0);
    auto& channel = getChannel(queueIdx);
    constexpr auto kExclusive = PortChannelDeviceHandle::Concurrency::Exclusive;
    const uint64_t dstOffset = transport_.symmetricOffset(recvBuffer_);
    const uint64_t srcOffset = transport_.symmetricOffset(sendBuffer_);
    const uint64_t flagOffset = transport_.symmetricOffset(qpSignals_ + queueIdx);
    if (requireFlushQueue(queueIdx, tokenIdx < nTokens)) {
      // Drain prior-round SQ completions while producers/reducers prepare the first chunk.
      channel.flush<kExclusive>(-1);
    }
    resetFlushQueueFlag(queueIdx);
    const int chunkStride = tokenStride / nTokensPerChunk_;
    for (int chunkIdx = tokenIdx / nTokensPerChunk_, tokenBegin = tokenIdx; tokenBegin < nTokens;
         chunkIdx += chunkStride, tokenBegin += tokenStride) {
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

  // Wait on local waitSignal, then notify the peer that forwarding is complete so the next round can reuse recvBuffer_.
  template <Scope Mode = Scope::Thread>
  MSCCLPP_DEVICE_INLINE void signalForwardComplete(int queueIdx, int* waitSignal) {
    static_assert(Mode == Scope::Thread);
    EP_DEVICE_ASSERT(completionSignal_ != nullptr && waitSignal != nullptr);
    auto& channel = getChannel(queueIdx);
    const uint64_t signalOffset = transport_.symmetricOffset(completionSignal_);
    constexpr auto kExclusive = PortChannelDeviceHandle::Concurrency::Exclusive;
    while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(waitSignal, mscclpp::memoryOrderAcquire) != 1) {
    }
    channel.accumulate<kExclusive, MaxQpsPerPeer>(signalOffset, 1, queueIdx);
  }

  // Wait for and consume the peer's forward-completion signal before starting the next round.
  template <Scope Mode = Scope::Thread>
  MSCCLPP_DEVICE_INLINE void waitForwardComplete() const {
    static_assert(Mode == Scope::Thread);
    EP_DEVICE_ASSERT(completionSignal_ != nullptr);
    while (mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(completionSignal_, mscclpp::memoryOrderAcquire) == 0) {
    }
    *completionSignal_ = 0;
  }

  // 1-dst RDMA receiver poll/wait.
  template <Scope Mode = Scope::Thread>
  MSCCLPP_DEVICE_INLINE bool poll(int tokenIdx) const {
    static_assert(Mode == Scope::Thread);
    const int nQps = transport_.portChannelCtx_->numQpsPerPeer;
    const int chunkIdx = tokenIdx / nTokensPerChunk_;
    const int queueIdx = chunkIdx % nQps;
    const uint64_t target = static_cast<uint64_t>(chunkIdx / nQps + 1);
    return mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(qpSignals_ + queueIdx, mscclpp::memoryOrderAcquire) >=
           target;
  }

  template <Scope Mode = Scope::Thread>
  MSCCLPP_DEVICE_INLINE void wait(int tokenIdx) const {
    static_assert(Mode == Scope::Thread);
    while (!poll(tokenIdx)) {
    }
  }

  // Intra-domain poll/wait (paired with signalIntraDomain).
  template <Scope Mode = Scope::Thread>
  MSCCLPP_DEVICE_INLINE static bool pollIntraDomain(const uint32_t* peerReadyFlags, int sourceRank, uint32_t flag) {
    static_assert(Mode == Scope::Thread);
    return mscclpp::atomicLoad<uint32_t, mscclpp::scopeSystem>(const_cast<uint32_t*>(peerReadyFlags + sourceRank),
                                                               mscclpp::memoryOrderAcquire) == flag;
  }

  template <Scope Mode = Scope::Thread>
  MSCCLPP_DEVICE_INLINE static void waitIntraDomain(const uint32_t* peerReadyFlags, int sourceRank, uint32_t flag) {
    static_assert(Mode == Scope::Thread);
    while (!pollIntraDomain(peerReadyFlags, sourceRank, flag)) {
    }
  }

  // ---------------------------------------------------------------------------
  // Warp-cooperative operations (Scope::Warp, 32 convergent lanes)
  // ---------------------------------------------------------------------------

  // Multi-dst intra-domain forward: all 32 lanes pass the same (tokenIdx, tokenStride, nTokens) and per-warp workspace;
  // lane 0 waits for incoming RDMA chunks and TMA-loads each token, lanes resolve destinations during bulkLoad, unique
  // intra-domain leaders TMA-store payload (and optional AuxBytes), and overlapFn runs while bulkStore is in flight.
  template <Scope Mode, size_t PrimaryBytes, size_t AuxBytes = 0, typename ResolveRouteFn, typename OverlapFn>
  MSCCLPP_DEVICE_INLINE void forward(int tokenIdx, int tokenStride, int nTokens,
                                     WarpForwardWorkspace<PrimaryBytes, AuxBytes>& workspace, void* dstBuffer,
                                     void* dstAuxBuffer, ResolveRouteFn&& resolveRoute, OverlapFn&& overlapFn) const {
    static_assert(Mode == Scope::Warp);
    static_assert(PrimaryBytes > 0 && PrimaryBytes % 16 == 0);
    EP_DEVICE_ASSERT(__activemask() == 0xffffffffu);
    EP_DEVICE_ASSERT(tokenIdx >= 0 && tokenStride > 0 && nTokens >= 0);
    EP_DEVICE_ASSERT(__shfl_sync(0xffffffffu, tokenIdx, 0) == tokenIdx &&
                     __shfl_sync(0xffffffffu, tokenStride, 0) == tokenStride &&
                     __shfl_sync(0xffffffffu, nTokens, 0) == nTokens);
    constexpr size_t LoadBytes = WarpForwardWorkspace<PrimaryBytes, AuxBytes>::LoadBytes;
    const int laneIdx = getLaneId();
    if (laneIdx == 0) workspace.barrier.init();
    uint32_t phase = 0;
    for (int idx = tokenIdx; idx < nTokens; idx += tokenStride) {
      if (laneIdx == 0) wait(idx);
      __syncwarp();

      auto* landing = static_cast<uint8_t*>(recvBuffer_) + static_cast<size_t>(idx) * tokenBytes_;
      if (laneIdx == 0) {
        workspace.barrier.arriveAndExpect(LoadBytes);
        mscclpp::bulkLoad(workspace.staged, landing, LoadBytes, workspace.barrier);
      }
      const ForwardRoute route = resolveRoute(idx, landing);
      const bool isLeader = isFirstLaneForRank(route.dstRank, laneIdx);
      const bool isLocalLeader = route.dstRank >= 0 && transport_.isIntraDomainPeer(route.dstRank) && isLeader;
      if (laneIdx == 0) workspace.barrier.wait(phase);
      __syncwarp();
      mscclpp::bulkFence();

      if (isLocalLeader) {
        auto* dstPrimary = static_cast<uint8_t*>(transport_.mappedBuffer(dstBuffer, route.dstRank));
        mscclpp::bulkStore(dstPrimary + route.dstSlotIndex * PrimaryBytes, workspace.staged, PrimaryBytes);
        if constexpr (AuxBytes > 0) {
          auto* dstAux = static_cast<uint8_t*>(transport_.mappedBuffer(dstAuxBuffer, route.dstRank)) +
                         route.dstSlotIndex * AuxBytes;
          const auto* stagedAux = workspace.staged + PrimaryBytes;
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
      overlapFn(idx, route, leaderMask);
      if (isLocalLeader) mscclpp::bulkStoreWait();
      __syncwarp();
    }
    if (laneIdx == 0) workspace.barrier.invalidate();
  }

  // Intra-domain completion signal: the last forwarder resets incoming QP counters, marks local forwarding complete,
  // and notifies intra-domain receivers across warp lanes.
  template <Scope Mode>
  MSCCLPP_DEVICE_INLINE void signalIntraDomain(int* nCompletedForwarders, int nForwarders, int* forwardComplete,
                                               uint32_t* peerReadyFlags, uint32_t flag) const {
    static_assert(Mode == Scope::Warp);
    EP_DEVICE_ASSERT(__activemask() == 0xffffffffu);
    EP_DEVICE_ASSERT(nCompletedForwarders != nullptr && nForwarders > 0 && forwardComplete != nullptr &&
                     peerReadyFlags != nullptr);
    const int laneIdx = getLaneId();
    const int nRanks = transport_.numRanks_;
    const int nQps = transport_.portChannelCtx_->numQpsPerPeer;
    bool isLast = false;
    if (laneIdx == 0) {
      const int nCompleted =
          mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(nCompletedForwarders, 1, mscclpp::memoryOrderAcqRel);
      isLast = nCompleted == nForwarders - 1;
      if (isLast) {
        for (int queueIdx = 0; queueIdx < nQps; ++queueIdx) qpSignals_[queueIdx] = 0;
        mscclpp::atomicStore<int, mscclpp::scopeDevice>(forwardComplete, 1, mscclpp::memoryOrderRelease);
      }
    }
    if (__shfl_sync(0xffffffffu, isLast, 0)) {
      for (int peerIdx = laneIdx; peerIdx < nRanks; peerIdx += WARP_SIZE) {
        if (transport_.isIntraDomainPeer(peerIdx)) {
          auto* target = static_cast<uint32_t*>(transport_.mappedBuffer(peerReadyFlags, peerIdx));
          mscclpp::atomicStore<uint32_t, mscclpp::scopeSystem>(target + peerRank_, flag, mscclpp::memoryOrderRelease);
        }
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Thread-group-cooperative operations (Scope::ThreadGroup, tid in [0, nThreads))
  // ---------------------------------------------------------------------------

  // Reset per-round chunk counters and, when requested, the incoming QP counters.
  template <Scope Mode>
  MSCCLPP_DEVICE_INLINE void reset(int maxTokens, int tid, int nThreads, bool resetRecv = true) const {
    static_assert(Mode == Scope::ThreadGroup);
    EP_DEVICE_ASSERT(nThreads > 0 && tid >= 0 && tid < nThreads);
    if (resetRecv) resetRecvSignals(tid, nThreads);
    resetSendSignals(maxTokens, tid, nThreads);
  }

  // Copy payload into sendBuffer_ slot and enqueue it for RDMA chunk transmission.
  template <Scope Mode, size_t PayloadBytes, typename SyncFn>
  MSCCLPP_DEVICE_INLINE void enqueue(int tokenIdx, const void* staged, int tid, int nThreads,
                                     SyncFn&& threadBarrierFn) const {
    static_assert(Mode == Scope::ThreadGroup);
    static_assert(PayloadBytes % sizeof(mscclpp::u32x4) == 0);
    if (tokenIdx < 0) return;
    EP_DEVICE_ASSERT(nThreads > 0 && tid >= 0 && tid < nThreads);
    EP_DEVICE_ASSERT(PayloadBytes <= tokenBytes_);
    auto* sendToken = static_cast<uint8_t*>(sendBuffer_) + static_cast<size_t>(tokenIdx) * tokenBytes_;
    for (int vectorIdx = tid; vectorIdx < static_cast<int>(PayloadBytes / sizeof(mscclpp::u32x4));
         vectorIdx += nThreads) {
      reinterpret_cast<mscclpp::u32x4*>(sendToken)[vectorIdx] =
          reinterpret_cast<const mscclpp::u32x4*>(staged)[vectorIdx];
    }
    threadBarrierFn();
    if (tid == 0) ready(tokenIdx);
  }

  // Cooperative reduction into sendBuffer_: all nThreads threads pass the same (tokenIdx, tokenStride, nTokens),
  // wave-pace intra-domain TMA loads across chunks, synchronize via threadBarrierFn, and publish ready chunks.
  template <Scope Mode, typename SyncFn, typename ReduceTokenFn>
  MSCCLPP_DEVICE_INLINE void reduce(int tokenIdx, int tokenStride, int nTokens, int tid, int nThreads,
                                    SyncFn&& threadBarrierFn, ReduceTokenFn&& reduceTokenFn) const {
    static_assert(Mode == Scope::ThreadGroup);
    EP_DEVICE_ASSERT(chunkLoadReadySignals_ != nullptr);
    EP_DEVICE_ASSERT(tokenIdx >= 0 && tokenStride > 0 && nTokens >= 0);
    EP_DEVICE_ASSERT(nThreads > 0 && nThreads % WARP_SIZE == 0 && static_cast<int>(blockDim.x) >= nThreads);
    EP_DEVICE_ASSERT(tid >= 0 && tid < nThreads);
    EP_DEVICE_ASSERT(__activemask() == 0xffffffffu);
    EP_DEVICE_ASSERT(__shfl_sync(0xffffffffu, tokenIdx, 0) == tokenIdx &&
                     __shfl_sync(0xffffffffu, tokenStride, 0) == tokenStride &&
                     __shfl_sync(0xffffffffu, nTokens, 0) == nTokens);
    const int nGroupsPerBlock = static_cast<int>(blockDim.x) / nThreads;
    const int nWorkerBlocks = max(1, tokenStride / nGroupsPerBlock);
    const int loadWindowChunks = max(1, nWorkerBlocks / nTokensPerChunk_);
    auto* sendBase = static_cast<uint8_t*>(sendBuffer_);
    for (int idx = tokenIdx; idx < nTokens; idx += tokenStride) {
      const int chunkIdx = idx / nTokensPerChunk_;
      if (chunkIdx > 0) {
        if (tid == 0) {
          // Launch chunk 0 uncontested to kick off RDMA immediately, then pace by one 1-CTA-per-SM grid wave.
          const int prevLoadChunk = max(0, chunkIdx - loadWindowChunks);
          while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(chunkLoadReadySignals_ + prevLoadChunk,
                                                                mscclpp::memoryOrderAcquire) != nTokensPerChunk_) {
          }
        }
        threadBarrierFn();
      }
      auto markLoadReady = [&] { ready(chunkLoadReadySignals_, idx); };
      reduceTokenFn(sendBase + static_cast<size_t>(idx) * tokenBytes_, idx, markLoadReady);
      // Join all writers in the group before tid 0 publishes chunk readiness.
      threadBarrierFn();
      if (tid == 0) ready(idx);
    }
    // Wait for priority gateway TMA loads before local reductions compete intra-domain, skipping any partial tail wave.
    const int nPriorityTokens = nTokens <= tokenStride ? nTokens : (nTokens / tokenStride) * tokenStride;
    if (tid == 0) {
      for (int tokenBegin = 0; tokenBegin < nPriorityTokens; tokenBegin += nTokensPerChunk_) {
        if (nTokens > tokenStride && tokenBegin + nTokensPerChunk_ > nPriorityTokens) break;
        const int nChunkTokens = min(nTokensPerChunk_, nTokens - tokenBegin);
        while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(chunkLoadReadySignals_ + tokenBegin / nTokensPerChunk_,
                                                              mscclpp::memoryOrderAcquire) != nChunkTokens) {
        }
      }
    }
    threadBarrierFn();
  }

 private:
  MSCCLPP_DEVICE_INLINE PortChannelDeviceHandle& getChannel(int queueIdx) {
    if (cachedQueueIdx_ != queueIdx) {
      cachedChannel_ = transport_.portChannelCtx_->channel(peerRank_, queueIdx);
      cachedQueueIdx_ = queueIdx;
    }
    return cachedChannel_;
  }

  MSCCLPP_DEVICE_INLINE bool requireFlushQueue(int queueIdx, bool hasTokens) const {
    return (flushQueueFlags_ & (1u << queueIdx)) != 0 && hasTokens;
  }

  MSCCLPP_DEVICE_INLINE void resetFlushQueueFlag(int queueIdx) { flushQueueFlags_ &= ~(1u << queueIdx); }

  MSCCLPP_DEVICE_INLINE void resetRecvSignals(int tid, int nThreads) const {
    const int nQps = transport_.portChannelCtx_->numQpsPerPeer;
    for (int queueIdx = tid; queueIdx < nQps; queueIdx += nThreads) {
      qpSignals_[queueIdx] = 0;
    }
  }

  MSCCLPP_DEVICE_INLINE void resetSendSignals(int maxTokens, int tid, int nThreads) const {
    const int nChunks = configCellDiv(maxTokens, nTokensPerChunk_);
    for (int chunkIdx = tid; chunkIdx < nChunks; chunkIdx += nThreads) {
      chunkReadySignals_[chunkIdx] = 0;
      if (chunkLoadReadySignals_ != nullptr) chunkLoadReadySignals_[chunkIdx] = 0;
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
  int* chunkLoadReadySignals_;
  PortChannelDeviceHandle cachedChannel_{};
  int cachedQueueIdx_ = -1;
  uint32_t flushQueueFlags_ = ~0u;
};

#endif  // defined(MSCCLPP_DEVICE_CUDA) && defined(USE_IBVERBS) && MSCCLPP_BULK_AVAILABLE

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_COMMON_GATEWAY_CUH_
