// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_HYBRID_GPU_NET_IO_HPP_
#define MSCCLPP_EP_HYBRID_GPU_NET_IO_HPP_

#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/port_channel_device.hpp>
#include <string>

namespace mscclpp::ep {

/// Device view of the EP GPUNetIO channels: one PortChannel per (peer, queue), each bound to its own queue pair over
/// an EndpointConfig::Ib::Mode::GpuNetIo connection. Every channel's dst_/src_ are the peer's / local symmetric buffer,
/// so offsets below are symmetric-buffer offsets. Unlike PortChannelDeviceHandle::put(), these helpers do not fence;
/// callers order their source writes (see the EP row-fence modes). EP posts to a QP from many threads concurrently,
/// so these helpers use GPU resource-sharing mode and ring the doorbell explicitly (GPU_SM_DB): the core creates
/// GpuNetIo QPs for exclusive BlueFlame submission, and DOCA's NIC_HANDLER_AUTO submit has no BlueFlame path.
///
/// Optional modes (host env, see EpGpuNetIoService::setup):
///  - exclusive: each post holds the QP's sq_lock across reserve, WQE preparation, ready and submit, so the QP is
///    driven with DOCA EXCLUSIVE-mode bookkeeping (plain index updates, no atomic reservation, no ready-index CAS
///    chain, no second lock for the doorbell). The last WQE is built in registers and, with blueFlame, written to
///    the BlueFlame register directly from them. Completion polling stays in GPU sharing mode.
///  - blueFlame: submit through the QP's BlueFlame register (64B WQE write) plus the doorbell.
///  - inlineSignal: a remote flag increment becomes a strong-ordered 8-byte inline RDMA write of a sender-side
///    counter instead of an RDMA atomic. Valid because every EP flag slot is written by exactly one sender rank,
///    always through the same QP, by one thread at a time; the receiver only waits for `flag >= target`.
struct EpGpuNetIoDeviceContext {
  /// Flag slots per (kind, sender) in the dispatch/combine flag buffers (GpuNetIoMaxQpsPerPeer).
  static constexpr int SignalQueues = 64;
  /// Shadow counters per peer: dispatch flags, combine flags, then the expanded-sync flag.
  static constexpr int SignalTokensPerPeer = 2 * SignalQueues + 1;
  static constexpr uint64_t NoOffset = ~uint64_t{0};

  PortChannelDeviceHandle* channels;
  int numPeers;
  int numQpsPerPeer;
  int numHcas;
  int rank = -1;
  int blueFlame = 0;
  int inlineSignal = 0;
  int exclusive = 0;
  /// [numPeers][SignalTokensPerPeer] last value written to each remote flag slot owned by this rank.
  uint64_t* signalTokens = nullptr;
  /// Symmetric offset of the dispatch flag buffer; the combine flag buffer follows at + signalFlagsBytes.
  uint64_t signalFlagsOffset = NoOffset;
  uint64_t signalFlagsBytes = 0;
  /// Symmetric offset of the expanded-sync flag buffer (one uint64_t per sender rank).
  uint64_t signalSyncOffset = NoOffset;

#if defined(MSCCLPP_DEVICE_COMPILE)
  MSCCLPP_DEVICE_INLINE static void require(bool valid) { GpuNetIoDeviceHandle::require(valid); }

  MSCCLPP_DEVICE_INLINE PortChannelDeviceHandle channel(int peer, int queue) const {
    require(channels != nullptr && peer >= 0 && peer < numPeers && queue >= 0 && queue < numQpsPerPeer);
    const auto handle = channels[static_cast<size_t>(peer) * numQpsPerPeer + queue];
    require(handle.backend_ == PortChannelBackend::GpuNetIo && handle.gpuNetIo_.qp != nullptr);
    return handle;
  }

#if defined(MSCCLPP_USE_GPUNETIO)
  MSCCLPP_DEVICE_INLINE static doca_gpu_dev_verbs_qp* qp(const PortChannelDeviceHandle& handle) {
    return static_cast<doca_gpu_dev_verbs_qp*>(handle.gpuNetIo_.qp);
  }

  /// Remote (peer) address of `bytes` at `offset`, bounds-checked against the channel's destination memory.
  MSCCLPP_DEVICE_INLINE static doca_gpu_dev_verbs_addr remoteAddr(const PortChannelDeviceHandle& handle,
                                                                  uint64_t offset, uint64_t bytes) {
    const auto memory = handle.gpuNetIo_.memory(handle.dst_, false, offset, bytes);
    return {memory.addr + offset, memory.key};
  }

  /// Local address of `bytes` at `offset`, bounds-checked against the channel's source memory.
  MSCCLPP_DEVICE_INLINE static doca_gpu_dev_verbs_addr localAddr(const PortChannelDeviceHandle& handle, uint64_t offset,
                                                                 uint64_t bytes) {
    const auto memory = handle.gpuNetIo_.memory(handle.src_, true, offset, bytes);
    return {memory.addr + offset, memory.key};
  }

  MSCCLPP_DEVICE_INLINE static doca_gpu_dev_verbs_addr atomicScratch(const PortChannelDeviceHandle& handle) {
    return {handle.gpuNetIo_.atomicScratchAddr, handle.gpuNetIo_.atomicScratchKey};
  }
#endif

  MSCCLPP_DEVICE_INLINE void put(int peer, uint64_t destination, uint64_t source, uint64_t bytes, int queue = 0) {
#if defined(MSCCLPP_USE_GPUNETIO)
    const auto handle = channel(peer, queue);
    auto* q = qp(handle);
    if (exclusive) {
      putExclusive(q, remoteAddr(handle, destination, bytes), localAddr(handle, source, bytes), bytes);
      return;
    }
    doca_gpu_dev_verbs_ticket_t ticket;
    doca_gpu_dev_verbs_put<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU, DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB>(
        q, remoteAddr(handle, destination, bytes), localAddr(handle, source, bytes), bytes, &ticket,
        DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_DB_RINGING);
    submit(q, ticket + 1);
#else
    require(false);
#endif
  }

  MSCCLPP_DEVICE_INLINE void atomicAdd(int peer, uint64_t destination, int64_t value, int queue = 0) {
#if defined(MSCCLPP_USE_GPUNETIO)
    const auto handle = channel(peer, queue);
    const auto remote = remoteAddr(handle, destination, sizeof(uint64_t));
    require(remote.addr % sizeof(uint64_t) == 0);
    auto* q = qp(handle);
    if (exclusive) {
      auto* token = signalToken(peer, destination);
      const auto scratch = atomicScratch(handle);
      const uint64_t index = beginExclusive(q, 1);
      ExclusiveWqe last;
      if (token != nullptr) {
        prepareSignal(q, &last.wqe, index, remote, nextToken(token, value));
      } else {
        doca_gpu_dev_verbs_wqe_prepare_atomic(q, &last.wqe, index, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
                                              DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, remote.addr, remote.key,
                                              scratch.addr, scratch.key, sizeof(uint64_t),
                                              static_cast<uint64_t>(value), 0);
      }
      endExclusive(q, index + 1, &last);
      return;
    }
    const uint64_t index = doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
        q, 1, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT);
    auto* wqe = doca_gpu_dev_verbs_get_wqe_ptr(q, index);
    if (auto* token = signalToken(peer, destination)) {
      prepareSignal(q, wqe, index, remote, nextToken(token, value));
    } else {
      const auto scratch = atomicScratch(handle);
      doca_gpu_dev_verbs_wqe_prepare_atomic(q, wqe, index, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
                                            DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, remote.addr, remote.key,
                                            scratch.addr, scratch.key, sizeof(uint64_t), static_cast<uint64_t>(value),
                                            0);
    }
    doca_gpu_dev_verbs_mark_wqes_ready<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(q, index, index);
    submit(q, index + 1);
#else
    require(false);
#endif
  }

  /// RDMA write followed by an ordered remote atomic add of `signalValue` at symmetric offset `signalOffset`.
  MSCCLPP_DEVICE_INLINE void putWithSignal(int peer, uint64_t destination, uint64_t source, uint64_t bytes,
                                           uint64_t signalOffset, uint64_t signalValue, int queue = 0) {
#if defined(MSCCLPP_USE_GPUNETIO)
    const auto handle = channel(peer, queue);
    const auto signal = remoteAddr(handle, signalOffset, sizeof(uint64_t));
    require(signal.addr % sizeof(uint64_t) == 0);
    auto* q = qp(handle);
    if (exclusive) {
      auto* token = signalToken(peer, signalOffset);
      if (bytes == 0 || bytes > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE) {
        put(peer, destination, source, bytes, queue);
        atomicAdd(peer, signalOffset, static_cast<int64_t>(signalValue), queue);
        return;
      }
      const auto remote = remoteAddr(handle, destination, bytes);
      const auto local = localAddr(handle, source, bytes);
      const auto scratch = atomicScratch(handle);
      const uint64_t index = beginExclusive(q, 2);
      doca_gpu_dev_verbs_wqe_prepare_write(q, doca_gpu_dev_verbs_get_wqe_ptr(q, index), index,
                                           DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                                           DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0, remote.addr, remote.key,
                                           local.addr, local.key, bytes);
      ExclusiveWqe last;
      if (token != nullptr) {
        prepareSignal(q, &last.wqe, index + 1, signal, nextToken(token, static_cast<int64_t>(signalValue)));
      } else {
        doca_gpu_dev_verbs_wqe_prepare_atomic(q, &last.wqe, index + 1, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
                                              DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, signal.addr, signal.key,
                                              scratch.addr, scratch.key, sizeof(uint64_t), signalValue, 0);
      }
      endExclusive(q, index + 2, &last);
      return;
    }
    if (auto* token = signalToken(peer, signalOffset)) {
      if (bytes == 0 || bytes > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE) {
        put(peer, destination, source, bytes, queue);
        atomicAdd(peer, signalOffset, static_cast<int64_t>(signalValue), queue);
        return;
      }
      const auto remote = remoteAddr(handle, destination, bytes);
      const auto local = localAddr(handle, source, bytes);
      const uint64_t index = doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
          q, 2, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT);
      doca_gpu_dev_verbs_wqe_prepare_write(q, doca_gpu_dev_verbs_get_wqe_ptr(q, index), index,
                                           DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                                           DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0, remote.addr, remote.key,
                                           local.addr, local.key, bytes);
      prepareSignal(q, doca_gpu_dev_verbs_get_wqe_ptr(q, index + 1), index + 1, signal,
                    nextToken(token, static_cast<int64_t>(signalValue)));
      doca_gpu_dev_verbs_mark_wqes_ready<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(q, index, index + 1);
      submit(q, index + 2);
      return;
    }
    doca_gpu_dev_verbs_ticket_t ticket;
    doca_gpu_dev_verbs_put_signal<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU,
                                  DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB>(
        q, remoteAddr(handle, destination, bytes), localAddr(handle, source, bytes), bytes, signal,
        atomicScratch(handle), signalValue, &ticket, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_DB_RINGING);
    submit(q, ticket + 1);
#else
    require(false);
#endif
  }

  /// Wait for every WQE posted so far on (peer, queue). Unlike PortChannelDeviceHandle::flush(), this polls in GPU
  /// resource-sharing mode because EP posts to a QP from many threads concurrently.
  MSCCLPP_DEVICE_INLINE void flush(int peer, int queue = 0) { waitTicket(peer, reservedTicket(peer, queue), queue); }

  /// Poll for completion of every WQE posted so far on (peer, queue) for at most `spins` iterations.
  /// @return 0 when drained, EBUSY if still pending, or the completion error status.
  MSCCLPP_DEVICE_INLINE int tryFlush(int peer, uint64_t spins, int queue = 0) {
#if defined(MSCCLPP_USE_GPUNETIO)
    auto* q = qp(channel(peer, queue));
    const uint64_t ticket =
        doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(&q->sq_rsvd_index);
    if (ticket == 0) return 0;
    for (uint64_t spin = 0; spin < spins; ++spin) {
      const int status =
          doca_gpu_dev_verbs_poll_one_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(q, ticket - 1);
      if (status != EBUSY) return status;
    }
    return EBUSY;
#else
    require(false);
    return 0;
#endif
  }

  /// Send-queue reservation index of (peer, queue): every WQE posted so far lies below it.
  MSCCLPP_DEVICE_INLINE uint64_t reservedTicket(int peer, int queue = 0) {
#if defined(MSCCLPP_USE_GPUNETIO)
    return doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
        &qp(channel(peer, queue))->sq_rsvd_index);
#else
    require(false);
    return 0;
#endif
  }

  /// Wait until every WQE below `ticket` (from reservedTicket) on (peer, queue) has completed.
  MSCCLPP_DEVICE_INLINE void waitTicket(int peer, uint64_t ticket, int queue = 0) {
#if defined(MSCCLPP_USE_GPUNETIO)
    if (ticket == 0) return;
    auto* q = qp(channel(peer, queue));
    int status;
    do {
      status = doca_gpu_dev_verbs_poll_one_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(q, ticket - 1);
    } while (status == EBUSY);
    require(status == 0);
#else
    require(false);
#endif
  }

  MSCCLPP_DEVICE_INLINE void putBatched3(int peer, int queue, uint64_t dst0, uint64_t src0, uint64_t size0,
                                         uint64_t dst1, uint64_t src1, uint64_t size1, uint64_t dst2, uint64_t src2,
                                         uint64_t size2) {
#if defined(MSCCLPP_USE_GPUNETIO)
    const auto handle = channel(peer, queue);
    const uint64_t destinations[3] = {dst0, dst1, dst2};
    const uint64_t sources[3] = {src0, src1, src2};
    const uint64_t sizes[3] = {size0, size1, size2};
    doca_gpu_dev_verbs_addr remote[3], local[3];
    for (int index = 0; index < 3; ++index) {
      require(sizes[index] <= DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE);
      remote[index] = remoteAddr(handle, destinations[index], sizes[index]);
      local[index] = localAddr(handle, sources[index], sizes[index]);
    }
    auto* q = qp(handle);
    if (exclusive) {
      const uint64_t base = beginExclusive(q, 3);
#pragma unroll
      for (int index = 0; index < 2; ++index) {
        doca_gpu_dev_verbs_wqe_prepare_write(q, doca_gpu_dev_verbs_get_wqe_ptr(q, base + index), base + index,
                                             DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                                             DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0, remote[index].addr,
                                             remote[index].key, local[index].addr, local[index].key, sizes[index]);
      }
      ExclusiveWqe last;
      doca_gpu_dev_verbs_wqe_prepare_write(q, &last.wqe, base + 2, DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                                           DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0, remote[2].addr, remote[2].key,
                                           local[2].addr, local[2].key, sizes[2]);
      endExclusive(q, base + 3, &last);
      return;
    }
    const uint64_t base = doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
        q, 3, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT);
#pragma unroll
    for (int index = 0; index < 3; ++index) {
      auto* wqe = doca_gpu_dev_verbs_get_wqe_ptr(q, base + index);
      doca_gpu_dev_verbs_wqe_prepare_write(q, wqe, base + index, DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                                           DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0, remote[index].addr,
                                           remote[index].key, local[index].addr, local[index].key, sizes[index]);
    }
    doca_gpu_dev_verbs_mark_wqes_ready<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(q, base, base + 2);
    submit(q, base + 3);
#else
    require(false);
#endif
  }

#if defined(MSCCLPP_USE_GPUNETIO)
  /// Hand every WQE below `prodIndex` (already marked ready) to the NIC: BlueFlame or doorbell, in GPU sharing mode.
  MSCCLPP_DEVICE_INLINE void submit(doca_gpu_dev_verbs_qp* q, uint64_t prodIndex) const {
    if (blueFlame) {
      submitBlueFlame(q, prodIndex);
      return;
    }
    doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU, DOCA_GPUNETIO_VERBS_SYNC_SCOPE_THREAD,
                              DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB>(q, prodIndex,
                                                                         DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT);
  }

  struct alignas(16) ExclusiveWqe {
    doca_gpu_dev_verbs_wqe wqe;
  };

  /// Exclusive mode: take the QP lock and reserve `count` WQE slots. Must be paired with endExclusive().
  MSCCLPP_DEVICE_INLINE static uint64_t beginExclusive(doca_gpu_dev_verbs_qp* q, uint32_t count) {
    doca_gpu_dev_verbs_lock<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(&q->sq_lock);
    cuda::atomic_ref<uint64_t, cuda::thread_scope_device> reserved(q->sq_rsvd_index);
    const uint64_t index = reserved.load(cuda::memory_order_relaxed);
    doca_gpu_dev_verbs_wait_until_slot_available<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(q,
                                                                                               index + count - 1);
    reserved.store(index + count, cuda::memory_order_relaxed);
    return index;
  }

  /// Exclusive mode: store `last` (the WQE at prodIndex - 1; nullptr if already in the SQ), publish, ring, unlock.
  MSCCLPP_DEVICE_INLINE void endExclusive(doca_gpu_dev_verbs_qp* q, uint64_t prodIndex,
                                          const ExclusiveWqe* last) const {
    ExclusiveWqe copy;
    auto* slot = reinterpret_cast<uint64_t*>(doca_gpu_dev_verbs_get_wqe_ptr(q, prodIndex - 1));
    if (last != nullptr) {
      auto* words = const_cast<uint64_t*>(reinterpret_cast<const uint64_t*>(&last->wqe));
#pragma unroll
      for (int index = 0; index < 8; index += 2) doca_gpu_dev_verbs_store_wqe_seg(slot + index, words + index);
    } else if (blueFlame) {
#pragma unroll
      for (int index = 0; index < 4; ++index)
        reinterpret_cast<uint4*>(&copy.wqe)[index] = reinterpret_cast<const uint4*>(slot)[index];
      last = &copy;
    }
    cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(q->sq_ready_index).store(prodIndex,
                                                                                    cuda::memory_order_relaxed);
    cuda::atomic_ref<uint64_t, cuda::thread_scope_device>(q->sq_wqe_pi).store(prodIndex, cuda::memory_order_relaxed);
    if (blueFlame) {
      auto* bf = reinterpret_cast<uint4*>(doca_gpu_dev_verbs_load_const((uintptr_t*)&q->sq_db));
      const auto* words = reinterpret_cast<const uint4*>(&last->wqe);
      doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
#pragma unroll
      for (int index = 0; index < 4; ++index) bf[index] = words[index];
      doca_priv_gpu_dev_verbs_update_dbr<DOCA_GPUNETIO_VERBS_QP_SQ>(q, prodIndex);
      doca_gpu_dev_verbs_ring_db<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>(q, prodIndex);
    } else {
      doca_gpu_dev_verbs_ring_db<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>(q, prodIndex);
      doca_priv_gpu_dev_verbs_update_dbr<DOCA_GPUNETIO_VERBS_QP_SQ>(q, prodIndex);
      doca_gpu_dev_verbs_ring_db<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>(q, prodIndex);
    }
    doca_gpu_dev_verbs_unlock<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(&q->sq_lock);
  }

 private:
  /// Exclusive-mode RDMA write; transfers above the per-WQE limit are split.
  MSCCLPP_DEVICE_INLINE void putExclusive(doca_gpu_dev_verbs_qp* q, doca_gpu_dev_verbs_addr remote,
                                          doca_gpu_dev_verbs_addr local, uint64_t bytes) const {
    do {
      const uint64_t chunk =
          bytes > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE ? DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE : bytes;
      const uint64_t index = beginExclusive(q, 1);
      ExclusiveWqe last;
      doca_gpu_dev_verbs_wqe_prepare_write(q, &last.wqe, index, DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                                           DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0, remote.addr, remote.key,
                                           local.addr, local.key, chunk);
      endExclusive(q, index + 1, &last);
      remote.addr += chunk;
      local.addr += chunk;
      bytes -= chunk;
    } while (bytes > 0);
  }

  /// Sender-side shadow of the remote flag at symmetric `offset` on `peer`, or nullptr when the slot is not an
  /// EP flag owned by this rank (those keep RDMA atomics).
  MSCCLPP_DEVICE_INLINE uint64_t* signalToken(int peer, uint64_t offset) const {
    if (!inlineSignal || signalTokens == nullptr) return nullptr;
    uint64_t* tokens = signalTokens + static_cast<size_t>(peer) * SignalTokensPerPeer;
    if (signalFlagsOffset != NoOffset && offset >= signalFlagsOffset &&
        offset < signalFlagsOffset + 2 * signalFlagsBytes) {
      const uint64_t relative = offset - signalFlagsOffset;
      const uint64_t kind = relative / signalFlagsBytes;
      const int64_t slot = static_cast<int64_t>((relative % signalFlagsBytes) / sizeof(uint64_t)) -
                           static_cast<int64_t>(rank) * SignalQueues;
      if (slot < 0 || slot >= SignalQueues) return nullptr;
      return tokens + kind * SignalQueues + slot;
    }
    if (signalSyncOffset != NoOffset && offset == signalSyncOffset + static_cast<uint64_t>(rank) * sizeof(uint64_t))
      return tokens + 2 * SignalQueues;
    return nullptr;
  }

  MSCCLPP_DEVICE_INLINE static uint64_t nextToken(uint64_t* token, int64_t value) {
    return ::atomicAdd(reinterpret_cast<unsigned long long*>(token), static_cast<unsigned long long>(value)) +
           static_cast<uint64_t>(value);
  }

  /// Strong-ordered inline write of `token` to `signal`: placed only after every earlier WQE on the QP.
  MSCCLPP_DEVICE_INLINE static void prepareSignal(doca_gpu_dev_verbs_qp* q, doca_gpu_dev_verbs_wqe* wqe,
                                                  uint64_t index, const doca_gpu_dev_verbs_addr& signal,
                                                  uint64_t token) {
    constexpr auto flags = static_cast<doca_gpu_dev_verbs_wqe_ctrl_flags>(
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE | DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_STRONG_ORDERING);
    doca_gpu_dev_verbs_prepare_inl_rdma_write_wqe_header(q, wqe, static_cast<uint16_t>(index), flags, signal.addr,
                                                         signal.key, sizeof(uint64_t));
    doca_gpu_dev_verbs_prepare_inl_rdma_write_wqe_data<uint64_t>(q, wqe, token);
  }

  /// BlueFlame submit in GPU sharing mode: under the QP lock, copy the last WQE (64B) from the SQ straight to the
  /// BlueFlame register with 16B stores and ring the doorbell (covers any earlier WQEs). No shared memory needed.
  MSCCLPP_DEVICE_INLINE static void submitBlueFlame(doca_gpu_dev_verbs_qp* q, uint64_t prodIndex) {
    const uint4* source = reinterpret_cast<const uint4*>(doca_gpu_dev_verbs_get_wqe_ptr(q, prodIndex - 1));
    doca_gpu_dev_verbs_lock<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(&q->sq_lock);
    const uint64_t previous =
        doca_gpu_dev_verbs_atomic_max<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU, true>(&q->sq_wqe_pi,
                                                                                                    prodIndex);
    if (previous < prodIndex) {
      uint4 wqe[4];
#pragma unroll
      for (int index = 0; index < 4; ++index) wqe[index] = source[index];
      uint4* bf = reinterpret_cast<uint4*>(doca_gpu_dev_verbs_load_const((uintptr_t*)&q->sq_db));
      doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
#pragma unroll
      for (int index = 0; index < 4; ++index) bf[index] = wqe[index];
      doca_priv_gpu_dev_verbs_update_dbr<DOCA_GPUNETIO_VERBS_QP_SQ>(q, prodIndex);
      doca_gpu_dev_verbs_ring_db<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>(q, prodIndex);
    }
    doca_gpu_dev_verbs_unlock<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(&q->sq_lock);
  }
#endif  // defined(MSCCLPP_USE_GPUNETIO)
#endif  // defined(MSCCLPP_DEVICE_COMPILE)
};

class EpGpuNetIoService {
 public:
  EpGpuNetIoService(std::shared_ptr<Bootstrap> bootstrap, const std::string& devices, int cudaDevice);
  ~EpGpuNetIoService();
  EpGpuNetIoService(const EpGpuNetIoService&) = delete;
  EpGpuNetIoService& operator=(const EpGpuNetIoService&) = delete;
  void setup(void* buffer, size_t bytes);
  /// Register the symmetric offsets of the EP flag buffers whose increments may become inline signals.
  /// Pass EpGpuNetIoDeviceContext::NoOffset for an absent buffer.
  void setSignalRegions(uint64_t flagsOffset, uint64_t flagsBytes, uint64_t syncOffset);
  EpGpuNetIoDeviceContext* deviceContext() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mscclpp::ep

#endif