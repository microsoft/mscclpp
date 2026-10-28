// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_PORT_CHANNEL_GPUNETIO_DEVICE_HPP_
#define MSCCLPP_PORT_CHANNEL_GPUNETIO_DEVICE_HPP_

#include <cstdint>

#include "assert_device.hpp"
#include "device.hpp"

#if defined(MSCCLPP_DEVICE_COMPILE) && defined(MSCCLPP_USE_GPUNETIO)
#include <doca_gpunetio_device.h>

#include "bulk_device.hpp"
#endif

namespace mscclpp {

#if defined(MSCCLPP_DEVICE_COMPILE) && defined(MSCCLPP_USE_GPUNETIO)
namespace detail {
// Locks for the shared-memory WQE slot pair used by connections without a dedicated slot, hashed by grid and block.
#if defined(__CUDACC_RDC__)
inline
#else
static
#endif
    __device__ unsigned int gpuNetIoSharedWqeLocks[1024] = {};
}  // namespace detail
#endif

/// A registered memory region that a GPUNetIO PortChannel can address by MemoryId.
struct GpuNetIoMemoryDeviceHandle {
  /// Base address of the region in the owner's address space.
  uint64_t addr;
  /// Size of the region in bytes. Zero marks a MemoryId that is not usable on this channel.
  uint64_t size;
  /// Local key (local regions) or remote key (remote regions), in network byte order.
  uint32_t key;
  /// Nonzero if the region belongs to the local process.
  uint32_t isLocal;
};

/// Device-side state of a PortChannel whose connection uses EndpointConfig::Ib::Mode::GpuNetIo.
/// The calling GPU thread builds work requests and rings the NIC doorbell itself (DOCA GPUNetIO).
/// Callers must make source data ready before put operations; DOCA orders queue publication.
/// All operations on a QP must be serialized, including across channels and submitting threads.
struct GpuNetIoDeviceHandle {
  /// GPU-resident DOCA queue pair (`doca_gpu_dev_verbs_qp*`) of the connection.
  void* qp;
  /// Remote semaphore token targeted by signals.
  uint64_t signalAddr;
  /// Remote key of the semaphore token, in network byte order.
  uint32_t signalKey;
  /// Local key of the connection's atomic fetch-result scratch, in network byte order.
  uint32_t atomicScratchKey;
  /// Connection-owned 8-byte buffer that receives atomic fetch results.
  uint64_t atomicScratchAddr;
  /// Device-resident outbound semaphore counter for inline signal writes.
  uint64_t* outboundToken;
  /// Snapshot of the ProxyService memory table, indexed by MemoryId.
  const GpuNetIoMemoryDeviceHandle* memories;
  /// Number of entries in `memories`.
  uint32_t numMemories;
  /// Shared-memory WQE slot pair of the connection. Values below kNumDedicatedWqeSlots are unique among the live
  /// GPUNetIO connections of the process and used without locking; kSharedWqeSlot is shared by all other
  /// connections and guarded by a per-block lock.
  uint32_t wqeSlot;

  /// Number of shared-memory WQE slot pairs reserved for one connection each.
  static constexpr uint32_t kNumDedicatedWqeSlots = 63;
  /// The slot pair used, under a lock, by connections that did not get a dedicated slot.
  static constexpr uint32_t kSharedWqeSlot = kNumDedicatedWqeSlots;

#if defined(MSCCLPP_DEVICE_COMPILE)
  MSCCLPP_DEVICE_INLINE static void trap() {
#if defined(MSCCLPP_DEVICE_CUDA)
    __trap();
#else
    __builtin_trap();
#endif
  }

  /// Trap on an invalid access even when device assertions are disabled.
  MSCCLPP_DEVICE_INLINE static void require(bool valid) {
    if (!valid) {
      MSCCLPP_ASSERT_DEVICE(false, "Invalid GPUNetIO PortChannel memory access");
      trap();
    }
  }

  /// Resolve a MemoryId and check its owner and bounds.
  MSCCLPP_DEVICE_INLINE GpuNetIoMemoryDeviceHandle memory(uint32_t id, bool local, uint64_t offset,
                                                          uint64_t bytes) const {
    require(id < numMemories);
    const GpuNetIoMemoryDeviceHandle mem = memories[id];
    require((mem.isLocal != 0) == local && offset <= mem.size && bytes <= mem.size - offset);
    return mem;
  }

  /// RDMA write of `size` bytes from a local region to a remote region.
  MSCCLPP_DEVICE_INLINE void put(const GpuNetIoMemoryDeviceHandle& dst, uint64_t dstOffset,
                                 const GpuNetIoMemoryDeviceHandle& src, uint64_t srcOffset, uint64_t size) {
#if defined(MSCCLPP_USE_GPUNETIO)
    putImpl(dst, dstOffset, src, srcOffset, size);
#else
    trap();
#endif
  }

  /// Increment the outbound semaphore counter and inline-write the new token value to the remote semaphore.
  MSCCLPP_DEVICE_INLINE void signal() {
#if defined(MSCCLPP_USE_GPUNETIO)
    const uint64_t token = ::atomicAdd(reinterpret_cast<unsigned long long*>(outboundToken), 1ULL) + 1ULL;
    signalAddrKey(signalAddr, signalKey, token);
#else
    trap();
#endif
  }

  /// Write an 8-byte strong-ordered inline signal `token` to `dst` at `dstOffset`.
  MSCCLPP_DEVICE_INLINE void signalAt(const GpuNetIoMemoryDeviceHandle& dst, uint64_t dstOffset, uint64_t token) {
#if defined(MSCCLPP_USE_GPUNETIO)
    require((dst.addr + dstOffset) % sizeof(uint64_t) == 0);
    signalAddrKey(dst.addr + dstOffset, dst.key, token);
#else
    trap();
#endif
  }

  /// RDMA write followed by an inline semaphore token write, ordered after the payload write.
  MSCCLPP_DEVICE_INLINE void putWithSignal(const GpuNetIoMemoryDeviceHandle& dst, uint64_t dstOffset,
                                           const GpuNetIoMemoryDeviceHandle& src, uint64_t srcOffset, uint64_t size) {
#if defined(MSCCLPP_USE_GPUNETIO)
    if (size > 0 && size <= DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE) {
      const uint64_t token = ::atomicAdd(reinterpret_cast<unsigned long long*>(outboundToken), 1ULL) + 1ULL;
      putWithSignalAddrKey(dst.addr + dstOffset, dst.key, src.addr + srcOffset, src.key, size, signalAddr, signalKey,
                           token);
      return;
    }
    putImpl(dst, dstOffset, src, srcOffset, size);
    signal();
#else
    trap();
#endif
  }

  /// RDMA write followed by an ordered inline signal `token` write to `sig` at `sigOffset`.
  MSCCLPP_DEVICE_INLINE void putWithSignalAt(const GpuNetIoMemoryDeviceHandle& dst, uint64_t dstOffset,
                                             const GpuNetIoMemoryDeviceHandle& src, uint64_t srcOffset, uint64_t size,
                                             const GpuNetIoMemoryDeviceHandle& sig, uint64_t sigOffset,
                                             uint64_t token) {
#if defined(MSCCLPP_USE_GPUNETIO)
    require((sig.addr + sigOffset) % sizeof(uint64_t) == 0);
    if (size > 0 && size <= DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE) {
      putWithSignalAddrKey(dst.addr + dstOffset, dst.key, src.addr + srcOffset, src.key, size, sig.addr + sigOffset,
                           sig.key, token);
      return;
    }
    putImpl(dst, dstOffset, src, srcOffset, size);
    signalAddrKey(sig.addr + sigOffset, sig.key, token);
#else
    trap();
#endif
  }

  /// Post three RDMA writes in a single batched doorbell submission.
  MSCCLPP_DEVICE_INLINE void putBatched3(const GpuNetIoMemoryDeviceHandle (&dst)[3], const uint64_t (&dstOffsets)[3],
                                         const GpuNetIoMemoryDeviceHandle (&src)[3], const uint64_t (&srcOffsets)[3],
                                         const uint64_t (&sizes)[3]) {
#if defined(MSCCLPP_USE_GPUNETIO)
    for (int i = 0; i < 3; ++i) {
      require(sizes[i] <= DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE);
    }
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    constexpr auto sharing = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE;
    const uint64_t base =
        doca_gpu_dev_verbs_reserve_wq_slots<sharing, DOCA_GPUNETIO_VERBS_QP_SQ, DOCA_GPUNETIO_VERBS_CQ_64B>(q, 3);
    withSharedWqe([&](doca_gpu_dev_verbs_wqe* sharedWqe) {
      prepareWriteWqe(q, &sharedWqe[0], base, dst[0].addr + dstOffsets[0], dst[0].key, src[0].addr + srcOffsets[0],
                      src[0].key, sizes[0]);
      copyWqeToGlobal(doca_gpu_dev_verbs_get_wqe_ptr(q, base), &sharedWqe[0]);

      prepareWriteWqe(q, &sharedWqe[1], base + 1, dst[1].addr + dstOffsets[1], dst[1].key, src[1].addr + srcOffsets[1],
                      src[1].key, sizes[1]);
      copyWqeToGlobal(doca_gpu_dev_verbs_get_wqe_ptr(q, base + 1), &sharedWqe[1]);

      prepareWriteWqe(q, &sharedWqe[1], base + 2, dst[2].addr + dstOffsets[2], dst[2].key, src[2].addr + srcOffsets[2],
                      src[2].key, sizes[2]);
      copyWqeToGlobal(doca_gpu_dev_verbs_get_wqe_ptr(q, base + 2), &sharedWqe[1]);
      doca_gpu_dev_verbs_mark_wqes_ready<sharing>(q, base, base + 2);
      submitPreparedFromShared(q, base + 2, &sharedWqe[0]);
    });
#else
    trap();
#endif
  }

  /// Remote 64-bit atomic add.
  MSCCLPP_DEVICE_INLINE void atomicAdd(uint64_t addr, uint32_t key, int64_t value) {
#if defined(MSCCLPP_USE_GPUNETIO)
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    constexpr auto sharing = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE;
    const auto index =
        doca_gpu_dev_verbs_reserve_wq_slots<sharing, DOCA_GPUNETIO_VERBS_QP_SQ, DOCA_GPUNETIO_VERBS_CQ_64B>(q, 1);
    withSharedWqe([&](doca_gpu_dev_verbs_wqe* sharedWqe) {
      doca_gpu_dev_verbs_wqe_prepare_atomic(q, sharedWqe, index, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
                                            DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, addr, key, atomicScratchAddr,
                                            atomicScratchKey, sizeof(uint64_t), static_cast<uint64_t>(value), 0);
      copyWqeToGlobal(doca_gpu_dev_verbs_get_wqe_ptr(q, index), sharedWqe);
      doca_gpu_dev_verbs_mark_wqes_ready<sharing>(q, index, index);
      submitPreparedFromShared(q, index, sharedWqe);
    });
#else
    trap();
#endif
  }

  /// Poll for completion of every WQE posted so far on the queue pair for at most `spins` iterations.
  /// @return 0 when drained, EBUSY if still pending, or the completion error status.
  MSCCLPP_DEVICE_INLINE int tryFlush(uint64_t spins) {
#if defined(MSCCLPP_USE_GPUNETIO)
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    const uint64_t ticket =
        doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
            &q->sq_rsvd_index);
    if (ticket == 0) return 0;
    for (uint64_t spin = 0; spin < spins; ++spin) {
      const int status =
          doca_gpu_dev_verbs_poll_one_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE,
                                            DOCA_GPUNETIO_VERBS_QP_SQ, DOCA_GPUNETIO_VERBS_CQ_64B>(q, ticket - 1);
      if (status != EBUSY) return status;
    }
    return EBUSY;
#else
    trap();
    return 0;
#endif
  }

  /// Send-queue reservation index: every WQE posted so far lies below it.
  MSCCLPP_DEVICE_INLINE uint64_t reservedTicket() const {
#if defined(MSCCLPP_USE_GPUNETIO)
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    return doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
        &q->sq_rsvd_index);
#else
    trap();
    return 0;
#endif
  }

  /// Return whether all WQEs below `ticket` have already been observed as completed in `cqe_ci`.
  MSCCLPP_DEVICE_INLINE bool isTicketDone(uint64_t ticket) const {
#if defined(MSCCLPP_USE_GPUNETIO)
    if (ticket == 0) return true;
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    return doca_gpu_dev_verbs_load_relaxed<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(&q->cq_sq.cqe_ci) >= ticket;
#else
    trap();
    return false;
#endif
  }

  /// Wait until every WQE below `ticket` (from `reservedTicket()`) has completed.
  MSCCLPP_DEVICE_INLINE void waitTicket(uint64_t ticket) {
#if defined(MSCCLPP_USE_GPUNETIO)
    if (ticket == 0) return;
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    int status;
    do {
      status = doca_gpu_dev_verbs_poll_one_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE,
                                                 DOCA_GPUNETIO_VERBS_QP_SQ, DOCA_GPUNETIO_VERBS_CQ_64B>(q, ticket - 1);
    } while (status == EBUSY);
    require(status == 0);
#else
    trap();
#endif
  }

  /// Wait until every work request posted so far on the queue pair has completed.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  MSCCLPP_DEVICE_INLINE void flush([[maybe_unused]] int64_t maxSpinCount) {
#if defined(MSCCLPP_USE_GPUNETIO)
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    const uint64_t ticket =
        doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
            &q->sq_rsvd_index);
    if (ticket == 0) return;
    int status;
    [[maybe_unused]] int64_t spinCount = 0;
    while ((status = doca_gpu_dev_verbs_poll_one_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE,
                                                       DOCA_GPUNETIO_VERBS_QP_SQ, DOCA_GPUNETIO_VERBS_CQ_64B>(
                q, ticket - 1)) == EBUSY) {
      MSCCLPP_ASSERT_DEVICE((maxSpinCount < 0 || spinCount++ != maxSpinCount), "GPUNetIO flush timed out");
    }
    if (status != 0) {
      MSCCLPP_ASSERT_DEVICE(false, "GPUNetIO completion queue reported an error");
      trap();
    }
#else
    trap();
#endif
  }

 private:
#if defined(MSCCLPP_USE_GPUNETIO)
  MSCCLPP_DEVICE_INLINE void signalAddrKey(uint64_t addr, uint32_t key, uint64_t token) {
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    constexpr auto sharing = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE;
    const auto index =
        doca_gpu_dev_verbs_reserve_wq_slots<sharing, DOCA_GPUNETIO_VERBS_QP_SQ, DOCA_GPUNETIO_VERBS_CQ_64B>(q, 1);
    withSharedWqe([&](doca_gpu_dev_verbs_wqe* sharedWqe) {
      prepareSignalWqeAt(q, sharedWqe, index, addr, key, token);
      copyWqeToGlobal(doca_gpu_dev_verbs_get_wqe_ptr(q, index), sharedWqe);
      doca_gpu_dev_verbs_mark_wqes_ready<sharing>(q, index, index);
      submitPreparedFromShared(q, index, sharedWqe);
    });
  }

  MSCCLPP_DEVICE_INLINE void putWithSignalAddrKey(uint64_t dstAddr, uint32_t dstKey, uint64_t srcAddr, uint32_t srcKey,
                                                  uint64_t size, uint64_t sigAddr, uint32_t sigKey, uint64_t token) {
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    constexpr auto sharing = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE;
    const auto index =
        doca_gpu_dev_verbs_reserve_wq_slots<sharing, DOCA_GPUNETIO_VERBS_QP_SQ, DOCA_GPUNETIO_VERBS_CQ_64B>(q, 2);
    withSharedWqe([&](doca_gpu_dev_verbs_wqe* sharedWqe) {
      prepareWriteWqe(q, &sharedWqe[0], index, dstAddr, dstKey, srcAddr, srcKey, size);
      if (size <= DOCA_GPUNETIO_VERBS_MAX_INLINE_SIZE) {
        submitInlineFromSharedNoWait(q, &sharedWqe[0]);
      } else {
        submitDmaFromSharedNoWait(q, &sharedWqe[0]);
      }
      copyWqeToGlobal(doca_gpu_dev_verbs_get_wqe_ptr(q, index), &sharedWqe[0]);
      prepareSignalWqeAt(q, &sharedWqe[1], index + 1, sigAddr, sigKey, token);
      copyWqeToGlobal(doca_gpu_dev_verbs_get_wqe_ptr(q, index + 1), &sharedWqe[1]);
      doca_gpu_dev_verbs_mark_wqes_ready<sharing>(q, index, index + 1);
      submitPreparedFromShared(q, index + 1, &sharedWqe[1]);
    });
  }

  MSCCLPP_DEVICE_INLINE void putImpl(const GpuNetIoMemoryDeviceHandle& dst, uint64_t dstOffset,
                                     const GpuNetIoMemoryDeviceHandle& src, uint64_t srcOffset, uint64_t size) {
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    constexpr auto sharing = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE;
    if (size == 0 || size > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE) {
      doca_gpu_dev_verbs_ticket_t ticket;
      doca_gpu_dev_verbs_put<sharing, DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB, DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD,
                             DOCA_GPUNETIO_VERBS_CQ_64B>(q, {dst.addr + dstOffset, dst.key},
                                                         {src.addr + srcOffset, src.key}, size, &ticket,
                                                         DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_DB_RINGING);
      submit(ticket);
      return;
    }

    const auto index =
        doca_gpu_dev_verbs_reserve_wq_slots<sharing, DOCA_GPUNETIO_VERBS_QP_SQ, DOCA_GPUNETIO_VERBS_CQ_64B>(q, 1);
    withSharedWqe([&](doca_gpu_dev_verbs_wqe* sharedWqe) {
      prepareWriteWqe(q, sharedWqe, index, dst.addr + dstOffset, dst.key, src.addr + srcOffset, src.key, size);
      copyWqeToGlobal(doca_gpu_dev_verbs_get_wqe_ptr(q, index), sharedWqe);
      doca_gpu_dev_verbs_mark_wqes_ready<sharing>(q, index, index);
      submitPreparedFromShared(q, index, sharedWqe);
    });
  }

  MSCCLPP_DEVICE_INLINE static void prepareSignalWqeAt(doca_gpu_dev_verbs_qp* q, doca_gpu_dev_verbs_wqe* wqePtr,
                                                       uint16_t wqeIdx, uint64_t addr, uint32_t key, uint64_t token) {
    constexpr auto ctrlFlags = static_cast<doca_gpu_dev_verbs_wqe_ctrl_flags>(
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE | DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_STRONG_ORDERING);
    doca_gpu_dev_verbs_prepare_inl_rdma_write_wqe_header(q, wqePtr, wqeIdx, ctrlFlags, addr, key, sizeof(uint64_t));
    doca_gpu_dev_verbs_prepare_inl_rdma_write_wqe_data<uint64_t>(q, wqePtr, token);
  }

  MSCCLPP_DEVICE_INLINE static void prepareWriteWqe(doca_gpu_dev_verbs_qp* q, doca_gpu_dev_verbs_wqe* wqePtr,
                                                    uint16_t wqeIdx, uint64_t dstAddr, uint32_t dstKey,
                                                    uint64_t srcAddr, uint32_t srcKey, uint64_t size) {
    if (size <= DOCA_GPUNETIO_VERBS_MAX_INLINE_SIZE) {
      // Inline copies the payload into the WQE; the NIC does not DMA-read the original source buffer.
      doca_gpu_dev_verbs_wqe_prepare_write_inl(q, wqePtr, wqeIdx, DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, dstAddr,
                                               dstKey, srcAddr, size);
    } else {
      doca_gpu_dev_verbs_wqe_prepare_write(q, wqePtr, wqeIdx, DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                                           DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0, dstAddr, dstKey, srcAddr,
                                           srcKey, size);
    }
  }

  MSCCLPP_DEVICE_INLINE static void copyWqeToGlobal(doca_gpu_dev_verbs_wqe* dstGlobal,
                                                    const doca_gpu_dev_verbs_wqe* src) {
    auto* dst = reinterpret_cast<uint64_t*>(dstGlobal);
    auto* s = const_cast<uint64_t*>(reinterpret_cast<const uint64_t*>(src));
    doca_gpu_dev_verbs_store_wqe_seg(dst + 0, s + 0);
    doca_gpu_dev_verbs_store_wqe_seg(dst + 2, s + 2);
    doca_gpu_dev_verbs_store_wqe_seg(dst + 4, s + 4);
    doca_gpu_dev_verbs_store_wqe_seg(dst + 6, s + 6);
  }

  MSCCLPP_DEVICE_INLINE static void submitPreparedFromShared(doca_gpu_dev_verbs_qp* q, uint64_t index,
                                                             doca_gpu_dev_verbs_wqe* sharedWqe) {
#if MSCCLPP_BULK_AVAILABLE
    bulkFence();
    doca_gpu_dev_verbs_submit_bf<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE,
                                 DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>(q, index + 1, sharedWqe);
    bulkStoreCommit();
    bulkStoreWaitSource();
#else
    MSCCLPP_ASSERT_DEVICE(false, "GPUNetIO BlueFlame requires device code compiled for sm_90 or newer");
    trap();
#endif
  }

  MSCCLPP_DEVICE_INLINE static void submitDmaFromSharedNoWait(doca_gpu_dev_verbs_qp* q,
                                                              doca_gpu_dev_verbs_wqe* sharedWqe) {
#if MSCCLPP_BULK_AVAILABLE
    bulkFence();
    doca_gpu_dev_verbs_ring_bf<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>(q, sharedWqe);
#else
    MSCCLPP_ASSERT_DEVICE(false, "GPUNetIO BlueFlame requires device code compiled for sm_90 or newer");
    trap();
#endif
  }

  MSCCLPP_DEVICE_INLINE static void submitInlineFromSharedNoWait(doca_gpu_dev_verbs_qp* q,
                                                                 doca_gpu_dev_verbs_wqe* sharedWqe) {
#if MSCCLPP_BULK_AVAILABLE
    bulkFence();
    void* bfPtr = reinterpret_cast<void*>(doca_gpu_dev_verbs_load_const(reinterpret_cast<uintptr_t*>(&q->sq_db)));
    asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], 64;" : : "l"(bfPtr), "l"(sharedWqe));
#else
    MSCCLPP_ASSERT_DEVICE(false, "GPUNetIO BlueFlame requires device code compiled for sm_90 or newer");
    trap();
#endif
  }

  // Per-block lock of the shared WQE slot pair, hashed by grid and block. The lock helpers are kept out of line to
  // limit register pressure in the inlined submission paths.
  MSCCLPP_DEVICE_INLINE static unsigned int* sharedWqeLock() {
    uint64_t gridId;
    asm volatile("mov.u64 %0, %%gridid;" : "=l"(gridId));
    const uint64_t block = blockIdx.x + uint64_t(gridDim.x) * (blockIdx.y + uint64_t(gridDim.y) * blockIdx.z);
    constexpr uint64_t numLocks = sizeof(detail::gpuNetIoSharedWqeLocks) / sizeof(detail::gpuNetIoSharedWqeLocks[0]);
    static_assert((numLocks & (numLocks - 1)) == 0, "the number of shared WQE slot locks must be a power of two");
    return &detail::gpuNetIoSharedWqeLocks[(gridId ^ (block * 67)) & (numLocks - 1)];
  }

  __device__ __noinline__ static void acquireSharedWqe() {
    unsigned int* lock = sharedWqeLock();
    while (atomicCAS(lock, 0u, 1u) != 0u) {
    }
    doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
  }

  __device__ __noinline__ static void releaseSharedWqe() {
    doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
    atomicExch(sharedWqeLock(), 0u);
  }

  template <typename Fn>
  MSCCLPP_DEVICE_INLINE void withSharedWqe(Fn&& fn) const {
#if MSCCLPP_BULK_AVAILABLE
    __shared__ __align__(16) doca_gpu_dev_verbs_wqe sharedWqes[(kNumDedicatedWqeSlots + 1) * 2];
    // Callers serialize all operations on a connection, so a dedicated slot has at most one user at a time, and every
    // submission waits for the asynchronous BlueFlame read of its slot before returning.
    const bool shared = wqeSlot >= kNumDedicatedWqeSlots;
    if (shared) acquireSharedWqe();
    fn(&sharedWqes[(shared ? kSharedWqeSlot : wqeSlot) * 2]);
    if (shared) releaseSharedWqe();
#else
    MSCCLPP_ASSERT_DEVICE(false, "GPUNetIO BlueFlame requires device code compiled for sm_90 or newer");
    trap();
#endif
  }

  MSCCLPP_DEVICE_INLINE void submit(uint64_t index) {
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    withSharedWqe([&](doca_gpu_dev_verbs_wqe* sharedWqe) {
      doca_gpu_dev_verbs_memcpy_data(sharedWqe, doca_gpu_dev_verbs_get_wqe_ptr(q, index), sizeof(*sharedWqe));
      submitPreparedFromShared(q, index, sharedWqe);
    });
  }
#endif  // defined(MSCCLPP_USE_GPUNETIO)
#endif  // defined(MSCCLPP_DEVICE_COMPILE)
};

}  // namespace mscclpp

#endif  // MSCCLPP_PORT_CHANNEL_GPUNETIO_DEVICE_HPP_
