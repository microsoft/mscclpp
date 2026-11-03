// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_GPUNETIO_DEVICE_HPP_
#define MSCCLPP_GPUNETIO_DEVICE_HPP_

#include <cstdint>

#include "assert_device.hpp"
#include "device.hpp"

#if defined(MSCCLPP_DEVICE_COMPILE)
#include "atomic_device.hpp"
#endif

#if defined(MSCCLPP_DEVICE_COMPILE) && defined(MSCCLPP_USE_GPUNETIO)
#include <doca_gpunetio_device.h>

#include "bulk_device.hpp"
#endif

namespace mscclpp {

/// Caller-owned shared-memory storage for a single 64-byte BlueFlame work request (WQE).
/// Concurrent submitters must use separate instances unless guarded by a lock. A posting operation waits until
/// BlueFlame has consumed this storage before returning, so sequential operations may reuse it even across different
/// queue pairs.
struct alignas(16) GpuNetIoWqeSlot {
  /// Opaque work-request storage; callers must not access its contents.
  uint64_t storage[8];
};

/// A registered memory region that a GPUNetIO PortChannel can address.
struct GpuNetIoMemoryDeviceHandle {
  /// Base address of the region in the owner's address space.
  uint64_t addr;
  /// Local key (local regions) or remote key (remote regions), in network byte order.
  uint32_t key;
};

/// Device-side state of a PortChannel whose connection uses EndpointConfig::Ib::Mode::Device.
/// The calling GPU thread builds work requests and rings the NIC doorbell itself (DOCA GPUNetIO).
/// Callers must make source data ready before put operations; DOCA orders queue publication.
struct GpuNetIoDeviceHandle {
  /// GPU-resident DOCA queue pair (`doca_gpu_dev_verbs_qp*`) of the connection.
  void* qp;
  /// Local key of the connection's atomic fetch-result buffer, in network byte order.
  uint32_t atomicFetchKey;
  /// Connection-owned 8-byte buffer that receives RDMA atomic fetch results.
  uint64_t atomicFetchAddr;
  /// Maximum payload size (in bytes) submitted via BlueFlame; larger payloads use regular doorbell submission
  /// without shared-memory WQE staging. Zero disables BlueFlame.
  uint32_t blueFlameLimit;

  /// Default BlueFlame payload size limit (4 KiB).
  static constexpr uint32_t kDefaultBlueFlameLimit = 4096;

 private:
  GpuNetIoWqeSlot* wqeSlotShm_;
  uint32_t* wqeSlotLock_;
  bool reuseSharedWqe_;

 public:
#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Bind a caller-owned shared-memory WQE slot for BlueFlame submissions.
  /// In non-Exclusive sharing modes or when `reuseSharedWqe` is true, `wqeSlotLock` must be non-null; otherwise
  /// BlueFlame is disabled and operations fall back to regular doorbell submission.
  template <PortChannelSharingMode Mode = PortChannelSharingMode::Gpu>
  MSCCLPP_DEVICE_INLINE void setWqeSlot(GpuNetIoWqeSlot* wqeSlotShm, uint32_t* wqeSlotLock = nullptr,
                                        bool reuseSharedWqe = false) {
    const bool needLock = (Mode != PortChannelSharingMode::Exclusive) || reuseSharedWqe;
    if (needLock && wqeSlotLock == nullptr) {
      wqeSlotShm_ = nullptr;
      wqeSlotLock_ = nullptr;
      reuseSharedWqe_ = false;
    } else {
      wqeSlotShm_ = wqeSlotShm;
      wqeSlotLock_ = wqeSlotLock;
      reuseSharedWqe_ = needLock;
    }
  }

  /// RDMA write of `size` bytes from a local region to a remote region.
  template <PortChannelSharingMode Mode = PortChannelSharingMode::Gpu>
  MSCCLPP_DEVICE_INLINE void put([[maybe_unused]] const GpuNetIoMemoryDeviceHandle& dst,
                                 [[maybe_unused]] uint64_t dstOffset,
                                 [[maybe_unused]] const GpuNetIoMemoryDeviceHandle& src,
                                 [[maybe_unused]] uint64_t srcOffset, [[maybe_unused]] uint64_t size) {
#if defined(MSCCLPP_USE_GPUNETIO)
    putImpl<docaSharingMode<Mode>()>(dst, dstOffset, src, srcOffset, size);
#else
    MSCCLPP_ASSERT_DEVICE(false, "GPUNetIO is not enabled");
#endif
  }

  /// Increment `*outboundToken` in queue order and inline-write the new value to the 8-byte token `dst`.
  template <PortChannelSharingMode Mode = PortChannelSharingMode::Gpu>
  MSCCLPP_DEVICE_INLINE void signal([[maybe_unused]] const GpuNetIoMemoryDeviceHandle& dst,
                                    [[maybe_unused]] uint64_t* outboundToken) {
#if defined(MSCCLPP_USE_GPUNETIO)
    signalAddrKey<docaSharingMode<Mode>()>(dst.addr, dst.key, 0, outboundToken);
#else
    MSCCLPP_ASSERT_DEVICE(false, "GPUNetIO is not enabled");
#endif
  }

  /// Write an 8-byte strong-ordered inline signal `token` to `dst` at `dstOffset`.
  template <PortChannelSharingMode Mode = PortChannelSharingMode::Gpu>
  MSCCLPP_DEVICE_INLINE void signalAt([[maybe_unused]] const GpuNetIoMemoryDeviceHandle& dst,
                                      [[maybe_unused]] uint64_t dstOffset, [[maybe_unused]] uint64_t token) {
#if defined(MSCCLPP_USE_GPUNETIO)
    MSCCLPP_ASSERT_DEVICE((dst.addr + dstOffset) % sizeof(uint64_t) == 0, "signalAt address must be 8-byte aligned");
    signalAddrKey<docaSharingMode<Mode>()>(dst.addr + dstOffset, dst.key, token);
#else
    MSCCLPP_ASSERT_DEVICE(false, "GPUNetIO is not enabled");
#endif
  }

  /// RDMA write followed by a signal to the token `sig`, ordered after the payload write. See signal().
  template <PortChannelSharingMode Mode = PortChannelSharingMode::Gpu>
  MSCCLPP_DEVICE_INLINE void putWithSignal([[maybe_unused]] const GpuNetIoMemoryDeviceHandle& dst,
                                           [[maybe_unused]] uint64_t dstOffset,
                                           [[maybe_unused]] const GpuNetIoMemoryDeviceHandle& src,
                                           [[maybe_unused]] uint64_t srcOffset, [[maybe_unused]] uint64_t size,
                                           [[maybe_unused]] const GpuNetIoMemoryDeviceHandle& sig,
                                           [[maybe_unused]] uint64_t* outboundToken) {
#if defined(MSCCLPP_USE_GPUNETIO)
    if (size > 0 && size <= DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE) {
      putWithSignalAddrKey<docaSharingMode<Mode>()>(dst.addr + dstOffset, dst.key, src.addr + srcOffset, src.key, size,
                                                    sig.addr, sig.key, 0, outboundToken);
      return;
    }
    putImpl<docaSharingMode<Mode>()>(dst, dstOffset, src, srcOffset, size);
    signal<Mode>(sig, outboundToken);
#else
    MSCCLPP_ASSERT_DEVICE(false, "GPUNetIO is not enabled");
#endif
  }

  /// RDMA write followed by a signal to the token `sig` and local completion flush. See signal().
  template <PortChannelSharingMode Mode = PortChannelSharingMode::Gpu>
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(const GpuNetIoMemoryDeviceHandle& dst, uint64_t dstOffset,
                                                   const GpuNetIoMemoryDeviceHandle& src, uint64_t srcOffset,
                                                   uint64_t size, const GpuNetIoMemoryDeviceHandle& sig,
                                                   uint64_t* outboundToken, int64_t maxSpinCount = 1000000) {
    putWithSignal<Mode>(dst, dstOffset, src, srcOffset, size, sig, outboundToken);
    flush<Mode>(maxSpinCount);
  }

  /// RDMA write followed by an ordered inline signal `token` write to `sig` at `sigOffset`.
  template <PortChannelSharingMode Mode = PortChannelSharingMode::Gpu>
  MSCCLPP_DEVICE_INLINE void putWithSignalAt([[maybe_unused]] const GpuNetIoMemoryDeviceHandle& dst,
                                             [[maybe_unused]] uint64_t dstOffset,
                                             [[maybe_unused]] const GpuNetIoMemoryDeviceHandle& src,
                                             [[maybe_unused]] uint64_t srcOffset, [[maybe_unused]] uint64_t size,
                                             [[maybe_unused]] const GpuNetIoMemoryDeviceHandle& sig,
                                             [[maybe_unused]] uint64_t sigOffset, [[maybe_unused]] uint64_t token) {
#if defined(MSCCLPP_USE_GPUNETIO)
    MSCCLPP_ASSERT_DEVICE((sig.addr + sigOffset) % sizeof(uint64_t) == 0,
                          "putWithSignalAt signal address must be 8-byte aligned");
    if (size > 0 && size <= DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE) {
      putWithSignalAddrKey<docaSharingMode<Mode>()>(dst.addr + dstOffset, dst.key, src.addr + srcOffset, src.key, size,
                                                    sig.addr + sigOffset, sig.key, token);
      return;
    }
    putImpl<docaSharingMode<Mode>()>(dst, dstOffset, src, srcOffset, size);
    signalAddrKey<docaSharingMode<Mode>()>(sig.addr + sigOffset, sig.key, token);
#else
    MSCCLPP_ASSERT_DEVICE(false, "GPUNetIO is not enabled");
#endif
  }

  /// Remote 64-bit atomic add into `dst` at `dstOffset`.
  template <PortChannelSharingMode Mode = PortChannelSharingMode::Gpu>
  MSCCLPP_DEVICE_INLINE void accumulate([[maybe_unused]] const GpuNetIoMemoryDeviceHandle& dst,
                                        [[maybe_unused]] uint64_t dstOffset, [[maybe_unused]] int64_t value) {
#if defined(MSCCLPP_USE_GPUNETIO)
    const uint64_t addr = dst.addr + dstOffset;
    MSCCLPP_ASSERT_DEVICE(addr % sizeof(uint64_t) == 0, "accumulate address must be 8-byte aligned");
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    constexpr auto sharing = docaSharingMode<Mode>();
    if (useBlueFlame(sizeof(uint64_t))) {
      submitAtomicBlueFlame<sharing>(q, addr, dst.key, value);
    } else {
      const auto index =
          doca_gpu_dev_verbs_reserve_wq_slots<sharing, DOCA_GPUNETIO_VERBS_QP_SQ, DOCA_GPUNETIO_VERBS_CQ_64B>(q, 1);
      auto* wqePtr = doca_gpu_dev_verbs_get_wqe_ptr(q, index);
      doca_gpu_dev_verbs_wqe_prepare_atomic(q, wqePtr, index, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
                                            DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, addr, dst.key, atomicFetchAddr,
                                            atomicFetchKey, sizeof(uint64_t), static_cast<uint64_t>(value), 0);
      doca_gpu_dev_verbs_mark_wqes_ready<sharing>(q, index, index);
      submitDb<sharing>(q, index + 1);
    }
#else
    MSCCLPP_ASSERT_DEVICE(false, "GPUNetIO is not enabled");
#endif
  }

  /// Wait until every work request posted so far on the queue pair has completed.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  template <PortChannelSharingMode Mode = PortChannelSharingMode::Gpu>
  MSCCLPP_DEVICE_INLINE void flush([[maybe_unused]] int64_t maxSpinCount) {
#if defined(MSCCLPP_USE_GPUNETIO)
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    constexpr auto sharing = docaSharingMode<Mode>();
    const uint64_t ticket = doca_gpu_dev_verbs_atomic_read<uint64_t, sharing>(&q->sq_rsvd_index);
    if (ticket == 0) return;
    [[maybe_unused]] int status;
    [[maybe_unused]] int64_t spinCount = 0;
    while ((status = doca_gpu_dev_verbs_poll_one_cq_at<sharing, DOCA_GPUNETIO_VERBS_QP_SQ, DOCA_GPUNETIO_VERBS_CQ_64B>(
                q, ticket - 1)) == EBUSY) {
      MSCCLPP_ASSERT_DEVICE((maxSpinCount < 0 || spinCount++ != maxSpinCount), "GPUNetIO flush timed out");
    }
    MSCCLPP_ASSERT_DEVICE(status == 0, "GPUNetIO completion queue error");
#else
    MSCCLPP_ASSERT_DEVICE(false, "GPUNetIO is not enabled");
#endif
  }

 private:
#if defined(MSCCLPP_USE_GPUNETIO)
  template <PortChannelSharingMode Mode>
  MSCCLPP_HOST_DEVICE_INLINE static constexpr doca_gpu_dev_verbs_resource_sharing_mode docaSharingMode() {
    if constexpr (Mode == PortChannelSharingMode::Exclusive) {
      return DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE;
    } else if constexpr (Mode == PortChannelSharingMode::Cta) {
      return DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA;
    } else {
      return DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU;
    }
  }

  MSCCLPP_DEVICE_INLINE bool useBlueFlame(uint64_t size) const {
#if MSCCLPP_BULK_AVAILABLE
    return wqeSlotShm_ != nullptr && size <= blueFlameLimit;
#else
    (void)size;
    return false;
#endif
  }

  template <doca_gpu_dev_verbs_resource_sharing_mode Sharing>
  MSCCLPP_DEVICE_INLINE static void submitDb(doca_gpu_dev_verbs_qp* q, uint64_t prodIndex) {
    constexpr auto syncScope = (Sharing == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU)
                                   ? DOCA_GPUNETIO_VERBS_SYNC_SCOPE_THREAD
                                   : DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU;
    doca_gpu_dev_verbs_submit_db<Sharing, syncScope, DOCA_GPUNETIO_VERBS_QP_SQ>(q, prodIndex);
  }

  struct WqeSlotLockGuard {
    uint32_t* lock;

    MSCCLPP_DEVICE_INLINE explicit WqeSlotLockGuard(const GpuNetIoDeviceHandle& handle)
        : lock(handle.reuseSharedWqe_ ? handle.wqeSlotLock_ : nullptr) {
      if (!lock) return;
      uint32_t expected = 0;
      while (!cuda::atomic_ref<uint32_t, cuda::thread_scope_block>{*lock}.compare_exchange_weak(
          expected, 1u, memoryOrderAcquire, memoryOrderRelaxed)) {
        expected = 0;
      }
    }

    MSCCLPP_DEVICE_INLINE ~WqeSlotLockGuard() {
      if (lock) cuda::atomic_ref<uint32_t, cuda::thread_scope_block>{*lock}.store(0u, memoryOrderRelease);
    }

    WqeSlotLockGuard(const WqeSlotLockGuard&) = delete;
    WqeSlotLockGuard& operator=(const WqeSlotLockGuard&) = delete;
  };

  template <doca_gpu_dev_verbs_resource_sharing_mode Sharing>
  MSCCLPP_DEVICE_INLINE void signalAddrKey(uint64_t addr, uint32_t key, uint64_t token,
                                           uint64_t* outboundToken = nullptr) {
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    if (useBlueFlame(sizeof(uint64_t))) {
      submitSignalBlueFlame<Sharing>(q, addr, key, token, outboundToken);
    } else {
      const auto index =
          doca_gpu_dev_verbs_reserve_wq_slots<Sharing, DOCA_GPUNETIO_VERBS_QP_SQ, DOCA_GPUNETIO_VERBS_CQ_64B>(q, 1);
      auto* wqePtr = doca_gpu_dev_verbs_get_wqe_ptr(q, index);
      prepareSignalWqeAt<Sharing>(q, wqePtr, index, index, addr, key, token, outboundToken);
      doca_gpu_dev_verbs_mark_wqes_ready<Sharing>(q, index, index);
      submitDb<Sharing>(q, index + 1);
    }
  }

  template <doca_gpu_dev_verbs_resource_sharing_mode Sharing>
  MSCCLPP_DEVICE_INLINE void putWithSignalAddrKey(uint64_t dstAddr, uint32_t dstKey, uint64_t srcAddr, uint32_t srcKey,
                                                  uint64_t size, uint64_t sigAddr, uint32_t sigKey, uint64_t token,
                                                  uint64_t* outboundToken = nullptr) {
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    if (useBlueFlame(size)) {
      submitPutWithSignalBlueFlame<Sharing>(q, dstAddr, dstKey, srcAddr, srcKey, size, sigAddr, sigKey, token,
                                            outboundToken);
    } else {
      const auto index =
          doca_gpu_dev_verbs_reserve_wq_slots<Sharing, DOCA_GPUNETIO_VERBS_QP_SQ, DOCA_GPUNETIO_VERBS_CQ_64B>(q, 2);
      submitPutWithSignalDb<Sharing>(q, index, dstAddr, dstKey, srcAddr, srcKey, size, sigAddr, sigKey, token,
                                     outboundToken);
    }
  }

  template <doca_gpu_dev_verbs_resource_sharing_mode Sharing>
  MSCCLPP_DEVICE_INLINE void putImpl(const GpuNetIoMemoryDeviceHandle& dst, uint64_t dstOffset,
                                     const GpuNetIoMemoryDeviceHandle& src, uint64_t srcOffset, uint64_t size) {
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    if (size == 0 || size > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE) {
      doca_gpu_dev_verbs_ticket_t ticket;
      doca_gpu_dev_verbs_put<Sharing, DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB, DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD,
                             DOCA_GPUNETIO_VERBS_CQ_64B>(q, {dst.addr + dstOffset, dst.key},
                                                         {src.addr + srcOffset, src.key}, size, &ticket);
      return;
    }

    if (useBlueFlame(size)) {
      submitPutBlueFlame<Sharing>(q, dst.addr + dstOffset, dst.key, src.addr + srcOffset, src.key, size);
    } else {
      const auto index =
          doca_gpu_dev_verbs_reserve_wq_slots<Sharing, DOCA_GPUNETIO_VERBS_QP_SQ, DOCA_GPUNETIO_VERBS_CQ_64B>(q, 1);
      submitPutDb<Sharing>(q, index, dst.addr + dstOffset, dst.key, src.addr + srcOffset, src.key, size);
    }
  }

  template <doca_gpu_dev_verbs_resource_sharing_mode Sharing>
  MSCCLPP_DEVICE_INLINE void submitAtomicBlueFlame(doca_gpu_dev_verbs_qp* q, uint64_t addr, uint32_t key,
                                                   int64_t value) {
    withWqeSlot<Sharing>(q, 1, [&](doca_gpu_dev_verbs_wqe* sharedWqe, uint64_t index) {
      doca_gpu_dev_verbs_wqe_prepare_atomic(q, sharedWqe, index, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
                                            DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, addr, key, atomicFetchAddr,
                                            atomicFetchKey, sizeof(uint64_t), static_cast<uint64_t>(value), 0);
      copyWqeToGlobal(doca_gpu_dev_verbs_get_wqe_ptr(q, index), sharedWqe);
      doca_gpu_dev_verbs_mark_wqes_ready<Sharing>(q, index, index);
      submitPreparedFromShared<Sharing>(q, index, sharedWqe);
    });
  }

  template <doca_gpu_dev_verbs_resource_sharing_mode Sharing>
  MSCCLPP_DEVICE_INLINE void submitSignalBlueFlame(doca_gpu_dev_verbs_qp* q, uint64_t addr, uint32_t key,
                                                   uint64_t token, uint64_t* outboundToken) {
    withWqeSlot<Sharing>(q, 1, [&](doca_gpu_dev_verbs_wqe* sharedWqe, uint64_t index) {
      prepareSignalWqeAt<Sharing>(q, sharedWqe, index, index, addr, key, token, outboundToken);
      copyWqeToGlobal(doca_gpu_dev_verbs_get_wqe_ptr(q, index), sharedWqe);
      doca_gpu_dev_verbs_mark_wqes_ready<Sharing>(q, index, index);
      submitPreparedFromShared<Sharing>(q, index, sharedWqe);
    });
  }

  template <doca_gpu_dev_verbs_resource_sharing_mode Sharing>
  MSCCLPP_DEVICE_INLINE void submitPutBlueFlame(doca_gpu_dev_verbs_qp* q, uint64_t dstAddr, uint32_t dstKey,
                                                uint64_t srcAddr, uint32_t srcKey, uint64_t size) {
    withWqeSlot<Sharing>(q, 1, [&](doca_gpu_dev_verbs_wqe* sharedWqe, uint64_t index) {
      prepareWriteWqe(q, sharedWqe, index, dstAddr, dstKey, srcAddr, srcKey, size);
      copyWqeToGlobal(doca_gpu_dev_verbs_get_wqe_ptr(q, index), sharedWqe);
      doca_gpu_dev_verbs_mark_wqes_ready<Sharing>(q, index, index);
      submitPreparedFromShared<Sharing>(q, index, sharedWqe);
    });
  }

  template <doca_gpu_dev_verbs_resource_sharing_mode Sharing>
  __device__ __noinline__ static void submitPutDb(doca_gpu_dev_verbs_qp* q, uint64_t index, uint64_t dstAddr,
                                                  uint32_t dstKey, uint64_t srcAddr, uint32_t srcKey, uint64_t size) {
    auto* wqePtr = doca_gpu_dev_verbs_get_wqe_ptr(q, index);
    prepareWriteWqe(q, wqePtr, index, dstAddr, dstKey, srcAddr, srcKey, size);
    doca_gpu_dev_verbs_mark_wqes_ready<Sharing>(q, index, index);
    submitDb<Sharing>(q, index + 1);
  }

  template <doca_gpu_dev_verbs_resource_sharing_mode Sharing>
  MSCCLPP_DEVICE_INLINE void submitPutWithSignalBlueFlame(doca_gpu_dev_verbs_qp* q, uint64_t dstAddr, uint32_t dstKey,
                                                          uint64_t srcAddr, uint32_t srcKey, uint64_t size,
                                                          uint64_t sigAddr, uint32_t sigKey, uint64_t token,
                                                          uint64_t* outboundToken) {
    withWqeSlot<Sharing>(q, 2, [&](doca_gpu_dev_verbs_wqe* sharedWqe, uint64_t index) {
      prepareWriteWqe(q, sharedWqe, index, dstAddr, dstKey, srcAddr, srcKey, size);
      copyWqeToGlobal(doca_gpu_dev_verbs_get_wqe_ptr(q, index), sharedWqe);
      if constexpr (Sharing == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE) {
#if MSCCLPP_BULK_AVAILABLE
        bulkFence();
        doca_gpu_dev_verbs_ring_bf<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>(q, sharedWqe);
        bulkStoreCommit();
        bulkStoreWaitSource();
#endif
      }
      prepareSignalWqeAt<Sharing>(q, sharedWqe, index, index + 1, sigAddr, sigKey, token, outboundToken);
      copyWqeToGlobal(doca_gpu_dev_verbs_get_wqe_ptr(q, index + 1), sharedWqe);
      doca_gpu_dev_verbs_mark_wqes_ready<Sharing>(q, index, index + 1);
      submitPreparedFromShared<Sharing>(q, index + 1, sharedWqe);
    });
  }

  template <doca_gpu_dev_verbs_resource_sharing_mode Sharing>
  __device__ __noinline__ static void submitPutWithSignalDb(doca_gpu_dev_verbs_qp* q, uint64_t index, uint64_t dstAddr,
                                                            uint32_t dstKey, uint64_t srcAddr, uint32_t srcKey,
                                                            uint64_t size, uint64_t sigAddr, uint32_t sigKey,
                                                            uint64_t token, uint64_t* outboundToken) {
    auto* wqe0 = doca_gpu_dev_verbs_get_wqe_ptr(q, index);
    prepareWriteWqe(q, wqe0, index, dstAddr, dstKey, srcAddr, srcKey, size);
    auto* wqe1 = doca_gpu_dev_verbs_get_wqe_ptr(q, index + 1);
    prepareSignalWqeAt<Sharing>(q, wqe1, index, index + 1, sigAddr, sigKey, token, outboundToken);
    doca_gpu_dev_verbs_mark_wqes_ready<Sharing>(q, index, index + 1);
    submitDb<Sharing>(q, index + 2);
  }

  // When `outboundToken` is non-null, the token is taken from it after all earlier WQEs from `fromWqeIdx` on are
  // ready, so concurrent signals reach the peer in increasing order.
  template <doca_gpu_dev_verbs_resource_sharing_mode Sharing>
  MSCCLPP_DEVICE_INLINE static void prepareSignalWqeAt(doca_gpu_dev_verbs_qp* q, doca_gpu_dev_verbs_wqe* wqePtr,
                                                       [[maybe_unused]] uint64_t fromWqeIdx, uint16_t wqeIdx,
                                                       uint64_t addr, uint32_t key, uint64_t token,
                                                       uint64_t* outboundToken) {
    if (outboundToken != nullptr) {
      if constexpr (Sharing == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA) {
        cuda::atomic_ref<uint64_t, cuda::thread_scope_block> readyIndex(q->sq_ready_index);
        while (readyIndex.load(cuda::memory_order_acquire) != fromWqeIdx) {
        }
      } else if constexpr (Sharing == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU) {
        cuda::atomic_ref<uint64_t, cuda::thread_scope_device> readyIndex(q->sq_ready_index);
        while (readyIndex.load(cuda::memory_order_acquire) != fromWqeIdx) {
        }
      }
      token = atomicFetchAdd<uint64_t, scopeDevice>(outboundToken, uint64_t{1}, memoryOrderRelaxed) + 1ULL;
    }
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

  template <doca_gpu_dev_verbs_resource_sharing_mode Sharing>
  MSCCLPP_DEVICE_INLINE static void submitPreparedFromShared([[maybe_unused]] doca_gpu_dev_verbs_qp* q,
                                                             [[maybe_unused]] uint64_t index,
                                                             [[maybe_unused]] doca_gpu_dev_verbs_wqe* sharedWqe) {
#if MSCCLPP_BULK_AVAILABLE
    bulkFence();
    doca_gpu_dev_verbs_submit_bf<Sharing, DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>(q, index + 1, sharedWqe);
    bulkStoreCommit();
    bulkStoreWaitSource();
#else
    MSCCLPP_ASSERT_DEVICE(false, "BlueFlame requires SM90 bulk copy");
#endif
  }

  template <doca_gpu_dev_verbs_resource_sharing_mode Sharing, typename Fn>
  MSCCLPP_DEVICE_INLINE void withWqeSlot(doca_gpu_dev_verbs_qp* q, uint32_t count, Fn&& fn) const {
    static_assert(sizeof(GpuNetIoWqeSlot) == sizeof(doca_gpu_dev_verbs_wqe));
    MSCCLPP_ASSERT_DEVICE(wqeSlotShm_ != nullptr &&
                              reinterpret_cast<uintptr_t>(wqeSlotShm_) % alignof(GpuNetIoWqeSlot) == 0 &&
                              __isShared(wqeSlotShm_),
                          "wqeSlotShm must be a 16-byte-aligned shared-memory pointer");
    WqeSlotLockGuard lockGuard(*this);
    const auto index =
        doca_gpu_dev_verbs_reserve_wq_slots<Sharing, DOCA_GPUNETIO_VERBS_QP_SQ, DOCA_GPUNETIO_VERBS_CQ_64B>(q, count);
    fn(reinterpret_cast<doca_gpu_dev_verbs_wqe*>(wqeSlotShm_->storage), index);
  }
#endif  // defined(MSCCLPP_USE_GPUNETIO)
#endif  // defined(MSCCLPP_DEVICE_COMPILE)
};

/// Device-side memory handle used by PortChannelDeviceHandle.
using DeviceMemoryHandle = GpuNetIoMemoryDeviceHandle;

/// Device-side backend handle used by BasePortChannelDeviceHandle.
using DeviceBackendHandle = GpuNetIoDeviceHandle;

}  // namespace mscclpp

#endif  // MSCCLPP_GPUNETIO_DEVICE_HPP_
