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

#if defined(MSCCLPP_DEVICE_CUDA) && defined(USE_IBVERBS)
#include <doca_gpunetio_device.h>

#include "bulk_device.hpp"
#endif

namespace mscclpp {

namespace detail {
struct alignas(16) GpuNetIoWqeSlot {
  uint64_t storage[8];
};

#if defined(MSCCLPP_DEVICE_CUDA) && defined(USE_IBVERBS) && MSCCLPP_BULK_AVAILABLE
template <uint32_t MaxExclusiveThreadsPerBlock>
MSCCLPP_DEVICE_INLINE doca_gpu_dev_verbs_wqe* gpuNetIoWqeSlot(uint32_t slotIdx) {
  static_assert(MaxExclusiveThreadsPerBlock > 0 && MaxExclusiveThreadsPerBlock <= 1024);
  static_assert(sizeof(GpuNetIoWqeSlot) == sizeof(doca_gpu_dev_verbs_wqe));
  __shared__ GpuNetIoWqeSlot slots[MaxExclusiveThreadsPerBlock];
  return reinterpret_cast<doca_gpu_dev_verbs_wqe*>(slots[slotIdx].storage);
}
#endif

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

  /// Default BlueFlame payload size limit (512 KiB).
  static constexpr uint32_t kDefaultBlueFlameLimit = 512 * 1024;

  /// Default maximum number of threads per block concurrently posting in Exclusive mode with BlueFlame.
  static constexpr uint32_t kDefaultMaxExclusiveThreadsPerBlock = 32;

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// RDMA write of `size` bytes from a local region to a remote region.
  /// @tparam Mode Caller concurrency scope supplied by the PortChannel device handle.
  /// @tparam MaxExclusiveThreadsPerBlock Maximum number of threads per block concurrently posting in Exclusive mode;
  /// zero disables BlueFlame.
  /// @param slotIdx Slot index, or -1 for the linear thread index. Out-of-range indices use regular doorbells.
  template <auto Mode, uint32_t MaxExclusiveThreadsPerBlock = kDefaultMaxExclusiveThreadsPerBlock>
  MSCCLPP_DEVICE_INLINE void put([[maybe_unused]] const GpuNetIoMemoryDeviceHandle& dst,
                                 [[maybe_unused]] uint64_t dstOffset,
                                 [[maybe_unused]] const GpuNetIoMemoryDeviceHandle& src,
                                 [[maybe_unused]] uint64_t srcOffset, [[maybe_unused]] uint64_t size,
                                 [[maybe_unused]] int slotIdx = -1) {
    static_assert(MaxExclusiveThreadsPerBlock <= 1024, "At most 1024 exclusive threads per block are supported");
#if defined(MSCCLPP_DEVICE_CUDA) && defined(USE_IBVERBS)
    putImpl<docaConcurrency<Mode>(), MaxExclusiveThreadsPerBlock>(dst, dstOffset, src, srcOffset, size, slotIdx);
#else
    MSCCLPP_ASSERT_DEVICE(false, "GPUNetIO is not enabled");
#endif
  }

  /// Atomically add `value` to the 64-bit word at `dst` + `dstOffset` (8-byte aligned) with an RDMA fetch-and-add.
  /// Any number of writers may update the same word. The peer applies the addition only after placing the data of
  /// earlier RDMA writes on this queue pair, so observing the new value publishes those writes.
  /// @param dst The destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param value The 64-bit signed value to add.
  /// @tparam Mode Caller concurrency scope supplied by the PortChannel device handle.
  /// @tparam MaxExclusiveThreadsPerBlock Maximum number of threads per block concurrently posting in Exclusive mode;
  /// zero disables BlueFlame.
  /// @param slotIdx Slot index, or -1 for the linear thread index. Out-of-range indices use regular doorbells.
  template <auto Mode, uint32_t MaxExclusiveThreadsPerBlock = kDefaultMaxExclusiveThreadsPerBlock>
  MSCCLPP_DEVICE_INLINE void accumulate([[maybe_unused]] const GpuNetIoMemoryDeviceHandle& dst,
                                        [[maybe_unused]] uint64_t dstOffset, [[maybe_unused]] int64_t value,
                                        [[maybe_unused]] int slotIdx = -1) {
    static_assert(MaxExclusiveThreadsPerBlock <= 1024, "At most 1024 exclusive threads per block are supported");
#if defined(MSCCLPP_DEVICE_CUDA) && defined(USE_IBVERBS)
    accumulateImpl<Mode, MaxExclusiveThreadsPerBlock>(dst, dstOffset, value, slotIdx, [] {});
#else
    MSCCLPP_ASSERT_DEVICE(false, "GPUNetIO is not enabled");
#endif
  }

  /// accumulate() that prepares the work request, invokes `wait()`, and then submits it, so the preparation overlaps
  /// the wait. Requires exclusive ownership of the queue pair and shared-memory slot.
  /// @param dst The destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param value The 64-bit signed value to add.
  /// @param wait Called once on the posting thread before submission. It must establish readiness and memory
  /// ordering, and must not access the same queue pair or shared-memory slot.
  /// @param slotIdx Slot index, or -1 for the linear thread index. Out-of-range indices use regular doorbells.
  /// @tparam Mode Caller concurrency scope supplied by the PortChannel device handle (must be Exclusive).
  /// @tparam MaxExclusiveThreadsPerBlock Maximum number of threads per block concurrently posting in Exclusive mode;
  /// zero disables BlueFlame.
  template <auto Mode, uint32_t MaxExclusiveThreadsPerBlock = kDefaultMaxExclusiveThreadsPerBlock, typename WaitFn>
  MSCCLPP_DEVICE_INLINE void accumulateAfter([[maybe_unused]] const GpuNetIoMemoryDeviceHandle& dst,
                                             [[maybe_unused]] uint64_t dstOffset, [[maybe_unused]] int64_t value,
                                             [[maybe_unused]] WaitFn&& wait, [[maybe_unused]] int slotIdx = -1) {
    static_assert(Mode == decltype(Mode)::Exclusive, "accumulateAfter requires Concurrency::Exclusive");
    static_assert(MaxExclusiveThreadsPerBlock <= 1024, "At most 1024 exclusive threads per block are supported");
#if defined(MSCCLPP_DEVICE_CUDA) && defined(USE_IBVERBS)
    accumulateImpl<Mode, MaxExclusiveThreadsPerBlock>(dst, dstOffset, value, slotIdx, static_cast<WaitFn&&>(wait));
#else
    MSCCLPP_ASSERT_DEVICE(false, "GPUNetIO is not enabled");
#endif
  }

  /// RDMA write followed by an accumulate to `acc` at `accOffset`, posted together with the write. Observing the
  /// accumulated value publishes the written data. See accumulate().
  /// @param dst The destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param src The source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  /// @param acc The memory region of the accumulated word.
  /// @param accOffset The offset of the accumulated word in `acc`.
  /// @param value The 64-bit signed value to add.
  /// @tparam Mode Caller concurrency scope supplied by the PortChannel device handle.
  /// @tparam MaxExclusiveThreadsPerBlock Maximum number of threads per block concurrently posting in Exclusive mode;
  /// zero disables BlueFlame.
  /// @param slotIdx Slot index, or -1 for the linear thread index. Out-of-range indices use regular doorbells.
  template <auto Mode, uint32_t MaxExclusiveThreadsPerBlock = kDefaultMaxExclusiveThreadsPerBlock>
  MSCCLPP_DEVICE_INLINE void putWithAccumulate([[maybe_unused]] const GpuNetIoMemoryDeviceHandle& dst,
                                               [[maybe_unused]] uint64_t dstOffset,
                                               [[maybe_unused]] const GpuNetIoMemoryDeviceHandle& src,
                                               [[maybe_unused]] uint64_t srcOffset, [[maybe_unused]] uint64_t size,
                                               [[maybe_unused]] const GpuNetIoMemoryDeviceHandle& acc,
                                               [[maybe_unused]] uint64_t accOffset, [[maybe_unused]] int64_t value,
                                               [[maybe_unused]] int slotIdx = -1) {
    static_assert(MaxExclusiveThreadsPerBlock <= 1024, "At most 1024 exclusive threads per block are supported");
#if defined(MSCCLPP_DEVICE_CUDA) && defined(USE_IBVERBS)
    putWithAccumulateImpl<Mode, MaxExclusiveThreadsPerBlock>(dst, dstOffset, src, srcOffset, size, acc, accOffset,
                                                             value, slotIdx, [] {});
#else
    MSCCLPP_ASSERT_DEVICE(false, "GPUNetIO is not enabled");
#endif
  }

  /// putWithAccumulate() that prepares the work requests, invokes `wait()`, and then submits them, so the preparation
  /// overlaps the wait. An inline payload is read only after `wait()` returns. Requires exclusive ownership of the
  /// queue pair and shared-memory slot.
  /// @param dst The destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param src The source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  /// @param acc The memory region of the accumulated word.
  /// @param accOffset The offset of the accumulated word in `acc`.
  /// @param value The 64-bit signed value to add.
  /// @param wait Called once on the posting thread before submission. It must establish payload readiness and memory
  /// ordering, and must not access the same queue pair or shared-memory slot.
  /// @param slotIdx Slot index, or -1 for the linear thread index. Out-of-range indices use regular doorbells.
  /// @tparam Mode Caller concurrency scope supplied by the PortChannel device handle (must be Exclusive).
  /// @tparam MaxExclusiveThreadsPerBlock Maximum number of threads per block concurrently posting in Exclusive mode;
  /// zero disables BlueFlame.
  template <auto Mode, uint32_t MaxExclusiveThreadsPerBlock = kDefaultMaxExclusiveThreadsPerBlock, typename WaitFn>
  MSCCLPP_DEVICE_INLINE void putWithAccumulateAfter([[maybe_unused]] const GpuNetIoMemoryDeviceHandle& dst,
                                                    [[maybe_unused]] uint64_t dstOffset,
                                                    [[maybe_unused]] const GpuNetIoMemoryDeviceHandle& src,
                                                    [[maybe_unused]] uint64_t srcOffset, [[maybe_unused]] uint64_t size,
                                                    [[maybe_unused]] const GpuNetIoMemoryDeviceHandle& acc,
                                                    [[maybe_unused]] uint64_t accOffset, [[maybe_unused]] int64_t value,
                                                    [[maybe_unused]] WaitFn&& wait, [[maybe_unused]] int slotIdx = -1) {
    static_assert(Mode == decltype(Mode)::Exclusive, "putWithAccumulateAfter requires Concurrency::Exclusive");
    static_assert(MaxExclusiveThreadsPerBlock <= 1024, "At most 1024 exclusive threads per block are supported");
#if defined(MSCCLPP_DEVICE_CUDA) && defined(USE_IBVERBS)
    putWithAccumulateImpl<Mode, MaxExclusiveThreadsPerBlock>(dst, dstOffset, src, srcOffset, size, acc, accOffset,
                                                             value, slotIdx, static_cast<WaitFn&&>(wait));
#else
    MSCCLPP_ASSERT_DEVICE(false, "GPUNetIO is not enabled");
#endif
  }

  /// Wait until every work request posted so far on the queue pair has completed.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  /// @tparam Mode Caller concurrency scope supplied by the PortChannel device handle.
  template <auto Mode>
  MSCCLPP_DEVICE_INLINE void flush([[maybe_unused]] int64_t maxSpinCount) {
#if defined(MSCCLPP_DEVICE_CUDA) && defined(USE_IBVERBS)
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    constexpr auto sharing = docaConcurrency<Mode>();
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
#if defined(MSCCLPP_DEVICE_CUDA) && defined(USE_IBVERBS)
  template <auto Mode>
  MSCCLPP_HOST_DEVICE_INLINE static constexpr doca_gpu_dev_verbs_resource_sharing_mode docaConcurrency() {
    using Concurrency = decltype(Mode);
    if constexpr (Mode == Concurrency::Exclusive) {
      return DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE;
    } else if constexpr (Mode == Concurrency::Block) {
      return DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA;
    } else {
      return DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU;
    }
  }

  MSCCLPP_DEVICE_INLINE static uint32_t resolveSlotIndex(int slotIdx) {
    if (slotIdx == -1) return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    return static_cast<uint32_t>(slotIdx);
  }

  // Only an exclusively owned queue pair submits through BlueFlame, staging each WQE in an internal slot. On a shared
  // queue pair, the NIC would take a BlueFlame WQE directly only if every earlier WQE were already submitted.
  template <doca_gpu_dev_verbs_resource_sharing_mode Sharing, uint32_t MaxExclusiveThreadsPerBlock>
  MSCCLPP_DEVICE_INLINE bool useBlueFlame([[maybe_unused]] uint64_t size, [[maybe_unused]] uint32_t slotIdx) const {
#if MSCCLPP_BULK_AVAILABLE
    if constexpr (Sharing == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE && MaxExclusiveThreadsPerBlock > 0) {
      return slotIdx < MaxExclusiveThreadsPerBlock && size <= blueFlameLimit;
    }
#endif
    return false;
  }

  template <doca_gpu_dev_verbs_resource_sharing_mode Sharing>
  MSCCLPP_DEVICE_INLINE static void submitDb(doca_gpu_dev_verbs_qp* q, uint64_t prodIndex) {
    constexpr auto syncScope = (Sharing == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU)
                                   ? DOCA_GPUNETIO_VERBS_SYNC_SCOPE_THREAD
                                   : DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU;
    doca_gpu_dev_verbs_submit_db<Sharing, syncScope, DOCA_GPUNETIO_VERBS_QP_SQ>(q, prodIndex);
  }

  template <doca_gpu_dev_verbs_resource_sharing_mode Sharing, uint32_t MaxExclusiveThreadsPerBlock>
  MSCCLPP_DEVICE_INLINE void putImpl(const GpuNetIoMemoryDeviceHandle& dst, uint64_t dstOffset,
                                     const GpuNetIoMemoryDeviceHandle& src, uint64_t srcOffset, uint64_t size,
                                     int slotIdx) {
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    if (size == 0 || size > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE) {
      doca_gpu_dev_verbs_ticket_t ticket;
      doca_gpu_dev_verbs_put<Sharing, DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB, DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD,
                             DOCA_GPUNETIO_VERBS_CQ_64B>(q, {dst.addr + dstOffset, dst.key},
                                                         {src.addr + srcOffset, src.key}, size, &ticket);
      return;
    }

    const uint32_t resolvedSlotIdx = resolveSlotIndex(slotIdx);
    if (useBlueFlame<Sharing, MaxExclusiveThreadsPerBlock>(size, resolvedSlotIdx)) {
      submitPutBlueFlame<Sharing, MaxExclusiveThreadsPerBlock>(q, dst.addr + dstOffset, dst.key, src.addr + srcOffset,
                                                               src.key, size, resolvedSlotIdx);
    } else {
      const auto index =
          doca_gpu_dev_verbs_reserve_wq_slots<Sharing, DOCA_GPUNETIO_VERBS_QP_SQ, DOCA_GPUNETIO_VERBS_CQ_64B>(q, 1);
      submitPutDb<Sharing>(q, index, dst.addr + dstOffset, dst.key, src.addr + srcOffset, src.key, size);
    }
  }

  template <doca_gpu_dev_verbs_resource_sharing_mode Sharing, uint32_t MaxExclusiveThreadsPerBlock>
  MSCCLPP_DEVICE_INLINE void submitPutBlueFlame(doca_gpu_dev_verbs_qp* q, uint64_t dstAddr, uint32_t dstKey,
                                                uint64_t srcAddr, uint32_t srcKey, uint64_t size, uint32_t slotIdx) {
    withWqeSlot<Sharing, MaxExclusiveThreadsPerBlock>(
        q, 1, slotIdx, [&](doca_gpu_dev_verbs_wqe* sharedWqe, uint64_t index) {
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

  template <auto Mode, uint32_t MaxExclusiveThreadsPerBlock, typename WaitFn>
  MSCCLPP_DEVICE_INLINE void accumulateImpl(const GpuNetIoMemoryDeviceHandle& dst, uint64_t dstOffset, int64_t value,
                                            int slotIdx, WaitFn&& wait) {
    const uint64_t addr = dst.addr + dstOffset;
    MSCCLPP_ASSERT_DEVICE(addr % sizeof(uint64_t) == 0, "accumulate address must be 8-byte aligned");
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    constexpr auto sharing = docaConcurrency<Mode>();
    const uint32_t resolvedSlotIdx = resolveSlotIndex(slotIdx);
    if (useBlueFlame<sharing, MaxExclusiveThreadsPerBlock>(sizeof(uint64_t), resolvedSlotIdx)) {
      withWqeSlot<sharing, MaxExclusiveThreadsPerBlock>(
          q, 1, resolvedSlotIdx, [&](doca_gpu_dev_verbs_wqe* sharedWqe, uint64_t index) {
            prepareAtomicWqe(q, sharedWqe, index, addr, dst.key, value);
            copyWqeToGlobal(doca_gpu_dev_verbs_get_wqe_ptr(q, index), sharedWqe);
            doca_gpu_dev_verbs_mark_wqes_ready<sharing>(q, index, index);
            wait();
            submitPreparedFromShared<sharing>(q, index, sharedWqe);
          });
    } else {
      const auto index =
          doca_gpu_dev_verbs_reserve_wq_slots<sharing, DOCA_GPUNETIO_VERBS_QP_SQ, DOCA_GPUNETIO_VERBS_CQ_64B>(q, 1);
      prepareAtomicWqe(q, doca_gpu_dev_verbs_get_wqe_ptr(q, index), index, addr, dst.key, value);
      doca_gpu_dev_verbs_mark_wqes_ready<sharing>(q, index, index);
      wait();
      submitDb<sharing>(q, index + 1);
    }
  }

  template <auto Mode, uint32_t MaxExclusiveThreadsPerBlock, typename WaitFn>
  MSCCLPP_DEVICE_INLINE void putWithAccumulateImpl(const GpuNetIoMemoryDeviceHandle& dst, uint64_t dstOffset,
                                                   const GpuNetIoMemoryDeviceHandle& src, uint64_t srcOffset,
                                                   uint64_t size, const GpuNetIoMemoryDeviceHandle& acc,
                                                   uint64_t accOffset, int64_t value, int slotIdx, WaitFn&& wait) {
    const uint64_t accAddr = acc.addr + accOffset;
    MSCCLPP_ASSERT_DEVICE(accAddr % sizeof(uint64_t) == 0, "putWithAccumulate address must be 8-byte aligned");
    if (size == 0 || size > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE) {
      wait();
      putImpl<docaConcurrency<Mode>(), MaxExclusiveThreadsPerBlock>(dst, dstOffset, src, srcOffset, size, slotIdx);
      accumulate<Mode, MaxExclusiveThreadsPerBlock>(acc, accOffset, value, slotIdx);
      return;
    }
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    constexpr auto sharing = docaConcurrency<Mode>();
    const uint64_t dstAddr = dst.addr + dstOffset;
    const uint64_t srcAddr = src.addr + srcOffset;
    const uint32_t resolvedSlotIdx = resolveSlotIndex(slotIdx);
    if (useBlueFlame<sharing, MaxExclusiveThreadsPerBlock>(size, resolvedSlotIdx)) {
      submitPutWithAccumulateBlueFlame<sharing, MaxExclusiveThreadsPerBlock>(
          q, dstAddr, dst.key, srcAddr, src.key, size, accAddr, acc.key, value, resolvedSlotIdx,
          static_cast<WaitFn&&>(wait));
    } else {
      const auto index =
          doca_gpu_dev_verbs_reserve_wq_slots<sharing, DOCA_GPUNETIO_VERBS_QP_SQ, DOCA_GPUNETIO_VERBS_CQ_64B>(q, 2);
      submitPutWithAccumulateDb<sharing>(q, index, dstAddr, dst.key, srcAddr, src.key, size, accAddr, acc.key, value,
                                         static_cast<WaitFn&&>(wait));
    }
  }

  // An inline payload is copied into the WQE, so it must be ready before the write WQE is prepared; otherwise the NIC
  // reads it after submission.
  template <doca_gpu_dev_verbs_resource_sharing_mode Sharing, uint32_t MaxExclusiveThreadsPerBlock, typename WaitFn>
  MSCCLPP_DEVICE_INLINE void submitPutWithAccumulateBlueFlame(doca_gpu_dev_verbs_qp* q, uint64_t dstAddr,
                                                              uint32_t dstKey, uint64_t srcAddr, uint32_t srcKey,
                                                              uint64_t size, uint64_t accAddr, uint32_t accKey,
                                                              int64_t value, uint32_t slotIdx, WaitFn&& wait) {
    withWqeSlot<Sharing, MaxExclusiveThreadsPerBlock>(
        q, 2, slotIdx, [&](doca_gpu_dev_verbs_wqe* sharedWqe, uint64_t index) {
          const bool inlinePayload = size <= DOCA_GPUNETIO_VERBS_MAX_INLINE_SIZE;
          if (inlinePayload) wait();
          prepareWriteWqe(q, sharedWqe, index, dstAddr, dstKey, srcAddr, srcKey, size);
          copyWqeToGlobal(doca_gpu_dev_verbs_get_wqe_ptr(q, index), sharedWqe);
          if (!inlinePayload) wait();
        // Ring the write by itself so that the NIC takes both WQEs through BlueFlame.
#if MSCCLPP_BULK_AVAILABLE
          bulkFence();
          doca_gpu_dev_verbs_ring_bf<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>(q, sharedWqe);
          bulkStoreCommit();
          bulkStoreWaitSource();
#endif
          prepareAtomicWqe(q, sharedWqe, index + 1, accAddr, accKey, value);
          copyWqeToGlobal(doca_gpu_dev_verbs_get_wqe_ptr(q, index + 1), sharedWqe);
          doca_gpu_dev_verbs_mark_wqes_ready<Sharing>(q, index, index + 1);
          submitPreparedFromShared<Sharing>(q, index + 1, sharedWqe);
        });
  }

  template <doca_gpu_dev_verbs_resource_sharing_mode Sharing, typename WaitFn>
  __device__ __noinline__ void submitPutWithAccumulateDb(doca_gpu_dev_verbs_qp* q, uint64_t index, uint64_t dstAddr,
                                                         uint32_t dstKey, uint64_t srcAddr, uint32_t srcKey,
                                                         uint64_t size, uint64_t accAddr, uint32_t accKey,
                                                         int64_t value, WaitFn&& wait) {
    auto* wqe0 = doca_gpu_dev_verbs_get_wqe_ptr(q, index);
    auto* wqe1 = doca_gpu_dev_verbs_get_wqe_ptr(q, index + 1);
    prepareAtomicWqe(q, wqe1, index + 1, accAddr, accKey, value);
    const bool inlinePayload = size <= DOCA_GPUNETIO_VERBS_MAX_INLINE_SIZE;
    if (inlinePayload) wait();
    prepareWriteWqe(q, wqe0, index, dstAddr, dstKey, srcAddr, srcKey, size);
    doca_gpu_dev_verbs_mark_wqes_ready<Sharing>(q, index, index + 1);
    if (!inlinePayload) wait();
    submitDb<Sharing>(q, index + 2);
  }

  // An RDMA atomic, unlike an RDMA write, is applied by the peer only after earlier writes on the queue pair are
  // placed, so it can publish them even when packets arrive out of order.
  MSCCLPP_DEVICE_INLINE void prepareAtomicWqe(doca_gpu_dev_verbs_qp* q, doca_gpu_dev_verbs_wqe* wqePtr, uint16_t wqeIdx,
                                              uint64_t addr, uint32_t key, int64_t value) const {
    doca_gpu_dev_verbs_wqe_prepare_atomic(q, wqePtr, wqeIdx, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA,
                                          DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, addr, key, atomicFetchAddr,
                                          atomicFetchKey, sizeof(uint64_t), static_cast<uint64_t>(value), 0);
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

  template <doca_gpu_dev_verbs_resource_sharing_mode Sharing, uint32_t MaxExclusiveThreadsPerBlock, typename Fn>
  MSCCLPP_DEVICE_INLINE void withWqeSlot([[maybe_unused]] doca_gpu_dev_verbs_qp* q, [[maybe_unused]] uint32_t count,
                                         [[maybe_unused]] uint32_t slotIdx, [[maybe_unused]] Fn&& fn) const {
#if MSCCLPP_BULK_AVAILABLE
    if constexpr (Sharing == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE && MaxExclusiveThreadsPerBlock > 0) {
      const auto index =
          doca_gpu_dev_verbs_reserve_wq_slots<Sharing, DOCA_GPUNETIO_VERBS_QP_SQ, DOCA_GPUNETIO_VERBS_CQ_64B>(q, count);
      fn(gpuNetIoWqeSlot<MaxExclusiveThreadsPerBlock>(slotIdx), index);
    }
#endif
  }
#endif  // defined(MSCCLPP_DEVICE_CUDA) && defined(USE_IBVERBS)
#endif  // defined(MSCCLPP_DEVICE_COMPILE)
};

}  // namespace detail

/// Device-side memory handle used by PortChannel device-mode operations.
using DeviceMemoryHandle = detail::GpuNetIoMemoryDeviceHandle;

/// Device-side backend handle used by BasePortChannelDeviceHandle.
using DeviceBackendHandle = detail::GpuNetIoDeviceHandle;

}  // namespace mscclpp

#endif  // MSCCLPP_GPUNETIO_DEVICE_HPP_
