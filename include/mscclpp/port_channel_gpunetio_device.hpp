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
// Non-leader lanes share per-warp BF scratch; these locks protect it across divergent calls.
#if defined(__CUDACC_RDC__)
inline
#else
static
#endif
    __device__ unsigned int gpuNetIoWqeLocks[1024] = {};
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
    auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
    constexpr auto sharing = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE;
    const auto index =
        doca_gpu_dev_verbs_reserve_wq_slots<sharing, DOCA_GPUNETIO_VERBS_QP_SQ, DOCA_GPUNETIO_VERBS_CQ_64B>(q, 1);
    const uint64_t token = ::atomicAdd(reinterpret_cast<unsigned long long*>(outboundToken), 1ULL) + 1ULL;
    withSharedWqe([&](doca_gpu_dev_verbs_wqe* sharedWqe) {
      prepareSignalWqe(q, sharedWqe, index, token);
      copyWqeToGlobal(doca_gpu_dev_verbs_get_wqe_ptr(q, index), sharedWqe);
      doca_gpu_dev_verbs_mark_wqes_ready<sharing>(q, index, index);
      submitPreparedFromShared(q, index, sharedWqe);
    });
#else
    trap();
#endif
  }

  /// RDMA write followed by an inline semaphore token write, ordered after the payload write.
  MSCCLPP_DEVICE_INLINE void putWithSignal(const GpuNetIoMemoryDeviceHandle& dst, uint64_t dstOffset,
                                           const GpuNetIoMemoryDeviceHandle& src, uint64_t srcOffset, uint64_t size) {
#if defined(MSCCLPP_USE_GPUNETIO)
    if (size > 0 && size <= DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE) {
      auto* q = static_cast<doca_gpu_dev_verbs_qp*>(qp);
      constexpr auto sharing = DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE;
      const auto index =
          doca_gpu_dev_verbs_reserve_wq_slots<sharing, DOCA_GPUNETIO_VERBS_QP_SQ, DOCA_GPUNETIO_VERBS_CQ_64B>(q, 2);
      withSharedWqe([&](doca_gpu_dev_verbs_wqe* sharedWqe) {
        if (size <= DOCA_GPUNETIO_VERBS_MAX_INLINE_SIZE) {
          doca_gpu_dev_verbs_wqe_prepare_write_inl(q, &sharedWqe[0], index, DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
                                                   dst.addr + dstOffset, dst.key, src.addr + srcOffset, size);
          submitInlineFromSharedNoWait(q, &sharedWqe[0]);
        } else {
          doca_gpu_dev_verbs_wqe_prepare_write(q, &sharedWqe[0], index, DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                                               DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0, dst.addr + dstOffset,
                                               dst.key, src.addr + srcOffset, src.key, size);
          submitDmaFromSharedNoWait(q, &sharedWqe[0]);
        }

        const uint64_t token = ::atomicAdd(reinterpret_cast<unsigned long long*>(outboundToken), 1ULL) + 1ULL;
        copyWqeToGlobal(doca_gpu_dev_verbs_get_wqe_ptr(q, index), &sharedWqe[0]);
        prepareSignalWqe(q, &sharedWqe[1], index + 1, token);
        copyWqeToGlobal(doca_gpu_dev_verbs_get_wqe_ptr(q, index + 1), &sharedWqe[1]);
        doca_gpu_dev_verbs_mark_wqes_ready<sharing>(q, index, index + 1);
        submitPreparedFromShared(q, index + 1, &sharedWqe[1]);
      });
      return;
    }
    putImpl(dst, dstOffset, src, srcOffset, size);
    signal();
#else
    trap();
#endif
  }

  /// Remote 64-bit atomic add.
  MSCCLPP_DEVICE_INLINE void atomicAdd(uint64_t addr, uint32_t key, int64_t value) {
#if defined(MSCCLPP_USE_GPUNETIO)
    __threadfence_system();
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
      if (size <= DOCA_GPUNETIO_VERBS_MAX_INLINE_SIZE) {
        // Inline copies the payload into the WQE; the NIC does not DMA-read the original source buffer.
        doca_gpu_dev_verbs_wqe_prepare_write_inl(q, sharedWqe, index, DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
                                                 dst.addr + dstOffset, dst.key, src.addr + srcOffset, size);
      } else {
        doca_gpu_dev_verbs_wqe_prepare_write(q, sharedWqe, index, DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                                             DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0, dst.addr + dstOffset, dst.key,
                                             src.addr + srcOffset, src.key, size);
      }
      copyWqeToGlobal(doca_gpu_dev_verbs_get_wqe_ptr(q, index), sharedWqe);
      doca_gpu_dev_verbs_mark_wqes_ready<sharing>(q, index, index);
      submitPreparedFromShared(q, index, sharedWqe);
    });
  }

  MSCCLPP_DEVICE_INLINE void prepareSignalWqe(doca_gpu_dev_verbs_qp* q, doca_gpu_dev_verbs_wqe* wqePtr, uint16_t wqeIdx,
                                              uint64_t token) const {
    constexpr auto ctrlFlags = static_cast<doca_gpu_dev_verbs_wqe_ctrl_flags>(
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE | DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_STRONG_ORDERING);
    doca_gpu_dev_verbs_prepare_inl_rdma_write_wqe_header(q, wqePtr, wqeIdx, ctrlFlags, signalAddr, signalKey,
                                                         sizeof(uint64_t));
    doca_gpu_dev_verbs_prepare_inl_rdma_write_wqe_data<uint64_t>(q, wqePtr, token);
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

  template <typename Fn>
  MSCCLPP_DEVICE_INLINE static void withSharedWqe(Fn&& fn) {
#if MSCCLPP_BULK_AVAILABLE
    const unsigned tid = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const unsigned numThreads = blockDim.x * blockDim.y * blockDim.z;
    __shared__ __align__(16) doca_gpu_dev_verbs_wqe sharedWqes[128];
    // Thread 0 and blocks with <= 64 threads have dedicated 2-WQE slots and need no workspace lock.
    if (tid == 0 || numThreads <= 64) {
      fn(&sharedWqes[tid * 2]);
      return;
    }

    const unsigned slot = 1 + ((tid - 1) % 63);
    uint64_t gridId;
    asm volatile("mov.u64 %0, %%gridid;" : "=l"(gridId));
    const uint64_t block = blockIdx.x + uint64_t(gridDim.x) * (blockIdx.y + uint64_t(gridDim.y) * blockIdx.z);
    auto* lock = &detail::gpuNetIoWqeLocks[(gridId ^ (block * 67) ^ slot) & 1023];
    bool submitted = false;
    do {
      if (atomicCAS(lock, 0u, 1u) == 0u) {
        doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
        fn(&sharedWqes[slot * 2]);
        doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
        atomicExch(lock, 0u);
        submitted = true;
      }
    } while (!submitted);
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
