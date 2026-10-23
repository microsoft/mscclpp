// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_PORT_CHANNEL_GPUNETIO_DEVICE_HPP_
#define MSCCLPP_PORT_CHANNEL_GPUNETIO_DEVICE_HPP_

#include <cstdint>

#include "assert_device.hpp"
#include "device.hpp"

#if defined(MSCCLPP_DEVICE_COMPILE) && defined(MSCCLPP_USE_GPUNETIO)
#include <doca_gpunetio_device.h>
#endif

namespace mscclpp {
namespace detail {

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
    __threadfence_system();
    doca_gpu_dev_verbs_ticket_t ticket;
    doca_gpu_dev_verbs_put<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
        static_cast<doca_gpu_dev_verbs_qp*>(qp), {dst.addr + dstOffset, dst.key}, {src.addr + srcOffset, src.key}, size,
        &ticket);
#else
    trap();
#endif
  }

  /// RDMA write followed by a +1 atomic on the remote semaphore token, ordered after the write.
  MSCCLPP_DEVICE_INLINE void putWithSignal(const GpuNetIoMemoryDeviceHandle& dst, uint64_t dstOffset,
                                           const GpuNetIoMemoryDeviceHandle& src, uint64_t srcOffset, uint64_t size) {
#if defined(MSCCLPP_USE_GPUNETIO)
    __threadfence_system();
    doca_gpu_dev_verbs_ticket_t ticket;
    doca_gpu_dev_verbs_put_signal<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
        static_cast<doca_gpu_dev_verbs_qp*>(qp), {dst.addr + dstOffset, dst.key}, {src.addr + srcOffset, src.key}, size,
        {signalAddr, signalKey}, {atomicScratchAddr, atomicScratchKey}, 1, &ticket);
#else
    trap();
#endif
  }

  /// Remote 64-bit atomic add (a zero-byte write fused with the atomic).
  MSCCLPP_DEVICE_INLINE void atomicAdd(uint64_t addr, uint32_t key, int64_t value) {
#if defined(MSCCLPP_USE_GPUNETIO)
    const doca_gpu_dev_verbs_addr remote{addr, key};
    const doca_gpu_dev_verbs_addr scratch{atomicScratchAddr, atomicScratchKey};
    doca_gpu_dev_verbs_ticket_t ticket;
    doca_gpu_dev_verbs_put_signal<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
        static_cast<doca_gpu_dev_verbs_qp*>(qp), remote, scratch, 0, remote, scratch, static_cast<uint64_t>(value),
        &ticket);
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
        doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(&q->sq_rsvd_index);
    if (ticket == 0) return;
    int status;
    [[maybe_unused]] int64_t spinCount = 0;
    while ((status = doca_gpu_dev_verbs_poll_one_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(q, ticket - 1)) ==
           EBUSY) {
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
#endif  // defined(MSCCLPP_DEVICE_COMPILE)
};

}  // namespace detail
}  // namespace mscclpp

#endif  // MSCCLPP_PORT_CHANNEL_GPUNETIO_DEVICE_HPP_
