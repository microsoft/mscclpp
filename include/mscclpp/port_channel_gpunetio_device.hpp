// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_PORT_CHANNEL_GPUNETIO_DEVICE_HPP_
#define MSCCLPP_PORT_CHANNEL_GPUNETIO_DEVICE_HPP_

#include <cstdint>

#include "device.hpp"

namespace mscclpp {

/// Device-side context for the GPU-initiated networking (GPUNetIO / GDAKI)
/// PortChannel backend. This is the kernel-issued RDMA path: instead of pushing
/// a ProxyTrigger to the host FIFO, the calling thread/warp builds the WQE and
/// rings the NIC doorbell directly (via the vendored DOCA GPUNetIO device
/// verbs), mirroring NCCL GIN's `gdaki` backend.
///
/// All remote addressing uses the same symmetric-memory model as the rest of
/// EP: a MemoryId selects a peer's registered symmetric buffer, and offsets are
/// identical on every rank. `qps`, `rkeys`, and `peerBase` are indexed by peer
/// rank; `lkey`/`localBase` describe this rank's registered buffer.
struct GpuNetIoDeviceContext {
  /// Per-peer GPU-mapped DOCA GDAKI queue pairs (type doca_gpu_dev_verbs_qp*).
  /// Kept as void* here so this public header does not pull in the DOCA device
  /// headers; the implementation reinterprets it.
  void* qps;
  /// Per-peer remote keys (network byte order), device array of length numPeers.
  const uint32_t* rkeys;
  /// Per-peer remote symmetric-buffer base addresses, device array.
  const uintptr_t* peerBase;
  /// This rank's local memory-region key.
  uint32_t lkey;
  /// This rank's symmetric-buffer base address.
  uintptr_t localBase;
  /// Number of peers (== world size); indices into the arrays above.
  int numPeers;

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Kernel-initiated RDMA write of [srcOffset, srcOffset+size) from the local
  /// symmetric buffer into peer `peer`'s symmetric buffer at dstOffset.
  MSCCLPP_DEVICE_INLINE void put(int peer, uint64_t dstOffset, uint64_t srcOffset, uint64_t size);

  /// Kernel-initiated RDMA write followed by a fused remote atomic-add signal
  /// (visible only after the payload settles).
  MSCCLPP_DEVICE_INLINE void putWithSignal(int peer, uint64_t dstOffset, uint64_t srcOffset, uint64_t size,
                                           uint64_t signalOffset, uint64_t signalValue);

  /// Kernel-initiated remote 64-bit atomic add.
  MSCCLPP_DEVICE_INLINE void atomicAdd(int peer, uint64_t dstOffset, int64_t value);

  /// Wait for locally-issued RDMA to this peer to complete (device CQ poll).
  MSCCLPP_DEVICE_INLINE void flush(int peer);

  /// Bounded version of `flush` for tests/diagnostics. Returns 0 on completion,
  /// EBUSY on timeout, or a negative CQ error status.
  MSCCLPP_DEVICE_INLINE int tryFlush(int peer, uint64_t maxSpinCount);
#endif  // defined(MSCCLPP_DEVICE_COMPILE)
};

}  // namespace mscclpp

#if defined(MSCCLPP_DEVICE_COMPILE)
// The real DOCA GPUNetIO device-verb implementations live in an internal header
// that is only pulled in when the backend is compiled in, keeping the heavy DOCA
// device headers out of this widely-included public header. When the backend is
// not compiled in, the methods are defined as unreachable traps so that any
// translation unit still links; the host never selects the GpuNetIo backend in
// that configuration.
#if defined(MSCCLPP_USE_GPUNETIO)
#include "internal/port_channel_gpunetio_device_impl.hpp"
#else
namespace mscclpp {
MSCCLPP_DEVICE_INLINE void GpuNetIoDeviceContext::put(int, uint64_t, uint64_t, uint64_t) {
#if defined(MSCCLPP_DEVICE_CUDA)
  __trap();
#endif
}
MSCCLPP_DEVICE_INLINE void GpuNetIoDeviceContext::putWithSignal(int, uint64_t, uint64_t, uint64_t, uint64_t, uint64_t) {
#if defined(MSCCLPP_DEVICE_CUDA)
  __trap();
#endif
}
MSCCLPP_DEVICE_INLINE void GpuNetIoDeviceContext::atomicAdd(int, uint64_t, int64_t) {
#if defined(MSCCLPP_DEVICE_CUDA)
  __trap();
#endif
}
MSCCLPP_DEVICE_INLINE void GpuNetIoDeviceContext::flush(int) {
#if defined(MSCCLPP_DEVICE_CUDA)
  __trap();
#endif
}
MSCCLPP_DEVICE_INLINE int GpuNetIoDeviceContext::tryFlush(int, uint64_t) {
#if defined(MSCCLPP_DEVICE_CUDA)
  __trap();
#endif
  return -1;
}
}  // namespace mscclpp
#endif  // defined(MSCCLPP_USE_GPUNETIO)
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

#endif  // MSCCLPP_PORT_CHANNEL_GPUNETIO_DEVICE_HPP_
