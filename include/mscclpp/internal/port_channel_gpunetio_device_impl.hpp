// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// GPU-initiated networking (GPUNetIO / GDAKI) device implementations for the
// PortChannel GpuNetIo backend. Included only when MSCCLPP_USE_GPUNETIO is set
// and compiling device code. Wraps the vendored DOCA GPUNetIO device verbs.

#ifndef MSCCLPP_INTERNAL_PORT_CHANNEL_GPUNETIO_DEVICE_IMPL_HPP_
#define MSCCLPP_INTERNAL_PORT_CHANNEL_GPUNETIO_DEVICE_IMPL_HPP_

// The vendored DOCA GPUNetIO device headers depend on a few loop-unroll pragma
// macros that upstream NCCL supplies from its own nccl_device/utility.h (not part
// of the vendored doca-gpunetio subtree). Provide them here before including the
// DOCA device umbrella so the headers are self-contained in mscclpp.
#ifndef DO_PRAGMA
#define DO_PRAGMA(x) _Pragma(#x)
#endif
#ifndef NVCC_PRAGMA_UNROLL
#define NVCC_PRAGMA_UNROLL(trip_count) DO_PRAGMA(unroll trip_count)
#endif
#ifndef NVCC_PRAGMA_UNROLL_AUTO
#define NVCC_PRAGMA_UNROLL_AUTO DO_PRAGMA(unroll)
#endif
#ifndef NVCC_PRAGMA_UNROLL_DISABLED
#define NVCC_PRAGMA_UNROLL_DISABLED NVCC_PRAGMA_UNROLL(1)
#endif

#include "doca_gpunetio_device.h"

namespace mscclpp {

namespace detail {
MSCCLPP_DEVICE_INLINE doca_gpu_dev_verbs_qp* ginQp(void* qps, int peer) {
  return reinterpret_cast<doca_gpu_dev_verbs_qp*>(qps) + peer;
}
MSCCLPP_DEVICE_INLINE __be32 ginHtobe32(uint32_t v) {
  return static_cast<__be32>(__byte_perm(v, 0, 0x0123));
}
}  // namespace detail

MSCCLPP_DEVICE_INLINE void GpuNetIoDeviceContext::put(int peer, uint64_t dstOffset, uint64_t srcOffset, uint64_t size) {
  doca_gpu_dev_verbs_addr raddr{peerBase[peer] + dstOffset, rkeys[peer]};
  doca_gpu_dev_verbs_addr laddr{localBase + srcOffset, detail::ginHtobe32(lkey)};
  doca_gpu_dev_verbs_ticket_t ticket;
  doca_gpu_dev_verbs_put<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(detail::ginQp(qps, peer), raddr, laddr, size,
                                                                        &ticket);
}

MSCCLPP_DEVICE_INLINE void GpuNetIoDeviceContext::putWithSignal(int peer, uint64_t dstOffset, uint64_t srcOffset,
                                                               uint64_t size, uint64_t signalOffset,
                                                               uint64_t signalValue) {
  doca_gpu_dev_verbs_addr raddr{peerBase[peer] + dstOffset, rkeys[peer]};
  doca_gpu_dev_verbs_addr laddr{localBase + srcOffset, detail::ginHtobe32(lkey)};
  doca_gpu_dev_verbs_addr sigR{peerBase[peer] + signalOffset, rkeys[peer]};
  doca_gpu_dev_verbs_addr sigL{localBase, detail::ginHtobe32(lkey)};
  doca_gpu_dev_verbs_ticket_t ticket;
  doca_gpu_dev_verbs_put_signal<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
      detail::ginQp(qps, peer), raddr, laddr, size, sigR, sigL, signalValue, &ticket);
}

MSCCLPP_DEVICE_INLINE void GpuNetIoDeviceContext::atomicAdd(int peer, uint64_t dstOffset, int64_t value) {
  doca_gpu_dev_verbs_addr raddr{peerBase[peer] + dstOffset, rkeys[peer]};
  doca_gpu_dev_verbs_addr laddr{localBase, detail::ginHtobe32(lkey)};
  doca_gpu_dev_verbs_ticket_t ticket;
  // Fused zero-byte write + remote atomic-add expresses a standalone atomic add.
  doca_gpu_dev_verbs_put_signal<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
      detail::ginQp(qps, peer), raddr, laddr, /*size=*/0, raddr, laddr, static_cast<uint64_t>(value), &ticket);
}

MSCCLPP_DEVICE_INLINE void GpuNetIoDeviceContext::flush(int peer) {
  doca_gpu_dev_verbs_wait(detail::ginQp(qps, peer));
}

}  // namespace mscclpp

#endif  // MSCCLPP_INTERNAL_PORT_CHANNEL_GPUNETIO_DEVICE_IMPL_HPP_
