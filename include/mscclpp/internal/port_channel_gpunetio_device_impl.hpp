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
  // Drain the send CQ by polling each completion up to the latest reserved
  // ticket. doca_gpu_dev_verbs_wait leaves the shared dispatch+combine CQ
  // under-drained across iterations, exhausting the send/completion queues; the
  // single-shot poll loop advances the CQ consumer index so the rings recycle.
  doca_gpu_dev_verbs_qp* qp = detail::ginQp(qps, peer);
  uint64_t ticket = doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
      &qp->sq_rsvd_index);
  if (ticket == 0) return;
  doca_gpu_dev_verbs_cq* cq = doca_gpu_dev_verbs_qp_get_cq_sq(qp);
  while (doca_gpu_dev_verbs_poll_one_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(cq, ticket - 1) == EBUSY) {
  }
}

MSCCLPP_DEVICE_INLINE void GpuNetIoDeviceContext::get(int peer, uint64_t remoteOffset, uint64_t localOffset,
                                                     uint64_t size) {
  doca_gpu_dev_verbs_addr raddr{peerBase[peer] + remoteOffset, rkeys[peer]};
  doca_gpu_dev_verbs_addr laddr{localBase + localOffset, detail::ginHtobe32(lkey)};
  doca_gpu_dev_verbs_qp* qp = detail::ginQp(qps, peer);
  doca_gpu_dev_verbs_ticket_t ticket;
  doca_gpu_dev_verbs_get_thread<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(qp, raddr, laddr, size, laddr, &ticket);
  // Wait for THIS read's own ticket, not the QP's global latest reserved slot
  // (doca_gpu_dev_verbs_wait(qp) polls sq_rsvd_index-1, so a caller sharing the
  // QP would block on another caller's outstanding op).
  doca_gpu_dev_verbs_wait(qp, ticket);
}

MSCCLPP_DEVICE_INLINE int GpuNetIoDeviceContext::tryFlush(int peer, uint64_t maxSpinCount) {
  doca_gpu_dev_verbs_qp* qp = detail::ginQp(qps, peer);
  uint64_t ticket = doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
      &qp->sq_rsvd_index);
  if (ticket == 0) return 0;
  --ticket;

  doca_gpu_dev_verbs_cq* cq = doca_gpu_dev_verbs_qp_get_cq_sq(qp);
  for (uint64_t spin = 0; spin < maxSpinCount; ++spin) {
    int status = doca_gpu_dev_verbs_poll_one_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(cq, ticket);
    if (status != EBUSY) return status;
  }
  return EBUSY;
}

}  // namespace mscclpp

#endif  // MSCCLPP_INTERNAL_PORT_CHANNEL_GPUNETIO_DEVICE_IMPL_HPP_
