// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// GPU-initiated networking (GPUNetIO / GDAKI) device implementations for the
// PortChannel GpuNetIo backend. Included only when MSCCLPP_USE_GPUNETIO is set
// and compiling device code. Wraps the DOCA GPUNetIO device verbs.

#ifndef MSCCLPP_INTERNAL_PORT_CHANNEL_GPUNETIO_DEVICE_IMPL_HPP_
#define MSCCLPP_INTERNAL_PORT_CHANNEL_GPUNETIO_DEVICE_IMPL_HPP_

// The DOCA GPUNetIO device headers depend on a few loop-unroll pragma
// macros that upstream NCCL supplies from its own nccl_device/utility.h (not part
// of the GPUNetIO repository). Provide them here before including the
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

#include "../assert_device.hpp"
#include "doca_gpunetio_device.h"

namespace mscclpp {

namespace detail {
MSCCLPP_DEVICE_INLINE doca_gpu_dev_verbs_qp* ginQp(const GpuNetIoDeviceContext& context, int peer, int qpIndex) {
  MSCCLPP_ASSERT_DEVICE(peer >= 0 && peer < context.numPeers, "GPUNetIO peer out of range");
  MSCCLPP_ASSERT_DEVICE(qpIndex >= 0 && qpIndex < context.numQpsPerPeer, "GPUNetIO QP index out of range");
  return reinterpret_cast<doca_gpu_dev_verbs_qp*>(context.qps) + static_cast<uintptr_t>(peer) * context.numQpsPerPeer +
         qpIndex;
}
MSCCLPP_DEVICE_INLINE __be32 ginHtobe32(uint32_t v) { return static_cast<__be32>(__byte_perm(v, 0, 0x0123)); }
MSCCLPP_DEVICE_INLINE doca_gpu_dev_verbs_addr ginAtomicResult(const GpuNetIoDeviceContext& context, int peer,
                                                              int qpIndex) {
  MSCCLPP_ASSERT_DEVICE(context.atomicResultBase != 0, "GPUNetIO atomics require registered result scratch");
  return {
      context.atomicResultBase + (static_cast<uintptr_t>(peer) * context.numQpsPerPeer + qpIndex) * sizeof(uint64_t),
      ginHtobe32(context.atomicResultLkey)};
}
}  // namespace detail

MSCCLPP_DEVICE_INLINE void GpuNetIoDeviceContext::put(int peer, uint64_t dstOffset, uint64_t srcOffset, uint64_t size,
                                                      int qpIndex) {
  doca_gpu_dev_verbs_addr raddr{peerBase[peer] + dstOffset, rkeys[peer]};
  doca_gpu_dev_verbs_addr laddr{localBase + srcOffset, detail::ginHtobe32(lkey)};
  doca_gpu_dev_verbs_ticket_t ticket;
  doca_gpu_dev_verbs_put<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(detail::ginQp(*this, peer, qpIndex), raddr,
                                                                        laddr, size, &ticket);
}

MSCCLPP_DEVICE_INLINE void GpuNetIoDeviceContext::putWithSignal(int peer, uint64_t dstOffset, uint64_t srcOffset,
                                                                uint64_t size, uint64_t signalOffset,
                                                                uint64_t signalValue, int qpIndex) {
  doca_gpu_dev_verbs_addr raddr{peerBase[peer] + dstOffset, rkeys[peer]};
  doca_gpu_dev_verbs_addr laddr{localBase + srcOffset, detail::ginHtobe32(lkey)};
  doca_gpu_dev_verbs_addr sigR{peerBase[peer] + signalOffset, rkeys[peer]};
  const auto sigL = detail::ginAtomicResult(*this, peer, qpIndex);
  doca_gpu_dev_verbs_ticket_t ticket;
  doca_gpu_dev_verbs_put_signal<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
      detail::ginQp(*this, peer, qpIndex), raddr, laddr, size, sigR, sigL, signalValue, &ticket);
}

MSCCLPP_DEVICE_INLINE void GpuNetIoDeviceContext::atomicAdd(int peer, uint64_t dstOffset, int64_t value, int qpIndex) {
  doca_gpu_dev_verbs_addr raddr{peerBase[peer] + dstOffset, rkeys[peer]};
  const auto laddr = detail::ginAtomicResult(*this, peer, qpIndex);
  doca_gpu_dev_verbs_ticket_t ticket;
  // Fused zero-byte write + remote atomic-add expresses a standalone atomic add.
  doca_gpu_dev_verbs_put_signal<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
      detail::ginQp(*this, peer, qpIndex), raddr, laddr, /*size=*/0, raddr, laddr, static_cast<uint64_t>(value),
      &ticket);
}

MSCCLPP_DEVICE_INLINE void GpuNetIoDeviceContext::flush(int peer, int qpIndex) {
  doca_gpu_dev_verbs_qp* qp = detail::ginQp(*this, peer, qpIndex);
  const uint64_t ticket =
      doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(&qp->sq_rsvd_index);
  if (ticket == 0) return;
  int status;
  do {
    status = doca_gpu_dev_verbs_poll_one_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(qp, ticket - 1);
  } while (status == EBUSY);
  if (status != 0) __trap();
}

MSCCLPP_DEVICE_INLINE int GpuNetIoDeviceContext::tryFlush(int peer, uint64_t maxSpinCount, int qpIndex) {
  doca_gpu_dev_verbs_qp* qp = detail::ginQp(*this, peer, qpIndex);
  uint64_t ticket =
      doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(&qp->sq_rsvd_index);
  if (ticket == 0) return 0;
  --ticket;

  for (uint64_t spin = 0; spin < maxSpinCount; ++spin) {
    int status = doca_gpu_dev_verbs_poll_one_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(qp, ticket);
    if (status != EBUSY) return status;
  }
  return EBUSY;
}

}  // namespace mscclpp

#endif  // MSCCLPP_INTERNAL_PORT_CHANNEL_GPUNETIO_DEVICE_IMPL_HPP_
