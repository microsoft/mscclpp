// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// GPU-initiated networking (GPUNetIO / GDAKI) device implementations for the
// PortChannel GpuNetIo backend. Included only when MSCCLPP_USE_GPUNETIO is set
// and compiling device code. Wraps the DOCA GPUNetIO device verbs.

#ifndef MSCCLPP_INTERNAL_PORT_CHANNEL_GPUNETIO_DEVICE_IMPL_HPP_
#define MSCCLPP_INTERNAL_PORT_CHANNEL_GPUNETIO_DEVICE_IMPL_HPP_

#include "../assert_device.hpp"
#include "doca_gpunetio_device.h"

namespace mscclpp {

namespace detail {
MSCCLPP_DEVICE_INLINE doca_gpu_dev_verbs_qp* gpuNetIoQp(const GpuNetIoDeviceContext& context, int peer, int qpIndex) {
  MSCCLPP_ASSERT_DEVICE(peer >= 0 && peer < context.numPeers, "GPUNetIO peer out of range");
  MSCCLPP_ASSERT_DEVICE(qpIndex >= 0 && qpIndex < context.numQpsPerPeer, "GPUNetIO QP index out of range");
  return reinterpret_cast<doca_gpu_dev_verbs_qp*>(context.qps) + static_cast<uintptr_t>(peer) * context.numQpsPerPeer +
         qpIndex;
}
MSCCLPP_DEVICE_INLINE __be32 gpuNetIoHtobe32(uint32_t v) { return static_cast<__be32>(__byte_perm(v, 0, 0x0123)); }
MSCCLPP_DEVICE_INLINE doca_gpu_dev_verbs_addr gpuNetIoAtomicResult(const GpuNetIoDeviceContext& context, int peer,
                                                                   int qpIndex) {
  MSCCLPP_ASSERT_DEVICE(context.atomicResultBase != 0, "GPUNetIO atomics require registered result scratch");
  return {
      context.atomicResultBase + (static_cast<uintptr_t>(peer) * context.numQpsPerPeer + qpIndex) * sizeof(uint64_t),
      gpuNetIoHtobe32(context.atomicResultLkey)};
}
}  // namespace detail

MSCCLPP_DEVICE_INLINE void GpuNetIoDeviceContext::putRegistered(int peer, int qpIndex, GpuNetIoMemoryDeviceHandle dst,
                                                                uint64_t dstOffset, GpuNetIoMemoryDeviceHandle src,
                                                                uint64_t srcOffset, uint64_t size) {
  doca_gpu_dev_verbs_ticket_t ticket;
  doca_gpu_dev_verbs_put<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
      detail::gpuNetIoQp(*this, peer, qpIndex), {dst.base + dstOffset, detail::gpuNetIoHtobe32(dst.key)},
      {src.base + srcOffset, detail::gpuNetIoHtobe32(src.key)}, size, &ticket);
}

MSCCLPP_DEVICE_INLINE void GpuNetIoDeviceContext::putRegisteredWithSignal(
    int peer, int qpIndex, GpuNetIoMemoryDeviceHandle dst, uint64_t dstOffset, GpuNetIoMemoryDeviceHandle src,
    uint64_t srcOffset, uint64_t size, GpuNetIoMemoryDeviceHandle signal) {
  doca_gpu_dev_verbs_ticket_t ticket;
  doca_gpu_dev_verbs_put_signal<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
      detail::gpuNetIoQp(*this, peer, qpIndex), {dst.base + dstOffset, detail::gpuNetIoHtobe32(dst.key)},
      {src.base + srcOffset, detail::gpuNetIoHtobe32(src.key)}, size,
      {signal.base, detail::gpuNetIoHtobe32(signal.key)}, detail::gpuNetIoAtomicResult(*this, peer, qpIndex), 1,
      &ticket);
}

MSCCLPP_DEVICE_INLINE void GpuNetIoDeviceContext::atomicAddRegistered(int peer, int qpIndex,
                                                                      GpuNetIoMemoryDeviceHandle dst,
                                                                      uint64_t dstOffset, int64_t value) {
  const doca_gpu_dev_verbs_addr remote{dst.base + dstOffset, detail::gpuNetIoHtobe32(dst.key)};
  const auto scratch = detail::gpuNetIoAtomicResult(*this, peer, qpIndex);
  doca_gpu_dev_verbs_ticket_t ticket;
  doca_gpu_dev_verbs_put_signal<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
      detail::gpuNetIoQp(*this, peer, qpIndex), remote, scratch, 0, remote, scratch, static_cast<uint64_t>(value),
      &ticket);
}

MSCCLPP_DEVICE_INLINE void GpuNetIoDeviceContext::put(int peer, uint64_t dstOffset, uint64_t srcOffset, uint64_t size,
                                                      int qpIndex) {
  auto* qp = detail::gpuNetIoQp(*this, peer, qpIndex);
  doca_gpu_dev_verbs_addr raddr{peerBase[peer] + dstOffset, rkeys[peer]};
  doca_gpu_dev_verbs_addr laddr{localBase + srcOffset, detail::gpuNetIoHtobe32(lkey)};
  doca_gpu_dev_verbs_ticket_t ticket;
  doca_gpu_dev_verbs_put<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(qp, raddr, laddr, size, &ticket);
}

MSCCLPP_DEVICE_INLINE void GpuNetIoDeviceContext::putWithSignal(int peer, uint64_t dstOffset, uint64_t srcOffset,
                                                                uint64_t size, uint64_t signalOffset,
                                                                uint64_t signalValue, int qpIndex) {
  doca_gpu_dev_verbs_addr raddr{peerBase[peer] + dstOffset, rkeys[peer]};
  doca_gpu_dev_verbs_addr laddr{localBase + srcOffset, detail::gpuNetIoHtobe32(lkey)};
  doca_gpu_dev_verbs_addr sigR{peerBase[peer] + signalOffset, rkeys[peer]};
  const auto sigL = detail::gpuNetIoAtomicResult(*this, peer, qpIndex);
  doca_gpu_dev_verbs_ticket_t ticket;
  doca_gpu_dev_verbs_put_signal<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
      detail::gpuNetIoQp(*this, peer, qpIndex), raddr, laddr, size, sigR, sigL, signalValue, &ticket);
}

MSCCLPP_DEVICE_INLINE void GpuNetIoDeviceContext::atomicAdd(int peer, uint64_t dstOffset, int64_t value, int qpIndex) {
  doca_gpu_dev_verbs_addr raddr{peerBase[peer] + dstOffset, rkeys[peer]};
  const auto laddr = detail::gpuNetIoAtomicResult(*this, peer, qpIndex);
  doca_gpu_dev_verbs_ticket_t ticket;
  // Fused zero-byte write + remote atomic-add expresses a standalone atomic add.
  doca_gpu_dev_verbs_put_signal<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
      detail::gpuNetIoQp(*this, peer, qpIndex), raddr, laddr, /*size=*/0, raddr, laddr, static_cast<uint64_t>(value),
      &ticket);
}

MSCCLPP_DEVICE_INLINE void GpuNetIoDeviceContext::flush(int peer, int qpIndex) {
  doca_gpu_dev_verbs_qp* qp = detail::gpuNetIoQp(*this, peer, qpIndex);
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
  doca_gpu_dev_verbs_qp* qp = detail::gpuNetIoQp(*this, peer, qpIndex);
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
