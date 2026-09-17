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
MSCCLPP_DEVICE_INLINE doca_gpu_dev_verbs_qp* ginQp(void* qps, int flatQpIndex) {
  return reinterpret_cast<doca_gpu_dev_verbs_qp*>(qps) + flatQpIndex;
}
MSCCLPP_DEVICE_INLINE int ginHcaIndex(const GpuNetIoDeviceContext& context, int qpIndex) {
  return context.numHcas > 1 ? qpIndex % context.numHcas : 0;
}
MSCCLPP_DEVICE_INLINE uint32_t ginRemoteKey(const GpuNetIoDeviceContext& context, int peer, int qpIndex) {
  return context.rkeys[static_cast<size_t>(ginHcaIndex(context, qpIndex)) * context.numPeers + peer];
}
MSCCLPP_DEVICE_INLINE uint32_t ginLocalKey(const GpuNetIoDeviceContext& context, int qpIndex) {
  return context.lkeys == nullptr ? context.lkey : context.lkeys[ginHcaIndex(context, qpIndex)];
}
MSCCLPP_DEVICE_INLINE __be32 ginHtobe32(uint32_t v) {
  return static_cast<__be32>(__byte_perm(v, 0, 0x0123));
}
}  // namespace detail

MSCCLPP_DEVICE_INLINE void GpuNetIoDeviceContext::put(int peer, uint64_t dstOffset, uint64_t srcOffset, uint64_t size,
                                                      int qpIndex) {
  doca_gpu_dev_verbs_addr raddr{peerBase[peer] + dstOffset, detail::ginRemoteKey(*this, peer, qpIndex)};
  doca_gpu_dev_verbs_addr laddr{localBase + srcOffset, detail::ginHtobe32(detail::ginLocalKey(*this, qpIndex))};
  doca_gpu_dev_verbs_ticket_t ticket;
  doca_gpu_dev_verbs_put<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
      detail::ginQp(qps, peer * numQpsPerPeer + qpIndex), raddr, laddr, size, &ticket);
}

MSCCLPP_DEVICE_INLINE void GpuNetIoDeviceContext::putWithSignal(int peer, uint64_t dstOffset, uint64_t srcOffset,
                                                                uint64_t size, uint64_t signalOffset,
                                                                uint64_t signalValue, int qpIndex) {
  const uint32_t remoteKey = detail::ginRemoteKey(*this, peer, qpIndex);
  const __be32 localKey = detail::ginHtobe32(detail::ginLocalKey(*this, qpIndex));
  doca_gpu_dev_verbs_addr raddr{peerBase[peer] + dstOffset, remoteKey};
  doca_gpu_dev_verbs_addr laddr{localBase + srcOffset, localKey};
  doca_gpu_dev_verbs_addr sigR{peerBase[peer] + signalOffset, remoteKey};
  doca_gpu_dev_verbs_addr sigL{localBase, localKey};
  doca_gpu_dev_verbs_ticket_t ticket;
  doca_gpu_dev_verbs_put_signal<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
      detail::ginQp(qps, peer * numQpsPerPeer + qpIndex), raddr, laddr, size, sigR, sigL, signalValue, &ticket);
}

MSCCLPP_DEVICE_INLINE void GpuNetIoDeviceContext::atomicAdd(int peer, uint64_t dstOffset, int64_t value, int qpIndex) {
  doca_gpu_dev_verbs_addr raddr{peerBase[peer] + dstOffset, detail::ginRemoteKey(*this, peer, qpIndex)};
  doca_gpu_dev_verbs_addr laddr{localBase, detail::ginHtobe32(detail::ginLocalKey(*this, qpIndex))};
  doca_gpu_dev_verbs_ticket_t ticket;
  // Fused zero-byte write + remote atomic-add expresses a standalone atomic add.
  doca_gpu_dev_verbs_put_signal<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
      detail::ginQp(qps, peer * numQpsPerPeer + qpIndex), raddr, laddr, /*size=*/0, raddr, laddr,
      static_cast<uint64_t>(value), &ticket);
}

MSCCLPP_DEVICE_INLINE void GpuNetIoDeviceContext::flush(int peer, int qpIndex) {
  // Drain through the latest reserved ticket so the shared dispatch/combine CQ
  // advances and its signaled entries can be recycled across iterations.
  doca_gpu_dev_verbs_qp* qp = detail::ginQp(qps, peer * numQpsPerPeer + qpIndex);
  uint64_t ticket =
      doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(&qp->sq_rsvd_index);
  if (ticket == 0) return;
  doca_gpu_dev_verbs_cq* cq = doca_gpu_dev_verbs_qp_get_cq_sq(qp);
  while (doca_gpu_dev_verbs_poll_one_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(cq, ticket - 1) == EBUSY) {
  }
}

MSCCLPP_DEVICE_INLINE void GpuNetIoDeviceContext::get(int peer, uint64_t remoteOffset, uint64_t localOffset,
                                                      uint64_t size, int qpIndex) {
  doca_gpu_dev_verbs_addr raddr{peerBase[peer] + remoteOffset, detail::ginRemoteKey(*this, peer, qpIndex)};
  doca_gpu_dev_verbs_addr laddr{localBase + localOffset, detail::ginHtobe32(detail::ginLocalKey(*this, qpIndex))};
  doca_gpu_dev_verbs_qp* qp = detail::ginQp(qps, peer * numQpsPerPeer + qpIndex);
  doca_gpu_dev_verbs_ticket_t ticket;
  doca_gpu_dev_verbs_get_thread<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(qp, raddr, laddr, size, laddr, &ticket);
  // Wait for this read, not unrelated operations concurrently reserved on the QP.
  doca_gpu_dev_verbs_wait(qp, ticket);
}

MSCCLPP_DEVICE_INLINE int GpuNetIoDeviceContext::tryFlush(int peer, uint64_t maxSpinCount, int qpIndex) {
  doca_gpu_dev_verbs_qp* qp = detail::ginQp(qps, peer * numQpsPerPeer + qpIndex);
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

MSCCLPP_DEVICE_INLINE void GpuNetIoDeviceContext::putBatched3(int peer, int qpIndex, uint64_t dst0, uint64_t src0,
                                                              uint64_t size0, uint64_t dst1, uint64_t src1,
                                                              uint64_t size1, uint64_t dst2, uint64_t src2,
                                                              uint64_t size2) {
  doca_gpu_dev_verbs_qp* qp = detail::ginQp(qps, peer * numQpsPerPeer + qpIndex);
  const uint64_t destinations[3] = {dst0, dst1, dst2};
  const uint64_t sources[3] = {src0, src1, src2};
  const uint64_t sizes[3] = {size0, size1, size2};
  const uint64_t base = doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
      qp, 3, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT);
#pragma unroll
  for (int index = 0; index < 3; ++index) {
    auto* wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, base + index);
    doca_gpu_dev_verbs_wqe_prepare_write(
        qp, wqe, base + index, DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE, DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0,
        peerBase[peer] + destinations[index], detail::ginRemoteKey(*this, peer, qpIndex), localBase + sources[index],
        detail::ginHtobe32(detail::ginLocalKey(*this, qpIndex)), sizes[index]);
  }
  doca_gpu_dev_verbs_mark_wqes_ready<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(qp, base, base + 2);
  doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU, DOCA_GPUNETIO_VERBS_SYNC_SCOPE_THREAD,
                            DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(qp, base + 3,
                                                                  DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT);
}

}  // namespace mscclpp

#endif  // MSCCLPP_INTERNAL_PORT_CHANNEL_GPUNETIO_DEVICE_IMPL_HPP_
