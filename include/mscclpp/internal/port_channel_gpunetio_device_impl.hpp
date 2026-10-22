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
MSCCLPP_DEVICE_INLINE __be32 ginHtobe32(uint32_t v) { return static_cast<__be32>(__byte_perm(v, 0, 0x0123)); }
MSCCLPP_DEVICE_INLINE doca_gpu_dev_verbs_addr ginAtomicResult(const GpuNetIoDeviceContext& context, int peer,
                                                              int qpIndex) {
  MSCCLPP_ASSERT_DEVICE(context.atomicResultBase != 0 && context.atomicResultLkeys != nullptr,
                        "GPUNetIO atomics require registered result scratch");
  const size_t flatIndex = static_cast<size_t>(peer) * context.numQpsPerPeer + qpIndex;
  return {context.atomicResultBase + flatIndex * sizeof(uint64_t),
          ginHtobe32(context.atomicResultLkeys[ginHcaIndex(context, qpIndex)])};
}
}  // namespace detail

#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
namespace detail {
struct GinSubmitTrace {
  uint64_t* stages;
  MSCCLPP_DEVICE_INLINE void operator()(unsigned int stage, uint64_t value = 0) const {
    if (stage >= 10) return;
    if (stage < 8) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value)::"memory");
    stages[stage] = value;
  }
};

struct GinSubmitCounts {
  uint64_t* values;
  MSCCLPP_DEVICE_INLINE void operator()(unsigned int stage, uint64_t value = 0) const {
    if (stage == 8) values[0] = value;
    if (stage == 9) values[1] = value;
    if (stage == 10) values[2] = value;
    if (stage == 13) values[3] = value;
    if (stage == 11) values[5] = value;
    if (stage == 12) values[6] = value;
    if (stage == 14) values[7] = value;
  }
};
}  // namespace detail
#endif

namespace detail {
template <class SubmitTrace = doca_gpu_dev_verbs_no_submit_trace>
MSCCLPP_DEVICE_INLINE void ginSubmitBatched(doca_gpu_dev_verbs_qp* qp, uint64_t end, GpuNetIoSubmitState* state,
                                            SubmitTrace trace = {}) {
  if (state != nullptr &&
      __ldg(reinterpret_cast<int*>(&qp->nic_handler)) == DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB) {
    trace(8, DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB);
    trace(14, 1);
    doca_gpu_dev_verbs_submit_coalesced_db(qp, end, &state->published, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT, trace);
  } else {
    doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU, DOCA_GPUNETIO_VERBS_SYNC_SCOPE_THREAD,
                              DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(qp, end, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT,
                                                                    trace);
  }
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
  const auto sigL = detail::ginAtomicResult(*this, peer, qpIndex);
  doca_gpu_dev_verbs_ticket_t ticket;
  doca_gpu_dev_verbs_put_signal<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
      detail::ginQp(qps, peer * numQpsPerPeer + qpIndex), raddr, laddr, size, sigR, sigL, signalValue, &ticket);
}

MSCCLPP_DEVICE_INLINE void GpuNetIoDeviceContext::atomicAdd(int peer, uint64_t dstOffset, int64_t value, int qpIndex) {
  doca_gpu_dev_verbs_addr raddr{peerBase[peer] + dstOffset, detail::ginRemoteKey(*this, peer, qpIndex)};
  const auto laddr = detail::ginAtomicResult(*this, peer, qpIndex);
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
  doca_gpu_dev_verbs_poll_cq_drain<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(cq, ticket);
}

MSCCLPP_DEVICE_INLINE uint64_t GpuNetIoDeviceContext::reservedTicket(int peer, int qpIndex) {
  doca_gpu_dev_verbs_qp* qp = detail::ginQp(qps, peer * numQpsPerPeer + qpIndex);
  return doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(&qp->sq_rsvd_index);
}

MSCCLPP_DEVICE_INLINE void GpuNetIoDeviceContext::waitTicket(int peer, uint64_t ticket, int qpIndex) {
  if (ticket == 0) return;
  doca_gpu_dev_verbs_cq* cq = doca_gpu_dev_verbs_qp_get_cq_sq(detail::ginQp(qps, peer * numQpsPerPeer + qpIndex));
  doca_gpu_dev_verbs_poll_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(cq, ticket - 1);
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
  uint64_t ticket =
      doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(&qp->sq_rsvd_index);
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
                                                              uint64_t size2, [[maybe_unused]] uint64_t* timing,
                                                              [[maybe_unused]] uint64_t* submitTiming,
                                                              [[maybe_unused]] uint64_t* submitCounts) {
  doca_gpu_dev_verbs_qp* qp = detail::ginQp(qps, peer * numQpsPerPeer + qpIndex);
  const uint64_t destinations[3] = {dst0, dst1, dst2};
  const uint64_t sources[3] = {src0, src1, src2};
  const uint64_t sizes[3] = {size0, size1, size2};
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  auto timestamp = []() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value)::"memory");
    return value;
  };
  if (timing != nullptr) timing[0] = timestamp();
#endif
  const uint64_t base = doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
      qp, 3, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  if (timing != nullptr) {
    timing[1] = timestamp();
    timing[5] = base;  // CQTRACE: first WQE index of this post
  }
#endif
#pragma unroll
  for (int index = 0; index < 3; ++index) {
    auto* wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, base + index);
    doca_gpu_dev_verbs_wqe_prepare_write(
        qp, wqe, base + index, DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE, DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0,
        peerBase[peer] + destinations[index], detail::ginRemoteKey(*this, peer, qpIndex), localBase + sources[index],
        detail::ginHtobe32(detail::ginLocalKey(*this, qpIndex)), sizes[index]);
  }
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  if (timing != nullptr) timing[2] = timestamp();
#endif
  doca_gpu_dev_verbs_mark_wqes_ready<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(qp, base, base + 2);
  auto* submitState =
      submitStates == nullptr ? nullptr : submitStates + static_cast<size_t>(peer) * numQpsPerPeer + qpIndex;
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  if (timing != nullptr) timing[3] = timestamp();
  if (submitCounts != nullptr) {
    submitCounts[4] = 3;
    detail::ginSubmitBatched(qp, base + 3, submitState, detail::GinSubmitCounts{submitCounts});
  } else if (submitTiming != nullptr) {
    detail::ginSubmitBatched(qp, base + 3, submitState, detail::GinSubmitTrace{submitTiming});
  } else
#endif
    detail::ginSubmitBatched(qp, base + 3, submitState);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  if (timing != nullptr) timing[4] = timestamp();
#endif
}

MSCCLPP_DEVICE_INLINE void GpuNetIoDeviceContext::putGatherRows(
    int peer, int qpIndex, uint64_t dstOffset, const uint64_t* srcAddrs, int rows, int sgePerWqe, uint32_t rowBytes,
    const uint32_t* srcLkeys, uint64_t dst1, uint64_t src1, uint64_t size1, uint64_t dst2, uint64_t src2,
    uint64_t size2, [[maybe_unused]] uint64_t* timing, [[maybe_unused]] uint64_t* submitTiming,
    [[maybe_unused]] uint64_t* submitCounts) {
  doca_gpu_dev_verbs_qp* qp = detail::ginQp(qps, peer * numQpsPerPeer + qpIndex);
  // sgePerWqe 4 -> one 2-WQEBB write per 4 rows (needs WQEBB-frontier CQ polling); 2 -> 1-WQEBB writes.
  // Negative sgePerWqe: |sgePerWqe| SGEs but force the multi-WQEBB CQ walk (diagnostic).
  const int rowsPerWqe = sgePerWqe < 0 ? -sgePerWqe : sgePerWqe;
  const int dataWqes = (rows + rowsPerWqe - 1) / rowsPerWqe;
  int dataBbs = 0;
  for (int index = 0; index < dataWqes; ++index) {
    const int sges = rows - index * rowsPerWqe < rowsPerWqe ? rows - index * rowsPerWqe : rowsPerWqe;
    dataBbs += (2 + sges + 3) / 4;
  }
  const int total = dataBbs + 2;
  if (dataBbs > dataWqes) {
    ::atomicAdd(&doca_gpu_dev_verbs_qp_get_cq_sq(qp)->multi_bb, static_cast<uint32_t>(dataBbs - dataWqes));
  }
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  auto timestamp = []() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value)::"memory");
    return value;
  };
  if (timing != nullptr) timing[0] = timestamp();
#endif
  const uint64_t base = doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
      qp, total, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  if (timing != nullptr) {
    timing[1] = timestamp();
    timing[5] = base;
  }
#endif
  const uint32_t remoteKey = detail::ginRemoteKey(*this, peer, qpIndex);
  const __be32 localKey = detail::ginHtobe32(detail::ginLocalKey(*this, qpIndex));
  const __be32 gatherKey =
      srcLkeys == nullptr ? localKey : detail::ginHtobe32(srcLkeys[detail::ginHcaIndex(*this, qpIndex)]);
  const uint64_t remoteBase = peerBase[peer];
  const __be32 qpnBase = doca_gpu_dev_verbs_bswap32(__ldg(&qp->sq_num_shift8_be)) & ~0xffu;
  uint64_t bb = base;
  for (int index = 0; index < dataWqes; ++index) {
    const int row = index * rowsPerWqe;
    const int sges = rows - row < rowsPerWqe ? rows - row : rowsPerWqe;
    const int ds = 2 + sges;
    uint64_t segs[6][2];
    doca_gpunetio_ib_mlx5_wqe_ctrl_seg cseg;
    memset(&cseg, 0, sizeof(cseg));
    cseg.opmod_idx_opcode =
        doca_gpu_dev_verbs_bswap32((static_cast<uint32_t>(bb & 0xffff) << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) |
                                   DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE);
    cseg.qpn_ds = doca_gpu_dev_verbs_bswap32(qpnBase | static_cast<uint32_t>(ds));
    cseg.fm_ce_se = DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE;
    cseg.imm = 0;
    memcpy(segs[0], &cseg, 16);
    doca_gpunetio_ib_mlx5_wqe_raddr_seg rseg;
    rseg.raddr = doca_gpu_dev_verbs_bswap64(remoteBase + dstOffset + static_cast<uint64_t>(row) * rowBytes);
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
    rseg.rkey = remoteKey;
#else
    rseg.rkey = doca_gpu_dev_verbs_bswap32(remoteKey);
#endif
    rseg.reserved = 0;
    memcpy(segs[1], &rseg, 16);
    for (int sge = 0; sge < sges; ++sge) {
      doca_gpunetio_ib_mlx5_wqe_data_seg dseg;
      dseg.byte_count = doca_gpu_dev_verbs_bswap32(rowBytes & uint32_t(DOCA_GPUNETIO_IB_MLX5_INLINE_SEG - 1));
#if DOCA_GPUNETIO_VERBS_MKEY_SWAPPED == 1
      dseg.lkey = gatherKey;
#else
      dseg.lkey = doca_gpu_dev_verbs_bswap32(gatherKey);
#endif
      dseg.addr = doca_gpu_dev_verbs_bswap64(srcAddrs[row + sge]);
      memcpy(segs[2 + sge], &dseg, 16);
    }
    for (int seg = 0; seg < ds; ++seg) {
      auto* wqe = reinterpret_cast<uint8_t*>(doca_gpu_dev_verbs_get_wqe_ptr(qp, static_cast<uint16_t>(bb + seg / 4)));
      doca_gpu_dev_verbs_store_wqe_seg(reinterpret_cast<uint64_t*>(wqe + (seg % 4) * 16), segs[seg]);
    }
    bb += (ds + 3) / 4;
  }
  doca_gpu_dev_verbs_wqe_prepare_write(qp, doca_gpu_dev_verbs_get_wqe_ptr(qp, bb), bb,
                                       DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE, DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE,
                                       0, remoteBase + dst1, remoteKey, localBase + src1, localKey,
                                       static_cast<uint32_t>(size1));
  doca_gpu_dev_verbs_wqe_prepare_write(qp, doca_gpu_dev_verbs_get_wqe_ptr(qp, bb + 1),
                                       bb + 1, DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                                       DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0, remoteBase + dst2, remoteKey,
                                       localBase + src2, localKey, static_cast<uint32_t>(size2));
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  if (timing != nullptr) timing[2] = timestamp();
#endif
  doca_gpu_dev_verbs_mark_wqes_ready<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(qp, base, base + total - 1);
  auto* submitState =
      submitStates == nullptr ? nullptr : submitStates + static_cast<size_t>(peer) * numQpsPerPeer + qpIndex;
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  if (timing != nullptr) timing[3] = timestamp();
  if (submitCounts != nullptr) {
    submitCounts[4] = total;
    detail::ginSubmitBatched(qp, base + total, submitState, detail::GinSubmitCounts{submitCounts});
  } else if (submitTiming != nullptr) {
    detail::ginSubmitBatched(qp, base + total, submitState, detail::GinSubmitTrace{submitTiming});
  } else
#endif
    detail::ginSubmitBatched(qp, base + total, submitState);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  if (timing != nullptr) timing[4] = timestamp();
#endif
}

namespace detail {
// Single-poster write post: no shared-QP lock, atomics or second doorbell. Valid only while the calling
// thread is the sole poster on the QP; later GPU-shared operations observe the advanced indices.
template <int N>
MSCCLPP_DEVICE_INLINE void ginPostWritesExclusive(const GpuNetIoDeviceContext& context, int peer, int qpIndex,
                                                  const uint64_t (&destinations)[N], const uint64_t (&sources)[N],
                                                  const uint64_t (&sizes)[N], [[maybe_unused]] uint64_t* timing) {
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  auto timestamp = []() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value)::"memory");
    return value;
  };
  if (timing != nullptr) timing[0] = timestamp();
#endif
  doca_gpu_dev_verbs_qp* qp = ginQp(context.qps, peer * context.numQpsPerPeer + qpIndex);
  const uint64_t base = doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(
      qp, N, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  if (timing != nullptr) {
    timing[1] = timestamp();
    timing[5] = base;  // CHUNKTRACE: first WQE index (exclusive path)
  }
#endif
  const uint32_t remoteKey = ginRemoteKey(context, peer, qpIndex);
  const __be32 localKey = ginHtobe32(ginLocalKey(context, qpIndex));
#pragma unroll
  for (int index = 0; index < N; ++index) {
    auto* wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, base + index);
    doca_gpu_dev_verbs_wqe_prepare_write(qp, wqe, base + index, DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                                         DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0,
                                         context.peerBase[peer] + destinations[index], remoteKey,
                                         context.localBase + sources[index], localKey, sizes[index]);
  }
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  if (timing != nullptr) timing[2] = timestamp();
#endif
  doca_gpu_dev_verbs_mark_wqes_ready<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE>(qp, base, base + N - 1);
  qp->sq_wqe_pi = base + N;
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  if (timing != nullptr) timing[3] = timestamp();
#endif
  doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
  doca_priv_gpu_dev_verbs_update_dbr<DOCA_GPUNETIO_VERBS_QP_SQ>(qp, base + N);
  doca_gpu_dev_verbs_ring_db<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>(qp, base + N);
#if defined(MSCCLPP_EP_GPUNETIO_TIMING)
  if (timing != nullptr) timing[4] = timestamp();
#endif
}
}  // namespace detail

MSCCLPP_DEVICE_INLINE void GpuNetIoDeviceContext::putExclusive(int peer, int qpIndex, uint64_t dstOffset,
                                                               uint64_t srcOffset, uint64_t size) {
  const uint64_t destinations[1] = {dstOffset}, sources[1] = {srcOffset}, sizes[1] = {size};
  detail::ginPostWritesExclusive<1>(*this, peer, qpIndex, destinations, sources, sizes, nullptr);
}

MSCCLPP_DEVICE_INLINE void GpuNetIoDeviceContext::putBatched3Exclusive(int peer, int qpIndex, uint64_t dst0,
                                                                       uint64_t src0, uint64_t size0, uint64_t dst1,
                                                                       uint64_t src1, uint64_t size1, uint64_t dst2,
                                                                       uint64_t src2, uint64_t size2,
                                                                       uint64_t* timing) {
  const uint64_t destinations[3] = {dst0, dst1, dst2}, sources[3] = {src0, src1, src2}, sizes[3] = {size0, size1, size2};
  detail::ginPostWritesExclusive<3>(*this, peer, qpIndex, destinations, sources, sizes, timing);
}

}  // namespace mscclpp

#endif  // MSCCLPP_INTERNAL_PORT_CHANNEL_GPUNETIO_DEVICE_IMPL_HPP_
