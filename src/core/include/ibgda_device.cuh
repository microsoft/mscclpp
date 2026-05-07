// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// Stage 2: device-side IBGDA primitives.
//
// Operates on an `IbgdaQpHandle` (see ibgda.hpp). All pointers in the handle
// must already be GPU-mapped by `IbgdaResources` on the host. This header is
// device-only and should be included from .cu translation units.
//
// State buffer layout (allocated by IbgdaResources, zero-initialised):
//   offset  0: uint64_t resv_head       // next free WQEBB slot (atomic)
//   offset  8: uint64_t ready_head      // next slot whose WQE write is done
//   offset 16: uint64_t prod_idx        // last value rung on the doorbell
//   offset 24: int      post_send_lock  // cta-level lock for ringing
//
// Conventions copied from NVSHMEM/DeepEP:
//   - wqe_idx is a 16-bit-truncated cyclic index into the SQ ring.
//   - one RDMA WRITE WQE = 1 WQEBB (64B): ctrl(16) + raddr(16) + data(16) + pad(16).
//     ctrl_seg.qpn_ds encodes ds=3 (three 16-byte segments).
//   - DBR record holds BE32(prod_idx & 0xFFFF) in the SQ slot (dbrec[1] for
//     mlx5; IbgdaResources sets handle.dbrec to that slot directly).
//   - BF doorbell: 64-bit write of {opmod_idx_opcode, qpn_ds} to bf_reg.

#ifndef MSCCLPP_IBGDA_DEVICE_CUH_
#define MSCCLPP_IBGDA_DEVICE_CUH_

#include <cstdint>
#include <cuda_runtime.h>

#include "ibgda.hpp"

namespace mscclpp {
namespace ibgda {

#ifndef MSCCLPP_IBGDA_OPCODE_RDMA_WRITE
#define MSCCLPP_IBGDA_OPCODE_RDMA_WRITE  0x08
#endif
#ifndef MSCCLPP_IBGDA_CTRL_CQ_UPDATE
#define MSCCLPP_IBGDA_CTRL_CQ_UPDATE     uint8_t(0x8)
#endif

// ---- BE swap helpers (PTX prmt) -------------------------------------------
__device__ static __forceinline__ uint32_t HtoBE32(uint32_t x) {
  uint32_t r;
  asm("{ .reg .b32 ig; prmt.b32 %0, %1, ig, 0x0123; }" : "=r"(r) : "r"(x));
  return r;
}

__device__ static __forceinline__ uint64_t HtoBE64(uint64_t x) {
  uint64_t r;
  asm("{\n\t"
      ".reg .b32 ig;\n\t"
      ".reg .b32 lo, hi, nl, nh;\n\t"
      "mov.b64 {lo,hi}, %1;\n\t"
      "prmt.b32 nh, lo, ig, 0x0123;\n\t"
      "prmt.b32 nl, hi, ig, 0x0123;\n\t"
      "mov.b64 %0, {nl,nh};\n\t" "}"
      : "=l"(r) : "l"(x));
  return r;
}

// ---- Memory ordering helpers ----------------------------------------------
// st.relaxed.sys + st.release.sys equivalents. We use `volatile` writes plus
// __threadfence_system() at the points the caller needs ordering, mirroring
// the NVSHMEM `st_na_*` style without depending on its internal headers.
__device__ static __forceinline__ void store_relaxed_u32(uint32_t* p, uint32_t v) {
  *reinterpret_cast<volatile uint32_t*>(p) = v;
}
__device__ static __forceinline__ void store_relaxed_u64(uint64_t* p, uint64_t v) {
  *reinterpret_cast<volatile uint64_t*>(p) = v;
}
__device__ static __forceinline__ void store_relaxed_int4(int4* p, int4 v) {
  asm volatile("st.relaxed.sys.v4.b32 [%0], {%1, %2, %3, %4};" ::
               "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}
__device__ static __forceinline__ void store_release_u32(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.b32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}
__device__ static __forceinline__ void store_release_u64(uint64_t* p, uint64_t v) {
  asm volatile("st.release.sys.b64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}

// ---- WQE segment layout (mirrors mlx5_wqe_*; redefined to avoid pulling
// libmlx5 headers into device code) ----------------------------------------
struct __align__(8) CtrlSeg {
  uint32_t opmod_idx_opcode;  // BE32: (wqe_idx << 8) | opcode
  uint32_t qpn_ds;            // BE32: (qpn << 8) | ds
  uint8_t  signature;
  uint8_t  rsvd0;
  uint8_t  rsvd1;
  uint8_t  fm_ce_se;          // CQ_UPDATE in bit 3 (0x8)
  uint32_t imm;               // BE32 immediate (or 0)
};
struct __align__(8) RaddrSeg {
  uint64_t raddr;             // BE64
  uint32_t rkey;              // BE32 (caller pre-swaps)
  uint32_t reserved;
};
struct __align__(8) DataSeg {
  uint32_t byte_count;        // BE32
  uint32_t lkey;              // BE32 (caller pre-swaps)
  uint64_t addr;              // BE64
};

static_assert(sizeof(CtrlSeg) == 16, "CtrlSeg must be 16B");
static_assert(sizeof(RaddrSeg) == 16, "RaddrSeg must be 16B");
static_assert(sizeof(DataSeg) == 16, "DataSeg must be 16B");

// ---- State accessors ------------------------------------------------------
struct StateView {
  unsigned long long* resv_head;
  unsigned long long* ready_head;
  unsigned long long* prod_idx;
  int*                post_send_lock;
};

__device__ static __forceinline__ StateView stateView(const IbgdaQpHandle& h) {
  auto* base = reinterpret_cast<uint64_t*>(h.state);
  StateView v;
  v.resv_head      = reinterpret_cast<unsigned long long*>(&base[0]);
  v.ready_head     = reinterpret_cast<unsigned long long*>(&base[1]);
  v.prod_idx       = reinterpret_cast<unsigned long long*>(&base[2]);
  v.post_send_lock = reinterpret_cast<int*>(&base[3]);
  return v;
}

// ---- WQE ring helpers -----------------------------------------------------
__device__ static __forceinline__
uint64_t reserve_wqe_slots(const IbgdaQpHandle& h, uint32_t num_wqes) {
  auto v = stateView(h);
  return atomicAdd(v.resv_head, static_cast<unsigned long long>(num_wqes));
}

__device__ static __forceinline__
void* get_wqe_ptr(const IbgdaQpHandle& h, uint16_t wqe_idx) {
  uint16_t mask = static_cast<uint16_t>(h.wqe_cnt - 1);
  uint16_t idx = wqe_idx & mask;
  return reinterpret_cast<uint8_t*>(h.sq_buf) +
         (static_cast<size_t>(idx) * h.stride);
}

// ---- Doorbell record + BF ring -------------------------------------------
__device__ static __forceinline__
void update_dbr(const IbgdaQpHandle& h, uint32_t dbrec_head) {
  // BE32(dbrec_head & 0xFFFF) — see DeepEP/NVSHMEM.
  uint32_t v;
  asm("{\n\t"
      ".reg .b32 lo16; .reg .b32 ig;\n\t"
      "and.b32 lo16, %1, 0xffff;\n\t"
      "prmt.b32 %0, lo16, ig, 0x123;\n\t"
      "}" : "=r"(v) : "r"(dbrec_head));
  // dbrec is the SQ slot directly (set by IbgdaResources).
  store_release_u32(reinterpret_cast<uint32_t*>(h.dbrec), v);
}

__device__ static __forceinline__
void ring_db(const IbgdaQpHandle& h, uint16_t prod_idx) {
  // 64-bit BF write of {opmod_idx_opcode, qpn_ds}.
  // opmod_idx_opcode = BE32(prod_idx << 8); qpn_ds = BE32(qpn << 8).
  uint32_t lo = HtoBE32(static_cast<uint32_t>(prod_idx) << 8);
  uint32_t hi = HtoBE32(h.qpn << 8);
  uint64_t v = static_cast<uint64_t>(lo) | (static_cast<uint64_t>(hi) << 32);
  store_release_u64(h.bf_reg, v);
}

__device__ static __forceinline__
void post_send(const IbgdaQpHandle& h, uint64_t new_prod_idx) {
  auto v = stateView(h);
  // CTA-level lock so concurrent posters serialise their DBR/BF writes.
  while (atomicCAS(v.post_send_lock, 0, 1) == 1) {}
  __threadfence();

  unsigned long long old_prod_idx =
      atomicMax(v.prod_idx, static_cast<unsigned long long>(new_prod_idx));
  if (new_prod_idx > old_prod_idx) {
    update_dbr(h, static_cast<uint32_t>(new_prod_idx));
    ring_db(h, static_cast<uint16_t>(new_prod_idx));
  }

  __threadfence();
  store_release_u32(reinterpret_cast<uint32_t*>(v.post_send_lock), 0);
}

// ---- Submit (publish ready_head, optionally ring DB) ----------------------
template <bool kAlwaysDoPostSend>
__device__ static __forceinline__
void submit_requests(const IbgdaQpHandle& h, uint64_t base_wqe_idx,
                     uint32_t num_wqes, int message_idx = 0) {
  auto v = stateView(h);
  uint64_t new_wqe_idx = base_wqe_idx + num_wqes;

  // The WQE writes themselves must be globally visible before we publish.
  __threadfence_system();

  // Wait for any earlier reservations to publish first, preserving order.
  auto base_ull = static_cast<unsigned long long>(base_wqe_idx);
  auto new_ull  = static_cast<unsigned long long>(new_wqe_idx);
  while (atomicCAS(v.ready_head, base_ull, new_ull) != base_ull) {}

  constexpr int kBatch = 4;
  if (kAlwaysDoPostSend || ((message_idx + 1) % kBatch == 0)) {
    post_send(h, new_wqe_idx);
  }
}

// Publish-only variant: advance ready_head in WQE-issue order (preserving the
// in-order chain that submit_requests builds) but DO NOT ring the doorbell.
// Use this for the data WRs in a coalesced "unsignaled body + signaled tail"
// pattern; the trailing WR's submit_requests<true> will atomicMax prod_idx
// past all earlier WRs and ring the doorbell once for the whole batch.
__device__ static __forceinline__
void submit_no_db(const IbgdaQpHandle& h, uint64_t base_wqe_idx,
                  uint32_t num_wqes) {
  auto v = stateView(h);
  uint64_t new_wqe_idx = base_wqe_idx + num_wqes;
  __threadfence_system();
  auto base_ull = static_cast<unsigned long long>(base_wqe_idx);
  auto new_ull  = static_cast<unsigned long long>(new_wqe_idx);
  while (atomicCAS(v.ready_head, base_ull, new_ull) != base_ull) {}
  // No post_send: the next signaled WR on this QP (or any concurrent ringer)
  // will atomicMax prod_idx past us and the NIC will pick up our slot.
}

// ---- WQE writers ----------------------------------------------------------
// All addresses BE-encoded inside; lkey/rkey expected pre-BE-swapped.
// `signal_cqe`: if true, set CQ_UPDATE so the NIC posts a CQE for this WR;
// if false, the WR is UNSIGNALED — useful for batched data WRs whose
// completion is implicitly bounded by a later signaled WR on the same QP.
__device__ static __forceinline__
void write_rdma_write_wqe(const IbgdaQpHandle& h, void* wqe_slot,
                          uint64_t laddr, uint32_t lkey_be,
                          uint64_t raddr, uint32_t rkey_be,
                          uint32_t bytes, uint16_t wqe_idx,
                          bool signal_cqe = true) {
  auto* ctrl  = reinterpret_cast<CtrlSeg*>(wqe_slot);
  auto* raddrp = reinterpret_cast<RaddrSeg*>(reinterpret_cast<uint8_t*>(wqe_slot) + 16);
  auto* datap  = reinterpret_cast<DataSeg*>(reinterpret_cast<uint8_t*>(wqe_slot) + 32);

  CtrlSeg c{};
  c.opmod_idx_opcode = HtoBE32((static_cast<uint32_t>(wqe_idx) << 8) |
                               MSCCLPP_IBGDA_OPCODE_RDMA_WRITE);
  c.qpn_ds = HtoBE32((h.qpn << 8) | 3u);
  c.fm_ce_se = signal_cqe ? MSCCLPP_IBGDA_CTRL_CQ_UPDATE : 0u;
  c.imm = 0;

  RaddrSeg r{};
  r.raddr = HtoBE64(raddr);
  r.rkey = rkey_be;
  r.reserved = 0;

  DataSeg d{};
  d.byte_count = HtoBE32(bytes);
  d.lkey = lkey_be;
  d.addr = HtoBE64(laddr);

  store_relaxed_int4(reinterpret_cast<int4*>(ctrl),   *reinterpret_cast<int4*>(&c));
  store_relaxed_int4(reinterpret_cast<int4*>(raddrp), *reinterpret_cast<int4*>(&r));
  store_relaxed_int4(reinterpret_cast<int4*>(datap),  *reinterpret_cast<int4*>(&d));
}

// One-shot helper: reserve a slot, write WQE, submit + ring doorbell.
// Returns the wqe_idx of the issued WR (caller can match against CQE wr_id
// if it embeds the index there; here we don't use wr_id).
__device__ static __forceinline__
uint64_t rdma_write(const IbgdaQpHandle& h,
                    uint64_t laddr, uint32_t lkey_be,
                    uint64_t raddr, uint32_t rkey_be,
                    uint32_t bytes,
                    bool signal_cqe = true, bool ring_db = true) {
  uint64_t base = reserve_wqe_slots(h, 1);
  void* slot = get_wqe_ptr(h, static_cast<uint16_t>(base));
  write_rdma_write_wqe(h, slot, laddr, lkey_be, raddr, rkey_be, bytes,
                       static_cast<uint16_t>(base), signal_cqe);
  if (ring_db) submit_requests<true>(h, base, 1);
  else         submit_no_db(h, base, 1);
  return base;
}

// ---- Inline 4-byte RDMA WRITE --------------------------------------------
// Single WQEBB carrying (ctrl=16) + (raddr=16) + (inline_hdr=4 + data=4)
// padded to 48B. ds=3 in ctrl_seg.
//
// MLX5_INLINE_SEG = 0x80000000 in inline_seg.byte_count signals "inline".
//
// The 4-byte payload `value` is written verbatim to remote `raddr`. The
// caller does NOT need a local MR for this op.
#ifndef MSCCLPP_IBGDA_INLINE_SEG
#define MSCCLPP_IBGDA_INLINE_SEG 0x80000000u
#endif

__device__ static __forceinline__
void write_rdma_write_inl4_wqe(const IbgdaQpHandle& h, void* wqe_slot,
                               uint32_t value,
                               uint64_t raddr, uint32_t rkey_be,
                               uint16_t wqe_idx,
                               bool signal_cqe = true) {
  auto* base8 = reinterpret_cast<uint8_t*>(wqe_slot);
  auto* ctrl   = reinterpret_cast<CtrlSeg*>(base8 +  0);
  auto* raddrp = reinterpret_cast<RaddrSeg*>(base8 + 16);
  // inline header is a 4-byte field (byte_count) at offset 32, followed by
  // 4 bytes of payload at offset 36.
  uint32_t* inl_hdr = reinterpret_cast<uint32_t*>(base8 + 32);
  uint32_t* inl_dat = reinterpret_cast<uint32_t*>(base8 + 36);

  CtrlSeg c{};
  c.opmod_idx_opcode = HtoBE32((static_cast<uint32_t>(wqe_idx) << 8) |
                               MSCCLPP_IBGDA_OPCODE_RDMA_WRITE);
  // ds=3 (3 * 16B = 48B; ctrl + raddr + 8B of inline hdr+data, padded).
  c.qpn_ds = HtoBE32((h.qpn << 8) | 3u);
  c.fm_ce_se = signal_cqe ? MSCCLPP_IBGDA_CTRL_CQ_UPDATE : 0u;
  c.imm = 0;

  RaddrSeg r{};
  r.raddr = HtoBE64(raddr);
  r.rkey = rkey_be;
  r.reserved = 0;

  store_relaxed_int4(reinterpret_cast<int4*>(ctrl),   *reinterpret_cast<int4*>(&c));
  store_relaxed_int4(reinterpret_cast<int4*>(raddrp), *reinterpret_cast<int4*>(&r));
  // inline byte_count: 4 | INLINE_SEG, BE32.
  store_relaxed_u32(inl_hdr, HtoBE32(4u | MSCCLPP_IBGDA_INLINE_SEG));
  // The 4-byte payload is written as-is in network order — the NIC treats
  // the inline data block as opaque bytes copied to remote memory.
  store_relaxed_u32(inl_dat, value);
}

__device__ static __forceinline__
uint64_t rdma_write_inl4(const IbgdaQpHandle& h,
                         uint32_t value,
                         uint64_t raddr, uint32_t rkey_be,
                         bool signal_cqe = true, bool ring_db = true) {
  uint64_t base = reserve_wqe_slots(h, 1);
  void* slot = get_wqe_ptr(h, static_cast<uint16_t>(base));
  write_rdma_write_inl4_wqe(h, slot, value, raddr, rkey_be,
                            static_cast<uint16_t>(base), signal_cqe);
  if (ring_db) submit_requests<true>(h, base, 1);
  else         submit_no_db(h, base, 1);
  return base;
}

// ---- Inline 8-byte RDMA WRITE --------------------------------------------
// Same WQE shape as inl4 (ds=3 / 48B): ctrl(16) + raddr(16) + inline_hdr(4)
// + payload(8) + pad(4). Used to publish 8B counters (e.g. dispatch
// per-(local_expert, src_rank) recv_count slot polled with int64 acquire).
__device__ static __forceinline__
void write_rdma_write_inl8_wqe(const IbgdaQpHandle& h, void* wqe_slot,
                               uint64_t value,
                               uint64_t raddr, uint32_t rkey_be,
                               uint16_t wqe_idx,
                               bool signal_cqe = true) {
  auto* base8 = reinterpret_cast<uint8_t*>(wqe_slot);
  auto* ctrl   = reinterpret_cast<CtrlSeg*>(base8 +  0);
  auto* raddrp = reinterpret_cast<RaddrSeg*>(base8 + 16);
  uint32_t* inl_hdr = reinterpret_cast<uint32_t*>(base8 + 32);
  // Two 32-bit halves of the 8B payload at offsets 36 / 40.
  uint32_t* inl_dat_lo = reinterpret_cast<uint32_t*>(base8 + 36);
  uint32_t* inl_dat_hi = reinterpret_cast<uint32_t*>(base8 + 40);

  CtrlSeg c{};
  c.opmod_idx_opcode = HtoBE32((static_cast<uint32_t>(wqe_idx) << 8) |
                               MSCCLPP_IBGDA_OPCODE_RDMA_WRITE);
  c.qpn_ds = HtoBE32((h.qpn << 8) | 3u);
  c.fm_ce_se = signal_cqe ? MSCCLPP_IBGDA_CTRL_CQ_UPDATE : 0u;
  c.imm = 0;

  RaddrSeg r{};
  r.raddr = HtoBE64(raddr);
  r.rkey = rkey_be;
  r.reserved = 0;

  store_relaxed_int4(reinterpret_cast<int4*>(ctrl),   *reinterpret_cast<int4*>(&c));
  store_relaxed_int4(reinterpret_cast<int4*>(raddrp), *reinterpret_cast<int4*>(&r));
  // inline byte_count: 8 | INLINE_SEG, BE32.
  store_relaxed_u32(inl_hdr, HtoBE32(8u | MSCCLPP_IBGDA_INLINE_SEG));
  // 8B payload as two 32-bit words; treated as opaque bytes by the NIC.
  store_relaxed_u32(inl_dat_lo, static_cast<uint32_t>(value & 0xffffffffull));
  store_relaxed_u32(inl_dat_hi, static_cast<uint32_t>(value >> 32));
}

__device__ static __forceinline__
uint64_t rdma_write_inl8(const IbgdaQpHandle& h,
                         uint64_t value,
                         uint64_t raddr, uint32_t rkey_be,
                         bool signal_cqe = true, bool ring_db = true) {
  uint64_t base = reserve_wqe_slots(h, 1);
  void* slot = get_wqe_ptr(h, static_cast<uint16_t>(base));
  write_rdma_write_inl8_wqe(h, slot, value, raddr, rkey_be,
                            static_cast<uint16_t>(base), signal_cqe);
  if (ring_db) submit_requests<true>(h, base, 1);
  else         submit_no_db(h, base, 1);
  return base;
}

// ---- Warp-coalesced burst (lane 0 issues N WRs as a single batch) --------
// This is the "warp granularity" pattern used by NVSHMEM IBGDA: a single
// thread reserves N contiguous slots (one atomic), writes N WQEs in a tight
// loop, then publishes ready_head and rings the doorbell exactly once. This
// amortises the ready_head CAS-spin and the BF doorbell MMIO across N WRs.
//
// Caller is responsible for ensuring that only one thread per warp invokes
// this with a given (laddr/raddr) layout — typically `if (lane_id == 0)`.
// All N WRs share the same lkey_be/rkey_be and have stride `bytes` (i.e. a
// contiguous chunk laddr_base..laddr_base+N*bytes -> raddr_base..).
__device__ static __forceinline__
uint64_t rdma_write_strided_burst(const IbgdaQpHandle& h,
                                  uint64_t laddr_base, uint32_t lkey_be,
                                  uint64_t raddr_base, uint32_t rkey_be,
                                  uint32_t bytes, uint32_t num_wrs) {
  if (num_wrs == 0) return 0;
  uint64_t base = reserve_wqe_slots(h, num_wrs);
  for (uint32_t i = 0; i < num_wrs; ++i) {
    uint16_t idx = static_cast<uint16_t>(base + i);
    void* slot = get_wqe_ptr(h, idx);
    write_rdma_write_wqe(h, slot,
                         laddr_base + size_t(i) * bytes, lkey_be,
                         raddr_base + size_t(i) * bytes, rkey_be,
                         bytes, idx);
  }
  submit_requests<true>(h, base, num_wrs);
  return base;
}

}  // namespace ibgda
}  // namespace mscclpp

#endif  // MSCCLPP_IBGDA_DEVICE_CUH_
