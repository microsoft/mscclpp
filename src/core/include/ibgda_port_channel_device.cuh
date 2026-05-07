// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// Stage 3: device-side methods that act on `IbgdaPortChannelDeviceHandle`.
// Split from the POD header so that the POD can be included by host code
// (which doesn't want PTX inline asm or CUDA-only types) while these inline
// device functions only appear in .cu translation units.

#ifndef MSCCLPP_IBGDA_PORT_CHANNEL_DEVICE_CUH_
#define MSCCLPP_IBGDA_PORT_CHANNEL_DEVICE_CUH_

#if defined(USE_IBVERBS) && defined(MSCCLPP_USE_MLX5DV) && !defined(MSCCLPP_USE_ROCM)

#include <cuda_runtime.h>

#include <mscclpp/ibgda_port_channel_device.hpp>

#include "ibgda_device.cuh"

namespace mscclpp {
namespace ibgda {

// ---------------- put -----------------------------------------------------
// `signal_cqe` / `ring_db` default to true (preserves prior call sites);
// pass false/false for batched data WRs whose completion is implicitly
// bounded by a later signaled WR on the same QP (e.g., the trailing count
// or flag write at the end of LL dispatch / combine).
__device__ static __forceinline__
void port_put(const IbgdaPortChannelDeviceHandle& ch,
              IbgdaMemoryId dst, uint64_t dstOffset,
              IbgdaMemoryId src, uint64_t srcOffset,
              uint64_t size,
              bool signal_cqe = true, bool ring_db = true) {
  IbgdaLocalMr  l = ch.local_mrs[src];
  IbgdaRemoteMr r = ch.remote_mrs[dst];
  rdma_write(ch.qp,
             l.addr + srcOffset, l.lkey_be,
             r.addr + dstOffset, r.rkey_be,
             static_cast<uint32_t>(size),
             signal_cqe, ring_db);
}

__device__ static __forceinline__
void port_put(const IbgdaPortChannelDeviceHandle& ch,
              uint64_t dstOffset, uint64_t srcOffset, uint64_t size,
              bool signal_cqe = true, bool ring_db = true) {
  port_put(ch, ch.dst, dstOffset, ch.src, srcOffset, size, signal_cqe, ring_db);
}

__device__ static __forceinline__
void port_put(const IbgdaPortChannelDeviceHandle& ch,
              uint64_t offset, uint64_t size,
              bool signal_cqe = true, bool ring_db = true) {
  port_put(ch, offset, offset, size, signal_cqe, ring_db);
}

// ---------------- signal --------------------------------------------------
// Increments the local sequence counter (system-wide atomic so multiple
// CTAs serialise) and writes the new value as an inline 4B RDMA WRITE to
// the peer's signal slot.
__device__ static __forceinline__
uint32_t port_signal(const IbgdaPortChannelDeviceHandle& ch) {
  // Use atomicAdd_system so the counter is consistent across CTAs sharing
  // this handle (rare but possible).
  uint32_t prev = atomicAdd(ch.sig_seq, 1u);
  uint32_t v = prev + 1u;
  rdma_write_inl4(ch.qp, v, ch.sig_remote_addr, ch.sig_rkey_be);
  return v;
}

// ---------------- wait ----------------------------------------------------
// Returns when the local signal slot's value reaches `expected` (sticky;
// callers track the expected counter externally — same pattern as
// PortChannelDeviceHandle::wait()).
__device__ static __forceinline__
void port_wait(const IbgdaPortChannelDeviceHandle& ch, uint32_t expected,
               int64_t maxSpinCount = 10000000) {
  volatile uint32_t* p = ch.sig_local_addr;
  if (maxSpinCount < 0) {
    while (*p < expected) { /* spin */ }
    return;
  }
  for (int64_t i = 0; i < maxSpinCount; ++i) {
    if (*p >= expected) return;
  }
  // Out of patience — caller can detect with a follow-up poll(); we do NOT
  // assert here to keep the device path lean.
}

__device__ static __forceinline__
bool port_poll(const IbgdaPortChannelDeviceHandle& ch, uint32_t expected) {
  return *(volatile uint32_t*)ch.sig_local_addr >= expected;
}

}  // namespace ibgda
}  // namespace mscclpp

#endif  // USE_IBVERBS && MSCCLPP_USE_MLX5DV && !MSCCLPP_USE_ROCM

#endif  // MSCCLPP_IBGDA_PORT_CHANNEL_DEVICE_CUH_
