// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
// Stage 3: device-side IBGDA "port channel" handle.
//
// Mirrors the device-facing API of mscclpp::PortChannelDeviceHandle (see
// include/mscclpp/port_channel_device.hpp) but issues RDMA traffic directly
// from the GPU via mscclpp::ibgda::rdma_write — no host proxy / FIFO.
//
//   put(dstOffset, srcOffset, size)  — issue 1 RDMA WRITE WQE
//   signal()                         — issue 1 inline 4-byte RDMA WRITE to a
//                                      remote 4B counter slot
//   wait()                           — spin on the local-side mirror of that
//                                      counter
//
// Memory model (each rank constructs):
//   - For every registered local memory region M, a flat row
//       local_mrs[M] = { base_addr, lkey_be }
//   - For every (peer rank R, registered region M) pair, a flat row
//       remote_mrs[M] = { base_addr_on_R, rkey_be_on_R }
//     (a channel is bound to a single peer R, so the rank dimension is
//     flattened away inside the handle).
//
// The signal counter:
//   - Each rank cudaMallocs a 4-byte "signal slot" registered with this NIC,
//     and exchanges its address+rkey with peers. The handle's
//     {sig_remote_addr, sig_rkey_be} points at the *peer's* slot, while
//     sig_local_addr is the local mirror used by wait() to compare against.
//
//   - signal() atomically bumps sig_seq (CTA-shared-by-channel u32 in GPU
//     memory), writes that value as a 4-byte inline RDMA WRITE to peer's
//     slot. Receiver polls its local slot.

#ifndef MSCCLPP_IBGDA_PORT_CHANNEL_DEVICE_HPP_
#define MSCCLPP_IBGDA_PORT_CHANNEL_DEVICE_HPP_

#if defined(USE_IBVERBS) && defined(MSCCLPP_USE_MLX5DV) && !defined(MSCCLPP_USE_ROCM)

#include <cstdint>

#include "ibgda.hpp"

namespace mscclpp {

using IbgdaMemoryId = uint32_t;

struct IbgdaLocalMr {
  uint64_t addr;       // host-order base virtual address (GPU-mapped)
  uint32_t lkey_be;    // BE-encoded local key
  uint32_t pad;
};

struct IbgdaRemoteMr {
  uint64_t addr;       // host-order base virtual address on the peer
  uint32_t rkey_be;    // BE-encoded remote key
  uint32_t pad;
};

// POD copied to GPU. Fields kept dense and 8-byte aligned.
struct IbgdaPortChannelDeviceHandle {
  IbgdaQpHandle qp;

  // Tables: indexed by IbgdaMemoryId. local_mrs[id] is for THIS rank;
  // remote_mrs[id] is for the peer rank this channel is bound to.
  // Both pointers must dereference from device code.
  const IbgdaLocalMr*  local_mrs;
  const IbgdaRemoteMr* remote_mrs;

  // Signal slot:
  //   sig_local_addr  — pointer to a 4-byte slot in this rank's GPU memory
  //                     used by wait() to poll. NIC writes here from the peer.
  //   sig_local_lkey  — BE-encoded lkey of the local 4B inline staging
  //                     buffer (same MR as sig_local_addr; we use it as both
  //                     the receive slot for wait() and as a backing MR for
  //                     completeness — but inline WRs don't actually need
  //                     a local MR. Kept for symmetry / future non-inline
  //                     fallback).
  //   sig_remote_addr — peer's 4-byte slot
  //   sig_rkey_be     — peer's rkey for sig_remote_addr
  //   sig_seq         — pointer to a 4-byte GPU-resident counter incremented
  //                     by signal(); the value sent is the post-increment.
  //                     Distinct from sig_local_addr.
  uint32_t* sig_local_addr;
  uint32_t  sig_local_lkey;  // unused for inline; kept for layout stability
  uint64_t  sig_remote_addr;
  uint32_t  sig_rkey_be;
  uint32_t* sig_seq;

  // Bound peer (informational); MemoryIds default to (0,0) in the simple
  // case but we keep them for parity with PortChannelDeviceHandle.
  IbgdaMemoryId dst;
  IbgdaMemoryId src;
  uint32_t      peer_rank;
  uint32_t      _pad;
};

}  // namespace mscclpp

#endif  // USE_IBVERBS && MSCCLPP_USE_MLX5DV && !MSCCLPP_USE_ROCM

#endif  // MSCCLPP_IBGDA_PORT_CHANNEL_DEVICE_HPP_
