// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Stage 4b.1: native IBGDA host-side plumbing for src/ext/ep.
//
// Owned by `Buffer` and built lazily in `Buffer::sync()` when the env var
// `MSCCLPP_EP_USE_IBGDA=1` is set AND the run is cross-node
// (num_rdma_ranks > 1). Builds a parallel "shadow" of the existing
// PortChannel layout: a flat (channel × num_ranks) array of
// `IbgdaPortChannelDeviceHandle` POD structs sitting on the GPU, plus the
// host-side resources (IbCtx, QPs, IbgdaResources, MRs, signal slots) that
// keep them valid for the lifetime of the Buffer.
//
// 4b.1 only constructs the state — the kernels do NOT consume it yet. That
// happens in 4b.2 behind a `kIbgdaPath` template branch.

#pragma once

#if defined(USE_IBVERBS) && defined(MSCCLPP_USE_MLX5DV) && !defined(MSCCLPP_USE_ROCM)

#include <cstdint>
#include <memory>
#include <vector>
#include <atomic>
#include <thread>

#include <mscclpp/core.hpp>
#include <mscclpp/ibgda_port_channel_device.hpp>

#include "../../core/include/ib.hpp"
#include "../../core/include/ibgda.hpp"

struct ibv_mr;

namespace mscclpp {
namespace ep {

// One owning bundle of host-side resources backing the device handle array.
struct IbgdaSetup {
  IbgdaSetup() = default;
  ~IbgdaSetup();
  IbgdaSetup(const IbgdaSetup&) = delete;
  IbgdaSetup& operator=(const IbgdaSetup&) = delete;
  // Layout constants (must match the existing port_channel_handles layout in
  // Buffer::sync, so kernels can reuse the same channel × num_ranks index
  // arithmetic in the IBGDA branch).
  int num_channels = 0;     // == num_port_channels_per_rank in Buffer::sync
  int num_ranks = 0;
  int rank = 0;

  // IB context + QPs + Stage-1 GPU mappings.
  std::unique_ptr<IbCtx> ib_ctx;
  // Indexed [channel * num_ranks + peer]; entries with peer == rank are null.
  std::vector<std::shared_ptr<IbQp>> qps;
  std::vector<std::unique_ptr<IbgdaResources>> resources;

  // RDMA buffer MR. We register the *same* `rdma_buffer_ptr` that the
  // existing PortChannel path uses, so dst/src offsets in the kernel work
  // unchanged.
  std::unique_ptr<const IbMr> rdma_mr;

  // Signal slots (GPU-resident).
  // Layout: 4 bytes per (channel, peer) on the receiving side. `sig_slots`
  // is the local mirror polled by `port_wait()`; remote peers RDMA-WRITE
  // into the corresponding slot of *their* signal MR for *us*. Each rank's
  // signal slot for (channel, peer=A) is at offset
  //   offset_in_buf(channel, peer) = (channel * num_ranks + peer) * 4
  // within rank A's signal buffer (i.e. peer=A's view of "messages from
  // rank=A's POV indexed by channel × peer"). See the per-handle wiring
  // in `IbgdaSetup::populate_handles`.
  uint32_t* sig_slots = nullptr;       // GPU mem; size = num_channels * num_ranks * 4
  ibv_mr* sig_mr = nullptr;            // raw verbs (we keep the IbMr only for rdma_mr)
  uint32_t* sig_seq = nullptr;         // GPU mem; per-handle outbound counters

  // Per-peer (addr, rkey) for the RDMA buffer and the signal buffer. Filled
  // by an allgather over the bootstrap.
  struct PeerMr {
    uint64_t addr = 0;
    uint32_t rkey = 0;
    uint32_t pad = 0;
  };
  std::vector<PeerMr> peer_rdma;       // size = num_ranks
  std::vector<PeerMr> peer_sig;        // size = num_ranks

  // Flat device-side handle array: num_channels * num_ranks entries.
  // Self entries (peer == rank) are zeroed and unused.
  std::shared_ptr<IbgdaPortChannelDeviceHandle> device_handles;

  // Underlying GPU-side MR-table arrays referenced by every device handle
  // (see IbgdaPortChannelDeviceHandle::local_mrs / remote_mrs).
  std::shared_ptr<IbgdaLocalMr>  d_local_mrs;     // length 1 (we have a single MR)
  std::shared_ptr<IbgdaRemoteMr> d_remote_mrs;    // length num_ranks

  // CQ drain thread. The kernel-side rdma_write paths set CQ_UPDATE on
  // every WR, so without periodic ibv_poll_cq() the send CQ would fill up
  // (CQ size = kIbgdaMaxSendWr * 2). One thread polls all per-QP send CQs
  // round-robin until `cq_poller_stop` is set in the destructor.
  std::atomic<bool> cq_poller_stop{false};
  std::thread cq_poller_thread;
};

// Build the full IBGDA setup. Bootstrap is used for cross-rank exchange of
// QP info and MR keys; rdma_buffer_ptr/num_rdma_bytes is the same buffer the
// existing PortChannel path uses. `ib_transport_index` is the IB device
// index this rank will use (== `device_id` on NDv5).
//
// Throws on any irrecoverable error. On success returns a fully-initialised
// IbgdaSetup with all QPs in RTS and the device-side handle array populated.
std::unique_ptr<IbgdaSetup> build_ibgda_setup(int rank, int num_ranks, int ib_transport_index, int num_channels,
                                              void* rdma_buffer_ptr, std::size_t num_rdma_bytes,
                                              std::shared_ptr<TcpBootstrap> bootstrap);

}  // namespace ep
}  // namespace mscclpp

#endif  // USE_IBVERBS && MSCCLPP_USE_MLX5DV && !MSCCLPP_USE_ROCM
