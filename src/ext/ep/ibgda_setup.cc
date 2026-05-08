// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Stage 4b.1 implementation. See ibgda_setup.hpp for the contract.

#include "ibgda_setup.hpp"

#if defined(USE_IBVERBS) && defined(MSCCLPP_USE_MLX5DV) && !defined(MSCCLPP_USE_ROCM)

#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <infiniband/verbs.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <thread>
#include <chrono>

#include <mscclpp/core.hpp>
#include <mscclpp/errors.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/utils.hpp>

#include "kernels/exception.cuh"

namespace mscclpp {
namespace ep {

IbgdaSetup::~IbgdaSetup() {
  // Stop the CQ poller first so it doesn't race with QP teardown.
  if (cq_poller_thread.joinable()) {
    cq_poller_stop.store(true, std::memory_order_release);
    cq_poller_thread.join();
  }
  // Tear down in reverse order of construction:
  //   - device_handles, d_local_mrs, d_remote_mrs: shared_ptr from
  //     gpuCallocShared, auto-freed via custom deleter.
  //   - resources / qps / rdma_mr: smart ptrs, auto-freed.
  //   - sig_mr (raw ibv_mr) + sig_slots / sig_seq (raw cudaMalloc): explicit.
  if (sig_mr != nullptr) {
    ibv_dereg_mr(sig_mr);
    sig_mr = nullptr;
  }
  if (sig_slots != nullptr) {
    cudaFree(sig_slots);
    sig_slots = nullptr;
  }
  if (sig_seq != nullptr) {
    cudaFree(sig_seq);
    sig_seq = nullptr;
  }
}

namespace {

// Tunables that mirror the existing PortChannel path:
//   - 16 channels (== num_port_channels_per_rank in Buffer::sync)
//   - port=1, gid_index=3 (matches the value used in Stage 0..4a probes and
//     the typical mscclpp/NDv5 default)
constexpr int kIbgdaPort = 1;
constexpr int kIbgdaGid = 3;
// SQ depth per QP. Each LL dispatch+combine iteration posts up to a few
// dozen WRs per QP at default LL configs; the bench loop runs 50 iters
// without explicit per-iter drain, so we size the SQ at 8192 to avoid
// wrap-around overwriting in-flight WQEs (we have no device-side
// back-pressure check in reserve_wqe_slots — the CQ poller drains
// asynchronously). 8k entries × 64B stride = 512 KiB SQ buf per QP, well
// under typical mlx5 HCA per-QP caps.
constexpr int kIbgdaMaxSendWr = 8192;

}  // namespace

std::unique_ptr<IbgdaSetup> build_ibgda_setup(int rank, int num_ranks, int ib_transport_index, int num_channels,
                                              void* rdma_buffer_ptr, std::size_t num_rdma_bytes,
                                              std::shared_ptr<TcpBootstrap> bootstrap) {
  EP_HOST_ASSERT(rdma_buffer_ptr != nullptr);
  EP_HOST_ASSERT(num_rdma_bytes > 0);
  EP_HOST_ASSERT(num_ranks > 1);
  EP_HOST_ASSERT(num_channels > 0);
  EP_HOST_ASSERT(ib_transport_index >= 0 && ib_transport_index < 8);

  auto setup = std::make_unique<IbgdaSetup>();
  setup->rank = rank;
  setup->num_ranks = num_ranks;
  setup->num_channels = num_channels;

  // 1. Resolve IB device name and build the IbCtx.
  //    `MSCCLPP_EP_IB_DEVICE_OVERRIDE` may force a specific IB transport
  //    index (0..7) for diagnostic NIC-affinity sweeps. Default = use the
  //    NUMA-affine NIC selected by the caller (== local rank on NDv5).
  int effective_ib_index = ib_transport_index;
  if (const char* e = std::getenv("MSCCLPP_EP_IB_DEVICE_OVERRIDE")) {
    int v = std::atoi(e);
    if (v >= 0 && v < 8) effective_ib_index = v;
  }
  auto ib_transport = static_cast<Transport>(static_cast<int>(Transport::IB0) + effective_ib_index);
  std::string dev_name = getIBDeviceName(ib_transport);
  setup->ib_ctx = std::make_unique<IbCtx>(dev_name);
  EP_HOST_ASSERT(setup->ib_ctx->isMlx5() && "IBGDA requires an mlx5 NIC");
  fprintf(stderr, "[mscclpp_ep] rank %d -> IB device %s (transport_index=%d, override=%s)\n",
          rank, dev_name.c_str(), effective_ib_index,
          effective_ib_index == ib_transport_index ? "no" : "yes");
  fflush(stderr);

  // 2. Create QPs. Layout: qps[channel * num_ranks + peer].
  // Self entries are nullptr.
  const int total_slots = num_channels * num_ranks;
  setup->qps.resize(total_slots);
  setup->resources.resize(total_slots);

  for (int c = 0; c < num_channels; ++c) {
    for (int r = 0; r < num_ranks; ++r) {
      if (r == rank) continue;
      auto qp = setup->ib_ctx->createQp(/*port=*/kIbgdaPort, /*gidIndex=*/kIbgdaGid,
                                        /*maxSendCqSize=*/kIbgdaMaxSendWr * 2,
                                        /*maxSendCqPollNum=*/64,
                                        /*maxSendWr=*/kIbgdaMaxSendWr,
                                        /*maxRecvWr=*/1,
                                        /*maxWrPerSend=*/1,
                                        /*noAtomic=*/true);
      setup->qps[c * num_ranks + r] = qp;
    }
  }

  // 3. AllGather IbQpInfos so every rank can RTR every QP.
  // Layout per-rank: total_slots IbQpInfo records, in [c * num_ranks + peer]
  // order. Self entries (peer == rank) are zeroed; remote entries describe
  // the QP that THIS rank uses to TALK TO peer == r.
  std::vector<IbQpInfo> my_infos(total_slots);
  std::memset(my_infos.data(), 0, total_slots * sizeof(IbQpInfo));
  for (int c = 0; c < num_channels; ++c) {
    for (int r = 0; r < num_ranks; ++r) {
      if (r == rank) continue;
      my_infos[c * num_ranks + r] = setup->qps[c * num_ranks + r]->getInfo();
    }
  }
  // bootstrap->allGather expects each rank to fill its own slot in a
  // contiguous buffer of size num_ranks * record_bytes.
  const std::size_t record_bytes = total_slots * sizeof(IbQpInfo);
  std::vector<IbQpInfo> all_infos(num_ranks * total_slots);
  std::memcpy(&all_infos[rank * total_slots], my_infos.data(), record_bytes);
  bootstrap->allGather(all_infos.data(), record_bytes);

  // 4. RTR/RTS each local QP using the matching peer-side QP info. The peer
  // info we want is "rank r's QP for talking to us" == r's record indexed at
  // [c * num_ranks + rank].
  for (int c = 0; c < num_channels; ++c) {
    for (int r = 0; r < num_ranks; ++r) {
      if (r == rank) continue;
      const IbQpInfo& peer_info = all_infos[r * total_slots + c * num_ranks + rank];
      auto& qp = setup->qps[c * num_ranks + r];
      qp->rtr(peer_info);
      qp->rts();
    }
  }

  // 5. Wrap each QP with IbgdaResources (Stage 1) — produces GPU-mapped
  // sq_buf / dbrec / bf_reg / state pointers usable from the kernel.
  for (int c = 0; c < num_channels; ++c) {
    for (int r = 0; r < num_ranks; ++r) {
      if (r == rank) continue;
      ibv_qp* raw = setup->qps[c * num_ranks + r]->getRawQp();
      setup->resources[c * num_ranks + r] = std::make_unique<IbgdaResources>(raw);
    }
  }

  // 6. Register the existing rdma_buffer_ptr as an MR on this IbCtx, then
  // allgather (addr, rkey) so we can build per-peer remote_mrs entries.
  setup->rdma_mr = setup->ib_ctx->registerMr(rdma_buffer_ptr, num_rdma_bytes);
  IbgdaSetup::PeerMr my_rdma_mr{};
  {
    auto info = setup->rdma_mr->getInfo();
    my_rdma_mr.addr = info.addr;
    my_rdma_mr.rkey = info.rkey;
  }
  setup->peer_rdma.assign(num_ranks, IbgdaSetup::PeerMr{});
  setup->peer_rdma[rank] = my_rdma_mr;
  bootstrap->allGather(setup->peer_rdma.data(), sizeof(IbgdaSetup::PeerMr));

  // 7. Allocate signal slots: total_slots * 4 bytes on GPU. Layout:
  // sig_slots[c * num_ranks + sender_peer] — when peer P sends signal() to
  // us through channel c, it RDMA-WRITEs to *our* sig_slots[c * num_ranks + P].
  // Each peer therefore needs to know our sig MR (addr+rkey) AND the offset
  // within it that corresponds to (channel, sender=P) == "we are receiving
  // from P". We allgather the base addr+rkey only; the offset is derivable.
  const std::size_t sig_bytes = std::size_t(total_slots) * sizeof(uint32_t);
  CUDA_CHECK(cudaMalloc(&setup->sig_slots, sig_bytes));
  CUDA_CHECK(cudaMemset(setup->sig_slots, 0, sig_bytes));
  CUDA_CHECK(cudaMalloc(&setup->sig_seq, sig_bytes));
  CUDA_CHECK(cudaMemset(setup->sig_seq, 0, sig_bytes));

  // Use raw verbs for the signal MR (we need the rkey/lkey directly).
  ibv_pd* pd = setup->qps[(rank == 0 ? 1 : 0)]->getRawQp()->pd;  // any non-null QP shares the same pd
  for (int c = 0; c < num_channels && pd == nullptr; ++c)
    for (int r = 0; r < num_ranks && pd == nullptr; ++r)
      if (auto& q = setup->qps[c * num_ranks + r]; q) pd = q->getRawQp()->pd;
  EP_HOST_ASSERT(pd != nullptr);
  setup->sig_mr = ibv_reg_mr(pd, setup->sig_slots, sig_bytes,
                             IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  if (!setup->sig_mr) {
    throw std::runtime_error("ibv_reg_mr(sig_slots) failed errno=" + std::to_string(errno));
  }

  IbgdaSetup::PeerMr my_sig_mr{};
  my_sig_mr.addr = reinterpret_cast<uint64_t>(setup->sig_slots);
  my_sig_mr.rkey = setup->sig_mr->rkey;
  setup->peer_sig.assign(num_ranks, IbgdaSetup::PeerMr{});
  setup->peer_sig[rank] = my_sig_mr;
  bootstrap->allGather(setup->peer_sig.data(), sizeof(IbgdaSetup::PeerMr));

  // 8. Build the GPU-resident MR tables. We have a single local MR (the
  // rdma_buffer_ptr); remote_mrs has one entry per peer rank.
  std::vector<IbgdaLocalMr> h_local(1);
  h_local[0].addr = reinterpret_cast<uint64_t>(rdma_buffer_ptr);
  h_local[0].lkey_be = htonl(setup->rdma_mr->getLkey());
  h_local[0].pad = 0;

  std::vector<IbgdaRemoteMr> h_remote(num_ranks);
  for (int r = 0; r < num_ranks; ++r) {
    h_remote[r].addr = setup->peer_rdma[r].addr;
    h_remote[r].rkey_be = htonl(setup->peer_rdma[r].rkey);
    h_remote[r].pad = 0;
  }

  setup->d_local_mrs = mscclpp::detail::gpuCallocShared<IbgdaLocalMr>(1);
  setup->d_remote_mrs = mscclpp::detail::gpuCallocShared<IbgdaRemoteMr>(num_ranks);
  mscclpp::gpuMemcpy<IbgdaLocalMr>(setup->d_local_mrs.get(), h_local.data(), 1, cudaMemcpyHostToDevice);
  mscclpp::gpuMemcpy<IbgdaRemoteMr>(setup->d_remote_mrs.get(), h_remote.data(), num_ranks, cudaMemcpyHostToDevice);

  // 9. Build the device handle array (channel × num_ranks).
  std::vector<IbgdaPortChannelDeviceHandle> h_handles(total_slots);
  std::memset(h_handles.data(), 0, h_handles.size() * sizeof(IbgdaPortChannelDeviceHandle));
  for (int c = 0; c < num_channels; ++c) {
    for (int r = 0; r < num_ranks; ++r) {
      auto& h = h_handles[c * num_ranks + r];
      if (r == rank) continue;  // self-slot left zeroed
      h.qp = setup->resources[c * num_ranks + r]->getHandle();
      h.local_mrs = setup->d_local_mrs.get();
      h.remote_mrs = setup->d_remote_mrs.get();

      // Local sig slot WE poll for messages from peer r through channel c.
      h.sig_local_addr = &setup->sig_slots[c * num_ranks + r];
      h.sig_local_lkey = htonl(setup->sig_mr->lkey);

      // Remote sig slot peer r polls for messages FROM US through channel c.
      // Peer r's slot for "rank == us" is at offset (c * num_ranks + rank) * 4
      // inside their signal buffer.
      h.sig_remote_addr = setup->peer_sig[r].addr +
                          static_cast<uint64_t>(c * num_ranks + rank) * sizeof(uint32_t);
      h.sig_rkey_be = htonl(setup->peer_sig[r].rkey);

      // Outbound seq counter is per-handle; carve out one u32 from sig_seq
      // (sig_seq is num_channels × num_ranks u32s so we can reuse the same
      // index function — it is unrelated to the inbound sig_slots buffer
      // beyond reusing the size).
      h.sig_seq = &setup->sig_seq[c * num_ranks + r];

      h.dst = static_cast<uint32_t>(r);   // index into remote_mrs[] (peer-rank-based)
      h.src = 0;                          // single-entry local_mrs table
      h.peer_rank = static_cast<uint32_t>(r);
      h._pad = 0;
    }
  }

  setup->device_handles = mscclpp::detail::gpuCallocShared<IbgdaPortChannelDeviceHandle>(total_slots);
  mscclpp::gpuMemcpy<IbgdaPortChannelDeviceHandle>(setup->device_handles.get(), h_handles.data(), total_slots,
                                                   cudaMemcpyHostToDevice);

  // 10. Spawn CQ drain thread. Kernel-issued rdma_write paths set CQ_UPDATE
  // on every WR; without this drain, the send CQ (sized 2 × kIbgdaMaxSendWr)
  // would fill within a few iterations of LL dispatch+combine and the QP
  // would error out. We collect raw send_cq pointers from each QP up front.
  {
    std::vector<ibv_cq*> send_cqs;
    send_cqs.reserve(total_slots);
    for (int idx = 0; idx < total_slots; ++idx) {
      auto& qp = setup->qps[idx];
      if (!qp) continue;
      ibv_cq* cq = qp->getRawQp()->send_cq;
      if (cq) send_cqs.push_back(cq);
    }
    IbgdaSetup* raw = setup.get();
    int dbg_rank = rank;
    setup->cq_poller_thread = std::thread([raw, send_cqs, dbg_rank]() {
      // Tight loop polling all CQs round-robin. Each ibv_poll_cq is cheap
      // (one PCIe-mapped read of CQE buffer + valid bit). We don't inspect
      // wc fields beyond their existence — a status != IBV_WC_SUCCESS would
      // indicate a fatal QP error which we surface lazily through later
      // failures (the test path will hang and the user sees the error).
      constexpr int kBatch = 16;
      ibv_wc wc[kBatch];
      uint64_t total_polled = 0;
      uint64_t total_errors = 0;
      auto t_last_log = std::chrono::steady_clock::now();
      while (!raw->cq_poller_stop.load(std::memory_order_acquire)) {
        bool any = false;
        for (ibv_cq* cq : send_cqs) {
          int n = ibv_poll_cq(cq, kBatch, wc);
          if (n > 0) {
            any = true;
            total_polled += n;
            for (int i = 0; i < n; ++i) {
              if (wc[i].status != IBV_WC_SUCCESS) {
                total_errors++;
                fprintf(stderr,
                        "[mscclpp_ep][ibgda][rank=%d] CQE error: status=%d (%s) opcode=%d vendor_err=0x%x wr_id=%llu\n",
                        dbg_rank, wc[i].status, ibv_wc_status_str(wc[i].status), wc[i].opcode, wc[i].vendor_err,
                        static_cast<unsigned long long>(wc[i].wr_id));
                fflush(stderr);
              }
            }
          }
          for (int rep = 0; rep < 3 && n == kBatch; ++rep) {
            n = ibv_poll_cq(cq, kBatch, wc);
            if (n > 0) {
              any = true;
              total_polled += n;
              for (int i = 0; i < n; ++i) {
                if (wc[i].status != IBV_WC_SUCCESS) {
                  total_errors++;
                  fprintf(stderr,
                          "[mscclpp_ep][ibgda][rank=%d] CQE error: status=%d (%s) opcode=%d vendor_err=0x%x\n",
                          dbg_rank, wc[i].status, ibv_wc_status_str(wc[i].status), wc[i].opcode, wc[i].vendor_err);
                  fflush(stderr);
                }
              }
            }
          }
        }
        auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - t_last_log).count() >= 5) {
          fprintf(stderr, "[mscclpp_ep][ibgda][rank=%d] poller: polled=%llu errors=%llu\n", dbg_rank,
                  static_cast<unsigned long long>(total_polled), static_cast<unsigned long long>(total_errors));
          fflush(stderr);
          t_last_log = now;
        }
        if (!any) {
          std::this_thread::sleep_for(std::chrono::microseconds(20));
        }
      }
    });
  }

  return setup;
}

}  // namespace ep
}  // namespace mscclpp

#endif  // USE_IBVERBS && MSCCLPP_USE_MLX5DV && !MSCCLPP_USE_ROCM
