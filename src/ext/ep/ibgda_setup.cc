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
  //   - resources / qps / rdma_mrs: smart ptrs, auto-freed.
  //   - sig_mrs (raw ibv_mr) + sig_slots / sig_seq (raw cudaMalloc): explicit.
  for (ibv_mr* m : sig_mrs) {
    if (m) ibv_dereg_mr(m);
  }
  sig_mrs.clear();
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
                                              std::shared_ptr<TcpBootstrap> bootstrap, int num_nics) {
  EP_HOST_ASSERT(rdma_buffer_ptr != nullptr);
  EP_HOST_ASSERT(num_rdma_bytes > 0);
  EP_HOST_ASSERT(num_ranks > 1);
  EP_HOST_ASSERT(num_channels > 0);
  EP_HOST_ASSERT(ib_transport_index >= 0 && ib_transport_index < 8);
  EP_HOST_ASSERT(num_nics >= 1 && num_nics <= 8);

  auto setup = std::make_unique<IbgdaSetup>();
  setup->rank = rank;
  setup->num_ranks = num_ranks;
  setup->num_channels = num_channels;
  setup->num_nics = num_nics;

  // 1. Resolve IB device(s) and build IbCtx per NIC.
  //    `MSCCLPP_EP_IB_DEVICE_OVERRIDE` may force a specific IB transport
  //    index (0..7) for diagnostic NIC-affinity sweeps. Default = use the
  //    NUMA-affine NIC selected by the caller (== local rank on NDv5).
  //    With num_nics > 1, the additional NICs are picked starting from the
  //    base index and wrapping mod 8: nic[i] = (base + i) % 8.
  int effective_base = ib_transport_index;
  if (const char* e = std::getenv("MSCCLPP_EP_IB_DEVICE_OVERRIDE")) {
    if (e[0] != '\0') {
      int v = std::atoi(e);
      if (v >= 0 && v < 8) effective_base = v;
    }
  }
  setup->ib_ctxs.resize(num_nics);
  std::vector<int> nic_idx(num_nics);
  // NUMA-aware striping: on NDv5, NICs 0-3 belong to NUMA 0 and 4-7 to NUMA 1.
  // Crossing NUMA for IBGDA doorbells/PCIe DMA is ~3× slower (verified
  // empirically). Constrain each rank's NIC stripe to its own 4-NIC NUMA
  // group: nic[i] = numa_base + (effective_base + i) % 4.
  // DEBUG: MSCCLPP_EP_NIC_DUP=1 forces all stripe slots to the SAME NIC
  // (same as effective_base) — used to isolate multi-IbCtx overhead from
  // actual multi-NIC routing cost.
  const int numa_base = (effective_base / 4) * 4;
  const int local_off = effective_base % 4;
  bool nic_dup = false;
  if (const char* e = std::getenv("MSCCLPP_EP_NIC_DUP"); e && e[0] != '\0' && std::atoi(e) > 0) {
    nic_dup = true;
  }
  for (int n = 0; n < num_nics; ++n) {
    nic_idx[n] = nic_dup ? effective_base : (numa_base + (local_off + n) % 4);
    auto ib_transport = static_cast<Transport>(static_cast<int>(Transport::IB0) + nic_idx[n]);
    std::string dev_name = getIBDeviceName(ib_transport);
    setup->ib_ctxs[n] = std::make_unique<IbCtx>(dev_name);
    EP_HOST_ASSERT(setup->ib_ctxs[n]->isMlx5() && "IBGDA requires an mlx5 NIC");
    fprintf(stderr, "[mscclpp_ep] rank %d -> IB device[%d/%d] %s (transport_index=%d, numa_base=%d, dup=%d)\n",
            rank, n, num_nics, dev_name.c_str(), nic_idx[n], numa_base, (int)nic_dup);
  }
  fflush(stderr);

  auto nic_for_channel = [num_nics](int c) { return c % num_nics; };

  // 2. Create QPs. Layout: qps[channel * num_ranks + peer]. Each QP lives on
  // ib_ctxs[nic_for_channel(channel)]. Self entries are nullptr.
  const int total_slots = num_channels * num_ranks;
  setup->qps.resize(total_slots);
  setup->resources.resize(total_slots);

  for (int c = 0; c < num_channels; ++c) {
    auto& ctx = setup->ib_ctxs[nic_for_channel(c)];
    for (int r = 0; r < num_ranks; ++r) {
      if (r == rank) continue;
      auto qp = ctx->createQp(/*port=*/kIbgdaPort, /*gidIndex=*/kIbgdaGid,
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
  // [c * num_ranks + rank]. Both sides agree on the channel index and
  // therefore on which NIC pair carries the connection.
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

  // 6. Register the rdma buffer as an MR on EACH NIC, then allgather
  //    (addr, rkey[N]) so we can build per-(nic, peer) remote_mrs entries.
  setup->rdma_mrs.resize(num_nics);
  std::vector<uint32_t> my_rdma_rkeys(num_nics);
  uint64_t my_rdma_addr = 0;
  for (int n = 0; n < num_nics; ++n) {
    setup->rdma_mrs[n] = setup->ib_ctxs[n]->registerMr(rdma_buffer_ptr, num_rdma_bytes);
    auto info = setup->rdma_mrs[n]->getInfo();
    my_rdma_addr = info.addr;  // same across NICs (single allocation)
    my_rdma_rkeys[n] = info.rkey;
  }
  // Allgather record per rank: addr (8B) + N rkeys (4B each), packed.
  const std::size_t rdma_rec_bytes = sizeof(uint64_t) + num_nics * sizeof(uint32_t);
  std::vector<uint8_t> rdma_all(num_ranks * rdma_rec_bytes, 0);
  {
    uint8_t* p = rdma_all.data() + rank * rdma_rec_bytes;
    std::memcpy(p, &my_rdma_addr, sizeof(uint64_t));
    std::memcpy(p + sizeof(uint64_t), my_rdma_rkeys.data(), num_nics * sizeof(uint32_t));
  }
  bootstrap->allGather(rdma_all.data(), rdma_rec_bytes);
  setup->peer_rdma.assign(num_nics * num_ranks, IbgdaSetup::PeerMr{});
  for (int r = 0; r < num_ranks; ++r) {
    const uint8_t* p = rdma_all.data() + r * rdma_rec_bytes;
    uint64_t addr;
    std::memcpy(&addr, p, sizeof(uint64_t));
    for (int n = 0; n < num_nics; ++n) {
      uint32_t rk;
      std::memcpy(&rk, p + sizeof(uint64_t) + n * sizeof(uint32_t), sizeof(uint32_t));
      auto& pr = setup->peer_rdma[n * num_ranks + r];
      pr.addr = addr;
      pr.rkey = rk;
    }
  }

  // 7. Allocate signal slots and register the same buffer as MR on each NIC.
  // Layout: sig_slots[c * num_ranks + sender_peer] — when peer P sends signal()
  // to us through channel c, it RDMA-WRITEs to *our* sig_slots[c * num_ranks
  // + P]. The base address is identical across NICs; rkey differs.
  const std::size_t sig_bytes = std::size_t(total_slots) * sizeof(uint32_t);
  CUDA_CHECK(cudaMalloc(&setup->sig_slots, sig_bytes));
  CUDA_CHECK(cudaMemset(setup->sig_slots, 0, sig_bytes));
  CUDA_CHECK(cudaMalloc(&setup->sig_seq, sig_bytes));
  CUDA_CHECK(cudaMemset(setup->sig_seq, 0, sig_bytes));

  setup->sig_mrs.assign(num_nics, nullptr);
  std::vector<uint32_t> my_sig_lkeys(num_nics);
  std::vector<uint32_t> my_sig_rkeys(num_nics);
  for (int n = 0; n < num_nics; ++n) {
    // Pick any QP on this NIC to grab its PD.
    ibv_pd* pd = nullptr;
    for (int c = 0; c < num_channels && pd == nullptr; ++c) {
      if (nic_for_channel(c) != n) continue;
      for (int r = 0; r < num_ranks && pd == nullptr; ++r) {
        if (auto& q = setup->qps[c * num_ranks + r]; q) pd = q->getRawQp()->pd;
      }
    }
    EP_HOST_ASSERT(pd != nullptr);
    ibv_mr* m = ibv_reg_mr(pd, setup->sig_slots, sig_bytes,
                           IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!m) {
      throw std::runtime_error("ibv_reg_mr(sig_slots) failed errno=" + std::to_string(errno));
    }
    setup->sig_mrs[n] = m;
    my_sig_lkeys[n] = m->lkey;
    my_sig_rkeys[n] = m->rkey;
  }

  uint64_t my_sig_addr = reinterpret_cast<uint64_t>(setup->sig_slots);
  const std::size_t sig_rec_bytes = sizeof(uint64_t) + num_nics * sizeof(uint32_t);
  std::vector<uint8_t> sig_all(num_ranks * sig_rec_bytes, 0);
  {
    uint8_t* p = sig_all.data() + rank * sig_rec_bytes;
    std::memcpy(p, &my_sig_addr, sizeof(uint64_t));
    std::memcpy(p + sizeof(uint64_t), my_sig_rkeys.data(), num_nics * sizeof(uint32_t));
  }
  bootstrap->allGather(sig_all.data(), sig_rec_bytes);
  setup->peer_sig.assign(num_nics * num_ranks, IbgdaSetup::PeerMr{});
  for (int r = 0; r < num_ranks; ++r) {
    const uint8_t* p = sig_all.data() + r * sig_rec_bytes;
    uint64_t addr;
    std::memcpy(&addr, p, sizeof(uint64_t));
    for (int n = 0; n < num_nics; ++n) {
      uint32_t rk;
      std::memcpy(&rk, p + sizeof(uint64_t) + n * sizeof(uint32_t), sizeof(uint32_t));
      auto& ps = setup->peer_sig[n * num_ranks + r];
      ps.addr = addr;
      ps.rkey = rk;
    }
  }

  // 8. Build the GPU-resident MR tables.
  //    d_local_mrs[n]   = (rdma_buffer_ptr, lkey on NIC n)  — one per NIC.
  //    d_remote_mrs[n*num_ranks + r] = (peer r's addr, peer r's rkey on NIC n).
  std::vector<IbgdaLocalMr> h_local(num_nics);
  for (int n = 0; n < num_nics; ++n) {
    h_local[n].addr = reinterpret_cast<uint64_t>(rdma_buffer_ptr);
    h_local[n].lkey_be = htonl(setup->rdma_mrs[n]->getLkey());
    h_local[n].pad = 0;
  }

  std::vector<IbgdaRemoteMr> h_remote(num_nics * num_ranks);
  for (int n = 0; n < num_nics; ++n) {
    for (int r = 0; r < num_ranks; ++r) {
      auto& pr = setup->peer_rdma[n * num_ranks + r];
      auto& hr = h_remote[n * num_ranks + r];
      hr.addr = pr.addr;
      hr.rkey_be = htonl(pr.rkey);
      hr.pad = 0;
    }
  }

  setup->d_local_mrs = mscclpp::detail::gpuCallocShared<IbgdaLocalMr>(num_nics);
  setup->d_remote_mrs = mscclpp::detail::gpuCallocShared<IbgdaRemoteMr>(num_nics * num_ranks);
  mscclpp::gpuMemcpy<IbgdaLocalMr>(setup->d_local_mrs.get(), h_local.data(), num_nics, cudaMemcpyHostToDevice);
  mscclpp::gpuMemcpy<IbgdaRemoteMr>(setup->d_remote_mrs.get(), h_remote.data(),
                                    num_nics * num_ranks, cudaMemcpyHostToDevice);

  // 9. Build the device handle array (channel × num_ranks).
  std::vector<IbgdaPortChannelDeviceHandle> h_handles(total_slots);
  std::memset(h_handles.data(), 0, h_handles.size() * sizeof(IbgdaPortChannelDeviceHandle));
  for (int c = 0; c < num_channels; ++c) {
    const int n = nic_for_channel(c);
    for (int r = 0; r < num_ranks; ++r) {
      auto& h = h_handles[c * num_ranks + r];
      if (r == rank) continue;  // self-slot left zeroed
      h.qp = setup->resources[c * num_ranks + r]->getHandle();
      h.local_mrs = setup->d_local_mrs.get();
      h.remote_mrs = setup->d_remote_mrs.get();

      // Local sig slot WE poll for messages from peer r through channel c.
      h.sig_local_addr = &setup->sig_slots[c * num_ranks + r];
      h.sig_local_lkey = htonl(my_sig_lkeys[n]);

      // Remote sig slot peer r polls for messages FROM US through channel c.
      // Peer r's slot for "rank == us" is at offset (c * num_ranks + rank) * 4
      // inside their signal buffer. The rkey must be the one peer r registered
      // on its own NIC n (matching channel-to-NIC pairing).
      h.sig_remote_addr = setup->peer_sig[n * num_ranks + r].addr +
                          static_cast<uint64_t>(c * num_ranks + rank) * sizeof(uint32_t);
      h.sig_rkey_be = htonl(setup->peer_sig[n * num_ranks + r].rkey);

      // Outbound seq counter is per-handle.
      h.sig_seq = &setup->sig_seq[c * num_ranks + r];

      // src/dst index into the local/remote MR tables. With multi-NIC,
      // src = nic, dst = nic * num_ranks + peer_rank.
      h.src = static_cast<uint32_t>(n);
      h.dst = static_cast<uint32_t>(n * num_ranks + r);
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
  // would error out. We collect raw send_cq pointers from each QP up front
  // (across ALL NICs).
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
