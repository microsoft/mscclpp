// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "host/gpu_net_io_service.hpp"

#include <cuda_runtime.h>
#include <endian.h>
#include <infiniband/verbs.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include <mscclpp/errors.hpp>

#include "ib.hpp"  // mscclpp core IbCtx / IbMr (ibverbs context + pd + MR)

// Vendored DOCA GPUNetIO host API.
extern "C" {
#include "host/doca_gpunetio.h"
#include "host/doca_gpunetio_high_level.h"
#include "host/doca_verbs.h"
}

namespace mscclpp {

namespace {

#define MSCCLPP_DOCA_THROW(expr)                                                                  \
  do {                                                                                            \
    doca_error_t _st = (expr);                                                                    \
    if (_st != DOCA_SUCCESS) {                                                                    \
      throw mscclpp::Error(std::string("DOCA GPUNetIO call failed: ") + #expr + " status=" +      \
                               std::to_string(static_cast<int>(_st)),                             \
                           mscclpp::ErrorCode::SystemError);                                       \
    }                                                                                             \
  } while (0)

#define MSCCLPP_CUDA_THROW(expr)                                                                          \
  do {                                                                                                    \
    cudaError_t _e = (expr);                                                                              \
    if (_e != cudaSuccess) {                                                                              \
      throw mscclpp::Error(std::string("CUDA call failed: ") + #expr + " : " + cudaGetErrorString(_e),    \
                           mscclpp::ErrorCode::SystemError);                                              \
    }                                                                                                     \
  } while (0)

// Per-peer connection info exchanged via the bootstrap all-gather.
struct QpExchangeInfo {
  uint32_t qpn;
  uint16_t lid;
  uint16_t gidIndex;
  uint8_t gid[16];
  uint8_t linkLayer;
  uint8_t grhRequired;
  uint8_t activeMtu;
  uint8_t pad;
};

// Per-rank memory info exchanged via the bootstrap all-gather.
struct MemExchangeInfo {
  uint64_t base;  // symmetric-buffer base address on that rank
  uint32_t rkey;  // remote key for that rank's registered MR
  uint32_t pad;
};

}  // namespace

struct GpuNetIoService::Impl {
  std::shared_ptr<Bootstrap> bootstrap;
  std::string ibDeviceName;
  int cudaDeviceId;
  int rank = -1;
  int worldSize = 0;
  bool didSetup = false;

  std::unique_ptr<IbCtx> ibCtx;
  std::unique_ptr<const IbMr> mr;  // registration of the symmetric buffer

  struct doca_gpu* gpuDev = nullptr;
  int portNum = 1;
  int gidIndex = 0;
  int numQpsPerPeer = 1;  // MSCCLPP_EP_GPUNETIO_QPS_PER_PEER (>= 1)

  // One high-level QP per remote rank (self entry is null).
  std::vector<struct doca_gpu_verbs_qp_hl*> qpHl;
  doca_gpu_verbs_service_t cpuProxyService = nullptr;
  struct doca_gpu_dev_verbs_qp* qpFlatGpu = nullptr;  // GPU array from flat_list

  // Device-side arrays referenced by GpuNetIoDeviceContext.
  uint32_t* rkeysGpu = nullptr;
  uintptr_t* peerBaseGpu = nullptr;
  GpuNetIoDeviceContext* ctxGpu = nullptr;

  ~Impl() {
    if (cpuProxyService) (void)doca_gpu_verbs_destroy_service(cpuProxyService);
    if (ctxGpu) (void)cudaFree(ctxGpu);
    if (rkeysGpu) (void)cudaFree(rkeysGpu);
    if (peerBaseGpu) (void)cudaFree(peerBaseGpu);
    if (qpFlatGpu) (void)doca_gpu_verbs_qp_flat_list_destroy_hl(qpFlatGpu);
    for (auto* q : qpHl) {
      if (q) (void)doca_gpu_verbs_destroy_qp_hl(q);
    }
    if (gpuDev) (void)doca_gpu_destroy(gpuDev);
  }

  void queryLocalPort(QpExchangeInfo& info) {
    struct ibv_port_attr portAttr;
    std::memset(&portAttr, 0, sizeof(portAttr));
    if (ibv_query_port(ibCtx->getContext(), portNum, &portAttr) != 0) {
      throw Error("ibv_query_port failed for GPUNetIO service", ErrorCode::SystemError);
    }
    info.lid = portAttr.lid;
    info.linkLayer = portAttr.link_layer;
    info.grhRequired = (portAttr.flags & IBV_QPF_GRH_REQUIRED) ? 1 : 0;
    info.activeMtu = portAttr.active_mtu;
  }

  void queryLocalGid(uint8_t outGid[16]) {
    union ibv_gid gid;
    std::memset(&gid, 0, sizeof(gid));
    if (ibv_query_gid(ibCtx->getContext(), portNum, gidIndex, &gid) != 0) {
      throw Error("ibv_query_gid failed for GPUNetIO service", ErrorCode::SystemError);
    }
    std::memcpy(outGid, gid.raw, 16);
  }

  doca_verbs_mtu_size pathMtu(const QpExchangeInfo& remote) const {
    uint8_t mtu = remote.activeMtu;
    if (mtu == 0) {
      mtu = IBV_MTU_1024;
    }
    switch (mtu) {
      case IBV_MTU_256:
        return DOCA_VERBS_MTU_SIZE_256_BYTES;
      case IBV_MTU_512:
        return DOCA_VERBS_MTU_SIZE_512_BYTES;
      case IBV_MTU_1024:
        return DOCA_VERBS_MTU_SIZE_1K_BYTES;
      case IBV_MTU_2048:
        return DOCA_VERBS_MTU_SIZE_2K_BYTES;
      case IBV_MTU_4096:
        return DOCA_VERBS_MTU_SIZE_4K_BYTES;
      default:
        return DOCA_VERBS_MTU_SIZE_1K_BYTES;
    }
  }

  // INIT -> RTR -> RTS for one QP, targeting the given remote info.
  void connectQp(struct doca_gpu_verbs_qp_hl* qp, const QpExchangeInfo& remote) {
    struct doca_verbs_ah_attr* ah = nullptr;
    MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_create(ibCtx->getContext(), &ah));
    struct doca_verbs_gid vgid;
    std::memcpy(vgid.raw, remote.gid, 16);
    MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_set_gid(ah, vgid));
    MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_set_dlid(ah, remote.lid));
    MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_set_sl(ah, 0));
    MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_set_sgid_index(ah, gidIndex));
    MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_set_hop_limit(ah, 255));
    if (remote.linkLayer == IBV_LINK_LAYER_INFINIBAND && !remote.grhRequired) {
      MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_set_addr_type(ah, DOCA_VERBS_ADDR_TYPE_IB_NO_GRH));
    } else if (remote.linkLayer == IBV_LINK_LAYER_INFINIBAND) {
      MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_set_addr_type(ah, DOCA_VERBS_ADDR_TYPE_IB_GRH));
    } else {
      MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_set_addr_type(ah, DOCA_VERBS_ADDR_TYPE_IPv6));
    }

    struct doca_verbs_qp_attr* attr = nullptr;
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_create(&attr));

    // RST -> INIT
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_next_state(attr, DOCA_VERBS_QP_STATE_INIT));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_port_num(attr, portNum));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_allow_remote_write(attr, 1));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_allow_remote_read(attr, 1));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_allow_remote_atomic(attr, DOCA_VERBS_QP_ATOMIC_MODE_IB_SPEC));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_modify(qp->qp, attr,
                                            DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE |
                                                DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ | DOCA_VERBS_QP_ATTR_PKEY_INDEX |
                                                DOCA_VERBS_QP_ATTR_PORT_NUM));

    // INIT -> RTR
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_next_state(attr, DOCA_VERBS_QP_STATE_RTR));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_rq_psn(attr, 0));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_dest_qp_num(attr, remote.qpn));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_path_mtu(attr, pathMtu(remote)));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_ah_attr(attr, ah));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_min_rnr_timer(attr, 12));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_max_dest_rd_atomic(attr, 1));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_modify(qp->qp, attr,
                                            DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_RQ_PSN |
                                                DOCA_VERBS_QP_ATTR_DEST_QP_NUM | DOCA_VERBS_QP_ATTR_PATH_MTU |
                                                DOCA_VERBS_QP_ATTR_AH_ATTR | DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER |
                                                DOCA_VERBS_QP_ATTR_MAX_DEST_RD_ATOMIC));

    // RTR -> RTS
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_next_state(attr, DOCA_VERBS_QP_STATE_RTS));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_sq_psn(attr, 0));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_ack_timeout(attr, 18));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_retry_cnt(attr, 7));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_rnr_retry(attr, 7));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_max_rd_atomic(attr, 1));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_modify(qp->qp, attr,
                                            DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_SQ_PSN |
                                                DOCA_VERBS_QP_ATTR_ACK_TIMEOUT | DOCA_VERBS_QP_ATTR_RETRY_CNT |
                                                DOCA_VERBS_QP_ATTR_RNR_RETRY | DOCA_VERBS_QP_ATTR_MAX_QP_RD_ATOMIC));

    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_destroy(attr));
    MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_destroy(ah));
  }
};

GpuNetIoService::GpuNetIoService(std::shared_ptr<Bootstrap> bootstrap, const std::string& ibDeviceName,
                                 int cudaDeviceId)
    : pimpl_(std::make_unique<Impl>()) {
  pimpl_->bootstrap = bootstrap;
  pimpl_->ibDeviceName = ibDeviceName;
  pimpl_->cudaDeviceId = cudaDeviceId;
  pimpl_->rank = bootstrap->getRank();
  pimpl_->worldSize = bootstrap->getNranks();
}

GpuNetIoService::~GpuNetIoService() = default;

void GpuNetIoService::setup(void* symmetricBuffer, size_t bytes) {
  auto& s = *pimpl_;
  if (s.didSetup) {
    throw Error("GpuNetIoService::setup called more than once", ErrorCode::InvalidUsage);
  }
  s.didSetup = true;

  // 1. ibverbs context + pd (reuse mscclpp's dlopen-based IbCtx), and register
  //    the symmetric buffer (IbMr already handles DMA-BUF / Data Direct on GB200).
  s.ibCtx = std::make_unique<IbCtx>(s.ibDeviceName);
  s.mr = s.ibCtx->registerMr(symmetricBuffer, bytes);

  // 2. DOCA GPU device handle from the CUDA device's PCI bus id.
  char pciBusId[32] = {0};
  MSCCLPP_CUDA_THROW(cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), s.cudaDeviceId));
  MSCCLPP_DOCA_THROW(doca_gpu_create(pciBusId, &s.gpuDev));

  // 3. Create numQpsPerPeer high-level GDAKI QPs per remote rank (skip self).
  //    Spreading a peer's WQEs across multiple send queues lets the NIC drain
  //    them in parallel. K comes from MSCCLPP_EP_GPUNETIO_QPS_PER_PEER (>= 1).
  {
    int k = 1;
    if (const char* env = std::getenv("MSCCLPP_EP_GPUNETIO_QPS_PER_PEER")) {
      k = std::atoi(env);
      if (k < 1) k = 1;
    }
    s.numQpsPerPeer = k;
  }
  const int nQp = s.numQpsPerPeer;
  s.qpHl.assign(static_cast<size_t>(s.worldSize) * nQp, nullptr);
  struct doca_gpu_verbs_qp_init_attr_hl initAttr;
  std::memset(&initAttr, 0, sizeof(initAttr));
  initAttr.gpu_dev = s.gpuDev;
  initAttr.ibpd = s.ibCtx->getPd();
  initAttr.sq_nwqe = 1024;
  initAttr.nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO;
  initAttr.mreg_type = DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT;
  for (int r = 0; r < s.worldSize; ++r) {
    if (r == s.rank) continue;
    for (int k = 0; k < nQp; ++k) {
      MSCCLPP_DOCA_THROW(doca_gpu_verbs_create_qp_hl(&initAttr, &s.qpHl[static_cast<size_t>(r) * nQp + k]));
    }
  }

  // 4. Exchange QP info (all-gather) and connect INIT -> RTR -> RTS.
  const size_t rowLen = static_cast<size_t>(s.worldSize) * nQp;  // QPs this rank publishes
  std::vector<QpExchangeInfo> qpInfo(rowLen);
  std::memset(qpInfo.data(), 0, qpInfo.size() * sizeof(QpExchangeInfo));
  for (int r = 0; r < s.worldSize; ++r) {
    if (r == s.rank) continue;
    for (int k = 0; k < nQp; ++k) {
      QpExchangeInfo& info = qpInfo[static_cast<size_t>(r) * nQp + k];
      info.qpn = doca_verbs_qp_get_qpn(s.qpHl[static_cast<size_t>(r) * nQp + k]->qp);
      info.gidIndex = static_cast<uint16_t>(s.gidIndex);
      s.queryLocalPort(info);
      s.queryLocalGid(info.gid);
    }
  }
  // Each rank publishes, for every peer, its nQp QPs targeting that peer. The
  // all-gather delivers a [worldSize][worldSize*nQp] table; entry [src][dst*nQp+k]
  // is src's k-th QP that talks to dst. We read [r][rank*nQp+k] to pair k<->k.
  std::vector<QpExchangeInfo> qpAll(static_cast<size_t>(s.worldSize) * rowLen);
  std::memcpy(&qpAll[static_cast<size_t>(s.rank) * rowLen], qpInfo.data(), qpInfo.size() * sizeof(QpExchangeInfo));
  s.bootstrap->allGather(qpAll.data(), static_cast<int>(rowLen * sizeof(QpExchangeInfo)));

  for (int r = 0; r < s.worldSize; ++r) {
    if (r == s.rank) continue;
    for (int k = 0; k < nQp; ++k) {
      // Remote peer r's k-th QP that targets this rank.
      const QpExchangeInfo& remote = qpAll[static_cast<size_t>(r) * rowLen + static_cast<size_t>(s.rank) * nQp + k];
      s.connectQp(s.qpHl[static_cast<size_t>(r) * nQp + k], remote);
    }
  }

  bool needsCpuProxy = false;
  for (auto* q : s.qpHl) {
    if (q && q->qp_gverbs && q->qp_gverbs->cpu_proxy) {
      needsCpuProxy = true;
      break;
    }
  }
  if (needsCpuProxy) {
    MSCCLPP_DOCA_THROW(doca_gpu_verbs_create_service(&s.cpuProxyService));
    for (auto* q : s.qpHl) {
      if (q && q->qp_gverbs && q->qp_gverbs->cpu_proxy) {
        MSCCLPP_DOCA_THROW(doca_gpu_verbs_service_monitor_qp(s.cpuProxyService, q->qp_gverbs));
      }
    }
  }

  // 5. Flatten the per-peer device QPs into a GPU array (peer-major: peer*nQp+k).
  MSCCLPP_DOCA_THROW(doca_gpu_verbs_qp_flat_list_create_hl(
      s.qpHl.data(), static_cast<uint32_t>(s.qpHl.size()), &s.qpFlatGpu));

  // 6. Exchange rkeys + symmetric base addresses.
  std::vector<MemExchangeInfo> memAll(s.worldSize);
  std::memset(memAll.data(), 0, memAll.size() * sizeof(MemExchangeInfo));
  memAll[s.rank].base = reinterpret_cast<uint64_t>(symmetricBuffer);
  memAll[s.rank].rkey = s.mr->getInfo().rkey;
  s.bootstrap->allGather(memAll.data(), static_cast<int>(sizeof(MemExchangeInfo)));

  std::vector<uint32_t> rkeysHost(s.worldSize);
  std::vector<uintptr_t> baseHost(s.worldSize);
  for (int r = 0; r < s.worldSize; ++r) {
    rkeysHost[r] = htobe32(memAll[r].rkey);
    baseHost[r] = static_cast<uintptr_t>(memAll[r].base);
  }

  // 7. Publish device-side arrays + context.
  MSCCLPP_CUDA_THROW(cudaMalloc(&s.rkeysGpu, sizeof(uint32_t) * s.worldSize));
  MSCCLPP_CUDA_THROW(cudaMalloc(&s.peerBaseGpu, sizeof(uintptr_t) * s.worldSize));
  MSCCLPP_CUDA_THROW(
      cudaMemcpy(s.rkeysGpu, rkeysHost.data(), sizeof(uint32_t) * s.worldSize, cudaMemcpyHostToDevice));
  MSCCLPP_CUDA_THROW(
      cudaMemcpy(s.peerBaseGpu, baseHost.data(), sizeof(uintptr_t) * s.worldSize, cudaMemcpyHostToDevice));

  GpuNetIoDeviceContext ctxHost;
  ctxHost.qps = s.qpFlatGpu;
  ctxHost.rkeys = s.rkeysGpu;
  ctxHost.peerBase = s.peerBaseGpu;
  ctxHost.lkey = s.mr->getLkey();
  ctxHost.localBase = reinterpret_cast<uintptr_t>(symmetricBuffer);
  ctxHost.numPeers = s.worldSize;
  ctxHost.numQpsPerPeer = s.numQpsPerPeer;
  MSCCLPP_CUDA_THROW(cudaMalloc(&s.ctxGpu, sizeof(GpuNetIoDeviceContext)));
  MSCCLPP_CUDA_THROW(cudaMemcpy(s.ctxGpu, &ctxHost, sizeof(GpuNetIoDeviceContext), cudaMemcpyHostToDevice));
}

GpuNetIoDeviceContext* GpuNetIoService::deviceContext() const { return pimpl_->ctxGpu; }

}  // namespace mscclpp
