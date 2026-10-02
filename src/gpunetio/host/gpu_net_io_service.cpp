// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cuda_runtime.h>
#include <endian.h>
#include <infiniband/verbs.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <mscclpp/env.hpp>
#include <mscclpp/errors.hpp>
#include <mscclpp/gpu_net_io_service.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/port_channel.hpp>
#include <optional>
#include <stdexcept>
#include <vector>

#include "api.h"
#include "gpu_net_io_binding.hpp"
#include "gpu_net_io_policy.hpp"
#include "gpu_net_io_qp_table.hpp"
#include "ib.hpp"  // mscclpp core IbCtx / IbMr (ibverbs context + pd + MR)

// DOCA GPUNetIO host API.
extern "C" {
#include "host/doca_gpunetio.h"
#include "host/doca_gpunetio_high_level.h"
#include "host/doca_verbs.h"
}

namespace mscclpp {

namespace {

#define MSCCLPP_DOCA_THROW(expr)                                                                                   \
  do {                                                                                                             \
    doca_error_t _st = (expr);                                                                                     \
    if (_st != DOCA_SUCCESS) {                                                                                     \
      throw mscclpp::Error(                                                                                        \
          std::string("DOCA GPUNetIO call failed: ") + #expr + " status=" + std::to_string(static_cast<int>(_st)), \
          mscclpp::ErrorCode::SystemError);                                                                        \
    }                                                                                                              \
  } while (0)

#define MSCCLPP_CUDA_THROW(expr)                                                                       \
  do {                                                                                                 \
    cudaError_t _e = (expr);                                                                           \
    if (_e != cudaSuccess) {                                                                           \
      throw mscclpp::Error(std::string("CUDA call failed: ") + #expr + " : " + cudaGetErrorString(_e), \
                           mscclpp::ErrorCode::SystemError);                                           \
    }                                                                                                  \
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

struct ConfigExchangeInfo {
  uint64_t bytes;
  int32_t tag;
  uint32_t valid;
};

}  // namespace

struct GpuNetIoService::Impl {
  std::shared_ptr<Bootstrap> bootstrap;
  std::string ibDeviceName;
  int cudaDeviceId = -1;
  int rank = -1;
  int worldSize = 0;
  int numQpsPerPeer = 1;
  std::vector<int> peerQpOffsets;
  std::vector<detail::GpuNetIoSetupStatus> setupStatuses;
  bool didSetup = false;
  bool deviceSelected = false;
  bool setupComplete = false;

  std::unique_ptr<IbCtx> ibCtx;
  std::unique_ptr<const IbMr> mr;  // registration of the symmetric buffer
  std::unique_ptr<const IbMr> atomicResultMr;

  doca_gpu_t* gpuDev = nullptr;
  doca_dev_t* netDev = nullptr;
  int portNum = 1;
  int gidIndex = -1;
  QpExchangeInfo localPortInfo{};

  // Compact QPs indexed by peerQpOffsets; peers with no requested queues have no entries.
  std::vector<struct doca_gpu_verbs_qp_hl*> qpHl;
  struct doca_gpu_dev_verbs_qp* qpFlatGpu = nullptr;
  int* peerQpOffsetsGpu = nullptr;

  // Device-side arrays referenced by GpuNetIoDeviceContext.
  uint32_t* rkeysGpu = nullptr;
  uintptr_t* peerBaseGpu = nullptr;
  GpuNetIoDeviceContext* ctxGpu = nullptr;
  uint64_t* atomicResultsGpu = nullptr;

  ~Impl() {
    if (!deviceSelected) return;
    CudaDeviceGuard deviceGuard(cudaDeviceId);
    if (ctxGpu) (void)cudaFree(ctxGpu);
    if (rkeysGpu) (void)cudaFree(rkeysGpu);
    if (peerBaseGpu) (void)cudaFree(peerBaseGpu);
    if (qpFlatGpu) (void)cudaFree(qpFlatGpu);
    if (peerQpOffsetsGpu) (void)cudaFree(peerQpOffsetsGpu);
    for (auto* q : qpHl) {
      if (q) (void)doca_gpu_verbs_destroy_qp_hl(q);
    }
    atomicResultMr.reset();
    if (atomicResultsGpu) (void)cudaFree(atomicResultsGpu);
    if (gpuDev) (void)doca_gpu_destroy(gpuDev);
    if (netDev) (void)doca_verbs_dev_close(netDev);
    mr.reset();
    ibCtx.reset();
  }

  void validateLocalPort() {
    const auto info = detail::queryGpuNetIoPort(
        ibCtx->getContext(), portNum, gidIndex,
        [](ibv_context* context, uint8_t port, ibv_port_attr* attributes) {
          return ibv_query_port(context, port, attributes);
        },
        [](ibv_context* context, uint8_t port, int index, ibv_gid* gid) {
          return ibv_query_gid(context, port, index, gid);
        });
    localPortInfo.gidIndex = static_cast<uint16_t>(gidIndex);
    localPortInfo.lid = info.port.lid;
    localPortInfo.linkLayer = info.port.link_layer;
    localPortInfo.grhRequired = (info.port.flags & IBV_QPF_GRH_REQUIRED) ? 1 : 0;
    localPortInfo.activeMtu = info.port.active_mtu;
    std::memcpy(localPortInfo.gid, info.gid.raw, sizeof(localPortInfo.gid));
  }

  doca_verbs_mtu_size pathMtu(uint8_t localMtu, const QpExchangeInfo& remote) const {
    if (localMtu < IBV_MTU_256 || localMtu > IBV_MTU_4096 || remote.activeMtu < IBV_MTU_256 ||
        remote.activeMtu > IBV_MTU_4096) {
      throw Error("Invalid active MTU for GPUNetIO QP", ErrorCode::InvalidUsage);
    }
    const uint8_t mtu = std::min(localMtu, remote.activeMtu);
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
    const auto mtu = pathMtu(localPortInfo.activeMtu, remote);
    doca_verbs_ah_attr_t* ah = nullptr;
    MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_create(netDev, &ah));
    std::unique_ptr<doca_verbs_ah_attr_t, decltype(&doca_verbs_ah_attr_destroy)> ahGuard(ah,
                                                                                         doca_verbs_ah_attr_destroy);
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

    doca_verbs_qp_attr_t* attr = nullptr;
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_create(&attr));
    std::unique_ptr<doca_verbs_qp_attr_t, decltype(&doca_verbs_qp_attr_destroy)> attrGuard(attr,
                                                                                           doca_verbs_qp_attr_destroy);

    // RST -> INIT
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_next_state(attr, DOCA_VERBS_QP_STATE_INIT));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_port_num(attr, portNum));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_allow_remote_write(attr, 1));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_allow_remote_read(attr, 1));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_atomic_mode(attr, DOCA_VERBS_QP_ATOMIC_MODE_IB_SPEC));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_modify(qp->qp, attr,
                                            DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE |
                                                DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ | DOCA_VERBS_QP_ATTR_ATOMIC_MODE |
                                                DOCA_VERBS_QP_ATTR_PKEY_INDEX | DOCA_VERBS_QP_ATTR_PORT_NUM));

    // INIT -> RTR
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_next_state(attr, DOCA_VERBS_QP_STATE_RTR));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_rq_psn(attr, 0));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_dest_qp_num(attr, remote.qpn));
    MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_path_mtu(attr, mtu));
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
  }
};

MSCCLPP_API_CPP GpuNetIoService::GpuNetIoService(std::shared_ptr<Bootstrap> bootstrap, const std::string& ibDeviceName,
                                                 int cudaDeviceId)
    : GpuNetIoService(bootstrap, ibDeviceName, cudaDeviceId, 1) {}

MSCCLPP_API_CPP GpuNetIoService::GpuNetIoService(std::shared_ptr<Bootstrap> bootstrap, const std::string& ibDeviceName,
                                                 int cudaDeviceId, int numQpsPerPeer)
    : pimpl_(std::make_shared<Impl>()) {
  if (!bootstrap || ibDeviceName.empty() || cudaDeviceId < 0) {
    throw Error("GPUNetIO requires a bootstrap, explicit IB device and CUDA device ordinal", ErrorCode::InvalidUsage);
  }
  pimpl_->bootstrap = bootstrap;
  pimpl_->ibDeviceName = ibDeviceName;
  pimpl_->cudaDeviceId = cudaDeviceId;
  pimpl_->rank = bootstrap->getRank();
  pimpl_->worldSize = bootstrap->getNranks();
  pimpl_->numQpsPerPeer = numQpsPerPeer;
  if (pimpl_->worldSize < 1 || pimpl_->rank < 0 || pimpl_->rank >= pimpl_->worldSize) {
    throw Error("Invalid GPUNetIO bootstrap geometry", ErrorCode::InvalidUsage);
  }
  pimpl_->setupStatuses.resize(pimpl_->worldSize);
}

MSCCLPP_API_CPP GpuNetIoService::~GpuNetIoService() = default;

MSCCLPP_API_CPP void GpuNetIoService::setup() { setup(nullptr, 0); }

MSCCLPP_API_CPP void GpuNetIoService::setup(void* symmetricBuffer, size_t bytes) {
  setupImpl(symmetricBuffer, bytes, {}, 0, true);
}

MSCCLPP_API_CPP void GpuNetIoService::setup(const std::vector<int>& peerQpCounts, int tag) {
  setupImpl(nullptr, 0, peerQpCounts, tag);
}

void GpuNetIoService::setupImpl(void* symmetricBuffer, size_t bytes, const std::vector<int>& peerQpCounts, int tag,
                                bool fullMesh) {
  auto& s = *pimpl_;
  const auto phase = [&](const char* name, auto operation) {
    detail::runGpuNetIoSetupPhase(*s.bootstrap, s.setupStatuses, name, operation);
  };
  std::optional<CudaDeviceGuard> deviceGuard;
  std::vector<int> counts;
  std::vector<ConfigExchangeInfo> configs;
  phase("configuration preparation", [&] {
    if (s.didSetup) throw std::invalid_argument("GpuNetIoService::setup called more than once");
    s.didSetup = true;
    deviceGuard.emplace(s.cudaDeviceId);
    s.deviceSelected = true;
    if (fullMesh) {
      if (s.numQpsPerPeer < 1 || s.numQpsPerPeer > 64) throw std::invalid_argument("Invalid uniform QP count");
      counts.assign(s.worldSize, s.numQpsPerPeer);
      counts[s.rank] = 0;
    } else {
      counts = peerQpCounts;
    }
    if (counts.size() != static_cast<size_t>(s.worldSize) || tag < 0 || (symmetricBuffer == nullptr) != (bytes == 0))
      throw std::invalid_argument("Invalid GPUNetIO setup arguments");
    for (size_t peer = 0; peer < counts.size(); ++peer) {
      if (counts[peer] < 0 || counts[peer] > 64 || (peer == static_cast<size_t>(s.rank) && counts[peer] != 0))
        throw std::invalid_argument("Invalid GPUNetIO per-peer QP count");
    }
    configs.resize(s.worldSize);
    configs[s.rank] = {bytes, tag, 1};
  });
  s.bootstrap->allGather(configs.data(), sizeof(ConfigExchangeInfo));
  const size_t ranks = static_cast<size_t>(s.worldSize);
  std::vector<int> plans;
  phase("connection plan preparation", [&] {
    for (const auto& config : configs) {
      if (!config.valid || config.bytes != bytes || config.tag != tag)
        throw std::invalid_argument("GPUNetIO ranks must agree on optional buffer size and exchange tag");
    }
    if (ranks > static_cast<size_t>(std::numeric_limits<int>::max()) / sizeof(int) ||
        ranks > std::numeric_limits<size_t>::max() / sizeof(int) / ranks)
      throw std::invalid_argument("GPUNetIO connection plan exceeds bootstrap size limit");
    plans.resize(ranks * ranks);
    std::copy(counts.begin(), counts.end(), plans.begin() + static_cast<size_t>(s.rank) * ranks);
  });
  s.bootstrap->allGather(plans.data(), static_cast<int>(ranks * sizeof(int)));
  phase("connection plan validation", [&] {
    s.peerQpOffsets = detail::gpuNetIoQpOffsets(plans, s.rank, s.worldSize);
    s.numQpsPerPeer = *std::max_element(counts.begin(), counts.end());
  });
  const size_t rowLength = static_cast<size_t>(s.peerQpOffsets.back());
  phase("HCA, port/GID and atomic capability", [&] {
    s.gidIndex = env()->ibGidIndex;
    s.ibCtx = std::make_unique<IbCtx>(s.ibDeviceName);
    if (!s.ibCtx->supportsRdmaAtomics()) {
      throw std::invalid_argument("GPUNetIO requires RDMA atomics on HCA " + s.ibDeviceName +
                                  "; no fallback is available");
    }
    try {
      s.validateLocalPort();
    } catch (const std::invalid_argument& error) {
      throw Error("HCA " + s.ibDeviceName + ", port " + std::to_string(s.portNum) +
                      ", MSCCLPP_IB_GID_INDEX=" + std::to_string(s.gidIndex) + ": " + error.what(),
                  ErrorCode::InvalidUsage);
    } catch (const std::exception& error) {
      throw Error("HCA " + s.ibDeviceName + ", port " + std::to_string(s.portNum) +
                      ", MSCCLPP_IB_GID_INDEX=" + std::to_string(s.gidIndex) + ": " + error.what(),
                  ErrorCode::SystemError);
    }
  });
  phase("memory registration and device resources", [&] {
    if (bytes != 0) s.mr = s.ibCtx->registerMr(symmetricBuffer, bytes);
    if (rowLength != 0) {
      MSCCLPP_DOCA_THROW(doca_verbs_dev_open(s.ibCtx->getPd(), &s.netDev));
      MSCCLPP_CUDA_THROW(cudaMalloc(&s.atomicResultsGpu, rowLength * sizeof(uint64_t)));
      MSCCLPP_CUDA_THROW(cudaMemset(s.atomicResultsGpu, 0, rowLength * sizeof(uint64_t)));
      s.atomicResultMr = s.ibCtx->registerMr(s.atomicResultsGpu, rowLength * sizeof(uint64_t));
      char pciBusId[32] = {0};
      MSCCLPP_CUDA_THROW(cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), s.cudaDeviceId));
      MSCCLPP_DOCA_THROW(doca_gpu_create(pciBusId, &s.gpuDev));
    }
  });
  phase("direct GPU QP creation", [&] {
    s.qpHl.assign(rowLength, nullptr);
    for (size_t index = 0; index < rowLength; ++index)
      MSCCLPP_DOCA_THROW(detail::createDirectGpuNetIoQp(s.gpuDev, s.netDev, s.ibCtx->getPd(), &s.qpHl[index]));
  });
  std::vector<QpExchangeInfo> qpInfo;
  std::vector<QpExchangeInfo> remoteQps;
  phase("QP metadata preparation", [&] {
    qpInfo.resize(rowLength);
    remoteQps.resize(rowLength);
    for (size_t index = 0; index < rowLength; ++index) {
      qpInfo[index] = s.localPortInfo;
      MSCCLPP_DOCA_THROW(doca_verbs_qp_get_qpn(s.qpHl[index]->qp, &qpInfo[index].qpn));
    }
  });
  detail::exchangeGpuNetIoQpMetadata(*s.bootstrap, qpInfo, remoteQps, s.peerQpOffsets, s.rank, tag);
  phase("QP transitions", [&] {
    for (size_t index = 0; index < rowLength; ++index) s.connectQp(s.qpHl[index], remoteQps[index]);
  });
  std::vector<MemExchangeInfo> memAll;
  phase("device QP table and memory metadata preparation", [&] {
    const auto qpTable = detail::buildGpuNetIoQpTable(s.qpHl);
    const size_t qpTableBytes = qpTable.size() * sizeof(doca_gpu_dev_verbs_qp);
    if (qpTableBytes != 0) {
      MSCCLPP_CUDA_THROW(cudaMalloc(&s.qpFlatGpu, qpTableBytes));
      MSCCLPP_CUDA_THROW(cudaMemcpy(s.qpFlatGpu, qpTable.data(), qpTableBytes, cudaMemcpyHostToDevice));
    }
    const size_t offsetBytes = s.peerQpOffsets.size() * sizeof(int);
    MSCCLPP_CUDA_THROW(cudaMalloc(&s.peerQpOffsetsGpu, offsetBytes));
    MSCCLPP_CUDA_THROW(cudaMemcpy(s.peerQpOffsetsGpu, s.peerQpOffsets.data(), offsetBytes, cudaMemcpyHostToDevice));
    memAll.resize(s.worldSize);
    memAll[s.rank].base = reinterpret_cast<uint64_t>(symmetricBuffer);
    memAll[s.rank].rkey = s.mr ? s.mr->getInfo().rkey : 0;
  });
  if (bytes != 0) s.bootstrap->allGather(memAll.data(), static_cast<int>(sizeof(MemExchangeInfo)));
  phase("device context publication", [&] {
    std::vector<uint32_t> rkeysHost(s.worldSize);
    std::vector<uintptr_t> baseHost(s.worldSize);
    for (int peer = 0; peer < s.worldSize; ++peer) {
      rkeysHost[peer] = htobe32(memAll[peer].rkey);
      baseHost[peer] = static_cast<uintptr_t>(memAll[peer].base);
    }
    MSCCLPP_CUDA_THROW(cudaMalloc(&s.rkeysGpu, sizeof(uint32_t) * s.worldSize));
    MSCCLPP_CUDA_THROW(cudaMalloc(&s.peerBaseGpu, sizeof(uintptr_t) * s.worldSize));
    MSCCLPP_CUDA_THROW(
        cudaMemcpy(s.rkeysGpu, rkeysHost.data(), sizeof(uint32_t) * s.worldSize, cudaMemcpyHostToDevice));
    MSCCLPP_CUDA_THROW(
        cudaMemcpy(s.peerBaseGpu, baseHost.data(), sizeof(uintptr_t) * s.worldSize, cudaMemcpyHostToDevice));
    GpuNetIoDeviceContext ctxHost{};
    ctxHost.qps = s.qpFlatGpu;
    ctxHost.rkeys = s.rkeysGpu;
    ctxHost.peerBase = s.peerBaseGpu;
    ctxHost.lkey = s.mr ? s.mr->getLkey() : 0;
    ctxHost.localBase = reinterpret_cast<uintptr_t>(symmetricBuffer);
    ctxHost.numPeers = s.worldSize;
    ctxHost.numQpsPerPeer = s.numQpsPerPeer;
    ctxHost.peerQpOffsets = s.peerQpOffsetsGpu;
    ctxHost.atomicResultBase = reinterpret_cast<uintptr_t>(s.atomicResultsGpu);
    ctxHost.atomicResultLkey = s.atomicResultMr ? s.atomicResultMr->getLkey() : 0;
    MSCCLPP_CUDA_THROW(cudaMalloc(&s.ctxGpu, sizeof(GpuNetIoDeviceContext)));
    MSCCLPP_CUDA_THROW(cudaMemcpy(s.ctxGpu, &ctxHost, sizeof(GpuNetIoDeviceContext), cudaMemcpyHostToDevice));
    MSCCLPP_CUDA_THROW(cudaDeviceSynchronize());
  });
  s.setupComplete = true;
}

MSCCLPP_API_CPP GpuNetIoDeviceContext* GpuNetIoService::deviceContext() const {
  return pimpl_->setupComplete ? pimpl_->ctxGpu : nullptr;
}

MSCCLPP_API_CPP GpuNetIoDeviceContext* GpuNetIoService::deviceContext(int peer) const {
  if (peer < 0 || peer >= pimpl_->worldSize || peer == pimpl_->rank) {
    throw Error("GPUNetIO channel requires a remote bootstrap rank within the world size", ErrorCode::InvalidUsage);
  }
  if (!pimpl_->setupComplete) {
    throw Error("GPUNetIO channel requires successful service setup", ErrorCode::InvalidUsage);
  }
  if (pimpl_->peerQpOffsets[peer] == pimpl_->peerQpOffsets[peer + 1]) {
    throw Error("GPUNetIO peer was not requested in the connection plan", ErrorCode::InvalidUsage);
  }
  return pimpl_->ctxGpu;
}

namespace detail {
struct GpuNetIoConnectionState {
  std::shared_ptr<GpuNetIoService::Impl> service;
  int peer;
  int qpIndex;
};

struct GpuNetIoMemoryState {
  std::shared_ptr<GpuNetIoService::Impl> service;
  std::shared_ptr<void> owner;
  std::shared_ptr<GpuNetIoMemoryState> exportedLocal;
  std::unique_ptr<const IbMr> registration;
  GpuNetIoMemoryDeviceHandle descriptor{};
  bool local = false;
  ~GpuNetIoMemoryState() {
    CudaDeviceGuard guard(service->cudaDeviceId);
    registration.reset();
    owner.reset();
  }
};

struct GpuNetIoSemaphoreState {
  std::shared_ptr<GpuNetIoConnectionState> connection;
  GpuNetIoMemory inbound;
  GpuNetIoMemory remote;
  uint64_t* inboundCounter;
  uint64_t* expectedCounter;
  GpuNetIoMemoryDeviceHandle remoteSignal;
};

struct GpuNetIoChannelState {
  std::shared_ptr<GpuNetIoSemaphoreState> semaphore;
  std::vector<GpuNetIoMemory> memories;
  GpuNetIoMemoryDeviceHandle* table = nullptr;
  ~GpuNetIoChannelState() {
    CudaDeviceGuard guard(semaphore->connection->service->cudaDeviceId);
    if (table) (void)cudaFree(table);
  }
};
}  // namespace detail

MSCCLPP_API_CPP GpuNetIoConnection GpuNetIoService::connect(int peer, int qpIndex) const {
  (void)deviceContext(peer);
  if (qpIndex < 0 || qpIndex >= pimpl_->peerQpOffsets[peer + 1] - pimpl_->peerQpOffsets[peer]) {
    throw Error("GPUNetIO connection QP index out of range", ErrorCode::InvalidUsage);
  }
  GpuNetIoConnection connection;
  connection.state_ =
      std::make_shared<detail::GpuNetIoConnectionState>(detail::GpuNetIoConnectionState{pimpl_, peer, qpIndex});
  return connection;
}

MSCCLPP_API_CPP GpuNetIoMemory GpuNetIoService::registerMemory(void* buffer, size_t bytes,
                                                               std::shared_ptr<void> owner) const {
  if (!pimpl_->setupComplete || buffer == nullptr || bytes == 0 ||
      bytes > UINTPTR_MAX - reinterpret_cast<uintptr_t>(buffer)) {
    throw Error("GPUNetIO registration requires setup and a valid nonempty CUDA buffer", ErrorCode::InvalidUsage);
  }
  CudaDeviceGuard guard(pimpl_->cudaDeviceId);
  cudaPointerAttributes attributes{};
  MSCCLPP_CUDA_THROW(cudaPointerGetAttributes(&attributes, buffer));
  if (attributes.type != cudaMemoryTypeDevice || attributes.device != pimpl_->cudaDeviceId) {
    throw Error("GPUNetIO buffer must belong to the service CUDA device", ErrorCode::InvalidUsage);
  }
  GpuNetIoMemory memory;
  memory.state_ = std::make_shared<detail::GpuNetIoMemoryState>();
  memory.state_->service = pimpl_;
  memory.state_->owner = std::move(owner);
  memory.state_->registration = pimpl_->ibCtx->registerMr(buffer, bytes);
  memory.state_->descriptor = {reinterpret_cast<uintptr_t>(buffer), bytes, memory.state_->registration->getLkey(),
                               pimpl_->rank};
  memory.state_->local = true;
  return memory;
}

GpuNetIoMemory GpuNetIoService::exchangeMemoryImpl(const GpuNetIoConnection& connection, const GpuNetIoMemory& local,
                                                   int tag, uint32_t kind) const {
  if (!connection.state_ || connection.state_->service != pimpl_ || !local.state_ || local.state_->service != pimpl_ ||
      !local.state_->local || tag < 0) {
    throw Error("GPUNetIO exchange requires a local registration and connection from this service",
                ErrorCode::InvalidUsage);
  }
  const auto& binding = *connection.state_;
  detail::GpuNetIoMemoryExchange outgoing{local.state_->descriptor.base,
                                          local.state_->descriptor.bytes,
                                          local.state_->registration->getInfo().rkey,
                                          kind,
                                          pimpl_->rank,
                                          binding.peer,
                                          binding.qpIndex,
                                          1};
  const auto incoming = detail::exchangeGpuNetIoMemory(*pimpl_->bootstrap, outgoing, tag);
  GpuNetIoMemory memory;
  memory.state_ = std::make_shared<detail::GpuNetIoMemoryState>();
  memory.state_->service = pimpl_;
  memory.state_->exportedLocal = local.state_;
  memory.state_->descriptor = {incoming.base, incoming.bytes, incoming.rkey, incoming.rank};
  return memory;
}

MSCCLPP_API_CPP GpuNetIoMemory GpuNetIoService::exchangeMemory(const GpuNetIoConnection& connection,
                                                               const GpuNetIoMemory& local, int tag) const {
  return exchangeMemoryImpl(connection, local, tag, 1);
}

MSCCLPP_API_CPP GpuNetIoSemaphore GpuNetIoService::buildSemaphore(const GpuNetIoConnection& connection, int tag) const {
  if (!connection.state_ || connection.state_->service != pimpl_ || tag < 0) {
    throw Error("GPUNetIO semaphore requires a connection from this service and a valid tag", ErrorCode::InvalidUsage);
  }
  CudaDeviceGuard guard(pimpl_->cudaDeviceId);
  uint64_t* counters = nullptr;
  MSCCLPP_CUDA_THROW(cudaMalloc(&counters, 2 * sizeof(uint64_t)));
  std::shared_ptr<void> owner(counters, [device = pimpl_->cudaDeviceId](void* pointer) {
    CudaDeviceGuard deviceGuard(device);
    (void)cudaFree(pointer);
  });
  MSCCLPP_CUDA_THROW(cudaMemset(counters, 0, 2 * sizeof(uint64_t)));
  MSCCLPP_CUDA_THROW(cudaStreamSynchronize(nullptr));
  auto inbound = registerMemory(counters, sizeof(uint64_t), owner);
  auto remote = exchangeMemoryImpl(connection, inbound, tag, 2);
  if (remote.state_->descriptor.bytes != sizeof(uint64_t) || remote.state_->descriptor.base % alignof(uint64_t) != 0) {
    throw Error("Invalid GPUNetIO semaphore counter metadata", ErrorCode::InvalidUsage);
  }
  GpuNetIoSemaphore semaphore;
  semaphore.state_ = std::make_shared<detail::GpuNetIoSemaphoreState>(detail::GpuNetIoSemaphoreState{
      connection.state_, inbound, remote, counters, counters + 1, remote.state_->descriptor});
  return semaphore;
}

MSCCLPP_API_CPP BasePortChannel::BasePortChannel(const GpuNetIoSemaphore& semaphore) : BasePortChannel(semaphore, {}) {}

MSCCLPP_API_CPP BasePortChannel::BasePortChannel(const GpuNetIoSemaphore& semaphore,
                                                 const std::vector<GpuNetIoMemory>& memories)
    : semaphoreId_(0) {
  if (!semaphore.state_ || memories.size() > std::numeric_limits<uint32_t>::max()) {
    throw Error("Invalid GPUNetIO semaphore or memory table", ErrorCode::InvalidUsage);
  }
  const auto& binding = *semaphore.state_;
  const auto& connection = *binding.connection;
  const auto& service = *connection.service;
  std::vector<GpuNetIoMemoryDeviceHandle> descriptors;
  for (const auto& memory : memories) {
    if (!memory.state_ || memory.state_->service != connection.service ||
        (memory.state_->descriptor.rank != service.rank && memory.state_->descriptor.rank != connection.peer)) {
      throw Error("GPUNetIO channel memory belongs to another service or peer", ErrorCode::InvalidUsage);
    }
    descriptors.push_back(memory.state_->descriptor);
  }
  auto state = std::make_shared<detail::GpuNetIoChannelState>();
  state->semaphore = semaphore.state_;
  state->memories = memories;
  CudaDeviceGuard guard(service.cudaDeviceId);
  if (!descriptors.empty()) {
    MSCCLPP_CUDA_THROW(cudaMalloc(&state->table, descriptors.size() * sizeof(GpuNetIoMemoryDeviceHandle)));
    MSCCLPP_CUDA_THROW(cudaMemcpy(state->table, descriptors.data(),
                                  descriptors.size() * sizeof(GpuNetIoMemoryDeviceHandle), cudaMemcpyHostToDevice));
  }
  gpuNetIoHandle_ = BasePortChannelDeviceHandle(service.ctxGpu, connection.peer, connection.qpIndex, service.rank,
                                                binding.remoteSignal, binding.inboundCounter, binding.expectedCounter,
                                                state->table, static_cast<uint32_t>(memories.size()));
  gpuNetIoState_ = std::move(state);
}

MSCCLPP_API_CPP PortChannel::PortChannel(const GpuNetIoSemaphore& semaphore, const GpuNetIoMemory& dst,
                                         const GpuNetIoMemory& src)
    : BasePortChannel(semaphore, {dst, src}), dst_(0), src_(1) {
  if (dst.state_->local || !src.state_->local) {
    throw Error("GPUNetIO PortChannel requires a remote destination and local source", ErrorCode::InvalidUsage);
  }
}

}  // namespace mscclpp
