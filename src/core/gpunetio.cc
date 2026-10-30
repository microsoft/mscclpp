// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "gpunetio.hpp"

#include <endian.h>

#include <algorithm>
#include <cstring>
#include <mscclpp/gpu_utils.hpp>
#include <string>

#include "endpoint.hpp"
#include "logger.hpp"
#include "registered_memory.hpp"

#if defined(MSCCLPP_USE_GPUNETIO)
#include <infiniband/verbs.h>

#include "common/doca_gpunetio_verbs_dev.h"

#define MSCCLPP_DOCA_THROW(expr)                                                                            \
  do {                                                                                                      \
    doca_error_t status__ = (expr);                                                                         \
    if (status__ != DOCA_SUCCESS) {                                                                         \
      THROW(NET, Error, ErrorCode::SystemError, "DOCA GPUNetIO call failed: ", #expr, " (", status__, ")"); \
    }                                                                                                       \
  } while (0)
#endif  // defined(MSCCLPP_USE_GPUNETIO)

namespace mscclpp {

#if defined(MSCCLPP_USE_GPUNETIO)

GpuNetIoContext::GpuNetIoContext(IbCtx& ibCtx) { MSCCLPP_DOCA_THROW(doca_verbs_dev_open(ibCtx.getPd(), &dev_)); }

GpuNetIoContext::~GpuNetIoContext() {
  for (auto& [id, gpu] : gpus_) (void)doca_gpu_destroy(gpu);
  if (dev_) (void)doca_verbs_dev_close(dev_);
}

doca_gpu_t* GpuNetIoContext::gpu(int gpuId) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = gpus_.find(gpuId);
  if (it != gpus_.end()) return it->second;
  char pciBusId[32] = {};
  MSCCLPP_CUDATHROW(cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), gpuId));
  doca_gpu_t* gpu = nullptr;
  MSCCLPP_DOCA_THROW(doca_gpu_create(pciBusId, &gpu));
  gpus_.emplace(gpuId, gpu);
  return gpu;
}

namespace {

doca_verbs_mtu_size docaMtu(int mtu) {
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
      THROW(NET, Error, ErrorCode::InvalidUsage, "unsupported IB MTU ", mtu, " for GPUNetIO");
  }
}

}  // namespace

GpuNetIoQp::GpuNetIoQp(IbCtx& ibCtx, std::shared_ptr<GpuNetIoContext> context, int gpuId, int port, int gidIndex)
    : context_(std::move(context)), gpuId_(gpuId), gidIndex_(gidIndex < 0 ? 0 : gidIndex) {
  if (!ibCtx.supportsRdmaAtomics()) {
    THROW(NET, Error, ErrorCode::InvalidUsage, "IB Device mode requires RDMA atomics, which device ",
          ibCtx.getDevName(), " does not support");
  }
  info_ = ibCtx.getPortInfo(port, gidIndex_);
  port_ = port;

  int major = 0;
  MSCCLPP_CUDATHROW(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, gpuId_));
  if (major < 9) {
    THROW(NET, Error, ErrorCode::InvalidUsage, "IB Device mode requires Hopper or newer for BlueFlame submission");
  }

  CudaDeviceGuard deviceGuard(gpuId_);
  doca_gpu_verbs_qp_init_attr_hl attr{};
  attr.gpu_dev = context_->gpu(gpuId_);
  attr.net_dev = context_->dev_;
  attr.ibpd = ibCtx.getPd();
  attr.sq_nwqe = 1024;
  attr.nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF;
  attr.mreg_type = DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT;
  attr.send_dbr_mode_ext = DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_VALID_DBR;
  attr.cq_type = DOCA_GPUNETIO_VERBS_CQ_64B;
  attr.cq_collapsed = false;
  attr.enable_umem_cpu = false;
  attr.ordering_semantic = DOCA_VERBS_QP_ORDERING_SEMANTIC_IBTA;
  MSCCLPP_DOCA_THROW(doca_gpu_verbs_create_qp_hl(&attr, &qp_));
  MSCCLPP_DOCA_THROW(doca_verbs_qp_get_qpn(qp_->qp, &info_.qpn));
}

GpuNetIoQp::~GpuNetIoQp() {
  if (qp_) {
    CudaDeviceGuard deviceGuard(gpuId_);
    (void)doca_gpu_verbs_destroy_qp_hl(qp_);
  }
}

void GpuNetIoQp::connect(const IbQpInfo& remote) {
  doca_verbs_ah_attr_t* ah = nullptr;
  MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_create(context_->dev_, &ah));
  std::unique_ptr<doca_verbs_ah_attr_t, decltype(&doca_verbs_ah_attr_destroy)> ahGuard(ah, doca_verbs_ah_attr_destroy);
  doca_verbs_gid gid;
  std::memcpy(gid.raw, &remote.spn, sizeof(remote.spn));
  std::memcpy(gid.raw + sizeof(remote.spn), &remote.iid, sizeof(remote.iid));
  MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_set_gid(ah, gid));
  MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_set_dlid(ah, remote.lid));
  MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_set_sl(ah, 0));
  MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_set_sgid_index(ah, gidIndex_));
  MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_set_hop_limit(ah, 255));
  if (remote.linkLayer == IBV_LINK_LAYER_INFINIBAND && !info_.isGrh && !remote.isGrh) {
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
  auto* qp = qp_->qp;

  MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_next_state(attr, DOCA_VERBS_QP_STATE_INIT));
  MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_port_num(attr, port_));
  MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_allow_remote_write(attr, 1));
  MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_allow_remote_read(attr, 1));
  MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_atomic_mode(attr, DOCA_VERBS_QP_ATOMIC_MODE_IB_SPEC));
  MSCCLPP_DOCA_THROW(doca_verbs_qp_modify(qp, attr,
                                          DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE |
                                              DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ | DOCA_VERBS_QP_ATTR_ATOMIC_MODE |
                                              DOCA_VERBS_QP_ATTR_PKEY_INDEX | DOCA_VERBS_QP_ATTR_PORT_NUM));

  MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_next_state(attr, DOCA_VERBS_QP_STATE_RTR));
  MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_rq_psn(attr, 0));
  MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_dest_qp_num(attr, remote.qpn));
  MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_path_mtu(attr, docaMtu(std::min(info_.mtu, remote.mtu))));
  MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_ah_attr(attr, ah));
  MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_min_rnr_timer(attr, 12));
  MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_max_dest_rd_atomic(attr, 1));
  MSCCLPP_DOCA_THROW(doca_verbs_qp_modify(qp, attr,
                                          DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_RQ_PSN |
                                              DOCA_VERBS_QP_ATTR_DEST_QP_NUM | DOCA_VERBS_QP_ATTR_PATH_MTU |
                                              DOCA_VERBS_QP_ATTR_AH_ATTR | DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER |
                                              DOCA_VERBS_QP_ATTR_MAX_DEST_RD_ATOMIC));

  MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_next_state(attr, DOCA_VERBS_QP_STATE_RTS));
  MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_sq_psn(attr, 0));
  MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_ack_timeout(attr, 18));
  MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_retry_cnt(attr, 7));
  MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_rnr_retry(attr, 7));
  MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_max_rd_atomic(attr, 1));
  MSCCLPP_DOCA_THROW(doca_verbs_qp_modify(qp, attr,
                                          DOCA_VERBS_QP_ATTR_NEXT_STATE | DOCA_VERBS_QP_ATTR_SQ_PSN |
                                              DOCA_VERBS_QP_ATTR_ACK_TIMEOUT | DOCA_VERBS_QP_ATTR_RETRY_CNT |
                                              DOCA_VERBS_QP_ATTR_RNR_RETRY | DOCA_VERBS_QP_ATTR_MAX_QP_RD_ATOMIC));

  // Publish the QP to GPU memory; the copy is owned and released by DOCA together with the QP.
  CudaDeviceGuard deviceGuard(gpuId_);
  doca_gpu_dev_verbs_qp* devQp = nullptr;
  MSCCLPP_DOCA_THROW(doca_gpu_verbs_get_qp_dev(qp_->qp_gverbs, &devQp));
  deviceQp_ = devQp;
}

#else  // !defined(MSCCLPP_USE_GPUNETIO)

GpuNetIoContext::GpuNetIoContext(IbCtx&) {}

GpuNetIoContext::~GpuNetIoContext() = default;

GpuNetIoQp::GpuNetIoQp(IbCtx&, std::shared_ptr<GpuNetIoContext>, int, int, int) {
  THROW(NET, Error, ErrorCode::InvalidUsage,
        "IB Device mode is unavailable: MSCCL++ was built without GPUNetIO (requires CUDA and InfiniBand)");
}

GpuNetIoQp::~GpuNetIoQp() = default;

void GpuNetIoQp::connect(const IbQpInfo&) {}

#endif  // !defined(MSCCLPP_USE_GPUNETIO)

GpuNetIoConnection::GpuNetIoConnection(std::shared_ptr<Context> context, const Endpoint& localEndpoint,
                                       const Endpoint& remoteEndpoint)
    : BaseConnection(context, localEndpoint),
      transport_(localEndpoint.transport()),
      remoteTransport_(remoteEndpoint.transport()),
      qp_(getImpl(localEndpoint).gpuNetIoQp_) {
  if (!qp_) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "local endpoint was not created in IB Device mode");
  }
  if (getImpl(remoteEndpoint).ibNoAtomic_) {
    THROW(CONN, Error, ErrorCode::InvalidUsage,
          "IB Device mode signals with RDMA atomics; the remote endpoint must not use IB HostNoAtomic mode");
  }
  qp_->connect(getImpl(remoteEndpoint).ibQpInfo_);
  CudaDeviceGuard deviceGuard(localEndpoint.device().id);
  atomicFetchBuff_ = detail::gpuCallocShared<uint64_t>();
  atomicFetchMem_ = context->registerMemory(atomicFetchBuff_.get(), sizeof(uint64_t), transport_);
}

static const std::string kGpuNetIoHostOpError =
    "GPUNetIO connections only support GPU-initiated PortChannel operations; host-side Connection operations are not "
    "supported";

void GpuNetIoConnection::write(RegisteredMemory, uint64_t, RegisteredMemory, uint64_t, uint64_t) {
  THROW(CONN, Error, ErrorCode::InvalidUsage, kGpuNetIoHostOpError);
}

void GpuNetIoConnection::updateAndSync(RegisteredMemory, uint64_t, uint64_t*, uint64_t) {
  THROW(CONN, Error, ErrorCode::InvalidUsage, kGpuNetIoHostOpError);
}

void GpuNetIoConnection::accumulate(RegisteredMemory, uint64_t, int64_t) {
  THROW(CONN, Error, ErrorCode::InvalidUsage, kGpuNetIoHostOpError);
}

void GpuNetIoConnection::flush(int64_t) { THROW(CONN, Error, ErrorCode::InvalidUsage, kGpuNetIoHostOpError); }

DeviceBackendHandle GpuNetIoConnection::getDeviceBackendInfo(const RegisteredMemory& signal,
                                                             uint64_t* outboundToken) const {
  const auto& signalImpl = getImpl(signal);
  if (!signalImpl.transports.has(remoteTransport_)) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "GPUNetIO semaphore token is not registered for ", remoteTransport_);
  }
  const auto& signalInfo = signalImpl.getTransportInfo(remoteTransport_);
  if (signalInfo.ibLocal || signalImpl.size < sizeof(uint64_t) || signalInfo.ibMrInfo.addr % sizeof(uint64_t) != 0) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "invalid GPUNetIO semaphore token");
  }
  const auto& fetchInfo = getImpl(atomicFetchMem_).getTransportInfo(transport_);

  DeviceBackendHandle handle = {};
  handle.qp = qp_->deviceQp();
  handle.signalAddr = signalInfo.ibMrInfo.addr;
  handle.signalKey = htobe32(signalInfo.ibMrInfo.rkey);
  handle.atomicFetchAddr = fetchInfo.ibMrInfo.addr;
  handle.atomicFetchKey = htobe32(fetchInfo.ibMr->getLkey());
  handle.outboundToken = outboundToken;
  handle.blueFlameLimit = DeviceBackendHandle::kDefaultBlueFlameLimit;
  return handle;
}

DeviceMemoryHandle GpuNetIoConnection::getDeviceMemoryInfo(const RegisteredMemory& memory) const {
  const auto& impl = getImpl(memory);
  if (impl.transports.has(transport_)) {
    const auto& info = impl.getTransportInfo(transport_);
    if (info.ibLocal) return {info.ibMrInfo.addr, htobe32(info.ibMr->getLkey())};
  }
  if (impl.transports.has(remoteTransport_)) {
    const auto& info = impl.getTransportInfo(remoteTransport_);
    if (!info.ibLocal) return {info.ibMrInfo.addr, htobe32(info.ibMrInfo.rkey)};
  }
  return {};
}

}  // namespace mscclpp
