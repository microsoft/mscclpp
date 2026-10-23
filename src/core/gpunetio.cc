// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "gpunetio.hpp"

#include <endian.h>

#include <cstring>
#include <mscclpp/gpu_utils.hpp>
#include <mutex>
#include <string>
#include <unordered_map>

#include "endpoint.hpp"
#include "logger.hpp"
#include "registered_memory.hpp"

#if defined(MSCCLPP_USE_GPUNETIO)
#include <infiniband/verbs.h>

extern "C" {
#include "host/doca_gpunetio.h"
#include "host/doca_gpunetio_high_level.h"
#include "host/doca_verbs.h"
}
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

// DOCA device handles shared by all GPUNetIO QPs of a Context.
struct GpuNetIoContext {
  std::mutex mutex;
  std::unordered_map<Transport, doca_dev_t*> devs;
  std::unordered_map<int, doca_gpu_t*> gpus;

  ~GpuNetIoContext() {
    for (auto& [id, gpu] : gpus) (void)doca_gpu_destroy(gpu);
    for (auto& [transport, dev] : devs) (void)doca_verbs_dev_close(dev);
  }

  doca_dev_t* dev(IbCtx* ibCtx, Transport transport) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = devs.find(transport);
    if (it != devs.end()) return it->second;
    doca_dev_t* dev = nullptr;
    MSCCLPP_DOCA_THROW(doca_verbs_dev_open(ibCtx->getPd(), &dev));
    devs.emplace(transport, dev);
    return dev;
  }

  doca_gpu_t* gpu(int gpuId) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = gpus.find(gpuId);
    if (it != gpus.end()) return it->second;
    char pciBusId[32] = {};
    MSCCLPP_CUDATHROW(cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), gpuId));
    doca_gpu_t* gpu = nullptr;
    MSCCLPP_DOCA_THROW(doca_gpu_create(pciBusId, &gpu));
    gpus.emplace(gpuId, gpu);
    return gpu;
  }
};

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

void checkMtu(int mtu) { (void)docaMtu(mtu); }

}  // namespace

struct GpuNetIoQp::Impl {
  std::shared_ptr<GpuNetIoContext> context;
  doca_dev_t* dev = nullptr;
  doca_gpu_verbs_qp_hl* qp = nullptr;
  doca_gpu_dev_verbs_qp* deviceQp = nullptr;
  int gpuId;
  int port;
  int gidIndex;
  IbQpInfo info;
};

GpuNetIoQp::GpuNetIoQp(IbCtx* ibCtx, std::shared_ptr<GpuNetIoContext>& sharedContext, Transport transport, int gpuId,
                       int port, int gidIndex)
    : pimpl_(std::make_unique<Impl>()) {
  if (!ibCtx->supportsRdmaAtomics()) {
    THROW(NET, Error, ErrorCode::InvalidUsage, "IB GpuNetIo mode requires RDMA atomics, which device ",
          ibCtx->getDevName(), " does not support");
  }
  pimpl_->gpuId = gpuId;
  pimpl_->gidIndex = gidIndex < 0 ? 0 : gidIndex;
  pimpl_->info = ibCtx->getPortInfo(port, pimpl_->gidIndex);
  pimpl_->port = port;
  if ((pimpl_->info.linkLayer != IBV_LINK_LAYER_INFINIBAND || pimpl_->info.isGrh) && pimpl_->info.spn == 0 &&
      pimpl_->info.iid == 0) {
    THROW(NET, Error, ErrorCode::InvalidUsage, "IB GpuNetIo mode requires a nonzero GID for RoCE or GRH routing (",
          ibCtx->getDevName(), " port ", port, " GID index ", pimpl_->gidIndex, ")");
  }
  checkMtu(pimpl_->info.mtu);

  if (!sharedContext) sharedContext = std::make_shared<GpuNetIoContext>();
  pimpl_->context = sharedContext;

  CudaDeviceGuard deviceGuard(gpuId);
  pimpl_->dev = pimpl_->context->dev(ibCtx, transport);
  doca_gpu_t* gpu = pimpl_->context->gpu(gpuId);

  doca_gpu_verbs_qp_init_attr_hl attr{};
  attr.gpu_dev = gpu;
  attr.net_dev = pimpl_->dev;
  attr.ibpd = ibCtx->getPd();
  attr.sq_nwqe = 1024;
  attr.nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB;
  attr.mreg_type = DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT;
  attr.send_dbr_mode_ext = DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_VALID_DBR;
  attr.cq_type = DOCA_GPUNETIO_VERBS_CQ_64B;
  attr.cq_collapsed = false;
  attr.enable_umem_cpu = false;
  attr.ordering_semantic = DOCA_VERBS_QP_ORDERING_SEMANTIC_IBTA;
  MSCCLPP_DOCA_THROW(doca_gpu_verbs_create_qp_hl(&attr, &pimpl_->qp));

  // The device code relies on GPU-rung doorbells with valid DBR records and 64B CQEs in GPU memory.
  const auto* qp = pimpl_->qp;
  if (qp == nullptr || qp->qp_gverbs == nullptr || qp->qp_gverbs->qp_cpu == nullptr ||
      qp->nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB ||
      qp->send_dbr_mode_ext != DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_VALID_DBR || qp->qp_gverbs->cpu_proxy ||
      qp->qp_gverbs->send_dbr_mode_ext != DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_VALID_DBR ||
      qp->qp_gverbs->cq_type != DOCA_GPUNETIO_VERBS_CQ_64B ||
      qp->qp_gverbs->qp_cpu->nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB ||
      qp->qp_gverbs->qp_cpu->mem_type != DOCA_GPUNETIO_VERBS_MEM_TYPE_GPU) {
    if (pimpl_->qp) (void)doca_gpu_verbs_destroy_qp_hl(pimpl_->qp);
    THROW(NET, Error, ErrorCode::InvalidUsage, "device ", ibCtx->getDevName(),
          " does not support GPU-initiated (GPU_SM_DB) GPUNetIO queue pairs");
  }
  MSCCLPP_DOCA_THROW(doca_verbs_qp_get_qpn(pimpl_->qp->qp, &pimpl_->info.qpn));
}

GpuNetIoQp::~GpuNetIoQp() {
  if (pimpl_->qp) {
    CudaDeviceGuard deviceGuard(pimpl_->gpuId);
    (void)doca_gpu_verbs_destroy_qp_hl(pimpl_->qp);
  }
}

const IbQpInfo& GpuNetIoQp::info() const { return pimpl_->info; }

void* GpuNetIoQp::deviceQp() const { return pimpl_->deviceQp; }

void GpuNetIoQp::connect(const IbQpInfo& remote) {
  const auto& local = pimpl_->info;
  checkMtu(remote.mtu);
  doca_verbs_ah_attr_t* ah = nullptr;
  MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_create(pimpl_->dev, &ah));
  std::unique_ptr<doca_verbs_ah_attr_t, decltype(&doca_verbs_ah_attr_destroy)> ahGuard(ah, doca_verbs_ah_attr_destroy);
  doca_verbs_gid gid;
  std::memcpy(gid.raw, &remote.spn, sizeof(remote.spn));
  std::memcpy(gid.raw + sizeof(remote.spn), &remote.iid, sizeof(remote.iid));
  MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_set_gid(ah, gid));
  MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_set_dlid(ah, remote.lid));
  MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_set_sl(ah, 0));
  MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_set_sgid_index(ah, pimpl_->gidIndex));
  MSCCLPP_DOCA_THROW(doca_verbs_ah_attr_set_hop_limit(ah, 255));
  if (remote.linkLayer == IBV_LINK_LAYER_INFINIBAND && !local.isGrh && !remote.isGrh) {
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
  auto* qp = pimpl_->qp->qp;

  MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_next_state(attr, DOCA_VERBS_QP_STATE_INIT));
  MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_port_num(attr, pimpl_->port));
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
  MSCCLPP_DOCA_THROW(doca_verbs_qp_attr_set_path_mtu(attr, docaMtu(std::min(local.mtu, remote.mtu))));
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
  CudaDeviceGuard deviceGuard(pimpl_->gpuId);
  MSCCLPP_DOCA_THROW(doca_gpu_verbs_get_qp_dev(pimpl_->qp->qp_gverbs, &pimpl_->deviceQp));
}

#else  // !defined(MSCCLPP_USE_GPUNETIO)

struct GpuNetIoContext {};

struct GpuNetIoQp::Impl {
  IbQpInfo info;
};

GpuNetIoQp::GpuNetIoQp(IbCtx*, std::shared_ptr<GpuNetIoContext>&, Transport, int, int, int) {
  THROW(NET, Error, ErrorCode::InvalidUsage,
        "IB GpuNetIo mode is unavailable: MSCCL++ was built without GPUNetIO (MSCCLPP_USE_GPUNETIO=OFF)");
}

GpuNetIoQp::~GpuNetIoQp() = default;

const IbQpInfo& GpuNetIoQp::info() const { return pimpl_->info; }

void* GpuNetIoQp::deviceQp() const { return nullptr; }

void GpuNetIoQp::connect(const IbQpInfo&) {}

#endif  // !defined(MSCCLPP_USE_GPUNETIO)

GpuNetIoConnection::GpuNetIoConnection(std::shared_ptr<Context> context, const Endpoint& localEndpoint,
                                       const Endpoint& remoteEndpoint)
    : BaseConnection(context, localEndpoint),
      transport_(localEndpoint.transport()),
      remoteTransport_(remoteEndpoint.transport()),
      remoteHostHash_(getImpl(remoteEndpoint).hostHash_),
      remotePidHash_(getImpl(remoteEndpoint).pidHash_),
      qp_(getImpl(localEndpoint).gpuNetIoQp_) {
  if (!qp_) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "local endpoint was not created in IB GpuNetIo mode");
  }
  if (getImpl(remoteEndpoint).ibNoAtomic_) {
    THROW(CONN, Error, ErrorCode::InvalidUsage,
          "IB GpuNetIo mode signals with RDMA atomics; the remote endpoint must not use IB HostNoAtomic mode");
  }
  qp_->connect(getImpl(remoteEndpoint).ibQpInfo_);
  CudaDeviceGuard deviceGuard(localEndpoint.device().id);
  atomicScratch_ = detail::gpuCallocShared<uint64_t>();
  atomicScratchMem_ = context->registerMemory(atomicScratch_.get(), sizeof(uint64_t), transport_);
  INFO(CONN, "GPUNetIO connection created");
}

Transport GpuNetIoConnection::transport() const { return transport_; }

Transport GpuNetIoConnection::remoteTransport() const { return remoteTransport_; }

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

detail::GpuNetIoDeviceHandle GpuNetIoConnection::deviceHandle(const RegisteredMemory& signal,
                                                              std::vector<detail::GpuNetIoMemoryDeviceHandle>& table,
                                                              const std::vector<RegisteredMemory>& memories) const {
  const auto& signalImpl = getImpl(signal);
  if (!signalImpl.transports.has(remoteTransport_)) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "GPUNetIO semaphore token is not registered for ", remoteTransport_);
  }
  const auto& signalInfo = signalImpl.getTransportInfo(remoteTransport_);
  if (signalInfo.ibLocal || signalImpl.size < sizeof(uint64_t) || signalInfo.ibMrInfo.addr % sizeof(uint64_t) != 0) {
    THROW(CONN, Error, ErrorCode::InvalidUsage, "invalid GPUNetIO semaphore token");
  }
  const auto& scratchInfo = getImpl(atomicScratchMem_).getTransportInfo(transport_);

  const uint64_t localHostHash = getImpl(localEndpoint_).hostHash_;
  const uint64_t localPidHash = getImpl(localEndpoint_).pidHash_;
  table.assign(memories.size(), detail::GpuNetIoMemoryDeviceHandle{});
  for (size_t i = 0; i < memories.size(); ++i) {
    const auto& impl = getImpl(memories[i]);
    if (impl.hostHash == localHostHash && impl.pidHash == localPidHash && impl.transports.has(transport_)) {
      const auto& info = impl.getTransportInfo(transport_);
      if (info.ibLocal) table[i] = {info.ibMrInfo.addr, impl.size, htobe32(info.ibMr->getLkey()), 1};
    } else if (impl.hostHash == remoteHostHash_ && impl.pidHash == remotePidHash_ &&
               impl.transports.has(remoteTransport_)) {
      const auto& info = impl.getTransportInfo(remoteTransport_);
      if (!info.ibLocal) table[i] = {info.ibMrInfo.addr, impl.size, htobe32(info.ibMrInfo.rkey), 0};
    }
  }

  detail::GpuNetIoDeviceHandle handle{};
  handle.qp = qp_->deviceQp();
  handle.signalAddr = signalInfo.ibMrInfo.addr;
  handle.signalKey = htobe32(signalInfo.ibMrInfo.rkey);
  handle.atomicScratchKey = htobe32(scratchInfo.ibMr->getLkey());
  handle.atomicScratchAddr = scratchInfo.ibMrInfo.addr;
  handle.numMemories = static_cast<uint32_t>(table.size());
  return handle;
}

namespace detail {

std::shared_ptr<GpuNetIoChannel> GpuNetIoChannel::bind(Host2DeviceSemaphore& semaphore,
                                                       const std::vector<RegisteredMemory>& memories) {
  auto connection = std::dynamic_pointer_cast<GpuNetIoConnection>(BaseConnection::getImpl(semaphore.connection()));
  if (!connection) return nullptr;
  auto channel = std::make_shared<GpuNetIoChannel>();
  channel->connection = connection;
  channel->memories = memories;
  std::vector<GpuNetIoMemoryDeviceHandle> table;
  channel->deviceHandle = connection->deviceHandle(semaphore.semaphore_.remoteMemory(), table, memories);
  if (!table.empty()) {
    CudaDeviceGuard deviceGuard(connection->localDevice().id);
    channel->table = gpuCallocShared<GpuNetIoMemoryDeviceHandle>(table.size());
    mscclpp::gpuMemcpy<GpuNetIoMemoryDeviceHandle>(channel->table.get(), table.data(), table.size(),
                                                   cudaMemcpyHostToDevice);
  }
  channel->deviceHandle.memories = channel->table.get();
  return channel;
}

}  // namespace detail
}  // namespace mscclpp
