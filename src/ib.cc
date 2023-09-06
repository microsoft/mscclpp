// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ib.hpp"

#include <infiniband/verbs.h>
#include <malloc.h>
#include <unistd.h>

#include <cstring>
#include <mscclpp/core.hpp>
#include <mscclpp/fifo.hpp>
#include <sstream>
#include <string>

#include "api.h"
#include "debug.h"

static ibv_device_attr getDeviceAttr(ibv_context* ctx) {
  ibv_device_attr devAttr;
  if (ibv_query_device(ctx, &devAttr) != 0) {
    std::stringstream err;
    err << "ibv_query_device failed (errno " << errno << ")";
    throw mscclpp::IbError(err.str(), errno);
  }
  return devAttr;
}

static ibv_qp_attr createQpAttr() {
  ibv_qp_attr qpAttr;
  std::memset(&qpAttr, 0, sizeof(qpAttr));
  return qpAttr;
}

namespace mscclpp {

IbMr::IbMr(ibv_pd* pd, void* buff, std::size_t size) : buff(buff) {
  if (size == 0) {
    throw std::invalid_argument("invalid size: " + std::to_string(size));
  }
  static __thread uintptr_t pageSize = 0;
  if (pageSize == 0) {
    pageSize = sysconf(_SC_PAGESIZE);
  }
  uintptr_t addr = reinterpret_cast<uintptr_t>(buff) & -pageSize;
  std::size_t pages = (size + (reinterpret_cast<uintptr_t>(buff) - addr) + pageSize - 1) / pageSize;
  this->mr = ibv_reg_mr(pd, reinterpret_cast<void*>(addr), pages * pageSize,
                        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                            IBV_ACCESS_RELAXED_ORDERING | IBV_ACCESS_REMOTE_ATOMIC);
  if (this->mr == nullptr) {
    std::stringstream err;
    err << "ibv_reg_mr failed (errno " << errno << ")";
    throw mscclpp::IbError(err.str(), errno);
  }
  this->size = pages * pageSize;
}

IbMr::~IbMr() { ibv_dereg_mr(this->mr); }

IbMrInfo IbMr::getInfo() const {
  IbMrInfo info;
  info.addr = reinterpret_cast<uint64_t>(this->buff);
  info.rkey = this->mr->rkey;
  return info;
}

const void* IbMr::getBuff() const { return this->buff; }

uint32_t IbMr::getLkey() const { return this->mr->lkey; }

IbQp::IbQp(ibv_context* ctx, ibv_pd* pd, int port, int maxCqSize, int maxCqPollNum, int maxSendWr, int maxRecvWr,
           int maxWrPerSend, int maxNumSgesPerWr)
    : maxCqPollNum_(maxCqPollNum), maxWrPerSend_(maxWrPerSend), maxNumSgesPerWr_(maxNumSgesPerWr) {
  this->cq = ibv_create_cq(ctx, maxCqSize, nullptr, nullptr, 0);
  if (this->cq == nullptr) {
    std::stringstream err;
    err << "ibv_create_cq failed (errno " << errno << ")";
    throw mscclpp::IbError(err.str(), errno);
  }

  struct ibv_qp_init_attr qpInitAttr;
  std::memset(&qpInitAttr, 0, sizeof(qpInitAttr));
  qpInitAttr.sq_sig_all = 0;
  qpInitAttr.send_cq = this->cq;
  qpInitAttr.recv_cq = this->cq;
  qpInitAttr.qp_type = IBV_QPT_RC;
  qpInitAttr.cap.max_send_wr = maxSendWr;
  qpInitAttr.cap.max_recv_wr = maxRecvWr;
  qpInitAttr.cap.max_send_sge = 1;
  qpInitAttr.cap.max_recv_sge = 1;
  qpInitAttr.cap.max_inline_data = 0;

  struct ibv_qp* _qp = ibv_create_qp(pd, &qpInitAttr);
  if (_qp == nullptr) {
    std::stringstream err;
    err << "ibv_create_qp failed (errno " << errno << ")";
    throw mscclpp::IbError(err.str(), errno);
  }

  struct ibv_port_attr portAttr;
  if (ibv_query_port(ctx, port, &portAttr) != 0) {
    std::stringstream err;
    err << "ibv_query_port failed (errno " << errno << ")";
    throw mscclpp::IbError(err.str(), errno);
  }
  this->info.lid = portAttr.lid;
  this->info.port = port;
  this->info.linkLayer = portAttr.link_layer;
  this->info.qpn = _qp->qp_num;
  this->info.mtu = portAttr.active_mtu;
  this->info.is_grh = (portAttr.flags & IBV_QPF_GRH_REQUIRED);

  if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND || this->info.is_grh) {
    union ibv_gid gid;
    if (ibv_query_gid(ctx, port, 0, &gid) != 0) {
      std::stringstream err;
      err << "ibv_query_gid failed (errno " << errno << ")";
      throw mscclpp::IbError(err.str(), errno);
    }
    this->info.spn = gid.global.subnet_prefix;
    this->info.iid = gid.global.interface_id;
  }

  ibv_qp_attr qpAttr = createQpAttr();
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.pkey_index = 0;
  qpAttr.port_num = port;
  qpAttr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
  if (ibv_modify_qp(_qp, &qpAttr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
    std::stringstream err;
    err << "ibv_modify_qp failed (errno " << errno << ")";
    throw mscclpp::IbError(err.str(), errno);
  }
  this->qp = _qp;
  this->wrs = std::make_unique<ibv_send_wr[]>(maxWrPerSend_);
  this->sges = std::make_unique<ibv_sge[]>(maxWrPerSend_ * maxNumSgesPerWr_);
  this->wcs = std::make_unique<ibv_wc[]>(maxCqPollNum_);
  numStagedWrs_ = 0;
  numStagedSges_ = 0;
}

IbQp::~IbQp() {
  ibv_destroy_qp(this->qp);
  ibv_destroy_cq(this->cq);
}

void IbQp::rtr(const IbQpInfo& info) {
  ibv_qp_attr qpAttr = createQpAttr();
  qpAttr.qp_state = IBV_QPS_RTR;
  qpAttr.path_mtu = static_cast<ibv_mtu>(info.mtu);
  qpAttr.dest_qp_num = info.qpn;
  qpAttr.rq_psn = 0;
  qpAttr.max_dest_rd_atomic = 1;
  qpAttr.min_rnr_timer = 0x12;
  if (info.linkLayer == IBV_LINK_LAYER_ETHERNET || info.is_grh) {
    qpAttr.ah_attr.is_global = 1;
    qpAttr.ah_attr.grh.dgid.global.subnet_prefix = info.spn;
    qpAttr.ah_attr.grh.dgid.global.interface_id = info.iid;
    qpAttr.ah_attr.grh.flow_label = 0;
    qpAttr.ah_attr.grh.sgid_index = 0;
    qpAttr.ah_attr.grh.hop_limit = 255;
    qpAttr.ah_attr.grh.traffic_class = 0;
  } else {
    qpAttr.ah_attr.is_global = 0;
  }
  qpAttr.ah_attr.dlid = info.lid;
  qpAttr.ah_attr.sl = 0;
  qpAttr.ah_attr.src_path_bits = 0;
  qpAttr.ah_attr.port_num = info.port;
  int ret = ibv_modify_qp(this->qp, &qpAttr,
                          IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                              IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
  if (ret != 0) {
    std::stringstream err;
    err << "ibv_modify_qp failed (errno " << errno << ")";
    throw mscclpp::IbError(err.str(), errno);
  }
}

void IbQp::rts() {
  ibv_qp_attr qpAttr = createQpAttr();
  qpAttr.qp_state = IBV_QPS_RTS;
  qpAttr.timeout = 18;
  qpAttr.retry_cnt = 7;
  qpAttr.rnr_retry = 7;
  qpAttr.sq_psn = 0;
  qpAttr.max_rd_atomic = 1;
  int ret = ibv_modify_qp(
      this->qp, &qpAttr,
      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
  if (ret != 0) {
    std::stringstream err;
    err << "ibv_modify_qp failed (errno " << errno << ")";
    throw mscclpp::IbError(err.str(), errno);
  }
}

IbQp::WrInfo IbQp::getNewWrInfo(int numSges) {
  if (numStagedWrs_ >= maxWrPerSend_) {
    std::stringstream err;
    err << "too many outstanding work requests. limit is " << maxWrPerSend_;
    throw mscclpp::Error(err.str(), ErrorCode::InvalidUsage);
  }
  if (numSges > maxNumSgesPerWr_) {
    std::stringstream err;
    err << "too many sges per work request. limit is " << maxNumSgesPerWr_;
    throw mscclpp::Error(err.str(), ErrorCode::InvalidUsage);
  }

  ibv_send_wr* wr_ = &this->wrs[numStagedWrs_];
  ibv_sge* sge_ = &this->sges[numStagedSges_];
  wr_->sg_list = sge_;
  wr_->num_sge = numSges;
  wr_->next = nullptr;
  if (numStagedWrs_ > 0) {
    this->wrs[numStagedWrs_ - 1].next = wr_;
  }
  numStagedWrs_++;
  numStagedSges_ += numSges;
  return IbQp::WrInfo{wr_, sge_};
}

void IbQp::stageSend(const IbMr* mr, const IbMrInfo& info, uint32_t size, uint64_t wrId, uint64_t srcOffset,
                     uint64_t dstOffset, bool signaled) {
  auto wrInfo = this->getNewWrInfo(1);
  wrInfo.wr->wr_id = wrId;
  wrInfo.wr->opcode = IBV_WR_RDMA_WRITE;
  wrInfo.wr->send_flags = signaled ? IBV_SEND_SIGNALED : 0;
  wrInfo.wr->wr.rdma.remote_addr = (uint64_t)(info.addr) + dstOffset;
  wrInfo.wr->wr.rdma.rkey = info.rkey;
  wrInfo.sge->addr = (uint64_t)(mr->getBuff()) + srcOffset;
  wrInfo.sge->length = size;
  wrInfo.sge->lkey = mr->getLkey();
}

void IbQp::stageAtomicAdd(const IbMr* mr, const IbMrInfo& info, uint64_t wrId, uint64_t dstOffset, uint64_t addVal) {
  auto wrInfo = this->getNewWrInfo(1);
  wrInfo.wr->wr_id = wrId;
  wrInfo.wr->opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
  wrInfo.wr->send_flags = 0;  // atomic op cannot be signaled
  wrInfo.wr->wr.atomic.remote_addr = (uint64_t)(info.addr) + dstOffset;
  wrInfo.wr->wr.atomic.rkey = info.rkey;
  wrInfo.wr->wr.atomic.compare_add = addVal;
  wrInfo.sge->addr = (uint64_t)(mr->getBuff());
  wrInfo.sge->length = sizeof(uint64_t);  // atomic op is always on uint64_t
  wrInfo.sge->lkey = mr->getLkey();
}

void IbQp::stageSendWithImm(const IbMr* mr, const IbMrInfo& info, uint32_t size, uint64_t wrId, uint64_t srcOffset,
                            uint64_t dstOffset, bool signaled, unsigned int immData) {
  auto wrInfo = this->getNewWrInfo(1);
  wrInfo.wr->wr_id = wrId;
  wrInfo.wr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wrInfo.wr->send_flags = signaled ? IBV_SEND_SIGNALED : 0;
  wrInfo.wr->wr.rdma.remote_addr = (uint64_t)(info.addr) + dstOffset;
  wrInfo.wr->wr.rdma.rkey = info.rkey;
  wrInfo.wr->imm_data = immData;
  wrInfo.sge->addr = (uint64_t)(mr->getBuff()) + srcOffset;
  wrInfo.sge->length = size;
  wrInfo.sge->lkey = mr->getLkey();
}

void IbQp::stageSendGather(const std::vector<IbMr*>& srcMrs, const IbMrInfo& dstInfo,
                           const std::vector<uint32_t>& srcSizes, uint64_t wrId,
                           const std::vector<uint64_t>& srcOffsets, uint64_t dstOffset, bool signaled) {
  size_t numSrcs = srcMrs.size();
  if (numSrcs != srcSizes.size() || numSrcs != srcOffsets.size()) {
    std::stringstream err;
    err << "invalid srcs: srcMrs.size()=" << numSrcs << ", srcSizes.size()=" << srcSizes.size()
        << ", srcOffsets.size()=" << srcOffsets.size();
    throw mscclpp::Error(err.str(), ErrorCode::InvalidUsage);
  }
  auto wrInfo = this->getNewWrInfo(numSrcs);
  wrInfo.wr->wr_id = wrId;
  wrInfo.wr->opcode = IBV_WR_RDMA_READ;
  wrInfo.wr->send_flags = signaled ? IBV_SEND_SIGNALED : 0;
  wrInfo.wr->wr.rdma.remote_addr = (uint64_t)(dstInfo.addr) + dstOffset;
  wrInfo.wr->wr.rdma.rkey = dstInfo.rkey;
  for (size_t i = 0; i < numSrcs; ++i) {
    wrInfo.sge[i].addr = (uint64_t)(srcMrs[i]->getBuff()) + srcOffsets[i];
    wrInfo.sge[i].length = srcSizes[i];
    wrInfo.sge[i].lkey = srcMrs[i]->getLkey();
  }
}

void IbQp::postSend() {
  if (numStagedWrs_ == 0) {
    return;
  }
  struct ibv_send_wr* bad_wr;
  int ret = ibv_post_send(this->qp, this->wrs.get(), &bad_wr);
  if (ret != 0) {
    std::stringstream err;
    err << "ibv_post_send failed (errno " << errno << ")";
    throw mscclpp::IbError(err.str(), errno);
  }
  numStagedWrs_ = 0;
  numStagedSges_ = 0;
}

void IbQp::postRecv(uint64_t wrId) {
  struct ibv_recv_wr wr, *bad_wr;
  wr.wr_id = wrId;
  wr.sg_list = nullptr;
  wr.num_sge = 0;
  wr.next = nullptr;
  int ret = ibv_post_recv(this->qp, &wr, &bad_wr);
  if (ret != 0) {
    std::stringstream err;
    err << "ibv_post_recv failed (errno " << errno << ")";
    throw mscclpp::IbError(err.str(), errno);
  }
}

int IbQp::pollCq() { return ibv_poll_cq(this->cq, maxCqPollNum_, this->wcs.get()); }

IbQpInfo& IbQp::getInfo() { return this->info; }

const ibv_wc* IbQp::getWc(int idx) const { return &this->wcs[idx]; }

IbCtx::IbCtx(const std::string& devName) : devName(devName) {
  int num;
  struct ibv_device** devices = ibv_get_device_list(&num);
  for (int i = 0; i < num; ++i) {
    if (std::string(devices[i]->name) == devName) {
      this->ctx = ibv_open_device(devices[i]);
      break;
    }
  }
  ibv_free_device_list(devices);
  if (this->ctx == nullptr) {
    std::stringstream err;
    err << "ibv_open_device failed (errno " << errno << ", device name << " << devName << ")";
    throw mscclpp::IbError(err.str(), errno);
  }
  this->pd = ibv_alloc_pd(this->ctx);
  if (this->pd == nullptr) {
    std::stringstream err;
    err << "ibv_alloc_pd failed (errno " << errno << ")";
    throw mscclpp::IbError(err.str(), errno);
  }
}

IbCtx::~IbCtx() {
  this->mrs.clear();
  this->qps.clear();
  if (this->pd != nullptr) {
    ibv_dealloc_pd(this->pd);
  }
  if (this->ctx != nullptr) {
    ibv_close_device(this->ctx);
  }
}

bool IbCtx::isPortUsable(int port) const {
  struct ibv_port_attr portAttr;
  if (ibv_query_port(this->ctx, port, &portAttr) != 0) {
    std::stringstream err;
    err << "ibv_query_port failed (errno " << errno << ", port << " << port << ")";
    throw mscclpp::IbError(err.str(), errno);
  }
  return portAttr.state == IBV_PORT_ACTIVE &&
         (portAttr.link_layer == IBV_LINK_LAYER_ETHERNET || portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND);
}

int IbCtx::getAnyActivePort() const {
  ibv_device_attr devAttr = getDeviceAttr(this->ctx);
  for (uint8_t port = 1; port <= devAttr.phys_port_cnt; ++port) {
    if (this->isPortUsable(port)) {
      return port;
    }
  }
  return -1;
}

void IbCtx::validateConfig(int maxCqSize, int maxCqPollNum, int maxSendWr, int maxRecvWr, int maxWrPerSend,
                           int maxNumSgesPerWr, int port) const {
  if (!this->isPortUsable(port)) {
    throw mscclpp::Error("invalid IB port: " + std::to_string(port), ErrorCode::InvalidUsage);
  }
  ibv_device_attr devAttr = getDeviceAttr(this->ctx);
  if (maxCqSize > devAttr.max_cqe || maxCqSize < 1) {
    throw mscclpp::Error("invalid maxCqSize: " + std::to_string(maxCqSize), ErrorCode::InvalidUsage);
  }
  if (maxCqPollNum > maxCqSize || maxCqPollNum < 1) {
    throw mscclpp::Error("invalid maxCqPollNum: " + std::to_string(maxCqPollNum), ErrorCode::InvalidUsage);
  }
  if (maxSendWr > devAttr.max_qp_wr || maxSendWr < 1) {
    throw mscclpp::Error("invalid maxSendWr: " + std::to_string(maxSendWr), ErrorCode::InvalidUsage);
  }
  if (maxRecvWr > devAttr.max_qp_wr || maxRecvWr < 1) {
    throw mscclpp::Error("invalid maxRecvWr: " + std::to_string(maxRecvWr), ErrorCode::InvalidUsage);
  }
  if (maxWrPerSend > maxSendWr || maxWrPerSend < 1) {
    throw mscclpp::Error("invalid maxWrPerSend: " + std::to_string(maxWrPerSend), ErrorCode::InvalidUsage);
  }
  if (maxNumSgesPerWr > devAttr.max_sge || maxNumSgesPerWr < 1) {
    throw mscclpp::Error("invalid maxNumSgesPerWr: " + std::to_string(maxNumSgesPerWr), ErrorCode::InvalidUsage);
  }
}

IbQp* IbCtx::createQp(int maxCqSize, int maxCqPollNum, int maxSendWr, int maxRecvWr, int maxWrPerSend,
                      int maxNumSgesPerWr, int port /*=-1*/) {
  if (port == -1) {
    port = this->getAnyActivePort();
    if (port == -1) {
      throw mscclpp::Error("No active port found", ErrorCode::InternalError);
    }
  }
  this->validateConfig(maxCqSize, maxCqPollNum, maxSendWr, maxRecvWr, maxWrPerSend, maxNumSgesPerWr, port);
  qps.emplace_back(new IbQp(this->ctx, this->pd, port, maxCqSize, maxCqPollNum, maxSendWr, maxRecvWr, maxWrPerSend,
                            maxNumSgesPerWr));
  return qps.back().get();
}

const IbMr* IbCtx::registerMr(void* buff, std::size_t size) {
  mrs.emplace_back(new IbMr(this->pd, buff, size));
  return mrs.back().get();
}

const std::string& IbCtx::getDevName() const { return this->devName; }

MSCCLPP_API_CPP int getIBDeviceCount() {
  int num;
  ibv_get_device_list(&num);
  return num;
}

MSCCLPP_API_CPP std::string getIBDeviceName(Transport ibTransport) {
  int num;
  struct ibv_device** devices = ibv_get_device_list(&num);
  int ibTransportIndex;
  switch (ibTransport) {  // TODO: get rid of this ugly switch
    case Transport::IB0:
      ibTransportIndex = 0;
      break;
    case Transport::IB1:
      ibTransportIndex = 1;
      break;
    case Transport::IB2:
      ibTransportIndex = 2;
      break;
    case Transport::IB3:
      ibTransportIndex = 3;
      break;
    case Transport::IB4:
      ibTransportIndex = 4;
      break;
    case Transport::IB5:
      ibTransportIndex = 5;
      break;
    case Transport::IB6:
      ibTransportIndex = 6;
      break;
    case Transport::IB7:
      ibTransportIndex = 7;
      break;
    default:
      throw std::invalid_argument("Not an IB transport");
  }
  if (ibTransportIndex >= num) {
    std::stringstream ss;
    ss << "IB transport out of range: " << ibTransportIndex << " >= " << num;
    throw std::out_of_range(ss.str());
  }
  return devices[ibTransportIndex]->name;
}

MSCCLPP_API_CPP Transport getIBTransportByDeviceName(const std::string& ibDeviceName) {
  int num;
  struct ibv_device** devices = ibv_get_device_list(&num);
  for (int i = 0; i < num; ++i) {
    if (ibDeviceName == devices[i]->name) {
      switch (i) {  // TODO: get rid of this ugly switch
        case 0:
          return Transport::IB0;
        case 1:
          return Transport::IB1;
        case 2:
          return Transport::IB2;
        case 3:
          return Transport::IB3;
        case 4:
          return Transport::IB4;
        case 5:
          return Transport::IB5;
        case 6:
          return Transport::IB6;
        case 7:
          return Transport::IB7;
        default:
          throw std::out_of_range("IB device index out of range");
      }
    }
  }
  throw std::invalid_argument("IB device not found");
}

}  // namespace mscclpp
