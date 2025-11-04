// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "ib.hpp"

#include <malloc.h>
#include <unistd.h>

#include <cstring>
#include <fstream>
#include <mscclpp/core.hpp>
#include <mscclpp/env.hpp>
#include <mscclpp/errors.hpp>
#include <mscclpp/fifo.hpp>
#include <sstream>
#include <string>

#include "api.h"
#include "context.hpp"
#if defined(USE_IBVERBS)
#include "ibverbs_wrapper.hpp"
#endif  // defined(USE_IBVERBS)
#include "logger.hpp"

#if !defined(__HIP_PLATFORM_AMD__)

// Check if nvidia_peermem kernel module is loaded
[[maybe_unused]] static bool checkNvPeerMemLoaded() {
  std::ifstream file("/proc/modules");
  std::string line;
  while (std::getline(file, line)) {
    if (line.find("nvidia_peermem") != std::string::npos) return true;
  }
  return false;
}

#endif  // !defined(__HIP_PLATFORM_AMD__)

namespace mscclpp {

#if defined(USE_IBVERBS)

IbMr::IbMr(ibv_pd* pd, void* buff, std::size_t size) : mr_(nullptr), buff_(buff), size_(0) {
  if (size == 0) {
    THROW(NET, Error, ErrorCode::InvalidUsage, "invalid MR size: 0");
  }
  static __thread uintptr_t pageSize = 0;
  if (pageSize == 0) {
    pageSize = sysconf(_SC_PAGESIZE);
  }
  uintptr_t addr = reinterpret_cast<uintptr_t>(buff_) & -pageSize;
  std::size_t pages = (size + (reinterpret_cast<uintptr_t>(buff_) - addr) + pageSize - 1) / pageSize;

  CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(buff_);
  bool cuMemAlloc = isCuMemMapAllocated((void*)dptr);
  int dmaBufSupported = 0;
#if !defined(__HIP_PLATFORM_AMD__)
  CUdevice dev;
  MSCCLPP_CUTHROW(cuCtxGetDevice(&dev));
  MSCCLPP_CUTHROW(cuDeviceGetAttribute(&dmaBufSupported, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, dev));
#endif  // !defined(__HIP_PLATFORM_AMD__)
  if (cuMemAlloc && dmaBufSupported) {
#if !defined(__HIP_PLATFORM_AMD__)
    int fd;
    MSCCLPP_CUTHROW(cuMemGetHandleForAddressRange(&fd, addr, pages * pageSize, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0));

    size_t offsetInDmaBuf = dptr % pageSize;
    mr_ = IBVerbs::ibv_reg_dmabuf_mr(pd, offsetInDmaBuf, size, (uint64_t)dptr, fd,
                                     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                                         IBV_ACCESS_RELAXED_ORDERING | IBV_ACCESS_REMOTE_ATOMIC);
    close(fd);
    if (mr_ == nullptr) {
      THROW(NET, IbError, errno, "ibv_reg_dmabuf_mr failed (errno ", errno, ")");
    }
#else
    THROW(NET, Error, ErrorCode::InvalidUsage, "Registration of DMA_BUF based memory region failed on HIP platform");
#endif  // !defined(__HIP_PLATFORM_AMD__)
  } else {
#if !defined(__HIP_PLATFORM_AMD__)
    // nvidia-peermem is needed only when DMA_BUF is not supported
    if (cuMemAlloc) {
      WARN(NET, "DMA_BUF is not supported; falling back to nvidia_peermem");
    }
    if (!checkNvPeerMemLoaded()) {
      THROW(NET, Error, ErrorCode::SystemError, "nvidia_peermem kernel module is not loaded");
    }
#endif  // !defined(__HIP_PLATFORM_AMD__)
    mr_ = IBVerbs::ibv_reg_mr2(pd, reinterpret_cast<void*>(addr), pages * pageSize,
                               IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                                   IBV_ACCESS_RELAXED_ORDERING | IBV_ACCESS_REMOTE_ATOMIC);
    if (mr_ == nullptr) {
      THROW(NET, IbError, errno, "ibv_reg_mr failed (errno ", errno, ")");
    }
  }

  size_ = pages * pageSize;
}

IbMr::~IbMr() { IBVerbs::ibv_dereg_mr(mr_); }

IbMrInfo IbMr::getInfo() const {
  IbMrInfo info;
  info.addr = reinterpret_cast<uint64_t>(buff_);
  info.rkey = mr_->rkey;
  return info;
}

const void* IbMr::getBuff() const { return buff_; }

uint32_t IbMr::getLkey() const { return mr_->lkey; }

IbQp::IbQp(ibv_context* ctx, ibv_pd* pd, int port, int maxCqSize, int maxCqPollNum, int maxSendWr, int maxRecvWr,
           int maxWrPerSend)
    : info_(),
      qp_(nullptr),
      cq_(nullptr),
      wcs_(),
      wrs_(),
      sges_(),
      wrn_(0),
      numSignaledPostedItems_(0),
      numSignaledStagedItems_(0),
      maxCqPollNum_(maxCqPollNum),
      maxWrPerSend_(maxWrPerSend) {
  cq_ = IBVerbs::ibv_create_cq(ctx, maxCqSize, nullptr, nullptr, 0);
  if (cq_ == nullptr) {
    THROW(NET, IbError, errno, "ibv_create_cq failed (errno ", errno, ")");
  }

  struct ibv_qp_init_attr qpInitAttr;
  std::memset(&qpInitAttr, 0, sizeof(qpInitAttr));
  qpInitAttr.sq_sig_all = 0;
  qpInitAttr.send_cq = cq_;
  qpInitAttr.recv_cq = cq_;
  qpInitAttr.qp_type = IBV_QPT_RC;
  qpInitAttr.cap.max_send_wr = maxSendWr;
  qpInitAttr.cap.max_recv_wr = maxRecvWr;
  qpInitAttr.cap.max_send_sge = 1;
  qpInitAttr.cap.max_recv_sge = 1;
  qpInitAttr.cap.max_inline_data = 0;

  struct ibv_qp* qp = IBVerbs::ibv_create_qp(pd, &qpInitAttr);
  if (qp == nullptr) {
    THROW(NET, IbError, errno, "ibv_create_qp failed (errno ", errno, ")");
  }

  struct ibv_port_attr portAttr;
  if (IBVerbs::ibv_query_port_w(ctx, port, &portAttr) != 0) {
    THROW(NET, IbError, errno, "ibv_query_port failed (errno ", errno, ")");
  }
  info_.lid = portAttr.lid;
  info_.port = port;
  info_.linkLayer = portAttr.link_layer;
  info_.qpn = qp->qp_num;
  info_.mtu = portAttr.active_mtu;
  info_.is_grh = (portAttr.flags & IBV_QPF_GRH_REQUIRED);

  if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND || info_.is_grh) {
    union ibv_gid gid;
    if (IBVerbs::ibv_query_gid(ctx, port, 0, &gid) != 0) {
      THROW(NET, IbError, errno, "ibv_query_gid failed (errno ", errno, ")");
    }
    info_.spn = gid.global.subnet_prefix;
    info_.iid = gid.global.interface_id;
  }

  struct ibv_qp_attr qpAttr;
  memset(&qpAttr, 0, sizeof(qpAttr));
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.pkey_index = 0;
  qpAttr.port_num = port;
  qpAttr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
  if (IBVerbs::ibv_modify_qp(qp, &qpAttr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS) != 0) {
    THROW(NET, IbError, errno, "ibv_modify_qp failed (errno ", errno, ")");
  }
  qp_ = qp;
  wrs_ = std::make_shared<std::vector<ibv_send_wr>>(maxWrPerSend_);
  sges_ = std::make_shared<std::vector<ibv_sge>>(maxWrPerSend_);
  wcs_ = std::make_shared<std::vector<ibv_wc>>(maxCqPollNum_);
}

IbQp::~IbQp() {
  IBVerbs::ibv_destroy_qp(qp_);
  IBVerbs::ibv_destroy_cq(cq_);
}

void IbQp::rtr(const IbQpInfo& info) {
  struct ibv_qp_attr qp_attr;
  std::memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
  qp_attr.qp_state = IBV_QPS_RTR;
  qp_attr.path_mtu = static_cast<ibv_mtu>(info.mtu);
  qp_attr.dest_qp_num = info.qpn;
  qp_attr.rq_psn = 0;
  qp_attr.max_dest_rd_atomic = 1;
  qp_attr.min_rnr_timer = 0x12;
  if (info.linkLayer == IBV_LINK_LAYER_ETHERNET || info.is_grh) {
    qp_attr.ah_attr.is_global = 1;
    qp_attr.ah_attr.grh.dgid.global.subnet_prefix = info.spn;
    qp_attr.ah_attr.grh.dgid.global.interface_id = info.iid;
    qp_attr.ah_attr.grh.flow_label = 0;
    qp_attr.ah_attr.grh.sgid_index = 0;
    qp_attr.ah_attr.grh.hop_limit = 255;
    qp_attr.ah_attr.grh.traffic_class = 0;
  } else {
    qp_attr.ah_attr.is_global = 0;
  }
  qp_attr.ah_attr.dlid = info.lid;
  qp_attr.ah_attr.sl = 0;
  qp_attr.ah_attr.src_path_bits = 0;
  qp_attr.ah_attr.port_num = info.port;
  int ret = IBVerbs::ibv_modify_qp(qp_, &qp_attr,
                                   IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                       IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
  if (ret != 0) {
    THROW(NET, IbError, errno, "ibv_modify_qp failed (errno ", errno, ")");
  }
}

void IbQp::rts() {
  struct ibv_qp_attr qp_attr;
  std::memset(&qp_attr, 0, sizeof(struct ibv_qp_attr));
  qp_attr.qp_state = IBV_QPS_RTS;
  qp_attr.timeout = 18;
  qp_attr.retry_cnt = 7;
  qp_attr.rnr_retry = 7;
  qp_attr.sq_psn = 0;
  qp_attr.max_rd_atomic = 1;
  int ret = IBVerbs::ibv_modify_qp(
      qp_, &qp_attr,
      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
  if (ret != 0) {
    THROW(NET, IbError, errno, "ibv_modify_qp failed (errno ", errno, ")");
  }
}

IbQp::WrInfo IbQp::getNewWrInfo() {
  if (wrn_ >= maxWrPerSend_) {
    THROW(NET, Error, ErrorCode::InvalidUsage, "too many outstanding work requests. limit is ", maxWrPerSend_);
  }
  ibv_send_wr* wr_ = &wrs_->data()[wrn_];
  ibv_sge* sge_ = &sges_->data()[wrn_];
  wr_->sg_list = sge_;
  wr_->num_sge = 1;
  wr_->next = nullptr;
  if (wrn_ > 0) {
    (*wrs_)[wrn_ - 1].next = wr_;
  }
  wrn_++;
  return IbQp::WrInfo{wr_, sge_};
}

void IbQp::stageSend(const IbMr* mr, const IbMrInfo& info, uint32_t size, uint64_t wrId, uint64_t srcOffset,
                     uint64_t dstOffset, bool signaled) {
  auto wrInfo = this->getNewWrInfo();
  wrInfo.wr->wr_id = wrId;
  wrInfo.wr->opcode = IBV_WR_RDMA_WRITE;
  wrInfo.wr->send_flags = signaled ? IBV_SEND_SIGNALED : 0;
  wrInfo.wr->wr.rdma.remote_addr = (uint64_t)(info.addr) + dstOffset;
  wrInfo.wr->wr.rdma.rkey = info.rkey;
  wrInfo.sge->addr = (uint64_t)(mr->getBuff()) + srcOffset;
  wrInfo.sge->length = size;
  wrInfo.sge->lkey = mr->getLkey();
  if (signaled) numSignaledStagedItems_++;
}

void IbQp::stageAtomicAdd(const IbMr* mr, const IbMrInfo& info, uint64_t wrId, uint64_t dstOffset, uint64_t addVal,
                          bool signaled) {
  auto wrInfo = this->getNewWrInfo();
  wrInfo.wr->wr_id = wrId;
  wrInfo.wr->opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
  wrInfo.wr->send_flags = signaled ? IBV_SEND_SIGNALED : 0;
  wrInfo.wr->wr.atomic.remote_addr = (uint64_t)(info.addr) + dstOffset;
  wrInfo.wr->wr.atomic.rkey = info.rkey;
  wrInfo.wr->wr.atomic.compare_add = addVal;
  wrInfo.sge->addr = (uint64_t)(mr->getBuff());
  wrInfo.sge->length = sizeof(uint64_t);  // atomic op is always on uint64_t
  wrInfo.sge->lkey = mr->getLkey();
  if (signaled) numSignaledStagedItems_++;
}

void IbQp::stageSendWithImm(const IbMr* mr, const IbMrInfo& info, uint32_t size, uint64_t wrId, uint64_t srcOffset,
                            uint64_t dstOffset, bool signaled, unsigned int immData) {
  auto wrInfo = this->getNewWrInfo();
  wrInfo.wr->wr_id = wrId;
  wrInfo.wr->opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wrInfo.wr->send_flags = signaled ? IBV_SEND_SIGNALED : 0;
  wrInfo.wr->wr.rdma.remote_addr = (uint64_t)(info.addr) + dstOffset;
  wrInfo.wr->wr.rdma.rkey = info.rkey;
  wrInfo.wr->imm_data = immData;
  wrInfo.sge->addr = (uint64_t)(mr->getBuff()) + srcOffset;
  wrInfo.sge->length = size;
  wrInfo.sge->lkey = mr->getLkey();
  if (signaled) numSignaledStagedItems_++;
}

void IbQp::postSend() {
  if (wrn_ == 0) {
    return;
  }
  struct ibv_send_wr* bad_wr;
  int ret = IBVerbs::ibv_post_send(qp_, wrs_->data(), &bad_wr);
  if (ret != 0) {
    THROW(NET, IbError, errno, "ibv_post_send failed (errno ", errno, ")");
  }
  wrn_ = 0;
  numSignaledPostedItems_ += numSignaledStagedItems_;
  numSignaledStagedItems_ = 0;
  if (numSignaledPostedItems_ + 4 > cq_->cqe) {
    WARN(NET, "IB: CQ is almost full ( ", numSignaledPostedItems_, " / ", cq_->cqe,
         " ). The connection needs to be flushed to prevent timeout errors.");
  }
}

int IbQp::pollCq() {
  int wcNum = IBVerbs::ibv_poll_cq(cq_, maxCqPollNum_, wcs_->data());
  if (wcNum > 0) {
    numSignaledPostedItems_ -= wcNum;
  }
  return wcNum;
}

int IbQp::getWcStatus(int idx) const { return (*wcs_)[idx].status; }

int IbQp::getNumCqItems() const { return numSignaledPostedItems_; }

IbCtx::IbCtx(const std::string& devName) : devName_(devName), ctx_(nullptr), pd_(nullptr) {
  int num;
  struct ibv_device** devices = IBVerbs::ibv_get_device_list(&num);
  for (int i = 0; i < num; ++i) {
    if (std::string(devices[i]->name) == devName_) {
      ctx_ = IBVerbs::ibv_open_device(devices[i]);
      break;
    }
  }
  IBVerbs::ibv_free_device_list(devices);
  if (ctx_ == nullptr) {
    THROW(NET, IbError, errno, "ibv_open_device failed (errno ", errno, ", device name ", devName_, ")");
  }
  pd_ = IBVerbs::ibv_alloc_pd(ctx_);
  if (pd_ == nullptr) {
    THROW(NET, IbError, errno, "ibv_alloc_pd failed (errno ", errno, ")");
  }
}

IbCtx::~IbCtx() {
  if (pd_ != nullptr) {
    IBVerbs::ibv_dealloc_pd(pd_);
  }
  if (ctx_ != nullptr) {
    IBVerbs::ibv_close_device(ctx_);
  }
}

bool IbCtx::isPortUsable(int port) const {
  struct ibv_port_attr portAttr;
  if (IBVerbs::ibv_query_port_w(ctx_, port, &portAttr) != 0) {
    THROW(NET, IbError, errno, "ibv_query_port failed (errno ", errno, ", port ", port, ")");
  }
  return portAttr.state == IBV_PORT_ACTIVE &&
         (portAttr.link_layer == IBV_LINK_LAYER_ETHERNET || portAttr.link_layer == IBV_LINK_LAYER_INFINIBAND);
}

int IbCtx::getAnyActivePort() const {
  struct ibv_device_attr devAttr;
  if (IBVerbs::ibv_query_device(ctx_, &devAttr) != 0) {
    THROW(NET, IbError, errno, "ibv_query_device failed (errno ", errno, ")");
  }
  for (uint8_t port = 1; port <= devAttr.phys_port_cnt; ++port) {
    if (this->isPortUsable(port)) {
      return port;
    }
  }
  return -1;
}

std::shared_ptr<IbQp> IbCtx::createQp(int maxCqSize, int maxCqPollNum, int maxSendWr, int maxRecvWr, int maxWrPerSend,
                                      int port /*=-1*/) {
  if (port == -1) {
    port = this->getAnyActivePort();
    if (port == -1) {
      THROW(NET, Error, ErrorCode::InvalidUsage, "No active port found");
    }
  } else if (!this->isPortUsable(port)) {
    THROW(NET, Error, ErrorCode::InvalidUsage, "invalid IB port: ", port);
  }
  return std::shared_ptr<IbQp>(new IbQp(ctx_, pd_, port, maxCqSize, maxCqPollNum, maxSendWr, maxRecvWr, maxWrPerSend));
}

std::unique_ptr<const IbMr> IbCtx::registerMr(void* buff, std::size_t size) {
  return std::unique_ptr<const IbMr>(new IbMr(pd_, buff, size));
}

MSCCLPP_API_CPP int getIBDeviceCount() {
  int num;
  IBVerbs::ibv_get_device_list(&num);
  return num;
}

std::string getHcaDevices(int deviceIndex) {
  std::string envStr = env()->hcaDevices;
  if (envStr != "") {
    std::vector<std::string> devices;
    std::stringstream ss(envStr);
    std::string device;
    while (std::getline(ss, device, ',')) {
      devices.push_back(device);
    }
    if (deviceIndex >= (int)devices.size()) {
      THROW(NET, Error, ErrorCode::InvalidUsage,
            "Not enough HCA devices are defined with MSCCLPP_HCA_DEVICES: ", envStr);
    }
    return devices[deviceIndex];
  }
  return "";
}

MSCCLPP_API_CPP std::string getIBDeviceName(Transport ibTransport) {
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
      THROW(NET, Error, ErrorCode::InvalidUsage, "Not an IB transport");
  }
  std::string userHcaDevice = getHcaDevices(ibTransportIndex);
  if (!userHcaDevice.empty()) {
    return userHcaDevice;
  }

  int num;
  struct ibv_device** devices = IBVerbs::ibv_get_device_list(&num);
  if (ibTransportIndex >= num) {
    THROW(NET, Error, ErrorCode::InvalidUsage, "IB transport out of range: ", ibTransportIndex, " >= ", num);
  }
  return devices[ibTransportIndex]->name;
}

MSCCLPP_API_CPP Transport getIBTransportByDeviceName(const std::string& ibDeviceName) {
  int num;
  struct ibv_device** devices = IBVerbs::ibv_get_device_list(&num);
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
          THROW(NET, Error, ErrorCode::InvalidUsage, "IB device index out of range");
      }
    }
  }
  THROW(NET, Error, ErrorCode::InvalidUsage, "IB device not found");
}

#else  // !defined(USE_IBVERBS)

MSCCLPP_API_CPP int getIBDeviceCount() { return 0; }

MSCCLPP_API_CPP std::string getIBDeviceName(Transport) { return ""; }

MSCCLPP_API_CPP Transport getIBTransportByDeviceName(const std::string&) { return Transport::Unknown; }

#endif  // !defined(USE_IBVERBS)

}  // namespace mscclpp
