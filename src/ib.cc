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
#include <unordered_map>

#include "api.h"
#include "context.hpp"
#if defined(USE_IBVERBS)
#include "ibverbs_wrapper.hpp"
#endif  // defined(USE_IBVERBS)
#include "logger.hpp"

#if !defined(MSCCLPP_USE_ROCM)

// Check if nvidia_peermem kernel module is loaded
[[maybe_unused]] static bool checkNvPeerMemLoaded() {
  std::ifstream file("/proc/modules");
  std::string line;
  while (std::getline(file, line)) {
    if (line.find("nvidia_peermem") != std::string::npos) return true;
  }
  return false;
}

#endif  // !defined(MSCCLPP_USE_ROCM)

namespace mscclpp {

#if defined(USE_IBVERBS)

static inline bool isDmabufSupportedByGpu(int gpuId) {
  static std::unordered_map<int, bool> cache;
  if (gpuId < 0 || !IBVerbs::isDmabufSupported()) {
    return false;
  }
  if (cache.find(gpuId) != cache.end()) {
    return cache[gpuId];
  }
  int dmaBufSupported = 0;
#if !defined(MSCCLPP_USE_ROCM)
  CUdevice dev;
  MSCCLPP_CUTHROW(cuDeviceGet(&dev, gpuId));
  MSCCLPP_CUTHROW(cuDeviceGetAttribute(&dmaBufSupported, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, dev));
#endif  // !defined(MSCCLPP_USE_ROCM)
  bool ret = dmaBufSupported != 0;
  if (!ret) {
    DEBUG(NET, "GPU ", gpuId, " does not support DMABUF");
  }
  cache[gpuId] = ret;
  return ret;
}

IbMr::IbMr(ibv_pd* pd, void* buff, std::size_t size) : mr_(nullptr), buff_(buff), size_(0) {
  if (size == 0) {
    THROW(NET, Error, ErrorCode::InvalidUsage, "invalid MR size: 0");
  }
  static __thread uintptr_t pageSize = 0;
  if (pageSize == 0) {
    pageSize = sysconf(_SC_PAGESIZE);
  }
  uintptr_t buffIntPtr = reinterpret_cast<uintptr_t>(buff_);
  uintptr_t addr = buffIntPtr & -pageSize;
  std::size_t pages = (size + (buffIntPtr - addr) + pageSize - 1) / pageSize;

  int gpuId = detail::gpuIdFromAddress(buff_);
  bool isGpuBuff = (gpuId != -1);
  if (isGpuBuff && isDmabufSupportedByGpu(gpuId)) {
#if !defined(MSCCLPP_USE_ROCM)
    int fd;
    MSCCLPP_CUTHROW(cuMemGetHandleForAddressRange(&fd, addr, pages * pageSize, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0));

    size_t offsetInDmaBuf = buffIntPtr % pageSize;
    mr_ = IBVerbs::ibv_reg_dmabuf_mr(pd, offsetInDmaBuf, size, buffIntPtr, fd,
                                     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ |
                                         IBV_ACCESS_RELAXED_ORDERING | IBV_ACCESS_REMOTE_ATOMIC);
    ::close(fd);
    if (mr_ == nullptr) {
      THROW(NET, IbError, errno, "ibv_reg_dmabuf_mr failed (errno ", errno, ")");
    }
#else   // defined(MSCCLPP_USE_ROCM)
    THROW(NET, Error, ErrorCode::InvalidUsage, "We don't support DMABUF on HIP platforms yet");
#endif  // defined(MSCCLPP_USE_ROCM)
  } else {
#if !defined(MSCCLPP_USE_ROCM)
    if (isGpuBuff) {
      if (isCuMemMapAllocated(buff_)) {
        THROW(NET, Error, ErrorCode::InvalidUsage, "DMABUF is required but is not supported in this platform.");
      }
      // Need nvidia-peermem when DMABUF is not supported
      if (!checkNvPeerMemLoaded()) {
        THROW(NET, Error, ErrorCode::SystemError, "nvidia_peermem kernel module is not loaded");
      }
    }
#endif  // !defined(MSCCLPP_USE_ROCM)
    mr_ = IBVerbs::ibv_reg_mr(pd, reinterpret_cast<void*>(addr), pages * pageSize,
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

IbQp::IbQp(ibv_context* ctx, ibv_pd* pd, int portNum, int gidIndex, int maxCqSize, int maxCqPollNum, int maxSendWr,
           int maxRecvWr, int maxWrPerSend)
    : portNum_(portNum),
      gidIndex_(gidIndex),
      info_(),
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

  struct ibv_qp_init_attr qpInitAttr = {};
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
  if (IBVerbs::ibv_query_port(ctx, portNum_, &portAttr) != 0) {
    THROW(NET, IbError, errno, "ibv_query_port failed (errno ", errno, ")");
  }
  info_.lid = portAttr.lid;
  info_.linkLayer = portAttr.link_layer;
  info_.qpn = qp->qp_num;
  info_.mtu = portAttr.active_mtu;
  info_.is_grh = (portAttr.flags & IBV_QPF_GRH_REQUIRED);

  if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND || info_.is_grh) {
    if (gidIndex_ >= portAttr.gid_tbl_len) {
      THROW(NET, Error, ErrorCode::InvalidUsage, "invalid GID index ", gidIndex_, " for port ", portNum_,
            " (max index is ", portAttr.gid_tbl_len - 1, ")");
    }

    union ibv_gid gid = {};
    if (IBVerbs::ibv_query_gid(ctx, portNum_, gidIndex_, &gid) != 0) {
      THROW(NET, IbError, errno, "ibv_query_gid failed for port ", portNum_, " index ", gidIndex_, " (errno ", errno,
            ")");
    }
    info_.spn = gid.global.subnet_prefix;
    info_.iid = gid.global.interface_id;
  }

  struct ibv_qp_attr qpAttr = {};
  qpAttr.qp_state = IBV_QPS_INIT;
  qpAttr.pkey_index = 0;
  qpAttr.port_num = portNum_;
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
  struct ibv_qp_attr qp_attr = {};
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
    qp_attr.ah_attr.grh.sgid_index = gidIndex_;
    qp_attr.ah_attr.grh.hop_limit = 255;
    qp_attr.ah_attr.grh.traffic_class = 0;
  } else {
    qp_attr.ah_attr.is_global = 0;
  }
  qp_attr.ah_attr.dlid = info.lid;
  qp_attr.ah_attr.sl = 0;
  qp_attr.ah_attr.src_path_bits = 0;
  qp_attr.ah_attr.port_num = portNum_;
  int ret = IBVerbs::ibv_modify_qp(qp_, &qp_attr,
                                   IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                       IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
  if (ret != 0) {
    THROW(NET, IbError, errno, "ibv_modify_qp failed (errno ", errno, ")");
  }
}

void IbQp::rts() {
  struct ibv_qp_attr qp_attr = {};
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
  int err = IBVerbs::ibv_post_send(qp_, wrs_->data(), &bad_wr);
  if (err != 0) {
    THROW(NET, IbError, err, "ibv_post_send failed (errno ", err, ")");
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

std::string IbQp::getWcStatusString(int idx) const { return IBVerbs::ibv_wc_status_str((*wcs_)[idx].status); }

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

bool IbCtx::isPortUsable(int port, int gidIndex) const {
  struct ibv_port_attr portAttr = {};
  if (IBVerbs::ibv_query_port(ctx_, port, &portAttr) != 0) {
    THROW(NET, IbError, errno, "ibv_query_port failed (errno ", errno, ", port ", port, ")");
  }

  // Check if port is active and has a supported link layer
  if (portAttr.state != IBV_PORT_ACTIVE) {
    return false;
  }
  if (portAttr.link_layer != IBV_LINK_LAYER_ETHERNET && portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND) {
    return false;
  }

  if (gidIndex >= 0) {
    // For Ethernet/RoCE or InfiniBand with GRH, check if GID table has entries
    if (portAttr.link_layer == IBV_LINK_LAYER_ETHERNET || (portAttr.flags & IBV_QPF_GRH_REQUIRED)) {
      if (gidIndex >= portAttr.gid_tbl_len) {
        return false;
      }
      union ibv_gid gid = {};
      if (IBVerbs::ibv_query_gid(ctx_, port, gidIndex, &gid) != 0) {
        return false;
      }
    }
  }

  return true;
}

int IbCtx::getAnyUsablePort(int gidIndex) const {
  struct ibv_device_attr devAttr;
  if (IBVerbs::ibv_query_device(ctx_, &devAttr) != 0) {
    THROW(NET, IbError, errno, "ibv_query_device failed (errno ", errno, ")");
  }
  for (uint8_t port = 1; port <= devAttr.phys_port_cnt; ++port) {
    if (this->isPortUsable(port, gidIndex)) {
      return port;
    }
  }
  return -1;
}

std::shared_ptr<IbQp> IbCtx::createQp(int port, int gidIndex, int maxCqSize, int maxCqPollNum, int maxSendWr,
                                      int maxRecvWr, int maxWrPerSend) {
  if (port == -1) {
    port = this->getAnyUsablePort(gidIndex);
    if (port == -1) {
      THROW(NET, Error, ErrorCode::InvalidUsage, "No usable port found (device: ", devName_, ")");
    }
  } else if (!this->isPortUsable(port, gidIndex)) {
    THROW(NET, Error, ErrorCode::InvalidUsage, "invalid IB port: ", port);
  }
  return std::shared_ptr<IbQp>(
      new IbQp(ctx_, pd_, port, gidIndex, maxCqSize, maxCqPollNum, maxSendWr, maxRecvWr, maxWrPerSend));
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
