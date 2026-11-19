// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "mrc.hpp"

#if defined(MSCCLPP_USE_MRC)

#include <arpa/inet.h>
#include <mrc.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <mscclpp/core.hpp>
#include <mscclpp/env.hpp>
#include <mscclpp/errors.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <sstream>

#include "logger.hpp"
#include "serialization.hpp"
#include "utils_internal.hpp"

#if defined(MSCCLPP_USE_CUDA)
#include <filesystem>
#include <fstream>
#endif

#if defined(MSCCLPP_USE_MLX5DV)
#include "mlx5dv_wrapper.hpp"
#endif

namespace mscclpp {

static_assert(MRC_API_CURRENT_VERSION == MRC_API_VER(1, 0, 0));
static_assert(MRC_API_VER_USED == MRC_API_VER(1, 0, 0));

std::string getMrcDeviceName(Transport transport) {
  if (!AllMrcTransports.has(transport)) {
    throw Error("Not a native MRC transport", ErrorCode::InvalidUsage);
  }
  int index = static_cast<int>(transport) - static_cast<int>(Transport::MRC0);
  if (!env()->hcaDevices.empty()) {
    std::istringstream names(env()->hcaDevices);
    std::string name;
    for (int i = 0; i <= index; ++i) {
      if (!std::getline(names, name, ',')) {
        throw Error("Not enough HCA devices in MSCCLPP_HCA_DEVICES for native MRC", ErrorCode::InvalidUsage);
      }
    }
    if (!name.empty()) return name;
  }
  int count = 0;
  std::unique_ptr<ibv_device*, decltype(&ibv_free_device_list)> devices(ibv_get_device_list(&count),
                                                                        ibv_free_device_list);
  if (!devices) throw MrcError("ibv_get_device_list failed for native MRC selection", errno);
  if (index >= count) {
    throw Error("Native MRC device index out of range: " + std::to_string(index), ErrorCode::InvalidUsage);
  }
  return ibv_get_device_name(devices.get()[index]);
}

#if defined(MSCCLPP_USE_CUDA)
static void checkGpuDmaTopology(CUdevice device, const std::string& dmaDevicePath) {
  char busId[32]{};
  MSCCLPP_CUTHROW(cuDeviceGetPCIBusId(busId, sizeof(busId), device));
  std::error_code error;
  auto gpuPath = std::filesystem::canonical(std::string("/sys/bus/pci/devices/") + busId, error);
  if (error) {
    throw Error("Unable to resolve GPU PCIe path for native MRC: " + error.message(), ErrorCode::SystemError);
  }
  auto dmaPath = std::filesystem::canonical(dmaDevicePath, error);
  if (error) {
    throw Error("Unable to resolve DMA device PCIe path for native MRC: " + error.message(), ErrorCode::SystemError);
  }
  std::filesystem::path common;
  for (auto gpu = gpuPath.begin(), dma = dmaPath.begin(); gpu != gpuPath.end() && dma != dmaPath.end() && *gpu == *dma;
       ++gpu, ++dma) {
    common /= *gpu;
  }
  // Require a shared PCI bridge, not merely a common sysfs or PCI-domain directory.
  std::ifstream pciClass(common / "class");
  unsigned int classCode = 0;
  if (!(pciClass >> std::hex >> classCode) || (classCode >> 8) != 0x0604) {
    throw Error("Native MRC GPU " + std::string(busId) + " and DMA device " + dmaPath.string() +
                    " do not share a PCIe bridge; select a compatible GPU/NIC pair",
                ErrorCode::InvalidUsage);
  }
}
#endif

MrcMr::MrcMr(ibv_pd* pd, void* buff, std::size_t size, [[maybe_unused]] const std::string& dataDirectPath)
    : mr_(nullptr), buff_(buff), size_(size) {
  if (buff == nullptr || size == 0) {
    throw Error("Native MRC requires a nonnull MR buffer and positive size", ErrorCode::InvalidUsage);
  }
  constexpr int accessFlags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;

  long pageSize = sysconf(_SC_PAGESIZE);
  if (pageSize <= 0) {
    throw Error("Unable to query page size for native MRC registration", ErrorCode::SystemError);
  }
  uintptr_t bufferAddress = reinterpret_cast<uintptr_t>(buff);
  std::size_t offset = bufferAddress % pageSize;
  if (size > std::numeric_limits<uintptr_t>::max() - bufferAddress ||
      size > std::numeric_limits<std::size_t>::max() - offset - (pageSize - 1)) {
    throw Error("Native MRC registration range is too large", ErrorCode::InvalidUsage);
  }
  uintptr_t address = bufferAddress - offset;
  std::size_t rangeSize = ((size + offset + pageSize - 1) / pageSize) * pageSize;

  int gpuId = -1;
#if defined(MSCCLPP_USE_CUDA)
  CUmemorytype memoryType{};
  CUresult result =
      cuPointerGetAttribute(&memoryType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, reinterpret_cast<CUdeviceptr>(buff));
  if (result != CUDA_ERROR_NOT_INITIALIZED && result != CUDA_ERROR_INVALID_VALUE) {
    MSCCLPP_CUTHROW(result);
    if (memoryType == CU_MEMORYTYPE_DEVICE || memoryType == CU_MEMORYTYPE_UNIFIED) {
      gpuId = detail::gpuIdFromAddress(buff);
    }
  }
#else
  gpuId = detail::gpuIdFromAddress(buff);
#endif
  CudaDeviceGuard deviceGuard(gpuId);

#if defined(MSCCLPP_USE_CUDA)
  if (gpuId >= 0) {
    CUdevice device;
    MSCCLPP_CUTHROW(cuDeviceGet(&device, gpuId));
    int dmaBufSupported = 0;
    MSCCLPP_CUTHROW(cuDeviceGetAttribute(&dmaBufSupported, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, device));
    std::string nicPath = std::string("/sys/class/infiniband/") + ibv_get_device_name(pd->context->device) + "/device";
    if (dmaBufSupported) {
      bool useDataDirect = !dataDirectPath.empty();
      checkGpuDmaTopology(device, useDataDirect ? dataDirectPath : nicPath);
      unsigned long long exportFlags = 0;
      if (useDataDirect) {
#if CUDA_VERSION >= 12080
        exportFlags = CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE;
#else
        throw Error("Native MRC Data Direct registration requires CUDA 12.8 or newer", ErrorCode::InvalidUsage);
#endif
      }
      int fd = -1;
      auto closeFd = [](int* descriptor) {
        if (*descriptor >= 0) ::close(*descriptor);
      };
      std::unique_ptr<int, decltype(closeFd)> fdGuard(&fd, closeFd);
      MSCCLPP_CUTHROW(
          cuMemGetHandleForAddressRange(&fd, address, rangeSize, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, exportFlags));
      if (fd < 0) {
        throw Error("GPU DMA-BUF export returned an invalid descriptor", ErrorCode::SystemError);
      }

#if defined(MSCCLPP_USE_MLX5DV)
      if (useDataDirect) {
        // Data Direct requires an aligned MR; retain the logical buffer address in buff_.
        mr_ = MLX5DV::mlx5dv_reg_dmabuf_mr(pd, 0, rangeSize, address, fd, accessFlags);
      } else
#endif
      {
        mr_ = ibv_reg_dmabuf_mr(pd, offset, size, bufferAddress, fd, accessFlags);
      }
      if (mr_ == nullptr) {
        throw MrcError("DMA-BUF registration failed for native MRC", errno);
      }
      return;
    }
    if (isCuMemMapAllocated(buff)) {
      throw Error("Native MRC VMM registration requires DMA-BUF support", ErrorCode::InvalidUsage);
    }
    checkGpuDmaTopology(device, nicPath);
    if (::access("/sys/module/nvidia_peermem", F_OK) != 0) {
      throw Error("nvidia_peermem kernel module is not loaded", ErrorCode::SystemError);
    }
  }
#endif

  mr_ = ibv_reg_mr(pd, reinterpret_cast<void*>(address), rangeSize, accessFlags);
  if (mr_ == nullptr) {
    throw MrcError("ibv_reg_mr failed for native MRC", errno);
  }
}

MrcMr::~MrcMr() { ibv_dereg_mr(mr_); }

MrcBuffer MrcMr::getBuffer() const { return {{reinterpret_cast<uint64_t>(buff_), mr_->rkey}, size_}; }

MrcQp::MrcQp(ibv_context* context, mrc_context* mrcContext, ibv_pd* pd, int port, int gidIndex, uint64_t evProfileId,
             int maxSendCqSize, int maxSendWr, int maxRecvWr)
    : pd_(pd),
      portNum_(port),
      gidIndex_(gidIndex),
      info_{},
      sendCq_(nullptr),
      recvCq_(nullptr),
      qp_(nullptr),
      numStagedSend_(0),
      numStagedSignaledSend_(0),
      numSendWcs_(0),
      numStagedRecv_(0),
      maxSendCqSize_(maxSendCqSize),
      maxSendWr_(maxSendWr),
      maxRecvWr_(maxRecvWr) {
  if (maxSendCqSize <= 0 || maxSendWr <= 0 || maxRecvWr < 0) {
    throw Error("Native MRC requires positive send CQ/WR limits and a nonnegative receive WR limit",
                ErrorCode::InvalidUsage);
  }
  if (port <= 0 || port > 255) {
    throw Error("Native MRC requires a port between 1 and 255", ErrorCode::InvalidUsage);
  }

  sendWcs_ = std::make_unique<ibv_wc[]>(maxSendCqSize_);
  sendWrs_ = std::make_unique<ibv_send_wr[]>(maxSendWr_);
  sendSges_ = std::make_unique<ibv_sge[]>(maxSendWr_);

  std::unique_ptr<mrc_cq, decltype(&mrc_destroy_cq)> sendCq(
      mrc_create_cq(mrcContext, maxSendCqSize, nullptr, nullptr, 0), mrc_destroy_cq);
  if (!sendCq) {
    throw MrcError("mrc_create_cq failed for send CQ", errno);
  }

  std::unique_ptr<mrc_cq, decltype(&mrc_destroy_cq)> recvCq(nullptr, mrc_destroy_cq);
  if (maxRecvWr > 0) {
    recvWcs_ = std::make_unique<ibv_wc[]>(maxRecvWr_);
    recvWrs_ = std::make_unique<ibv_recv_wr[]>(maxRecvWr_);
    recvCq.reset(mrc_create_cq(mrcContext, maxRecvWr, nullptr, nullptr, 0));
    if (!recvCq) {
      throw MrcError("mrc_create_cq failed for receive CQ", errno);
    }
  }

  mrc_qp_init_attr attrs{};
  attrs.send_cq = sendCq.get();
  attrs.recv_cq = recvCq ? recvCq.get() : sendCq.get();
  attrs.cap.max_send_wr = maxSendWr;
  attrs.cap.max_recv_wr = maxRecvWr;
  attrs.cap.max_send_sge = 1;
  attrs.cap.max_recv_sge = 1;
  attrs.pd = pd;
  std::unique_ptr<mrc_qp, decltype(&mrc_destroy_qp)> qp(mrc_create_qp(mrcContext, &attrs), mrc_destroy_qp);
  if (!qp) {
    throw MrcError("mrc_create_qp failed", errno);
  }

  ibv_port_attr portAttr{};
  int ret = ibv_query_port(context, portNum_, &portAttr);
  if (ret != 0) {
    throw MrcError("ibv_query_port failed for native MRC QP", ret > 0 ? ret : errno);
  }
  info_.lid = portAttr.lid;
  info_.linkLayer = portAttr.link_layer;
  info_.mtu = portAttr.active_mtu;
  info_.isGrh = (portAttr.flags & IBV_QPF_GRH_REQUIRED);

  ret = mrc_get_qpn(qp.get(), &info_.qpn);
  if (ret != 0) {
    throw MrcError("mrc_get_qpn failed", ret > 0 ? ret : errno);
  }

  if (portAttr.link_layer != IBV_LINK_LAYER_INFINIBAND || info_.isGrh) {
    if (gidIndex_ < 0 || gidIndex_ >= portAttr.gid_tbl_len) {
      throw Error("invalid native MRC GID index " + std::to_string(gidIndex_) + " for port " +
                      std::to_string(portNum_) + " (GID table length " + std::to_string(portAttr.gid_tbl_len) + ")",
                  ErrorCode::InvalidUsage);
    }
    ibv_gid gid{};
    ret = ibv_query_gid(context, portNum_, gidIndex_, &gid);
    if (ret != 0) {
      throw MrcError("ibv_query_gid failed for native MRC QP", ret > 0 ? ret : errno);
    }
    info_.spn = gid.global.subnet_prefix;
    info_.iid = gid.global.interface_id;
  }

  ibv_qp_attr qpAttrs{};
  qpAttrs.qp_state = IBV_QPS_INIT;
  qpAttrs.port_num = portNum_;
  qpAttrs.pkey_index = 0;
  qpAttrs.qp_access_flags = IBV_ACCESS_REMOTE_WRITE;

  mrc_qp_attr mrcAttrs{};
  mrcAttrs.ev_profile_id = evProfileId;
  ret = mrc_modify_qp(qp.get(), &qpAttrs, IBV_QP_STATE | IBV_QP_PORT | IBV_QP_PKEY_INDEX | IBV_QP_ACCESS_FLAGS,
                      &mrcAttrs, MRC_QP_EV_PROFILE_ID);
  if (ret != 0) {
    throw MrcError("mrc_modify_qp failed transitioning to INIT", ret > 0 ? ret : errno);
  }

  sendCq_ = sendCq.release();
  recvCq_ = recvCq.release();
  qp_ = qp.release();
}

MrcQp::~MrcQp() {
  mrc_destroy_qp(qp_);
  mrc_destroy_cq(sendCq_);
  if (recvCq_ != nullptr) {
    mrc_destroy_cq(recvCq_);
  }
}

void MrcQp::rtr(const MrcQpInfo& info) {
  ibv_qp_attr qpAttrs{};
  qpAttrs.qp_state = IBV_QPS_RTR;
  qpAttrs.path_mtu = static_cast<ibv_mtu>(info.mtu);
  qpAttrs.dest_qp_num = info.qpn;
  qpAttrs.rq_psn = 0;
  if (info.linkLayer == IBV_LINK_LAYER_ETHERNET || info.isGrh) {
    qpAttrs.ah_attr.is_global = 1;
    qpAttrs.ah_attr.grh.dgid.global.subnet_prefix = info.spn;
    qpAttrs.ah_attr.grh.dgid.global.interface_id = info.iid;
    qpAttrs.ah_attr.grh.sgid_index = gidIndex_;
    qpAttrs.ah_attr.grh.hop_limit = 255;
    // Match the reference program's traffic class on the installed provider.
    qpAttrs.ah_attr.grh.traffic_class = 1;
  }
  qpAttrs.ah_attr.dlid = info.lid;
  qpAttrs.ah_attr.port_num = portNum_;

  mrc_qp_attr mrcAttrs{};
  mrcAttrs.version = MRC_VERSION_1;
  int ret = mrc_modify_qp(qp_, &qpAttrs, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN,
                          &mrcAttrs, MRC_QP_PROTOCOL_VERSION);
  if (ret != 0) {
    throw MrcError("mrc_modify_qp failed transitioning to RTR", ret > 0 ? ret : errno);
  }
}

void MrcQp::rts() {
  ibv_qp_attr qpAttrs{};
  qpAttrs.qp_state = IBV_QPS_RTS;
  qpAttrs.sq_psn = 0;
  // The installed provider accepts these verbs retry fields despite the SDK's unsupported-fields table.
  // FIXME: should this be set dynamically based on some environment variable?
  qpAttrs.retry_cnt = 7;
  qpAttrs.rnr_retry = 7;
  mrc_qp_attr mrcAttrs{};
  mrcAttrs.timeout = 14;
  int ret = mrc_modify_qp(qp_, &qpAttrs, IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY, &mrcAttrs,
                          MRC_QP_TIMEOUT);
  if (ret != 0) {
    throw MrcError("mrc_modify_qp failed transitioning to RTS", ret > 0 ? ret : errno);
  }
}

void MrcQp::stageSendWrite(const MrcMr* mr, const MrcBuffer& dst, uint64_t size, uint64_t wrId, uint64_t srcOffset,
                           uint64_t dstOffset, bool signaled) {
  if (mr == nullptr) {
    throw Error("Native MRC WRITE requires an MR", ErrorCode::InvalidUsage);
  }
  stageSend(mr, dst, size, wrId, srcOffset, dstOffset, signaled);
}

void MrcQp::stageSendWriteWithImm(const MrcMr* mr, const MrcBuffer& dst, uint64_t size, uint64_t wrId,
                                  uint64_t srcOffset, uint64_t dstOffset, bool signaled, unsigned int immData) {
  // TODO: Define how the Connection handles zero peer receive credits during one-way traffic.
  // A local send completion does not mean the peer has reposted a receive.
  auto& wr = stageSend(mr, dst, size, wrId, srcOffset, dstOffset, signaled);
  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.imm_data = htonl(immData);
}

static uint64_t checkedWriteAddress(const MrcBuffer& buffer, uint64_t offset, uint64_t size) {
  if (buffer.info.addr == 0 || buffer.size == 0 ||
      buffer.size > std::numeric_limits<uint64_t>::max() - buffer.info.addr || offset > buffer.size ||
      size > buffer.size - offset) {
    throw Error("Native MRC WRITE outside logical memory range", ErrorCode::InvalidUsage);
  }
  return buffer.info.addr + offset;
}

ibv_send_wr& MrcQp::stageSend(const MrcMr* mr, const MrcBuffer& dst, uint64_t size, uint64_t wrId, uint64_t srcOffset,
                              uint64_t dstOffset, bool signaled) {
  const int pending = sendProgress_.pendingSignaledSends();
  if (mr == nullptr && size != 0) {
    throw Error("Native MRC WRITE requires an MR for nonzero size", ErrorCode::InvalidUsage);
  }
  // On an all-signaled QP, pending accounts for every posted WR not yet polled.
  if (numStagedSend_ >= maxSendWr_ - pending) {
    throw Error("Native MRC send queue is full; post staged requests and poll send completions",
                ErrorCode::InvalidUsage);
  }
  if (signaled && numStagedSignaledSend_ >= maxSendCqSize_ - pending) {
    throw Error("Native MRC send CQ is full; poll completions before staging another signal", ErrorCode::InvalidUsage);
  }
  if (size > std::numeric_limits<uint32_t>::max()) {
    throw Error("Native MRC WRITE exceeds the maximum SGE length", ErrorCode::InvalidUsage);
  }
  if (mr && mr->mr_->pd != pd_) {
    throw Error("Native MRC WRITE source and QP belong to different protection domains", ErrorCode::InvalidUsage);
  }
  const uint64_t dstAddress = checkedWriteAddress(dst, dstOffset, size);
  const uint64_t srcAddress = mr ? checkedWriteAddress(mr->getBuffer(), srcOffset, size) : 0;
  auto& wr = sendWrs_[numStagedSend_];
  auto& sge = sendSges_[numStagedSend_];
  wr = {};
  if (mr != nullptr) {
    sge.addr = srcAddress;
    sge.length = static_cast<uint32_t>(size);
    sge.lkey = mr->mr_->lkey;
  }
  wr.wr_id = wrId;
  wr.sg_list = mr != nullptr ? &sge : nullptr;
  wr.num_sge = mr != nullptr ? 1 : 0;
  wr.opcode = IBV_WR_RDMA_WRITE;
  wr.send_flags = signaled ? IBV_SEND_SIGNALED : 0;
  wr.wr.rdma.remote_addr = dstAddress;
  wr.wr.rdma.rkey = dst.info.rkey;
  if (numStagedSend_ > 0) {
    sendWrs_[numStagedSend_ - 1].next = &wr;
  }
  ++numStagedSend_;
  if (signaled) ++numStagedSignaledSend_;
  return wr;
}

void MrcQp::postSend() {
  sendProgress_.check();
  if (numStagedSend_ == 0) return;
  ibv_send_wr* badWr = nullptr;
  int ret = mrc_post_send(qp_, sendWrs_.get(), &badWr);
  // A failed post may have submitted a prefix: never replay it or report it drained.
  sendProgress_.posted(ret, numStagedSend_, numStagedSignaledSend_);
  numStagedSend_ = 0;
  numStagedSignaledSend_ = 0;
}

void MrcQp::stageRecv(uint64_t wrId) {
  if (numStagedRecv_ >= maxRecvWr_ - recvProgress_.pendingRecvs()) {
    throw Error("Native MRC receive queue is full or disabled; poll completions before staging another receive",
                ErrorCode::InvalidUsage);
  }
  auto& wr = recvWrs_[numStagedRecv_];
  wr = {};
  wr.wr_id = wrId;
  if (numStagedRecv_ > 0) {
    recvWrs_[numStagedRecv_ - 1].next = &wr;
  }
  ++numStagedRecv_;
}

void MrcQp::postRecv() {
  recvProgress_.check();
  if (numStagedRecv_ == 0) return;
  ibv_recv_wr* badWr = nullptr;
  int ret = mrc_post_recv(qp_, recvWrs_.get(), &badWr);
  // A failed post may have accepted a prefix. Never replay that batch.
  recvProgress_.posted(ret, numStagedRecv_);
  numStagedRecv_ = 0;
}

void MrcSendProgress::check() const {
  if (failure_) std::rethrow_exception(failure_);
}

int MrcSendProgress::pendingSignaledSends() const {
  check();
  return pending_;
}

bool MrcSendProgress::drained() const {
  check();
  if (hasUnsignaled_) {
    throw Error("Cannot verify a native MRC send drain after an unsignaled send", ErrorCode::InvalidUsage);
  }
  return pending_ == 0;
}

void MrcSendProgress::posted(int result, int staged, int signaled) try {
  check();
  if (result != 0) throw MrcError("mrc_post_send failed", result > 0 ? result : errno);
  pending_ += signaled;
  if (staged != signaled) hasUnsignaled_ = true;
} catch (...) {
  failure_ = std::current_exception();
  throw;
}

void MrcSendProgress::polled(int count, const ibv_wc* wcs) try {
  check();
  if (count < 0) throw MrcError("mrc_poll_cq failed for native MRC send CQ", errno);
  for (int i = 0; i < count; ++i) {
    // An unsignaled send can also produce an error CQE. Never credit it to a signaled send.
    if (wcs[i].status != IBV_WC_SUCCESS) {
      throw Error("Native MRC send WR " + std::to_string(wcs[i].wr_id) + " failed: " + ibv_wc_status_str(wcs[i].status),
                  ErrorCode::SystemError);
    }
  }
  if (count > pending_) {
    throw Error("Unexpected native MRC send completion count", ErrorCode::InternalError);
  }
  pending_ -= count;
} catch (...) {
  failure_ = std::current_exception();
  throw;
}

int MrcQp::pollSendCq() {
  sendProgress_.check();
  int count = mrc_poll_cq(sendCq_, maxSendCqSize_, sendWcs_.get());
  sendProgress_.polled(count, sendWcs_.get());
  numSendWcs_ = count;
  return count;
}

bool MrcQp::progressSend() {
  if (sendProgress_.drained()) return numStagedSend_ == 0;
  pollSendCq();
  return numStagedSend_ == 0 && sendProgress_.drained();
}

void MrcRecvProgress::check() const {
  if (failure_) std::rethrow_exception(failure_);
}

int MrcRecvProgress::pendingRecvs() const {
  check();
  return pending_;
}

void MrcRecvProgress::posted(int result, int staged) try {
  check();
  if (result != 0) throw MrcError("mrc_post_recv failed", result > 0 ? result : errno);
  pending_ += staged;
} catch (...) {
  failure_ = std::current_exception();
  throw;
}

void MrcRecvProgress::polled(int count, const ibv_wc* wcs) try {
  check();
  if (count < 0) throw MrcError("mrc_poll_cq failed for native MRC receive CQ", errno);
  if (count > pending_) throw Error("Unexpected native MRC receive completion count", ErrorCode::InternalError);
  // Validate the entire batch before making any notification visible.
  for (int i = 0; i < count; ++i) {
    if (wcs[i].status != IBV_WC_SUCCESS) {
      throw Error(
          "Native MRC receive WR " + std::to_string(wcs[i].wr_id) + " failed: " + ibv_wc_status_str(wcs[i].status),
          ErrorCode::SystemError);
    }
    if (!(wcs[i].wc_flags & IBV_WC_WITH_IMM)) {
      throw Error("Native MRC receive WR " + std::to_string(wcs[i].wr_id) + " has no immediate data",
                  ErrorCode::SystemError);
    }
  }
  numWcs_ = count;
  pending_ -= count;
} catch (...) {
  failure_ = std::current_exception();
  throw;
}

const ibv_wc& MrcRecvProgress::completion(int idx, const ibv_wc* wcs) const {
  check();
  if (idx < 0 || idx >= numWcs_) throw Error("Invalid native MRC receive completion index", ErrorCode::InvalidUsage);
  return wcs[idx];
}

int MrcQp::pollRecvCq() {
  recvProgress_.check();
  if (recvCq_ == nullptr) return 0;
  int count = mrc_poll_cq(recvCq_, maxRecvWr_, recvWcs_.get());
  recvProgress_.polled(count, recvWcs_.get());
  return count;
}

int MrcQp::getSendWcStatus(int idx) const {
  if (idx < 0 || idx >= numSendWcs_) {
    throw Error("Invalid native MRC send completion index", ErrorCode::InvalidUsage);
  }
  return sendWcs_[idx].status;
}

std::string MrcQp::getSendWcStatusString(int idx) const {
  return ibv_wc_status_str(static_cast<ibv_wc_status>(getSendWcStatus(idx)));
}

int MrcQp::getNumSendCqItems() const { return sendProgress_.pendingSignaledSends(); }

int MrcQp::getRecvWcStatus(int idx) const { return recvProgress_.completion(idx, recvWcs_.get()).status; }

std::string MrcQp::getRecvWcStatusString(int idx) const {
  return ibv_wc_status_str(static_cast<ibv_wc_status>(getRecvWcStatus(idx)));
}

unsigned int MrcQp::getRecvWcImmData(int idx) const {
  return ntohl(recvProgress_.completion(idx, recvWcs_.get()).imm_data);
}

MrcCtx::MrcCtx(const std::string& devName) : devName_(devName), context_(nullptr), pd_(nullptr), mrcContext_(nullptr) {
  if (devName.empty()) {
    throw Error("Native MRC requires a device name", ErrorCode::InvalidUsage);
  }

  int numDevices = 0;
  std::unique_ptr<ibv_device*, decltype(&ibv_free_device_list)> devices(ibv_get_device_list(&numDevices),
                                                                        ibv_free_device_list);
  if (!devices) {
    throw MrcError("ibv_get_device_list failed", errno);
  }
  std::unique_ptr<ibv_context, decltype(&ibv_close_device)> context(nullptr, ibv_close_device);
  for (int i = 0; i < numDevices; ++i) {
    if (devName == ibv_get_device_name(devices.get()[i])) {
      context.reset(ibv_open_device(devices.get()[i]));
      if (!context) {
        throw MrcError("ibv_open_device failed for " + devName, errno);
      }
      break;
    }
  }
  if (!context) {
    throw Error("Native MRC device not found: " + devName, ErrorCode::InvalidUsage);
  }

  // TODO: Use mrc_query_device_ex and its attribute layout when adding newer SDK support.
  mrc_attr capabilities{};
  int supported = 0;
  if (mrc_query_device(context.get(), &capabilities, &supported) != 0) {
    throw MrcError("mrc_query_device failed for " + devName, errno);
  }
  if (!supported) {
    throw Error("Device does not support native MRC: " + devName, ErrorCode::InvalidUsage);
  }

#if defined(MSCCLPP_USE_MLX5DV)
  if (MLX5DV::isAvailable() && MLX5DV::mlx5dv_is_supported(context->device)) {
    char sysfsPath[256];
    if (MLX5DV::mlx5dv_get_data_direct_sysfs_path(context.get(), sysfsPath, sizeof(sysfsPath)) == 0) {
      dataDirectPath_ = std::string("/sys") + sysfsPath;
    }
  }
#endif

  std::unique_ptr<ibv_pd, decltype(&ibv_dealloc_pd)> pd(ibv_alloc_pd(context.get()), ibv_dealloc_pd);
  if (!pd) {
    throw MrcError("ibv_alloc_pd failed for " + devName, errno);
  }

  // TODO: Add runtime compatibility checks with mrc_query_version when the SDK provides it.
  mrc_context_attr attrs{};
  attrs.mrc_api_version_used = MRC_API_VER(1, 0, 0);
  std::unique_ptr<mrc_context, decltype(&mrc_destroy_context)> mrcContext(mrc_create_context(context.get(), &attrs),
                                                                          mrc_destroy_context);
  if (!mrcContext) {
    throw MrcError("mrc_create_context failed for " + devName, errno);
  }

  context_ = context.release();
  pd_ = pd.release();
  mrcContext_ = mrcContext.release();
}

MrcCtx::~MrcCtx() {
  mrc_destroy_context(mrcContext_);
  ibv_dealloc_pd(pd_);
  ibv_close_device(context_);
}

std::shared_ptr<MrcQp> MrcCtx::createQp(int port, int gidIndex, uint64_t evProfileId, int maxSendCqSize, int maxSendWr,
                                        int maxRecvWr) {
  return std::shared_ptr<MrcQp>(
      new MrcQp(context_, mrcContext_, pd_, port, gidIndex, evProfileId, maxSendCqSize, maxSendWr, maxRecvWr));
}

std::unique_ptr<const MrcMr> MrcCtx::registerMr(void* buff, std::size_t size) {
  return std::unique_ptr<const MrcMr>(new MrcMr(pd_, buff, size, dataDirectPath_));
}

namespace {

void validateMrcConfig(const EndpointConfig& config) {
  const auto& mrc = config.mrc;
  if (mrc.port <= 0 || mrc.port > 255 || mrc.gidIndex < 0 || mrc.maxCqSize <= 0 || mrc.maxSendWr <= 0 ||
      mrc.maxRecvWr < 0 || config.maxWriteQueueSize <= 0) {
    throw Error("Invalid native MRC endpoint configuration", ErrorCode::InvalidUsage);
  }
}

}  // namespace

MrcMemory::MrcMemory(std::shared_ptr<RdmaProvider> provider, std::unique_ptr<const MrcMr> mr)
    : RdmaMemory(std::move(provider)), mr_(std::move(mr)), buffer_(mr_->getBuffer()) {}

MrcMemory::MrcMemory(const MrcMrInfo& info, size_t size) : buffer_{} {
  // Preserve padding too: existing metadata uses the raw object representation.
  std::memcpy(&buffer_.info, &info, sizeof(info));
  buffer_.size = size;
}

void MrcMemory::serialize(std::vector<char>& data) const { detail::serialize(data, buffer_.info); }

MrcEndpoint::MrcEndpoint(std::shared_ptr<RdmaProvider> provider, MrcCtx& context, EndpointConfig& config)
    : RdmaEndpoint(std::move(provider)) {
  qp_ = context.createQp(config.mrc.port, config.mrc.gidIndex, config.mrc.evProfileId, config.mrc.maxCqSize,
                         config.mrc.maxSendWr, config.mrc.maxRecvWr);
  info_ = qp_->getInfo();
}

void prepareMrcEndpointConfig(EndpointConfig& config) {
  if (config.maxWriteQueueSize <= 0) config.maxWriteQueueSize = config.mrc.maxCqSize;
  if (config.mrc.gidIndex < 0) config.mrc.gidIndex = env()->ibGidIndex;
  validateMrcConfig(config);
}

MrcEndpoint::MrcEndpoint(const MrcQpInfo& info) { std::memcpy(&info_, &info, sizeof(info_)); }

void MrcEndpoint::serialize(std::vector<char>& data) const { detail::serialize(data, info_); }

void MrcEndpoint::connect(const RdmaEndpoint& remote) {
  auto peer = dynamic_cast<const MrcEndpoint*>(&remote);
  if (!peer) throw Error("Not a native MRC endpoint", ErrorCode::InvalidUsage);
  if (!isLocal()) throw Error("Cannot connect an imported native MRC endpoint", ErrorCode::InvalidUsage);
  qp_->rtr(peer->info_);
  qp_->rts();
}

std::shared_ptr<BaseConnection> MrcEndpoint::createConnection(std::shared_ptr<Context>, const Endpoint&,
                                                              const Endpoint&) const {
  // TODO: Before enabling native signaling, agree on how the sender learns the peer's
  // initial posted receive count and when newly posted receives become available to it.
  throw Error("Native MRC connections are not implemented yet", ErrorCode::InvalidUsage);
}

MrcConnection::MrcConnection(std::shared_ptr<Context> context, const Endpoint& localEndpoint,
                             const Endpoint& remoteEndpoint)
    : BaseConnection(std::move(context), localEndpoint), remoteEndpoint_(remoteEndpoint) {
  auto* local = dynamic_cast<MrcEndpoint*>(getImpl(localEndpoint).rdmaEndpoint_.get());
  const auto* remote = dynamic_cast<const MrcEndpoint*>(getImpl(remoteEndpoint).rdmaEndpoint_.get());
  if (!local || !local->isLocal() || !remote || remote->isLocal()) {
    throw Error("Native MRC requires a local endpoint and an imported peer endpoint", ErrorCode::InvalidUsage);
  }
  const auto& config = localEndpoint.config().mrc;
  maxWriteQueueSize_ = std::min({maxWriteQueueSize_, config.maxSendWr, config.maxCqSize});
  qp_ = local->qp();
  qp_->writeSources_.reserve(maxWriteQueueSize_);
  local->connect(*remote);
}

void MrcConnection::write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
                          uint64_t size) {
  validateTransport(src, transport());
  validateTransport(dst, remoteTransport());
  const auto& srcOwner = getImpl(src).getTransportInfo(transport()).rdmaMemory;
  const auto& dstImpl = getImpl(dst);
  const auto* source = dynamic_cast<const MrcMemory*>(srcOwner.get());
  const auto* target = dynamic_cast<const MrcMemory*>(dstImpl.getTransportInfo(remoteTransport()).rdmaMemory.get());
  if (!source || !source->isLocal() || !target || target->isLocal() || dstImpl.hostHash != remoteEndpoint_.hostHash() ||
      dstImpl.pidHash != remoteEndpoint_.pidHash()) {
    throw Error("Native MRC write requires a local source and memory imported from the connected peer",
                ErrorCode::InvalidUsage);
  }

  if (qp_->writeSources_.size() >= static_cast<size_t>(maxWriteQueueSize_)) flush();
  qp_->stageSendWrite(source->mr(), target->buffer(), size, /*wrId=*/0, srcOffset, dstOffset, /*signaled=*/true);
  // Space was reserved before connecting. A post failure can leave accepted work on the QP.
  qp_->writeSources_.push_back(srcOwner);
  qp_->postSend();
}

void MrcConnection::updateAndSync(RegisteredMemory, uint64_t, uint64_t*, uint64_t) {
  throw Error("Native MRC signaling is not implemented yet", ErrorCode::InvalidUsage);
}

void MrcConnection::accumulate(RegisteredMemory, uint64_t, int64_t) {
  throw Error("Native MRC does not support remote atomics", ErrorCode::InvalidUsage);
}

void MrcConnection::requestFlush() {
  // As in IB, write() already posted the work. The proxy drives completion with progressFlush().
}

bool MrcConnection::progressFlush() {
  if (!qp_->progressSend()) return false;
  qp_->writeSources_.clear();
  return true;
}

void MrcConnection::flush(int64_t timeoutUsec) {
  Timer timer;
  while (!progressFlush()) {
    if (timeoutUsec >= 0 && timer.elapsed() >= timeoutUsec) {
      throw Error("Native MRC send completion timed out", ErrorCode::Timeout);
    }
  }
}

MrcProvider::MrcProvider(Transport transport) : context_(getMrcDeviceName(transport)) {}

std::shared_ptr<RdmaEndpoint> MrcProvider::createEndpoint(EndpointConfig& config) {
  prepareMrcEndpointConfig(config);
  return std::make_shared<MrcEndpoint>(shared_from_this(), context_, config);
}

std::shared_ptr<RdmaMemory> MrcProvider::registerMemory(void* data, size_t size) {
  auto memory = std::make_shared<MrcMemory>(shared_from_this(), context_.registerMr(data, size));
  INFO(NET, "Native MRC mr for address ", data, " with size ", size, " is registered");
  return memory;
}

std::shared_ptr<RdmaEndpoint> deserializeMrcEndpoint(const EndpointConfig& config,
                                                     std::vector<char>::const_iterator& it,
                                                     std::vector<char>::const_iterator end) {
  validateMrcConfig(config);
  MrcQpInfo info;
  if (end - it < static_cast<std::ptrdiff_t>(sizeof(info))) {
    throw Error("Truncated native MRC endpoint metadata", ErrorCode::InvalidUsage);
  }
  it = detail::deserialize(it, info);
  return std::make_shared<MrcEndpoint>(info);
}

std::shared_ptr<RdmaMemory> deserializeMrcMemory(Transport transport, std::vector<char>::const_iterator& it,
                                                 std::vector<char>::const_iterator end, void* originalData, size_t size,
                                                 TransportFlags advertised, TransportFlags& seen) {
  if (!advertised.has(transport) || seen.has(transport)) {
    throw Error("Unexpected or duplicate native MRC memory record", ErrorCode::InvalidUsage);
  }
  MrcMrInfo info;
  if (end - it < static_cast<std::ptrdiff_t>(sizeof(info))) {
    throw Error("Truncated registered memory metadata", ErrorCode::InvalidUsage);
  }
  it = detail::deserialize(it, info);
  if (size == 0 || info.addr == 0 || info.addr != reinterpret_cast<uintptr_t>(originalData) ||
      size > std::numeric_limits<uint64_t>::max() - info.addr) {
    throw Error("Invalid native MRC memory range", ErrorCode::InvalidUsage);
  }
  seen |= transport;
  return std::make_shared<MrcMemory>(info, size);
}

}  // namespace mscclpp

#endif  // defined(MSCCLPP_USE_MRC)
