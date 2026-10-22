// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_GPUNETIO_HOST_GPU_NET_IO_POLICY_HPP_
#define MSCCLPP_GPUNETIO_HOST_GPU_NET_IO_POLICY_HPP_

#include <infiniband/verbs.h>

#include <stdexcept>

#include "common/doca_gpunetio_verbs_dev.h"
#include "host/doca_gpunetio_high_level.h"

namespace mscclpp::detail {

struct GpuNetIoPortInfo {
  ibv_port_attr port{};
  ibv_gid gid{};
};

template <class QueryPort, class QueryGid>
GpuNetIoPortInfo queryGpuNetIoPort(ibv_context* context, int port, int gidIndex, QueryPort queryPort,
                                   QueryGid queryGid) {
  if (port < 1 || port > 255 || gidIndex < 0 || gidIndex > 255) {
    throw std::invalid_argument("GPUNetIO port must be in [1, 255] and GID index in [0, 255]");
  }
  GpuNetIoPortInfo info;
  if (queryPort(context, static_cast<uint8_t>(port), &info.port) != 0) {
    throw std::runtime_error("ibv_query_port failed for GPUNetIO");
  }
  if (info.port.state != IBV_PORT_ACTIVE ||
      (info.port.link_layer != IBV_LINK_LAYER_ETHERNET && info.port.link_layer != IBV_LINK_LAYER_INFINIBAND) ||
      info.port.active_mtu < IBV_MTU_256 || info.port.active_mtu > IBV_MTU_4096 || gidIndex >= info.port.gid_tbl_len) {
    throw std::invalid_argument("GPUNetIO requires an active IB/RoCE port, valid MTU and an in-range GID index");
  }
  if (queryGid(context, static_cast<uint8_t>(port), gidIndex, &info.gid) != 0) {
    throw std::runtime_error("ibv_query_gid failed for GPUNetIO");
  }
  if (info.port.link_layer == IBV_LINK_LAYER_ETHERNET || (info.port.flags & IBV_QPF_GRH_REQUIRED)) {
    bool nonzero = false;
    for (const auto byte : info.gid.raw) nonzero = nonzero || byte != 0;
    if (!nonzero) throw std::invalid_argument("GPUNetIO requires a nonzero GID for RoCE or GRH routing");
  }
  return info;
}

inline doca_error_t createDirectGpuNetIoQp(doca_gpu_t* gpu, doca_dev_t* device, ibv_pd* pd,
                                           doca_gpu_verbs_qp_hl** output) {
  if (output == nullptr) return DOCA_ERROR_INVALID_VALUE;
  *output = nullptr;
  doca_gpu_verbs_qp_init_attr_hl attributes{};
  attributes.gpu_dev = gpu;
  attributes.net_dev = device;
  attributes.ibpd = pd;
  attributes.sq_nwqe = 1024;
  attributes.nic_handler = DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB;
  attributes.mreg_type = DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT;
  attributes.send_dbr_mode_ext = DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_VALID_DBR;
  attributes.cq_type = DOCA_GPUNETIO_VERBS_CQ_64B;
  attributes.cq_collapsed = false;
  attributes.enable_umem_cpu = false;
  attributes.ordering_semantic = DOCA_VERBS_QP_ORDERING_SEMANTIC_IBTA;
  const auto status = doca_gpu_verbs_create_qp_hl(&attributes, output);
  if (status != DOCA_SUCCESS) return status;
  const auto* qp = *output;
  if (qp == nullptr || qp->qp_gverbs == nullptr || qp->qp_gverbs->qp_cpu == nullptr ||
      qp->nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB ||
      qp->send_dbr_mode_ext != DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_VALID_DBR || qp->qp_gverbs->cpu_proxy ||
      qp->qp_gverbs->send_dbr_mode_ext != DOCA_GPUNETIO_VERBS_SEND_DBR_MODE_EXT_VALID_DBR ||
      qp->qp_gverbs->cq_type != DOCA_GPUNETIO_VERBS_CQ_64B ||
      qp->qp_gverbs->qp_cpu->nic_handler != DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB ||
      qp->qp_gverbs->qp_cpu->mem_type != DOCA_GPUNETIO_VERBS_MEM_TYPE_GPU) {
    return DOCA_ERROR_NOT_SUPPORTED;
  }
  return DOCA_SUCCESS;
}

}  // namespace mscclpp::detail

#endif