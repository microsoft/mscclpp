// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_GPUNETIO_HOST_GPU_NET_IO_POLICY_HPP_
#define MSCCLPP_GPUNETIO_HOST_GPU_NET_IO_POLICY_HPP_

#include "common/doca_gpunetio_verbs_dev.h"
#include "host/doca_gpunetio_high_level.h"

namespace mscclpp::detail {

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