#ifndef MSCCLPP_GPUNETIO_HOST_GPU_NET_IO_QP_TABLE_HPP_
#define MSCCLPP_GPUNETIO_HOST_GPU_NET_IO_QP_TABLE_HPP_

#include <cstring>
#include <stdexcept>
#include <vector>

#include "common/doca_gpunetio_verbs_dev.h"
#include "host/doca_gpunetio_high_level.h"

namespace mscclpp::detail {

inline std::vector<doca_gpu_dev_verbs_qp> buildGpuNetIoQpTable(const std::vector<doca_gpu_verbs_qp_hl*>& qps, int rank,
                                                               int numQpsPerPeer) {
  if (numQpsPerPeer < 1 || numQpsPerPeer > 64 || qps.empty() || qps.size() % numQpsPerPeer != 0 || rank < 0 ||
      static_cast<size_t>(rank) >= qps.size() / numQpsPerPeer) {
    throw std::invalid_argument("Invalid GPUNetIO peer QP table geometry");
  }
  std::vector<doca_gpu_dev_verbs_qp> table(qps.size());
  for (size_t index = 0; index < qps.size(); ++index) {
    if (index / numQpsPerPeer == static_cast<size_t>(rank)) {
      if (qps[index] != nullptr) throw std::invalid_argument("GPUNetIO self QP must be empty");
      continue;
    }
    const auto* qp = qps[index];
    if (qp == nullptr || qp->qp_gverbs == nullptr || qp->qp_gverbs->qp_cpu == nullptr) {
      throw std::invalid_argument("Missing GPUNetIO remote QP descriptor");
    }
    std::memcpy(&table[index], qp->qp_gverbs->qp_cpu, sizeof(table[index]));
  }
  return table;
}

}  // namespace mscclpp::detail

#endif