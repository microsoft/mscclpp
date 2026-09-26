// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_GPUNETIO_HOST_GPU_NET_IO_QP_TABLE_HPP_
#define MSCCLPP_GPUNETIO_HOST_GPU_NET_IO_QP_TABLE_HPP_

#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "common/doca_gpunetio_verbs_dev.h"
#include "host/doca_gpunetio_high_level.h"

namespace mscclpp::detail {

inline std::vector<int> gpuNetIoQpOffsets(const std::vector<int>& plans, int rank, int ranks) {
  if (ranks < 1 || rank < 0 || rank >= ranks || plans.size() != static_cast<size_t>(ranks) * ranks) {
    throw std::invalid_argument("Invalid GPUNetIO connection plan geometry");
  }
  for (int source = 0; source < ranks; ++source) {
    int total = 0;
    for (int peer = 0; peer < ranks; ++peer) {
      const int count = plans[static_cast<size_t>(source) * ranks + peer];
      if (count < 0 || count > 64 || (source == peer && count != 0) ||
          count != plans[static_cast<size_t>(peer) * ranks + source] ||
          total > std::numeric_limits<int>::max() - count) {
        throw std::invalid_argument("GPUNetIO plans require reciprocal counts in [0, 64] and zero self counts");
      }
      total += count;
    }
  }
  std::vector<int> offsets(ranks + 1);
  for (int peer = 0; peer < ranks; ++peer) {
    offsets[peer + 1] = offsets[peer] + plans[static_cast<size_t>(rank) * ranks + peer];
  }
  return offsets;
}

template <class BootstrapType, class Metadata>
void exchangeGpuNetIoQpMetadata(BootstrapType& bootstrap, std::vector<Metadata>& local, std::vector<Metadata>& remote,
                                const std::vector<int>& offsets, int rank, int tag) {
  if (offsets.empty() || rank < 0 || static_cast<size_t>(rank + 1) >= offsets.size() || offsets.front() != 0 ||
      offsets.back() < 0 || static_cast<size_t>(offsets.back()) != local.size() || remote.size() != local.size() ||
      tag < 0) {
    throw std::invalid_argument("Invalid GPUNetIO metadata exchange geometry");
  }
  for (size_t peer = 0; peer + 1 < offsets.size(); ++peer) {
    if (offsets[peer] < 0 || offsets[peer + 1] < offsets[peer] ||
        static_cast<size_t>(offsets[peer + 1] - offsets[peer]) >
            static_cast<size_t>(std::numeric_limits<int>::max()) / sizeof(Metadata)) {
      throw std::invalid_argument("Invalid GPUNetIO metadata exchange size");
    }
  }
  for (int peer = 0; static_cast<size_t>(peer + 1) < offsets.size(); ++peer) {
    const int count = offsets[peer + 1] - offsets[peer];
    if (peer == rank || count == 0) continue;
    const int bytes = static_cast<int>(count * sizeof(Metadata));
    if (rank < peer) {
      bootstrap.send(local.data() + offsets[peer], bytes, peer, tag);
      bootstrap.recv(remote.data() + offsets[peer], bytes, peer, tag);
    } else {
      bootstrap.recv(remote.data() + offsets[peer], bytes, peer, tag);
      bootstrap.send(local.data() + offsets[peer], bytes, peer, tag);
    }
  }
}

template <class BootstrapType, class Metadata>
std::vector<Metadata> exchangeGpuNetIoQpMetadata(BootstrapType& bootstrap, std::vector<Metadata>& local,
                                                 const std::vector<int>& offsets, int rank, int tag) {
  std::vector<Metadata> remote(local.size());
  exchangeGpuNetIoQpMetadata(bootstrap, local, remote, offsets, rank, tag);
  return remote;
}

inline std::vector<doca_gpu_dev_verbs_qp> buildGpuNetIoQpTable(const std::vector<doca_gpu_verbs_qp_hl*>& qps) {
  std::vector<doca_gpu_dev_verbs_qp> table(qps.size());
  for (size_t index = 0; index < qps.size(); ++index) {
    const auto* qp = qps[index];
    if (qp == nullptr || qp->qp_gverbs == nullptr || qp->qp_gverbs->qp_cpu == nullptr) {
      throw std::invalid_argument("Missing GPUNetIO requested QP descriptor");
    }
    std::memcpy(&table[index], qp->qp_gverbs->qp_cpu, sizeof(table[index]));
  }
  return table;
}

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