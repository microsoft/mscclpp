// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Placeholder launchers for the not-yet-ported internode HT and low-latency
// kernels. `get_dispatch_layout` and `get_source_meta_bytes` ARE ported in
// `kernels/internode_layout.cu`.

#include <stdexcept>

#include "kernels/api.cuh"

namespace mscclpp {
namespace ep {

namespace internode_ll {

void clean_low_latency_buffer(int* /*clean_0*/, int /*n0*/, int* /*clean_1*/, int /*n1*/, cudaStream_t /*stream*/) {
  throw std::runtime_error(
      "mscclpp::ep::internode_ll::clean_low_latency_buffer: not yet ported. "
      "See nccl/contrib/nccl_ep/device/low_latency.cu and DeepEP "
      "csrc/kernels/internode_ll.cu for the reference implementation.");
}

}  // namespace internode_ll

}  // namespace ep
}  // namespace mscclpp
