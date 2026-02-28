// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_ENDPOINT_HPP_
#define MSCCLPP_ENDPOINT_HPP_

#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <vector>

#include "ib.hpp"
#include "socket.h"

#define MAX_IF_NAME_SIZE 16

namespace mscclpp {

struct Endpoint::Impl {
  Impl(const EndpointConfig& config, Context::Impl& contextImpl);
  Impl(const std::vector<char>& serialization);

  EndpointConfig config_;
  uint64_t hostHash_;
  uint64_t pidHash_;

  // The following are only used for IB and are undefined for other transports.
  bool ibLocal_;
  bool ibNoAtomic_;
  std::shared_ptr<IbQp> ibQp_;
  IbQpInfo ibQpInfo_;

  // Signal GPU buffer for write-with-imm data payload (ibNoAtomic_ only).
  // Each endpoint allocates a 64-bit GPU buffer and registers it as an IB MR.
  // The MR info is serialized/exchanged so the remote can RDMA-write to it.
  std::shared_ptr<uint64_t> ibSignalGpuBuffer_;
  std::unique_ptr<const IbMr> ibSignalGpuMr_;
  IbMrInfo ibSignalGpuMrInfo_;

  // The following are only used for Ethernet and are undefined for other transports.
  std::unique_ptr<Socket> socket_;
  SocketAddress socketAddress_;
  volatile uint32_t* abortFlag_;
  char netIfName_[MAX_IF_NAME_SIZE + 1];
};

}  // namespace mscclpp

#endif  // MSCCLPP_ENDPOINT_HPP_
