// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_ENDPOINT_HPP_
#define MSCCLPP_ENDPOINT_HPP_

#include <memory>
#include <mscclpp/core.hpp>
#include <vector>

#include "ib.hpp"
#include "socket.h"

#define MAX_IF_NAME_SIZE 16

namespace mscclpp {

/// Data structure for write-with-imm signaling: each entry contains (dstGpuAddr, newValue)
struct WriteImmData {
  uint64_t dstGpuAddr;
  uint64_t newValue;
};

/// Buffer size for write-with-imm signaling
static constexpr int kWriteImmBufSize = 16;

struct Endpoint::Impl {
  Impl(const EndpointConfig& config, Context::Impl& contextImpl);
  Impl(const std::vector<char>& serialization);

  EndpointConfig config_;
  uint64_t hostHash_;
  uint64_t pidHash_;

  // The following are only used for IB and are undefined for other transports.
  bool ibLocal_;
  std::shared_ptr<IbQp> ibQp_;
  IbQpInfo ibQpInfo_;

  // The following are only used for IB with write-with-imm signaling enabled.
  bool useWriteImmSignal_;
  std::unique_ptr<WriteImmData[]> writeImmRecvBuf_;
  std::unique_ptr<const IbMr> writeImmRecvBufIbMr_;
  IbMrInfo writeImmRecvBufMrInfo_;

  // The following are only used for Ethernet and are undefined for other transports.
  std::unique_ptr<Socket> socket_;
  SocketAddress socketAddress_;
  volatile uint32_t* abortFlag_;
  char netIfName_[MAX_IF_NAME_SIZE + 1];
};

}  // namespace mscclpp

#endif  // MSCCLPP_ENDPOINT_HPP_
