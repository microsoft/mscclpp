// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "endpoint.hpp"

#include <algorithm>

#include "api.h"
#include "context.hpp"
#include "serialization.hpp"
#include "socket.h"
#include "utils_internal.hpp"

namespace mscclpp {

Endpoint::Impl::Impl(const EndpointConfig& config, Context::Impl& contextImpl)
    : config_(config), hostHash_(getHostHash()), pidHash_(getPidHash()) {
  if (config_.device.type == DeviceType::GPU && config_.device.id < 0) {
    MSCCLPP_CUDATHROW(cudaGetDevice(&(config_.device.id)));
  }
  if (AllIBTransports.has(config_.transport)) {
    ibLocal_ = true;
    if (config_.maxWriteQueueSize <= 0) {
      config_.maxWriteQueueSize = config_.ib.maxCqSize;
    }
    ibQp_ = contextImpl.getIbContext(config_.transport)
                ->createQp(config_.ib.port, config_.ib.gidIndex, config_.ib.maxCqSize, config_.ib.maxCqPollNum,
                           config_.ib.maxSendWr, 0, config_.ib.maxWrPerSend);
    ibQpInfo_ = ibQp_->getInfo();
  } else if (config_.transport == Transport::Ethernet) {
    // Configuring Ethernet Interfaces
    abortFlag_ = 0;
    int ret = FindInterfaces(netIfName_, &socketAddress_, MAX_IF_NAME_SIZE, 1);
    if (ret <= 0) throw Error("Failed to find network interfaces", ErrorCode::InternalError);

    // Starting Server Socket
    socket_ = std::make_unique<Socket>(&socketAddress_, MSCCLPP_SOCKET_MAGIC, SocketTypeBootstrap, abortFlag_);
    socket_->bindAndListen();
    socketAddress_ = socket_->getAddr();
  }
}

Endpoint::Impl::Impl(const std::vector<char>& serialization) {
  auto it = serialization.begin();
  it = detail::deserialize(it, config_);
  it = detail::deserialize(it, hostHash_);
  it = detail::deserialize(it, pidHash_);
  if (AllIBTransports.has(config_.transport)) {
    ibLocal_ = false;
    it = detail::deserialize(it, ibQpInfo_);
  } else if (config_.transport == Transport::Ethernet) {
    it = detail::deserialize(it, socketAddress_);
  }
  if (it != serialization.end()) {
    throw Error("Endpoint deserialization failed", ErrorCode::Aborted);
  }
}

MSCCLPP_API_CPP Endpoint::Endpoint(std::shared_ptr<Endpoint::Impl> pimpl) : pimpl_(pimpl) {}

MSCCLPP_API_CPP const EndpointConfig& Endpoint::config() const { return pimpl_->config_; }

MSCCLPP_API_CPP Transport Endpoint::transport() const { return pimpl_->config_.transport; }

MSCCLPP_API_CPP const Device& Endpoint::device() const { return pimpl_->config_.device; }

MSCCLPP_API_CPP uint64_t Endpoint::hostHash() const { return pimpl_->hostHash_; }

MSCCLPP_API_CPP uint64_t Endpoint::pidHash() const { return pimpl_->pidHash_; }

MSCCLPP_API_CPP int Endpoint::maxWriteQueueSize() const { return pimpl_->config_.maxWriteQueueSize; }

MSCCLPP_API_CPP std::vector<char> Endpoint::serialize() const {
  std::vector<char> data;
  detail::serialize(data, pimpl_->config_);
  detail::serialize(data, pimpl_->hostHash_);
  detail::serialize(data, pimpl_->pidHash_);
  if (AllIBTransports.has(pimpl_->config_.transport)) {
    detail::serialize(data, pimpl_->ibQpInfo_);
  } else if (pimpl_->config_.transport == Transport::Ethernet) {
    detail::serialize(data, pimpl_->socketAddress_);
  }
  return data;
}

MSCCLPP_API_CPP Endpoint Endpoint::deserialize(const std::vector<char>& data) {
  return Endpoint(std::make_shared<Impl>(data));
}

}  // namespace mscclpp
