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

Endpoint::Impl::Impl(EndpointConfig config, Context::Impl& contextImpl)
    : transport_(config.transport),
      device_(config.device),
      hostHash_(getHostHash()),
      pidHash_(getPidHash()),
      maxWriteQueueSize_(config.maxWriteQueueSize) {
  if (device_.type == DeviceType::GPU && device_.id < 0) {
    MSCCLPP_CUDATHROW(cudaGetDevice(&(device_.id)));
  }
  if (AllIBTransports.has(transport_)) {
    ibLocal_ = true;
    ibQp_ = contextImpl.getIbContext(transport_)
                ->createQp(config.ibMaxCqSize, config.ibMaxCqPollNum, config.ibMaxSendWr, 0, config.ibMaxWrPerSend);
    ibQpInfo_ = ibQp_->getInfo();
  } else if (transport_ == Transport::Ethernet) {
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
  it = detail::deserialize(it, transport_);
  it = detail::deserialize(it, device_);
  it = detail::deserialize(it, hostHash_);
  it = detail::deserialize(it, pidHash_);
  if (AllIBTransports.has(transport_)) {
    ibLocal_ = false;
    it = detail::deserialize(it, ibQpInfo_);
  }
  if (transport_ == Transport::Ethernet) {
    it = detail::deserialize(it, socketAddress_);
  }
}

MSCCLPP_API_CPP Endpoint::Endpoint(std::shared_ptr<Endpoint::Impl> pimpl) : pimpl_(pimpl) {}

MSCCLPP_API_CPP Transport Endpoint::transport() const { return pimpl_->transport_; }

MSCCLPP_API_CPP const Device& Endpoint::device() const { return pimpl_->device_; }

MSCCLPP_API_CPP uint64_t Endpoint::hostHash() const { return pimpl_->hostHash_; }

MSCCLPP_API_CPP uint64_t Endpoint::pidHash() const { return pimpl_->pidHash_; }

MSCCLPP_API_CPP int Endpoint::maxWriteQueueSize() const { return pimpl_->maxWriteQueueSize_; }

MSCCLPP_API_CPP std::vector<char> Endpoint::serialize() const {
  std::vector<char> data;
  detail::serialize(data, pimpl_->transport_);
  detail::serialize(data, pimpl_->device_);
  detail::serialize(data, pimpl_->hostHash_);
  detail::serialize(data, pimpl_->pidHash_);
  if (AllIBTransports.has(pimpl_->transport_)) {
    detail::serialize(data, pimpl_->ibQpInfo_);
  }
  if ((pimpl_->transport_) == Transport::Ethernet) {
    detail::serialize(data, pimpl_->socketAddress_);
  }
  return data;
}

MSCCLPP_API_CPP Endpoint Endpoint::deserialize(const std::vector<char>& data) {
  return Endpoint(std::make_shared<Impl>(data));
}

}  // namespace mscclpp
