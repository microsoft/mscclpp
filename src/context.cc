// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "context.hpp"

#include <mscclpp/env.hpp>
#include <sstream>

#include "api.h"
#include "connection.hpp"
#include "debug.h"
#include "endpoint.hpp"
#include "registered_memory.hpp"

namespace mscclpp {

CudaIpcStream::CudaIpcStream(int deviceId)
    : stream_(std::make_shared<CudaStreamWithFlags>()), deviceId_(deviceId), dirty_(false) {}

void CudaIpcStream::setStreamIfNeeded() {
  if (!env()->cudaIpcUseDefaultStream && stream_->empty()) {
    MSCCLPP_CUDATHROW(cudaSetDevice(deviceId_));
    stream_->set(cudaStreamNonBlocking);
  }
}

void CudaIpcStream::memcpyD2D(void *dst, const void *src, size_t nbytes) {
  setStreamIfNeeded();
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, *stream_));
  dirty_ = true;
}

void CudaIpcStream::memcpyH2D(void *dst, const void *src, size_t nbytes) {
  setStreamIfNeeded();
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice, *stream_));
  dirty_ = true;
}

void CudaIpcStream::sync() {
  setStreamIfNeeded();
  if (dirty_) {
    MSCCLPP_CUDATHROW(cudaStreamSynchronize(*stream_));
    dirty_ = false;
  }
}

Context::Impl::Impl() {}

IbCtx *Context::Impl::getIbContext(Transport ibTransport) {
  // Find IB context or create it
  auto it = ibContexts_.find(ibTransport);
  if (it == ibContexts_.end()) {
    auto ibDev = getIBDeviceName(ibTransport);
    ibContexts_[ibTransport] = std::make_unique<IbCtx>(ibDev);
    return ibContexts_[ibTransport].get();
  }
  return it->second.get();
}

MSCCLPP_API_CPP Context::Context() : pimpl_(std::make_unique<Impl>()) {}

MSCCLPP_API_CPP Context::~Context() = default;

MSCCLPP_API_CPP RegisteredMemory Context::registerMemory(void *ptr, size_t size, TransportFlags transports) {
  return RegisteredMemory(std::make_shared<RegisteredMemory::Impl>(ptr, size, transports, *pimpl_));
}

MSCCLPP_API_CPP Endpoint Context::createEndpoint(EndpointConfig config) {
  return Endpoint(std::make_shared<Endpoint::Impl>(config, *pimpl_));
}

MSCCLPP_API_CPP std::shared_ptr<Connection> Context::connect(const Endpoint &localEndpoint,
                                                             const Endpoint &remoteEndpoint) {
  if (localEndpoint.device().type == DeviceType::GPU && localEndpoint.device().id < 0) {
    throw Error("No GPU device ID provided for local endpoint", ErrorCode::InvalidUsage);
  }
  if (remoteEndpoint.device().type == DeviceType::GPU && remoteEndpoint.device().id < 0) {
    throw Error("No GPU device ID provided for remote endpoint", ErrorCode::InvalidUsage);
  }
  auto localTransport = localEndpoint.transport();
  auto remoteTransport = remoteEndpoint.transport();
  if (localTransport != remoteTransport &&
      !(AllIBTransports.has(localTransport) && AllIBTransports.has(remoteTransport))) {
    std::stringstream ss;
    ss << "Transport mismatch between local (" << std::to_string(localTransport) << ") and remote ("
       << std::to_string(remoteEndpoint.transport()) << ") endpoints";
    throw Error(ss.str(), ErrorCode::InvalidUsage);
  }
  std::shared_ptr<Connection> conn;
  if (localTransport == Transport::CudaIpc) {
    conn = std::make_shared<CudaIpcConnection>(shared_from_this(), localEndpoint, remoteEndpoint);
  } else if (AllIBTransports.has(localTransport)) {
    conn = std::make_shared<IBConnection>(shared_from_this(), localEndpoint, remoteEndpoint);
  } else if (localTransport == Transport::Ethernet) {
    conn = std::make_shared<EthernetConnection>(shared_from_this(), localEndpoint, remoteEndpoint);
  } else {
    throw Error("Unsupported transport", ErrorCode::InternalError);
  }
  return conn;
}

}  // namespace mscclpp
