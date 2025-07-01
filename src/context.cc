// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "context.hpp"

#include <mscclpp/env.hpp>

#include "api.h"
#include "connection.hpp"
#include "debug.h"
#include "endpoint.hpp"
#include "registered_memory.hpp"

namespace mscclpp {

CudaIpcStream::CudaIpcStream() : stream_(std::make_shared<CudaStreamWithFlags>()), dirty_(false) {}

void CudaIpcStream::setStreamIfNeeded() {
  if (!env()->cudaIpcUseDefaultStream && stream_->empty()) stream_->set(cudaStreamNonBlocking);
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
  } else {
    return it->second.get();
  }
}

MSCCLPP_API_CPP Context::Context() : pimpl_(std::make_unique<Impl>()) {}

MSCCLPP_API_CPP Context::~Context() = default;

MSCCLPP_API_CPP RegisteredMemory Context::registerMemory(void *ptr, size_t size, TransportFlags transports) {
  return RegisteredMemory(std::make_shared<RegisteredMemory::Impl>(ptr, size, transports, *pimpl_));
}

MSCCLPP_API_CPP Endpoint Context::createEndpoint(EndpointConfig config) {
  return Endpoint(std::make_shared<Endpoint::Impl>(config, *pimpl_));
}

MSCCLPP_API_CPP std::shared_ptr<Connection> Context::connect(Endpoint localEndpoint, Endpoint remoteEndpoint) {
  std::shared_ptr<Connection> conn;
  if (localEndpoint.transport() == Transport::CudaIpc) {
    if (remoteEndpoint.transport() != Transport::CudaIpc) {
      throw mscclpp::Error("Local transport is CudaIpc but remote is not", ErrorCode::InvalidUsage);
    }
#if defined(MSCCLPP_DEVICE_HIP)
    pimpl_->ipcStreams_.emplace_back(std::make_shared<CudaIpcStream>());
#else   // !defined(MSCCLPP_DEVICE_HIP)
    if (pimpl_->ipcStreams_.empty()) {
      pimpl_->ipcStreams_.emplace_back(std::make_shared<CudaIpcStream>());
    }
#endif  // !defined(MSCCLPP_DEVICE_HIP)
    conn = std::make_shared<CudaIpcConnection>(shared_from_this(), localEndpoint, remoteEndpoint,
                                               pimpl_->ipcStreams_.back());
  } else if (AllIBTransports.has(localEndpoint.transport())) {
    if (!AllIBTransports.has(remoteEndpoint.transport())) {
      throw mscclpp::Error("Local transport is IB but remote is not", ErrorCode::InvalidUsage);
    }
    conn = std::make_shared<IBConnection>(shared_from_this(), localEndpoint, remoteEndpoint);
  } else if (localEndpoint.transport() == Transport::Ethernet) {
    if (remoteEndpoint.transport() != Transport::Ethernet) {
      throw mscclpp::Error("Local transport is Ethernet but remote is not", ErrorCode::InvalidUsage);
    }
    conn = std::make_shared<EthernetConnection>(shared_from_this(), localEndpoint, remoteEndpoint);
  } else {
    throw mscclpp::Error("Unsupported transport", ErrorCode::InternalError);
  }
  return conn;
}

}  // namespace mscclpp
