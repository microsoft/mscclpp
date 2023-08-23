// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "context.hpp"

#include <sstream>

#include "api.h"
#include "connection.hpp"
#include "debug.h"
#include "endpoint.hpp"
#include "registered_memory.hpp"
#include "utils_internal.hpp"

namespace mscclpp {

Context::Impl::Impl() : ipcStream_(cudaStreamNonBlocking), hostHash_(getHostHash()) {
  INFO(MSCCLPP_INIT, "Host hash: %lx", hostHash_);
}

IbCtx* Context::Impl::getIbContext(Transport ibTransport) {
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

MSCCLPP_API_CPP Context::Context() : pimpl(std::make_unique<Impl>()) {}

MSCCLPP_API_CPP Context::~Context() = default;

MSCCLPP_API_CPP RegisteredMemory Context::registerMemory(void* ptr, size_t size, TransportFlags transports) {
  return RegisteredMemory(std::make_shared<RegisteredMemory::Impl>(ptr, size, transports, *pimpl));
}

MSCCLPP_API_CPP Endpoint Context::createEndpoint(Transport transport, int ibMaxCqSize, int ibMaxCqPollNum,
                                                 int ibMaxSendWr, int ibMaxWrPerSend) {
  return Endpoint(
      std::make_shared<Endpoint::Impl>(transport, ibMaxCqSize, ibMaxCqPollNum, ibMaxSendWr, ibMaxWrPerSend, *pimpl));
}

MSCCLPP_API_CPP std::shared_ptr<Connection> Context::connect(Endpoint localEndpoint, Endpoint remoteEndpoint) {
  std::shared_ptr<Connection> conn;
  if (localEndpoint.transport() == Transport::CudaIpc) {
    conn = std::make_shared<CudaIpcConnection>(localEndpoint, remoteEndpoint, pimpl->ipcStream_);
  } else if (AllIBTransports.has(localEndpoint.transport())) {
    conn = std::make_shared<IBConnection>(localEndpoint, remoteEndpoint, *this);
  } else {
    throw mscclpp::Error("Unsupported transport", ErrorCode::InternalError);
  }
  pimpl->connections_.push_back(conn);
  return conn;
}

}  // namespace mscclpp
