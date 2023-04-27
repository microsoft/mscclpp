#include "mscclpp.hpp"
#include "communicator.hpp"
#include "host_connection.hpp"
#include "comm.h"
#include "basic_proxy_handler.hpp"
#include "api.h"
#include "utils.h"
#include "checks.hpp"
#include "debug.h"
#include "connection.hpp"
#include "registered_memory.hpp"

namespace mscclpp {

Communicator::Impl::Impl(std::shared_ptr<BaseBootstrap> bootstrap) : bootstrap_(bootstrap) {}

Communicator::Impl::~Impl() {
  ibContexts.clear();
}

IbCtx* Communicator::Impl::getIbContext(Transport ibTransport) {
  // Find IB context or create it
  auto it = ibContexts.find(ibTransport);
  if (it == ibContexts.end()) {
    auto ibDev = getIBDeviceName(ibTransport);
    ibContexts[ibTransport] = std::make_unique<IbCtx>(ibDev);
    return ibContexts[ibTransport].get();
  } else {
    return it->second.get();
  }
}

MSCCLPP_API_CPP Communicator::~Communicator() = default;

MSCCLPP_API_CPP Communicator::Communicator(std::shared_ptr<BaseBootstrap> bootstrap) : pimpl(std::make_unique<Impl>(bootstrap)) {}

MSCCLPP_API_CPP void Communicator::bootstrapAllGather(void* data, int size) {
  mscclppBootstrapAllGather(pimpl->comm, data, size);
}

MSCCLPP_API_CPP void Communicator::bootstrapBarrier() {
  mscclppBootstrapBarrier(pimpl->comm);
}

RegisteredMemory Communicator::registerMemory(void* ptr, size_t size, TransportFlags transports) {
  return RegisteredMemory(std::make_shared<RegisteredMemory::Impl>(ptr, size, pimpl->comm->rank, transports, *pimpl));
}

MSCCLPP_API_CPP std::shared_ptr<Connection> Communicator::connect(int remoteRank, int tag, Transport transport) {
  std::shared_ptr<ConnectionBase> conn;
  if (transport == Transport::CudaIpc) {
    auto cudaIpcConn = std::make_shared<CudaIpcConnection>();
    conn = cudaIpcConn;
  } else if (AllIBTransports.has(transport)) {
    auto ibConn = std::make_shared<IBConnection>(remoteRank, tag, transport, *pimpl);
    conn = ibConn;
  } else {
    throw std::runtime_error("Unsupported transport");
  }
  pimpl->connections.push_back(conn);
  return conn;
}

MSCCLPP_API_CPP void Communicator::connectionSetup() {
  for (auto& conn : pimpl->connections) {
    conn->startSetup(pimpl->bootstrap_);
  }
  for (auto& conn : pimpl->connections) {
    conn->endSetup(pimpl->bootstrap_);
  }
}

} // namespace mscclpp
