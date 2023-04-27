#include <sstream>

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

Communicator::Impl::Impl(std::shared_ptr<BaseBootstrap> bootstrap) : bootstrap_(bootstrap) {
  rankToHash_.resize(bootstrap->getNranks());
  auto hostHash = getHostHash();
  INFO(MSCCLPP_INIT, "Host hash: %lx", hostHash);
  rankToHash_[bootstrap->getRank()] = hostHash;
  bootstrap->allGather(rankToHash_.data(), sizeof(uint64_t));
}

Communicator::Impl::~Impl() {
  ibContexts.clear();
}

IbCtx* Communicator::Impl::getIbContext(TransportFlags ibTransport) {
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

static mscclppTransport_t transportToCStyle(TransportFlags flags) {
  switch (flags) {
    case TransportIB0:
    case TransportIB1:
    case TransportIB2:
    case TransportIB3:
    case TransportIB4:
    case TransportIB5:
    case TransportIB6:
    case TransportIB7:
      return mscclppTransportIB;
    case TransportCudaIpc:
      return mscclppTransportP2P;
    default:
      throw std::runtime_error("Unsupported conversion");
  }
}

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

MSCCLPP_API_CPP std::shared_ptr<Connection> Communicator::connect(int remoteRank, int tag, TransportFlags transport) {
  std::shared_ptr<ConnectionBase> conn;
  if (transport | TransportCudaIpc) {
    // sanity check: make sure the IPC connection is being made within a node
    if (pimpl->rankToHash_[remoteRank] != pimpl->rankToHash_[pimpl->bootstrap_->getRank()]) {
      std::stringstream ss;
      ss << "Cuda IPC connection can only be made within a node: " << remoteRank << " != " << pimpl->bootstrap_->getRank();
      throw std::runtime_error(ss.str());
    }
    auto cudaIpcConn = std::make_shared<CudaIpcConnection>();
    conn = cudaIpcConn;
    INFO(MSCCLPP_INIT, "Cuda IPC connection between %d(%lx) and %d(%lx) created", pimpl->bootstrap_->getRank(), pimpl->rankToHash_[pimpl->bootstrap_->getRank()], 
          remoteRank, pimpl->rankToHash_[remoteRank]);
  } else if (transport | TransportAllIB) {
    auto ibConn = std::make_shared<IBConnection>(remoteRank, tag, transport, *pimpl);
    conn = ibConn;
    INFO(MSCCLPP_INIT, "IB connection between %d(%lx) via %s and %d(%lx) created", pimpl->bootstrap_->getRank(), pimpl->rankToHash_[pimpl->bootstrap_->getRank()], 
          getIBDeviceName(transport).c_str(), remoteRank, pimpl->rankToHash_[remoteRank]);
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
