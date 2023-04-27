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
  for (auto& entry : ibContexts) {
    delete entry.second;
  }
  ibContexts.clear();
}

IbCtx* Communicator::Impl::getIbContext(TransportFlags ibTransport) {
  // Find IB context or create it
  auto it = ibContexts.find(ibTransport);
  if (it == ibContexts.end()) {
    auto ibDev = getIBDeviceName(ibTransport);
    IbCtx* ibCtx = new IbCtx(ibDev);
    ibContexts[ibTransport] = ibCtx;
    return ibCtx;
  } else {
    return it->second;
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

MSCCLPP_API_CPP Communicator::Communicator(int nranks, const char* ipPortPair, int rank) : pimpl(std::make_unique<Impl>()) {
  mscclppCommInitRank(&pimpl->comm, nranks, ipPortPair, rank);
}

MSCCLPP_API_CPP Communicator::Communicator(int nranks, UniqueId id, int rank) : pimpl(std::make_unique<Impl>()) {
  static_assert(sizeof(mscclppUniqueId) == sizeof(UniqueId), "UniqueId size mismatch");
  mscclppUniqueId *cstyle_id = reinterpret_cast<mscclppUniqueId*>(&id);
  mscclppCommInitRankFromId(&pimpl->comm, nranks, *cstyle_id, rank);
}

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
    auto cudaIpcConn = std::make_shared<CudaIpcConnection>();
    conn = cudaIpcConn;
  } else if (transport | TransportAllIB) {
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
    conn->startSetup(*this);
  }
  for (auto& conn : pimpl->connections) {
    conn->endSetup(*this);
  }
}

MSCCLPP_API_CPP int Communicator::rank() {
  int result;
  mscclppCommRank(pimpl->comm, &result);
  return result;
}

MSCCLPP_API_CPP int Communicator::size() {
  int result;
  mscclppCommSize(pimpl->comm, &result);
  return result;
}

} // namespace mscclpp
