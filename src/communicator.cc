#include "communicator.hpp"
#include "host_connection.hpp"
#include "comm.h"
#include "basic_proxy_handler.hpp"
#include "api.h"

namespace mscclpp {

Communicator::Impl::Impl() : comm(nullptr), proxy(makeBasicProxyHandler(*this)) {}

Communicator::Impl::~Impl() {
  if (comm) {
    mscclppCommDestroy(comm);
  }
}

MSCCLPP_API_CPP Communicator::Communicator() = default;
MSCCLPP_API_CPP Communicator::~Communicator() = default;

mscclppTransport_t transportTypeToCStyle(TransportType type) {
  switch (type) {
    case TransportType::IB:
      return mscclppTransportIB;
    case TransportType::P2P:
      return mscclppTransportP2P;
    default:
      throw std::runtime_error("Unknown transport type");
  }
}

MSCCLPP_API_CPP void Communicator::initRank(int nranks, const char* ipPortPair, int rank) {
  if (pimpl) {
    throw std::runtime_error("Communicator already initialized");
  }
  pimpl = std::make_unique<Impl>();
  mscclppCommInitRank(&pimpl->comm, nranks, ipPortPair, rank);
}

MSCCLPP_API_CPP void Communicator::initRankFromId(int nranks, UniqueId id, int rank) {
  if (pimpl) {
    throw std::runtime_error("Communicator already initialized");
  }
  pimpl = std::make_unique<Impl>();
  static_assert(sizeof(mscclppUniqueId) == sizeof(UniqueId), "UniqueId size mismatch");
  mscclppUniqueId *cstyle_id = reinterpret_cast<mscclppUniqueId*>(&id);
  mscclppCommInitRankFromId(&pimpl->comm, nranks, *cstyle_id, rank);
}

MSCCLPP_API_CPP void Communicator::bootstrapAllGather(void* data, int size) {
  if (!pimpl) {
    throw std::runtime_error("Communicator not initialized");
  }
  mscclppBootstrapAllGather(pimpl->comm, data, size);
}

MSCCLPP_API_CPP void Communicator::bootstrapBarrier() {
  if (!pimpl) {
    throw std::runtime_error("Communicator not initialized");
  }
  mscclppBootstrapBarrier(pimpl->comm);
}

MSCCLPP_API_CPP std::shared_ptr<HostConnection> Communicator::connect(int remoteRank, int tag,
                                                      TransportType transportType, const char* ibDev) {
  if (!pimpl) {
    throw std::runtime_error("Communicator not initialized");
  }
  mscclppConnectWithoutBuffer(pimpl->comm, remoteRank, tag, transportTypeToCStyle(transportType), ibDev);
  auto connIdx = pimpl->connections.size();
  auto conn = std::make_shared<HostConnection>(std::make_unique<HostConnection::Impl>(this, &pimpl->comm->conns[connIdx]));
  pimpl->connections.push_back(conn);
  return conn;
}

MSCCLPP_API_CPP void Communicator::connectionSetup() {
  if (!pimpl) {
    throw std::runtime_error("Communicator not initialized");
  }
  mscclppConnectionSetup(pimpl->comm);
}

MSCCLPP_API_CPP void Communicator::startProxying() {
  if (!pimpl) {
    throw std::runtime_error("Communicator not initialized");
  }
  pimpl->proxy.start();
}

MSCCLPP_API_CPP void Communicator::stopProxying() {
  if (!pimpl) {
    throw std::runtime_error("Communicator not initialized");
  }
  pimpl->proxy.stop();
}

MSCCLPP_API_CPP int Communicator::rank() {
  if (!pimpl) {
    throw std::runtime_error("Communicator not initialized");
  }
  int result;
  mscclppCommRank(pimpl->comm, &result);
  return result;
}

MSCCLPP_API_CPP int Communicator::size() {
  if (!pimpl) {
    throw std::runtime_error("Communicator not initialized");
  }
  int result;
  mscclppCommSize(pimpl->comm, &result);
  return result;
}

} // namespace mscclpp