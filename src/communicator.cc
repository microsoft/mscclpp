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

MSCCLPP_API_CPP std::shared_ptr<HostConnection> Communicator::connect(int remoteRank, int tag,
                                                      TransportType transportType, const char* ibDev) {
  mscclppConnectWithoutBuffer(pimpl->comm, remoteRank, tag, transportTypeToCStyle(transportType), ibDev);
  auto connIdx = pimpl->connections.size();
  auto conn = std::make_shared<HostConnection>(std::make_unique<HostConnection::Impl>(this, &pimpl->comm->conns[connIdx]));
  pimpl->connections.push_back(conn);
  return conn;
}

MSCCLPP_API_CPP void Communicator::connectionSetup() {
  mscclppConnectionSetup(pimpl->comm);
}

MSCCLPP_API_CPP void Communicator::startProxying() {
  pimpl->proxy.start();
}

MSCCLPP_API_CPP void Communicator::stopProxying() {
  pimpl->proxy.stop();
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