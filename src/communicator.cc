#include "mscclpp.hpp"
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

MSCCLPP_API_CPP std::shared_ptr<Connection> Communicator::connect(int remoteRank, int tag, TransportFlags transport) {
  std::string ibDev;
  switch (transport) {
    case TransportIB0:
    case TransportIB1:
    case TransportIB2:
    case TransportIB3:
    case TransportIB4:
    case TransportIB5:
    case TransportIB6:
    case TransportIB7:
      ibDev = getIBDeviceName(transport);
      break;
  }
  mscclppConnectWithoutBuffer(pimpl->comm, remoteRank, tag, transportToCStyle(transport), ibDev.c_str());
  auto connIdx = pimpl->connections.size();
  auto conn = std::make_shared<Connection>(std::make_unique<Connection::Impl>(this, &pimpl->comm->conns[connIdx]));
  pimpl->connections.push_back(conn);
  return conn;
}

MSCCLPP_API_CPP void Communicator::connectionSetup() {
  mscclppConnectionSetup(pimpl->comm);
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