#include "mscclpp.h"
#include "mscclpp.hpp"

namespace mscclpp {

mscclppTransport_t transportTypeToCStyle(TransportType type)
{
  switch (type) {
  case TransportType::IB:
    return mscclppTransportIB;
  case TransportType::P2P:
    return mscclppTransportP2P;
  default:
    throw std::runtime_error("Unknown transport type");
  }
}

struct Communicator::Impl
{
  mscclppComm_t comm;
  std::vector<std::shared_ptr<HostConnection>> connections;

  Impl() : comm(nullptr)
  {
  }

  ~Impl()
  {
    if (comm) {
      mscclppCommDestroy(comm);
    }
  }
};

void Communicator::initRank(int nranks, const char* ipPortPair, int rank)
{
  if (pimpl) {
    throw std::runtime_error("Communicator already initialized");
  }
  pimpl = std::make_unique<Impl>();
  mscclppCommInitRank(&pimpl->comm, nranks, ipPortPair, rank);
}

void Communicator::initRankFromId(int nranks, UniqueId id, int rank)
{
  if (pimpl) {
    throw std::runtime_error("Communicator already initialized");
  }
  pimpl = std::make_unique<Impl>();
  static_assert(sizeof(mscclppUniqueId) == sizeof(UniqueId), "UniqueId size mismatch");
  mscclppUniqueId* cstyle_id = reinterpret_cast<mscclppUniqueId*>(&id);
  mscclppCommInitRankFromId(&pimpl->comm, nranks, *cstyle_id, rank);
}

void Communicator::bootstrapAllGather(void* data, int size)
{
  mscclppBootstrapAllGather(pimpl->comm, data, size);
}

void Communicator::bootstrapBarrier()
{
  mscclppBootstrapBarrier(pimpl->comm);
}

std::shared_ptr<HostConnection> Communicator::connect(int remoteRank, int tag, TransportType transportType,
                                                      const char* ibDev = 0)
{
  mscclppConnectWithoutBuffer(pimpl->comm, remoteRank, tag, transportTypeToCStyle(transportType), ibDev);
  auto conn = std::make_shared<HostConnection>();
  auto connIdx = pimpl->connections.size();
  pimpl->connections.push_back(conn);
  return conn;
}

void Communicator::connectionSetup()
{
  mscclppConnectionSetup(pimpl->comm);
  mscclppHostConn_t* hostConns;
  int numHostConns;
  mscclppGetAllHostConnections(pimpl->comm, &hostConns, &numHostConns);
  if (numHostConns != pimpl->connections.size()) {
    throw std::logic_error("Number of HostConnections didn't match number of mscclppHostConns");
  }
  for (int connIdx = 0; connIdx < pimpl->connections.size(); ++connIdx) {
    pimpl->connections[connIdx]->pimpl->setup(hostConns[connIdx]);
  }
}

int Communicator::rank()
{
  int result;
  mscclppCommRank(pimpl->comm, &result);
  return result;
}

int Communicator::size()
{
  int result;
  mscclppCommSize(pimpl->comm, &result);
  return result;
}

} // namespace mscclpp