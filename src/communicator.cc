#include <sstream>

#include "api.h"
#include "checks.hpp"
#include "comm.h"
#include "communicator.hpp"
#include "connection.hpp"
#include "debug.h"
#include "mscclpp.hpp"
#include "registered_memory.hpp"
#include "utils.h"

namespace mscclpp {

Communicator::Impl::Impl(std::shared_ptr<BaseBootstrap> bootstrap) : bootstrap_(bootstrap)
{
  rankToHash_.resize(bootstrap->getNranks());
  auto hostHash = getHostHash();
  INFO(MSCCLPP_INIT, "Host hash: %lx", hostHash);
  rankToHash_[bootstrap->getRank()] = hostHash;
  bootstrap->allGather(rankToHash_.data(), sizeof(uint64_t));
}

Communicator::Impl::~Impl()
{
  ibContexts.clear();
}

IbCtx* Communicator::Impl::getIbContext(Transport ibTransport)
{
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

MSCCLPP_API_CPP Communicator::Communicator(std::shared_ptr<BaseBootstrap> bootstrap)
  : pimpl(std::make_unique<Impl>(bootstrap))
{
}

MSCCLPP_API_CPP std::shared_ptr<BaseBootstrap> Communicator::bootstrapper()
{
  return pimpl->bootstrap_;
}

MSCCLPP_API_CPP RegisteredMemory Communicator::registerMemory(void* ptr, size_t size, TransportFlags transports)
{
  return RegisteredMemory(
    std::make_shared<RegisteredMemory::Impl>(ptr, size, pimpl->bootstrap_->getRank(), transports, *pimpl));
}

MSCCLPP_API_CPP std::shared_ptr<Connection> Communicator::connect(int remoteRank, int tag, Transport transport)
{
  std::shared_ptr<ConnectionBase> conn;
  if (transport == Transport::CudaIpc) {
    // sanity check: make sure the IPC connection is being made within a node
    if (pimpl->rankToHash_[remoteRank] != pimpl->rankToHash_[pimpl->bootstrap_->getRank()]) {
      std::stringstream ss;
      ss << "Cuda IPC connection can only be made within a node: " << remoteRank << "(" << std::hex
         << pimpl->rankToHash_[pimpl->bootstrap_->getRank()] << ")"
         << " != " << pimpl->bootstrap_->getRank() << "(" << std::hex
         << pimpl->rankToHash_[pimpl->bootstrap_->getRank()] << ")";
      throw std::runtime_error(ss.str());
    }
    auto cudaIpcConn = std::make_shared<CudaIpcConnection>(remoteRank, tag);
    conn = cudaIpcConn;
    INFO(MSCCLPP_P2P, "Cuda IPC connection between rank %d(%lx) and remoteRank %d(%lx) created",
         pimpl->bootstrap_->getRank(), pimpl->rankToHash_[pimpl->bootstrap_->getRank()], remoteRank,
         pimpl->rankToHash_[remoteRank]);
  } else if (AllIBTransports.has(transport)) {
    auto ibConn = std::make_shared<IBConnection>(remoteRank, tag, transport, *pimpl);
    conn = ibConn;
    INFO(MSCCLPP_NET, "IB connection between rank %d(%lx) via %s and remoteRank %d(%lx) created",
         pimpl->bootstrap_->getRank(), pimpl->rankToHash_[pimpl->bootstrap_->getRank()],
         getIBDeviceName(transport).c_str(), remoteRank, pimpl->rankToHash_[remoteRank]);
  } else {
    throw std::runtime_error("Unsupported transport");
  }
  pimpl->connections.push_back(conn);
  return conn;
}

MSCCLPP_API_CPP void Communicator::connectionSetup()
{
  for (auto& conn : pimpl->connections) {
    conn->startSetup(pimpl->bootstrap_);
  }
  for (auto& conn : pimpl->connections) {
    conn->endSetup(pimpl->bootstrap_);
  }
}

} // namespace mscclpp
