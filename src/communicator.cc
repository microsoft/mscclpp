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

namespace mscclpp {

Communicator::Impl::Impl() : comm(nullptr) {}

Communicator::Impl::~Impl() {
  for (auto& entry : ibContexts) {
    mscclppIbContextDestroy(entry.second);
  }
  ibContexts.clear();
  if (comm) {
    mscclppCommDestroy(comm);
  }
}

mscclppIbContext* Communicator::Impl::getIbContext(TransportFlags ibTransport) {
  // Find IB context or create it
  auto it = ibContexts.find(ibTransport);
  if (it == ibContexts.end()) {
    auto ibDev = getIBDeviceName(ibTransport);
    mscclppIbContext* ibCtx;
    MSCCLPPTHROW(mscclppIbContextCreate(&ibCtx, ibDev.c_str()));
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

// TODO: move these elsewhere

int getIBDeviceCount() {
  int num;
  struct ibv_device** devices = ibv_get_device_list(&num);
  return num;
}

std::string getIBDeviceName(TransportFlags ibTransport) {
  int num;
  struct ibv_device** devices = ibv_get_device_list(&num);
  int ibTransportIndex;
  switch (ibTransport) { // TODO: get rid of this ugly switch
    case TransportIB0:
      ibTransportIndex = 0;
      break;
    case TransportIB1:
      ibTransportIndex = 1;
      break;
    case TransportIB2:
      ibTransportIndex = 2;
      break;
    case TransportIB3:
      ibTransportIndex = 3;
      break;
    case TransportIB4:
      ibTransportIndex = 4;
      break;
    case TransportIB5:
      ibTransportIndex = 5;
      break;
    case TransportIB6:
      ibTransportIndex = 6;
      break;
    case TransportIB7:
      ibTransportIndex = 7;
      break;
    default:
      throw std::runtime_error("Not an IB transport");
  }
  if (ibTransportIndex >= num) {
    throw std::runtime_error("IB transport out of range");
  }
  return devices[ibTransportIndex]->name;
}

TransportFlags getIBTransportByDeviceName(const std::string& ibDeviceName) {
  int num;
  struct ibv_device** devices = ibv_get_device_list(&num);
  for (int i = 0; i < num; ++i) {
    if (ibDeviceName == devices[i]->name) {
      switch (i) { // TODO: get rid of this ugly switch
        case 0:
          return TransportIB0;
        case 1:
          return TransportIB1;
        case 2:
          return TransportIB2;
        case 3:
          return TransportIB3;
        case 4:
          return TransportIB4;
        case 5:
          return TransportIB5;
        case 6:
          return TransportIB6;
        case 7:
          return TransportIB7;
        default:
          throw std::runtime_error("IB device index out of range");
      }
    }
  }
  throw std::runtime_error("IB device not found");
}


} // namespace mscclpp