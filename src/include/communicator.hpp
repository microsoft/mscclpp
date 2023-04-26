#ifndef MSCCL_COMMUNICATOR_HPP_
#define MSCCL_COMMUNICATOR_HPP_

#include "mscclpp.hpp"
#include "mscclpp.h"
#include "channel.hpp"
#include "proxy.hpp"
#include "ib.h"
#include <unordered_map>

namespace mscclpp {

class ConnectionBase;

struct Communicator::Impl {
  mscclppComm_t comm;
  std::vector<std::shared_ptr<ConnectionBase>> connections;
  std::unordered_map<TransportFlags, mscclppIbContext*> ibContexts;

  Impl();

  ~Impl();

  mscclppIbContext* getIbContext(TransportFlags ibTransport);
};

} // namespace mscclpp

#endif // MSCCL_COMMUNICATOR_HPP_
