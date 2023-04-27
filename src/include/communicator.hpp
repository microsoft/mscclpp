#ifndef MSCCL_COMMUNICATOR_HPP_
#define MSCCL_COMMUNICATOR_HPP_

#include "mscclpp.hpp"
#include "mscclpp.h"
#include "channel.hpp"
#include "proxy.hpp"
#include "ib.hpp"
#include <unordered_map>
#include <memory>

namespace mscclpp {

class ConnectionBase;

struct Communicator::Impl {
  mscclppComm_t comm;
  std::vector<std::shared_ptr<ConnectionBase>> connections;
  std::unordered_map<TransportFlags, std::unique_ptr<IbCtx>> ibContexts;
  std::shared_ptr<BaseBootstrap> bootstrap_;

  Impl(std::shared_ptr<BaseBootstrap> bootstrap);

  ~Impl();

  IbCtx* getIbContext(TransportFlags ibTransport);
};

} // namespace mscclpp

#endif // MSCCL_COMMUNICATOR_HPP_
