#ifndef MSCCL_COMMUNICATOR_HPP_
#define MSCCL_COMMUNICATOR_HPP_

#include "ib.hpp"
#include "mscclpp.h"
#include <mscclpp/core.hpp>
#include <mscclpp/proxy.hpp>
#include <memory>
#include <unordered_map>

namespace mscclpp {

class ConnectionBase;

struct Communicator::Impl
{
  std::vector<std::shared_ptr<ConnectionBase>> connections_;
  std::vector<std::shared_ptr<Setuppable>> toSetup_;
  std::unordered_map<Transport, std::unique_ptr<IbCtx>> ibContexts_;
  std::shared_ptr<BaseBootstrap> bootstrap_;
  std::vector<uint64_t> rankToHash_;

  Impl(std::shared_ptr<BaseBootstrap> bootstrap);

  ~Impl();

  IbCtx* getIbContext(Transport ibTransport);
};

} // namespace mscclpp

#endif // MSCCL_COMMUNICATOR_HPP_
