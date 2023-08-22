// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCL_COMMUNICATOR_HPP_
#define MSCCL_COMMUNICATOR_HPP_

#include <cuda_runtime.h>

#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/proxy.hpp>
#include <unordered_map>

#include "ib.hpp"

namespace mscclpp {

class ConnectionBase;

struct ConnectionInfo {
  int remoteRank;
  int tag;
};

struct Communicator::Impl {
  std::shared_ptr<Bootstrap> bootstrap_;
  std::shared_ptr<Context> context_;
  std::unordered_map<const Connection*, ConnectionInfo> connectionInfos_;
  std::vector<std::shared_ptr<Setuppable>> toSetup_;
  std::vector<uint64_t> rankToHash_;

  Impl(std::shared_ptr<Bootstrap> bootstrap, std::shared_ptr<Context> context);
};

}  // namespace mscclpp

#endif  // MSCCL_COMMUNICATOR_HPP_
