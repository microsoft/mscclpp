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

struct Communicator::Impl {
  std::vector<std::shared_ptr<ConnectionBase>> connections_;
  std::vector<std::shared_ptr<Setuppable>> toSetup_;
  std::unordered_map<Transport, std::unique_ptr<IbCtx>> ibContexts_;
  cudaStream_t ipcStream_;
  std::shared_ptr<BaseBootstrap> bootstrap_;
  std::vector<uint64_t> rankToHash_;

  Impl(std::shared_ptr<BaseBootstrap> bootstrap);

  ~Impl();

  IbCtx* getIbContext(Transport ibTransport);
  cudaStream_t getIpcStream();
};

}  // namespace mscclpp

#endif  // MSCCL_COMMUNICATOR_HPP_
