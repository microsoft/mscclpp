// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCL_CONTEXT_HPP_
#define MSCCL_CONTEXT_HPP_

#include <cuda_runtime.h>

#include <mscclpp/core.hpp>
#include <mscclpp/cuda_utils.hpp>
#include <unordered_map>

#include "ib.hpp"

namespace mscclpp {

class ConnectionBase;

struct Communicator::Impl {
  std::vector<std::shared_ptr<ConnectionBase>> connections_;
  std::unordered_map<Transport, std::unique_ptr<IbCtx>> ibContexts_;
  CudaStreamWithFlags ipcStream_;
  uint64_t hostHash_;

  Impl();

  IbCtx* getIbContext(Transport ibTransport);
};

}  // namespace mscclpp

#endif  // MSCCL_CONTEXT_HPP_
