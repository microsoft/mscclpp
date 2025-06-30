// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_CONTEXT_HPP_
#define MSCCLPP_CONTEXT_HPP_

#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <unordered_map>
#include <vector>

#include "ib.hpp"

namespace mscclpp {

class CudaIpcStream {
 private:
  std::shared_ptr<CudaStreamWithFlags> stream_;
  bool dirty_;

  void setStreamIfNeeded();

 public:
  CudaIpcStream();

  void memcpyD2D(void *dst, const void *src, size_t nbytes);

  void memcpyH2D(void *dst, const void *src, size_t nbytes);

  void sync();

  operator cudaStream_t() const { return *stream_; }
};

struct Context::Impl {
  std::unordered_map<Transport, std::unique_ptr<IbCtx>> ibContexts_;
  std::vector<std::shared_ptr<CudaIpcStream>> ipcStreams_;

  Impl();

  IbCtx *getIbContext(Transport ibTransport);
};

}  // namespace mscclpp

#endif  // MSCCLPP_CONTEXT_HPP_
