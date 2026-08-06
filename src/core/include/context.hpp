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
  int deviceId_;
  bool dirty_;

  void setStreamIfNeeded();

 public:
  CudaIpcStream(int deviceId);

  void memcpyD2D(void* dst, const void* src, size_t nbytes);

  void memcpyH2D(void* dst, const void* src, size_t nbytes);

#if defined(MSCCLPP_USE_ROCM)
  /// Add a value to a 64-bit integer in peer memory, with a kernel on this stream. ROCm only:
  /// on CUDA such a kernel cannot be scheduled while the caller's kernel spins.
  void accumulate(int64_t* dst, int64_t value);
#endif  // defined(MSCCLPP_USE_ROCM)

  void sync();

  operator cudaStream_t() const { return *stream_; }

  int deviceId() const { return deviceId_; }
};

class TokenPool;
struct Context::Impl {
  std::unordered_map<Transport, std::unique_ptr<IbCtx>> ibContexts_;
  std::vector<std::shared_ptr<CudaIpcStream>> ipcStreams_;
  std::shared_ptr<TokenPool> tokenPool_;
  const size_t maxNumTokens_ = 1 << 15;  // 32K tokens

  IbCtx* getIbContext(Transport ibTransport);
  std::shared_ptr<uint64_t> getToken();
};

}  // namespace mscclpp

#endif  // MSCCLPP_CONTEXT_HPP_
