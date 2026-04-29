// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <ATen/ATen.h>

#include <comms/torchcomms/TorchCommTypes.hpp>
#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu.hpp>

namespace torch::comms {

/// dlopen-based NCCL fallback for collectives MSCCL++ does not natively
/// implement (reduce_scatter, broadcast, barrier on certain configs).
///
/// The backend creates one of these in init() via tryCreate(). If libnccl
/// can't be found or its symbols can't be resolved, tryCreate() returns
/// nullptr and the backend operates without a fallback (unsupported
/// collectives then throw).
///
/// All ABI/dlsym ugliness lives in NcclFallback.cpp; the backend never
/// touches NCCL types or symbols directly.
class NcclFallback {
 public:
  /// Try to dlopen libnccl.so.2, resolve the symbols we need, and create a
  /// parallel NCCL communicator. Returns nullptr (without throwing) if any
  /// step fails — callers should treat that as "fallback unavailable".
  ///
  /// Search order for the shared library:
  ///   1. $MSCCLPP_NCCL_LIB_PATH (matches src/ext/nccl/nccl.cc behavior)
  ///   2. libnccl.so.2 (rtld-resolved; finds PyTorch's bundled NCCL)
  ///   3. libnccl.so
  static std::unique_ptr<NcclFallback> tryCreate(const std::shared_ptr<mscclpp::Communicator>& comm, int rank,
                                                 int worldSize);

  ~NcclFallback();

  NcclFallback(const NcclFallback&) = delete;
  NcclFallback& operator=(const NcclFallback&) = delete;

  /// reduce_scatter: input is the full buffer, output is rank's chunk.
  /// recvCount is the number of elements in the per-rank output chunk.
  void reduceScatter(const void* sendbuf, void* recvbuf, size_t recvCount, at::ScalarType dtype, const ReduceOp& op,
                     cudaStream_t stream);

  /// broadcast from `root` to all ranks. count is element count.
  void broadcast(const void* sendbuf, void* recvbuf, size_t count, at::ScalarType dtype, int root,
                 cudaStream_t stream);

  /// barrier emulated as a 1-element ncclAllReduce on a persistent device int.
  void barrier(cudaStream_t stream);

 private:
  NcclFallback() = default;

  // Opaque to the header — concrete state lives in NcclFallback.cpp.
  void* dlHandle_ = nullptr;
  void* ncclComm_ = nullptr;
  void* getUniqueIdFn_ = nullptr;
  void* commInitRankFn_ = nullptr;
  void* commDestroyFn_ = nullptr;
  void* reduceScatterFn_ = nullptr;
  void* broadcastFn_ = nullptr;
  void* allReduceFn_ = nullptr;
  void* barrierBuf_ = nullptr;  // persistent 4-byte device buffer for barrier()
};

}  // namespace torch::comms
