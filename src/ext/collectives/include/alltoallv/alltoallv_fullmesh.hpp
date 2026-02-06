// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <mscclpp/algorithm.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/port_channel.hpp>

namespace mscclpp {
namespace collective {

/**
 * AllToAllV collective operation builder.
 *
 * This class builds an AllToAllV algorithm that handles variable element counts
 * per rank, similar to MPI_Alltoallv. Unlike NCCL's ncclGroupStart/ncclGroupEnd
 * approach, mscclpp uses explicit put/signal/wait operations on PortChannels.
 *
 * The implementation uses a ring-based exchange pattern to avoid deadlocks.
 *
 * Usage:
 *   auto builder = std::make_shared<AlltoallvFullmesh>();
 *   auto algorithm = builder->build();
 *   // Then execute with extras containing sendCounts, sendDispls, recvCounts, recvDispls
 */
class AlltoallvFullmesh : public AlgorithmBuilder {
 public:
  AlltoallvFullmesh() = default;
  ~AlltoallvFullmesh();

  std::shared_ptr<Algorithm> build() override;

 private:
  void initialize(std::shared_ptr<Communicator> comm);

  CommResult alltoallvKernelFunc(const std::shared_ptr<void> ctx, const void* input, void* output,
                                 size_t inputSize, size_t outputSize, DataType dtype, cudaStream_t stream,
                                 int nBlocks, int nThreadsPerBlock,
                                 const std::unordered_map<std::string, uintptr_t>& extras);

  std::shared_ptr<void> initAlltoallvContext(std::shared_ptr<Communicator> comm, const void* input,
                                             void* output, size_t inputSize, size_t outputSize,
                                             DataType dtype);

  AlgorithmCtxKey generateAlltoallvContextKey(const void* input, void* output, size_t inputSize,
                                              size_t outputSize, DataType dtype);

  std::vector<Connection> conns_;
  std::shared_ptr<ProxyService> proxyService_;
  int worldSize_;
};

}  // namespace collective
}  // namespace mscclpp
