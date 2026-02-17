// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <mscclpp/algorithm.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/semaphore.hpp>

namespace mscclpp {
namespace collective {

/**
 * AllToAllV collective operation builder.
 *
 * This class builds an AllToAllV algorithm that handles variable element counts
 * per rank. Uses direct MemoryChannel for low-latency
 * intra-node communication instead of proxy-based PortChannel.
 *
 * The implementation uses a parallel warp-based exchange pattern where each warp
 * handles communication with one peer for maximum throughput.
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
  int worldSize_;
};

}  // namespace collective
}  // namespace mscclpp
