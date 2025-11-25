// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/algorithm.hpp>

#include "algorithms/allreduce/common.hpp"

namespace mscclpp {
namespace algorithm {
class AllreducePacket : public AlgorithmBuilder {
 public:
  AllreducePacket(uintptr_t scratchBuffer, size_t scratchBufferSize)
      : scratchBuffer_((void*)scratchBuffer), scratchBufferSize_(scratchBufferSize) {};
  std::shared_ptr<Algorithm> build() override;
 private:
  void initialize(std::shared_ptr<Communicator> comm);
  CommResult allreduceKernelFunc(const std::shared_ptr<AlgorithmCtx> ctx, const void* input, void* output,
                                 size_t inputSize, DataType dtype, cudaStream_t stream,
                                 std::unordered_map<std::string, uintptr_t>& extras);

  std::shared_ptr<AlgorithmCtx> initAllreduceContext(std::shared_ptr<Communicator> comm, const void*, void* output,
                                                     size_t, DataType);
  AlgorithmCtxKey generateAllreduceContextKey(const void*, void*, size_t, DataType);

  size_t scratchBufferSize_;
  void* scratchBuffer_;
  const int nSegmentsForScratchBuffer_ = 2;
  const int maxBlockNum_ = 28;
  std::vector<Connection> conns_;
  std::vector<std::shared_ptr<MemoryDevice2DeviceSemaphore>> memorySemaphores_;
  std::vector<RegisteredMemory> registeredMemories_;
  std::shared_ptr<LL8Packet> flags_;
};
}  // namespace algorithm
}  // namespace mscclpp