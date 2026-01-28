// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/algorithm.hpp>

#include "allreduce/common.hpp"

namespace mscclpp {
namespace collective {
class AllreducePacket : public AlgorithmBuilder {
 public:
  AllreducePacket(uintptr_t scratchBuffer, size_t scratchBufferSize)
      : scratchBuffer_((void*)scratchBuffer), scratchBufferSize_(scratchBufferSize){};
  std::shared_ptr<Algorithm> build() override;

 private:
  void initialize(std::shared_ptr<Communicator> comm);
  CommResult allreduceKernelFunc(const std::shared_ptr<void> ctx, const void* input, void* output, size_t inputSize,
                                 DataType dtype, ReduceOp op, cudaStream_t stream, int nBlocks, int nThreadsPerBlock,
                                 const std::unordered_map<std::string, uintptr_t>& extras);

  std::shared_ptr<void> initAllreduceContext(std::shared_ptr<Communicator> comm, const void*, void* output, size_t,
                                             DataType);
  AlgorithmCtxKey generateAllreduceContextKey(const void*, void*, size_t, DataType, bool);

  void* scratchBuffer_;
  size_t scratchBufferSize_;
  const int nSegmentsForScratchBuffer_ = 2;
  const int maxBlockNum_ = 56;
  std::vector<Connection> conns_;
  std::vector<std::shared_ptr<MemoryDevice2DeviceSemaphore>> memorySemaphores_;
  std::vector<RegisteredMemory> registeredMemories_;
  std::shared_ptr<LL8Packet> flags_;
};
}  // namespace collective
}  // namespace mscclpp