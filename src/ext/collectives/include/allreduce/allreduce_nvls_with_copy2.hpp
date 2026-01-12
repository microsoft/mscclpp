// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <mscclpp/algorithm.hpp>

namespace mscclpp {
namespace collective {

class AllreduceNvlsWithCopy2 : public AlgorithmBuilder {
 public:
  AllreduceNvlsWithCopy2(uintptr_t scratchBuffer, size_t scratchBufferSize)
      : scratchBuffer_(reinterpret_cast<void*>(scratchBuffer)), scratchBufferSize_(scratchBufferSize){};
  std::shared_ptr<Algorithm> build() override;

 private:
  void initialize(std::shared_ptr<Communicator> comm);
  CommResult allreduceKernelFunc(const std::shared_ptr<AlgorithmCtx> ctx, const void* input, void* output,
                                 size_t inputSize, DataType dtype, ReduceOp op, cudaStream_t stream, int nBlocks,
                                 int nThreadsPerBlock, const std::unordered_map<std::string, uintptr_t>& extras);

  std::shared_ptr<AlgorithmCtx> initAllreduceContext(std::shared_ptr<Communicator> comm, const void*, void* output,
                                                     size_t, DataType);
  AlgorithmCtxKey generateAllreduceContextKey(const void*, void*, size_t, DataType);

  const size_t nvlsBufferSize_ = (1 << 30);
  void* scratchBuffer_;
  size_t scratchBufferSize_;
  uint32_t nSwitchChannels_;
  std::shared_ptr<DeviceHandle<BaseMemoryChannel>> memoryChannelsDeviceHandle_;
  std::vector<BaseMemoryChannel> baseChannels_;
  std::vector<Connection> conns_;
};
}  // namespace collective
}  // namespace mscclpp