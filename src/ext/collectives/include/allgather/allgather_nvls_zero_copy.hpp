// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_ALLGATHER_NVLS_ZERO_COPY_HPP_
#define MSCCLPP_ALLGATHER_NVLS_ZERO_COPY_HPP_

#include <mscclpp/algorithm.hpp>

namespace mscclpp {
namespace collective {

class AllgatherNvls : public AlgorithmBuilder {
 public:
  AllgatherNvls() = default;
  std::shared_ptr<Algorithm> build() override;

 private:
  bool symmetricMemory_ = false;
  void initialize(std::shared_ptr<Communicator> comm);
  CommResult allgatherKernelFunc(const std::shared_ptr<void> ctx, const void* input, void* output, size_t inputSize,
                                 cudaStream_t stream, int nBlocks, int nThreadsPerBlock,
                                 const std::unordered_map<std::string, uintptr_t>& extras);

  std::shared_ptr<void> initAllgatherContext(std::shared_ptr<Communicator> comm, const void* input, void* output,
                                             size_t inputSize, DataType);
  AlgorithmCtxKey generateAllgatherContextKey(const void*, void* output, size_t, DataType, bool);

  // Large buffer size because cuMemMap requires offset=0 for multicast handles, so the entire
  // user allocation must be mapped. This only reserves virtual address space; no physical memory
  // is consumed beyond what is actually bound.
  const size_t nvlsBufferSize_ = (1UL << 34);
  uint32_t nSwitchChannels_{0};
  std::shared_ptr<DeviceHandle<BaseMemoryChannel>> memoryChannelsDeviceHandle_;
  std::vector<BaseMemoryChannel> baseChannels_;
  std::vector<Connection> conns_;
  std::vector<std::shared_ptr<NvlsConnection>> nvlsConnections_;
  int computeCapabilityMajor_{0};
};

}  // namespace collective
}  // namespace mscclpp

#endif  // MSCCLPP_ALLGATHER_NVLS_ZERO_COPY_HPP_
