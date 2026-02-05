// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_ALLREDUCE_NVLS_PACKET_HPP_
#define MSCCLPP_ALLREDUCE_NVLS_PACKET_HPP_

#include <mscclpp/algorithm.hpp>

namespace mscclpp {
namespace collective {
class AllreduceNvlsPacket : public mscclpp::AlgorithmBuilder {
 public:
  AllreduceNvlsPacket(uintptr_t scratchBuffer, size_t scratchBufferSize)
      : scratchBuffer_((void*)scratchBuffer), scratchBufferSize_(scratchBufferSize){};
  std::shared_ptr<mscclpp::Algorithm> build() override;

 private:
  void initialize(std::shared_ptr<mscclpp::Communicator> comm);
  CommResult allreduceKernelFunc(const std::shared_ptr<void> ctx, const void* input, void* output, size_t inputSize,
                                 mscclpp::DataType dtype, ReduceOp op, cudaStream_t stream, int nBlocks,
                                 int nThreadsPerBlock, const std::unordered_map<std::string, uintptr_t>& extras);

  std::shared_ptr<void> initAllreduceContext(std::shared_ptr<mscclpp::Communicator> comm, const void*, void* output,
                                             size_t, mscclpp::DataType);
  mscclpp::AlgorithmCtxKey generateAllreduceContextKey(const void*, void*, size_t, mscclpp::DataType);

  void* scratchBuffer_;
  size_t scratchBufferSize_;
  const size_t nvlsBufferSize_ = (1 << 30);
  const int maxBlockNum_ = 16;
  std::shared_ptr<LL8Packet> flags_;
  std::shared_ptr<uint32_t> flags4_;
  std::shared_ptr<uint32_t> flags8_;
};
}  // namespace collective
}  // namespace mscclpp
#endif