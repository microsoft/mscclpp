// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/algorithm.hpp>

#include "allreduce_common.hpp"

namespace mscclpp {

class AllreducePacket : public mscclpp::AlgorithmBuilder {
 public:
  AllreducePacket(std::shared_ptr<void> scratchBuffer, size_t scratchBufferSize)
      : scratchBuffer_(std::static_pointer_cast<char>(scratchBuffer)), scratchBufferSize_(scratchBufferSize){};
  std::shared_ptr<mscclpp::Algorithm> build() override;

 private:
  void initialize(std::shared_ptr<mscclpp::Communicator> comm);
  CommResult allreduceKernelFunc(const std::shared_ptr<mscclpp::AlgorithmCtx> ctx, const void* input, void* output,
                                 size_t inputSize, mscclpp::DataType dtype, cudaStream_t stream,
                                 std::unordered_map<std::string, uintptr_t>& extras);

  std::shared_ptr<mscclpp::AlgorithmCtx> initAllreduceContext(std::shared_ptr<mscclpp::Communicator> comm, const void*,
                                                              void* output, size_t, mscclpp::DataType);
  mscclpp::AlgorithmCtxKey generateAllreduceContextKey(const void*, void*, size_t, mscclpp::DataType);

  size_t scratchBufferSize_;
  std::weak_ptr<char> scratchBuffer_;
  const int nSegmentsForScratchBuffer_ = 2;
  const int maxBlockNum_ = 56;
  std::vector<mscclpp::Connection> conns_;

  std::shared_ptr<uint32_t> deviceFlag7_;
  std::shared_ptr<uint32_t> deviceFlag28_;
  std::shared_ptr<uint32_t> deviceFlag56_;
  std::shared_ptr<mscclpp::AlgorithmCtx> ctx_;
};
}  // namespace mscclpp