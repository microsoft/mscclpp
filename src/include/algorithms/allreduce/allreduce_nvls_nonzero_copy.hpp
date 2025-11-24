#include <mscclpp/algorithm.hpp>

namespace mscclpp {
class AllreduceNvlsWithCopy : public mscclpp::AlgorithmBuilder {
 public:
  AllreduceNvlsWithCopy(std::shared_ptr<void> scratchBuffer, size_t scratchBufferSize)
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

  const size_t nvlsBufferSize_ = (1 << 30);
  size_t scratchBufferSize_;
  std::weak_ptr<char> scratchBuffer_;
  uint32_t nSwitchChannels_;
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::BaseMemoryChannel>> memoryChannelsDeviceHandle_;
  std::vector<mscclpp::BaseMemoryChannel> baseChannels_;
  std::vector<mscclpp::Connection> conns_;
};
}  // namespace mscclpp