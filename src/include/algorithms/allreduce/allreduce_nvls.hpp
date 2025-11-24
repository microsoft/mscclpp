#include <mscclpp/algorithm.hpp>

namespace mscclpp {
class AllreduceNvls : public mscclpp::AlgorithmBuilder {
 public:
  AllreduceNvls() = default;
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
  uint32_t nSwitchChannels_;
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::BaseMemoryChannel>> memoryChannelsDeviceHandle_;
  std::vector<mscclpp::BaseMemoryChannel> baseChannels_;
  std::vector<mscclpp::Connection> conns_;
};

}  // namespace mscclpp