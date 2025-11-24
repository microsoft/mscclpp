
#include <mscclpp/algorithm.hpp>

namespace mscclpp {

class Allreduce8 : public mscclpp::AlgorithmBuilder {
 public:
  Allreduce8(std::shared_ptr<void> scratchBuffer, size_t scratchBufferSize)
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
  std::shared_ptr<mscclpp::Communicator> comm_;
  int nChannelsPerConnection_;
  std::vector<mscclpp::Connection> conns_;
  std::weak_ptr<char> scratchBuffer_;
  std::vector<std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>> outputSemaphores_;
  std::vector<std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>> inputScratchSemaphores_;
  std::vector<mscclpp::RegisteredMemory> remoteScratchMemories_;
  mscclpp::RegisteredMemory localScratchMemory_;
  std::unordered_map<const void*, std::pair<std::vector<mscclpp::MemoryChannel>,
                                            std::shared_ptr<mscclpp::DeviceHandle<mscclpp::MemoryChannel>>>>
      memoryChannelsMap_;
};
}  // namespace mscclpp