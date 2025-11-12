// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_ALGORITHM_HPP_
#define MSCCLPP_ALGORITHM_HPP_

#include <memory>
#include <mscclpp/executor.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/switch_channel.hpp>
#include <vector>

namespace mscclpp {

enum class CollectiveBufferMode {
  ANY = 0,
  IN_PLACE,
  OUT_OF_PLACE,
};

enum class AlgorithmType {
  NATIVE = 0,
  DSL,
};

class Algorithm {
 public:
  struct Constraint {
    int worldSize;
    int nRanksPerNode;
  };

  virtual ~Algorithm() = default;

  virtual const std::string& name() const = 0;
  virtual const std::string& collective() const = 0;
  virtual const std::pair<size_t, size_t>& messageRange() const = 0;
  virtual const std::unordered_map<std::string, uint64_t>& tags() const = 0;
  virtual const CollectiveBufferMode& bufferMode() const = 0;
  virtual AlgorithmType type() const = 0;
  virtual Constraint constraint() const = 0;
  virtual int execute(std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output, size_t inputSize,
                      size_t outputSize, int dtype, cudaStream_t stream, std::shared_ptr<Executor> executor,
                      std::unordered_map<std::string, void*>& extras) = 0;
};

class AlgorithmBuilder {
 public:
  virtual ~AlgorithmBuilder() = default;
  virtual std::shared_ptr<Algorithm> build() = 0;
};

class AlgorithmCtx {
 public:
  int rank;
  int workSize;
  int nRanksPerNode;

  std::vector<mscclpp::RegisteredMemory> registeredMemories;
  std::vector<mscclpp::MemoryChannel> memoryChannels;
  std::vector<mscclpp::SwitchChannel> switchChannels;
  std::vector<mscclpp::PortChannel> portChannels;
  std::vector<std::shared_ptr<mscclpp::NvlsConnection>> nvlsConnections;
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::MemoryChannel>> memoryChannelDeviceHandles;
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SwitchChannel>> switchChannelDeviceHandles;
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::PortChannel>> portChannelDeviceHandles;
  std::vector<std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>> memorySemaphores;
  std::vector<std::shared_ptr<mscclpp::Host2DeviceSemaphore>> hostSemaphores;

  std::unordered_map<std::string, std::shared_ptr<void>> extras;
};

struct AlgorithmCtxKey {
  void* baseSendBuff;
  void* baseRecvBuff;
  size_t baseSendSize;
  size_t baseRecvSize;
  int tag;

  bool operator==(const AlgorithmCtxKey& other) const {
    return baseSendBuff == other.baseSendBuff && baseRecvBuff == other.baseRecvBuff &&
           baseSendSize == other.baseSendSize && baseRecvSize == other.baseRecvSize && tag == other.tag;
  }
};

}  // namespace mscclpp

namespace std {

// Refer https://www.boost.org/doc/libs/1_86_0/libs/container_hash/doc/html/hash.html#combine
template <typename T>
inline void hash_combine(std::size_t& seed, const T& value) {
  std::hash<T> hasher;
  seed ^= hasher(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <>
struct hash<mscclpp::AlgorithmCtxKey> {
  std::size_t operator()(const mscclpp::AlgorithmCtxKey& key) const {
    std::size_t seed = 42;
    hash_combine(seed, key.baseSendBuff);
    hash_combine(seed, key.baseRecvBuff);
    hash_combine(seed, key.baseSendSize);
    hash_combine(seed, key.baseRecvSize);
    hash_combine(seed, key.tag);
    return seed;
  }
};
}  // namespace std

namespace mscclpp {

class NativeAlgorithm : public Algorithm {
 public:
  using InitFunc = std::function<void(std::shared_ptr<mscclpp::Communicator>)>;
  using KernelFunc = std::function<int(const std::shared_ptr<AlgorithmCtx>, const void*, void*, size_t, size_t, int,
                                       cudaStream_t, std::unordered_map<std::string, void*>&)>;
  using ContextInitFunc = std::function<std::shared_ptr<AlgorithmCtx>(std::shared_ptr<mscclpp::Communicator>,
                                                                      const void*, void*, size_t, size_t, int)>;
  using ContextKeyGenFunc =
      std::function<AlgorithmCtxKey(const void* input, void* output, size_t inputSize, size_t outputSize, int dtype)>;
  NativeAlgorithm(std::string name, std::string collective, InitFunc initFunc, KernelFunc kernelFunc,
                  ContextInitFunc contextInitFunc, ContextKeyGenFunc contextKeyGenFunc, size_t minMessageSize = 0,
                  size_t maxMessageSize = UINT64_MAX, CollectiveBufferMode bufferMode = CollectiveBufferMode::ANY,
                  std::unordered_map<std::string, uint64_t> tags = {}, Constraint constraint = {});

  /// @brief Execute the algorithm.
  /// @brief comm The communicator.
  /// @param input The input buffer.
  /// @param output The output buffer.
  /// @param count The number of elements.
  /// @param dtype The data type.
  /// @param stream The CUDA stream.
  /// @details This method will call ContextKeyGenFunc to generate a context key based on the input parameters,
  /// and then use the context key to retrieve or create an AlgorithmCtx. The kernel function
  /// will be launched with the AlgorithmCtx.
  int execute(std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output, size_t inputSize,
              size_t outputSize, int dtype, cudaStream_t stream, std::shared_ptr<Executor> executor,
              std::unordered_map<std::string, void*>& extras) override;
  const std::string& name() const override;
  const std::string& collective() const override;
  const std::pair<size_t, size_t>& messageRange() const override;
  const std::unordered_map<std::string, uint64_t>& tags() const override;
  const CollectiveBufferMode& bufferMode() const override;
  AlgorithmType type() const override { return AlgorithmType::NATIVE; }
  Constraint constraint() const override;

 private:
  std::string name_;
  std::string collective_;
  NativeAlgorithm::InitFunc initFunc_;
  NativeAlgorithm::KernelFunc kernelLaunchFunc_;
  NativeAlgorithm::ContextInitFunc contextInitFunc_;
  NativeAlgorithm::ContextKeyGenFunc contextKeyGenFunc_;
  size_t minMessageSize_;
  size_t maxMessageSize_;
  CollectiveBufferMode bufferMode_;
  std::unordered_map<std::string, uint64_t> tags_;
  Constraint constraint_;
  std::unordered_map<AlgorithmCtxKey, std::shared_ptr<AlgorithmCtx>> contexts_;

  bool initialized_ = false;
};

class DslAlgorithm : public Algorithm, public AlgorithmBuilder, public std::enable_shared_from_this<DslAlgorithm> {
 public:
  DslAlgorithm(std::string id, std::shared_ptr<ExecutionPlan> plan, std::unordered_map<std::string, uint64_t> tags = {},
               Constraint constraint = {});
  const std::string& name() const override;
  const std::string& collective() const override;
  const std::pair<size_t, size_t>& messageRange() const override;
  const std::unordered_map<std::string, uint64_t>& tags() const override;
  const CollectiveBufferMode& bufferMode() const override;
  int execute(std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output, size_t inputSize,
              size_t outputSize, int dtype, cudaStream_t stream, std::shared_ptr<Executor> executor,
              std::unordered_map<std::string, void*>& extras) override;
  AlgorithmType type() const override { return AlgorithmType::DSL; }
  Constraint constraint() const override;

  std::shared_ptr<Algorithm> build() override;

 private:
  std::shared_ptr<ExecutionPlan> plan_;
  std::string id_;
  std::unordered_map<std::string, uint64_t> tags_;
  Constraint constraint_;
};

struct CollectiveRequest {
  int worldSize;
  int nRanksPerNode;
  int rank;
  const void* inputBuffer;
  void* outputBuffer;
  size_t messageSize;
  const std::string& collective;
  const int dtype;
  const std::unordered_map<std::string, std::vector<uint64_t>>& hints;

  CollectiveBufferMode bufferMode() const;
};
}  // namespace mscclpp

namespace mscclpp {

using AlgoSelectFunc = std::function<std::shared_ptr<Algorithm>(
    const std::unordered_map<std::string, std::unordered_map<std::string, std::shared_ptr<Algorithm>>>&
        algoMapByCollective,
    const CollectiveRequest& request)>;

class AlgorithmCollection {
 public:
  AlgorithmCollection() = default;

  /// @brief Select an algorithm based on the collective operation name and message size.
  /// @param request The collective request containing all necessary parameters.
  /// @return The selected algorithm. If no suitable algorithm is found, a nullptr will be returned.
  std::shared_ptr<Algorithm> selectAlgorithm(const CollectiveRequest& request);

  /// @brief Register a new algorithm.
  /// @param collective The collective operation name.
  /// @param algoName The algorithm name.
  /// @param algorithm The algorithm implementation.
  void registerAlgorithm(const std::string collective, const std::string algoName,
                         std::shared_ptr<Algorithm> algorithm);

  std::unordered_map<std::string, std::shared_ptr<Algorithm>> getAlgorithmsByCollective(
      const std::string& collective) const;

 private:
  std::unordered_map<std::string, std::unordered_map<std::string, std::shared_ptr<Algorithm>>> algoMapByCollective_;
  AlgoSelectFunc algoSelector_ = nullptr;
  AlgoSelectFunc fallbackAlgoSelector_ = nullptr;

  friend class AlgorithmCollectionBuilder;
};

class AlgorithmCollectionBuilder {
 public:
  static std::shared_ptr<AlgorithmCollectionBuilder> getInstance();

  /// @brief Add a new algorithm builder for a specific collective operation.
  /// @param builder The algorithm builder.
  void addAlgorithmBuilder(std::shared_ptr<AlgorithmBuilder> builder);

  /// @brief Set a new algorithm selection function.
  /// @param selector The algorithm selection function.
  void setAlgorithmSelector(AlgoSelectFunc selector);

  /// @brief Set a fallback algorithm selection function.
  /// @param selector The fallback algorithm selection function.
  /// @details The fallback selector will be used if the primary selector returns an empty algorithm. MSCCL++ will
  /// assign a predefined selector as the fallback selector.
  void setFallbackAlgorithmSelector(AlgoSelectFunc selector);

  /// @brief Build the AlgorithmCollection instance.
  /// @return The AlgorithmCollection instance.
  std::shared_ptr<AlgorithmCollection> build();

 private:
  AlgorithmCollectionBuilder() = default;
  std::vector<std::shared_ptr<AlgorithmBuilder>> algoBuilders_;
  AlgoSelectFunc algoSelector_ = nullptr;
  AlgoSelectFunc fallbackAlgoSelector_ = nullptr;
};

}  // namespace mscclpp

#endif  // MSCCLPP_ALGORITHM_HPP_