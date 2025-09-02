// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_ALGORITHM_HPP_
#define MSCCLPP_ALGORITHM_HPP_

#include <memory>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/nvls.hpp>
#include <mscclpp/port_channel.hpp>
#include <vector>

namespace mscclpp {

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

class AlgorithmImpl;

class Algorithm {
 public:
  using InitFunc = std::function<void(std::shared_ptr<mscclpp::Communicator>,
                                      std::unordered_map<std::string, std::shared_ptr<void>>&)>;
  using KernelFunc = std::function<int(const std::shared_ptr<AlgorithmCtx>, const void*, void*, size_t, int,
                                       cudaStream_t, std::unordered_map<std::string, std::shared_ptr<void>>&)>;
  using ContextInitFunc = std::function<std::shared_ptr<AlgorithmCtx>(std::shared_ptr<mscclpp::Communicator>,
                                                                      const void*, void*, size_t, int)>;
  using ContextKeyGenFunc = std::function<AlgorithmCtxKey(const void* input, void* output, size_t count, int dtype)>;
  Algorithm(std::string name, InitFunc initFunc, KernelFunc kernelFunc, ContextInitFunc contextInitFunc,
            ContextKeyGenFunc contextKeyGenFunc);
  Algorithm() = default;

  /// @brief Launch the algorithm.
  /// @brief comm The communicator.
  /// @param input The input buffer.
  /// @param output The output buffer.
  /// @param count The number of elements.
  /// @param dtype The data type.
  /// @param stream The CUDA stream.
  /// @details This method will call ContextKeyGenFunc to generate a context key based on the input parameters,
  /// and then use the context key to retrieve or create an AlgorithmCtx. The kernel function
  /// will be launched with the AlgorithmCtx.
  int launch(std::shared_ptr<mscclpp::Communicator> comm, const void* input, void* output, size_t count, int dtype,
             cudaStream_t stream, std::unordered_map<std::string, std::shared_ptr<void>>& extras);
  bool isEmpty();

 private:
  std::shared_ptr<AlgorithmImpl> impl_;
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

class AlgorithmFactory {
 public:
  using AlgoSelectFunc = std::function<Algorithm(
      const std::unordered_map<std::string, std::unordered_map<std::string, Algorithm>>& algoMapByCollective,
      std::string collective, size_t messageSize, int nRanksPerNode, int worldSize)>;

  static std::shared_ptr<AlgorithmFactory> getInstance();

  /// @brief Register a new algorithm.
  /// @param collective The collective operation name.
  /// @param algoName The algorithm name.
  /// @param algorithm The algorithm implementation.
  void registerAlgorithm(const std::string collective, const std::string algoName, Algorithm algorithm);

  /// @brief Select an algorithm based on the collective operation name and message size.
  /// @param collective The collective operation name.
  /// @param messageSize The message size.
  /// @param nRanksPerNode The number of ranks per node.
  /// @param worldSize The total number of ranks.
  /// @return The selected algorithm. If no suitable algorithm is found, an empty Algorithm object is returned.
  Algorithm selectAlgorithm(const std::string& collective, size_t messageSize, int nRanksPerNode, int worldSize);

  /// @brief Set a new algorithm selection function.
  /// @param selector The algorithm selection function.
  void setAlgorithmSelector(AlgoSelectFunc selector);

  /// @brief Set a fallback algorithm selection function.
  /// @param selector The fallback algorithm selection function.
  /// @details The fallback selector will be used if the primary selector returns an empty algorithm. MSCCL++ will
  /// assign a predefined selector as the fallback selector.
  void setFallbackAlgorithmSelector(AlgoSelectFunc selector);

  void destroy();

 private:
  AlgorithmFactory() = default;
  std::unordered_map<std::string, std::unordered_map<std::string, Algorithm>> algoMapByCollective_;
  AlgoSelectFunc algoSelector_ = nullptr;
  AlgoSelectFunc fallbackAlgoSelector_ = nullptr;
};

}  // namespace mscclpp

#endif  // MSCCLPP_ALGORITHM_HPP_