// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_ALGORITHM_HPP_
#define MSCCLPP_ALGORITHM_HPP_

#include <mscclpp/nccl.h>

#include <memory>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/nvls.hpp>
#include <mscclpp/port_channel.hpp>
#include <vector>

namespace mscclpp {
enum class AlgorithmFeature { NonZeroCopy, NVLS };

class AlgorithmCtx {
 public:
  int rank;
  int workSize;
  int nRanksPerNode;

  std::vector<mscclpp::RegisteredMemory> registeredMemories;
  std::vector<mscclpp::MemoryChannel> memoryChannels;
  std::vector<mscclpp::SwitchChannel> switchChannels;
  std::vector<mscclpp::PortChannel> portChannels;
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::MemoryChannel>> memoryChannelDeviceHandles;
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::SwitchChannel>> switchChannelDeviceHandles;
  std::shared_ptr<mscclpp::DeviceHandle<mscclpp::PortChannel>> portChannelDeviceHandles;
  std::shared_ptr<char> scratchBuffer;
  std::vector<AlgorithmFeature> supportFeatures;
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

struct AlgorithmKey {
  std::string collective;
  std::string name;
  bool operator==(const AlgorithmKey& other) const { return name == other.name && collective == other.collective; }
};

class AlgorithmImpl;

class Algorithm {
 public:
  using KernelFunc = std::function<ncclResult_t(const std::shared_ptr<AlgorithmCtx>, void*, void*, size_t,
                                                ncclDataType_t, cudaStream_t)>;
  using ContextInitFunc = std::function<std::shared_ptr<AlgorithmCtx>(std::shared_ptr<mscclpp::Communicator>)>;
  using ContextKeyGenFunc =
      std::function<AlgorithmCtxKey(void* input, void* output, size_t count, ncclDataType_t dtype)>;
  Algorithm(std::shared_ptr<Communicator> comm, std::string name, KernelFunc kernelFunc,
            ContextInitFunc contextInitFunc, ContextKeyGenFunc contextKeyGenFunc);

  /// @brief Launch the algorithm.
  /// @param input The input buffer.
  /// @param output The output buffer.
  /// @param count The number of elements.
  /// @param dtype The data type.
  /// @param stream The CUDA stream.
  /// @details This method will call ContextKeyGenFunc to generate a context key based on the input parameters,
  /// and then use the context key to retrieve or create an AlgorithmCtx. The kernel function
  /// will be launched with the AlgorithmCtx.
  ncclResult_t launch(void* input, void* output, size_t count, ncclDataType_t dtype, cudaStream_t stream);

 private:
  /// The algorithm name.
  std::string name;

  std::shared_ptr<AlgorithmImpl> impl;
};
}

namespace std {

// Refer https://www.boost.org/doc/libs/1_86_0/libs/container_hash/doc/html/hash.html#combine
template <typename T>
inline void hash_combine(std::size_t& seed, const T& value) {
  std::hash<T> hasher;
  seed ^= hasher(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <>
struct hash<mscclpp::AlgorithmKey> {
  std::size_t operator()(const mscclpp::AlgorithmKey& key) const {
    std::size_t seed = 42;
    hash_combine(seed, key.name);
    hash_combine(seed, key.collective);
    return seed;
  }
};

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
  static void registerAlgorithm(const std::string collective, const std::string algoName, Algorithm algorithm);

  static Algorithm getAlgorithm(const AlgorithmKey& algoKey);

  static std::shared_ptr<AlgorithmFactory> getInstance() {
    static std::shared_ptr<AlgorithmFactory> instance(new AlgorithmFactory());
    return instance;
  }

 private:
  AlgorithmFactory() = default;
  std::unordered_map<AlgorithmKey, Algorithm> algoMap;
};

}  // namespace mscclpp

#endif  // MSCCLPP_ALGORITHM_HPP_