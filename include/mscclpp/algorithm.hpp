// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_ALGORITHM_HPP_
#define MSCCLPP_ALGORITHM_HPP_

#include <memory>
#include <mscclpp/executor.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/switch_channel.hpp>
#include <mscclpp/utils.hpp>
#include <vector>

namespace mscclpp {

/// Capsule name for native algorithm pointers used in Python bindings.
constexpr char ALGORITHM_NATIVE_CAPSULE_NAME[] = "mscclpp::AlgorithmPtr";

enum class CollectiveBufferMode {
  Any = 0,
  InPlace,
  OutOfPlace,
};

enum class AlgorithmType {
  Native = 0,
  DSL,
};

enum class CommResult {
  CommSuccess = 0,
  CommUnhandledCudaError = 1,
  CommSystemError = 2,
  CommInternalError = 3,
  CommInvalidArgument = 4,
  CommInvalidUsage = 5,
  CommRemoteError = 6,
  CommInProgress = 7,
  CommNumResults = 8
};

enum ReduceOp { SUM = 0, MIN = 3, NOP = 255 };

/// Base class for collective communication algorithms.
///
/// This abstract class defines the interface for implementing collective communication
/// algorithms such as allreduce, allgather, and reduce-scatter. Concrete implementations
/// can be either native C++/CUDA algorithms or DSL-defined algorithms.
class Algorithm {
 public:
  struct Constraint {
    int worldSize;
    int nRanksPerNode;
  };

  virtual ~Algorithm() = default;

  /// Get the name of the algorithm.
  /// @return A reference to the algorithm name string.
  virtual const std::string& name() const = 0;

  /// Get the collective operation this algorithm implements.
  /// @return A reference to the collective name (e.g., "allreduce", "allgather").
  virtual const std::string& collective() const = 0;

  /// Get the valid message size range for this algorithm.
  /// @return A pair of (minMessageSize, maxMessageSize) in bytes.
  virtual const std::pair<size_t, size_t>& messageRange() const = 0;

  /// Get the tags associated with this algorithm.
  /// @return An unordered map of tag names to tag values.
  virtual const std::unordered_map<std::string, uint64_t>& tags() const = 0;

  /// Get the buffer mode supported by this algorithm.
  /// @return The CollectiveBufferMode indicating in-place, out-of-place, or any.
  virtual const CollectiveBufferMode& bufferMode() const = 0;

  /// Get the type of this algorithm.
  /// @return AlgorithmType::Native or AlgorithmType::DSL.
  virtual AlgorithmType type() const = 0;

  /// Get the execution constraints for this algorithm.
  /// @return The Constraint struct specifying worldSize and nRanksPerNode requirements.
  virtual Constraint constraint() const = 0;

  /// Execute the algorithm.
  /// @param comm The communicator to use.
  /// @param input Pointer to the input buffer.
  /// @param output Pointer to the output buffer.
  /// @param inputSize Size of the input buffer in bytes.
  /// @param outputSize Size of the output buffer in bytes.
  /// @param dtype The data type of the elements.
  /// @param op The reduction operation (for reduce-type collectives).
  /// @param stream The CUDA stream to execute on.
  /// @param executor The executor for DSL algorithms (may be nullptr for native).
  /// @param nBlocks Number of CUDA blocks (0 for auto-selection).
  /// @param nThreadsPerBlock Number of threads per block (0 for auto-selection).
  /// @param extras Additional parameters for algorithm-specific customization.
  /// @return The result of the operation.
  virtual CommResult execute(std::shared_ptr<Communicator> comm, const void* input, void* output, size_t inputSize,
                             size_t outputSize, DataType dtype, ReduceOp op, cudaStream_t stream,
                             std::shared_ptr<Executor> executor, int nBlocks = 0, int nThreadsPerBlock = 0,
                             const std::unordered_map<std::string, uintptr_t>& extras = {}) = 0;

  /// Reset the algorithm state, clearing any cached contexts.
  virtual void reset() = 0;
};

/// Interface for building Algorithm instances.
///
/// Implement this interface to create custom algorithm factories that can be
/// registered with the AlgorithmCollectionBuilder.
class AlgorithmBuilder {
 public:
  virtual ~AlgorithmBuilder() = default;

  /// Build and return an Algorithm instance.
  /// @return A shared pointer to the constructed Algorithm.
  virtual std::shared_ptr<Algorithm> build() = 0;
};

/// Context holding resources for algorithm execution.
///
/// This struct contains all the channels, semaphores, and memory handles
/// needed for executing a native algorithm. It is created once per unique
/// buffer configuration and cached for reuse.
/// @note This struct may be changed in future releases.
class AlgorithmCtx {
 public:
  int rank;
  int workSize;
  int nRanksPerNode;

  std::vector<RegisteredMemory> registeredMemories;
  std::vector<MemoryChannel> memoryChannels;
  std::vector<SwitchChannel> switchChannels;
  std::vector<PortChannel> portChannels;
  std::vector<std::shared_ptr<NvlsConnection>> nvlsConnections;
  std::shared_ptr<DeviceHandle<MemoryChannel>> memoryChannelDeviceHandles;
  std::shared_ptr<DeviceHandle<SwitchChannel>> switchChannelDeviceHandles;
  std::shared_ptr<DeviceHandle<PortChannel>> portChannelDeviceHandles;
  std::vector<std::shared_ptr<MemoryDevice2DeviceSemaphore>> memorySemaphores;
  std::vector<std::shared_ptr<Host2DeviceSemaphore>> hostSemaphores;
  std::unordered_map<std::string, std::shared_ptr<void>> extras;
};

/// Key for identifying cached AlgorithmCtx instances.
///
/// The context key uniquely identifies a buffer configuration, allowing
/// the algorithm to cache and reuse contexts for repeated operations with
/// the same buffers.
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

template <>
struct hash<mscclpp::AlgorithmCtxKey> {
  std::size_t operator()(const mscclpp::AlgorithmCtxKey& key) const {
    std::size_t seed = 42;
    mscclpp::detail::hashCombine(seed, key.baseSendBuff);
    mscclpp::detail::hashCombine(seed, key.baseRecvBuff);
    mscclpp::detail::hashCombine(seed, key.baseSendSize);
    mscclpp::detail::hashCombine(seed, key.baseRecvSize);
    mscclpp::detail::hashCombine(seed, key.tag);
    return seed;
  }
};
}  // namespace std

namespace mscclpp {

/// Native C++/CUDA implementation of a collective algorithm.
///
/// NativeAlgorithm allows users to implement custom collective algorithms in C++/CUDA.
/// It provides a framework for initialization, context management, and kernel execution.
/// Contexts are cached based on buffer configurations to avoid redundant setup.
class NativeAlgorithm : public Algorithm {
 public:
  using InitFunc = std::function<void(std::shared_ptr<Communicator>)>;

  /// Function type for the kernel that executes the collective operation.
  /// @param ctx The algorithm context containing channels and semaphores.
  /// @param input Pointer to the input buffer.
  /// @param output Pointer to the output buffer.
  /// @param inputSize Size of the input buffer in bytes.
  /// @param outputSize Size of the output buffer in bytes.
  /// @param dtype Data type of the elements.
  /// @param op Reduction operation (for reduce-type collectives).
  /// @param stream CUDA stream to execute on.
  /// @param nBlocks Number of CUDA blocks.
  /// @param nThreadsPerBlock Number of threads per block.
  /// @param extras Additional algorithm-specific parameters.
  /// @return The result of the operation.
  using KernelFunc =
      std::function<CommResult(const std::shared_ptr<AlgorithmCtx>, const void*, void*, size_t, size_t, DataType,
                               ReduceOp, cudaStream_t, int, int, const std::unordered_map<std::string, uintptr_t>&)>;

  /// Function type for creating algorithm contexts.
  /// @param comm The communicator.
  /// @param input Pointer to the input buffer.
  /// @param output Pointer to the output buffer.
  /// @param inputSize Size of the input buffer.
  /// @param outputSize Size of the output buffer.
  /// @param dtype Data type of the elements.
  /// @return A shared pointer to the created context.
  using ContextInitFunc = std::function<std::shared_ptr<AlgorithmCtx>(std::shared_ptr<Communicator>, const void*, void*,
                                                                      size_t, size_t, DataType)>;

  /// Function type for generating context keys.
  /// @param input Pointer to the input buffer.
  /// @param output Pointer to the output buffer.
  /// @param inputSize Size of the input buffer.
  /// @param outputSize Size of the output buffer.
  /// @param dtype Data type of the elements.
  /// @return A key uniquely identifying this buffer configuration.
  using ContextKeyGenFunc = std::function<AlgorithmCtxKey(const void* input, void* output, size_t inputSize,
                                                          size_t outputSize, DataType dtype)>;

  /// Construct a NativeAlgorithm.
  /// @param name Human-readable name of the algorithm.
  /// @param collective The collective operation (e.g., "allreduce").
  /// @param initFunc Function called once to initialize the algorithm.
  /// @param kernelFunc Function that launches the CUDA kernel.
  /// @param contextInitFunc Function that creates execution contexts.
  /// @param contextKeyGenFunc Function that generates cache keys for contexts.
  /// @param minMessageSize Minimum supported message size in bytes (default: 0).
  /// @param maxMessageSize Maximum supported message size in bytes (default: UINT64_MAX).
  /// @param bufferMode Buffer mode supported by this algorithm (default: ANY).
  /// @param tags Tags for algorithm selection hints.
  /// @param constraint Execution constraints (worldSize, nRanksPerNode).
  NativeAlgorithm(std::string name, std::string collective, InitFunc initFunc, KernelFunc kernelFunc,
                  ContextInitFunc contextInitFunc, ContextKeyGenFunc contextKeyGenFunc, size_t minMessageSize = 0,
                  size_t maxMessageSize = UINT64_MAX, CollectiveBufferMode bufferMode = CollectiveBufferMode::Any,
                  std::unordered_map<std::string, uint64_t> tags = {}, Constraint constraint = {});

  CommResult execute(std::shared_ptr<Communicator> comm, const void* input, void* output, size_t inputSize,
                     size_t outputSize, DataType dtype, ReduceOp op, cudaStream_t stream,
                     std::shared_ptr<Executor> executor, int nBlocks = 0, int nThreadsPerBlock = 0,
                     const std::unordered_map<std::string, uintptr_t>& extras = {}) override;
  const std::string& name() const override;
  const std::string& collective() const override;
  const std::pair<size_t, size_t>& messageRange() const override;
  const std::unordered_map<std::string, uint64_t>& tags() const override;
  const CollectiveBufferMode& bufferMode() const override;
  AlgorithmType type() const override { return AlgorithmType::Native; }
  Constraint constraint() const override;
  void reset() override;

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

/// DSL-based implementation of a collective algorithm.
///
/// DslAlgorithm wraps an ExecutionPlan loaded from a DSL specification file.
/// It implements both Algorithm and AlgorithmBuilder interfaces, allowing it
/// to be used directly or registered with AlgorithmCollectionBuilder.
class DslAlgorithm : public Algorithm, public AlgorithmBuilder, public std::enable_shared_from_this<DslAlgorithm> {
 public:
  /// Construct a DslAlgorithm from an execution plan.
  /// @param id Identifier for this algorithm instance.
  /// @param plan The execution plan defining the algorithm.
  /// @param tags Tags for algorithm selection hints.
  /// @param constraint Execution constraints (worldSize, nRanksPerNode).
  DslAlgorithm(std::string id, ExecutionPlan plan, std::unordered_map<std::string, uint64_t> tags = {},
               Constraint constraint = {});
  const std::string& name() const override;
  const std::string& collective() const override;
  const std::pair<size_t, size_t>& messageRange() const override;
  const std::unordered_map<std::string, uint64_t>& tags() const override;
  const CollectiveBufferMode& bufferMode() const override;
  CommResult execute(std::shared_ptr<Communicator> comm, const void* input, void* output, size_t inputSize,
                     size_t outputSize, DataType dtype, ReduceOp op, cudaStream_t stream,
                     std::shared_ptr<Executor> executor, int nBlocks = 0, int nThreadsPerBlock = 0,
                     const std::unordered_map<std::string, uintptr_t>& extras = {}) override;
  AlgorithmType type() const override { return AlgorithmType::DSL; }
  Constraint constraint() const override;
  void reset() override;

  std::shared_ptr<Algorithm> build() override;

 private:
  ExecutionPlan plan_;
  std::string id_;
  std::unordered_map<std::string, uint64_t> tags_;
  Constraint constraint_;
};

/// Request parameters for selecting and executing a collective operation.
///
/// This struct encapsulates all the information needed to select an appropriate
/// algorithm for a collective operation.
struct CollectiveRequest {
  int worldSize;
  int nRanksPerNode;
  int rank;
  const void* inputBuffer;
  void* outputBuffer;
  size_t messageSize;
  const std::string& collective;
  const DataType dtype;
  const std::unordered_map<std::string, std::vector<uint64_t>>& hints;

  CollectiveBufferMode bufferMode() const;
};

/// Function type for custom algorithm selection.
/// @param algoMapByCollective Map of collective names to available algorithms.
/// @param request The collective request parameters.
/// @return The selected algorithm, or nullptr if no suitable algorithm is found.
using AlgoSelectFunc = std::function<std::shared_ptr<Algorithm>(
    const std::unordered_map<std::string, std::unordered_map<std::string, std::shared_ptr<Algorithm>>>&
        algoMapByCollective,
    const CollectiveRequest& request)>;

/// Collection of algorithms for collective operations.
///
/// AlgorithmCollection manages a set of algorithms indexed by collective operation
/// name and algorithm name. It provides methods to select the best algorithm for
/// a given request and to register new algorithms.
class AlgorithmCollection {
 public:
  AlgorithmCollection() = default;

  /// Select an algorithm based on the collective operation name and message size.
  /// @param request The collective request containing all necessary parameters.
  /// @return The selected algorithm. If no suitable algorithm is found, nullptr is returned.
  std::shared_ptr<Algorithm> selectAlgorithm(const CollectiveRequest& request);

  /// Register a new algorithm.
  /// @param collective The collective operation name (e.g., "allreduce").
  /// @param algoName The algorithm name.
  /// @param algorithm The algorithm implementation.
  void registerAlgorithm(const std::string collective, const std::string algoName,
                         std::shared_ptr<Algorithm> algorithm);

  /// Get all algorithms for a specific collective operation.
  /// @param collective The collective operation name.
  /// @return A map of algorithm names to algorithm instances.
  std::unordered_map<std::string, std::shared_ptr<Algorithm>> getAlgorithmsByCollective(
      const std::string& collective) const;

  /// Get all registered algorithms.
  /// @return A vector containing all algorithm instances.
  std::vector<std::shared_ptr<Algorithm>> getAllAlgorithms() const;

  /// Extend this collection with algorithms from another collection.
  /// @param other The other AlgorithmCollection to merge in.
  void extend(const AlgorithmCollection& other);

  void setSelectors(AlgoSelectFunc algoSelector, AlgoSelectFunc fallbackAlgoSelector);

 private:
  std::unordered_map<std::string, std::unordered_map<std::string, std::shared_ptr<Algorithm>>> algoMapByCollective_;
  AlgoSelectFunc algoSelector_ = nullptr;
  AlgoSelectFunc fallbackAlgoSelector_ = nullptr;
};

}  // namespace mscclpp

#endif  // MSCCLPP_ALGORITHM_HPP_