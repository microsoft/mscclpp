// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_EXECUTOR_HPP_
#define MSCCLPP_EXECUTOR_HPP_

#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu.hpp>
#include <unordered_map>

namespace mscclpp {

/// Data types supported by the executor.
enum class DataType {
  INT32,     // 32-bit signed integer.
  UINT32,    // 32-bit unsigned integer.
  FLOAT16,   // IEEE 754 half precision.
  FLOAT32,   // IEEE 754 single precision.
  BFLOAT16,  // bfloat16 precision.
  FP8_E4M3,  // FP8 with E4M3 layout.
  FP8_E5M2,  // FP8 with E5M2 layout.
};

/// Packet formats used by low-latency transport.
enum class PacketType {
  LL8,   // 8-byte low-latency packet.
  LL16,  // 16-byte low-latency packet.
};

/// Represents a compiled execution plan loaded from disk.
///
/// An ExecutionPlan encapsulates metadata about a collective algorithm such as its name, the
/// collective it implements, and the supported message-size range. The concrete implementation
/// is hidden behind the PIMPL pointer.
class ExecutionPlan {
 public:
  /// Construct an ExecutionPlan by loading the plan file at `planPath`.
  /// @param planPath Filesystem path to the serialized plan.
  /// @param rank The rank of the current process.
  ExecutionPlan(const std::string& planPath, int rank);

  /// Destructor.
  ~ExecutionPlan() = default;

  /// Return the human-readable name of the plan.
  const std::string& name() const;

  /// Return the collective implemented by this plan (e.g., "allreduce", "allgather").
  const std::string& collective() const;

  /// Minimum message size (in bytes) for which this plan is valid.
  size_t minMessageSize() const;

  /// Maximum message size (in bytes) for which this plan is valid.
  size_t maxMessageSize() const;

  /// Whether this plan performs the operation in-place.
  bool isInPlace() const;

 private:
  struct Impl;
  std::shared_ptr<Impl> impl_;

  friend class Executor;
};


/// A handle representing a specific execution plan along with its constraints and metadata.
struct ExecutionPlanHandle {
  /// Constraints that must be satisfied for the plan to be valid.
  struct Constraint {
    int worldSize;
    int nRanksPerNode;
  };

  std::string id;                                  /// Unique identifier for the handle.
  Constraint constraint;                           /// Constraints for plan applicability.
  std::shared_ptr<ExecutionPlan> plan;             /// Backing ExecutionPlan instance.
  std::unordered_map<std::string, uint64_t> tags;  /// Optional tags/metadata used by selector.

  /// Create a new ExecutionPlanHandle.
  /// @param id Unique id for the handle.
  /// @param worldSize Required world size for the plan.
  /// @param nRanksPerNode Required ranks-per-node for the plan.
  /// @param plan The associated ExecutionPlan.
  /// @param tags Optional tags used for selection.
  static std::shared_ptr<ExecutionPlanHandle> create(const std::string& id, int worldSize, int nRanksPerNode,
                                                     std::shared_ptr<ExecutionPlan> plan,
                                                     const std::unordered_map<std::string, uint64_t>& tags = {});

};


/// High-level executor responsible for invoking execution plans on a communicator.
class Executor {
 public:
  /// Construct an Executor using the provided communicator.
  /// @param comm Communicator instance used for underlying communication.
  /// @param defaultScratchBuffer Optional scratch buffer used by some plans (may be nullptr).
  Executor(std::shared_ptr<Communicator> comm, std::shared_ptr<char> defaultScratchBuffer = nullptr);

  /// Copy construction is disabled for Executor.
  Executor(const Executor&) = delete;

  /// Copy assignment is disabled for Executor.
  Executor& operator=(const Executor&) = delete;

  /// Destructor. Cleans up internal resources held by the Executor.
  ~Executor();

  /// Execute a plan.
  ///
  /// This method dispatches the given plan on the provided CUDA stream.
  ///
  /// @param rank Rank of the calling process.
  /// @param sendbuff Pointer to the send buffer.
  /// @param recvBuff Pointer to the receive buffer.
  /// @param sendBuffSize Size of the send buffer in bytes.
  /// @param recvBuffSize Size of the receive buffer in bytes.
  /// @param dataType Data type of elements in the buffers.
  /// @param plan The execution plan to run.
  /// @param stream CUDA stream to execute kernels/operations on.
  /// @param packetType Packet type used for low-latency transports (default: LL16).
  void execute(int rank, void* sendbuff, void* recvBuff, size_t sendBuffSize, size_t recvBuffSize, DataType dataType,
               const ExecutionPlan& plan, cudaStream_t stream, PacketType packetType = PacketType::LL16);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace mscclpp

#endif  // MSCCLPP_EXECUTOR_HPP_
