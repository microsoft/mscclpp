// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_GROUP_EXECUTION_PLAN_HPP_
#define MSCCLPP_GROUP_EXECUTION_PLAN_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <mscclpp/core.hpp>
#include <mscclpp/errors.hpp>
#include <mscclpp/group.hpp>
#include <mscclpp/executor.hpp>

namespace mscclpp {

/// Forward declaration for ExecutionPlan
class ExecutionPlan;

/// Chunk size specification for variable-size operations
struct ChunkSizeSpec {
  int rank;                    ///< Source rank for this chunk
  int destRank;               ///< Destination rank for this chunk
  size_t sendSize;            ///< Size to send to destRank
  size_t recvSize;            ///< Size to receive from rank
  size_t sendOffset;          ///< Offset in send buffer
  size_t recvOffset;          ///< Offset in receive buffer
};

/// Information for all_to_allv operation extracted from execution plan
struct AllToAllvInfo {
  std::vector<ChunkSizeSpec> chunkSpecs;  ///< Per-rank chunk specifications
  size_t totalSendSize;                   ///< Total send buffer size needed
  size_t totalRecvSize;                   ///< Total receive buffer size needed
  uint32_t maxChunks;                     ///< Maximum number of chunks
  bool isVariable;                        ///< Whether chunk sizes are variable
};

/// Extended operation for execution plan-based all_to_allv operations
class ExecutionPlanAllToAllvOperation : public Operation {
 public:
  ExecutionPlanAllToAllvOperation(std::shared_ptr<Communicator> comm, 
                                 std::shared_ptr<ExecutionPlan> plan,
                                 void* sendBuffer, void* recvBuffer,
                                 size_t inputSize, size_t outputSize,
                                 int tag = 0);
  
  GroupResult execute() override;
  bool isComplete() const override;
  void cancel() override;
  
  /// Get the all_to_allv information extracted from execution plan
  const AllToAllvInfo& getAllToAllvInfo() const { return allToAllvInfo_; }

 private:
  std::shared_ptr<ExecutionPlan> plan_;
  void* sendBuffer_;
  void* recvBuffer_;
  size_t inputSize_;
  size_t outputSize_;
  AllToAllvInfo allToAllvInfo_;
  std::atomic<bool> cancelled_{false};
  
  /// Extract chunk size information from execution plan
  void extractChunkSizes();
  
  /// Validate that buffers are sufficient for the operation
  bool validateBuffers() const;
};

/// Extended GroupManager with execution plan support
class ExecutionPlanGroupManager : public GroupManager {
 public:
  /// Add an all_to_allv operation based on execution plan to the current group
  /// @param comm The communicator to use
  /// @param plan The execution plan containing chunk size information
  /// @param sendBuffer Buffer containing data to send
  /// @param recvBuffer Buffer to receive data into
  /// @param inputSize Total input buffer size
  /// @param outputSize Total output buffer size
  /// @param tag Tag for this operation
  /// @return Shared pointer to the all_to_allv operation
  static std::shared_ptr<ExecutionPlanAllToAllvOperation> addExecutionPlanAllToAllv(
    std::shared_ptr<Communicator> comm,
    std::shared_ptr<ExecutionPlan> plan,
    void* sendBuffer, void* recvBuffer,
    size_t inputSize, size_t outputSize,
    int tag = 0);

  /// Add a custom operation with execution plan context
  /// @param comm The communicator to use
  /// @param plan The execution plan for context
  /// @param executeFunc Function to execute the operation
  /// @param isCompleteFunc Function to check if operation is complete
  /// @param cancelFunc Optional function to cancel the operation
  /// @param tag Tag for this operation
  /// @return Shared pointer to the custom operation
  static std::shared_ptr<CustomOperation> addExecutionPlanCustom(
    std::shared_ptr<Communicator> comm,
    std::shared_ptr<ExecutionPlan> plan,
    CustomOperation::ExecuteFunction executeFunc,
    CustomOperation::IsCompleteFunction isCompleteFunc,
    CustomOperation::CancelFunction cancelFunc = nullptr,
    int tag = 0);

  /// Get execution plan information for current group
  /// @return Map of execution plans by operation tag
  static std::unordered_map<int, std::shared_ptr<ExecutionPlan>> getExecutionPlans();

 private:
  static thread_local std::unordered_map<int, std::shared_ptr<ExecutionPlan>> executionPlans_;
};

/// Utility functions for execution plan chunk size extraction

/// Extract chunk size specifications from an execution plan
/// @param plan The execution plan to analyze
/// @param inputSize Total input size
/// @param outputSize Total output size
/// @return AllToAllvInfo containing chunk specifications
AllToAllvInfo extractAllToAllvInfo(const ExecutionPlan& plan, size_t inputSize, size_t outputSize);

/// Calculate send/receive sizes for each rank pair from execution plan
/// @param plan The execution plan to analyze
/// @param inputSize Total input size
/// @param outputSize Total output size
/// @return Vector of chunk size specifications
std::vector<ChunkSizeSpec> calculateChunkSizes(const ExecutionPlan& plan, size_t inputSize, size_t outputSize);

/// Get maximum chunk size that might be needed based on execution plan
/// @param plan The execution plan to analyze
/// @param inputSize Total input size
/// @param outputSize Total output size
/// @return Maximum chunk size
size_t getMaxChunkSize(const ExecutionPlan& plan, size_t inputSize, size_t outputSize);

/// Check if execution plan supports variable chunk sizes (for all_to_allv)
/// @param plan The execution plan to check
/// @return True if variable chunk sizes are supported
bool supportsVariableChunkSizes(const ExecutionPlan& plan);

/// RAII helper for execution plan-aware group management
class ExecutionPlanGroupScope : public GroupScope {
 public:
  /// Constructor that sets up execution plan context
  /// @param plan The execution plan to use for this group
  /// @param blocking Whether the group should be blocking
  ExecutionPlanGroupScope(std::shared_ptr<ExecutionPlan> plan, bool blocking = true);
  
  /// Destructor that cleans up execution plan context
  ~ExecutionPlanGroupScope();

  /// Get the execution plan for this scope
  std::shared_ptr<ExecutionPlan> getExecutionPlan() const { return plan_; }

 private:
  std::shared_ptr<ExecutionPlan> plan_;
};

/// Convenience functions for execution plan-based group operations

/// Execute multiple all_to_allv operations based on execution plans
/// @param operations Vector of all_to_allv parameters with execution plans
/// @param blocking Whether to wait for completion
/// @return Vector of operation futures
std::vector<std::shared_ptr<ExecutionPlanAllToAllvOperation>>
groupExecutionPlanAllToAllv(
  const std::vector<std::tuple<std::shared_ptr<Communicator>, std::shared_ptr<ExecutionPlan>, 
                              void*, void*, size_t, size_t, int>>& operations,
  bool blocking = true);

}  // namespace mscclpp

#endif  // MSCCLPP_GROUP_EXECUTION_PLAN_HPP_