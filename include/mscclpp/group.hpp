// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_GROUP_HPP_
#define MSCCLPP_GROUP_HPP_

#include <atomic>
#include <functional>
#include <future>
#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/errors.hpp>
#include <queue>
#include <thread>
#include <vector>

namespace mscclpp {

/// Result type for group operations.
enum class GroupResult {
  Success = 0,      ///< Operation completed successfully
  InProgress,       ///< Operation is still in progress
  InvalidUsage,     ///< Invalid usage of group API
  InternalError,    ///< Internal error occurred
  Timeout          ///< Operation timed out
};

/// Converts GroupResult to ErrorCode for exception handling.
ErrorCode groupResultToErrorCode(GroupResult result);

/// Operation types that can be batched in a group.
enum class OperationType {
  Connect,         ///< Connection establishment
  SendMemory,      ///< Memory send operation
  RecvMemory,      ///< Memory receive operation
  Barrier,         ///< Barrier synchronization
  AllGather,       ///< AllGather collective
  Custom          ///< Custom user-defined operation
};

/// Base class for batched operations.
class Operation {
 public:
  Operation(OperationType type, std::shared_ptr<Communicator> comm, int tag = 0);
  virtual ~Operation() = default;

  /// Execute the operation.
  /// @return GroupResult indicating success or failure.
  virtual GroupResult execute() = 0;

  /// Cancel the operation if possible.
  virtual void cancel() {}

  /// Check if the operation is complete.
  /// @return True if operation is complete.
  virtual bool isComplete() const = 0;

  /// Get the operation type.
  OperationType getType() const { return type_; }

  /// Get the communicator associated with this operation.
  std::shared_ptr<Communicator> getCommunicator() const { return comm_; }

  /// Get the operation tag.
  int getTag() const { return tag_; }

 protected:
  OperationType type_;
  std::shared_ptr<Communicator> comm_;
  int tag_;
  std::atomic<bool> completed_{false};
};

/// Connection operation for batching.
class ConnectOperation : public Operation {
 public:
  ConnectOperation(std::shared_ptr<Communicator> comm, EndpointConfig localConfig, int remoteRank, int tag = 0);
  
  GroupResult execute() override;
  bool isComplete() const override;
  
  /// Get the connection future (available after execute() is called).
  std::shared_future<std::shared_ptr<Connection>> getFuture() const { return future_; }

 private:
  EndpointConfig localConfig_;
  int remoteRank_;
  std::shared_future<std::shared_ptr<Connection>> future_;
};

/// Memory send operation for batching.
class SendMemoryOperation : public Operation {
 public:
  SendMemoryOperation(std::shared_ptr<Communicator> comm, std::shared_ptr<RegisteredMemory> memory, 
                     int remoteRank, int tag = 0);
  
  GroupResult execute() override;
  bool isComplete() const override;
  
  /// Get the send future (available after execute() is called).
  std::shared_future<RegisteredMemory> getFuture() const { return future_; }

 private:
  std::shared_ptr<RegisteredMemory> memory_;
  int remoteRank_;
  std::shared_future<RegisteredMemory> future_;
};

/// Memory receive operation for batching.
class RecvMemoryOperation : public Operation {
 public:
  RecvMemoryOperation(std::shared_ptr<Communicator> comm, int remoteRank, int tag = 0);
  
  GroupResult execute() override;
  bool isComplete() const override;
  
  /// Get the receive future (available after execute() is called).
  std::shared_future<RegisteredMemory> getFuture() const { return future_; }

 private:
  int remoteRank_;
  std::shared_future<RegisteredMemory> future_;
};

/// Custom operation for user-defined batching.
class CustomOperation : public Operation {
 public:
  using ExecuteFunction = std::function<GroupResult()>;
  using IsCompleteFunction = std::function<bool()>;
  using CancelFunction = std::function<void()>;

  CustomOperation(std::shared_ptr<Communicator> comm, ExecuteFunction executeFunc, 
                 IsCompleteFunction isCompleteFunc, CancelFunction cancelFunc = nullptr, int tag = 0);
  
  GroupResult execute() override;
  bool isComplete() const override;
  void cancel() override;

 private:
  ExecuteFunction executeFunc_;
  IsCompleteFunction isCompleteFunc_;
  CancelFunction cancelFunc_;
};

/// Job state for tracking group execution.
enum class GroupJobState {
  Created,      ///< Job created but not started
  Running,      ///< Job is running
  Done,         ///< Job completed successfully
  Failed,       ///< Job failed
  Cancelled     ///< Job was cancelled
};

/// Internal job structure for async execution.
struct GroupJob {
  std::atomic<GroupJobState> state{GroupJobState::Created};
  std::vector<std::unique_ptr<Operation>> operations;
  std::vector<std::shared_ptr<Communicator>> communicators;
  std::thread executionThread;
  std::atomic<bool> abortFlag{false};
  GroupResult result{GroupResult::Success};
  
  // Synchronization
  std::mutex operationsMutex;
  std::condition_variable completionCV;
  
  // For blocking/non-blocking mode
  bool isBlocking{true};
  
  GroupJob() = default;
  ~GroupJob();
  
  // Non-copyable, non-movable
  GroupJob(const GroupJob&) = delete;
  GroupJob& operator=(const GroupJob&) = delete;
  GroupJob(GroupJob&&) = delete;
  GroupJob& operator=(GroupJob&&) = delete;
};

/// Main group management class.
class GroupManager {
 public:
  GroupManager();
  ~GroupManager();

  /// Start a new group. Operations added after this call will be batched together.
  /// @return GroupResult indicating success or failure.
  static GroupResult groupStart();

  /// End the current group and execute all batched operations.
  /// @param blocking If true, wait for all operations to complete before returning.
  /// @return GroupResult indicating success or failure.
  static GroupResult groupEnd(bool blocking = true);

  /// Add an operation to the current group.
  /// @param operation The operation to add.
  /// @return GroupResult indicating success or failure.
  static GroupResult addOperation(std::unique_ptr<Operation> operation);

  /// Add a connection operation to the current group.
  /// @param comm The communicator to use.
  /// @param localConfig The local endpoint configuration.
  /// @param remoteRank The remote rank to connect to.
  /// @param tag The tag for this operation.
  /// @return Shared pointer to the connect operation.
  static std::shared_ptr<ConnectOperation> addConnect(std::shared_ptr<Communicator> comm, 
                                                     EndpointConfig localConfig, int remoteRank, int tag = 0);

  /// Add a send memory operation to the current group.
  /// @param comm The communicator to use.
  /// @param memory The memory to send.
  /// @param remoteRank The remote rank to send to.
  /// @param tag The tag for this operation.
  /// @return Shared pointer to the send operation.
  static std::shared_ptr<SendMemoryOperation> addSendMemory(std::shared_ptr<Communicator> comm,
                                                           std::shared_ptr<RegisteredMemory> memory,
                                                           int remoteRank, int tag = 0);

  /// Add a receive memory operation to the current group.
  /// @param comm The communicator to use.
  /// @param remoteRank The remote rank to receive from.
  /// @param tag The tag for this operation.
  /// @return Shared pointer to the receive operation.
  static std::shared_ptr<RecvMemoryOperation> addRecvMemory(std::shared_ptr<Communicator> comm,
                                                           int remoteRank, int tag = 0);

  /// Add a custom operation to the current group.
  /// @param comm The communicator to use.
  /// @param executeFunc Function to execute the operation.
  /// @param isCompleteFunc Function to check if operation is complete.
  /// @param cancelFunc Optional function to cancel the operation.
  /// @param tag The tag for this operation.
  /// @return Shared pointer to the custom operation.
  static std::shared_ptr<CustomOperation> addCustom(std::shared_ptr<Communicator> comm,
                                                   CustomOperation::ExecuteFunction executeFunc,
                                                   CustomOperation::IsCompleteFunction isCompleteFunc,
                                                   CustomOperation::CancelFunction cancelFunc = nullptr,
                                                   int tag = 0);

  /// Check if we are currently in a group.
  /// @return True if in a group, false otherwise.
  static bool inGroup();

  /// Get the current group depth (for nested groups).
  /// @return The current nesting depth.
  static int getGroupDepth();

  /// Wait for the current group to complete (if non-blocking).
  /// @param timeoutMs Timeout in milliseconds (-1 for no timeout).
  /// @return GroupResult indicating success or failure.
  static GroupResult waitForCompletion(int timeoutMs = -1);

  /// Abort the current group execution.
  static void abortGroup();

  /// Cleanup group state (for testing).
  static void cleanupGroup();

  /// Get the current group error state.
  /// @return The current group error.
  static GroupResult getGroupError();

 private:
  /// Execute a group job.
  /// @param job The job to execute.
  static void executeGroupJob(std::shared_ptr<GroupJob> job);

  // Thread-local state
  static thread_local int groupDepth_;
  static thread_local std::shared_ptr<GroupJob> currentJob_;
  static thread_local GroupResult groupError_;
  static thread_local bool groupBlocking_;
};

/// RAII helper for group management.
class GroupScope {
 public:
  /// Constructor that calls groupStart().
  /// @param blocking Whether the group should be blocking.
  GroupScope(bool blocking = true);

  /// Destructor that calls groupEnd().
  ~GroupScope();

  /// Check if the group is valid.
  /// @return True if group was started successfully.
  bool isValid() const { return valid_; }

  /// Get the result of group operations.
  /// @return The current group result.
  GroupResult getResult() const;

  // Non-copyable, non-movable
  GroupScope(const GroupScope&) = delete;
  GroupScope& operator=(const GroupScope&) = delete;
  GroupScope(GroupScope&&) = delete;
  GroupScope& operator=(GroupScope&&) = delete;

 private:
  bool valid_;
  bool blocking_;
};

/// Convenience functions for group operations.

/// Execute multiple connection operations in a group.
/// @param connections Vector of connection parameters (comm, localConfig, remoteRank, tag).
/// @param blocking Whether to wait for completion.
/// @return Vector of connection futures.
std::vector<std::shared_future<std::shared_ptr<Connection>>> 
groupConnect(const std::vector<std::tuple<std::shared_ptr<Communicator>, EndpointConfig, int, int>>& connections,
             bool blocking = true);

/// Execute multiple send memory operations in a group.
/// @param sends Vector of send parameters (comm, memory, remoteRank, tag).
/// @param blocking Whether to wait for completion.
/// @return Vector of send futures.
std::vector<std::shared_future<RegisteredMemory>>
groupSendMemory(const std::vector<std::tuple<std::shared_ptr<Communicator>, std::shared_ptr<RegisteredMemory>, int, int>>& sends,
                bool blocking = true);

/// Execute multiple receive memory operations in a group.
/// @param recvs Vector of receive parameters (comm, remoteRank, tag).
/// @param blocking Whether to wait for completion.
/// @return Vector of receive futures.
std::vector<std::shared_future<RegisteredMemory>>
groupRecvMemory(const std::vector<std::tuple<std::shared_ptr<Communicator>, int, int>>& recvs,
                bool blocking = true);

}  // namespace mscclpp

#endif  // MSCCLPP_GROUP_HPP_