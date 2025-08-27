// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/group.hpp>

#include <algorithm>
#include <chrono>
#include <mscclpp/core.hpp>

#include "debug.h"

namespace mscclpp {

// Thread-local variables for group management
thread_local int GroupManager::groupDepth_ = 0;
thread_local std::shared_ptr<GroupJob> GroupManager::currentJob_ = nullptr;
thread_local GroupResult GroupManager::groupError_ = GroupResult::Success;
thread_local bool GroupManager::groupBlocking_ = true;

ErrorCode groupResultToErrorCode(GroupResult result) {
  switch (result) {
    case GroupResult::Success:
      // Since there's no Success in ErrorCode, we'll use a cast to represent success
      // In practice, successful operations shouldn't need to convert to ErrorCode
      // We'll use a special case where we return the first ErrorCode value
      return static_cast<ErrorCode>(0);  // Cast to first enum value
    case GroupResult::InProgress:
      return ErrorCode::InternalError;  // Should not be exposed as exception
    case GroupResult::InvalidUsage:
      return ErrorCode::InvalidUsage;
    case GroupResult::InternalError:
      return ErrorCode::InternalError;
    case GroupResult::Timeout:
      return ErrorCode::Timeout;
    default:
      return ErrorCode::InternalError;
  }
}

//=============================================================================
// Operation implementations
//=============================================================================

Operation::Operation(OperationType type, std::shared_ptr<Communicator> comm, int tag)
    : type_(type), comm_(comm), tag_(tag) {}

ConnectOperation::ConnectOperation(std::shared_ptr<Communicator> comm, EndpointConfig localConfig, 
                                 int remoteRank, int tag)
    : Operation(OperationType::Connect, comm, tag), localConfig_(localConfig), remoteRank_(remoteRank) {}

GroupResult ConnectOperation::execute() {
  try {
    future_ = comm_->connect(localConfig_, remoteRank_, tag_);
    return GroupResult::Success;
  } catch (const Error& e) {
    INFO(MSCCLPP_INIT, "ConnectOperation failed: %s", e.what());
    return GroupResult::InternalError;
  } catch (...) {
    INFO(MSCCLPP_INIT, "ConnectOperation failed with unknown error");
    return GroupResult::InternalError;
  }
}

bool ConnectOperation::isComplete() const {
  if (!future_.valid()) return false;
  return future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

SendMemoryOperation::SendMemoryOperation(std::shared_ptr<Communicator> comm, 
                                       std::shared_ptr<RegisteredMemory> memory,
                                       int remoteRank, int tag)
    : Operation(OperationType::SendMemory, comm, tag), memory_(memory), remoteRank_(remoteRank) {}

GroupResult SendMemoryOperation::execute() {
  try {
    // sendMemory expects RegisteredMemory by value, not shared_ptr
    comm_->sendMemory(*memory_, remoteRank_, tag_);
    // Since sendMemory is void, we create a fulfilled future
    std::promise<RegisteredMemory> promise;
    promise.set_value(*memory_);
    future_ = promise.get_future().share();
    return GroupResult::Success;
  } catch (const Error& e) {
    INFO(MSCCLPP_INIT, "SendMemoryOperation failed: %s", e.what());
    return GroupResult::InternalError;
  } catch (...) {
    INFO(MSCCLPP_INIT, "SendMemoryOperation failed with unknown error");
    return GroupResult::InternalError;
  }
}

bool SendMemoryOperation::isComplete() const {
  if (!future_.valid()) return false;
  return future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

RecvMemoryOperation::RecvMemoryOperation(std::shared_ptr<Communicator> comm, int remoteRank, int tag)
    : Operation(OperationType::RecvMemory, comm, tag), remoteRank_(remoteRank) {}

GroupResult RecvMemoryOperation::execute() {
  try {
    future_ = comm_->recvMemory(remoteRank_, tag_);
    return GroupResult::Success;
  } catch (const Error& e) {
    INFO(MSCCLPP_INIT, "RecvMemoryOperation failed: %s", e.what());
    return GroupResult::InternalError;
  } catch (...) {
    INFO(MSCCLPP_INIT, "RecvMemoryOperation failed with unknown error");
    return GroupResult::InternalError;
  }
}

bool RecvMemoryOperation::isComplete() const {
  if (!future_.valid()) return false;
  return future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready;
}

CustomOperation::CustomOperation(std::shared_ptr<Communicator> comm, ExecuteFunction executeFunc,
                                IsCompleteFunction isCompleteFunc, CancelFunction cancelFunc, int tag)
    : Operation(OperationType::Custom, comm, tag), 
      executeFunc_(executeFunc), 
      isCompleteFunc_(isCompleteFunc), 
      cancelFunc_(cancelFunc) {}

GroupResult CustomOperation::execute() {
  if (executeFunc_) {
    return executeFunc_();
  }
  return GroupResult::InternalError;
}

bool CustomOperation::isComplete() const {
  if (isCompleteFunc_) {
    return isCompleteFunc_();
  }
  return completed_.load();
}

void CustomOperation::cancel() {
  if (cancelFunc_) {
    cancelFunc_();
  }
}

//=============================================================================
// GroupJob implementation
//=============================================================================

GroupJob::~GroupJob() {
  abortFlag.store(true);
  if (executionThread.joinable()) {
    executionThread.join();
  }
}

//=============================================================================
// GroupManager implementation
//=============================================================================

GroupManager::GroupManager() {}

GroupManager::~GroupManager() {}

GroupResult GroupManager::groupStart() {
  if (groupDepth_ == 0) {
    groupError_ = GroupResult::Success;
    currentJob_ = std::make_shared<GroupJob>();
  }
  groupDepth_++;
  
  INFO(MSCCLPP_INIT, "GroupStart: depth=%d", groupDepth_);
  return GroupResult::Success;
}

GroupResult GroupManager::groupEnd(bool blocking) {
  if (groupDepth_ == 0) {
    INFO(MSCCLPP_INIT, "GroupEnd: not in a group call");
    return GroupResult::InvalidUsage;
  }

  groupDepth_--;
  
  if (groupDepth_ == 0) {
    // This is the outermost group, execute it
    if (currentJob_ && !currentJob_->operations.empty()) {
      currentJob_->isBlocking = blocking;
      
      if (blocking) {
        // Execute synchronously
        executeGroupJob(currentJob_);
        GroupResult result = currentJob_->result;
        currentJob_ = nullptr;
        groupError_ = GroupResult::Success;
        INFO(MSCCLPP_INIT, "GroupEnd: blocking execution completed with result %d", static_cast<int>(result));
        return result;
      } else {
        // Execute asynchronously
        auto job = currentJob_;
        currentJob_ = nullptr;
        groupError_ = GroupResult::Success;
        
        job->executionThread = std::thread([job]() {
          executeGroupJob(job);
        });
        
        INFO(MSCCLPP_INIT, "GroupEnd: non-blocking execution started");
        return GroupResult::InProgress;
      }
    } else {
      currentJob_ = nullptr;
      groupError_ = GroupResult::Success;
      INFO(MSCCLPP_INIT, "GroupEnd: no operations to execute");
      return GroupResult::Success;
    }
  }
  
  INFO(MSCCLPP_INIT, "GroupEnd: nested group, depth=%d", groupDepth_);
  return groupError_;
}

GroupResult GroupManager::addOperation(std::unique_ptr<Operation> operation) {
  if (groupDepth_ == 0) {
    INFO(MSCCLPP_INIT, "AddOperation: not in a group");
    return GroupResult::InvalidUsage;
  }
  
  if (!currentJob_) {
    INFO(MSCCLPP_INIT, "AddOperation: no current job");
    return GroupResult::InternalError;
  }
  
  std::lock_guard<std::mutex> lock(currentJob_->operationsMutex);
  currentJob_->operations.push_back(std::move(operation));
  
  INFO(MSCCLPP_INIT, "AddOperation: added operation, total count=%zu", currentJob_->operations.size());
  return GroupResult::Success;
}

std::shared_ptr<ConnectOperation> GroupManager::addConnect(std::shared_ptr<Communicator> comm,
                                                          EndpointConfig localConfig, int remoteRank, int tag) {
  auto op = std::make_shared<ConnectOperation>(comm, localConfig, remoteRank, tag);
  
  // Create a unique_ptr without taking ownership from shared_ptr
  auto result = addOperation(std::unique_ptr<Operation>(new ConnectOperation(comm, localConfig, remoteRank, tag)));
  
  if (result == GroupResult::Success) {
    return op;
  }
  
  return nullptr;
}

std::shared_ptr<SendMemoryOperation> GroupManager::addSendMemory(std::shared_ptr<Communicator> comm,
                                                                std::shared_ptr<RegisteredMemory> memory,
                                                                int remoteRank, int tag) {
  auto op = std::make_shared<SendMemoryOperation>(comm, memory, remoteRank, tag);
  
  auto result = addOperation(std::unique_ptr<Operation>(new SendMemoryOperation(comm, memory, remoteRank, tag)));
  
  if (result == GroupResult::Success) {
    return op;
  }
  
  return nullptr;
}

std::shared_ptr<RecvMemoryOperation> GroupManager::addRecvMemory(std::shared_ptr<Communicator> comm,
                                                                int remoteRank, int tag) {
  auto op = std::make_shared<RecvMemoryOperation>(comm, remoteRank, tag);
  
  auto result = addOperation(std::unique_ptr<Operation>(new RecvMemoryOperation(comm, remoteRank, tag)));
  
  if (result == GroupResult::Success) {
    return op;
  }
  
  return nullptr;
}

std::shared_ptr<CustomOperation> GroupManager::addCustom(std::shared_ptr<Communicator> comm,
                                                        CustomOperation::ExecuteFunction executeFunc,
                                                        CustomOperation::IsCompleteFunction isCompleteFunc,
                                                        CustomOperation::CancelFunction cancelFunc,
                                                        int tag) {
  auto op = std::make_shared<CustomOperation>(comm, executeFunc, isCompleteFunc, cancelFunc, tag);
  
  auto result = addOperation(std::unique_ptr<Operation>(new CustomOperation(comm, executeFunc, isCompleteFunc, cancelFunc, tag)));
  
  if (result == GroupResult::Success) {
    return op;
  }
  
  return nullptr;
}

bool GroupManager::inGroup() {
  return groupDepth_ > 0;
}

int GroupManager::getGroupDepth() {
  return groupDepth_;
}

GroupResult GroupManager::waitForCompletion(int timeoutMs) {
  if (!currentJob_) {
    return GroupResult::Success;
  }
  
  if (currentJob_->executionThread.joinable()) {
    if (timeoutMs < 0) {
      // No timeout
      currentJob_->executionThread.join();
    } else {
      // Wait with timeout
      auto start = std::chrono::steady_clock::now();
      while (currentJob_->state.load() == GroupJobState::Running || 
             currentJob_->state.load() == GroupJobState::Created) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        if (elapsed >= timeoutMs) {
          return GroupResult::Timeout;
        }
      }
      if (currentJob_->executionThread.joinable()) {
        currentJob_->executionThread.join();
      }
    }
  }
  
  return currentJob_->result;
}

void GroupManager::abortGroup() {
  if (currentJob_) {
    currentJob_->abortFlag.store(true);
  }
}

void GroupManager::cleanupGroup() {
  groupDepth_ = 0;
  currentJob_ = nullptr;
  groupError_ = GroupResult::Success;
  groupBlocking_ = true;
}

GroupResult GroupManager::getGroupError() {
  return groupError_;
}

void GroupManager::executeGroupJob(std::shared_ptr<GroupJob> job) {
  job->state.store(GroupJobState::Running);
  
  INFO(MSCCLPP_INIT, "ExecuteGroupJob: starting execution of %zu operations", job->operations.size());
  
  try {
    // Phase 1: Execute all operations
    for (auto& operation : job->operations) {
      if (job->abortFlag.load()) {
        INFO(MSCCLPP_INIT, "ExecuteGroupJob: aborted during execution");
        job->result = GroupResult::InternalError;
        job->state.store(GroupJobState::Cancelled);
        return;
      }
      
      auto result = operation->execute();
      if (result != GroupResult::Success) {
        INFO(MSCCLPP_INIT, "ExecuteGroupJob: operation failed with result %d", static_cast<int>(result));
        job->result = result;
        job->state.store(GroupJobState::Failed);
        return;
      }
    }
    
    // Phase 2: Wait for completion if blocking
    if (job->isBlocking) {
      bool allComplete = false;
      const int maxWaitMs = 30000;  // 30 second timeout
      auto startTime = std::chrono::steady_clock::now();
      
      while (!allComplete && !job->abortFlag.load()) {
        allComplete = true;
        for (auto& operation : job->operations) {
          if (!operation->isComplete()) {
            allComplete = false;
            break;
          }
        }
        
        if (!allComplete) {
          auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - startTime).count();
          if (elapsed > maxWaitMs) {
            INFO(MSCCLPP_INIT, "ExecuteGroupJob: timeout waiting for operations to complete");
            job->result = GroupResult::Timeout;
            job->state.store(GroupJobState::Failed);
            return;
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
      }
      
      if (job->abortFlag.load()) {
        INFO(MSCCLPP_INIT, "ExecuteGroupJob: aborted during wait");
        job->result = GroupResult::InternalError;
        job->state.store(GroupJobState::Cancelled);
        return;
      }
    }
    
    job->result = GroupResult::Success;
    job->state.store(GroupJobState::Done);
    INFO(MSCCLPP_INIT, "ExecuteGroupJob: completed successfully");
    
  } catch (const Error& e) {
    INFO(MSCCLPP_INIT, "ExecuteGroupJob: exception: %s", e.what());
    job->result = GroupResult::InternalError;
    job->state.store(GroupJobState::Failed);
  } catch (...) {
    INFO(MSCCLPP_INIT, "ExecuteGroupJob: unknown exception");
    job->result = GroupResult::InternalError;
    job->state.store(GroupJobState::Failed);
  }
}

//=============================================================================
// GroupScope implementation
//=============================================================================

GroupScope::GroupScope(bool blocking) : blocking_(blocking) {
  auto result = GroupManager::groupStart();
  valid_ = (result == GroupResult::Success);
  if (!valid_) {
    INFO(MSCCLPP_INIT, "GroupScope: failed to start group");
  }
}

GroupScope::~GroupScope() {
  if (valid_) {
    auto result = GroupManager::groupEnd(blocking_);
    if (result != GroupResult::Success && result != GroupResult::InProgress) {
      INFO(MSCCLPP_INIT, "GroupScope: group end failed with result %d", static_cast<int>(result));
    }
  }
}

GroupResult GroupScope::getResult() const {
  return GroupManager::getGroupError();
}

//=============================================================================
// Convenience functions
//=============================================================================

std::vector<std::shared_future<std::shared_ptr<Connection>>> 
groupConnect(const std::vector<std::tuple<std::shared_ptr<Communicator>, EndpointConfig, int, int>>& connections,
             bool blocking) {
  std::vector<std::shared_future<std::shared_ptr<Connection>>> futures;
  
  GroupScope scope(blocking);
  if (!scope.isValid()) {
    return futures;
  }
  
  for (const auto& conn : connections) {
    auto op = GroupManager::addConnect(std::get<0>(conn), std::get<1>(conn), std::get<2>(conn), std::get<3>(conn));
    if (op) {
      futures.push_back(op->getFuture());
    }
  }
  
  return futures;
}

std::vector<std::shared_future<RegisteredMemory>>
groupSendMemory(const std::vector<std::tuple<std::shared_ptr<Communicator>, std::shared_ptr<RegisteredMemory>, int, int>>& sends,
                bool blocking) {
  std::vector<std::shared_future<RegisteredMemory>> futures;
  
  GroupScope scope(blocking);
  if (!scope.isValid()) {
    return futures;
  }
  
  for (const auto& send : sends) {
    auto op = GroupManager::addSendMemory(std::get<0>(send), std::get<1>(send), std::get<2>(send), std::get<3>(send));
    if (op) {
      futures.push_back(op->getFuture());
    }
  }
  
  return futures;
}

std::vector<std::shared_future<RegisteredMemory>>
groupRecvMemory(const std::vector<std::tuple<std::shared_ptr<Communicator>, int, int>>& recvs,
                bool blocking) {
  std::vector<std::shared_future<RegisteredMemory>> futures;
  
  GroupScope scope(blocking);
  if (!scope.isValid()) {
    return futures;
  }
  
  for (const auto& recv : recvs) {
    auto op = GroupManager::addRecvMemory(std::get<0>(recv), std::get<1>(recv), std::get<2>(recv));
    if (op) {
      futures.push_back(op->getFuture());
    }
  }
  
  return futures;
}

}  // namespace mscclpp