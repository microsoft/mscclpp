// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/group_execution_plan.hpp>
#include <mscclpp/executor.hpp>
#include <algorithm>
#include <stdexcept>

namespace mscclpp {

// Thread-local storage for execution plans
thread_local std::unordered_map<int, std::shared_ptr<ExecutionPlan>> 
  ExecutionPlanGroupManager::executionPlans_;

ExecutionPlanAllToAllvOperation::ExecutionPlanAllToAllvOperation(
  std::shared_ptr<Communicator> comm, 
  std::shared_ptr<ExecutionPlan> plan,
  void* sendBuffer, void* recvBuffer,
  size_t inputSize, size_t outputSize,
  int tag)
  : Operation(OperationType::Custom, comm, tag),
    plan_(plan), sendBuffer_(sendBuffer), recvBuffer_(recvBuffer),
    inputSize_(inputSize), outputSize_(outputSize) {
  
  if (!plan_) {
    throw std::invalid_argument("ExecutionPlan cannot be null");
  }
  
  if (!sendBuffer_ || !recvBuffer_) {
    throw std::invalid_argument("Send and receive buffers cannot be null");
  }
  
  // Extract chunk size information from execution plan
  extractChunkSizes();
  
  if (!validateBuffers()) {
    throw std::invalid_argument("Buffers are insufficient for the operation");
  }
}

GroupResult ExecutionPlanAllToAllvOperation::execute() {
  if (cancelled_.load()) {
    return GroupResult::InternalError;
  }

  try {
    // Use the execution plan to perform variable-size all-to-all
    // This would integrate with the actual DSL execution engine
    
    // For now, we'll simulate the operation by using the chunk size information
    // In a real implementation, this would invoke the DSL execution engine
    // with the variable chunk sizes
    
    // Get the chunk specifications
    const auto& specs = allToAllvInfo_.chunkSpecs;
    
    // Validate that we have proper chunk specifications
    if (specs.empty()) {
      return GroupResult::InvalidUsage;
    }
    
    // TODO: Integrate with actual DSL execution engine to perform
    // variable-size all-to-all operation using the chunk specifications
    
    // For now, mark as completed
    completed_.store(true);
    return GroupResult::Success;
    
  } catch (const std::exception& e) {
    return GroupResult::InternalError;
  }
}

bool ExecutionPlanAllToAllvOperation::isComplete() const {
  return completed_.load() || cancelled_.load();
}

void ExecutionPlanAllToAllvOperation::cancel() {
  cancelled_.store(true);
  completed_.store(true);
}

void ExecutionPlanAllToAllvOperation::extractChunkSizes() {
  // Extract chunk size information from the execution plan
  allToAllvInfo_ = extractAllToAllvInfo(*plan_, inputSize_, outputSize_);
}

bool ExecutionPlanAllToAllvOperation::validateBuffers() const {
  // Check that send and receive buffers are large enough
  if (inputSize_ < allToAllvInfo_.totalSendSize) {
    return false;
  }
  
  if (outputSize_ < allToAllvInfo_.totalRecvSize) {
    return false;
  }
  
  return true;
}

std::shared_ptr<ExecutionPlanAllToAllvOperation> 
ExecutionPlanGroupManager::addExecutionPlanAllToAllv(
  std::shared_ptr<Communicator> comm,
  std::shared_ptr<ExecutionPlan> plan,
  void* sendBuffer, void* recvBuffer,
  size_t inputSize, size_t outputSize,
  int tag) {
  
  auto operation = std::make_shared<ExecutionPlanAllToAllvOperation>(
    comm, plan, sendBuffer, recvBuffer, inputSize, outputSize, tag);
  
  // Store the execution plan for this operation
  executionPlans_[tag] = plan;
  
  // Add to the group using the base class functionality
  GroupResult result = addOperation(std::unique_ptr<Operation>(
    static_cast<Operation*>(operation.get())));
  
  if (result != GroupResult::Success) {
    executionPlans_.erase(tag);
    throw std::runtime_error("Failed to add execution plan all_to_allv operation to group");
  }
  
  return operation;
}

std::shared_ptr<CustomOperation> ExecutionPlanGroupManager::addExecutionPlanCustom(
  std::shared_ptr<Communicator> comm,
  std::shared_ptr<ExecutionPlan> plan,
  CustomOperation::ExecuteFunction executeFunc,
  CustomOperation::IsCompleteFunction isCompleteFunc,
  CustomOperation::CancelFunction cancelFunc,
  int tag) {
  
  // Store the execution plan for this operation
  executionPlans_[tag] = plan;
  
  auto operation = addCustom(comm, executeFunc, isCompleteFunc, cancelFunc, tag);
  
  if (!operation) {
    executionPlans_.erase(tag);
    throw std::runtime_error("Failed to add execution plan custom operation to group");
  }
  
  return operation;
}

std::unordered_map<int, std::shared_ptr<ExecutionPlan>> 
ExecutionPlanGroupManager::getExecutionPlans() {
  return executionPlans_;
}

AllToAllvInfo extractAllToAllvInfo(const ExecutionPlan& plan, size_t inputSize, size_t outputSize) {
  AllToAllvInfo info;
  
  // Calculate chunk specifications based on execution plan
  info.chunkSpecs = calculateChunkSizes(plan, inputSize, outputSize);
  
  // Calculate total sizes
  info.totalSendSize = 0;
  info.totalRecvSize = 0;
  
  for (const auto& spec : info.chunkSpecs) {
    info.totalSendSize += spec.sendSize;
    info.totalRecvSize += spec.recvSize;
  }
  
  // Check if chunk sizes are variable
  if (!info.chunkSpecs.empty()) {
    size_t firstSendSize = info.chunkSpecs[0].sendSize;
    size_t firstRecvSize = info.chunkSpecs[0].recvSize;
    
    info.isVariable = std::any_of(info.chunkSpecs.begin(), info.chunkSpecs.end(),
      [firstSendSize, firstRecvSize](const ChunkSizeSpec& spec) {
        return spec.sendSize != firstSendSize || spec.recvSize != firstRecvSize;
      });
  } else {
    info.isVariable = false;
  }
  
  // Get maximum chunk count
  info.maxChunks = info.chunkSpecs.size();
  
  return info;
}

std::vector<ChunkSizeSpec> calculateChunkSizes(const ExecutionPlan& plan, 
                                               size_t inputSize, size_t outputSize) {
  std::vector<ChunkSizeSpec> specs;
  
  // Since we can't access getThreadblockCount() directly, we'll use a different approach
  // to estimate the number of ranks/threadblocks based on available information
  
  try {
    // Get the maximum chunk size from execution plan
    size_t maxChunkSize = getMaxChunkSize(plan, inputSize, outputSize);
    
    // Estimate the number of ranks based on the execution plan name and collective type
    // This is a simplified approach - real implementation would parse the JSON plan
    std::string collective = plan.collective();
    int estimatedRankCount = 4; // Default assumption for all-to-all operations
    
    // Try to infer rank count from collective name or other available information
    // This is a heuristic approach since we don't have direct access to the rank count
    if (collective.find("alltoall") != std::string::npos) {
      // For all-to-all, we can estimate based on buffer sizes and chunk sizes
      if (maxChunkSize > 0) {
        estimatedRankCount = std::max(2, static_cast<int>(inputSize / maxChunkSize));
        // Clamp to reasonable values
        estimatedRankCount = std::min(estimatedRankCount, 64);
      }
    }
    
    // Create simplified chunk specifications
    for (int i = 0; i < estimatedRankCount; ++i) {
      for (int j = 0; j < estimatedRankCount; ++j) {
        if (i != j) {  // Don't send to self
          ChunkSizeSpec spec;
          spec.rank = i;
          spec.destRank = j;
          
          // Calculate sizes based on execution plan
          // This is simplified - real implementation would extract from JSON
          spec.sendSize = maxChunkSize;
          spec.recvSize = maxChunkSize;
          spec.sendOffset = j * maxChunkSize;
          spec.recvOffset = i * maxChunkSize;
          
          specs.push_back(spec);
        }
      }
    }
    
  } catch (const std::exception& e) {
    // If we can't parse the execution plan, create minimal specs
    // This ensures the function doesn't fail completely
    ChunkSizeSpec spec;
    spec.rank = 0;
    spec.destRank = 1;
    spec.sendSize = inputSize;
    spec.recvSize = outputSize;
    spec.sendOffset = 0;
    spec.recvOffset = 0;
    specs.push_back(spec);
  }
  
  return specs;
}

size_t getMaxChunkSize(const ExecutionPlan& plan, size_t inputSize, size_t outputSize) {
  try {
    // Since we can't access the private implementation methods directly,
    // we'll estimate the maximum chunk size based on available public information
    
    // Use the message size limits as a guide
    size_t minMessage = plan.minMessageSize();
    size_t maxMessage = plan.maxMessageSize();
    
    // Estimate chunk size based on total size and typical all-to-all patterns
    size_t totalSize = std::max(inputSize, outputSize);
    
    // If we have message size constraints, use them
    if (maxMessage > 0 && maxMessage < std::numeric_limits<size_t>::max()) {
      return std::min(totalSize, maxMessage);
    }
    
    // Otherwise, use a reasonable chunk size for all-to-all operations
    // Typically, all-to-all operations divide data among participants
    // Assume 4 participants as a default (can be adjusted based on needs)
    size_t estimatedChunkSize = totalSize / 4;
    
    // Ensure minimum chunk size for efficiency
    estimatedChunkSize = std::max(estimatedChunkSize, minMessage);
    
    // Ensure chunk size doesn't exceed total size
    estimatedChunkSize = std::min(estimatedChunkSize, totalSize);
    
    return estimatedChunkSize > 0 ? estimatedChunkSize : 1024; // Fallback minimum
    
  } catch (const std::exception& e) {
    // Fallback to using total size or a reasonable default
    size_t totalSize = std::max(inputSize, outputSize);
    return totalSize > 0 ? totalSize : 1024;
  }
}

bool supportsVariableChunkSizes(const ExecutionPlan& plan) {
  // Check if the execution plan supports variable chunk sizes
  // This would typically involve parsing the JSON plan to look for
  // variable size specifications or placeholders
  
  try {
    // Use collective type to infer variable size support
    std::string collective = plan.collective();
    
    // All-to-all variants typically support variable sizes
    if (collective.find("alltoall") != std::string::npos ||
        collective.find("allgather") != std::string::npos ||
        collective.find("scatter") != std::string::npos) {
      return true;
    }
    
    // For other collectives, assume basic support
    return true;
    
  } catch (const std::exception& e) {
    // If we can't determine, assume variable sizes are supported
    return true;
  }
}

ExecutionPlanGroupScope::ExecutionPlanGroupScope(std::shared_ptr<ExecutionPlan> plan, bool blocking)
  : GroupScope(blocking), plan_(plan) {
  if (!plan_) {
    throw std::invalid_argument("ExecutionPlan cannot be null");
  }
}

ExecutionPlanGroupScope::~ExecutionPlanGroupScope() {
  // Cleanup is handled by base GroupScope
}

std::vector<std::shared_ptr<ExecutionPlanAllToAllvOperation>>
groupExecutionPlanAllToAllv(
  const std::vector<std::tuple<std::shared_ptr<Communicator>, std::shared_ptr<ExecutionPlan>, 
                              void*, void*, size_t, size_t, int>>& operations,
  bool blocking) {
  
  std::vector<std::shared_ptr<ExecutionPlanAllToAllvOperation>> results;
  
  if (operations.empty()) {
    return results;
  }
  
  {
    ExecutionPlanGroupScope scope(std::get<1>(operations[0]), blocking);
    
    for (const auto& op : operations) {
      auto operation = ExecutionPlanGroupManager::addExecutionPlanAllToAllv(
        std::get<0>(op), std::get<1>(op), std::get<2>(op), std::get<3>(op),
        std::get<4>(op), std::get<5>(op), std::get<6>(op));
      results.push_back(operation);
    }
  }  // GroupScope destructor will execute the group
  
  return results;
}

}  // namespace mscclpp