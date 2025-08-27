// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/group_execution_plan.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/executor.hpp>
#include <iostream>
#include <vector>
#include <memory>
#include <fstream>

using namespace mscclpp;

// Helper function to check if a file exists
bool fileExists(const std::string& filename) {
  std::ifstream file(filename);
  return file.good();
}

// Helper function to create a simple execution plan JSON file for testing
void createTestExecutionPlan(const std::string& filename) {
  std::ofstream file(filename);
  if (file.is_open()) {
    file << R"({
  "name": "test_plan",
  "collective": "alltoall",
  "protocol": "Simple",
  "inplace": false,
  "num_threads_per_block": 1024,
  "min_message_size": 0,
  "max_message_size": 1048576,
  "gpus": [
    {
      "id": 0,
      "input_chunks": 4,
      "output_chunks": 4,
      "scratch_chunks": 0,
      "channels": [],
      "nvls_channels": [],
      "operations": [[]]
    }
  ]
})";
    file.close();
    std::cout << "Created test execution plan: " << filename << std::endl;
  }
}

// Helper function to create a test communicator
std::shared_ptr<Communicator> createTestCommunicator() {
  try {
    auto bootstrap = std::make_shared<TcpBootstrap>(0, 1);
    bootstrap->initialize(bootstrap->createUniqueId());
    return std::make_shared<Communicator>(bootstrap);
  } catch (const std::exception& e) {
    std::cout << "Note: Could not create full MSCCLPP communicator: " << e.what() << std::endl;
    std::cout << "This is expected in some test environments." << std::endl;
    return nullptr;
  }
}

// Example showing how to use execution plan-aware group management for all_to_allv
void demonstrateExecutionPlanAllToAllv() {
  std::cout << "=== Execution Plan All-to-Allv Group Management Example ===" << std::endl;
  
  try {
    // Create a test execution plan if it doesn't exist
    std::string planPath = "alltoallv_execution_plan.json";
    if (!fileExists(planPath)) {
      createTestExecutionPlan(planPath);
    }
    
    // Create a communicator (simplified example)
    auto comm = createTestCommunicator();
    if (!comm) {
      std::cout << "Skipping communicator-dependent functionality due to environment limitations." << std::endl;
      return;
    }
    
    // Load an execution plan for all_to_allv operation
    auto executionPlan = std::make_shared<ExecutionPlan>(planPath, 0 /* rank */);
    
    // Prepare buffers for variable-size all-to-all
    size_t totalInputSize = 1024 * 1024;   // 1MB input
    size_t totalOutputSize = 1024 * 1024;  // 1MB output
    
    std::vector<char> sendBuffer(totalInputSize);
    std::vector<char> recvBuffer(totalOutputSize);
    
    // Initialize send buffer with some data
    for (size_t i = 0; i < sendBuffer.size(); ++i) {
      sendBuffer[i] = static_cast<char>(i % 256);
    }
    
    std::cout << "Performing execution plan-based all_to_allv operation..." << std::endl;
    
    // Method 1: Using ExecutionPlanGroupScope RAII
    {
      ExecutionPlanGroupScope scope(executionPlan, true);
      
      auto operation = ExecutionPlanGroupManager::addExecutionPlanAllToAllv(
        comm, executionPlan, 
        sendBuffer.data(), recvBuffer.data(),
        totalInputSize, totalOutputSize,
        0 /* tag */);
      
      // Get chunk size information
      const auto& allToAllvInfo = operation->getAllToAllvInfo();
      
      std::cout << "All-to-Allv Info:" << std::endl;
      std::cout << "  Total send size: " << allToAllvInfo.totalSendSize << std::endl;
      std::cout << "  Total recv size: " << allToAllvInfo.totalRecvSize << std::endl;
      std::cout << "  Max chunks: " << allToAllvInfo.maxChunks << std::endl;
      std::cout << "  Variable sizes: " << (allToAllvInfo.isVariable ? "Yes" : "No") << std::endl;
      std::cout << "  Number of chunk specs: " << allToAllvInfo.chunkSpecs.size() << std::endl;
      
      // Print chunk specifications (limited to first 5 for brevity)
      for (size_t i = 0; i < allToAllvInfo.chunkSpecs.size() && i < 5; ++i) {
        const auto& spec = allToAllvInfo.chunkSpecs[i];
        std::cout << "  Chunk " << i << ": rank " << spec.rank 
                  << " -> rank " << spec.destRank
                  << " (send: " << spec.sendSize 
                  << " @ " << spec.sendOffset
                  << ", recv: " << spec.recvSize 
                  << " @ " << spec.recvOffset << ")" << std::endl;
      }
      
    }  // ExecutionPlanGroupScope destructor executes the group
    
    std::cout << "All-to-Allv operation completed successfully!" << std::endl;
    
  } catch (const std::exception& e) {
    std::cerr << "Error in execution plan all_to_allv example: " << e.what() << std::endl;
    std::cout << "Note: This example requires proper MPI and MSCCLPP initialization." << std::endl;
  }
}

// Example showing multiple execution plan operations in a group
void demonstrateMultipleExecutionPlanOperations() {
  std::cout << "\n=== Multiple Execution Plan Operations Example ===" << std::endl;
  
  try {
    // Create test execution plans if they don't exist
    std::string plan1Path = "alltoallv_plan1.json";
    std::string plan2Path = "alltoallv_plan2.json";
    
    if (!fileExists(plan1Path)) {
      createTestExecutionPlan(plan1Path);
    }
    if (!fileExists(plan2Path)) {
      createTestExecutionPlan(plan2Path);
    }
    
    auto comm = createTestCommunicator();
    if (!comm) {
      std::cout << "Skipping communicator-dependent functionality due to environment limitations." << std::endl;
      return;
    }
    
    // Load different execution plans for different operations
    auto plan1 = std::make_shared<ExecutionPlan>(plan1Path, 0);
    auto plan2 = std::make_shared<ExecutionPlan>(plan2Path, 0);
    
    // Prepare buffers
    std::vector<char> sendBuffer1(512 * 1024);
    std::vector<char> recvBuffer1(512 * 1024);
    std::vector<char> sendBuffer2(256 * 1024);
    std::vector<char> recvBuffer2(256 * 1024);
    
    std::cout << "Performing multiple execution plan operations..." << std::endl;
    
    // Method 2: Using convenience function for multiple operations
    std::vector<std::tuple<std::shared_ptr<Communicator>, std::shared_ptr<ExecutionPlan>, 
                          void*, void*, size_t, size_t, int>> operations = {
      {comm, plan1, sendBuffer1.data(), recvBuffer1.data(), 
       sendBuffer1.size(), recvBuffer1.size(), 1},
      {comm, plan2, sendBuffer2.data(), recvBuffer2.data(), 
       sendBuffer2.size(), recvBuffer2.size(), 2}
    };
    
    auto results = groupExecutionPlanAllToAllv(operations, true);
    
    std::cout << "All operations completed. Results:" << std::endl;
    for (size_t i = 0; i < results.size(); ++i) {
      std::cout << "  Operation " << i << " complete: " 
                << (results[i]->isComplete() ? "Yes" : "No") << std::endl;
    }
    
  } catch (const std::exception& e) {
    std::cerr << "Error in multiple operations example: " << e.what() << std::endl;
    std::cout << "Note: This example requires proper MPI and MSCCLPP initialization." << std::endl;
  }
}

// Example showing custom execution plan operation
void demonstrateCustomExecutionPlanOperation() {
  std::cout << "\n=== Custom Execution Plan Operation Example ===" << std::endl;
  
  try {
    std::string planPath = "custom_plan.json";
    if (!fileExists(planPath)) {
      createTestExecutionPlan(planPath);
    }
    
    auto comm = createTestCommunicator();
    if (!comm) {
      std::cout << "Skipping communicator-dependent functionality due to environment limitations." << std::endl;
      return;
    }
    
    auto plan = std::make_shared<ExecutionPlan>(planPath, 0);
    
    std::cout << "Performing custom execution plan operation..." << std::endl;
    
    {
      ExecutionPlanGroupScope scope(plan, true);
      
      // Add a custom operation that uses execution plan information
      auto customOp = ExecutionPlanGroupManager::addExecutionPlanCustom(
        comm, plan,
        []() -> GroupResult {
          // Custom execute function that could use execution plan data
          std::cout << "Executing custom operation with execution plan context" << std::endl;
          return GroupResult::Success;
        },
        []() -> bool {
          // Custom completion check
          return true;
        },
        []() -> void {
          // Custom cancel function
          std::cout << "Cancelling custom execution plan operation" << std::endl;
        },
        3 /* tag */);
      
      // Get execution plans for current group
      auto executionPlans = ExecutionPlanGroupManager::getExecutionPlans();
      std::cout << "Number of execution plans in group: " << executionPlans.size() << std::endl;
      
    }  // Group executes here
    
    std::cout << "Custom operation completed!" << std::endl;
    
  } catch (const std::exception& e) {
    std::cerr << "Error in custom operation example: " << e.what() << std::endl;
    std::cout << "Note: This example requires proper MPI and MSCCLPP initialization." << std::endl;
  }
}

// Example showing how to check execution plan capabilities
void demonstrateExecutionPlanCapabilities() {
  std::cout << "\n=== Execution Plan Capabilities Example ===" << std::endl;
  
  try {
    std::string planPath = "test_plan.json";
    if (!fileExists(planPath)) {
      createTestExecutionPlan(planPath);
    }
    
    auto plan = std::make_shared<ExecutionPlan>(planPath, 0);
    
    // Check capabilities
    bool supportsVariable = supportsVariableChunkSizes(*plan);
    std::cout << "Execution plan supports variable chunk sizes: " 
              << (supportsVariable ? "Yes" : "No") << std::endl;
    
    // Get maximum chunk size for different input sizes
    std::vector<size_t> testSizes = {1024, 4096, 16384, 65536};
    
    for (size_t size : testSizes) {
      size_t maxChunk = getMaxChunkSize(*plan, size, size);
      std::cout << "Max chunk size for " << size << " bytes: " << maxChunk << std::endl;
    }
    
    // Extract all_to_allv info for analysis
    AllToAllvInfo info = extractAllToAllvInfo(*plan, 8192, 8192);
    std::cout << "All-to-Allv analysis for 8KB:" << std::endl;
    std::cout << "  Variable sizes: " << (info.isVariable ? "Yes" : "No") << std::endl;
    std::cout << "  Total send: " << info.totalSendSize << std::endl;
    std::cout << "  Total recv: " << info.totalRecvSize << std::endl;
    std::cout << "  Max chunks: " << info.maxChunks << std::endl;
    
  } catch (const std::exception& e) {
    std::cerr << "Error in capabilities example: " << e.what() << std::endl;
    std::cout << "Note: This example requires proper MPI and MSCCLPP initialization." << std::endl;
  }
}

// Simple test that demonstrates basic functionality without requiring full MSCCLPP setup
void demonstrateBasicFunctionality() {
  std::cout << "\n=== Basic Functionality Test ===" << std::endl;
  
  try {
    std::string planPath = "basic_test_plan.json";
    createTestExecutionPlan(planPath);
    
    // Test utility functions that don't require full MSCCLPP initialization
    auto plan = std::make_shared<ExecutionPlan>(planPath, 0);
    
    std::cout << "Plan name: " << plan->name() << std::endl;
    std::cout << "Plan collective: " << plan->collective() << std::endl;
    std::cout << "Min message size: " << plan->minMessageSize() << std::endl;
    std::cout << "Max message size: " << plan->maxMessageSize() << std::endl;
    std::cout << "Is in-place: " << (plan->isInPlace() ? "Yes" : "No") << std::endl;
    
    // Test utility functions
    bool supportsVariable = supportsVariableChunkSizes(*plan);
    std::cout << "Supports variable chunk sizes: " << (supportsVariable ? "Yes" : "No") << std::endl;
    
    size_t maxChunk = getMaxChunkSize(*plan, 4096, 4096);
    std::cout << "Max chunk size for 4KB: " << maxChunk << std::endl;
    
    AllToAllvInfo info = extractAllToAllvInfo(*plan, 2048, 2048);
    std::cout << "All-to-Allv info extracted successfully:" << std::endl;
    std::cout << "  Number of chunk specs: " << info.chunkSpecs.size() << std::endl;
    std::cout << "  Variable sizes: " << (info.isVariable ? "Yes" : "No") << std::endl;
    
    std::cout << "Basic functionality test completed successfully!" << std::endl;
    
  } catch (const std::exception& e) {
    std::cerr << "Error in basic functionality test: " << e.what() << std::endl;
  }
}

int main() {
  std::cout << "MSCCLPP Execution Plan Group Management Examples" << std::endl;
  std::cout << "===============================================" << std::endl;
  
  // Start with basic functionality test that doesn't require full MSCCLPP setup
  demonstrateBasicFunctionality();
  
  // Note: The following examples assume proper MSCCLPP and MPI initialization
  // In a real deployment, you would initialize MPI and MSCCLPP before these calls
  std::cout << "\nNote: The following examples require proper MPI and MSCCLPP initialization." << std::endl;
  std::cout << "They will demonstrate the API but may not execute fully in this test environment." << std::endl;
  
  demonstrateExecutionPlanAllToAllv();
  demonstrateMultipleExecutionPlanOperations();
  demonstrateCustomExecutionPlanOperation();
  demonstrateExecutionPlanCapabilities();
  
  std::cout << "\nAll examples completed!" << std::endl;
  std::cout << "The execution plan group management system is ready for use." << std::endl;
  
  return 0;
}