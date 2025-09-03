// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/dynamic_execution_plan.hpp>
#include <mscclpp/core.hpp>
#include <vector>
#include <iostream>
#include <memory>
#include <numeric>

int main(int argc, char* argv[]) {
  // Initialize MSCCLPP (implementation-specific)
  // ...
  
  try {
    // Create bootstrap (implementation-specific)
    // auto bootstrap = std::make_shared<SomeBootstrap>(/* parameters */);
    
    // Create communicator
    // auto comm = std::make_shared<mscclpp::Communicator>(bootstrap);
    
    // For demonstration, we'll assume we have a communicator
    std::shared_ptr<mscclpp::Communicator> comm = nullptr;
    
    if (!comm) {
      std::cout << "This example requires a properly initialized communicator." << std::endl;
      std::cout << "Please set up bootstrap and communicator according to your environment." << std::endl;
      return 0;
    }
    
    // Get rank and size from bootstrap
    int rank = comm->bootstrap()->getRank();
    int numRanks = comm->bootstrap()->getNranks();
    
    // Load dynamic execution plan template
    auto dynamicPlan = std::make_shared<mscclpp::DynamicExecutionPlan>(
      "test/dynamic_alltoallv_plan.json", rank);
    
    // Setup variable send/recv sizes for each peer
    std::vector<size_t> sendSizes(numRanks);
    std::vector<size_t> recvSizes(numRanks);
    
    // Example: different message sizes per peer based on some algorithm
    for (int i = 0; i < numRanks; ++i) {
      sendSizes[i] = 1024 * (i + 1);  // Variable sizes
      recvSizes[i] = 2048 * (i + 1);  // Variable sizes
    }
    
    // Allocate buffers
    size_t totalSendSize = std::accumulate(sendSizes.begin(), sendSizes.end(), 0UL);
    size_t totalRecvSize = std::accumulate(recvSizes.begin(), recvSizes.end(), 0UL);
    
    void* sendBuffer = nullptr;  // Allocate appropriate buffer
    void* recvBuffer = nullptr;  // Allocate appropriate buffer
    
    // Execute dynamic all-to-allv
    bool success = mscclpp::DynamicAllToAllv::execute(
      comm, dynamicPlan, sendBuffer, recvBuffer, sendSizes, recvSizes);
    
    if (success) {
      std::cout << "Dynamic all-to-allv completed successfully!" << std::endl;
    } else {
      std::cout << "Dynamic all-to-allv failed!" << std::endl;
    }
    
    // Cleanup buffers
    // ...
    
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  
  return 0;
}