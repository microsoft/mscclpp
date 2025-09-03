// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/dynamic_execution_plan.hpp>
#include <mscclpp/core.hpp>
#include <vector>
#include <iostream>
#include <memory>
#include <numeric>
#include <mpi.h>

int main(int argc, char* argv[]) {
  // Initialize MPI
  MPI_Init(&argc, &argv);
  
  int rank, numRanks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
  
  try {
    std::cout << "Rank " << rank << " of " << numRanks << " starting..." << std::endl;
    
    // Create TcpBootstrap for MSCCLPP
    auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, numRanks);
    
    // Create a unique ID (rank 0 creates and broadcasts)
    mscclpp::UniqueId uniqueId;
    if (rank == 0) {
      uniqueId = bootstrap->createUniqueId();
    }
    MPI_Bcast(&uniqueId, sizeof(uniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    // Initialize bootstrap with the unique ID
    bootstrap->initialize(uniqueId);
    
    // Create communicator
    auto comm = std::make_shared<mscclpp::Communicator>(bootstrap);
    
    std::cout << "Rank " << rank << ": MSCCLPP communicator initialized" << std::endl;
    
    // Load dynamic execution plan template
    std::string planPath = "test/dynamic_alltoallv_plan.json";
    auto dynamicPlan = std::make_shared<mscclpp::DynamicExecutionPlan>(planPath, rank);
    
    std::cout << "Rank " << rank << ": Dynamic execution plan loaded" << std::endl;
    
    // Setup variable send/recv sizes for each peer
    std::vector<size_t> sendSizes(numRanks);
    std::vector<size_t> recvSizes(numRanks);
    
    // Example: different message sizes per peer
    for (int i = 0; i < numRanks; ++i) {
      // Each rank sends different amounts to different peers
      sendSizes[i] = 1024 * (rank + 1) * (i + 1);  // Variable sizes based on rank and peer
      recvSizes[i] = 1024 * (i + 1) * (rank + 1);  // Corresponding receive sizes
    }
    
    // Calculate total buffer sizes
    size_t totalSendSize = std::accumulate(sendSizes.begin(), sendSizes.end(), 0UL);
    size_t totalRecvSize = std::accumulate(recvSizes.begin(), recvSizes.end(), 0UL);
    
    std::cout << "Rank " << rank << ": Total send size: " << totalSendSize
              << ", Total recv size: " << totalRecvSize << std::endl;
    
    // Print send sizes for debugging
    std::cout << "Rank " << rank << " send sizes: ";
    for (int i = 0; i < numRanks; ++i) {
      std::cout << sendSizes[i] << " ";
    }
    std::cout << std::endl;
    
    // Allocate CPU buffers for testing (in real use, these would be GPU buffers)
    std::vector<char> sendBuffer(totalSendSize);
    std::vector<char> recvBuffer(totalRecvSize);
    
    // Initialize send buffer with rank-specific pattern
    for (size_t i = 0; i < totalSendSize; ++i) {
      sendBuffer[i] = static_cast<char>((rank * 0x10) + (i % 256));
    }
    
    std::cout << "Rank " << rank << ": Buffers allocated and initialized" << std::endl;
    
    // Synchronize all ranks before starting the test
    MPI_Barrier(MPI_COMM_WORLD);
    
    std::cout << "Rank " << rank << ": Starting dynamic all-to-allv execution..." << std::endl;
    
    // Execute dynamic all-to-allv
    bool success = mscclpp::DynamicAllToAllv::execute(
      comm, dynamicPlan, sendBuffer.data(), recvBuffer.data(), sendSizes, recvSizes);
    
    if (success) {
      std::cout << "Rank " << rank << ": Dynamic all-to-allv completed successfully!" << std::endl;
      
      // Verify some received data
      std::cout << "Rank " << rank << ": First few received bytes: ";
      for (int i = 0; i < std::min(10, static_cast<int>(totalRecvSize)); ++i) {
        std::cout << std::hex << static_cast<int>(static_cast<unsigned char>(recvBuffer[i])) << " ";
      }
      std::cout << std::dec << std::endl;
    } else {
      std::cout << "Rank " << rank << ": Dynamic all-to-allv failed!" << std::endl;
    }
    
    // Final synchronization
    MPI_Barrier(MPI_COMM_WORLD);
    
    if (rank == 0) {
      std::cout << "All ranks completed dynamic all-to-allv test" << std::endl;
    }
    
  } catch (const std::exception& e) {
    std::cerr << "Rank " << rank << " Error: " << e.what() << std::endl;
    MPI_Finalize();
    return 1;
  }
  
  MPI_Finalize();
  return 0;
}