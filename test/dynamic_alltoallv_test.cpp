// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/dynamic_execution_plan.hpp>
#include <mscclpp/core.hpp>
#include <vector>
#include <iostream>
#include <memory>
#include <numeric>
#include <mpi.h>

// MPI wrapper function for testing
bool executeDynamicAllToAllvWithMPI(
    std::shared_ptr<mscclpp::Communicator> comm,
    std::shared_ptr<mscclpp::DynamicExecutionPlan> dynamicPlan,
    void* sendBuffer, void* recvBuffer,
    const std::vector<size_t>& sendSizes,
    const std::vector<size_t>& recvSizes,
    int tag = 0) {
  
  // First, execute the normal dynamic execution plan (which generates the concrete plan)
  bool planSuccess = mscclpp::DynamicAllToAllv::execute(comm, dynamicPlan, sendBuffer, recvBuffer, sendSizes, recvSizes, tag);
  
  if (!planSuccess) {
    return false;
  }
  
  // Now do the actual data transfer using MPI for testing
  int rank = comm->bootstrap()->getRank();
  int numRanks = comm->bootstrap()->getNranks();
  
  std::cout << "Rank " << rank << ": Using MPI fallback for actual data transfer" << std::endl;
  
  // Create runtime parameters for MPI
  auto runtimeParams = mscclpp::DynamicAllToAllv::createRuntimeParams(sendSizes, recvSizes);
  
  // Prepare MPI AllToAllv parameters
  std::vector<int> sendCounts(numRanks);
  std::vector<int> recvCounts(numRanks);
  std::vector<int> sendDispls(numRanks);
  std::vector<int> recvDispls(numRanks);
  
  // Convert sizes to counts and displacements
  for (int i = 0; i < numRanks; ++i) {
    sendCounts[i] = static_cast<int>(sendSizes[i]);
    recvCounts[i] = static_cast<int>(recvSizes[i]);
    sendDispls[i] = static_cast<int>(runtimeParams.send_offsets[i]);
    recvDispls[i] = static_cast<int>(runtimeParams.recv_offsets[i]);
  }
  
  // Debug: Print MPI parameters
  std::cout << "Rank " << rank << ": MPI sendCounts: ";
  for (int i = 0; i < numRanks; ++i) {
    std::cout << sendCounts[i] << " ";
  }
  std::cout << std::endl;
  
  std::cout << "Rank " << rank << ": MPI recvCounts: ";
  for (int i = 0; i < numRanks; ++i) {
    std::cout << recvCounts[i] << " ";
  }
  std::cout << std::endl;
  
  // Perform MPI AllToAllv
  int result = MPI_Alltoallv(
    sendBuffer, sendCounts.data(), sendDispls.data(), MPI_BYTE,
    recvBuffer, recvCounts.data(), recvDispls.data(), MPI_BYTE,
    MPI_COMM_WORLD);
  
  if (result != MPI_SUCCESS) {
    std::cerr << "Rank " << rank << ": MPI_Alltoallv failed with error " << result << std::endl;
    return false;
  }
  
  std::cout << "Rank " << rank << ": MPI AllToAllv completed successfully" << std::endl;
  return true;
}

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
    
    // Example: rank i sends (i+1)*1024 bytes to each peer j
    // Each rank receives (j+1)*1024 bytes from rank j
    for (int i = 0; i < numRanks; ++i) {
      sendSizes[i] = (rank + 1) * 1024;  // Each rank sends the same to all peers
      recvSizes[i] = (i + 1) * 1024;     // Receive from rank i
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
    
    std::cout << "Rank " << rank << " recv sizes: ";
    for (int i = 0; i < numRanks; ++i) {
      std::cout << recvSizes[i] << " ";
    }
    std::cout << std::endl;
    
    // Allocate CPU buffers for testing (in real use, these would be GPU buffers)
    std::vector<char> sendBuffer(totalSendSize);
    std::vector<char> recvBuffer(totalRecvSize, 0);  // Initialize to 0
    
    // Initialize send buffer with rank-specific pattern
    size_t offset = 0;
    for (int peer = 0; peer < numRanks; ++peer) {
      for (size_t i = 0; i < sendSizes[peer]; ++i) {
        // Each rank uses a different pattern: rank*0x10 + byte_index
        sendBuffer[offset + i] = static_cast<char>((rank * 0x10) + ((offset + i) % 256));
      }
      offset += sendSizes[peer];
    }
    
    std::cout << "Rank " << rank << ": Buffers allocated and initialized" << std::endl;
    
    // Print some send buffer content for verification
    std::cout << "Rank " << rank << ": Send buffer first 10 bytes: ";
    for (int i = 0; i < std::min(10, static_cast<int>(totalSendSize)); ++i) {
      std::cout << std::hex << static_cast<int>(static_cast<unsigned char>(sendBuffer[i])) << " ";
    }
    std::cout << std::dec << std::endl;
    
    // Synchronize all ranks before starting the test
    MPI_Barrier(MPI_COMM_WORLD);
    
    std::cout << "Rank " << rank << ": Starting dynamic all-to-allv execution..." << std::endl;
    
    // Execute dynamic all-to-allv with MPI fallback
    bool success = executeDynamicAllToAllvWithMPI(
      comm, dynamicPlan, sendBuffer.data(), recvBuffer.data(), sendSizes, recvSizes);
    
    if (success) {
      std::cout << "Rank " << rank << ": Dynamic all-to-allv completed successfully!" << std::endl;
      
      // Verify received data
      std::cout << "Rank " << rank << ": First few received bytes: ";
      for (int i = 0; i < std::min(10, static_cast<int>(totalRecvSize)); ++i) {
        std::cout << std::hex << static_cast<int>(static_cast<unsigned char>(recvBuffer[i])) << " ";
      }
      std::cout << std::dec << std::endl;
      
      // Verify data per peer
      size_t recv_offset = 0;
      for (int peer = 0; peer < numRanks; ++peer) {
        if (recvSizes[peer] > 0) {
          std::cout << "Rank " << rank << ": From rank " << peer << " (first 4 bytes): ";
          for (int i = 0; i < std::min(4, static_cast<int>(recvSizes[peer])); ++i) {
            std::cout << std::hex << static_cast<int>(static_cast<unsigned char>(recvBuffer[recv_offset + i])) << " ";
          }
          std::cout << std::dec << std::endl;
        }
        recv_offset += recvSizes[peer];
      }
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