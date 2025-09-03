// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/dynamic_execution_plan.hpp>
#include <mscclpp/core.hpp>
#include <vector>
#include <iostream>
#include <memory>
#include <numeric>
#include <cuda_runtime.h>

int main(int argc, char* argv[]) {
  
  try {
    // For single-GPU testing, use rank 0
    int rank = 0;
    int numRanks = 1;
    
    std::cout << "Rank " << rank << " of " << numRanks << " starting MSCCLPP execution test..." << std::endl;
    
    // Initialize CUDA
    cudaSetDevice(0);
    
    // Create TcpBootstrap for MSCCLPP (single rank for testing)
    auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank, numRanks);
    auto uniqueId = bootstrap->createUniqueId();
    bootstrap->initialize(uniqueId);
    
    // Create communicator
    auto comm = std::make_shared<mscclpp::Communicator>(bootstrap);
    
    std::cout << "Rank " << rank << ": MSCCLPP communicator initialized" << std::endl;
    
    // Load dynamic execution plan template
    std::string planPath = "test/dynamic_alltoallv_plan.json";
    auto dynamicPlan = std::make_shared<mscclpp::DynamicExecutionPlan>(planPath, rank);
    
    std::cout << "Rank " << rank << ": Dynamic execution plan loaded" << std::endl;
    
    // Setup variable send/recv sizes for testing
    std::vector<size_t> sendSizes(numRanks);
    std::vector<size_t> recvSizes(numRanks);
    
    // Example: each rank sends/receives 4KB
    for (int i = 0; i < numRanks; ++i) {
      sendSizes[i] = 4096;  // 4KB per peer
      recvSizes[i] = 4096;  // 4KB per peer
    }
    
    // Calculate total buffer sizes
    size_t totalSendSize = std::accumulate(sendSizes.begin(), sendSizes.end(), 0UL);
    size_t totalRecvSize = std::accumulate(recvSizes.begin(), recvSizes.end(), 0UL);
    
    std::cout << "Rank " << rank << ": Total send size: " << totalSendSize
              << ", Total recv size: " << totalRecvSize << std::endl;
    
    // Allocate GPU buffers
    void* d_sendBuffer;
    void* d_recvBuffer;
    
    cudaMalloc(&d_sendBuffer, totalSendSize);
    cudaMalloc(&d_recvBuffer, totalRecvSize);
    
    // Initialize send buffer with test pattern
    std::vector<char> h_sendBuffer(totalSendSize);
    for (size_t i = 0; i < totalSendSize; ++i) {
      h_sendBuffer[i] = static_cast<char>((rank * 0x10) + (i % 256));
    }
    
    cudaMemcpy(d_sendBuffer, h_sendBuffer.data(), totalSendSize, cudaMemcpyHostToDevice);
    cudaMemset(d_recvBuffer, 0, totalRecvSize);
    
    std::cout << "Rank " << rank << ": GPU buffers allocated and initialized" << std::endl;
    
    std::cout << "Rank " << rank << ": Starting dynamic all-to-allv execution with MSCCLPP..." << std::endl;
    
    // Execute dynamic all-to-allv with MSCCLPP execution engine
    bool success = mscclpp::DynamicAllToAllv::execute(
      comm, dynamicPlan, d_sendBuffer, d_recvBuffer, sendSizes, recvSizes);
    
    if (success) {
      std::cout << "Rank " << rank << ": Dynamic all-to-allv completed successfully!" << std::endl;
      
      // Copy results back to host for verification
      std::vector<char> h_recvBuffer(totalRecvSize);
      cudaMemcpy(h_recvBuffer.data(), d_recvBuffer, totalRecvSize, cudaMemcpyDeviceToHost);
      
      // Verify received data
      std::cout << "Rank " << rank << ": First few received bytes: ";
      for (int i = 0; i < std::min(10, static_cast<int>(totalRecvSize)); ++i) {
        std::cout << std::hex << static_cast<int>(static_cast<unsigned char>(h_recvBuffer[i])) << " ";
      }
      std::cout << std::dec << std::endl;
      
    } else {
      std::cout << "Rank " << rank << ": Dynamic all-to-allv failed!" << std::endl;
    }
    
    // Cleanup GPU memory
    cudaFree(d_sendBuffer);
    cudaFree(d_recvBuffer);
    
    std::cout << "Rank " << rank << ": Test completed" << std::endl;
    
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  
  return 0;
}