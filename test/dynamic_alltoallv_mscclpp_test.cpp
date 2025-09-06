// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/dynamic_execution_plan.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>  // For GpuBuffer
#include <vector>
#include <iostream>
#include <memory>
#include <numeric>
#include <cuda_runtime.h>
#include <mpi.h>
#include <unistd.h>  // for getcwd
#include <fstream>   // For file existence check
#include <chrono>    // For sleep_for
#include <thread>    // For this_thread

int main(int argc, char* argv[]) {
  
  // Initialize MPI for multi-GPU coordination
  MPI_Init(&argc, &argv);
  
  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  
  // Declare variables outside try block so they're accessible in catch block
  std::shared_ptr<mscclpp::Communicator> comm = nullptr;
  std::shared_ptr<mscclpp::DynamicExecutionPlan> dynamicPlan = nullptr;
  
  // Declare GPU buffers outside try block so we can control their lifetime
  std::unique_ptr<mscclpp::GpuBuffer<char>> sendGpuBuffer = nullptr;
  std::unique_ptr<mscclpp::GpuBuffer<char>> recvGpuBuffer = nullptr;
  
  try {
    // Set CUDA device based on MPI rank
    cudaSetDevice(mpi_rank % 8);  // Assuming up to 8 GPUs per node
    
    int device;
    cudaGetDevice(&device);
    std::cout << "Rank " << mpi_rank << ": Using CUDA device " << device << std::endl;
    
    // Initialize TcpBootstrap for communication setup
    std::cout << "Rank " << mpi_rank << ": Creating TcpBootstrap..." << std::endl;
    
    auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(mpi_rank, mpi_size);
    
    // Create a unique ID (rank 0 creates and broadcasts)
    mscclpp::UniqueId uniqueId;
    if (mpi_rank == 0) {
      uniqueId = bootstrap->createUniqueId();
      std::cout << "Rank " << mpi_rank << ": Created unique ID for bootstrap" << std::endl;
    }
    
    // Broadcast the unique ID to all ranks
    MPI_Bcast(&uniqueId, sizeof(uniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    std::cout << "Rank " << mpi_rank << ": Received unique ID, initializing bootstrap..." << std::endl;
    
    // Initialize bootstrap with the unique ID
    bootstrap->initialize(uniqueId);
    
    // Create Communicator (without ProxyService for simplicity)
    comm = std::make_shared<mscclpp::Communicator>(bootstrap);
    
    if (!comm) {
      throw std::runtime_error("Failed to create Communicator");
    }
    
    std::cout << "Rank " << mpi_rank << ": Communicator created successfully" << std::endl;
    
    // Load dynamic execution plan template with better path handling
    std::string planPath = "test/dynamic_alltoallv_plan.json";
    dynamicPlan = std::make_shared<mscclpp::DynamicExecutionPlan>(planPath, mpi_rank);
    
    std::cout << "Rank " << mpi_rank << ": Dynamic execution plan loaded" << std::endl;
    
    // Setup variable send/recv sizes for multi-GPU all-to-allv
    std::vector<size_t> sendSizes(mpi_size);
    std::vector<size_t> recvSizes(mpi_size);
    
    // Example: each rank sends different amounts to different peers
    for (int i = 0; i < mpi_size; ++i) {
      // Variable message sizes: rank r sends (r+1)*1024 bytes to peer i
      sendSizes[i] = (mpi_rank + 1) * 1024;  // Variable sizes based on sender rank
      recvSizes[i] = (i + 1) * 1024;         // Variable sizes based on sender rank (peer i)
    }
    
    // Calculate total buffer sizes
    size_t totalSendSize = std::accumulate(sendSizes.begin(), sendSizes.end(), 0UL);
    size_t totalRecvSize = std::accumulate(recvSizes.begin(), recvSizes.end(), 0UL);
    
    std::cout << "Rank " << mpi_rank << ": Total send size: " << totalSendSize
              << ", Total recv size: " << totalRecvSize << std::endl;
    
    // Print send/recv patterns for debugging
    std::cout << "Rank " << mpi_rank << " send sizes: ";
    for (int i = 0; i < mpi_size; ++i) {
      std::cout << sendSizes[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Rank " << mpi_rank << " recv sizes: ";
    for (int i = 0; i < mpi_size; ++i) {
      std::cout << recvSizes[i] << " ";
    }
    std::cout << std::endl;
    
    // Create MSCCLPP GpuBuffer objects with proper lifetime management
    sendGpuBuffer = std::make_unique<mscclpp::GpuBuffer<char>>(totalSendSize);
    recvGpuBuffer = std::make_unique<mscclpp::GpuBuffer<char>>(totalRecvSize);
    
    char* d_sendBuffer = sendGpuBuffer->data();
    char* d_recvBuffer = recvGpuBuffer->data();
    
    std::cout << "Rank " << mpi_rank << ": GPU buffers allocated - send: " << totalSendSize 
              << " bytes, recv: " << totalRecvSize << " bytes" << std::endl;
    
    // Initialize send buffer with test data
    if (totalSendSize > 0) {
      std::vector<char> h_sendBuffer(totalSendSize);
      
      // Fill with pattern: rank ID + offset
      for (size_t i = 0; i < totalSendSize; ++i) {
        h_sendBuffer[i] = static_cast<char>((mpi_rank * 16 + i) % 256);
      }
      
      cudaMemcpy(d_sendBuffer, h_sendBuffer.data(), totalSendSize, cudaMemcpyHostToDevice);
    }
    
    if (totalRecvSize > 0) {
      cudaMemset(d_recvBuffer, 0, totalRecvSize);
    }
    
    std::cout << "Rank " << mpi_rank << ": GPU buffers initialized" << std::endl;
    
    // Synchronize all ranks before starting the test
    MPI_Barrier(MPI_COMM_WORLD);
    
    std::cout << "Rank " << mpi_rank << ": Starting dynamic all-to-allv execution with MSCCLPP..." << std::endl;
    
    // Execute dynamic all-to-allv with MSCCLPP execution engine
    bool success = mscclpp::DynamicAllToAllv::execute(
      comm, dynamicPlan, d_sendBuffer, d_recvBuffer, sendSizes, recvSizes);
    
    if (success) {
      std::cout << "Rank " << mpi_rank << ": Dynamic all-to-allv completed successfully!" << std::endl;
      
      // Copy results back to host for verification
      if (totalRecvSize > 0) {
        std::vector<char> h_recvBuffer(totalRecvSize);
        cudaMemcpy(h_recvBuffer.data(), d_recvBuffer, totalRecvSize, cudaMemcpyDeviceToHost);
        
        // Verify received data
        std::cout << "Rank " << mpi_rank << ": First few received bytes: ";
        for (int i = 0; i < std::min(10, static_cast<int>(totalRecvSize)); ++i) {
          std::cout << std::hex << static_cast<int>(static_cast<unsigned char>(h_recvBuffer[i])) << " ";
        }
        std::cout << std::dec << std::endl;
        
        // Verify data per source rank
        size_t recv_offset = 0;
        for (int src_rank = 0; src_rank < mpi_size; ++src_rank) {
          if (recvSizes[src_rank] > 0) {
            std::cout << "Rank " << mpi_rank << ": From rank " << src_rank << " (first 4 bytes): ";
            for (int i = 0; i < std::min(4, static_cast<int>(recvSizes[src_rank])); ++i) {
              std::cout << std::hex << static_cast<int>(static_cast<unsigned char>(h_recvBuffer[recv_offset + i])) << " ";
            }
            std::cout << std::dec << std::endl;
          }
          recv_offset += recvSizes[src_rank];
        }
      }
      
    } else {
      std::cout << "Rank " << mpi_rank << ": Dynamic all-to-allv failed!" << std::endl;
    }
    
    // Synchronize all ranks before cleanup
    MPI_Barrier(MPI_COMM_WORLD);
    
    std::cout << "Rank " << mpi_rank << ": Starting proper cleanup..." << std::endl;
    
    // Explicit cleanup in the correct order to avoid memory issues
    // 1. Clean up dynamic plan first (this may hold references to buffers)
    if (dynamicPlan) {
      dynamicPlan->cleanup();
      dynamicPlan.reset();
    }
    
    // 2. Reset communicator (this may unregister memory)
    if (comm) {
      comm.reset();
    }
    
    // 3. CUDA synchronize before releasing buffers
    cudaDeviceSynchronize();
    
    // 4. Finally release GPU buffers
    sendGpuBuffer.reset();
    recvGpuBuffer.reset();
    
    std::cout << "Rank " << mpi_rank << ": Cleanup completed successfully" << std::endl;
    
  } catch (const std::exception& e) {
    std::cerr << "Rank " << mpi_rank << " Error: " << e.what() << std::endl;
    
    // Cleanup in catch block with extra safety
    try {
      std::cout << "Rank " << mpi_rank << ": Exception cleanup starting..." << std::endl;
      
      if (dynamicPlan) {
        dynamicPlan->cleanup();
        dynamicPlan.reset();
      }
      
      if (comm) {
        comm.reset();
      }
      
      // CUDA synchronize before releasing buffers
      cudaDeviceSynchronize();
      
      sendGpuBuffer.reset();
      recvGpuBuffer.reset();
      
      std::cout << "Rank " << mpi_rank << ": Exception cleanup completed" << std::endl;
      
    } catch (const std::exception& cleanup_error) {
      std::cerr << "Rank " << mpi_rank << " Cleanup error: " << cleanup_error.what() << std::endl;
    }
    
    MPI_Finalize();
    return 1;
  }
  
  std::cout << "Rank " << mpi_rank << ": Test completed successfully" << std::endl;
  MPI_Finalize();
  return 0;
}