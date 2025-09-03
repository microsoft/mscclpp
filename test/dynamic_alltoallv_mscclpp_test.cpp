// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/dynamic_execution_plan.hpp>
#include <mscclpp/core.hpp>
#include <vector>
#include <iostream>
#include <memory>
#include <numeric>
#include <cuda_runtime.h>
#include <mpi.h>
#include <unistd.h>  // for getcwd
#include <fstream>   // For file existence check

int main(int argc, char* argv[]) {
  
  // Initialize MPI for multi-GPU coordination
  MPI_Init(&argc, &argv);
  
  int mpi_rank, mpi_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  
  try {
    std::cout << "MPI Rank " << mpi_rank << " of " << mpi_size << " starting MSCCLPP execution test..." << std::endl;
    
    // Determine GPU device based on rank
    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    
    if (num_gpus == 0) {
      std::cerr << "No CUDA devices found!" << std::endl;
      MPI_Finalize();
      return 1;
    }
    
    // Map MPI rank to GPU device (round-robin if more ranks than GPUs)
    int device_id = mpi_rank % num_gpus;
    cudaSetDevice(device_id);
    
    std::cout << "Rank " << mpi_rank << ": Using GPU device " << device_id << " (total GPUs: " << num_gpus << ")" << std::endl;
    
    // Initialize CUDA context
    cudaFree(0);  // Force CUDA context initialization
    
    // Create TcpBootstrap for MSCCLPP with multiple ranks
    auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(mpi_rank, mpi_size);
    
    // Create unique ID and broadcast from rank 0
    mscclpp::UniqueId uniqueId;
    if (mpi_rank == 0) {
      uniqueId = bootstrap->createUniqueId();
    }
    MPI_Bcast(&uniqueId, sizeof(uniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    
    bootstrap->initialize(uniqueId);
    
    // Create communicator
    auto comm = std::make_shared<mscclpp::Communicator>(bootstrap);
    
    std::cout << "Rank " << mpi_rank << ": MSCCLPP communicator initialized" << std::endl;
    
    // Load dynamic execution plan template with better path handling
    std::string planPath = "test/dynamic_alltoallv_plan.json";
    auto dynamicPlan = std::make_shared<mscclpp::DynamicExecutionPlan>(planPath, mpi_rank);
    
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
    
    // Allocate GPU buffers
    void* d_sendBuffer = nullptr;
    void* d_recvBuffer = nullptr;
    
    if (totalSendSize > 0) {
      cudaError_t err = cudaMalloc(&d_sendBuffer, totalSendSize);
      if (err != cudaSuccess) {
        std::cerr << "Rank " << mpi_rank << ": Failed to allocate send buffer: " << cudaGetErrorString(err) << std::endl;
        MPI_Finalize();
        return 1;
      }
    }
    
    if (totalRecvSize > 0) {
      cudaError_t err = cudaMalloc(&d_recvBuffer, totalRecvSize);
      if (err != cudaSuccess) {
        std::cerr << "Rank " << mpi_rank << ": Failed to allocate recv buffer: " << cudaGetErrorString(err) << std::endl;
        if (d_sendBuffer) cudaFree(d_sendBuffer);
        MPI_Finalize();
        return 1;
      }
    }
    
    // Initialize send buffer with rank-specific test pattern
    if (totalSendSize > 0) {
      std::vector<char> h_sendBuffer(totalSendSize);
      
      // Initialize different patterns for different destination ranks
      size_t offset = 0;
      for (int dest_rank = 0; dest_rank < mpi_size; ++dest_rank) {
        for (size_t i = 0; i < sendSizes[dest_rank]; ++i) {
          // Pattern: (source_rank * 0x10) + (dest_rank * 0x01) + (i % 256)
          h_sendBuffer[offset + i] = static_cast<char>(
            (mpi_rank * 0x10) + (dest_rank * 0x01) + ((offset + i) % 0x10));
        }
        offset += sendSizes[dest_rank];
      }
      
      cudaMemcpy(d_sendBuffer, h_sendBuffer.data(), totalSendSize, cudaMemcpyHostToDevice);
    }
    
    if (totalRecvSize > 0) {
      cudaMemset(d_recvBuffer, 0, totalRecvSize);
    }
    
    std::cout << "Rank " << mpi_rank << ": GPU buffers allocated and initialized" << std::endl;
    
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
    
    // Cleanup GPU memory
    if (d_sendBuffer) cudaFree(d_sendBuffer);
    if (d_recvBuffer) cudaFree(d_recvBuffer);
    
    // Reset CUDA device
    cudaDeviceReset();
    
    std::cout << "Rank " << mpi_rank << ": Test completed" << std::endl;
    
  } catch (const std::exception& e) {
    std::cerr << "Rank " << mpi_rank << " Error: " << e.what() << std::endl;
    MPI_Finalize();
    return 1;
  }
  
  MPI_Finalize();
  return 0;
}