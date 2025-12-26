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
  std::shared_ptr<mscclpp::Executor> executor = nullptr;
  
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
      uniqueId = mscclpp::TcpBootstrap::createUniqueId();
      std::cout << "Rank " << mpi_rank << ": Created unique ID" << std::endl;
    }
    
    // Broadcast unique ID to all ranks
    MPI_Bcast(&uniqueId, sizeof(uniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
    std::cout << "Rank " << mpi_rank << ": Received unique ID" << std::endl;
    
    // Initialize TcpBootstrap with the unique ID
    bootstrap->initialize(uniqueId);
    std::cout << "Rank " << mpi_rank << ": TcpBootstrap initialized" << std::endl;
    
    // Create communicator
    comm = std::make_shared<mscclpp::Communicator>(bootstrap);
    std::cout << "Rank " << mpi_rank << ": Communicator created" << std::endl;
    
    // Create executor
    executor = std::make_shared<mscclpp::Executor>(comm);
    std::cout << "Rank " << mpi_rank << ": Executor created" << std::endl;
    
    // Load dynamic execution plan from enhanced DSL-generated JSON with comprehensive template variables
    std::string dsl_plan_path = "test/dynamic_alltoallv_plan.json";
    
    // Check if DSL file exists
    std::ifstream test_file(dsl_plan_path);
    if (!test_file.good()) {
      std::cout << "Rank " << mpi_rank << ": DSL file not found at: " << dsl_plan_path << std::endl;
      std::cout << "Rank " << mpi_rank << ": Please ensure the comprehensive template file exists with:" << std::endl;
      std::cout << "- operation_template section with variables: ${operation_type}, ${chunk_id}, ${peer_rank}, ${channel_id}, ${tb_count}" << std::endl;
      std::cout << "- Enhanced buffer template variables: ${src_chunk_index}, ${dst_chunk_index}, ${src_chunk_size}, ${dst_chunk_size}" << std::endl;
      std::cout << "- Dynamic operation variables: ${chunk_size}, ${step_id}" << std::endl;
      throw std::runtime_error("Enhanced DSL execution plan file not found");
    }
    test_file.close();
    
    std::cout << "Rank " << mpi_rank << ": Loading enhanced DSL execution plan from: " << dsl_plan_path << std::endl;
    dynamicPlan = std::make_shared<mscclpp::DynamicExecutionPlan>(dsl_plan_path, mpi_rank);
    std::cout << "Rank " << mpi_rank << ": Enhanced dynamic execution plan loaded from DSL with comprehensive template support" << std::endl;
    
    // Create DynamicAllToAllv instance
    auto dynamicAllToAllv = dynamicPlan->createAllToAllv();
    std::cout << "Rank " << mpi_rank << ": DynamicAllToAllv created with enhanced template variable support" << std::endl;
    
    // Define test message sizes for alltoallv (variable sizes per peer)
    std::vector<size_t> sendSizes(mpi_size);
    std::vector<size_t> recvSizes(mpi_size);
    std::vector<size_t> sendOffsets(mpi_size);
    std::vector<size_t> recvOffsets(mpi_size);
    
    // Create variable message sizes: send (rank+1)*1024 bytes to each peer
    size_t sendOffset = 0;
    for (int i = 0; i < mpi_size; ++i) {
      sendSizes[i] = (mpi_rank + 1) * 1024;  // This rank sends (rank+1)*1KB to peer i
      sendOffsets[i] = sendOffset;
      sendOffset += sendSizes[i];
    }
    
    // Receive sizes: receive (peer+1)*1024 bytes from each peer
    size_t recvOffset = 0;
    for (int i = 0; i < mpi_size; ++i) {
      recvSizes[i] = (i + 1) * 1024;  // Receive (peer+1)*1KB from peer i
      recvOffsets[i] = recvOffset;
      recvOffset += recvSizes[i];
    }
    
    size_t totalSendSize = std::accumulate(sendSizes.begin(), sendSizes.end(), 0ULL);
    size_t totalRecvSize = std::accumulate(recvSizes.begin(), recvSizes.end(), 0ULL);
    
    std::cout << "Rank " << mpi_rank << ": Total send size: " << totalSendSize 
              << ", total recv size: " << totalRecvSize << std::endl;
    
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
    
    std::cout << "Rank " << mpi_rank << ": Starting enhanced DSL-based dynamic all-to-allv execution with comprehensive template variable support..." << std::endl;
    
    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    // Execute dynamic all-to-allv with enhanced DSL template supporting all template variables
    dynamicAllToAllv->execute(
      d_sendBuffer, sendSizes, sendOffsets,
      d_recvBuffer, recvSizes, recvOffsets,
      comm, executor, stream);
    
    // Synchronize the stream
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    
    std::cout << "Rank " << mpi_rank << ": Enhanced DSL-based dynamic all-to-allv completed successfully with comprehensive template variable substitution!" << std::endl;
    
    // Copy results back to host for verification
    if (totalRecvSize > 0) {
      std::vector<char> h_recvBuffer(totalRecvSize);
      cudaMemcpy(h_recvBuffer.data(), d_recvBuffer, totalRecvSize, cudaMemcpyDeviceToHost);
      
      std::cout << "Rank " << mpi_rank << ": First 20 received bytes: ";
      for (size_t i = 0; i < std::min(totalRecvSize, size_t(20)); ++i) {
        std::cout << static_cast<int>(h_recvBuffer[i]) << " ";
      }
      std::cout << std::endl;
    }
    
    // Cleanup
    dynamicPlan->cleanup();
    
    std::cout << "Rank " << mpi_rank << ": Enhanced template variable test completed successfully!" << std::endl;
    
  } catch (const std::exception& e) {
    std::cout << "Rank " << mpi_rank << ": Error occurred: " << e.what() << std::endl;
    
    // Cleanup in case of error
    if (dynamicPlan) {
      try {
        dynamicPlan->cleanup();
      } catch (...) {
        // Ignore cleanup errors
      }
    }
    
    MPI_Finalize();
    return 1;
  }
  
  MPI_Finalize();
  return 0;
}