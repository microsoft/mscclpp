// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <getopt.h>

#include <iostream>
#include <map>
#include <memory>
#include <mscclpp/fifo.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/numa.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/port_channel_device.hpp>
#include <mscclpp/proxy.hpp>
#include <sstream>
#include <stdexcept>

#include "framework.hpp"

using namespace mscclpp::test;

__constant__ mscclpp::PortChannelDeviceHandle gPortChannel;

// New kernels for bidirectional data transfer
__global__ void kernelPutData(int* sendBuffer, mscclpp::PortChannelDeviceHandle portHandle, int numElements, int rank) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid == 0) {
    portHandle.put(0, 0, numElements * sizeof(int));
  }

  // Only thread 0 signals completion
  if (tid == 0) {
    portHandle.signal();
  }
}

__global__ void kernelGetData(int* recvBuffer, mscclpp::PortChannelDeviceHandle portHandle, int numElements, int rank,
                              int expectedValue) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int totalThreads = blockDim.x * gridDim.x;

  // Wait for signal from sender - only global thread 0 should wait
  if (tid == 0) {
    portHandle.wait();
  }

  __shared__ int errorCount;
  if (threadIdx.x == 0) {
    errorCount = 0;
  }
  __syncthreads();

  int localErrors = 0;

  // Each thread validates a portion of the received data
  for (int i = tid; i < numElements; i += totalThreads) {
    if (recvBuffer[i] != expectedValue) {
      localErrors++;
    }
  }

  // Accumulate errors from all threads
  if (localErrors > 0) {
    atomicAdd(&errorCount, localErrors);
  }

  // Report validation results from thread 0
  __syncthreads();
  if (tid == 0) {
    if (errorCount == 0) {
      printf("GPU%d: Data validation PASSED - all %d elements correct (expected value: %d)\n", rank, numElements,
             expectedValue);
    } else {
      printf("GPU%d: Data validation FAILED - %d errors found out of %d elements (expected value: %d)\n", rank,
             errorCount, numElements, expectedValue);
    }
    assert(errorCount == 0);
  }
}

static void setupCuda(int& cudaDevice, int& numaNode) {
  utils::CUDA_CHECK(cudaGetDevice(&cudaDevice));
  numaNode = mscclpp::getDeviceNumaNode(cudaDevice);
  mscclpp::numaBind(numaNode);
}

std::tuple<double, double, int> runDataTransferKernelVariant(cudaStream_t stream, int numParallel, int rank,
                                                             mscclpp::PortChannelDeviceHandle portChannelHandle,
                                                             int* sendBuffer, int* recvBuffer, int numElements) {
  int threadsPerBlock = std::min(numParallel, 512);
  int threadBlocks = (numParallel + threadsPerBlock - 1) / threadsPerBlock;
  threadBlocks = std::max(1, threadBlocks);  // Ensure at least 1 block

  // Benchmark
  utils::Timer timer;
  timer.start();

  // Launch both put and get operations simultaneously on each GPU for bidirectional transfer
  if (rank == 0) {
    // GPU0: Send data (value 1) and receive data (expecting value 2 from GPU1)
    kernelPutData<<<threadBlocks, threadsPerBlock, 0, stream>>>(sendBuffer, portChannelHandle, numElements, rank);
    utils::CUDA_CHECK(cudaGetLastError());

    kernelGetData<<<threadBlocks, threadsPerBlock, 0, stream>>>(recvBuffer, portChannelHandle, numElements, rank, 2);
    utils::CUDA_CHECK(cudaGetLastError());
  } else if (rank == 1) {
    // GPU1: Send data (value 2) and receive data (expecting value 1 from GPU0)
    kernelPutData<<<threadBlocks, threadsPerBlock, 0, stream>>>(sendBuffer, portChannelHandle, numElements, rank);
    utils::CUDA_CHECK(cudaGetLastError());

    kernelGetData<<<threadBlocks, threadsPerBlock, 0, stream>>>(recvBuffer, portChannelHandle, numElements, rank, 1);
    utils::CUDA_CHECK(cudaGetLastError());
  }

  utils::CUDA_CHECK(cudaStreamSynchronize(stream));

  timer.stop();

  const int totalElements = numElements;
  double throughput = totalElements / timer.elapsedSeconds();
  double duration_us = timer.elapsedMicroseconds();

  utils::CUDA_CHECK(cudaDeviceSynchronize());

  return {throughput, duration_us, totalElements};
}

void runDataTransferTestVariant(cudaStream_t stream, int numParallel, nlohmann::ordered_json& combinedMetrics, int rank,
                                mscclpp::PortChannelDeviceHandle portChannelHandle, int* sendBuffer, int* recvBuffer,
                                int numElements) {
  // Run simultaneous bidirectional data transfer
  printf("=== Running simultaneous bidirectional GPU0 ↔ GPU1 transfer ===\n");
  auto [throughput, duration, totalElements] =
      runDataTransferKernelVariant(stream, numParallel, rank, portChannelHandle, sendBuffer, recvBuffer, numElements);

  auto formatThroughput = [](double thru) {
    return double(int(thru * 10)) / 10.0;  // Round to 1 decimal place
  };

  std::string prefix = "p" + std::to_string(numParallel) + "_";
  combinedMetrics[prefix + "data_throughput_elements_per_sec"] = formatThroughput(throughput);
  combinedMetrics[prefix + "data_duration_us"] = duration;
  combinedMetrics[prefix + "total_elements"] = totalElements;
  combinedMetrics[prefix + "bandwidth_GB_per_sec"] =
      formatThroughput((totalElements * sizeof(int)) / (duration / 1e6) / 1e9);
}

struct FifoTestConfig {
  int fifoSize;
  std::vector<int> parallelismLevels;

  // Constructor with default parallelism levels
  FifoTestConfig(int size, const std::vector<int>& parallel = {1, 2, 4, 8, 16})
      : fifoSize(size), parallelismLevels(parallel) {}
};

void runDataTransferTest(const FifoTestConfig& config, const mscclpp::test::TestContext& context) {
  int rank = context.rank;
  int worldSize = context.size;
  auto communicator = context.communicator;
  auto bootstrap = context.bootstrap;

  if (config.fifoSize <= 0) {
    throw std::invalid_argument("FIFO size must be positive");
  }
  if (config.parallelismLevels.empty()) {
    throw std::invalid_argument("At least one parallelism level must be specified");
  }

  // Set the device for this process
  cudaSetDevice(rank);

  // Define buffer size and allocate memory
  const int nElem = 1024;
  const int halfElements = nElem / 2;
  std::shared_ptr<int> buff = mscclpp::GpuBuffer<int>(nElem).memory();

  // Split buffer into send and receive halves
  int* sendBuffer = buff.get();
  int* recvBuffer = buff.get() + halfElements;

  // Initialize send buffer with test data
  int initValue = rank + 1;  // GPU0 uses 1, GPU1 uses 2
  std::vector<int> hostBuffer(halfElements, initValue);
  cudaMemcpy(sendBuffer, hostBuffer.data(), halfElements * sizeof(int), cudaMemcpyHostToDevice);

  // Initialize receive buffer to zero
  cudaMemset(recvBuffer, 0, halfElements * sizeof(int));

  // Setup transport
  mscclpp::TransportFlags transport = mscclpp::Transport::CudaIpc;
  std::vector<std::shared_ptr<mscclpp::Connection>> connections;
  if (worldSize > 1) {
    for (int i = 0; i < worldSize; i++) {
      if (i == rank) {
        continue;
      }
      // Use different IB transports for different ranks
      std::vector<mscclpp::Transport> ibTransports{mscclpp::Transport::IB0, mscclpp::Transport::IB1};
      mscclpp::Transport selectedTransport = ibTransports[rank % ibTransports.size()];
      transport |= selectedTransport;
      connections.push_back(communicator->connect(selectedTransport, i).get());
    }
  }

  // Wait for all connections to be established
  bootstrap->barrier();

  // Create and start proxy service with specified FIFO size
  auto proxyService = std::make_shared<mscclpp::ProxyService>(config.fifoSize);
  proxyService->startProxy();

  // Register send buffer memory (first half)
  mscclpp::RegisteredMemory sendBufRegMem =
      communicator->registerMemory(sendBuffer, halfElements * sizeof(int), transport);

  // Register receive buffer memory (second half)
  mscclpp::RegisteredMemory recvBufRegMem =
      communicator->registerMemory(recvBuffer, halfElements * sizeof(int), transport);

  // Exchange memory with other ranks
  std::vector<std::shared_future<mscclpp::RegisteredMemory>> remoteSendMemFutures(worldSize);
  std::vector<std::shared_future<mscclpp::RegisteredMemory>> remoteRecvMemFutures(worldSize);

  for (int r = 0; r < worldSize; r++) {
    if (r == rank) {
      continue;
    }
    // Send our buffer info to other ranks
    communicator->sendMemory(sendBufRegMem, r, 0);  // tag 0 for send buffer
    communicator->sendMemory(recvBufRegMem, r, 1);  // tag 1 for recv buffer

    // Receive other ranks' buffer info
    remoteSendMemFutures[r] = communicator->recvMemory(r, 0);
    remoteRecvMemFutures[r] = communicator->recvMemory(r, 1);
  }

  // Allocate and setup local semaphore flag
  uint64_t* localSemaphoreFlag;
  cudaMalloc(&localSemaphoreFlag, sizeof(uint64_t));
  cudaMemset(localSemaphoreFlag, 0, sizeof(uint64_t));

  // Register semaphore flag
  auto localFlagRegmem = communicator->registerMemory(localSemaphoreFlag, sizeof(uint64_t), transport);

  int cudaDevice, numaNode;
  setupCuda(cudaDevice, numaNode);

  cudaStream_t stream;
  utils::CUDA_CHECK(cudaStreamCreate(&stream));

  // Create test name with parallelism range
  std::string testName = "FifoDataTransferTest_Size" + std::to_string(config.fifoSize) + "_Parallel";

  // Add parallelism range to test name (e.g., "P1-16" or "P1-4-16-64")
  if (!config.parallelismLevels.empty()) {
    testName += std::to_string(config.parallelismLevels.front());
    if (config.parallelismLevels.size() > 1) {
      testName += "-" + std::to_string(config.parallelismLevels.back());

      // If parallelism levels have non-standard steps, include more detail
      if (config.parallelismLevels.size() > 2 &&
          (config.parallelismLevels[1] != 2 * config.parallelismLevels[0] || config.parallelismLevels.size() > 3)) {
        testName = "FifoTest_Size" + std::to_string(config.fifoSize) + "_ParallelCustom";
      }
    }
  }

  // Print test configuration
  if (utils::isMainRank()) {
    std::stringstream ss;
    ss << "Running FIFO test with size=" << config.fifoSize << ", parallelism_levels=[";
    for (size_t i = 0; i < config.parallelismLevels.size(); ++i) {
      if (i > 0) ss << ",";
      ss << config.parallelismLevels[i];
    }
    ss << "]";
    std::cout << ss.str() << std::endl;
  }

  nlohmann::ordered_json combinedMetrics;

  // Prepare variables for the test variant
  mscclpp::SemaphoreId semaphoreId = 0;
  std::shared_ptr<mscclpp::Connection> connection = nullptr;
  mscclpp::RegisteredMemory remoteFlagRegMem = localFlagRegmem;
  mscclpp::PortChannelDeviceHandle portChannelHandle;

  if (worldSize >= 2 && !connections.empty()) {
    int peerRank = (rank == 0) ? 1 : 0;
    int connIndex = peerRank < rank ? peerRank : peerRank - 1;
    if (connIndex < connections.size()) {
      connection = connections[connIndex];
      semaphoreId = proxyService->buildAndAddSemaphore(*communicator, connection);
      // Setup port channel to copy from our send buffer to remote's receive buffer
      auto portChannel =
          proxyService->portChannel(semaphoreId, proxyService->addMemory(remoteRecvMemFutures[peerRank].get()),
                                    proxyService->addMemory(sendBufRegMem));
      portChannelHandle = portChannel.deviceHandle();
      cudaMemcpyToSymbol(gPortChannel, &portChannelHandle, sizeof(portChannelHandle), 0, cudaMemcpyHostToDevice);
    }
  }

  for (int numParallel : config.parallelismLevels) {
    // Add synchronization before each test iteration
    MPI_Barrier(MPI_COMM_WORLD);

    runDataTransferTestVariant(stream, numParallel, combinedMetrics, rank, portChannelHandle, sendBuffer, recvBuffer,
                               halfElements);

    // Add synchronization after each test iteration
    MPI_Barrier(MPI_COMM_WORLD);
  }

  std::map<std::string, std::string> testParams;
  testParams["fifo_size"] = std::to_string(static_cast<int>(config.fifoSize));
  testParams["elements_per_gpu"] = std::to_string(halfElements);

  // Add parallelism levels to test parameters
  std::stringstream parallelismStream;
  for (size_t i = 0; i < config.parallelismLevels.size(); ++i) {
    if (i > 0) parallelismStream << ",";
    parallelismStream << config.parallelismLevels[i];
  }
  testParams["parallelism_levels"] = parallelismStream.str();

  utils::recordResult(testName, "fifo_data_transfer", combinedMetrics, testParams);

  // Cleanup
  utils::CUDA_CHECK(cudaStreamDestroy(stream));
  cudaFree(localSemaphoreFlag);

  proxyService->stopProxy();
}

void runAllDataTransferTests(const mscclpp::test::TestContext& context) {
  // clang-format off
  std::vector<FifoTestConfig> configs = {
      {1, {1}},
      {128, {1, 8, 64, 128}},
      {512, {1, 8, 64, 256, 512}},
  };
  // clang-format on

  for (const auto& config : configs) {
    runDataTransferTest(config, context);
  }
}

static void printUsage(char* argv0) {
  std::stringstream ss;
  ss << "Usage: " << argv0 << " [OPTIONS]\n"
     << "\n"
     << "Options:\n"
     << "  -o, --output-format FORMAT   Output format: human or json (default: human)\n"
     << "  -f, --output-file FILE       JSON output file path (default: report.jsonl)\n"
     << "  -v, --verbose                Increase verbosity\n"
     << "  -h, --help                   Show this help message\n";
  std::cout << ss.str();
}

int main(int argc, char* argv[]) {
  std::string outputFormat = "human";
  std::string outputFile = "report.jsonl";
  bool verbose = false;

  static struct option longOptions[] = {{"output-format", required_argument, 0, 'o'},
                                        {"output-file", required_argument, 0, 'f'},
                                        {"verbose", no_argument, 0, 'v'},
                                        {"help", no_argument, 0, 'h'},
                                        {0, 0, 0, 0}};

  int c;
  while ((c = getopt_long(argc, argv, "o:f:vh", longOptions, nullptr)) != -1) {
    switch (c) {
      case 'o':
        outputFormat = optarg;
        break;
      case 'f':
        outputFile = optarg;
        break;
      case 'v':
        verbose = true;
        break;
      case 'h':
        printUsage(argv[0]);
        return 0;
      default:
        printUsage(argv[0]);
        return 1;
    }
  }

  std::vector<std::tuple<std::string, std::string, std::function<void(const mscclpp::test::TestContext&)>>> tests = {
      {"AllDataTransferTests", "Data transfer tests with multiple configurations", runAllDataTransferTests}};

  int result = utils::runMultipleTests(argc, argv, tests);

  if (utils::isMainRank()) {
    if (outputFormat == "json") {
      utils::writeResultsToFile(outputFile);
    } else {
      utils::printResults(verbose);
    }
  }

  utils::cleanupMPI();

  return result;
}
