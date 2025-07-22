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

// Constants for timeout and trigger calculation
constexpr uint64_t TIMEOUT_SPINS = 1000000;
constexpr int MIN_TRIGGERS = 1000;
constexpr int MIN_WARMUP_TRIGGERS = 100;
constexpr int TRIGGERS_PER_FIFO_SIZE = 10;
constexpr int WARMUP_TRIGGERS_PER_FIFO_SIZE = 2;

__constant__ mscclpp::FifoDeviceHandle gFifoDeviceHandle;
__constant__ mscclpp::PortChannelDeviceHandle gPortChannel;

__global__ void kernelFifoPushAndSignal(mscclpp::PortChannelDeviceHandle portHandle, size_t numTriggers,
                                        mscclpp::SemaphoreId semaphoreId) {
  int tid = threadIdx.x;

  if (tid == 0) {
    portHandle.signal();
  }
}

__global__ void kernelWaitAndCheck(mscclpp::PortChannelDeviceHandle portHandle, uint64_t* localFlag, int* result) {
  int tid = threadIdx.x;

  // Check if flag was updated
  if (tid == 0) {
    portHandle.wait();
    if (*localFlag != 1ULL) {
      *result = 1;  // Place holder for error check
    }
  }
}

static void setupCuda(int& cudaDevice, int& numaNode) {
  utils::CUDA_CHECK(cudaGetDevice(&cudaDevice));
  numaNode = mscclpp::getDeviceNumaNode(cudaDevice);
  mscclpp::numaBind(numaNode);
}

// Helper function to run a single kernel variant and return performance metrics
std::tuple<double, double, int, int> runSingleKernelVariant(
    std::unique_ptr<mscclpp::Fifo>& hostFifo, cudaStream_t stream, int numParallel, int rank,
    mscclpp::PortChannelDeviceHandle portChannelHandle, mscclpp::SemaphoreId semaphoreId, uint64_t* localSemaphoreFlag,
    std::shared_ptr<mscclpp::Connection> connection, mscclpp::RegisteredMemory remoteFlagRegMem) {
  // Calculate triggers based on FIFO size
  const int numTriggers = std::max(MIN_TRIGGERS, static_cast<int>(hostFifo->size() * TRIGGERS_PER_FIFO_SIZE));
  const int warmupTriggers =
      std::max(MIN_WARMUP_TRIGGERS, static_cast<int>(hostFifo->size() * WARMUP_TRIGGERS_PER_FIFO_SIZE));

  int threadBlocks = std::min(8, numParallel);

  // Benchmark
  utils::Timer timer;
  timer.start();

  if (rank == 0) {
    // Launch on GPU0
    kernelFifoPushAndSignal<<<threadBlocks, numParallel / threadBlocks, 0, stream>>>(portChannelHandle, numTriggers,
                                                                                     semaphoreId);

    utils::CUDA_CHECK(cudaGetLastError());
  } else if (rank == 1) {
    // Allocate result variable for this function
    int* dResult;
    cudaMalloc(&dResult, sizeof(int));
    cudaMemset(dResult, 0, sizeof(int));

    // Launch on GPU1
    kernelWaitAndCheck<<<threadBlocks, numParallel / threadBlocks, 0, stream>>>(portChannelHandle, localSemaphoreFlag,
                                                                                dResult);
    utils::CUDA_CHECK(cudaGetLastError());

    // Clean up
    cudaFree(dResult);
  }

  utils::CUDA_CHECK(cudaStreamSynchronize(stream));

  timer.stop();

  const int totalTriggers = numTriggers * numParallel;
  double throughput = totalTriggers / timer.elapsedSeconds();
  double duration_us = timer.elapsedMicroseconds();

  utils::CUDA_CHECK(cudaDeviceSynchronize());

  return {throughput, duration_us, totalTriggers, warmupTriggers * numParallel};
}

// void runFifoTestVariant(std::shared_ptr<mscclpp::ProxyService> proxyService, int fifoSize,
void runFifoTestVariant(std::unique_ptr<mscclpp::Fifo>& hostFifo, cudaStream_t stream, int numParallel,
                        nlohmann::ordered_json& combinedMetrics, int rank,
                        mscclpp::PortChannelDeviceHandle gPortChannel, mscclpp::SemaphoreId semaphoreId,
                        uint64_t* localSemaphoreFlag, std::shared_ptr<mscclpp::Connection> connection,
                        mscclpp::RegisteredMemory remoteFlagRegMem) {
  auto [pushThroughput, pushDuration, numTriggers, warmupTriggers] = runSingleKernelVariant(
      hostFifo, stream, numParallel, rank, gPortChannel, semaphoreId, localSemaphoreFlag, connection, remoteFlagRegMem);

  auto formatThroughput = [](double thru) {
    return double(int(thru * 10)) / 10.0;  // Round to 1 decimal place
  };

  std::string prefix = "p" + std::to_string(numParallel) + "_";
  combinedMetrics[prefix + "push_throughput"] = formatThroughput(pushThroughput);
  combinedMetrics[prefix + "push_duration_us"] = pushDuration;
  combinedMetrics[prefix + "num_triggers"] = numTriggers;
  combinedMetrics[prefix + "warmup_triggers"] = warmupTriggers;
}

struct FifoTestConfig {
  int fifoSize;
  std::vector<int> parallelismLevels;

  // Constructor with default parallelism levels
  FifoTestConfig(int size, const std::vector<int>& parallel = {1, 2, 4, 8, 16})
      : fifoSize(size), parallelismLevels(parallel) {}
};

void runFifoTest(const FifoTestConfig& config, const mscclpp::test::TestContext& context) {
  int rank = context.rank;
  int worldSize = context.size;
  int localRank = context.local_rank;
  auto communicator = context.communicator;
  auto connections = context.connections;
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
  std::shared_ptr<int> buff = mscclpp::GpuBuffer<int>(nElem).memory();

  // Setup transport
  mscclpp::TransportFlags transport = mscclpp::Transport::CudaIpc;
  if (worldSize > 1) {
    // Add IB transport for multi-node scenarios
    transport |= mscclpp::Transport::IB0;
  }

  // Create and start proxy service with specified FIFO size
  auto proxyService = std::make_shared<mscclpp::ProxyService>(config.fifoSize);
  proxyService->startProxy();

  // Register send buffer memory
  mscclpp::RegisteredMemory sendBufRegMem = communicator->registerMemory(buff.get(), nElem * sizeof(int), transport);

  // Register receive buffer memory (same as send for simplicity)
  mscclpp::RegisteredMemory recvBufRegMem = communicator->registerMemory(buff.get(), nElem * sizeof(int), transport);

  // Exchange memory with other ranks
  std::vector<std::shared_future<mscclpp::RegisteredMemory>> remoteMemFutures(worldSize);
  for (int r = 0; r < worldSize; r++) {
    if (r == rank) {
      continue;
    }
    communicator->sendMemory(recvBufRegMem, r, 0);
    remoteMemFutures[r] = communicator->recvMemory(r, 0);
  }

  // Allocate and setup local semaphore flag
  uint64_t* localSemaphoreFlag;
  cudaMalloc(&localSemaphoreFlag, sizeof(uint64_t));
  cudaMemset(localSemaphoreFlag, 0, sizeof(uint64_t));

  // Register semaphore flag
  auto localFlagRegmem = communicator->registerMemory(localSemaphoreFlag, sizeof(uint64_t), transport);

  int* dResult;
  cudaMalloc(&dResult, sizeof(int));
  cudaMemset(dResult, 0, sizeof(int));

  int cudaDevice, numaNode;
  setupCuda(cudaDevice, numaNode);

  auto hostFifo = std::make_unique<mscclpp::Fifo>(config.fifoSize);

  mscclpp::FifoDeviceHandle hostHandle = hostFifo->deviceHandle();
  utils::CUDA_CHECK(cudaMemcpyToSymbol(gFifoDeviceHandle, &hostHandle, sizeof(mscclpp::FifoDeviceHandle)));

  cudaStream_t stream;
  utils::CUDA_CHECK(cudaStreamCreate(&stream));

  // Create test name with parallelism range
  std::string testName = "FifoTest_Size" + std::to_string(config.fifoSize) + "_Parallel";

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
      auto portChannel =
          proxyService->portChannel(semaphoreId, proxyService->addMemory(remoteMemFutures[peerRank].get()),
                                    proxyService->addMemory(localFlagRegmem));
      portChannelHandle = portChannel.deviceHandle();
      cudaMemcpyToSymbol(gPortChannel, &portChannelHandle, sizeof(portChannelHandle), 0, cudaMemcpyHostToDevice);
    }
  }

  for (int numParallel : config.parallelismLevels) {
    // Add synchronization before each test iteration
    MPI_Barrier(MPI_COMM_WORLD);

    runFifoTestVariant(hostFifo, stream, numParallel, combinedMetrics, rank, portChannelHandle, semaphoreId,
                       localSemaphoreFlag, connection, remoteFlagRegMem);

    // Add synchronization after each test iteration
    MPI_Barrier(MPI_COMM_WORLD);
  }

  std::map<std::string, std::string> testParams;
  testParams["fifo_size"] = std::to_string(static_cast<int>(hostFifo->size()));

  // Add parallelism levels to test parameters
  std::stringstream parallelismStream;
  for (size_t i = 0; i < config.parallelismLevels.size(); ++i) {
    if (i > 0) parallelismStream << ",";
    parallelismStream << config.parallelismLevels[i];
  }
  testParams["parallelism_levels"] = parallelismStream.str();

  utils::recordResult(testName, "fifo", combinedMetrics, testParams);

  // Cleanup
  utils::CUDA_CHECK(cudaStreamDestroy(stream));
  cudaFree(localSemaphoreFlag);
  cudaFree(dResult);

  proxyService->stopProxy();
}

void runAllFifoTests(const mscclpp::test::TestContext& context) {
  int rank = context.rank;
  int worldSize = context.size;
  int localRank = context.local_rank;
  // clang-format off
  std::vector<FifoTestConfig> configs = {
      {1, {1}},
      {128, {1, 8, 64, 128}},
      {512, {1, 8, 64, 256, 512}},
  };
  // clang-format on

  for (const auto& config : configs) {
    runFifoTest(config, context);
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
      {"AllFifoTests", "FIFO performance tests with multiple configurations", runAllFifoTests}};

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
