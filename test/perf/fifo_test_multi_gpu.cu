// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <getopt.h>

#include <iostream>
#include <map>
#include <memory>
#include <mscclpp/fifo.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/numa.hpp>
#include <mscclpp/proxy.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/port_channel_device.hpp>
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

__global__ void kernelFifoPush(size_t numTriggers) {
  mscclpp::FifoDeviceHandle& fifo = gFifoDeviceHandle;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  mscclpp::ProxyTrigger trigger;
  for (size_t i = 1; i <= numTriggers; ++i) {
    trigger.fst = i;
    trigger.snd = tid ^ i;
    fifo.push(trigger);
  }
}

__global__ void kernelFifoPushSync(size_t numTriggers) {
  mscclpp::FifoDeviceHandle& fifo = gFifoDeviceHandle;
  mscclpp::ProxyTrigger trigger;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (size_t i = 1; i <= numTriggers; ++i) {
    trigger.fst = i;
    trigger.snd = tid ^ i;
    fifo.sync(fifo.push(trigger));
  }
}

__constant__ mscclpp::PortChannelDeviceHandle gPortChannel;

__global__ void kernelFifoPushAndSignal(mscclpp::PortChannelDeviceHandle portHandle, size_t numTriggers, mscclpp::SemaphoreId semaphoreId) {
  mscclpp::FifoDeviceHandle& fifo = gFifoDeviceHandle;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  mscclpp::ProxyTrigger trigger;
  for (size_t i = 1; i <= numTriggers; ++i) {
    trigger.fst = semaphoreId;
    trigger.snd = tid ^ i;
    fifo.push(trigger);
  }
  __syncthreads();

  if (tid == 0) {
      portHandle.signal();
  }
}

__global__ void kernelWaitAndCheck(mscclpp::PortChannelDeviceHandle portHandle, uint64_t* localFlag, int* result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Check if flag was updated
    if (tid == 0) {
      portHandle.wait();
      if (*localFlag != 1ULL) {
          *result = 1; // Error
      }
    }
}

static void setupCuda(int& cudaDevice, int& numaNode) {
  utils::CUDA_CHECK(cudaGetDevice(&cudaDevice));
  numaNode = mscclpp::getDeviceNumaNode(cudaDevice);
  mscclpp::numaBind(numaNode);
}

// Helper function to consume triggers from FIFO
// static bool consumeTriggers(std::shared_ptr<mscclpp::ProxyService> proxyService, int numTriggers, int parallel,
static bool consumeTriggers(std::unique_ptr<mscclpp::Fifo>& hostFifo, int numTriggers, int parallel,
    std::shared_ptr<mscclpp::Connection> connection,
    mscclpp::RegisteredMemory remoteFlagRegMem) {
  int totalTriggers = numTriggers * parallel;
  std::unordered_map<int, int> triggerCounts;
  printf("Consume trigger from proxy service\n");
  for (int i = 0; i < totalTriggers; ++i) {
    mscclpp::ProxyTrigger trigger;
    uint64_t spin = 0;
    do {
      trigger = hostFifo->poll();
      // trigger = proxyService->fifo().poll();
      if (spin++ > TIMEOUT_SPINS) {
        return false;
      }
    } while (trigger.fst == 0 && trigger.snd == 0);
    printf("i: %d, finish hostFifo->poll()\n", i);

    // Process trigger (see src/proxy.cc)
    trigger.snd ^= ((uint64_t)1 << (uint64_t)63);
    trigger.snd = trigger.snd ^ trigger.fst;
    assert(triggerCounts[trigger.snd] + 1 == trigger.fst);
    triggerCounts[trigger.snd]++;
    hostFifo->pop();
    // Pop from proxy service FIFO
    // proxyService->fifo().pop();

    printf("i: %d, totalTriggers: %d, Consumed trigger: %d, count: %d\n", i, totalTriggers, trigger.snd, triggerCounts[trigger.snd]);
    // Host-side atomicAdd for each trigger
    //connection->atomicAdd(remoteFlagRegMem, 0, 1);
  }
  return true;
}

// Helper function to run a single kernel variant and return performance metrics
std::tuple<double, double, int, int> runSingleKernelVariant(void (*kernel)(size_t),
                                                            std::unique_ptr<mscclpp::Fifo>& hostFifo,
                                                            cudaStream_t stream, int numParallel, int rank,
                                                            mscclpp::PortChannelDeviceHandle portChannelHandle,
                                                            mscclpp::SemaphoreId semaphoreId, uint64_t* localSemaphoreFlag,
                                                            std::shared_ptr<mscclpp::Connection> connection,
                                                            mscclpp::RegisteredMemory remoteFlagRegMem) {
  // Calculate triggers based on FIFO size
  const int numTriggers = std::max(MIN_TRIGGERS, static_cast<int>(hostFifo->size() * TRIGGERS_PER_FIFO_SIZE));
  const int warmupTriggers =
      std::max(MIN_WARMUP_TRIGGERS, static_cast<int>(hostFifo->size() * WARMUP_TRIGGERS_PER_FIFO_SIZE));

  /* 
  // Warmup
  kernel<<<numParallel, 1, 0, stream>>>(warmupTriggers);
  utils::CUDA_CHECK(cudaGetLastError());

  // Process warmup triggers (note: total triggers = warmupTriggers * numParallel)
  if (!consumeTriggers(hostFifo, warmupTriggers, numParallel)) {
    return {0.0, 0.0, 0, 0};  // Return error values
  }
  utils::CUDA_CHECK(cudaStreamSynchronize(stream));
  */

  // Benchmark
  utils::Timer timer;
  timer.start();

  if (rank == 0) {
    // Launch on GPU0
    kernelFifoPushAndSignal<<<numParallel, 1, 0, stream>>>(portChannelHandle, numTriggers, semaphoreId);

    printf("Launching kernel with %d triggers on rank %d\n", numTriggers, rank);
    utils::CUDA_CHECK(cudaGetLastError());

    // Process all triggers
    if (!consumeTriggers(hostFifo, numTriggers, numParallel, connection, remoteFlagRegMem)) {
      return {0.0, 0.0, 0, 0};
    }
    printf("Finished consuming %d triggers on rank %d\n", numTriggers * numParallel, rank);
  } else if (rank == 1) {
    // Allocate result variable for this function
    int* dResult;
    cudaMalloc(&dResult, sizeof(int));
    cudaMemset(dResult, 0, sizeof(int));
    
    // Launch on GPU1
    kernelWaitAndCheck<<<numParallel, 1, 0, stream>>>(portChannelHandle, localSemaphoreFlag, dResult);
    printf("Launching kernel with %d triggers on rank %d\n", numTriggers, rank);
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
                        mscclpp::PortChannelDeviceHandle gPortChannel,
                        mscclpp::SemaphoreId semaphoreId, uint64_t* localSemaphoreFlag,
                        std::shared_ptr<mscclpp::Connection> connection,
                        mscclpp::RegisteredMemory remoteFlagRegMem) {
      // runSingleKernelVariant(kernelFifoPush, proxyService, fifoSize, stream, numParallel, rank, gPortChannel, semaphoreId, localSemaphoreFlag, connection, remoteFlagRegMem);
  auto [pushThroughput, pushDuration, numTriggers, warmupTriggers] =
      runSingleKernelVariant(kernelFifoPush, hostFifo, stream, numParallel, rank, gPortChannel, semaphoreId, localSemaphoreFlag, connection, remoteFlagRegMem);

  // auto [syncThroughput, syncDuration, syncNumTriggers, syncWarmupTriggers] =
  //     runSingleKernelVariant(kernelFifoPushSync, hostFifo, stream, numParallel, rank, gPortChannel, semaphoreId, localSemaphoreFlag, connection, remoteFlagRegMem);

  auto formatThroughput = [](double thru) {
    return double(int(thru * 10)) / 10.0;  // Round to 1 decimal place
  };

  std::string prefix = "p" + std::to_string(numParallel) + "_";
  combinedMetrics[prefix + "push_throughput"] = formatThroughput(pushThroughput);
  // combinedMetrics[prefix + "push_sync_throughput"] = formatThroughput(syncThroughput);
  combinedMetrics[prefix + "push_duration_us"] = pushDuration;
  // combinedMetrics[prefix + "push_sync_duration_us"] = syncDuration;
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
  const int nElem = 1024;  // Example size, adjust as needed
  // std::shared_ptr<int> buff = mscclpp::allocSharedCuda<int>(nElem);
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

  // // Setup port channels
  // std::vector<mscclpp::PortChannel> portChannels;
  // for (int r = 0; r < worldSize; r++) {
  //   if (r == rank) {
  //     continue;
  //   }
  //   mscclpp::SemaphoreId cid = proxyService->buildAndAddSemaphore(*communicator, connections[r < rank ? r : r-1]);
  //   portChannels.emplace_back(proxyService->portChannel(cid, proxyService->addMemory(remoteMemFutures[r].get()),
  //                                                       proxyService->addMemory(sendBufRegMem)));
  // }
  // printf("Finished port channel setup for rank %d\n", rank);

  // std::vector<mscclpp::PortChannelDeviceHandle> portChannelHandles;
  // for (auto& ch : portChannels) portChannelHandles.push_back(ch.deviceHandle());

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
  // mscclpp::FifoDeviceHandle proxyFifoHandle = proxyService->fifo().deviceHandle();
  utils::CUDA_CHECK(cudaMemcpyToSymbol(gFifoDeviceHandle, &hostHandle, sizeof(mscclpp::FifoDeviceHandle)));
  // utils::CUDA_CHECK(cudaMemcpyToSymbol(gFifoDeviceHandle, &proxyFifoHandle, sizeof(mscclpp::FifoDeviceHandle)));

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
  mscclpp::RegisteredMemory remoteFlagRegMem = localFlagRegmem; // Use local as placeholder
  mscclpp::PortChannelDeviceHandle portChannelHandle;
  
  if (worldSize >= 2 && !connections.empty()) {
    int peerRank = (rank == 0) ? 1 : 0;
    int connIndex = peerRank < rank ? peerRank : peerRank - 1;
    if (connIndex < connections.size()) {
      connection = connections[connIndex];
      semaphoreId = proxyService->buildAndAddSemaphore(*communicator, connection);
      printf("Semaphore ID for rank %d: %d\n", rank, semaphoreId);
      auto portChannel = proxyService->portChannel(semaphoreId, proxyService->addMemory(remoteMemFutures[peerRank].get()), proxyService->addMemory(localFlagRegmem));
      portChannelHandle = portChannel.deviceHandle();
      cudaMemcpyToSymbol(gPortChannel, &portChannelHandle, sizeof(portChannelHandle), 0, cudaMemcpyHostToDevice);
    }
  }
  printf("Finished gPortChannel setup for rank %d\n", rank);

  for (int numParallel : config.parallelismLevels) {
    // Add synchronization before each test iteration
    MPI_Barrier(MPI_COMM_WORLD);

    runFifoTestVariant(hostFifo, stream, numParallel, combinedMetrics, rank, portChannelHandle, semaphoreId, localSemaphoreFlag, connection, remoteFlagRegMem);

    // Add synchronization after each test iteration
    MPI_Barrier(MPI_COMM_WORLD);
  }

  printf("Finished runFifoTestVariant for rank %d\n", rank);

  std::map<std::string, std::string> testParams;
  testParams["fifo_size"] = std::to_string(static_cast<int>(hostFifo->size()));
  // testParams["fifo_size"] = std::to_string(static_cast<int>(config.fifoSize));

  // Add parallelism levels to test parameters
  std::stringstream parallelismStream;
  for (size_t i = 0; i < config.parallelismLevels.size(); ++i) {
    if (i > 0) parallelismStream << ",";
    parallelismStream << config.parallelismLevels[i];
  }
  testParams["parallelism_levels"] = parallelismStream.str();

  utils::recordResult(testName, "fifo", combinedMetrics, testParams);
  printf("Finished recording results for rank %d\n", rank);

  // Cleanup
  utils::CUDA_CHECK(cudaStreamDestroy(stream));
  cudaFree(localSemaphoreFlag);
  cudaFree(dResult);
  
  proxyService->stopProxy();
  printf("Finished stopping proxy for rank %d\n", rank);
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
