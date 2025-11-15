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

// Constants for trigger calculation
constexpr int MIN_TRIGGERS = 1000;
constexpr int TRIGGERS_PER_FIFO_SIZE = 10;

__constant__ mscclpp::PortChannelDeviceHandle gPortChannel;

struct MultiGpuTestConfig {
  int fifoSize;
  int numGpus;    // Total number of GPUs
  int numGroups;  // Number of groups
  std::vector<int> parallelismLevels;

  MultiGpuTestConfig(int size, int gpus, int groups, const std::vector<int>& parallel = {64, 128, 256, 512})
      : fifoSize(size), numGpus(gpus), numGroups(groups), parallelismLevels(parallel) {
    if (numGpus % numGroups != 0) {
      throw std::invalid_argument("Number of GPUs must be divisible by number of groups");
    }
  }

  int getGroupSize() const { return numGpus / numGroups; }
  int getGroupIndex(int rank) const { return rank / getGroupSize(); }
  int getLocalRankInGroup(int rank) const { return rank % getGroupSize(); }

  // Get all ranks that participate in cross-group signaling (local rank 0 from each group)
  std::vector<int> getCrossGroupSignalingRanks() const {
    std::vector<int> signalingRanks;
    for (int group = 0; group < numGroups; group++) {
      int localRank0 = group * getGroupSize();  // First rank in each group
      signalingRanks.push_back(localRank0);
    }
    return signalingRanks;
  }

  // Check if this rank should participate in cross-group signaling
  bool shouldParticipateInSignaling(int rank) const {
    return getLocalRankInGroup(rank) == 0;  // Only local rank 0 in each group participates
  }
};

// Enhanced kernels for multi-GPU signaling
__global__ void kernelMultiGpuSignalSend(mscclpp::PortChannelDeviceHandle* portHandles, int numPeers, int numParallel) {
  int tid = threadIdx.x;

  // Each thread sends signals to all peers
  if (tid < numParallel) {
    for (int peer = 0; peer < numPeers; peer++) {
      portHandles[peer].signal();
    }
  }
}

__global__ void kernelMultiGpuSignalWait(mscclpp::PortChannelDeviceHandle* portHandles, int numPeers, int numParallel) {
  int tid = threadIdx.x;

  // Each thread waits for signals from all peers
  if (tid < numParallel) {
    for (int peer = 0; peer < numPeers; peer++) {
      portHandles[peer].wait();
    }
  }
}

static void setupCuda(int& cudaDevice, int& numaNode) {
  utils::CUDA_CHECK(cudaGetDevice(&cudaDevice));
  numaNode = mscclpp::getDeviceNumaNode(cudaDevice);
  mscclpp::numaBind(numaNode);
}

// Enhanced performance measurement function
std::tuple<double, double, int> runMultiGpuKernelVariant(
    cudaStream_t stream, int numParallel, int rank,
    const std::vector<mscclpp::PortChannelDeviceHandle>& sendPortHandles,
    const std::vector<mscclpp::PortChannelDeviceHandle>& recvPortHandles, const MultiGpuTestConfig& config) {
  // Calculate triggers based on FIFO size, but respect the limit
  const int maxParallel = std::min(numParallel, config.fifoSize);
  const int numTriggers = std::max(MIN_TRIGGERS, static_cast<int>(config.fifoSize * TRIGGERS_PER_FIFO_SIZE));

  // Configure kernel launch parameters
  int threadsPerBlock = std::min(maxParallel, 256);
  int threadBlocks = (maxParallel + threadsPerBlock - 1) / threadsPerBlock;

  // Copy port handles to device memory using MSCCLPP gpuCallocShared
  std::shared_ptr<mscclpp::PortChannelDeviceHandle> d_sendHandles = nullptr;
  std::shared_ptr<mscclpp::PortChannelDeviceHandle> d_recvHandles = nullptr;

  if (!sendPortHandles.empty()) {
    d_sendHandles = mscclpp::detail::gpuCallocShared<mscclpp::PortChannelDeviceHandle>(sendPortHandles.size());
    mscclpp::gpuMemcpy(d_sendHandles.get(), sendPortHandles.data(), sendPortHandles.size(), cudaMemcpyHostToDevice);
  }

  if (!recvPortHandles.empty()) {
    d_recvHandles = mscclpp::detail::gpuCallocShared<mscclpp::PortChannelDeviceHandle>(recvPortHandles.size());
    mscclpp::gpuMemcpy(d_recvHandles.get(), recvPortHandles.data(), recvPortHandles.size(), cudaMemcpyHostToDevice);
  }

  // Benchmark
  utils::Timer timer;
  timer.start();

  bool shouldSignal = config.shouldParticipateInSignaling(rank);

  if (shouldSignal) {
    // Launch signaling kernels
    if (!sendPortHandles.empty()) {
      kernelMultiGpuSignalSend<<<threadBlocks, threadsPerBlock, 0, stream>>>(d_sendHandles.get(),
                                                                             sendPortHandles.size(), maxParallel);
      utils::CUDA_CHECK(cudaGetLastError());
    }

    // Launch waiting kernels
    if (!recvPortHandles.empty()) {
      kernelMultiGpuSignalWait<<<threadBlocks, threadsPerBlock, 0, stream>>>(d_recvHandles.get(),
                                                                             recvPortHandles.size(), maxParallel);
      utils::CUDA_CHECK(cudaGetLastError());
    }
  }

  utils::CUDA_CHECK(cudaStreamSynchronize(stream));
  timer.stop();

  const int totalSignals = numTriggers * maxParallel * (sendPortHandles.size() + recvPortHandles.size());
  double throughput = totalSignals / timer.elapsedSeconds();
  double duration_us = timer.elapsedMicroseconds();

  utils::CUDA_CHECK(cudaDeviceSynchronize());

  return {throughput, duration_us, totalSignals};
}

// Main multi-GPU test function
void runMultiGpuTest(const MultiGpuTestConfig& config, const mscclpp::test::TestContext& context) {
  int rank = context.rank;
  int worldSize = context.size;
  auto communicator = context.communicator;
  auto bootstrap = context.bootstrap;

  if (worldSize != config.numGpus) {
    throw std::invalid_argument("World size must match number of GPUs in config");
  }

  // Set the device for this process
  cudaSetDevice(rank);

  // Setup transport
  mscclpp::TransportFlags transport = mscclpp::Transport::CudaIpc;
  std::vector<mscclpp::Transport> ibTransports{
      mscclpp::Transport::IB0, mscclpp::Transport::IB1, mscclpp::Transport::IB2, mscclpp::Transport::IB3,
      mscclpp::Transport::IB4, mscclpp::Transport::IB5, mscclpp::Transport::IB6, mscclpp::Transport::IB7};
  std::vector<std::shared_ptr<mscclpp::Connection>> connections;

  // Only create connections for GPUs that need to communicate
  if (config.shouldParticipateInSignaling(rank)) {
    mscclpp::Transport selectedTransport = ibTransports[rank % ibTransports.size()];
    transport |= selectedTransport;

    // Get all ranks that participate in cross-group signaling
    auto signalingRanks = config.getCrossGroupSignalingRanks();

    for (int peerRank : signalingRanks) {
      if (peerRank != rank) {
        connections.push_back(communicator->connect(selectedTransport, peerRank).get());
      }
    }
  }

  // Wait for all connections to be established
  bootstrap->barrier();

  // Create and start proxy service
  auto proxyService = std::make_shared<mscclpp::ProxyService>(config.fifoSize);
  proxyService->startProxy();

  // Setup semaphore flags
  uint64_t* localSemaphoreFlag;
  cudaMalloc(&localSemaphoreFlag, sizeof(uint64_t));
  cudaMemset(localSemaphoreFlag, 0, sizeof(uint64_t));
  auto localFlagRegmem = communicator->registerMemory(localSemaphoreFlag, sizeof(uint64_t), transport);

  int cudaDevice, numaNode;
  setupCuda(cudaDevice, numaNode);

  cudaStream_t stream;
  utils::CUDA_CHECK(cudaStreamCreate(&stream));

  // Setup port channels for communication
  std::vector<mscclpp::PortChannelDeviceHandle> sendPortHandles;
  std::vector<mscclpp::PortChannelDeviceHandle> recvPortHandles;

  if (config.shouldParticipateInSignaling(rank)) {
    // Get all ranks that participate in cross-group signaling
    auto signalingRanks = config.getCrossGroupSignalingRanks();
    int connIndex = 0;

    for (int peerRank : signalingRanks) {
      if (peerRank != rank && connIndex < connections.size()) {
        auto connection = connections[connIndex++];
        auto semaphoreId = proxyService->buildAndAddSemaphore(*communicator, connection);

        // Create port channels for bidirectional communication
        auto sendPortChannel = proxyService->portChannel(semaphoreId, proxyService->addMemory(localFlagRegmem),
                                                         proxyService->addMemory(localFlagRegmem));
        auto recvPortChannel = proxyService->portChannel(semaphoreId, proxyService->addMemory(localFlagRegmem),
                                                         proxyService->addMemory(localFlagRegmem));

        sendPortHandles.push_back(sendPortChannel.deviceHandle());
        recvPortHandles.push_back(recvPortChannel.deviceHandle());
      }
    }
  }

  // Create test name
  std::string testName = "MultiGpuTest_GPUs" + std::to_string(config.numGpus) + "_Groups" +
                         std::to_string(config.numGroups) + "_FifoSize" + std::to_string(config.fifoSize);

  // Print test configuration
  if (utils::isMainRank()) {
    std::cout << "Running Multi-GPU test: " << config.numGpus << " GPUs, " << config.numGroups
              << " groups, FIFO size=" << config.fifoSize << std::endl;

    // Print which ranks participate in cross-group signaling
    auto signalingRanks = config.getCrossGroupSignalingRanks();
    std::cout << "Cross-group signaling participants: ";
    for (size_t i = 0; i < signalingRanks.size(); ++i) {
      if (i > 0) std::cout << ", ";
      std::cout << "rank " << signalingRanks[i] << " (group " << config.getGroupIndex(signalingRanks[i]) << ")";
    }
    std::cout << std::endl;
  }

  nlohmann::ordered_json combinedMetrics;

  // Run tests for different parallelism levels
  for (int numParallel : config.parallelismLevels) {
    // Ensure parallelism doesn't exceed FIFO size
    int effectiveParallel = std::min(numParallel, config.fifoSize);

    // Add synchronization before each test iteration
    MPI_Barrier(MPI_COMM_WORLD);

    if (config.shouldParticipateInSignaling(rank)) {
      auto [throughput, duration, totalSignals] =
          runMultiGpuKernelVariant(stream, effectiveParallel, rank, sendPortHandles, recvPortHandles, config);

      std::string prefix = "p" + std::to_string(effectiveParallel) + "_";
      combinedMetrics[prefix + "throughput_signals_per_sec"] = double(int(throughput * 10)) / 10.0;
      combinedMetrics[prefix + "duration_us"] = duration;
      combinedMetrics[prefix + "total_signals"] = totalSignals;
      combinedMetrics[prefix + "participating_gpus"] = config.numGpus;
    }

    // Add synchronization after each test iteration
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Record results
  std::map<std::string, std::string> testParams;
  testParams["num_gpus"] = std::to_string(config.numGpus);
  testParams["num_groups"] = std::to_string(config.numGroups);
  testParams["group_size"] = std::to_string(config.getGroupSize());
  testParams["fifo_size"] = std::to_string(config.fifoSize);
  testParams["participating_in_signaling"] = config.shouldParticipateInSignaling(rank) ? "true" : "false";

  // Add information about cross-group signaling ranks
  if (config.shouldParticipateInSignaling(rank)) {
    auto signalingRanks = config.getCrossGroupSignalingRanks();
    std::stringstream ss;
    for (size_t i = 0; i < signalingRanks.size(); ++i) {
      if (i > 0) ss << ",";
      ss << signalingRanks[i];
    }
    testParams["cross_group_signaling_ranks"] = ss.str();
  }

  utils::recordResult(testName, "multi_gpu_signaling", combinedMetrics, testParams);

  // Cleanup
  utils::CUDA_CHECK(cudaStreamDestroy(stream));
  cudaFree(localSemaphoreFlag);
  proxyService->stopProxy();
}

void runAllMultiGpuTests(const mscclpp::test::TestContext& context) {
  std::vector<MultiGpuTestConfig> configs = {
      // 8 GPUs, 2 groups (4 GPUs per group) - local rank 0 participates in signaling
      MultiGpuTestConfig(512, 8, 2, {1, 8, 64, 128, 256, 512}),

      // 8 GPUs, 4 groups (2 GPUs per group) - local rank 0 participates in signaling
      MultiGpuTestConfig(512, 8, 4, {1, 8, 64, 128, 256, 512}),

      // 8 GPUs, 8 groups (1 GPU per group) - local rank 0 participates in signaling
      MultiGpuTestConfig(512, 8, 8, {1, 8, 64, 128, 256, 512}),
  };

  for (const auto& config : configs) {
    // Only run if we have the right number of GPUs
    if (context.size == config.numGpus) {
      runMultiGpuTest(config, context);
    }
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
      {"AllMultiGpuTests", "Multi-GPU signaling tests with configurable groups", runAllMultiGpuTests}};

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
