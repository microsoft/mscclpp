// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <getopt.h>

#include <iostream>
#include <map>
#include <memory>
#include <mscclpp/fifo.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/numa.hpp>
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

static void setupCuda(int& cudaDevice, int& numaNode) {
  CUDA_CHECK(cudaGetDevice(&cudaDevice));
  numaNode = mscclpp::getDeviceNumaNode(cudaDevice);
  mscclpp::numaBind(numaNode);
}

// Helper function to consume triggers from FIFO
static bool consumeTriggers(std::unique_ptr<mscclpp::Fifo>& hostFifo, int numTriggers, int parallel) {
  int totalTriggers = numTriggers * parallel;
  std::unordered_map<int, int> triggerCounts;
  for (int i = 0; i < totalTriggers; ++i) {
    mscclpp::ProxyTrigger trigger;
    uint64_t spin = 0;
    do {
      trigger = hostFifo->poll();
      if (spin++ > TIMEOUT_SPINS) {
        return false;
      }
    } while (trigger.fst == 0 || trigger.snd == 0);

    // Process trigger (see src/proxy.cc)
    trigger.snd ^= ((uint64_t)1 << (uint64_t)63);
    trigger.snd = trigger.snd ^ trigger.fst;
    assert(triggerCounts[trigger.snd] + 1 == trigger.fst);
    triggerCounts[trigger.snd]++;
    hostFifo->pop();
  }
  return true;
}

// Helper function to run a single kernel variant and return performance metrics
std::tuple<double, double, int, int> runSingleKernelVariant(void (*kernel)(size_t),
                                                            std::unique_ptr<mscclpp::Fifo>& hostFifo,
                                                            cudaStream_t stream, int numParallel) {
  // Calculate triggers based on FIFO size
  const int numTriggers = std::max(MIN_TRIGGERS, static_cast<int>(hostFifo->size() * TRIGGERS_PER_FIFO_SIZE));
  const int warmupTriggers =
      std::max(MIN_WARMUP_TRIGGERS, static_cast<int>(hostFifo->size() * WARMUP_TRIGGERS_PER_FIFO_SIZE));

  // Warmup
  kernel<<<numParallel, 1, 0, stream>>>(warmupTriggers);
  CUDA_CHECK(cudaGetLastError());

  // Process warmup triggers (note: total triggers = warmupTriggers * numParallel)
  if (!consumeTriggers(hostFifo, warmupTriggers, numParallel)) {
    return {0.0, 0.0, 0, 0};  // Return error values
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // Benchmark
  utils::Timer timer;
  timer.start();

  kernel<<<numParallel, 1, 0, stream>>>(numTriggers);
  CUDA_CHECK(cudaGetLastError());

  // Process all triggers
  if (!consumeTriggers(hostFifo, numTriggers, numParallel)) {
    return {0.0, 0.0, 0, 0};
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  timer.stop();

  const int totalTriggers = numTriggers * numParallel;
  double throughput = totalTriggers / timer.elapsedSeconds();
  double duration_us = timer.elapsedMicroseconds();

  CUDA_CHECK(cudaDeviceSynchronize());

  return {throughput, duration_us, totalTriggers, warmupTriggers * numParallel};
}

void runFifoTestVariant(std::unique_ptr<mscclpp::Fifo>& hostFifo, cudaStream_t stream, int numParallel,
                        nlohmann::ordered_json& combinedMetrics) {
  auto [pushThroughput, pushDuration, numTriggers, warmupTriggers] =
      runSingleKernelVariant(kernelFifoPush, hostFifo, stream, numParallel);

  auto [syncThroughput, syncDuration, syncNumTriggers, syncWarmupTriggers] =
      runSingleKernelVariant(kernelFifoPushSync, hostFifo, stream, numParallel);

  auto formatThroughput = [](double thru) {
    return double(int(thru * 10)) / 10.0;  // Round to 1 decimal place
  };

  std::string prefix = "p" + std::to_string(numParallel) + "_";
  combinedMetrics[prefix + "push_throughput"] = formatThroughput(pushThroughput);
  combinedMetrics[prefix + "push_sync_throughput"] = formatThroughput(syncThroughput);
  combinedMetrics[prefix + "push_duration_us"] = pushDuration;
  combinedMetrics[prefix + "push_sync_duration_us"] = syncDuration;
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

void runFifoTest(const FifoTestConfig& config, [[maybe_unused]] int rank, [[maybe_unused]] int worldSize,
                 [[maybe_unused]] int localRank) {
  if (config.fifoSize <= 0) {
    throw std::invalid_argument("FIFO size must be positive");
  }
  if (config.parallelismLevels.empty()) {
    throw std::invalid_argument("At least one parallelism level must be specified");
  }

  int cudaDevice, numaNode;
  setupCuda(cudaDevice, numaNode);

  auto hostFifo = std::make_unique<mscclpp::Fifo>(config.fifoSize);

  mscclpp::FifoDeviceHandle hostHandle = hostFifo->deviceHandle();
  CUDA_CHECK(cudaMemcpyToSymbol(gFifoDeviceHandle, &hostHandle, sizeof(mscclpp::FifoDeviceHandle)));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

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

  for (int numParallel : config.parallelismLevels) {
    runFifoTestVariant(hostFifo, stream, numParallel, combinedMetrics);
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

  CUDA_CHECK(cudaStreamDestroy(stream));
}

void runAllFifoTests([[maybe_unused]] int rank, [[maybe_unused]] int worldSize, [[maybe_unused]] int localRank) {
  // clang-format off
  std::vector<FifoTestConfig> configs = {
      {1, {1}},
      {128, {1, 8, 64, 128}},
      {512, {1, 8, 64, 256, 512}},
  };
  // clang-format on

  for (const auto& config : configs) {
    runFifoTest(config, rank, worldSize, localRank);
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

  std::vector<std::tuple<std::string, std::string, std::function<void(int, int, int)>>> tests = {
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
