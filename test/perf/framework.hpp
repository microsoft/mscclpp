// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_TEST_PERF_FRAMEWORK_HPP_
#define MSCCLPP_TEST_PERF_FRAMEWORK_HPP_

#include <mpi.h>

#include <chrono>
#include <fstream>
#include <functional>
#include <map>
#include <mscclpp/gpu.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <tuple>
#include <vector>

namespace mscclpp {
namespace test {

// Test result structure
struct TestResult {
  std::string test_name;
  std::string test_category;
  std::map<std::string, std::string> test_params;
  nlohmann::ordered_json metrics;
  int num_processes;
  int process_rank;
  std::string timestamp;
};

// Simple utility functions for testing
namespace utils {

// Test execution utilities
int runMultipleTests(
    int argc, char* argv[],
    const std::vector<std::tuple<std::string, std::string, std::function<void(int, int, int)>>>& tests);

// MPI management
void initializeMPI(int argc, char* argv[]);
void cleanupMPI();
bool isMainRank();

// Result recording
void recordResult(const std::string& test_name, const std::string& test_category, const nlohmann::ordered_json& metrics,
                  const std::map<std::string, std::string>& test_params = {});

// Output utilities
void writeResultsToFile(const std::string& filename);
void printResults(bool verbose = false);
void cleanupMPI();

// Timing utilities
class Timer {
 public:
  Timer();
  void start();
  void stop();
  double elapsedMicroseconds() const;
  double elapsedMilliseconds() const;
  double elapsedSeconds() const;

 private:
  std::chrono::high_resolution_clock::time_point start_time_;
  std::chrono::high_resolution_clock::time_point end_time_;
  bool is_running_;
};

// CUDA utilities
void cudaCheck(cudaError_t err, const char* file, int line);
#define CUDA_CHECK(call) cudaCheck(call, __FILE__, __LINE__)

}  // namespace utils

}  // namespace test
}  // namespace mscclpp

#endif  // MSCCLPP_TEST_PERF_FRAMEWORK_HPP_
