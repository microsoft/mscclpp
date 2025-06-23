// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "framework.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace mscclpp {
namespace test {

// Global state for results
static std::vector<TestResult> g_results;
static int g_mpi_rank = 0;
static int g_mpi_size = 1;
static bool g_mpi_initialized = false;

namespace utils {

// Internal MPI helper functions (not exposed in header)
void initializeMPI(int argc, char* argv[]) {
  if (g_mpi_initialized) return;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_mpi_size);
  g_mpi_initialized = true;
}

static void finalizeMPI() {
  if (!g_mpi_initialized) return;

  MPI_Finalize();
  g_mpi_initialized = false;
}

static int getMPIRank() { return g_mpi_rank; }

static int getMPISize() { return g_mpi_size; }

static bool isMainProcess() { return g_mpi_rank == 0; }

// Public utility functions for test output
bool isMainRank() { return g_mpi_rank == 0; }

void cleanupMPI() { finalizeMPI(); }

std::string getCurrentTimestamp() {
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%S");
  return ss.str();
}

void recordResult(const std::string& test_name, const std::string& test_category, const nlohmann::ordered_json& metrics,
                  const std::map<std::string, std::string>& test_params) {
  TestResult result;
  result.test_name = test_name;
  result.test_category = test_category;
  result.test_params = test_params;
  result.metrics = metrics;
  result.num_processes = g_mpi_size;
  result.process_rank = g_mpi_rank;
  result.timestamp = getCurrentTimestamp();

  g_results.push_back(result);
}

void writeResultsToFile(const std::string& filename) {
  std::ofstream file(filename);
  if (!file) {
    throw std::runtime_error("Cannot open output file: " + filename);
  }

  for (const auto& result : g_results) {
    nlohmann::ordered_json j;
    j["test_name"] = result.test_name;
    j["test_category"] = result.test_category;
    j["test_config"] = result.test_params;
    j["metrics"] = result.metrics;
    j["num_processes"] = result.num_processes;
    j["process_rank"] = result.process_rank;
    j["timestamp"] = result.timestamp;

    file << j.dump() << std::endl;
  }
}

void printResults(bool verbose) {
  if (!isMainProcess()) return;

  std::cout << "\n=== Test Results ===" << std::endl;

  for (const auto& result : g_results) {
    std::cout << "\nTest: " << result.test_name << " (" << result.test_category << ")" << std::endl;

    if (verbose && !result.test_params.empty()) {
      std::cout << "  Parameters:" << std::endl;
      for (const auto& param : result.test_params) {
        std::cout << "    " << param.first << ": " << param.second << std::endl;
      }
    }

    std::cout << "  Metrics:" << std::endl;
    for (auto it = result.metrics.begin(); it != result.metrics.end(); ++it) {
      std::cout << "    " << it.key() << ": " << it.value() << std::endl;
    }
  }
  std::cout << std::endl;
}

// Timer implementation
Timer::Timer() : is_running_(false) {}

void Timer::start() {
  start_time_ = std::chrono::high_resolution_clock::now();
  is_running_ = true;
}

void Timer::stop() {
  end_time_ = std::chrono::high_resolution_clock::now();
  is_running_ = false;
}

double Timer::elapsedMicroseconds() const {
  if (is_running_) {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(now - start_time_).count();
  }
  return std::chrono::duration_cast<std::chrono::microseconds>(end_time_ - start_time_).count();
}

double Timer::elapsedMilliseconds() const { return elapsedMicroseconds() / 1000.0; }

double Timer::elapsedSeconds() const { return elapsedMicroseconds() / 1000000.0; }

void cudaCheck(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    std::string msg =
        std::string("CUDA error at ") + file + ":" + std::to_string(line) + " - " + cudaGetErrorString(err);
    throw std::runtime_error(msg);
  }
}

int runMultipleTests(
    int argc, char* argv[],
    const std::vector<std::tuple<std::string, std::string, std::function<void(int, int, int)>>>& tests) {
  int totalResult = 0;

  // Initialize MPI once for all tests
  initializeMPI(argc, argv);

  try {
    // Get MPI information
    int rank = getMPIRank();
    int size = getMPISize();
    int local_rank = rank;  // For simplicity, assume local_rank = rank

    for (const auto& test : tests) {
      const std::string& testName = std::get<0>(test);
      const std::string& testDescription = std::get<1>(test);
      const std::function<void(int, int, int)>& testFunction = std::get<2>(test);

      if (rank == 0) {
        std::cout << "Running test: " << testName << std::endl;
        if (!testDescription.empty()) {
          std::cout << "  " << testDescription << std::endl;
        }
      }

      // Don't clear results - accumulate them for all tests in the same file
      // g_results.clear();  // Commented out to accumulate results

      try {
        // Run the individual test function with MPI information
        testFunction(rank, size, local_rank);

        // Synchronize before moving to next test
        MPI_Barrier(MPI_COMM_WORLD);

      } catch (const std::exception& e) {
        if (rank == 0) {
          std::cerr << "Error in test " << testName << ": " << e.what() << std::endl;
        }
        totalResult = 1;
      }
    }

    // Don't cleanup MPI here - let the caller handle it
    // finalizeMPI();

  } catch (const std::exception& e) {
    if (g_mpi_rank == 0) {
      std::cerr << "Error: " << e.what() << std::endl;
    }
    finalizeMPI();
    return 1;
  }

  return totalResult;
}

}  // namespace utils
}  // namespace test
}  // namespace mscclpp
