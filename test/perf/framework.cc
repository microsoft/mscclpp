// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "framework.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace mscclpp {
namespace test {

// Global state for performance test results
static std::vector<struct PerfTestResult {
  std::string test_name;
  std::string test_category;
  std::map<std::string, std::string> test_params;
  nlohmann::ordered_json metrics;
  int num_processes;
  int process_rank;
  std::string timestamp;
}> g_perf_results;

static std::string getCurrentTimestamp() {
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%S");
  return ss.str();
}

void recordResult(const std::string& test_name, const std::string& test_category, const nlohmann::ordered_json& metrics,
                  const std::map<std::string, std::string>& test_params) {
  PerfTestResult result;
  result.test_name = test_name;
  result.test_category = test_category;
  result.test_params = test_params;
  result.metrics = metrics;
  result.num_processes = utils::getMPISize();
  result.process_rank = utils::getMPIRank();
  result.timestamp = getCurrentTimestamp();

  g_perf_results.push_back(result);
}

void writeResultsToFile(const std::string& filename) {
  std::ofstream file(filename);
  if (!file) {
    throw std::runtime_error("Cannot open output file: " + filename);
  }

  for (const auto& result : g_perf_results) {
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
  if (!utils::isMainRank()) return;

  std::cout << "\n=== Test Results ===" << std::endl;

  for (const auto& result : g_perf_results) {
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

}  // namespace test
}  // namespace mscclpp
