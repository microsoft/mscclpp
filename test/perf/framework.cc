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
static std::vector<TestResult> gPerfResults;

namespace {
std::string getCurrentTimestamp() {
  auto now = std::chrono::system_clock::now();
  auto time_t = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  ss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%S");
  return ss.str();
}
}  // namespace

namespace utils {

void recordResult(const std::string& testName, const std::string& testCategory, const nlohmann::ordered_json& metrics,
                  const std::map<std::string, std::string>& testParams) {
  TestResult result;
  result.testName = testName;
  result.testCategory = testCategory;
  result.testParams = testParams;
  result.metrics = metrics;
  result.numProcesses = getMPISize();
  result.processRank = getMPIRank();
  result.timestamp = getCurrentTimestamp();

  gPerfResults.push_back(result);
}

void writeResultsToFile(const std::string& filename) {
  std::ofstream file(filename);
  if (!file) {
    throw std::runtime_error("Cannot open output file: " + filename);
  }

  for (const auto& result : gPerfResults) {
    nlohmann::ordered_json j;
    j["test_name"] = result.testName;
    j["test_category"] = result.testCategory;
    j["test_config"] = result.testParams;
    j["metrics"] = result.metrics;
    j["num_processes"] = result.numProcesses;
    j["process_rank"] = result.processRank;
    j["timestamp"] = result.timestamp;

    file << j.dump() << std::endl;
  }
}

void printResults(bool verbose) {
  if (!isMainRank()) return;

  std::cout << "\n=== Test Results ===" << std::endl;

  for (const auto& result : gPerfResults) {
    std::cout << "\nTest: " << result.testName << " (" << result.testCategory << ")" << std::endl;

    if (verbose && !result.testParams.empty()) {
      std::cout << "  Parameters:" << std::endl;
      for (const auto& param : result.testParams) {
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

}  // namespace utils
}  // namespace test
}  // namespace mscclpp
