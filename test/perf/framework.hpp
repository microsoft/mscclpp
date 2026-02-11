// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_TEST_PERF_FRAMEWORK_HPP_
#define MSCCLPP_TEST_PERF_FRAMEWORK_HPP_

// This file is kept for backwards compatibility with perf tests
// The actual framework is now in test/framework.hpp

#include <nlohmann/json.hpp>

#include "../framework.hpp"

namespace mscclpp {
namespace test {
namespace utils {

// Additional performance test utilities not in the base framework

// Result recording for performance tests
void recordResult(const std::string& test_name, const std::string& test_category, const nlohmann::ordered_json& metrics,
                  const std::map<std::string, std::string>& test_params = {});

// Output utilities for performance tests
void writeResultsToFile(const std::string& filename);
void printResults(bool verbose = false);

}  // namespace utils
}  // namespace test
}  // namespace mscclpp

#endif  // MSCCLPP_TEST_PERF_FRAMEWORK_HPP_
