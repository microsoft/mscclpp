// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "framework.hpp"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace mscclpp {
namespace test {

// Global state
static int gMpiRank = 0;
static int gMpiSize = 1;
static bool gMpiInitialized = false;
static bool gCurrentTestPassed = true;
static std::string gCurrentTestFailureMessage;

namespace utils {

// Internal MPI helper functions (not exposed in header)
void initializeMPI(int argc, char* argv[]) {
  if (gMpiInitialized) return;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &gMpiRank);
  MPI_Comm_size(MPI_COMM_WORLD, &gMpiSize);
  gMpiInitialized = true;
}

static void finalizeMPI() {
  if (!gMpiInitialized) return;

  MPI_Finalize();
  gMpiInitialized = false;
}

static bool isMainProcess() { return gMpiRank == 0; }

// Public utility functions for test output
bool isMainRank() { return gMpiRank == 0; }

int getMPIRank() { return gMpiRank; }

int getMPISize() { return gMpiSize; }

void cleanupMPI() { finalizeMPI(); }

void reportFailure(const char* file, int line, const std::string& message) {
  gCurrentTestPassed = false;
  std::ostringstream oss;
  oss << file << ":" << line << ": " << message;
  if (!gCurrentTestFailureMessage.empty()) {
    gCurrentTestFailureMessage += "\n";
  }
  gCurrentTestFailureMessage += oss.str();
  std::cerr << oss.str() << std::endl;
}

void reportSuccess() {
  gCurrentTestPassed = true;
  gCurrentTestFailureMessage.clear();
}

// Timer implementation
Timer::Timer() : isRunning_(false) {}

void Timer::start() {
  startTime_ = std::chrono::high_resolution_clock::now();
  isRunning_ = true;
}

void Timer::stop() {
  endTime_ = std::chrono::high_resolution_clock::now();
  isRunning_ = false;
}

double Timer::elapsedMicroseconds() const {
  if (isRunning_) {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(now - startTime_).count();
  }
  return std::chrono::duration_cast<std::chrono::microseconds>(endTime_ - startTime_).count();
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
    if (gMpiRank == 0) {
      std::cerr << "Error: " << e.what() << std::endl;
    }
    finalizeMPI();
    return 1;
  }

  return totalResult;
}

}  // namespace utils

// UnitTest implementation
UnitTest* UnitTest::GetInstance() {
  static UnitTest instance;
  return &instance;
}

// TestRegistry implementation
TestRegistry& TestRegistry::instance() {
  static TestRegistry registry;
  return registry;
}

void TestRegistry::registerTest(const std::string& test_suite, const std::string& test_name, TestFactory factory,
                                bool isPerfTest) {
  TestInfoInternal info;
  info.suiteName = test_suite;
  info.testName = test_name;
  info.factory = factory;
  info.isPerfTest = isPerfTest;
  tests_.push_back(info);
}

void TestRegistry::addGlobalTestEnvironment(Environment* env) { environments_.push_back(env); }

void TestRegistry::initGoogleTest(int* argc, char** argv) {
  // Parse command-line arguments if needed
  // For now, this is a no-op placeholder for compatibility
}

int TestRegistry::runAllTests(int argc, char* argv[]) {
  // Initialize MPI if not already initialized
  if (!gMpiInitialized) {
    utils::initializeMPI(argc, argv);
  }

  // Parse command line arguments
  std::string filter = "";
  bool excludePerfTests = false;
  
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.find("--gtest_filter=") == 0) {
      filter = arg.substr(15);  // Length of "--gtest_filter="
    } else if (arg == "--gtest_filter" && i + 1 < argc) {
      filter = argv[i + 1];
      ++i;
    } else if (arg == "--exclude-perf-tests") {
      excludePerfTests = true;
    }
  }

  // Set up global test environments
  for (auto* env : environments_) {
    try {
      env->SetUp();
    } catch (const std::exception& e) {
      if (gMpiRank == 0) {
        std::cerr << "Failed to set up test environment: " << e.what() << std::endl;
      }
      return 1;
    }
  }

  int passed = 0;
  int failed = 0;
  int skipped = 0;

  // Count tests to run
  int total_to_run = 0;
  for (const auto& test_info : tests_) {
    std::string full_name = test_info.suiteName + "." + test_info.testName;
    
    // Skip performance tests if requested
    if (excludePerfTests && test_info.isPerfTest) {
      skipped++;
      continue;
    }
    
    if (!filter.empty() && full_name.find(filter) == std::string::npos) {
      skipped++;
      continue;
    }
    total_to_run++;
  }

  if (gMpiRank == 0) {
    std::cout << "[==========] Running " << total_to_run << " tests";
    if (skipped > 0) {
      std::cout << " (" << skipped << " skipped)";
    }
    std::cout << ".\n";
  }

  for (const auto& test_info : tests_) {
    std::string full_name = test_info.suiteName + "." + test_info.testName;

    // Skip performance tests if requested
    if (excludePerfTests && test_info.isPerfTest) {
      continue;
    }
    
    // Apply filter
    if (!filter.empty() && full_name.find(filter) == std::string::npos) {
      continue;
    }

    gCurrentTestPassed = true;
    gCurrentTestFailureMessage.clear();

    if (gMpiRank == 0) {
      std::cout << "[ RUN      ] " << full_name << std::endl;
    }

    // Set current test info for UnitTest::GetInstance()->current_test_info()
    TestInfo current_info(test_info.suiteName, test_info.testName);
    UnitTest::GetInstance()->set_current_test_info(&current_info);

    TestCase* test_case = nullptr;
    try {
      test_case = test_info.factory();
      test_case->SetUp();
      test_case->TestBody();
      test_case->TearDown();
    } catch (const std::exception& e) {
      gCurrentTestPassed = false;
      if (gCurrentTestFailureMessage.empty()) {
        gCurrentTestFailureMessage = e.what();
      }
    } catch (...) {
      gCurrentTestPassed = false;
      if (gCurrentTestFailureMessage.empty()) {
        gCurrentTestFailureMessage = "Unknown exception";
      }
    }

    delete test_case;

    // Clear current test info
    UnitTest::GetInstance()->set_current_test_info(nullptr);

    // Synchronize test status across all MPI processes
    int local_passed = gCurrentTestPassed ? 1 : 0;
    int global_passed = 1;
    if (gMpiInitialized) {
      MPI_Allreduce(&local_passed, &global_passed, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    } else {
      global_passed = local_passed;
    }

    if (gMpiRank == 0) {
      if (global_passed) {
        std::cout << "[       OK ] " << full_name << std::endl;
        passed++;
      } else {
        std::cout << "[  FAILED  ] " << full_name << std::endl;
        failed++;
      }
    }
  }

  if (gMpiRank == 0) {
    std::cout << "[==========] " << total_to_run << " tests ran.\n";
    if (passed > 0) {
      std::cout << "[  PASSED  ] " << passed << " tests.\n";
    }
    if (failed > 0) {
      std::cout << "[  FAILED  ] " << failed << " tests.\n";
    }
  }

  // Tear down global test environments (in reverse order)
  for (auto it = environments_.rbegin(); it != environments_.rend(); ++it) {
    try {
      (*it)->TearDown();
    } catch (const std::exception& e) {
      if (gMpiRank == 0) {
        std::cerr << "Failed to tear down test environment: " << e.what() << std::endl;
      }
    }
  }

  return failed > 0 ? 1 : 0;
}

}  // namespace test
}  // namespace mscclpp
