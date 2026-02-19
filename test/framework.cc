// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "framework.hpp"

#include <algorithm>
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
static std::string gCurrentTestName;

std::string currentTestName() { return gCurrentTestName; }

namespace utils {

void initializeMPI(int argc, char* argv[]) {
  if (gMpiInitialized) return;

  int initialized = 0;
  MPI_Initialized(&initialized);
  if (!initialized) {
    MPI_Init(&argc, &argv);
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &gMpiRank);
  MPI_Comm_size(MPI_COMM_WORLD, &gMpiSize);
  gMpiInitialized = true;
}

static void finalizeMPI() {
  if (!gMpiInitialized) return;

  MPI_Finalize();
  gMpiInitialized = false;
}

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

}  // namespace utils

// TestRegistry implementation
TestRegistry& TestRegistry::instance() {
  static TestRegistry registry;
  return registry;
}

void TestRegistry::registerTest(const std::string& suiteName, const std::string& testName, TestFactory factory,
                                bool isPerfTest) {
  tests_.push_back({suiteName, testName, std::move(factory), isPerfTest});
}

void TestRegistry::addEnvironment(Environment* env) { environments_.push_back(env); }

// Returns true if the test should run given the filter string.
// Filter syntax:
//   ""          -> run all
//   "Pattern"   -> run only tests whose full name contains Pattern
//   "-Pattern"  -> run all tests EXCEPT those whose full name contains Pattern
static bool matchesFilter(const std::string& fullName, const std::string& filter) {
  if (filter.empty()) return true;
  if (filter[0] == '-') {
    // Negative filter: exclude matching tests
    std::string pattern = filter.substr(1);
    return fullName.find(pattern) == std::string::npos;
  }
  // Positive filter: include only matching tests
  return fullName.find(filter) != std::string::npos;
}

int TestRegistry::runAllTests(int argc, char* argv[]) {
  // Initialize MPI if not already initialized
  if (!gMpiInitialized) {
    utils::initializeMPI(argc, argv);
  }

  // Parse command line arguments
  std::string filter;
  bool excludePerfTests = false;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.find("--filter=") == 0) {
      filter = arg.substr(9);  // Length of "--filter="
    } else if (arg == "--filter" && i + 1 < argc) {
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
  int totalToRun = 0;
  int skippedByFilter = 0;
  for (const auto& entry : tests_) {
    std::string fullName = entry.suiteName + "." + entry.testName;
    if (excludePerfTests && entry.isPerfTest) {
      skippedByFilter++;
      continue;
    }
    if (!matchesFilter(fullName, filter)) {
      skippedByFilter++;
      continue;
    }
    totalToRun++;
  }

  if (gMpiRank == 0) {
    std::cout << "[==========] Running " << totalToRun << " tests";
    if (skippedByFilter > 0) {
      std::cout << " (" << skippedByFilter << " skipped by filter)";
    }
    std::cout << ".\n";
  }

  for (const auto& entry : tests_) {
    std::string fullName = entry.suiteName + "." + entry.testName;

    if (excludePerfTests && entry.isPerfTest) continue;
    if (!matchesFilter(fullName, filter)) continue;

    gCurrentTestPassed = true;
    gCurrentTestFailureMessage.clear();
    gCurrentTestName = fullName;

    if (gMpiRank == 0) {
      std::cout << "[ RUN      ] " << fullName << std::endl;
    }

    TestCase* testCase = nullptr;
    bool testSkipped = false;
    try {
      testCase = entry.factory();
      testCase->SetUp();
      testCase->TestBody();
      testCase->TearDown();
    } catch (const SkipException& e) {
      gCurrentTestPassed = true;
      testSkipped = true;
      if (gMpiRank == 0) {
        std::cout << "[  SKIPPED ] " << fullName << ": " << e.what() << std::endl;
      }
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

    delete testCase;
    gCurrentTestName.clear();

    if (testSkipped) {
      skipped++;
      continue;
    }

    // Synchronize test status across all MPI processes
    int localPassed = gCurrentTestPassed ? 1 : 0;
    int globalPassed = 1;
    if (gMpiInitialized) {
      MPI_Allreduce(&localPassed, &globalPassed, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    } else {
      globalPassed = localPassed;
    }

    if (gMpiRank == 0) {
      if (globalPassed) {
        std::cout << "[       OK ] " << fullName << std::endl;
        passed++;
      } else {
        std::cout << "[  FAILED  ] " << fullName << std::endl;
        failed++;
      }
    }
  }

  if (gMpiRank == 0) {
    std::cout << "[==========] " << totalToRun << " tests ran.\n";
    if (passed > 0) {
      std::cout << "[  PASSED  ] " << passed << " tests.\n";
    }
    if (skipped > 0) {
      std::cout << "[  SKIPPED ] " << skipped << " tests.\n";
    }
    if (failed > 0) {
      std::cout << "[  FAILED  ] " << failed << " tests.\n";
    }
  }

  // Tear down global test environments (in reverse order)
  for (auto it = environments_.rbegin(); it != environments_.rend(); ++it) {
    try {
      (*it)->TearDown();
      delete *it;
    } catch (const std::exception& e) {
      if (gMpiRank == 0) {
        std::cerr << "Failed to tear down test environment: " << e.what() << std::endl;
      }
    }
  }
  environments_.clear();

  return failed > 0 ? 1 : 0;
}

}  // namespace test
}  // namespace mscclpp
