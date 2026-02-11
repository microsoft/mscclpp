// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_TEST_FRAMEWORK_HPP_
#define MSCCLPP_TEST_FRAMEWORK_HPP_

#include <mpi.h>

#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <mscclpp/gpu.hpp>
#include <sstream>
#include <stdexcept>
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
  int num_processes;
  int process_rank;
  std::string timestamp;
  bool passed;
  std::string failure_message;
};

// Forward declarations
class Environment;
class TestCase;
class TestInfo;
class UnitTest;

// Test case base class
class TestCase {
 public:
  virtual ~TestCase() = default;
  virtual void SetUp() {}
  virtual void TearDown() {}
  virtual void TestBody() = 0;
};

// Environment base class (for global test setup/teardown)
class Environment {
 public:
  virtual ~Environment() = default;
  virtual void SetUp() {}
  virtual void TearDown() {}
};

// Test info class (for getting current test information)
class TestInfo {
 public:
  TestInfo(const std::string& suite, const std::string& name) : test_suite_name_(suite), test_name_(name) {}

  const char* test_suite_name() const { return test_suite_name_.c_str(); }
  const char* name() const { return test_name_.c_str(); }

 private:
  std::string test_suite_name_;
  std::string test_name_;
};

// UnitTest singleton (for getting test information)
class UnitTest {
 public:
  static UnitTest* GetInstance();

  const TestInfo* current_test_info() const { return current_test_info_; }
  void set_current_test_info(const TestInfo* info) { current_test_info_ = info; }

 private:
  UnitTest() = default;
  const TestInfo* current_test_info_ = nullptr;
};

// Test registry and runner
class TestRegistry {
 public:
  using TestFactory = std::function<TestCase*()>;

  static TestRegistry& instance();

  void registerTest(const std::string& test_suite, const std::string& test_name, TestFactory factory);
  void addGlobalTestEnvironment(Environment* env);
  int runAllTests(int argc, char* argv[]);
  void initGoogleTest(int* argc, char** argv);

 private:
  TestRegistry() = default;
  struct TestInfoInternal {
    std::string suite_name;
    std::string test_name;
    TestFactory factory;
  };
  std::vector<TestInfoInternal> tests_;
  std::vector<Environment*> environments_;
};

// Simple utility functions for testing
namespace utils {

// Test execution utilities (for performance tests)
int runMultipleTests(
    int argc, char* argv[],
    const std::vector<std::tuple<std::string, std::string, std::function<void(int, int, int)>>>& tests);

// MPI management
void initializeMPI(int argc, char* argv[]);
void cleanupMPI();
bool isMainRank();
int getMPIRank();
int getMPISize();

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
#define CUDA_CHECK(call) mscclpp::test::utils::cudaCheck(call, __FILE__, __LINE__)

// Test assertion helpers
void reportFailure(const char* file, int line, const std::string& message);
void reportSuccess();

}  // namespace utils

}  // namespace test
}  // namespace mscclpp

// Test registration macros
#define TEST(test_suite, test_name)                                                            \
  class test_suite##_##test_name##_Test : public ::mscclpp::test::TestCase {                   \
   public:                                                                                     \
    test_suite##_##test_name##_Test() {}                                                       \
    void TestBody() override;                                                                  \
  };                                                                                           \
  static bool test_suite##_##test_name##_registered = []() {                                   \
    ::mscclpp::test::TestRegistry::instance().registerTest(                                    \
        #test_suite, #test_name,                                                               \
        []() -> ::mscclpp::test::TestCase* { return new test_suite##_##test_name##_Test(); }); \
    return true;                                                                               \
  }();                                                                                         \
  void test_suite##_##test_name##_Test::TestBody()

#define TEST_F(test_fixture, test_name)                                                          \
  class test_fixture##_##test_name##_Test : public test_fixture {                                \
   public:                                                                                       \
    test_fixture##_##test_name##_Test() {}                                                       \
    void TestBody() override;                                                                    \
  };                                                                                             \
  static bool test_fixture##_##test_name##_registered = []() {                                   \
    ::mscclpp::test::TestRegistry::instance().registerTest(                                      \
        #test_fixture, #test_name,                                                               \
        []() -> ::mscclpp::test::TestCase* { return new test_fixture##_##test_name##_Test(); }); \
    return true;                                                                                 \
  }();                                                                                           \
  void test_fixture##_##test_name##_Test::TestBody()

// Test runner macro
#define RUN_ALL_TESTS() ::mscclpp::test::TestRegistry::instance().runAllTests(argc, argv)

// Assertion macros
#define EXPECT_TRUE(condition)                                                                          \
  do {                                                                                                  \
    if (!(condition)) {                                                                                 \
      ::mscclpp::test::utils::reportFailure(__FILE__, __LINE__, "Expected: " #condition " to be true"); \
    }                                                                                                   \
  } while (0)

#define EXPECT_FALSE(condition)                                                                          \
  do {                                                                                                   \
    if (condition) {                                                                                     \
      ::mscclpp::test::utils::reportFailure(__FILE__, __LINE__, "Expected: " #condition " to be false"); \
    }                                                                                                    \
  } while (0)

#define EXPECT_EQ(val1, val2)                                                         \
  do {                                                                                \
    auto v1 = (val1);                                                                 \
    auto v2 = (val2);                                                                 \
    if (!(v1 == v2)) {                                                                \
      std::ostringstream oss;                                                         \
      oss << "Expected: " #val1 " == " #val2 << "\n  Actual: " << v1 << " vs " << v2; \
      ::mscclpp::test::utils::reportFailure(__FILE__, __LINE__, oss.str());           \
    }                                                                                 \
  } while (0)

#define EXPECT_NE(val1, val2)                                                         \
  do {                                                                                \
    auto v1 = (val1);                                                                 \
    auto v2 = (val2);                                                                 \
    if (!(v1 != v2)) {                                                                \
      std::ostringstream oss;                                                         \
      oss << "Expected: " #val1 " != " #val2 << "\n  Actual: " << v1 << " vs " << v2; \
      ::mscclpp::test::utils::reportFailure(__FILE__, __LINE__, oss.str());           \
    }                                                                                 \
  } while (0)

#define EXPECT_LT(val1, val2)                                                        \
  do {                                                                               \
    auto v1 = (val1);                                                                \
    auto v2 = (val2);                                                                \
    if (!(v1 < v2)) {                                                                \
      std::ostringstream oss;                                                        \
      oss << "Expected: " #val1 " < " #val2 << "\n  Actual: " << v1 << " vs " << v2; \
      ::mscclpp::test::utils::reportFailure(__FILE__, __LINE__, oss.str());          \
    }                                                                                \
  } while (0)

#define EXPECT_LE(val1, val2)                                                         \
  do {                                                                                \
    auto v1 = (val1);                                                                 \
    auto v2 = (val2);                                                                 \
    if (!(v1 <= v2)) {                                                                \
      std::ostringstream oss;                                                         \
      oss << "Expected: " #val1 " <= " #val2 << "\n  Actual: " << v1 << " vs " << v2; \
      ::mscclpp::test::utils::reportFailure(__FILE__, __LINE__, oss.str());           \
    }                                                                                 \
  } while (0)

#define EXPECT_GT(val1, val2)                                                        \
  do {                                                                               \
    auto v1 = (val1);                                                                \
    auto v2 = (val2);                                                                \
    if (!(v1 > v2)) {                                                                \
      std::ostringstream oss;                                                        \
      oss << "Expected: " #val1 " > " #val2 << "\n  Actual: " << v1 << " vs " << v2; \
      ::mscclpp::test::utils::reportFailure(__FILE__, __LINE__, oss.str());          \
    }                                                                                \
  } while (0)

#define EXPECT_GE(val1, val2)                                                         \
  do {                                                                                \
    auto v1 = (val1);                                                                 \
    auto v2 = (val2);                                                                 \
    if (!(v1 >= v2)) {                                                                \
      std::ostringstream oss;                                                         \
      oss << "Expected: " #val1 " >= " #val2 << "\n  Actual: " << v1 << " vs " << v2; \
      ::mscclpp::test::utils::reportFailure(__FILE__, __LINE__, oss.str());           \
    }                                                                                 \
  } while (0)

#define ASSERT_TRUE(condition)                                                                          \
  do {                                                                                                  \
    if (!(condition)) {                                                                                 \
      ::mscclpp::test::utils::reportFailure(__FILE__, __LINE__, "Expected: " #condition " to be true"); \
      throw std::runtime_error("Test assertion failed");                                                \
    }                                                                                                   \
  } while (0)

#define ASSERT_FALSE(condition)                                                                          \
  do {                                                                                                   \
    if (condition) {                                                                                     \
      ::mscclpp::test::utils::reportFailure(__FILE__, __LINE__, "Expected: " #condition " to be false"); \
      throw std::runtime_error("Test assertion failed");                                                 \
    }                                                                                                    \
  } while (0)

#define ASSERT_EQ(val1, val2)                                                         \
  do {                                                                                \
    auto v1 = (val1);                                                                 \
    auto v2 = (val2);                                                                 \
    if (!(v1 == v2)) {                                                                \
      std::ostringstream oss;                                                         \
      oss << "Expected: " #val1 " == " #val2 << "\n  Actual: " << v1 << " vs " << v2; \
      ::mscclpp::test::utils::reportFailure(__FILE__, __LINE__, oss.str());           \
      throw std::runtime_error("Test assertion failed");                              \
    }                                                                                 \
  } while (0)

#define ASSERT_NE(val1, val2)                                                         \
  do {                                                                                \
    auto v1 = (val1);                                                                 \
    auto v2 = (val2);                                                                 \
    if (!(v1 != v2)) {                                                                \
      std::ostringstream oss;                                                         \
      oss << "Expected: " #val1 " != " #val2 << "\n  Actual: " << v1 << " vs " << v2; \
      ::mscclpp::test::utils::reportFailure(__FILE__, __LINE__, oss.str());           \
      throw std::runtime_error("Test assertion failed");                              \
    }                                                                                 \
  } while (0)

#define ASSERT_LT(val1, val2)                                                        \
  do {                                                                               \
    auto v1 = (val1);                                                                \
    auto v2 = (val2);                                                                \
    if (!(v1 < v2)) {                                                                \
      std::ostringstream oss;                                                        \
      oss << "Expected: " #val1 " < " #val2 << "\n  Actual: " << v1 << " vs " << v2; \
      ::mscclpp::test::utils::reportFailure(__FILE__, __LINE__, oss.str());          \
      throw std::runtime_error("Test assertion failed");                             \
    }                                                                                \
  } while (0)

#define ASSERT_LE(val1, val2)                                                         \
  do {                                                                                \
    auto v1 = (val1);                                                                 \
    auto v2 = (val2);                                                                 \
    if (!(v1 <= v2)) {                                                                \
      std::ostringstream oss;                                                         \
      oss << "Expected: " #val1 " <= " #val2 << "\n  Actual: " << v1 << " vs " << v2; \
      ::mscclpp::test::utils::reportFailure(__FILE__, __LINE__, oss.str());           \
      throw std::runtime_error("Test assertion failed");                              \
    }                                                                                 \
  } while (0)

#define ASSERT_GT(val1, val2)                                                        \
  do {                                                                               \
    auto v1 = (val1);                                                                \
    auto v2 = (val2);                                                                \
    if (!(v1 > v2)) {                                                                \
      std::ostringstream oss;                                                        \
      oss << "Expected: " #val1 " > " #val2 << "\n  Actual: " << v1 << " vs " << v2; \
      ::mscclpp::test::utils::reportFailure(__FILE__, __LINE__, oss.str());          \
      throw std::runtime_error("Test assertion failed");                             \
    }                                                                                \
  } while (0)

#define ASSERT_GE(val1, val2)                                                         \
  do {                                                                                \
    auto v1 = (val1);                                                                 \
    auto v2 = (val2);                                                                 \
    if (!(v1 >= v2)) {                                                                \
      std::ostringstream oss;                                                         \
      oss << "Expected: " #val1 " >= " #val2 << "\n  Actual: " << v1 << " vs " << v2; \
      ::mscclpp::test::utils::reportFailure(__FILE__, __LINE__, oss.str());           \
      throw std::runtime_error("Test assertion failed");                              \
    }                                                                                 \
  } while (0)

#define ASSERT_NO_THROW(statement)                                                                         \
  do {                                                                                                     \
    try {                                                                                                  \
      statement;                                                                                           \
    } catch (const std::exception& e) {                                                                    \
      std::ostringstream oss;                                                                              \
      oss << "Expected: " #statement " not to throw\n  Actual: threw " << e.what();                        \
      ::mscclpp::test::utils::reportFailure(__FILE__, __LINE__, oss.str());                                \
      throw std::runtime_error("Test assertion failed");                                                   \
    } catch (...) {                                                                                        \
      ::mscclpp::test::utils::reportFailure(                                                               \
          __FILE__, __LINE__, "Expected: " #statement " not to throw\n  Actual: threw unknown exception"); \
      throw std::runtime_error("Test assertion failed");                                                   \
    }                                                                                                      \
  } while (0)

#define FAIL()                                                                \
  do {                                                                        \
    ::mscclpp::test::utils::reportFailure(__FILE__, __LINE__, "Test failed"); \
    throw std::runtime_error("Test failed");                                  \
  } while (0)

// Helper class for GTEST_SKIP functionality
class SkipHelper {
 public:
  explicit SkipHelper(const char* file, int line) : file_(file), line_(line) {}
  template <typename T>
  SkipHelper& operator<<(const T& value) {
    message_ << value;
    return *this;
  }
  ~SkipHelper() noexcept(false) {
    std::string msg = message_.str();
    if (!msg.empty()) {
      ::mscclpp::test::utils::reportFailure(file_, line_, "Test skipped: " + msg);
    } else {
      ::mscclpp::test::utils::reportFailure(file_, line_, "Test skipped");
    }
    throw std::runtime_error("Test skipped");
  }

 private:
  const char* file_;
  int line_;
  std::ostringstream message_;
};

#define GTEST_SKIP() ::SkipHelper(__FILE__, __LINE__)

// Create a namespace alias for compatibility with GTest code
namespace testing = ::mscclpp::test;

// Helper functions for compatibility with GTest API
inline void InitGoogleTest(int* argc, char** argv) {
  ::mscclpp::test::TestRegistry::instance().initGoogleTest(argc, argv);
}

inline ::mscclpp::test::Environment* AddGlobalTestEnvironment(::mscclpp::test::Environment* env) {
  ::mscclpp::test::TestRegistry::instance().addGlobalTestEnvironment(env);
  return env;
}

#endif  // MSCCLPP_TEST_FRAMEWORK_HPP_
