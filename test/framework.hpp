// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_TEST_FRAMEWORK_HPP_
#define MSCCLPP_TEST_FRAMEWORK_HPP_

#include <mpi.h>

#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mscclpp/gpu.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mscclpp {
namespace test {

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

// Test registry and runner
class TestRegistry {
 public:
  using TestFactory = std::function<TestCase*()>;

  static TestRegistry& instance();

  void registerTest(const std::string& suiteName, const std::string& testName, TestFactory factory,
                    bool isPerfTest = false);
  void addEnvironment(Environment* env);
  int runAllTests(int argc, char* argv[]);

 private:
  TestRegistry() = default;
  struct TestEntry {
    std::string suiteName;
    std::string testName;
    TestFactory factory;
    bool isPerfTest;
  };
  std::vector<TestEntry> tests_;
  std::vector<Environment*> environments_;
};

// Returns "Suite.Name" for the currently running test, or "" if none.
std::string currentTestName();

// Utility functions
namespace utils {

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
  std::chrono::high_resolution_clock::time_point startTime_;
  std::chrono::high_resolution_clock::time_point endTime_;
  bool isRunning_;
};

// CUDA utilities
void cudaCheck(cudaError_t err, const char* file, int line);
#define CUDA_CHECK(call) mscclpp::test::utils::cudaCheck(call, __FILE__, __LINE__)

// Test assertion helpers
void reportFailure(const char* file, int line, const std::string& message);
void reportSuccess();

}  // namespace utils

// Exception for test skips
class SkipException : public std::runtime_error {
 public:
  explicit SkipException(const std::string& message) : std::runtime_error(message) {}
};

// Helper class for FAIL() macro — supports message streaming via operator<<
class FailHelper {
 public:
  explicit FailHelper(const char* file, int line) : file_(file), line_(line) {}
  template <typename T>
  FailHelper& operator<<(const T& value) {
    message_ << value;
    return *this;
  }
  ~FailHelper() noexcept(false) {
    std::string msg = message_.str();
    if (!msg.empty()) {
      ::mscclpp::test::utils::reportFailure(file_, line_, "Test failed: " + msg);
    } else {
      ::mscclpp::test::utils::reportFailure(file_, line_, "Test failed");
    }
    throw std::runtime_error("Test failed");
  }

 private:
  const char* file_;
  int line_;
  std::ostringstream message_;
};

// Helper class for SKIP_TEST() macro — supports message streaming via operator<<
// Usage: SKIP_TEST() << "Reason for skipping";
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
      throw SkipException("Test skipped: " + msg);
    } else {
      throw SkipException("Test skipped");
    }
  }

 private:
  const char* file_;
  int line_;
  std::ostringstream message_;
};

// SFINAE helper: resolves to T if T is a complete type (user-defined fixture),
// otherwise falls back to TestCase. This lets TEST() work with or without a fixture class.
namespace detail {
template <typename...>
using void_t = void;

template <typename T, typename = void_t<>>
struct FixtureOf {
  using type = TestCase;
};
template <typename T>
struct FixtureOf<T, void_t<decltype(sizeof(T))>> {
  using type = T;
};
}  // namespace detail

}  // namespace test
}  // namespace mscclpp

// --- Test registration macros ---
// TEST(Suite, Name): if Suite is a previously-defined class, the test inherits from it (fixture).
// Otherwise, the test inherits from TestCase (no fixture needed).

#define TEST(test_fixture, test_name)                                                                       \
  class test_fixture;                                                                                       \
  class test_fixture##_##test_name##_Test : public ::mscclpp::test::detail::FixtureOf<test_fixture>::type { \
   public:                                                                                                  \
    void TestBody() override;                                                                               \
  };                                                                                                        \
  static bool test_fixture##_##test_name##_registered = []() {                                              \
    ::mscclpp::test::TestRegistry::instance().registerTest(                                                 \
        #test_fixture, #test_name,                                                                          \
        []() -> ::mscclpp::test::TestCase* { return new test_fixture##_##test_name##_Test(); });            \
    return true;                                                                                            \
  }();                                                                                                      \
  void test_fixture##_##test_name##_Test::TestBody()

#define PERF_TEST(test_fixture, test_name)                                                                  \
  class test_fixture;                                                                                       \
  class test_fixture##_##test_name##_Test : public ::mscclpp::test::detail::FixtureOf<test_fixture>::type { \
   public:                                                                                                  \
    void TestBody() override;                                                                               \
  };                                                                                                        \
  static bool test_fixture##_##test_name##_registered = []() {                                              \
    ::mscclpp::test::TestRegistry::instance().registerTest(                                                 \
        #test_fixture, #test_name,                                                                          \
        []() -> ::mscclpp::test::TestCase* { return new test_fixture##_##test_name##_Test(); }, true);      \
    return true;                                                                                            \
  }();                                                                                                      \
  void test_fixture##_##test_name##_Test::TestBody()

// --- Test runner macro ---
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

// --- Test control macros ---

// Fail the current test immediately. Usage: FAIL() << "reason";
#define FAIL() ::mscclpp::test::FailHelper(__FILE__, __LINE__)

// Skip the current test. Usage: SKIP_TEST() << "reason";
#define SKIP_TEST() ::mscclpp::test::SkipHelper(__FILE__, __LINE__)

#endif  // MSCCLPP_TEST_FRAMEWORK_HPP_
