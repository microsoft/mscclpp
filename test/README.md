# MSCCL++ C++ Test Framework

A lightweight, GTest-like test framework with MPI support for testing MSCCL++ C++ APIs. Defined in `framework.hpp` / `framework.cc`.

## Adding a New Test (Step-by-Step)

### Single-process test (unit/)

1. **Create the test file** `test/unit/my_feature_tests.cc` (or `.cu` for CUDA):

    ```cpp
    #include "../framework.hpp"
    #include <mscclpp/my_feature.hpp>

    TEST(MyFeatureTest, BasicUsage) {
      EXPECT_EQ(myFunction(), 42);
    }
    ```

2. **Register it in CMake** — add the filename to `test/unit/CMakeLists.txt`:

    ```cmake
    target_sources(unit_tests PRIVATE
        ...
        my_feature_tests.cc   # <-- add here
    )
    ```

3. **Build and run**:

    ```bash
    cmake --build build -j
    ./build/test/unit_tests --filter=MyFeatureTest
    ```

### Multi-process test (mp_unit/)

1. **Create the test file** `test/mp_unit/my_feature_tests.cc` (or `.cu`):

    ```cpp
    #include "mp_unit_tests.hpp"

    TEST(MyFeatureTest, MultiRank) {
      int rank = gEnv->rank;
      EXPECT_GE(rank, 0);
    }
    ```

    Use fixtures from `mp_unit_tests.hpp` (e.g., `CommunicatorTest`) if you need pre-established connections.

2. **Register it in CMake** — add the filename to `test/mp_unit/CMakeLists.txt`:

    ```cmake
    target_sources(mp_unit_tests PRIVATE
        ...
        my_feature_tests.cc   # <-- add here
    )
    ```

3. **Build and run**:

    ```bash
    cmake --build build -j
    mpirun -np 2 ./build/test/mp_unit_tests --filter=MyFeatureTest
    ```

### Notes

- No separate test registration step is needed — `TEST()` auto-registers via static initialization.
- The `test_framework` static library is built from `framework.cc` in the top-level `test/CMakeLists.txt` and linked into both `unit_tests` and `mp_unit_tests`. You do not need to modify it.
- Use `.cu` extension for files that contain CUDA kernel code; use `.cc` for host-only tests.
- Each test binary needs a `main()` that calls `RUN_ALL_TESTS()`. See `unit/unit_tests_main.cc` (single-process) and `mp_unit/mp_unit_tests.cc` (multi-process with `Environment` setup).
- Additional run options: `--filter=-Pattern` (exclude), `--exclude-perf-tests` (skip `PERF_TEST`s).

## Macros

| Macro | Behavior |
|---|---|
| `TEST(Suite, Name)` | Register a test. If `Suite` is a defined class, it's used as a fixture. |
| `PERF_TEST(Suite, Name)` | Same as `TEST` but marked as perf (skippable via `--exclude-perf-tests`). |
| `EXPECT_*` | Non-fatal assertions: `EXPECT_TRUE`, `EXPECT_FALSE`, `EXPECT_EQ`, `EXPECT_NE`, `EXPECT_LT`, `EXPECT_LE`, `EXPECT_GT`, `EXPECT_GE` |
| `ASSERT_*` | Fatal assertions (abort test on failure): same variants as `EXPECT_*`, plus `ASSERT_NO_THROW` |
| `FAIL()` | Fail immediately. Supports streaming: `FAIL() << "reason";` |
| `SKIP_TEST()` | Skip the current test. Supports streaming: `SKIP_TEST() << "reason";` |
| `CUDA_CHECK(call)` | Check a CUDA API return code, throw on error. |

## Fixtures

Define a class inheriting from `mscclpp::test::TestCase` with `SetUp()` / `TearDown()`, then use the class name as the suite name:

```cpp
class MyFixture : public mscclpp::test::TestCase {
 public:
  void SetUp() override { /* per-test setup */ }
  void TearDown() override { /* per-test cleanup */ }
 protected:
  int sharedState_ = 0;
};

TEST(MyFixture, SomeTest) {
  sharedState_ = 42;
  EXPECT_EQ(sharedState_, 42);
}
```

See `mp_unit/mp_unit_tests.hpp` (`BootstrapTest`, `CommunicatorTest`, etc.) for real fixture examples.

## Global Environments

Register an `Environment` subclass for one-time global setup/teardown (e.g., MPI bootstrap):

```cpp
class MyEnv : public mscclpp::test::Environment {
 public:
  void SetUp() override { /* global init */ }
  void TearDown() override { /* global cleanup */ }
};

// In main(), before RUN_ALL_TESTS():
mscclpp::test::TestRegistry::instance().addEnvironment(new MyEnv());
```

See `mp_unit/mp_unit_tests.cc` for the `MultiProcessTestEnv` example.

## Utilities

- `mscclpp::test::utils::isMainRank()` — true on MPI rank 0
- `mscclpp::test::utils::getMPIRank()` / `getMPISize()`
- `mscclpp::test::utils::Timer` — high-resolution timer with `start()`, `stop()`, `elapsedMilliseconds()`
- `mscclpp::test::currentTestName()` — returns `"Suite.Name"` for the running test