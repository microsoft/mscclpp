# Group Execution Plan Testing

This document describes how to build and run the group execution plan tests and examples.

## Building

To build the group execution plan example and tests:

```bash
# From the build directory
make group_execution_plan_example

# To build all tests including unit tests
make unit_tests
```

## Running the Example

### Simple Execution

```bash
# Run the standalone example
./test/group_execution_plan_example
```

### Using CTest

```bash
# Run through CTest
ctest -R group_execution_plan_example -V
```

### Using the Test Runner Script

```bash
# Make the script executable and run
chmod +x test/run_group_execution_plan_test.sh
./test/run_group_execution_plan_test.sh
```

## Running Unit Tests

```bash
# Run all unit tests
ctest -R unit_tests -V

# Run only group-related unit tests
./build/test/unit_tests --gtest_filter="*Group*" -V
```

## Test Structure

- `group_execution_plan_example.cpp` - Standalone example demonstrating the API
- `unit/group_execution_plan_test.cpp` - Unit tests for the group execution plan functionality
- `unit/group_test.cpp` - Unit tests for basic group management

## Example Output

The example will demonstrate:

1. **Basic Functionality**: Tests utility functions without requiring full MSCCLPP setup
2. **Execution Plan All-to-Allv**: Shows how to use execution plans for variable-size operations
3. **Multiple Operations**: Demonstrates batching multiple execution plan operations
4. **Custom Operations**: Shows how to create custom operations with execution plan context
5. **Capabilities Analysis**: Demonstrates how to analyze execution plan capabilities

## Generated Files

The example creates temporary JSON execution plan files for testing:
- `alltoallv_execution_plan.json`
- `alltoallv_plan1.json`
- `alltoallv_plan2.json`
- `custom_plan.json`
- `test_plan.json`
- `basic_test_plan.json`

These files are automatically cleaned up after the test runs.

## Notes

- Some functionality requires proper MPI and MSCCLPP initialization
- The example includes graceful error handling for missing dependencies
- Unit tests provide more comprehensive coverage of edge cases and error conditions
- Tests are designed to handle environments where full MSCCLPP setup is not available

## Compilation Fixes

The tests have been updated to handle common compilation issues:

### Multiple Main Functions
- **Fixed**: Both unit test files had their own `main` functions causing linker conflicts
- **Solution**: Removed `main` functions from test files since GoogleTest's `gtest_main` provides it
- **Configuration**: CMake links unit tests with `GTest::gtest_main` automatically

### Unused Variables
- **Fixed**: Unused variable warnings in test code
- **Solution**: Either use the variables or replace with `EXPECT_NO_THROW` for testing without assignment

### Communicator Creation
- **Fixed**: Communicator constructor requires Bootstrap and Context parameters
- **Solution**: Uses TcpBootstrap for test environments with proper initialization
- **Fallback**: Gracefully handles environments where full initialization isn't possible

### ErrorCode Enum
- **Fixed**: ErrorCode doesn't have a Success member
- **Solution**: Updated groupResultToErrorCode function to handle Success appropriately
- **Tests**: Updated to test actual ErrorCode mappings rather than non-existent Success

### RegisteredMemory Creation
- **Fixed**: RegisteredMemory should be created using `communicator->registerMemory()`
- **Solution**: Use proper pattern with `comm_->registerMemory(buffer, size, transport)`
- **Transport**: Use `mscclpp::NoTransports` for simple test cases

### Pointer vs Object Access
- **Fixed**: Incorrect usage of `.` operator on shared_ptr objects
- **Solution**: Use `->` operator for shared_ptr member access (e.g., `customOp->isComplete()`)

## Troubleshooting

If you encounter build errors:

1. **Multiple main function errors**: Ensure you're not defining main functions in unit test files
   ```cpp
   // Don't do this in unit test files:
   // int main(int argc, char** argv) {
   //   ::testing::InitGoogleTest(&argc, argv);
   //   return RUN_ALL_TESTS();
   // }
   
   // GoogleTest's gtest_main provides this automatically
   ```

2. **Communicator constructor errors**: Make sure you're providing proper Bootstrap and Context
   ```cpp
   auto bootstrap = std::make_shared<TcpBootstrap>(0, 1);
   bootstrap->initialize(bootstrap->createUniqueId());
   auto comm = std::make_shared<Communicator>(bootstrap);
   ```

3. **RegisteredMemory creation errors**: Use the communicator to register memory properly
   ```cpp
   void* buffer = malloc(1024);
   auto registeredMemory = comm->registerMemory(buffer, 1024, mscclpp::NoTransports);
   auto memory = std::make_shared<RegisteredMemory>(std::move(registeredMemory));
   free(buffer); // Clean up when done
   ```

4. **ErrorCode::Success errors**: Use the mapped error codes or handle success cases separately
   ```cpp
   // Don't use ErrorCode::Success (doesn't exist)
   // Instead, use the actual ErrorCode enum values
   EXPECT_EQ(groupResultToErrorCode(GroupResult::InvalidUsage), ErrorCode::InvalidUsage);
   ```

5. **Pointer access errors**: Use correct operators for shared_ptr objects
   ```cpp
   // Correct: Use -> for shared_ptr member access
   EXPECT_TRUE(customOp->isComplete());
   
   // Incorrect: Don't use . operator on shared_ptr
   // EXPECT_TRUE(customOp.isComplete());  // This will fail
   ```

6. **Unused variable warnings**: Use variables or test without assignment
   ```cpp
   // Instead of storing unused variables:
   // auto successCode = groupResultToErrorCode(GroupResult::Success);
   
   // Just test that it doesn't crash:
   EXPECT_NO_THROW(groupResultToErrorCode(GroupResult::Success));
   ```

7. **Missing CUDA types**: Tests avoid CUDA-specific types to work in more environments
   ```cpp
   // Use simple transport flags for testing
   auto memory = comm->registerMemory(buffer, size, mscclpp::NoTransports);
   ```

8. **Missing source files**: Make sure the group management source files are present:
   - `src/group.cc`
   - `src/group_execution_plan.cc`
   - `include/mscclpp/group.hpp`
   - `include/mscclpp/group_execution_plan.hpp`

9. **CMakeLists.txt updates**: Verify that the CMakeLists.txt files have been updated to include the new test targets

10. **Dependencies**: Check that all dependencies are properly linked (MPI, CUDA, etc.)

## Environment Considerations

The tests are designed to work in various environments:
- **Full MSCCLPP environment**: All functionality works
- **Limited environment**: Basic functionality tests work, communicator-dependent tests are skipped
- **CI/CD environment**: Robust error handling prevents test failures due to missing resources
- **CUDA/ROCm optional**: Tests avoid CUDA-specific types when possible for broader compatibility

## Common Patterns

### Unit Test Structure
```cpp
// Unit tests don't need main functions - GoogleTest provides gtest_main
TEST_F(GroupTest, SomeTest) {
  // Test implementation...
}

// For tests requiring setup, use the test fixture
class GroupTest : public ::testing::Test {
 protected:
  void SetUp() override { /* setup */ }
  void TearDown() override { /* cleanup */ }
};
```

### Creating Test Communicators
```cpp
std::shared_ptr<Communicator> createTestCommunicator() {
  try {
    auto bootstrap = std::make_shared<TcpBootstrap>(0, 1);
    bootstrap->initialize(bootstrap->createUniqueId());
    return std::make_shared<Communicator>(bootstrap);
  } catch (const std::exception& e) {
    return nullptr; // Handle gracefully
  }
}
```

### Registering Memory for Tests
```cpp
void* buffer = malloc(size);
auto registeredMemory = comm->registerMemory(buffer, size, mscclpp::NoTransports);
auto memory = std::make_shared<RegisteredMemory>(std::move(registeredMemory));
// Use memory...
free(buffer); // Clean up
```

### Safe Test Execution
```cpp
TEST_F(GroupTest, SomeTest) {
  if (!comm_) {
    GTEST_SKIP() << "Communicator not available for testing";
  }
  
  // Test logic here...
}
```

### Testing Functions That May Throw Without Using Return Values
```cpp
// Instead of:
// auto result = someFunction();  // If result is unused
// 
// Use:
EXPECT_NO_THROW(someFunction());
```