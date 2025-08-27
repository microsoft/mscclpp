# MSCCLPP Group Management System

The MSCCLPP Group Management System provides APIs for batching multiple operations together, enabling more efficient execution through reduced synchronization overhead and improved kernel fusion opportunities. This system is inspired by NCCL's group management but designed specifically for MSCCLPP's architecture.

## Overview

The group management system allows you to:
- Batch multiple operations (connections, memory transfers, etc.) for efficient execution
- Reduce synchronization overhead by executing operations together
- Enable kernel fusion opportunities for better performance
- Support both blocking and non-blocking execution modes
- Handle complex nested workflows
- Provide robust error handling and recovery mechanisms

## Key Components

### 1. GroupManager
The main interface for group management operations:
- `groupStart()` - Start a new operation group
- `groupEnd(blocking)` - Execute all batched operations
- `addOperation()` - Add operations to the current group
- Helper methods for common operation types

### 2. Operation Classes
Base classes for different types of operations:
- `ConnectOperation` - Connection establishment
- `SendMemoryOperation` - Memory send operations
- `RecvMemoryOperation` - Memory receive operations
- `CustomOperation` - User-defined operations

### 3. GroupScope (RAII)
Automatic group management using RAII pattern:
- Constructor calls `groupStart()`
- Destructor calls `groupEnd()`
- Exception-safe operation batching

## Basic Usage

### Simple Group with RAII
```cpp
#include <mscclpp/group.hpp>

// RAII style - automatic group management
{
    GroupScope group(true);  // blocking execution
    if (group.isValid()) {
        // Add operations
        auto op1 = GroupManager::addConnect(comm, config1, rank1, tag1);
        auto op2 = GroupManager::addConnect(comm, config2, rank2, tag2);
        
        // Operations execute when scope ends
    }
}
```

### Manual Group Management
```cpp
// Manual control
auto result = GroupManager::groupStart();
if (result == GroupResult::Success) {
    // Add operations
    GroupManager::addConnect(comm, config, rank, tag);
    
    // Execute non-blocking
    result = GroupManager::groupEnd(false);
    if (result == GroupResult::InProgress) {
        // Wait for completion
        result = GroupManager::waitForCompletion(5000);
    }
}
```

### Convenience Functions
```cpp
// Batch multiple connections
std::vector<std::tuple<std::shared_ptr<Communicator>, EndpointConfig, int, int>> connections = {
    {comm1, config1, rank1, tag1},
    {comm2, config2, rank2, tag2}
};

auto futures = groupConnect(connections, true);  // blocking execution
```

## Advanced Features

### Nested Groups
```cpp
{
    GroupScope outerGroup(true);
    // Outer group operations
    
    {
        GroupScope innerGroup(true);
        // Inner group operations
        // Inner group executes first
    }
    
    // More outer group operations
    // Outer group executes after inner group completes
}
```

### Custom Operations
```cpp
{
    GroupScope group(true);
    
    auto customOp = GroupManager::addCustom(
        comm,
        []() -> GroupResult {
            // Your custom logic here
            return GroupResult::Success;
        },
        []() -> bool {
            // Check if operation is complete
            return true;
        },
        []() {
            // Optional cleanup/cancel logic
        }
    );
}
```

## Performance Benefits

### 1. Reduced Synchronization Overhead
Traditional approach:
```cpp
// Each operation has individual synchronization overhead
auto conn1 = comm->connect(config1, rank1, tag1);
conn1.wait();  // Synchronization point

auto conn2 = comm->connect(config2, rank2, tag2);
conn2.wait();  // Another synchronization point
```

Grouped approach:
```cpp
// Single synchronization point for all operations
{
    GroupScope group(true);
    auto op1 = GroupManager::addConnect(comm, config1, rank1, tag1);
    auto op2 = GroupManager::addConnect(comm, config2, rank2, tag2);
    // All operations execute together with single synchronization
}
```

### 2. Kernel Fusion Opportunities
The group system analyzes operations and can:
- Combine multiple small data transfers into larger, more efficient transfers
- Merge compatible kernels to reduce launch overhead
- Optimize memory access patterns across operations
- Reduce the number of synchronization barriers

## Integration with Existing MSCCLPP Code

The group management system integrates seamlessly with existing MSCCLPP code:

### Before (Traditional)
```cpp
auto context = std::make_shared<Context>();
auto comm = std::make_shared<Communicator>(context, bootstrap);

// Individual operations
auto conn1 = comm->connect(EndpointConfig(Transport::IB0), 1, 0);
auto conn2 = comm->connect(EndpointConfig(Transport::IB0), 1, 1);

// Wait for each operation
auto connection1 = conn1.get();
auto connection2 = conn2.get();
```

### After (With Groups)
```cpp
auto context = std::make_shared<Context>();
auto comm = std::make_shared<Communicator>(context, bootstrap);

// Grouped operations
{
    GroupScope group(true);
    auto op1 = GroupManager::addConnect(comm, EndpointConfig(Transport::IB0), 1, 0);
    auto op2 = GroupManager::addConnect(comm, EndpointConfig(Transport::IB0), 1, 1);
    
    // All operations execute together when scope ends
    // Access results through operation futures
    auto connection1 = op1->getFuture().get();
    auto connection2 = op2->getFuture().get();
}
```

## Best Practices

1. **Use GroupScope When Possible** - RAII ensures proper cleanup
2. **Batch Related Operations** - Group operations with similar resource requirements
3. **Choose Appropriate Execution Mode** - Blocking vs non-blocking based on workflow
4. **Handle Errors Appropriately** - Always check validity and use try-catch blocks
5. **Optimize for Your Workload** - Profile to identify bottlenecks

## Thread Safety

The group management system is designed with thread safety in mind:
- Each thread has its own group state (thread-local storage)
- Operations within a group are thread-safe
- Multiple threads can have concurrent groups
- Proper synchronization for shared resources

This implementation provides a comprehensive group management system for MSCCLPP that enables significant performance optimizations while maintaining ease of use and robust error handling.

# Execution Plan-Aware Group Management for All_to_Allv

This extension to the MSCCLPP group management system adds support for variable chunk sizes obtained from execution plans, enabling efficient all_to_allv operations using the DSL.

## Overview

The execution plan-aware group management extends the existing group management system to:

1. **Extract chunk size information** from JSON execution plans
2. **Support variable-size operations** like all_to_allv with per-rank different data sizes
3. **Integrate with DSL execution engine** for optimized collective operations
4. **Maintain the same API patterns** as the original group management system

## Key Components

### ExecutionPlanAllToAllvOperation

A specialized operation class that extracts variable chunk size information from execution plans and performs all_to_allv operations.

```cpp
// Create an all_to_allv operation based on execution plan
auto operation = std::make_shared<ExecutionPlanAllToAllvOperation>(
    comm, executionPlan, 
    sendBuffer, recvBuffer,
    inputSize, outputSize, tag);

// Get chunk size information
const auto& info = operation->getAllToAllvInfo();
```

### ExecutionPlanGroupManager

Extended group manager with execution plan support:

```cpp
// Add execution plan-based all_to_allv to group
auto operation = ExecutionPlanGroupManager::addExecutionPlanAllToAllv(
    comm, plan, sendBuffer, recvBuffer, inputSize, outputSize, tag);

// Add custom operation with execution plan context
auto customOp = ExecutionPlanGroupManager::addExecutionPlanCustom(
    comm, plan, executeFunc, isCompleteFunc, cancelFunc, tag);
```

### AllToAllvInfo Structure

Contains extracted chunk size specifications:

```cpp
struct AllToAllvInfo {
    std::vector<ChunkSizeSpec> chunkSpecs;  // Per-rank specifications
    size_t totalSendSize;                   // Total send buffer size
    size_t totalRecvSize;                   // Total receive buffer size
    uint32_t maxChunks;                     // Maximum chunks
    bool isVariable;                        // Whether sizes are variable
};
```

### ChunkSizeSpec Structure

Defines variable chunk sizes for each rank pair:

```cpp
struct ChunkSizeSpec {
    int rank;           // Source rank
    int destRank;       // Destination rank
    size_t sendSize;    // Data size to send
    size_t recvSize;    // Data size to receive
    size_t sendOffset;  // Offset in send buffer
    size_t recvOffset;  // Offset in receive buffer
};
```

## Usage Examples

### Basic All_to_Allv with Execution Plan

```cpp
#include <mscclpp/group_execution_plan.hpp>

// Load execution plan with variable chunk sizes
auto plan = std::make_shared<ExecutionPlan>("alltoallv_plan.json", rank);
auto comm = std::make_shared<Communicator>();

// Prepare buffers
std::vector<char> sendBuffer(totalSendSize);
std::vector<char> recvBuffer(totalRecvSize);

// Use RAII group scope
{
    ExecutionPlanGroupScope scope(plan, true);
    
    auto operation = ExecutionPlanGroupManager::addExecutionPlanAllToAllv(
        comm, plan, 
        sendBuffer.data(), recvBuffer.data(),
        sendBuffer.size(), recvBuffer.size(), 0);
    
    // Operation executes when scope ends
}
```

### Multiple Operations with Different Plans

```cpp
// Multiple operations with different execution plans
std::vector<std::tuple<std::shared_ptr<Communicator>, 
                      std::shared_ptr<ExecutionPlan>, 
                      void*, void*, size_t, size_t, int>> operations = {
    {comm1, plan1, send1, recv1, size1, size1, 1},
    {comm2, plan2, send2, recv2, size2, size2, 2}
};

auto results = groupExecutionPlanAllToAllv(operations, true);
```

### Custom Operation with Execution Plan Context

```cpp
{
    ExecutionPlanGroupScope scope(plan, true);
    
    auto customOp = ExecutionPlanGroupManager::addExecutionPlanCustom(
        comm, plan,
        []() -> GroupResult {
            // Custom logic using execution plan context
            return GroupResult::Success;
        },
        []() -> bool { return true; },
        nullptr, tag);
}
```

## Utility Functions

### Chunk Size Analysis

```cpp
// Extract all_to_allv information from execution plan
AllToAllvInfo info = extractAllToAllvInfo(plan, inputSize, outputSize);

// Check if plan supports variable chunk sizes
bool supportsVariable = supportsVariableChunkSizes(plan);

// Get maximum chunk size for buffer allocation
size_t maxChunk = getMaxChunkSize(plan, inputSize, outputSize);

// Calculate detailed chunk specifications
auto specs = calculateChunkSizes(plan, inputSize, outputSize);
```

## Integration with DSL

The execution plan-aware group management is designed to integrate with the MSCCLPP DSL for optimized execution:

1. **JSON Execution Plans**: Plans contain variable chunk size specifications
2. **Runtime Size Determination**: Chunk sizes are extracted at runtime from execution plans
3. **DSL Integration**: Operations use DSL execution engine with variable chunk information
4. **Placeholder Support**: Plans can contain placeholders that are replaced with actual sizes

## Benefits

1. **Variable Size Support**: Enables all_to_allv operations with per-rank variable data sizes
2. **Execution Plan Integration**: Leverages DSL execution plans for optimized collective operations
3. **Consistent API**: Maintains the same group management patterns as the base system
4. **Performance**: Reduces synchronization overhead through batching
5. **Flexibility**: Supports custom operations with execution plan context

## Error Handling

The system provides comprehensive error handling:

- **Invalid execution plans**: Throws `std::invalid_argument` for null or invalid plans
- **Buffer validation**: Ensures buffers are sufficient for the operation
- **Group state**: Validates group state before adding operations
- **Cancellation support**: Operations can be cancelled if needed

## Thread Safety

The execution plan-aware group management maintains the same thread safety guarantees as the base group management system, using thread-local storage for group state and execution plan information.

## Testing

Comprehensive unit tests are provided to verify:

- Operation creation and validation
- Group management with execution plans
- Chunk size extraction and calculation
- Error handling and edge cases
- Integration with base group management functionality

This extension enables MSCCLPP to support sophisticated all_to_allv operations with variable chunk sizes while maintaining the familiar group management API patterns.