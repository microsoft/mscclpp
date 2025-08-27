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