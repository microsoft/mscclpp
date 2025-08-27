// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <iostream>
#include <mscclpp/core.hpp>
#include <mscclpp/group.hpp>
#include <vector>

using namespace mscclpp;

int main() {
    try {
        std::cout << "=== MSCCLPP Group Management Example ===" << std::endl;
        
        // Example 1: Using GroupScope (RAII style)
        std::cout << "\nExample 1: RAII Group Management" << std::endl;
        {
            GroupScope group(true);  // blocking group
            if (group.isValid()) {
                std::cout << "  Group started successfully" << std::endl;
                std::cout << "  Current group depth: " << GroupManager::getGroupDepth() << std::endl;
                
                // Add a custom operation for demonstration
                auto customOp = GroupManager::addCustom(
                    nullptr,  // No communicator needed for demo
                    []() -> GroupResult {
                        std::cout << "    Executing custom operation..." << std::endl;
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        return GroupResult::Success;
                    },
                    []() -> bool {
                        return true;  // Always complete immediately for demo
                    }
                );
                
                if (customOp) {
                    std::cout << "  Added custom operation to group" << std::endl;
                } else {
                    std::cout << "  Failed to add custom operation" << std::endl;
                }
                
                // Group will be executed when GroupScope destructor is called
            } else {
                std::cout << "  Failed to start group" << std::endl;
            }
        }
        std::cout << "  Group completed" << std::endl;
        
        // Example 2: Manual group management
        std::cout << "\nExample 2: Manual Group Management" << std::endl;
        auto result = GroupManager::groupStart();
        if (result == GroupResult::Success) {
            std::cout << "  Group started manually" << std::endl;
            
            // Add multiple operations
            for (int i = 0; i < 3; ++i) {
                auto op = GroupManager::addCustom(
                    nullptr,
                    [i]() -> GroupResult {
                        std::cout << "    Executing operation " << i+1 << std::endl;
                        return GroupResult::Success;
                    },
                    []() -> bool { return true; }
                );
                
                if (op) {
                    std::cout << "  Added operation " << i+1 << " to group" << std::endl;
                }
            }
            
            // Execute blocking
            result = GroupManager::groupEnd(true);
            if (result == GroupResult::Success) {
                std::cout << "  Group executed successfully" << std::endl;
            } else {
                std::cout << "  Group execution failed with result: " << static_cast<int>(result) << std::endl;
            }
        } else {
            std::cout << "  Failed to start group manually" << std::endl;
        }
        
        // Example 3: Nested groups
        std::cout << "\nExample 3: Nested Groups" << std::endl;
        {
            GroupScope outerGroup(true);
            if (outerGroup.isValid()) {
                std::cout << "  In outer group (depth: " << GroupManager::getGroupDepth() << ")" << std::endl;
                
                auto outerOp = GroupManager::addCustom(
                    nullptr,
                    []() -> GroupResult {
                        std::cout << "    Outer group operation" << std::endl;
                        return GroupResult::Success;
                    },
                    []() -> bool { return true; }
                );
                
                {
                    GroupScope innerGroup(true);
                    if (innerGroup.isValid()) {
                        std::cout << "  In inner group (depth: " << GroupManager::getGroupDepth() << ")" << std::endl;
                        
                        auto innerOp = GroupManager::addCustom(
                            nullptr,
                            []() -> GroupResult {
                                std::cout << "    Inner group operation" << std::endl;
                                return GroupResult::Success;
                            },
                            []() -> bool { return true; }
                        );
                        
                        std::cout << "  Added operation to inner group" << std::endl;
                    }
                }
                std::cout << "  Inner group completed, back to outer group (depth: " << GroupManager::getGroupDepth() << ")" << std::endl;
                
                std::cout << "  Added operation to outer group" << std::endl;
            }
        }
        std::cout << "  All nested groups completed" << std::endl;
        
        // Example 4: Error handling
        std::cout << "\nExample 4: Error Handling" << std::endl;
        try {
            GroupScope group(true);
            if (group.isValid()) {
                // Add an operation that will fail
                auto failingOp = GroupManager::addCustom(
                    nullptr,
                    []() -> GroupResult {
                        std::cout << "    Operation that fails..." << std::endl;
                        return GroupResult::InternalError;
                    },
                    []() -> bool { return true; }
                );
                
                if (failingOp) {
                    std::cout << "  Added failing operation to group" << std::endl;
                }
            }
        } catch (const Error& e) {
            std::cout << "  Caught MSCCLPP error: " << e.what() << std::endl;
        }
        std::cout << "  Error handling example completed" << std::endl;
        
    } catch (const Error& e) {
        std::cerr << "MSCCLPP Error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Standard Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n=== All examples completed successfully! ===" << std::endl;
    std::cout << "\nGroup Management Benefits:" << std::endl;
    std::cout << "✓ Reduced synchronization overhead" << std::endl;
    std::cout << "✓ Improved kernel fusion opportunities" << std::endl;
    std::cout << "✓ Better resource utilization" << std::endl;
    std::cout << "✓ Simplified error handling" << std::endl;
    std::cout << "✓ Support for nested groups" << std::endl;
    
    return 0;
}