// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <gtest/gtest.h>
#include <mscclpp/core.hpp>
#include <mscclpp/group.hpp>
#include <thread>
#include <chrono>

using namespace mscclpp;

class GroupTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset group state before each test
        GroupManager::cleanupGroup();
    }
    
    void TearDown() override {
        // Cleanup after each test
        GroupManager::cleanupGroup();
    }
};

TEST_F(GroupTest, BasicGroupStartEnd) {
    EXPECT_FALSE(GroupManager::inGroup());
    EXPECT_EQ(GroupManager::getGroupDepth(), 0);
    
    auto result = GroupManager::groupStart();
    EXPECT_EQ(result, GroupResult::Success);
    EXPECT_TRUE(GroupManager::inGroup());
    EXPECT_EQ(GroupManager::getGroupDepth(), 1);
    
    result = GroupManager::groupEnd();
    EXPECT_EQ(result, GroupResult::Success);
    EXPECT_FALSE(GroupManager::inGroup());
    EXPECT_EQ(GroupManager::getGroupDepth(), 0);
}

TEST_F(GroupTest, NestedGroups) {
    EXPECT_EQ(GroupManager::getGroupDepth(), 0);
    
    auto result = GroupManager::groupStart();
    EXPECT_EQ(result, GroupResult::Success);
    EXPECT_EQ(GroupManager::getGroupDepth(), 1);
    
    result = GroupManager::groupStart();
    EXPECT_EQ(result, GroupResult::Success);
    EXPECT_EQ(GroupManager::getGroupDepth(), 2);
    
    result = GroupManager::groupEnd();
    EXPECT_EQ(result, GroupResult::Success);
    EXPECT_EQ(GroupManager::getGroupDepth(), 1);
    
    result = GroupManager::groupEnd();
    EXPECT_EQ(result, GroupResult::Success);
    EXPECT_EQ(GroupManager::getGroupDepth(), 0);
}

TEST_F(GroupTest, GroupEndWithoutStart) {
    auto result = GroupManager::groupEnd();
    EXPECT_EQ(result, GroupResult::InvalidUsage);
}

TEST_F(GroupTest, GroupScope) {
    EXPECT_FALSE(GroupManager::inGroup());
    
    {
        GroupScope scope(true);
        EXPECT_TRUE(scope.isValid());
        EXPECT_TRUE(GroupManager::inGroup());
        EXPECT_EQ(GroupManager::getGroupDepth(), 1);
    }
    
    EXPECT_FALSE(GroupManager::inGroup());
    EXPECT_EQ(GroupManager::getGroupDepth(), 0);
}

TEST_F(GroupTest, CustomOperation) {
    bool executed = false;
    bool completed = false;
    
    {
        GroupScope scope(true);
        ASSERT_TRUE(scope.isValid());
        
        auto customOp = GroupManager::addCustom(
            nullptr,  // No communicator needed for this test
            [&executed]() -> GroupResult {
                executed = true;
                return GroupResult::Success;
            },
            [&completed]() -> bool {
                return completed;
            }
        );
        
        EXPECT_NE(customOp, nullptr);
        
        // Simulate completion
        completed = true;
    }
    
    EXPECT_TRUE(executed);
}

TEST_F(GroupTest, OperationTypes) {
    // Test that we can create operations of different types
    
    CustomOperation::ExecuteFunction execFunc = []() { return GroupResult::Success; };
    CustomOperation::IsCompleteFunction completeFunc = []() { return true; };
    
    CustomOperation customOp(nullptr, execFunc, completeFunc);
    EXPECT_EQ(customOp.getType(), OperationType::Custom);
    EXPECT_EQ(customOp.getCommunicator(), nullptr);
    EXPECT_EQ(customOp.getTag(), 0);
    
    // Test execution
    auto result = customOp.execute();
    EXPECT_EQ(result, GroupResult::Success);
    EXPECT_TRUE(customOp.isComplete());
}

TEST_F(GroupTest, GroupResultToErrorCode) {
    EXPECT_EQ(groupResultToErrorCode(GroupResult::Success), ErrorCode::Success);
    EXPECT_EQ(groupResultToErrorCode(GroupResult::InvalidUsage), ErrorCode::InvalidUsage);
    EXPECT_EQ(groupResultToErrorCode(GroupResult::InternalError), ErrorCode::InternalError);
    EXPECT_EQ(groupResultToErrorCode(GroupResult::Timeout), ErrorCode::Timeout);
}

TEST_F(GroupTest, MultipleOperationsInGroup) {
    int execCount = 0;
    
    {
        GroupScope scope(true);
        ASSERT_TRUE(scope.isValid());
        
        // Add multiple custom operations
        for (int i = 0; i < 5; ++i) {
            auto customOp = GroupManager::addCustom(
                nullptr,
                [&execCount]() -> GroupResult {
                    execCount++;
                    return GroupResult::Success;
                },
                []() -> bool {
                    return true;
                }
            );
            EXPECT_NE(customOp, nullptr);
        }
    }
    
    EXPECT_EQ(execCount, 5);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}