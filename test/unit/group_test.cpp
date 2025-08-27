// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <gtest/gtest.h>
#include <mscclpp/group.hpp>
#include <mscclpp/core.hpp>
#include <memory>

using namespace mscclpp;

class GroupTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a proper bootstrap for testing
    try {
      bootstrap_ = std::make_shared<TcpBootstrap>(0, 1);
      bootstrap_->initialize(bootstrap_->createUniqueId());
      comm_ = std::make_shared<Communicator>(bootstrap_);
    } catch (const std::exception& e) {
      // If we can't create a real communicator, set to nullptr
      comm_ = nullptr;
      std::cout << "Warning: Could not create test communicator: " << e.what() << std::endl;
    }
  }

  void TearDown() override {
    GroupManager::cleanupGroup();
  }

  std::shared_ptr<TcpBootstrap> bootstrap_;
  std::shared_ptr<Communicator> comm_;
};

TEST_F(GroupTest, GroupStartAndEnd) {
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

TEST_F(GroupTest, NestedGroups) {
  EXPECT_EQ(GroupManager::groupStart(), GroupResult::Success);
  EXPECT_EQ(GroupManager::getGroupDepth(), 1);
  
  EXPECT_EQ(GroupManager::groupStart(), GroupResult::Success);
  EXPECT_EQ(GroupManager::getGroupDepth(), 2);
  
  EXPECT_EQ(GroupManager::groupEnd(), GroupResult::Success);
  EXPECT_EQ(GroupManager::getGroupDepth(), 1);
  
  EXPECT_EQ(GroupManager::groupEnd(), GroupResult::Success);
  EXPECT_EQ(GroupManager::getGroupDepth(), 0);
}

TEST_F(GroupTest, AddConnectOperation) {
  if (!comm_) {
    GTEST_SKIP() << "Communicator not available for testing";
  }

  EXPECT_EQ(GroupManager::groupStart(), GroupResult::Success);
  
  EndpointConfig config{};
  auto operation = GroupManager::addConnect(comm_, config, 1, 0);
  
  EXPECT_NE(operation, nullptr);
  EXPECT_EQ(operation->getType(), OperationType::Connect);
  EXPECT_EQ(operation->getTag(), 0);
  
  EXPECT_EQ(GroupManager::groupEnd(), GroupResult::Success);
}

TEST_F(GroupTest, AddSendMemoryOperation) {
  if (!comm_) {
    GTEST_SKIP() << "Communicator not available for testing";
  }

  EXPECT_EQ(GroupManager::groupStart(), GroupResult::Success);
  
  // Create a buffer and register it properly using the communicator
  void* buffer = malloc(1024);
  auto registeredMemory = comm_->registerMemory(buffer, 1024, mscclpp::NoTransports);
  auto memory = std::make_shared<RegisteredMemory>(std::move(registeredMemory));
  
  auto operation = GroupManager::addSendMemory(comm_, memory, 1, 0);
  
  EXPECT_NE(operation, nullptr);
  EXPECT_EQ(operation->getType(), OperationType::SendMemory);
  EXPECT_EQ(operation->getTag(), 0);
  
  EXPECT_EQ(GroupManager::groupEnd(), GroupResult::Success);
  
  free(buffer);  // Clean up
}

TEST_F(GroupTest, AddRecvMemoryOperation) {
  if (!comm_) {
    GTEST_SKIP() << "Communicator not available for testing";
  }

  EXPECT_EQ(GroupManager::groupStart(), GroupResult::Success);
  
  auto operation = GroupManager::addRecvMemory(comm_, 1, 0);
  
  EXPECT_NE(operation, nullptr);
  EXPECT_EQ(operation->getType(), OperationType::RecvMemory);
  EXPECT_EQ(operation->getTag(), 0);
  
  EXPECT_EQ(GroupManager::groupEnd(), GroupResult::Success);
}

TEST_F(GroupTest, AddCustomOperation) {
  if (!comm_) {
    GTEST_SKIP() << "Communicator not available for testing";
  }

  EXPECT_EQ(GroupManager::groupStart(), GroupResult::Success);
  
  bool executed = false;
  bool completed = false;
  
  auto customOp = GroupManager::addCustom(
    comm_,
    [&executed]() -> GroupResult {
      executed = true;
      return GroupResult::Success;
    },
    [&completed]() -> bool {
      return completed = true;
    }
  );
  
  EXPECT_NE(customOp, nullptr);
  EXPECT_EQ(customOp->getType(), OperationType::Custom);
  
  auto result = GroupManager::groupEnd();
  EXPECT_EQ(result, GroupResult::Success);
  EXPECT_TRUE(customOp->isComplete());
}

TEST_F(GroupTest, GroupResultToErrorCode) {
    // Since ErrorCode doesn't have Success, we test the actual mappings
    EXPECT_EQ(groupResultToErrorCode(GroupResult::InvalidUsage), ErrorCode::InvalidUsage);
    EXPECT_EQ(groupResultToErrorCode(GroupResult::InternalError), ErrorCode::InternalError);
    EXPECT_EQ(groupResultToErrorCode(GroupResult::Timeout), ErrorCode::Timeout);
    
    // For success, the function should handle it gracefully (implementation dependent)
    // We just test that it doesn't crash when called with Success
    EXPECT_NO_THROW(groupResultToErrorCode(GroupResult::Success));
}

TEST_F(GroupTest, ErrorHandling) {
  if (!comm_) {
    GTEST_SKIP() << "Communicator not available for testing";
  }

  // Test adding operation without starting group
  EXPECT_THROW({
    GroupManager::addConnect(comm_, EndpointConfig{}, 1, 0);
  }, std::runtime_error);
}

TEST_F(GroupTest, ConvenienceFunctions) {
  if (!comm_) {
    GTEST_SKIP() << "Communicator not available for testing";
  }

  std::vector<std::tuple<std::shared_ptr<Communicator>, EndpointConfig, int, int>> connections = {
    {comm_, EndpointConfig{}, 1, 0}
  };
  
  // This should not throw (may not execute fully due to test environment)
  EXPECT_NO_THROW({
    auto futures = groupConnect(connections, false);  // Use non-blocking to avoid hanging
  });
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}