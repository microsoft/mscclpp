// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <mscclpp/core.hpp>
#include <mscclpp/port_channel.hpp>

class LocalCommunicatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    bootstrap = std::make_shared<mscclpp::TcpBootstrap>(0, 1);
    bootstrap->initialize(bootstrap->createUniqueId());
    comm = std::make_shared<mscclpp::Communicator>(bootstrap);
    proxyService = std::make_shared<mscclpp::ProxyService>();
  }

  std::shared_ptr<mscclpp::TcpBootstrap> bootstrap;
  std::shared_ptr<mscclpp::Communicator> comm;
  std::shared_ptr<mscclpp::ProxyService> proxyService;
};

TEST_F(LocalCommunicatorTest, RegisterMemory) {
  int dummy[42];
  auto memory = comm->registerMemory(&dummy, sizeof(dummy), mscclpp::NoTransports);
  EXPECT_EQ(memory.data(), &dummy);
  EXPECT_EQ(memory.size(), sizeof(dummy));
  EXPECT_EQ(memory.transports(), mscclpp::NoTransports);
}

TEST_F(LocalCommunicatorTest, SendMemoryToSelf) {
  int dummy[42];
  auto memory = comm->registerMemory(&dummy, sizeof(dummy), mscclpp::NoTransports);
  comm->sendMemory(memory, 0, 0);
  auto memoryFuture = comm->recvMemory(0, 0);
  auto sameMemory = memoryFuture.get();
  EXPECT_EQ(sameMemory.data(), memory.data());
  EXPECT_EQ(sameMemory.size(), memory.size());
  EXPECT_EQ(sameMemory.transports(), memory.transports());
}

TEST_F(LocalCommunicatorTest, ProxyServiceAddRemoveMemory) {
  auto memory = mscclpp::RegisteredMemory();
  auto memoryId = proxyService->addMemory(memory);
  EXPECT_EQ(memoryId, 0);
  proxyService->removeMemory(memoryId);
  memoryId = proxyService->addMemory(memory);
  EXPECT_EQ(memoryId, 0);
}
