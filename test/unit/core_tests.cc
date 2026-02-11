// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/core.hpp>

#include "../framework.hpp"

class LocalCommunicatorTest : public ::mscclpp::test::TestCase {
 protected:
  void SetUp() override {
    bootstrap = std::make_shared<mscclpp::TcpBootstrap>(0, 1);
    bootstrap->initialize(bootstrap->createUniqueId());
    comm = std::make_shared<mscclpp::Communicator>(bootstrap);
  }

  std::shared_ptr<mscclpp::TcpBootstrap> bootstrap;
  std::shared_ptr<mscclpp::Communicator> comm;
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
  comm->sendMemory(memory, 0);
  auto memoryFuture = comm->recvMemory(0);
  auto sameMemory = memoryFuture.get();
  EXPECT_EQ(sameMemory.data(), memory.data());
  EXPECT_EQ(sameMemory.size(), memory.size());
  EXPECT_EQ(sameMemory.transports(), memory.transports());
}
