// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <mscclpp/core.hpp>

#include <sstream>

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

TEST(TransportFlagsTest, OutputStream) {
  auto toString = [](const mscclpp::TransportFlags& transportFlags) {
    std::ostringstream os;
    os << transportFlags;
    return os.str();
  };

  EXPECT_EQ(toString(mscclpp::NoTransports), std::string("{}"));
  EXPECT_EQ(toString(mscclpp::Transport::CudaIpc), std::string("{IPC}"));
  EXPECT_EQ(toString(mscclpp::Transport::CudaIpc | mscclpp::Transport::IB0 | mscclpp::Transport::Ethernet),
            std::string("{IPC, IB0, ETH}"));
}

TEST(LocalCommunicatorTest, RegisterMemory) {
  int dummy[42];
  auto memory = comm->registerMemory(&dummy, sizeof(dummy), mscclpp::NoTransports);
  EXPECT_EQ(memory.data(), &dummy);
  EXPECT_EQ(memory.size(), sizeof(dummy));
  EXPECT_EQ(memory.transports(), mscclpp::NoTransports);
}

TEST(LocalCommunicatorTest, SendMemoryToSelf) {
  int dummy[42];
  auto memory = comm->registerMemory(&dummy, sizeof(dummy), mscclpp::NoTransports);
  comm->sendMemory(memory, 0);
  auto memoryFuture = comm->recvMemory(0);
  auto sameMemory = memoryFuture.get();
  EXPECT_EQ(sameMemory.data(), memory.data());
  EXPECT_EQ(sameMemory.size(), memory.size());
  EXPECT_EQ(sameMemory.transports(), memory.transports());
}
