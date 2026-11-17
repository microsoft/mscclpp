// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <mscclpp/errors.hpp>
#include <thread>

#include "connection.hpp"
#include "mp_unit_tests.hpp"
#include "mrc.hpp"

class MrcPeerToPeerTest : public MultiProcessTest {
 protected:
  void SetUp() override {
    const char* device = std::getenv("MSCCLPP_TEST_MRC_DEVICE");
    if (device == nullptr || gEnv->worldSize != 2) {
      SKIP_TEST() << "Requires two ranks and MSCCLPP_TEST_MRC_DEVICE naming an MRC-capable NIC.";
    }
    context = std::make_unique<mscclpp::MrcCtx>(device);
    buffer.resize(4096);
    mr = context->registerMr(buffer.data(), buffer.size());
    qp = context->createQp(1, std::stoi(gEnv->args["ib_gid_index"]), 0, 8, 8, 8);
    auto info = qp->getInfo();
    std::array<mscclpp::MrcQpInfo, 2> peers;
    MPI_Allgather(&info, sizeof(info), MPI_BYTE, peers.data(), sizeof(info), MPI_BYTE, MPI_COMM_WORLD);
    qp->rtr(peers[1 - gEnv->rank]);
    qp->rts();
    auto localMr = mr->getInfo();
    std::array<mscclpp::MrcMrInfo, 2> memories;
    MPI_Allgather(&localMr, sizeof(localMr), MPI_BYTE, memories.data(), sizeof(localMr), MPI_BYTE, MPI_COMM_WORLD);
    remoteMr = memories[1 - gEnv->rank];
  }

  template <typename F>
  void expectInvalid(F function) {
    bool caught = false;
    try {
      function();
    } catch (const mscclpp::Error& error) {
      EXPECT_TRUE(error.getErrorCode() == mscclpp::ErrorCode::InvalidUsage);
      caught = true;
    }
    EXPECT_TRUE(caught);
  }

  void drain(int expectedReceives, int expectedSends, const std::vector<unsigned int>& immediates) {
    int received = 0;
    int sent = 0;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
    while (received < expectedReceives || sent < expectedSends) {
      int count = qp->pollRecvCq();
      ASSERT_GE(count, 0);
      for (int i = 0; i < count; ++i) {
        ASSERT_EQ(qp->getRecvWcStatus(i), 0);
        ASSERT_LT(received, expectedReceives);
        EXPECT_EQ(qp->getRecvWcImmData(i), immediates[received++]);
      }
      count = qp->pollSendCq();
      ASSERT_GE(count, 0);
      for (int i = 0; i < count; ++i) {
        ASSERT_EQ(qp->getSendWcStatus(i), 0);
      }
      sent += count;
      ASSERT_TRUE(std::chrono::steady_clock::now() < deadline);
      std::this_thread::yield();
    }
    EXPECT_EQ(received, expectedReceives);
    EXPECT_EQ(sent, expectedSends);
    EXPECT_EQ(qp->getNumSendCqItems(), 0);
  }

  void testImmediateNotifications(bool precedingWrite);

  std::unique_ptr<mscclpp::MrcCtx> context;
  std::vector<unsigned char> buffer;
  std::unique_ptr<const mscclpp::MrcMr> mr;
  std::shared_ptr<mscclpp::MrcQp> qp;
  mscclpp::MrcMrInfo remoteMr;
};

void MrcPeerToPeerTest::testImmediateNotifications(bool precedingWrite) {
  const unsigned int tokens[] = {0, 0x01020304, 0xffffffff, 0x80000000};
  expectInvalid([&] { qp->stageSendWriteWithImm(nullptr, remoteMr, 1, 0, 0, 0, true, 0); });
  expectInvalid([&] { qp->getRecvWcStatus(0); });
  expectInvalid([&] { qp->getRecvWcImmData(0); });
  qp->postRecv();
  for (int round = 0; round < 8; ++round) {
    std::fill(buffer.begin(), buffer.end(), 0x5a);
    for (int i = 0; i < 4; ++i) {
      std::fill_n(buffer.begin() + 13 + i * 64, 17, static_cast<unsigned char>(gEnv->rank * 31 + round + i));
      qp->stageRecv(round * 4 + i);
    }
    std::vector<unsigned char> expected = buffer;
    qp->postRecv();
    MPI_Barrier(MPI_COMM_WORLD);
    std::vector<unsigned int> immediates;
    for (int i = 0; i < 4; ++i) {
      unsigned int value = tokens[i] ^ static_cast<unsigned int>(round << 8);
      if (precedingWrite) {
        qp->stageSendWrite(mr.get(), remoteMr, 17, i * 2, 13 + i * 64, 1024 + i * 64, false);
        qp->stageSendWriteWithImm(nullptr, remoteMr, 0, i * 2 + 1, 0, 1024 + i * 64, i == 3, value);
      } else {
        qp->stageSendWriteWithImm(mr.get(), remoteMr, 17, i, 13 + i * 64, 1024 + i * 64, true, value);
      }
      immediates.push_back(value);
    }
    qp->postSend();
    drain(4, precedingWrite ? 1 : 4, immediates);
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < 4; ++i) {
      std::fill_n(expected.begin() + 1024 + i * 64, 17, static_cast<unsigned char>((1 - gEnv->rank) * 31 + round + i));
    }
    EXPECT_TRUE(buffer == expected);
  }
}

TEST(MrcPeerToPeerTest, WriteWithImmediatePayload) { testImmediateNotifications(false); }

TEST(MrcPeerToPeerTest, WriteThenZeroByteNotification) { testImmediateNotifications(true); }

TEST(MrcPeerToPeerTest, ReceiveCapacityAndReplenishment) {
  for (int round = 0; round < 3; ++round) {
    for (int i = 0; i < 8; ++i) qp->stageRecv(round * 8 + i);
    expectInvalid([&] { qp->stageRecv(100); });
    qp->postRecv();
    expectInvalid([&] { qp->stageRecv(100); });
    MPI_Barrier(MPI_COMM_WORLD);
    std::vector<unsigned int> immediates;
    for (int i = 0; i < 8; ++i) {
      unsigned int value = 0x01020304 + round * 8 + i;
      qp->stageSendWriteWithImm(nullptr, remoteMr, 0, i, 0, 0, true, value);
      immediates.push_back(value);
    }

    qp->postSend();
    drain(8, 8, immediates);
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

struct MrcMetadataAccess : mscclpp::BaseConnection {
  using BaseConnection::getImpl;
};

TEST(MultiProcessTest, MrcEndpointAndMemoryExchange) {
  const char* device = std::getenv("MSCCLPP_TEST_MRC_DEVICE");
  if (device == nullptr || gEnv->worldSize != 2) {
    SKIP_TEST() << "Requires two ranks and MSCCLPP_TEST_MRC_DEVICE naming an MRC-capable NIC.";
  }
  auto transport = mscclpp::Transport::Unknown;
  for (auto candidate :
       {mscclpp::Transport::MRC0, mscclpp::Transport::MRC1, mscclpp::Transport::MRC2, mscclpp::Transport::MRC3}) {
    if (mscclpp::getMrcDeviceName(candidate) == device) {
      transport = candidate;
      break;
    }
  }
  ASSERT_TRUE(transport != mscclpp::Transport::Unknown);
  auto context = mscclpp::Context::create();
  std::vector<unsigned char> buffer(4096, 0x5a);
  std::fill_n(buffer.begin(), 64, static_cast<unsigned char>(gEnv->rank + 17));
  auto memory = context->registerMemory(buffer.data(), buffer.size(), transport);
  auto endpoint = context->createEndpoint(
      {transport, mscclpp::DeviceType::CPU, -1, {}, {1, std::stoi(gEnv->args["ib_gid_index"]), 0, 8, 8, 8}});
  auto exchange = [](const std::vector<char>& local) {
    std::vector<char> peers(local.size() * 2);
    MPI_Allgather(local.data(), local.size(), MPI_BYTE, peers.data(), local.size(), MPI_BYTE, MPI_COMM_WORLD);
    auto begin = peers.begin() + (1 - gEnv->rank) * local.size();
    return std::vector<char>(begin, begin + local.size());
  };
  auto peerEndpoint = mscclpp::Endpoint::deserialize(exchange(endpoint.serialize()));
  auto peerMemory = mscclpp::RegisteredMemory::deserialize(exchange(memory.serialize()));
  auto& local = MrcMetadataAccess::getImpl(memory).getTransportInfo(transport);
  auto& remote = MrcMetadataAccess::getImpl(peerMemory).getTransportInfo(peerEndpoint.transport());
  EXPECT_TRUE(local.rdmaMemory->isLocal());
  EXPECT_FALSE(remote.rdmaMemory->isLocal());
  const auto& localMemory = dynamic_cast<const mscclpp::MrcMemory&>(*local.rdmaMemory);
  const auto& remoteMemory = dynamic_cast<const mscclpp::MrcMemory&>(*remote.rdmaMemory);
  EXPECT_TRUE(remoteMemory.mr() == nullptr);
  auto& contextImpl = MrcMetadataAccess::getImpl(*context);
  EXPECT_EQ(contextImpl.rdmaProviders_.size(), size_t{1});
  EXPECT_TRUE(contextImpl.getRdmaProvider(transport) == contextImpl.getRdmaProvider(transport));
  std::weak_ptr<mscclpp::RdmaProvider> providerOwner = contextImpl.rdmaProviders_.at(transport);
  auto localEndpoint = MrcMetadataAccess::getImpl(endpoint).rdmaEndpoint_;
  auto remoteEndpoint = MrcMetadataAccess::getImpl(peerEndpoint).rdmaEndpoint_;
  EXPECT_TRUE(localEndpoint->isLocal());
  EXPECT_FALSE(remoteEndpoint->isLocal());
  auto& nativeEndpoint = dynamic_cast<mscclpp::MrcEndpoint&>(*localEndpoint);
  nativeEndpoint.connect(*remoteEndpoint);
  auto qp = nativeEndpoint.qp();
  context.reset();
  EXPECT_FALSE(providerOwner.expired());
  qp->stageRecv(0);
  qp->postRecv();
  MPI_Barrier(MPI_COMM_WORLD);
  qp->stageSendWriteWithImm(localMemory.mr(), remoteMemory.info(), 64, 0, 0, 1024, true, 0x01020304 + gEnv->rank);
  qp->postSend();
  bool received = false;
  bool sent = false;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
  while (!received || !sent) {
    int count = qp->pollRecvCq();
    ASSERT_GE(count, 0);
    for (int i = 0; i < count; ++i) {
      ASSERT_EQ(qp->getRecvWcStatus(i), 0);
      EXPECT_EQ(qp->getRecvWcImmData(i), static_cast<unsigned int>(0x01020304 + 1 - gEnv->rank));
      received = true;
    }
    count = qp->pollSendCq();
    ASSERT_GE(count, 0);
    for (int i = 0; i < count; ++i) {
      ASSERT_EQ(qp->getSendWcStatus(i), 0);
      sent = true;
    }
    ASSERT_TRUE(std::chrono::steady_clock::now() < deadline);
    std::this_thread::yield();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  std::vector<unsigned char> expected(4096, 0x5a);
  std::fill_n(expected.begin(), 64, static_cast<unsigned char>(gEnv->rank + 17));
  std::fill_n(expected.begin() + 1024, 64, static_cast<unsigned char>(1 - gEnv->rank + 17));
  EXPECT_TRUE(buffer == expected);
}
