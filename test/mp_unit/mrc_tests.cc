// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <limits>
#include <mscclpp/errors.hpp>
#include <thread>

#include "connection.hpp"
#include "mp_unit_tests.hpp"

namespace {

template <typename T>
T exchangeWithPeer(const T& local) {
  std::array<T, 2> peers;
  MPI_Allgather(&local, sizeof(T), MPI_BYTE, peers.data(), sizeof(T), MPI_BYTE, MPI_COMM_WORLD);
  return peers[1 - gEnv->rank];
}

std::vector<char> exchangeWithPeer(const std::vector<char>& local) {
  std::vector<char> peers(local.size() * 2);
  MPI_Allgather(local.data(), local.size(), MPI_BYTE, peers.data(), local.size(), MPI_BYTE, MPI_COMM_WORLD);
  auto begin = peers.begin() + (1 - gEnv->rank) * local.size();
  return {begin, begin + local.size()};
}

mscclpp::Transport mrcTransportForDevice(const char* device) {
  for (auto candidate :
       {mscclpp::Transport::MRC0, mscclpp::Transport::MRC1, mscclpp::Transport::MRC2, mscclpp::Transport::MRC3}) {
    if (mscclpp::getMrcDeviceName(candidate) == device) return candidate;
  }
  return mscclpp::Transport::Unknown;
}

template <typename F>
bool rejectsInvalidUsage(F function) {
  try {
    function();
  } catch (const mscclpp::Error& error) {
    return error.getErrorCode() == mscclpp::ErrorCode::InvalidUsage;
  }
  return false;
}

}  // namespace

void MrcTestBase::SetUp() {
  if (gEnv->worldSize != 2) SKIP_TEST() << "Requires two ranks.";
  mrcDevice = std::getenv("MSCCLPP_TEST_MRC_DEVICE");
  int configured = mrcDevice != nullptr && mrcDevice[0] != '\0';
  int peerConfigured = exchangeWithPeer(configured);
  if (!configured && !peerConfigured) SKIP_TEST() << "Requires MSCCLPP_TEST_MRC_DEVICE naming an MRC-capable NIC.";
  if (!configured || !peerConfigured) FAIL() << "MSCCLPP_TEST_MRC_DEVICE must be configured on both ranks.";
}

void MrcTestBase::runOnBothRanks(const char* phase, const std::function<void()>& run) {
  // Agree before either rank throws; otherwise the other rank could enter the next collective alone.
  std::exception_ptr error;
  try {
    run();
  } catch (...) {
    error = std::current_exception();
  }
  int peerFailed = exchangeWithPeer(static_cast<int>(error != nullptr));
  if (error) std::rethrow_exception(error);
  if (peerFailed) FAIL() << phase << " failed on the peer.";
}

mscclpp::MrcBuffer MrcTestBase::connectPeer(mscclpp::MrcQp& qp, const mscclpp::MrcMr& destination) {
  auto peer = exchangeWithPeer(qp.getInfo());
  runOnBothRanks("native QP connection", [&] {
    qp.rtr(peer);
    qp.rts();
  });
  return exchangeWithPeer(destination.getBuffer());
}

class MrcPeerToPeerTest : public MrcTestBase {
 protected:
  void SetUp() override { setUpWithSendDepth(queueDepth); }

  void setUpWithSendDepth(int sendDepth) {
    MrcTestBase::SetUp();
    runOnBothRanks("native resource creation", [&] {
      context = std::make_unique<mscclpp::MrcCtx>(mrcDevice);
      buffer.resize(4096);
      mr = context->registerMr(buffer.data(), buffer.size());
      qp = context->createQp(1, std::stoi(gEnv->args["ib_gid_index"]), 0, queueDepth, sendDepth, queueDepth);
    });
    remoteMr = connectPeer(*qp, *mr);
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
  mscclpp::MrcBuffer remoteMr;
};

void MrcPeerToPeerTest::testImmediateNotifications(bool precedingWrite) {
  const unsigned int tokens[] = {0, 0x01020304, 0xffffffff, 0x80000000};
  EXPECT_TRUE(rejectsInvalidUsage([&] { qp->stageSendWriteWithImm(nullptr, remoteMr, 1, 0, 0, 0, true, 0); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { qp->getRecvWcStatus(0); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { qp->getRecvWcImmData(0); }));
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

class MrcSendQueueTest : public MrcPeerToPeerTest {
 protected:
  void SetUp() override { setUpWithSendDepth(2); }  // Leave room in the CQ when the SQ is full.
};

TEST(MrcSendQueueTest, StagedAndPostedWritesShareCapacity) {
  runOnBothRanks("exercise native send queue capacity", [&] {
    auto stage = [&](int i) { qp->stageSendWrite(mr.get(), remoteMr, 1, i, i, 1024 + i, true); };
    stage(0);
    qp->postSend();
    stage(1);
    ASSERT_TRUE(rejectsInvalidUsage([&] { stage(2); }));

    drain(0, 1, {});  // Only the first write was posted; the second still occupies a staging slot.
    stage(2);
    ASSERT_TRUE(rejectsInvalidUsage([&] { stage(3); }));
    qp->postSend();
    ASSERT_TRUE(rejectsInvalidUsage([&] { stage(3); }));
    drain(0, 2, {});
  });
}

TEST(MrcPeerToPeerTest, CheckedSendDrainAllowsSourceReuse) {
  runOnBothRanks("post receives for checked send drain", [&] {
    qp->stageRecv(0);
    qp->stageRecv(1);
    qp->postRecv();
  });
  runOnBothRanks("drain both native sends before source reuse", [&] {
    for (int i = 0; i < 2; ++i) {
      buffer[i] = static_cast<unsigned char>(0x40 + 2 * gEnv->rank + i);
      qp->stageSendWriteWithImm(mr.get(), remoteMr, 1, i, i, 1024 + i, true, i);
      ASSERT_FALSE(qp->progressSend());  // Staging is not submission.
      qp->postSend();
    }
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
    while (!qp->progressSend()) {
      ASSERT_TRUE(std::chrono::steady_clock::now() < deadline);
      std::this_thread::yield();
    }
    std::fill_n(buffer.begin(), 2, 0xff);
    ASSERT_TRUE(qp->progressSend());
    drain(2, 0, {0, 1});
    for (int i = 0; i < 2; ++i) {
      EXPECT_EQ(buffer[1024 + i], static_cast<unsigned char>(0x40 + 2 * (1 - gEnv->rank) + i));
    }
  });
}

TEST(MrcPeerToPeerTest, InvalidWritesPreserveStagedBatch) {
  std::unique_ptr<const mscclpp::MrcMr> source;
  std::vector<unsigned int> tokens;
  runOnBothRanks("stage checked native writes", [&] {
    // The logical source is shorter than its page-aligned registration.
    source = context->registerMr(buffer.data() + 1, 2);
    mscclpp::MrcCtx other(mrcDevice);
    auto foreign = other.registerMr(buffer.data(), buffer.size());
    buffer[2] = static_cast<unsigned char>(0x40 + gEnv->rank);
    qp->stageSendWrite(source.get(), remoteMr, 1, 0, 1, remoteMr.size - 1, true);
    ASSERT_TRUE(rejectsInvalidUsage([&] { qp->stageSendWrite(foreign.get(), remoteMr, 1, 0, 0, 0, true); }));
    ASSERT_TRUE(rejectsInvalidUsage([&] { qp->stageSendWrite(source.get(), remoteMr, 2, 0, 1, 0, true); }));
    ASSERT_TRUE(
        rejectsInvalidUsage([&] { qp->stageSendWrite(source.get(), remoteMr, 2, 0, 0, remoteMr.size - 1, true); }));
    ASSERT_TRUE(
        rejectsInvalidUsage([&] { qp->stageSendWrite(source.get(), remoteMr, uint64_t{1} << 32, 0, 0, 0, true); }));
    auto wrapped = remoteMr;
    wrapped.info.addr = std::numeric_limits<uint64_t>::max() - 1;
    ASSERT_TRUE(rejectsInvalidUsage([&] { qp->stageSendWriteWithImm(source.get(), wrapped, 1, 0, 0, 2, true, 0); }));
    ASSERT_TRUE(rejectsInvalidUsage(
        [&] { qp->stageSendWriteWithImm(nullptr, remoteMr, 0, 0, 0, std::numeric_limits<uint64_t>::max(), true, 0); }));

    // Fill every remaining slot before posting: a rejected WR must not consume space or break the chain.
    for (int i = 1; i < queueDepth; ++i) {
      qp->stageRecv(i);
      qp->stageSendWriteWithImm(nullptr, remoteMr, 0, i, 0, 0, true, i);
      tokens.push_back(i);
    }
    qp->postRecv();
  });
  runOnBothRanks("complete checked native writes", [&] {
    qp->postSend();
    drain(queueDepth - 1, queueDepth, tokens);
    EXPECT_EQ(buffer.back(), static_cast<unsigned char>(0x40 + 1 - gEnv->rank));
  });
}

TEST(MrcPeerToPeerTest, ReceiveCapacityAndReplenishment) {
  for (int round = 0; round < 3; ++round) {
    for (int i = 0; i < 8; ++i) qp->stageRecv(round * 8 + i);
    EXPECT_TRUE(rejectsInvalidUsage([&] { qp->stageRecv(100); }));
    qp->postRecv();
    EXPECT_TRUE(rejectsInvalidUsage([&] { qp->stageRecv(100); }));
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

TEST(MrcPeerToPeerTest, ReceiveCreditsWithDelayedReceiver) {
  constexpr int rounds = 64;
  runOnBothRanks("initial receive credit acknowledgement", [&] {
    for (int i = 0; i < queueDepth; ++i) qp->stageRecv(i);
    qp->postRecv();
  });
  for (int round = 0; round < rounds; ++round) {
    const unsigned int first = round * queueDepth;
    runOnBothRanks("send completion without receive replenishment", [&] {
      for (int i = 0; i < queueDepth; ++i) {
        qp->stageSendWriteWithImm(nullptr, remoteMr, 0, first + i, 0, 0, true, first + i);
      }
      qp->postSend();
      ASSERT_TRUE(rejectsInvalidUsage([&] { qp->stageSendWriteWithImm(nullptr, remoteMr, 0, 0, 0, 0, true, 0); }));
      int completed = 0;
      auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
      while (completed < queueDepth) {
        int count = qp->pollSendCq();
        ASSERT_GE(count, 0);
        for (int i = 0; i < count; ++i) {
          ASSERT_EQ(qp->getSendWcStatus(i), 0);
        }
        completed += count;
        ASSERT_TRUE(std::chrono::steady_clock::now() < deadline);
        std::this_thread::yield();
      }
      ASSERT_EQ(completed, queueDepth);
      ASSERT_EQ(qp->getNumSendCqItems(), 0);
      ASSERT_TRUE(rejectsInvalidUsage([&] { qp->stageRecv(0); }));
    });

    // Both send CQs are empty, but neither peer has polled or reposted a receive.
    // Only the following explicit test-side repost acknowledgement grants the next batch.
    runOnBothRanks("receive replenishment acknowledgement", [&] {
      if (gEnv->rank == 1) std::this_thread::sleep_for(std::chrono::milliseconds(2));
      std::array<bool, queueDepth> seen{};
      int received = 0;
      auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
      while (received < queueDepth) {
        int count = qp->pollRecvCq();
        ASSERT_GE(count, 0);
        for (int i = 0; i < count; ++i) {
          ASSERT_EQ(qp->getRecvWcStatus(i), 0);
          unsigned int value = qp->getRecvWcImmData(i);
          ASSERT_GE(value, first);
          ASSERT_LT(value, first + queueDepth);
          unsigned int index = value - first;
          ASSERT_FALSE(seen[index]);
          seen[index] = true;
          ++received;
          if (round + 1 < rounds) qp->stageRecv(value + queueDepth);
        }
        ASSERT_TRUE(std::chrono::steady_clock::now() < deadline);
        std::this_thread::yield();
      }
      ASSERT_EQ(received, queueDepth);
      if (round + 1 < rounds) qp->postRecv();
    });
  }
}

struct MrcMetadataAccess : mscclpp::BaseConnection {
  using BaseConnection::getImpl;
};

class MrcConnectionTest : public MrcTestBase {
 protected:
  void SetUp() override {
    MrcTestBase::SetUp();
    mscclpp::Endpoint endpoint;
    runOnBothRanks("native Connection resources", [&] {
      transport = mrcTransportForDevice(mrcDevice);
      ASSERT_TRUE(transport != mscclpp::Transport::Unknown);
      context = mscclpp::Context::create();
      local = context->registerMemory(buffer.data(), buffer.size(), transport);
      endpoint = context->createEndpoint({transport,
                                          mscclpp::DeviceType::CPU,
                                          queueDepth,
                                          {},
                                          {1, std::stoi(gEnv->args["ib_gid_index"]), 0, queueDepth, 2, 0}});
    });
    auto peer = mscclpp::Endpoint::deserialize(exchangeWithPeer(endpoint.serialize()));
    remote = mscclpp::RegisteredMemory::deserialize(exchangeWithPeer(local.serialize()));
    runOnBothRanks("connect private native data path", [&] {
      ASSERT_TRUE(rejectsInvalidUsage([&] { context->connect(endpoint, peer); }));
      connection = std::make_unique<mscclpp::MrcConnection>(context, endpoint, peer);
    });
  }

  std::array<unsigned char, 32> buffer{};
  mscclpp::Transport transport;
  std::shared_ptr<mscclpp::Context> context;
  mscclpp::RegisteredMemory local, remote;
  std::unique_ptr<mscclpp::BaseConnection> connection;
};

TEST(MrcConnectionTest, WritesAndFlushesThroughBothHooks) {
  std::array<unsigned char, 8> source;
  for (bool async : {false, true}) {
    runOnBothRanks("flush posted writes and reuse source", [&] {
      auto src = context->registerMemory(source.data(), source.size(), transport);
      std::weak_ptr<mscclpp::RdmaMemory> srcOwner =
          MrcMetadataAccess::getImpl(src).getTransportInfo(transport).rdmaMemory;
      for (int i = 0; i < 3; ++i) {
        source[1 + i] = 0x40 + async * 16 + gEnv->rank * 4 + i;
        // The third write exceeds the two-slot SQ unless the Connection progresses prior writes.
        connection->write(remote, 8 + async * 4 + i, src, 1 + i, 1);
      }
      src = {};
      if (async) {
        connection->requestFlush();
        ASSERT_FALSE(srcOwner.expired());
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
        while (!connection->progressFlush()) {
          ASSERT_TRUE(std::chrono::steady_clock::now() < deadline);
          std::this_thread::yield();
        }
      } else {
        ASSERT_FALSE(srcOwner.expired());
        connection->flush(8000000);
      }
      source.fill(0xff);
      ASSERT_TRUE(srcOwner.expired());
    });
    for (int i = 0; i < 3; ++i) {
      EXPECT_EQ(buffer[8 + async * 4 + i], static_cast<unsigned char>(0x40 + async * 16 + (1 - gEnv->rank) * 4 + i));
    }
  }
}

TEST(MrcConnectionTest, RejectsInvalidWritesBeforePosting) {
  runOnBothRanks("reject invalid Connection arguments", [&] {
    auto other = mscclpp::Context::create();
    auto foreign = other->registerMemory(buffer.data(), buffer.size(), transport);
    auto wrongPeer = mscclpp::RegisteredMemory::deserialize(local.serialize());
    auto unregistered = context->registerMemory(buffer.data(), buffer.size(), mscclpp::NoTransports);
    ASSERT_TRUE(rejectsInvalidUsage([&] { connection->write(local, 0, local, 0, 1); }));
    ASSERT_TRUE(rejectsInvalidUsage([&] { connection->write(remote, 0, remote, 0, 1); }));
    ASSERT_TRUE(rejectsInvalidUsage([&] { connection->write(remote, 0, unregistered, 0, 1); }));
    ASSERT_TRUE(rejectsInvalidUsage([&] { connection->write(wrongPeer, 0, local, 0, 1); }));
    ASSERT_TRUE(rejectsInvalidUsage([&] { connection->write(remote, 0, foreign, 0, 1); }));
    ASSERT_TRUE(
        rejectsInvalidUsage([&] { connection->write(remote, std::numeric_limits<uint64_t>::max(), local, 0, 1); }));
    uint64_t token = 0;
    ASSERT_TRUE(rejectsInvalidUsage([&] { connection->updateAndSync(remote, 0, &token, 1); }));
    ASSERT_EQ(token, uint64_t{0});
    ASSERT_TRUE(rejectsInvalidUsage([&] { connection->accumulate(remote, 0, 1); }));
    buffer[0] = 0x30 + gEnv->rank;
    connection->write(remote, 1, local, 0, 1);
    connection->flush(8000000);
  });
  EXPECT_EQ(buffer[1], static_cast<unsigned char>(0x30 + 1 - gEnv->rank));
}

TEST(MrcConnectionTest, ReusesGpuSourceAfterFlush) {
  constexpr size_t bytes = 4096;
  std::shared_ptr<unsigned char> src, dst;
  mscclpp::RegisteredMemory srcMemory, dstMemory;
  runOnBothRanks("register native GPU source and destination", [&] {
    int devices = 0;
    MSCCLPP_CUDATHROW(cudaGetDeviceCount(&devices));
    ASSERT_GT(devices, 0);
    MSCCLPP_CUDATHROW(cudaSetDevice(rankToLocalRank(gEnv->rank) % devices));
    src = mscclpp::detail::gpuCallocShared<unsigned char>(bytes);
    dst = mscclpp::detail::gpuCallocShared<unsigned char>(bytes);
    srcMemory = context->registerMemory(src.get(), bytes, transport);
    dstMemory = context->registerMemory(dst.get(), bytes, transport);
  });
  auto target = mscclpp::RegisteredMemory::deserialize(exchangeWithPeer(dstMemory.serialize()));
  for (bool async : {false, true}) {
    runOnBothRanks("write GPU memory and reuse source after flush", [&] {
      MSCCLPP_CUDATHROW(cudaMemset(src.get(), 0x40 + async * 16 + gEnv->rank, bytes));
      MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
      connection->write(target, 0, srcMemory, 0, bytes);
      if (async) {
        connection->requestFlush();
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(8);
        while (!connection->progressFlush()) {
          ASSERT_TRUE(std::chrono::steady_clock::now() < deadline);
          std::this_thread::yield();
        }
      } else {
        connection->flush(8000000);
      }
      MSCCLPP_CUDATHROW(cudaMemset(src.get(), 0xff, bytes));
      MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
    });
    runOnBothRanks("verify GPU destination before the next write", [&] {
      std::vector<unsigned char> observed(bytes);
      MSCCLPP_CUDATHROW(cudaMemcpy(observed.data(), dst.get(), bytes, cudaMemcpyDeviceToHost));
      EXPECT_TRUE(observed == std::vector<unsigned char>(bytes, 0x40 + async * 16 + 1 - gEnv->rank));
    });
  }
}

TEST(MultiProcessTest, MrcEndpointAndMemoryExchange) {
  const char* device = std::getenv("MSCCLPP_TEST_MRC_DEVICE");
  if (device == nullptr || gEnv->worldSize != 2) {
    SKIP_TEST() << "Requires two ranks and MSCCLPP_TEST_MRC_DEVICE naming an MRC-capable NIC.";
  }
  auto transport = mrcTransportForDevice(device);
  ASSERT_TRUE(transport != mscclpp::Transport::Unknown);
  auto context = mscclpp::Context::create();
  std::vector<unsigned char> buffer(4096, 0x5a);
  std::fill_n(buffer.begin(), 64, static_cast<unsigned char>(gEnv->rank + 17));
  auto memory = context->registerMemory(buffer.data(), buffer.size(), transport);
  auto endpoint = context->createEndpoint(
      {transport, mscclpp::DeviceType::CPU, -1, {}, {1, std::stoi(gEnv->args["ib_gid_index"]), 0, 8, 8, 8}});
  auto peerEndpoint = mscclpp::Endpoint::deserialize(exchangeWithPeer(endpoint.serialize()));
  auto peerMemory = mscclpp::RegisteredMemory::deserialize(exchangeWithPeer(memory.serialize()));
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
  qp->stageSendWriteWithImm(localMemory.mr(), remoteMemory.buffer(), 64, 0, 0, 1024, true, 0x01020304 + gEnv->rank);
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
