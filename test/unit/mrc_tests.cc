// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <infiniband/verbs.h>

#include <cerrno>
#include <limits>
#include <mscclpp/core.hpp>
#include <mscclpp/errors.hpp>

#include "../framework.hpp"
#include "connection.hpp"
#include "ib.hpp"
#include "mrc.hpp"
#include "serialization.hpp"

class MrcSendProgressTest : public mscclpp::test::TestCase {
 protected:
  template <typename F>
  std::string failureMessage(F run) {
    try {
      run();
    } catch (const mscclpp::BaseError& error) {
      return error.what();
    }
    FAIL() << "Expected native MRC send progress to fail";
    return {};
  }
};

TEST(MrcSendProgressTest, RequiresEverySignaledCompletion) {
  mscclpp::MrcSendProgress sends;
  sends.posted(0, 2, 2);
  sends.polled(0, nullptr);
  EXPECT_FALSE(sends.drained());
  ibv_wc wc{};
  wc.status = IBV_WC_SUCCESS;
  sends.polled(1, &wc);
  EXPECT_FALSE(sends.drained());
  sends.polled(1, &wc);
  EXPECT_TRUE(sends.drained());

  sends.posted(0, 2, 1);
  sends.polled(1, &wc);
  EXPECT_EQ(sends.pending(), 0);
  EXPECT_TRUE(failureMessage([&] { sends.drained(); }).find("unsignaled") != std::string::npos);
}

TEST(MrcSendProgressTest, FailedPostOrPollCannotBecomeADrain) {
  mscclpp::MrcSendProgress post;
  // Even the first failed post may have submitted a prefix.
  auto failure = failureMessage([&] { post.posted(EIO, 2, 2); });
  EXPECT_EQ(failureMessage([&] { post.posted(0, 1, 1); }), failure);
  EXPECT_EQ(failureMessage([&] { post.drained(); }), failure);
  mscclpp::MrcSendProgress poll;
  errno = EIO;
  failure = failureMessage([&] { poll.polled(-1, nullptr); });
  EXPECT_EQ(failureMessage([&] { poll.polled(0, nullptr); }), failure);
  EXPECT_EQ(failureMessage([&] { poll.drained(); }), failure);
}

TEST(MrcSendProgressTest, FailedCompletionDoesNotRetireSignaledSends) {
  mscclpp::MrcSendProgress sends;
  sends.posted(0, 3, 2);
  ibv_wc wcs[2]{};
  wcs[0].status = IBV_WC_SUCCESS;
  wcs[1].wr_id = 42;
  wcs[1].status = IBV_WC_WR_FLUSH_ERR;
  auto failure = failureMessage([&] { sends.polled(2, wcs); });
  EXPECT_TRUE(failure.find("42") != std::string::npos);
  EXPECT_EQ(failureMessage([&] { sends.pending(); }), failure);
  EXPECT_EQ(failureMessage([&] { sends.drained(); }), failure);
}

struct MrcUnitMetadataAccess : mscclpp::BaseConnection {
  using BaseConnection::getImpl;
};

class MrcMetadataTest : public mscclpp::test::TestCase {
 protected:
  template <typename F>
  void expectInvalid(F function, const char* message = nullptr) {
    bool caught = false;
    try {
      function();
    } catch (const mscclpp::Error& error) {
      EXPECT_TRUE(error.getErrorCode() == mscclpp::ErrorCode::InvalidUsage);
      if (message) EXPECT_TRUE(std::string(error.what()).find(message) != std::string::npos);
      caught = true;
    }
    EXPECT_TRUE(caught);
  }

  std::vector<char> endpointRecord(mscclpp::Transport transport = mscclpp::Transport::MRC1) {
    mscclpp::EndpointConfig config(transport, mscclpp::DeviceType::CPU, 8, {}, {1, 3, 0, 8, 8, 8});
    std::vector<char> data;
    mscclpp::detail::serialize(data, config);
    mscclpp::detail::serialize(data, uint64_t{1});
    mscclpp::detail::serialize(data, uint64_t{2});
    mscclpp::detail::serialize(data, qpInfo);
    return data;
  }

  std::vector<char> memoryRecord(int8_t count = 1, mscclpp::TransportFlags advertised = mscclpp::Transport::MRC1) {
    std::vector<char> data;
    mscclpp::detail::serialize(data, reinterpret_cast<void*>(mrInfo.addr));
    mscclpp::detail::serialize(data, size_t{64});
    mscclpp::detail::serialize(data, uint64_t{1});
    mscclpp::detail::serialize(data, uint64_t{2});
    mscclpp::detail::serialize(data, advertised);
    mscclpp::detail::serialize(data, count);
    for (int i = 0; i < count; ++i) {
      mscclpp::detail::serialize(data, mscclpp::Transport::MRC1);
      mscclpp::detail::serialize(data, mrInfo);
    }
    return data;
  }

  mscclpp::MrcQpInfo qpInfo{0x1234, 2, 0x123456, 0x0102030405060708, 3, 0x1122334455667788, true};
  mscclpp::MrcMrInfo mrInfo{0x0102030405060708, 0x90a0b0c0};
};

TEST(MrcMetadataTest, EndpointRoundtripAndConnectionGate) {
  auto data = endpointRecord();
  auto endpoint = mscclpp::Endpoint::deserialize(data);
  EXPECT_TRUE(endpoint.transport() == mscclpp::Transport::MRC1);
  EXPECT_EQ(endpoint.config().mrc.gidIndex, 3);
  EXPECT_TRUE(endpoint.serialize() == data);
  auto context = mscclpp::Context::create();
  const char* message = "Native MRC connections are not implemented yet";
  expectInvalid([&] { context->connect(endpoint, endpoint); }, message);
  auto other = mscclpp::Endpoint::deserialize(endpointRecord(mscclpp::Transport::MRC3));
  expectInvalid([&] { context->connect(endpoint, other); }, message);
  expectInvalid([&] { context->connect(other, endpoint); }, message);
  auto rdma = MrcUnitMetadataAccess::getImpl(endpoint).rdmaEndpoint_;
  EXPECT_FALSE(rdma->isLocal());
  auto& nativeEndpoint = dynamic_cast<mscclpp::MrcEndpoint&>(*rdma);
  expectInvalid([&] { nativeEndpoint.connect(nativeEndpoint); });
  for (size_t size = 0; size < data.size(); ++size) {
    expectInvalid([&] { mscclpp::Endpoint::deserialize(std::vector<char>(data.begin(), data.begin() + size)); });
  }
}

TEST(MrcMetadataTest, RejectsIbConnectionRouting) {
  std::vector<char> bytes;
  mscclpp::detail::serialize(bytes, mscclpp::EndpointConfig(mscclpp::Transport::IB7, mscclpp::DeviceType::CPU));
  mscclpp::detail::serialize(bytes, uint64_t{1});
  mscclpp::detail::serialize(bytes, uint64_t{2});
  mscclpp::detail::serialize(bytes, mscclpp::IbQpInfo{});
  mscclpp::detail::serialize(bytes, false);
  auto ib = mscclpp::Endpoint::deserialize(bytes);
  auto mrc = mscclpp::Endpoint::deserialize(endpointRecord());
  auto context = mscclpp::Context::create();
  expectInvalid([&] { context->connect(ib, mrc); }, "Transport mismatch");
  expectInvalid([&] { context->connect(mrc, ib); }, "Transport mismatch");
}

TEST(MrcMetadataTest, MemoryRoundtripAndRecordValidation) {
  auto data = memoryRecord();
  auto memory = mscclpp::RegisteredMemory::deserialize(data);
  EXPECT_EQ(memory.size(), size_t{64});
  EXPECT_TRUE(memory.transports() == mscclpp::TransportFlags(mscclpp::Transport::MRC1));
  EXPECT_TRUE(memory.serialize() == data);
  auto rdma = MrcUnitMetadataAccess::getImpl(memory).getTransportInfo(mscclpp::Transport::MRC1).rdmaMemory;
  EXPECT_FALSE(rdma->isLocal());
  EXPECT_EQ(dynamic_cast<const mscclpp::MrcMemory&>(*rdma).address(), mrInfo.addr);
  for (size_t size = 0; size < data.size(); ++size) {
    expectInvalid(
        [&] { mscclpp::RegisteredMemory::deserialize(std::vector<char>(data.begin(), data.begin() + size)); });
  }
  expectInvalid([&] { mscclpp::RegisteredMemory::deserialize(memoryRecord(0)); });
  expectInvalid([&] { mscclpp::RegisteredMemory::deserialize(memoryRecord(2)); });
  expectInvalid([&] { mscclpp::RegisteredMemory::deserialize(memoryRecord(1, mscclpp::NoTransports)); });
  expectInvalid([&] {
    mscclpp::RegisteredMemory::deserialize(memoryRecord(1, mscclpp::Transport::MRC1 | mscclpp::Transport::MRC2));
  });
  mrInfo.addr = std::numeric_limits<uint64_t>::max() - 31;
  expectInvalid([&] { mscclpp::RegisteredMemory::deserialize(memoryRecord()); });
  mrInfo.addr = 0;
  expectInvalid([&] { mscclpp::RegisteredMemory::deserialize(memoryRecord()); });
}
