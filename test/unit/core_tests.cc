// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <array>
#include <cstring>
#include <mscclpp/core.hpp>
#include <mscclpp/errors.hpp>
#include <sstream>

#include "../framework.hpp"
#include "connection.hpp"
#include "ib.hpp"
#include "serialization.hpp"
#include "utils_internal.hpp"

namespace {

struct RdmaTestAccess : mscclpp::BaseConnection {
  using BaseConnection::getImpl;
};

class OpaqueRdmaMemory : public mscclpp::RdmaMemory {
 public:
  OpaqueRdmaMemory(std::shared_ptr<mscclpp::RdmaProvider> provider, char metadata)
      : RdmaMemory(std::move(provider)), metadata_(metadata) {}
  bool isLocal() const override { return true; }
  void serialize(std::vector<char>& data) const override { data.push_back(metadata_); }

 private:
  char metadata_;
};

class OpaqueRdmaProvider : public mscclpp::RdmaProvider {
 public:
  OpaqueRdmaProvider(mscclpp::Transport transport, std::vector<mscclpp::Transport>& calls, void* data, size_t size)
      : transport_(transport), calls_(calls), data_(data), size_(size) {}
  std::shared_ptr<mscclpp::RdmaEndpoint> createEndpoint(mscclpp::EndpointConfig&) override { return {}; }
  std::shared_ptr<mscclpp::RdmaMemory> registerMemory(void* data, size_t size) override {
    EXPECT_EQ(data, data_);
    EXPECT_EQ(size, size_);
    calls_.push_back(transport_);
    return std::make_shared<OpaqueRdmaMemory>(shared_from_this(), static_cast<char>(transport_));
  }

 private:
  mscclpp::Transport transport_;
  std::vector<mscclpp::Transport>& calls_;
  void* data_;
  size_t size_;
};

template <typename F>
void expectInvalidUsage(F function, const char* message = nullptr) {
  try {
    function();
    FAIL() << "Expected invalid usage";
  } catch (const mscclpp::Error& error) {
    EXPECT_TRUE(error.getErrorCode() == mscclpp::ErrorCode::InvalidUsage);
    if (message) EXPECT_TRUE(std::string(error.what()).find(message) != std::string::npos);
  }
}

}  // namespace

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
  EXPECT_EQ(toString(mscclpp::Transport::MRC0 | mscclpp::Transport::MRC3), std::string("{MRC0, MRC3}"));
}

TEST(TransportFlagsTest, MrcIsExplicitOptIn) {
  EXPECT_TRUE((mscclpp::AllTransports & mscclpp::AllMrcTransports).none());
  EXPECT_TRUE((mscclpp::AllIBTransports & mscclpp::AllMrcTransports).none());
  EXPECT_TRUE(mscclpp::AllTransports ==
              (mscclpp::AllIBTransports | mscclpp::Transport::CudaIpc | mscclpp::Transport::Ethernet));
  auto selected = mscclpp::Transport::CudaIpc | mscclpp::Transport::MRC1;
  EXPECT_TRUE(selected.has(mscclpp::Transport::MRC1));
  EXPECT_FALSE(selected.has(mscclpp::Transport::IB1));
  EXPECT_EQ((selected & mscclpp::AllMrcTransports).count(), size_t{1});
}

TEST(RdmaProviderTest, FailedCreationDoesNotPopulateCache) {
  auto owner = mscclpp::Context::create();
  auto& context = RdmaTestAccess::getImpl(*owner);
  std::vector<mscclpp::Transport> rejected{mscclpp::Transport::Unknown, mscclpp::Transport::CudaIpc,
                                           mscclpp::Transport::Ethernet};
#if !defined(USE_IBVERBS)
  rejected.push_back(mscclpp::Transport::IB0);
#endif
#if !defined(MSCCLPP_USE_MRC)
  rejected.push_back(mscclpp::Transport::MRC0);
#endif
  for (auto transport : rejected) {
    for (int attempt = 0; attempt < 2; ++attempt) {
      expectInvalidUsage([&] { context.getRdmaProvider(transport); });
      EXPECT_TRUE(context.rdmaProviders_.empty());
    }
  }
}

#if !defined(USE_IBVERBS)
TEST(IbDisabledTest, ExplicitRequestsFail) {
  auto context = mscclpp::Context::create();
  unsigned char buffer[64]{};
  const char* message = "IB support is not enabled";
  expectInvalidUsage([&] { context->registerMemory(buffer, sizeof(buffer), mscclpp::Transport::IB0); }, message);
  expectInvalidUsage([&] { context->createEndpoint({mscclpp::Transport::IB0, mscclpp::DeviceType::CPU}); }, message);
  expectInvalidUsage([&] { context->createEndpoint({mscclpp::Transport::IB0}); }, message);
}
#endif

#if !defined(MSCCLPP_USE_MRC)
TEST(MrcDisabledTest, ExplicitRequestsFail) {
  auto context = mscclpp::Context::create();
  unsigned char buffer[64]{};
  const char* message = "Native MRC support is disabled";
  expectInvalidUsage([&] { context->registerMemory(buffer, sizeof(buffer), mscclpp::Transport::MRC0); }, message);
  expectInvalidUsage([&] { context->createEndpoint({mscclpp::Transport::MRC0, mscclpp::DeviceType::CPU}); }, message);
  expectInvalidUsage([&] { context->createEndpoint({mscclpp::Transport::MRC0}); }, message);
}

TEST(MrcDisabledTest, RejectsNativeMetadata) {
  const char* message = "Native MRC support is disabled";
  std::vector<char> endpoint;
  mscclpp::detail::serialize(endpoint, mscclpp::EndpointConfig(mscclpp::Transport::MRC0, mscclpp::DeviceType::CPU));
  mscclpp::detail::serialize(endpoint, uint64_t{1});
  mscclpp::detail::serialize(endpoint, uint64_t{2});
  expectInvalidUsage([&] { mscclpp::Endpoint::deserialize(endpoint); }, message);

  for (auto flags : {mscclpp::TransportFlags(mscclpp::Transport::MRC0), mscclpp::NoTransports}) {
    std::vector<char> memory;
    mscclpp::detail::serialize(memory, reinterpret_cast<void*>(uintptr_t{0x1000}));
    mscclpp::detail::serialize(memory, size_t{64});
    mscclpp::detail::serialize(memory, uint64_t{1});
    mscclpp::detail::serialize(memory, uint64_t{2});
    mscclpp::detail::serialize(memory, flags);
    int8_t count = flags.none() ? 1 : 0;
    mscclpp::detail::serialize(memory, count);
    if (count != 0) mscclpp::detail::serialize(memory, mscclpp::Transport::MRC0);
    expectInvalidUsage([&] { mscclpp::RegisteredMemory::deserialize(memory); }, message);
  }
}
#endif

TEST(EndpointMetadataTest, ExistingTransportRoundtrip) {
  auto context = mscclpp::Context::create();
  auto endpoint = context->createEndpoint({mscclpp::Transport::CudaIpc, mscclpp::DeviceType::CPU});
  auto imported = mscclpp::Endpoint::deserialize(endpoint.serialize());
  EXPECT_TRUE(imported.transport() == mscclpp::Transport::CudaIpc);
  EXPECT_TRUE(imported.device().type == mscclpp::DeviceType::CPU);
  EXPECT_TRUE(imported.serialize() == endpoint.serialize());
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

TEST(RdmaProviderTest, ImportedEndpointPreservesWireBytes) {
  using namespace mscclpp;
  IbQpInfo info;
  std::memset(&info, 0xa5, sizeof(info));
  info.lid = 17;
  info.linkLayer = 1;
  info.qpn = 23;
  info.spn = 31;
  info.mtu = 4;
  info.iid = 41;
  info.isGrh = true;
  for (bool noAtomic : {false, true}) {
    EndpointConfig config(Transport::IB7, DeviceType::CPU, 13);
    config.ib.mode = noAtomic ? EndpointConfig::Ib::Mode::HostNoAtomic : EndpointConfig::Ib::Mode::Host;
    config.ib.gidIndex = 3;
    uint64_t hostHash = 43;
    uint64_t pidHash = 47;
    std::vector<char> bytes;
    detail::serialize(bytes, config);
    detail::serialize(bytes, hostHash);
    detail::serialize(bytes, pidHash);
    detail::serialize(bytes, info);
    detail::serialize(bytes, noAtomic);

    auto endpoint = Endpoint::deserialize(bytes);
    EXPECT_TRUE(endpoint.serialize() == bytes);
    EXPECT_FALSE(RdmaTestAccess::getImpl(endpoint).rdmaEndpoint_->isLocal());
    bytes.pop_back();
    expectInvalidUsage([&] { Endpoint::deserialize(bytes); });
  }
}

TEST(RdmaProviderTest, ImportedMemoryPreservesRecordOrderAndBytes) {
  using namespace mscclpp;
  uint64_t buffer = 0;
  void* ptr = &buffer;
  size_t size = sizeof(buffer);
  uint64_t hostHash = getHostHash() ^ 1;
  uint64_t pidHash = getPidHash();
  TransportFlags transports = Transport::IB0 | Transport::IB1;
  int8_t transportCount = 3;
  std::vector<char> bytes;
  detail::serialize(bytes, ptr);
  detail::serialize(bytes, size);
  detail::serialize(bytes, hostHash);
  detail::serialize(bytes, pidHash);
  detail::serialize(bytes, transports);
  detail::serialize(bytes, transportCount);
  uint64_t address = 0x1000;
  for (auto transport : {Transport::IB1, Transport::IB0, Transport::IB1}) {
    IbMrInfo info;
    std::memset(&info, 0x5a, sizeof(info));
    info.addr = address;
    info.rkey = 53;
    detail::serialize(bytes, transport);
    detail::serialize(bytes, info);
    address += 0x100;
  }

  auto memory = RegisteredMemory::deserialize(bytes);
  EXPECT_TRUE(memory.serialize() == bytes);
  EXPECT_EQ(memory.data(), nullptr);
  const auto& firstIb1 = getIbMemory(RdmaTestAccess::getImpl(memory).getTransportInfo(Transport::IB1));
  EXPECT_EQ(firstIb1.info().addr, uint64_t{0x1000});
  EXPECT_FALSE(firstIb1.isLocal());
  bytes.pop_back();
  expectInvalidUsage([&] { RegisteredMemory::deserialize(bytes); });
}

TEST(RdmaProviderTest, ImportedMemoryPreservesMixedIpcMetadataBytes) {
  using namespace mscclpp;
  uint64_t buffer = 0;
  void* ptr = &buffer;
  size_t size = sizeof(buffer);
  uint64_t hostHash = getHostHash();
  uint64_t pidHash = getPidHash();
  TransportFlags transports = Transport::CudaIpc | Transport::IB1;
  int8_t transportCount = 3;
  GpuIpcMemHandle ipc;
  std::memset(&ipc, 0xa7, sizeof(ipc));
  ipc.typeFlags = GpuIpcMemHandle::Type::RuntimeIpc;
  ipc.baseSize = size;
  ipc.offsetFromBase = 0;
  IbMrInfo mr;
  std::memset(&mr, 0x5a, sizeof(mr));
  mr.addr = 0x2000;
  mr.rkey = 59;
  std::vector<char> bytes;
  detail::serialize(bytes, ptr);
  detail::serialize(bytes, size);
  detail::serialize(bytes, hostHash);
  detail::serialize(bytes, pidHash);
  detail::serialize(bytes, transports);
  detail::serialize(bytes, transportCount);
  detail::serialize(bytes, Transport::IB1);
  detail::serialize(bytes, mr);
  detail::serialize(bytes, Transport::CudaIpc);
  detail::serialize(bytes, ipc);
  detail::serialize(bytes, Transport::IB1);
  detail::serialize(bytes, mr);

  // Same-process runtime IPC metadata requires no GPU import or address query.
  auto memory = RegisteredMemory::deserialize(bytes);
  EXPECT_TRUE(memory.serialize() == bytes);
  EXPECT_EQ(memory.data(), ptr);
  const auto& ipcInfo = RdmaTestAccess::getImpl(memory).getTransportInfo(Transport::CudaIpc);
  try {
    getIbMemory(ipcInfo);
    FAIL() << "Expected IB memory access to reject an IPC record";
  } catch (const Error& error) {
    EXPECT_TRUE(error.getErrorCode() == ErrorCode::InvalidUsage);
  }
}

TEST(RdmaProviderTest, RegisteredMemoryKeepsProviderMetadataOpaque) {
  using namespace mscclpp;
  uint64_t buffer = 0;
  void* ptr = &buffer;
  size_t size = sizeof(buffer);
  std::vector<Transport> calls;
  std::vector<std::weak_ptr<RdmaProvider>> providers;
  auto context = Context::create();
  // These test doubles are opaque to generic code and are never used for IB I/O.
  for (auto transport : {Transport::IB7, Transport::IB1, Transport::IB0}) {
    auto provider = std::make_shared<OpaqueRdmaProvider>(transport, calls, ptr, size);
    providers.push_back(provider);
    RdmaTestAccess::getImpl(*context).rdmaProviders_.emplace(transport, std::move(provider));
  }
  TransportFlags transports =
      Transport::IB7 | Transport::IB0 | Transport::IB1 | Transport::Ethernet | Transport::Unknown;
  auto memory = context->registerMemory(ptr, size, transports);
  EXPECT_TRUE(calls == (std::vector<Transport>{Transport::IB0, Transport::IB1, Transport::IB7}));
  EXPECT_EQ(memory.data(), ptr);
  EXPECT_EQ(memory.size(), size);
  EXPECT_EQ(memory.transports(), transports);

  uint64_t hostHash = getHostHash();
  uint64_t pidHash = getPidHash();
  int8_t transportCount = 3;
  std::vector<char> bytes;
  detail::serialize(bytes, ptr);
  detail::serialize(bytes, size);
  detail::serialize(bytes, hostHash);
  detail::serialize(bytes, pidHash);
  detail::serialize(bytes, transports);
  detail::serialize(bytes, transportCount);
  for (auto transport : calls) {
    detail::serialize(bytes, transport);
    detail::serialize(bytes, static_cast<char>(transport));
  }
  context.reset();
  for (const auto& provider : providers) EXPECT_FALSE(provider.expired());
  EXPECT_TRUE(memory.serialize() == bytes);
  memory = RegisteredMemory();
  for (const auto& provider : providers) EXPECT_TRUE(provider.expired());
}

TEST(RdmaProviderTest, EndpointRoutingPreservesIbDeviceCompatibility) {
  using namespace mscclpp;
  auto importEndpoint = [](Transport transport) {
    EndpointConfig config(transport, DeviceType::CPU);
    uint64_t hostHash = 61;
    uint64_t pidHash = 67;
    std::vector<char> bytes;
    detail::serialize(bytes, config);
    detail::serialize(bytes, hostHash);
    detail::serialize(bytes, pidHash);
    if (AllIBTransports.has(transport)) {
      detail::serialize(bytes, IbQpInfo{});
      detail::serialize(bytes, false);
    }
    return Endpoint::deserialize(bytes);
  };
  auto context = Context::create();
  for (auto pair : {std::pair{Transport::IB0, Transport::IB7}, std::pair{Transport::IB7, Transport::IB0},
                    std::pair{Transport::IB0, Transport::CudaIpc}, std::pair{Transport::CudaIpc, Transport::IB0}}) {
    auto local = importEndpoint(pair.first);
    auto remote = importEndpoint(pair.second);
    try {
      context->connect(local, remote);
      FAIL() << "Expected endpoint routing to reject connection construction";
    } catch (const Error& error) {
      EXPECT_TRUE(error.getErrorCode() == ErrorCode::InvalidUsage);
      // Imported local IB endpoints stop at the provider factory, before NIC/GPU access.
      const char* expected = AllIBTransports.has(pair.first) && AllIBTransports.has(pair.second)
                                 ? "Cannot create a connection from a remote IB endpoint"
                                 : "Transport mismatch";
      EXPECT_TRUE(std::string(error.what()).find(expected) != std::string::npos);
    }
  }
}

TEST(RdmaProviderTest, LocalResourcesRetainProvider) {
  using namespace mscclpp;
  if (getIBDeviceCount() == 0) {
    SKIP_TEST() << "Requires an IB device";
  }
  MSCCLPP_CUDATHROW(cudaSetDevice(0));
  for (bool memoryFirst : {false, true}) {
    std::array<uint64_t, 8> buffer{};
    auto context = Context::create();
    EndpointConfig::Ib ib;
    ib.maxCqSize = 8;
    ib.maxCqPollNum = 8;
    ib.maxSendWr = 16;
    ib.maxRecvWr = 8;
    ib.maxWrPerSend = 4;
    auto endpoint = context->createEndpoint({Transport::IB0, DeviceType::CPU, -1, ib});
    auto memory = context->registerMemory(buffer.data(), sizeof(buffer), Transport::IB0);
    EXPECT_EQ(RdmaTestAccess::getImpl(*context).rdmaProviders_.size(), size_t{1});
    std::weak_ptr<RdmaProvider> provider = RdmaTestAccess::getImpl(*context).rdmaProviders_.at(Transport::IB0);
    std::weak_ptr<Context> outerContext = context;
    auto endpointBytes = endpoint.serialize();
    auto memoryBytes = memory.serialize();

    context.reset();
    EXPECT_TRUE(outerContext.expired());
    ASSERT_FALSE(provider.expired());
    EXPECT_TRUE(endpoint.serialize() == endpointBytes);
    EXPECT_TRUE(memory.serialize() == memoryBytes);
    if (memoryFirst) {
      memory = RegisteredMemory();
    } else {
      endpoint = Endpoint();
    }
    ASSERT_FALSE(provider.expired());
    memory = RegisteredMemory();
    endpoint = Endpoint();
    EXPECT_TRUE(provider.expired());
  }
}
