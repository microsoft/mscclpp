// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <endian.h>

#include <array>
#include <cstring>
#include <mscclpp/core.hpp>
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
  EXPECT_EQ(toString(mscclpp::Transport::CudaIpc | mscclpp::Transport::RDMA0 | mscclpp::Transport::Ethernet),
            std::string("{IPC, RDMA0, ETH}"));
  EXPECT_EQ(toString(mscclpp::Transport::RDMA0 | mscclpp::Transport::RDMA7), std::string("{RDMA0, RDMA7}"));
}

TEST(TransportFlagsTest, RdmaTransportWireEncoding) {
  using namespace mscclpp;
  const std::array rdma = {Transport::RDMA0, Transport::RDMA1, Transport::RDMA2, Transport::RDMA3,
                           Transport::RDMA4, Transport::RDMA5, Transport::RDMA6, Transport::RDMA7};
  for (size_t i = 0; i < rdma.size(); ++i) {
    auto flags = rdma[i] | Transport::Ethernet;
    // Historical wire tags are 2-9, with Ethernet in bit 10.
    int32_t tag = static_cast<int32_t>(i + 2);
    std::bitset<11> bits((uint64_t{1} << (i + 2)) | (uint64_t{1} << 10));
    std::vector<char> generic, legacy;
    detail::serialize(generic, rdma[i]);
    detail::serialize(generic, flags);
    detail::serialize(legacy, tag);
    detail::serialize(legacy, bits);
    EXPECT_TRUE(generic == legacy);
    TransportFlags restored;
    detail::deserialize(generic.cbegin() + sizeof(Transport), restored);
    EXPECT_TRUE(restored.has(rdma[i]));
    EXPECT_TRUE(restored.has(Transport::Ethernet));
    EXPECT_EQ(restored.count(), size_t{2});
  }
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
  for (auto mode :
       {EndpointConfig::Ib::Mode::Host, EndpointConfig::Ib::Mode::HostNoAtomic, EndpointConfig::Ib::Mode::Device}) {
    bool noAtomic = mode == EndpointConfig::Ib::Mode::HostNoAtomic;
    EndpointConfig config(Transport::RDMA7, DeviceType::CPU, 13);
    config.ib.mode = mode;
    if (mode == EndpointConfig::Ib::Mode::Device) config.device = Device(DeviceType::GPU, 0);
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
  }
}

TEST(RdmaProviderTest, ImportedMemoryPreservesRecordOrderAndBytes) {
  using namespace mscclpp;
  uint64_t buffer = 0;
  void* ptr = &buffer;
  size_t size = sizeof(buffer);
  uint64_t hostHash = getHostHash() ^ 1;
  uint64_t pidHash = getPidHash();
  TransportFlags transports = Transport::RDMA0 | Transport::RDMA1;
  int8_t transportCount = 3;
  std::vector<char> bytes;
  detail::serialize(bytes, ptr);
  detail::serialize(bytes, size);
  detail::serialize(bytes, hostHash);
  detail::serialize(bytes, pidHash);
  detail::serialize(bytes, transports);
  detail::serialize(bytes, transportCount);
  uint64_t address = 0x1000;
  for (auto transport : {Transport::RDMA1, Transport::RDMA0, Transport::RDMA1}) {
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
  const auto& firstIb1 = getIbMemory(RdmaTestAccess::getImpl(memory).getTransportInfo(Transport::RDMA1));
  EXPECT_EQ(firstIb1.info().addr, uint64_t{0x1000});
  EXPECT_FALSE(firstIb1.isLocal());
}

TEST(RdmaProviderTest, ImportedMemoryPreservesMixedIpcMetadataBytes) {
  using namespace mscclpp;
  uint64_t buffer = 0;
  void* ptr = &buffer;
  size_t size = sizeof(buffer);
  uint64_t hostHash = getHostHash();
  uint64_t pidHash = getPidHash();
  TransportFlags transports = Transport::CudaIpc | Transport::RDMA1;
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
  detail::serialize(bytes, Transport::RDMA1);
  detail::serialize(bytes, mr);
  detail::serialize(bytes, Transport::CudaIpc);
  detail::serialize(bytes, ipc);
  detail::serialize(bytes, Transport::RDMA1);
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
  for (auto transport : {Transport::RDMA7, Transport::RDMA1, Transport::RDMA0}) {
    auto provider = std::make_shared<OpaqueRdmaProvider>(transport, calls, ptr, size);
    providers.push_back(provider);
    RdmaTestAccess::getImpl(*context).rdmaProviders_.emplace(transport, std::move(provider));
  }
  TransportFlags transports =
      Transport::RDMA7 | Transport::RDMA0 | Transport::RDMA1 | Transport::Ethernet | Transport::Unknown;
  auto memory = context->registerMemory(ptr, size, transports);
  EXPECT_TRUE(calls == (std::vector<Transport>{Transport::RDMA0, Transport::RDMA1, Transport::RDMA7}));
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
  auto importEndpoint = [](Transport transport, EndpointConfig::Ib::Mode mode) {
    EndpointConfig config(transport, DeviceType::CPU);
    config.ib.mode = mode;
    if (mode == EndpointConfig::Ib::Mode::Device) config.device = Device(DeviceType::GPU, 0);
    uint64_t hostHash = 61;
    uint64_t pidHash = 67;
    std::vector<char> bytes;
    detail::serialize(bytes, config);
    detail::serialize(bytes, hostHash);
    detail::serialize(bytes, pidHash);
    if (AllRDMATransports.has(transport)) {
      detail::serialize(bytes, IbQpInfo{});
      detail::serialize(bytes, false);
    }
    return Endpoint::deserialize(bytes);
  };
  auto context = Context::create();
  for (auto mode : {EndpointConfig::Ib::Mode::Host, EndpointConfig::Ib::Mode::Device}) {
    for (auto pair :
         {std::pair{Transport::RDMA0, Transport::RDMA7}, std::pair{Transport::RDMA7, Transport::RDMA0},
          std::pair{Transport::RDMA0, Transport::CudaIpc}, std::pair{Transport::CudaIpc, Transport::RDMA0}}) {
      auto local = importEndpoint(pair.first, mode);
      auto remote = importEndpoint(pair.second, mode);
      try {
        context->connect(local, remote);
        FAIL() << "Expected endpoint routing to reject connection construction";
      } catch (const Error& error) {
        EXPECT_TRUE(error.getErrorCode() == ErrorCode::InvalidUsage);
        // Imported local IB endpoints stop at the provider factory, before NIC/GPU access.
        const char* expected = AllRDMATransports.has(pair.first) && AllRDMATransports.has(pair.second)
                                   ? "Cannot create a connection from a remote IB endpoint"
                                   : "Transport mismatch";
        EXPECT_TRUE(std::string(error.what()).find(expected) != std::string::npos);
      }
    }
  }
}

TEST(RdmaProviderTest, DeviceModeRequiresGpuEndpoint) {
  using namespace mscclpp;
  if (getIBDeviceCount() == 0) {
    SKIP_TEST() << "Requires an IB device";
  }
  auto context = Context::create();
  EndpointConfig config(Transport::RDMA0, DeviceType::CPU);
  config.ib.mode = EndpointConfig::Ib::Mode::Device;
  try {
    context->createEndpoint(config);
    FAIL() << "Expected IB Device mode to reject a CPU endpoint";
  } catch (const Error& error) {
    EXPECT_TRUE(error.getErrorCode() == ErrorCode::InvalidUsage);
    EXPECT_TRUE(std::string(error.what()).find("IB Device mode requires a GPU endpoint") != std::string::npos);
  }
}

#if !defined(__HIP_PLATFORM_AMD__)
TEST(RdmaProviderTest, DeviceConnectionUsesProviderResources) {
  using namespace mscclpp;
  if (getIBDeviceCount() == 0) {
    SKIP_TEST() << "Requires an IB device";
  }
  {
    IbCtx ibContext(getIBDeviceName(Transport::RDMA0));
    if (!ibContext.supportsRdmaAtomics()) {
      SKIP_TEST() << "Requires an IB device with RDMA atomics";
    }
  }
  MSCCLPP_CUDATHROW(cudaSetDevice(0));
  int major = 0;
  MSCCLPP_CUDATHROW(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));
  if (major < 9) {
    SKIP_TEST() << "Requires Hopper or newer";
  }
  auto context = Context::create();
  EndpointConfig config(Transport::RDMA0, Device(DeviceType::GPU, 0));
  config.ib.mode = EndpointConfig::Ib::Mode::Device;
  auto endpoint = context->createEndpoint(config);
  ASSERT_TRUE(RdmaTestAccess::getImpl(endpoint).rdmaEndpoint_->isLocal());
  auto remote = Endpoint::deserialize(endpoint.serialize());
  EXPECT_FALSE(RdmaTestAccess::getImpl(remote).rdmaEndpoint_->isLocal());
  auto connection = context->connect(endpoint, remote);
  auto& connImpl = RdmaTestAccess::getImpl(connection);
  ASSERT_TRUE(connImpl->isDeviceBackend());
  ASSERT_NE(connImpl->getDeviceBackendInfo().qp, nullptr);

  auto buffer = detail::gpuCallocShared<uint64_t>();
  auto memory = context->registerMemory(buffer.get(), sizeof(uint64_t), Transport::RDMA0);
  auto remoteMemory = RegisteredMemory::deserialize(memory.serialize());
  const auto& ibMemory = getIbMemory(RdmaTestAccess::getImpl(memory).getTransportInfo(Transport::RDMA0));
  auto localHandle = connImpl->getDeviceMemoryInfo(memory);
  auto remoteHandle = connImpl->getDeviceMemoryInfo(remoteMemory);
  EXPECT_EQ(localHandle.addr, ibMemory.info().addr);
  EXPECT_EQ(localHandle.key, htobe32(ibMemory.mr()->getLkey()));
  EXPECT_EQ(remoteHandle.addr, ibMemory.info().addr);
  EXPECT_EQ(remoteHandle.key, htobe32(ibMemory.info().rkey));

  std::weak_ptr<Context> outerContext = context;
  std::weak_ptr<RdmaProvider> provider = RdmaTestAccess::getImpl(*context).rdmaProviders_.at(Transport::RDMA0);
  context.reset();
  EXPECT_FALSE(outerContext.expired());
  connection = Connection();
  EXPECT_TRUE(outerContext.expired());
  endpoint = Endpoint();
  EXPECT_FALSE(provider.expired());
  memory = RegisteredMemory();
  EXPECT_TRUE(provider.expired());
}
#endif

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
    auto endpoint = context->createEndpoint({Transport::RDMA0, DeviceType::CPU, -1, ib});
    auto memory = context->registerMemory(buffer.data(), sizeof(buffer), Transport::RDMA0);
    EXPECT_EQ(RdmaTestAccess::getImpl(*context).rdmaProviders_.size(), size_t{1});
    std::weak_ptr<RdmaProvider> provider = RdmaTestAccess::getImpl(*context).rdmaProviders_.at(Transport::RDMA0);
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
