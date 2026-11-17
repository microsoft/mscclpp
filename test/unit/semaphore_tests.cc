// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <mscclpp/memory_channel.hpp>
#include <mscclpp/port_channel.hpp>
#include <type_traits>
#include <utility>

#include "../framework.hpp"
#include "connection.hpp"

namespace {
using Backend = mscclpp::PortChannelBackend;
using Host = mscclpp::PortChannel<Backend::Host>;
using Device = mscclpp::PortChannel<Backend::Device>;
using PortSemaphore = mscclpp::PortDevice2DeviceSemaphore;
using MemorySemaphore = mscclpp::MemoryDevice2DeviceSemaphore;
static_assert(std::is_same_v<mscclpp::DeviceHandle<Host>, mscclpp::PortChannelDeviceHandle>);
static_assert(std::is_same_v<mscclpp::DeviceHandle<Device>, mscclpp::PortChannelDeviceHandleT<Backend::Device>>);
static_assert(std::is_same_v<decltype(mscclpp::deviceHandle(std::declval<Device&>())), Device::DeviceHandle>);
static_assert(std::is_same_v<decltype(mscclpp::PortChannel(std::declval<std::shared_ptr<PortSemaphore>>(),
                                                           std::declval<mscclpp::RegisteredMemory>(),
                                                           std::declval<mscclpp::RegisteredMemory>())),
                             Device>);
static_assert(!std::is_constructible_v<Host, std::shared_ptr<PortSemaphore>, mscclpp::RegisteredMemory,
                                       mscclpp::RegisteredMemory>);
static_assert(
    !std::is_constructible_v<Device, const mscclpp::Semaphore&, mscclpp::RegisteredMemory, mscclpp::RegisteredMemory>);
static_assert(!std::is_assignable_v<Host&, Device>);
static_assert(!std::is_copy_constructible_v<PortSemaphore> && std::is_move_constructible_v<PortSemaphore>);
static_assert(!std::is_copy_constructible_v<MemorySemaphore> && std::is_move_constructible_v<MemorySemaphore>);

template <class Func>
bool rejectsInvalidUsage(Func&& func) {
  try {
    func();
  } catch (const mscclpp::Error& error) {
    return error.getErrorCode() == mscclpp::ErrorCode::InvalidUsage;
  }
  return false;
}

// Metadata-only connection. Counters use real GPU allocations; no transport success is simulated.
class MetadataConnection : public mscclpp::BaseConnection {
 public:
  MetadataConnection(std::shared_ptr<mscclpp::Context> context, const mscclpp::Endpoint& endpoint, bool device)
      : BaseConnection(std::move(context), endpoint), device_(device) {}
  bool isDeviceBackend() const override { return device_; }
  mscclpp::DeviceMemoryHandle getDeviceMemoryInfo(const mscclpp::RegisteredMemory& memory) const override {
    BaseConnection::getImpl(memory);  // Exercise the real empty-registration check.
    return {0x1000, 17};
  }
  mscclpp::Transport transport() const override { return mscclpp::Transport::Ethernet; }
  mscclpp::Transport remoteTransport() const override { return transport(); }
  void write(mscclpp::RegisteredMemory, uint64_t, mscclpp::RegisteredMemory, uint64_t, uint64_t) override { fail(); }
  void updateAndSync(mscclpp::RegisteredMemory, uint64_t, uint64_t*, uint64_t) override { fail(); }
  void accumulate(mscclpp::RegisteredMemory, uint64_t, int64_t) override { fail(); }
  void flush(int64_t) override { fail(); }

 private:
  static void fail() { throw mscclpp::Error("metadata-only connection", mscclpp::ErrorCode::InvalidUsage); }
  bool device_;
};
}  // namespace

class SemaphoreValidationTest : public mscclpp::test::TestCase {
 protected:
  void SetUp() override {
    MSCCLPP_CUDATHROW(cudaSetDevice(0));
    context = mscclpp::Context::create();
  }
  mscclpp::Semaphore makeSemaphore(mscclpp::Device device, bool deviceMode) {
    mscclpp::EndpointConfig config;
    config.transport = mscclpp::Transport::Unknown;
    config.device = device;
    auto endpoint = context->createEndpoint(config);
    mscclpp::Connection connection;
    mscclpp::BaseConnection::getImpl(connection) = std::make_shared<MetadataConnection>(context, endpoint, deviceMode);
    mscclpp::SemaphoreStub stub(connection);
    return mscclpp::Semaphore(stub, stub);
  }
  std::shared_ptr<mscclpp::Context> context;
};

TEST(SemaphoreValidationTest, EmptyAndCpuInputs) {
  mscclpp::Semaphore empty;
  EXPECT_TRUE(rejectsInvalidUsage([&] { empty.connection(); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { empty.localMemory(); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { empty.remoteMemory(); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { MemorySemaphore invalid(empty); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { PortSemaphore invalid(empty); }));
  auto cpu = makeSemaphore({mscclpp::DeviceType::CPU, 0}, true);
  EXPECT_TRUE(rejectsInvalidUsage([&] { MemorySemaphore invalid(cpu); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { PortSemaphore invalid(cpu); }));
  auto memoryOnly = makeSemaphore({mscclpp::DeviceType::GPU, 0}, false);
  EXPECT_TRUE(rejectsInvalidUsage([&] { PortSemaphore invalid(memoryOnly); }));
}

TEST(SemaphoreValidationTest, DeviceChannelOwnershipAndRegistrations) {
  auto raw = makeSemaphore({mscclpp::DeviceType::GPU, 0}, true);
  auto owner = std::make_shared<PortSemaphore>(raw);
  const auto expected = owner->deviceHandle().expectedInboundToken;
  EXPECT_TRUE(rejectsInvalidUsage([&] { Device invalid(owner, {}, raw.localMemory()); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { Device invalid(owner, raw.remoteMemory(), {}); }));
  Device channel(owner, raw.remoteMemory(), raw.localMemory());
  auto copy = channel;
  mscclpp::BasePortChannel<Backend::Device> baseCopy = channel;
  std::weak_ptr<PortSemaphore> weak = owner;
  owner.reset();
  channel = {};
  EXPECT_TRUE(!weak.expired());
  EXPECT_EQ(mscclpp::deviceHandle(copy).semaphore_.expectedInboundToken, expected);
  EXPECT_EQ(mscclpp::deviceHandle(baseCopy).semaphore_.expectedInboundToken, expected);
  copy = {};
  EXPECT_TRUE(!weak.expired());
  baseCopy = {};
  EXPECT_TRUE(weak.expired());
  PortSemaphore first(raw);
  PortSemaphore second(raw);
  EXPECT_TRUE(first.deviceHandle().expectedInboundToken != second.deviceHandle().expectedInboundToken);
  auto moved = std::move(first);
  EXPECT_TRUE(rejectsInvalidUsage([&] { first.deviceHandle(); }));
  EXPECT_TRUE(moved.deviceHandle().expectedInboundToken != nullptr);
}

TEST(SemaphoreValidationTest, MemoryCounterMoveAndOwnership) {
  auto raw = makeSemaphore({mscclpp::DeviceType::GPU, 0}, false);
  MemorySemaphore first(raw);
  const auto expected = first.deviceHandle().expectedInboundToken;
  auto moved = std::move(first);
  EXPECT_EQ(moved.deviceHandle().expectedInboundToken, expected);
  EXPECT_TRUE(rejectsInvalidUsage([&] { first.deviceHandle(); }));
  MemorySemaphore assigned(raw);
  assigned = std::move(moved);
  EXPECT_EQ(assigned.deviceHandle().expectedInboundToken, expected);
  EXPECT_TRUE(rejectsInvalidUsage([&] { moved.deviceHandle(); }));
  auto owner = std::make_shared<MemorySemaphore>(std::move(assigned));
  std::weak_ptr<MemorySemaphore> weak = owner;
  mscclpp::BaseMemoryChannel channel(owner);
  auto copy = channel;
  owner.reset();
  channel = {};
  EXPECT_TRUE(!weak.expired());
  EXPECT_EQ(mscclpp::deviceHandle(copy).semaphore_.expectedInboundToken, expected);
  copy = {};
  EXPECT_TRUE(weak.expired());
}

TEST(SemaphoreValidationTest, EndpointAllocationRestoresCaller) {
  int count = 0;
  MSCCLPP_CUDATHROW(cudaGetDeviceCount(&count));
  if (count < 2) SKIP_TEST() << "Requires two GPUs for endpoint allocation and caller-device restoration";
  auto rawMemory = makeSemaphore({mscclpp::DeviceType::GPU, 0}, false);
  auto rawPort = makeSemaphore({mscclpp::DeviceType::GPU, 0}, true);
  mscclpp::CudaDeviceGuard caller(1);
  MemorySemaphore memory(rawMemory);
  PortSemaphore port(rawPort);
  for (auto* token : {memory.deviceHandle().expectedInboundToken, port.deviceHandle().expectedInboundToken}) {
    int current = -1;
    MSCCLPP_CUDATHROW(cudaGetDevice(&current));
    EXPECT_EQ(current, 1);
    int allocated = -1;
    MSCCLPP_CUTHROW(
        cuPointerGetAttribute(&allocated, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, reinterpret_cast<CUdeviceptr>(token)));
    EXPECT_EQ(allocated, 0);
  }
}
