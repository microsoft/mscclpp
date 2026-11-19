// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <mscclpp/memory_channel.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/switch_channel.hpp>
#include <type_traits>
#include <utility>

#include "../framework.hpp"
#include "connection.hpp"
#include "logger.hpp"

namespace {
using Backend = mscclpp::PortChannelBackend;
using Host = mscclpp::PortChannel<Backend::Host>;
using Device = mscclpp::PortChannel<Backend::Device>;
using PortSemaphore = mscclpp::PortDevice2DeviceSemaphore;
static_assert(std::is_same_v<mscclpp::BasePortChannel<>::DeviceHandle, mscclpp::BasePortChannelDeviceHandle<>>);
static_assert(std::is_same_v<mscclpp::BasePortChannel<Backend::Device>::DeviceHandle,
                             mscclpp::BasePortChannelDeviceHandle<Backend::Device>>);
static_assert(std::is_same_v<mscclpp::BaseMemoryChannel::DeviceHandle, mscclpp::BaseMemoryChannelDeviceHandle>);
static_assert(std::is_same_v<mscclpp::MemoryChannel::DeviceHandle, mscclpp::MemoryChannelDeviceHandle>);
static_assert(std::is_same_v<mscclpp::SwitchChannel::DeviceHandle, mscclpp::SwitchChannelDeviceHandle>);
static_assert(std::is_same_v<mscclpp::DeviceHandle<Host>, mscclpp::PortChannelDeviceHandle<>>);
static_assert(std::is_same_v<mscclpp::DeviceHandle<Device>, mscclpp::PortChannelDeviceHandle<Backend::Device>>);
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

static_assert(std::is_trivially_copyable_v<mscclpp::BaseMemoryChannelDeviceHandle> &&
              std::is_trivially_copyable_v<mscclpp::MemoryChannelDeviceHandle> &&
              std::is_trivially_copyable_v<mscclpp::SwitchChannelDeviceHandle>);

// The pre-existing Sm alias remains deprecated.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
static_assert(std::is_same_v<mscclpp::SmChannelDeviceHandle, mscclpp::MemoryChannelDeviceHandle>);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

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
  static void fail() {
    THROW(mscclpp::CONN, mscclpp::Error, mscclpp::ErrorCode::InvalidUsage, "metadata-only connection");
  }
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

TEST(SemaphoreValidationTest, PortAdmission) {
  auto cpu = makeSemaphore({mscclpp::DeviceType::CPU, 0}, true);
  EXPECT_TRUE(rejectsInvalidUsage([&] { PortSemaphore invalid(cpu); }));
  auto nonDevice = makeSemaphore({mscclpp::DeviceType::GPU, 0}, false);
  EXPECT_TRUE(rejectsInvalidUsage([&] { PortSemaphore invalid(nonDevice); }));
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

  auto liveOwner = std::make_shared<PortSemaphore>(raw);
  Device liveChannel(liveOwner, raw.remoteMemory(), raw.localMemory());
  mscclpp::BasePortChannel<Backend::Device> liveBase = liveChannel;
  const auto liveExpected = liveOwner->deviceHandle().expectedInboundToken;
  auto movedOwner = std::move(*liveOwner);
  EXPECT_EQ(movedOwner.deviceHandle().expectedInboundToken, liveExpected);
  // Extraction must reject before accessing the moved-from raw semaphore's connection.
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::deviceHandle(liveBase); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::deviceHandle(liveChannel); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::BasePortChannel<Backend::Device> invalid(liveOwner); }));
}

TEST(SemaphoreValidationTest, EndpointAllocationRestoresCaller) {
  int count = 0;
  MSCCLPP_CUDATHROW(cudaGetDeviceCount(&count));
  if (count < 2) SKIP_TEST() << "Requires two GPUs for endpoint allocation and caller-device restoration";
  auto rawPort = makeSemaphore({mscclpp::DeviceType::GPU, 0}, true);
  mscclpp::CudaDeviceGuard caller(1);
  PortSemaphore port(rawPort);
  int current = -1;
  MSCCLPP_CUDATHROW(cudaGetDevice(&current));
  EXPECT_EQ(current, 1);
  int allocated = -1;
  MSCCLPP_CUTHROW(cuPointerGetAttribute(&allocated, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                                        reinterpret_cast<CUdeviceptr>(port.deviceHandle().expectedInboundToken)));
  EXPECT_EQ(allocated, 0);
}
