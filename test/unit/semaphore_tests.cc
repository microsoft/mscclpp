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
using Channel = mscclpp::PortChannel;
using Base = mscclpp::BasePortChannel;
using PortSemaphore = mscclpp::PortDevice2DeviceSemaphore;
static_assert(std::is_same_v<mscclpp::BaseMemoryChannel::DeviceHandle, mscclpp::BaseMemoryChannelDeviceHandle>);
static_assert(std::is_same_v<mscclpp::MemoryChannel::DeviceHandle, mscclpp::MemoryChannelDeviceHandle>);
static_assert(std::is_same_v<mscclpp::SwitchChannel::DeviceHandle, mscclpp::SwitchChannelDeviceHandle>);
static_assert(std::is_same_v<Base::DeviceHandle<Backend::Host>, mscclpp::BasePortChannelDeviceHandle<Backend::Host>>);
static_assert(
    std::is_same_v<Channel::DeviceHandle<Backend::Device>, mscclpp::PortChannelDeviceHandle<Backend::Device>>);
static_assert(std::is_same_v<decltype(std::declval<const Channel&>().deviceHandle<Backend::Host>()),
                             mscclpp::PortChannelDeviceHandle<Backend::Host>>);
static_assert(std::is_same_v<decltype(std::declval<const Channel&>().deviceHandle<Backend::Device>()),
                             mscclpp::PortChannelDeviceHandle<Backend::Device>>);
static_assert(std::is_constructible_v<Channel, mscclpp::SemaphoreId, std::shared_ptr<mscclpp::Host2DeviceSemaphore>,
                                      std::shared_ptr<mscclpp::Proxy>, mscclpp::MemoryId, mscclpp::MemoryId>);
static_assert(std::is_constructible_v<Channel, std::shared_ptr<PortSemaphore>, mscclpp::RegisteredMemory,
                                      mscclpp::RegisteredMemory>);
static_assert(
    !std::is_constructible_v<Channel, const mscclpp::Semaphore&, mscclpp::RegisteredMemory, mscclpp::RegisteredMemory>);
static_assert(std::is_copy_assignable_v<Channel>);  // One owner type can hold either backend.
static_assert(!std::is_assignable_v<mscclpp::PortChannelDeviceHandle<Backend::Host>&,
                                    mscclpp::PortChannelDeviceHandle<Backend::Device>>);
template <class T>
concept NoArgumentMember = requires(const T& owner) { owner.deviceHandle(); };
template <class T>
concept GenericExtraction = requires(const T& owner) { mscclpp::deviceHandle(owner); };
template <class T>
concept GenericAssociatedHandle = requires { typename mscclpp::DeviceHandle<T>; };
static_assert(!NoArgumentMember<Channel> && !NoArgumentMember<Base>);
static_assert(!GenericExtraction<Channel> && !GenericExtraction<Base>);
static_assert(!GenericAssociatedHandle<Channel> && !GenericAssociatedHandle<Base>);
static_assert(GenericExtraction<mscclpp::MemoryChannel> && GenericExtraction<mscclpp::SwitchChannel>);
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
  EXPECT_TRUE(rejectsInvalidUsage([&] { Channel invalid(owner, {}, raw.localMemory()); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { Channel invalid(owner, raw.remoteMemory(), {}); }));
  auto buffer = mscclpp::GpuBuffer(64).memory();
  auto region = context->registerMemory(buffer.get(), 64, mscclpp::Transport::Ethernet);
  Channel channel(owner, region, region);
  region = {};  // The Device binding retains its own registration metadata.
  auto copy = channel;
  Base baseCopy = channel;
  std::weak_ptr<PortSemaphore> weak = owner;
  owner.reset();
  channel = {};
  EXPECT_TRUE(!weak.expired());
  EXPECT_EQ(copy.deviceHandle<Backend::Device>().semaphore_.expectedInboundToken, expected);
  EXPECT_EQ(baseCopy.deviceHandle<Backend::Device>().semaphore_.expectedInboundToken, expected);
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
  Channel liveChannel(liveOwner, raw.remoteMemory(), raw.localMemory());
  Base liveBase = liveChannel;
  const auto liveExpected = liveOwner->deviceHandle().expectedInboundToken;
  auto movedOwner = std::move(*liveOwner);
  EXPECT_EQ(movedOwner.deviceHandle().expectedInboundToken, liveExpected);
  // Extraction must reject before accessing the moved-from raw semaphore's connection.
  EXPECT_TRUE(rejectsInvalidUsage([&] { liveBase.deviceHandle<Backend::Device>(); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { liveChannel.deviceHandle<Backend::Device>(); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { Base invalid(liveOwner); }));
}

TEST(SemaphoreValidationTest, ExplicitBackendAndIndependentBindingState) {
  auto rawHost = makeSemaphore({mscclpp::DeviceType::GPU, 0}, false);
  mscclpp::ProxyService service;
  auto hostSemaphore = std::make_shared<mscclpp::Host2DeviceSemaphore>(rawHost);
  auto sid = service.addSemaphore(hostSemaphore);
  EXPECT_EQ(service.addMemory(rawHost.localMemory()), 0u);
  Channel host = service.portChannel(sid, 0, 0);
  Base hostBase = service.basePortChannel(sid);
  EXPECT_EQ(host.deviceHandle<Backend::Host>().dst_, 0u);
  EXPECT_EQ(mscclpp::deviceHandle<Backend::Host>(host).dst_, 0u);
  EXPECT_EQ(mscclpp::deviceHandle<Backend::Host>(hostBase).semaphoreId_, sid);
  EXPECT_TRUE(rejectsInvalidUsage([&] { host.deviceHandle<Backend::Device>(); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { hostBase.deviceHandle<Backend::Device>(); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::deviceHandle<Backend::Device>(host); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::deviceHandle<Backend::Device>(hostBase); }));
  auto rawDevice = makeSemaphore({mscclpp::DeviceType::GPU, 0}, true);
  auto deviceSemaphore = std::make_shared<PortSemaphore>(rawDevice);
  Channel device(deviceSemaphore, rawDevice.remoteMemory(), rawDevice.localMemory());
  Base deviceBase = device;
  EXPECT_TRUE(rejectsInvalidUsage([&] { device.deviceHandle<Backend::Host>(); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { deviceBase.deviceHandle<Backend::Host>(); }));
  EXPECT_EQ(mscclpp::deviceHandle<Backend::Device>(device).semaphore_.expectedInboundToken,
            deviceSemaphore->deviceHandle().expectedInboundToken);
  EXPECT_EQ(mscclpp::deviceHandle<Backend::Device>(deviceBase).semaphore_.expectedInboundToken,
            deviceSemaphore->deviceHandle().expectedInboundToken);
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::deviceHandle<Backend::Host>(device); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::deviceHandle<Backend::Host>(deviceBase); }));
  Channel unassigned;
  Base unassignedBase;
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::deviceHandle<Backend::Host>(unassigned); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::deviceHandle<Backend::Device>(unassigned); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::deviceHandle<Backend::Host>(unassignedBase); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::deviceHandle<Backend::Device>(unassignedBase); }));
  Channel unboundHost, unboundDevice;
  static_cast<Base&>(unboundHost) = hostBase;
  static_cast<Base&>(unboundDevice) = deviceBase;
  EXPECT_TRUE(rejectsInvalidUsage([&] { unboundHost.deviceHandle<Backend::Host>(); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { unboundDevice.deviceHandle<Backend::Device>(); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::deviceHandle<Backend::Host>(unboundHost); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::deviceHandle<Backend::Device>(unboundDevice); }));
  Channel hostWithDeviceBase = host, deviceWithHostBase = device;
  static_cast<Base&>(hostWithDeviceBase) = deviceBase;
  static_cast<Base&>(deviceWithHostBase) = hostBase;
  EXPECT_TRUE(rejectsInvalidUsage([&] { hostWithDeviceBase.deviceHandle<Backend::Device>(); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { deviceWithHostBase.deviceHandle<Backend::Host>(); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::deviceHandle<Backend::Device>(hostWithDeviceBase); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::deviceHandle<Backend::Host>(deviceWithHostBase); }));
  Channel selected = host;
  selected = device;  // Whole-owner cross-backend assignment is now supported, not a compile failure.
  EXPECT_EQ(selected.deviceHandle<Backend::Device>().semaphore_.expectedInboundToken,
            deviceSemaphore->deviceHandle().expectedInboundToken);
  EXPECT_TRUE(rejectsInvalidUsage([&] { selected.deviceHandle<Backend::Host>(); }));
  selected = host;
  auto movedOwner = std::move(selected);
  EXPECT_EQ(movedOwner.deviceHandle<Backend::Host>().src_, 0u);
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
