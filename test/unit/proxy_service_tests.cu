// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <chrono>
#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/port_channel.hpp>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "../framework.hpp"
#include "port_channel_handle_access.hpp"

using Access = mscclpp::detail::PortChannelHandleAccess;

static_assert(std::is_nothrow_default_constructible_v<mscclpp::BasePortChannel>);
static_assert(std::is_nothrow_default_constructible_v<mscclpp::PortChannel>);
static_assert(std::is_same_v<mscclpp::DeviceHandle<mscclpp::BasePortChannel>, mscclpp::BasePortChannelDeviceHandle>);
static_assert(std::is_same_v<mscclpp::DeviceHandle<mscclpp::PortChannel>, mscclpp::PortChannelDeviceHandle>);
static_assert(std::is_same_v<decltype(&mscclpp::BasePortChannel::deviceHandle),
                             mscclpp::BasePortChannelDeviceHandle (mscclpp::BasePortChannel::*)() const>);
static_assert(std::is_same_v<decltype(&mscclpp::PortChannel::deviceHandle),
                             mscclpp::PortChannelDeviceHandle (mscclpp::PortChannel::*)() const>);

namespace {
template <typename Func>
bool rejectsInvalidUsage(Func func) {
  try {
    func();
  } catch (const mscclpp::Error& e) {
    return e.getErrorCode() == mscclpp::ErrorCode::InvalidUsage;
  }
  return false;
}
}  // namespace

static_assert(std::is_trivially_copy_assignable_v<mscclpp::BasePortChannelDeviceHandle>);
static_assert(std::is_trivially_copy_assignable_v<mscclpp::PortChannelDeviceHandle>);
static_assert(std::is_trivially_move_assignable_v<mscclpp::PortChannelDeviceHandle>);

using DeviceBase = mscclpp::BasePortChannelDeviceHandleT<mscclpp::PortChannelBackend::Device>;
using DeviceBound = mscclpp::PortChannelDeviceHandleT<mscclpp::PortChannelBackend::Device>;
static_assert(!std::is_assignable_v<mscclpp::BasePortChannelDeviceHandle&, DeviceBase>);
static_assert(!std::is_assignable_v<DeviceBase&, mscclpp::BasePortChannelDeviceHandle>);
static_assert(!std::is_assignable_v<mscclpp::PortChannelDeviceHandle&, DeviceBound>);
static_assert(!std::is_assignable_v<DeviceBound&, mscclpp::PortChannelDeviceHandle>);

TEST(PortChannelHandleTest, TypedCopyMovePreservesBounds) {
  const mscclpp::PortChannelDeviceHandle host(3, {}, {}, 11, 13, nullptr);
  auto hostCopy = host;
  auto hostMoved = std::move(hostCopy);
  EXPECT_EQ(Access::hostBound(hostMoved).dst, 11u);
  EXPECT_EQ(Access::hostBound(hostMoved).src, 13u);
  const auto device = Access::unavailableBoundDeviceForTest();
  auto copy = device;
  DeviceBound assigned;
  assigned = copy;
  auto moved = std::move(copy);
  DeviceBound moveAssigned;
  moveAssigned = std::move(assigned);
  for (const auto* roundtrip : {&moved, &moveAssigned}) {
    EXPECT_EQ(Access::deviceBound(*roundtrip).dst.addr, uint64_t{0x1000});
    EXPECT_EQ(Access::deviceBound(*roundtrip).src.key, 19u);
  }
}

TEST(PortChannelHandleTest, HostBoundsRequireIndependentInitialization) {
  using Base = mscclpp::BasePortChannelDeviceHandle;
  using Bound = mscclpp::PortChannelDeviceHandle;
  Bound unbound;
  EXPECT_TRUE(rejectsInvalidUsage([&] { Access::hostBound(std::as_const(unbound)); }));
  static_cast<Base&>(unbound) = Base(3, {}, {}, nullptr);
  EXPECT_TRUE(rejectsInvalidUsage([&] { Access::hostBound(std::as_const(unbound)); }));
  auto copy = unbound;
  Bound assigned;
  assigned = copy;
  auto moved = std::move(copy);
  Bound moveAssigned;
  moveAssigned = std::move(assigned);
  for (const auto* invalid : {&unbound, &moved, &moveAssigned}) {
    EXPECT_TRUE(rejectsInvalidUsage([&] { Access::hostBound(std::as_const(*invalid)); }));
  }

  const Bound zero(0, {}, {}, 0, 0, nullptr);
  auto zeroCopy = zero;
  assigned = zeroCopy;
  moved = std::move(zeroCopy);
  moveAssigned = std::move(assigned);
  for (const auto* valid : {&moved, &moveAssigned}) {
    EXPECT_EQ(Access::hostBound(std::as_const(*valid)).dst, 0u);
    EXPECT_EQ(Access::hostBound(std::as_const(*valid)).src, 0u);
  }
}

TEST(PortChannelHandleTest, DefaultHostProperties) {
  for (bool dstFirst : {false, true}) {
    mscclpp::PortChannelDeviceHandle handle;
    Access::host(handle).id = 7;
    EXPECT_TRUE(rejectsInvalidUsage([&] { Access::hostBound(std::as_const(handle)); }));
    // Match either Python setter: writable access intentionally initializes zero IDs too.
    if (dstFirst)
      Access::hostBound(handle).dst = 0;
    else
      Access::hostBound(handle).src = 0;
    EXPECT_EQ(Access::host(std::as_const(handle)).id, 7u);
    EXPECT_EQ(Access::hostBound(std::as_const(handle)).dst, 0u);
    EXPECT_EQ(Access::hostBound(std::as_const(handle)).src, 0u);
  }
}

TEST(ProxyServiceTest, InvalidChannels) {
  mscclpp::BasePortChannel base;
  mscclpp::PortChannel channel;
  EXPECT_TRUE(rejectsInvalidUsage([&] { base.deviceHandle(); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { channel.deviceHandle(); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { base.deviceBackendHandle(); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { channel.deviceBackendHandle(); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::deviceHandle(base); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::deviceHandle(channel); }));

  mscclpp::Semaphore raw;
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::BasePortChannel invalid(raw); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::PortChannel invalid(raw, {}, {}); }));

  std::shared_ptr<mscclpp::Host2DeviceSemaphore> hostSemaphore;
  std::shared_ptr<mscclpp::Device2DeviceSemaphore> deviceSemaphore;
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::BasePortChannel invalid(0, hostSemaphore, nullptr); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::PortChannel invalid(0, hostSemaphore, nullptr, 0, 0); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::BasePortChannel invalid(deviceSemaphore); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::PortChannel invalid(deviceSemaphore, {}, {}); }));
}

TEST(ProxyServiceTest, HostChannelHandlesAndOwnership) {
  MSCCLPP_CUDATHROW(cudaSetDevice(0));
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(0, 1);
  bootstrap->initialize(mscclpp::TcpBootstrap::createUniqueId());
  mscclpp::Communicator communicator(bootstrap);
  auto connection = communicator.connect(mscclpp::Transport::CudaIpc, 0).get();
  auto service = std::make_unique<mscclpp::ProxyService>();
  auto id = service->buildAndAddSemaphore(communicator, connection);
  auto semaphore = service->semaphore(id);
  std::weak_ptr<mscclpp::Host2DeviceSemaphore> weakSemaphore = semaphore;
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::BasePortChannel invalid(id, semaphore, nullptr); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::PortChannel invalid(id, semaphore, nullptr, 0, 0); }));

  auto base = service->basePortChannel(id);
  auto channel = service->portChannel(id, 17, 23);
  const auto baseHandle = mscclpp::deviceHandle(base);
  const auto handle = mscclpp::deviceHandle(channel);
  const auto zeroHandle = service->portChannel(id, 0, 0).deviceHandle();
  EXPECT_EQ(Access::hostBound(zeroHandle).dst, 0u);
  EXPECT_EQ(Access::hostBound(zeroHandle).src, 0u);
  EXPECT_EQ(Access::host(handle).id, id);
  EXPECT_EQ(Access::hostBound(handle).dst, 17u);
  EXPECT_EQ(Access::hostBound(handle).src, 23u);
  EXPECT_EQ(Access::receive(baseHandle).expectedInboundToken, Access::receive(handle).expectedInboundToken);
  EXPECT_TRUE(rejectsInvalidUsage([&] { base.deviceBackendHandle(); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { channel.deviceBackendHandle(); }));
  auto raw = communicator.buildSemaphore(connection, 0, 87).get();
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::BasePortChannel invalid(raw); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::PortChannel invalid(raw, {}, {}); }));

  auto baseCopy = base;
  auto channelCopy = channel;
  base = {};
  channel = {};
  semaphore.reset();
  service.reset();
  EXPECT_FALSE(weakSemaphore.expired());
  EXPECT_EQ(Access::receive(baseCopy.deviceHandle()).expectedInboundToken,
            Access::receive(baseHandle).expectedInboundToken);
  EXPECT_EQ(Access::receive(channelCopy.deviceHandle()).expectedInboundToken,
            Access::receive(handle).expectedInboundToken);
  EXPECT_EQ(Access::hostBound(channelCopy.deviceHandle()).dst, Access::hostBound(handle).dst);
  EXPECT_EQ(Access::hostBound(channelCopy.deviceHandle()).src, Access::hostBound(handle).src);
  baseCopy = {};
  EXPECT_FALSE(weakSemaphore.expired());
  channelCopy = {};
  EXPECT_TRUE(weakSemaphore.expired());
}

__global__ void kernelProxyServiceTransfer(mscclpp::PortChannelDeviceHandle channel, size_t bytes) {
  channel.putWithSignalAndFlush(0, 0, bytes);
  channel.wait();
}

TEST(ProxyServiceTest, Lifecycle) {
  MSCCLPP_CUDATHROW(cudaSetDevice(0));
  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(0, 1);
  bootstrap->initialize(mscclpp::TcpBootstrap::createUniqueId());
  mscclpp::Communicator communicator(bootstrap);
  auto connection = communicator.connect(mscclpp::Transport::CudaIpc, 0).get();
  constexpr size_t count = 1024;
  constexpr size_t bytes = count * sizeof(int);
  auto src = mscclpp::GpuBuffer(bytes).memory();
  auto dst = mscclpp::GpuBuffer(bytes).memory();
  auto srcMemory = communicator.registerMemory(src.get(), bytes, mscclpp::Transport::CudaIpc);
  auto dstMemory = communicator.registerMemory(dst.get(), bytes, mscclpp::Transport::CudaIpc);
  std::vector<int> expected(count, 777);
  std::vector<int> actual(count);
  MSCCLPP_CUDATHROW(cudaMemcpy(src.get(), expected.data(), bytes, cudaMemcpyHostToDevice));

  {
    mscclpp::ProxyService service;
    service.stopProxy();
    service.stopProxy();
  }

  for (int iteration = 0; iteration < 12; ++iteration) {
    auto service = std::make_unique<mscclpp::ProxyService>();
    auto srcId = service->addMemory(srcMemory);
    auto dstId = service->addMemory(dstMemory);
    auto semaphoreId = service->buildAndAddSemaphore(communicator, connection);
    auto original = service->portChannel(semaphoreId, dstId, srcId);
    auto channel = std::make_unique<mscclpp::PortChannel>(original);
    original = {};
    for (int run = 0; run < 2; ++run) {
      MSCCLPP_CUDATHROW(cudaMemset(dst.get(), 0, bytes));
      service->startProxy(true);
      kernelProxyServiceTransfer<<<1, 1>>>(channel->deviceHandle(), bytes);
      MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
      MSCCLPP_CUDATHROW(cudaMemcpy(actual.data(), dst.get(), bytes, cudaMemcpyDeviceToHost));
      EXPECT_TRUE(actual == expected);
      if (run == 0) {
        service->stopProxy();
        service->stopProxy();
      }
    }
    if (iteration % 3 == 0) {
      service->stopProxy();
      service->stopProxy();
      service.reset();
    } else if (iteration % 3 == 1) {
      channel.reset();
      service.reset();
    } else {
      // Retain the channel while destroying the service through its virtual base.
      std::unique_ptr<mscclpp::BaseProxyService> base = std::move(service);
      base.reset();
      // Give stale callbacks time to execute under host memory sanitizers.
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      channel.reset();
    }
  }
}
