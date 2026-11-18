// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <chrono>
#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/port_channel.hpp>
#include <thread>
#include <type_traits>
#include <vector>

#include "../framework.hpp"

namespace {
template <class Func>
bool rejectsInvalidUsage(Func&& func) {
  try {
    func();
  } catch (const mscclpp::Error& error) {
    return error.getErrorCode() == mscclpp::ErrorCode::InvalidUsage;
  }
  return false;
}
}  // namespace

TEST(ProxyServiceTest, UnassignedChannels) {
  using Backend = mscclpp::PortChannelBackend;
  mscclpp::BasePortChannel<> hostBase;
  mscclpp::BasePortChannel<Backend::Device> deviceBase;
  mscclpp::PortChannel<> host;
  mscclpp::PortChannel<Backend::Device> device;
  auto hostCopy = host;
  auto deviceCopy = device;
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::deviceHandle(hostBase); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::deviceHandle(deviceBase); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::deviceHandle(hostCopy); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] { mscclpp::deviceHandle(deviceCopy); }));
  EXPECT_TRUE(rejectsInvalidUsage(
      [&] { mscclpp::BasePortChannel<> invalid(0, std::shared_ptr<mscclpp::Host2DeviceSemaphore>{}, nullptr); }));
  EXPECT_TRUE(rejectsInvalidUsage([&] {
    mscclpp::BasePortChannel<Backend::Device> invalid(std::shared_ptr<mscclpp::PortDevice2DeviceSemaphore>{});
  }));
}

TEST(ProxyServiceTest, PlainHandleDefaults) {
  using Host = mscclpp::PortChannelDeviceHandle<>;
  using Device = mscclpp::PortChannelDeviceHandle<mscclpp::PortChannelBackend::Device>;
  static_assert(std::is_trivially_copyable_v<Host> && std::is_trivially_copyable_v<Device>);
  Host host;
  Device device;
  EXPECT_EQ(host.semaphoreId_, 0u);
  EXPECT_EQ(host.src_, 0u);
  EXPECT_EQ(host.dst_, 0u);
  EXPECT_TRUE(host.fifo_.head == nullptr && host.flushDonePos_ == nullptr);
  EXPECT_TRUE(device.device_.qp == nullptr);
  EXPECT_EQ(device.srcMemory_.addr, uint64_t{0});
  EXPECT_EQ(device.dstMemory_.addr, uint64_t{0});
  // Public writable fields remain usable, including registry ID0.
  host.src_ = 0;
  host.dst_ = 0;
  auto copy = host;
  EXPECT_EQ(copy.src_, 0u);
}

__global__ void kernelProxyServiceTransfer(mscclpp::PortChannelDeviceHandle<> channel, size_t bytes) {
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
    auto channel = std::make_unique<mscclpp::PortChannel<>>(service->portChannel(semaphoreId, dstId, srcId));
    const auto zero = service->portChannel(semaphoreId, 0, 0).deviceHandle();
    EXPECT_EQ(zero.src_, 0u);
    EXPECT_EQ(zero.dst_, 0u);
    EXPECT_TRUE(rejectsInvalidUsage(
        [&] { mscclpp::BasePortChannel<> invalid(semaphoreId, service->semaphore(semaphoreId), nullptr); }));
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
