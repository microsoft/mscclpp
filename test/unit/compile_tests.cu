// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "../framework.hpp"

#undef NDEBUG
#ifndef DEBUG_BUILD
#define DEBUG_BUILD
#endif  // DEBUG_BUILD
#include <assert.h>

#include <mscclpp/poll_device.hpp>
#include <mscclpp/port_channel.hpp>
#include <type_traits>

TEST(CompileTest, Assert) { assert(true); }

namespace port_channel_compile_test {
using Backend = mscclpp::PortChannelBackend;
using HostHandle = mscclpp::PortChannelDeviceHandle<>;
using DeviceHandle = mscclpp::PortChannelDeviceHandle<Backend::Device>;
using DeviceBase = mscclpp::BasePortChannelDeviceHandle<Backend::Device>;
using C = HostHandle::Concurrency;
constexpr HostHandle defaultHost;
constexpr DeviceHandle defaultDevice;
constexpr mscclpp::BasePortChannelDeviceHandle<> defaultHostBase;
constexpr DeviceBase defaultDeviceBase;
static_assert(defaultHost.src_ == 0 && defaultDevice.srcMemory_.addr == 0);
static_assert(defaultHostBase.semaphoreId_ == 0 && defaultDeviceBase.device_.qp == nullptr);
static_assert(!std::is_assignable_v<mscclpp::BasePortChannelDeviceHandle<>&, DeviceBase>);
static_assert(!std::is_assignable_v<DeviceBase&, mscclpp::BasePortChannelDeviceHandle<>>);

template <class Handle>
concept HasCombinedPut = requires(Handle channel) { channel.putWithAccumulate(0, 0, 8, 0, 1); };
template <class Handle>
concept HasMemoryIdPut = requires(Handle channel) {
  channel.put(mscclpp::MemoryId{}, uint64_t{}, mscclpp::MemoryId{}, uint64_t{}, uint64_t{});
};
static_assert(!HasCombinedPut<HostHandle> && HasCombinedPut<DeviceHandle>);
static_assert(HasMemoryIdPut<mscclpp::BasePortChannelDeviceHandle<>> && !HasMemoryIdPut<DeviceBase>);

__constant__ HostHandle constantHost;
__constant__ DeviceHandle constantDevice;
__constant__ mscclpp::BasePortChannelDeviceHandle<> constantHostBase;
__constant__ DeviceBase constantDeviceBase;

// Compiled but not launched by the test runner; these operations require live channels.
template <class Handle>
__global__ void portAlgorithm(Handle channel) {
  channel.putWithSignal(0, 8);
  channel.flush();
}
void genericExtraction(const mscclpp::PortChannel<>& host, const mscclpp::PortChannel<Backend::Device>& device) {
  portAlgorithm<<<1, 1>>>(mscclpp::deviceHandle(host));
  portAlgorithm<<<1, 1>>>(mscclpp::deviceHandle(device));
}

template <C Mode, uint32_t Capacity>
__global__ void deviceCapabilities(DeviceBase base, DeviceHandle bound, mscclpp::DeviceMemoryHandle dst,
                                   mscclpp::DeviceMemoryHandle src) {
  base.put<Mode, Capacity>(dst, 0, src, 0, 8);
  base.putWithSignal<Mode, Capacity>(dst, src, 0, 8);
  bound.putWithAccumulate<Mode, Capacity>(0, 0, 8, 0, 1);
  bound.signal<Mode, Capacity>();
  bound.flush<Mode>();
}
template __global__ void deviceCapabilities<C::Exclusive, 0>(DeviceBase, DeviceHandle, mscclpp::DeviceMemoryHandle,
                                                             mscclpp::DeviceMemoryHandle);
template __global__ void deviceCapabilities<C::Exclusive, 8>(DeviceBase, DeviceHandle, mscclpp::DeviceMemoryHandle,
                                                             mscclpp::DeviceMemoryHandle);
template __global__ void deviceCapabilities<C::Block, 32>(DeviceBase, DeviceHandle, mscclpp::DeviceMemoryHandle,
                                                          mscclpp::DeviceMemoryHandle);
template __global__ void deviceCapabilities<C::Device, 1024>(DeviceBase, DeviceHandle, mscclpp::DeviceMemoryHandle,
                                                             mscclpp::DeviceMemoryHandle);
}  // namespace port_channel_compile_test
