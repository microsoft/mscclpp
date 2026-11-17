// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Compile the actual public owner/handle API without the library target's private build context.
#ifdef USE_IBVERBS
#error "This probe must not inherit USE_IBVERBS"
#endif
#include <mscclpp/port_channel.hpp>
#include <type_traits>
#include <utility>

using Backend = mscclpp::PortChannelBackend;
using Host = mscclpp::PortChannel<Backend::Host>;
using Device = mscclpp::PortChannel<Backend::Device>;
using HostHandle = mscclpp::PortChannelDeviceHandle;
using DeviceHandle = mscclpp::PortChannelDeviceHandleT<Backend::Device>;
using DeviceBase = mscclpp::BasePortChannelDeviceHandleT<Backend::Device>;
using C = HostHandle::Concurrency;
constexpr HostHandle defaultHost;
constexpr DeviceHandle defaultDevice;
static_assert(defaultHost.src_ == 0 && defaultDevice.srcMemory_.addr == 0);
__constant__ HostHandle constantHost;
__constant__ DeviceHandle constantDevice;
static_assert(std::is_same_v<mscclpp::DeviceHandle<Host>, HostHandle>);
static_assert(std::is_same_v<mscclpp::DeviceHandle<Device>, DeviceHandle>);
static_assert(std::is_trivially_copyable_v<HostHandle> && std::is_trivially_copyable_v<DeviceHandle>);
static_assert(!std::is_assignable_v<Host&, Device>);
static_assert(
    !std::is_constructible_v<Device, const mscclpp::Semaphore&, mscclpp::RegisteredMemory, mscclpp::RegisteredMemory>);
static_assert(std::is_same_v<decltype(mscclpp::PortChannel(
                                 std::declval<std::shared_ptr<mscclpp::PortDevice2DeviceSemaphore>>(),
                                 std::declval<mscclpp::RegisteredMemory>(), std::declval<mscclpp::RegisteredMemory>())),
                             Device>);

template <class Handle>
__global__ void portAlgorithm(Handle channel) {
  channel.putWithSignal(0, 8);
  channel.flush();
}
void genericExtraction(const Host& host, const Device& device) {
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

// Define one macro at a time for independent expected compile failures.
#ifdef PROBE_HOST_EXCLUSIVE
__global__ void negative(HostHandle channel) { channel.signal<C::Exclusive>(); }
#endif
#ifdef PROBE_HOST_BLOCK
__global__ void negative(HostHandle channel) { channel.flush<C::Block>(); }
#endif
#ifdef PROBE_HOST_CAPABILITY
__global__ void negative(HostHandle channel) { channel.putWithAccumulate(0, 0, 8, 0, 1); }
#endif
#ifdef PROBE_DEVICE_IDS
__global__ void negative(DeviceBase channel) { channel.put(0, 0, 0, 0, 8); }
#endif
#ifdef PROBE_CROSS_BACKEND
void negative(mscclpp::BasePortChannelDeviceHandle& host, DeviceBase device) { host = device; }
#endif
#ifdef PROBE_SLOTS
__global__ void negative(DeviceHandle channel) { channel.signal<C::Exclusive, 1025>(); }
#endif
