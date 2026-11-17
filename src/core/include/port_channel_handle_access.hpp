// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_PORT_CHANNEL_HANDLE_ACCESS_HPP_
#define MSCCLPP_PORT_CHANNEL_HANDLE_ACCESS_HPP_

#include <mscclpp/errors.hpp>
#include <mscclpp/port_channel_device.hpp>

namespace mscclpp::detail {

// Host construction/binding/test access only. Not installed or part of the device API.
struct PortChannelHandleAccess {
  static BasePortChannelDeviceHandleT<PortChannelBackend::Device> unavailableDeviceForTest() {
    return BasePortChannelDeviceHandleT<PortChannelBackend::Device>(
        Host2DeviceSemaphoreDeviceHandle{}, GpuNetIoDeviceHandle{}, GpuNetIoMemoryDeviceHandle{});
  }
  static PortChannelDeviceHandleT<PortChannelBackend::Device> unavailableBoundDeviceForTest() {
    using Handle = PortChannelDeviceHandleT<PortChannelBackend::Device>;
    return Handle(unavailableDeviceForTest(),
                  Handle::DeviceBound{GpuNetIoMemoryDeviceHandle{0x1000, 17}, GpuNetIoMemoryDeviceHandle{0x2000, 19}});
  }
  static auto& host(BasePortChannelDeviceHandle& handle) { return handle.host_; }
  static const auto& host(const BasePortChannelDeviceHandle& handle) { return handle.host_; }
  static auto& hostReceive(BasePortChannelDeviceHandle& handle) { return handle.receive_; }
  template <PortChannelBackend Backend>
  static const auto& receive(const BasePortChannelDeviceHandleT<Backend>& handle) {
    return handle.receive_;
  }
  static const auto& device(const BasePortChannelDeviceHandleT<PortChannelBackend::Device>& handle) {
    return handle.device_;
  }
  static auto& hostBound(PortChannelDeviceHandle& handle) {
    // Writable Host field setup intentionally initializes the bound IDs, including zero IDs.
    handle.boundsInitialized_ = true;
    return handle.bound_;
  }
  static const auto& hostBound(const PortChannelDeviceHandle& handle) {
    if (!handle.boundsInitialized_) {
      throw Error("unassigned Host bound regions", ErrorCode::InvalidUsage);
    }
    return handle.bound_;
  }
  static const auto& deviceBound(const PortChannelDeviceHandleT<PortChannelBackend::Device>& handle) {
    return handle.bound_;
  }
};

}  // namespace mscclpp::detail

#endif  // MSCCLPP_PORT_CHANNEL_HANDLE_ACCESS_HPP_
