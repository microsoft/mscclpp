// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_NVLS_DEVICE_HPP_
#define MSCCLPP_NVLS_DEVICE_HPP_

namespace mscclpp {

/// Device-side handle for @ref Host2DeviceSemaphore.
struct DeviceMulticastPointerDeviceHandle {
  void* devicePtr;
  void* mcPtr;
  size_t bufferSize;
};

}  // namespace mscclpp

#endif  // MSCCLPP_SEMAPHORE_DEVICE_HPP_
