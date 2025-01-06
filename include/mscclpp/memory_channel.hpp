// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_MEMORY_CHANNEL_HPP_
#define MSCCLPP_MEMORY_CHANNEL_HPP_

#include <type_traits>

#include "core.hpp"
#include "memory_channel_device.hpp"
#include "semaphore.hpp"

namespace mscclpp {

/// Channel for accessing peer memory directly from GPU threads.
struct MemoryChannel {
 private:
  std::shared_ptr<MemoryDevice2DeviceSemaphore> semaphore_;
  RegisteredMemory dst_;
  void* src_;
  void* getPacketBuffer_;

 public:
  /// Constructor.
  MemoryChannel() = default;

  /// Constructor.
  /// @param semaphore The semaphore used to synchronize the communication.
  /// @param dst Registered memory of the destination.
  /// @param src The source memory address.
  /// @param getPacketBuffer The optional buffer used for @ref getPackets().
  MemoryChannel(std::shared_ptr<MemoryDevice2DeviceSemaphore> semaphore, RegisteredMemory dst, void* src,
                void* getPacketBuffer = nullptr);

  /// Device-side handle for @ref MemoryChannel.
  using DeviceHandle = MemoryChannelDeviceHandle;

  /// Returns the device-side handle.
  ///
  /// User should make sure the MemoryChannel is not released when using the returned handle.
  ///
  DeviceHandle deviceHandle() const;
};

/// @deprecated Use @ref MemoryChannel instead.
[[deprecated("Use MemoryChannel instead.")]] typedef MemoryChannel SmChannel;

}  // namespace mscclpp

#endif  // MSCCLPP_MEMORY_CHANNEL_HPP_
