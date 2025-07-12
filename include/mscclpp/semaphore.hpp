// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_SEMAPHORE_HPP_
#define MSCCLPP_SEMAPHORE_HPP_

#include <memory>

#include "core.hpp"
#include "gpu_utils.hpp"
#include "semaphore_device.hpp"

namespace mscclpp {

/// A semaphore for sending signals from the host to the device.
class Host2DeviceSemaphore {
 private:
  Semaphore semaphore_;
  detail::UniqueGpuPtr<uint64_t> expectedInboundToken_;
  std::unique_ptr<uint64_t> outboundToken_;

 public:
  /// Constructor.
  /// @param semaphore
  Host2DeviceSemaphore(const Semaphore& semaphore);

  /// Constructor.
  /// @param communicator The communicator.
  /// @param connection The connection associated with this semaphore.
  Host2DeviceSemaphore(Communicator& communicator, std::shared_ptr<Connection> connection);

  /// Returns the connection.
  /// @return The connection associated with this semaphore.
  std::shared_ptr<Connection> connection() const;

  /// Signal the device.
  void signal();

  /// Device-side handle for Host2DeviceSemaphore.
  using DeviceHandle = Host2DeviceSemaphoreDeviceHandle;

  /// Returns the device-side handle.
  DeviceHandle deviceHandle() const;
};

/// A semaphore for sending signals from the local host to a remote host.
class Host2HostSemaphore {
 private:
  Semaphore semaphore_;
  std::unique_ptr<uint64_t> expectedInboundToken_;
  std::unique_ptr<uint64_t> outboundToken_;

 public:
  /// Constructor.
  /// @param semaphore
  Host2HostSemaphore(const Semaphore& semaphore);

  /// Constructor.
  /// @param communicator The communicator.
  /// @param connection The connection associated with this semaphore. Transport::CudaIpc is not allowed for
  /// Host2HostSemaphore.
  Host2HostSemaphore(Communicator& communicator, std::shared_ptr<Connection> connection);

  /// Returns the connection.
  /// @return The connection associated with this semaphore.
  std::shared_ptr<Connection> connection() const;

  /// Signal the remote host.
  void signal();

  /// Check if the remote host has signaled.
  /// @return true if the remote host has signaled.
  bool poll();

  /// Wait for the remote host to signal.
  /// @param maxSpinCount The maximum number of spin counts before throwing an exception. Never throws if negative.
  void wait(int64_t maxSpinCount = 10000000);
};

/// A semaphore for sending signals from the local device to a peer device via a GPU thread.
class MemoryDevice2DeviceSemaphore {
 private:
  Semaphore semaphore_;
  detail::UniqueGpuPtr<uint64_t> expectedInboundToken_;
  detail::UniqueGpuPtr<uint64_t> outboundToken_;

 public:
  /// Constructor.
  /// @param semaphore
  MemoryDevice2DeviceSemaphore(const Semaphore& semaphore);

  /// Constructor.
  /// @param communicator The communicator.
  /// @param connection The connection associated with this semaphore.
  MemoryDevice2DeviceSemaphore(Communicator& communicator, std::shared_ptr<Connection> connection);

  /// Returns the connection.
  /// @return The connection associated with this semaphore.
  std::shared_ptr<Connection> connection() const;

  /// Device-side handle for MemoryDevice2DeviceSemaphore.
  using DeviceHandle = MemoryDevice2DeviceSemaphoreDeviceHandle;

  /// Returns the device-side handle.
  DeviceHandle deviceHandle() const;
};

/// @deprecated Use MemoryDevice2DeviceSemaphore instead.
[[deprecated(
    "Use MemoryDevice2DeviceSemaphore instead.")]] typedef MemoryDevice2DeviceSemaphore SmDevice2DeviceSemaphore;

}  // namespace mscclpp

#endif  // MSCCLPP_SEMAPHORE_HPP_
