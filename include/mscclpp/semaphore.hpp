// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_SEMAPHORE_HPP_
#define MSCCLPP_SEMAPHORE_HPP_

#include <memory>

#include "core.hpp"
#include "gpu_utils.hpp"
#include "semaphore_device.hpp"

namespace mscclpp {

/// Base class for semaphores.
///
/// A semaphore is a synchronization mechanism that allows the local peer to wait for the remote peer to complete a
/// data transfer. The local peer signals the remote peer that it has completed a data transfer by incrementing the
/// outbound semaphore ID. The incremented outbound semaphore ID is copied to the remote peer's inbound semaphore ID so
/// that the remote peer can wait for the local peer to complete a data transfer. Vice versa, the remote peer signals
/// the local peer that it has completed a data transfer by incrementing the remote peer's outbound semaphore ID and
/// copying the incremented value to the local peer's inbound semaphore ID.
///
/// @tparam InboundDeleter The deleter for inbound semaphore IDs. Either `std::default_delete` for host memory
/// or CudaDeleter for device memory.
/// @tparam OutboundDeleter The deleter for outbound semaphore IDs. Either `std::default_delete` for host memory
/// or CudaDeleter for device memory.
template <template <typename> typename InboundDeleter, template <typename> typename OutboundDeleter>
class BaseSemaphore {
 protected:
  /// The registered memory for the remote peer's inbound semaphore ID.
  std::shared_future<RegisteredMemory> remoteInboundSemaphoreIdsRegMem_;

  /// The inbound semaphore ID that is incremented by the remote peer and waited on by the local peer.
  /// The location can be either on the host or on the device.
  std::unique_ptr<uint64_t, InboundDeleter<uint64_t>> localInboundSemaphore_;

  /// The expected inbound semaphore ID to be incremented by the local peer and compared to the
  /// localInboundSemaphore_.
  /// The location can be either on the host or on the device.
  std::unique_ptr<uint64_t, InboundDeleter<uint64_t>> expectedInboundSemaphore_;

  /// The outbound semaphore ID that is incremented by the local peer and copied to the remote peer's
  /// localInboundSemaphore_.
  /// The location can be either on the host or on the device.
  std::unique_ptr<uint64_t, OutboundDeleter<uint64_t>> outboundSemaphore_;

 public:
  /// Constructs a BaseSemaphore.
  /// @param localInboundSemaphoreId The inbound semaphore ID
  /// @param expectedInboundSemaphoreId The expected inbound semaphore ID
  /// @param outboundSemaphoreId The outbound semaphore ID
  BaseSemaphore(std::unique_ptr<uint64_t, InboundDeleter<uint64_t>> localInboundSemaphoreId,
                std::unique_ptr<uint64_t, InboundDeleter<uint64_t>> expectedInboundSemaphoreId,
                std::unique_ptr<uint64_t, OutboundDeleter<uint64_t>> outboundSemaphoreId)
      : localInboundSemaphore_(std::move(localInboundSemaphoreId)),
        expectedInboundSemaphore_(std::move(expectedInboundSemaphoreId)),
        outboundSemaphore_(std::move(outboundSemaphoreId)) {}
};

/// Semaphore for sending signals from host to device.
class Host2DeviceSemaphore : public BaseSemaphore<detail::GpuDeleter, std::default_delete> {
 private:
  std::shared_ptr<Connection> connection_;

 public:
  /// Constructor.
  /// @param communicator The communicator.
  /// @param connection The connection associated with this semaphore.
  Host2DeviceSemaphore(Communicator& communicator, std::shared_ptr<Connection> connection);

  /// Returns the connection.
  /// @return The connection associated with this semaphore.
  std::shared_ptr<Connection> connection();

  /// Signal the device.
  void signal();

  /// Device-side handle for Host2DeviceSemaphore.
  using DeviceHandle = Host2DeviceSemaphoreDeviceHandle;

  /// Returns the device-side handle.
  DeviceHandle deviceHandle();
};

/// Semaphore for sending signals from local host to remote host.
class Host2HostSemaphore : public BaseSemaphore<std::default_delete, std::default_delete> {
 public:
  /// Constructor
  /// @param communicator The communicator.
  /// @param connection The connection associated with this semaphore. Transport::CudaIpc is not allowed.
  Host2HostSemaphore(Communicator& communicator, std::shared_ptr<Connection> connection);

  /// Returns the connection.
  /// @return The connection associated with this semaphore.
  std::shared_ptr<Connection> connection();

  /// Signal the remote host.
  void signal();

  /// Check if the remote host has signaled.
  /// @return true if the remote host has signaled.
  bool poll();

  /// Wait for the remote host to signal.
  /// @param maxSpinCount The maximum number of spin counts before throwing an exception. Never throws if negative.
  void wait(int64_t maxSpinCount = 10000000);

 private:
  std::shared_ptr<Connection> connection_;
};

/// Semaphore for sending signals from local device to peer device via GPU thread.
class MemoryDevice2DeviceSemaphore : public BaseSemaphore<detail::GpuDeleter, detail::GpuDeleter> {
 public:
  /// Constructor.
  /// @param communicator The communicator.
  /// @param connection The connection associated with this semaphore.
  MemoryDevice2DeviceSemaphore(Communicator& communicator, std::shared_ptr<Connection> connection);

  /// Constructor.
  MemoryDevice2DeviceSemaphore() = delete;

  /// Device-side handle for MemoryDevice2DeviceSemaphore.
  using DeviceHandle = MemoryDevice2DeviceSemaphoreDeviceHandle;

  /// Returns the device-side handle.
  DeviceHandle deviceHandle() const;

  bool isRemoteInboundSemaphoreIdSet_;
};

/// @deprecated Use MemoryDevice2DeviceSemaphore instead.
[[deprecated(
    "Use MemoryDevice2DeviceSemaphore instead.")]] typedef MemoryDevice2DeviceSemaphore SmDevice2DeviceSemaphore;

}  // namespace mscclpp

#endif  // MSCCLPP_SEMAPHORE_HPP_
