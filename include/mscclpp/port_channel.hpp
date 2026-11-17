// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_PORT_CHANNEL_HPP_
#define MSCCLPP_PORT_CHANNEL_HPP_

#include <variant>

#include "core.hpp"
#include "port_channel_device.hpp"
#include "proxy.hpp"
#include "semaphore.hpp"

namespace mscclpp {

struct BasePortChannel;
struct PortChannel;

/// Base class for proxy services. Proxy services are used to proxy data between devices.
class BaseProxyService {
 public:
  BaseProxyService() = default;
  virtual ~BaseProxyService() = default;
  virtual void startProxy(bool blocking = false) = 0;
  virtual void stopProxy() = 0;
};

/// Proxy service implementation.
class ProxyService : public BaseProxyService {
 public:
  /// Constructor.
  /// @param fifoSize Size of the FIFO used by the proxy service (default: DEFAULT_FIFO_SIZE).
  ProxyService(int fifoSize = DEFAULT_FIFO_SIZE);

  /// Destructor. Stops the proxy before releasing the service state.
  ~ProxyService() noexcept override;

  /// Build and add a semaphore to the proxy service.
  /// @param connection The connection associated with the semaphore.
  /// @return The ID of the semaphore.
  SemaphoreId buildAndAddSemaphore(Communicator& communicator, const Connection& connection);

  /// Add a semaphore to the proxy service.
  /// @param semaphore The semaphore to be added
  /// @return The ID of the semaphore.
  SemaphoreId addSemaphore(const Semaphore& semaphore);

  /// Add a semaphore to the proxy service.
  /// @param semaphore The semaphore to be added
  /// @return The ID of the semaphore.
  SemaphoreId addSemaphore(std::shared_ptr<Host2DeviceSemaphore> semaphore);

  /// Register a memory region with the proxy service.
  /// @param memory The memory region to register.
  /// @return The ID of the memory region.
  MemoryId addMemory(RegisteredMemory memory);

  /// Get the next available memory ID.
  /// @param count The number of consecutive IDs required (default: 1).
  /// @return The first ID of an available range [first, first + count).
  MemoryId nextMemoryId(uint32_t count = 1) const;

  /// Get a semaphore by ID.
  /// @param id The ID of the semaphore.
  /// @return The semaphore.
  std::shared_ptr<Host2DeviceSemaphore> semaphore(SemaphoreId id) const;

  /// Get a base port channel by semaphore ID.
  /// @param id The ID of the semaphore.
  /// @return The base port channel.
  BasePortChannel basePortChannel(SemaphoreId id);

  /// Get a port channel by semaphore ID and memory regions.
  /// @param id The ID of the semaphore.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  /// @return The port channel.
  PortChannel portChannel(SemaphoreId id, MemoryId dst, MemoryId src);

  /// Start the proxy service.
  /// @param blocking Whether to block until the proxy thread has started (default: false).
  void startProxy(bool blocking = false);

  /// Stop the proxy service.
  void stopProxy();

 private:
  std::vector<std::shared_ptr<Host2DeviceSemaphore>> semaphores_;
  std::vector<RegisteredMemory> memories_;
  std::shared_ptr<Proxy> proxy_;
  std::unordered_map<std::shared_ptr<BaseConnection>, int> inflightRequests_;
  // Latest pending TriggerFlush FIFO position per connection. Proxy publishes pos+1 to the
  // connection's gpuFlushDonePos_ when the CQ drains, then erases the entry.
  std::unordered_map<std::shared_ptr<BaseConnection>, uint64_t> pendingFlushPos_;

  ProxyHandlerResult handleTrigger(ProxyTrigger triggerRaw);
  void progressFlushes();
  void drainPendingOperations();
};

/// Port channel without specifying source/destination memory regions.
struct BasePortChannel {
 private:
  SemaphoreId semaphoreId_;
  std::variant<std::shared_ptr<Host2DeviceSemaphore>, std::shared_ptr<Device2DeviceSemaphore>> semaphore_;
  std::shared_ptr<Proxy> proxy_;

  friend struct PortChannel;

 public:
  /// Constructor. Handle extraction throws InvalidUsage until a channel is assigned.
  BasePortChannel() noexcept : semaphoreId_(0) {}

  /// Constructor for a host-proxy channel.
  /// @param semaphoreId The ID of the semaphore.
  /// @param semaphore The non-null semaphore used to synchronize the communication.
  /// @param proxy The non-null proxy used for communication.
  /// @throws Error with ErrorCode::InvalidUsage for null state or a device-mode connection.
  BasePortChannel(SemaphoreId semaphoreId, std::shared_ptr<Host2DeviceSemaphore> semaphore,
                  std::shared_ptr<Proxy> proxy);

  /// Constructor for a channel in device mode (PortChannelBackend::Device).
  /// @param semaphore The non-null semaphore of a PortChannel device-mode connection.
  /// @throws Error with ErrorCode::InvalidUsage for a null semaphore, non-device-mode connection, or invalid remote
  /// token.
  explicit BasePortChannel(std::shared_ptr<Device2DeviceSemaphore> semaphore);

  /// Construct a Device channel from a raw semaphore, without a proxy.
  /// Creates an internal owner; channel copies share its expected counter.
  /// @throws Error with ErrorCode::InvalidUsage unless the connection is in Device mode.
  explicit BasePortChannel(const Semaphore& semaphore);

  /// Constructor for a host-proxy channel.
  /// @param semaphoreId The ID of the semaphore.
  /// @param semaphore The semaphore used to synchronize the communication.
  /// @param proxy The non-null proxy used for communication.
  BasePortChannel(SemaphoreId semaphoreId, const Semaphore& semaphore, std::shared_ptr<Proxy> proxy);

  /// Copy constructor. Shares the semaphore and proxy ownership.
  /// @param other The other BasePortChannel to copy from.
  BasePortChannel(const BasePortChannel& other) = default;

  /// Assignment operator.
  /// @param other The other BasePortChannel to assign from.
  BasePortChannel& operator=(const BasePortChannel& other) = default;

  /// Host-proxy device view, retained for the generic deviceHandle helper.
  using DeviceHandle = BasePortChannelDeviceHandle;

  /// Returns a Host-proxy view, valid while this channel remains alive.
  /// @throws Error with ErrorCode::InvalidUsage for uninitialized or Device-mode state.
  DeviceHandle deviceHandle() const;

  /// Returns a direct GPU-posting view, valid while this channel remains alive.
  /// @throws Error with ErrorCode::InvalidUsage for uninitialized or Host-mode state.
  BasePortChannelDeviceHandleT<PortChannelBackend::Device> deviceBackendHandle() const;
};

/// Port channel.
struct PortChannel : public BasePortChannel {
 private:
  MemoryId dst_;
  MemoryId src_;
  RegisteredMemory dstMemory_;
  RegisteredMemory srcMemory_;

 public:
  /// Constructor. Handle extraction throws InvalidUsage until a channel is assigned.
  PortChannel() noexcept : dst_(0), src_(0) {}

  /// Constructor for a host-proxy channel.
  /// @param semaphoreId The ID of the semaphore.
  /// @param semaphore The non-null semaphore.
  /// @param proxy The non-null proxy.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  PortChannel(SemaphoreId semaphoreId, std::shared_ptr<Host2DeviceSemaphore> semaphore, std::shared_ptr<Proxy> proxy,
              MemoryId dst, MemoryId src);

  /// Constructor for a channel in device mode (PortChannelBackend::Device).
  /// @param semaphore The non-null semaphore of a PortChannel device-mode connection.
  /// @param dst The destination registered memory.
  /// @param src The source registered memory.
  PortChannel(std::shared_ptr<Device2DeviceSemaphore> semaphore, RegisteredMemory dst, RegisteredMemory src);

  /// Construct a bound Device channel from the raw semaphore and registered regions, without a proxy.
  /// Registered memory retains metadata, not ownership of application buffers.
  PortChannel(const Semaphore& semaphore, RegisteredMemory dst, RegisteredMemory src);

  /// Constructor for a host-proxy channel.
  /// @param semaphoreId The ID of the semaphore.
  /// @param semaphore The semaphore.
  /// @param proxy The non-null proxy.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  PortChannel(SemaphoreId semaphoreId, const Semaphore& semaphore, std::shared_ptr<Proxy> proxy, MemoryId dst,
              MemoryId src);

  /// Copy constructor. Shares semaphore ownership and retains registered memories.
  /// @param other The other PortChannel to copy from.
  PortChannel(const PortChannel& other) = default;

  /// Assignment operator.
  /// @param other The other PortChannel to assign from.
  PortChannel& operator=(const PortChannel& other) = default;

  /// Host-proxy device view, retained for the generic deviceHandle helper.
  using DeviceHandle = PortChannelDeviceHandle;

  /// Returns a Host-proxy view. Keep this channel and its application buffers alive during GPU use.
  /// @throws Error with ErrorCode::InvalidUsage for uninitialized or Device-mode state.
  DeviceHandle deviceHandle() const;

  /// Returns a direct GPU-posting view. Keep this channel and its application buffers alive during GPU use.
  /// @throws Error with ErrorCode::InvalidUsage for uninitialized or Host-mode state.
  PortChannelDeviceHandleT<PortChannelBackend::Device> deviceBackendHandle() const;
};

/// @deprecated Use BasePortChannel instead.
[[deprecated("Use BasePortChannel instead.")]] typedef BasePortChannel BaseProxyChannel;

/// @deprecated Use PortChannel instead.
[[deprecated("Use PortChannel instead.")]] typedef PortChannel ProxyChannel;

}  // namespace mscclpp

#endif  // MSCCLPP_PORT_CHANNEL_HPP_
