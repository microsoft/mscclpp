// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_PORT_CHANNEL_HPP_
#define MSCCLPP_PORT_CHANNEL_HPP_

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
  virtual void startProxy() = 0;
  virtual void stopProxy() = 0;
};

/// Proxy service implementation.
class ProxyService : public BaseProxyService {
 public:
  /// Constructor.
  ProxyService(size_t fifoSize = DEFAULT_FIFO_SIZE);

  /// Build and add a semaphore to the proxy service.
  /// @param connection The connection associated with the semaphore.
  /// @return The ID of the semaphore.
  SemaphoreId buildAndAddSemaphore(Communicator& communicator, std::shared_ptr<Connection> connection);

  /// Add a semaphore to the proxy service.
  /// @param semaphore The semaphore to be added
  /// @return The ID of the semaphore.
  SemaphoreId addSemaphore(std::shared_ptr<Host2DeviceSemaphore> semaphore);

  /// Register a memory region with the proxy service.
  /// @param memory The memory region to register.
  /// @return The ID of the memory region.
  MemoryId addMemory(RegisteredMemory memory);

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
  void startProxy();

  /// Stop the proxy service.
  void stopProxy();

 private:
  std::vector<std::shared_ptr<Host2DeviceSemaphore>> semaphores_;
  std::vector<RegisteredMemory> memories_;
  std::shared_ptr<Proxy> proxy_;
  int deviceNumaNode;
  std::unordered_map<std::shared_ptr<Connection>, int> inflightRequests;

  void bindThread();

  ProxyHandlerResult handleTrigger(ProxyTrigger triggerRaw);
};

/// Port channel without specifying source/destination memory regions.
struct BasePortChannel {
 protected:
  SemaphoreId semaphoreId_;

  std::shared_ptr<Host2DeviceSemaphore> semaphore_;

  std::shared_ptr<Proxy> proxy_;

 public:
  BasePortChannel() = default;

  BasePortChannel(SemaphoreId semaphoreId, std::shared_ptr<Host2DeviceSemaphore> semaphore,
                  std::shared_ptr<Proxy> proxy);

  BasePortChannel(const BasePortChannel& other) = default;

  BasePortChannel& operator=(BasePortChannel& other) = default;

  /// Device-side handle for @ref BasePortChannel.
  using DeviceHandle = BasePortChannelDeviceHandle;

  /// Returns the device-side handle.
  ///
  /// User should make sure the BasePortChannel is not released when using the returned handle.
  ///
  DeviceHandle deviceHandle() const;
};

/// Port channel.
struct PortChannel : public BasePortChannel {
 private:
  MemoryId dst_;
  MemoryId src_;

 public:
  /// Default constructor.
  PortChannel() = default;

  /// Constructor.
  /// @param semaphoreId The ID of the semaphore.
  /// @param semaphore The semaphore.
  /// @param proxy The proxy.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  PortChannel(SemaphoreId semaphoreId, std::shared_ptr<Host2DeviceSemaphore> semaphore, std::shared_ptr<Proxy> proxy,
              MemoryId dst, MemoryId src);

  /// Copy constructor.
  PortChannel(const PortChannel& other) = default;

  /// Assignment operator.
  PortChannel& operator=(PortChannel& other) = default;

  /// Device-side handle for @ref PortChannel.
  using DeviceHandle = PortChannelDeviceHandle;

  /// Returns the device-side handle.
  ///
  /// User should make sure the PortChannel is not released when using the returned handle.
  ///
  DeviceHandle deviceHandle() const;
};

/// @deprecated Use @ref BasePortChannel instead.
[[deprecated("Use BasePortChannel instead.")]] typedef BasePortChannel BaseProxyChannel;

/// @deprecated Use @ref PortChannel instead.
[[deprecated("Use PortChannel instead.")]] typedef PortChannel ProxyChannel;

}  // namespace mscclpp

#endif  // MSCCLPP_PORT_CHANNEL_HPP_
