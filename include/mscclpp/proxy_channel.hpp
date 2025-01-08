// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_PROXY_CHANNEL_HPP_
#define MSCCLPP_PROXY_CHANNEL_HPP_

#include "core.hpp"
#include "proxy.hpp"
#include "proxy_channel_device.hpp"
#include "semaphore.hpp"

namespace mscclpp {

struct BaseProxyChannel;
struct ProxyChannel;

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

  /// Get a base proxy channel by semaphore ID.
  /// @param id The ID of the semaphore.
  /// @return The base proxy channel.
  BaseProxyChannel baseProxyChannel(SemaphoreId id);

  /// Get a proxy channel by semaphore ID and memory regions.
  /// @param id The ID of the semaphore.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  /// @return The proxy channel.
  ProxyChannel proxyChannel(SemaphoreId id, MemoryId dst, MemoryId src);

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

/// Proxy channel.
struct BaseProxyChannel {
 protected:
  SemaphoreId semaphoreId_;

  std::shared_ptr<Host2DeviceSemaphore> semaphore_;

  std::shared_ptr<Proxy> proxy_;

 public:
  BaseProxyChannel() = default;

  BaseProxyChannel(SemaphoreId semaphoreId, std::shared_ptr<Host2DeviceSemaphore> semaphore,
                   std::shared_ptr<Proxy> proxy);

  BaseProxyChannel(const BaseProxyChannel& other) = default;

  BaseProxyChannel& operator=(BaseProxyChannel& other) = default;

  /// Device-side handle for @ref BaseProxyChannel.
  using DeviceHandle = BaseProxyChannelDeviceHandle;

  /// Returns the device-side handle.
  ///
  /// User should make sure the BaseProxyChannel is not released when using the returned handle.
  ///
  DeviceHandle deviceHandle() const;
};

/// A common form of proxy channel with a single destination and source memory region.
struct ProxyChannel : public BaseProxyChannel {
 private:
  MemoryId dst_;
  MemoryId src_;

 public:
  /// Default constructor.
  ProxyChannel() = default;

  /// Constructor.
  /// @param semaphoreId The ID of the semaphore.
  /// @param semaphore The semaphore.
  /// @param proxy The proxy.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  ProxyChannel(SemaphoreId semaphoreId, std::shared_ptr<Host2DeviceSemaphore> semaphore, std::shared_ptr<Proxy> proxy,
               MemoryId dst, MemoryId src);

  /// Copy constructor.
  ProxyChannel(const ProxyChannel& other) = default;

  /// Assignment operator.
  ProxyChannel& operator=(ProxyChannel& other) = default;

  /// Device-side handle for @ref ProxyChannel.
  using DeviceHandle = ProxyChannelDeviceHandle;

  /// Returns the device-side handle.
  ///
  /// User should make sure the ProxyChannel is not released when using the returned handle.
  ///
  DeviceHandle deviceHandle() const;
};

}  // namespace mscclpp

#endif  // MSCCLPP_PROXY_CHANNEL_HPP_
