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
struct GpuNetIoChannel;

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
  /// If the semaphore's connection uses EndpointConfig::Ib::Mode::Device, the channel captures the memories added
  /// so far; add every memory the channel will address with addMemory() before calling this.
  /// @param id The ID of the semaphore.
  /// @return The base port channel.
  BasePortChannel basePortChannel(SemaphoreId id);

  /// Get a port channel by semaphore ID and memory regions.
  /// If the semaphore's connection uses EndpointConfig::Ib::Mode::Device, the channel captures the memories added
  /// so far; add every memory the channel will address with addMemory() before calling this.
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
};

/// Port channel without specifying source/destination memory regions.
struct BasePortChannel {
 protected:
  SemaphoreId semaphoreId_;

  std::shared_ptr<Host2DeviceSemaphore> semaphore_;

  std::shared_ptr<Proxy> proxy_;

  std::shared_ptr<GpuNetIoChannel> gpuNetIo_;

  /// Internal constructor used by ProxyService. Binds the channel to its connection's backend, giving it the
  /// complete memory table that its device handle may address.
  /// @param semaphoreId The ID of the semaphore.
  /// @param semaphore The semaphore used to synchronize the communication.
  /// @param proxy The proxy used for communication.
  /// @param memories The ProxyService memory table, indexed by MemoryId.
  BasePortChannel(SemaphoreId semaphoreId, std::shared_ptr<Host2DeviceSemaphore> semaphore,
                  std::shared_ptr<Proxy> proxy, const std::vector<RegisteredMemory>& memories);

  friend class ProxyService;

 public:
  /// Constructor.
  BasePortChannel() = default;

  /// Constructor.
  /// Channels over EndpointConfig::Ib::Mode::Device connections must be created with
  /// ProxyService::basePortChannel() instead; deviceHandle() throws for directly constructed ones.
  /// @param semaphoreId The ID of the semaphore.
  /// @param semaphore The semaphore used to synchronize the communication.
  /// @param proxy The proxy used for communication.
  BasePortChannel(SemaphoreId semaphoreId, std::shared_ptr<Host2DeviceSemaphore> semaphore,
                  std::shared_ptr<Proxy> proxy);

  /// Constructor.
  /// @param semaphoreId The ID of the semaphore.
  /// @param semaphore The semaphore used to synchronize the communication.
  /// @param proxy The proxy used for communication.
  BasePortChannel(SemaphoreId semaphoreId, const Semaphore& semaphore, std::shared_ptr<Proxy> proxy);

  /// Copy constructor.
  /// @param other The other BasePortChannel to copy from.
  BasePortChannel(const BasePortChannel& other) = default;

  /// Assignment operator.
  /// @param other The other BasePortChannel to assign from.
  BasePortChannel& operator=(BasePortChannel& other) = default;

  /// Device-side handle for BasePortChannel.
  using DeviceHandle = BasePortChannelDeviceHandle;

  /// Returns the device-side handle.
  /// User should make sure the BasePortChannel is not released when using the returned handle.
  /// @return The device-side handle.
  DeviceHandle deviceHandle() const;
};

/// Port channel.
struct PortChannel : public BasePortChannel {
 private:
  MemoryId dst_;
  MemoryId src_;

  /// Internal constructor used by ProxyService; see the BasePortChannel overload taking a memory table.
  PortChannel(SemaphoreId semaphoreId, std::shared_ptr<Host2DeviceSemaphore> semaphore, std::shared_ptr<Proxy> proxy,
              MemoryId dst, MemoryId src, const std::vector<RegisteredMemory>& memories);

  friend class ProxyService;

 public:
  /// Constructor.
  PortChannel() = default;

  /// Constructor.
  /// Channels over EndpointConfig::Ib::Mode::Device connections must be created with ProxyService::portChannel()
  /// instead; deviceHandle() throws for directly constructed ones.
  /// @param semaphoreId The ID of the semaphore.
  /// @param semaphore The semaphore.
  /// @param proxy The proxy.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  PortChannel(SemaphoreId semaphoreId, std::shared_ptr<Host2DeviceSemaphore> semaphore, std::shared_ptr<Proxy> proxy,
              MemoryId dst, MemoryId src);

  /// Constructor.
  /// @param semaphoreId The ID of the semaphore.
  /// @param semaphore The semaphore.
  /// @param proxy The proxy.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  PortChannel(SemaphoreId semaphoreId, const Semaphore& semaphore, std::shared_ptr<Proxy> proxy, MemoryId dst,
              MemoryId src);

  /// Copy constructor.
  /// @param other The other PortChannel to copy from.
  PortChannel(const PortChannel& other) = default;

  /// Assignment operator.
  /// @param other The other PortChannel to assign from.
  PortChannel& operator=(PortChannel& other) = default;

  /// Device-side handle for PortChannel.
  using DeviceHandle = PortChannelDeviceHandle;

  /// Returns the device-side handle.
  /// User should make sure the PortChannel is not released when using the returned handle.
  /// @return The device-side handle.
  DeviceHandle deviceHandle() const;
};

/// @deprecated Use BasePortChannel instead.
[[deprecated("Use BasePortChannel instead.")]] typedef BasePortChannel BaseProxyChannel;

/// @deprecated Use PortChannel instead.
[[deprecated("Use PortChannel instead.")]] typedef PortChannel ProxyChannel;

}  // namespace mscclpp

#endif  // MSCCLPP_PORT_CHANNEL_HPP_
