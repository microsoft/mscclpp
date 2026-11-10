// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_PORT_CHANNEL_HPP_
#define MSCCLPP_PORT_CHANNEL_HPP_

#include "core.hpp"
#include "port_channel_device.hpp"
#include "proxy.hpp"
#include "semaphore.hpp"

namespace mscclpp {

template <PortChannelBackend Backend = PortChannelBackend::Host>
struct BasePortChannel;
template <PortChannelBackend Backend = PortChannelBackend::Host>
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
  BasePortChannel<> basePortChannel(SemaphoreId id);

  /// Get a port channel by semaphore ID and memory regions.
  /// @param id The ID of the semaphore.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  /// @return The port channel.
  PortChannel<> portChannel(SemaphoreId id, MemoryId dst, MemoryId src);

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
template <PortChannelBackend Backend>
struct BasePortChannel {
 protected:
  using SemaphoreType =
      std::conditional_t<Backend == PortChannelBackend::Host, Host2DeviceSemaphore, PortDevice2DeviceSemaphore>;

  SemaphoreId semaphoreId_;

  std::shared_ptr<SemaphoreType> semaphore_;

  std::shared_ptr<Proxy> proxy_;

 public:
  /// Constructor.
  BasePortChannel() = default;

  /// Constructor.
  /// @param semaphoreId The ID of the semaphore.
  /// @param semaphore The semaphore used to synchronize the communication.
  /// @param proxy The proxy used for communication.
  BasePortChannel(SemaphoreId semaphoreId, std::shared_ptr<Host2DeviceSemaphore> semaphore,
                  std::shared_ptr<Proxy> proxy)
    requires(Backend == PortChannelBackend::Host);

  /// Constructor for a channel in device mode (PortChannelBackend::Device).
  /// @param semaphore The semaphore of a PortChannel device-mode connection.
  explicit BasePortChannel(std::shared_ptr<PortDevice2DeviceSemaphore> semaphore)
    requires(Backend == PortChannelBackend::Device);

  /// Constructor.
  /// @param semaphoreId The ID of the semaphore.
  /// @param semaphore The semaphore used to synchronize the communication.
  /// @param proxy The proxy used for communication.
  BasePortChannel(SemaphoreId semaphoreId, const Semaphore& semaphore, std::shared_ptr<Proxy> proxy)
    requires(Backend == PortChannelBackend::Host);

  /// Copy constructor.
  /// @param other The other BasePortChannel to copy from.
  BasePortChannel(const BasePortChannel& other) = default;

  /// Assignment operator.
  /// @param other The other BasePortChannel to assign from.
  BasePortChannel& operator=(const BasePortChannel& other) = default;

  /// Device-side handle for BasePortChannel.
  using DeviceHandle = BasePortChannelDeviceHandleT<Backend>;

  /// Returns the device-side handle.
  /// User should make sure the BasePortChannel is not released when using the returned handle.
  /// @return The device-side handle.
  DeviceHandle deviceHandle() const;

  /// Returns a device-side handle with `SmemSlots` shared-memory slots per CTA.
  /// @tparam SmemSlots Number of 64-byte shared-memory slots per CTA (0 disables BlueFlame).
  /// @return The device-side handle, valid while this BasePortChannel remains alive.
  template <uint32_t SmemSlots>
  BasePortChannelDeviceHandleT<Backend, SmemSlots> deviceHandle() const
    requires(Backend == PortChannelBackend::Device)
  {
    const auto h = deviceHandle();
    return {h.semaphore_, h.device_};
  }
};

BasePortChannel(SemaphoreId, std::shared_ptr<Host2DeviceSemaphore>,
                std::shared_ptr<Proxy>) -> BasePortChannel<PortChannelBackend::Host>;
BasePortChannel(SemaphoreId, const Semaphore&, std::shared_ptr<Proxy>) -> BasePortChannel<PortChannelBackend::Host>;
BasePortChannel(std::shared_ptr<PortDevice2DeviceSemaphore>) -> BasePortChannel<PortChannelBackend::Device>;

/// Port channel.
template <PortChannelBackend Backend>
struct PortChannel : public BasePortChannel<Backend> {
 private:
  MemoryId dst_;
  MemoryId src_;
  RegisteredMemory dstMemory_;
  RegisteredMemory srcMemory_;

 public:
  /// Constructor.
  PortChannel() = default;

  /// Constructor.
  /// @param semaphoreId The ID of the semaphore.
  /// @param semaphore The semaphore.
  /// @param proxy The proxy.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  PortChannel(SemaphoreId semaphoreId, std::shared_ptr<Host2DeviceSemaphore> semaphore, std::shared_ptr<Proxy> proxy,
              MemoryId dst, MemoryId src)
    requires(Backend == PortChannelBackend::Host);

  /// Constructor for a channel in device mode (PortChannelBackend::Device).
  /// @param semaphore The semaphore of a PortChannel device-mode connection.
  /// @param dst The destination registered memory.
  /// @param src The source registered memory.
  PortChannel(std::shared_ptr<PortDevice2DeviceSemaphore> semaphore, RegisteredMemory dst, RegisteredMemory src)
    requires(Backend == PortChannelBackend::Device);

  /// Constructor.
  /// @param semaphoreId The ID of the semaphore.
  /// @param semaphore The semaphore.
  /// @param proxy The proxy.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  PortChannel(SemaphoreId semaphoreId, const Semaphore& semaphore, std::shared_ptr<Proxy> proxy, MemoryId dst,
              MemoryId src)
    requires(Backend == PortChannelBackend::Host);

  /// Copy constructor.
  /// @param other The other PortChannel to copy from.
  PortChannel(const PortChannel& other) = default;

  /// Assignment operator.
  /// @param other The other PortChannel to assign from.
  PortChannel& operator=(const PortChannel& other) = default;

  /// Device-side handle for PortChannel.
  using DeviceHandle = PortChannelDeviceHandleT<Backend>;

  /// Returns the device-side handle.
  /// User should make sure the PortChannel is not released when using the returned handle.
  /// @return The device-side handle.
  DeviceHandle deviceHandle() const;

  /// Returns a device-side handle with `SmemSlots` shared-memory slots per CTA.
  /// @tparam SmemSlots Number of 64-byte shared-memory slots per CTA (0 disables BlueFlame).
  /// @return The device-side handle, valid while this PortChannel remains alive.
  template <uint32_t SmemSlots>
  PortChannelDeviceHandleT<Backend, SmemSlots> deviceHandle() const
    requires(Backend == PortChannelBackend::Device)
  {
    const auto h = deviceHandle();
    return {h.semaphore_, h.device_, h.dstMemory_, h.srcMemory_};
  }
};

PortChannel(SemaphoreId, std::shared_ptr<Host2DeviceSemaphore>, std::shared_ptr<Proxy>, MemoryId,
            MemoryId) -> PortChannel<PortChannelBackend::Host>;
PortChannel(SemaphoreId, const Semaphore&, std::shared_ptr<Proxy>, MemoryId,
            MemoryId) -> PortChannel<PortChannelBackend::Host>;
PortChannel(std::shared_ptr<PortDevice2DeviceSemaphore>, RegisteredMemory,
            RegisteredMemory) -> PortChannel<PortChannelBackend::Device>;

/// @deprecated Use BasePortChannel instead.
[[deprecated("Use BasePortChannel instead.")]] typedef BasePortChannel<> BaseProxyChannel;

/// @deprecated Use PortChannel instead.
[[deprecated("Use PortChannel instead.")]] typedef PortChannel<> ProxyChannel;

}  // namespace mscclpp

#endif  // MSCCLPP_PORT_CHANNEL_HPP_
