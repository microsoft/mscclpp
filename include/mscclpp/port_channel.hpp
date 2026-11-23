// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_PORT_CHANNEL_HPP_
#define MSCCLPP_PORT_CHANNEL_HPP_

#include <utility>
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

/// Port channel owner without source/destination memory regions.
/// Construct with a Host semaphore and proxy, or with a Device semaphore.
/// GPU handle extraction requires an explicit backend.
struct BasePortChannel {
 protected:
  struct HostState {
    SemaphoreId id;
    std::shared_ptr<Host2DeviceSemaphore> semaphore;
    std::shared_ptr<Proxy> proxy;
    HostState(SemaphoreId id, std::shared_ptr<Host2DeviceSemaphore> semaphore, std::shared_ptr<Proxy> proxy)
        : id(id), semaphore(std::move(semaphore)), proxy(std::move(proxy)) {}
  };
  struct DeviceState {
    std::shared_ptr<PortDevice2DeviceSemaphore> semaphore;
    explicit DeviceState(std::shared_ptr<PortDevice2DeviceSemaphore> semaphore) : semaphore(std::move(semaphore)) {}
  };
  std::variant<std::monostate, HostState, DeviceState> state_;

 public:
  /// Construct an unassigned owner. Every extraction fails until assigned.
  BasePortChannel() noexcept : state_(std::monostate{}) {}

  /// Construct a Host-backed channel.
  /// @param semaphoreId The ID of the semaphore in the proxy service; ID0 is valid.
  /// @param semaphore The semaphore used to synchronize the communication.
  /// @param proxy The proxy used for communication.
  /// @throws Error with ErrorCode::InvalidUsage if semaphore or proxy is null, or the
  /// semaphore uses a device-mode connection.
  BasePortChannel(SemaphoreId semaphoreId, std::shared_ptr<Host2DeviceSemaphore> semaphore,
                  std::shared_ptr<Proxy> proxy);

  /// Construct a Device-backed channel without a proxy.
  /// @param semaphore The semaphore of a PortChannel device-mode connection.
  /// @throws Error with ErrorCode::InvalidUsage if semaphore is null or moved from.
  explicit BasePortChannel(std::shared_ptr<PortDevice2DeviceSemaphore> semaphore);

  /// Construct a Host-backed channel from an initialized raw Semaphore.
  /// Creates a synchronization owner whose counters are shared by channel copies.
  /// @pre semaphore must be initialized; default or moved-from raw semaphores are not supported.
  /// @param semaphoreId The ID of the semaphore in the proxy service; ID0 is valid.
  /// @param semaphore The semaphore used to synchronize the communication.
  /// @param proxy The proxy used for communication.
  /// @throws Error with ErrorCode::InvalidUsage if proxy is null, the local endpoint is not a GPU,
  /// or the semaphore uses a device-mode connection.
  BasePortChannel(SemaphoreId semaphoreId, const Semaphore& semaphore, std::shared_ptr<Proxy> proxy);

  /// Copy constructor. Copies share the semaphore and proxy ownership.
  /// @param other The other BasePortChannel to copy from.
  BasePortChannel(const BasePortChannel& other) = default;

  /// Assignment operator. The source may use either backend.
  /// @param other The other BasePortChannel to assign from.
  BasePortChannel& operator=(const BasePortChannel& other) = default;

  /// Backend-specific device-side handle type.
  /// @tparam Backend PortChannelBackend::Host or PortChannelBackend::Device.
  template <PortChannelBackend Backend>
  using DeviceHandle = BasePortChannelDeviceHandle<Backend>;

  /// Return a non-owning device-side handle for the selected backend.
  /// Keep this owner and application buffers alive while using the returned handle.
  /// @tparam Backend The backend matching the constructor used for this owner. No default is provided.
  /// @return The backend-specific device-side handle.
  /// @throws Error with ErrorCode::InvalidUsage if the owner is unassigned, the requested
  /// backend does not match, or its Port semaphore has been moved from.
  template <PortChannelBackend Backend>
  DeviceHandle<Backend> deviceHandle() const
    requires(Backend == PortChannelBackend::Host || Backend == PortChannelBackend::Device);
};

/// Port channel owner with source/destination memory regions.
/// Copies retain both synchronization and memory bindings. Replace the whole owner to rebind.
struct PortChannel : public BasePortChannel {
 private:
  struct HostBinding {
    MemoryId dst;
    MemoryId src;
    HostBinding(MemoryId dst, MemoryId src) : dst(dst), src(src) {}
  };
  struct DeviceBinding {
    RegisteredMemory dst;
    RegisteredMemory src;
    DeviceBinding(RegisteredMemory dst, RegisteredMemory src) : dst(std::move(dst)), src(std::move(src)) {}
  };
  std::variant<std::monostate, HostBinding, DeviceBinding> binding_;

 public:
  /// Construct an unassigned channel. Handle extraction throws InvalidUsage until assigned.
  PortChannel() noexcept : BasePortChannel(), binding_(std::monostate{}) {}

  /// Construct a Host-backed channel.
  /// @param semaphoreId The ID of the semaphore in the proxy service; ID0 is valid.
  /// @param semaphore The semaphore used to synchronize the communication.
  /// @param proxy The proxy used for communication.
  /// @param dst The destination memory ID in the proxy service; ID0 is valid.
  /// @param src The source memory ID in the proxy service; ID0 is valid.
  /// @throws Error with ErrorCode::InvalidUsage if semaphore or proxy is null, or the
  /// semaphore uses a device-mode connection.
  PortChannel(SemaphoreId semaphoreId, std::shared_ptr<Host2DeviceSemaphore> semaphore, std::shared_ptr<Proxy> proxy,
              MemoryId dst, MemoryId src);

  /// Construct a Device-backed channel without a proxy.
  /// @param semaphore The semaphore of a PortChannel device-mode connection.
  /// @param dst The destination registered memory, retained by this owner.
  /// @param src The source registered memory, retained by this owner.
  /// @throws Error with ErrorCode::InvalidUsage if semaphore is null or moved from, or
  /// memory registrations are uninitialized or incompatible with the connection.
  PortChannel(std::shared_ptr<PortDevice2DeviceSemaphore> semaphore, RegisteredMemory dst, RegisteredMemory src);

  /// Construct a Host-backed channel from an initialized raw Semaphore.
  /// Creates a synchronization owner whose counters are shared by channel copies.
  /// @pre semaphore must be initialized; default or moved-from raw semaphores are not supported.
  /// @param semaphoreId The ID of the semaphore in the proxy service; ID0 is valid.
  /// @param semaphore The semaphore used to synchronize the communication.
  /// @param proxy The proxy used for communication.
  /// @param dst The destination memory ID in the proxy service; ID0 is valid.
  /// @param src The source memory ID in the proxy service; ID0 is valid.
  /// @throws Error with ErrorCode::InvalidUsage if proxy is null, the local endpoint is not a GPU,
  /// or the semaphore uses a device-mode connection.
  PortChannel(SemaphoreId semaphoreId, const Semaphore& semaphore, std::shared_ptr<Proxy> proxy, MemoryId dst,
              MemoryId src);

  /// Copy constructor. Copies share synchronization ownership and retain memory bindings.
  /// @param other The other PortChannel to copy from.
  PortChannel(const PortChannel& other) = default;

  /// Assignment operator. Copies both synchronization and binding state, including across backends.
  /// @param other The other PortChannel to assign from.
  PortChannel& operator=(const PortChannel& other) = default;

  /// Backend-specific device-side handle type.
  /// @tparam Backend PortChannelBackend::Host or PortChannelBackend::Device.
  template <PortChannelBackend Backend>
  using DeviceHandle = PortChannelDeviceHandle<Backend>;

  /// Return a non-owning device-side handle for the selected backend.
  /// Keep this owner and application buffers alive while using the returned handle.
  /// Assigning only the base does not update memory bindings. Same-backend base replacement
  /// requires compatible proxy/connection/regions; replace the whole owner to rebind.
  /// @tparam Backend The backend matching both synchronization and memory bindings. No default is provided.
  /// @return The backend-specific device-side handle.
  /// @throws Error with ErrorCode::InvalidUsage if synchronization or binding state is unassigned,
  /// either state does not match Backend, the Port semaphore has been moved from, or
  /// Device memory registrations are uninitialized or incompatible with the connection.
  template <PortChannelBackend Backend>
  DeviceHandle<Backend> deviceHandle() const
    requires(Backend == PortChannelBackend::Host || Backend == PortChannelBackend::Device);
};

/// @deprecated Use BasePortChannel instead.
[[deprecated("Use BasePortChannel instead.")]] typedef BasePortChannel BaseProxyChannel;

/// @deprecated Use PortChannel instead.
[[deprecated("Use PortChannel instead.")]] typedef PortChannel ProxyChannel;

}  // namespace mscclpp

#endif  // MSCCLPP_PORT_CHANNEL_HPP_
