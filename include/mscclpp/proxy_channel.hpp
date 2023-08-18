// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_PROXY_CHANNEL_HPP_
#define MSCCLPP_PROXY_CHANNEL_HPP_

#include <mscclpp/core.hpp>
#include <mscclpp/proxy.hpp>
#include <mscclpp/proxy_channel_device.hpp>
#include <mscclpp/semaphore.hpp>

namespace mscclpp {

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
  ProxyService();

  /// Build and add a semaphore to the proxy service.
  /// @param communicator The communicator for bootstrapping.
  /// @param connection The connection associated with the semaphore.
  /// @return The ID of the semaphore.
  SemaphoreId buildAndAddSemaphore(Communicator& communicator, std::shared_ptr<Connection> connection);

  /// Build and add a semaphore with pitch to the proxy service. This is used for 2D transfers.
  /// @param communicator The communicator for bootstrapping.
  /// @param connection The connection associated with the channel.
  /// @param pitch The pitch pair.
  SemaphoreId buildAndAddSemaphore(Communicator& communicator, std::shared_ptr<Connection> connection,
                                   std::pair<uint64_t, uint64_t> pitch);

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

  /// Get a proxy channel by semaphore ID.
  /// @param id The ID of the semaphore.
  /// @return The proxy channel.
  ProxyChannel proxyChannel(SemaphoreId id);

  /// Start the proxy service.
  void startProxy();

  /// Stop the proxy service.
  void stopProxy();

 private:
  std::vector<std::shared_ptr<Host2DeviceSemaphore>> semaphores_;
  std::vector<RegisteredMemory> memories_;
  std::vector<std::pair<uint64_t, uint64_t>> pitches_;
  Proxy proxy_;
  int deviceNumaNode;

  void bindThread();

  ProxyHandlerResult handleTrigger(ProxyTrigger triggerRaw);
};

/// Proxy channel.
struct ProxyChannel {
 private:
  SemaphoreId semaphoreId_;

  Host2DeviceSemaphore::DeviceHandle semaphore_;

  // this is a concurrent fifo which is multiple threads from the device
  // can produce for and the sole proxy thread consumes it.
  FifoDeviceHandle fifo_;

 public:
  ProxyChannel() = default;

  ProxyChannel(SemaphoreId semaphoreId, Host2DeviceSemaphore::DeviceHandle semaphore, FifoDeviceHandle fifo);

  ProxyChannel(const ProxyChannel& other) = default;

  ProxyChannel& operator=(ProxyChannel& other) = default;

  /// Device-side handle for @ref ProxyChannel.
  using DeviceHandle = ProxyChannelDeviceHandle;

  /// Returns the device-side handle.
  ///
  /// User should make sure the ProxyChannel is not released when using the returned handle.
  ///
  DeviceHandle deviceHandle() const;
};

/// Simple proxy channel with a single destination and source memory region.
struct SimpleProxyChannel {
 private:
  ProxyChannel proxyChan_;
  MemoryId dst_;
  MemoryId src_;

 public:
  /// Default constructor.
  SimpleProxyChannel() = default;

  /// Constructor.
  /// @param proxyChan The proxy channel.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  SimpleProxyChannel(ProxyChannel proxyChan, MemoryId dst, MemoryId src);

  /// Constructor.
  /// @param proxyChan The proxy channel.
  SimpleProxyChannel(ProxyChannel proxyChan) : proxyChan_(proxyChan) {}

  /// Copy constructor.
  SimpleProxyChannel(const SimpleProxyChannel& other) = default;

  /// Assignment operator.
  SimpleProxyChannel& operator=(SimpleProxyChannel& other) = default;

  /// Device-side handle for @ref SimpleProxyChannel.
  using DeviceHandle = SimpleProxyChannelDeviceHandle;

  /// Returns the device-side handle.
  ///
  /// User should make sure the SimpleProxyChannel is not released when using the returned handle.
  ///
  DeviceHandle deviceHandle() const;
};

}  // namespace mscclpp

#endif  // MSCCLPP_PROXY_CHANNEL_HPP_
