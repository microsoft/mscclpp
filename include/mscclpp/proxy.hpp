// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_PROXY_HPP_
#define MSCCLPP_PROXY_HPP_

#include <functional>
#include <memory>

#include "fifo.hpp"

namespace mscclpp {

/// Possible return values of a ProxyHandler.
enum class ProxyHandlerResult {
  /// Move to the next trigger in the FIFO.
  Continue,
  /// Flush the FIFO and continue to the next trigger.
  FlushFifoTailAndContinue,
  /// Stop the proxy and exit.
  Stop,
};

class Proxy;

/// Type of handler function for the proxy.
using ProxyHandler = std::function<ProxyHandlerResult(ProxyTrigger)>;

/// Host-side proxy for PortChannels.
class Proxy {
 public:
  /// Constructor of Proxy.
  /// @param handler The handler function to be called for each trigger in the FIFO.
  /// @param threadInit Optional function to be called in the proxy thread before starting the FIFO consumption.
  /// @param fifoSize The size of the FIFO. Default is DEFAULT_FIFO_SIZE.
  Proxy(ProxyHandler handler, std::function<void()> threadInit, size_t fifoSize = DEFAULT_FIFO_SIZE);

  /// Constructor of Proxy.
  /// @param handler The handler function to be called for each trigger in the FIFO.
  /// @param fifoSize The size of the FIFO. Default is DEFAULT_FIFO_SIZE.
  Proxy(ProxyHandler handler, size_t fifoSize = DEFAULT_FIFO_SIZE);

  /// Destructor of Proxy.
  /// This will stop the proxy if it is running.
  ~Proxy();

  /// Start the proxy.
  void start();

  /// Stop the proxy.
  void stop();

  /// This is a concurrent fifo which is multiple threads from the device
  /// can produce for and the sole proxy thread consumes it.
  /// @return A reference to the FIFO object used by the proxy.
  Fifo& fifo();

 private:
  struct Impl;
  std::unique_ptr<Impl> pimpl;
};

}  // namespace mscclpp

#endif  // MSCCLPP_PROXY_HPP_
