// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_PROXY_HPP_
#define MSCCLPP_PROXY_HPP_

#include <functional>
#include <memory>
#include <mscclpp/fifo.hpp>

namespace mscclpp {

enum class ProxyHandlerResult {
  Continue,
  FlushFifoTailAndContinue,
  Stop,
};

class Proxy;
using ProxyHandler = std::function<ProxyHandlerResult(ProxyTrigger)>;

class Proxy {
 public:
  Proxy(ProxyHandler handler, std::function<void()> threadInit);
  Proxy(ProxyHandler handler);
  ~Proxy();

  void start();
  void stop();

  HostProxyFifo& fifo();

 private:
  struct Impl;
  std::unique_ptr<Impl> pimpl;
};

}  // namespace mscclpp

#endif  // MSCCLPP_PROXY_HPP_