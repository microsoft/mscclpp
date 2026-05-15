// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_PROXY_IMPL_HPP_
#define MSCCLPP_PROXY_IMPL_HPP_

#include <atomic>
#include <functional>
#include <memory>
#include <mscclpp/fifo.hpp>
#include <mscclpp/proxy.hpp>
#include <thread>

namespace mscclpp {

struct Proxy::Impl {
  ProxyHandler handler;
  std::function<void()> threadInit;
  std::function<void()> progressHandler;
  std::shared_ptr<Fifo> fifo;
  std::atomic_bool threadStarted;
  std::thread service;
  std::atomic_bool running;

  Impl(ProxyHandler handler, std::function<void()> threadInit, int fifoSize)
      : handler(handler),
        threadInit(threadInit),
        fifo(std::make_shared<Fifo>(fifoSize)),
        threadStarted(false),
        running(false) {}

  // Must be called before start() — the proxy thread captures progressHandler at start time.
  void setProgressHandler(std::function<void()> h) { progressHandler = std::move(h); }
};

}  // namespace mscclpp

#endif  // MSCCLPP_PROXY_IMPL_HPP_
