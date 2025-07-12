// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <atomic>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/numa.hpp>
#include <mscclpp/proxy.hpp>
#include <mscclpp/utils.hpp>
#include <thread>

#include "api.h"

namespace mscclpp {

constexpr int ProxyStopCheckPeriod = 1000;

struct Proxy::Impl {
  ProxyHandler handler;
  std::function<void()> threadInit;
  std::shared_ptr<Fifo> fifo;
  std::thread service;
  std::atomic_bool running;

  Impl(ProxyHandler handler, std::function<void()> threadInit, int fifoSize)
      : handler(handler), threadInit(threadInit), fifo(std::make_shared<Fifo>(fifoSize)), running(false) {}
};

MSCCLPP_API_CPP Proxy::Proxy(ProxyHandler handler, std::function<void()> threadInit, int fifoSize) {
  pimpl_ = std::make_unique<Impl>(handler, threadInit, fifoSize);
}

MSCCLPP_API_CPP Proxy::Proxy(ProxyHandler handler, int fifoSize) {
  int cudaDevice;
  MSCCLPP_CUDATHROW(cudaGetDevice(&cudaDevice));
  int deviceNumaNode = getDeviceNumaNode(cudaDevice);
  auto initFunc = [cudaDevice, deviceNumaNode]() {
    MSCCLPP_CUDATHROW(cudaSetDevice(cudaDevice));
    if (deviceNumaNode >= 0) {
      numaBind(deviceNumaNode);
    }
  };
  pimpl_ = std::make_unique<Impl>(handler, initFunc, fifoSize);
}

MSCCLPP_API_CPP Proxy::~Proxy() {
  if (pimpl_) {
    stop();
  }
}

MSCCLPP_API_CPP void Proxy::start() {
  pimpl_->running = true;
  pimpl_->service = std::thread([this] {
    // never capture in a proxy thread
    auto mode = cudaStreamCaptureModeRelaxed;
    MSCCLPP_CUDATHROW(cudaThreadExchangeStreamCaptureMode(&mode));

    pimpl_->threadInit();

    ProxyHandler handler = this->pimpl_->handler;
    auto fifo = this->pimpl_->fifo;
    std::atomic_bool& running = this->pimpl_->running;
    ProxyTrigger trigger;

    int runCnt = ProxyStopCheckPeriod;
    for (;;) {
      if (runCnt-- == 0) {
        runCnt = ProxyStopCheckPeriod;
        if (!running) {
          break;
        }
      }
      // Poll to see if we are ready to send anything
      trigger = fifo->poll();
      if (trigger.fst == 0 || trigger.snd == 0) {  // TODO: this check is a potential pitfall for custom triggers
        continue;                                  // there is one in progress
      }
      trigger.snd ^= (uint64_t{1} << uint64_t{63});  // this is where the last bit of snd is reverted.

      ProxyHandlerResult result = handler(trigger);

      // Send completion: reset only the high 64 bits
      fifo->pop();

      if (result == ProxyHandlerResult::Stop) {
        break;
      }
    }
  });
}

MSCCLPP_API_CPP void Proxy::stop() {
  pimpl_->running = false;
  if (pimpl_->service.joinable()) {
    pimpl_->service.join();
  }
}

MSCCLPP_API_CPP std::shared_ptr<Fifo> Proxy::fifo() { return pimpl_->fifo; }

}  // namespace mscclpp
