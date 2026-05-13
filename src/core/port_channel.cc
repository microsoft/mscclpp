// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/atomic_device.hpp>
#include <mscclpp/numa.hpp>
#include <mscclpp/port_channel.hpp>

#include "api.h"
#include "connection.hpp"
#include "logger.hpp"
#include "proxy_impl.hpp"

namespace mscclpp {

MSCCLPP_API_CPP BasePortChannel::BasePortChannel(SemaphoreId semaphoreId,
                                                 std::shared_ptr<Host2DeviceSemaphore> semaphore,
                                                 std::shared_ptr<Proxy> proxy)
    : semaphoreId_(semaphoreId), semaphore_(semaphore), proxy_(proxy) {}

MSCCLPP_API_CPP BasePortChannel::BasePortChannel(SemaphoreId semaphoreId, const Semaphore& semaphore,
                                                 std::shared_ptr<Proxy> proxy)
    : BasePortChannel(semaphoreId, std::make_shared<Host2DeviceSemaphore>(semaphore), proxy) {}

MSCCLPP_API_CPP PortChannel::PortChannel(SemaphoreId semaphoreId, std::shared_ptr<Host2DeviceSemaphore> semaphore,
                                         std::shared_ptr<Proxy> proxy, MemoryId dst, MemoryId src)
    : BasePortChannel(semaphoreId, semaphore, proxy), dst_(dst), src_(src) {}

MSCCLPP_API_CPP PortChannel::PortChannel(SemaphoreId semaphoreId, const Semaphore& semaphore,
                                         std::shared_ptr<Proxy> proxy, MemoryId dst, MemoryId src)
    : BasePortChannel(semaphoreId, semaphore, proxy), dst_(dst), src_(src) {}

MSCCLPP_API_CPP ProxyService::ProxyService(int fifoSize) {
  int cudaDevice;
  MSCCLPP_CUDATHROW(cudaGetDevice(&cudaDevice));
  int deviceNumaNode = getDeviceNumaNode(cudaDevice);
  auto initFunc = [cudaDevice, deviceNumaNode]() {
    MSCCLPP_CUDATHROW(cudaSetDevice(cudaDevice));
    if (deviceNumaNode >= 0) {
      numaBind(deviceNumaNode);
      INFO(CONN, "NUMA node of ProxyService proxy thread is set to ", deviceNumaNode);
    }
  };
  auto handlerFunc = [&](ProxyTrigger triggerRaw) { return handleTrigger(triggerRaw); };
  proxy_ = std::make_shared<Proxy>(handlerFunc, initFunc, fifoSize);
  proxy_->pimpl_->setProgressHandler([this]() { progressFlushes(); });
}

MSCCLPP_API_CPP SemaphoreId ProxyService::buildAndAddSemaphore(Communicator& communicator,
                                                               const Connection& connection) {
  semaphores_.push_back(std::make_shared<Host2DeviceSemaphore>(communicator, connection));
  return semaphores_.size() - 1;
}

MSCCLPP_API_CPP SemaphoreId ProxyService::addSemaphore(const Semaphore& semaphore) {
  semaphores_.push_back(std::make_shared<Host2DeviceSemaphore>(semaphore));
  return semaphores_.size() - 1;
}

MSCCLPP_API_CPP SemaphoreId ProxyService::addSemaphore(std::shared_ptr<Host2DeviceSemaphore> semaphore) {
  semaphores_.push_back(semaphore);
  return semaphores_.size() - 1;
}

MSCCLPP_API_CPP MemoryId ProxyService::addMemory(RegisteredMemory memory) {
  memories_.push_back(memory);
  return memories_.size() - 1;
}

MSCCLPP_API_CPP MemoryId ProxyService::nextMemoryId([[maybe_unused]] uint32_t count) const {
  if (count == 0) {
    throw Error("count must be greater than 0", ErrorCode::InvalidUsage);
  }
  MemoryId firstId = memories_.size();
  return firstId;
}

MSCCLPP_API_CPP std::shared_ptr<Host2DeviceSemaphore> ProxyService::semaphore(SemaphoreId id) const {
  return semaphores_[id];
}

MSCCLPP_API_CPP BasePortChannel ProxyService::basePortChannel(SemaphoreId id) {
  return BasePortChannel(id, semaphores_[id], proxy_);
}

MSCCLPP_API_CPP PortChannel ProxyService::portChannel(SemaphoreId id, MemoryId dst, MemoryId src) {
  return PortChannel(id, semaphores_[id], proxy_, dst, src);
}

MSCCLPP_API_CPP void ProxyService::startProxy(bool blocking) { proxy_->start(blocking); }

MSCCLPP_API_CPP void ProxyService::stopProxy() {
  proxy_->stop();
  // Drain pending flushes. After a bounded loop, force-unblock any still-pending GPU
  // waiters with a sentinel write (UINT64_MAX >= any expected generation).
  for (int i = 0; i < 1000 && !pendingFlushConns_.empty(); ++i) {
    progressFlushes();
  }
  if (!pendingFlushConns_.empty()) {
    WARN(CONN, "stopProxy: ", pendingFlushConns_.size(), " connections still pending; writing sentinel");
    for (auto& conn : pendingFlushConns_) {
      if (uint64_t* ptr = conn->getFlushDonePtr()) atomicStore(ptr, UINT64_MAX, memoryOrderRelease);
    }
    pendingFlushConns_.clear();
  }
}

ProxyHandlerResult ProxyService::handleTrigger(ProxyTrigger trigger) {
  std::shared_ptr<Host2DeviceSemaphore> semaphore = semaphores_[trigger.fields.semaphoreId];

  auto& conn = semaphore->connection();
  int maxWriteQueueSize = conn.getMaxWriteQueueSize();
  auto& numRequests = inflightRequests_[conn.impl_];

  if (trigger.fields.type & TriggerData) {
    RegisteredMemory& dst = memories_[trigger.fields.dstMemoryId];
    RegisteredMemory& src = memories_[trigger.fields.srcMemoryId];
    conn.write(dst, trigger.fields.dstOffset, src, trigger.fields.srcOffset, trigger.fields.size);
    numRequests++;
  }

  if (trigger.fields.type & TriggerFlag) {
    semaphore->signal();
    numRequests++;
  }

  if (trigger.fields.type & TriggerSync) {
    // Always requestFlush on TriggerSync, even when numRequests == 0. The GPU increments
    // expectedFlushGen_ on every flush() push, so the proxy must always increment
    // flushRequestGen_ to match. An empty CQ drain completes immediately.
    conn.impl_->requestFlush();
    pendingFlushConns_.insert(conn.impl_);
    numRequests = 0;
  } else if (maxWriteQueueSize != -1 && numRequests >= maxWriteQueueSize) {
    conn.flush();  // flow-control flush stays blocking
    numRequests = 0;
  }

  return ProxyHandlerResult::Continue;
}

MSCCLPP_API_CPP BasePortChannel::DeviceHandle BasePortChannel::deviceHandle() const {
  auto& conn = semaphore_->connection();
  return BasePortChannel::DeviceHandle(semaphoreId_, semaphore_->deviceHandle(), proxy_->fifo()->deviceHandle(),
                                       conn.impl_->getFlushDonePtr(), conn.impl_->getExpectedFlushPtr());
}

MSCCLPP_API_CPP PortChannel::DeviceHandle PortChannel::deviceHandle() const {
  auto& conn = semaphore_->connection();
  return PortChannel::DeviceHandle(semaphoreId_, semaphore_->deviceHandle(), proxy_->fifo()->deviceHandle(), dst_, src_,
                                   conn.impl_->getFlushDonePtr(), conn.impl_->getExpectedFlushPtr());
}

void ProxyService::progressFlushes() {
  for (auto it = pendingFlushConns_.begin(); it != pendingFlushConns_.end();) {
    if ((*it)->progressFlush()) {
      it = pendingFlushConns_.erase(it);
    } else {
      ++it;
    }
  }
}

}  // namespace mscclpp
