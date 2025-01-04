// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/numa.hpp>
#include <mscclpp/proxy_channel.hpp>

#include "api.h"
#include "debug.h"

namespace mscclpp {

MSCCLPP_API_CPP BaseProxyChannel::BaseProxyChannel(SemaphoreId semaphoreId,
                                                   std::shared_ptr<Host2DeviceSemaphore> semaphore,
                                                   std::shared_ptr<Proxy> proxy)
    : semaphoreId_(semaphoreId), semaphore_(semaphore), proxy_(proxy) {}

MSCCLPP_API_CPP ProxyChannel::ProxyChannel(SemaphoreId semaphoreId, std::shared_ptr<Host2DeviceSemaphore> semaphore,
                                           std::shared_ptr<Proxy> proxy, MemoryId dst, MemoryId src)
    : BaseProxyChannel(semaphoreId, semaphore, proxy), dst_(dst), src_(src) {}

MSCCLPP_API_CPP ProxyService::ProxyService(size_t fifoSize)
    : proxy_(std::make_shared<Proxy>([&](ProxyTrigger triggerRaw) { return handleTrigger(triggerRaw); },
                                     [&]() { bindThread(); }, fifoSize)) {
  int cudaDevice;
  MSCCLPP_CUDATHROW(cudaGetDevice(&cudaDevice));
  deviceNumaNode = getDeviceNumaNode(cudaDevice);
}

MSCCLPP_API_CPP SemaphoreId ProxyService::buildAndAddSemaphore(Communicator& communicator,
                                                               std::shared_ptr<Connection> connection) {
  semaphores_.push_back(std::make_shared<Host2DeviceSemaphore>(communicator, connection));
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

MSCCLPP_API_CPP std::shared_ptr<Host2DeviceSemaphore> ProxyService::semaphore(SemaphoreId id) const {
  return semaphores_[id];
}

MSCCLPP_API_CPP BaseProxyChannel ProxyService::baseProxyChannel(SemaphoreId id) {
  return BaseProxyChannel(id, semaphores_[id], proxy_);
}

MSCCLPP_API_CPP ProxyChannel ProxyService::proxyChannel(SemaphoreId id, MemoryId dst, MemoryId src) {
  return ProxyChannel(id, semaphores_[id], proxy_, dst, src);
}

MSCCLPP_API_CPP void ProxyService::startProxy() { proxy_->start(); }

MSCCLPP_API_CPP void ProxyService::stopProxy() { proxy_->stop(); }

MSCCLPP_API_CPP void ProxyService::bindThread() {
  if (deviceNumaNode >= 0) {
    numaBind(deviceNumaNode);
    INFO(MSCCLPP_INIT, "NUMA node of ProxyService proxy thread is set to %d", deviceNumaNode);
  }
}

ProxyHandlerResult ProxyService::handleTrigger(ProxyTrigger triggerRaw) {
  ChannelTrigger* trigger = reinterpret_cast<ChannelTrigger*>(&triggerRaw);
  std::shared_ptr<Host2DeviceSemaphore> semaphore = semaphores_[trigger->fields.chanId];

  auto result = ProxyHandlerResult::Continue;
  int maxWriteQueueSize = semaphore->connection()->getMaxWriteQueueSize();

  if (trigger->fields.type & TriggerData) {
    RegisteredMemory& dst = memories_[trigger->fields.dstMemoryId];
    RegisteredMemory& src = memories_[trigger->fields.srcMemoryId];
    semaphore->connection()->write(dst, trigger->fields.dstOffset, src, trigger->fields.srcOffset,
                                   trigger->fields.size);
    inflightRequests[semaphore->connection()]++;
  }

  if (trigger->fields.type & TriggerFlag) {
    semaphore->signal();
    inflightRequests[semaphore->connection()]++;
  }

  if (trigger->fields.type & TriggerSync ||
      (maxWriteQueueSize != -1 && inflightRequests[semaphore->connection()] > maxWriteQueueSize)) {
    semaphore->connection()->flush();
    result = ProxyHandlerResult::FlushFifoTailAndContinue;
    inflightRequests[semaphore->connection()] = 0;
  }

  return result;
}

MSCCLPP_API_CPP BaseProxyChannel::DeviceHandle BaseProxyChannel::deviceHandle() const {
  return BaseProxyChannel::DeviceHandle(semaphoreId_, semaphore_->deviceHandle(), proxy_->fifo().deviceHandle());
}

MSCCLPP_API_CPP ProxyChannel::DeviceHandle ProxyChannel::deviceHandle() const {
  return ProxyChannel::DeviceHandle(semaphoreId_, semaphore_->deviceHandle(), proxy_->fifo().deviceHandle(), dst_,
                                    src_);
}

}  // namespace mscclpp
