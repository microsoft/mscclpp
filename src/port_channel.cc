// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/numa.hpp>
#include <mscclpp/port_channel.hpp>

#include "api.h"
#include "debug.h"

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
      INFO(MSCCLPP_INIT, "NUMA node of ProxyService proxy thread is set to %d", deviceNumaNode);
    }
  };
  auto handlerFunc = [&](ProxyTrigger triggerRaw) { return handleTrigger(triggerRaw); };
  proxy_ = std::make_shared<Proxy>(handlerFunc, initFunc, fifoSize);
}

MSCCLPP_API_CPP SemaphoreId ProxyService::buildAndAddSemaphore(Communicator& communicator,
                                                               std::shared_ptr<Connection> connection) {
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

MSCCLPP_API_CPP std::shared_ptr<Host2DeviceSemaphore> ProxyService::semaphore(SemaphoreId id) const {
  return semaphores_[id];
}

MSCCLPP_API_CPP BasePortChannel ProxyService::basePortChannel(SemaphoreId id) {
  return BasePortChannel(id, semaphores_[id], proxy_);
}

MSCCLPP_API_CPP PortChannel ProxyService::portChannel(SemaphoreId id, MemoryId dst, MemoryId src) {
  return PortChannel(id, semaphores_[id], proxy_, dst, src);
}

MSCCLPP_API_CPP void ProxyService::startProxy() { proxy_->start(); }

MSCCLPP_API_CPP void ProxyService::stopProxy() { proxy_->stop(); }

ProxyHandlerResult ProxyService::handleTrigger(ProxyTrigger triggerRaw) {
  ChannelTrigger* trigger = reinterpret_cast<ChannelTrigger*>(&triggerRaw);
  std::shared_ptr<Host2DeviceSemaphore> semaphore = semaphores_[trigger->fields.semaphoreId];

  int maxWriteQueueSize = semaphore->connection()->getMaxWriteQueueSize();
  auto& numRequests = inflightRequests_[semaphore->connection()];

  if (trigger->fields.type & TriggerData) {
    RegisteredMemory& dst = memories_[trigger->fields.dstMemoryId];
    RegisteredMemory& src = memories_[trigger->fields.srcMemoryId];
    semaphore->connection()->write(dst, trigger->fields.dstOffset, src, trigger->fields.srcOffset,
                                   trigger->fields.size);
    numRequests++;
  }

  if (trigger->fields.type & TriggerFlag) {
    semaphore->signal();
    numRequests++;
  }

  if (((trigger->fields.type & TriggerSync) && numRequests > 0) ||
      (maxWriteQueueSize != -1 && numRequests > maxWriteQueueSize)) {
    semaphore->connection()->flush();
    numRequests = 0;
  }

  return ProxyHandlerResult::Continue;
}

MSCCLPP_API_CPP BasePortChannel::DeviceHandle BasePortChannel::deviceHandle() const {
  return BasePortChannel::DeviceHandle(semaphoreId_, semaphore_->deviceHandle(), proxy_->fifo()->deviceHandle());
}

MSCCLPP_API_CPP PortChannel::DeviceHandle PortChannel::deviceHandle() const {
  return PortChannel::DeviceHandle(semaphoreId_, semaphore_->deviceHandle(), proxy_->fifo()->deviceHandle(), dst_,
                                   src_);
}

}  // namespace mscclpp
