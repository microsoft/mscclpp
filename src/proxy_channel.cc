// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/proxy_channel.hpp>

#include "api.h"
#include "debug.h"
#include "numa.hpp"

namespace mscclpp {
namespace channel {

MSCCLPP_API_CPP DeviceChannelHandle::DeviceChannelHandle(SemaphoreId semaphoreId,
                                                         Host2DeviceSemaphore::DeviceHandle semaphore,
                                                         DeviceProxyFifo fifo)
    : semaphoreId_(semaphoreId), semaphore_(semaphore), fifo_(fifo) {}

MSCCLPP_API_CPP SimpleDeviceChannelHandle::SimpleDeviceChannelHandle(DeviceChannelHandle devChan, MemoryId dst,
                                                                     MemoryId src)
    : devChan_(devChan), dst_(dst), src_(src) {}

MSCCLPP_API_CPP ProxyService::ProxyService(Communicator& communicator)
    : communicator_(communicator),
      proxy_([&](ProxyTrigger triggerRaw) { return handleTrigger(triggerRaw); }, [&]() { bindThread(); }) {
  int cudaDevice;
  MSCCLPP_CUDATHROW(cudaGetDevice(&cudaDevice));
  deviceNumaNode = getDeviceNumaNode(cudaDevice);
}

MSCCLPP_API_CPP SemaphoreId ProxyService::addSemaphore(std::shared_ptr<Connection> connection) {
  semaphores_.push_back(std::make_shared<Host2DeviceSemaphore>(communicator_, connection));
  return semaphores_.size() - 1;
}

MSCCLPP_API_CPP MemoryId ProxyService::addMemory(RegisteredMemory memory) {
  memories_.push_back(memory);
  return memories_.size() - 1;
}

MSCCLPP_API_CPP std::shared_ptr<Host2DeviceSemaphore> ProxyService::semaphore(SemaphoreId id) const {
  return semaphores_[id];
}

MSCCLPP_API_CPP DeviceChannelHandle ProxyService::deviceChannel(SemaphoreId id) {
  return DeviceChannelHandle(id, semaphores_[id]->deviceHandle(), proxy_.fifo().deviceFifo());
}

MSCCLPP_API_CPP void ProxyService::startProxy() { proxy_.start(); }

MSCCLPP_API_CPP void ProxyService::stopProxy() { proxy_.stop(); }

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

  if (trigger->fields.type & TriggerData) {
    RegisteredMemory& dst = memories_[trigger->fields.dstMemoryId];
    RegisteredMemory& src = memories_[trigger->fields.srcMemoryId];
    semaphore->connection()->write(dst, trigger->fields.dstOffset, src, trigger->fields.srcOffset,
                                   trigger->fields.size);
  }

  if (trigger->fields.type & TriggerFlag) {
    semaphore->signal();
  }

  if (trigger->fields.type & TriggerSync) {
    semaphore->connection()->flush();
    result = ProxyHandlerResult::FlushFifoTailAndContinue;
  }

  return result;
}

}  // namespace channel
}  // namespace mscclpp
