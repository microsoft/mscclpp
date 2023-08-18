// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/numa.hpp>
#include <mscclpp/proxy_channel.hpp>

#include "api.h"
#include "debug.h"

namespace mscclpp {

MSCCLPP_API_CPP ProxyChannel::ProxyChannel(SemaphoreId semaphoreId, Host2DeviceSemaphore::DeviceHandle semaphore,
                                           FifoDeviceHandle fifo)
    : semaphoreId_(semaphoreId), semaphore_(semaphore), fifo_(fifo) {}

MSCCLPP_API_CPP SimpleProxyChannel::SimpleProxyChannel(ProxyChannel proxyChan, MemoryId dst, MemoryId src)
    : proxyChan_(proxyChan), dst_(dst), src_(src) {}

MSCCLPP_API_CPP ProxyService::ProxyService()
    : proxy_([&](ProxyTrigger triggerRaw) { return handleTrigger(triggerRaw); }, [&]() { bindThread(); }) {
  int cudaDevice;
  MSCCLPP_CUDATHROW(cudaGetDevice(&cudaDevice));
  deviceNumaNode = getDeviceNumaNode(cudaDevice);
}

MSCCLPP_API_CPP SemaphoreId ProxyService::buildAndAddSemaphore(Communicator& communicator,
                                                               std::shared_ptr<Connection> connection) {
  semaphores_.push_back(std::make_shared<Host2DeviceSemaphore>(communicator, connection));
  return semaphores_.size() - 1;
}

MSCCLPP_API_CPP SemaphoreId ProxyService::buildAndAddSemaphore(Communicator& communicator,
                                                               std::shared_ptr<Connection> connection,
                                                               std::pair<uint64_t, uint64_t> pitch) {
  semaphores_.push_back(std::make_shared<Host2DeviceSemaphore>(communicator, connection));
  SemaphoreId id = semaphores_.size() - 1;
  if (id >= pitches_.size()) pitches_.resize(id + 1, std::pair<uint64_t, uint64_t>(0, 0));
  pitches_[id] = pitch;
  return id;
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

MSCCLPP_API_CPP ProxyChannel ProxyService::proxyChannel(SemaphoreId id) {
  return ProxyChannel(id, semaphores_[id]->deviceHandle(), proxy_.fifo().deviceHandle());
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
    if (trigger->fields2D.multiDimensionFlag) {
      std::pair<uint64_t, uint64_t>& pitch = pitches_.at(trigger->fields.chanId);
      semaphore->connection()->write2D(dst, trigger->fields.dstOffset, pitch.first, src, trigger->fields.srcOffset,
                                       pitch.second, trigger->fields2D.width, trigger->fields2D.height);
    } else {
      semaphore->connection()->write(dst, trigger->fields.dstOffset, src, trigger->fields.srcOffset,
                                     trigger->fields.size);
    }
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

MSCCLPP_API_CPP ProxyChannel::DeviceHandle ProxyChannel::deviceHandle() const {
  return ProxyChannel::DeviceHandle{.semaphoreId_ = semaphoreId_, .semaphore_ = semaphore_, .fifo_ = fifo_};
}

MSCCLPP_API_CPP SimpleProxyChannel::DeviceHandle SimpleProxyChannel::deviceHandle() const {
  return SimpleProxyChannel::DeviceHandle{.proxyChan_ = proxyChan_.deviceHandle(), .dst_ = dst_, .src_ = src_};
}

}  // namespace mscclpp
