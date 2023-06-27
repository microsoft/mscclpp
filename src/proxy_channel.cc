// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/proxy_channel.hpp>

#include "api.h"
#include "debug.h"
#include "numa.hpp"

namespace mscclpp {
namespace channel {

MSCCLPP_API_CPP DeviceChannelHandle::DeviceChannelHandle(EpochId epochId, Host2DeviceEpoch::DeviceHandle epoch,
                                                         DeviceProxyFifo fifo)
    : epochId_(epochId), epoch_(epoch), fifo_(fifo) {}

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

MSCCLPP_API_CPP EpochId ProxyService::addEpoch(std::shared_ptr<Connection> connection) {
  epochs_.push_back(std::make_shared<Host2DeviceEpoch>(communicator_, connection));
  return epochs_.size() - 1;
}

MSCCLPP_API_CPP MemoryId ProxyService::addMemory(RegisteredMemory memory) {
  memories_.push_back(memory);
  return memories_.size() - 1;
}

MSCCLPP_API_CPP std::shared_ptr<Host2DeviceEpoch> ProxyService::epoch(EpochId id) const { return epochs_[id]; }

MSCCLPP_API_CPP DeviceChannelHandle ProxyService::deviceChannel(EpochId id) {
  return DeviceChannelHandle(id, epochs_[id]->deviceHandle(), proxy_.fifo().deviceFifo());
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
  std::shared_ptr<Host2DeviceEpoch> epoch = epochs_[trigger->fields.chanId];

  auto result = ProxyHandlerResult::Continue;

  if (trigger->fields.type & TriggerData) {
    RegisteredMemory& dst = memories_[trigger->fields.dstMemoryId];
    RegisteredMemory& src = memories_[trigger->fields.srcMemoryId];
    epoch->connection()->write(dst, trigger->fields.dstOffset, src, trigger->fields.srcOffset, trigger->fields.size);
  }

  if (trigger->fields.type & TriggerFlag) {
    epoch->signal();
  }

  if (trigger->fields.type & TriggerSync) {
    epoch->connection()->flush();
    result = ProxyHandlerResult::FlushFifoTailAndContinue;
  }

  return result;
}

}  // namespace channel
}  // namespace mscclpp
