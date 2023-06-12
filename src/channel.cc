#include <mscclpp/channel.hpp>

#include "api.h"
#include "checks_internal.hpp"
#include "debug.h"
#include "numa.hpp"

namespace mscclpp {
namespace channel {

MSCCLPP_API_CPP DeviceChannelService::DeviceChannelService(Communicator& communicator)
    : communicator_(communicator),
      proxy_([&](ProxyTrigger triggerRaw) { return handleTrigger(triggerRaw); }, [&]() { bindThread(); }) {
  int cudaDevice;
  MSCCLPP_CUDATHROW(cudaGetDevice(&cudaDevice));
  deviceNumaNode = getDeviceNumaNode(cudaDevice);
}

MSCCLPP_API_CPP ChannelId DeviceChannelService::addChannel(std::shared_ptr<Connection> connection) {
  channels_.push_back(Channel(communicator_, connection));
  return channels_.size() - 1;
}

MSCCLPP_API_CPP MemoryId DeviceChannelService::addMemory(RegisteredMemory memory) {
  memories_.push_back(memory);
  return memories_.size() - 1;
}

MSCCLPP_API_CPP Channel DeviceChannelService::channel(ChannelId id) const { return channels_[id]; }

MSCCLPP_API_CPP DeviceChannel DeviceChannelService::deviceChannel(ChannelId id) {
  return DeviceChannel(id, channels_[id].epoch().deviceHandle(), proxy_.fifo().deviceFifo());
}

MSCCLPP_API_CPP void DeviceChannelService::startProxy() { proxy_.start(); }

MSCCLPP_API_CPP void DeviceChannelService::stopProxy() { proxy_.stop(); }

MSCCLPP_API_CPP void DeviceChannelService::bindThread() {
  if (deviceNumaNode >= 0) {
    numaBind(deviceNumaNode);
    INFO(MSCCLPP_INIT, "NUMA node of DeviceChannelService proxy thread is set to %d", deviceNumaNode);
  }
}

ProxyHandlerResult DeviceChannelService::handleTrigger(ProxyTrigger triggerRaw) {
  ChannelTrigger* trigger = reinterpret_cast<ChannelTrigger*>(&triggerRaw);
  Channel& channel = channels_[trigger->fields.chanId];

  auto result = ProxyHandlerResult::Continue;

  if (trigger->fields.type & TriggerData) {
    RegisteredMemory& dst = memories_[trigger->fields.dstMemoryId];
    RegisteredMemory& src = memories_[trigger->fields.srcMemoryId];
    channel.connection().write(dst, trigger->fields.dstOffset, src, trigger->fields.srcOffset, trigger->fields.size);
  }

  if (trigger->fields.type & TriggerFlag) {
    channel.epoch().signal();
  }

  if (trigger->fields.type & TriggerSync) {
    channel.connection().flush();
    result = ProxyHandlerResult::FlushFifoTailAndContinue;
  }

  return result;
}

}  // namespace channel
}  // namespace mscclpp
