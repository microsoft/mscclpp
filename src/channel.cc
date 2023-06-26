// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/channel.hpp>

#include "api.h"
#include "debug.h"
#include "numa.hpp"

namespace mscclpp {
namespace channel {

MSCCLPP_API_CPP DeviceChannel::DeviceChannel(ChannelId channelId, DeviceEpoch::DeviceHandle epoch, DeviceProxyFifo fifo)
    : channelId_(channelId), epoch_(epoch), fifo_(fifo) {}

MSCCLPP_API_CPP SmDeviceChannel::SmDeviceChannel(uint32_t epochId, SmEpoch::DeviceHandle epoch, DeviceProxyFifo fifo)
    : epochId_(epochId), epoch_(epoch), fifo_(fifo) {}

MSCCLPP_API_CPP SimpleDeviceChannel::SimpleDeviceChannel(DeviceChannel devChan, MemoryId dst, MemoryId src)
    : devChan_(devChan), dst_(dst), src_(src) {}

MSCCLPP_API_CPP SimpleSmDeviceChannel::SimpleSmDeviceChannel(SmDeviceChannel devChan, MemoryId remoteGetPacketMem,
                                                             MemoryId localPutPacketMem, void* putPacketBuffer,
                                                             void* getPacketBuffer)
    : devChan_(devChan),
      remoteGetPacketMem_(remoteGetPacketMem),
      localPutPacketMem_(localPutPacketMem),
      putPacketBuffer_(putPacketBuffer),
      getPacketBuffer_(getPacketBuffer) {}

MSCCLPP_API_CPP SmChannel::SmChannel(SmEpoch::DeviceHandle epoch, RegisteredMemory dst, void* src,
                                     void* getPacketBuffer)
    : epoch_(epoch), src_(src), getPacketBuffer_(getPacketBuffer) {
  if (!dst.transports().has(Transport::CudaIpc)) {
    throw Error("SmChannel: dst must be registered with CudaIpc", ErrorCode::InvalidUsage);
  }
  dst_ = dst.data();
}

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

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// TODO(chhwang): reduce redundancy below

MSCCLPP_API_CPP SmDeviceChannelService::SmDeviceChannelService(Communicator& communicator)
    : communicator_(communicator),
      proxy_([&](ProxyTrigger triggerRaw) { return handleTrigger(triggerRaw); }, [&]() { bindThread(); }) {
  int cudaDevice;
  MSCCLPP_CUDATHROW(cudaGetDevice(&cudaDevice));
  deviceNumaNode = getDeviceNumaNode(cudaDevice);
}

MSCCLPP_API_CPP uint32_t SmDeviceChannelService::addEpoch(std::shared_ptr<Connection> connection) {
  epochs_.emplace_back(communicator_, connection);
  connections_.emplace_back(connection);
  return epochs_.size() - 1;
}

MSCCLPP_API_CPP MemoryId SmDeviceChannelService::addMemory(RegisteredMemory memory) {
  memories_.push_back(memory);
  return memories_.size() - 1;
}

MSCCLPP_API_CPP const SmEpoch& SmDeviceChannelService::epoch(uint32_t id) const { return epochs_[id]; }

MSCCLPP_API_CPP SmDeviceChannel SmDeviceChannelService::deviceChannel(uint32_t id) {
  return SmDeviceChannel(id, epochs_[id].deviceHandle(), proxy_.fifo().deviceFifo());
}

MSCCLPP_API_CPP void SmDeviceChannelService::startProxy() { proxy_.start(); }

MSCCLPP_API_CPP void SmDeviceChannelService::stopProxy() { proxy_.stop(); }

MSCCLPP_API_CPP void SmDeviceChannelService::bindThread() {
  if (deviceNumaNode >= 0) {
    numaBind(deviceNumaNode);
    INFO(MSCCLPP_INIT, "NUMA node of SmDeviceChannelService proxy thread is set to %d", deviceNumaNode);
  }
}

ProxyHandlerResult SmDeviceChannelService::handleTrigger(ProxyTrigger triggerRaw) {
  ChannelTrigger* trigger = reinterpret_cast<ChannelTrigger*>(&triggerRaw);
  auto& conn = connections_[trigger->fields.chanId];

  auto result = ProxyHandlerResult::Continue;

  if (trigger->fields.type & TriggerData) {
    RegisteredMemory& dst = memories_[trigger->fields.dstMemoryId];
    RegisteredMemory& src = memories_[trigger->fields.srcMemoryId];
    conn->write(dst, trigger->fields.dstOffset, src, trigger->fields.srcOffset, trigger->fields.size);
  }

  if (trigger->fields.type & TriggerSync) {
    conn->flush();
    result = ProxyHandlerResult::FlushFifoTailAndContinue;
  }

  return result;
}

}  // namespace channel
}  // namespace mscclpp
