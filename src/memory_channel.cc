// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/memory_channel.hpp>

#include "api.h"
#include "debug.h"

namespace mscclpp {

MSCCLPP_API_CPP MemoryChannel::MemoryChannel(std::shared_ptr<MemoryDevice2DeviceSemaphore> semaphore,
                                             RegisteredMemory dst, void* src, void* getPacketBuffer)
    : semaphore_(semaphore), dst_(dst), src_(src), getPacketBuffer_(getPacketBuffer) {
  if (!dst.transports().has(Transport::CudaIpc)) {
    throw Error("MemoryChannel: dst must be registered with CudaIpc", ErrorCode::InvalidUsage);
  }
}

MSCCLPP_API_CPP MemoryChannel::DeviceHandle MemoryChannel::deviceHandle() const {
  return DeviceHandle{.semaphore_ = semaphore_->deviceHandle(),
                      .src_ = src_,
                      .dst_ = dst_.data(),
                      .getPacketBuffer_ = getPacketBuffer_,
                      .dstSize_ = dst_.size()};
}

}  // namespace mscclpp
