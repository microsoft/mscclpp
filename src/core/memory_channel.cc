// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/memory_channel.hpp>

#include "api.h"
#include "debug.h"

namespace mscclpp {

MSCCLPP_API_CPP BaseMemoryChannel::BaseMemoryChannel(std::shared_ptr<MemoryDevice2DeviceSemaphore> semaphore)
    : semaphore_(semaphore) {}

MSCCLPP_API_CPP BaseMemoryChannel::BaseMemoryChannel(const Semaphore& semaphore)
    : BaseMemoryChannel(std::make_shared<MemoryDevice2DeviceSemaphore>(semaphore)) {}

MSCCLPP_API_CPP MemoryChannel::MemoryChannel(std::shared_ptr<MemoryDevice2DeviceSemaphore> semaphore,
                                             RegisteredMemory dst, RegisteredMemory src, void* packetBuffer)
    : BaseMemoryChannel(semaphore), dst_(dst), src_(src), packetBuffer_(packetBuffer) {
  if (!dst.transports().has(Transport::CudaIpc)) {
    throw Error("MemoryChannel: dst must be registered with CudaIpc", ErrorCode::InvalidUsage);
  }
}

MSCCLPP_API_CPP MemoryChannel::MemoryChannel(const Semaphore& semaphore, RegisteredMemory dst, RegisteredMemory src,
                                             void* packetBuffer)
    : MemoryChannel(std::make_shared<MemoryDevice2DeviceSemaphore>(semaphore), dst, src, packetBuffer) {}

MSCCLPP_API_CPP BaseMemoryChannel::DeviceHandle BaseMemoryChannel::deviceHandle() const {
  return BaseMemoryChannel::DeviceHandle(semaphore_->deviceHandle());
}

MSCCLPP_API_CPP MemoryChannel::DeviceHandle MemoryChannel::deviceHandle() const {
  return MemoryChannel::DeviceHandle(semaphore_->deviceHandle(), dst_.data(), src_.data(), packetBuffer_);
}

}  // namespace mscclpp
