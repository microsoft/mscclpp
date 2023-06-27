// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/sm_channel.hpp>

#include "api.h"
#include "debug.h"

namespace mscclpp {
namespace channel {

MSCCLPP_API_CPP SmChannel::SmChannel(SmDevice2DeviceSemaphore::DeviceHandle semaphore, RegisteredMemory dst, void* src,
                                     void* getPacketBuffer)
    : semaphore_(semaphore), src_(src), getPacketBuffer_(getPacketBuffer) {
  if (!dst.transports().has(Transport::CudaIpc)) {
    throw Error("SmChannel: dst must be registered with CudaIpc", ErrorCode::InvalidUsage);
  }
  dst_ = dst.data();
}

}  // namespace channel
}  // namespace mscclpp
