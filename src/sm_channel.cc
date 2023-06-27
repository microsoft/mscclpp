// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/sm_channel.hpp>

#include "api.h"
#include "debug.h"

namespace mscclpp {
namespace channel {
namespace sm {

MSCCLPP_API_CPP SmChannel::SmChannel(SmDevice2DeviceEpoch::DeviceHandle epoch, RegisteredMemory dst, void* src,
                                     void* getPacketBuffer)
    : epoch_(epoch), src_(src), getPacketBuffer_(getPacketBuffer) {
  if (!dst.transports().has(Transport::CudaIpc)) {
    throw Error("SmChannel: dst must be registered with CudaIpc", ErrorCode::InvalidUsage);
  }
  dst_ = dst.data();
}

}  // namespace sm
}  // namespace channel
}  // namespace mscclpp
