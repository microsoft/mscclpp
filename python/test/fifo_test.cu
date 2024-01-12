// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/fifo_device.hpp>

extern "C" __global__ void LAUNCH_BOUNDS fifo(mscclpp::FifoDeviceHandle fifo) {
  mscclpp::ProxyTrigger trigger;
  trigger.fst = 123;
  fifo.push(trigger);
}
