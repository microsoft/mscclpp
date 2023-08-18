// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <stdio.h>

#include "mscclpp/fifo_device.hpp"
#include "mscclpp_common.h"

// BEGIN_DEFINES //

#ifndef PARAMETRIZE
#define KERNEL fifo
#endif

// END_DEFINES //

extern "C" __global__ void __launch_bounds__(1024, 1) KERNEL(mscclpp::FifoDeviceHandle fifo) {
  mscclpp::ProxyTrigger trigger;
  trigger.fst = 123;
  fifo.push(trigger);
}
