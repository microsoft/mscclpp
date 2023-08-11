#include "mscclpp_common.h"
#include <mscclpp/fifo_device.hpp>
#include <stdio.h>

// BEGIN_DEFINES //

#define KERNEL fifo

// END_DEFINES //

extern "C"
__global__ void __launch_bounds__(1024,1) KERNEL(mscclpp::FifoDeviceHandle fifo)
{
    mscclpp::ProxyTrigger trigger;
    trigger.fst = 123;
    printf("push trigger\n");
    fifo.push(trigger);
}
