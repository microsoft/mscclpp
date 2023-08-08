// #include "mscclpp_common.h"
#include "fifo_device.hpp"

// BEGIN_DEFINES //

#define KERNEL fifo

// END_DEFINES //

extern "C"
__global__ void __launch_bounds__(1024,1) KERNEL(mscclpp::FifoDeviceHandle fifo)
{
    mscclpp::ProxyTrigger trigger;
    trigger.fst = 123;
    fifo.push(trigger);
}