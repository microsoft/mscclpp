// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/fifo_device.hpp>
#include <mscclpp/semaphore_device.hpp>

#include "mscclpp_common.h"

// BEGIN_DEFINES //

#ifndef PARAMETRIZE
#define KERNEL proxy
#define N_SHARDS 8
#endif

// END_DEFINES //

extern "C" __global__ void __launch_bounds__(1024, 1)
    KERNEL(int my_rank, int nranks, mscclpp::FifoDeviceHandle fifo,
           Plist<mscclpp::Host2DeviceSemaphoreDeviceHandle, N_SHARDS> semaphores) {
  int tid = threadIdx.x;
  if (tid == 0) {
    mscclpp::ProxyTrigger trigger;
    trigger.fst = 123;
    trigger.snd = 0;
    uint64_t currentFifoHead = fifo.push(trigger);
    // wait for the work to be done in cpu side
    fifo.sync(currentFifoHead);
  }
  __syncthreads();
  if (tid < nranks && tid != my_rank) {
    semaphores[tid].wait();
  }
}
