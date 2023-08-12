#include <mscclpp/semaphore_device.hpp>

#include "mscclpp_common.h"

// BEGIN_DEFINES //

#ifndef PARAMETRIZE
#define KERNEL h2d_semaphore
#define N_SHARDS 8
#endif

// END_DEFINES //

// be careful about using semaphore[my_rank] as it is an invalid semaphore and it is there just for simplicity of
// indexing
extern "C" __global__ void __launch_bounds__(1024, 1)
    KERNEL(Plist<mscclpp::Host2DeviceSemaphoreDeviceHandle, N_SHARDS> semaphores, int my_rank, int nranks) {
  int tid = threadIdx.x;
  if (tid < nranks && tid != my_rank) semaphores[tid].wait();
}
