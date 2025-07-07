// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/poll_device.hpp>
#include <mscclpp/semaphore_device.hpp>
#include <mscclpp/switch_channel_device.hpp>

__device__ mscclpp::DeviceSyncer deviceSyncer;

extern "C" __global__ void __launch_bounds__(1024, 1)
    nvls_test(mscclpp::SwitchChannelDeviceHandle nvlsPtrs,
              mscclpp::MemoryDevice2DeviceSemaphoreDeviceHandle* semaphores, int my_rank, int nranks, int nbytes) {
  int nelem = nbytes / sizeof(float);
  float* dev_ptr = (float*)nvlsPtrs.devicePtr;
  mscclpp::f32x4* mc_ptr = (mscclpp::f32x4*)nvlsPtrs.mcPtr;
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  for (int idx = bid * blockDim.x + tid; idx < nelem; idx += blockDim.x * gridDim.x) {
    dev_ptr[idx] = my_rank;
  }
  deviceSyncer.sync(gridDim.x);
  if (tid == 0 && bid == 0) {
    __threadfence_system();
  }

  if (bid == 0) {
    if (tid < nranks && tid != my_rank) {
      semaphores[tid].signal();
      semaphores[tid].wait();
    }
  }
  deviceSyncer.sync(gridDim.x);

  int my_st = ((int64_t)nelem / 4 * (int64_t)my_rank) / (int64_t)nranks;
  int my_en = ((int64_t)nelem / 4 * (int64_t)(my_rank + 1)) / (int64_t)nranks;

  int my_offset = (tid + bid * blockDim.x);
  int my_step = blockDim.x * gridDim.x;

  for (int idx = my_st + my_offset; idx < my_en; idx += my_step) {
    mscclpp::f32x4 val = mscclpp::SwitchChannelDeviceHandle::multimemLoadReduce(mc_ptr + idx);
    mscclpp::SwitchChannelDeviceHandle::multimemStore(val, mc_ptr + idx);
  }

  deviceSyncer.sync(gridDim.x);
  if (tid == 0 && bid == 0) {
    __threadfence_system();
  }

  if (bid == 0) {
    if (tid < nranks && tid != my_rank) {
      semaphores[tid].signal();
      semaphores[tid].wait();
    }
  }
  deviceSyncer.sync(gridDim.x);

  for (int idx = bid * blockDim.x + tid; idx < nelem; idx += blockDim.x * gridDim.x) {
    MSCCLPP_ASSERT_DEVICE(dev_ptr[idx] == ((nranks * (nranks - 1)) / 2), "dev_ptr[idx] != nranks");
  }
}
