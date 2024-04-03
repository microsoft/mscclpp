// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/device.hpp>

#include "execution_kernel.hpp"

#if defined(MSCCLPP_DEVICE_HIP)
#define __synclds() asm volatile("s_waitcnt lgkmcnt(0) \n s_barrier");
#endif  // defined(MSCCLPP_DEVICE_HIP)

namespace mscclpp {
__global__ void kernel(DeviceExecutionPlan* plan) {
  extern __shared__ int sharedMem[];
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  DeviceExecutionPlan* localPlan = plan + bid;
  for (int i = tid; i < sizeof(DeviceExecutionPlan); i += blockDim.x) {
    sharedMem[i] = ((int*)localPlan)[i];
  }
#if defined(MSCCLPP_DEVICE_HIP)
  __synclds();
#else   // !defined(MSCCLPP_DEVICE_HIP)
  __syncthreads();
#endif  // !defined(MSCCLPP_DEVICE_HIP)
}

void ExecutionKernel::launchKernel(int nthreadblocks, int nthreads, DeviceExecutionPlan* plan, size_t sharedMemSize,
                                   cudaStream_t stream) {
  kernel<<<nthreadblocks, nthreads, sharedMemSize, stream>>>(plan);
}
}  // namespace mscclpp
