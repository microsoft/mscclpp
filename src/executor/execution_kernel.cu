// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "execution_kernel.hpp"

namespace mscclpp {
__global__ void kernel(DeviceExecutionPlan* plan) {}

void ExecutionKernel::launchKernel(int nthreadblocks, int nthreads, DeviceExecutionPlan* plan, cudaStream_t stream) {
  kernel<<<nthreadblocks, nthreads, 0, stream>>>(plan);
}
}  // namespace mscclpp
