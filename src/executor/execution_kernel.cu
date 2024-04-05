// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "execution_kernel.hpp"

namespace mscclpp {
void ExecutionKernel::launchKernel(int rank, int nthreadblocks, int nthreads, void* src, void* dst, void* scratch,
                                   DataType dataType, DeviceExecutionPlan* plan, size_t sharedMemSize,
                                   cudaStream_t stream) {
  switch (dataType) {
    case DataType::INT32:
      executionKernel<int32_t><<<nthreadblocks, nthreads, sharedMemSize, stream>>>(rank, (int32_t*)src, (int32_t*)dst,
                                                                                   (int32_t*)scratch, plan);
      break;
    case DataType::UINT32:
      executionKernel<uint32_t><<<nthreadblocks, nthreads, sharedMemSize, stream>>>(
          rank, (uint32_t*)src, (uint32_t*)dst, (uint32_t*)scratch, plan);
      break;
    case DataType::FLOAT16:
      executionKernel<half>
          <<<nthreadblocks, nthreads, sharedMemSize, stream>>>(rank, (half*)src, (half*)dst, (half*)scratch, plan);
      break;
    case DataType::FLOAT32:
      executionKernel<float>
          <<<nthreadblocks, nthreads, sharedMemSize, stream>>>(rank, (float*)src, (float*)dst, (float*)scratch, plan);
      break;
  }
}
}  // namespace mscclpp
