// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/atomic_device.hpp>

#include "connection_kernel.hpp"

namespace mscclpp {

__global__ void connectionAtomicAddKernel(uint64_t *dst, uint64_t value) {
  atomicFetchAdd(dst, value, memoryOrderRelease);
}

cudaError_t connectionAtomicAdd(uint64_t *dst, uint64_t value, cudaStream_t stream) {
  connectionAtomicAddKernel<<<1, 1, 0, stream>>>(dst, value);
  return cudaGetLastError();
}

}  // namespace mscclpp
