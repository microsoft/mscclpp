// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/atomic_device.hpp>

#include "connection_kernels.hpp"

namespace mscclpp {

__global__ void connectionAtomicAddKernel(uint64_t* dst, uint64_t value) {
  atomicFetchAdd(dst, value, memoryOrderRelaxed);
}

const void* connectionAtomicAddKernelFunc() {
  static const void* func = reinterpret_cast<const void*>(&connectionAtomicAddKernel);
  return func;
}

}  // namespace mscclpp
