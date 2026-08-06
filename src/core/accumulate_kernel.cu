// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <mscclpp/gpu.hpp>

#if defined(MSCCLPP_USE_ROCM)

#include <mscclpp/atomic_device.hpp>
#include <mscclpp/gpu_utils.hpp>

#include "context.hpp"

namespace mscclpp {

// System-scope atomic add on a signed 64-bit value.
__global__ void accumulateI64Kernel(int64_t* dst, int64_t value) {
  (void)atomicFetchAdd<int64_t, scopeSystem>(dst, value, memoryOrderRelaxed);
}

void CudaIpcStream::accumulate(int64_t* dst, int64_t value) {
  CudaDeviceGuard deviceGuard(deviceId_);
  setStreamIfNeeded();
  // Submit to this connection's stream, which orders the add ahead of any signal or flush that
  // follows. On ROCm a kernel runs while the caller's kernel occupies the GPU, so the proxy does
  // not wait for the caller.
  accumulateI64Kernel<<<1, 1, 0, *stream_>>>(dst, value);
  MSCCLPP_CUDATHROW(cudaGetLastError());
  dirty_ = true;
}

}  // namespace mscclpp

#endif  // defined(MSCCLPP_USE_ROCM)
