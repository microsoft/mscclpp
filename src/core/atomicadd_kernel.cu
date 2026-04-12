// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <mscclpp/atomic_device.hpp>
#include <mscclpp/gpu.hpp>
#include <mscclpp/gpu_utils.hpp>

#include "context.hpp"

namespace mscclpp {

// Kernel for atomic add on a signed 64-bit value.
// Uses mscclpp's atomicFetchAdd which wraps cuda::atomic_ref (CUDA) or __atomic_fetch_add (HIP).
__global__ void atomicAddI64Kernel(int64_t* dst, int64_t value) {
  (void)atomicFetchAdd<int64_t, scopeSystem>(dst, value, memoryOrderRelaxed);
}

void CudaIpcStream::atomicAdd(uint64_t* dst, int64_t value) {
  CudaDeviceGuard deviceGuard(deviceId_);

#if !defined(MSCCLPP_DEVICE_HIP)
  // On CUDA, the proxy thread cannot launch kernels or perform stream operations on the
  // primary context without deadlocking with the main thread's cudaStreamSynchronize().
  // The CUDA runtime uses a per-context lock; the main thread holds it while waiting for
  // the test kernel, and the proxy thread needs it to launch the atomicAdd kernel.
  // A separate CUDA context avoids this contention.
  if (!proxyAtomicCtx_) {
    CUdevice cuDevice;
    CUresult res = cuDeviceGet(&cuDevice, deviceId_);
    if (res != CUDA_SUCCESS) throw Error("cuDeviceGet failed", ErrorCode::InternalError);

    res = cuCtxCreate(&proxyAtomicCtx_, 0, cuDevice);
    if (res != CUDA_SUCCESS) throw Error("cuCtxCreate failed", ErrorCode::InternalError);

    cuCtxPopCurrent(nullptr);
  }

  cuCtxPushCurrent(proxyAtomicCtx_);

  if (!proxyAtomicStream_) {
    MSCCLPP_CUDATHROW(cudaStreamCreateWithFlags(&proxyAtomicStream_, cudaStreamNonBlocking));
  }

  int64_t* dstI64 = reinterpret_cast<int64_t*>(dst);
  atomicAddI64Kernel<<<1, 1, 0, proxyAtomicStream_>>>(dstI64, value);

  cuCtxPopCurrent(nullptr);
#else
  // On HIP, contexts do not provide true isolation (hipDeviceSynchronize blocks all streams
  // on the device regardless of context). However, hipStreamSynchronize on a specific stream
  // does NOT block kernel launches on other streams, so we can launch directly.
  if (!proxyAtomicStream_) {
    MSCCLPP_CUDATHROW(cudaStreamCreateWithFlags(&proxyAtomicStream_, cudaStreamNonBlocking));
  }

  int64_t* dstI64 = reinterpret_cast<int64_t*>(dst);
  atomicAddI64Kernel<<<1, 1, 0, proxyAtomicStream_>>>(dstI64, value);
#endif
}

}  // namespace mscclpp
