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
  //
  // TODO(#796): `dst` is a CUDA-IPC mapping registered in the primary/runtime context, so
  // launching this kernel from `proxyAtomicCtx_` is technically UB (device pointers are
  // context-scoped). It works in practice on current drivers because the IPC handle aliases
  // the same physical allocation, but a correct fix would either (a) avoid the separate
  // context (e.g. break the deadlock differently) or (b) re-open the IPC mapping inside
  // `proxyAtomicCtx_`. Carried over from the DeepEP `chhwang/dev-atomic-add-cleanup`
  // cherry-pick; revisit before this lands on `main`.
  if (!proxyAtomicCtx_) {
    CUdevice cuDevice;
    CUresult res = cuDeviceGet(&cuDevice, deviceId_);
    if (res != CUDA_SUCCESS) throw Error("cuDeviceGet failed", ErrorCode::InternalError);

    // cuCtxCreate added a `paramsArray` argument in CUDA 12.5 — use the
    // 4-arg form on new toolkits, fall back to the legacy 3-arg form on
    // CUDA < 12.5 so we keep compiling against older drivers/toolkits.
#if CUDA_VERSION >= 12050
    res = cuCtxCreate_v4(&proxyAtomicCtx_, NULL, 0, cuDevice);
#else
    res = cuCtxCreate(&proxyAtomicCtx_vice);
#endif
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
