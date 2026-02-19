// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <cuda.h>
#include <mscclpp/gpu_utils.hpp>

#include "context.hpp"

namespace mscclpp {

// PTX for a minimal atomicAdd kernel, compiled for sm_70 (forward-compatible with sm_80, sm_90).
// The kernel takes a uint64_t pointer and a uint64_t value, and atomically adds the value.
static const char* kAtomicAddPtx = R"(
.version 7.0
.target sm_70
.address_size 64
.visible .entry atomicAdd_u64(
    .param .u64 dst_ptr,
    .param .u64 value
)
{
    .reg .u64 %rd<3>;
    ld.param.u64 %rd0, [dst_ptr];
    ld.param.u64 %rd1, [value];
    atom.global.add.u64 %rd2, [%rd0], %rd1;
    ret;
}
)";

void CudaIpcStream::atomicAdd(uint64_t* dst, uint64_t value) {
#if !defined(MSCCLPP_DEVICE_HIP)
  // Lazy initialization of the separate CUDA context for proxy atomic operations.
  // This context is separate from the primary context, so kernel launches here
  // do not deadlock with cudaDeviceSynchronize on the main thread.
  if (!proxyAtomicCtx_) {
    CUdevice cuDevice;
    CUresult res = cuDeviceGet(&cuDevice, deviceId_);
    if (res != CUDA_SUCCESS) throw Error("cuDeviceGet failed", ErrorCode::InternalError);

    res = cuCtxCreate(&proxyAtomicCtx_, 0, cuDevice);
    if (res != CUDA_SUCCESS) throw Error("cuCtxCreate failed", ErrorCode::InternalError);

    res = cuStreamCreate(&proxyAtomicStream_, CU_STREAM_NON_BLOCKING);
    if (res != CUDA_SUCCESS) throw Error("cuStreamCreate failed", ErrorCode::InternalError);

    CUmodule module;
    res = cuModuleLoadData(&module, kAtomicAddPtx);
    if (res != CUDA_SUCCESS) throw Error("cuModuleLoadData failed", ErrorCode::InternalError);

    res = cuModuleGetFunction(&proxyAtomicFunc_, module, "atomicAdd_u64");
    if (res != CUDA_SUCCESS) throw Error("cuModuleGetFunction failed", ErrorCode::InternalError);

    // Pop the context so we don't leave it current on this thread.
    cuCtxPopCurrent(nullptr);
  }

  // Push the atomic context, launch the kernel, pop the context.
  cuCtxPushCurrent(proxyAtomicCtx_);

  void* args[] = {&dst, &value};
  CUresult res = cuLaunchKernel(proxyAtomicFunc_,
                                1, 1, 1,   // grid
                                1, 1, 1,   // block
                                0,         // shared mem
                                proxyAtomicStream_,
                                args, nullptr);
  if (res != CUDA_SUCCESS) throw Error("cuLaunchKernel failed", ErrorCode::InternalError);

  cuCtxPopCurrent(nullptr);
  atomicDirty_ = true;
#else
  // HIP fallback: use D2H/H2D approach
  CudaDeviceGuard deviceGuard(deviceId_);
  setStreamIfNeeded();
  uint64_t current;
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(&current, dst, sizeof(uint64_t), cudaMemcpyDeviceToHost, *stream_));
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(*stream_));
  current += value;
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(dst, &current, sizeof(uint64_t), cudaMemcpyHostToDevice, *stream_));
  dirty_ = true;
#endif
}

}  // namespace mscclpp
