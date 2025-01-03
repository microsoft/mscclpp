// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <cstring>
#include <mscclpp/gpu.hpp>
#include <mscclpp/gpu_utils.hpp>

namespace mscclpp {

namespace detail {

/// set memory access permission to read-write
/// @param base Base memory pointer.
/// @param size Size of the memory.
void setReadWriteMemoryAccess(void* base, size_t size) {
  CUmemAccessDesc accessDesc = {};
  int deviceId;
  MSCCLPP_CUDATHROW(cudaGetDevice(&deviceId));
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = deviceId;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  MSCCLPP_CUTHROW(cuMemSetAccess((CUdeviceptr)base, size, &accessDesc, 1));
}

void* gpuCalloc(size_t bytes) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  void* ptr;
  CudaStreamWithFlags stream(cudaStreamNonBlocking);
  MSCCLPP_CUDATHROW(cudaMalloc(&ptr, bytes));
  MSCCLPP_CUDATHROW(cudaMemsetAsync(ptr, 0, bytes, stream));
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
  return ptr;
}

void* gpuCallocHost(size_t bytes) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  void* ptr;
  MSCCLPP_CUDATHROW(cudaHostAlloc(&ptr, bytes, cudaHostAllocMapped | cudaHostAllocWriteCombined));
  ::memset(ptr, 0, bytes);
  return ptr;
}

#if defined(__HIP_PLATFORM_AMD__)
void* gpuCallocUncached(size_t bytes) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  void* ptr;
  CudaStreamWithFlags stream(cudaStreamNonBlocking);
  MSCCLPP_CUDATHROW(hipExtMallocWithFlags((void**)&ptr, bytes, hipDeviceMallocUncached));
  MSCCLPP_CUDATHROW(cudaMemsetAsync(ptr, 0, bytes, stream));
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
  return ptr;
}
#endif  // defined(__HIP_PLATFORM_AMD__)

#if (CUDA_NVLS_SUPPORTED)
void* gpuCallocPhysical(size_t bytes, size_t gran) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  int deviceId = -1;
  CUdevice currentDevice;
  MSCCLPP_CUDATHROW(cudaGetDevice(&deviceId));
  MSCCLPP_CUTHROW(cuDeviceGet(&currentDevice, deviceId));

  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.requestedHandleTypes =
      (CUmemAllocationHandleType)(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR | CU_MEM_HANDLE_TYPE_FABRIC);
  prop.location.id = currentDevice;

  // allocate physical memory
  CUmemGenericAllocationHandle memHandle;
  size_t nbytes = (bytes + gran - 1) / gran * gran;
  MSCCLPP_CUTHROW(cuMemCreate(&memHandle, nbytes, &prop, 0 /*flags*/));

  void* devicePtr = nullptr;
  MSCCLPP_CUTHROW(cuMemAddressReserve((CUdeviceptr*)&devicePtr, nbytes, gran, 0U, 0));
  MSCCLPP_CUTHROW(cuMemMap((CUdeviceptr)devicePtr, nbytes, 0, memHandle, 0));
  setReadWriteMemoryAccess(devicePtr, nbytes);
  CudaStreamWithFlags stream(cudaStreamNonBlocking);
  MSCCLPP_CUDATHROW(cudaMemsetAsync(devicePtr, 0, nbytes, stream));
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));

  return devicePtr;
}
#endif  // CUDA_NVLS_SUPPORTED

void gpuFree(void* ptr) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  MSCCLPP_CUDATHROW(cudaFree(ptr));
}

void gpuFreeHost(void* ptr) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  MSCCLPP_CUDATHROW(cudaFreeHost(ptr));
}

#if (CUDA_NVLS_SUPPORTED)
void gpuFreePhysical(void* ptr) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  CUmemGenericAllocationHandle handle;
  size_t size = 0;
  MSCCLPP_CUTHROW(cuMemRetainAllocationHandle(&handle, ptr));
  MSCCLPP_CUTHROW(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));
  MSCCLPP_CUTHROW(cuMemUnmap((CUdeviceptr)ptr, size));
  MSCCLPP_CUTHROW(cuMemRelease(handle));
  MSCCLPP_CUTHROW(cuMemAddressFree((CUdeviceptr)ptr, size));
}
#endif  // CUDA_NVLS_SUPPORTED

#if (CUDA_NVLS_SUPPORTED)
size_t getMulticastGranularity(size_t size, CUmulticastGranularity_flags granFlag) {
  size_t gran = 0;
  int numDevices = 0;
  MSCCLPP_CUDATHROW(cudaGetDeviceCount(&numDevices));

  CUmulticastObjectProp prop = {};
  prop.size = size;
  // This is a dummy value, it might affect the granularity in the future
  prop.numDevices = numDevices;
  prop.handleTypes = (CUmemAllocationHandleType)(CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR | CU_MEM_HANDLE_TYPE_FABRIC);
  prop.flags = 0;
  MSCCLPP_CUTHROW(cuMulticastGetGranularity(&gran, &prop, granFlag));
  return gran;
}
#endif  // CUDA_NVLS_SUPPORTED

void gpuMemcpyAsync(void* dst, const void* src, size_t bytes, cudaStream_t stream, cudaMemcpyKind kind) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(dst, src, bytes, kind, stream));
}

void gpuMemcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  CudaStreamWithFlags stream(cudaStreamNonBlocking);
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(dst, src, bytes, kind, stream));
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
}

}  // namespace detail

}  // namespace mscclpp
