// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_GPU_UTILS_HPP_
#define MSCCLPP_GPU_UTILS_HPP_

#include <memory>

#include "errors.hpp"
#include "gpu.hpp"
#include "utils.hpp"

/// Throw @ref mscclpp::CudaError if @p cmd does not return cudaSuccess.
/// @param cmd The command to execute.
#define MSCCLPP_CUDATHROW(cmd)                                                                                       \
  do {                                                                                                               \
    cudaError_t err = cmd;                                                                                           \
    if (err != cudaSuccess) {                                                                                        \
      throw mscclpp::CudaError(std::string("Call to " #cmd " failed. ") + __FILE__ + ":" + std::to_string(__LINE__), \
                               err);                                                                                 \
    }                                                                                                                \
  } while (false)

/// Throw @ref mscclpp::CuError if @p cmd does not return CUDA_SUCCESS.
/// @param cmd The command to execute.
#define MSCCLPP_CUTHROW(cmd)                                                                                      \
  do {                                                                                                            \
    CUresult err = cmd;                                                                                           \
    if (err != CUDA_SUCCESS) {                                                                                    \
      throw mscclpp::CuError(std::string("Call to " #cmd " failed.") + __FILE__ + ":" + std::to_string(__LINE__), \
                             err);                                                                                \
    }                                                                                                             \
  } while (false)

namespace mscclpp {

/// A RAII guard that will cudaThreadExchangeStreamCaptureMode to cudaStreamCaptureModeRelaxed on construction and
/// restore the previous mode on destruction. This is helpful when we want to avoid CUDA graph capture.
struct AvoidCudaGraphCaptureGuard {
  AvoidCudaGraphCaptureGuard();
  ~AvoidCudaGraphCaptureGuard();
  cudaStreamCaptureMode mode_;
};

/// A RAII wrapper around cudaStream_t that will call cudaStreamDestroy on destruction.
struct CudaStreamWithFlags {
  CudaStreamWithFlags(unsigned int flags);
  ~CudaStreamWithFlags();
  operator cudaStream_t() const { return stream_; }
  cudaStream_t stream_;
};

namespace detail {

void setReadWriteMemoryAccess(void* base, size_t size);

void* gpuCalloc(size_t bytes);
void* gpuCallocHost(size_t bytes);
#if defined(__HIP_PLATFORM_AMD__)
void* gpuCallocUncached(size_t bytes);
#endif  // defined(__HIP_PLATFORM_AMD__)
#if (CUDA_NVLS_SUPPORTED)
void* gpuCallocPhysical(size_t bytes, size_t gran);
#endif  // CUDA_NVLS_SUPPORTED

void gpuFree(void* ptr);
void gpuFreeHost(void* ptr);
#if (CUDA_NVLS_SUPPORTED)
void gpuFreePhysical(void* ptr);
#endif  // CUDA_NVLS_SUPPORTED

void gpuMemcpyAsync(void* dst, const void* src, size_t bytes, cudaStream_t stream,
                    cudaMemcpyKind kind = cudaMemcpyDefault);
void gpuMemcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind = cudaMemcpyDefault);

/// A template function that allocates memory while ensuring that the memory will be freed when the returned object is
/// destroyed.
/// @tparam T Type of each element in the allocated memory.
/// @tparam alloc A function that allocates memory.
/// @tparam Deleter A deleter that will be used to free the allocated memory.
/// @tparam Memory The type of the returned object.
/// @param nelems Number of elements to allocate.
/// @return An object of type @p Memory that will free the allocated memory when destroyed.
///
template <class T, void*(alloc)(size_t), class Deleter, class Memory>
Memory safeAlloc(size_t nelems) {
  T* ptr = nullptr;
  try {
    ptr = reinterpret_cast<T*>(alloc(nelems * sizeof(T)));
  } catch (...) {
    if (ptr) {
      Deleter()(ptr);
    }
    throw;
  }
  return Memory(ptr, Deleter());
}

#if (CUDA_NVLS_SUPPORTED)

size_t getMulticastGranularity(size_t size, CUmulticastGranularity_flags granFlag);

template <class T, void*(alloc)(size_t, size_t), class Deleter, class Memory>
Memory safeAlloc(size_t nelems, size_t gran) {
  if (gran == 0) {
    gran = getMulticastGranularity(nelems * sizeof(T), CU_MULTICAST_GRANULARITY_RECOMMENDED);
  }
  nelems = ((nelems * sizeof(T) + gran - 1) / gran * gran) / sizeof(T);
  if ((nelems * sizeof(T)) % gran) {
    throw Error("The request allocation size is not divisible by the required granularity:" +
                    std::to_string(nelems * sizeof(T)) + " vs " + std::to_string(gran),
                ErrorCode::InvalidUsage);
  }
  T* ptr = nullptr;
  try {
    ptr = reinterpret_cast<T*>(alloc(nelems * sizeof(T), gran));
  } catch (...) {
    if (ptr) {
      Deleter()(ptr);
    }
    throw;
  }
  return Memory(ptr, Deleter());
}

#endif  // CUDA_NVLS_SUPPORTED

/// A deleter that calls gpuFree for use with std::unique_ptr or std::shared_ptr.
/// @tparam T Type of each element in the allocated memory.
template <class T = void>
struct GpuDeleter {
  void operator()(void* ptr) { gpuFree(ptr); }
};

/// A deleter that calls gpuFreeHost for use with std::unique_ptr or std::shared_ptr.
/// @tparam T Type of each element in the allocated memory.
template <class T = void>
struct GpuHostDeleter {
  void operator()(void* ptr) { gpuFreeHost(ptr); }
};

#if (CUDA_NVLS_SUPPORTED)
template <class T = void>
struct GpuPhysicalDeleter {
  void operator()(void* ptr) { gpuFreePhysical(ptr); }
};
#endif  // CUDA_NVLS_SUPPORTED

template <class T>
using UniqueGpuPtr = std::unique_ptr<T, detail::GpuDeleter<T>>;

template <class T>
using UniqueGpuHostPtr = std::unique_ptr<T, detail::GpuHostDeleter<T>>;

template <class T>
auto gpuCallocShared(size_t nelems = 1) {
  return detail::safeAlloc<T, detail::gpuCalloc, detail::GpuDeleter<T>, std::shared_ptr<T>>(nelems);
}

template <class T>
auto gpuCallocUnique(size_t nelems = 1) {
  return detail::safeAlloc<T, detail::gpuCalloc, detail::GpuDeleter<T>, UniqueGpuPtr<T>>(nelems);
}

template <class T>
auto gpuCallocHostShared(size_t nelems = 1) {
  return detail::safeAlloc<T, detail::gpuCallocHost, detail::GpuHostDeleter<T>, std::shared_ptr<T>>(nelems);
}

template <class T>
auto gpuCallocHostUnique(size_t nelems = 1) {
  return detail::safeAlloc<T, detail::gpuCallocHost, detail::GpuHostDeleter<T>, UniqueGpuHostPtr<T>>(nelems);
}

#if defined(__HIP_PLATFORM_AMD__)

template <class T>
auto gpuCallocUncachedShared(size_t nelems = 1) {
  return detail::safeAlloc<T, detail::gpuCallocUncached, detail::GpuDeleter<T>, std::shared_ptr<T>>(nelems);
}

template <class T>
auto gpuCallocUncachedUnique(size_t nelems = 1) {
  return detail::safeAlloc<T, detail::gpuCallocUncached, detail::GpuDeleter<T>, UniqueGpuPtr<T>>(nelems);
}

#endif  // defined(__HIP_PLATFORM_AMD__)

#if (CUDA_NVLS_SUPPORTED)

template <class T>
using UniqueGpuPhysicalPtr = std::unique_ptr<T, detail::GpuPhysicalDeleter<T>>;

template <class T>
auto gpuCallocPhysicalShared(size_t nelems = 1, size_t gran = 0) {
  return detail::safeAlloc<T, detail::gpuCallocPhysical, detail::GpuPhysicalDeleter<T>, std::shared_ptr<T>>(nelems,
                                                                                                            gran);
}

template <class T>
auto gpuCallocPhysicalUnique(size_t nelems = 1, size_t gran = 0) {
  return detail::safeAlloc<T, detail::gpuCallocPhysical, detail::GpuPhysicalDeleter<T>, UniqueGpuPhysicalPtr<T>>(nelems,
                                                                                                                 gran);
}

#endif  // CUDA_NVLS_SUPPORTED

}  // namespace detail

/// Allocates memory on the device and returns a std::shared_ptr to it. The memory is zeroed out.
/// The allocated memory space is specialized for MSCCL++ communication.
/// @tparam T Type of each element in the allocated memory.
/// @param nelems Number of elements to allocate.
/// @return A std::shared_ptr to the allocated memory.
template <class T = char>
std::shared_ptr<T> gpuMemAlloc(size_t nelems = 1) {
  if (nelems == 0) {
    return nullptr;
  }
#if (CUDA_NVLS_SUPPORTED)
  if (mscclpp::isNvlsSupported()) {
    return detail::gpuCallocPhysicalShared<T>(nelems);
  }
#endif  // CUDA_NVLS_SUPPORTED

#if defined(__HIP_PLATFORM_AMD__)
  return detail::gpuCallocUncachedShared<T>(nelems);
#else   // !defined(__HIP_PLATFORM_AMD__)
  return detail::gpuCallocShared<T>(nelems);
#endif  // !defined(__HIP_PLATFORM_AMD__)
}

template <class T = char>
void gpuMemcpyAsync(T* dst, const T* src, size_t nelems, cudaStream_t stream, cudaMemcpyKind kind = cudaMemcpyDefault) {
  detail::gpuMemcpyAsync(dst, src, nelems * sizeof(T), stream, kind);
}

template <class T = char>
void gpuMemcpy(T* dst, const T* src, size_t nelems, cudaMemcpyKind kind = cudaMemcpyDefault) {
  detail::gpuMemcpy(dst, src, nelems * sizeof(T), kind);
}

}  // namespace mscclpp

#endif  // MSCCLPP_GPU_UTILS_HPP_
