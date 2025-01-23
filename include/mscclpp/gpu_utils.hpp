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
  CudaStreamWithFlags() : stream_(nullptr) {}
  CudaStreamWithFlags(unsigned int flags);
  ~CudaStreamWithFlags();
  void set(unsigned int flags);
  bool empty() const;
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
void* gpuCallocPhysical(size_t bytes, size_t gran = 0, size_t align = 0);
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
/// @tparam Deleter A deleter that will be used to free the allocated memory.
/// @tparam Memory The type of the returned object.
/// @tparam Alloc A function type that allocates memory.
/// @tparam Args Input types of the @p alloc function variables.
/// @param alloc A function that allocates memory.
/// @param nelems Number of elements to allocate.
/// @param args Extra input variables for the @p alloc function.
/// @return An object of type @p Memory that will free the allocated memory when destroyed.
///
template <class T, class Deleter, class Memory, typename Alloc, typename... Args>
Memory safeAlloc(Alloc alloc, size_t nelems, Args&&... args) {
  T* ptr = nullptr;
  try {
    ptr = reinterpret_cast<T*>(alloc(nelems * sizeof(T), std::forward<Args>(args)...));
  } catch (...) {
    if (ptr) {
      Deleter()(ptr);
    }
    throw;
  }
  return Memory(ptr, Deleter());
}

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
  return detail::safeAlloc<T, detail::GpuDeleter<T>, std::shared_ptr<T>>(detail::gpuCalloc, nelems);
}

template <class T>
auto gpuCallocUnique(size_t nelems = 1) {
  return detail::safeAlloc<T, detail::GpuDeleter<T>, UniqueGpuPtr<T>>(detail::gpuCalloc, nelems);
}

template <class T>
auto gpuCallocHostShared(size_t nelems = 1) {
  return detail::safeAlloc<T, detail::GpuHostDeleter<T>, std::shared_ptr<T>>(detail::gpuCallocHost, nelems);
}

template <class T>
auto gpuCallocHostUnique(size_t nelems = 1) {
  return detail::safeAlloc<T, detail::GpuHostDeleter<T>, UniqueGpuHostPtr<T>>(detail::gpuCallocHost, nelems);
}

#if defined(__HIP_PLATFORM_AMD__)

template <class T>
auto gpuCallocUncachedShared(size_t nelems = 1) {
  return detail::safeAlloc<T, detail::GpuDeleter<T>, std::shared_ptr<T>>(detail::gpuCallocUncached, nelems);
}

template <class T>
auto gpuCallocUncachedUnique(size_t nelems = 1) {
  return detail::safeAlloc<T, detail::GpuDeleter<T>, UniqueGpuPtr<T>>(detail::gpuCallocUncached, nelems);
}

#endif  // defined(__HIP_PLATFORM_AMD__)

#if (CUDA_NVLS_SUPPORTED)

template <class T>
using UniqueGpuPhysicalPtr = std::unique_ptr<T, detail::GpuPhysicalDeleter<T>>;

template <class T>
auto gpuCallocPhysicalShared(size_t nelems = 1, size_t gran = 0, size_t align = 0) {
  return detail::safeAlloc<T, detail::GpuPhysicalDeleter<T>, std::shared_ptr<T>>(detail::gpuCallocPhysical, nelems,
                                                                                 gran, align);
}

template <class T>
auto gpuCallocPhysicalUnique(size_t nelems = 1, size_t gran = 0, size_t align = 0) {
  return detail::safeAlloc<T, detail::GpuPhysicalDeleter<T>, UniqueGpuPhysicalPtr<T>>(detail::gpuCallocPhysical, nelems,
                                                                                      gran, align);
}

size_t getMulticastGranularity(size_t size, CUmulticastGranularity_flags granFlag);

#endif  // CUDA_NVLS_SUPPORTED

}  // namespace detail

template <class T = char>
void gpuMemcpyAsync(T* dst, const T* src, size_t nelems, cudaStream_t stream, cudaMemcpyKind kind = cudaMemcpyDefault) {
  detail::gpuMemcpyAsync(dst, src, nelems * sizeof(T), stream, kind);
}

template <class T = char>
void gpuMemcpy(T* dst, const T* src, size_t nelems, cudaMemcpyKind kind = cudaMemcpyDefault) {
  detail::gpuMemcpy(dst, src, nelems * sizeof(T), kind);
}

/// Check if NVLink SHARP (NVLS) is supported.
///
/// @return True if NVLink SHARP (NVLS) is supported, false otherwise.
bool isNvlsSupported();

/// Allocates a GPU memory space specialized for communication. The memory is zeroed out. Get the device pointer by
/// `GpuBuffer::data()`.
///
/// Use this function for communication buffers, i.e., only when other devices (CPU, GPU, NIC, etc.) may access this
/// memory space at the same time with the local device (GPU). Running heavy computation over this memory space
/// may perform bad and is not recommended in general.
///
/// The allocated memory space is managed by the `memory_` object, not by the class instance. Which means,
/// the class destructor will NOT free the allocated memory if `memory_` is shared with and alive in other contexts.
///
/// @tparam T Type of each element in the allocated memory. Default is `char`.
///
template <class T = char>
class GpuBuffer {
 public:
  /// Constructs a GpuBuffer with the specified number of elements.
  /// @param nelems Number of elements to allocate. If it is zero, `data()` will return a null pointer.
  GpuBuffer(size_t nelems) : nelems_(nelems) {
    if (nelems == 0) {
      bytes_ = 0;
      return;
    }
#if (CUDA_NVLS_SUPPORTED)
    if (isNvlsSupported()) {
      size_t gran = detail::getMulticastGranularity(nelems * sizeof(T), CU_MULTICAST_GRANULARITY_RECOMMENDED);
      bytes_ = (nelems * sizeof(T) + gran - 1) / gran * gran / sizeof(T) * sizeof(T);
      memory_ = detail::gpuCallocPhysicalShared<T>(nelems, gran);
      return;
    }
#endif  // CUDA_NVLS_SUPPORTED

    bytes_ = nelems * sizeof(T);
#if defined(__HIP_PLATFORM_AMD__)
    memory_ = detail::gpuCallocUncachedShared<T>(nelems);
#else   // !defined(__HIP_PLATFORM_AMD__)
    memory_ = detail::gpuCallocShared<T>(nelems);
#endif  // !defined(__HIP_PLATFORM_AMD__)
  }

  /// Returns the number of elements in the allocated memory.
  /// @return The number of elements.
  size_t nelems() const { return nelems_; }

  /// Returns the number of bytes that is actually allocated. This may be larger than `nelems() * sizeof(T)`.
  /// @return The number of bytes.
  size_t bytes() const { return bytes_; }

  /// Returns the shared pointer to the allocated memory.
  /// If `nelems()` is zero, this function will return an empty shared pointer.
  /// @return A `std::shared_ptr` to the allocated memory.
  std::shared_ptr<T> memory() { return memory_; }

  /// Returns the device pointer to the allocated memory. Equivalent to `memory().get()`.
  /// If `nelems()` is zero, this function will return a null pointer.
  /// @return A device pointer to the allocated memory.
  T* data() { return memory_.get(); }

 private:
  size_t nelems_;
  size_t bytes_;
  std::shared_ptr<T> memory_;
};

}  // namespace mscclpp

#endif  // MSCCLPP_GPU_UTILS_HPP_
