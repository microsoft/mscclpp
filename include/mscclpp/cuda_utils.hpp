#ifndef MSCCLPP_CUDA_UTILS_HPP_
#define MSCCLPP_CUDA_UTILS_HPP_

// #include <type_traits>
#include <cuda_runtime.h>

#include <cstring>
#include <memory>
#include <mscclpp/checks.hpp>

namespace mscclpp {

// A RAII guard that will cudaThreadExchangeStreamCaptureMode to cudaStreamCaptureModeRelaxed on construction and
// restore the previous mode on destruction. This is helpful when we want to avoid CUDA graph capture.
struct AvoidCudaGraphCaptureGuard {
  AvoidCudaGraphCaptureGuard() : mode_(cudaStreamCaptureModeRelaxed) {
    MSCCLPP_CUDATHROW(cudaThreadExchangeStreamCaptureMode(&mode_));
  }
  ~AvoidCudaGraphCaptureGuard() { cudaThreadExchangeStreamCaptureMode(&mode_); }
  cudaStreamCaptureMode mode_;
};

// A RAII wrapper around cudaStream_t that will call cudaStreamDestroy on destruction.
struct CudaStreamWithFlags {
  CudaStreamWithFlags(unsigned int flags) { MSCCLPP_CUDATHROW(cudaStreamCreateWithFlags(&stream_, flags)); }
  ~CudaStreamWithFlags() { cudaStreamDestroy(stream_); }
  operator cudaStream_t() const { return stream_; }
  cudaStream_t stream_;
};

namespace detail {

template <class T>
T* cudaCalloc(size_t nelem) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  T* ptr;
  CudaStreamWithFlags stream(cudaStreamNonBlocking);
  MSCCLPP_CUDATHROW(cudaMalloc(&ptr, nelem * sizeof(T)));
  MSCCLPP_CUDATHROW(cudaMemsetAsync(ptr, 0, nelem * sizeof(T), stream));
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
  return ptr;
}

template <class T>
T* cudaHostCalloc(size_t nelem) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  T* ptr;
  MSCCLPP_CUDATHROW(cudaHostAlloc(&ptr, nelem * sizeof(T), cudaHostAllocMapped | cudaHostAllocWriteCombined));
  memset(ptr, 0, nelem * sizeof(T));
  return ptr;
}

template <class T, T*(alloc)(size_t), class Deleter, class Memory>
Memory safeAlloc(size_t nelem) {
  T* ptr = nullptr;
  try {
    ptr = alloc(nelem);
  } catch (...) {
    if (ptr) {
      Deleter()(ptr);
    }
    throw;
  }
  return Memory(ptr, Deleter());
}

}  // namespace detail

// A deleter that calls cudaFree for use with std::unique_ptr/std::shared_ptr.
template <class T>
struct CudaDeleter {
  using TPtrOrArray = std::conditional_t<std::is_array_v<T>, T, T*>;
  void operator()(TPtrOrArray ptr) {
    AvoidCudaGraphCaptureGuard cgcGuard;
    MSCCLPP_CUDATHROW(cudaFree(ptr));
  }
};

// A deleter that calls cudaFreeHost for use with std::unique_ptr/std::shared_ptr.
template <class T>
struct CudaHostDeleter {
  using TPtrOrArray = std::conditional_t<std::is_array_v<T>, T, T*>;
  void operator()(TPtrOrArray ptr) {
    AvoidCudaGraphCaptureGuard cgcGuard;
    MSCCLPP_CUDATHROW(cudaFreeHost(ptr));
  }
};

// Allocates memory on the device and returns a std::shared_ptr to it. The memory is zeroed out.
template <class T>
std::shared_ptr<T> allocSharedCuda(size_t count = 1) {
  return detail::safeAlloc<T, detail::cudaCalloc<T>, CudaDeleter<T>, std::shared_ptr<T>>(count);
}

template <class T>
using UniqueCudaPtr = std::unique_ptr<T, CudaDeleter<T>>;

// Allocates memory on the device and returns a std::unique_ptr to it. The memory is zeroed out.
template <class T>
UniqueCudaPtr<T> allocUniqueCuda(size_t count = 1) {
  return detail::safeAlloc<T, detail::cudaCalloc<T>, CudaDeleter<T>, UniqueCudaPtr<T>>(count);
}

// Allocates memory with cudaHostAlloc, constructs an object of type T in it and returns a std::shared_ptr to it.
template <class T, typename... Args>
std::shared_ptr<T> makeSharedCudaHost(Args&&... args) {
  auto ptr = detail::safeAlloc<T, detail::cudaHostCalloc<T>, CudaHostDeleter<T>, std::shared_ptr<T>>(1);
  new (ptr.get()) T(std::forward<Args>(args)...);
  return ptr;
}

// Allocates an array of objects of type T with cudaHostAlloc, default constructs each element and returns a
// std::shared_ptr to it.
template <class T>
std::shared_ptr<T[]> makeSharedCudaHost(size_t count) {
  using TElem = std::remove_extent_t<T>;
  auto ptr = detail::safeAlloc<T, detail::cudaHostCalloc<T>, CudaHostDeleter<TElem>, std::shared_ptr<T[]>>(count);
  for (size_t i = 0; i < count; ++i) {
    new (&ptr[i]) TElem();
  }
  return ptr;
}

template <class T>
using UniqueCudaHostPtr = std::unique_ptr<T, CudaHostDeleter<T>>;

// Allocates memory with cudaHostAlloc, constructs an object of type T in it and returns a std::unique_ptr to it.
template <class T, typename... Args, std::enable_if_t<false == std::is_array_v<T>, bool> = true>
UniqueCudaHostPtr<T> makeUniqueCudaHost(Args&&... args) {
  auto ptr = detail::safeAlloc<T, detail::cudaHostCalloc<T>, CudaHostDeleter<T>, UniqueCudaHostPtr<T>>(1);
  new (ptr.get()) T(std::forward<Args>(args)...);
  return ptr;
}

// Allocates an array of objects of type T with cudaHostAlloc, default constructs each element and returns a
// std::unique_ptr to it.
template <class T, std::enable_if_t<true == std::is_array_v<T>, bool> = true>
UniqueCudaHostPtr<T> makeUniqueCudaHost(size_t count) {
  using TElem = std::remove_extent_t<T>;
  auto ptr = detail::safeAlloc<TElem, detail::cudaHostCalloc<TElem>, CudaHostDeleter<T>, UniqueCudaHostPtr<T>>(count);
  for (size_t i = 0; i < count; ++i) {
    new (&ptr[i]) TElem();
  }
  return ptr;
}

// Asynchronous cudaMemcpy without capture into a CUDA graph.
template <class T>
void memcpyCudaAsync(T* dst, const T* src, size_t count, cudaStream_t stream, cudaMemcpyKind kind = cudaMemcpyDefault) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(dst, src, count * sizeof(T), kind, stream));
}

// Synchronous cudaMemcpy without capture into a CUDA graph.
template <class T>
void memcpyCuda(T* dst, const T* src, size_t count, cudaMemcpyKind kind = cudaMemcpyDefault) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  CudaStreamWithFlags stream(cudaStreamNonBlocking);
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(dst, src, count * sizeof(T), kind, stream));
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
}

}  // namespace mscclpp

#endif  // MSCCLPP_CUDA_UTILS_HPP_
