#ifndef MSCCLPP_CUDA_UTILS_HPP_
#define MSCCLPP_CUDA_UTILS_HPP_

// #include <type_traits>
#include <cuda_runtime.h>

#include <cstring>
#include <memory>
#include <mscclpp/checks.hpp>

namespace mscclpp {

struct AvoidCudaGraphCaptureGuard {
  AvoidCudaGraphCaptureGuard() : mode_(cudaStreamCaptureModeRelaxed) {
    MSCCLPP_CUDATHROW(cudaThreadExchangeStreamCaptureMode(&mode_));
  }
  ~AvoidCudaGraphCaptureGuard() { cudaThreadExchangeStreamCaptureMode(&mode_); }
  cudaStreamCaptureMode mode_;
};

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
Memory safeMake(size_t nelem) {
  T* ptr = nullptr;
  try {
    ptr = alloc(nelem);
  } catch (...) {
    if (ptr) {
      Deleter()(ptr);
    }
  }
  return Memory(ptr, Deleter());
}

}  // namespace detail

template <class T>
struct CudaDeleter {
  void operator()(T* ptr) {
    AvoidCudaGraphCaptureGuard cgcGuard;
    MSCCLPP_CUDATHROW(cudaFree(ptr));
  }
};

template <class T>
struct CudaHostDeleter {
  void operator()(T* ptr) {
    AvoidCudaGraphCaptureGuard cgcGuard;
    MSCCLPP_CUDATHROW(cudaFreeHost(ptr));
  }
};

template <class T>
std::shared_ptr<T> makeSharedCuda(size_t count = 1) {
  return detail::safeMake<T, detail::cudaCalloc<T>, CudaDeleter<T>, std::shared_ptr<T>>(count);
}

template <class T>
using UniqueCudaPtr = std::unique_ptr<T, CudaDeleter<T>>;

template <class T>
UniqueCudaPtr<T> makeUniqueCuda(size_t count = 1) {
  return detail::safeMake<T, detail::cudaCalloc<T>, CudaDeleter<T>, UniqueCudaPtr<T>>(count);
}

template <class T>
std::shared_ptr<T> makeSharedCudaHost(size_t count = 1) {
  return detail::safeMake<T, detail::cudaHostCalloc<T>, CudaHostDeleter<T>, std::shared_ptr<T>>(count);
}

template <class T>
using UniqueCudaHostPtr = std::unique_ptr<T, CudaHostDeleter<T>>;

template <class T>
UniqueCudaHostPtr<T> makeUniqueCudaHost(size_t count = 1) {
  return detail::safeMake<T, detail::cudaHostCalloc<T>, CudaHostDeleter<T>, UniqueCudaHostPtr<T>>(count);
}

}  // namespace mscclpp

#endif  // MSCCLPP_CUDA_UTILS_HPP_