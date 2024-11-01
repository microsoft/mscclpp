// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_GPU_UTILS_HPP_
#define MSCCLPP_GPU_UTILS_HPP_

#include <cstring>
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

#define MSCCLPP_CULOG_WARN(cmd)                             \
  do {                                                      \
    CUresult err = cmd;                                     \
    if (err != CUDA_SUCCESS) {                              \
      const char* errStr;                                   \
      if (cuGetErrorString(err, &errStr) != CUDA_SUCCESS) { \
        errStr = "failed to get error string";              \
      }                                                     \
      WARN("Call to " #cmd " failed, error is %s", errStr); \
    }                                                       \
  } while (false)

namespace mscclpp {

/// set memory access permission to read-write
/// @param base Base memory pointer.
/// @param size Size of the memory.
inline void setReadWriteMemoryAccess(void* base, size_t size) {
  CUmemAccessDesc accessDesc = {};
  int deviceId;
  MSCCLPP_CUDATHROW(cudaGetDevice(&deviceId));
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = deviceId;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  MSCCLPP_CUTHROW(cuMemSetAccess((CUdeviceptr)base, size, &accessDesc, 1));
}

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

template <class T>
struct CudaDeleter;

namespace detail {

/// A wrapper of cudaMalloc that sets the allocated memory to zero.
/// @tparam T Type of each element in the allocated memory.
/// @param nelem Number of elements to allocate.
/// @return A pointer to the allocated memory.
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

#if (CUDA_FABRIC_SUPPORTED)
template <class T>
T* cudaPhysicalCalloc(size_t nbytes, size_t gran) {
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
  MSCCLPP_CUTHROW(cuMemCreate(&memHandle, nbytes, &prop, 0 /*flags*/));

  T* devicePtr = nullptr;
  MSCCLPP_CUTHROW(cuMemAddressReserve((CUdeviceptr*)&devicePtr, nbytes, gran, 0U, 0));
  MSCCLPP_CUTHROW(cuMemMap((CUdeviceptr)devicePtr, nbytes, 0, memHandle, 0));
  setReadWriteMemoryAccess(devicePtr, nbytes);
  CudaStreamWithFlags stream(cudaStreamNonBlocking);
  MSCCLPP_CUDATHROW(cudaMemsetAsync(devicePtr, 0, nbytes, stream));
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));

  return devicePtr;
}
#endif

template <class T>
T* cudaExtCalloc(size_t nelem) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  T* ptr;
  CudaStreamWithFlags stream(cudaStreamNonBlocking);
#if defined(__HIP_PLATFORM_AMD__)
  MSCCLPP_CUDATHROW(hipExtMallocWithFlags((void**)&ptr, nelem * sizeof(T), hipDeviceMallocUncached));
#else
  MSCCLPP_CUDATHROW(cudaMalloc(&ptr, nelem * sizeof(T)));
#endif
  MSCCLPP_CUDATHROW(cudaMemsetAsync(ptr, 0, nelem * sizeof(T), stream));
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
  return ptr;
}

/// A wrapper of cudaHostAlloc that sets the allocated memory to zero.
/// @tparam T Type of each element in the allocated memory.
/// @param nelem Number of elements to allocate.
/// @return A pointer to the allocated memory.
template <class T>
T* cudaHostCalloc(size_t nelem) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  T* ptr;
  MSCCLPP_CUDATHROW(cudaHostAlloc(&ptr, nelem * sizeof(T), cudaHostAllocMapped | cudaHostAllocWriteCombined));
  memset(ptr, 0, nelem * sizeof(T));
  return ptr;
}

/// A template function that allocates memory while ensuring that the memory will be freed when the returned object is
/// destroyed.
/// @tparam T Type of each element in the allocated memory.
/// @tparam alloc A function that allocates memory.
/// @tparam Deleter A deleter that will be used to free the allocated memory.
/// @tparam Memory The type of the returned object.
/// @param nelem Number of elements to allocate.
/// @return An object of type @p Memory that will free the allocated memory when destroyed.
///
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

template <class T, T*(alloc)(size_t, size_t), class Deleter, class Memory>
Memory safeAlloc(size_t nelem, size_t gran) {
  if ((nelem * sizeof(T)) % gran) {
    throw Error("The request allocation size is not divisible by the required granularity:" +
                    std::to_string(nelem * sizeof(T)) + " vs " + std::to_string(gran),
                ErrorCode::InvalidUsage);
  }
  T* ptr = nullptr;
  try {
    ptr = alloc(nelem, gran);
  } catch (...) {
    if (ptr) {
      Deleter()(ptr);
    }
    throw;
  }
  return Memory(ptr, Deleter());
}

}  // namespace detail

/// A deleter that calls cudaFree for use with std::unique_ptr or std::shared_ptr.
/// @tparam T Type of each element in the allocated memory.
template <class T>
struct CudaDeleter {
  using TPtrOrArray = std::conditional_t<std::is_array_v<T>, T, T*>;
  void operator()(TPtrOrArray ptr) {
    AvoidCudaGraphCaptureGuard cgcGuard;
    MSCCLPP_CUDATHROW(cudaFree(ptr));
  }
};

template <class T>
struct CudaPhysicalDeleter {
  static_assert(!std::is_array_v<T>, "T must not be an array");
  void operator()(T* ptr) {
    AvoidCudaGraphCaptureGuard cgcGuard;
    CUmemGenericAllocationHandle handle;
    size_t size = 0;
    MSCCLPP_CUTHROW(cuMemRetainAllocationHandle(&handle, ptr));
    MSCCLPP_CUTHROW(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));
    MSCCLPP_CUTHROW(cuMemUnmap((CUdeviceptr)ptr, size));
    MSCCLPP_CUTHROW(cuMemRelease(handle));
    MSCCLPP_CUTHROW(cuMemAddressFree((CUdeviceptr)ptr, size));
  }
};

/// A deleter that calls cudaFreeHost for use with std::unique_ptr or std::shared_ptr.
/// @tparam T Type of each element in the allocated memory.
template <class T>
struct CudaHostDeleter {
  using TPtrOrArray = std::conditional_t<std::is_array_v<T>, T, T*>;
  void operator()(TPtrOrArray ptr) {
    AvoidCudaGraphCaptureGuard cgcGuard;
    MSCCLPP_CUDATHROW(cudaFreeHost(ptr));
  }
};

/// Allocates memory on the device and returns a std::shared_ptr to it. The memory is zeroed out.
/// @tparam T Type of each element in the allocated memory.
/// @param count Number of elements to allocate.
/// @return A std::shared_ptr to the allocated memory.
template <class T>
std::shared_ptr<T> allocSharedCuda(size_t count = 1) {
  return detail::safeAlloc<T, detail::cudaCalloc<T>, CudaDeleter<T>, std::shared_ptr<T>>(count);
}

#if (CUDA_FABRIC_SUPPORTED)
static inline size_t getMulticastGranularity(size_t size, CUmulticastGranularity_flags granFlag) {
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
#endif

/// Allocates physical memory on the device and returns a std::shared_ptr to it. The memory is zeroed out.
/// @tparam T Type of each element in the allocated memory.
/// @param count Number of elements to allocate.
/// @param gran the granularity of the allocation.
/// @return A std::shared_ptr to the allocated memory.
template <class T>
std::shared_ptr<T> allocSharedPhysicalCuda([[maybe_unused]] size_t count, [[maybe_unused]] size_t gran = 0) {
#if (CUDA_FABRIC_SUPPORTED)
  if (!isFabricSupported()) {
    throw Error("Only suupport GPU with Fabric support", ErrorCode::InvalidUsage);
  }
  if (count == 0) {
    return nullptr;
  }

  if (gran == 0) {
    gran = getMulticastGranularity(count * sizeof(T), CU_MULTICAST_GRANULARITY_RECOMMENDED);
  }
  size_t nbytes = (count * sizeof(T) + gran - 1) / gran * gran;
  return detail::safeAlloc<T, detail::cudaPhysicalCalloc<T>, CudaPhysicalDeleter<T>, std::shared_ptr<T>>(nbytes, gran);
#else
  throw Error("Only support GPU with Fabric support", ErrorCode::InvalidUsage);
#endif
}

/// Allocates memory on the device and returns a std::shared_ptr to it. The memory is zeroed out.
/// @tparam T Type of each element in the allocated memory.
/// @param count Number of elements to allocate.
/// @return A std::shared_ptr to the allocated memory.
template <class T>
std::shared_ptr<T> allocExtSharedCuda(size_t count = 1) {
  return detail::safeAlloc<T, detail::cudaExtCalloc<T>, CudaDeleter<T>, std::shared_ptr<T>>(count);
}

/// Unique device pointer that will call cudaFree on destruction.
/// @tparam T Type of each element in the allocated memory.
template <class T>
using UniqueCudaPtr = std::unique_ptr<T, CudaDeleter<T>>;

/// Allocates memory on the device and returns a std::unique_ptr to it. The memory is zeroed out.
/// @tparam T Type of each element in the allocated memory.
/// @param count Number of elements to allocate.
/// @return A std::unique_ptr to the allocated memory.
template <class T>
UniqueCudaPtr<T> allocUniqueCuda(size_t count = 1) {
  return detail::safeAlloc<T, detail::cudaCalloc<T>, CudaDeleter<T>, UniqueCudaPtr<T>>(count);
}

/// Allocates memory on the device and returns a std::unique_ptr to it. The memory is zeroed out.
/// @tparam T Type of each element in the allocated memory.
/// @param count Number of elements to allocate.
/// @return A std::unique_ptr to the allocated memory.
template <class T>
UniqueCudaPtr<T> allocExtUniqueCuda(size_t count = 1) {
  return detail::safeAlloc<T, detail::cudaExtCalloc<T>, CudaDeleter<T>, UniqueCudaPtr<T>>(count);
}

/// Allocates memory with cudaHostAlloc, constructs an object of type T in it and returns a std::shared_ptr to it.
/// @tparam T Type of the object to construct.
/// @tparam Args Types of the arguments to pass to the constructor.
/// @param args Arguments to pass to the constructor.
/// @return A std::shared_ptr to the allocated memory.
template <class T, typename... Args>
std::shared_ptr<T> makeSharedCudaHost(Args&&... args) {
  auto ptr = detail::safeAlloc<T, detail::cudaHostCalloc<T>, CudaHostDeleter<T>, std::shared_ptr<T>>(1);
  new (ptr.get()) T(std::forward<Args>(args)...);
  return ptr;
}

/// Allocates an array of objects of type T with cudaHostAlloc, default constructs each element and returns a
/// std::shared_ptr to it.
/// @tparam T Type of the object to construct.
/// @param count Number of elements to allocate.
/// @return A std::shared_ptr to the allocated memory.
template <class T>
std::shared_ptr<T[]> makeSharedCudaHost(size_t count) {
  using TElem = std::remove_extent_t<T>;
  auto ptr = detail::safeAlloc<T, detail::cudaHostCalloc<T>, CudaHostDeleter<TElem>, std::shared_ptr<T[]>>(count);
  for (size_t i = 0; i < count; ++i) {
    new (&ptr[i]) TElem();
  }
  return ptr;
}

/// Unique CUDA host pointer that will call cudaFreeHost on destruction.
/// @tparam T Type of each element in the allocated memory.
template <class T>
using UniqueCudaHostPtr = std::unique_ptr<T, CudaHostDeleter<T>>;

/// Allocates memory with cudaHostAlloc, constructs an object of type T in it and returns a std::unique_ptr to it.
/// @tparam T Type of the object to construct.
/// @tparam Args Types of the arguments to pass to the constructor.
/// @param args Arguments to pass to the constructor.
/// @return A std::unique_ptr to the allocated memory.
template <class T, typename... Args, std::enable_if_t<false == std::is_array_v<T>, bool> = true>
UniqueCudaHostPtr<T> makeUniqueCudaHost(Args&&... args) {
  auto ptr = detail::safeAlloc<T, detail::cudaHostCalloc<T>, CudaHostDeleter<T>, UniqueCudaHostPtr<T>>(1);
  new (ptr.get()) T(std::forward<Args>(args)...);
  return ptr;
}

/// Allocates an array of objects of type T with cudaHostAlloc, default constructs each element and returns a
/// std::unique_ptr to it.
/// @tparam T Type of the object to construct.
/// @param count Number of elements to allocate.
/// @return A std::unique_ptr to the allocated memory.
template <class T, std::enable_if_t<true == std::is_array_v<T>, bool> = true>
UniqueCudaHostPtr<T> makeUniqueCudaHost(size_t count) {
  using TElem = std::remove_extent_t<T>;
  auto ptr = detail::safeAlloc<TElem, detail::cudaHostCalloc<TElem>, CudaHostDeleter<T>, UniqueCudaHostPtr<T>>(count);
  for (size_t i = 0; i < count; ++i) {
    new (&ptr[i]) TElem();
  }
  return ptr;
}

/// Allocated physical memory on the device and returns a memory handle along with a virtual memory handle for it.
/// The memory is zeroed out.
/// @tparam T Type of each element in the allocated memory.
/// @param count Number of elements to allocate.
/// @param gran the granularity of the allocation.
/// @return A std::unique_ptr to the allocated memory.
template <class T>
std::unique_ptr<T> allocUniquePhysicalCuda([[maybe_unused]] size_t count, [[maybe_unused]] size_t gran = 0) {
#if (CUDA_FABRIC_SUPPORTED)
  if (!isFabricSupported()) {
    throw Error("Only suupport GPU with Fabric support", ErrorCode::InvalidUsage);
  }
  if (count == 0) {
    return nullptr;
  }

  if (gran == 0) {
    gran = getMulticastGranularity(count * sizeof(T), CU_MULTICAST_GRANULARITY_RECOMMENDED);
  }
  return detail::safeAlloc<T, detail::cudaPhysicalCalloc<T>, CudaPhysicalDeleter<T>,
                           std::unique_ptr<CudaPhysicalDeleter<T>, CudaDeleter<CudaPhysicalDeleter<T>>>>(count, gran);
#else
  throw Error("Only support GPU with Fabric support", ErrorCode::InvalidUsage);
#endif
}

/// Asynchronous cudaMemcpy without capture into a CUDA graph.
/// @tparam T Type of each element in the allocated memory.
/// @param dst Destination pointer.
/// @param src Source pointer.
/// @param count Number of elements to copy.
/// @param stream CUDA stream to use.
/// @param kind Type of cudaMemcpy to perform.
template <class T>
void memcpyCudaAsync(T* dst, const T* src, size_t count, cudaStream_t stream, cudaMemcpyKind kind = cudaMemcpyDefault) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(dst, src, count * sizeof(T), kind, stream));
}

/// Synchronous cudaMemcpy without capture into a CUDA graph.
/// @tparam T Type of each element in the allocated memory.
/// @param dst Destination pointer.
/// @param src Source pointer.
/// @param count Number of elements to copy.
/// @param kind Type of cudaMemcpy to perform.
template <class T>
void memcpyCuda(T* dst, const T* src, size_t count, cudaMemcpyKind kind = cudaMemcpyDefault) {
  AvoidCudaGraphCaptureGuard cgcGuard;
  CudaStreamWithFlags stream(cudaStreamNonBlocking);
  MSCCLPP_CUDATHROW(cudaMemcpyAsync(dst, src, count * sizeof(T), kind, stream));
  MSCCLPP_CUDATHROW(cudaStreamSynchronize(stream));
}

}  // namespace mscclpp

#endif  // MSCCLPP_GPU_UTILS_HPP_
