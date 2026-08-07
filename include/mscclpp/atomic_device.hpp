// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_ATOMIC_DEVICE_HPP_
#define MSCCLPP_ATOMIC_DEVICE_HPP_

#include "device.hpp"

#if defined(MSCCLPP_DEVICE_CUDA)
#include <cuda/atomic>
#endif  // defined(MSCCLPP_DEVICE_CUDA)

namespace mscclpp {

namespace detail {

template <typename T>
MSCCLPP_HOST_DEVICE_INLINE void atomicStoreBuiltin(T* ptr, const T& val, int memoryOrder) {
  // Keep the order compile-time constant: a runtime order becomes a locked
  // xchg in unoptimized host builds, which is prohibitively slow on BAR mappings.
  switch (memoryOrder) {
    case __ATOMIC_RELAXED:
      __atomic_store(ptr, &val, __ATOMIC_RELAXED);
      break;
    case __ATOMIC_RELEASE:
      __atomic_store(ptr, &val, __ATOMIC_RELEASE);
      break;
    default:
      __atomic_store(ptr, &val, __ATOMIC_SEQ_CST);
      break;
  }
}

}  // namespace detail

#if defined(MSCCLPP_DEVICE_CUDA)

constexpr cuda::memory_order memoryOrderRelaxed = cuda::memory_order_relaxed;
constexpr cuda::memory_order memoryOrderAcquire = cuda::memory_order_acquire;
constexpr cuda::memory_order memoryOrderRelease = cuda::memory_order_release;
constexpr cuda::memory_order memoryOrderAcqRel = cuda::memory_order_acq_rel;
constexpr cuda::memory_order memoryOrderSeqCst = cuda::memory_order_seq_cst;

constexpr cuda::thread_scope scopeSystem = cuda::thread_scope_system;
constexpr cuda::thread_scope scopeDevice = cuda::thread_scope_device;

template <typename T, cuda::thread_scope Scope = cuda::thread_scope_system>
MSCCLPP_HOST_DEVICE_INLINE T atomicLoad(T* ptr, cuda::memory_order memoryOrder) {
  return cuda::atomic_ref<T, Scope>{*ptr}.load(memoryOrder);
}

template <typename T, cuda::thread_scope Scope = cuda::thread_scope_system>
MSCCLPP_HOST_DEVICE_INLINE void atomicStore(T* ptr, const T& val, cuda::memory_order memoryOrder) {
#if defined(__CUDA_ARCH__)
  cuda::atomic_ref<T, Scope>{*ptr}.store(val, memoryOrder);
#else   // !defined(__CUDA_ARCH__)
  detail::atomicStoreBuiltin(ptr, val, static_cast<int>(memoryOrder));
#endif  // !defined(__CUDA_ARCH__)
}

template <typename T, cuda::thread_scope Scope = cuda::thread_scope_system>
MSCCLPP_HOST_DEVICE_INLINE T atomicFetchAdd(T* ptr, const T& val, cuda::memory_order memoryOrder) {
  return cuda::atomic_ref<T, Scope>{*ptr}.fetch_add(val, memoryOrder);
}

#else  // !defined(MSCCLPP_DEVICE_CUDA)

constexpr auto memoryOrderRelaxed = __ATOMIC_RELAXED;
constexpr auto memoryOrderAcquire = __ATOMIC_ACQUIRE;
constexpr auto memoryOrderRelease = __ATOMIC_RELEASE;
constexpr auto memoryOrderAcqRel = __ATOMIC_ACQ_REL;
constexpr auto memoryOrderSeqCst = __ATOMIC_SEQ_CST;

constexpr auto scopeSystem = 0;
constexpr auto scopeDevice = 0;

template <typename T, int scope = scopeSystem>
MSCCLPP_HOST_DEVICE_INLINE T atomicLoad(const T* ptr, int memoryOrder) {
  return __atomic_load_n(ptr, memoryOrder);
}

template <typename T, int scope = scopeSystem>
MSCCLPP_HOST_DEVICE_INLINE void atomicStore(T* ptr, const T& val, int memoryOrder) {
  detail::atomicStoreBuiltin(ptr, val, memoryOrder);
}

template <typename T, int scope = scopeSystem>
MSCCLPP_HOST_DEVICE_INLINE T atomicFetchAdd(T* ptr, const T& val, int memoryOrder) {
  return __atomic_fetch_add(ptr, val, memoryOrder);
}

#endif  // !defined(MSCCLPP_DEVICE_CUDA)

}  // namespace mscclpp

#endif  // MSCCLPP_ATOMIC_DEVICE_HPP_
