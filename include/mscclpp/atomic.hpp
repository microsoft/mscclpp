// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_ATOMIC_HPP_
#define MSCCLPP_ATOMIC_HPP_

#include "device.hpp"

#if defined(MSCCLPP_CUDA) || defined(MSCCLPP_CUDA_HOST)
#include <cuda/atomic>
#endif  // defined(MSCCLPP_CUDA) || defined(MSCCLPP_CUDA_HOST)

namespace mscclpp {

#if defined(MSCCLPP_CUDA) || defined(MSCCLPP_CUDA_HOST)

constexpr auto memoryOrderRelaxed = cuda::memory_order_relaxed;
constexpr auto memoryOrderAcquire = cuda::memory_order_acquire;
constexpr auto memoryOrderRelease = cuda::memory_order_release;
constexpr auto memoryOrderAcqRel = cuda::memory_order_acq_rel;
constexpr auto memoryOrderSeqCst = cuda::memory_order_seq_cst;

template <typename T>
MSCCLPP_HOST_DEVICE_INLINE T atomicLoad(const T* ptr, int memoryOrder) {
  return cuda::atomic_ref<T, cuda::thread_scope_system>{*ptr}.load(memoryOrder);
}

template <typename T>
MSCCLPP_HOST_DEVICE_INLINE void atomicStore(T* ptr, const T& val, int memoryOrder) {
  cuda::atomic_ref<T, cuda::thread_scope_system>{*ptr}.store(val, memoryOrder);
}

template <typename T>
MSCCLPP_HOST_DEVICE_INLINE T atomicFetchAdd(T* ptr, const T& val, int memoryOrder) {
  return cuda::atomic_ref<T, cuda::thread_scope_system>{*ptr}.fetch_add(val, memoryOrder);
}

#else

constexpr auto memoryOrderRelaxed = __ATOMIC_RELAXED;
constexpr auto memoryOrderAcquire = __ATOMIC_ACQUIRE;
constexpr auto memoryOrderRelease = __ATOMIC_RELEASE;
constexpr auto memoryOrderAcqRel = __ATOMIC_ACQ_REL;
constexpr auto memoryOrderSeqCst = __ATOMIC_SEQ_CST;

template <typename T>
MSCCLPP_HOST_DEVICE_INLINE T atomicLoad(const T* ptr, int memoryOrder) {
  return __atomic_load_n(ptr, memoryOrder);
}

template <typename T>
MSCCLPP_HOST_DEVICE_INLINE void atomicStore(T* ptr, const T& val, int memoryOrder) {
  __atomic_store_n(ptr, val, memoryOrder);
}

template <typename T>
MSCCLPP_HOST_DEVICE_INLINE T atomicFetchAdd(T* ptr, const T& val, int memoryOrder) {
  return __atomic_fetch_add(ptr, val, memoryOrder);
}

#endif

}  // namespace mscclpp

#endif  // MSCCLPP_ATOMIC_HPP_
