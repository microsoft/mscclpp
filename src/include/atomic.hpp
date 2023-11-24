// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_ATOMIC_HPP_
#define MSCCLPP_ATOMIC_HPP_

#if !defined(__HIP_PLATFORM_AMD__)
#include <cuda/atomic>
#endif  // !defined(__HIP_PLATFORM_AMD__)

namespace mscclpp {

#if !defined(__HIP_PLATFORM_AMD__)

constexpr cuda::memory_order memoryOrderRelaxed = cuda::memory_order_relaxed;
constexpr cuda::memory_order memoryOrderAcquire = cuda::memory_order_acquire;
constexpr cuda::memory_order memoryOrderRelease = cuda::memory_order_release;
constexpr cuda::memory_order memoryOrderAcqRel = cuda::memory_order_acq_rel;
constexpr cuda::memory_order memoryOrderSeqCst = cuda::memory_order_seq_cst;

template <typename T>
inline T atomicLoad(T* ptr, cuda::memory_order memoryOrder) {
  return cuda::atomic_ref<T, cuda::thread_scope_system>{*ptr}.load(memoryOrder);
}

template <typename T>
inline void atomicStore(T* ptr, const T& val, cuda::memory_order memoryOrder) {
  cuda::atomic_ref<T, cuda::thread_scope_system>{*ptr}.store(val, memoryOrder);
}

template <typename T>
inline T atomicFetchAdd(T* ptr, const T& val, cuda::memory_order memoryOrder) {
  return cuda::atomic_ref<T, cuda::thread_scope_system>{*ptr}.fetch_add(val, memoryOrder);
}

#else

// TODO: use `std::atomic` in C++20 if we are ready to upgrade the C++ standard.

constexpr auto memoryOrderRelaxed = __ATOMIC_RELAXED;
constexpr auto memoryOrderAcquire = __ATOMIC_ACQUIRE;
constexpr auto memoryOrderRelease = __ATOMIC_RELEASE;
constexpr auto memoryOrderAcqRel = __ATOMIC_ACQ_REL;
constexpr auto memoryOrderSeqCst = __ATOMIC_SEQ_CST;

template <typename T>
inline T atomicLoad(const T* ptr, int memoryOrder) {
  return __atomic_load_n(ptr, memoryOrder);
}

template <typename T>
inline void atomicStore(T* ptr, const T& val, int memoryOrder) {
  __atomic_store_n(ptr, val, memoryOrder);
}

template <typename T>
inline T atomicFetchAdd(T* ptr, const T& val, int memoryOrder) {
  return __atomic_fetch_add(ptr, val, memoryOrder);
}

#endif

}  // namespace mscclpp

#endif  // MSCCLPP_ATOMIC_HPP_
