// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_ATOMIC_DEVICE_HPP_
#define MSCCLPP_ATOMIC_DEVICE_HPP_

#include <type_traits>

#include "device.hpp"

#if defined(MSCCLPP_DEVICE_CUDA)
#include <cuda/atomic>
#endif  // defined(MSCCLPP_DEVICE_CUDA)

namespace mscclpp {

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
  cuda::atomic_ref<T, Scope>{*ptr}.store(val, memoryOrder);
}

template <typename T, cuda::thread_scope Scope = cuda::thread_scope_system>
MSCCLPP_HOST_DEVICE_INLINE T atomicFetchAdd(T* ptr, const T& val, cuda::memory_order memoryOrder) {
  return cuda::atomic_ref<T, Scope>{*ptr}.fetch_add(val, memoryOrder);
}

/// Fire-and-forget atomic add using PTX `red` (reduce) — no return value, more efficient than `atom`.
/// Supports uint32_t, int32_t, uint64_t, float, double with explicit PTX; other types fall back to fetch_add.
#if defined(MSCCLPP_DEVICE_COMPILE)

// PTX `red` only supports relaxed and release semantics — all other memory orders (acquire, acq_rel,
// seq_cst) are mapped to release, which is the strongest ordering `red` provides.
#define MSCCLPP_RED_(TYPE, CSTR)                                                                          \
  do {                                                                                                    \
    if constexpr (Scope == cuda::thread_scope_block) {                                                    \
      if (memoryOrder == cuda::memory_order_relaxed)                                                      \
        asm volatile("red.relaxed.cta.global.add." #TYPE " [%0], %1;" ::"l"(ptr), #CSTR(val) : "memory"); \
      else                                                                                                \
        asm volatile("red.release.cta.global.add." #TYPE " [%0], %1;" ::"l"(ptr), #CSTR(val) : "memory"); \
    } else if constexpr (Scope == cuda::thread_scope_device) {                                            \
      if (memoryOrder == cuda::memory_order_relaxed)                                                      \
        asm volatile("red.relaxed.gpu.global.add." #TYPE " [%0], %1;" ::"l"(ptr), #CSTR(val) : "memory"); \
      else                                                                                                \
        asm volatile("red.release.gpu.global.add." #TYPE " [%0], %1;" ::"l"(ptr), #CSTR(val) : "memory"); \
    } else {                                                                                              \
      if (memoryOrder == cuda::memory_order_relaxed)                                                      \
        asm volatile("red.relaxed.sys.global.add." #TYPE " [%0], %1;" ::"l"(ptr), #CSTR(val) : "memory"); \
      else                                                                                                \
        asm volatile("red.release.sys.global.add." #TYPE " [%0], %1;" ::"l"(ptr), #CSTR(val) : "memory"); \
    }                                                                                                     \
  } while (0)

template <typename T, cuda::thread_scope Scope = cuda::thread_scope_system>
MSCCLPP_DEVICE_INLINE void atomicAdd(T* ptr, const T& val, cuda::memory_order memoryOrder) {
  if constexpr (std::is_same_v<T, uint32_t>) {
    MSCCLPP_RED_(u32, r);
  } else if constexpr (std::is_same_v<T, int32_t>) {
    MSCCLPP_RED_(s32, r);
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    MSCCLPP_RED_(u64, l);
  } else if constexpr (std::is_same_v<T, float>) {
    MSCCLPP_RED_(f32, f);
  } else if constexpr (std::is_same_v<T, double>) {
    MSCCLPP_RED_(f64, d);
  } else {
    // Generic fallback: nvcc won't optimize (void)fetch_add to red, but it's correct.
    (void)cuda::atomic_ref<T, Scope>{*ptr}.fetch_add(val, memoryOrder);
  }
}

#undef MSCCLPP_RED_

#endif  // defined(MSCCLPP_DEVICE_COMPILE)

#elif defined(MSCCLPP_DEVICE_HIP)

constexpr auto memoryOrderRelaxed = __ATOMIC_RELAXED;
constexpr auto memoryOrderAcquire = __ATOMIC_ACQUIRE;
constexpr auto memoryOrderRelease = __ATOMIC_RELEASE;
constexpr auto memoryOrderAcqRel = __ATOMIC_ACQ_REL;
constexpr auto memoryOrderSeqCst = __ATOMIC_SEQ_CST;

// HIP does not have thread scope enums like CUDA
constexpr auto scopeSystem = 0;
constexpr auto scopeDevice = 0;

template <typename T, int scope = scopeSystem>
MSCCLPP_HOST_DEVICE_INLINE T atomicLoad(const T* ptr, int memoryOrder) {
  return __atomic_load_n(ptr, memoryOrder);
}

template <typename T, int scope = scopeSystem>
MSCCLPP_HOST_DEVICE_INLINE void atomicStore(T* ptr, const T& val, int memoryOrder) {
  __atomic_store_n(ptr, val, memoryOrder);
}

template <typename T, int scope = scopeSystem>
MSCCLPP_HOST_DEVICE_INLINE T atomicFetchAdd(T* ptr, const T& val, int memoryOrder) {
  return __atomic_fetch_add(ptr, val, memoryOrder);
}

/// Fire-and-forget atomic add — hipcc optimizes (void)fetch_add to no-return atomic.
#if defined(MSCCLPP_DEVICE_COMPILE)
template <typename T, int scope = scopeSystem>
MSCCLPP_DEVICE_INLINE void atomicAdd(T* ptr, const T& val, int memoryOrder) {
  (void)__atomic_fetch_add(ptr, val, memoryOrder);
}
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

#endif  // defined(MSCCLPP_DEVICE_HIP)

}  // namespace mscclpp

#endif  // MSCCLPP_ATOMIC_DEVICE_HPP_
