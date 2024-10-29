// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_NVLS_DEVICE_HPP_
#define MSCCLPP_NVLS_DEVICE_HPP_

#include <mscclpp/gpu.hpp>
#include <type_traits>

#if defined(MSCCLPP_DEVICE_CUDA)
#include <cuda_fp16.h>
#endif  // defined(MSCCLPP_DEVICE_CUDA)

#include <mscclpp/gpu_data_types.hpp>

#include "device.hpp"

namespace mscclpp {

template <class>
constexpr bool dependentFalse = false;  // workaround before CWG2518/P2593R1

/// Device-side handle for @ref Host2DeviceSemaphore.
struct DeviceMulticastPointerDeviceHandle {
  void* devicePtr;
  void* mcPtr;
  size_t bufferSize;

#if defined(MSCCLPP_DEVICE_CUDA)
  template <typename TValue, typename T>
  MSCCLPP_DEVICE_INLINE static void multimemLoadReduce(TValue& val, T* ptr) {
    if constexpr (std::is_same_v<TValue, int32_t> && std::is_same_v<T, int32_t>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.s32 %0, [%1];" : "=r"(val) : "l"(ptr) : "memory");
    } else if constexpr (std::is_same_v<TValue, uint32_t> && std::is_same_v<T, uint32_t>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.u32 %0, [%1];" : "=r"(val) : "l"(ptr) : "memory");
    } else if constexpr (std::is_same_v<TValue, uint4> && std::is_same_v<T, float>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.v4.f32 {%0,%1,%2,%3}, [%4];"
          : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
          : "l"(ptr)
          : "memory");
    } else if constexpr (std::is_same_v<TValue, uint2> && std::is_same_v<T, float>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.v2.f32 {%0,%1}, [%2];"
          : "=r"(val.x), "=r"(val.y)
          : "l"(ptr)
          : "memory");
    } else if constexpr (std::is_same_v<TValue, uint1> && std::is_same_v<T, float>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.f32 {%0}, [%1];" : "=r"(val.x) : "l"(ptr) : "memory");
    } else if constexpr (std::is_same_v<TValue, uint4> && std::is_same_v<T, __half2>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.v4.f16x2 {%0,%1,%2,%3}, [%4];"
          : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
          : "l"(ptr)
          : "memory");
    } else if constexpr (std::is_same_v<TValue, uint2> && std::is_same_v<T, __half2>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.v2.f16x2 {%0,%1}, [%2];"
          : "=r"(val.x), "=r"(val.y)
          : "l"(ptr)
          : "memory");
    } else if constexpr (std::is_same_v<TValue, uint1> && std::is_same_v<T, __half2>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.f16x2 {%0}, [%1];" : "=r"(val.x) : "l"(ptr) : "memory");
    } else if constexpr (std::is_same_v<TValue, uint4> && std::is_same_v<T, __bfloat162>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
          : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
          : "l"(ptr)
          : "memory");
    } else if constexpr (std::is_same_v<TValue, uint1> && std::is_same_v<T, __bfloat162>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.bf16x2 {%0}, [%1];" : "=r"(val.x) : "l"(ptr) : "memory");
    } else {
      static_assert(dependentFalse<T>, "Not supported type");
    }
  };

  template <typename TValue, typename T>
  MSCCLPP_DEVICE_INLINE static void multimemStore(const TValue& val, T* ptr) {
    if constexpr (std::is_same_v<TValue, int32_t> && std::is_same_v<T, int32_t>) {
      asm volatile("multimem.st.relaxed.sys.global.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
    } else if constexpr (std::is_same_v<TValue, uint32_t> && std::is_same_v<T, uint32_t>) {
      asm volatile("multimem.st.relaxed.sys.global.u32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
    } else if constexpr (std::is_same_v<TValue, uint4> && std::is_same_v<T, float>) {
      asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(val.x), "r"(val.y),
                   "r"(val.z), "r"(val.w)
                   : "memory");
    } else if constexpr (std::is_same_v<TValue, uint2> && std::is_same_v<T, float>) {
      asm volatile("multimem.st.relaxed.sys.global.v2.f32 [%0], {%1,%2};" ::"l"(ptr), "r"(val.x), "r"(val.y)
                   : "memory");
    } else if constexpr (std::is_same_v<TValue, uint1> && std::is_same_v<T, float>) {
      asm volatile("multimem.st.relaxed.sys.global.f32 [%0], {%1};" ::"l"(ptr), "r"(val.x) : "memory");
    } else if constexpr (std::is_same_v<TValue, uint4> && std::is_same_v<T, __half2>) {
      asm volatile("multimem.st.relaxed.sys.global.v4.f16x2 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(val.x), "r"(val.y),
                   "r"(val.z), "r"(val.w)
                   : "memory");
    } else if constexpr (std::is_same_v<TValue, uint2> && std::is_same_v<T, __half2>) {
      asm volatile("multimem.st.relaxed.sys.global.v2.f16x2 [%0], {%1,%2};" ::"l"(ptr), "r"(val.x), "r"(val.y)
                   : "memory");
    } else if constexpr (std::is_same_v<TValue, uint1> && std::is_same_v<T, __half2>) {
      asm volatile("multimem.st.relaxed.sys.global.f16x2 [%0], {%1};" ::"l"(ptr), "r"(val.x) : "memory");
    } else if constexpr (std::is_same_v<TValue, uint4> && std::is_same_v<T, __bfloat162>) {
      asm volatile("multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(val.x), "r"(val.y),
                   "r"(val.z), "r"(val.w)
                   : "memory");
    } else if constexpr (std::is_same_v<TValue, uint1> && std::is_same_v<T, __bfloat162>) {
      asm volatile("multimem.st.relaxed.sys.global.bf16x2 [%0], {%1};" ::"l"(ptr), "r"(val.x) : "memory");
    } else {
      static_assert(dependentFalse<T>, "Not supported type");
    }
  };

  template <typename TValue, typename T>
  MSCCLPP_DEVICE_INLINE static void multimemStoreReduce(const TValue& val, T* ptr) {
    if constexpr (std::is_same_v<TValue, float4> && std::is_same_v<T, float>) {
      asm volatile("multimem.red.relaxed.sys.global.add.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(val.x), "r"(val.y),
                   "r"(val.z), "r"(val.w)
                   : "memory");
    } else if constexpr (std::is_same_v<TValue, uint2> && std::is_same_v<T, float>) {
      asm volatile("multimem.red.relaxed.sys.global.add.v2.f32 [%0], {%1,%2};" ::"l"(ptr), "r"(val.x), "r"(val.y)
                   : "memory");
    } else if constexpr (std::is_same_v<TValue, uint1> && std::is_same_v<T, float>) {
      asm volatile("multimem.red.relaxed.sys.global.add.f32 [%0], {%1};" ::"l"(ptr), "r"(val.x) : "memory");
    } else if constexpr (std::is_same_v<TValue, uint4> && std::is_same_v<T, __half2>) {
      asm volatile("multimem.red.relaxed.sys.global.add.v4.f16x2 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(val.x),
                   "r"(val.y), "r"(val.z), "r"(val.w)
                   : "memory");
    } else if constexpr (std::is_same_v<TValue, uint2> && std::is_same_v<T, __half2>) {
      asm volatile("multimem.red.relaxed.sys.global.add.v2.f16x2 [%0], {%1,%2};" ::"l"(ptr), "r"(val.x), "r"(val.y)
                   : "memory");
    } else if constexpr (std::is_same_v<TValue, uint1> && std::is_same_v<T, __half2>) {
      asm volatile("multimem.red.relaxed.sys.global.add.f16x2 [%0], {%1};" ::"l"(ptr), "r"(val.x) : "memory");
    } else {
      static_assert(dependentFalse<T>, "Not supported type");
    }
  };
#endif  // defined(MSCCLPP_DEVICE_CUDA)
};

}  // namespace mscclpp

#endif  // MSCCLPP_NVLS_DEVICE_HPP_
