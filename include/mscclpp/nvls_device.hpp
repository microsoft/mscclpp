// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_NVLS_DEVICE_HPP_
#define MSCCLPP_NVLS_DEVICE_HPP_

#include <mscclpp/gpu.hpp>
#include <mscclpp/gpu_data_types.hpp>
#include <type_traits>

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
  template <int NElemPerThread = 4, typename TVaule = float4, typename T = float>
  MSCCLPP_DEVICE_INLINE void multimemLoad(TVaule& val, T* ptr) {
    static_assert(NElemPerThread == 4, "Only support NElemPerThread == 4");
    if constexpr (std::is_same<T, float>::value) {
      asm("multimem.ld_reduce.global.add.v4.f32 {%0,%1,%2,%3}, [%4];"
          : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
          : "l"(ptr)
          : "memory");
    } else if constexpr (std::is_same<T, half2>::value) {
      asm("multimem.ld_reduce.global.add.v4.f16x2 {%0,%1,%2,%3}, [%4];"
          : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
          : "l"(ptr)
          : "memory");
    } else {
      static_assert(dependentFalse<T>, "Not supported type");
    }
  };

  template <int NElemPerThread = 4, typename TVaule, typename T>
  MSCCLPP_DEVICE_INLINE void multimemStore(const TVaule& val, T* ptr) {
    static_assert(NElemPerThread == 4, "Only support NElemPerThread == 4");
    if constexpr (std::is_same<T, float>::value) {
      asm volatile("multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z),
                   "r"(val.w)
                   : "memory");
    } else if constexpr (std::is_same<T, half2>::value) {
      asm volatile("multimem.st.global.v4.f16x2 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z),
                   "r"(val.w)
                   : "memory");
    } else {
      static_assert(dependentFalse<T>, "Not supported type");
    }
  };
#endif  // defined(MSCCLPP_DEVICE_CUDA)
};

}  // namespace mscclpp

#endif  // MSCCLPP_SEMAPHORE_DEVICE_HPP_
