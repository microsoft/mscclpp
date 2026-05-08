// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_SWITCH_CHANNEL_DEVICE_HPP_
#define MSCCLPP_SWITCH_CHANNEL_DEVICE_HPP_

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

/// Device-side handle for SwitchChannel.
struct SwitchChannelDeviceHandle {
  void* devicePtr;
  void* mcPtr;
  size_t bufferSize;

#if defined(MSCCLPP_DEVICE_CUDA)
  template <typename T>
  MSCCLPP_DEVICE_INLINE T reduce(uint64_t index) {
    return SwitchChannelDeviceHandle::multimemLoadReduce(reinterpret_cast<T*>(mcPtr) + index);
  }

  template <typename T>
  MSCCLPP_DEVICE_INLINE void broadcast(uint64_t index, const T& val) {
    SwitchChannelDeviceHandle::multimemStore(val, reinterpret_cast<T*>(mcPtr) + index);
  }

  /// Vectorized multimem load+reduce. The optional `AccumT` template parameter selects the
  /// accumulator: when `AccumT == __half` and `VectorType` is an FP8 vector type, the
  /// `.acc::f16` variant of the instruction is used (faster but lower precision than the
  /// default FP32 accumulator). For all other types `AccumT` is ignored.
  template <typename VectorType, typename AccumT = void>
  MSCCLPP_DEVICE_INLINE static VectorType multimemLoadReduce(VectorType* ptr) {
    VectorType val;
    if constexpr (std::is_same_v<VectorType, i32x1>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.s32 %0, [%1];" : "=r"(val.words[0]) : "l"(ptr) : "memory");
    } else if constexpr (std::is_same_v<VectorType, u32x1>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.u32 %0, [%1];" : "=r"(val.words[0]) : "l"(ptr) : "memory");
    } else if constexpr (std::is_same_v<VectorType, f32x1>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];" : "=r"(val.words[0]) : "l"(ptr) : "memory");
    } else if constexpr (std::is_same_v<VectorType, f32x2>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.v2.f32 {%0,%1}, [%2];"
          : "=r"(val.words[0]), "=r"(val.words[1])
          : "l"(ptr)
          : "memory");
    } else if constexpr (std::is_same_v<VectorType, f32x4>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.v4.f32 {%0,%1,%2,%3}, [%4];"
          : "=r"(val.words[0]), "=r"(val.words[1]), "=r"(val.words[2]), "=r"(val.words[3])
          : "l"(ptr)
          : "memory");
    } else if constexpr (std::is_same_v<VectorType, f16x2>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.f16x2 %0, [%1];" : "=r"(val.words[0]) : "l"(ptr) : "memory");
    } else if constexpr (std::is_same_v<VectorType, f16x4>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.v2.f16x2 {%0,%1}, [%2];"
          : "=r"(val.words[0]), "=r"(val.words[1])
          : "l"(ptr)
          : "memory");
    } else if constexpr (std::is_same_v<VectorType, f16x8>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.v4.f16x2 {%0,%1,%2,%3}, [%4];"
          : "=r"(val.words[0]), "=r"(val.words[1]), "=r"(val.words[2]), "=r"(val.words[3])
          : "l"(ptr)
          : "memory");
    } else if constexpr (std::is_same_v<VectorType, bf16x2>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.bf16x2 %0, [%1];" : "=r"(val.words[0]) : "l"(ptr) : "memory");
    } else if constexpr (std::is_same_v<VectorType, bf16x4>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.v2.bf16x2 {%0,%1}, [%2];"
          : "=r"(val.words[0]), "=r"(val.words[1])
          : "l"(ptr)
          : "memory");
    } else if constexpr (std::is_same_v<VectorType, bf16x8>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
          : "=r"(val.words[0]), "=r"(val.words[1]), "=r"(val.words[2]), "=r"(val.words[3])
          : "l"(ptr)
          : "memory");
    }
#if (defined(__CUDA_ARCH_SPECIFIC__) || defined(__CUDA_ARCH_FAMILY_SPECIFIC__)) && (__CUDA_ARCH__ >= 1000)
    else if constexpr (std::is_same_v<VectorType, f8_e4m3x4>) {
      if constexpr (std::is_same_v<AccumT, __half>) {
        asm("multimem.ld_reduce.relaxed.sys.global.add.acc::f16.e4m3x4 %0, [%1];"
            : "=r"(val.words[0])
            : "l"(ptr)
            : "memory");
      } else {
        asm("multimem.ld_reduce.relaxed.sys.global.add.e4m3x4 %0, [%1];" : "=r"(val.words[0]) : "l"(ptr) : "memory");
      }
    } else if constexpr (std::is_same_v<VectorType, f8_e4m3x8>) {
      if constexpr (std::is_same_v<AccumT, __half>) {
        asm("multimem.ld_reduce.relaxed.sys.global.add.acc::f16.v2.e4m3x4 {%0,%1}, [%2];"
            : "=r"(val.words[0]), "=r"(val.words[1])
            : "l"(ptr)
            : "memory");
      } else {
        asm("multimem.ld_reduce.relaxed.sys.global.add.v2.e4m3x4 {%0,%1}, [%2];"
            : "=r"(val.words[0]), "=r"(val.words[1])
            : "l"(ptr)
            : "memory");
      }
    } else if constexpr (std::is_same_v<VectorType, f8_e4m3x16>) {
      if constexpr (std::is_same_v<AccumT, __half>) {
        asm("multimem.ld_reduce.relaxed.sys.global.add.acc::f16.v4.e4m3x4 {%0,%1,%2,%3}, [%4];"
            : "=r"(val.words[0]), "=r"(val.words[1]), "=r"(val.words[2]), "=r"(val.words[3])
            : "l"(ptr)
            : "memory");
      } else {
        asm("multimem.ld_reduce.relaxed.sys.global.add.v4.e4m3x4 {%0,%1,%2,%3}, [%4];"
            : "=r"(val.words[0]), "=r"(val.words[1]), "=r"(val.words[2]), "=r"(val.words[3])
            : "l"(ptr)
            : "memory");
      }
    } else if constexpr (std::is_same_v<VectorType, f8_e5m2x4>) {
      if constexpr (std::is_same_v<AccumT, __half>) {
        asm("multimem.ld_reduce.relaxed.sys.global.add.acc::f16.e5m2x4 %0, [%1];"
            : "=r"(val.words[0])
            : "l"(ptr)
            : "memory");
      } else {
        asm("multimem.ld_reduce.relaxed.sys.global.add.e5m2x4 %0, [%1];" : "=r"(val.words[0]) : "l"(ptr) : "memory");
      }
    } else if constexpr (std::is_same_v<VectorType, f8_e5m2x8>) {
      if constexpr (std::is_same_v<AccumT, __half>) {
        asm("multimem.ld_reduce.relaxed.sys.global.add.acc::f16.v2.e5m2x4 {%0,%1}, [%2];"
            : "=r"(val.words[0]), "=r"(val.words[1])
            : "l"(ptr)
            : "memory");
      } else {
        asm("multimem.ld_reduce.relaxed.sys.global.add.v2.e5m2x4 {%0,%1}, [%2];"
            : "=r"(val.words[0]), "=r"(val.words[1])
            : "l"(ptr)
            : "memory");
      }
    } else if constexpr (std::is_same_v<VectorType, f8_e5m2x16>) {
      if constexpr (std::is_same_v<AccumT, __half>) {
        asm("multimem.ld_reduce.relaxed.sys.global.add.acc::f16.v4.e5m2x4 {%0,%1,%2,%3}, [%4];"
            : "=r"(val.words[0]), "=r"(val.words[1]), "=r"(val.words[2]), "=r"(val.words[3])
            : "l"(ptr)
            : "memory");
      } else {
        asm("multimem.ld_reduce.relaxed.sys.global.add.v4.e5m2x4 {%0,%1,%2,%3}, [%4];"
            : "=r"(val.words[0]), "=r"(val.words[1]), "=r"(val.words[2]), "=r"(val.words[3])
            : "l"(ptr)
            : "memory");
      }
    }
#endif
    else {
      assert(false && "Unsupported vector type for multimemLoadReduce");
    }
    return val;
  };

  template <typename VectorType, typename T>
  MSCCLPP_DEVICE_INLINE static void multimemStore(const VectorType& val, T* ptr) {
    if constexpr (std::is_same_v<VectorType, i32x1>) {
      asm volatile("multimem.st.relaxed.sys.global.s32 [%0], %1;" ::"l"(ptr), "r"(val.words[0]) : "memory");
    } else if constexpr (std::is_same_v<VectorType, u32x1>) {
      asm volatile("multimem.st.relaxed.sys.global.u32 [%0], %1;" ::"l"(ptr), "r"(val.words[0]) : "memory");
    } else if constexpr (std::is_same_v<VectorType, f64x1>) {
      asm volatile("multimem.st.relaxed.sys.global.f64 [%0], %1;" ::"l"(ptr), "d"(val.words[0]) : "memory");
    } else if constexpr (std::is_same_v<VectorType, f32x1>) {
      asm volatile("multimem.st.relaxed.sys.global.f32 [%0], %1;" ::"l"(ptr), "r"(val.words[0]) : "memory");
    } else if constexpr (std::is_same_v<VectorType, f32x2>) {
      asm volatile("multimem.st.relaxed.sys.global.v2.f32 [%0], {%1,%2};" ::"l"(ptr), "r"(val.words[0]),
                   "r"(val.words[1])
                   : "memory");
    } else if constexpr (std::is_same_v<VectorType, f32x4>) {
      asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(val.words[0]),
                   "r"(val.words[1]), "r"(val.words[2]), "r"(val.words[3])
                   : "memory");
    } else if constexpr (std::is_same_v<VectorType, f16x2>) {
      asm volatile("multimem.st.relaxed.sys.global.f16x2 [%0], %1;" ::"l"(ptr), "r"(val.words[0]) : "memory");
    } else if constexpr (std::is_same_v<VectorType, f16x4>) {
      asm volatile("multimem.st.relaxed.sys.global.v2.f16x2 [%0], {%1,%2};" ::"l"(ptr), "r"(val.words[0]),
                   "r"(val.words[1])
                   : "memory");
    } else if constexpr (std::is_same_v<VectorType, f16x8>) {
      asm volatile("multimem.st.relaxed.sys.global.v4.f16x2 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(val.words[0]),
                   "r"(val.words[1]), "r"(val.words[2]), "r"(val.words[3])
                   : "memory");
    } else if constexpr (std::is_same_v<VectorType, bf16x2>) {
      asm volatile("multimem.st.relaxed.sys.global.bf16x2 [%0], %1;" ::"l"(ptr), "r"(val.words[0]) : "memory");
    } else if constexpr (std::is_same_v<VectorType, bf16x4>) {
      asm volatile("multimem.st.relaxed.sys.global.v2.bf16x2 [%0], {%1,%2};" ::"l"(ptr), "r"(val.words[0]),
                   "r"(val.words[1])
                   : "memory");
    } else if constexpr (std::is_same_v<VectorType, bf16x8>) {
      asm volatile("multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(val.words[0]),
                   "r"(val.words[1]), "r"(val.words[2]), "r"(val.words[3])
                   : "memory");
    }
#if (defined(__CUDA_ARCH_SPECIFIC__) || defined(__CUDA_ARCH_FAMILY_SPECIFIC__)) && (__CUDA_ARCH__ >= 1000)
    else if constexpr (std::is_same_v<VectorType, f8_e4m3x4>) {
      asm volatile("multimem.st.relaxed.sys.global.e4m3x4 [%0], %1;" ::"l"(ptr), "r"(val.words[0]) : "memory");
    } else if constexpr (std::is_same_v<VectorType, f8_e4m3x8>) {
      asm volatile("multimem.st.relaxed.sys.global.v2.e4m3x4  [%0], {%1,%2};" ::"l"(ptr), "r"(val.words[0]),
                   "r"(val.words[1])
                   : "memory");
    } else if constexpr (std::is_same_v<VectorType, f8_e4m3x16>) {
      asm volatile("multimem.st.relaxed.sys.global.v4.e4m3x4 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(val.words[0]),
                   "r"(val.words[1]), "r"(val.words[2]), "r"(val.words[3])
                   : "memory");
    } else if constexpr (std::is_same_v<VectorType, f8_e5m2x4>) {
      asm volatile("multimem.st.relaxed.sys.global.e5m2x4 [%0], %1;" ::"l"(ptr), "r"(val.words[0]) : "memory");
    } else if constexpr (std::is_same_v<VectorType, f8_e5m2x8>) {
      asm volatile("multimem.st.relaxed.sys.global.v2.e5m2x4  [%0], {%1,%2};" ::"l"(ptr), "r"(val.words[0]),
                   "r"(val.words[1])
                   : "memory");
    } else if constexpr (std::is_same_v<VectorType, f8_e5m2x16>) {
      asm volatile("multimem.st.relaxed.sys.global.v4.e5m2x4 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(val.words[0]),
                   "r"(val.words[1]), "r"(val.words[2]), "r"(val.words[3])
                   : "memory");
    }
#endif
    else {
      assert(false && "Unsupported vector type for multimemStore");
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
      assert(false && "Unsupported vector type for multimemStoreReduce");
    }
  };
#endif  // defined(MSCCLPP_DEVICE_CUDA)
};

}  // namespace mscclpp

#endif  // MSCCLPP_SWITCH_CHANNEL_DEVICE_HPP_
