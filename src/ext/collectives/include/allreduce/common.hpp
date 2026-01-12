// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_ALLREDUCE_COMMOM_HPP_
#define MSCCLPP_ALLREDUCE_COMMOM_HPP_

#include <cmath>
#include <mscclpp/algorithm.hpp>
#include <mscclpp/gpu_data_types.hpp>
#include <mscclpp/packet_device.hpp>
#include <type_traits>

#if defined(ENABLE_NPKIT)
#include <mscclpp/npkit/npkit.hpp>
#endif

namespace mscclpp {

namespace collective {
constexpr ReduceOp SUM = ReduceOp::SUM;
constexpr ReduceOp MIN = ReduceOp::MIN;

#if defined(MSCCLPP_DEVICE_COMPILE)

template <typename To, typename From>
__forceinline__ __device__ To bit_cast(const From& src) {
  static_assert(sizeof(To) == sizeof(From), "Size mismatch for bit_cast");

  union {
    From f;
    To t;
  } u;
  u.f = src;
  return u.t;
}

template <typename T>
__forceinline__ __device__ T clip(T val) {
  return val;
}

template <>
__forceinline__ __device__ __half clip(__half val) {
  val = __hmax(val, bit_cast<__half, unsigned short>(0xfbff));
  val = __hmin(val, bit_cast<__half, unsigned short>(0x7bff));

  return val;
}

template <>
__forceinline__ __device__ __half2 clip(__half2 val) {
  val.x = __hmax(val.x, bit_cast<__half, unsigned short>(0xfbff));
  val.x = __hmin(val.x, bit_cast<__half, unsigned short>(0x7bff));
  val.y = __hmax(val.y, bit_cast<__half, unsigned short>(0xfbff));
  val.y = __hmin(val.y, bit_cast<__half, unsigned short>(0x7bff));
  return val;
}

template <>
__forceinline__ __device__ __bfloat16 clip(__bfloat16 val) {
  val = __hmax(val, bit_cast<__bfloat16, unsigned short>(0xff80));
  val = __hmin(val, bit_cast<__bfloat16, unsigned short>(0x7f80));
  return val;
}

template <>
__forceinline__ __device__ __bfloat162 clip(__bfloat162 val) {
  val.x = __hmax(val.x, bit_cast<__bfloat16, unsigned short>(0xff80));
  val.x = __hmin(val.x, bit_cast<__bfloat16, unsigned short>(0x7f80));
  val.y = __hmax(val.y, bit_cast<__bfloat16, unsigned short>(0xff80));
  val.y = __hmin(val.y, bit_cast<__bfloat16, unsigned short>(0x7f80));
  return val;
}

template <typename T, bool UseClip = true>
__forceinline__ __device__ T add_elements(T a, T b) {
  if constexpr (UseClip) {
    return clip(a + b);
  } else {
    return a + b;
  }
}

template <bool UseClip = true>
__forceinline__ __device__ __half2 add_elements(__half2 a, __half2 b) {
  if constexpr (UseClip) {
    return clip(__hadd2(a, b));
  } else {
    return __hadd2(a, b);
  }
}

template <bool UseClip = true>
__forceinline__ __device__ __bfloat162 add_elements(__bfloat162 a, __bfloat162 b) {
  if constexpr (UseClip) {
    return clip(__hadd2(a, b));
  } else {
    return __hadd2(a, b);
  }
}

template <typename T>
__forceinline__ __device__ T min_elements(T a, T b) {
  return (a < b ? a : b);
}

template <>
__forceinline__ __device__ __half2 min_elements(__half2 a, __half2 b) {
#if defined(__HIP_PLATFORM_AMD__)
  __half2 val;
  val.x = __hmin(a.x, b.x);
  val.y = __hmin(a.y, b.y);
  return val;
#else
  return __hmin2(a, b);
#endif
}

template <>
__forceinline__ __device__ __bfloat162 min_elements(__bfloat162 a, __bfloat162 b) {
  return __hmin2(a, b);
}

#if defined(__FP8_TYPES_EXIST__)
// FP8 E4M3 clipping function
template <>
__forceinline__ __device__ __fp8_e4m3 clip(__fp8_e4m3 val) {
  // FP8 E4M3 has range [-448, 448], no infinities
  // Built-in saturation in FP8 arithmetic
  return val;
}

// FP8 E5M2 clipping function - prevent infinities by clamping to max finite value
template <>
__forceinline__ __device__ __fp8_e5m2 clip(__fp8_e5m2 val) {
  // FP8 E5M2 has infinities - clamp to max finite value to prevent overflow
  // Max finite value for E5M2 is 57344.0f (0x7B), min is -57344.0f (0xFB)
  float fval = float(val);
  fval = fmaxf(fval, -57344.0f);
  fval = fminf(fval, 57344.0f);
  return __fp8_e5m2(fval);
}

// FP8 E4M3 addition using __hadd for efficiency (single element)
template <bool UseClip = true>
__forceinline__ __device__ __fp8_e4m3 add_elements(__fp8_e4m3 a, __fp8_e4m3 b) {
#if defined(__HIP_PLATFORM_AMD__) && defined(__gfx942__)
  // Optimized assembly for gfx942
  float2 v;
  uint32_t ival = 0;
  asm volatile("v_pk_add_f32 %0, %1, %2"
               : "=v"(v)
               : "v"(__builtin_amdgcn_cvt_pk_f32_fp8(a.__x, 0)), "v"(__builtin_amdgcn_cvt_pk_f32_fp8(b.__x, 0)));
  return __builtin_amdgcn_cvt_pk_fp8_f32(v.x, v.x, ival, false);
#elif !defined(__HIP_PLATFORM_AMD__)
  // NVIDIA CUDA FP8 addition (CUDA 11.8+)
  __fp8_e4m3 result = __fp8_e4m3(__hadd(__half(a), __half(b)));
  return UseClip ? clip(result) : result;
#else
  // Fallback for non-gfx942 HIP platforms
  __fp8_e4m3 result = __fp8_e4m3(float(a) + float(b));
  return UseClip ? clip(result) : result;
#endif
}

// FP8 E4M3 vectorized addition for 2 elements
template <bool UseClip = true>
__forceinline__ __device__ __fp8x2_e4m3 add_elements(__fp8x2_e4m3 a, __fp8x2_e4m3 b) {
#if defined(__HIP_PLATFORM_AMD__) && defined(__gfx942__)
  float2 v;
  uint32_t ival = 0;
  asm volatile("v_pk_add_f32 %0, %1, %2"
               : "=v"(v)
               : "v"(__builtin_amdgcn_cvt_pk_f32_fp8(a, 0)), "v"(__builtin_amdgcn_cvt_pk_f32_fp8(b, 0)));
  return __builtin_amdgcn_cvt_pk_fp8_f32(v.x, v.y, ival, false);
#elif !defined(__HIP_PLATFORM_AMD__)
  // CUDA: Convert to half2, add using optimized __hadd2, convert back
  __fp8x2_e4m3 result = __fp8x2_e4m3(__hadd2(__half2(a), __half2(b)));
  return result;
#else
  // Fallback for non-gfx942 HIP: element-wise using single-element operations
  union {
    __fp8_e4m3 fp8[2];
    __fp8x2_e4m3 fp8x2;
  } ua, ub, result;
  ua.fp8x2 = a;
  ub.fp8x2 = b;
  result.fp8[0] = add_elements<UseClip>(ua.fp8[0], ub.fp8[0]);
  result.fp8[1] = add_elements<UseClip>(ua.fp8[1], ub.fp8[1]);
  return result.fp8x2;
#endif
}

// FP8 E4M3 vectorized addition for 4 elements (via 2x __fp8x2_e4m3)
template <bool UseClip = true>
__forceinline__ __device__ __fp8x4_e4m3 add_elements(__fp8x4_e4m3 a, __fp8x4_e4m3 b) {
  // Process as two __fp8x2_e4m3 using add_elements for 2 elements
  __fp8x2_e4m3* a_pair = reinterpret_cast<__fp8x2_e4m3*>(&a);
  __fp8x2_e4m3* b_pair = reinterpret_cast<__fp8x2_e4m3*>(&b);

  __fp8x2_e4m3 result[2];
  result[0] = add_elements<UseClip>(a_pair[0], b_pair[0]);
  result[1] = add_elements<UseClip>(a_pair[1], b_pair[1]);

  return *reinterpret_cast<__fp8x4_e4m3*>(result);
}

// FP8 E5M2 addition using __hadd for efficiency (single element)
template <bool UseClip = true>
__forceinline__ __device__ __fp8_e5m2 add_elements(__fp8_e5m2 a, __fp8_e5m2 b) {
#if defined(__HIP_PLATFORM_AMD__) && defined(__gfx942__)
  // Optimized assembly for gfx942 (bfloat8)
  float2 v;
  uint32_t ival = 0;
  asm volatile("v_pk_add_f32 %0, %1, %2"
               : "=v"(v)
               : "v"(__builtin_amdgcn_cvt_pk_f32_bf8(a.__x, 0)), "v"(__builtin_amdgcn_cvt_pk_f32_bf8(b.__x, 0)));
  return __builtin_amdgcn_cvt_pk_bf8_f32(v.x, v.x, ival, false);
#elif !defined(__HIP_PLATFORM_AMD__)
  // NVIDIA CUDA FP8 addition
  __fp8_e5m2 result = __fp8_e5m2(__hadd(__half(a), __half(b)));
  return UseClip ? clip(result) : result;
#else
  // Fallback for non-gfx942 HIP platforms
  __fp8_e5m2 result = __fp8_e5m2(float(a) + float(b));
  return UseClip ? clip(result) : result;
#endif
}

#if !defined(__HIP_PLATFORM_AMD__)
// FP8 E5M2 vectorized addition for 2 elements (CUDA only)
template <bool UseClip = true>
__forceinline__ __device__ __fp8x2_e5m2 add_elements(__fp8x2_e5m2 a, __fp8x2_e5m2 b) {
  // CUDA: Convert to half2, add using optimized __hadd2, convert back
  __fp8x2_e5m2 result = __fp8x2_e5m2(__hadd2(__half2(a), __half2(b)));
  return result;
}

// FP8 E5M2 vectorized addition for 4 elements (CUDA only - via 2x __fp8x2_e5m2)
template <bool UseClip = true>
__forceinline__ __device__ __fp8x4_e5m2 add_elements(__fp8x4_e5m2 a, __fp8x4_e5m2 b) {
  // Process as two __fp8x2_e5m2 using add_elements for 2 elements
  __fp8x2_e5m2* a_pair = reinterpret_cast<__fp8x2_e5m2*>(&a);
  __fp8x2_e5m2* b_pair = reinterpret_cast<__fp8x2_e5m2*>(&b);

  __fp8x2_e5m2 result[2];
  result[0] = add_elements<UseClip>(a_pair[0], b_pair[0]);
  result[1] = add_elements<UseClip>(a_pair[1], b_pair[1]);

  return *reinterpret_cast<__fp8x4_e5m2*>(result);
}
#endif  // !defined(__HIP_PLATFORM_AMD__)

// FP8 E4M3 min operation (single element)
template <>
__forceinline__ __device__ __fp8_e4m3 min_elements(__fp8_e4m3 a, __fp8_e4m3 b) {
#if defined(__HIP_PLATFORM_AMD__)
  return __fp8_e4m3(fminf(float(a), float(b)));
#else
  return __fp8_e4m3(__hmin(__half(a), __half(b)));
#endif
}

// FP8 E4M3 vectorized min for 2 elements
__forceinline__ __device__ __fp8x2_e4m3 min_elements(__fp8x2_e4m3 a, __fp8x2_e4m3 b) {
#if defined(__HIP_PLATFORM_AMD__)
  // HIP implementation: use union and process element-wise
  union {
    __fp8_e4m3 fp8[2];
    __fp8x2_e4m3 fp8x2;
  } ua, ub, result;
  ua.fp8x2 = a;
  ub.fp8x2 = b;
  result.fp8[0] = min_elements(ua.fp8[0], ub.fp8[0]);
  result.fp8[1] = min_elements(ua.fp8[1], ub.fp8[1]);
  return result.fp8x2;
#else
  return __fp8x2_e4m3(__hmin2(__half2(a), __half2(b)));
#endif
}

// FP8 E4M3 vectorized min for 4 elements
__forceinline__ __device__ __fp8x4_e4m3 min_elements(__fp8x4_e4m3 a, __fp8x4_e4m3 b) {
  // Process as two __fp8x2_e4m3 using min_elements for 2 elements
  union {
    __fp8x4_e4m3 vec4;
    __fp8x2_e4m3 vec2[2];
  } ua, ub, uresult;
  ua.vec4 = a;
  ub.vec4 = b;

  uresult.vec2[0] = min_elements(ua.vec2[0], ub.vec2[0]);
  uresult.vec2[1] = min_elements(ua.vec2[1], ub.vec2[1]);

  return uresult.vec4;
}

// FP8 E5M2 min operation (single element)
template <>
__forceinline__ __device__ __fp8_e5m2 min_elements(__fp8_e5m2 a, __fp8_e5m2 b) {
#if defined(__HIP_PLATFORM_AMD__)
  return __fp8_e5m2(fminf(float(a), float(b)));
#else
  return __fp8_e5m2(__hmin(__half(a), __half(b)));
#endif
}

#if !defined(__HIP_PLATFORM_AMD__)
// FP8 E5M2 vectorized min for 2 elements (CUDA only)
__forceinline__ __device__ __fp8x2_e5m2 min_elements(__fp8x2_e5m2 a, __fp8x2_e5m2 b) {
  return __fp8x2_e5m2(__hmin2(__half2(a), __half2(b)));
}

// FP8 E5M2 vectorized min for 4 elements (CUDA only)
__forceinline__ __device__ __fp8x4_e5m2 min_elements(__fp8x4_e5m2 a, __fp8x4_e5m2 b) {
  // Process as two __fp8x2_e5m2 using min_elements for 2 elements
  union {
    __fp8x4_e5m2 vec4;
    __fp8x2_e5m2 vec2[2];
  } ua, ub, uresult;
  ua.vec4 = a;
  ub.vec4 = b;

  uresult.vec2[0] = min_elements(ua.vec2[0], ub.vec2[0]);
  uresult.vec2[1] = min_elements(ua.vec2[1], ub.vec2[1]);

  return uresult.vec4;
}
#endif  // !defined(__HIP_PLATFORM_AMD__)
#endif  // __FP8_TYPES_EXIST__

template <typename T, ReduceOp OpType>
__forceinline__ __device__ T cal_elements(T a, T b) {
  if constexpr (OpType == SUM) {
    return add_elements(a, b);
  } else if constexpr (OpType == MIN) {
    return min_elements(a, b);
  }
  // Should never reach here
  return a;
}

template <typename T, ReduceOp OpType>
__forceinline__ __device__ int4 cal_vectors_helper(int4 a, int4 b) {
  int4 ret;
  ret.w = bit_cast<int, T>(cal_elements<T, OpType>(bit_cast<T, int>(a.w), bit_cast<T, int>(b.w)));
  ret.x = bit_cast<int, T>(cal_elements<T, OpType>(bit_cast<T, int>(a.x), bit_cast<T, int>(b.x)));
  ret.y = bit_cast<int, T>(cal_elements<T, OpType>(bit_cast<T, int>(a.y), bit_cast<T, int>(b.y)));
  ret.z = bit_cast<int, T>(cal_elements<T, OpType>(bit_cast<T, int>(a.z), bit_cast<T, int>(b.z)));
  return ret;
}

template <typename T, ReduceOp OpType>
__forceinline__ __device__ uint2 cal_vectors_helper(uint2 a, uint2 b) {
  uint2 ret;
  ret.x = bit_cast<int, T>(cal_elements<T, OpType>(bit_cast<T, int>(a.x), bit_cast<T, int>(b.x)));
  ret.y = bit_cast<int, T>(cal_elements<T, OpType>(bit_cast<T, int>(a.y), bit_cast<T, int>(b.y)));
  return ret;
}

template <typename T, ReduceOp OpType>
__forceinline__ __device__ int cal_vectors_helper(int a, int b) {
  return bit_cast<int, T>(cal_elements<T, OpType>(bit_cast<T, int>(a), bit_cast<T, int>(b)));
}

#if defined(__HIP_PLATFORM_AMD__) && defined(__FP8_TYPES_EXIST__) && defined(__gfx942__)
// Helper function to perform FP8 vector addition - dispatches based on scalar type
// Uses AMD builtins from hip/amd_detail/amd_hip_fp8.h:
//   - __builtin_amdgcn_cvt_pk_f32_fp8/bf8: Convert 2 FP8 values to 2 floats
//   - __builtin_amdgcn_cvt_pk_fp8/bf8_f32: Convert 2 floats to 2 FP8 values
// The 'word' parameter (false/true) selects low/high 16-bit word from uint32_t
template <typename ScalarT>
__forceinline__ __device__ int add_fp8x4_hip(int a, int b) {
  uint32_t a32 = static_cast<uint32_t>(a);
  uint32_t b32 = static_cast<uint32_t>(b);

  float2 v_low, v_high;
  uint32_t ival = 0;

  if constexpr (std::is_same_v<ScalarT, __fp8_e4m3>) {
    // E4M3 using fp8 conversion - process low word (false) and high word (true)
    asm volatile("v_pk_add_f32 %0, %1, %2"
                 : "=v"(v_low)
                 : "v"(__builtin_amdgcn_cvt_pk_f32_fp8(a32, false)), "v"(__builtin_amdgcn_cvt_pk_f32_fp8(b32, false)));
    uint16_t result_low = __builtin_amdgcn_cvt_pk_fp8_f32(v_low.x, v_low.y, ival, false);

    asm volatile("v_pk_add_f32 %0, %1, %2"
                 : "=v"(v_high)
                 : "v"(__builtin_amdgcn_cvt_pk_f32_fp8(a32, true)), "v"(__builtin_amdgcn_cvt_pk_f32_fp8(b32, true)));
    uint16_t result_high = __builtin_amdgcn_cvt_pk_fp8_f32(v_high.x, v_high.y, ival, false);

    uint32_t result = (static_cast<uint32_t>(result_high) << 16) | result_low;
    return static_cast<int>(result);
  } else {  // __fp8_e5m2
    // E5M2 using bf8 conversion - process low word (false) and high word (true)
    asm volatile("v_pk_add_f32 %0, %1, %2"
                 : "=v"(v_low)
                 : "v"(__builtin_amdgcn_cvt_pk_f32_bf8(a32, false)), "v"(__builtin_amdgcn_cvt_pk_f32_bf8(b32, false)));
    uint16_t result_low = __builtin_amdgcn_cvt_pk_bf8_f32(v_low.x, v_low.y, ival, false);

    asm volatile("v_pk_add_f32 %0, %1, %2"
                 : "=v"(v_high)
                 : "v"(__builtin_amdgcn_cvt_pk_f32_bf8(a32, true)), "v"(__builtin_amdgcn_cvt_pk_f32_bf8(b32, true)));
    uint16_t result_high = __builtin_amdgcn_cvt_pk_bf8_f32(v_high.x, v_high.y, ival, false);

    uint32_t result = (static_cast<uint32_t>(result_high) << 16) | result_low;
    return static_cast<int>(result);
  }
}
#endif

template <typename T, ReduceOp OpType, typename DataType>
__forceinline__ __device__ DataType cal_vectors(DataType a, DataType b) {
#if defined(__HIP_PLATFORM_AMD__) && defined(__FP8_TYPES_EXIST__) && defined(__gfx942__)
  // For FP8 types on HIP gfx942, use specialized helper that dispatches based on scalar type
  if constexpr (std::is_same_v<T, __fp8_e4m3> || std::is_same_v<T, __fp8_e5m2>) {
    if constexpr (OpType == SUM) {
      if constexpr (std::is_same_v<DataType, int> || std::is_same_v<DataType, uint32_t>) {
        // Handle int/uint32_t (4 FP8 elements)
        return add_fp8x4_hip<T>(a, b);
      } else if constexpr (std::is_same_v<DataType, int4>) {
        // Handle int4 (16 FP8 elements) - process as 4 ints
        int4 ret;
        ret.w = add_fp8x4_hip<T>(a.w, b.w);
        ret.x = add_fp8x4_hip<T>(a.x, b.x);
        ret.y = add_fp8x4_hip<T>(a.y, b.y);
        ret.z = add_fp8x4_hip<T>(a.z, b.z);
        return ret;
      } else if constexpr (std::is_same_v<DataType, uint2>) {
        // Handle uint2 (8 FP8 elements) - process as 2 ints
        uint2 ret;
        ret.x = add_fp8x4_hip<T>(a.x, b.x);
        ret.y = add_fp8x4_hip<T>(a.y, b.y);
        return ret;
      }
    }
  }
#endif

  // Define the vectorized computation type based on the element type
  using CompType = typename std::conditional_t<
      std::is_same_v<T, __half>, __half2,
      std::conditional_t<std::is_same_v<T, __bfloat16>, __bfloat162,
#if defined(__FP8_TYPES_EXIST__)
                         std::conditional_t<std::is_same_v<T, __fp8_e4m3>, __fp8x4_e4m3,
                                            std::conditional_t<std::is_same_v<T, __fp8_e5m2>, __fp8x4_e5m2,
#endif
                                                               T
#if defined(__FP8_TYPES_EXIST__)
                                                               >>>>;
#else
                         >>;
#endif
  return cal_vectors_helper<CompType, OpType>(a, b);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
template <class T>
MSCCLPP_DEVICE_INLINE constexpr std::size_t calcVectorSize() {
  using U = std::remove_cv_t<std::remove_reference_t<T>>;
  if constexpr (std::is_same_v<U, std::int32_t> || std::is_same_v<U, std::uint32_t>) {
    return 1;
  } else {
    static_assert(16 % sizeof(U) == 0, "16 bytes must be divisible by sizeof(T).");
    return 16 / sizeof(U);
  }
}

template <typename T>
MSCCLPP_DEVICE_INLINE void handleMultiLoadReduceStore(T* src, T* dst, size_t srcOffset, size_t dstOffset, size_t size,
                                                      int tid, int nThreads) {
  // nvls can only handle 4 bytes alignment
  MSCCLPP_ASSERT_DEVICE(size % 4 == 0, "size must be 4 bytes aligned");
  constexpr size_t nElem = calcVectorSize<T>();
  using vectorType = mscclpp::VectorType<T, nElem>;
  const size_t nVec = size / sizeof(vectorType);
  const size_t srcOffset4 = srcOffset / sizeof(vectorType);
  const size_t dstOffset4 = dstOffset / sizeof(vectorType);
  vectorType* src4 = (vectorType*)src;
  vectorType* dst4 = (vectorType*)dst;
  for (size_t idx = tid; idx < nVec; idx += nThreads) {
    auto val = mscclpp::SwitchChannelDeviceHandle::multimemLoadReduce(src4 + srcOffset4 + idx);
    mscclpp::SwitchChannelDeviceHandle::multimemStore(val, dst4 + dstOffset4 + idx);
  }
  // handle rest of data
  size_t processed = nVec * sizeof(vectorType);
  constexpr size_t nRestElem = 4 / sizeof(T);
  using restVectorType = mscclpp::VectorType<T, nRestElem>;
  const size_t startIdx = (srcOffset + processed) / sizeof(restVectorType);
  const size_t endIdx = (srcOffset + size) / sizeof(restVectorType);
  for (size_t idx = tid + startIdx; idx < endIdx; idx += nThreads) {
    auto val = mscclpp::SwitchChannelDeviceHandle::multimemLoadReduce((restVectorType*)src + idx);
    mscclpp::SwitchChannelDeviceHandle::multimemStore(val, (restVectorType*)dst + idx);
  }
}
#endif  // defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

#endif  // defined(MSCCLPP_DEVICE_COMPILE)

using AllreduceFunc =
    std::function<cudaError_t(const void*, void*, void*, void*, void*, mscclpp::DeviceHandle<mscclpp::SwitchChannel>*,
                              mscclpp::DeviceHandle<mscclpp::SwitchChannel>*, size_t, size_t, size_t, int, int, int,
                              size_t, cudaStream_t, void*, uint32_t, int, int)>;

template <template <ReduceOp, typename> class Adapter>
AllreduceFunc dispatch(ReduceOp op, mscclpp::DataType dtype) {
  if (op == SUM) {
    if (dtype == mscclpp::DataType::FLOAT16) {
      return Adapter<SUM, half>::call;
    } else if (dtype == mscclpp::DataType::FLOAT32) {
      return Adapter<SUM, float>::call;
#if defined(__CUDA_BF16_TYPES_EXIST__)
    } else if (dtype == mscclpp::DataType::BFLOAT16) {
      return Adapter<SUM, __bfloat16>::call;
#endif
#if defined(__FP8_TYPES_EXIST__)
    } else if (dtype == mscclpp::DataType::FP8_E4M3) {
      return Adapter<SUM, __fp8_e4m3>::call;
    } else if (dtype == mscclpp::DataType::FP8_E5M2) {
      return Adapter<SUM, __fp8_e5m2>::call;
#endif
    } else if (dtype == mscclpp::DataType::INT32 || dtype == mscclpp::DataType::UINT32) {
      return Adapter<SUM, int>::call;
    } else {
      return nullptr;
    }
  } else if (op == MIN) {
    if (dtype == mscclpp::DataType::FLOAT16) {
      return Adapter<MIN, half>::call;
    } else if (dtype == mscclpp::DataType::FLOAT32) {
      return Adapter<MIN, float>::call;
#if defined(__CUDA_BF16_TYPES_EXIST__)
    } else if (dtype == mscclpp::DataType::BFLOAT16) {
      return Adapter<MIN, __bfloat16>::call;
#endif
#if defined(__FP8_TYPES_EXIST__)
    } else if (dtype == mscclpp::DataType::FP8_E4M3) {
      return Adapter<MIN, __fp8_e4m3>::call;
    } else if (dtype == mscclpp::DataType::FP8_E5M2) {
      return Adapter<MIN, __fp8_e5m2>::call;
#endif
    } else if (dtype == mscclpp::DataType::INT32 || dtype == mscclpp::DataType::UINT32) {
      return Adapter<MIN, int>::call;
    } else {
      return nullptr;
    }
  }
  return nullptr;
}
}  // namespace collective
}  // namespace mscclpp

#endif  // MSCCLPP_ALLREDUCE_COMMON_HPP_