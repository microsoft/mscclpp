// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_REDUCE_KERNEL_HPP_
#define MSCCLPP_REDUCE_KERNEL_HPP_

#include <mscclpp/algorithm.hpp>
#include <mscclpp/gpu_data_types.hpp>
#include <type_traits>

namespace mscclpp {

#if defined(MSCCLPP_DEVICE_COMPILE)

template <typename To, typename From>
MSCCLPP_DEVICE_INLINE To bit_cast(const From& src) {
  static_assert(sizeof(To) == sizeof(From), "Size mismatch for bit_cast");

  union {
    From f;
    To t;
  } u;
  u.f = src;
  return u.t;
}

template <typename T>
MSCCLPP_DEVICE_INLINE T clip(T val) {
  return val;
}

template <>
MSCCLPP_DEVICE_INLINE __half clip(__half val) {
  val = __hmax(val, bit_cast<__half, unsigned short>(0xfbff));
  val = __hmin(val, bit_cast<__half, unsigned short>(0x7bff));

  return val;
}

template <>
MSCCLPP_DEVICE_INLINE __half2 clip(__half2 val) {
  val.x = __hmax(val.x, bit_cast<__half, unsigned short>(0xfbff));
  val.x = __hmin(val.x, bit_cast<__half, unsigned short>(0x7bff));
  val.y = __hmax(val.y, bit_cast<__half, unsigned short>(0xfbff));
  val.y = __hmin(val.y, bit_cast<__half, unsigned short>(0x7bff));
  return val;
}

template <>
MSCCLPP_DEVICE_INLINE __bfloat16 clip(__bfloat16 val) {
  val = __hmax(val, bit_cast<__bfloat16, unsigned short>(0xff80));
  val = __hmin(val, bit_cast<__bfloat16, unsigned short>(0x7f80));
  return val;
}

template <>
MSCCLPP_DEVICE_INLINE __bfloat162 clip(__bfloat162 val) {
  val.x = __hmax(val.x, bit_cast<__bfloat16, unsigned short>(0xff80));
  val.x = __hmin(val.x, bit_cast<__bfloat16, unsigned short>(0x7f80));
  val.y = __hmax(val.y, bit_cast<__bfloat16, unsigned short>(0xff80));
  val.y = __hmin(val.y, bit_cast<__bfloat16, unsigned short>(0x7f80));
  return val;
}

// FP8 E4M3 clipping function
#if defined(__FP8_TYPES_EXIST__)
template <>
MSCCLPP_DEVICE_INLINE __fp8_e4m3 clip(__fp8_e4m3 val) {
  // FP8 E4M3 has range [-448, 448], no infinities
  // Built-in saturation in FP8 arithmetic
  return val;
}

// FP8 E5M2 clipping function - prevent infinities by clamping to max finite value
template <>
MSCCLPP_DEVICE_INLINE __fp8_e5m2 clip(__fp8_e5m2 val) {
  // FP8 E5M2 has infinities - clamp to max finite value to prevent overflow
  // Max finite value for E5M2 is 57344.0f (0x7B), min is -57344.0f (0xFB)
  float fval = float(val);
  fval = fmaxf(fval, -57344.0f);
  fval = fminf(fval, 57344.0f);
  return __fp8_e5m2(fval);
}
#endif

template <typename T, bool UseClip = true>
MSCCLPP_DEVICE_INLINE T add_elements(T a, T b) {
  if constexpr (UseClip) {
    return clip(a + b);
  } else {
    return a + b;
  }
}

template <bool UseClip = true>
MSCCLPP_DEVICE_INLINE __half2 add_elements(__half2 a, __half2 b) {
  if constexpr (UseClip) {
    return clip(__hadd2(a, b));
  } else {
    return __hadd2(a, b);
  }
}

template <bool UseClip = true>
MSCCLPP_DEVICE_INLINE __bfloat162 add_elements(__bfloat162 a, __bfloat162 b) {
  if constexpr (UseClip) {
    return clip(__hadd2(a, b));
  } else {
    return __hadd2(a, b);
  }
}

#if defined(__FP8_TYPES_EXIST__)
template <bool UseClip = true>
MSCCLPP_DEVICE_INLINE __fp8_e4m3 add_elements(__fp8_e4m3 a, __fp8_e4m3 b) {
#if defined(MSCCLPP_DEVICE_HIP) && defined(__gfx942__)
  // Optimized assembly for gfx942
  float2 v;
  uint32_t ival = 0;
  asm volatile("v_pk_add_f32 %0, %1, %2"
               : "=v"(v)
               : "v"(__builtin_amdgcn_cvt_pk_f32_fp8(a.__x, 0)), "v"(__builtin_amdgcn_cvt_pk_f32_fp8(b.__x, 0)));
  return __builtin_amdgcn_cvt_pk_fp8_f32(v.x, v.x, ival, false);
#elif defined(MSCCLPP_DEVICE_CUDA)
  // NVIDIA CUDA FP8 addition (CUDA 11.8+)
  __fp8_e4m3 result = __fp8_e4m3(__hadd(__half(a), __half(b)));
  return UseClip ? clip(result) : result;
#else
  // Fallback for other devices
  __fp8_e4m3 result = __fp8_e4m3(float(a) + float(b));
  return UseClip ? clip(result) : result;
#endif
}

template <bool UseClip = true>
MSCCLPP_DEVICE_INLINE __fp8x2_e4m3 add_elements(__fp8x2_e4m3 a, __fp8x2_e4m3 b) {
#if defined(MSCCLPP_DEVICE_HIP) && defined(__gfx942__)
  float2 v;
  uint32_t ival = 0;
  asm volatile("v_pk_add_f32 %0, %1, %2"
               : "=v"(v)
               : "v"(__builtin_amdgcn_cvt_pk_f32_fp8(a, 0)), "v"(__builtin_amdgcn_cvt_pk_f32_fp8(b, 0)));
  return __fp8x2_e4m3(__builtin_amdgcn_cvt_pk_fp8_f32(v.x, v.y, ival, false));
#elif defined(MSCCLPP_DEVICE_CUDA)
  // CUDA: Convert to half2, add using optimized __hadd2, convert back
  __fp8x2_e4m3 result = __fp8x2_e4m3(__hadd2(__half2(a), __half2(b)));
  return result;
#else
  // Fallback for other devices: element-wise using single-element operations
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

template <bool UseClip = true>
MSCCLPP_DEVICE_INLINE __fp8x4_e4m3 add_elements(__fp8x4_e4m3 a, __fp8x4_e4m3 b) {
#if defined(MSCCLPP_DEVICE_HIP) && defined(__gfx942__)
  uint32_t a32 = *reinterpret_cast<uint32_t*>(&a);
  uint32_t b32 = *reinterpret_cast<uint32_t*>(&b);
  float2 v_low, v_high;
  // E4M3 using fp8 conversion - process low word (false) and high word (true)
  asm volatile("v_pk_add_f32 %0, %1, %2"
               : "=v"(v_low)
               : "v"(__builtin_amdgcn_cvt_pk_f32_fp8(a32, false)), "v"(__builtin_amdgcn_cvt_pk_f32_fp8(b32, false)));
  uint32_t result_packed = __builtin_amdgcn_cvt_pk_fp8_f32(v_low.x, v_low.y, 0, false);

  asm volatile("v_pk_add_f32 %0, %1, %2"
               : "=v"(v_high)
               : "v"(__builtin_amdgcn_cvt_pk_f32_fp8(a32, true)), "v"(__builtin_amdgcn_cvt_pk_f32_fp8(b32, true)));
  result_packed = __builtin_amdgcn_cvt_pk_fp8_f32(v_high.x, v_high.y, result_packed, true);
  return *reinterpret_cast<__fp8x4_e4m3*>(&result_packed);
#else
  // Process as two __fp8x2_e4m3 using add_elements for 2 elements
  __fp8x2_e4m3* a_pair = reinterpret_cast<__fp8x2_e4m3*>(&a);
  __fp8x2_e4m3* b_pair = reinterpret_cast<__fp8x2_e4m3*>(&b);

  __fp8x2_e4m3 result[2];
  result[0] = add_elements<UseClip>(a_pair[0], b_pair[0]);
  result[1] = add_elements<UseClip>(a_pair[1], b_pair[1]);

  return *reinterpret_cast<__fp8x4_e4m3*>(result);
#endif
}

template <bool UseClip = true>
MSCCLPP_DEVICE_INLINE __fp8_e5m2 add_elements(__fp8_e5m2 a, __fp8_e5m2 b) {
#if defined(MSCCLPP_DEVICE_HIP) && defined(__gfx942__)
  // Optimized assembly for gfx942 (bfloat8)
  float2 v;
  uint32_t ival = 0;
  asm volatile("v_pk_add_f32 %0, %1, %2"
               : "=v"(v)
               : "v"(__builtin_amdgcn_cvt_pk_f32_bf8(a.__x, 0)), "v"(__builtin_amdgcn_cvt_pk_f32_bf8(b.__x, 0)));
  return __builtin_amdgcn_cvt_pk_bf8_f32(v.x, v.x, ival, false);
#elif defined(MSCCLPP_DEVICE_CUDA)
  // NVIDIA CUDA FP8 addition
  __fp8_e5m2 result = __fp8_e5m2(__hadd(__half(a), __half(b)));
  return UseClip ? clip(result) : result;
#else
  __fp8_e5m2 result = __fp8_e5m2(float(a) + float(b));
  return UseClip ? clip(result) : result;
#endif
}

template <bool UseClip = true>
MSCCLPP_DEVICE_INLINE __fp8x2_e5m2 add_elements(__fp8x2_e5m2 a, __fp8x2_e5m2 b) {
#if defined(MSCCLPP_DEVICE_CUDA)
  // CUDA: Convert to half2, add using optimized __hadd2, convert back
  __fp8x2_e5m2 result = __fp8x2_e5m2(__hadd2(__half2(a), __half2(b)));
  return UseClip ? clip(result) : result;
#elif defined(MSCCLPP_DEVICE_HIP) && defined(__gfx942__)
  // HIP gfx942: Use BF8 assembly instructions
  float2 v;
  uint32_t ival = 0;
  asm volatile("v_pk_add_f32 %0, %1, %2"
               : "=v"(v)
               : "v"(__builtin_amdgcn_cvt_pk_f32_bf8(a, 0)), "v"(__builtin_amdgcn_cvt_pk_f32_bf8(b, 0)));
  return __fp8x2_e5m2(__builtin_amdgcn_cvt_pk_bf8_f32(v.x, v.y, ival, false));
#else
  // Fallback: element-wise using single-element operations
  union {
    __fp8_e5m2 fp8[2];
    __fp8x2_e5m2 fp8x2;
  } ua, ub, result;
  ua.fp8x2 = a;
  ub.fp8x2 = b;
  result.fp8[0] = add_elements<UseClip>(ua.fp8[0], ub.fp8[0]);
  result.fp8[1] = add_elements<UseClip>(ua.fp8[1], ub.fp8[1]);
  return result.fp8x2;
#endif
}

template <bool UseClip = true>
MSCCLPP_DEVICE_INLINE __fp8x4_e5m2 add_elements(__fp8x4_e5m2 a, __fp8x4_e5m2 b) {
#if defined(MSCCLPP_DEVICE_HIP) && defined(__gfx942__)
  uint32_t a32 = *reinterpret_cast<uint32_t*>(&a);
  uint32_t b32 = *reinterpret_cast<uint32_t*>(&b);
  float2 v_low, v_high;
  // E5M2 using bf8 conversion - process low word (false) and high word (true)
  asm volatile("v_pk_add_f32 %0, %1, %2"
               : "=v"(v_low)
               : "v"(__builtin_amdgcn_cvt_pk_f32_bf8(a32, false)), "v"(__builtin_amdgcn_cvt_pk_f32_bf8(b32, false)));
  uint32_t result_packed = __builtin_amdgcn_cvt_pk_bf8_f32(v_low.x, v_low.y, 0, false);

  asm volatile("v_pk_add_f32 %0, %1, %2"
               : "=v"(v_high)
               : "v"(__builtin_amdgcn_cvt_pk_f32_bf8(a32, true)), "v"(__builtin_amdgcn_cvt_pk_f32_bf8(b32, true)));
  result_packed = __builtin_amdgcn_cvt_pk_bf8_f32(v_high.x, v_high.y, result_packed, true);
  return *reinterpret_cast<__fp8x4_e5m2*>(&result_packed);
#else
  // Process as two __fp8x2_e5m2 using add_elements for 2 elements
  __fp8x2_e5m2* a_pair = reinterpret_cast<__fp8x2_e5m2*>(&a);
  __fp8x2_e5m2* b_pair = reinterpret_cast<__fp8x2_e5m2*>(&b);

  __fp8x2_e5m2 result[2];
  result[0] = add_elements<UseClip>(a_pair[0], b_pair[0]);
  result[1] = add_elements<UseClip>(a_pair[1], b_pair[1]);

  return *reinterpret_cast<__fp8x4_e5m2*>(result);
#endif
}
#endif  // defined(__FP8_TYPES_EXIST__)

template <typename T>
MSCCLPP_DEVICE_INLINE T min_elements(T a, T b) {
  return (a < b ? a : b);
}

template <>
MSCCLPP_DEVICE_INLINE __half2 min_elements(__half2 a, __half2 b) {
#if defined(MSCCLPP_DEVICE_HIP)
  __half2 val;
  val.x = __hmin(a.x, b.x);
  val.y = __hmin(a.y, b.y);
  return val;
#else
  return __hmin2(a, b);
#endif
}

template <>
MSCCLPP_DEVICE_INLINE __bfloat162 min_elements(__bfloat162 a, __bfloat162 b) {
  return __hmin2(a, b);
}

#if defined(__FP8_TYPES_EXIST__)
template <>
MSCCLPP_DEVICE_INLINE __fp8_e4m3 min_elements(__fp8_e4m3 a, __fp8_e4m3 b) {
#if defined(MSCCLPP_DEVICE_HIP)
  return __fp8_e4m3(fminf(float(a), float(b)));
#else
  return __fp8_e4m3(__hmin(__half(a), __half(b)));
#endif
}

MSCCLPP_DEVICE_INLINE __fp8x2_e4m3 min_elements(__fp8x2_e4m3 a, __fp8x2_e4m3 b) {
#if defined(MSCCLPP_DEVICE_HIP)
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

MSCCLPP_DEVICE_INLINE __fp8x4_e4m3 min_elements(__fp8x4_e4m3 a, __fp8x4_e4m3 b) {
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

template <>
MSCCLPP_DEVICE_INLINE __fp8_e5m2 min_elements(__fp8_e5m2 a, __fp8_e5m2 b) {
#if defined(MSCCLPP_DEVICE_HIP)
  return __fp8_e5m2(fminf(float(a), float(b)));
#else
  return __fp8_e5m2(__hmin(__half(a), __half(b)));
#endif
}

MSCCLPP_DEVICE_INLINE __fp8x2_e5m2 min_elements(__fp8x2_e5m2 a, __fp8x2_e5m2 b) {
#if defined(MSCCLPP_DEVICE_HIP)
  union {
    __fp8_e5m2 fp8[2];
    __fp8x2_e5m2 fp8x2;
  } ua, ub, result;
  ua.fp8x2 = a;
  ub.fp8x2 = b;
  result.fp8[0] = min_elements(ua.fp8[0], ub.fp8[0]);
  result.fp8[1] = min_elements(ua.fp8[1], ub.fp8[1]);
  return result.fp8x2;
#else
  return __fp8x2_e5m2(__hmin2(__half2(a), __half2(b)));
#endif
}

MSCCLPP_DEVICE_INLINE __fp8x4_e5m2 min_elements(__fp8x4_e5m2 a, __fp8x4_e5m2 b) {
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
#endif  // defined(__FP8_TYPES_EXIST__)

// Generic element-wise calculation helper
template <typename T, ReduceOp OpType>
MSCCLPP_DEVICE_INLINE T cal_elements(T a, T b) {
  if constexpr (OpType == SUM) {
    return add_elements(a, b);
  } else if constexpr (OpType == MIN) {
    return min_elements(a, b);
  }
  static_assert(OpType == SUM || OpType == MIN, "Unsupported ReduceOp");
}

// Generic vector reduction helpers
template <typename T, ReduceOp OpType>
MSCCLPP_DEVICE_INLINE int4 cal_vector_helper(int4 a, int4 b) {
  int4 ret;
  ret.w = bit_cast<int, T>(cal_elements<T, OpType>(bit_cast<T, int>(a.w), bit_cast<T, int>(b.w)));
  ret.x = bit_cast<int, T>(cal_elements<T, OpType>(bit_cast<T, int>(a.x), bit_cast<T, int>(b.x)));
  ret.y = bit_cast<int, T>(cal_elements<T, OpType>(bit_cast<T, int>(a.y), bit_cast<T, int>(b.y)));
  ret.z = bit_cast<int, T>(cal_elements<T, OpType>(bit_cast<T, int>(a.z), bit_cast<T, int>(b.z)));
  return ret;
}

template <typename T, ReduceOp OpType>
MSCCLPP_DEVICE_INLINE uint2 cal_vector_helper(uint2 a, uint2 b) {
  uint2 ret;
  ret.x = bit_cast<int, T>(cal_elements<T, OpType>(bit_cast<T, int>(a.x), bit_cast<T, int>(b.x)));
  ret.y = bit_cast<int, T>(cal_elements<T, OpType>(bit_cast<T, int>(a.y), bit_cast<T, int>(b.y)));
  return ret;
}

template <typename T, ReduceOp OpType>
MSCCLPP_DEVICE_INLINE int cal_vector_helper(int a, int b) {
  return bit_cast<int, T>(cal_elements<T, OpType>(bit_cast<T, int>(a), bit_cast<T, int>(b)));
}

template <typename T, ReduceOp OpType>
MSCCLPP_DEVICE_INLINE uint32_t cal_vector_helper(uint32_t a, uint32_t b) {
  return bit_cast<uint32_t, T>(cal_elements<T, OpType>(bit_cast<T, uint32_t>(a), bit_cast<T, uint32_t>(b)));
}

// cal_vector wrapper - converts scalar types to vector types and calls cal_vector_helper
template <typename T, ReduceOp OpType, typename DataType>
MSCCLPP_DEVICE_INLINE DataType cal_vector(DataType a, DataType b) {
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
  return cal_vector_helper<CompType, OpType>(a, b);
}

#endif  // defined(MSCCLPP_DEVICE_COMPILE)

}  // namespace mscclpp

#endif  // MSCCLPP_REDUCE_KERNEL_HPP_
