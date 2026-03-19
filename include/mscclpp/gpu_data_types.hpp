// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_GPU_DATA_TYPES_HPP_
#define MSCCLPP_GPU_DATA_TYPES_HPP_

#include <mscclpp/device.hpp>

#if defined(MSCCLPP_DEVICE_HIP)

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_version.h>

using __bfloat16 = __hip_bfloat16;
using __bfloat162 = __hip_bfloat162;
#define __CUDA_BF16_TYPES_EXIST__

// AMD FP8 support - Use fnuz types for HIP 6.0 or when HIP_FP8_TYPE_FNUZ is enabled and HIP_FP8_TYPE_OCP is not
// enabled. Otherwise, use the standard FP8 types.
#if defined(HIP_VERSION_MAJOR) && (HIP_VERSION_MAJOR >= 6)
#include <hip/hip_fp8.h>

// Create aliases matching CUDA naming convention for cross-platform compatibility
#if (HIP_VERSION_MAJOR == 6) || (HIP_VERSION_MAJOR > 6 && HIP_FP8_TYPE_FNUZ && !HIP_FP8_TYPE_OCP)
using __fp8_e4m3 = __hip_fp8_e4m3_fnuz;
using __fp8_e5m2 = __hip_fp8_e5m2_fnuz;
using __fp8x2_e4m3 = __hip_fp8x2_e4m3_fnuz;
using __fp8x2_e5m2 = __hip_fp8x2_e5m2_fnuz;
using __fp8x4_e4m3 = __hip_fp8x4_e4m3_fnuz;
using __fp8x4_e5m2 = __hip_fp8x4_e5m2_fnuz;
#else
using __fp8_e4m3 = __hip_fp8_e4m3;
using __fp8_e5m2 = __hip_fp8_e5m2;
using __fp8x2_e4m3 = __hip_fp8x2_e4m3;
using __fp8x2_e5m2 = __hip_fp8x2_e5m2;
using __fp8x4_e4m3 = __hip_fp8x4_e4m3;
using __fp8x4_e5m2 = __hip_fp8x4_e5m2;
#endif

#define __FP8_TYPES_EXIST__
#endif  // HIP_VERSION_MAJOR >= 6

#else  // NVIDIA

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#if (CUDART_VERSION >= 11000)
#include <cuda_bf16.h>
#endif
#if (CUDART_VERSION >= 11080)
#include <cuda_fp8.h>
using __fp8_e4m3 = __nv_fp8_e4m3;
using __fp8_e5m2 = __nv_fp8_e5m2;
using __fp8x2_e4m3 = __nv_fp8x2_e4m3;
using __fp8x2_e5m2 = __nv_fp8x2_e5m2;
using __fp8x4_e4m3 = __nv_fp8x4_e4m3;
using __fp8x4_e5m2 = __nv_fp8x4_e5m2;
#define __FP8_TYPES_EXIST__
#endif

using __bfloat16 = __nv_bfloat16;
using __bfloat162 = __nv_bfloat162;

#endif

namespace mscclpp {

/// Data types supported by mscclpp operations.
enum class DataType {
  INT32,        // 32-bit signed integer.
  UINT32,       // 32-bit unsigned integer.
  FLOAT16,      // IEEE 754 half precision.
  FLOAT32,      // IEEE 754 single precision.
  BFLOAT16,     // bfloat16 precision.
  FLOAT8_E4M3,  // float8 with E4M3 layout.
  FLOAT8_E5M2,  // float8 with E5M2 layout.
  UINT8,        // 8-bit unsigned integer.
};

/// Word array.
template <int Bytes, bool Enabled = (Bytes >= 4 && Bytes % 4 == 0)>
struct alignas(Bytes) Words {
  uint32_t w[Bytes / 4];

  MSCCLPP_HOST_DEVICE_INLINE Words() {}

  MSCCLPP_HOST_DEVICE_INLINE uint32_t& operator[](int i) { return w[i]; }

  MSCCLPP_HOST_DEVICE_INLINE const uint32_t& operator[](int i) const { return w[i]; }
};

template <int Bytes>
struct alignas(Bytes) Words<Bytes, false> {};

/// Vector type implementation (internal).
template <typename T, int N, typename StorageT>
union alignas(sizeof(T) * N) VectorTypeImpl {
  static_assert(N > 0, "N must be greater than 0");

  T data[N];
  Words<sizeof(T) * N> words;
  StorageT storage;

  using ElementType = T;
  constexpr static int Size = N;

  MSCCLPP_HOST_DEVICE_INLINE VectorTypeImpl() {}

  MSCCLPP_HOST_DEVICE_INLINE VectorTypeImpl(const StorageT& value) : storage(value) {}

  MSCCLPP_HOST_DEVICE_INLINE VectorTypeImpl(const VectorTypeImpl& other) { storage = other.storage; }

  MSCCLPP_HOST_DEVICE_INLINE VectorTypeImpl& operator=(const VectorTypeImpl& other) {
    storage = other.storage;
    return *this;
  }

  /// Converting constructor from a different element type with the same vector size.
  /// Enables e.g. `f32x2 a = some_f8_e4m3x2_value;`
  /// Uses element-wise static_cast. For the optimized path (e.g. hardware FP8->F32 intrinsics),
  /// use mscclpp::to<TargetType>(value) explicitly.
  template <typename OtherT, typename OtherStorageT,
            typename = std::enable_if_t<!std::is_same_v<T, OtherT>>>
  MSCCLPP_HOST_DEVICE_INLINE VectorTypeImpl(const VectorTypeImpl<OtherT, N, OtherStorageT>& other) {
#if defined(MSCCLPP_DEVICE_COMPILE)
#pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      data[i] = static_cast<T>(other.data[i]);
    }
  }

  /// Converting assignment from a different element type with the same vector size.
  template <typename OtherT, typename OtherStorageT,
            typename = std::enable_if_t<!std::is_same_v<T, OtherT>>>
  MSCCLPP_HOST_DEVICE_INLINE VectorTypeImpl& operator=(const VectorTypeImpl<OtherT, N, OtherStorageT>& other) {
#if defined(MSCCLPP_DEVICE_COMPILE)
#pragma unroll
#endif
    for (int i = 0; i < N; ++i) {
      data[i] = static_cast<T>(other.data[i]);
    }
    return *this;
  }

  MSCCLPP_HOST_DEVICE_INLINE operator StorageT() const { return storage; }

  MSCCLPP_HOST_DEVICE_INLINE operator T*() { return data; }

  MSCCLPP_HOST_DEVICE_INLINE operator const T*() const { return data; }

  MSCCLPP_HOST_DEVICE_INLINE T& operator[](int i) { return data[i]; }

  MSCCLPP_HOST_DEVICE_INLINE const T& operator[](int i) const { return data[i]; }
};

// Helper template to get the appropriate vector type for a given element type and count
template <typename T, int N>
struct VectorTypeHelper {
  using type =
      VectorTypeImpl<T, N,
                     typename std::conditional_t<N * sizeof(T) == 4, uint32_t,
                                                 typename std::conditional_t<N * sizeof(T) == 8, uint2, uint4>>>;
};

/// Vector type - clean user interface (automatically selects appropriate storage type)
template <typename T, int N>
using VectorType = typename VectorTypeHelper<T, N>::type;

// Macro to define specialization AND alias in one go
#define DEFINE_VEC(Alias, T, N, Storage)        \
  template <>                                   \
  struct VectorTypeHelper<T, N> {               \
    using type = VectorTypeImpl<T, N, Storage>; \
  };                                            \
  using Alias = VectorType<T, N>

DEFINE_VEC(i32x1, int32_t, 1, int32_t);
DEFINE_VEC(u32x1, uint32_t, 1, uint32_t);
DEFINE_VEC(f32x1, float, 1, float);
DEFINE_VEC(f64x1, double, 1, double);

DEFINE_VEC(i32x2, int32_t, 2, int2);
DEFINE_VEC(u32x2, uint32_t, 2, uint2);
DEFINE_VEC(u8x2, uint8_t, 2, uint16_t);
DEFINE_VEC(f32x2, float, 2, float2);
DEFINE_VEC(f16x2, __half, 2, __half2);
DEFINE_VEC(bf16x2, __bfloat16, 2, __bfloat162);

DEFINE_VEC(i32x4, int32_t, 4, int4);
DEFINE_VEC(u32x4, uint32_t, 4, uint4);
DEFINE_VEC(u8x4, uint8_t, 4, uint32_t);
DEFINE_VEC(f32x4, float, 4, float4);
DEFINE_VEC(f16x4, __half, 4, uint2);
DEFINE_VEC(bf16x4, __bfloat16, 4, uint2);

DEFINE_VEC(f16x8, __half, 8, uint4);
DEFINE_VEC(bf16x8, __bfloat16, 8, uint4);

#if defined(__FP8_TYPES_EXIST__)
DEFINE_VEC(f8_e4m3x2, __fp8_e4m3, 2, __fp8x2_e4m3);
DEFINE_VEC(f8_e4m3x4, __fp8_e4m3, 4, __fp8x4_e4m3);
DEFINE_VEC(f8_e4m3x8, __fp8_e4m3, 8, uint2);
DEFINE_VEC(f8_e4m3x16, __fp8_e4m3, 16, uint4);

DEFINE_VEC(f8_e5m2x2, __fp8_e5m2, 2, __fp8x2_e5m2);
DEFINE_VEC(f8_e5m2x4, __fp8_e5m2, 4, __fp8x4_e5m2);
DEFINE_VEC(f8_e5m2x8, __fp8_e5m2, 8, uint2);
DEFINE_VEC(f8_e5m2x16, __fp8_e5m2, 16, uint4);
#endif
#undef DEFINE_VEC

#if defined(MSCCLPP_DEVICE_COMPILE)
template <typename To, typename From>
MSCCLPP_DEVICE_INLINE To bit_cast(const From& src) {
  static_assert(sizeof(To) == sizeof(From), "Size mismatch for bit_cast");

  union {
    From f;
    To t;
  } u{.f = src};
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

template <bool UseClip = true>
MSCCLPP_DEVICE_INLINE f16x2 operator+(const f16x2& a, const f16x2& b) {
  __half2 result;
  if constexpr (UseClip) {
    result = clip(__hadd2(a, b));
  } else {
    result = __hadd2(a, b);
  }
  return result;
}

template <bool UseClip = true>
MSCCLPP_DEVICE_INLINE bf16x2 operator+(const bf16x2& a, const bf16x2& b) {
  __bfloat162 result;
  if constexpr (UseClip) {
    result = clip(__hadd2(a, b));
  } else {
    result = __hadd2(a, b);
  }
  return result;
}

#if defined(__FP8_TYPES_EXIST__)
template <bool UseClip = true>
MSCCLPP_DEVICE_INLINE __fp8_e4m3 operator+(const __fp8_e4m3& a, const __fp8_e4m3& b) {
#if defined(MSCCLPP_DEVICE_HIP) && defined(__gfx942__)
  // Optimized assembly for gfx942
  float2 v;
  uint32_t ival = 0;
  asm volatile("v_pk_add_f32 %0, %1, %2"
               : "=v"(v)
               : "v"(__builtin_amdgcn_cvt_pk_f32_fp8(a.__x, 0)), "v"(__builtin_amdgcn_cvt_pk_f32_fp8(b.__x, 0)));
  return static_cast<__hip_fp8_storage_t>(__builtin_amdgcn_cvt_pk_fp8_f32(v.x, v.x, ival, false));
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
MSCCLPP_DEVICE_INLINE f8_e4m3x2 operator+(const f8_e4m3x2& a, const f8_e4m3x2& b) {
#if defined(MSCCLPP_DEVICE_HIP) && defined(__gfx942__)
  float2 v;
  uint32_t ival = 0;
  asm volatile("v_pk_add_f32 %0, %1, %2"
               : "=v"(v)
               : "v"(__builtin_amdgcn_cvt_pk_f32_fp8(a.storage.__x, 0)),
                 "v"(__builtin_amdgcn_cvt_pk_f32_fp8(b.storage.__x, 0)));
  return bit_cast<f8_e4m3x2>(
      static_cast<__hip_fp8x2_storage_t>(__builtin_amdgcn_cvt_pk_fp8_f32(v.x, v.y, ival, false)));
#elif defined(MSCCLPP_DEVICE_CUDA)
  // CUDA: Convert to half2, add using optimized __hadd2, convert back
  return __fp8x2_e4m3(__hadd2(__half2(static_cast<__fp8x2_e4m3>(a)), __half2(static_cast<__fp8x2_e4m3>(b))));
#else
  // Fallback for other devices: element-wise using single-element operations
  f8_e4m3x2 result;
  result.data[0] = a.data[0] + b.data[0];
  result.data[1] = a.data[1] + b.data[1];
  return result;
#endif
}

template <bool UseClip = true>
MSCCLPP_DEVICE_INLINE f8_e4m3x4 operator+(const f8_e4m3x4& a, const f8_e4m3x4& b) {
#if defined(MSCCLPP_DEVICE_HIP) && defined(__gfx942__)
  float2 v_low, v_high;
  // E4M3 using fp8 conversion - process low word (false) and high word (true)
  asm volatile("v_pk_add_f32 %0, %1, %2"
               : "=v"(v_low)
               : "v"(__builtin_amdgcn_cvt_pk_f32_fp8(a.storage.__x, false)),
                 "v"(__builtin_amdgcn_cvt_pk_f32_fp8(b.storage.__x, false)));
  uint32_t result_packed = __builtin_amdgcn_cvt_pk_fp8_f32(v_low.x, v_low.y, 0, false);

  asm volatile("v_pk_add_f32 %0, %1, %2"
               : "=v"(v_high)
               : "v"(__builtin_amdgcn_cvt_pk_f32_fp8(a.storage.__x, true)),
                 "v"(__builtin_amdgcn_cvt_pk_f32_fp8(b.storage.__x, true)));
  result_packed = __builtin_amdgcn_cvt_pk_fp8_f32(v_high.x, v_high.y, result_packed, true);
  return bit_cast<f8_e4m3x4>(result_packed);
#else
  // Process as two f8_e4m3x2 using operator+ for 2 elements
  const f8_e4m3x2* a_pair = reinterpret_cast<const f8_e4m3x2*>(&a);
  const f8_e4m3x2* b_pair = reinterpret_cast<const f8_e4m3x2*>(&b);

  f8_e4m3x2 result[2];
  result[0] = a_pair[0] + b_pair[0];
  result[1] = a_pair[1] + b_pair[1];

  return *reinterpret_cast<f8_e4m3x4*>(result);
#endif
}

template <bool UseClip = true>
MSCCLPP_DEVICE_INLINE __fp8_e5m2 operator+(const __fp8_e5m2& a, const __fp8_e5m2& b) {
#if defined(MSCCLPP_DEVICE_HIP) && defined(__gfx942__)
  // Optimized assembly for gfx942 (bfloat8)
  float2 v;
  uint32_t ival = 0;
  asm volatile("v_pk_add_f32 %0, %1, %2"
               : "=v"(v)
               : "v"(__builtin_amdgcn_cvt_pk_f32_bf8(a.__x, 0)), "v"(__builtin_amdgcn_cvt_pk_f32_bf8(b.__x, 0)));
  return static_cast<__hip_fp8_storage_t>(__builtin_amdgcn_cvt_pk_bf8_f32(v.x, v.x, ival, false));
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
MSCCLPP_DEVICE_INLINE f8_e5m2x2 operator+(const f8_e5m2x2& a, const f8_e5m2x2& b) {
#if defined(MSCCLPP_DEVICE_CUDA)
  // CUDA: Convert to half2, add using optimized __hadd2, convert back
  f8_e5m2x2 result =
      __fp8x2_e5m2(__hadd2(__half2(static_cast<__fp8x2_e5m2>(a)), __half2(static_cast<__fp8x2_e5m2>(b))));
  if constexpr (UseClip) {
    result = clip(result);
  }
  return result;
#elif defined(MSCCLPP_DEVICE_HIP) && defined(__gfx942__)
  // HIP gfx942: Use BF8 assembly instructions
  float2 v;
  uint32_t ival = 0;
  asm volatile("v_pk_add_f32 %0, %1, %2"
               : "=v"(v)
               : "v"(__builtin_amdgcn_cvt_pk_f32_bf8(a.data[0].__x, 0)),
                 "v"(__builtin_amdgcn_cvt_pk_f32_bf8(b.data[0].__x, 0)));
  return bit_cast<f8_e5m2x2>(
      static_cast<__hip_fp8x2_storage_t>(__builtin_amdgcn_cvt_pk_bf8_f32(v.x, v.y, ival, false)));
#else
  // Fallback: element-wise using single-element operations
  f8_e5m2x2 result;
  result.data[0] = a.data[0] + b.data[0];
  result.data[1] = a.data[1] + b.data[1];
  return result;
#endif
}

template <bool UseClip = true>
MSCCLPP_DEVICE_INLINE f8_e5m2x4 operator+(const f8_e5m2x4& a, const f8_e5m2x4& b) {
#if defined(MSCCLPP_DEVICE_HIP) && defined(__gfx942__)
  float2 v_low, v_high;
  // E5M2 using bf8 conversion - process low word (false) and high word (true)
  asm volatile("v_pk_add_f32 %0, %1, %2"
               : "=v"(v_low)
               : "v"(__builtin_amdgcn_cvt_pk_f32_bf8(a.storage.__x, false)),
                 "v"(__builtin_amdgcn_cvt_pk_f32_bf8(b.storage.__x, false)));
  uint32_t result_packed = __builtin_amdgcn_cvt_pk_bf8_f32(v_low.x, v_low.y, 0, false);

  asm volatile("v_pk_add_f32 %0, %1, %2"
               : "=v"(v_high)
               : "v"(__builtin_amdgcn_cvt_pk_f32_bf8(a.storage.__x, true)),
                 "v"(__builtin_amdgcn_cvt_pk_f32_bf8(b.storage.__x, true)));
  result_packed = __builtin_amdgcn_cvt_pk_bf8_f32(v_high.x, v_high.y, result_packed, true);
  return bit_cast<f8_e5m2x4>(result_packed);
#else
  // Process as two f8_e5m2x2 using operator+ for 2 elements
  const f8_e5m2x2* a_pair = reinterpret_cast<const f8_e5m2x2*>(&a);
  const f8_e5m2x2* b_pair = reinterpret_cast<const f8_e5m2x2*>(&b);
  f8_e5m2x2 result[2];
  result[0] = a_pair[0] + b_pair[0];
  result[1] = a_pair[1] + b_pair[1];

  return *reinterpret_cast<f8_e5m2x4*>(result);
#endif
}
#endif  // defined(__FP8_TYPES_EXIST__)

MSCCLPP_DEVICE_INLINE u8x4 operator+(const u8x4& a, const u8x4& b) {
#if defined(MSCCLPP_DEVICE_HIP)
  // Optimized uint8_t x 4 sum using byte permute to avoid overflow between adjacent bytes
  constexpr uint32_t even = 0x00ff00ffu;
  uint32_t ua = a.storage;
  uint32_t ub = b.storage;
  uint32_t x = (ua & even) + (ub & even);
  uint32_t y = (ua & ~even) + (ub & ~even);
  return __byte_perm(x, y, 0x7250);
#else
  return __vadd4(a.storage, b.storage);
#endif
}

template <typename T>
MSCCLPP_DEVICE_INLINE T min(const T& a, const T& b) {
  return (a < b ? a : b);
}

template <>
MSCCLPP_DEVICE_INLINE f16x2 min(const f16x2& a, const f16x2& b) {
#if defined(MSCCLPP_DEVICE_HIP)
  f16x2 val;
  val[0] = __hmin(a[0], b[0]);
  val[1] = __hmin(a[1], b[1]);
  return val;
#else
  __half2 ret = __hmin2(a, b);
  return ret;
#endif
}

template <>
MSCCLPP_DEVICE_INLINE bf16x2 min(const bf16x2& a, const bf16x2& b) {
  return __hmin2(a, b);
}

template <>
MSCCLPP_DEVICE_INLINE u8x4 min(const u8x4& a, const u8x4& b) {
#if defined(MSCCLPP_DEVICE_HIP)
  // Optimized uint8_t x 4 min using 9-bit arithmetic
  constexpr uint32_t ones = 0x01010101u;
  constexpr uint32_t even = 0x00ff00ffu;  // even byte mask
  uint32_t ua = a.storage;
  uint32_t ub = b.storage;
  // Use 9-bit arithmetic to compute d=a-b for each byte
  uint32_t d0 = (ua & even) + (~ub & even) + ones;
  uint32_t d1 = ((ua >> 8) & even) + (~(ub >> 8) & even) + ones;
  // Move sign bit of each 9-bit delta into the least bit of origin byte
  uint32_t s = __byte_perm(d0, d1, 0x7351) & ones;
  // Broadcast least bit across whole byte
  s *= 0xffu;
  // Compose result by selecting bytes via: signbit(a-b)==1 ? a : b
  return (ua & s) | (ub & ~s);
#else
  return __vminu4(a.storage, b.storage);
#endif
}

#if defined(__FP8_TYPES_EXIST__)
template <>
MSCCLPP_DEVICE_INLINE __fp8_e4m3 min(const __fp8_e4m3& a, const __fp8_e4m3& b) {
#if defined(MSCCLPP_DEVICE_HIP)
  return __fp8_e4m3(fminf(float(a), float(b)));
#else
  return __fp8_e4m3(__hmin(__half(a), __half(b)));
#endif
}

MSCCLPP_DEVICE_INLINE f8_e4m3x2 min(const f8_e4m3x2& a, const f8_e4m3x2& b) {
  // Process element-wise using single-element operations
  f8_e4m3x2 result;
  result.data[0] = mscclpp::min(a.data[0], b.data[0]);
  result.data[1] = mscclpp::min(a.data[1], b.data[1]);
  return result;
}

MSCCLPP_DEVICE_INLINE f8_e4m3x4 min(const f8_e4m3x4& a, const f8_e4m3x4& b) {
  // Process as two f8_e4m3x2 using min for 2 elements
  const f8_e4m3x2* a_ptr = reinterpret_cast<const f8_e4m3x2*>(&a);
  const f8_e4m3x2* b_ptr = reinterpret_cast<const f8_e4m3x2*>(&b);

  f8_e4m3x4 result;
  f8_e4m3x2* result_ptr = reinterpret_cast<f8_e4m3x2*>(&result);

  result_ptr[0] = mscclpp::min(a_ptr[0], b_ptr[0]);
  result_ptr[1] = mscclpp::min(a_ptr[1], b_ptr[1]);

  return result;
}

template <>
MSCCLPP_DEVICE_INLINE __fp8_e5m2 min(const __fp8_e5m2& a, const __fp8_e5m2& b) {
#if defined(MSCCLPP_DEVICE_HIP)
  return __fp8_e5m2(fminf(float(a), float(b)));
#else
  return __fp8_e5m2(__hmin(__half(a), __half(b)));
#endif
}

MSCCLPP_DEVICE_INLINE f8_e5m2x2 min(const f8_e5m2x2& a, const f8_e5m2x2& b) {
  // Process element-wise using single-element operations
  f8_e5m2x2 result;
  result.data[0] = mscclpp::min(a.data[0], b.data[0]);
  result.data[1] = mscclpp::min(a.data[1], b.data[1]);
  return result;
}

MSCCLPP_DEVICE_INLINE f8_e5m2x4 min(const f8_e5m2x4& a, const f8_e5m2x4& b) {
  // Process as two f8_e5m2x2 using min for 2 elements
  const f8_e5m2x2* a_ptr = reinterpret_cast<const f8_e5m2x2*>(&a);
  const f8_e5m2x2* b_ptr = reinterpret_cast<const f8_e5m2x2*>(&b);

  f8_e5m2x4 result;
  f8_e5m2x2* result_ptr = reinterpret_cast<f8_e5m2x2*>(&result);

  result_ptr[0] = mscclpp::min(a_ptr[0], b_ptr[0]);
  result_ptr[1] = mscclpp::min(a_ptr[1], b_ptr[1]);

  return result;
}

/// Convert a vector type From to vector type To.
/// Primary template: element-wise conversion via static_cast.
/// Specialized below for optimized FP8 conversion paths.
template <typename To, typename From>
MSCCLPP_DEVICE_INLINE To to(const From& v) {
  static_assert(To::Size == From::Size, "to<To, From>: vector sizes must match");
  To result;
#pragma unroll
  for (int i = 0; i < From::Size; ++i) {
    result.data[i] = static_cast<typename To::ElementType>(v.data[i]);
  }
  return result;
}

// --- f8_e4m3 -> f32 specializations ---

/// f8_e4m3x2 -> f32x2.
/// NVIDIA: fp8 -> half (via __nv_cvt_fp8x2_to_halfraw2) -> float.
/// HIP gfx942: fp8 -> float (via __builtin_amdgcn_cvt_pk_f32_fp8).
template <>
MSCCLPP_DEVICE_INLINE f32x2 to<f32x2, f8_e4m3x2>(const f8_e4m3x2& v) {
#if defined(MSCCLPP_DEVICE_HIP) && defined(__gfx942__)
  float2 f = __builtin_amdgcn_cvt_pk_f32_fp8(v.storage.__x, 0);
  return f;
#elif defined(MSCCLPP_DEVICE_CUDA) && __CUDA_ARCH__ >= 900
  __half2_raw h2 = __nv_cvt_fp8x2_to_halfraw2(bit_cast<__nv_fp8x2_storage_t>(v.storage), __NV_E4M3);
  f32x2 result;
  result.data[0] = __half2float(bit_cast<__half>(h2.x));
  result.data[1] = __half2float(bit_cast<__half>(h2.y));
  return result;
#else
  f32x2 result;
  result.data[0] = float(v.data[0]);
  result.data[1] = float(v.data[1]);
  return result;
#endif
}

/// f8_e4m3x4 -> f32x4.
template <>
MSCCLPP_DEVICE_INLINE f32x4 to<f32x4, f8_e4m3x4>(const f8_e4m3x4& v) {
#if defined(MSCCLPP_DEVICE_HIP) && defined(__gfx942__)
  float2 lo = __builtin_amdgcn_cvt_pk_f32_fp8(v.storage.__x, false);
  float2 hi = __builtin_amdgcn_cvt_pk_f32_fp8(v.storage.__x, true);
  f32x4 result;
  result.data[0] = lo.x;
  result.data[1] = lo.y;
  result.data[2] = hi.x;
  result.data[3] = hi.y;
  return result;
#else
  const f8_e4m3x2* pair = reinterpret_cast<const f8_e4m3x2*>(&v);
  f32x2 lo = to<f32x2>(pair[0]);
  f32x2 hi = to<f32x2>(pair[1]);
  f32x4 result;
  result.data[0] = lo.data[0];
  result.data[1] = lo.data[1];
  result.data[2] = hi.data[0];
  result.data[3] = hi.data[1];
  return result;
#endif
}

// --- f8_e5m2 -> f32 specializations ---

/// f8_e5m2x2 -> f32x2.
/// NVIDIA: fp8 -> half (via __nv_cvt_fp8x2_to_halfraw2) -> float.
/// HIP gfx942: bf8 -> float (via __builtin_amdgcn_cvt_pk_f32_bf8).
template <>
MSCCLPP_DEVICE_INLINE f32x2 to<f32x2, f8_e5m2x2>(const f8_e5m2x2& v) {
#if defined(MSCCLPP_DEVICE_HIP) && defined(__gfx942__)
  float2 f = __builtin_amdgcn_cvt_pk_f32_bf8(v.storage.__x, 0);
  return f;
#elif defined(MSCCLPP_DEVICE_CUDA) && __CUDA_ARCH__ >= 900
  __half2_raw h2 = __nv_cvt_fp8x2_to_halfraw2(bit_cast<__nv_fp8x2_storage_t>(v.storage), __NV_E5M2);
  f32x2 result;
  result.data[0] = __half2float(bit_cast<__half>(h2.x));
  result.data[1] = __half2float(bit_cast<__half>(h2.y));
  return result;
#else
  f32x2 result;
  result.data[0] = float(v.data[0]);
  result.data[1] = float(v.data[1]);
  return result;
#endif
}

/// f8_e5m2x4 -> f32x4.
template <>
MSCCLPP_DEVICE_INLINE f32x4 to<f32x4, f8_e5m2x4>(const f8_e5m2x4& v) {
#if defined(MSCCLPP_DEVICE_HIP) && defined(__gfx942__)
  float2 lo = __builtin_amdgcn_cvt_pk_f32_bf8(v.storage.__x, false);
  float2 hi = __builtin_amdgcn_cvt_pk_f32_bf8(v.storage.__x, true);
  f32x4 result;
  result.data[0] = lo.x;
  result.data[1] = lo.y;
  result.data[2] = hi.x;
  result.data[3] = hi.y;
  return result;
#else
  const f8_e5m2x2* pair = reinterpret_cast<const f8_e5m2x2*>(&v);
  f32x2 lo = to<f32x2>(pair[0]);
  f32x2 hi = to<f32x2>(pair[1]);
  f32x4 result;
  result.data[0] = lo.data[0];
  result.data[1] = lo.data[1];
  result.data[2] = hi.data[0];
  result.data[3] = hi.data[1];
  return result;
#endif
}

// --- f32 -> f8_e4m3 specializations (downcast) ---

/// f32x2 -> f8_e4m3x2.
/// HIP gfx942: float -> fp8 (via __builtin_amdgcn_cvt_pk_fp8_f32).
/// NVIDIA SM90+: float -> half -> fp8 (via __nv_cvt_halfraw2_to_fp8x2).
/// NVIDIA pre-SM90: float -> half -> fp8 (via __nv_cvt_halfraw_to_fp8, element-wise).
template <>
MSCCLPP_DEVICE_INLINE f8_e4m3x2 to<f8_e4m3x2, f32x2>(const f32x2& v) {
#if defined(MSCCLPP_DEVICE_HIP) && defined(__gfx942__)
  uint32_t packed = __builtin_amdgcn_cvt_pk_fp8_f32(v.data[0], v.data[1], 0, false);
  return bit_cast<f8_e4m3x2>(static_cast<__hip_fp8x2_storage_t>(packed));
#elif defined(MSCCLPP_DEVICE_CUDA) && __CUDA_ARCH__ >= 900
  __half2_raw h2;
  h2.x = bit_cast<unsigned short>(__float2half_rn(v.data[0]));
  h2.y = bit_cast<unsigned short>(__float2half_rn(v.data[1]));
  __nv_fp8x2_storage_t fp8x2 = __nv_cvt_halfraw2_to_fp8x2(h2, __NV_SATFINITE, __NV_E4M3);
  return bit_cast<f8_e4m3x2>(fp8x2);
#elif defined(MSCCLPP_DEVICE_CUDA)
  __half_raw h0, h1;
  h0.x = bit_cast<unsigned short>(__float2half_rn(v.data[0]));
  h1.x = bit_cast<unsigned short>(__float2half_rn(v.data[1]));
  f8_e4m3x2 result;
  result.data[0] = bit_cast<__fp8_e4m3>(__nv_cvt_halfraw_to_fp8(h0, __NV_SATFINITE, __NV_E4M3));
  result.data[1] = bit_cast<__fp8_e4m3>(__nv_cvt_halfraw_to_fp8(h1, __NV_SATFINITE, __NV_E4M3));
  return result;
#else
  f8_e4m3x2 result;
  result.data[0] = static_cast<__fp8_e4m3>(v.data[0]);
  result.data[1] = static_cast<__fp8_e4m3>(v.data[1]);
  return result;
#endif
}

/// f32x4 -> f8_e4m3x4.
template <>
MSCCLPP_DEVICE_INLINE f8_e4m3x4 to<f8_e4m3x4, f32x4>(const f32x4& v) {
#if defined(MSCCLPP_DEVICE_HIP) && defined(__gfx942__)
  uint32_t packed = __builtin_amdgcn_cvt_pk_fp8_f32(v.data[0], v.data[1], 0, false);
  packed = __builtin_amdgcn_cvt_pk_fp8_f32(v.data[2], v.data[3], packed, true);
  return bit_cast<f8_e4m3x4>(packed);
#else
  f32x2 lo, hi;
  lo.data[0] = v.data[0];
  lo.data[1] = v.data[1];
  hi.data[0] = v.data[2];
  hi.data[1] = v.data[3];
  f8_e4m3x2 lo_fp8 = to<f8_e4m3x2>(lo);
  f8_e4m3x2 hi_fp8 = to<f8_e4m3x2>(hi);
  f8_e4m3x4 result;
  result.data[0] = lo_fp8.data[0];
  result.data[1] = lo_fp8.data[1];
  result.data[2] = hi_fp8.data[0];
  result.data[3] = hi_fp8.data[1];
  return result;
#endif
}

// --- f32 -> f8_e5m2 specializations (downcast) ---

/// f32x2 -> f8_e5m2x2.
/// HIP gfx942: float -> bf8 (via __builtin_amdgcn_cvt_pk_bf8_f32).
/// NVIDIA SM90+: float -> half -> fp8 (via __nv_cvt_halfraw2_to_fp8x2 with __NV_E5M2).
/// NVIDIA pre-SM90: float -> half -> fp8 (via __nv_cvt_halfraw_to_fp8, element-wise).
template <>
MSCCLPP_DEVICE_INLINE f8_e5m2x2 to<f8_e5m2x2, f32x2>(const f32x2& v) {
#if defined(MSCCLPP_DEVICE_HIP) && defined(__gfx942__)
  uint32_t packed = __builtin_amdgcn_cvt_pk_bf8_f32(v.data[0], v.data[1], 0, false);
  return bit_cast<f8_e5m2x2>(static_cast<__hip_fp8x2_storage_t>(packed));
#elif defined(MSCCLPP_DEVICE_CUDA) && __CUDA_ARCH__ >= 900
  __half2_raw h2;
  h2.x = bit_cast<unsigned short>(__float2half_rn(v.data[0]));
  h2.y = bit_cast<unsigned short>(__float2half_rn(v.data[1]));
  __nv_fp8x2_storage_t fp8x2 = __nv_cvt_halfraw2_to_fp8x2(h2, __NV_SATFINITE, __NV_E5M2);
  return bit_cast<f8_e5m2x2>(fp8x2);
#elif defined(MSCCLPP_DEVICE_CUDA)
  __half_raw h0, h1;
  h0.x = bit_cast<unsigned short>(__float2half_rn(v.data[0]));
  h1.x = bit_cast<unsigned short>(__float2half_rn(v.data[1]));
  f8_e5m2x2 result;
  result.data[0] = bit_cast<__fp8_e5m2>(__nv_cvt_halfraw_to_fp8(h0, __NV_SATFINITE, __NV_E5M2));
  result.data[1] = bit_cast<__fp8_e5m2>(__nv_cvt_halfraw_to_fp8(h1, __NV_SATFINITE, __NV_E5M2));
  return result;
#else
  f8_e5m2x2 result;
  result.data[0] = static_cast<__fp8_e5m2>(v.data[0]);
  result.data[1] = static_cast<__fp8_e5m2>(v.data[1]);
  return result;
#endif
}

/// f32x4 -> f8_e5m2x4.
template <>
MSCCLPP_DEVICE_INLINE f8_e5m2x4 to<f8_e5m2x4, f32x4>(const f32x4& v) {
#if defined(MSCCLPP_DEVICE_HIP) && defined(__gfx942__)
  uint32_t packed = __builtin_amdgcn_cvt_pk_bf8_f32(v.data[0], v.data[1], 0, false);
  packed = __builtin_amdgcn_cvt_pk_bf8_f32(v.data[2], v.data[3], packed, true);
  return bit_cast<f8_e5m2x4>(packed);
#else
  f32x2 lo, hi;
  lo.data[0] = v.data[0];
  lo.data[1] = v.data[1];
  hi.data[0] = v.data[2];
  hi.data[1] = v.data[3];
  f8_e5m2x2 lo_fp8 = to<f8_e5m2x2>(lo);
  f8_e5m2x2 hi_fp8 = to<f8_e5m2x2>(hi);
  f8_e5m2x4 result;
  result.data[0] = lo_fp8.data[0];
  result.data[1] = lo_fp8.data[1];
  result.data[2] = hi_fp8.data[0];
  result.data[3] = hi_fp8.data[1];
  return result;
#endif
}
#endif  // defined(__FP8_TYPES_EXIST__)
#endif  // MSCCLPP_DEVICE_COMPILE
}  // namespace mscclpp

#endif  // MSCCLPP_GPU_DATA_TYPES_HPP_
