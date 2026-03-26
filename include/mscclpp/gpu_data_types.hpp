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

/// Software float8 with 4 exponent bits, 3 mantissa bits, exponent bias = 15.
/// Format (MSB first): [sign:1][exponent:4][mantissa:3]
/// No infinities; exp=15 is NaN. Negative zero is NaN (fnuz convention).
/// Max finite value: 0.9375, min normal: ~6.1e-5, min subnormal: ~7.6e-6.
struct alignas(1) __fp8_e4m3b15 {
  uint8_t __x;

  __fp8_e4m3b15() = default;

  /// Construct from raw bits (use __fp8_e4m3b15::fromRaw() for clarity).
  MSCCLPP_HOST_DEVICE_INLINE explicit __fp8_e4m3b15(uint8_t raw) : __x(raw) {}

  /// Construct from float32 (explicit to avoid ambiguous conversion chains).
  MSCCLPP_HOST_DEVICE_INLINE explicit __fp8_e4m3b15(float val) : __x(fromFloat(val)) {}

  /// Convert to float32.
  MSCCLPP_HOST_DEVICE_INLINE operator float() const { return toFloat(__x); }

  /// Construct from a raw bit pattern without conversion.
  static MSCCLPP_HOST_DEVICE_INLINE __fp8_e4m3b15 fromRaw(uint8_t bits) {
    __fp8_e4m3b15 r;
    r.__x = bits;
    return r;
  }

 private:
  /// Decode fp8_e4m3b15 bits → float32.
  ///
  /// Uses bit manipulation through fp16 as intermediate, adapted from the Triton compiler.
  /// fp8_e4m3b15 is identical to fp8_e4m3fn (NVIDIA) except exponent bias is 15 vs 7.
  /// Algorithm: reinterpret fp8 bits into an fp16 bit pattern with exponent shifted by -8,
  /// then convert fp16 → float32.
  static MSCCLPP_HOST_DEVICE_INLINE float toFloat(uint8_t bits) {
    // Handle special values: negative zero (0x80) → NaN, exponent=15 → NaN.
    uint32_t exp = (bits >> 3) & 0xFu;
    if (bits == 0x80 || exp == 15) {
      union {
        uint32_t u;
        float f;
      } nan_val = {0x7FC00000u};
      return nan_val.f;
    }
    if (bits == 0) return 0.0f;

    // Triton-style bit manipulation: fp8 → fp16 → fp32.
    // fp8 layout: [S:1][E:4][M:3]  (bias=15)
    // fp16 layout: [S:1][E:5][M:10] (bias=15)
    //
    // Place fp8 in upper byte of fp16, then right-shift exponent+mantissa by 1
    // to convert E4 → E5 (both share bias=15). Sign bit stays at bit 15.
    // Refer:
    // https://github.com/triton-lang/triton/blob/cf34004b8a67d290a962da166f5aa2fc66751326/python/triton/language/extra/cuda/utils.py#L34
    uint16_t h = (uint16_t)bits << 8;             // place fp8 in upper byte of fp16
    uint16_t sign16 = h & 0x8000u;                // extract sign at fp16 position
    uint16_t nosign = h & 0x7F00u;                // exponent + mantissa (no sign)
    uint16_t fp16_bits = sign16 | (nosign >> 1);  // shift exponent right by 1

    // For subnormals: when fp8 exponent=0, the above gives fp16 exponent=0
    // and fp16 mantissa = (fp8_mantissa << 7), which correctly represents
    // the subnormal fp16 value since both share bias=15.

    // Convert fp16 bits to float via __half (works on host and device, CUDA and HIP).
    union {
      uint16_t u;
      __half h;
    } cvt = {fp16_bits};
    return __half2float(cvt.h);
  }

  /// Encode float32 → fp8_e4m3b15 bits.
  ///
  /// Algorithm adapted from Triton: float32 → fp16 → bit-manipulate → fp8.
  /// The key insight is to convert to fp16 first (which shares bias=15 with e4m3b15),
  /// then pack the fp16 bits back into 8 bits by shifting the exponent left by 1.
  static MSCCLPP_HOST_DEVICE_INLINE uint8_t fromFloat(float val) {
    union {
      float f;
      uint32_t u;
    } in = {val};

    // NaN → 0x80 (negative-zero bit pattern = NaN in fnuz).
    if ((in.u & 0x7F800000u) == 0x7F800000u && (in.u & 0x007FFFFFu) != 0) return 0x80u;

    // Convert float32 → fp16 bits via __half (works on host and device, CUDA and HIP).
    __half h_val = __float2half_rn(val);
    union {
      __half h;
      uint16_t u;
    } cvt = {h_val};
    uint16_t fp16_bits = cvt.u;

    // Clamp absolute value to max finite e4m3b15: 0.9375 → fp16 = 0x3B80.
    uint16_t abs_fp16 = fp16_bits & 0x7FFFu;
    if (abs_fp16 > 0x3B80u) abs_fp16 = 0x3B80u;

    // Reconstruct with sign.
    uint16_t sign16 = fp16_bits & 0x8000u;

    // Triton-style: fp16 → fp8.
    // fp16 layout: [S:1][E:5][M:10] (bias=15)
    // fp8 layout:  [S:1][E:4][M:3]  (bias=15)
    //
    // mad.lo.u32 a0, a0, 2, 0x00800080  →  (abs_fp16 * 2 + 0x0080)
    // This shifts left by 1 (undoing the right-shift in decode) and adds rounding bias.
    // Then: lop3.b32 b0, $1, 0x80008000, a0, 0xea  →  (sign & 0x8000) | a0
    // Finally: prmt for byte extraction.
    //
    // Simplified for scalar: shift abs_fp16 left by 1, add rounding bias, take upper byte.
    uint16_t adjusted = (uint16_t)(abs_fp16 * 2u + 0x0080u);
    // The upper byte now contains [E:4][M:3][round_bit].
    // Combine with sign and extract.
    uint16_t with_sign = sign16 | adjusted;
    uint8_t result = (uint8_t)(with_sign >> 8);

    // Zero → 0x00 (ensure positive zero, not negative zero which is NaN).
    if ((result & 0x7Fu) == 0) result = 0x00u;

    return result;
  }
};

/// Packed 2x fp8_e4m3b15 storage.
struct alignas(2) __fp8x2_e4m3b15 {
  uint16_t __x;
};

/// Packed 4x fp8_e4m3b15 storage.
struct alignas(4) __fp8x4_e4m3b15 {
  uint32_t __x;
};

namespace mscclpp {

/// Data types supported by mscclpp operations.
enum class DataType {
  INT32,         // 32-bit signed integer.
  UINT32,        // 32-bit unsigned integer.
  FLOAT16,       // IEEE 754 half precision.
  FLOAT32,       // IEEE 754 single precision.
  BFLOAT16,      // bfloat16 precision.
  FLOAT8_E4M3,   // float8 with E4M3 layout.
  FLOAT8_E5M2,   // float8 with E5M2 layout.
  UINT8,         // 8-bit unsigned integer.
  FLOAT8_E4B15,  // float8 with E4M3 layout, bias=15 (software, no HW accel).
  AUTO = 255,    // Sentinel: resolve to the input dtype at runtime.
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
  static_assert(sizeof(StorageT) >= sizeof(T) * N, "StorageT must cover the full vector size");

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

  MSCCLPP_HOST_DEVICE_INLINE operator StorageT() const { return storage; }

  MSCCLPP_HOST_DEVICE_INLINE operator T*() { return data; }

  MSCCLPP_HOST_DEVICE_INLINE operator const T*() const { return data; }

  MSCCLPP_HOST_DEVICE_INLINE T& operator[](int i) { return data[i]; }

  MSCCLPP_HOST_DEVICE_INLINE const T& operator[](int i) const { return data[i]; }
};

// Helper template to get the appropriate vector type for a given element type and count.
template <typename T, int N>
struct VectorTypeHelper {
  static constexpr int Bytes = N * sizeof(T);
  using type = VectorTypeImpl<
      T, N,
      std::conditional_t<Bytes == 4, uint32_t,
                         std::conditional_t<Bytes == 8, uint2, std::conditional_t<Bytes <= 16, uint4, Words<Bytes>>>>>;
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

// Aliases for large vector types (>16 bytes) where no native CUDA storage type exists.
using f32x8 = VectorType<float, 8>;
using f32x16 = VectorType<float, 16>;
using f16x16 = VectorType<__half, 16>;

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

// fp8_e4m3b15 vectors (always available — software type, no HW dependency)
DEFINE_VEC(f8_e4m3b15x2, __fp8_e4m3b15, 2, __fp8x2_e4m3b15);
DEFINE_VEC(f8_e4m3b15x4, __fp8_e4m3b15, 4, __fp8x4_e4m3b15);
DEFINE_VEC(f8_e4m3b15x8, __fp8_e4m3b15, 8, uint2);
DEFINE_VEC(f8_e4m3b15x16, __fp8_e4m3b15, 16, uint4);
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

// --- f32x2 arithmetic ---

template <bool UseClip = true>
MSCCLPP_DEVICE_INLINE f32x2 operator+(const f32x2& a, const f32x2& b) {
#if defined(MSCCLPP_DEVICE_CUDA) && (__CUDA_ARCH__ >= 1000)
  // Blackwell (SM 10.0+): packed float2 add in a single instruction.
  return __fadd2_rn(a.storage, b.storage);
#else
  f32x2 result;
  result.data[0] = a.data[0] + b.data[0];
  result.data[1] = a.data[1] + b.data[1];
  return result;
#endif
}

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
MSCCLPP_DEVICE_INLINE f16x4 operator+(const f16x4& a, const f16x4& b) {
  // Decompose into 2× packed __hadd2 (2 instructions instead of 4 scalar __hadd).
  const f16x2* a2 = reinterpret_cast<const f16x2*>(&a);
  const f16x2* b2 = reinterpret_cast<const f16x2*>(&b);
  f16x4 result;
  f16x2* r2 = reinterpret_cast<f16x2*>(&result);
  r2[0] = a2[0] + b2[0];
  r2[1] = a2[1] + b2[1];
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

// --- fp8_e4m3b15 arithmetic (software, always available) ---

template <bool UseClip = true>
MSCCLPP_DEVICE_INLINE __fp8_e4m3b15 operator+(const __fp8_e4m3b15& a, const __fp8_e4m3b15& b) {
  return __fp8_e4m3b15(float(a) + float(b));
}

template <bool UseClip = true>
MSCCLPP_DEVICE_INLINE f8_e4m3b15x2 operator+(const f8_e4m3b15x2& a, const f8_e4m3b15x2& b) {
  f8_e4m3b15x2 result;
  result.data[0] = __fp8_e4m3b15(float(a.data[0]) + float(b.data[0]));
  result.data[1] = __fp8_e4m3b15(float(a.data[1]) + float(b.data[1]));
  return result;
}

template <bool UseClip = true>
MSCCLPP_DEVICE_INLINE f8_e4m3b15x4 operator+(const f8_e4m3b15x4& a, const f8_e4m3b15x4& b) {
  f8_e4m3b15x4 result;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    result.data[i] = __fp8_e4m3b15(float(a.data[i]) + float(b.data[i]));
  }
  return result;
}

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
MSCCLPP_DEVICE_INLINE f32x2 min(const f32x2& a, const f32x2& b) {
  f32x2 result;
  result.data[0] = fminf(a.data[0], b.data[0]);
  result.data[1] = fminf(a.data[1], b.data[1]);
  return result;
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

// --- fp8_e4m3b15 min (software) ---

template <>
MSCCLPP_DEVICE_INLINE __fp8_e4m3b15 min(const __fp8_e4m3b15& a, const __fp8_e4m3b15& b) {
  return __fp8_e4m3b15(fminf(float(a), float(b)));
}

MSCCLPP_DEVICE_INLINE f8_e4m3b15x2 min(const f8_e4m3b15x2& a, const f8_e4m3b15x2& b) {
  f8_e4m3b15x2 result;
  result.data[0] = mscclpp::min(a.data[0], b.data[0]);
  result.data[1] = mscclpp::min(a.data[1], b.data[1]);
  return result;
}

MSCCLPP_DEVICE_INLINE f8_e4m3b15x4 min(const f8_e4m3b15x4& a, const f8_e4m3b15x4& b) {
  f8_e4m3b15x4 result;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    result.data[i] = mscclpp::min(a.data[i], b.data[i]);
  }
  return result;
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
  auto f = __builtin_amdgcn_cvt_pk_f32_fp8(v.storage.__x, 0);
  f32x2 result;
  result.data[0] = f[0];
  result.data[1] = f[1];
  return result;
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
  auto lo = __builtin_amdgcn_cvt_pk_f32_fp8(v.storage.__x, false);
  auto hi = __builtin_amdgcn_cvt_pk_f32_fp8(v.storage.__x, true);
  f32x4 result;
  result.data[0] = lo[0];
  result.data[1] = lo[1];
  result.data[2] = hi[0];
  result.data[3] = hi[1];
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
  auto f = __builtin_amdgcn_cvt_pk_f32_bf8(v.storage.__x, 0);
  f32x2 result;
  result.data[0] = f[0];
  result.data[1] = f[1];
  return result;
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
  auto lo = __builtin_amdgcn_cvt_pk_f32_bf8(v.storage.__x, false);
  auto hi = __builtin_amdgcn_cvt_pk_f32_bf8(v.storage.__x, true);
  f32x4 result;
  result.data[0] = lo[0];
  result.data[1] = lo[1];
  result.data[2] = hi[0];
  result.data[3] = hi[1];
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

// --- fp8_e4m3b15 <-> f32 conversion specializations (software, always available) ---

/// f8_e4m3b15x2 -> f32x2.
/// NVIDIA CUDA: place 2 fp8 bytes into a packed fp16 pair, adjust exponent (E4→E5),
/// convert to float32 via __half22float2 (packed half2 → float2).
template <>
MSCCLPP_DEVICE_INLINE f32x2 to<f32x2, f8_e4m3b15x2>(const f8_e4m3b15x2& v) {
#if defined(MSCCLPP_DEVICE_CUDA)
  uint16_t in = v.storage.__x;
  // Spread 2 fp8 bytes into a packed fp16 pair: byte[0]→upper byte of lo16, byte[1]→upper byte of hi16.
  uint32_t a0 = ((uint32_t)(in & 0xFFu) << 8) | ((uint32_t)(in >> 8) << 24);
  uint32_t b0 = (a0 & 0x7f007f00u) >> 1;
  uint32_t out0 = b0 | (a0 & 0x80008000u);  // 2 packed fp16
  __half2 h;
  asm("mov.b32 %0, %1;" : "=r"(*reinterpret_cast<uint32_t*>(&h)) : "r"(out0));
  // Packed half2 → float2 conversion (single cvt.f32x2.f16x2 on Blackwell).
  float2 f2 = __half22float2(h);
  return bit_cast<f32x2>(f2);
#else
  f32x2 result;
  result.data[0] = float(v.data[0]);
  result.data[1] = float(v.data[1]);
  return result;
#endif
}

/// f8_e4m3b15x4 -> f32x4.
/// NVIDIA CUDA: Triton-style vectorized bit manipulation (fp8x4 → fp16x4 → fp32x4).
/// Uses __byte_perm to spread 4 fp8 bytes into 2 packed fp16 pairs with each fp8
/// in the upper byte of its 16-bit lane, then adjusts the exponent with a right-shift
/// (E4 bias=15 → E5 bias=15) and converts to float32 via hardware __half2float.
/// Optimized: uses lop3.b32 to fuse (shift & mask) | sign into one instruction per path,
/// and a second prmt to move lower-byte fp8 values into upper-byte positions so both
/// paths share identical conversion logic (saves 1 instruction vs add+shift approach).
template <>
MSCCLPP_DEVICE_INLINE f32x4 to<f32x4, f8_e4m3b15x4>(const f8_e4m3b15x4& v) {
#if defined(MSCCLPP_DEVICE_CUDA)
  uint32_t in = v.storage.__x;

  // Byte permute: spread 4 fp8 bytes into 2 pairs of (upper-byte, lower-byte).
  // Source: {in, 0} bytes: 0-3=0, 4-7=in[0..3]=fp8[0..3].
  // 0x5746: byte0=fp8[2], byte1=fp8[0], byte2=fp8[3], byte3=fp8[1]
  // → fp8[0] at byte1 (lo16 upper), fp8[1] at byte3 (hi16 upper).
  uint32_t a0 = __byte_perm(0u, in, 0x5746u);

  // Upper-byte path: fp8[0] and fp8[1] in upper-byte positions of each 16-bit lane.
  // Right-shift by 1 converts E4→E5 exponent, lop3 merges masked shift with sign.
  // out0 = ((a0 >> 1) & 0x3f803f80) | (a0 & 0x80008000)   [lop3 truth table 0xEA = (A&B)|C]
  uint32_t a0_shr = a0 >> 1;
  uint32_t a0_sign = a0 & 0x80008000u;
  uint32_t out0;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(out0) : "r"(a0_shr), "r"(0x3f803f80u), "r"(a0_sign));

  // Lower-byte path: swap bytes within each 16-bit lane so fp8[2] and fp8[3]
  // move to upper-byte positions, then apply identical conversion.
  // 0x2301: {byte2,byte3,byte0,byte1} → fp8[2] at byte1 (lo16 upper), fp8[3] at byte3 (hi16 upper).
  uint32_t a1 = __byte_perm(a0, 0u, 0x2301u);
  uint32_t a1_shr = a1 >> 1;
  uint32_t a1_sign = a1 & 0x80008000u;
  uint32_t out1;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(out1) : "r"(a1_shr), "r"(0x3f803f80u), "r"(a1_sign));

  // Convert 4 packed fp16 values to 4 float32 via __half22float2
  // (single cvt.f32x2.f16x2 per pair on Blackwell, 2× cvt.f32.f16 on older).
  __half2 h0, h1;
  asm("mov.b32 %0, %1;" : "=r"(*reinterpret_cast<uint32_t*>(&h0)) : "r"(out0));
  asm("mov.b32 %0, %1;" : "=r"(*reinterpret_cast<uint32_t*>(&h1)) : "r"(out1));
  float2 f0 = __half22float2(h0);
  float2 f1 = __half22float2(h1);
  f32x4 result;
  result.data[0] = f0.x;
  result.data[1] = f0.y;
  result.data[2] = f1.x;
  result.data[3] = f1.y;
  return result;
#else
  f32x4 result;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    result.data[i] = float(v.data[i]);
  }
  return result;
#endif
}

/// f32x2 -> f8_e4m3b15x2.
/// NVIDIA CUDA: convert to packed fp16 pair, clamp, shift exponent (E5→E4), pack bytes.
template <>
MSCCLPP_DEVICE_INLINE f8_e4m3b15x2 to<f8_e4m3b15x2, f32x2>(const f32x2& v) {
#if defined(MSCCLPP_DEVICE_CUDA)
  // Packed float2 → half2 conversion (single cvt.rn.f16x2.f32 on Blackwell).
  float2 f2 = {v.data[0], v.data[1]};
  __half2 h = __float22half2_rn(f2);
  uint32_t in0;
  asm("mov.b32 %0, %1;" : "=r"(in0) : "r"(*reinterpret_cast<uint32_t*>(&h)));
  // Clamp abs to max finite e4m3b15 (0x3B80 = 0.9375 in fp16).
  uint32_t lo = in0 & 0xFFFFu, hi = in0 >> 16;
  uint32_t alo = lo & 0x7FFFu, ahi = hi & 0x7FFFu;
  alo = alo < 0x3B80u ? alo : 0x3B80u;
  ahi = ahi < 0x3B80u ? ahi : 0x3B80u;
  uint32_t a0 = alo | (ahi << 16);
  // Shift left by 1 + rounding bias → fp16 E5 to fp8 E4.
  a0 = a0 * 2u + 0x00800080u;
  // Restore sign bits.
  uint32_t b0 = a0 | (in0 & 0x80008000u);
  // Extract upper byte of each 16-bit half → 2 fp8 bytes.
  uint16_t packed = (uint16_t)(((b0 >> 8) & 0xFFu) | ((b0 >> 16) & 0xFF00u));
  return bit_cast<f8_e4m3b15x2>(packed);
#else
  f8_e4m3b15x2 result;
  result.data[0] = __fp8_e4m3b15(v.data[0]);
  result.data[1] = __fp8_e4m3b15(v.data[1]);
  return result;
#endif
}

/// f32x4 -> f8_e4m3b15x4.
/// NVIDIA CUDA: Triton-style vectorized bit manipulation (fp32x4 → fp16x4 → fp8x4).
/// Converts to fp16 via hardware __float2half_rn, clamps abs to max finite 0.9375 (fp16 = 0x3B80),
/// shifts exponent left by 1 (E5→E4) with rounding, then packs via __byte_perm.
template <>
MSCCLPP_DEVICE_INLINE f8_e4m3b15x4 to<f8_e4m3b15x4, f32x4>(const f32x4& v) {
#if defined(MSCCLPP_DEVICE_CUDA)
  // Convert 4 float32 → 2 packed fp16 pairs via __float22half2_rn
  // (single cvt.rn.f16x2.f32 per pair on Blackwell, 2× cvt.rn.f16.f32 on older).
  float2 f01 = {v.data[0], v.data[1]};
  float2 f23 = {v.data[2], v.data[3]};
  __half2 h01 = __float22half2_rn(f01);
  __half2 h23 = __float22half2_rn(f23);
  uint32_t in0, in1;
  asm("mov.b32 %0, %1;" : "=r"(in0) : "r"(*reinterpret_cast<uint32_t*>(&h01)));
  asm("mov.b32 %0, %1;" : "=r"(in1) : "r"(*reinterpret_cast<uint32_t*>(&h23)));

  // Strip sign and clamp abs to max finite e4m3b15: 0x3B80 = 0.9375 in fp16.
  // __vminu2 does packed 2×16-bit unsigned min in one instruction, replacing
  // the split-compare-repack sequence (saves ~14 instructions).
  uint32_t abs0 = in0 & 0x7fff7fffu;
  uint32_t abs1 = in1 & 0x7fff7fffu;
  uint32_t a0 = __vminu2(abs0, 0x3B803B80u);
  uint32_t a1 = __vminu2(abs1, 0x3B803B80u);

  // Shift left by 1, add rounding bias: fp16 E5→fp8 E4 exponent adjustment.
  // Compiler emits mad.lo.u32 (1 instruction).
  a0 = a0 * 2u + 0x00800080u;
  a1 = a1 * 2u + 0x00800080u;

  // Restore sign bits using lop3: b = a | (in & 0x80008000).
  // lop3 truth table 0xf8 = A | (B & C), fuses AND+OR into 1 instruction.
  uint32_t b0, b1;
  asm("lop3.b32 %0, %1, %2, %3, 0xf8;" : "=r"(b0) : "r"(a0), "r"(in0), "r"(0x80008000u));
  asm("lop3.b32 %0, %1, %2, %3, 0xf8;" : "=r"(b1) : "r"(a1), "r"(in1), "r"(0x80008000u));

  // Pack upper byte of each 16-bit half → 4 fp8 bytes.
  uint32_t packed = __byte_perm(b0, b1, 0x7531u);
  return bit_cast<f8_e4m3b15x4>(packed);
#else
  f8_e4m3b15x4 result;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    result.data[i] = __fp8_e4m3b15(v.data[i]);
  }
  return result;
#endif
}

// --- fp8_e4m3b15 <-> f32 decomposed x8/x16 specializations ---
// Decompose into x4 chunks to use the optimized bit-manipulation specializations
// instead of the generic template (which has issues with large VectorType sizes > 16 bytes).

/// f8_e4m3b15x8 -> f32 (8 elements): decompose into 2x f8_e4m3b15x4 -> f32x4.
template <>
MSCCLPP_DEVICE_INLINE f32x8 to<f32x8, f8_e4m3b15x8>(const f8_e4m3b15x8& v) {
  const f8_e4m3b15x4* pair = reinterpret_cast<const f8_e4m3b15x4*>(&v);
  f32x8 result;
  f32x4* out = reinterpret_cast<f32x4*>(&result);
  out[0] = to<f32x4>(pair[0]);
  out[1] = to<f32x4>(pair[1]);
  return result;
}

/// f32 (8 elements) -> f8_e4m3b15x8: decompose into 2x f32x4 -> f8_e4m3b15x4.
template <>
MSCCLPP_DEVICE_INLINE f8_e4m3b15x8 to<f8_e4m3b15x8, f32x8>(const f32x8& v) {
  const f32x4* pair = reinterpret_cast<const f32x4*>(&v);
  f8_e4m3b15x8 result;
  f8_e4m3b15x4* out = reinterpret_cast<f8_e4m3b15x4*>(&result);
  out[0] = to<f8_e4m3b15x4>(pair[0]);
  out[1] = to<f8_e4m3b15x4>(pair[1]);
  return result;
}

/// f8_e4m3b15x16 -> f32 (16 elements): decompose into 4x f8_e4m3b15x4 -> f32x4.
template <>
MSCCLPP_DEVICE_INLINE f32x16 to<f32x16, f8_e4m3b15x16>(const f8_e4m3b15x16& v) {
  const f8_e4m3b15x4* quads = reinterpret_cast<const f8_e4m3b15x4*>(&v);
  f32x16 result;
  f32x4* out = reinterpret_cast<f32x4*>(&result);
  out[0] = to<f32x4>(quads[0]);
  out[1] = to<f32x4>(quads[1]);
  out[2] = to<f32x4>(quads[2]);
  out[3] = to<f32x4>(quads[3]);
  return result;
}

/// f32 (16 elements) -> f8_e4m3b15x16: decompose into 4x f32x4 -> f8_e4m3b15x4.
template <>
MSCCLPP_DEVICE_INLINE f8_e4m3b15x16 to<f8_e4m3b15x16, f32x16>(const f32x16& v) {
  const f32x4* quads = reinterpret_cast<const f32x4*>(&v);
  f8_e4m3b15x16 result;
  f8_e4m3b15x4* out = reinterpret_cast<f8_e4m3b15x4*>(&result);
  out[0] = to<f8_e4m3b15x4>(quads[0]);
  out[1] = to<f8_e4m3b15x4>(quads[1]);
  out[2] = to<f8_e4m3b15x4>(quads[2]);
  out[3] = to<f8_e4m3b15x4>(quads[3]);
  return result;
}

// --- fp8_e4m3b15 <-> fp16 direct conversion specializations ---
// These avoid the fp32 detour: fp8_b15 <-> fp16 is just a 1-bit exponent shift
// (E4 bias=15 <-> E5 bias=15), no precision loss since fp16 has 10 mantissa bits
// vs fp8's 3. Enables fp16 accumulation for allreduce.

/// f8_e4m3b15x2 -> f16x2.
/// Direct fp8 -> fp16 via branch-free bit manipulation (same as the to<f32x2> path
/// but stops at fp16 instead of converting further to fp32).
template <>
MSCCLPP_DEVICE_INLINE f16x2 to<f16x2, f8_e4m3b15x2>(const f8_e4m3b15x2& v) {
#if defined(MSCCLPP_DEVICE_CUDA)
  uint16_t in = v.storage.__x;
  // Spread 2 fp8 bytes into packed fp16 pair, adjust exponent E4->E5.
  uint32_t a0 = ((uint32_t)(in & 0xFFu) << 8) | ((uint32_t)(in >> 8) << 24);
  uint32_t b0 = (a0 & 0x7f007f00u) >> 1;
  uint32_t out0 = b0 | (a0 & 0x80008000u);
  __half2 h;
  asm("mov.b32 %0, %1;" : "=r"(*reinterpret_cast<uint32_t*>(&h)) : "r"(out0));
  return h;
#else
  f16x2 result;
  result.data[0] = __float2half(float(v.data[0]));
  result.data[1] = __float2half(float(v.data[1]));
  return result;
#endif
}

/// f8_e4m3b15x4 -> f16x4.
/// Uses __byte_perm + lop3 for branch-free vectorized conversion.
template <>
MSCCLPP_DEVICE_INLINE f16x4 to<f16x4, f8_e4m3b15x4>(const f8_e4m3b15x4& v) {
#if defined(MSCCLPP_DEVICE_CUDA)
  uint32_t in = v.storage.__x;
  uint32_t a0 = __byte_perm(0u, in, 0x5746u);
  uint32_t a0_shr = a0 >> 1;
  uint32_t a0_sign = a0 & 0x80008000u;
  uint32_t out0;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(out0) : "r"(a0_shr), "r"(0x3f803f80u), "r"(a0_sign));
  uint32_t a1 = __byte_perm(a0, 0u, 0x2301u);
  uint32_t a1_shr = a1 >> 1;
  uint32_t a1_sign = a1 & 0x80008000u;
  uint32_t out1;
  asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(out1) : "r"(a1_shr), "r"(0x3f803f80u), "r"(a1_sign));
  f16x4 result;
  asm("mov.b32 %0, %1;" : "=r"(result.words[0]) : "r"(out0));
  asm("mov.b32 %0, %1;" : "=r"(result.words[1]) : "r"(out1));
  return result;
#else
  f16x4 result;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    result.data[i] = __float2half(float(v.data[i]));
  }
  return result;
#endif
}

/// f16x2 -> f8_e4m3b15x2.
/// Direct fp16 -> fp8 via clamp + exponent shift E5->E4 + pack.
template <>
MSCCLPP_DEVICE_INLINE f8_e4m3b15x2 to<f8_e4m3b15x2, f16x2>(const f16x2& v) {
#if defined(MSCCLPP_DEVICE_CUDA)
  uint32_t in0;
  asm("mov.b32 %0, %1;" : "=r"(in0) : "r"(*reinterpret_cast<const uint32_t*>(&v)));
  // Clamp abs to max finite e4m3b15 (0x3B80 = 0.9375 in fp16).
  uint32_t lo = in0 & 0xFFFFu, hi = in0 >> 16;
  uint32_t alo = lo & 0x7FFFu, ahi = hi & 0x7FFFu;
  alo = alo < 0x3B80u ? alo : 0x3B80u;
  ahi = ahi < 0x3B80u ? ahi : 0x3B80u;
  uint32_t a0 = alo | (ahi << 16);
  a0 = a0 * 2u + 0x00800080u;
  uint32_t b0 = a0 | (in0 & 0x80008000u);
  uint16_t packed = (uint16_t)(((b0 >> 8) & 0xFFu) | ((b0 >> 16) & 0xFF00u));
  return bit_cast<f8_e4m3b15x2>(packed);
#else
  f8_e4m3b15x2 result;
  result.data[0] = __fp8_e4m3b15(__half2float(v.data[0]));
  result.data[1] = __fp8_e4m3b15(__half2float(v.data[1]));
  return result;
#endif
}

/// f16x4 -> f8_e4m3b15x4.
/// Uses __vminu2 + lop3 + __byte_perm for branch-free vectorized conversion.
template <>
MSCCLPP_DEVICE_INLINE f8_e4m3b15x4 to<f8_e4m3b15x4, f16x4>(const f16x4& v) {
#if defined(MSCCLPP_DEVICE_CUDA)
  uint32_t in0, in1;
  asm("mov.b32 %0, %1;" : "=r"(in0) : "r"(v.words[0]));
  asm("mov.b32 %0, %1;" : "=r"(in1) : "r"(v.words[1]));
  uint32_t abs0 = in0 & 0x7fff7fffu;
  uint32_t abs1 = in1 & 0x7fff7fffu;
  uint32_t a0 = __vminu2(abs0, 0x3B803B80u);
  uint32_t a1 = __vminu2(abs1, 0x3B803B80u);
  a0 = a0 * 2u + 0x00800080u;
  a1 = a1 * 2u + 0x00800080u;
  uint32_t b0, b1;
  asm("lop3.b32 %0, %1, %2, %3, 0xf8;" : "=r"(b0) : "r"(a0), "r"(in0), "r"(0x80008000u));
  asm("lop3.b32 %0, %1, %2, %3, 0xf8;" : "=r"(b1) : "r"(a1), "r"(in1), "r"(0x80008000u));
  uint32_t packed = __byte_perm(b0, b1, 0x7531u);
  return bit_cast<f8_e4m3b15x4>(packed);
#else
  f8_e4m3b15x4 result;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    result.data[i] = __fp8_e4m3b15(__half2float(v.data[i]));
  }
  return result;
#endif
}

/// f8_e4m3b15x8 -> f16x8: decompose into 2x f8_e4m3b15x4 -> f16x4.
template <>
MSCCLPP_DEVICE_INLINE f16x8 to<f16x8, f8_e4m3b15x8>(const f8_e4m3b15x8& v) {
  const f8_e4m3b15x4* pair = reinterpret_cast<const f8_e4m3b15x4*>(&v);
  f16x8 result;
  f16x4* out = reinterpret_cast<f16x4*>(&result);
  out[0] = to<f16x4>(pair[0]);
  out[1] = to<f16x4>(pair[1]);
  return result;
}

/// f16x8 -> f8_e4m3b15x8: decompose into 2x f16x4 -> f8_e4m3b15x4.
template <>
MSCCLPP_DEVICE_INLINE f8_e4m3b15x8 to<f8_e4m3b15x8, f16x8>(const f16x8& v) {
  const f16x4* pair = reinterpret_cast<const f16x4*>(&v);
  f8_e4m3b15x8 result;
  f8_e4m3b15x4* out = reinterpret_cast<f8_e4m3b15x4*>(&result);
  out[0] = to<f8_e4m3b15x4>(pair[0]);
  out[1] = to<f8_e4m3b15x4>(pair[1]);
  return result;
}

/// f8_e4m3b15x16 -> f16x16: decompose into 4x f8_e4m3b15x4 -> f16x4.
template <>
MSCCLPP_DEVICE_INLINE f16x16 to<f16x16, f8_e4m3b15x16>(const f8_e4m3b15x16& v) {
  const f8_e4m3b15x4* quads = reinterpret_cast<const f8_e4m3b15x4*>(&v);
  f16x16 result;
  f16x4* out = reinterpret_cast<f16x4*>(&result);
  out[0] = to<f16x4>(quads[0]);
  out[1] = to<f16x4>(quads[1]);
  out[2] = to<f16x4>(quads[2]);
  out[3] = to<f16x4>(quads[3]);
  return result;
}

/// f16x16 -> f8_e4m3b15x16: decompose into 4x f16x4 -> f8_e4m3b15x4.
template <>
MSCCLPP_DEVICE_INLINE f8_e4m3b15x16 to<f8_e4m3b15x16, f16x16>(const f16x16& v) {
  const f16x4* quads = reinterpret_cast<const f16x4*>(&v);
  f8_e4m3b15x16 result;
  f8_e4m3b15x4* out = reinterpret_cast<f8_e4m3b15x4*>(&result);
  out[0] = to<f8_e4m3b15x4>(quads[0]);
  out[1] = to<f8_e4m3b15x4>(quads[1]);
  out[2] = to<f8_e4m3b15x4>(quads[2]);
  out[3] = to<f8_e4m3b15x4>(quads[3]);
  return result;
}

#endif  // MSCCLPP_DEVICE_COMPILE
}  // namespace mscclpp

#endif  // MSCCLPP_GPU_DATA_TYPES_HPP_
