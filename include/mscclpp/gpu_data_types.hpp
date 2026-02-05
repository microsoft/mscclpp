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
  INT32,     // 32-bit signed integer.
  UINT32,    // 32-bit unsigned integer.
  FLOAT16,   // IEEE 754 half precision.
  FLOAT32,   // IEEE 754 single precision.
  BFLOAT16,  // bfloat16 precision.
  FP8_E4M3,  // FP8 with E4M3 layout.
  FP8_E5M2,  // FP8 with E5M2 layout.
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
DEFINE_VEC(f32x2, float, 2, float2);
DEFINE_VEC(f16x2, __half, 2, __half2);
DEFINE_VEC(bf16x2, __bfloat16, 2, __bfloat162);

DEFINE_VEC(i32x4, int32_t, 4, int4);
DEFINE_VEC(u32x4, uint32_t, 4, uint4);
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
}  // namespace mscclpp

#endif  // MSCCLPP_GPU_DATA_TYPES_HPP_
