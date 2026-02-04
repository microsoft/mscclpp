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

// AMD FP8 support - Use fnuz types for HIP 6.0 or when HIP_FP8_TYPE_FNUZ is enabled and HIP_FP8_TYPE_OCP is not enabled.
// Otherwise, use the standard AMD FP8 types.
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
template <int Bytes>
struct alignas(Bytes) Words {
  static_assert(Bytes > 0, "Bytes must be greater than 0");
  static_assert(Bytes % 4 == 0, "Bytes must be multiple of 4");
  uint32_t w[Bytes / 4];

  MSCCLPP_HOST_DEVICE_INLINE Words() {}

  MSCCLPP_HOST_DEVICE_INLINE uint32_t& operator[](int i) { return w[i]; }

  MSCCLPP_HOST_DEVICE_INLINE const uint32_t& operator[](int i) const { return w[i]; }
};

/// Vector type.
template <typename T, int N>
union alignas(sizeof(T) * N) VectorType {
  static_assert(N > 0, "N must be greater than 0");

  T data[N];
  Words<sizeof(T) * N> words;

  using ElementType = T;
  constexpr static int Size = N;

  MSCCLPP_HOST_DEVICE_INLINE VectorType() {}

  MSCCLPP_HOST_DEVICE_INLINE operator T*() { return data; }

  MSCCLPP_HOST_DEVICE_INLINE operator const T*() const { return data; }

  MSCCLPP_HOST_DEVICE_INLINE T& operator[](int i) { return data[i]; }

  MSCCLPP_HOST_DEVICE_INLINE const T& operator[](int i) const { return data[i]; }
};

using i32x1 = VectorType<int32_t, 1>;
using u32x1 = VectorType<uint32_t, 1>;
using f64x1 = VectorType<double, 1>;
using f32x1 = VectorType<float, 1>;

using i32x2 = VectorType<int32_t, 2>;
using u32x2 = VectorType<uint32_t, 2>;
using f32x2 = VectorType<float, 2>;
using f16x2 = VectorType<__half, 2>;
using bf16x2 = VectorType<__bfloat16, 2>;

using i32x4 = VectorType<int32_t, 4>;
using u32x4 = VectorType<uint32_t, 4>;
using f32x4 = VectorType<float, 4>;
using f16x4 = VectorType<__half, 4>;
using bf16x4 = VectorType<__bfloat16, 4>;

using f16x8 = VectorType<__half, 8>;
using bf16x8 = VectorType<__bfloat16, 8>;

#if defined(__FP8_TYPES_EXIST__)

using fp8_e4m3x4 = VectorType<__fp8_e4m3, 4>;
using fp8_e4m3x8 = VectorType<__fp8_e4m3, 8>;
using fp8_e4m3x16 = VectorType<__fp8_e4m3, 16>;

using fp8_e5m2x4 = VectorType<__fp8_e5m2, 4>;
using fp8_e5m2x8 = VectorType<__fp8_e5m2, 8>;
using fp8_e5m2x16 = VectorType<__fp8_e5m2, 16>;
#endif

}  // namespace mscclpp

#endif  // MSCCLPP_GPU_DATA_TYPES_HPP_
