// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_GPU_DATA_TYPES_HPP_
#define MSCCLPP_GPU_DATA_TYPES_HPP_

#if defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
// AMD FP8 support
#if defined(__HIP_FP8_TYPES_EXIST__)
#include <hip/hip_fp8.h>
#endif

using __bfloat16 = __hip_bfloat16;
using __bfloat162 = __hip_bfloat162;
#define __CUDA_BF16_TYPES_EXIST__

// Define FP8 types for AMD
#if defined(__HIP_FP8_TYPES_EXIST__)
using __fp8_e4m3 = __hip_fp8_e4m3_fnuz;
using __fp8_e5m2 = __hip_fp8_e5m2_fnuz;
#define __CUDA_FP8_TYPES_EXIST__
#endif

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
#define __CUDA_FP8_TYPES_EXIST__
#endif

using __bfloat16 = __nv_bfloat16;
using __bfloat162 = __nv_bfloat162;

#endif

#include <mscclpp/device.hpp>

namespace mscclpp {

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

#if defined(__CUDA_FP8_TYPES_EXIST__)
// FP8 vector types
using fp8_e4m3x2 = VectorType<__fp8_e4m3, 2>;
using fp8_e4m3x4 = VectorType<__fp8_e4m3, 4>;
using fp8_e4m3x8 = VectorType<__fp8_e4m3, 8>;
using fp8_e5m2x2 = VectorType<__fp8_e5m2, 2>;
using fp8_e5m2x4 = VectorType<__fp8_e5m2, 4>;
using fp8_e5m2x8 = VectorType<__fp8_e5m2, 8>;
#endif

}  // namespace mscclpp

#endif  // MSCCLPP_GPU_DATA_TYPES_HPP_
