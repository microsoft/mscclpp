// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <mscclpp/gpu_data_types.hpp>

#include "device_helpers.cuh"

namespace mscclpp {
namespace ep {

inline constexpr float Fp8E4M3MaxValue = 448.0f;

MSCCLPP_DEVICE_INLINE uint8_t encodeUe8m0RoundUp(float value) {
  if (!(value > 0.0f)) return 0;

#if CUDART_VERSION >= 12080
  return __nv_cvt_float_to_e8m0(value, __NV_SATFINITE, cudaRoundPosInf);
#else
  const uint32_t bits = __float_as_uint(value);
  uint32_t exponent = (bits >> 23) & 0xffu;
  const uint32_t mantissa = bits & 0x7fffffu;
  if (exponent >= 254) return 254;
  if (mantissa != 0 && !(exponent == 0 && mantissa <= 0x400000u)) ++exponent;
  return static_cast<uint8_t>(exponent);
#endif
}

MSCCLPP_DEVICE_INLINE float maxAbsF32x8(const mscclpp::f32x8& values, float seed) {
  float maxAbs = seed;
#pragma unroll
  for (int element = 0; element < mscclpp::f32x8::Size; ++element) {
    maxAbs = fmaxf(maxAbs, fabsf(values.data[element]));
  }
  return maxAbs;
}

template <int NumLanes>
MSCCLPP_DEVICE_INLINE float laneGroupMax(float value, int laneId) {
  EP_STATIC_ASSERT(NumLanes > 0 && NumLanes <= WARP_SIZE, "Invalid lane group size");
  EP_STATIC_ASSERT((NumLanes & (NumLanes - 1)) == 0, "Lane group size must be a power of two");

  unsigned int mask;
  if constexpr (NumLanes == WARP_SIZE) {
    mask = 0xffffffffu;
  } else {
    const int groupStart = laneId - laneId % NumLanes;
    mask = ((1u << NumLanes) - 1u) << groupStart;
  }

#pragma unroll
  for (int offset = NumLanes / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(mask, value, offset));
  }
  return value;
}

template <int NumElementsPerScale>
MSCCLPP_DEVICE_INLINE mscclpp::f8_e4m3x8 quantizeBf16x8ToFp8E4M3(const mscclpp::bf16x8& source, float* scaleOut,
                                                                 int laneId) {
  constexpr int NumElements = mscclpp::bf16x8::Size;
  constexpr int NumLanesPerScale = NumElementsPerScale / NumElements;
  constexpr float Margin = 1e-4f;

  EP_STATIC_ASSERT(NumElementsPerScale % NumElements == 0, "Invalid scale vectorization");
  EP_STATIC_ASSERT(NumLanesPerScale > 0 && NumLanesPerScale <= WARP_SIZE, "Invalid lanes per scale");
  EP_STATIC_ASSERT((NumLanesPerScale & (NumLanesPerScale - 1)) == 0, "Lanes per scale must be a power of two");

  const mscclpp::f32x8 values = mscclpp::to<mscclpp::f32x8>(source);
  float maxAbs = maxAbsF32x8(values, Margin);

  maxAbs = laneGroupMax<NumLanesPerScale>(maxAbs, laneId);
  const float quantScale = Fp8E4M3MaxValue / maxAbs;
  if (laneId % NumLanesPerScale == 0) {
    *scaleOut = maxAbs / Fp8E4M3MaxValue;
  }

  mscclpp::f32x8 scaledValues;
#pragma unroll
  for (int element = 0; element < NumElements; ++element) {
    scaledValues.data[element] = values.data[element] * quantScale;
  }
  return mscclpp::to<mscclpp::f8_e4m3x8>(scaledValues);
}

template <int NumElementsPerScale>
MSCCLPP_DEVICE_INLINE mscclpp::f8_e4m3x8 quantizeBf16x8ToMxFp8E4M3(const mscclpp::bf16x8& source, uint8_t* scaleOut,
                                                                   int laneId) {
  constexpr int NumElements = mscclpp::bf16x8::Size;
  constexpr int NumLanesPerScale = NumElementsPerScale / NumElements;

  EP_STATIC_ASSERT(NumElementsPerScale % NumElements == 0, "Invalid scale vectorization");
  EP_STATIC_ASSERT(NumLanesPerScale > 0 && NumLanesPerScale <= WARP_SIZE, "Invalid lanes per scale");
  EP_STATIC_ASSERT((NumLanesPerScale & (NumLanesPerScale - 1)) == 0, "Lanes per scale must be a power of two");

  const mscclpp::f32x8 values = mscclpp::to<mscclpp::f32x8>(source);
  float maxAbs = maxAbsF32x8(values, 0.0f);

  maxAbs = laneGroupMax<NumLanesPerScale>(maxAbs, laneId);
  const float normalizedMax = maxAbs * reciprocalApproximateFtz(Fp8E4M3MaxValue);
  const uint8_t scale = encodeUe8m0RoundUp(normalizedMax);
  if (laneId % NumLanesPerScale == 0) {
    *scaleOut = scale;
  }

  const float decodedScale = __uint_as_float(static_cast<uint32_t>(scale) << 23);
  const float inverseScale = scale == 0 ? 1.0f : reciprocalApproximateFtz(decodedScale);
  mscclpp::f32x8 scaledValues;
#pragma unroll
  for (int element = 0; element < NumElements; ++element) {
    scaledValues.data[element] = values.data[element] * inverseScale;
  }
  return mscclpp::to<mscclpp::f8_e4m3x8>(scaledValues);
}

MSCCLPP_DEVICE_INLINE float dequantizeFp8E4M3(typename mscclpp::f8_e4m3x2::ElementType value, float scale) {
  mscclpp::f8_e4m3x2 packed;
  packed.data[0] = value;
  packed.data[1] = value;
  return mscclpp::to<mscclpp::f32x2>(packed).data[0] * scale;
}

}  // namespace ep
}  // namespace mscclpp
