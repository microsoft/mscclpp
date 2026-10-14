// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_COMMON_QUANTIZATION_CUH_
#define MSCCLPP_EP_COMMON_QUANTIZATION_CUH_

#include <mscclpp/gpu_data_types.hpp>

#include "device_helpers.cuh"

namespace mscclpp {
namespace ep {

inline constexpr float Fp8E4M3MaxValue = 448.0f;

/// MXFP8 E8M0 block scale: the smallest power of two that maps the block maximum to at most 448 (the E4M3 maximum).
MSCCLPP_DEVICE_INLINE uint8_t quantizeE8M0Scale(float maximum) {
  if (maximum == 0.0f) return 127;
  const uint32_t bits = mscclpp::bit_cast<uint32_t>(maximum);
  const uint32_t exponent = (bits >> 23) & 0xff;
  if (exponent == 0xff) return 254;
  // 448 = 1.75 * 2^8; compare the significand directly instead of dividing.
  const int scale = int(exponent) - 8 + int((bits & 0x7fffff) > 0x600000);
  return uint8_t(scale < 1 ? 1 : (scale > 254 ? 254 : scale));
}

/// Reciprocal of an E8M0 block scale, applied to values before E4M3 conversion.
MSCCLPP_DEVICE_INLINE float inverseE8M0Scale(uint8_t scale) {
  // The reciprocal of 2^127 is a representable FP32 subnormal, not zero.
  return mscclpp::bit_cast<float>(scale == 254 ? 0x00400000u : uint32_t(254 - scale) << 23);
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

MSCCLPP_DEVICE_INLINE mscclpp::f8_e4m3x8 quantizeBf16x8ToMxfp8(const mscclpp::bf16x8& source, uint8_t& scale,
                                                               int laneId) {
  const mscclpp::f32x8 values = mscclpp::to<mscclpp::f32x8>(source);
  const float maximum = laneGroupMax<4>(maxAbsF32x8(values, 0.0f), laneId);
  scale = quantizeE8M0Scale(maximum);
  const float inverse = inverseE8M0Scale(scale);
  // Scale f32x2 pairs: Blackwell multiplies each pair with one packed instruction.
  const mscclpp::f32x2 inverses = make_float2(inverse, inverse);
  const auto* pairs = reinterpret_cast<const mscclpp::f32x2*>(&values);
  mscclpp::f32x8 scaled;
  auto* scaledPairs = reinterpret_cast<mscclpp::f32x2*>(&scaled);
#pragma unroll
  for (int pair = 0; pair < mscclpp::f32x8::Size / 2; ++pair) {
    scaledPairs[pair] = pairs[pair] * inverses;
  }
  return mscclpp::to<mscclpp::f8_e4m3x8>(scaled);
}

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_COMMON_QUANTIZATION_CUH_
