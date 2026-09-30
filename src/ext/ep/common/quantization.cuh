// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_COMMON_QUANTIZATION_CUH_
#define MSCCLPP_EP_COMMON_QUANTIZATION_CUH_

#include <cstdint>
#include <mscclpp/gpu_data_types.hpp>

#include "device_helpers.cuh"

namespace mscclpp {
namespace ep {

inline constexpr float Fp8E4M3MaxValue = 448.0f;

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
MSCCLPP_DEVICE_INLINE mscclpp::f8_e4m3x8 quantizeBf16x8ToFp8E4M3(const mscclpp::bf16x8& source, uint8_t* scaleOut,
                                                                 int laneId) {
  constexpr int NumElements = mscclpp::bf16x8::Size;
  constexpr int NumLanesPerScale = NumElementsPerScale / NumElements;

  EP_STATIC_ASSERT(NumElementsPerScale % NumElements == 0, "Invalid scale vectorization");
  EP_STATIC_ASSERT(NumLanesPerScale > 0 && NumLanesPerScale <= WARP_SIZE, "Invalid lanes per scale");
  EP_STATIC_ASSERT((NumLanesPerScale & (NumLanesPerScale - 1)) == 0, "Lanes per scale must be a power of two");

  const mscclpp::f32x8 values = mscclpp::to<mscclpp::f32x8>(source);
  float maxAbs = maxAbsF32x8(values, 0.0f);

  maxAbs = laneGroupMax<NumLanesPerScale>(maxAbs, laneId);
  const float dequantScale = maxAbs / Fp8E4M3MaxValue;
  const uint32_t roundedScaleBits = (__float_as_uint(dequantScale) + 0x007fffffu) & 0x7f800000u;
  const float roundedScale = __uint_as_float(roundedScaleBits);
  const float quantScale = roundedScaleBits == 0 ? 0.0f : 1.0f / roundedScale;
  if (laneId % NumLanesPerScale == 0) {
    *scaleOut = static_cast<uint8_t>(roundedScaleBits >> 23);
  }

  mscclpp::f32x8 scaledValues;
#pragma unroll
  for (int element = 0; element < NumElements; ++element) {
    scaledValues.data[element] = values.data[element] * quantScale;
  }
  return mscclpp::to<mscclpp::f8_e4m3x8>(scaledValues);
}

MSCCLPP_DEVICE_INLINE float decodeE8M0(uint8_t scale) { return __uint_as_float(static_cast<uint32_t>(scale) << 23); }

MSCCLPP_DEVICE_INLINE float dequantizeFp8E4M3(typename mscclpp::f8_e4m3x2::ElementType value, uint8_t scale) {
  mscclpp::f8_e4m3x2 packed;
  packed.data[0] = value;
  packed.data[1] = value;
  return mscclpp::to<mscclpp::f32x2>(packed).data[0] * decodeE8M0(scale);
}

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_COMMON_QUANTIZATION_CUH_
