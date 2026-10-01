// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_QUANTIZATION_CUH_
#define MSCCLPP_EXT_MEGAMOE_QUANTIZATION_CUH_

#include "megamoe_device.cuh"

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

__device__ __forceinline__ uint8_t quantizeE8M0Scale(float maximum) {
  if (maximum == 0.0f) return 127;
  uint32_t bits = mscclpp::bit_cast<uint32_t>(maximum);
  uint32_t exponent = (bits >> 23) & 0xff;
  if (exponent == 0xff) return 254;
  // 448 = 1.75 * 2^8; compare the significand directly instead of dividing.
  int scale = int(exponent) - 8 + int((bits & 0x7fffff) > 0x600000);
  return uint8_t(max(1, min(scale, 254)));
}

__device__ __forceinline__ float inverseE8M0Scale(uint8_t scale) {
  // The reciprocal of 2^127 is a representable FP32 subnormal, not zero.
  return __uint_as_float(scale == 254 ? 0x00400000u : uint32_t(254 - scale) << 23);
}

__device__ __forceinline__ uint16_t quantizeE4M3Pair(float first, float second, float inverse) {
  mscclpp::f32x2 values;
  values.data[0] = first * inverse;
  values.data[1] = second * inverse;
  return mscclpp::bit_cast<uint16_t>(mscclpp::to<mscclpp::f8_e4m3x2>(values));
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_QUANTIZATION_CUH_
