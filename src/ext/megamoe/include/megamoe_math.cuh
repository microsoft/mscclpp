// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_EXT_MEGAMOE_MATH_CUH_
#define MSCCLPP_EXT_MEGAMOE_MATH_CUH_

#include <cstdint>
#include <mscclpp/device.hpp>
#include <mscclpp/gpu_data_types.hpp>

#ifndef WARP_SIZE
#if defined(__HIP_PLATFORM_AMD__)
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif
#endif

namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail {

MSCCLPP_DEVICE_INLINE float2 swigluPair(float2 gate, float2 up) {
  float2 product = __fmul2_rn(gate, up);
  float2 exponent = __fmul2_rn(gate, make_float2(-1.4426950408889634f, -1.4426950408889634f));
  float2 denominator;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(denominator.x) : "f"(exponent.x));
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(denominator.y) : "f"(exponent.y));
  denominator = __fadd2_rn(denominator, make_float2(1.0f, 1.0f));
  float2 inverse;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inverse.x) : "f"(denominator.x));
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(inverse.y) : "f"(denominator.y));
  return __fmul2_rn(product, inverse);
}

// Exchanges half of the gate (lanes 0..15) and up (lanes 16..31) values intra-warp across WARP_SIZE / 2 so each
// half-warp holds complete (gate, up) pairs for HalfTokens tokens.
template <int HalfTokens, class Values>
MSCCLPP_DEVICE_INLINE void exchangeGateUpHalfIntraWarp(const Values& values, bool isUpHalf, float (&gate)[HalfTokens],
                                                       float (&up)[HalfTokens]) {
  static_assert(WARP_SIZE == 32);
#pragma unroll
  for (int i = 0; i < HalfTokens; ++i) {
    float sentValue = isUpHalf ? values(i) : values(i + HalfTokens);
    float peerValue = __shfl_xor_sync(0xffffffff, sentValue, WARP_SIZE / 2);
    gate[i] = isUpHalf ? peerValue : values(i);
    up[i] = isUpHalf ? values(i + HalfTokens) : peerValue;
  }
}

template <int HalfTokens>
MSCCLPP_DEVICE_INLINE void clampGateUpArray(float (&gate)[HalfTokens], float (&up)[HalfTokens], float clampLimit) {
  if (clampLimit < 0.0f) return;
#pragma unroll
  for (int i = 0; i < HalfTokens; ++i) {
    gate[i] = fminf(gate[i], clampLimit);
    up[i] = fmaxf(-clampLimit, fminf(up[i], clampLimit));
  }
}

// Computes the maximum absolute value across `values` and across each `ThreadsPerGroup`-lane subgroup within a warp.
// All 32 lanes of the warp must execute this convergently.
template <int ThreadsPerGroup, class Activations>
MSCCLPP_DEVICE_INLINE float warpSubgroupMaxAbs(const Activations& values) {
  constexpr int Features = Activations::Size;
  Activations magnitude;
#pragma unroll
  for (int i = 0; i < Features; ++i) magnitude[i] = fabsf(values[i]);
#pragma unroll
  for (int width = Features / 2; width > 0; width /= 2) {
#pragma unroll
    for (int i = 0; i < width; ++i) magnitude[i] = fmaxf(magnitude[i], magnitude[i + width]);
  }
  float maximum = magnitude[0];
#pragma unroll
  for (int offset = ThreadsPerGroup / 2; offset > 0; offset /= 2) {
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
  }
  return maximum;
}

// Transposes four lanes' token-pair FP8 values (lanes l, l+4, l+8, l+12) into two 4-feature uint32_t stores.
MSCCLPP_DEVICE_INLINE void storeTransposedFp8Quad(uint8_t* destination, int rowStride, int colBase, int laneIdx,
                                                  uint16_t fp8Pair) {
  uint32_t pair16 = (uint32_t(fp8Pair & 0xffu) | (uint32_t(fp8Pair & 0xff00u) << 8)) << ((laneIdx & 4) ? 8 : 0);
  pair16 |= __shfl_xor_sync(0xffffffff, pair16, 4);
  uint32_t quad0 = uint32_t(pair16 & 0xffffu) << ((laneIdx & 8) ? 16 : 0);
  uint32_t quad1 = uint32_t(pair16 >> 16) << ((laneIdx & 8) ? 16 : 0);
  quad0 |= __shfl_xor_sync(0xffffffff, quad0, 8);
  quad1 |= __shfl_xor_sync(0xffffffff, quad1, 8);
  if ((laneIdx & 12) == 0) {
    int col = colBase + ((laneIdx >> 4) * 4);
    *reinterpret_cast<uint32_t*>(destination + col) = quad0;
    *reinterpret_cast<uint32_t*>(destination + rowStride + col) = quad1;
  }
}

MSCCLPP_DEVICE_INLINE void zeroFloat8(mscclpp::f32x8& values) {
#pragma unroll
  for (int element = 0; element < mscclpp::f32x8::Size; ++element) values.data[element] = 0.0f;
}

template <bool Weighted>
MSCCLPP_DEVICE_INLINE void accumulateWeightedFloat8(mscclpp::f32x8& sum, const mscclpp::f32x8& values, float weight) {
#pragma unroll
  for (int element = 0; element < mscclpp::f32x8::Size; ++element) {
    sum.data[element] = Weighted ? __fmaf_rn(values.data[element], weight, sum.data[element])
                                 : sum.data[element] + values.data[element];
  }
}

MSCCLPP_DEVICE_INLINE void scaleFloat8(mscclpp::f32x8& values, float scale) {
#pragma unroll
  for (int element = 0; element < mscclpp::f32x8::Size; ++element) values.data[element] *= scale;
}

}  // namespace MSCCLPP_MEGAMOE_KERNEL_NAMESPACE::detail

#endif  // MSCCLPP_EXT_MEGAMOE_MATH_CUH_
