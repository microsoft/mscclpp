// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "utils.cuh"

namespace mscclpp {
namespace ep {

template <__nv_fp8_interpretation_t kFp8Format>
struct Fp8FormatTraits {};

template <>
struct Fp8FormatTraits<__NV_E4M3> {
  static constexpr float maxValue = 448.0f;
};

template <>
struct Fp8FormatTraits<__NV_E5M2> {
  static constexpr float maxValue = 57344.0f;
};

template <int kBytes>
struct Fp8PackedVector {
  using Type = typename VecInt<kBytes>::vec_t;
};

template <>
struct Fp8PackedVector<8> {
  using Type = int2;
};

template <typename SourceType>
struct Fp8VectorType {
  static_assert(sizeof(int4) % sizeof(SourceType) == 0, "Source type must divide int4");
  static constexpr int numElems = sizeof(int4) / sizeof(SourceType);
  using Type = typename Fp8PackedVector<numElems * sizeof(__nv_fp8_storage_t)>::Type;
};

template <int kNumLanes>
__device__ __forceinline__ float laneGroupReduceMax(float value, int laneId) {
  EP_STATIC_ASSERT(kNumLanes > 0 && kNumLanes <= 32, "Invalid lane group size");
  EP_STATIC_ASSERT((kNumLanes & (kNumLanes - 1)) == 0, "Lane group size must be a power of two");

  unsigned int mask;
  if constexpr (kNumLanes == 32) {
    mask = 0xffffffffu;
  } else {
    const int groupStart = laneId - laneId % kNumLanes;
    mask = ((1u << kNumLanes) - 1u) << groupStart;
  }

#pragma unroll
  for (int offset = kNumLanes / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(mask, value, offset));
  }
  return value;
}

template <typename SourceType, int kNumElemsPerScale, __nv_fp8_interpretation_t kFp8Format = __NV_E4M3>
__device__ __forceinline__ typename Fp8VectorType<SourceType>::Type quantizeToFp8(const int4& sourceValue,
                                                                                  float* scaleOut, int laneId) {
  constexpr int kNumElemsPerRead = Fp8VectorType<SourceType>::numElems;
  constexpr int kNumLanesPerScale = kNumElemsPerScale / kNumElemsPerRead;
  constexpr float kFp8Margin = 1e-4f;
  using OutputVec = typename Fp8VectorType<SourceType>::Type;

  EP_STATIC_ASSERT(kNumElemsPerRead % 2 == 0, "FP8 conversion packs pairs of elements");
  EP_STATIC_ASSERT(kNumElemsPerScale % kNumElemsPerRead == 0, "Invalid scale vectorization");
  EP_STATIC_ASSERT(kNumLanesPerScale > 0 && kNumLanesPerScale <= 32, "Invalid lanes per scale");
  EP_STATIC_ASSERT((kNumLanesPerScale & (kNumLanesPerScale - 1)) == 0, "Lanes per scale must be a power of two");

  auto sourceValues = reinterpret_cast<const SourceType*>(&sourceValue);
  float fp32Values[kNumElemsPerRead];
  float amax = kFp8Margin;

#pragma unroll
  for (int j = 0; j < kNumElemsPerRead; ++j) {
    fp32Values[j] = static_cast<float>(sourceValues[j]);
    amax = fmaxf(amax, fabsf(fp32Values[j]));
  }

  amax = laneGroupReduceMax<kNumLanesPerScale>(amax, laneId);
  const float scale = Fp8FormatTraits<kFp8Format>::maxValue / amax;
  const float scaleInv = amax / Fp8FormatTraits<kFp8Format>::maxValue;

  if (laneId % kNumLanesPerScale == 0) {
    *scaleOut = scaleInv;
  }

  OutputVec outputValue;
  auto fp8x2Values = reinterpret_cast<__nv_fp8x2_storage_t*>(&outputValue);
#pragma unroll
  for (int j = 0; j < kNumElemsPerRead; j += 2) {
    float2 fp32x2 = {fp32Values[j] * scale, fp32Values[j + 1] * scale};
    fp8x2Values[j / 2] = __nv_cvt_float2_to_fp8x2(fp32x2, __NV_SATFINITE, kFp8Format);
  }

  return outputValue;
}

template <__nv_fp8_interpretation_t kFp8Format = __NV_E4M3>
__device__ __forceinline__ float dequantizeFp8(__nv_fp8_storage_t value, float scale) {
  return static_cast<float>(__half(__nv_cvt_fp8_to_halfraw(value, kFp8Format))) * scale;
}

// MXFP8-style micro-scaled quantization: one E8M0 (8-bit exponent-only,
// power-of-two, bias 127) scale per ``kNumElemsPerScale`` block (32 for OCP
// MXFP8) with FP8-E4M3 elements. The shared scale is derived self-consistently
// from the block amax: pick the smallest power of two ``P = 2^exp`` such that
// ``amax / P <= maxValue`` (i.e. ``exp = ceil(log2(amax / maxValue))``), encode
// the biased exponent as the E8M0 byte, quantize with ``2^-exp`` and dequantize
// with ``2^exp``. This keeps the quant/dequant pair exact to FP8 precision while
// emitting a valid E8M0 micro-scale.
template <typename SourceType, int kNumElemsPerScale, __nv_fp8_interpretation_t kFp8Format = __NV_E4M3>
__device__ __forceinline__ typename Fp8VectorType<SourceType>::Type quantizeToMxFp8(const int4& sourceValue,
                                                                                    uint8_t* scaleOut, int laneId) {
  constexpr int kNumElemsPerRead = Fp8VectorType<SourceType>::numElems;
  constexpr int kNumLanesPerScale = kNumElemsPerScale / kNumElemsPerRead;
  constexpr float kFp8Margin = 1e-4f;
  using OutputVec = typename Fp8VectorType<SourceType>::Type;

  EP_STATIC_ASSERT(kNumElemsPerRead % 2 == 0, "FP8 conversion packs pairs of elements");
  EP_STATIC_ASSERT(kNumElemsPerScale % kNumElemsPerRead == 0, "Invalid scale vectorization");
  EP_STATIC_ASSERT(kNumLanesPerScale > 0 && kNumLanesPerScale <= 32, "Invalid lanes per scale");
  EP_STATIC_ASSERT((kNumLanesPerScale & (kNumLanesPerScale - 1)) == 0, "Lanes per scale must be a power of two");

  auto sourceValues = reinterpret_cast<const SourceType*>(&sourceValue);
  float fp32Values[kNumElemsPerRead];
  float amax = kFp8Margin;

#pragma unroll
  for (int j = 0; j < kNumElemsPerRead; ++j) {
    fp32Values[j] = static_cast<float>(sourceValues[j]);
    amax = fmaxf(amax, fabsf(fp32Values[j]));
  }

  amax = laneGroupReduceMax<kNumLanesPerScale>(amax, laneId);

  // Smallest power of two whose reciprocal keeps the block within FP8 range.
  const float scaleInv = amax / Fp8FormatTraits<kFp8Format>::maxValue;
  int exponent = static_cast<int>(ceilf(log2f(scaleInv)));
  // Clamp to the representable E8M0 exponent range [-127, 127] (biased
  // [0, 254]); the biased value 255 is reserved for NaN.
  exponent = exponent < -127 ? -127 : (exponent > 127 ? 127 : exponent);
  const float scale = exp2f(static_cast<float>(-exponent));  // 2^-exp, multiply to quantize

  if (laneId % kNumLanesPerScale == 0) {
    *scaleOut = static_cast<uint8_t>(exponent + 127);
  }

  OutputVec outputValue;
  auto fp8x2Values = reinterpret_cast<__nv_fp8x2_storage_t*>(&outputValue);
#pragma unroll
  for (int j = 0; j < kNumElemsPerRead; j += 2) {
    float2 fp32x2 = {fp32Values[j] * scale, fp32Values[j + 1] * scale};
    fp8x2Values[j / 2] = __nv_cvt_float2_to_fp8x2(fp32x2, __NV_SATFINITE, kFp8Format);
  }

  return outputValue;
}

// Dequantize an FP8 element using an E8M0 (power-of-two, bias 127) micro-scale.
template <__nv_fp8_interpretation_t kFp8Format = __NV_E4M3>
__device__ __forceinline__ float dequantizeMxFp8(__nv_fp8_storage_t value, uint8_t e8m0Scale) {
  const float scale = exp2f(static_cast<float>(static_cast<int>(e8m0Scale) - 127));
  return static_cast<float>(__half(__nv_cvt_fp8_to_halfraw(value, kFp8Format))) * scale;
}

}  // namespace ep
}  // namespace mscclpp
