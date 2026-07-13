// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "device_helpers.cuh"

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

}  // namespace ep
}  // namespace mscclpp
