// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mscclpp/device.hpp>
#include <mscclpp/gpu_data_types.hpp>
#include <mscclpp/packet_device.hpp>
#include <type_traits>

#include "constants.cuh"

namespace mscclpp {
namespace ep {

template <typename dtype_t>
MSCCLPP_HOST_DEVICE_INLINE constexpr dtype_t configCellDiv(dtype_t a, dtype_t b) {
  return (a + b - 1) / b;
}

template <typename dtype_t>
MSCCLPP_HOST_DEVICE_INLINE constexpr dtype_t configAlign(dtype_t a, dtype_t b) {
  return configCellDiv<dtype_t>(a, b) * b;
}

namespace low_latency {

using Bf16 = typename mscclpp::bf16x2::ElementType;
using Fp8E4M3 = typename mscclpp::f8_e4m3x2::ElementType;

// Rank-deduplicated dispatch payload layout:
//
//   [data: DataType[hidden]]
//   [optional scales: ScaleType[hidden / format scale block size]]
//   [topKIndices: int[topK]]
//   [topKValues: float[topK]]
//   [srcTokenGlobalIdx: int]
//
// The payload is 32-byte aligned as a whole.
template <typename DataType, typename ScaleType = void>
struct PayloadView {
  static constexpr bool HasScales = !std::is_void_v<ScaleType>;

  int topK_;
  size_t scaleOffset_;
  size_t metadataOffset_;
  size_t numBytes_;

  MSCCLPP_HOST_DEVICE_INLINE static int numScales([[maybe_unused]] int hidden, [[maybe_unused]] int scaleBlockSize) {
    if constexpr (HasScales) {
      return hidden / scaleBlockSize;
    }
    return 0;
  }

  MSCCLPP_HOST_DEVICE_INLINE static size_t hiddenBytes(int hidden) {
    return static_cast<size_t>(hidden) * sizeof(DataType);
  }

  MSCCLPP_HOST_DEVICE_INLINE static size_t scaleOffset(int hidden) {
    if constexpr (HasScales) {
      return configAlign<size_t>(hiddenBytes(hidden), alignof(ScaleType));
    }
    return 0;
  }

  MSCCLPP_HOST_DEVICE_INLINE static size_t scaleBytes([[maybe_unused]] int hidden,
                                                      [[maybe_unused]] int scaleBlockSize) {
    if constexpr (HasScales) {
      return static_cast<size_t>(numScales(hidden, scaleBlockSize)) * sizeof(ScaleType);
    }
    return 0;
  }

  MSCCLPP_HOST_DEVICE_INLINE static size_t metadataOffset(int hidden, int scaleBlockSize) {
    if constexpr (HasScales) {
      return configAlign<size_t>(scaleOffset(hidden) + scaleBytes(hidden, scaleBlockSize), alignof(int));
    }
    return configAlign<size_t>(hiddenBytes(hidden), alignof(int));
  }

  MSCCLPP_HOST_DEVICE_INLINE static size_t metadataBytes(int topK) {
    return static_cast<size_t>(topK) * sizeof(int) + static_cast<size_t>(topK) * sizeof(float) + sizeof(int);
  }

  MSCCLPP_HOST_DEVICE_INLINE static size_t numBytes(int hidden, int topK, int scaleBlockSize) {
    return configAlign<size_t>(metadataOffset(hidden, scaleBlockSize) + metadataBytes(topK), 32);
  }

  MSCCLPP_HOST_DEVICE_INLINE PayloadView(int hidden, int topK, int scaleBlockSize = (HasScales ? 128 : 0))
      : topK_(topK),
        scaleOffset_(scaleOffset(hidden)),
        metadataOffset_(metadataOffset(hidden, scaleBlockSize)),
        numBytes_(numBytes(hidden, topK, scaleBlockSize)) {}

  template <typename T>
  MSCCLPP_HOST_DEVICE_INLINE T* data(void* base) const {
    return reinterpret_cast<T*>(base);
  }

  MSCCLPP_HOST_DEVICE_INLINE ScaleType* scaleFactors(void* base) const {
    static_assert(HasScales, "Payload has no scale factors");
    return reinterpret_cast<ScaleType*>(reinterpret_cast<uint8_t*>(base) + scaleOffset_);
  }

  MSCCLPP_HOST_DEVICE_INLINE const ScaleType* scaleFactors(const void* base) const {
    static_assert(HasScales, "Payload has no scale factors");
    return reinterpret_cast<const ScaleType*>(reinterpret_cast<const uint8_t*>(base) + scaleOffset_);
  }

  MSCCLPP_HOST_DEVICE_INLINE int* topKIndices(void* base) const {
    return reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(base) + metadataOffset_);
  }

  MSCCLPP_HOST_DEVICE_INLINE const int* topKIndices(const void* base) const {
    return reinterpret_cast<const int*>(reinterpret_cast<const uint8_t*>(base) + metadataOffset_);
  }

  MSCCLPP_HOST_DEVICE_INLINE float* topKValues(void* base) const {
    return reinterpret_cast<float*>(topKIndices(base) + topK_);
  }

  MSCCLPP_HOST_DEVICE_INLINE const float* topKValues(const void* base) const {
    return reinterpret_cast<const float*>(topKIndices(base) + topK_);
  }

  MSCCLPP_HOST_DEVICE_INLINE int* srcTokenGlobalIdx(void* base) const {
    return reinterpret_cast<int*>(topKValues(base) + topK_);
  }

  MSCCLPP_HOST_DEVICE_INLINE const int* srcTokenGlobalIdx(const void* base) const {
    return reinterpret_cast<const int*>(topKValues(base) + topK_);
  }
};

struct Layout {
  size_t totalBytes_;
  void* dispatchRecvBuffer_;
  void* combineRecvBuffer_;

  Layout(void* symmetricBuffer, int maxTokensPerRank, int hidden, int numRanks, int numExperts, int numTopk) {
    const PayloadView<Bf16> bf16Payload(hidden, numTopk);
    const PayloadView<Fp8E4M3, float> fp8Payload128(hidden, numTopk, 128);
    const PayloadView<Fp8E4M3, float> fp8Payload32(hidden, numTopk, 32);
    const size_t dispatchMetadataBytes =
        configAlign<size_t>(static_cast<size_t>(numRanks + numExperts) * sizeof(uint64_t), 128);
    const size_t dispatchPayloadStride =
        configAlign<size_t>(std::max({bf16Payload.numBytes_, fp8Payload128.numBytes_, fp8Payload32.numBytes_}), 128);
    const size_t dispatchBufferBytes =
        dispatchMetadataBytes + static_cast<size_t>(numRanks) * maxTokensPerRank * dispatchPayloadStride;
    const size_t combineBufferBytes = static_cast<size_t>(numExperts) * maxTokensPerRank * hidden * sizeof(Bf16);
    const size_t recvBufferBytes = configAlign<size_t>(std::max(dispatchBufferBytes, combineBufferBytes), 128);
    totalBytes_ = 2 * recvBufferBytes;

    if (symmetricBuffer != nullptr) {
      auto* base = reinterpret_cast<uint8_t*>(symmetricBuffer);
      dispatchRecvBuffer_ = base;
      combineRecvBuffer_ = base + recvBufferBytes;
    }
  }
};

inline size_t symmetricBufferSize(int maxTokensPerRank, int hidden, int numRanks, int numExperts, int numTopk) {
  const auto numBytes = Layout(nullptr, maxTokensPerRank, hidden, numRanks, numExperts, numTopk).totalBytes_;
  return configAlign<size_t>(numBytes, NUM_BUFFER_ALIGNMENT_BYTES);
}

}  // namespace low_latency

}  // namespace ep
}  // namespace mscclpp
