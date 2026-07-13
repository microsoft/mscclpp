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
__host__ __device__ constexpr dtype_t configCellDiv(dtype_t a, dtype_t b) {
  return (a + b - 1) / b;
}

template <typename dtype_t>
__host__ __device__ constexpr dtype_t configAlign(dtype_t a, dtype_t b) {
  return configCellDiv<dtype_t>(a, b) * b;
}

namespace low_latency {

// Rank-deduplicated dispatch payload layout:
//
//   [data: DataType[hidden]]
//   [optional scales: ScaleType[hidden / scale_block_size]]
//   [topKIndices: int[topK]]
//   [topKValues: float[topK]]
//   [srcTokenGlobalIdx: int]
//
// The payload is 32-byte aligned as a whole.
template <typename DataType, typename ScaleType = void>
struct PayloadView {
  static constexpr bool kHasScales = !std::is_void_v<ScaleType>;

  int hidden_;
  int topK_;
  int scaleBlockSize_;
  int numScales_;
  size_t hiddenBytes_;
  size_t scaleOffset_;
  size_t metadataOffset_;
  size_t numBytes_;

  MSCCLPP_HOST_DEVICE_INLINE static int numScales([[maybe_unused]] int hidden, [[maybe_unused]] int scaleBlockSize) {
    if constexpr (kHasScales) {
      return hidden / scaleBlockSize;
    }
    return 0;
  }

  MSCCLPP_HOST_DEVICE_INLINE static size_t hiddenBytes(int hidden) {
    return static_cast<size_t>(hidden) * sizeof(DataType);
  }

  MSCCLPP_HOST_DEVICE_INLINE static size_t scaleOffset(int hidden) {
    if constexpr (kHasScales) {
      return configAlign<size_t>(hiddenBytes(hidden), alignof(ScaleType));
    }
    return 0;
  }

  MSCCLPP_HOST_DEVICE_INLINE static size_t scaleBytes([[maybe_unused]] int hidden,
                                                      [[maybe_unused]] int scaleBlockSize) {
    if constexpr (kHasScales) {
      return static_cast<size_t>(numScales(hidden, scaleBlockSize)) * sizeof(ScaleType);
    }
    return 0;
  }

  MSCCLPP_HOST_DEVICE_INLINE static size_t metadataOffset(int hidden, int scaleBlockSize) {
    if constexpr (kHasScales) {
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

  MSCCLPP_HOST_DEVICE_INLINE PayloadView(int hidden, int topK, int scaleBlockSize = (kHasScales ? 128 : 0))
      : hidden_(hidden),
        topK_(topK),
        scaleBlockSize_(scaleBlockSize),
        numScales_(numScales(hidden, scaleBlockSize)),
        hiddenBytes_(hiddenBytes(hidden)),
        scaleOffset_(scaleOffset(hidden)),
        metadataOffset_(metadataOffset(hidden, scaleBlockSize)),
        numBytes_(numBytes(hidden, topK, scaleBlockSize)) {}

  template <typename T>
  MSCCLPP_HOST_DEVICE_INLINE T* data(void* base) const {
    return reinterpret_cast<T*>(base);
  }

  MSCCLPP_HOST_DEVICE_INLINE ScaleType* scaleFactors(void* base) const {
    static_assert(kHasScales, "Payload has no scale factors");
    return reinterpret_cast<ScaleType*>(reinterpret_cast<uint8_t*>(base) + scaleOffset_);
  }

  MSCCLPP_HOST_DEVICE_INLINE const ScaleType* scaleFactors(const void* base) const {
    static_assert(kHasScales, "Payload has no scale factors");
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

struct Buffer {
  void* dispatchData_;
  void* combineData_;
  mscclpp::LL8Packet* combineReadyPackets_;
};

struct Layout {
  size_t totalBytes_;
  Buffer buffers_[2];

  Layout(void* rdmaBuffer, int maxTokensPerRank, int hidden, int numRanks, int numExperts, int numTopk) {
    const PayloadView<__bfloat16> bf16Payload(hidden, numTopk);
    const PayloadView<__nv_fp8_storage_t, float> fp8Payload(hidden, numTopk, 128);
    const size_t dispatchMetadataBytes =
        configAlign<size_t>(static_cast<size_t>(2 * numRanks + numExperts) * sizeof(uint64_t), 128);
    const size_t dispatchPayloadStride =
        configAlign<size_t>(std::max(bf16Payload.numBytes_, fp8Payload.numBytes_), 128);
    const size_t dispatchBufferBytes =
        dispatchMetadataBytes + static_cast<size_t>(numRanks) * maxTokensPerRank * dispatchPayloadStride;
    const size_t combineControlBytes =
        configAlign<size_t>(static_cast<size_t>(numRanks) * sizeof(mscclpp::LL8Packet), 128);
    const size_t combineBufferBytes =
        combineControlBytes + static_cast<size_t>(numExperts) * maxTokensPerRank * hidden * sizeof(__bfloat16);
    const size_t bufferBytes = configAlign<size_t>(std::max(dispatchBufferBytes, combineBufferBytes), 128);
    totalBytes_ = 2 * bufferBytes;

    if (rdmaBuffer != nullptr) {
      auto* base = reinterpret_cast<uint8_t*>(rdmaBuffer);
      for (int bufferIdx = 0; bufferIdx < 2; ++bufferIdx) {
        auto* bufferBase = base + static_cast<size_t>(bufferIdx) * bufferBytes;
        buffers_[bufferIdx] = {
            .dispatchData_ = bufferBase,
            .combineData_ = bufferBase + combineControlBytes,
            .combineReadyPackets_ = reinterpret_cast<mscclpp::LL8Packet*>(bufferBase),
        };
      }
    }
  }
};

inline size_t getRdmaSizeHint(int maxTokensPerRank, int hidden, int numRanks, int numExperts, int numTopk) {
  const auto numBytes = Layout(nullptr, maxTokensPerRank, hidden, numRanks, numExperts, numTopk).totalBytes_;
  return configAlign<size_t>(numBytes, NUM_BUFFER_ALIGNMENT_BYTES);
}

}  // namespace low_latency

}  // namespace ep
}  // namespace mscclpp
