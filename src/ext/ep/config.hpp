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

MSCCLPP_HOST_DEVICE_INLINE size_t combineDataBytes(int maxTokensPerRank, int hidden, int numExperts) {
  return static_cast<size_t>(numExperts) * maxTokensPerRank * hidden * sizeof(Bf16);
}

MSCCLPP_HOST_DEVICE_INLINE size_t combineStagingDataBytes(int maxTokensPerRank, int hidden, int numRanks, int numTopk) {
  return static_cast<size_t>(numRanks) * maxTokensPerRank * numTopk * hidden * sizeof(Bf16);
}

MSCCLPP_HOST_DEVICE_INLINE size_t dispatchMetadataRegionBytes(int numRanks, int numExperts) {
  return configAlign<size_t>(static_cast<size_t>(numRanks + numExperts) * sizeof(mscclpp::LL8Packet), 128);
}

MSCCLPP_HOST_DEVICE_INLINE size_t maxDispatchPayloadStride(int hidden, int numTopk) {
  const size_t bf16Bytes = PayloadView<Bf16>(hidden, numTopk).numBytes_;
  const size_t fp8Bytes128 = PayloadView<Fp8E4M3, float>(hidden, numTopk, 128).numBytes_;
  const size_t fp8Bytes64 = PayloadView<Fp8E4M3, float>(hidden, numTopk, 64).numBytes_;
  const size_t maxFp8Bytes = fp8Bytes128 > fp8Bytes64 ? fp8Bytes128 : fp8Bytes64;
  return configAlign<size_t>(bf16Bytes > maxFp8Bytes ? bf16Bytes : maxFp8Bytes, 128);
}

MSCCLPP_HOST_DEVICE_INLINE size_t dispatchPayloadRegionBytes(int maxTokensPerRank, int hidden, int numRanks,
                                                             int numTopk) {
  return static_cast<size_t>(numRanks) * maxTokensPerRank * maxDispatchPayloadStride(hidden, numTopk);
}

MSCCLPP_HOST_DEVICE_INLINE size_t rankTokenCompactSlotMapRegionBytes(int maxTokensPerRank, int numRanks) {
  return static_cast<size_t>(numRanks) * maxTokensPerRank * sizeof(int);
}

MSCCLPP_HOST_DEVICE_INLINE size_t bufferDataRegionBytes(int maxTokensPerRank, int hidden, int numRanks, int numExperts,
                                                        int numTopk) {
  const size_t dispatchBytes = dispatchMetadataRegionBytes(numRanks, numExperts) +
                               dispatchPayloadRegionBytes(maxTokensPerRank, hidden, numRanks, numTopk) +
                               rankTokenCompactSlotMapRegionBytes(maxTokensPerRank, numRanks);
  const size_t combineBytes = combineDataBytes(maxTokensPerRank, hidden, numExperts);
  return configAlign<size_t>(dispatchBytes > combineBytes ? dispatchBytes : combineBytes, 128);
}

MSCCLPP_HOST_DEVICE_INLINE int* rankTokenCompactSlotMap(void* buffer, int maxTokensPerRank, int hidden, int numRanks,
                                                        int numExperts, int numTopk) {
  return reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(buffer) + dispatchMetadataRegionBytes(numRanks, numExperts) +
                                dispatchPayloadRegionBytes(maxTokensPerRank, hidden, numRanks, numTopk));
}

MSCCLPP_HOST_DEVICE_INLINE const int* rankTokenCompactSlotMap(const void* buffer, int maxTokensPerRank, int hidden,
                                                              int numRanks, int numExperts, int numTopk) {
  return reinterpret_cast<const int*>(reinterpret_cast<const uint8_t*>(buffer) +
                                      dispatchMetadataRegionBytes(numRanks, numExperts) +
                                      dispatchPayloadRegionBytes(maxTokensPerRank, hidden, numRanks, numTopk));
}

MSCCLPP_HOST_DEVICE_INLINE void* dispatchPayloadStaging(void* buffer, int maxTokensPerRank, int hidden, int numRanks,
                                                        int numExperts, int numTopk) {
  return reinterpret_cast<uint8_t*>(buffer) +
         bufferDataRegionBytes(maxTokensPerRank, hidden, numRanks, numExperts, numTopk);
}

MSCCLPP_HOST_DEVICE_INLINE void* dispatchMetadataStaging(void* buffer, int maxTokensPerRank, int hidden, int numRanks,
                                                         int numExperts, int numTopk) {
  return reinterpret_cast<uint8_t*>(
             dispatchPayloadStaging(buffer, maxTokensPerRank, hidden, numRanks, numExperts, numTopk)) +
         dispatchPayloadRegionBytes(maxTokensPerRank, hidden, numRanks, numTopk);
}

MSCCLPP_HOST_DEVICE_INLINE void* combineStaging(void* buffer, int maxTokensPerRank, int hidden, int numRanks,
                                                int numExperts, int numTopk) {
  return reinterpret_cast<uint8_t*>(buffer) +
         bufferDataRegionBytes(maxTokensPerRank, hidden, numRanks, numExperts, numTopk);
}

struct Buffer {
  void* data_;
};

struct Layout {
  size_t totalBytes_;
  Buffer buffers_[2];

  Layout(void* symmetricBuffer, int maxTokensPerRank, int hidden, int numRanks, int numExperts, int numTopk) {
    const size_t dataBytes = bufferDataRegionBytes(maxTokensPerRank, hidden, numRanks, numExperts, numTopk);
    const size_t dispatchPayloadStagingBytes = dispatchPayloadRegionBytes(maxTokensPerRank, hidden, numRanks, numTopk);
    const size_t dispatchMetadataBlockBytes =
        static_cast<size_t>(1 + numExperts / numRanks) * sizeof(mscclpp::LL8Packet);
    const size_t dispatchMetadataStagingBytes = static_cast<size_t>(numRanks) * dispatchMetadataBlockBytes;
    const size_t dispatchStagingBytes = dispatchPayloadStagingBytes + dispatchMetadataStagingBytes;
    const size_t combineStagingBytes = combineStagingDataBytes(maxTokensPerRank, hidden, numRanks, numTopk);
    const size_t stagingBytes = std::max(dispatchStagingBytes, combineStagingBytes);
    const size_t bufferBytes = configAlign<size_t>(dataBytes + stagingBytes, 128);
    totalBytes_ = 2 * bufferBytes;

    if (symmetricBuffer != nullptr) {
      auto* base = reinterpret_cast<uint8_t*>(symmetricBuffer);
      for (int bufferIdx = 0; bufferIdx < 2; ++bufferIdx) {
        auto* bufferBase = base + static_cast<size_t>(bufferIdx) * bufferBytes;
        buffers_[bufferIdx] = {
            .data_ = bufferBase,
        };
      }
    }
  }
};

inline size_t getSymmetricBufferSizeHint(int maxTokensPerRank, int hidden, int numRanks, int numExperts, int numTopk) {
  const auto numBytes = Layout(nullptr, maxTokensPerRank, hidden, numRanks, numExperts, numTopk).totalBytes_;
  return configAlign<size_t>(numBytes, NUM_BUFFER_ALIGNMENT_BYTES);
}

}  // namespace low_latency

}  // namespace ep
}  // namespace mscclpp
