// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_CONFIG_HPP_
#define MSCCLPP_EP_CONFIG_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mscclpp/device.hpp>
#include <mscclpp/ext/ep/types.hpp>
#include <mscclpp/gpu_data_types.hpp>
#include <mscclpp/packet_device.hpp>
#include <type_traits>

#include "exception.hpp"

namespace mscclpp {
namespace ep {

inline constexpr size_t BufferAlignmentBytes = 128;
inline constexpr int MaxNumTopk = 8;

inline constexpr bool isSupportedThroughputRanks(int numRanks) {
  return numRanks == 2 || numRanks == 4 || numRanks == 8 || numRanks == 16 || numRanks == 32;
}

template <typename dtype_t>
MSCCLPP_HOST_DEVICE_INLINE constexpr dtype_t configCellDiv(dtype_t a, dtype_t b) {
  return (a + b - 1) / b;
}

template <typename dtype_t>
MSCCLPP_HOST_DEVICE_INLINE constexpr dtype_t configAlign(dtype_t a, dtype_t b) {
  return configCellDiv<dtype_t>(a, b) * b;
}

using Bf16 = typename mscclpp::bf16x2::ElementType;
using Fp8E4M3 = typename mscclpp::f8_e4m3x2::ElementType;

MSCCLPP_HOST_DEVICE_INLINE constexpr int dispatchElementBytes(DispatchDataType dispatchDataType) {
  return dispatchDataType == DispatchDataType::BF16 ? static_cast<int>(sizeof(Bf16))
                                                    : static_cast<int>(sizeof(Fp8E4M3));
}

MSCCLPP_HOST_DEVICE_INLINE constexpr int dispatchElementsPerScale(DispatchDataType dispatchDataType) {
  return dispatchDataType == DispatchDataType::FP8_E4M3 ? 128 : 0;
}

MSCCLPP_HOST_DEVICE_INLINE constexpr int dispatchNumScales(DispatchDataType dispatchDataType, int hidden) {
  return dispatchDataType == DispatchDataType::FP8_E4M3 ? hidden / dispatchElementsPerScale(dispatchDataType) : 0;
}

MSCCLPP_HOST_DEVICE_INLINE constexpr bool isSupportedDispatchDataType(DispatchDataType dataType) {
  return dataType == DispatchDataType::BF16 || dataType == DispatchDataType::FP8_E4M3;
}

// Latency rank-deduplicated dispatch payload layout:
//
//   [data: DataType[hidden]]
//   [optional scales: ScaleType[hidden / format scale block size]]
//   [topKIndices: int[topK]]
//   [topKValues: float[topK]]
//   [srcTokenGlobalIdx: int]
//
// The payload is 32-byte aligned as a whole.
template <typename DataType, typename ScaleType = void>
struct LatencyPayloadView {
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

  MSCCLPP_HOST_DEVICE_INLINE LatencyPayloadView(int hidden, int topK, int scaleBlockSize = (HasScales ? 128 : 0))
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

MSCCLPP_HOST_DEVICE_INLINE size_t rankMajorTopkIdsOffset(int numRanks, int numExperts) {
  return configAlign<size_t>(static_cast<size_t>(numRanks + numExperts) * sizeof(mscclpp::LL8Packet),
                             BufferAlignmentBytes);
}

MSCCLPP_HOST_DEVICE_INLINE size_t rankMajorTopkWeightsOffset(int numRanks, int numExperts, int maxTokensPerRank,
                                                             int numTopk) {
  const size_t numEntries = static_cast<size_t>(numRanks) * maxTokensPerRank * numTopk;
  return configAlign<size_t>(rankMajorTopkIdsOffset(numRanks, numExperts) + numEntries * sizeof(int),
                             BufferAlignmentBytes);
}

MSCCLPP_HOST_DEVICE_INLINE size_t rankMajorTokenOffset(int numRanks, int numExperts, int maxTokensPerRank,
                                                       int numTopk) {
  const size_t numEntries = static_cast<size_t>(numRanks) * maxTokensPerRank * numTopk;
  return configAlign<size_t>(
      rankMajorTopkWeightsOffset(numRanks, numExperts, maxTokensPerRank, numTopk) + numEntries * sizeof(float), 128);
}

struct LatencyStorageLayout {
  size_t totalBytes_;
  void* dispatchRecvBuffer_ = nullptr;
  // Rank-major expert input or expert-major receive staging.
  void* combineBuffer_ = nullptr;
  void* rankMajorTopkIdsBuffer_ = nullptr;
  void* rankMajorTopkWeightsBuffer_ = nullptr;
  void* dispatchOutputBuffer_ = nullptr;

  LatencyStorageLayout(void* symmetricBuffer, int maxTokensPerRank, int hidden, int numRanks, int numExperts,
                       int numTopk, DispatchLayout outputLayout, CombineMode combineMode) {
    const bool rankMajor = outputLayout == DispatchLayout::RANK_MAJOR;
    const bool rankMajorDirectSend = rankMajor && combineMode == CombineMode::DIRECT_SEND;
    const bool rankMajorLocalReduce = rankMajor && combineMode == CombineMode::RANK_LOCAL_REDUCE;
    const LatencyPayloadView<Bf16> bf16Payload(hidden, numTopk);
    const LatencyPayloadView<Fp8E4M3, float> fp8Payload128(hidden, numTopk, 128);
    const size_t dispatchMetadataBytes =
        configAlign<size_t>(static_cast<size_t>(numRanks + numExperts) * sizeof(uint64_t), BufferAlignmentBytes);
    const size_t dispatchPayloadStride =
        configAlign<size_t>(std::max(bf16Payload.numBytes_, fp8Payload128.numBytes_), BufferAlignmentBytes);
    const size_t dispatchBufferBytes =
        dispatchMetadataBytes + static_cast<size_t>(numRanks) * maxTokensPerRank * dispatchPayloadStride;
    const size_t rankMajorTokenOffsetBytes = rankMajorTokenOffset(numRanks, numExperts, maxTokensPerRank, numTopk);
    const size_t rankMajorDispatchOutputBytes =
        static_cast<size_t>(numRanks) * maxTokensPerRank * hidden * sizeof(Bf16);
    const size_t rankMajorDirectSendCombineInputBytes = rankMajorDispatchOutputBytes * numTopk;
    const size_t expertMajorDispatchOutputBytes =
        static_cast<size_t>(numExperts) * maxTokensPerRank * hidden * sizeof(Bf16);
    const size_t rankMajorDispatchBufferBytes = rankMajorTokenOffsetBytes + rankMajorDispatchOutputBytes;
    const size_t dispatchOutputBytes = rankMajor ? rankMajorDispatchOutputBytes : expertMajorDispatchOutputBytes;
    const size_t dispatchRecvBufferBytes =
        std::max({dispatchBufferBytes, rankMajorDispatchBufferBytes, dispatchOutputBytes});
    const size_t combineBufferBytes = rankMajorDirectSend    ? rankMajorDirectSendCombineInputBytes
                                      : rankMajorLocalReduce ? 0
                                                             : dispatchOutputBytes;
    const size_t alignedDispatchRecvBufferBytes = configAlign<size_t>(dispatchRecvBufferBytes, BufferAlignmentBytes);
    const size_t alignedCombineBufferBytes = configAlign<size_t>(combineBufferBytes, BufferAlignmentBytes);
    totalBytes_ = alignedDispatchRecvBufferBytes + alignedCombineBufferBytes +
                  (rankMajor ? 0 : configAlign<size_t>(dispatchOutputBytes, BufferAlignmentBytes));

    if (symmetricBuffer != nullptr) {
      auto* base = reinterpret_cast<uint8_t*>(symmetricBuffer);
      dispatchRecvBuffer_ = base;
      rankMajorTopkIdsBuffer_ = base + rankMajorTopkIdsOffset(numRanks, numExperts);
      rankMajorTopkWeightsBuffer_ = base + rankMajorTopkWeightsOffset(numRanks, numExperts, maxTokensPerRank, numTopk);
      dispatchOutputBuffer_ = rankMajor ? base + rankMajorTokenOffsetBytes
                                        : base + alignedDispatchRecvBufferBytes + alignedCombineBufferBytes;
      combineBuffer_ = rankMajorLocalReduce ? dispatchOutputBuffer_ : base + alignedDispatchRecvBufferBytes;
    }
  }
};

inline size_t latencyStorageSize(int maxTokensPerRank, int hidden, int numRanks, int numExperts, int numTopk,
                                 DispatchLayout outputLayout, CombineMode combineMode) {
  const auto numBytes =
      LatencyStorageLayout(nullptr, maxTokensPerRank, hidden, numRanks, numExperts, numTopk, outputLayout, combineMode)
          .totalBytes_;
  return configAlign<size_t>(numBytes, BufferAlignmentBytes);
}

// Unlike latency's packed per-token payload, throughput keeps dense token rows
// and fixed-stride metadata in separate slabs so GEMM can use the rows directly.
struct ThroughputPayloadView {
  int topK_;
  size_t metadataOffset_;
  size_t metadataSlotBytes_;
  size_t numBytes_;

  MSCCLPP_HOST_DEVICE_INLINE ThroughputPayloadView(size_t maxRows, int hidden, int topK)
      : topK_(topK),
        metadataOffset_(
            configAlign<size_t>(maxRows * static_cast<size_t>(hidden) * sizeof(Bf16), BufferAlignmentBytes)),
        metadataSlotBytes_(configAlign<size_t>(metadataBytes(dispatchNumScales(DispatchDataType::FP8_E4M3, hidden)),
                                               BufferAlignmentBytes)),
        numBytes_(metadataOffset_ + maxRows * metadataSlotBytes_) {}

  MSCCLPP_HOST_DEVICE_INLINE size_t metadataBytes(int numScales) const {
    return static_cast<size_t>(topK_) * (sizeof(int) + sizeof(float)) + static_cast<size_t>(numScales) * sizeof(float);
  }

  template <typename T>
  MSCCLPP_HOST_DEVICE_INLINE T* data(void* base) const {
    return static_cast<T*>(base);
  }

  template <typename T>
  MSCCLPP_HOST_DEVICE_INLINE const T* data(const void* base) const {
    return static_cast<const T*>(base);
  }

  MSCCLPP_HOST_DEVICE_INLINE int* topKIndices(void* base, int64_t row) const {
    return reinterpret_cast<int*>(static_cast<uint8_t*>(base) + metadataOffset_ + row * metadataSlotBytes_);
  }

  MSCCLPP_HOST_DEVICE_INLINE const int* topKIndices(const void* base, int64_t row) const {
    return reinterpret_cast<const int*>(static_cast<const uint8_t*>(base) + metadataOffset_ + row * metadataSlotBytes_);
  }

  MSCCLPP_HOST_DEVICE_INLINE float* topKValues(void* base, int64_t row) const {
    return reinterpret_cast<float*>(topKIndices(base, row) + topK_);
  }

  MSCCLPP_HOST_DEVICE_INLINE const float* topKValues(const void* base, int64_t row) const {
    return reinterpret_cast<const float*>(topKIndices(base, row) + topK_);
  }

  MSCCLPP_HOST_DEVICE_INLINE float* scaleFactors(void* base, int64_t row) const {
    return topKValues(base, row) + topK_;
  }

  MSCCLPP_HOST_DEVICE_INLINE const float* scaleFactors(const void* base, int64_t row) const {
    return topKValues(base, row) + topK_;
  }
};

struct ThroughputStorageLayout {
  // Allocation-derived offsets stay fixed when a request uses a smaller active capacity.
  ThroughputPayloadView payload_;
  size_t totalBytes_;
  void* recvBuffer_ = nullptr;

  ThroughputStorageLayout(void* symmetricBuffer, int maxTokensPerRank, int hidden, int numRanks, int numExperts,
                          int numTopk)
      : payload_(static_cast<size_t>(numRanks) * maxTokensPerRank, hidden, numTopk) {
    EP_HOST_ASSERT(isSupportedThroughputRanks(numRanks));
    EP_HOST_ASSERT(maxTokensPerRank > 0 && hidden > 0 && numExperts > 0 && numExperts % numRanks == 0);
    EP_HOST_ASSERT(numTopk > 0 && numTopk <= MaxNumTopk);

    const size_t ranks = static_cast<size_t>(numRanks);
    const size_t prefixBytes = ranks * ranks * sizeof(int);
    const size_t expertScratchBytes = static_cast<size_t>(numExperts) * sizeof(int);
    const size_t recvOffset = configAlign<size_t>(prefixBytes + expertScratchBytes, BufferAlignmentBytes);
    totalBytes_ = configAlign<size_t>(recvOffset + payload_.numBytes_, BufferAlignmentBytes);
    if (symmetricBuffer != nullptr) {
      recvBuffer_ = static_cast<uint8_t*>(symmetricBuffer) + recvOffset;
    }
  }
};

struct ThroughputWorkspaceLayout {
  size_t totalBytes_;
  // Local routing histograms produced before communicating with peers.
  int* numTokensPerRank_ = nullptr;
  int* numTokensPerExpert_ = nullptr;
  // Beginning of this source rank's receive range on each destination rank.
  int* rankOffsets_ = nullptr;
  // Stable offset within that range for [local token, destination rank], or -1.
  int* recvTokenOffsets_ = nullptr;
  int* numRecvTokens_ = nullptr;
  // Receive counts indexed by local expert or source rank, depending on output layout.
  int* recvCounts_ = nullptr;

  ThroughputWorkspaceLayout(void* workspace, int maxTokensPerRank, int numRanks, int numExperts) {
    size_t offset = 0;
    auto place = [&](size_t bytes, size_t alignment) -> void* {
      offset = configAlign<size_t>(offset, alignment);
      void* ptr = workspace == nullptr ? nullptr : reinterpret_cast<uint8_t*>(workspace) + offset;
      offset += bytes;
      return ptr;
    };

    numTokensPerRank_ = static_cast<int*>(place(static_cast<size_t>(numRanks) * sizeof(int), alignof(int)));
    numTokensPerExpert_ = static_cast<int*>(place(static_cast<size_t>(numExperts) * sizeof(int), alignof(int)));
    rankOffsets_ = static_cast<int*>(place(static_cast<size_t>(numRanks) * sizeof(int), alignof(int)));
    recvTokenOffsets_ =
        static_cast<int*>(place(static_cast<size_t>(maxTokensPerRank) * numRanks * sizeof(int), alignof(int)));
    numRecvTokens_ = static_cast<int*>(place(sizeof(int), alignof(int)));
    recvCounts_ = static_cast<int*>(
        place(static_cast<size_t>(std::max(numRanks, numExperts / numRanks)) * sizeof(int), alignof(int)));
    totalBytes_ = configAlign<size_t>(offset, BufferAlignmentBytes);
  }

  MSCCLPP_HOST_DEVICE_INLINE int recvTokenIndex(int token, int destinationRank, int numRanks) const {
    const int offset = recvTokenOffsets_[static_cast<size_t>(token) * numRanks + destinationRank];
    return offset < 0 ? -1 : rankOffsets_[destinationRank] + offset;
  }
};

inline size_t throughputWorkspaceSize(int maxTokensPerRank, int numRanks, int numExperts) {
  return ThroughputWorkspaceLayout(nullptr, maxTokensPerRank, numRanks, numExperts).totalBytes_;
}

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_CONFIG_HPP_
