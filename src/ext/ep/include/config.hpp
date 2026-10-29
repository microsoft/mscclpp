// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_CONFIG_HPP_
#define MSCCLPP_EP_CONFIG_HPP_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mscclpp/concurrency_device.hpp>
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

inline constexpr int GpuNetIoMaxQpsPerPeer = 64;
inline constexpr int GpuNetIoDistributeDefaultChunkRows = 8;
inline constexpr int GpuNetIoDistributeForwardBlocks = 8;
inline constexpr int GpuNetIoDistributeForwardWarps = 8;

inline constexpr bool isSupportedRanks(int numRanks) { return numRanks > 0 && numRanks <= 64; }

template <typename dtype_t>
MSCCLPP_HOST_DEVICE_INLINE constexpr dtype_t configCellDiv(dtype_t a, dtype_t b) {
  return (a + b - 1) / b;
}

template <typename dtype_t>
MSCCLPP_HOST_DEVICE_INLINE constexpr dtype_t configAlign(dtype_t a, dtype_t b) {
  return configCellDiv<dtype_t>(a, b) * b;
}

// Send-and-distribute owns separate registered send and landing rows. Routing slots
// are the source's freshly allocated compact slots, not input-token indices.
struct GpuNetIoDistributeRow {
  size_t stride_;
  int* ids_;
  float* weights_;
  int* slots_;

  MSCCLPP_HOST_DEVICE_INLINE static size_t stride(int hidden, int topk) {
    return configAlign<size_t>(static_cast<size_t>(hidden) * sizeof(uint16_t) + topk * 3 * sizeof(int),
                               BufferAlignmentBytes);
  }

  MSCCLPP_HOST_DEVICE_INLINE GpuNetIoDistributeRow(void* row, int hidden, int topk)
      : stride_(stride(hidden, topk)),
        ids_(reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(row) + hidden * sizeof(uint16_t))),
        weights_(reinterpret_cast<float*>(ids_ + topk)),
        slots_(reinterpret_cast<int*>(weights_ + topk)) {}
};

struct alignas(BufferAlignmentBytes) GpuNetIoDistributeHeader {
  mscclpp::LL8Packet rows;
  mscclpp::LL8Packet counts[64];

  MSCCLPP_HOST_DEVICE_INLINE static constexpr size_t wireBytes(int ranks) {
    return static_cast<size_t>(1 + ranks) * sizeof(mscclpp::LL8Packet);
  }
};
static_assert(offsetof(GpuNetIoDistributeHeader, counts) == sizeof(mscclpp::LL8Packet));

struct alignas(BufferAlignmentBytes) GpuNetIoDistributeControl {
  uint64_t chunks[4];
  uint64_t chunkBase[4];
  uint64_t ack;
  uint64_t ackBase;
  uint32_t forwardReady[64];
  GpuNetIoDistributeHeader sendHeader;
  GpuNetIoDistributeHeader recvHeader;
};

struct alignas(BufferAlignmentBytes) GpuNetIoAggregateControl {
  uint64_t chunks[4];
  uint64_t chunkBase[4];
  uint32_t lastCombineEpoch;
};
static_assert(sizeof(GpuNetIoAggregateControl) == BufferAlignmentBytes);

using Bf16 = typename mscclpp::bf16x2::ElementType;
using Fp8E4M3 = typename mscclpp::f8_e4m3x2::ElementType;

MSCCLPP_HOST_DEVICE_INLINE constexpr int dispatchElementBytes(DispatchDataType dispatchDataType) {
  return dispatchDataType == DispatchDataType::BF16 ? static_cast<int>(sizeof(Bf16))
                                                    : static_cast<int>(sizeof(Fp8E4M3));
}

MSCCLPP_HOST_DEVICE_INLINE constexpr int dispatchElementsPerScale(DispatchDataType dispatchDataType) {
  return dispatchDataType == DispatchDataType::MXFP8_E4M3 ? 32
         : dispatchDataType == DispatchDataType::FP8_E4M3 ? 128
                                                          : 0;
}

MSCCLPP_HOST_DEVICE_INLINE constexpr int dispatchNumScales(DispatchDataType dispatchDataType, int hidden) {
  const int blockSize = dispatchElementsPerScale(dispatchDataType);
  return blockSize == 0 ? 0 : hidden / blockSize;
}

MSCCLPP_HOST_DEVICE_INLINE constexpr bool isSupportedDispatchDataType(DispatchDataType dataType) {
  return dataType == DispatchDataType::BF16 || dataType == DispatchDataType::FP8_E4M3 ||
         dataType == DispatchDataType::MXFP8_E4M3 || dataType == DispatchDataType::FP8_E4M3_STATIC;
}

MSCCLPP_HOST_DEVICE_INLINE constexpr bool isSupportedCombineDataType(DispatchDataType dataType) {
  return dataType == DispatchDataType::BF16 || dataType == DispatchDataType::FP8_E4M3_STATIC;
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
  return configAlign<size_t>(static_cast<size_t>(numRanks + numExperts) * sizeof(uint64_t), BufferAlignmentBytes);
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
  void* gpuNetIoCombineFlagsBuffer_ = nullptr;
  void* gpuNetIoCombineLandingBuffer_ = nullptr;
  void* gpuNetIoDistributeSend_ = nullptr;
  void* gpuNetIoDistributeRecv_ = nullptr;
  GpuNetIoDistributeControl* gpuNetIoDistributeControl_ = nullptr;
  float* gpuNetIoAggregateSend_ = nullptr;
  float* gpuNetIoAggregateRecv_ = nullptr;
  GpuNetIoAggregateControl* gpuNetIoAggregateControl_ = nullptr;
  void* dispatchRecvBuffer_ = nullptr;
  // Rank-major expert input or expert-major receive staging.
  void* combineBuffer_ = nullptr;
  void* rankMajorTopkIdsBuffer_ = nullptr;
  void* rankMajorTopkWeightsBuffer_ = nullptr;
  void* rankMajorScalesBuffer_ = nullptr;
  void* dispatchOutputBuffer_ = nullptr;

  LatencyStorageLayout(void* symmetricBuffer, int maxTokensPerRank, int hidden, int numRanks, int numExperts,
                       int numTopk, DispatchLayout outputLayout, CombineMode combineMode, bool useGpuNetIo = false,
                       bool aggregateCombine = false) {
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
    const size_t rankMajorScalesOffset =
        configAlign<size_t>(rankMajorTokenOffsetBytes + rankMajorDispatchOutputBytes, BufferAlignmentBytes);
    const size_t rankMajorScalesBytes =
        static_cast<size_t>(numRanks) * maxTokensPerRank * dispatchNumScales(DispatchDataType::MXFP8_E4M3, hidden);
    const size_t rankMajorDirectSendCombineInputBytes = rankMajorDispatchOutputBytes * numTopk;
    const size_t expertMajorDispatchOutputBytes =
        static_cast<size_t>(numExperts) * maxTokensPerRank * hidden * sizeof(Bf16);
    const size_t rankLocalReduceCombineRecvBytes =
        static_cast<size_t>(numRanks) * maxTokensPerRank * hidden * sizeof(Bf16);
    const size_t rankMajorDispatchBufferBytes =
        rankMajorScalesOffset + (rankMajor && !useGpuNetIo ? rankMajorScalesBytes : 0);
    const size_t dispatchOutputBytes = rankMajor ? rankMajorDispatchOutputBytes : expertMajorDispatchOutputBytes;
    const size_t dispatchRecvBufferBytes =
        std::max({dispatchBufferBytes, rankMajorDispatchBufferBytes, dispatchOutputBytes});
    const size_t combineBufferBytes = rankMajorDirectSend    ? rankMajorDirectSendCombineInputBytes
                                      : rankMajorLocalReduce ? (useGpuNetIo ? 0 : rankMajorDispatchOutputBytes)
                                      : combineMode == CombineMode::RANK_LOCAL_REDUCE ? rankLocalReduceCombineRecvBytes
                                                                                      : expertMajorDispatchOutputBytes;
    const size_t alignedDispatchRecvBufferBytes = configAlign<size_t>(dispatchRecvBufferBytes, BufferAlignmentBytes);
    const size_t alignedCombineBufferBytes = configAlign<size_t>(combineBufferBytes, BufferAlignmentBytes);
    totalBytes_ = alignedDispatchRecvBufferBytes + alignedCombineBufferBytes +
                  (rankMajor ? 0 : configAlign<size_t>(dispatchOutputBytes, BufferAlignmentBytes));

    const bool directCombine = useGpuNetIo && !aggregateCombine;
    const size_t gpuNetIoCombineFlagsBytes =
        directCombine ? configAlign<size_t>(static_cast<size_t>(numRanks) * GpuNetIoMaxQpsPerPeer * sizeof(uint64_t),
                                            BufferAlignmentBytes)
                      : 0;
    const size_t gpuNetIoCombineLandingBytes =
        directCombine ? configAlign<size_t>(rankMajorDispatchOutputBytes, BufferAlignmentBytes) : 0;
    const size_t baseBytes = totalBytes_;
    totalBytes_ += gpuNetIoCombineFlagsBytes + gpuNetIoCombineLandingBytes;
    const size_t distributeOffset = totalBytes_;
    const size_t distributeBytes =
        useGpuNetIo ? static_cast<size_t>(maxTokensPerRank) * GpuNetIoDistributeRow::stride(hidden, numTopk) : 0;
    if (useGpuNetIo) totalBytes_ += 2 * distributeBytes + sizeof(GpuNetIoDistributeControl);
    const size_t aggregateOffset = totalBytes_;
    const size_t aggregateBytes = aggregateCombine ? static_cast<size_t>(maxTokensPerRank) * hidden * sizeof(float) : 0;
    if (aggregateCombine) totalBytes_ += 2 * aggregateBytes + sizeof(GpuNetIoAggregateControl);

    if (symmetricBuffer != nullptr) {
      auto* base = reinterpret_cast<uint8_t*>(symmetricBuffer);
      dispatchRecvBuffer_ = base;
      rankMajorTopkIdsBuffer_ = base + rankMajorTopkIdsOffset(numRanks, numExperts);
      rankMajorTopkWeightsBuffer_ = base + rankMajorTopkWeightsOffset(numRanks, numExperts, maxTokensPerRank, numTopk);
      rankMajorScalesBuffer_ = rankMajor && !useGpuNetIo ? base + rankMajorScalesOffset : nullptr;
      dispatchOutputBuffer_ = rankMajor ? base + rankMajorTokenOffsetBytes
                                        : base + alignedDispatchRecvBufferBytes + alignedCombineBufferBytes;
      combineBuffer_ =
          useGpuNetIo && rankMajorLocalReduce ? dispatchOutputBuffer_ : base + alignedDispatchRecvBufferBytes;
      if (directCombine) {
        gpuNetIoCombineFlagsBuffer_ = base + baseBytes;
        gpuNetIoCombineLandingBuffer_ = base + baseBytes + gpuNetIoCombineFlagsBytes;
      }
      if (useGpuNetIo) {
        gpuNetIoDistributeSend_ = base + distributeOffset;
        gpuNetIoDistributeRecv_ = base + distributeOffset + distributeBytes;
        gpuNetIoDistributeControl_ =
            reinterpret_cast<GpuNetIoDistributeControl*>(base + distributeOffset + 2 * distributeBytes);
      }
      if (aggregateCombine) {
        gpuNetIoAggregateSend_ = reinterpret_cast<float*>(base + aggregateOffset);
        gpuNetIoAggregateRecv_ = reinterpret_cast<float*>(base + aggregateOffset + aggregateBytes);
        gpuNetIoAggregateControl_ =
            reinterpret_cast<GpuNetIoAggregateControl*>(base + aggregateOffset + 2 * aggregateBytes);
      }
    }
  }
};

inline size_t latencyStorageSize(int maxTokensPerRank, int hidden, int numRanks, int numExperts, int numTopk,
                                 DispatchLayout outputLayout, CombineMode combineMode, bool useGpuNetIo = false,
                                 bool aggregateCombine = false) {
  const auto numBytes = LatencyStorageLayout(nullptr, maxTokensPerRank, hidden, numRanks, numExperts, numTopk,
                                             outputLayout, combineMode, useGpuNetIo, aggregateCombine)
                            .totalBytes_;
  return configAlign<size_t>(numBytes, BufferAlignmentBytes);
}

// Peer-visible token and metadata slabs within recvBuffer_.
// TOKEN_MAJOR is compact; RANK_MAJOR reserves source-rank ranges.
// maxRows is capacity; private routing state lives in ThroughputWorkspaceLayout.
//
// TOKEN_MAJOR: one dense tensor, with no gaps between source-rank batches.
//   [dense data: DataType[numRecvTokens][hidden]]
//   [unused data capacity]
//   [topKIndices: int[maxRows][topK]]
//   [topKValues: float[maxRows][topK]]
//   [FP8 scales: float[maxRows][hidden / 128]]
struct ThroughputPayloadView {
  int topK_;
  int numScales_;
  size_t topKIndicesOffset_;
  size_t topKValuesOffset_;
  size_t scaleFactorsOffset_;
  size_t numBytes_;

  MSCCLPP_HOST_DEVICE_INLINE ThroughputPayloadView(size_t maxRows, int hidden, int topK)
      : topK_(topK),
        numScales_(dispatchNumScales(DispatchDataType::FP8_E4M3, hidden)),
        topKIndicesOffset_(
            configAlign<size_t>(maxRows * static_cast<size_t>(hidden) * sizeof(Bf16), BufferAlignmentBytes)),
        topKValuesOffset_(configAlign<size_t>(topKIndicesOffset_ + maxRows * static_cast<size_t>(topK_) * sizeof(int),
                                              BufferAlignmentBytes)),
        scaleFactorsOffset_(configAlign<size_t>(
            topKValuesOffset_ + maxRows * static_cast<size_t>(topK_) * sizeof(float), BufferAlignmentBytes)),
        numBytes_(configAlign<size_t>(scaleFactorsOffset_ + maxRows * static_cast<size_t>(numScales_) * sizeof(float),
                                      BufferAlignmentBytes)) {}

  template <typename T>
  MSCCLPP_HOST_DEVICE_INLINE T* data(void* base) const {
    return static_cast<T*>(base);
  }

  template <typename T>
  MSCCLPP_HOST_DEVICE_INLINE const T* data(const void* base) const {
    return static_cast<const T*>(base);
  }

  MSCCLPP_HOST_DEVICE_INLINE int* topKIndices(void* base, int64_t row) const {
    return reinterpret_cast<int*>(static_cast<uint8_t*>(base) + topKIndicesOffset_) + row * topK_;
  }

  MSCCLPP_HOST_DEVICE_INLINE const int* topKIndices(const void* base, int64_t row) const {
    return reinterpret_cast<const int*>(static_cast<const uint8_t*>(base) + topKIndicesOffset_) + row * topK_;
  }

  MSCCLPP_HOST_DEVICE_INLINE float* topKValues(void* base, int64_t row) const {
    return reinterpret_cast<float*>(static_cast<uint8_t*>(base) + topKValuesOffset_) + row * topK_;
  }

  MSCCLPP_HOST_DEVICE_INLINE const float* topKValues(const void* base, int64_t row) const {
    return reinterpret_cast<const float*>(static_cast<const uint8_t*>(base) + topKValuesOffset_) + row * topK_;
  }

  MSCCLPP_HOST_DEVICE_INLINE float* scaleFactors(void* base, int64_t row) const {
    return reinterpret_cast<float*>(static_cast<uint8_t*>(base) + scaleFactorsOffset_) + row * numScales_;
  }

  MSCCLPP_HOST_DEVICE_INLINE const float* scaleFactors(const void* base, int64_t row) const {
    return reinterpret_cast<const float*>(static_cast<const uint8_t*>(base) + scaleFactorsOffset_) + row * numScales_;
  }
};

struct ThroughputStorageLayout {
  // Peer-visible storage: count scratch, dispatch payload, and combine rows.
  ThroughputPayloadView payload_;
  size_t totalBytes_;
  void* recvBuffer_ = nullptr;
  void* combineBuffer_ = nullptr;
  int* outputTopkIdsBuffer_ = nullptr;
  float* outputTopkWeightsBuffer_ = nullptr;
  float* outputScalesBuffer_ = nullptr;

  ThroughputStorageLayout(void* symmetricBuffer, int maxTokensPerRank, int hidden, int numRanks, int numExperts,
                          int numTopk)
      : payload_(static_cast<size_t>(numRanks) * maxTokensPerRank, hidden, numTopk) {
    EP_HOST_ASSERT(isSupportedRanks(numRanks));
    EP_HOST_ASSERT(maxTokensPerRank > 0 && hidden > 0 && numExperts > 0 && numExperts % numRanks == 0);
    EP_HOST_ASSERT(numTopk > 0 && numTopk <= MaxNumTopk);

    const size_t ranks = static_cast<size_t>(numRanks);
    const size_t prefixBytes = ranks * ranks * sizeof(int);
    const size_t expertScratchBytes = static_cast<size_t>(numExperts) * sizeof(int);
    const size_t recvOffset = configAlign<size_t>(prefixBytes + expertScratchBytes, BufferAlignmentBytes);
    const size_t recvBytes = configAlign<size_t>(payload_.numBytes_, BufferAlignmentBytes);
    const size_t combineOffset = recvOffset + recvBytes;
    const size_t maxRows = ranks * static_cast<size_t>(maxTokensPerRank);
    const size_t combineBytes =
        configAlign<size_t>(maxRows * static_cast<size_t>(hidden) * sizeof(Bf16), BufferAlignmentBytes);
    totalBytes_ = combineOffset + combineBytes;
    if (symmetricBuffer != nullptr) {
      recvBuffer_ = static_cast<uint8_t*>(symmetricBuffer) + recvOffset;
      combineBuffer_ = static_cast<uint8_t*>(symmetricBuffer) + combineOffset;
      outputTopkIdsBuffer_ = payload_.topKIndices(recvBuffer_, 0);
      outputTopkWeightsBuffer_ = payload_.topKValues(recvBuffer_, 0);
      outputScalesBuffer_ = payload_.scaleFactors(recvBuffer_, 0);
    }
  }
};

struct alignas(8) ThroughputTokenRoute {
  int rank_;
  // Stable token offset within this source rank's receive range.
  int offset_;
};

// Local-only routing state produced by prepare() and consumed by dispatch/combine.
struct ThroughputWorkspaceLayout {
  size_t totalBytes_;
  // Local route counts by destination rank and global expert.
  int* numTokensPerRank_ = nullptr;
  int* numTokensPerExpert_ = nullptr;
  // This source rank's compact receive offset on each destination rank.
  int* rankOffsets_ = nullptr;
  // Distinct destination ranks and stable offsets for each local token.
  ThroughputTokenRoute* tokenRoutes_ = nullptr;
  // Active TOKEN_MAJOR rows received by this rank.
  int* numRecvTokens_ = nullptr;
  // Counts by local expert (TOKEN_MAJOR) or source rank (RANK_MAJOR).
  int* recvCounts_ = nullptr;
  // Local software grid barrier.
  mscclpp::DeviceSyncer* syncer_ = nullptr;

  ThroughputWorkspaceLayout(void* workspace, int maxTokensPerRank, int numRanks, int numExperts, int numTopk) {
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
    tokenRoutes_ = static_cast<ThroughputTokenRoute*>(place(
        static_cast<size_t>(maxTokensPerRank) * numTopk * sizeof(ThroughputTokenRoute), alignof(ThroughputTokenRoute)));
    numRecvTokens_ = static_cast<int*>(place(sizeof(int), alignof(int)));
    recvCounts_ = static_cast<int*>(
        place(static_cast<size_t>(std::max(numRanks, numExperts / numRanks)) * sizeof(int), alignof(int)));
    syncer_ = static_cast<mscclpp::DeviceSyncer*>(place(sizeof(mscclpp::DeviceSyncer), alignof(mscclpp::DeviceSyncer)));
    totalBytes_ = configAlign<size_t>(offset, BufferAlignmentBytes);
  }

  MSCCLPP_HOST_DEVICE_INLINE int recvTokenIndex(const ThroughputTokenRoute& route) const {
    return route.offset_ < 0 ? -1 : rankOffsets_[route.rank_] + route.offset_;
  }
};

inline size_t throughputWorkspaceSize(int maxTokensPerRank, int numRanks, int numExperts, int numTopk) {
  return ThroughputWorkspaceLayout(nullptr, maxTokensPerRank, numRanks, numExperts, numTopk).totalBytes_;
}

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_CONFIG_HPP_
