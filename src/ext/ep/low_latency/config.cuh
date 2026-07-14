// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <cstdint>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/gpu_data_types.hpp>
#include <mscclpp/memory_channel_device.hpp>

#include "../config.hpp"
#include "api.cuh"
#include "device_helpers.cuh"

namespace mscclpp {
namespace ep {
namespace low_latency {
namespace detail {

constexpr int DispatchNWarps = 16;
constexpr int DispatchMinNWarpsPerGroup = 8;
constexpr int DispatchMaxNWarpGroups = DispatchNWarps / DispatchMinNWarpsPerGroup;
constexpr int DispatchNThreads = DispatchNWarps * WARP_SIZE;
constexpr int DispatchMaxNRecvTmaWorkers = DispatchNWarps;
constexpr int DispatchWarpGroupBarrierBase = 1;
constexpr int DispatchSchedulerPrefixBarrier = DispatchWarpGroupBarrierBase + DispatchMaxNWarpGroups;
constexpr int DispatchSchedulerReadyBarrier = DispatchSchedulerPrefixBarrier + 1;
constexpr size_t OptimizedDynamicSharedMemoryBytes = 226 * 1024;
constexpr size_t TmaWorkerControlBytes = DispatchMaxNWarpGroups * WARP_SIZE * sizeof(int);
static_assert(DispatchNWarps % DispatchMinNWarpsPerGroup == 0);
static_assert(sizeof(mscclpp::DeviceSemaphore) == sizeof(int));
static_assert(alignof(mscclpp::DeviceSemaphore) <= alignof(int));
static_assert(sizeof(mscclpp::DeviceSyncer) % sizeof(int) == 0);
static_assert(alignof(mscclpp::DeviceSyncer) <= alignof(int));

struct TransportView {
  void* symmetricBufferBase_;
  void* const* peerMappedBufferBases_;
  mscclpp::BaseMemoryChannelDeviceHandle* baseMemoryChannels_;
  mscclpp::PortChannelDeviceHandle* portChannels_;
  int rank_;

  MSCCLPP_HOST_DEVICE_INLINE explicit TransportView(const CommContext& comm)
      : symmetricBufferBase_(comm.symmetricBufferBase_),
        peerMappedBufferBases_(comm.peerMappedBufferBases_),
        baseMemoryChannels_(comm.baseMemoryChannels_),
        portChannels_(comm.portChannels_),
        rank_(comm.rank_) {}

  MSCCLPP_HOST_DEVICE_INLINE bool isSelf(int peerRank) const { return peerRank == rank_; }

  MSCCLPP_HOST_DEVICE_INLINE bool isMappedPeer(int peerRank) const {
    return peerMappedBufferBases_[peerRank] != nullptr;
  }

  MSCCLPP_HOST_DEVICE_INLINE bool isPortPeer(int peerRank) const {
    return !isSelf(peerRank) && !isMappedPeer(peerRank);
  }

  MSCCLPP_HOST_DEVICE_INLINE void* mappedBuffer(void* localBuffer, int peerRank) const {
    if (isSelf(peerRank)) return localBuffer;
    const auto offset = reinterpret_cast<uint8_t*>(localBuffer) - reinterpret_cast<uint8_t*>(symmetricBufferBase_);
    return reinterpret_cast<uint8_t*>(peerMappedBufferBases_[peerRank]) + offset;
  }

  MSCCLPP_HOST_DEVICE_INLINE uint64_t offset(const void* pointer) const {
    return reinterpret_cast<const uint8_t*>(pointer) - reinterpret_cast<const uint8_t*>(symmetricBufferBase_);
  }
};

struct BufferView {
  void* base_;
  Workload workload_;
  int nRanks_;

  MSCCLPP_HOST_DEVICE_INLINE int* rankTokenCompactSlotMap() const {
    return low_latency::rankTokenCompactSlotMap(base_, workload_.maxTokensPerRank_, workload_.hidden_, nRanks_,
                                                workload_.numExperts_, workload_.numTopk_);
  }

  MSCCLPP_HOST_DEVICE_INLINE void* dispatchPayloadStaging() const {
    return low_latency::dispatchPayloadStaging(base_, workload_.maxTokensPerRank_, workload_.hidden_, nRanks_,
                                               workload_.numExperts_, workload_.numTopk_);
  }

  MSCCLPP_HOST_DEVICE_INLINE void* dispatchMetadataStaging() const {
    return low_latency::dispatchMetadataStaging(base_, workload_.maxTokensPerRank_, workload_.hidden_, nRanks_,
                                                workload_.numExperts_, workload_.numTopk_);
  }

  MSCCLPP_HOST_DEVICE_INLINE void* combineStaging() const {
    return low_latency::combineStaging(base_, workload_.maxTokensPerRank_, workload_.hidden_, nRanks_,
                                       workload_.numExperts_, workload_.numTopk_);
  }
};

struct ConstBufferView {
  const void* base_;
  Workload workload_;
  int nRanks_;

  MSCCLPP_HOST_DEVICE_INLINE const int* rankTokenCompactSlotMap() const {
    return low_latency::rankTokenCompactSlotMap(base_, workload_.maxTokensPerRank_, workload_.hidden_, nRanks_,
                                                workload_.numExperts_, workload_.numTopk_);
  }
};

MSCCLPP_HOST_DEVICE_INLINE size_t dispatchMetadataBytes(int nRanks, int nExperts) {
  return dispatchMetadataRegionBytes(nRanks, nExperts);
}

MSCCLPP_HOST_DEVICE_INLINE size_t dispatchMetadataBlockBytes(int nLocalExperts) {
  return static_cast<size_t>(1 + nLocalExperts) * sizeof(mscclpp::LL8Packet);
}

MSCCLPP_HOST_DEVICE_INLINE size_t dispatchMetadataPacketIndex(int sourceRank, int nLocalExperts,
                                                              int localExpertIdx = -1) {
  return static_cast<size_t>(sourceRank) * (1 + nLocalExperts) + 1 + localExpertIdx;
}

template <DispatchDataType DataType>
struct DispatchDataTypeTraits;

template <>
struct DispatchDataTypeTraits<DispatchDataType::BF16> {
  using ElementType = Bf16;
  using ScaleType = void;
};

template <>
struct DispatchDataTypeTraits<DispatchDataType::FP8_E4M3> {
  using ElementType = Fp8E4M3;
  using ScaleType = float;
};

template <DispatchDataType DataType>
using DispatchElementType = typename DispatchDataTypeTraits<DataType>::ElementType;

template <DispatchDataType DataType>
using DispatchScaleType = typename DispatchDataTypeTraits<DataType>::ScaleType;

template <DispatchDataType DataType>
using DispatchPayloadView = PayloadView<DispatchElementType<DataType>, DispatchScaleType<DataType>>;

MSCCLPP_HOST_DEVICE_INLINE constexpr bool isSupportedDispatchDataType(DispatchDataType dataType) {
  return dataType == DispatchDataType::BF16 || dataType == DispatchDataType::FP8_E4M3;
}

template <DispatchDataType DataType>
MSCCLPP_HOST_DEVICE_INLINE size_t dispatchPayloadStride(int hidden, int nTopk, int scaleBlockSize) {
  return configAlign<size_t>(DispatchPayloadView<DataType>(hidden, nTopk, scaleBlockSize).numBytes_, 128);
}

MSCCLPP_HOST_DEVICE_INLINE constexpr int dispatchNWarpsPerGroup(int nTokens, int nBlocks) {
  return nTokens <= nBlocks ? DispatchNWarps
                            : (nTokens <= 2 * nBlocks ? DispatchNWarps / 2 : DispatchMinNWarpsPerGroup);
}

struct RecvTask {
  int sourceRank_;
  int tokenBegin_;
  int tokenEnd_;
};
static_assert(sizeof(RecvTask) % sizeof(int) == 0);
static_assert(alignof(RecvTask) <= alignof(int));

struct WorkspaceView {
  uint32_t* dispatchEpoch_;
  int* dispatchRankPayloadSlots_;
  int* dispatchRankPayloadCompletions_;
  mscclpp::DeviceSemaphore* dispatchLocalPayloadReady_;
  int* dispatchExpertCopiedCounts_;
  uint32_t* dispatchRankReadyEpochs_;
  RecvTask* dispatchRecvTasks_;
  uint32_t* dispatchTasksReadyEpoch_;
  int* dispatchNumRecvTasks_;
  int* combineRankPartialCompletions_;
  mscclpp::DeviceSyncer* combineSyncer_;

  MSCCLPP_HOST_DEVICE_INLINE WorkspaceView(void* workspace, int nRanks, int nExperts) {
    auto* cursor = reinterpret_cast<int*>(workspace);
    dispatchEpoch_ = reinterpret_cast<uint32_t*>(cursor++);
    dispatchRankPayloadSlots_ = cursor;
    cursor += nRanks;
    dispatchRankPayloadCompletions_ = cursor;
    cursor += nRanks;
    dispatchLocalPayloadReady_ = reinterpret_cast<mscclpp::DeviceSemaphore*>(cursor++);
    dispatchExpertCopiedCounts_ = cursor;
    cursor += nExperts;
    dispatchRankReadyEpochs_ = reinterpret_cast<uint32_t*>(cursor);
    cursor += nRanks;
    dispatchRecvTasks_ = reinterpret_cast<RecvTask*>(cursor);
    cursor += static_cast<size_t>(MaxWorkerBlocks) * sizeof(RecvTask) / sizeof(int);
    dispatchTasksReadyEpoch_ = reinterpret_cast<uint32_t*>(cursor++);
    dispatchNumRecvTasks_ = cursor++;
    combineRankPartialCompletions_ = cursor;
    cursor += nRanks;
    combineSyncer_ = reinterpret_cast<mscclpp::DeviceSyncer*>(cursor);
  }

  MSCCLPP_HOST_DEVICE_INLINE static size_t numBytes(int nRanks, int nExperts) {
    return sizeof(uint32_t) +                                       // dispatchEpoch_
           static_cast<size_t>(nRanks) * sizeof(int) +              // dispatchRankPayloadSlots_
           static_cast<size_t>(nRanks) * sizeof(int) +              // dispatchRankPayloadCompletions_
           sizeof(mscclpp::DeviceSemaphore) +                       // dispatchLocalPayloadReady_
           static_cast<size_t>(nExperts) * sizeof(int) +            // dispatchExpertCopiedCounts_
           static_cast<size_t>(nRanks) * sizeof(uint32_t) +         // dispatchRankReadyEpochs_
           static_cast<size_t>(MaxWorkerBlocks) * sizeof(RecvTask)  // dispatchRecvTasks_
           + sizeof(uint32_t) +                                     // dispatchTasksReadyEpoch_
           sizeof(int) +                                            // dispatchNumRecvTasks_
           static_cast<size_t>(nRanks) * sizeof(int) +              // combineRankPartialCompletions_
           sizeof(mscclpp::DeviceSyncer);                           // combineSyncer_
  }
};

struct KernelConfigCache {
  int deviceId_ = -1;
  size_t dynamicSharedBytes_ = 0;
  int residentBlocks_ = 0;
};

template <typename Kernel>
inline int configureKernel(Kernel kernel, int nThreads, size_t dynamicSharedBytes, const CommContext& comm,
                           KernelConfigCache& cache) {
  if (cache.deviceId_ != comm.deviceId_ || cache.dynamicSharedBytes_ < dynamicSharedBytes) {
    cudaFuncAttributes attributes;
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
    EP_HOST_ASSERT(dynamicSharedBytes + attributes.sharedSizeBytes <=
                   static_cast<size_t>(comm.maxSharedMemoryPerBlock_));
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(dynamicSharedBytes)));
    int blocksPerSm;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSm, kernel, nThreads, dynamicSharedBytes));
    cache.deviceId_ = comm.deviceId_;
    cache.dynamicSharedBytes_ = dynamicSharedBytes;
    cache.residentBlocks_ = blocksPerSm * comm.numSms_;
  }
  return cache.residentBlocks_;
}

MSCCLPP_HOST_DEVICE_INLINE size_t workspaceBytes(int nRanks, int nExperts) {
  return WorkspaceView::numBytes(nRanks, nExperts);
}

MSCCLPP_HOST_DEVICE_INLINE size_t dispatchSharedControlBytes(int nRanks) {
  constexpr int NSendSlots = DispatchMaxNWarpGroups * WARP_SIZE;
  const int nSlots = nRanks > NSendSlots ? nRanks : NSendSlots;
  return configAlign<size_t>(static_cast<size_t>(nSlots) * sizeof(int), 128);
}

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
MSCCLPP_HOST_DEVICE_INLINE size_t dispatchSendTmaBytes(int nTopk) {
  return DispatchMaxNWarpGroups * (dispatchPayloadStride<DataType>(Hidden, nTopk, ScaleBlockSize) + sizeof(uint64_t));
}

template <int Hidden, typename ElementType, int MaxWorkers>
MSCCLPP_HOST_DEVICE_INLINE constexpr int tmaWorkerCount() {
  static_assert(Hidden % 128 == 0);
  constexpr size_t workerBytes = static_cast<size_t>(Hidden) * sizeof(ElementType) + sizeof(uint64_t);
  constexpr int nWorkers = static_cast<int>((OptimizedDynamicSharedMemoryBytes - TmaWorkerControlBytes) / workerBytes);
  return nWorkers < MaxWorkers ? nWorkers : MaxWorkers;
}

template <int Hidden, DispatchDataType DataType>
MSCCLPP_HOST_DEVICE_INLINE size_t dispatchRecvTmaBytes() {
  using ElementType = DispatchElementType<DataType>;
  constexpr int NWorkers = tmaWorkerCount<Hidden, ElementType, DispatchMaxNRecvTmaWorkers>();
  constexpr size_t tileBytes = static_cast<size_t>(Hidden) * sizeof(ElementType);
  return static_cast<size_t>(NWorkers) * (tileBytes + sizeof(uint64_t));
}

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
MSCCLPP_HOST_DEVICE_INLINE size_t dispatchSharedBytes(int nRanks, int nExperts, int nTopk) {
  const size_t controlBytes = dispatchSharedControlBytes(nRanks);
  const size_t sendBytes = dispatchSendTmaBytes<Hidden, DataType, ScaleBlockSize>(nTopk);
  const size_t recvBytes = dispatchRecvTmaBytes<Hidden, DataType>();
  const size_t tmaBytes = controlBytes + (sendBytes > recvBytes ? sendBytes : recvBytes);
  const size_t metadataBytes = static_cast<size_t>(nRanks + nExperts) * sizeof(int);
  return tmaBytes > metadataBytes ? tmaBytes : metadataBytes;
}

}  // namespace detail
}  // namespace low_latency
}  // namespace ep
}  // namespace mscclpp
