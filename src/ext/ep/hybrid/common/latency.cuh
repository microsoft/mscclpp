// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_HYBRID_COMMON_LATENCY_CUH_
#define MSCCLPP_EP_HYBRID_COMMON_LATENCY_CUH_

#include <cstdint>
#include <mscclpp/bulk_device.hpp>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/gpu_data_types.hpp>
#include <mscclpp/memory_channel_device.hpp>

#include "hybrid/common/device_helpers.cuh"
#include "hybrid/include/config.hpp"
#include "hybrid/include/kernels.hpp"

namespace mscclpp {
namespace ep {
namespace hybrid {

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
  int rank_;
  int numRanks_;
  // GPU-initiated networking (GPUNetIO) cross-domain resources. gpuNetIo_ is null
  // unless the backend is active for this rank; peers with a null mapped-buffer
  // base (see isNvlinkPeer) are served over GPUNetIO instead of NVLink/IPC.
  EpGpuNetIoDeviceContext* gpuNetIo_;
  void* gpuNetIoStagingBuffer_;
  void* gpuNetIoFlagsBuffer_;
  void* gpuNetIoCombineFlagsBuffer_;
  void* gpuNetIoCombineLandingBuffer_;
  size_t gpuNetIoSlotStride_;
  bool gpuNetIoBatchedDispatch_;
  bool gpuNetIoBatchAllQps_;
  bool gpuNetIoDeferFlush_;
  uint64_t* gpuNetIoDispatchTickets_;
  int gpuNetIoPipelineChunkRows_;
  int gpuNetIoParallelStage_;
  int gpuNetIoRowFence_;
  int* gpuNetIoChunkReady_;
  int* gpuNetIoTailRows_;
  int gpuNetIoTmaStage_;
  GpuNetIoDispatchPeerState* gpuNetIoDispatchPeers_;
  int gpuNetIoCombineChunkRows_;
  uint64_t* gpuNetIoCombineChunkBase_;
  bool gpuNetIoCombineNoBarrier_;

  MSCCLPP_HOST_DEVICE_INLINE explicit TransportView(const DeviceContext* context)
      : symmetricBufferBase_(context->localBufferBase_),
        peerMappedBufferBases_(context->peerBufferBases_),
        baseMemoryChannels_(context->channels_),
        rank_(context->rank_),
        numRanks_(context->numRanks_),
        gpuNetIo_(context->gpuNetIo_),
        gpuNetIoStagingBuffer_(context->gpuNetIoStagingBuffer_),
        gpuNetIoFlagsBuffer_(context->gpuNetIoFlagsBuffer_),
        gpuNetIoCombineFlagsBuffer_(context->gpuNetIoCombineFlagsBuffer_),
        gpuNetIoCombineLandingBuffer_(context->gpuNetIoCombineLandingBuffer_),
        gpuNetIoSlotStride_(context->gpuNetIoSlotStride_),
        gpuNetIoBatchedDispatch_(context->gpuNetIoBatchedDispatch_),
        gpuNetIoBatchAllQps_(context->gpuNetIoBatchAllQps_),
        gpuNetIoDeferFlush_(context->gpuNetIoDeferFlush_),
        gpuNetIoDispatchTickets_(context->gpuNetIoDispatchTickets_),
        gpuNetIoPipelineChunkRows_(context->gpuNetIoPipelineChunkRows_),
        gpuNetIoParallelStage_(context->gpuNetIoParallelStage_),
        gpuNetIoRowFence_(context->gpuNetIoRowFence_),
        gpuNetIoChunkReady_(context->gpuNetIoChunkReady_),
        gpuNetIoTailRows_(context->gpuNetIoTailRows_),
        gpuNetIoTmaStage_(context->gpuNetIoTmaStage_),
        gpuNetIoDispatchPeers_(context->gpuNetIoDispatchPeers_),
        gpuNetIoCombineChunkRows_(context->gpuNetIoCombineChunkRows_),
        gpuNetIoCombineChunkBase_(context->gpuNetIoCombineChunkBase_),
        gpuNetIoCombineNoBarrier_(context->gpuNetIoCombineNoBarrier_) {}

  MSCCLPP_HOST_DEVICE_INLINE bool isSelf(int peerRank) const { return peerRank == rank_; }

  // Byte offset of a symmetric-buffer pointer from this rank's base; the same
  // offset addresses the identical region on any peer (used by GPUNetIO puts).
  MSCCLPP_HOST_DEVICE_INLINE uint64_t symmetricOffset(const void* ptr) const {
    return static_cast<uint64_t>(reinterpret_cast<const uint8_t*>(ptr) -
                                 reinterpret_cast<const uint8_t*>(symmetricBufferBase_));
  }

  // A peer is NVLink/IPC-reachable if it is self or its symmetric buffer is
  // directly mapped; otherwise it is a cross-domain peer served over GPUNetIO.
  MSCCLPP_HOST_DEVICE_INLINE bool isNvlinkPeer(int peerRank) const {
    return isSelf(peerRank) || peerMappedBufferBases_[peerRank] != nullptr;
  }

  MSCCLPP_HOST_DEVICE_INLINE void* mappedBuffer(void* localBuffer, int peerRank) const {
    if (isSelf(peerRank)) return localBuffer;
    const auto offset = reinterpret_cast<uint8_t*>(localBuffer) - reinterpret_cast<uint8_t*>(symmetricBufferBase_);
    return reinterpret_cast<uint8_t*>(peerMappedBufferBases_[peerRank]) + offset;
  }
};

MSCCLPP_HOST_DEVICE_INLINE size_t dispatchMetadataBytes(int nRanks, int nExperts) {
  return configAlign<size_t>(static_cast<size_t>(nRanks + nExperts) * sizeof(mscclpp::LL8Packet), BufferAlignmentBytes);
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
  return configAlign<size_t>(DispatchPayloadView<DataType>(hidden, nTopk, scaleBlockSize).numBytes_,
                             BufferAlignmentBytes);
}

MSCCLPP_HOST_DEVICE_INLINE constexpr int dispatchNWarpsPerGroup(int nTokens, int nBlocks) {
  return nTokens <= nBlocks ? DispatchNWarps
                            : (nTokens <= 2 * nBlocks ? DispatchNWarps / 2 : DispatchMinNWarpsPerGroup);
}

MSCCLPP_HOST_DEVICE_INLINE int dispatchSharedSendSlots(int nRanks) {
  constexpr int NSendSlots = DispatchMaxNWarpGroups * WARP_SIZE;
  return nRanks > NSendSlots ? nRanks : NSendSlots;
}

struct RecvTask {
  int sourceRank_;
  int tokenBegin_;
  int tokenEnd_;
};
static_assert(sizeof(RecvTask) % sizeof(int) == 0);
static_assert(alignof(RecvTask) <= alignof(int));

struct WorkspaceView {
  int* dispatchRankPayloadSlots_;
  int* dispatchRankPayloadCompletions_;
  mscclpp::DeviceSemaphore* dispatchLocalPayloadReady_;
  int* dispatchExpertCopiedCounts_;
  uint32_t* dispatchRankReadyEpochs_;
  RecvTask* dispatchRecvTasks_;
  uint32_t* dispatchTasksReadyEpoch_;
  int* dispatchNumRecvTasks_;
  uint32_t* combineRankReadyEpochs_;
  uint32_t* combineReadyEpoch_;
  // Device-side rank-major round epoch. Kernels derive their epoch from it so CUDA graph replays, which reuse the
  // captured Workload, still see a fresh epoch every dispatch/combine round. Dispatch reads rankMajorLastEpoch_ and
  // publishes the new epoch in rankMajorRoundEpoch_; combine reads that and retires it into rankMajorLastEpoch_.
  uint32_t* rankMajorLastEpoch_;
  uint32_t* rankMajorRoundEpoch_;
  mscclpp::DeviceSyncer* combineSyncer_;
  int* rankMajorSendIndices_;
  // GPUNetIO cross-domain: cumulative per-source arrival baseline (dispatch recv
  // flag poll target, never reset) and persisted per-source recv counts (read by
  // the combine PUSH sender after the recvBuffer count packet is cleared).
  uint64_t* dispatchArrivedBaseline_;
  uint64_t* combineArrivedBaseline_;
  int* dispatchRecvCounts_;

  MSCCLPP_HOST_DEVICE_INLINE WorkspaceView(void* workspace, int nRanks, int nExperts) {
    auto* cursor = reinterpret_cast<int*>(workspace);
    // 8-byte-aligned field first (workspace base is allocation-aligned).
    dispatchArrivedBaseline_ = reinterpret_cast<uint64_t*>(cursor);
    cursor += static_cast<size_t>(nRanks) * (sizeof(uint64_t) / sizeof(int));
    combineArrivedBaseline_ = reinterpret_cast<uint64_t*>(cursor);
    cursor += static_cast<size_t>(nRanks) * (sizeof(uint64_t) / sizeof(int));
    dispatchRecvCounts_ = cursor;
    cursor += nRanks;
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
    combineRankReadyEpochs_ = reinterpret_cast<uint32_t*>(cursor);
    cursor += nRanks;
    combineReadyEpoch_ = reinterpret_cast<uint32_t*>(cursor++);
    rankMajorLastEpoch_ = reinterpret_cast<uint32_t*>(cursor++);
    rankMajorRoundEpoch_ = reinterpret_cast<uint32_t*>(cursor++);
    combineSyncer_ = reinterpret_cast<mscclpp::DeviceSyncer*>(cursor);
    cursor += sizeof(mscclpp::DeviceSyncer) / sizeof(int);
    rankMajorSendIndices_ = cursor;
  }

  MSCCLPP_HOST_DEVICE_INLINE static size_t numBytes(int nRanks, int nExperts, int maxTokensPerRank, int nTopk) {
    return static_cast<size_t>(nRanks) * sizeof(uint64_t) +  // dispatchArrivedBaseline_
           static_cast<size_t>(nRanks) * sizeof(uint64_t) +  // combineArrivedBaseline_
           static_cast<size_t>(nRanks) * sizeof(int) +       // dispatchRecvCounts_
           static_cast<size_t>(nRanks) * sizeof(int) +       // dispatchRankPayloadSlots_
           static_cast<size_t>(nRanks) * sizeof(int) +       // dispatchRankPayloadCompletions_
           sizeof(mscclpp::DeviceSemaphore) +                // dispatchLocalPayloadReady_
           static_cast<size_t>(nExperts) * sizeof(int) +     // dispatchExpertCopiedCounts_
           static_cast<size_t>(nRanks) * sizeof(uint32_t) +  // dispatchRankReadyEpochs_
           static_cast<size_t>(MaxWorkerBlocks) * sizeof(RecvTask) + sizeof(uint32_t) +  // dispatchTasksReadyEpoch_
           sizeof(int) +                                                                 // dispatchNumRecvTasks_
           static_cast<size_t>(nRanks) * sizeof(uint32_t) +                              // combineRankReadyEpochs_
           sizeof(uint32_t) +                                                            // combineReadyEpoch_
           sizeof(uint32_t) +                                                            // rankMajorLastEpoch_
           sizeof(uint32_t) +                                                            // rankMajorRoundEpoch_
           sizeof(mscclpp::DeviceSyncer) +                                               // combineSyncer_
           static_cast<size_t>(maxTokensPerRank) * nTopk * sizeof(int);                  // rankMajorSendIndices_
  }
};

// Epochs stay in [1, RankMajorEpochPeriod] so 0 never matches a live epoch and epoch * 2 + 1 cannot wrap.
constexpr uint32_t RankMajorEpochPeriod = 1u << 30;

#if defined(MSCCLPP_DEVICE_COMPILE)
// Each kernel reads one word and only block 0 writes the other, so no barrier or atomic is needed: stream order
// makes the previous kernel's write visible, and no block of the same kernel reads the word it writes.
MSCCLPP_DEVICE_INLINE uint32_t beginRankMajorDispatchEpoch(const WorkspaceView& workspace) {
  const uint32_t epoch = *workspace.rankMajorLastEpoch_ % RankMajorEpochPeriod + 1;
  if (blockIdx.x == 0 && threadIdx.x == 0) *workspace.rankMajorRoundEpoch_ = epoch;
  return epoch;
}

MSCCLPP_DEVICE_INLINE uint32_t beginRankMajorCombineEpoch(const WorkspaceView& workspace) {
  const uint32_t epoch = *workspace.rankMajorRoundEpoch_;
  if (blockIdx.x == 0 && threadIdx.x == 0) *workspace.rankMajorLastEpoch_ = epoch;
  return epoch;
}
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

struct KernelConfigCache {
  int deviceId_ = -1;
  size_t dynamicSharedBytes_ = 0;
  int residentBlocks_ = 0;
};

template <typename Kernel>
inline int configureKernel(Kernel kernel, int nThreads, size_t dynamicSharedBytes, const DeviceContext& context,
                           KernelConfigCache& cache) {
  if (cache.deviceId_ != context.deviceId_ || cache.dynamicSharedBytes_ < dynamicSharedBytes) {
    cudaFuncAttributes attributes;
    MSCCLPP_CUDATHROW(cudaFuncGetAttributes(&attributes, kernel));
    EP_HOST_ASSERT(dynamicSharedBytes + attributes.sharedSizeBytes <=
                   static_cast<size_t>(context.maxSharedMemoryPerBlock_));
    MSCCLPP_CUDATHROW(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           static_cast<int>(dynamicSharedBytes)));
    int blocksPerSm;
    MSCCLPP_CUDATHROW(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSm, kernel, nThreads, dynamicSharedBytes));
    cache.deviceId_ = context.deviceId_;
    cache.dynamicSharedBytes_ = dynamicSharedBytes;
    cache.residentBlocks_ = blocksPerSm * context.numSms_;
  }
  return cache.residentBlocks_;
}

MSCCLPP_HOST_DEVICE_INLINE size_t workspaceBytes(int nRanks, int nExperts, int maxTokensPerRank, int nTopk) {
  return WorkspaceView::numBytes(nRanks, nExperts, maxTokensPerRank, nTopk);
}

MSCCLPP_HOST_DEVICE_INLINE size_t dispatchSharedControlBytes(int nRanks) {
  constexpr int NSendSlots = DispatchMaxNWarpGroups * WARP_SIZE;
  const int nSlots = nRanks > NSendSlots ? nRanks : NSendSlots;
  const int nAggregatedCompletionSlots = DispatchMaxNWarpGroups * nRanks;
  return configAlign<size_t>(static_cast<size_t>(nSlots + nAggregatedCompletionSlots) * sizeof(int),
                             BufferAlignmentBytes);
}

template <int Hidden, DispatchDataType DataType, int ScaleBlockSize>
MSCCLPP_HOST_DEVICE_INLINE size_t dispatchSendTmaBytes(int nTopk) {
  return DispatchMaxNWarpGroups *
         (dispatchPayloadStride<DataType>(Hidden, nTopk, ScaleBlockSize) + sizeof(mscclpp::BulkBarrier));
}

template <int Hidden, typename ElementType, int MaxWorkers>
MSCCLPP_HOST_DEVICE_INLINE constexpr int tmaWorkerCount() {
  static_assert(Hidden % 128 == 0);
  constexpr size_t workerBytes = static_cast<size_t>(Hidden) * sizeof(ElementType) + sizeof(mscclpp::BulkBarrier);
  constexpr int nWorkers = static_cast<int>((OptimizedDynamicSharedMemoryBytes - TmaWorkerControlBytes) / workerBytes);
  return nWorkers < MaxWorkers ? nWorkers : MaxWorkers;
}

template <int Hidden, DispatchDataType DataType>
MSCCLPP_HOST_DEVICE_INLINE size_t dispatchRecvTmaBytes() {
  using ElementType = DispatchElementType<DataType>;
  constexpr int NWorkers = tmaWorkerCount<Hidden, ElementType, DispatchMaxNRecvTmaWorkers>();
  constexpr size_t tileBytes = static_cast<size_t>(Hidden) * sizeof(ElementType);
  return static_cast<size_t>(NWorkers) * (tileBytes + sizeof(mscclpp::BulkBarrier));
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

}  // namespace hybrid
}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_HYBRID_COMMON_LATENCY_CUH_
