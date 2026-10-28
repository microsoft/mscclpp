// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_DEVICE_CONTEXT_HPP_
#define MSCCLPP_EP_DEVICE_CONTEXT_HPP_

#include <mscclpp/memory_channel_device.hpp>

namespace mscclpp {
namespace ep {

struct EpGpuNetIoDeviceContext;
struct GpuNetIoDispatchPeerState;
struct GpuNetIoDistributeControl;
struct GpuNetIoAggregateControl;

constexpr int GpuNetIoRowFenceSystem = 0;
constexpr int GpuNetIoRowFenceGpu = 1;
constexpr int GpuNetIoRowFenceGpuPoster = 2;

/// Persistent device resources shared by every dispatch and combine algorithm.
struct DeviceContext {
  /// Local base used to translate local buffer addresses into peer mappings.
  void* localBufferBase_;
  /// Peer mappings of the same symmetric allocation.
  void* const* peerBufferBases_;
  /// Peer synchronization channels.
  mscclpp::BaseMemoryChannelDeviceHandle* channels_;
  /// Optional algorithm workspace.
  void* workspace_;
  /// Maximum dynamic shared memory available to one block.
  int maxSharedMemoryPerBlock_;
  /// Number of SMs on the device.
  int numSms_;
  /// CUDA device ID.
  int deviceId_;
  /// Local rank.
  int rank_;
  /// Number of ranks.
  int numRanks_;
  /// Persistent device copy used by kernel launches. Host launch code only.
  DeviceContext* devicePtr_ = nullptr;
  /// GPU-initiated networking (GPUNetIO/GDAKI) device context for peers outside
  /// this rank's NVLink/IPC domain, or nullptr when every peer is NVLink-mapped
  /// (single-domain) or the GPUNetIO backend is disabled.
  EpGpuNetIoDeviceContext* gpuNetIo_ = nullptr;
  /// Symmetric inter-domain send-staging ring base (GpuNetIoStagingSlots slots).
  void* gpuNetIoStagingBuffer_ = nullptr;
  /// Symmetric per-source-rank, per-QP dispatch completion flag array base.
  void* gpuNetIoFlagsBuffer_ = nullptr;
  /// Symmetric per-peer-rank combine completion flag array base (kept separate
  /// from gpuNetIoFlagsBuffer_ so dispatch and combine signal counts never alias).
  void* gpuNetIoCombineFlagsBuffer_ = nullptr;
  /// Symmetric payload-only combine landing rows, null unless active.
  void* gpuNetIoCombineLandingBuffer_ = nullptr;
  /// Byte stride of one staging-ring slot (token + top-k metadata).
  size_t gpuNetIoSlotStride_ = 0;
  /// Pack compact dispatch tokens by remote rank before bulk publication.
  bool gpuNetIoBatchedDispatch_ = false;
  /// Rows per pipelined destination chunk; whole-destination batching uses the allocation capacity.
  int gpuNetIoPipelineChunkRows_ = 0;
  /// Maximum helper warps per remote destination row; zero keeps leader-serial staging.
  int gpuNetIoParallelStage_ = 0;
  /// Batched staged-row fence: GpuNetIoRowFenceSystem, GpuNetIoRowFenceGpu or GpuNetIoRowFenceGpuPoster.
  int gpuNetIoRowFence_ = 0;
  /// Local completed-row counters, reset on the dispatch stream before every invocation.
  int* gpuNetIoChunkReady_ = nullptr;
  /// Stage token rows with TMA bulk shared->global stores (MSCCLPP_EP_GPUNETIO_TMA_STAGE=1; 2 = debug: skip the
  /// payload copy).
  int gpuNetIoTmaStage_ = 0;
  /// Local completed-producer count, reset before each compact GPUNetIO dispatch.
  int* gpuNetIoDispatchDone_ = nullptr;
  /// Unbatched rank-major dispatch may record per-QP send tickets instead of draining sends.
  /// The next dispatch waits for these tickets before rewriting registered sources.
  /// Batched dispatch always drains its sends.
  bool gpuNetIoDeferFlush_ = false;
  /// [peer * GpuNetIoMaxQpsPerPeer + qp] send tickets of the previous dispatch (deferred flush only).
  uint64_t* gpuNetIoDispatchTickets_ = nullptr;
  /// Rank-major GPUNetIO combine pipeline (MSCCLPP_EP_GPUNETIO_COMBINE_CHUNK_ROWS): rows per combine chunk, striped
  /// round-robin over every peer QP; each chunk carries its own completion signal so the receiver reduces a token as
  /// soon as the chunk holding its row lands. Zero keeps one whole-peer put per HCA stripe.
  int gpuNetIoCombineChunkRows_ = 0;
  /// [peer * GpuNetIoMaxQpsPerPeer + qp] cumulative combine chunks already consumed (pipelined combine only).
  uint64_t* gpuNetIoCombineChunkBase_ = nullptr;
  /// Local destination counts and post-completion state, reset before graph replay.
  GpuNetIoDispatchPeerState* gpuNetIoDispatchPeers_ = nullptr;
  /// Optional single-copy dispatch to the paired gateway in the other NVLink domain.
  bool gpuNetIoSendDistribute_ = false;
  bool gpuNetIoDistributeStats_ = false;
  int gpuNetIoGateway_ = -1;
  int gpuNetIoDistributeCapacity_ = 0;
  void* gpuNetIoDistributeSend_ = nullptr;
  void* gpuNetIoDistributeRecv_ = nullptr;
  GpuNetIoDistributeControl* gpuNetIoDistributeControl_ = nullptr;
  /// Local [compact row allocator, completed forward CTAs, per-chunk staged rows].
  int* gpuNetIoDistributeScratch_ = nullptr;
  /// Optional reverse-gateway FP32 aggregation; dispatch caches live until the next local dispatch.
  bool gpuNetIoAggregateCombine_ = false;
  int* gpuNetIoAggregateTokenRows_ = nullptr;
  int* gpuNetIoAggregateIncomingRows_ = nullptr;
  /// Private [compact row, IDs then destination slots] cache, never exposed to remote writes.
  int* gpuNetIoAggregateRoutes_ = nullptr;
  float* gpuNetIoAggregateLocal_ = nullptr;
  /// Independent registered FP32 send/landing rows, indexed by compact remote row.
  float* gpuNetIoAggregateSend_ = nullptr;
  float* gpuNetIoAggregateRecv_ = nullptr;
  GpuNetIoAggregateControl* gpuNetIoAggregateControl_ = nullptr;
  /// Private completed-row counters, reset on the caller stream before each aggregate combine.
  int* gpuNetIoAggregateChunkReady_ = nullptr;
};

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
  bool gpuNetIoDeferFlush_;
  uint64_t* gpuNetIoDispatchTickets_;
  int gpuNetIoPipelineChunkRows_;
  int gpuNetIoParallelStage_;
  int gpuNetIoRowFence_;
  int* gpuNetIoChunkReady_;
  int gpuNetIoTmaStage_;
  GpuNetIoDispatchPeerState* gpuNetIoDispatchPeers_;
  int gpuNetIoCombineChunkRows_;
  uint64_t* gpuNetIoCombineChunkBase_;

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
        gpuNetIoDeferFlush_(context->gpuNetIoDeferFlush_),
        gpuNetIoDispatchTickets_(context->gpuNetIoDispatchTickets_),
        gpuNetIoPipelineChunkRows_(context->gpuNetIoPipelineChunkRows_),
        gpuNetIoParallelStage_(context->gpuNetIoParallelStage_),
        gpuNetIoRowFence_(context->gpuNetIoRowFence_),
        gpuNetIoChunkReady_(context->gpuNetIoChunkReady_),
        gpuNetIoTmaStage_(context->gpuNetIoTmaStage_),
        gpuNetIoDispatchPeers_(context->gpuNetIoDispatchPeers_),
        gpuNetIoCombineChunkRows_(context->gpuNetIoCombineChunkRows_),
        gpuNetIoCombineChunkBase_(context->gpuNetIoCombineChunkBase_) {}

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

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_DEVICE_CONTEXT_HPP_
