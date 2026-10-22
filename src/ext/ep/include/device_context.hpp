// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_DEVICE_CONTEXT_HPP_
#define MSCCLPP_EP_DEVICE_CONTEXT_HPP_

#include <mscclpp/memory_channel_device.hpp>

namespace mscclpp {

// Forward declaration; full definition in the GPUNetIO device header.
struct GpuNetIoDeviceContext;

namespace ep {

struct BufferedTiming;
struct GpuNetIoDispatchPeerState;

constexpr int GpuNetIoRowFenceSystem = 0;
constexpr int GpuNetIoRowFenceGpu = 1;
constexpr int GpuNetIoRowFenceGpuPoster = 2;

/// Persistent device resources shared by every dispatch and combine algorithm.
struct DeviceContext {
  /// Local base used to translate local buffer addresses into peer mappings.
  void* localBufferBase_;
  /// Peer-mapped control or symmetric-buffer bases.
  void* const* peerBufferBases_;
  /// Optional peer-mapped payload-pool bases.
  void* const* peerPayloadBases_;
  /// Peer synchronization channels.
  mscclpp::BaseMemoryChannelDeviceHandle* channels_;
  /// Optional algorithm workspace.
  void* workspace_;
  /// Optional token/rank receive indices used by throughput combine.
  int* combineRecvIdx_;
  /// Optional mapped receive count used by throughput preparation.
  int* mappedRecvCounter_;
  /// Optional mapped per-expert receive counts used by throughput preparation.
  int* mappedRecvExpertCounters_;
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
  /// GPU-initiated networking (GPUNetIO/GDAKI) device context for peers outside
  /// this rank's NVLink/IPC domain, or nullptr when every peer is NVLink-mapped
  /// (single-domain) or the GPUNetIO backend is disabled.
  mscclpp::GpuNetIoDeviceContext* gpuNetIo_ = nullptr;
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
  /// Expanded IPC specialization, collectively enabled when all peers are mapped.
  bool expandedIpcFastPath_ = false;
  /// Expanded network specialization, collectively enabled only with active GPUNetIO.
  bool expandedGpuNetIoFastPath_ = false;
  /// Persistent device copy used by kernel launches. Host launch code only.
  DeviceContext* devicePtr_ = nullptr;
  /// Optional local-only diagnostic storage, never registered with peers.
  BufferedTiming* timing_ = nullptr;
  /// Pack compact dispatch tokens by remote rank before bulk publication.
  bool gpuNetIoBatchedDispatch_ = false;
  /// Stripe batched dispatch across every peer QP instead of one QP per HCA.
  bool gpuNetIoBatchAllQps_ = false;
  /// Rows per pipelined destination chunk; zero retains whole-destination batching.
  int gpuNetIoPipelineChunkRows_ = 0;
  /// Maximum helper warps per remote destination row; zero keeps leader-serial staging.
  int gpuNetIoParallelStage_ = 0;
  /// Batched staged-row fence: GpuNetIoRowFenceSystem, GpuNetIoRowFenceGpu or GpuNetIoRowFenceGpuPoster.
  int gpuNetIoRowFence_ = 0;
  /// Local completed-row counters, reset on the dispatch stream before every invocation.
  int* gpuNetIoChunkReady_ = nullptr;
  /// Early tail post (MSCCLPP_EP_GPUNETIO_EARLY_TAIL=1): [rank] rows+1 words published by the control CTA once row
  /// counts are known; located right before gpuNetIoChunkReady_ and reset by the same memset.
  int* gpuNetIoTailRows_ = nullptr;
  /// Stage token rows with TMA bulk shared->global stores (MSCCLPP_EP_GPUNETIO_TMA_STAGE=1).
  int gpuNetIoTmaStage_ = 0;  // 1=TMA, 2=debug: skip payload copy
  /// Zero-copy dispatch (MSCCLPP_EP_GPUNETIO_ZERO_COPY=1): RDMA gathers token rows straight from the registered
  /// input tensor; staging only keeps top-k metadata. RowSrc maps (destination, slot) to the source token.
  int gpuNetIoZeroCopy_ = 0;
  uintptr_t gpuNetIoZcBase_ = 0;
  const uint32_t* gpuNetIoZcLkeys_ = nullptr;
  int* gpuNetIoZcRowSrc_ = nullptr;
  /// Local completed-producer count, reset before each compact GPUNetIO dispatch.
  int* gpuNetIoDispatchDone_ = nullptr;
  /// Select dedicated receive CTAs and destination-local network completion.
  bool gpuNetIoAsyncDispatch_ = false;
  /// Control-block threads own (peer, QP) pairs and post ready staged rows with single-poster submits;
  /// gpuNetIoChunkReady_ then holds one ready flag per (destination, row).
  bool gpuNetIoDedicatedPoster_ = false;
  /// Rank-major dispatch does not wait for its send completions at the end of the kernel; it records
  /// the per-(peer, QP) send tickets instead, and the next dispatch waits for them before rewriting
  /// any registered send source (default on with batched dispatch; MSCCLPP_EP_GPUNETIO_DEFER_FLUSH=0 disables).
  bool gpuNetIoDeferFlush_ = false;
  /// [peer * GpuNetIoMaxQpsPerPeer + qp] send tickets of the previous dispatch (deferred flush only).
  uint64_t* gpuNetIoDispatchTickets_ = nullptr;
  /// Rank-major GPUNetIO combine pipeline (MSCCLPP_EP_GPUNETIO_COMBINE_CHUNK_ROWS): rows per combine chunk, striped
  /// round-robin over every peer QP; each chunk carries its own completion signal so the receiver reduces a token as
  /// soon as the chunk holding its row lands. Zero keeps one whole-peer put per HCA stripe.
  int gpuNetIoCombineChunkRows_ = 0;
  /// [peer * GpuNetIoMaxQpsPerPeer + qp] cumulative combine chunks already consumed (pipelined combine only).
  uint64_t* gpuNetIoCombineChunkBase_ = nullptr;
  /// Skip the grid barrier between the combine ready publisher and the TMA workers
  /// (MSCCLPP_EP_GPUNETIO_COMBINE_NO_BARRIER=1); implied by the combine pipeline.
  bool gpuNetIoCombineNoBarrier_ = false;
  /// Local destination counts and post-completion state, reset before graph replay.
  GpuNetIoDispatchPeerState* gpuNetIoDispatchPeers_ = nullptr;
};

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_DEVICE_CONTEXT_HPP_
