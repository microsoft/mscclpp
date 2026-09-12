// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
// Adapted from feature/ep-experimental 9deec055 and 63fea77:
// dispatch/common.cuh and combine/common.cuh. Keep the old GPUNetIO runtime ABI.
// Upstream warp-group TMA scatter, dense row-aligned metadata, and weighted
// contributor-ready TMA gather are retained. Remote peers use registered PUSH
// staging/landing and fixed ordered markers rather than mapped pointer reads.

#include "config.cuh"
#include "exception.cuh"
#include "topk_expanded.cuh"

#if defined(MSCCLPP_USE_GPUNETIO)
#include <mscclpp/port_channel_gpunetio_device.hpp>
#endif

namespace mscclpp::ep::low_latency::topk_expanded {
namespace {
using namespace detail;
constexpr int CombineNThreads = 32 * WARP_SIZE;
constexpr int RankMajorTmaMaxNTopk = 8;
constexpr int CombineMaxNTopk = 9;

MSCCLPP_HOST_DEVICE_INLINE bool validExpert(int64_t expert, int nExperts) { return expert >= 0 && expert < nExperts; }

size_t dispatchControlBytes(int ranks) {
  const int sendSlots = std::max(ranks, DispatchMaxNWarpGroups * WARP_SIZE);
  return configAlign<size_t>((sendSlots + DispatchMaxNWarpGroups * ranks) * sizeof(int), 128);
}

template <int Hidden>
size_t dispatchSharedBytes(int ranks, int topk) {
  const size_t stride = dispatchPayloadStride<DispatchDataType::BF16>(Hidden, topk, 0);
  return std::max(dispatchControlBytes(ranks) + DispatchMaxNWarpGroups * (stride + sizeof(mscclpp::BulkBarrier)),
                  static_cast<size_t>(ranks) * sizeof(int));
}

template <int Hidden>
size_t combineSharedBytes(int topk) {
  if (topk > RankMajorTmaMaxNTopk) return 0;
  constexpr size_t bytes =
      RankMajorTmaMaxNTopk * (Hidden * sizeof(Bf16) + sizeof(mscclpp::BulkBarrier) + sizeof(int) + sizeof(float));
  static_assert(bytes <= OptimizedDynamicSharedMemoryBytes);
  return configAlign<size_t>(bytes, 128);
}

#if MSCCLPP_BULK_AVAILABLE
__device__ void validateGpuNetIo(const TransportView& transport) {
#if defined(MSCCLPP_USE_GPUNETIO)
  if (transport.gpuNetIo_ != nullptr) {
    const int qps = transport.gpuNetIo_->numQpsPerPeer;
    const int hcas = transport.gpuNetIo_->numHcas;
    // Release-build guard: invalid geometry must not index adjacent source flags.
    if (qps <= 0 || qps > GpuNetIoMaxQpsPerPeer || hcas <= 0 || hcas > qps || qps % hcas != 0) __trap();
  }
#endif
}

__device__ void signalLocal(uint64_t* flag) {
  mscclpp::atomicFetchAdd<uint64_t, mscclpp::scopeSystem>(flag, 1, mscclpp::memoryOrderRelease);
}

__device__ bool flagReady(uint64_t* flag, uint64_t target) {
  return mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(flag, mscclpp::memoryOrderAcquire) >= target;
}

__device__ int markerCount(const TransportView& transport, int peer, bool combine) {
#if defined(MSCCLPP_USE_GPUNETIO)
  if (!transport.isNvlinkPeer(peer)) return combine ? transport.gpuNetIo_->numHcas : transport.gpuNetIo_->numQpsPerPeer;
#endif
  return 1;
}

__device__ int markerQp(const TransportView& transport, int owner, int stripe) {
#if defined(MSCCLPP_USE_GPUNETIO)
  if (transport.gpuNetIo_ != nullptr)
    return (owner % transport.gpuNetIo_->numQpsPerPeer + stripe) % transport.gpuNetIo_->numQpsPerPeer;
#endif
  return 0;
}

__device__ bool sourceReady(const TransportView& transport, uint64_t* flags, int source, uint64_t target,
                            bool combine) {
  bool ready = true;
  for (int stripe = 0; stripe < markerCount(transport, source, combine); ++stripe) {
    const int q =
        transport.isNvlinkPeer(source) ? 0 : (combine ? markerQp(transport, transport.rank_, stripe) : stripe);
    ready &= flagReady(flags + static_cast<size_t>(source) * GpuNetIoMaxQpsPerPeer + q, target);
  }
  return ready;
}

__device__ void waitSource(const TransportView& transport, uint64_t* flags, int source, uint64_t target, bool combine) {
  while (!sourceReady(transport, flags, source, target, combine)) {
  }
}

// Called only by block zero AFTER all local readers/writers/drains reached a
// grid barrier. This ACK is necessary because expert output aliases dispatch
// storage: no peer may overwrite that storage while another still reads it.
// Device generations advance on replay; workload.epoch_ is deliberately unused.
__device__ void finishCollective(const TransportView& transport, const Layout& layout, int ranks) {
  auto* epoch = static_cast<uint64_t*>(layout.expandedSyncEpoch_);
  if (threadIdx.x == 0) ++*epoch;
  __syncthreads();
  const uint64_t target = *epoch;
  auto* flags = static_cast<uint64_t*>(layout.expandedSyncFlags_);
  for (int peer = threadIdx.x; peer < ranks; peer += blockDim.x) {
    if (transport.isNvlinkPeer(peer)) {
      auto* remote = static_cast<uint64_t*>(transport.mappedBuffer(flags, peer));
      signalLocal(remote + transport.rank_);
    } else {
#if defined(MSCCLPP_USE_GPUNETIO)
      transport.gpuNetIo_->atomicAdd(peer, transport.symmetricOffset(flags + transport.rank_), 1, 0);
#endif
    }
  }
  __syncthreads();
  for (int peer = threadIdx.x; peer < ranks; peer += blockDim.x) {
    while (!flagReady(flags + peer, target)) {
    }
#if defined(MSCCLPP_USE_GPUNETIO)
    if (!transport.isNvlinkPeer(peer)) transport.gpuNetIo_->flush(peer, 0);
#endif
  }
  __syncthreads();
}

template <int Hidden>
struct RankMajorSendState {
  int laneId_, warpGroupId_, tokenStride_, firstTokenIdx_;
  uint8_t* stagedToken_;
  mscclpp::BulkBarrier* bulkBarrier_;
  uint32_t bulkPhase_;
};

template <int Hidden>
__device__ bool initRankMajorSendState(RankMajorSendState<Hidden>& state, int nTokens, int nTopk, int nRanks,
                                       int nPayloadBlocks, int* sharedMem) {
  if (blockIdx.x == 0 || static_cast<int>(blockIdx.x) > nPayloadBlocks) return false;
  const int warpId = threadIdx.x / WARP_SIZE;
  const int nWarpsPerGroup = dispatchNWarpsPerGroup(nTokens, nPayloadBlocks);
  const int nWarpGroups = DispatchNWarps / nWarpsPerGroup;
  if (warpId % nWarpsPerGroup != 0) return false;
  const int sendSlots = nRanks > DispatchMaxNWarpGroups * WARP_SIZE ? nRanks : DispatchMaxNWarpGroups * WARP_SIZE;
  const size_t control = configAlign<size_t>((sendSlots + DispatchMaxNWarpGroups * nRanks) * sizeof(int), 128);
  const size_t stride = dispatchPayloadStride<DispatchDataType::BF16>(Hidden, nTopk, 0);
  auto* sharedTokenBase = reinterpret_cast<uint8_t*>(sharedMem) + control;
  auto* barriers = reinterpret_cast<mscclpp::BulkBarrier*>(sharedTokenBase + DispatchMaxNWarpGroups * stride);
  state.laneId_ = get_lane_id();
  state.warpGroupId_ = warpId / nWarpsPerGroup;
  state.tokenStride_ = nPayloadBlocks * nWarpGroups;
  state.firstTokenIdx_ = (static_cast<int>(blockIdx.x) - 1) * nWarpGroups + state.warpGroupId_;
  state.stagedToken_ = sharedTokenBase + state.warpGroupId_ * stride;
  state.bulkBarrier_ = barriers + state.warpGroupId_;
  state.bulkPhase_ = 0;
  if (state.firstTokenIdx_ < nTokens && state.laneId_ == 0) state.bulkBarrier_->init();
  return true;
}

// Upstream dense metadata publication and per-warp-group completion aggregation.
// Only payloads to valid selected slots are scattered; duplicate experts retain
// separate rows. Check the upper bound before narrowing int64 IDs.
template <int Hidden>
__device__ void dispatchSendRankMajorTopkExpandedBf16(void* output, int* outputIds, float* outputWeights,
                                                      const void* input, const int64_t* topkIds, const float* weights,
                                                      const Workload& work, const TransportView& transport,
                                                      const Layout& layout, WorkspaceView& workspace, int ranks,
                                                      int* sharedMem) {
  RankMajorSendState<Hidden> send;
  if (!initRankMajorSendState(send, work.numTokens_, work.numTopk_, ranks, gridDim.x - DispatchControlBlocks,
                              sharedMem))
    return;
  const int lane = send.laneId_;
  const int topk = work.numTopk_;
  const int localExperts = work.numExperts_ / ranks;
  const int sendSlots = ranks > DispatchMaxNWarpGroups * WARP_SIZE ? ranks : DispatchMaxNWarpGroups * WARP_SIZE;
  auto* completions = sharedMem + sendSlots + send.warpGroupId_ * ranks;
  for (int peer = lane; peer < ranks; peer += WARP_SIZE) completions[peer] = 0;
  __syncwarp();
  constexpr size_t bytes = Hidden * sizeof(Bf16);
  [[maybe_unused]] constexpr int vectors = bytes / sizeof(int4);
  const size_t rowsPerRank = static_cast<size_t>(work.maxTokensPerRank_) * topk;
  [[maybe_unused]] int tokensSinceFlush = 0;
  for (int token = send.firstTokenIdx_; token < work.numTokens_; token += send.tokenStride_) {
    if (lane == 0) {
      send.bulkBarrier_->arriveAndExpect(bytes);
      mscclpp::bulkLoad(send.stagedToken_, static_cast<const uint8_t*>(input) + token * bytes, bytes,
                        *send.bulkBarrier_);
    }
    const size_t selection = static_cast<size_t>(token) * topk + lane;
    const int64_t expert = lane < topk ? topkIds[selection] : -1;
    const int destination = validExpert(expert, work.numExperts_) ? static_cast<int>(expert / localExperts) : -1;
    const float weight = lane < topk ? (weights == nullptr ? 1.0f : weights[selection]) : 0.0f;
    const size_t row = static_cast<size_t>(transport.rank_) * rowsPerRank + selection;
    if (lane == 0) send.bulkBarrier_->wait(send.bulkPhase_);
    __syncwarp();
    mscclpp::bulkFence();
    if (lane < topk) {
      for (int peer = 0; peer < ranks; ++peer) {
        const bool local = peer == destination;
        int* ids;
        float* wgts;
        size_t offset;
        if (transport.isNvlinkPeer(peer)) {
          ids = static_cast<int*>(transport.mappedBuffer(outputIds, peer));
          wgts = static_cast<float*>(transport.mappedBuffer(outputWeights, peer));
          offset = row;
        } else {
          ids = static_cast<int*>(layout.expandedSendIds_);
          wgts = static_cast<float*>(layout.expandedSendWeights_);
          offset = static_cast<size_t>(peer) * rowsPerRank + selection;
        }
        ids[offset] = local ? static_cast<int>(expert) : work.invalidTokenExpertId_;
        wgts[offset] = local ? weight : 0.0f;
      }
    }
    if (destination >= 0 && transport.isNvlinkPeer(destination)) {
      auto* remote = static_cast<uint8_t*>(transport.mappedBuffer(output, destination)) + row * bytes;
      mscclpp::bulkStore(remote, send.stagedToken_, bytes);
      mscclpp::bulkStoreCommit();
      mscclpp::bulkStoreWait();
    }
#if defined(MSCCLPP_USE_GPUNETIO)
    const bool remote = destination >= 0 && !transport.isNvlinkPeer(destination);
    // Every lane executes this collective, including invalid/non-top-k lanes.
    if (__any_sync(0xffffffff, remote)) {
      auto* staged = reinterpret_cast<int4*>(static_cast<uint8_t*>(layout.gpuNetIoStagingBuffer_) +
                                             static_cast<size_t>(token) * layout.gpuNetIoSlotStride_);
      const auto* shared = reinterpret_cast<const int4*>(send.stagedToken_);
      for (int v = lane; v < vectors; v += WARP_SIZE) staged[v] = shared[v];
      __syncwarp();
      __threadfence_system();
      __syncwarp();
      if (remote) {
        const int q = static_cast<int>(expert % localExperts) % transport.gpuNetIo_->numQpsPerPeer;
        transport.gpuNetIo_->put(destination, transport.symmetricOffset(static_cast<uint8_t*>(output) + row * bytes),
                                 transport.symmetricOffset(staged), bytes, q);
      }
      __syncwarp();
    }
    // Bounded SQ pressure for capacities larger than one token per warp group.
    if (transport.gpuNetIo_ != nullptr && ++tokensSinceFlush >= GpuNetIoFlushInterval) {
      __syncwarp();
      for (int p = 0; p < ranks; ++p) {
        if (transport.isNvlinkPeer(p)) continue;
        for (int q = lane; q < transport.gpuNetIo_->numQpsPerPeer; q += WARP_SIZE) transport.gpuNetIo_->flush(p, q);
      }
      __syncwarp();
      tokensSinceFlush = 0;
    }
#endif
    if (lane < topk) {
      for (int peer = 0; peer < ranks; ++peer)
        if (transport.isNvlinkPeer(peer)) atomicAdd_block(completions + peer, 1);
    }
    __syncwarp();
  }
  __threadfence_system();
  __syncwarp();
  for (int peer = lane; peer < ranks; peer += WARP_SIZE) {
    if (completions[peer])
      mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(workspace.dispatchRankPayloadCompletions_ + peer,
                                                         completions[peer], mscclpp::memoryOrderRelease);
  }
}

__device__ void dispatchRankMajorTopkExpandedNotify(int* outputIds, float* outputWeights, const int64_t* topkIds,
                                                    const Workload& work, const TransportView& transport,
                                                    const Layout& layout, WorkspaceView& workspace, int ranks,
                                                    int* counts) {
  const int lane = get_lane_id();
  const int warp = threadIdx.x / WARP_SIZE;
  const int topk = work.numTopk_;
  const int localExperts = work.numExperts_ / ranks;
  for (int peer = threadIdx.x; peer < ranks; peer += blockDim.x) counts[peer] = 0;
  __syncthreads();
  for (int token = warp; token < work.numTokens_; token += DispatchNWarps) {
    const int64_t expert = lane < topk ? topkIds[static_cast<size_t>(token) * topk + lane] : -1;
    if (validExpert(expert, work.numExperts_)) atomicAdd_block(counts + expert / localExperts, 1);
  }
  __syncthreads();
  const size_t rows = static_cast<size_t>(work.maxTokensPerRank_) * topk;
  for (int peer = 0; peer < ranks; ++peer) {
    const bool mapped = transport.isNvlinkPeer(peer);
    auto* ids = mapped ? static_cast<int*>(transport.mappedBuffer(outputIds, peer))
                       : static_cast<int*>(layout.expandedSendIds_);
    auto* wgts = mapped ? static_cast<float*>(transport.mappedBuffer(outputWeights, peer))
                        : static_cast<float*>(layout.expandedSendWeights_);
    const size_t base = static_cast<size_t>(mapped ? transport.rank_ : peer) * rows;
    for (size_t i = static_cast<size_t>(work.numTokens_) * topk + threadIdx.x; i < rows; i += blockDim.x) {
      ids[base + i] = work.invalidTokenExpertId_;
      wgts[base + i] = 0.0f;
    }
  }
  __threadfence_system();
  __syncthreads();
  auto* flags = static_cast<uint64_t*>(layout.gpuNetIoFlagsBuffer_);
  for (int peer = threadIdx.x; peer < ranks; peer += blockDim.x) {
    if (transport.isNvlinkPeer(peer)) {
      const int expected = work.numTokens_ * topk;
      while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(workspace.dispatchRankPayloadCompletions_ + peer,
                                                            mscclpp::memoryOrderAcquire) != expected) {
      }
      workspace.dispatchRankPayloadCompletions_[peer] = 0;
      auto* remoteCount = static_cast<int*>(transport.mappedBuffer(layout.expandedCounts_, peer));
      remoteCount[transport.rank_] = counts[peer];
      __threadfence_system();
      auto* remoteFlags = static_cast<uint64_t*>(transport.mappedBuffer(flags, peer));
      // Even all-empty sources publish one generation: padding has changed.
      signalLocal(remoteFlags + static_cast<size_t>(transport.rank_) * GpuNetIoMaxQpsPerPeer);
    } else {
#if defined(MSCCLPP_USE_GPUNETIO)
      auto* staged = static_cast<int*>(layout.expandedCountStaging_) + peer;
      *staged = counts[peer];
      __threadfence_system();
      auto* remote = static_cast<int*>(layout.expandedCounts_) + transport.rank_;
      transport.gpuNetIo_->put(peer, transport.symmetricOffset(remote), transport.symmetricOffset(staged), sizeof(int),
                               0);
#endif
    }
  }
}

__device__ void postRemoteDispatchMetadataAndMarkers(const TransportView& transport, const Layout& layout,
                                                     const Workload& work, int ranks) {
#if defined(MSCCLPP_USE_GPUNETIO)
  auto* gin = transport.gpuNetIo_;
  if (gin == nullptr) return;
  const size_t rows = static_cast<size_t>(work.maxTokensPerRank_) * work.numTopk_;
  for (int peer = threadIdx.x; peer < ranks; peer += blockDim.x) {
    if (transport.isNvlinkPeer(peer)) continue;
    const size_t source = static_cast<size_t>(peer) * rows;
    const size_t dest = static_cast<size_t>(transport.rank_) * rows;
    gin->put(peer, transport.symmetricOffset(static_cast<int*>(layout.rankMajorTopkIdsBuffer_) + dest),
             transport.symmetricOffset(static_cast<int*>(layout.expandedSendIds_) + source), rows * sizeof(int), 0);
    gin->put(peer, transport.symmetricOffset(static_cast<float*>(layout.rankMajorTopkWeightsBuffer_) + dest),
             transport.symmetricOffset(static_cast<float*>(layout.expandedSendWeights_) + source), rows * sizeof(float),
             0);
  }
  __syncthreads();
  auto* flags = static_cast<uint64_t*>(layout.gpuNetIoFlagsBuffer_) +
                static_cast<size_t>(transport.rank_) * GpuNetIoMaxQpsPerPeer;
  for (int i = threadIdx.x; i < ranks * gin->numQpsPerPeer; i += blockDim.x) {
    const int peer = i / gin->numQpsPerPeer, q = i % gin->numQpsPerPeer;
    if (!transport.isNvlinkPeer(peer)) gin->atomicAdd(peer, transport.symmetricOffset(flags + q), 1, q);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < ranks * gin->numQpsPerPeer; i += blockDim.x) {
    const int peer = i / gin->numQpsPerPeer, q = i % gin->numQpsPerPeer;
    if (!transport.isNvlinkPeer(peer)) gin->flush(peer, q);
  }
#endif
}
#endif  // MSCCLPP_BULK_AVAILABLE

template <int Hidden>
__global__ __launch_bounds__(DispatchNThreads,
                             1) void dispatchTopkExpandedKernel(void* output, int* outputIds, float* outputWeights,
                                                                int* outputCount, const void* input,
                                                                const int64_t* topkIds, const float* weights,
                                                                Workload work, CommContext comm, void* workspace) {
#if MSCCLPP_BULK_AVAILABLE
  extern __shared__ __align__(128) uint8_t shared[];
  const TransportView transport(comm);
  validateGpuNetIo(transport);
  WorkspaceView state(workspace, comm.numRanks_, work.numExperts_);
  const Layout layout(comm.symmetricBufferBase_, work.maxTokensPerRank_, Hidden, comm.numRanks_, work.numExperts_,
                      work.numTopk_, true, true);
  const uint64_t target = state.dispatchArrivedBaseline_[comm.rank_] + 1;
  if (blockIdx.x < gridDim.x - 1)
    dispatchSendRankMajorTopkExpandedBf16<Hidden>(output, outputIds, outputWeights, input, topkIds, weights, work,
                                                  transport, layout, state, comm.numRanks_,
                                                  reinterpret_cast<int*>(shared));
  else
    dispatchRankMajorTopkExpandedNotify(outputIds, outputWeights, topkIds, work, transport, layout, state,
                                        comm.numRanks_, reinterpret_cast<int*>(shared));
  // Registered metadata is complete and every payload WQE has been posted.
  __threadfence_system();
  state.combineSyncer_->sync(gridDim.x);
  if (blockIdx.x == gridDim.x - 1) postRemoteDispatchMetadataAndMarkers(transport, layout, work, comm.numRanks_);
  if (blockIdx.x < comm.numRanks_ && threadIdx.x == 0) {
    waitSource(transport, static_cast<uint64_t*>(layout.gpuNetIoFlagsBuffer_), blockIdx.x, target, false);
    outputCount[blockIdx.x] = static_cast<int*>(layout.expandedCounts_)[blockIdx.x];
  }
  state.combineSyncer_->sync(gridDim.x);
  if (blockIdx.x == 0) {
    finishCollective(transport, layout, comm.numRanks_);
    if (threadIdx.x == 0) state.dispatchArrivedBaseline_[comm.rank_] = target;
  }
  state.combineSyncer_->sync(gridDim.x);
#endif
}

#if MSCCLPP_BULK_AVAILABLE
__device__ const uint8_t* expandedRow(const void* input, const Layout& layout, const TransportView& transport,
                                      int source, int token, int slot, int capacity, int topk, size_t bytes) {
  const bool mapped = transport.isNvlinkPeer(source);
  const auto* base = mapped ? static_cast<const uint8_t*>(transport.mappedBuffer(const_cast<void*>(input), source))
                            : static_cast<const uint8_t*>(layout.gpuNetIoCombineLandingBuffer_);
  const int rowRank = mapped ? transport.rank_ : source;
  return base + ((static_cast<size_t>(rowRank) * capacity + token) * topk + slot) * bytes;
}

// Upstream contributor-ready TMA gather. Only the source address and readiness
// predicate differ for GPUNetIO landing rows. A skipped lane retains its phase.
template <int Hidden>
__device__ void recvRankMajorTopkExpandedRemotePartialsTma(void* output, const void* input, const int64_t* topkIds,
                                                           const float* weights, const Workload& work,
                                                           const TransportView& transport, const Layout& layout,
                                                           int ranks, uint64_t target, uint8_t* shared) {
  constexpr size_t bytes = Hidden * sizeof(Bf16);
  constexpr int vectors = bytes / sizeof(int4);
  constexpr int pairsPerVector = sizeof(int4) / sizeof(mscclpp::bf16x2);
  auto* sharedRows = reinterpret_cast<int4*>(shared);
  auto* barriers = reinterpret_cast<mscclpp::BulkBarrier*>(shared + RankMajorTmaMaxNTopk * bytes);
  auto* validRows = reinterpret_cast<int*>(barriers + RankMajorTmaMaxNTopk);
  auto* slotWeights = reinterpret_cast<float*>(validRows + RankMajorTmaMaxNTopk);
  const int lane = get_lane_id(), warp = threadIdx.x / WARP_SIZE;
  const int localExperts = work.numExperts_ / ranks;
  auto* flags = static_cast<uint64_t*>(layout.gpuNetIoCombineFlagsBuffer_);
  uint32_t phase = 0;
  if (warp == 0) {
    if (lane < RankMajorTmaMaxNTopk) barriers[lane].init();
    __syncwarp();
  }
  for (int token = blockIdx.x - 1; token < work.numTokens_; token += gridDim.x - 1) {
    if (warp == 0) {
      const size_t index = static_cast<size_t>(token) * work.numTopk_ + lane;
      const int64_t expert = lane < work.numTopk_ ? topkIds[index] : -1;
      const float weight = validExpert(expert, work.numExperts_) ? (weights == nullptr ? 1.0f : weights[index]) : 0.0f;
      const int source =
          validExpert(expert, work.numExperts_) && weight != 0.0f ? static_cast<int>(expert / localExperts) : -1;
      const bool valid = lane < RankMajorTmaMaxNTopk && source >= 0;
      if (lane < RankMajorTmaMaxNTopk) {
        validRows[lane] = valid;
        slotWeights[lane] = valid ? weight : 0.0f;
      }
      __syncwarp();
      bool pending = valid;
      while (__any_sync(0xffffffff, pending)) {
        const bool ready = pending && sourceReady(transport, flags, source, target, true);
        if (ready) {
          const auto* src =
              expandedRow(input, layout, transport, source, token, lane, work.maxTokensPerRank_, work.numTopk_, bytes);
          barriers[lane].arriveAndExpect(bytes);
          mscclpp::bulkLoad(shared + lane * bytes, src, bytes, barriers[lane]);
          pending = false;
        }
      }
      if (valid) barriers[lane].wait(phase);
      __syncwarp();
      if (lane == 0) mscclpp::bulkFence();
    }
    __syncthreads();
    for (int v = threadIdx.x; v < vectors; v += CombineNThreads) {
      float2 reduced[pairsPerVector] = {};
#pragma unroll
      for (int k = 0; k < RankMajorTmaMaxNTopk; ++k) {
        if (!validRows[k]) continue;
        const int4 packed = sharedRows[k * vectors + v];
        const auto* pairs = reinterpret_cast<const mscclpp::bf16x2*>(&packed);
#pragma unroll
        for (int p = 0; p < pairsPerVector; ++p) {
          const auto values = mscclpp::to<mscclpp::f32x2>(pairs[p]);
          reduced[p].x = fmaf(values.data[0], slotWeights[k], reduced[p].x);
          reduced[p].y = fmaf(values.data[1], slotWeights[k], reduced[p].y);
        }
      }
      int4 packed;
      auto* pairs = reinterpret_cast<mscclpp::bf16x2*>(&packed);
#pragma unroll
      for (int p = 0; p < pairsPerVector; ++p) pairs[p] = mscclpp::to<mscclpp::bf16x2>(mscclpp::f32x2(reduced[p]));
      static_cast<int4*>(output)[static_cast<size_t>(token) * vectors + v] = packed;
    }
    __syncthreads();
  }
}

// K=9 fallback: fixed slots and original top-k order. Hidden specializations
// contain a multiple of WARP_SIZE int4s, so each vector-loop warp is full.
template <int Hidden>
__device__ void recvRankMajorTopkExpandedRemotePartials(void* output, const void* input, const int64_t* topkIds,
                                                        const float* weights, const Workload& work,
                                                        const TransportView& transport, const Layout& layout, int ranks,
                                                        uint64_t target) {
  constexpr size_t bytes = Hidden * sizeof(Bf16);
  constexpr int vectors = bytes / sizeof(int4);
  static_assert(vectors % WARP_SIZE == 0);
  constexpr int pairsPerVector = sizeof(int4) / sizeof(mscclpp::bf16x2);
  const int lane = get_lane_id();
  const int localExperts = work.numExperts_ / ranks;
  for (int token = blockIdx.x - 1; token < work.numTokens_; token += gridDim.x - 1) {
    const size_t index = static_cast<size_t>(token) * work.numTopk_ + lane;
    const int64_t expert = lane < work.numTopk_ ? topkIds[index] : -1;
    const float weight = validExpert(expert, work.numExperts_) ? (weights == nullptr ? 1.0f : weights[index]) : 0.0f;
    const int source =
        validExpert(expert, work.numExperts_) && weight != 0.0f ? static_cast<int>(expert / localExperts) : -1;
    if (source >= 0)
      waitSource(transport, static_cast<uint64_t*>(layout.gpuNetIoCombineFlagsBuffer_), source, target, true);
    __syncwarp();
    for (int v = threadIdx.x; v < vectors; v += CombineNThreads) {
      float2 reduced[pairsPerVector] = {};
      for (int k = 0; k < work.numTopk_; ++k) {
        const int owner = __shfl_sync(0xffffffff, source, k);
        const float w = __shfl_sync(0xffffffff, weight, k);
        if (owner < 0) continue;
        const int4 packed = reinterpret_cast<const int4*>(
            expandedRow(input, layout, transport, owner, token, k, work.maxTokensPerRank_, work.numTopk_, bytes))[v];
        const auto* pairs = reinterpret_cast<const mscclpp::bf16x2*>(&packed);
#pragma unroll
        for (int p = 0; p < pairsPerVector; ++p) {
          const auto values = mscclpp::to<mscclpp::f32x2>(pairs[p]);
          reduced[p].x = fmaf(values.data[0], w, reduced[p].x);
          reduced[p].y = fmaf(values.data[1], w, reduced[p].y);
        }
      }
      int4 packed;
      auto* pairs = reinterpret_cast<mscclpp::bf16x2*>(&packed);
#pragma unroll
      for (int p = 0; p < pairsPerVector; ++p) pairs[p] = mscclpp::to<mscclpp::bf16x2>(mscclpp::f32x2(reduced[p]));
      static_cast<int4*>(output)[static_cast<size_t>(token) * vectors + v] = packed;
    }
  }
}

template <int Hidden>
__device__ void pushExpandedCombine(const void* input, const Layout& layout, const TransportView& transport,
                                    const Workload& work, int owner) {
#if defined(MSCCLPP_USE_GPUNETIO)
  if (transport.isNvlinkPeer(owner)) return;
  auto* gin = transport.gpuNetIo_;
  const size_t rows = static_cast<size_t>(work.maxTokensPerRank_) * work.numTopk_;
  constexpr size_t bytes = Hidden * sizeof(Bf16);
  if (threadIdx.x < gin->numHcas) {
    const int stripe = threadIdx.x;
    const int q = markerQp(transport, owner, stripe);
    const auto* ids = static_cast<const int*>(layout.rankMajorTopkIdsBuffer_) + owner * rows;
    const auto* wgts = static_cast<const float*>(layout.rankMajorTopkWeightsBuffer_) + owner * rows;
    int sinceFlush = 0;
    for (size_t row = rows * stripe / gin->numHcas; row < rows * (stripe + 1) / gin->numHcas; ++row) {
      if (!validExpert(ids[row], work.numExperts_) || wgts[row] == 0.0f) continue;
      const auto* src = static_cast<const uint8_t*>(input) + (owner * rows + row) * bytes;
      auto* dst = static_cast<uint8_t*>(layout.gpuNetIoCombineLandingBuffer_) +
                  (static_cast<size_t>(transport.rank_) * rows + row) * bytes;
      gin->put(owner, transport.symmetricOffset(dst), transport.symmetricOffset(src), bytes, q);
      if (++sinceFlush >= GpuNetIoFlushInterval) {
        gin->flush(owner, q);
        sinceFlush = 0;
      }
    }
    auto* flag = static_cast<uint64_t*>(layout.gpuNetIoCombineFlagsBuffer_) +
                 static_cast<size_t>(transport.rank_) * GpuNetIoMaxQpsPerPeer + q;
    // Always mark every physical stripe, even for empty peers/zero-weight rows.
    gin->atomicAdd(owner, transport.symmetricOffset(flag), 1, q);
  }
#endif
}
#endif  // MSCCLPP_BULK_AVAILABLE

template <int Hidden, bool UseTma>
__global__ __launch_bounds__(CombineNThreads, 1) void combineTopkExpandedKernel(void* output, const void* input,
                                                                                const int64_t* topkIds,
                                                                                const float* weights, Workload work,
                                                                                CommContext comm, void* workspace) {
#if MSCCLPP_BULK_AVAILABLE
  extern __shared__ __align__(128) uint8_t shared[];
  const TransportView transport(comm);
  validateGpuNetIo(transport);
  WorkspaceView state(workspace, comm.numRanks_, work.numExperts_);
  const Layout layout(comm.symmetricBufferBase_, work.maxTokensPerRank_, Hidden, comm.numRanks_, work.numExperts_,
                      work.numTopk_, true, true);
  auto* flags = static_cast<uint64_t*>(layout.gpuNetIoCombineFlagsBuffer_);
  const uint64_t target = state.combineArrivedBaseline_[comm.rank_] + 1;
  // Inputs were produced by preceding work on this stream. Publish local
  // readiness before cross-domain pushes, without a starting grid barrier.
  if (blockIdx.x == 0) {
    for (int peer = threadIdx.x; peer < comm.numRanks_; peer += blockDim.x) {
      if (!transport.isNvlinkPeer(peer)) continue;
      auto* remote = static_cast<uint64_t*>(transport.mappedBuffer(flags, peer));
      signalLocal(remote + static_cast<size_t>(comm.rank_) * GpuNetIoMaxQpsPerPeer);
    }
  }
  if (blockIdx.x < comm.numRanks_) pushExpandedCombine<Hidden>(input, layout, transport, work, blockIdx.x);
  if (blockIdx.x == 0) {
    for (int source = threadIdx.x; source < comm.numRanks_; source += blockDim.x)
      waitSource(transport, flags, source, target, true);
  } else {
    if constexpr (UseTma)
      recvRankMajorTopkExpandedRemotePartialsTma<Hidden>(output, input, topkIds, weights, work, transport, layout,
                                                         comm.numRanks_, target, shared);
    else
      recvRankMajorTopkExpandedRemotePartials<Hidden>(output, input, topkIds, weights, work, transport, layout,
                                                      comm.numRanks_, target);
  }
#if defined(MSCCLPP_USE_GPUNETIO)
  if (blockIdx.x < comm.numRanks_ && !transport.isNvlinkPeer(blockIdx.x)) {
    auto* gin = transport.gpuNetIo_;
    if (threadIdx.x < gin->numHcas) gin->flush(blockIdx.x, markerQp(transport, blockIdx.x, threadIdx.x));
  }
#endif
  __threadfence_system();
  state.combineSyncer_->sync(gridDim.x);
  if (blockIdx.x == 0) {
    finishCollective(transport, layout, comm.numRanks_);
    if (threadIdx.x == 0) state.combineArrivedBaseline_[comm.rank_] = target;
  }
  state.combineSyncer_->sync(gridDim.x);
#endif
}

void validate(const Workload& work, const CommContext& comm, int blocks) {
  EP_HOST_ASSERT(work.outputLayout_ == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED);
  EP_HOST_ASSERT(work.dispatchDataType_ == DispatchDataType::BF16);
  EP_HOST_ASSERT(comm.numRanks_ > 0 && comm.numRanks_ <= 64);
  EP_HOST_ASSERT(work.numExperts_ > 0 && work.numExperts_ % comm.numRanks_ == 0);
  EP_HOST_ASSERT(work.numTopk_ > 0 && work.numTopk_ <= CombineMaxNTopk);
  EP_HOST_ASSERT(work.maxTokensPerRank_ > 0 && work.numTokens_ >= 0 && work.numTokens_ <= work.maxTokensPerRank_);
  EP_HOST_ASSERT(static_cast<int64_t>(work.maxTokensPerRank_) * work.numTopk_ <= INT32_MAX);
  EP_HOST_ASSERT(blocks > comm.numRanks_ && blocks <= MaxDispatchBlocks);
  EP_HOST_ASSERT(comm.symmetricBufferBase_ != nullptr && comm.peerMappedBufferBases_ != nullptr);
}

template <int Hidden>
void launchDispatch(void* output, int* ids, float* weightsOut, int* count, const void* input, const int64_t* topkIds,
                    const float* weights, const Workload& work, const CommContext& comm, void* workspace, int blocks,
                    cudaStream_t stream) {
  const size_t shared = dispatchSharedBytes<Hidden>(comm.numRanks_, work.numTopk_);
  static thread_local KernelConfigCache config;
  EP_HOST_ASSERT(configureKernel(dispatchTopkExpandedKernel<Hidden>, DispatchNThreads, shared, comm, config) >= blocks);
  dispatchTopkExpandedKernel<Hidden><<<blocks, DispatchNThreads, shared, stream>>>(
      output, ids, weightsOut, count, input, topkIds, weights, work, comm, workspace);
}

template <int Hidden, bool UseTma>
void launchCombine(void* output, const void* input, const int64_t* ids, const float* weights, const Workload& work,
                   const CommContext& comm, void* workspace, int blocks, cudaStream_t stream) {
  const size_t shared = combineSharedBytes<Hidden>(work.numTopk_);
  static thread_local KernelConfigCache config;
  EP_HOST_ASSERT(configureKernel(combineTopkExpandedKernel<Hidden, UseTma>, CombineNThreads, shared, comm, config) >=
                 blocks);
  combineTopkExpandedKernel<Hidden, UseTma>
      <<<blocks, CombineNThreads, shared, stream>>>(output, input, ids, weights, work, comm, workspace);
}
}  // namespace

void dispatch(void* output, int* outputIds, float* outputWeights, int* outputCount, const void* input,
              const int64_t* topkIds, const float* weights, const Workload& work, const CommContext& comm,
              void* workspace, int numBlocks, cudaStream_t stream) {
  validate(work, comm, numBlocks);
  EP_HOST_ASSERT(output && outputIds && outputWeights && outputCount && workspace);
  EP_HOST_ASSERT(work.numTokens_ == 0 || (input && topkIds));
#define EXPANDED_DISPATCH(H)                                                                                         \
  case H:                                                                                                            \
    launchDispatch<H>(output, outputIds, outputWeights, outputCount, input, topkIds, weights, work, comm, workspace, \
                      numBlocks, stream);                                                                            \
    break
  switch (work.hidden_) {
    EXPANDED_DISPATCH(2048);
    EXPANDED_DISPATCH(4096);
    EXPANDED_DISPATCH(4352);
    EXPANDED_DISPATCH(6656);
    EXPANDED_DISPATCH(7168);
    EXPANDED_DISPATCH(8192);
    EXPANDED_DISPATCH(8704);
    EXPANDED_DISPATCH(9216);
    default:
      EP_HOST_ASSERT(false && "unsupported expanded hidden size");
  }
#undef EXPANDED_DISPATCH
  CUDA_CHECK(cudaGetLastError());
}

void combine(void* output, const void* input, const int64_t* topkIds, const float* weights, const Workload& work,
             const CommContext& comm, void* workspace, int numBlocks, cudaStream_t stream) {
  const int blocks = numBlocks + 1;
  validate(work, comm, blocks);
  EP_HOST_ASSERT(input && workspace);
  EP_HOST_ASSERT(work.numTokens_ == 0 || (output && topkIds));
#define EXPANDED_COMBINE(H)                                                                            \
  case H:                                                                                              \
    if (work.numTopk_ <= RankMajorTmaMaxNTopk)                                                         \
      launchCombine<H, true>(output, input, topkIds, weights, work, comm, workspace, blocks, stream);  \
    else                                                                                               \
      launchCombine<H, false>(output, input, topkIds, weights, work, comm, workspace, blocks, stream); \
    break
  switch (work.hidden_) {
    EXPANDED_COMBINE(2048);
    EXPANDED_COMBINE(4096);
    EXPANDED_COMBINE(4352);
    EXPANDED_COMBINE(6656);
    EXPANDED_COMBINE(7168);
    EXPANDED_COMBINE(8192);
    EXPANDED_COMBINE(8704);
    EXPANDED_COMBINE(9216);
    default:
      EP_HOST_ASSERT(false && "unsupported expanded hidden size");
  }
#undef EXPANDED_COMBINE
  CUDA_CHECK(cudaGetLastError());
}
}  // namespace mscclpp::ep::low_latency::topk_expanded
