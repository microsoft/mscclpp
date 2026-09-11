// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "config.cuh"
#include "exception.cuh"
#include "topk_expanded.cuh"

#if defined(MSCCLPP_USE_GPUNETIO)
#include <mscclpp/port_channel_gpunetio_device.hpp>
#endif

namespace mscclpp::ep::low_latency::topk_expanded {
namespace {

constexpr int Threads = 256;
using detail::TransportView;
using detail::WorkspaceView;

__device__ bool validExpert(int64_t expert, int experts) { return expert >= 0 && expert < experts; }

__device__ void waitFlag(const uint64_t* flag, uint64_t target) {
  while (mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(const_cast<uint64_t*>(flag), mscclpp::memoryOrderAcquire) <
         target) {
  }
}

__device__ void signalLocal(uint64_t* flag) {
  mscclpp::atomicFetchAdd<uint64_t, mscclpp::scopeSystem>(flag, 1, mscclpp::memoryOrderRelease);
}

__device__ int markerCount(const TransportView& transport, int peer, bool combine) {
#if defined(MSCCLPP_USE_GPUNETIO)
  if (!transport.isNvlinkPeer(peer)) return combine ? transport.gpuNetIo_->numHcas : transport.gpuNetIo_->numQpsPerPeer;
#endif
  return 1;
}

__device__ int markerQp(const TransportView& transport, int owner, int stripe, bool combine) {
#if defined(MSCCLPP_USE_GPUNETIO)
  if (transport.gpuNetIo_ != nullptr && combine)
    return (owner % transport.gpuNetIo_->numQpsPerPeer + stripe) % transport.gpuNetIo_->numQpsPerPeer;
#endif
  return stripe;
}

__device__ void postMarkers(const TransportView& transport, uint64_t* flags, int ranks, bool combine) {
  for (int peer = threadIdx.x; peer < ranks; peer += blockDim.x) {
    if (transport.isNvlinkPeer(peer)) {
      auto* remote = static_cast<uint64_t*>(transport.mappedBuffer(flags, peer));
      signalLocal(remote + static_cast<size_t>(transport.rank_) * GpuNetIoMaxQpsPerPeer);
    } else {
#if defined(MSCCLPP_USE_GPUNETIO)
      for (int stripe = 0; stripe < markerCount(transport, peer, combine); ++stripe) {
        const int q = markerQp(transport, peer, stripe, combine);
        auto* flag = flags + static_cast<size_t>(transport.rank_) * GpuNetIoMaxQpsPerPeer + q;
        transport.gpuNetIo_->atomicAdd(peer, transport.symmetricOffset(flag), 1, q);
      }
#endif
    }
  }
  __syncthreads();
}

__device__ void drainMarkers(const TransportView& transport, int ranks, bool combine) {
#if defined(MSCCLPP_USE_GPUNETIO)
  for (int peer = threadIdx.x; peer < ranks; peer += blockDim.x) {
    if (transport.isNvlinkPeer(peer)) continue;
    for (int stripe = 0; stripe < markerCount(transport, peer, combine); ++stripe)
      transport.gpuNetIo_->flush(peer, markerQp(transport, peer, stripe, combine));
  }
#endif
  __syncthreads();
}

__device__ void waitSource(const TransportView& transport, const uint64_t* flags, int source, uint64_t target,
                           bool combine) {
  for (int stripe = 0; stripe < markerCount(transport, source, combine); ++stripe) {
    const int q = transport.isNvlinkPeer(source) ? 0 : markerQp(transport, transport.rank_, stripe, combine);
    waitFlag(flags + static_cast<size_t>(source) * GpuNetIoMaxQpsPerPeer + q, target);
  }
}

// A separate generation protects sparse metadata and remote expert-output readers across graph replays.
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
  for (int peer = threadIdx.x; peer < ranks; peer += blockDim.x) waitFlag(flags + peer, target);
#if defined(MSCCLPP_USE_GPUNETIO)
  for (int peer = threadIdx.x; peer < ranks; peer += blockDim.x)
    if (!transport.isNvlinkPeer(peer)) transport.gpuNetIo_->flush(peer, 0);
#endif
  __syncthreads();
}

__global__ void dispatchTopkExpandedKernel(void* output, int* outputIds, float* outputWeights, int* outputCount,
                                           const void* input, const int64_t* topkIds, const float* weights,
                                           Workload work, CommContext comm, void* workspace) {
  const TransportView transport(comm);
  WorkspaceView state(workspace, comm.numRanks_, work.numExperts_);
  const Layout layout(comm.symmetricBufferBase_, work.maxTokensPerRank_, work.hidden_, comm.numRanks_, work.numExperts_,
                      work.numTopk_, true, true);
  const int capacity = work.maxTokensPerRank_;
  const int topk = work.numTopk_;
  const int localExperts = work.numExperts_ / comm.numRanks_;
  const int vectors = work.hidden_ / (sizeof(int4) / sizeof(Bf16));
  [[maybe_unused]] const size_t bytes = static_cast<size_t>(work.hidden_) * sizeof(Bf16);
  auto* flags = static_cast<uint64_t*>(layout.gpuNetIoFlagsBuffer_);
  const uint64_t target = state.dispatchArrivedBaseline_[comm.rank_] + 1;

  for (int token = blockIdx.x; token < capacity; token += gridDim.x) {
    auto* staged = reinterpret_cast<int4*>(static_cast<uint8_t*>(layout.gpuNetIoStagingBuffer_) +
                                           static_cast<size_t>(token) * layout.gpuNetIoSlotStride_);
    if (token < work.numTokens_) {
      const auto* source = static_cast<const int4*>(input) + static_cast<size_t>(token) * vectors;
      for (int i = threadIdx.x; i < vectors; i += blockDim.x) staged[i] = source[i];
    }
    __syncthreads();
    for (int peer = 0; peer < comm.numRanks_; ++peer) {
      const size_t sendRow = (static_cast<size_t>(peer) * capacity + token) * topk;
      const size_t recvRow = (static_cast<size_t>(comm.rank_) * capacity + token) * topk;
      auto* ids = static_cast<int*>(layout.expandedSendIds_) + sendRow;
      auto* wgts = static_cast<float*>(layout.expandedSendWeights_) + sendRow;
      for (int k = threadIdx.x; k < topk; k += blockDim.x) {
        const int64_t expert = token < work.numTokens_ ? topkIds[static_cast<size_t>(token) * topk + k] : -1;
        const bool local = validExpert(expert, work.numExperts_) && expert / localExperts == peer;
        ids[k] = local ? static_cast<int>(expert) : work.invalidTokenExpertId_;
        wgts[k] = local ? (weights == nullptr ? 1.0f : weights[static_cast<size_t>(token) * topk + k]) : 0.0f;
      }
      __syncthreads();
      __threadfence_system();
      if (transport.isNvlinkPeer(peer)) {
        auto* remoteIds = static_cast<int*>(transport.mappedBuffer(outputIds, peer)) + recvRow;
        auto* remoteWeights = static_cast<float*>(transport.mappedBuffer(outputWeights, peer)) + recvRow;
        auto* remoteTokens = static_cast<int4*>(transport.mappedBuffer(output, peer)) + recvRow * vectors;
        for (int k = threadIdx.x; k < topk; k += blockDim.x) {
          remoteIds[k] = ids[k];
          remoteWeights[k] = wgts[k];
        }
        for (int i = threadIdx.x; i < topk * vectors; i += blockDim.x) {
          const int k = i / vectors;
          if (validExpert(ids[k], work.numExperts_)) remoteTokens[i] = staged[i % vectors];
        }
      } else {
#if defined(MSCCLPP_USE_GPUNETIO)
        if (threadIdx.x == 0) {
          auto* gin = transport.gpuNetIo_;
          const int q = token % gin->numQpsPerPeer;
          gin->put(peer, transport.symmetricOffset(outputIds + recvRow), transport.symmetricOffset(ids),
                   static_cast<size_t>(topk) * sizeof(int), q);
          gin->put(peer, transport.symmetricOffset(outputWeights + recvRow), transport.symmetricOffset(wgts),
                   static_cast<size_t>(topk) * sizeof(float), q);
          for (int k = 0; k < topk; ++k) {
            if (!validExpert(ids[k], work.numExperts_)) continue;
            const int payloadQp = (ids[k] % localExperts) % gin->numQpsPerPeer;
            auto* remote = static_cast<uint8_t*>(output) + (recvRow + k) * bytes;
            gin->put(peer, transport.symmetricOffset(remote), transport.symmetricOffset(staged), bytes, payloadQp);
          }
        }
#endif
      }
      __syncthreads();
    }
  }
  __threadfence_system();
  state.combineSyncer_->sync(gridDim.x);
  if (blockIdx.x == 0) {
    postMarkers(transport, flags, comm.numRanks_, false);
    drainMarkers(transport, comm.numRanks_, false);
  } else if (blockIdx.x <= comm.numRanks_) {
    const int source = blockIdx.x - 1;
    if (threadIdx.x == 0) {
      waitSource(transport, flags, source, target, false);
      outputCount[source] = 0;
    }
    __syncthreads();
    int count = 0;
    for (int row = threadIdx.x; row < capacity * topk; row += blockDim.x) {
      const int expert = outputIds[static_cast<size_t>(source) * capacity * topk + row];
      count += validExpert(expert, work.numExperts_);
    }
    if (count) atomicAdd(outputCount + source, count);
  }
  __threadfence_system();
  state.combineSyncer_->sync(gridDim.x);
  if (blockIdx.x == 0) {
    finishCollective(transport, layout, comm.numRanks_);
    if (threadIdx.x == 0) state.dispatchArrivedBaseline_[comm.rank_] = target;
  }
  state.combineSyncer_->sync(gridDim.x);
}

__global__ void combineTopkExpandedKernel(void* output, const void* input, const int64_t* topkIds, const float* weights,
                                          Workload work, CommContext comm, void* workspace) {
  const TransportView transport(comm);
  WorkspaceView state(workspace, comm.numRanks_, work.numExperts_);
  const Layout layout(comm.symmetricBufferBase_, work.maxTokensPerRank_, work.hidden_, comm.numRanks_, work.numExperts_,
                      work.numTopk_, true, true);
  const int topk = work.numTopk_;
  [[maybe_unused]] const int rowsPerRank = work.maxTokensPerRank_ * topk;
  const int localExperts = work.numExperts_ / comm.numRanks_;
  [[maybe_unused]] const size_t bytes = static_cast<size_t>(work.hidden_) * sizeof(Bf16);
  auto* flags = static_cast<uint64_t*>(layout.gpuNetIoCombineFlagsBuffer_);
  const uint64_t target = state.combineArrivedBaseline_[comm.rank_] + 1;

#if defined(MSCCLPP_USE_GPUNETIO)
  if (blockIdx.x < comm.numRanks_ && !transport.isNvlinkPeer(blockIdx.x)) {
    auto* gin = transport.gpuNetIo_;
    const int owner = blockIdx.x;
    if (threadIdx.x < gin->numHcas) {
      const int stripe = threadIdx.x;
      const int begin = rowsPerRank * stripe / gin->numHcas;
      const int end = rowsPerRank * (stripe + 1) / gin->numHcas;
      const int q = markerQp(transport, owner, stripe, true);
      const auto* ids =
          static_cast<const int*>(layout.rankMajorTopkIdsBuffer_) + static_cast<size_t>(owner) * rowsPerRank;
      const auto* wgts =
          static_cast<const float*>(layout.rankMajorTopkWeightsBuffer_) + static_cast<size_t>(owner) * rowsPerRank;
      for (int row = begin; row < end; ++row) {
        if (!validExpert(ids[row], work.numExperts_) || wgts[row] == 0.0f) continue;
        const auto* src = static_cast<const uint8_t*>(input) + (static_cast<size_t>(owner) * rowsPerRank + row) * bytes;
        auto* dst = static_cast<uint8_t*>(layout.gpuNetIoCombineLandingBuffer_) +
                    (static_cast<size_t>(comm.rank_) * rowsPerRank + row) * bytes;
        gin->put(owner, transport.symmetricOffset(dst), transport.symmetricOffset(src), bytes, q);
      }
    }
  }
#endif
  state.combineSyncer_->sync(gridDim.x);
  if (blockIdx.x == 0) {
    postMarkers(transport, flags, comm.numRanks_, true);
    drainMarkers(transport, comm.numRanks_, true);
    for (int source = threadIdx.x; source < comm.numRanks_; source += blockDim.x)
      waitSource(transport, flags, source, target, true);
  } else {
    constexpr int Elements = sizeof(int4) / sizeof(Bf16);
    const int vectors = work.hidden_ / Elements;
    for (int token = blockIdx.x - 1; token < work.numTokens_; token += gridDim.x - 1) {
      for (int v = threadIdx.x; v < vectors; v += blockDim.x) {
        float2 sum[Elements / 2] = {};
        for (int k = 0; k < topk; ++k) {
          const int64_t expert = topkIds[static_cast<size_t>(token) * topk + k];
          if (!validExpert(expert, work.numExperts_)) continue;
          const float weight = weights == nullptr ? 1.0f : weights[static_cast<size_t>(token) * topk + k];
          if (weight == 0.0f) continue;
          const int source = expert / localExperts;
          waitSource(transport, flags, source, target, true);
          const void* base = transport.isNvlinkPeer(source) ? transport.mappedBuffer(const_cast<void*>(input), source)
                                                            : layout.gpuNetIoCombineLandingBuffer_;
          const int rowRank = transport.isNvlinkPeer(source) ? comm.rank_ : source;
          const size_t row = (static_cast<size_t>(rowRank) * work.maxTokensPerRank_ + token) * topk + k;
          const int4 packed = static_cast<const int4*>(base)[row * vectors + v];
          const auto* pairs = reinterpret_cast<const mscclpp::bf16x2*>(&packed);
#pragma unroll
          for (int p = 0; p < Elements / 2; ++p) {
            const auto values = mscclpp::to<mscclpp::f32x2>(pairs[p]);
            sum[p].x = fmaf(values.data[0], weight, sum[p].x);
            sum[p].y = fmaf(values.data[1], weight, sum[p].y);
          }
        }
        int4 packed;
        auto* pairs = reinterpret_cast<mscclpp::bf16x2*>(&packed);
#pragma unroll
        for (int p = 0; p < Elements / 2; ++p) pairs[p] = mscclpp::to<mscclpp::bf16x2>(mscclpp::f32x2(sum[p]));
        static_cast<int4*>(output)[static_cast<size_t>(token) * vectors + v] = packed;
      }
    }
  }
  __threadfence_system();
  state.combineSyncer_->sync(gridDim.x);
  if (blockIdx.x == 0) {
    finishCollective(transport, layout, comm.numRanks_);
    if (threadIdx.x == 0) state.combineArrivedBaseline_[comm.rank_] = target;
  }
  state.combineSyncer_->sync(gridDim.x);
}

// Single-domain path: stage routing ONCE per source, push only valid expanded
// payload rows, then let each receiver materialize its own metadata. This avoids
// an all-rank clear-before-scatter barrier and never races remote metadata stores.
// The public [R * capacity * K, H] buffers and distinct duplicate slots remain.
size_t nvlinkDispatchSharedBytes(int hidden) {
  return static_cast<size_t>(hidden) * sizeof(Bf16) + sizeof(mscclpp::BulkBarrier);
}

size_t nvlinkCombineSharedBytes(int hidden, int topk) {
  return static_cast<size_t>(topk) *
         (static_cast<size_t>(hidden) * sizeof(Bf16) + sizeof(mscclpp::BulkBarrier) + sizeof(float));
}

__global__ void dispatchTopkExpandedNvlinkKernel(void* output, int* outputIds, float* outputWeights, int* outputCount,
                                                 const void* input, const int64_t* topkIds, const float* weights,
                                                 Workload work, CommContext comm, void* workspace) {
#if MSCCLPP_BULK_AVAILABLE
  extern __shared__ __align__(128) uint8_t shared[];
  const TransportView transport(comm);
  WorkspaceView state(workspace, comm.numRanks_, work.numExperts_);
  const Layout layout(comm.symmetricBufferBase_, work.maxTokensPerRank_, work.hidden_, comm.numRanks_, work.numExperts_,
                      work.numTopk_, true, true);
  const int capacity = work.maxTokensPerRank_;
  const int topk = work.numTopk_;
  const int localExperts = work.numExperts_ / comm.numRanks_;
  const size_t bytes = static_cast<size_t>(work.hidden_) * sizeof(Bf16);
  const size_t rowsPerSource = static_cast<size_t>(capacity) * topk;
  // Only the first capacity*K entries are used in this mode. All peers read this
  // same source-owned routing image after its publication; no per-peer copies.
  auto* stagedIds = static_cast<int*>(layout.expandedSendIds_);
  auto* stagedWeights = static_cast<float*>(layout.expandedSendWeights_);
  // Expanded dispatch does not use LL8 packets; the first R ints of the existing
  // receive metadata region can hold published counts without changing Layout.
  auto* publishedCounts = static_cast<int*>(layout.dispatchRecvBuffer_);
  auto* flags = static_cast<uint64_t*>(layout.gpuNetIoFlagsBuffer_);
  const uint64_t target = state.dispatchArrivedBaseline_[comm.rank_] + 1;
  auto* barrier = reinterpret_cast<mscclpp::BulkBarrier*>(shared + bytes);
  uint32_t phase = 0;
  if (threadIdx.x == 0) barrier->init();

  for (int token = blockIdx.x; token < capacity; token += gridDim.x) {
    const bool active = token < work.numTokens_;
    if (active && threadIdx.x == 0) {
      barrier->arriveAndExpect(static_cast<uint32_t>(bytes));
      mscclpp::bulkLoad(shared, static_cast<const uint8_t*>(input) + static_cast<size_t>(token) * bytes,
                        static_cast<uint32_t>(bytes), *barrier);
    }
    const int k = static_cast<int>(threadIdx.x);
    const size_t selection = static_cast<size_t>(token) * topk + k;
    const int64_t expert = active && k < topk ? topkIds[selection] : -1;
    const bool valid = k < topk && validExpert(expert, work.numExperts_);
    if (k < topk) {
      stagedIds[selection] = valid ? static_cast<int>(expert) : work.invalidTokenExpertId_;
      stagedWeights[selection] = valid ? (weights == nullptr ? 1.0f : weights[selection]) : 0.0f;
    }
    if (valid) atomicAdd(state.dispatchRankPayloadSlots_ + expert / localExperts, 1);
    if (active && threadIdx.x == 0) {
      barrier->wait(phase);
      mscclpp::bulkFence();
    }
    __syncthreads();
    if (valid) {
      const int peer = static_cast<int>(expert / localExperts);
      const size_t row = static_cast<size_t>(comm.rank_) * rowsPerSource + selection;
      auto* destination = static_cast<uint8_t*>(transport.mappedBuffer(output, peer)) + row * bytes;
      mscclpp::bulkStore(destination, shared, static_cast<uint32_t>(bytes));
      mscclpp::bulkStoreCommit();
      // Full destination completion, not merely source-read completion, before
      // publishing readiness. Each issuer waits for its own bulk group.
      mscclpp::bulkStoreWait();
    }
    __syncthreads();  // no shared tile reuse while another slot still reads it
  }
  __threadfence_system();
  state.combineSyncer_->sync(gridDim.x);
  if (blockIdx.x == 0) {
    // The runtime's existing stream-ordered memset resets these counters before
    // every dispatch. They count selections, including duplicate/zero-weight ones.
    if (threadIdx.x < comm.numRanks_) {
      const int peer = threadIdx.x;
      publishedCounts[peer] = state.dispatchRankPayloadSlots_[peer];
      __threadfence_system();
      auto* remote = static_cast<uint64_t*>(transport.mappedBuffer(flags, peer));
      signalLocal(remote + static_cast<size_t>(comm.rank_) * GpuNetIoMaxQpsPerPeer);
    }
  } else if (blockIdx.x <= comm.numRanks_) {
    const int source = blockIdx.x - 1;
    if (threadIdx.x == 0) waitFlag(flags + static_cast<size_t>(source) * GpuNetIoMaxQpsPerPeer, target);
    __syncthreads();
    const auto* sourceIds = static_cast<const int*>(transport.mappedBuffer(stagedIds, source));
    const auto* sourceWeights = static_cast<const float*>(transport.mappedBuffer(stagedWeights, source));
    if (threadIdx.x == 0) {
      const auto* counts = static_cast<const int*>(transport.mappedBuffer(publishedCounts, source));
      outputCount[source] = counts[comm.rank_];
    }
    for (size_t row = threadIdx.x; row < rowsPerSource; row += blockDim.x) {
      const int expert = sourceIds[row];
      const bool local = validExpert(expert, work.numExperts_) && expert / localExperts == comm.rank_;
      const size_t destination = static_cast<size_t>(source) * rowsPerSource + row;
      outputIds[destination] = local ? expert : work.invalidTokenExpertId_;
      outputWeights[destination] = local ? sourceWeights[row] : 0.0f;
    }
  }
  __threadfence_system();
  state.combineSyncer_->sync(gridDim.x);
  if (blockIdx.x == 0) {
    // All local metadata readers have arrived before acknowledging other ranks.
    // Keep the original lifetime protocol; a helper alone is not a grid barrier.
    finishCollective(transport, layout, comm.numRanks_);
    if (threadIdx.x == 0) state.dispatchArrivedBaseline_[comm.rank_] = target;
  }
  state.combineSyncer_->sync(gridDim.x);
#endif
}

__global__ void combineTopkExpandedNvlinkKernel(void* output, const void* input, const int64_t* topkIds,
                                                const float* weights, Workload work, CommContext comm,
                                                void* workspace) {
#if MSCCLPP_BULK_AVAILABLE
  extern __shared__ __align__(128) uint8_t shared[];
  const TransportView transport(comm);
  WorkspaceView state(workspace, comm.numRanks_, work.numExperts_);
  const Layout layout(comm.symmetricBufferBase_, work.maxTokensPerRank_, work.hidden_, comm.numRanks_, work.numExperts_,
                      work.numTopk_, true, true);
  const int topk = work.numTopk_;
  const int localExperts = work.numExperts_ / comm.numRanks_;
  constexpr int Elements = sizeof(int4) / sizeof(Bf16);
  const int vectors = work.hidden_ / Elements;
  const size_t bytes = static_cast<size_t>(work.hidden_) * sizeof(Bf16);
  auto* barriers = reinterpret_cast<mscclpp::BulkBarrier*>(shared + topk * bytes);
  auto* rowWeights = reinterpret_cast<float*>(barriers + topk);
  auto* flags = static_cast<uint64_t*>(layout.gpuNetIoCombineFlagsBuffer_);
  const uint64_t target = state.combineArrivedBaseline_[comm.rank_] + 1;

  // Expert output is produced by preceding stream work. Publish one local
  // readiness generation per source, never an RDMA marker/drain loop.
  if (blockIdx.x == 0) {
    if (threadIdx.x < comm.numRanks_) {
      auto* remote = static_cast<uint64_t*>(transport.mappedBuffer(flags, threadIdx.x));
      signalLocal(remote + static_cast<size_t>(comm.rank_) * GpuNetIoMaxQpsPerPeer);
    }
    __syncthreads();
    if (threadIdx.x < comm.numRanks_)
      waitFlag(flags + static_cast<size_t>(threadIdx.x) * GpuNetIoMaxQpsPerPeer, target);
  } else {
    const int k = static_cast<int>(threadIdx.x);
    uint32_t phase = 0;  // private per issuer; advances ONLY when this slot loads
    if (k < topk) barriers[k].init();
    for (int token = blockIdx.x - 1; token < work.numTokens_; token += gridDim.x - 1) {
      if (k < topk) {
        const size_t selection = static_cast<size_t>(token) * topk + k;
        const int64_t expert = topkIds[selection];
        const float weight =
            validExpert(expert, work.numExperts_) ? (weights == nullptr ? 1.0f : weights[selection]) : 0.0f;
        rowWeights[k] = weight;
        // Skip invalid AND zero-weight rows before mapping or reading payload;
        // their registered expert rows are allowed to contain NaNs.
        if (weight != 0.0f) {
          const int source = static_cast<int>(expert / localExperts);
          waitFlag(flags + static_cast<size_t>(source) * GpuNetIoMaxQpsPerPeer, target);
          const auto* base = static_cast<const uint8_t*>(transport.mappedBuffer(const_cast<void*>(input), source));
          const size_t row = (static_cast<size_t>(comm.rank_) * work.maxTokensPerRank_ + token) * topk + k;
          barriers[k].arriveAndExpect(static_cast<uint32_t>(bytes));
          mscclpp::bulkLoad(shared + k * bytes, base + row * bytes, static_cast<uint32_t>(bytes), barriers[k]);
          barriers[k].wait(phase);
          mscclpp::bulkFence();
        }
      }
      __syncthreads();
      for (int v = threadIdx.x; v < vectors; v += blockDim.x) {
        float2 sum[Elements / 2] = {};
        // Preserve original top-k order and source-side FP32 FMA semantics.
        // Duplicate experts still reference distinct slot rows and are not deduplicated.
        for (int slot = 0; slot < topk; ++slot) {
          const float weight = rowWeights[slot];
          if (weight == 0.0f) continue;
          const int4 packed = reinterpret_cast<const int4*>(shared)[static_cast<size_t>(slot) * vectors + v];
          const auto* pairs = reinterpret_cast<const mscclpp::bf16x2*>(&packed);
#pragma unroll
          for (int p = 0; p < Elements / 2; ++p) {
            const auto values = mscclpp::to<mscclpp::f32x2>(pairs[p]);
            sum[p].x = fmaf(values.data[0], weight, sum[p].x);
            sum[p].y = fmaf(values.data[1], weight, sum[p].y);
          }
        }
        int4 packed;
        auto* pairs = reinterpret_cast<mscclpp::bf16x2*>(&packed);
#pragma unroll
        for (int p = 0; p < Elements / 2; ++p) pairs[p] = mscclpp::to<mscclpp::bf16x2>(mscclpp::f32x2(sum[p]));
        static_cast<int4*>(output)[static_cast<size_t>(token) * vectors + v] = packed;
      }
      __syncthreads();  // all generic shared readers finish before the next load
    }
  }
  __threadfence_system();
  state.combineSyncer_->sync(gridDim.x);
  if (blockIdx.x == 0) {
    finishCollective(transport, layout, comm.numRanks_);
    if (threadIdx.x == 0) state.combineArrivedBaseline_[comm.rank_] = target;
  }
  state.combineSyncer_->sync(gridDim.x);
#endif
}

template <typename Kernel>
bool nvlinkKernelFits(Kernel kernel, size_t sharedBytes, const CommContext& comm, detail::KernelConfigCache& config) {
  cudaFuncAttributes attributes;
  CUDA_CHECK(cudaFuncGetAttributes(&attributes, kernel));
  if (sharedBytes + attributes.sharedSizeBytes > static_cast<size_t>(comm.maxSharedMemoryPerBlock_)) return false;
  return detail::configureKernel(kernel, Threads, sharedBytes, comm, config) >= MaxDispatchBlocks;
}

void validate(const Workload& work, const CommContext& comm, int blocks) {
  EP_HOST_ASSERT(work.outputLayout_ == DispatchLayout::RANK_MAJOR_TOPK_EXPANDED);
  EP_HOST_ASSERT(work.dispatchDataType_ == DispatchDataType::BF16);
  EP_HOST_ASSERT(comm.numRanks_ > 0 && comm.numRanks_ <= 64);
  EP_HOST_ASSERT(work.numExperts_ > 0 && work.numExperts_ % comm.numRanks_ == 0);
  EP_HOST_ASSERT(work.numTopk_ > 0 && work.numTopk_ <= 9);
  EP_HOST_ASSERT(work.maxTokensPerRank_ > 0 && work.numTokens_ >= 0 && work.numTokens_ <= work.maxTokensPerRank_);
  EP_HOST_ASSERT(work.hidden_ > 0 && work.hidden_ % (sizeof(int4) / sizeof(Bf16)) == 0);
  EP_HOST_ASSERT(blocks > comm.numRanks_ && blocks <= MaxDispatchBlocks);
  EP_HOST_ASSERT(comm.symmetricBufferBase_ != nullptr && comm.peerMappedBufferBases_ != nullptr);
}

}  // namespace

bool nvlinkFastPathAvailable(int hidden, int topk, const CommContext& comm) {
  if (hidden <= 0 || hidden % 8 != 0 || topk <= 0 || topk > 9) return false;
  static thread_local detail::KernelConfigCache dispatchConfig, combineConfig;
  return nvlinkKernelFits(dispatchTopkExpandedNvlinkKernel, nvlinkDispatchSharedBytes(hidden), comm, dispatchConfig) &&
         nvlinkKernelFits(combineTopkExpandedNvlinkKernel, nvlinkCombineSharedBytes(hidden, topk), comm, combineConfig);
}

void dispatch(void* output, int* outputIds, float* outputWeights, int* outputCount, const void* input,
              const int64_t* topkIds, const float* weights, const Workload& workload, const CommContext& comm,
              void* workspace, int numBlocks, cudaStream_t stream) {
  validate(workload, comm, numBlocks);
  EP_HOST_ASSERT(output && outputIds && outputWeights && outputCount && workspace);
  EP_HOST_ASSERT(workload.numTokens_ == 0 || (input && topkIds));
  if (comm.expandedNvlinkFastPath_) {
    const size_t sharedBytes = nvlinkDispatchSharedBytes(workload.hidden_);
    static thread_local detail::KernelConfigCache nvlinkConfig;
    EP_HOST_ASSERT(detail::configureKernel(dispatchTopkExpandedNvlinkKernel, Threads, sharedBytes, comm,
                                           nvlinkConfig) >= numBlocks);
    dispatchTopkExpandedNvlinkKernel<<<numBlocks, Threads, sharedBytes, stream>>>(
        output, outputIds, outputWeights, outputCount, input, topkIds, weights, workload, comm, workspace);
    CUDA_CHECK(cudaGetLastError());
    return;
  }
  static thread_local detail::KernelConfigCache config;
  EP_HOST_ASSERT(detail::configureKernel(dispatchTopkExpandedKernel, Threads, 0, comm, config) >= numBlocks);
  dispatchTopkExpandedKernel<<<numBlocks, Threads, 0, stream>>>(output, outputIds, outputWeights, outputCount, input,
                                                                topkIds, weights, workload, comm, workspace);
  CUDA_CHECK(cudaGetLastError());
}

void combine(void* output, const void* input, const int64_t* topkIds, const float* weights, const Workload& workload,
             const CommContext& comm, void* workspace, int numBlocks, cudaStream_t stream) {
  const int blocks = numBlocks + 1;
  validate(workload, comm, blocks);
  EP_HOST_ASSERT(input && workspace);
  EP_HOST_ASSERT(workload.numTokens_ == 0 || (output && topkIds));
  if (comm.expandedNvlinkFastPath_) {
    const size_t sharedBytes = nvlinkCombineSharedBytes(workload.hidden_, workload.numTopk_);
    static thread_local detail::KernelConfigCache nvlinkConfig;
    EP_HOST_ASSERT(detail::configureKernel(combineTopkExpandedNvlinkKernel, Threads, sharedBytes, comm, nvlinkConfig) >=
                   blocks);
    combineTopkExpandedNvlinkKernel<<<blocks, Threads, sharedBytes, stream>>>(output, input, topkIds, weights, workload,
                                                                              comm, workspace);
    CUDA_CHECK(cudaGetLastError());
    return;
  }
  static thread_local detail::KernelConfigCache config;
  EP_HOST_ASSERT(detail::configureKernel(combineTopkExpandedKernel, Threads, 0, comm, config) >= blocks);
  combineTopkExpandedKernel<<<blocks, Threads, 0, stream>>>(output, input, topkIds, weights, workload, comm, workspace);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace mscclpp::ep::low_latency::topk_expanded