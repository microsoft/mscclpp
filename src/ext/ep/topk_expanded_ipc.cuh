// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include "common/latency.cuh"

namespace mscclpp::ep::topk_expanded::ipc {
constexpr int CombineThreads = 1024;
constexpr int TmaSlots = 8;

#if MSCCLPP_BULK_AVAILABLE
__device__ __forceinline__ bool valid(int64_t expert, int experts) { return expert >= 0 && expert < experts; }

__device__ __forceinline__ void release(uint64_t* flag) {
  mscclpp::atomicFetchAdd<uint64_t, mscclpp::scopeSystem>(flag, 1, mscclpp::memoryOrderRelease);
}

__device__ __forceinline__ void wait(uint64_t* flag, uint64_t target) {
  while (mscclpp::atomicLoad<uint64_t, mscclpp::scopeSystem>(flag, mscclpp::memoryOrderAcquire) < target) {
  }
}

__device__ void retireAndAck(const TransportView& transport, const LatencyStorageLayout& layout, WorkspaceView& state,
                             int ranks) {
  __syncthreads();
  auto* retired = state.dispatchNumRecvTasks_;
  if (blockIdx.x != 0) {
    if (threadIdx.x == 0) mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(retired, 1, mscclpp::memoryOrderRelease);
    return;
  }
  if (threadIdx.x == 0) {
    while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(retired, mscclpp::memoryOrderAcquire) !=
           static_cast<int>(gridDim.x) - 1) {
    }
    *retired = 0;
    ++*static_cast<uint64_t*>(layout.expandedSyncEpoch_);
  }
  __syncthreads();
  const uint64_t target = *static_cast<uint64_t*>(layout.expandedSyncEpoch_);
  auto* flags = static_cast<uint64_t*>(layout.expandedSyncFlags_);
  for (int peer = threadIdx.x; peer < ranks; peer += blockDim.x) {
    auto* remote = static_cast<uint64_t*>(transport.mappedBuffer(flags, peer));
    release(remote + transport.rank_);
  }
  __syncthreads();
  for (int peer = threadIdx.x; peer < ranks; peer += blockDim.x) wait(flags + peer, target);
  __syncthreads();
}

template <int Hidden>
__device__ void send(void* output, int* outputIds, float* outputWeights, const void* input, const int64_t* topkIds,
                     const float* weights, const Workload& work, const TransportView& transport, WorkspaceView& state,
                     int ranks, int* sharedMem) {
  const int workers = gridDim.x - DispatchControlBlocks;
  if (blockIdx.x == 0 || static_cast<int>(blockIdx.x) > workers) return;
  const int warp = threadIdx.x / WARP_SIZE;
  const int perGroup = dispatchNWarpsPerGroup(work.numTokens_, workers);
  if (warp % perGroup != 0) return;
  const int lane = get_lane_id(), group = warp / perGroup;
  const int groups = DispatchNWarps / perGroup;
  const int sendSlots = ranks > DispatchMaxNWarpGroups * WARP_SIZE ? ranks : DispatchMaxNWarpGroups * WARP_SIZE;
  const size_t control = configAlign<size_t>((sendSlots + DispatchMaxNWarpGroups * ranks) * sizeof(int), 128);
  const size_t stride = dispatchPayloadStride<DispatchDataType::BF16>(Hidden, work.numTopk_, 0);
  auto* tiles = reinterpret_cast<uint8_t*>(sharedMem) + control;
  auto* tile = tiles + group * stride;
  auto* barrier = reinterpret_cast<mscclpp::BulkBarrier*>(tiles + DispatchMaxNWarpGroups * stride) + group;
  auto* completed = sharedMem + sendSlots + group * ranks;
  const int first = (blockIdx.x - 1) * groups + group;
  const int step = workers * groups;
  const int localExperts = work.numExperts_ / ranks;
  constexpr size_t bytes = Hidden * sizeof(Bf16);
  uint32_t phase = 0;
  if (lane == 0 && first < work.numTokens_) barrier->init();
  for (int peer = lane; peer < ranks; peer += WARP_SIZE) completed[peer] = 0;
  __syncwarp();
  for (int token = first; token < work.numTokens_; token += step) {
    if (lane == 0) {
      barrier->arriveAndExpect(bytes);
      mscclpp::bulkLoad(tile, static_cast<const uint8_t*>(input) + token * bytes, bytes, *barrier);
    }
    const size_t index = static_cast<size_t>(token) * work.numTopk_ + lane;
    const int64_t expert = lane < work.numTopk_ ? topkIds[index] : -1;
    const int destination = valid(expert, work.numExperts_) ? static_cast<int>(expert / localExperts) : -1;
    const float weight = lane < work.numTopk_ ? (weights == nullptr ? 1.0f : weights[index]) : 0.0f;
    const size_t row = (static_cast<size_t>(transport.rank_) * work.maxTokensPerRank_ + token) * work.numTopk_ + lane;
    if (lane == 0) barrier->wait(phase);
    __syncwarp();
    mscclpp::bulkFence();
    if (lane < work.numTopk_) {
      for (int peer = 0; peer < ranks; ++peer) {
        static_cast<int*>(transport.mappedBuffer(outputIds, peer))[row] =
            peer == destination ? static_cast<int>(expert) : work.invalidTokenExpertId_;
        static_cast<float*>(transport.mappedBuffer(outputWeights, peer))[row] = peer == destination ? weight : 0.0f;
      }
    }
    if (destination >= 0) {
      auto* dst = static_cast<uint8_t*>(transport.mappedBuffer(output, destination)) + row * bytes;
      mscclpp::bulkStore(dst, tile, bytes);
      mscclpp::bulkStoreCommit();
      mscclpp::bulkStoreWait();
    }
    if (lane < work.numTopk_)
      for (int peer = 0; peer < ranks; ++peer) atomicAdd_block(completed + peer, 1);
    __syncwarp();
  }
  __threadfence_system();
  __syncwarp();
  for (int peer = lane; peer < ranks; peer += WARP_SIZE)
    if (completed[peer])
      mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(state.dispatchRankPayloadCompletions_ + peer, completed[peer],
                                                         mscclpp::memoryOrderRelease);
}

__device__ void notify(int* outputIds, float* outputWeights, const int64_t* topkIds, const Workload& work,
                       const TransportView& transport, const LatencyStorageLayout& layout, WorkspaceView& state,
                       int ranks, int* counts) {
  const int lane = get_lane_id(), warp = threadIdx.x / WARP_SIZE;
  const int topk = work.numTopk_, localExperts = work.numExperts_ / ranks;
  for (int peer = threadIdx.x; peer < ranks; peer += blockDim.x) counts[peer] = 0;
  __syncthreads();
  for (int token = warp; token < work.numTokens_; token += DispatchNWarps) {
    const int64_t expert = lane < topk ? topkIds[static_cast<size_t>(token) * topk + lane] : -1;
    if (valid(expert, work.numExperts_)) atomicAdd_block(counts + expert / localExperts, 1);
  }
  __syncthreads();
  const size_t rows = static_cast<size_t>(work.maxTokensPerRank_) * topk;
  for (int peer = 0; peer < ranks; ++peer) {
    auto* ids = static_cast<int*>(transport.mappedBuffer(outputIds, peer)) + transport.rank_ * rows;
    auto* wgts = static_cast<float*>(transport.mappedBuffer(outputWeights, peer)) + transport.rank_ * rows;
    for (size_t index = static_cast<size_t>(work.numTokens_) * topk + threadIdx.x; index < rows; index += blockDim.x) {
      ids[index] = work.invalidTokenExpertId_;
      wgts[index] = 0.0f;
    }
  }
  __threadfence_system();
  __syncthreads();
  for (int peer = threadIdx.x; peer < ranks; peer += blockDim.x) {
    while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(state.dispatchRankPayloadCompletions_ + peer,
                                                          mscclpp::memoryOrderAcquire) != work.numTokens_ * topk) {
    }
    state.dispatchRankPayloadCompletions_[peer] = 0;
    static_cast<int*>(transport.mappedBuffer(layout.expandedCounts_, peer))[transport.rank_] = counts[peer];
    __threadfence_system();
    auto* flags = static_cast<uint64_t*>(transport.mappedBuffer(layout.gpuNetIoFlagsBuffer_, peer));
    release(flags + static_cast<size_t>(transport.rank_) * GpuNetIoMaxQpsPerPeer);
  }
}

__device__ void publishReady(const TransportView& transport, const LatencyStorageLayout& layout, WorkspaceView& state,
                             int ranks, uint64_t target) {
  auto* flags = static_cast<uint64_t*>(layout.gpuNetIoCombineFlagsBuffer_);
  for (int peer = threadIdx.x; peer < ranks; peer += blockDim.x) {
    auto* remote = static_cast<uint64_t*>(transport.mappedBuffer(flags, peer));
    release(remote + static_cast<size_t>(transport.rank_) * GpuNetIoMaxQpsPerPeer);
  }
  __syncthreads();
  for (int source = threadIdx.x; source < ranks; source += blockDim.x) {
    wait(flags + static_cast<size_t>(source) * GpuNetIoMaxQpsPerPeer, target);
    mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(state.combineRankReadyEpochs_ + source,
                                                         static_cast<uint32_t>(target), mscclpp::memoryOrderRelease);
  }
}

__device__ __forceinline__ bool ready(WorkspaceView& state, int source, uint64_t target) {
  return mscclpp::atomicLoad<uint32_t, mscclpp::scopeDevice>(
             state.combineRankReadyEpochs_ + source, mscclpp::memoryOrderAcquire) == static_cast<uint32_t>(target);
}

template <int Hidden, bool UseTma>
__device__ void gather(void* output, const void* input, const int64_t* topkIds, const float* weights,
                       const Workload& work, const TransportView& transport, WorkspaceView& state, int ranks,
                       uint64_t target, uint8_t* shared) {
  constexpr size_t bytes = Hidden * sizeof(Bf16);
  constexpr int vectors = bytes / sizeof(int4), pairsPerVector = sizeof(int4) / sizeof(mscclpp::bf16x2);
  static_assert(vectors % WARP_SIZE == 0);
  const int lane = get_lane_id(), warp = threadIdx.x / WARP_SIZE;
  const int localExperts = work.numExperts_ / ranks;
  auto* barriers = reinterpret_cast<mscclpp::BulkBarrier*>(shared + TmaSlots * bytes);
  auto* validRows = reinterpret_cast<int*>(barriers + TmaSlots);
  auto* slotWeights = reinterpret_cast<float*>(validRows + TmaSlots);
  uint32_t phase = 0;
  if constexpr (UseTma) {
    if (warp == 0) {
      if (lane < TmaSlots) barriers[lane].init();
      __syncwarp();
    }
  }
  for (int token = blockIdx.x - 1; token < work.numTokens_; token += gridDim.x - 1) {
    int source = -1;
    float weight = 0.0f;
    if (!UseTma || warp == 0) {
      const size_t index = static_cast<size_t>(token) * work.numTopk_ + lane;
      const int64_t expert = lane < work.numTopk_ ? topkIds[index] : -1;
      weight = valid(expert, work.numExperts_) ? (weights == nullptr ? 1.0f : weights[index]) : 0.0f;
      source = valid(expert, work.numExperts_) && weight != 0.0f ? static_cast<int>(expert / localExperts) : -1;
    }
    if constexpr (UseTma) {
      if (warp == 0) {
        const bool live = lane < TmaSlots && source >= 0;
        if (lane < TmaSlots) {
          validRows[lane] = live;
          slotWeights[lane] = live ? weight : 0.0f;
        }
        __syncwarp();
        bool pending = live;
        while (__any_sync(0xffffffff, pending)) {
          if (pending && ready(state, source, target)) {
            const auto* base = static_cast<const uint8_t*>(transport.mappedBuffer(const_cast<void*>(input), source));
            const size_t row =
                (static_cast<size_t>(transport.rank_) * work.maxTokensPerRank_ + token) * work.numTopk_ + lane;
            barriers[lane].arriveAndExpect(bytes);
            mscclpp::bulkLoad(shared + lane * bytes, base + row * bytes, bytes, barriers[lane]);
            pending = false;
          }
        }
        if (live) barriers[lane].wait(phase);
        __syncwarp();
        if (lane == 0) mscclpp::bulkFence();
      }
      __syncthreads();
    } else {
      if (source >= 0)
        while (!ready(state, source, target)) {
        }
      __syncwarp();
    }
    for (int vector = threadIdx.x; vector < vectors; vector += CombineThreads) {
      float2 sum[pairsPerVector] = {};
      const int slots = UseTma ? TmaSlots : work.numTopk_;
#pragma unroll
      for (int slot = 0; slot < slots; ++slot) {
        int4 packed;
        float weightValue;
        if constexpr (UseTma) {
          if (!validRows[slot]) continue;
          packed = reinterpret_cast<const int4*>(shared)[slot * vectors + vector];
          weightValue = slotWeights[slot];
        } else {
          const int owner = __shfl_sync(0xffffffff, source, slot);
          weightValue = __shfl_sync(0xffffffff, weight, slot);
          if (owner < 0) continue;
          const auto* base = static_cast<const int4*>(transport.mappedBuffer(const_cast<void*>(input), owner));
          const size_t row =
              (static_cast<size_t>(transport.rank_) * work.maxTokensPerRank_ + token) * work.numTopk_ + slot;
          packed = base[row * vectors + vector];
        }
        const auto* pairs = reinterpret_cast<const mscclpp::bf16x2*>(&packed);
#pragma unroll
        for (int pair = 0; pair < pairsPerVector; ++pair) {
          const auto values = mscclpp::to<mscclpp::f32x2>(pairs[pair]);
          sum[pair].x = fmaf(values.data[0], weightValue, sum[pair].x);
          sum[pair].y = fmaf(values.data[1], weightValue, sum[pair].y);
        }
      }
      int4 packed;
      auto* pairs = reinterpret_cast<mscclpp::bf16x2*>(&packed);
#pragma unroll
      for (int pair = 0; pair < pairsPerVector; ++pair)
        pairs[pair] = mscclpp::to<mscclpp::bf16x2>(mscclpp::f32x2(sum[pair]));
      static_cast<int4*>(output)[static_cast<size_t>(token) * vectors + vector] = packed;
    }
    if constexpr (UseTma) __syncthreads();
  }
}
#endif

template <int Hidden>
__global__ __launch_bounds__(DispatchNThreads, 1) void dispatchKernel(void* output, int* ids, float* weightsOut,
                                                                      int* count, const void* input,
                                                                      const int64_t* topkIds, const float* weights,
                                                                      Workload work, const DeviceContext* context) {
#if MSCCLPP_BULK_AVAILABLE
  extern __shared__ __align__(128) uint8_t shared[];
  const TransportView transport(context);
  WorkspaceView state(context->workspace_, context->numRanks_, work.numExperts_);
  const LatencyStorageLayout layout(context->localBufferBase_, work.maxTokensPerRank_, Hidden, context->numRanks_,
                                    work.numExperts_, work.numTopk_, DispatchLayout::RANK_MAJOR_TOPK_EXPANDED,
                                    CombineMode::RANK_LOCAL_REDUCE);
  const uint64_t target = state.dispatchArrivedBaseline_[context->rank_] + 1;
  if (blockIdx.x == gridDim.x - 1)
    notify(ids, weightsOut, topkIds, work, transport, layout, state, context->numRanks_,
           reinterpret_cast<int*>(shared));
  else
    send<Hidden>(output, ids, weightsOut, input, topkIds, weights, work, transport, state, context->numRanks_,
                 reinterpret_cast<int*>(shared));
  if (blockIdx.x < context->numRanks_ && threadIdx.x == 0) {
    wait(static_cast<uint64_t*>(layout.gpuNetIoFlagsBuffer_) + blockIdx.x * GpuNetIoMaxQpsPerPeer, target);
    count[blockIdx.x] = static_cast<int*>(layout.expandedCounts_)[blockIdx.x];
  }
  retireAndAck(transport, layout, state, context->numRanks_);
  if (blockIdx.x == 0 && threadIdx.x == 0) state.dispatchArrivedBaseline_[context->rank_] = target;
#endif
}

template <int Hidden, bool UseTma>
__global__ __launch_bounds__(CombineThreads, 1) void combineKernel(void* output, const void* input, const int64_t* ids,
                                                                   const float* weights, Workload work,
                                                                   const DeviceContext* context) {
#if MSCCLPP_BULK_AVAILABLE
  extern __shared__ __align__(128) uint8_t shared[];
  const TransportView transport(context);
  WorkspaceView state(context->workspace_, context->numRanks_, work.numExperts_);
  const LatencyStorageLayout layout(context->localBufferBase_, work.maxTokensPerRank_, Hidden, context->numRanks_,
                                    work.numExperts_, work.numTopk_, DispatchLayout::RANK_MAJOR_TOPK_EXPANDED,
                                    CombineMode::RANK_LOCAL_REDUCE);
  const uint64_t target = state.combineArrivedBaseline_[context->rank_] + 1;
  if (blockIdx.x == 0)
    publishReady(transport, layout, state, context->numRanks_, target);
  else
    gather<Hidden, UseTma>(output, input, ids, weights, work, transport, state, context->numRanks_, target, shared);
  retireAndAck(transport, layout, state, context->numRanks_);
  if (blockIdx.x == 0 && threadIdx.x == 0) state.combineArrivedBaseline_[context->rank_] = target;
#endif
}
}  // namespace mscclpp::ep::topk_expanded::ipc