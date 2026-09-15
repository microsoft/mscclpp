// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

// Included inside topk_expanded.cu's implementation namespace, after the
// baseline helpers. No public allocation, row, weight, or generation changes.
namespace gpunetio_fast {
#if MSCCLPP_BULK_AVAILABLE

template <int Hidden>
__device__ void send(void* output, int* outputIds, float* outputWeights, const void* input, const int64_t* topkIds,
                     const float* weights, const Workload& work, const TransportView& transport, const Layout& layout,
                     WorkspaceView& state, int ranks, int* sharedMem) {
  RankMajorSendState<Hidden> s;
  if (!initRankMajorSendState(s, work.numTokens_, work.numTopk_, ranks, gridDim.x - DispatchControlBlocks, sharedMem))
    return;
  const int lane = s.laneId_, topk = work.numTopk_, localExperts = work.numExperts_ / ranks;
  constexpr size_t bytes = Hidden * sizeof(Bf16);
  [[maybe_unused]] constexpr int vectors = bytes / sizeof(int4);
  const size_t rows = static_cast<size_t>(work.maxTokensPerRank_) * topk;
  int completedTokens = 0;
  [[maybe_unused]] int sinceFlush = 0;
  for (int token = s.firstTokenIdx_; token < work.numTokens_; token += s.tokenStride_) {
    if (lane == 0) {
      s.bulkBarrier_->arriveAndExpect(bytes);
      mscclpp::bulkLoad(s.stagedToken_, static_cast<const uint8_t*>(input) + static_cast<size_t>(token) * bytes, bytes,
                        *s.bulkBarrier_);
    }
    const size_t selection = static_cast<size_t>(token) * topk + lane;
    const int64_t expert = lane < topk ? topkIds[selection] : -1;
    const int dst = validExpert(expert, work.numExperts_) ? static_cast<int>(expert / localExperts) : -1;
    const float weight = lane < topk ? (weights == nullptr ? 1.0f : weights[selection]) : 0.0f;
    const size_t row = static_cast<size_t>(transport.rank_) * rows + selection;
    if (lane == 0) s.bulkBarrier_->wait(s.bulkPhase_);
    __syncwarp();
    mscclpp::bulkFence();
    // Start mapped payload stores before dense metadata, as compact dispatch
    // does. Wait below, before the tile can be reused or completion published.
    const bool mapped = dst >= 0 && transport.isNvlinkPeer(dst);
    if (mapped) {
      auto* remote = static_cast<uint8_t*>(transport.mappedBuffer(output, dst)) + row * bytes;
      mscclpp::bulkStore(remote, s.stagedToken_, bytes);
      mscclpp::bulkStoreCommit();
    }
    if (lane < topk) {
      for (int peer = 0; peer < ranks; ++peer) {
        const bool ipcPeer = transport.isNvlinkPeer(peer);
        auto* ids = ipcPeer ? static_cast<int*>(transport.mappedBuffer(outputIds, peer))
                            : static_cast<int*>(layout.expandedSendIds_);
        auto* wgts = ipcPeer ? static_cast<float*>(transport.mappedBuffer(outputWeights, peer))
                             : static_cast<float*>(layout.expandedSendWeights_);
        const size_t offset = ipcPeer ? row : static_cast<size_t>(peer) * rows + selection;
        ids[offset] = peer == dst ? static_cast<int>(expert) : work.invalidTokenExpertId_;
        wgts[offset] = peer == dst ? weight : 0.0f;
      }
    }
    if (mapped) mscclpp::bulkStoreWait();
#if defined(MSCCLPP_USE_GPUNETIO)
    {
      const bool remote = dst >= 0 && !transport.isNvlinkPeer(dst);
      if (__any_sync(0xffffffff, remote)) {
        auto* staged = reinterpret_cast<int4*>(static_cast<uint8_t*>(layout.gpuNetIoStagingBuffer_) +
                                               static_cast<size_t>(token) * layout.gpuNetIoSlotStride_);
        const auto* tile = reinterpret_cast<const int4*>(s.stagedToken_);
        for (int v = lane; v < vectors; v += WARP_SIZE) staged[v] = tile[v];
        __syncwarp();
        __threadfence_system();
        __syncwarp();
        if (remote) {
          const int q = static_cast<int>(expert % localExperts) % transport.gpuNetIo_->numQpsPerPeer;
          transport.gpuNetIo_->put(dst, transport.symmetricOffset(static_cast<uint8_t*>(output) + row * bytes),
                                   transport.symmetricOffset(staged), bytes, q);
        }
        __syncwarp();
      }
      if (++sinceFlush >= GpuNetIoFlushInterval) {
        for (int peer = 0; peer < ranks; ++peer) {
          if (transport.isNvlinkPeer(peer)) continue;
          for (int q = lane; q < transport.gpuNetIo_->numQpsPerPeer; q += WARP_SIZE)
            transport.gpuNetIo_->flush(peer, q);
        }
        __syncwarp();
        sinceFlush = 0;
      }
    }
#endif
    ++completedTokens;
    __syncwarp();
  }
  // Dense metadata covers every peer for every token. A single aggregate token
  // count proves the same completion as R identical K*T counters. Every writer
  // retains its SYSTEM fence; warp synchronization precedes the leader's RMW.
  __threadfence_system();
  __syncwarp();
  if (lane == 0 && completedTokens != 0)
    mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(state.dispatchRankPayloadCompletions_, completedTokens,
                                                       mscclpp::memoryOrderRelease);
}

__device__ void notify(int* outputIds, float* outputWeights, const int64_t* ids, const Workload& work,
                       const TransportView& transport, const Layout& layout, WorkspaceView& state, int ranks,
                       int* counts) {
  const int lane = get_lane_id(), warp = threadIdx.x / WARP_SIZE;
  const int topk = work.numTopk_, localExperts = work.numExperts_ / ranks;
  for (int peer = threadIdx.x; peer < ranks; peer += blockDim.x) counts[peer] = 0;
  __syncthreads();
  for (int token = warp; token < work.numTokens_; token += DispatchNWarps) {
    const int64_t expert = lane < topk ? ids[static_cast<size_t>(token) * topk + lane] : -1;
    if (validExpert(expert, work.numExperts_)) atomicAdd_block(counts + expert / localExperts, 1);
  }
  __syncthreads();
  const size_t rows = static_cast<size_t>(work.maxTokensPerRank_) * topk;
  for (int peer = 0; peer < ranks; ++peer) {
    const bool mapped = transport.isNvlinkPeer(peer);
    auto* dstIds = mapped ? static_cast<int*>(transport.mappedBuffer(outputIds, peer))
                          : static_cast<int*>(layout.expandedSendIds_);
    auto* dstWeights = mapped ? static_cast<float*>(transport.mappedBuffer(outputWeights, peer))
                              : static_cast<float*>(layout.expandedSendWeights_);
    const size_t base = static_cast<size_t>(mapped ? transport.rank_ : peer) * rows;
    for (size_t i = static_cast<size_t>(work.numTokens_) * topk + threadIdx.x; i < rows; i += blockDim.x) {
      dstIds[base + i] = work.invalidTokenExpertId_;
      dstWeights[base + i] = 0.0f;
    }
  }
  __threadfence_system();
  if (threadIdx.x == 0) {
    while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(state.dispatchRankPayloadCompletions_,
                                                          mscclpp::memoryOrderAcquire) != work.numTokens_) {
    }
    *state.dispatchRankPayloadCompletions_ = 0;
  }
  __syncthreads();
  for (int peer = threadIdx.x; peer < ranks; peer += blockDim.x) {
    if (transport.isNvlinkPeer(peer)) {
      static_cast<int*>(transport.mappedBuffer(layout.expandedCounts_, peer))[transport.rank_] = counts[peer];
      __threadfence_system();
      auto* flag = static_cast<uint64_t*>(transport.mappedBuffer(layout.gpuNetIoFlagsBuffer_, peer)) +
                   static_cast<size_t>(transport.rank_) * GpuNetIoMaxQpsPerPeer;
      signalLocal(flag);
    } else {
      static_cast<int*>(layout.expandedCountStaging_)[peer] = counts[peer];
    }
  }
  __threadfence_system();
}

// All local readers, payload writers and outgoing CQ drains retire before the
// controller sends the ORIGINAL all-peer lifetime ACK. No fence-scope weakening.
__device__ void retireNetwork(const TransportView& transport, const Layout& layout, WorkspaceView& state, int ranks) {
  __threadfence_system();
  __syncthreads();
  if (blockIdx.x != 0) {
    if (threadIdx.x == 0)
      mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(state.dispatchNumRecvTasks_, 1, mscclpp::memoryOrderRelease);
    return;
  }
  if (threadIdx.x == 0) {
    while (mscclpp::atomicLoad<int, mscclpp::scopeDevice>(state.dispatchNumRecvTasks_, mscclpp::memoryOrderAcquire) !=
           static_cast<int>(gridDim.x) - 1) {
    }
    *state.dispatchNumRecvTasks_ = 0;
  }
  __syncthreads();
  finishCollective(transport, layout, ranks);
}

__device__ void postDispatch(const TransportView& transport, const Layout& layout, const Workload& work, int ranks) {
#if defined(MSCCLPP_USE_GPUNETIO)
  auto* gin = transport.gpuNetIo_;
  const size_t rows = static_cast<size_t>(work.maxTokensPerRank_) * work.numTopk_;
  for (int peer = threadIdx.x; peer < ranks; peer += blockDim.x) {
    if (transport.isNvlinkPeer(peer)) continue;
    const size_t source = static_cast<size_t>(peer) * rows, dest = static_cast<size_t>(transport.rank_) * rows;
    if (rows * sizeof(int) > DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE) {
      // The ordinary put splits oversized transfers; the fixed three-WQE helper
      // deliberately does not. Preserve the baseline's large-capacity contract.
      gin->put(peer, transport.symmetricOffset(static_cast<int*>(layout.expandedCounts_) + transport.rank_),
               transport.symmetricOffset(static_cast<int*>(layout.expandedCountStaging_) + peer), sizeof(int), 0);
      gin->put(peer, transport.symmetricOffset(static_cast<int*>(layout.rankMajorTopkIdsBuffer_) + dest),
               transport.symmetricOffset(static_cast<int*>(layout.expandedSendIds_) + source), rows * sizeof(int), 0);
      gin->put(peer, transport.symmetricOffset(static_cast<float*>(layout.rankMajorTopkWeightsBuffer_) + dest),
               transport.symmetricOffset(static_cast<float*>(layout.expandedSendWeights_) + source),
               rows * sizeof(float), 0);
      continue;
    }
    gin->putBatched3(peer, 0, transport.symmetricOffset(static_cast<int*>(layout.expandedCounts_) + transport.rank_),
                     transport.symmetricOffset(static_cast<int*>(layout.expandedCountStaging_) + peer), sizeof(int),
                     transport.symmetricOffset(static_cast<int*>(layout.rankMajorTopkIdsBuffer_) + dest),
                     transport.symmetricOffset(static_cast<int*>(layout.expandedSendIds_) + source), rows * sizeof(int),
                     transport.symmetricOffset(static_cast<float*>(layout.rankMajorTopkWeightsBuffer_) + dest),
                     transport.symmetricOffset(static_cast<float*>(layout.expandedSendWeights_) + source),
                     rows * sizeof(float));
  }
  __syncthreads();
  auto* flags = static_cast<uint64_t*>(layout.gpuNetIoFlagsBuffer_) +
                static_cast<size_t>(transport.rank_) * GpuNetIoMaxQpsPerPeer;
  // Mark EVERY queue, including empty queues, only after all payload posts.
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

#if defined(MSCCLPP_USE_GPUNETIO)
// All 32 lanes participate, with potentially sparse valid rows. A single
// reservation/submit per tile replaces serial per-row doorbells. Each WQE is
// STILL signaled: CQ ticket-index recycling depends on this, including wrap.
__device__ int putWarpRows(mscclpp::GpuNetIoDeviceContext* gin, int peer, int q, bool live, uint64_t dstOffset,
                           uint64_t srcOffset, uint32_t bytes) {
  const int lane = get_lane_id();
  const unsigned mask = __ballot_sync(0xffffffff, live);
  const int count = __popc(mask);
  if (count == 0) return 0;
  auto* qp = mscclpp::detail::ginQp(gin->qps, peer * gin->numQpsPerPeer + q);
  uint64_t base = 0;
  if (lane == 0)
    base = doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
        qp, count, DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT);
  base = __shfl_sync(0xffffffff, base, 0);
  if (live) {
    const int offset = __popc(mask & ((1u << lane) - 1));
    const uint64_t ticket = base + offset;
    auto* wqe = doca_gpu_dev_verbs_get_wqe_ptr(qp, ticket);
    doca_gpu_dev_verbs_wqe_prepare_write(qp, wqe, ticket, DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE,
                                         DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE, 0, gin->peerBase[peer] + dstOffset,
                                         mscclpp::detail::ginRemoteKey(*gin, peer, q), gin->localBase + srcOffset,
                                         mscclpp::detail::ginHtobe32(mscclpp::detail::ginLocalKey(*gin, q)), bytes);
    __threadfence_system();
  }
  __syncwarp();
  if (lane == 0) {
    doca_gpu_dev_verbs_mark_wqes_ready<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(qp, base, base + count - 1);
    doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU, DOCA_GPUNETIO_VERBS_SYNC_SCOPE_THREAD,
                              DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO>(qp, base + count,
                                                                    DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT);
  }
  __syncwarp();
  return count;
}
#endif

template <int Hidden>
__device__ void pushCombine(const void* input, const Layout& layout, const TransportView& transport,
                            const Workload& work, int owner) {
#if defined(MSCCLPP_USE_GPUNETIO)
  if (transport.isNvlinkPeer(owner)) return;
  auto* gin = transport.gpuNetIo_;
  const int lane = get_lane_id(), warp = threadIdx.x / WARP_SIZE;
  const size_t rows = static_cast<size_t>(work.maxTokensPerRank_) * work.numTopk_;
  constexpr size_t bytes = Hidden * sizeof(Bf16);
  const auto* ids = static_cast<const int*>(layout.rankMajorTopkIdsBuffer_) + owner * rows;
  const auto* wgts = static_cast<const float*>(layout.rankMajorTopkWeightsBuffer_) + owner * rows;
  // Same HCA stripe boundaries and matching QPs as baseline. Support up to 64
  // HCAs by having a warp visit a second stripe when necessary.
  for (int stripe = warp; stripe < gin->numHcas; stripe += CombineNThreads / WARP_SIZE) {
    const int q = markerQp(transport, owner, stripe);
    const size_t begin = rows * stripe / gin->numHcas, end = rows * (stripe + 1) / gin->numHcas;
    int outstanding = 0;
    for (size_t tile = begin; tile < end; tile += WARP_SIZE) {
      const size_t row = tile + lane;
      const bool live = row < end && validExpert(ids[row], work.numExperts_) && wgts[row] != 0.0f;
      const uint64_t src = transport.symmetricOffset(const_cast<void*>(input)) + (owner * rows + row) * bytes;
      const uint64_t dst = transport.symmetricOffset(layout.gpuNetIoCombineLandingBuffer_) +
                           (static_cast<size_t>(transport.rank_) * rows + row) * bytes;
      outstanding += putWarpRows(gin, owner, q, live, dst, src, bytes);
      if (outstanding >= GpuNetIoFlushInterval) {
        if (lane == 0) gin->flush(owner, q);
        __syncwarp();
        outstanding = 0;
      }
    }
    if (lane == 0) {
      auto* flag = static_cast<uint64_t*>(layout.gpuNetIoCombineFlagsBuffer_) +
                   static_cast<size_t>(transport.rank_) * GpuNetIoMaxQpsPerPeer + q;
      // Even a completely empty stripe publishes its ordered generation.
      gin->atomicAdd(owner, transport.symmetricOffset(flag), 1, q);
    }
    __syncwarp();
  }
#endif
}
#endif  // MSCCLPP_BULK_AVAILABLE

template <int Hidden>
__global__ __launch_bounds__(DispatchNThreads,
                             1) void dispatchKernel(void* output, int* ids, float* weightsOut, int* counts,
                                                    const void* input, const int64_t* topkIds, const float* weights,
                                                    Workload work, CommContext comm, void* workspace) {
#if MSCCLPP_BULK_AVAILABLE
  extern __shared__ __align__(128) uint8_t shared[];
  const TransportView transport(comm);
  validateGpuNetIo(transport);
  WorkspaceView state(workspace, comm.numRanks_, work.numExperts_);
  const Layout layout(comm.symmetricBufferBase_, work.maxTokensPerRank_, Hidden, comm.numRanks_, work.numExperts_,
                      work.numTopk_, true, true);
  const uint64_t target = state.dispatchArrivedBaseline_[comm.rank_] + 1;
  if (blockIdx.x == gridDim.x - 1)
    notify(ids, weightsOut, topkIds, work, transport, layout, state, comm.numRanks_, reinterpret_cast<int*>(shared));
  else
    send<Hidden>(output, ids, weightsOut, input, topkIds, weights, work, transport, layout, state, comm.numRanks_,
                 reinterpret_cast<int*>(shared));
  // Retain the posting barrier: a marker must never overtake another CTA's
  // payload reservation. Only the redundant end-of-kernel grids are removed.
  __threadfence_system();
  state.combineSyncer_->sync(gridDim.x);
  if (blockIdx.x == gridDim.x - 1) postDispatch(transport, layout, work, comm.numRanks_);
  if (blockIdx.x < comm.numRanks_ && threadIdx.x == 0) {
    waitSource(transport, static_cast<uint64_t*>(layout.gpuNetIoFlagsBuffer_), blockIdx.x, target, false);
    counts[blockIdx.x] = static_cast<int*>(layout.expandedCounts_)[blockIdx.x];
  }
  retireNetwork(transport, layout, state, comm.numRanks_);
  if (blockIdx.x == 0 && threadIdx.x == 0) state.dispatchArrivedBaseline_[comm.rank_] = target;
#endif
}

template <int Hidden, bool UseTma>
__global__ __launch_bounds__(CombineNThreads, 1) void combineKernel(void* output, const void* input, const int64_t* ids,
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
  if (blockIdx.x == 0) {
    for (int peer = threadIdx.x; peer < comm.numRanks_; peer += blockDim.x) {
      if (!transport.isNvlinkPeer(peer)) continue;
      signalLocal(static_cast<uint64_t*>(transport.mappedBuffer(flags, peer)) +
                  static_cast<size_t>(comm.rank_) * GpuNetIoMaxQpsPerPeer);
    }
  }
  if (blockIdx.x < comm.numRanks_) pushCombine<Hidden>(input, layout, transport, work, blockIdx.x);
  if (blockIdx.x == 0) {
    // One system-scope poller per source, not one poller per top-k lane/worker.
    for (int source = threadIdx.x; source < comm.numRanks_; source += blockDim.x) {
      waitSource(transport, flags, source, target, true);
      mscclpp::atomicStore<uint32_t, mscclpp::scopeDevice>(state.combineRankReadyEpochs_ + source,
                                                           static_cast<uint32_t>(target), mscclpp::memoryOrderRelease);
    }
  } else if constexpr (UseTma) {
    recvRankMajorTopkExpandedRemotePartialsTma<Hidden, true>(output, input, ids, weights, work, transport, layout,
                                                             comm.numRanks_, target, shared, &state);
  } else {
    recvRankMajorTopkExpandedRemotePartials<Hidden, true>(output, input, ids, weights, work, transport, layout,
                                                          comm.numRanks_, target, &state);
  }
#if defined(MSCCLPP_USE_GPUNETIO)
  if (blockIdx.x < comm.numRanks_ && !transport.isNvlinkPeer(blockIdx.x)) {
    auto* gin = transport.gpuNetIo_;
    for (int stripe = threadIdx.x; stripe < gin->numHcas; stripe += blockDim.x)
      gin->flush(blockIdx.x, markerQp(transport, blockIdx.x, stripe));
  }
#endif
  retireNetwork(transport, layout, state, comm.numRanks_);
  if (blockIdx.x == 0 && threadIdx.x == 0) state.combineArrivedBaseline_[comm.rank_] = target;
#endif
}
}  // namespace gpunetio_fast
