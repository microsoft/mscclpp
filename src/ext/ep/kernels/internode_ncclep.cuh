// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// NCCL-EP-ported warp-specialized HT dispatch kernel (dispatch_ncclep).
// Guarded by EP_DISPATCH_NCCLEP. Included from internode.cu INSIDE the
// mscclpp::ep::internode namespace, AFTER all shared helpers/buffers are
// defined. When the guard is OFF this file is not compiled and the
// production dispatch<> kernel is used unchanged.
#ifndef MSCCLPP_EP_INTERNODE_NCCLEP_CUH_
#define MSCCLPP_EP_INTERNODE_NCCLEP_CUH_
#ifdef EP_DISPATCH_NCCLEP

template <bool kLowLatencyMode, int kNumRDMARanks, bool kCachedMode, int kNumDispatchRDMASenderWarps,
          int kNumTopkRDMARanks = get_num_topk_rdma_ranks(kNumRDMARanks)>
__global__ void __launch_bounds__(((kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS) * 32), 1)
    dispatch_ncclep(int4* recv_x, float* recv_x_scales, int64_t* recv_topk_idx, float* recv_topk_weights,
             SourceMeta* recv_src_meta, const int4* x, const float* x_scales, const int64_t* topk_idx,
             const float* topk_weights, int* send_rdma_head, int* send_nvl_head, int* recv_rdma_channel_prefix_matrix,
             int* recv_gbl_channel_prefix_matrix, const int* rdma_channel_prefix_matrix,
             const int* recv_rdma_rank_prefix_sum, const int* gbl_channel_prefix_matrix,
             const int* recv_gbl_rank_prefix_sum, int num_tokens, int hidden_int4, int num_scales, int num_topk,
             int num_experts, const bool* is_token_in_rank, void* rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens,
             int num_max_rdma_chunked_recv_tokens, void** buffer_ptrs, int num_max_nvl_chunked_send_tokens,
             int num_max_nvl_chunked_recv_tokens, int rank, int num_ranks,
             mscclpp::PortChannelDeviceHandle* port_channel_handles,
             mscclpp::MemoryChannelDeviceHandle* memory_channel_handles,
             // Phase 3 NVLS counter pointers (nullptr → fall back to PortChannel/atomicAdd path).
             void* nvls_head_mc, void* nvls_head_dev, void* nvls_tail_mc, void* nvls_tail_dev,
             // Phase 4: per-peer fabric-IPC base pointers; when non-null, cross-node data
             // PUTs go directly through NVL72 fabric VA instead of `handle.put` over IB.
             void* const* peer_rdma_bases) {
  enum class WarpRole {
    kRDMASender,
    kRDMASenderCoordinator,
    kRDMAAndNVLForwarder,
    kForwarderCoordinator,
    kNVLReceivers
  };

  const auto sm_id = static_cast<int>(blockIdx.x);
  const auto thread_id = static_cast<int>(threadIdx.x), warp_id = thread_id / 32, lane_id = get_lane_id();
  const auto num_channels = static_cast<int>(gridDim.x) / 2, channel_id = sm_id / 2;
  const bool is_forwarder = sm_id % 2 == 0;
  const auto rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;

  const auto role_meta = [=]() -> std::pair<WarpRole, int> {
    if (is_forwarder) {
      if (warp_id < NUM_MAX_NVL_PEERS) {
        return {WarpRole::kRDMAAndNVLForwarder, (warp_id + channel_id) % NUM_MAX_NVL_PEERS};
      } else {
        return {WarpRole::kForwarderCoordinator, warp_id - NUM_MAX_NVL_PEERS};
      }
    } else if (warp_id < kNumDispatchRDMASenderWarps) {
      return {WarpRole::kRDMASender, -1};
    } else if (warp_id == kNumDispatchRDMASenderWarps) {
      return {WarpRole::kRDMASenderCoordinator, -1};
    } else {
      return {WarpRole::kNVLReceivers, (warp_id + channel_id - kNumDispatchRDMASenderWarps) % NUM_MAX_NVL_PEERS};
    }
  }();
  auto warp_role = role_meta.first;
  auto target_rank = role_meta.second;  // Not applicable for RDMA senders
  EP_DEVICE_ASSERT(num_warps == kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS);
  if (thread_id == 0 && sm_id == 0) {
    (void)0;
  }

  // Data checks
  EP_DEVICE_ASSERT(num_topk <= 32);

  // RDMA symmetric layout (packed-bool size guard is at namespace scope via NvlPackT).
  // Snapshot the original base before SymBuffer constructors advance it; used
  // below to compute MR-relative offsets for handle.put().
  void* const rdma_buffer_ptr_base = rdma_buffer_ptr;
  auto hidden_bytes = hidden_int4 * sizeof(int4);
  auto num_bytes_per_rdma_token = get_num_bytes_per_rdma_token(hidden_int4, num_scales, num_topk, num_topk);
  auto rdma_channel_data =
      SymBuffer<int8_t>(rdma_buffer_ptr, num_max_rdma_chunked_recv_tokens * num_bytes_per_rdma_token, kNumRDMARanks,
                        channel_id, num_channels);
  auto rdma_channel_meta =
      SymBuffer<int>(rdma_buffer_ptr, NUM_MAX_NVL_PEERS * 2 + 2, kNumRDMARanks, channel_id, num_channels);
  auto rdma_channel_head = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);
  auto rdma_channel_tail = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);
  // Scratch slots (one uint64_t per (channel, peer) per kind) holding the
  // absolute counter values used as RDMA WRITE source. Replaces the broken
  // HW atomicAdd path on Azure CX-7 RoCE (IBV_ATOMIC_NONE) — each tail/head
  // slot has a single writer per peer, so atomicity is unnecessary; an
  // absolute-value RDMA WRITE through the same QP as the data PUT preserves
  // ordering by IB semantics.
  auto rdma_channel_tail_send_src =
      SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);
  auto rdma_channel_head_send_src =
      SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);

  auto data_send_offset =
      sizeof(int8_t) * (num_max_rdma_chunked_recv_tokens * num_bytes_per_rdma_token) * kNumRDMARanks * channel_id;
  auto data_recv_offset = sizeof(int8_t) * (num_max_rdma_chunked_recv_tokens * num_bytes_per_rdma_token) *
                          kNumRDMARanks * (channel_id + num_channels);
  auto meta_offset =
      sizeof(int8_t) * (num_max_rdma_chunked_recv_tokens * num_bytes_per_rdma_token) * kNumRDMARanks * num_channels * 2;
  auto meta_send_offset = meta_offset + sizeof(int) * (NUM_MAX_NVL_PEERS * 2 + 2) * kNumRDMARanks * channel_id;
  auto meta_recv_offset =
      meta_offset + sizeof(int) * (NUM_MAX_NVL_PEERS * 2 + 2) * kNumRDMARanks * (channel_id + num_channels);
  auto head_offset = meta_offset + sizeof(int) * (NUM_MAX_NVL_PEERS * 2 + 2) * kNumRDMARanks * num_channels * 2;
  auto head_send_offset = head_offset + sizeof(uint64_t) * kNumRDMARanks * channel_id;
  auto tail_offset = head_offset + sizeof(uint64_t) * kNumRDMARanks * num_channels;
  auto tail_send_offset = tail_offset + sizeof(uint64_t) * kNumRDMARanks * channel_id;

  // NVL buffer layouts
  // NOTES: `rs_wr_buffer_ptr` means "Read for Senders, Write for Receivers", `ws_rr_buffer_ptr` means "Write for
  // Senders, Read for Receivers"
  void *rs_wr_buffer_ptr = nullptr, *ws_rr_buffer_ptr = nullptr;
  int rs_wr_rank = 0, ws_rr_rank = 0;
  if (warp_role == WarpRole::kRDMAAndNVLForwarder)
    rs_wr_buffer_ptr = buffer_ptrs[nvl_rank], ws_rr_buffer_ptr = buffer_ptrs[target_rank], rs_wr_rank = nvl_rank,
    ws_rr_rank = target_rank;
  if (warp_role == WarpRole::kNVLReceivers)
    rs_wr_buffer_ptr = buffer_ptrs[target_rank], ws_rr_buffer_ptr = buffer_ptrs[nvl_rank], rs_wr_rank = target_rank,
    ws_rr_rank = nvl_rank;

  // Allocate buffers
  auto nvl_channel_x = AsymBuffer<int4>(ws_rr_buffer_ptr, num_max_nvl_chunked_recv_tokens * hidden_int4,
                                        NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank)
                           .advance_also(rs_wr_buffer_ptr);
  auto nvl_channel_src_meta = AsymBuffer<SourceMeta>(ws_rr_buffer_ptr, num_max_nvl_chunked_recv_tokens,
                                                     NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank)
                                  .advance_also(rs_wr_buffer_ptr);
  auto nvl_channel_x_scales = AsymBuffer<float>(ws_rr_buffer_ptr, num_max_nvl_chunked_recv_tokens * num_scales,
                                                NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank)
                                  .advance_also(rs_wr_buffer_ptr);
  auto nvl_channel_topk_idx = AsymBuffer<int>(ws_rr_buffer_ptr, num_max_nvl_chunked_recv_tokens * num_topk,
                                              NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank)
                                  .advance_also(rs_wr_buffer_ptr);
  auto nvl_channel_topk_weights = AsymBuffer<float>(ws_rr_buffer_ptr, num_max_nvl_chunked_recv_tokens * num_topk,
                                                    NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank)
                                      .advance_also(rs_wr_buffer_ptr);
  auto nvl_channel_prefix_start =
      AsymBuffer<int>(ws_rr_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank)
          .advance_also(rs_wr_buffer_ptr);
  auto nvl_channel_prefix_end =
      AsymBuffer<int>(ws_rr_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank)
          .advance_also(rs_wr_buffer_ptr);
  auto nvl_channel_head = AsymBuffer<int>(rs_wr_buffer_ptr, 1, NUM_MAX_NVL_PEERS, channel_id, num_channels, ws_rr_rank)
                              .advance_also(ws_rr_buffer_ptr);
  auto nvl_channel_tail = AsymBuffer<int>(ws_rr_buffer_ptr, 1, NUM_MAX_NVL_PEERS, channel_id, num_channels, rs_wr_rank)
                              .advance_also(rs_wr_buffer_ptr);

  // RDMA sender warp synchronization
  __shared__ volatile int rdma_send_next_token_idx;
  __shared__ volatile int rdma_send_channel_tail[kNumRDMARanks];
  __shared__ volatile int rdma_send_channel_next_tail[kNumRDMARanks];
  auto sync_rdma_sender_smem = []() { asm volatile("bar.sync 0, %0;" ::"r"((kNumDispatchRDMASenderWarps + 1) * 32)); };

  // Forward warp synchronization
  __shared__ volatile int forward_channel_head[NUM_MAX_NVL_PEERS][kNumRDMARanks];
  __shared__ volatile bool forward_channel_retired[NUM_MAX_NVL_PEERS];
  auto sync_forwarder_smem = []() { asm volatile("bar.sync 1, %0;" ::"r"((NUM_MAX_NVL_PEERS + 1) * 32)); };

  if (warp_role == WarpRole::kRDMASender) {
    // Get tasks
    int token_start_idx, token_end_idx;
    get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

    // Clean shared memory
    EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA ranks");
    (warp_id == 0 and lane_id == 0) ? (rdma_send_next_token_idx = token_start_idx) : 0;
    (warp_id == 0 and lane_id < kNumRDMARanks) ? (rdma_send_channel_tail[lane_id] = 0) : 0;
    (warp_id == 0 and lane_id < kNumRDMARanks) ? (rdma_send_channel_next_tail[lane_id] = 0) : 0;

    // Send number of tokens in this channel by `-value - 1`
    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS * 2 + 2 <= 32, "Invalid number of NVL peers");
    for (int dst_rdma_rank = warp_id; dst_rdma_rank < kNumRDMARanks; dst_rdma_rank += kNumDispatchRDMASenderWarps) {
      auto dst_ptr = dst_rdma_rank == rdma_rank ? rdma_channel_meta.recv_buffer(dst_rdma_rank)
                                                : rdma_channel_meta.send_buffer(dst_rdma_rank);
      if (lane_id < NUM_MAX_NVL_PEERS) {
        dst_ptr[lane_id] =
            -(channel_id == 0 ? 0
                              : gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS + lane_id) * num_channels +
                                                          channel_id - 1]) -
            1;
      } else if (lane_id < NUM_MAX_NVL_PEERS * 2) {
        dst_ptr[lane_id] =
            -gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS + lane_id - NUM_MAX_NVL_PEERS) *
                                           num_channels +
                                       channel_id] -
            1;
      } else if (lane_id == NUM_MAX_NVL_PEERS * 2) {
        dst_ptr[lane_id] =
            -(channel_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1]) - 1;
      } else if (lane_id == NUM_MAX_NVL_PEERS * 2 + 1) {
        dst_ptr[lane_id] = -rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id] - 1;
      }
      __syncwarp();

      if (dst_rdma_rank == rdma_rank) continue;

      // Issue RDMA for non-local ranks
      if (peer_rdma_bases != nullptr) {
        // Phase 4: deliver meta directly via NVL72 fabric VA (cooperative
        // int copy across the warp). Replaces port_channel.put which is
        // unreliable cross-node on Azure CX-7 RoCE without flush.
        const auto num_bytes = sizeof(int) * (NUM_MAX_NVL_PEERS * 2 + 2);
        const auto dst_offset = rdma_rank * num_bytes + meta_recv_offset;
        const auto src_offset = dst_rdma_rank * num_bytes + meta_send_offset;
        const int dst_rank_global = dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank;
        const int* src_p = reinterpret_cast<const int*>(reinterpret_cast<uint8_t*>(rdma_buffer_ptr_base) + src_offset);
        int* dst_p = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(peer_rdma_bases[dst_rank_global]) + dst_offset);
        const int n_int = (int)(num_bytes / sizeof(int));
        for (int k = lane_id; k < n_int; k += 32) {
          dst_p[k] = src_p[k];
        }
        __threadfence_system();
      } else if (lane_id == 0) {
        auto num_bytes = sizeof(int) * (NUM_MAX_NVL_PEERS * 2 + 2);
        auto dst_offset = rdma_rank * num_bytes + meta_recv_offset;
        auto src_offset = dst_rdma_rank * num_bytes + meta_send_offset;
        auto port_channel_idx = kLowLatencyMode
                                    ? (channel_id * kNumRDMARanks + dst_rdma_rank)
                                    : (channel_id * num_ranks + dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank);
        port_channel_handles[port_channel_idx].put(dst_offset, src_offset, num_bytes);
        // port_channel_handles[port_channel_idx].flush();
      }
      __syncwarp();
    }
    sync_rdma_sender_smem();

    // Iterate over tokens and copy into buffer
    int64_t token_idx;
    int cached_rdma_channel_head = 0, last_rdma_tail_idx = -1;
    auto send_buffer =
        lane_id == rdma_rank ? rdma_channel_data.recv_buffer(lane_id) : rdma_channel_data.send_buffer(lane_id);
    for (token_idx = token_start_idx + warp_id; token_idx < token_end_idx; token_idx += kNumDispatchRDMASenderWarps) {
      // Read RDMA rank existence
      NvlPackT is_token_in_rank_uint64 = 0;
      if (lane_id < kNumRDMARanks)
        is_token_in_rank_uint64 =
            *reinterpret_cast<const NvlPackT*>(is_token_in_rank + token_idx * num_ranks + lane_id * NUM_MAX_NVL_PEERS);

      // Acquire sequential lock
      while (lane_id == 0 and rdma_send_next_token_idx != token_idx)
        ;
      __syncwarp();

      // Acquire next tail
      int rdma_tail_idx = -1;
      if (is_token_in_rank_uint64 != 0) {
        rdma_tail_idx = rdma_send_channel_next_tail[lane_id]++;
        while (rdma_tail_idx - cached_rdma_channel_head >= num_max_rdma_chunked_recv_tokens) {
          // Phase 4: head feedback path \u2014 cross-node uses fabric-VA store,
          // self-loop uses local atomic. Both end up in rdma_channel_head;
          // single read via ld_volatile_global covers them. NVLS removed.
          cached_rdma_channel_head = static_cast<int>(ld_volatile_global(rdma_channel_head.buffer(lane_id)));
        }
      }
      __syncwarp();

      // Store RDMA head for combine
      if (lane_id < kNumRDMARanks and not kCachedMode)
        send_rdma_head[token_idx * kNumRDMARanks + lane_id] = rdma_tail_idx;

      // Update last token tail. In-loop writes are sequenced by the
      // per-channel sequential lock and the warp-stride property of the
      // token loop, so monotonicity is guaranteed and a plain
      // st_release_cta is correct AND faster than atomicMax (which
      // would serialize through L2 if the compiler can't infer shared
      // address space). The epilogue (out of the seq-lock contract for
      // the highest in-rank slot) needs atomicMax separately.
      if (last_rdma_tail_idx >= 0)
        st_release_cta(const_cast<const int*>(rdma_send_channel_tail + lane_id), last_rdma_tail_idx + 1);
      last_rdma_tail_idx = rdma_tail_idx;

      // Release sequential lock
      lane_id == 0 ? (rdma_send_next_token_idx += 1) : 0;

      // Broadcast tails
      SourceMeta src_meta;
      int num_topk_ranks = 0, topk_ranks[kNumTopkRDMARanks];
      void* dst_send_buffers[kNumTopkRDMARanks];
#pragma unroll
      for (int i = 0, slot_idx; i < kNumRDMARanks; ++i)
        if ((slot_idx = __shfl_sync(0xffffffff, rdma_tail_idx, i)) >= 0) {
          slot_idx = slot_idx % num_max_rdma_chunked_recv_tokens;
          topk_ranks[num_topk_ranks] = i;
          auto recv_is_token_in_rank_uint64 = broadcast(is_token_in_rank_uint64, i);
          auto recv_is_token_in_rank_values = reinterpret_cast<const bool*>(&recv_is_token_in_rank_uint64);
          if (lane_id == num_topk_ranks) src_meta = SourceMeta(rdma_rank, recv_is_token_in_rank_values);
          dst_send_buffers[num_topk_ranks++] =
              reinterpret_cast<uint8_t*>(broadcast(send_buffer, i)) + slot_idx * num_bytes_per_rdma_token;
        }
      EP_DEVICE_ASSERT(num_topk_ranks <= kNumTopkRDMARanks);

      // Copy `x` into symmetric send buffer
      auto st_broadcast = [=](const int key, const int4& value) {
#pragma unroll
        for (int j = 0; j < num_topk_ranks; ++j)
          st_na_global(reinterpret_cast<int4*>(dst_send_buffers[j]) + key, value);
      };
      UNROLLED_WARP_COPY(5, lane_id, hidden_int4, 0, x + token_idx * hidden_int4, ld_nc_global, st_broadcast);
#pragma unroll
      for (int i = 0; i < num_topk_ranks; ++i)
        dst_send_buffers[i] = reinterpret_cast<int4*>(dst_send_buffers[i]) + hidden_int4;

      // Copy source metadata into symmetric send buffer
      if (lane_id < num_topk_ranks) st_na_global(reinterpret_cast<SourceMeta*>(dst_send_buffers[lane_id]), src_meta);
#pragma unroll
      for (int i = 0; i < num_topk_ranks; ++i)
        dst_send_buffers[i] = reinterpret_cast<SourceMeta*>(dst_send_buffers[i]) + 1;

// Copy `x_scales` into symmetric send buffer
#pragma unroll
      for (int i = lane_id; i < num_scales; i += 32) {
        auto value = ld_nc_global(x_scales + token_idx * num_scales + i);
#pragma unroll
        for (int j = 0; j < num_topk_ranks; ++j) st_na_global(reinterpret_cast<float*>(dst_send_buffers[j]) + i, value);
      }
#pragma unroll
      for (int i = 0; i < num_topk_ranks; ++i)
        dst_send_buffers[i] = reinterpret_cast<float*>(dst_send_buffers[i]) + num_scales;

// Copy `topk_idx` and `topk_weights` into symmetric send buffer
#pragma unroll
      for (int i = lane_id; i < num_topk * num_topk_ranks; i += 32) {
        auto rank_idx = i / num_topk, copy_idx = i % num_topk;
        auto idx_value = static_cast<int>(ld_nc_global(topk_idx + token_idx * num_topk + copy_idx));
        auto weight_value = ld_nc_global(topk_weights + token_idx * num_topk + copy_idx);
        st_na_global(reinterpret_cast<int*>(dst_send_buffers[rank_idx]) + copy_idx, idx_value);
        st_na_global(reinterpret_cast<float*>(dst_send_buffers[rank_idx]) + num_topk + copy_idx, weight_value);
      }
    }

    // Epilogue
    // Acquire sequential lock
    while (lane_id == 0 and rdma_send_next_token_idx != token_idx)
      ;
    __syncwarp();

    // Update last token tail (epilogue). See in-loop note on atomicMax.
    if (last_rdma_tail_idx >= 0) atomicMax(const_cast<int*>(rdma_send_channel_tail + lane_id), last_rdma_tail_idx + 1);

    // Release sequential lock
    lane_id == 0 ? (rdma_send_next_token_idx += 1) : 0;
  } else if (warp_role == WarpRole::kRDMASenderCoordinator) {
    // NOTES: in case of splitting the issued put at the end of the buffer
    EP_DEVICE_ASSERT(num_max_rdma_chunked_recv_tokens % num_max_rdma_chunked_send_tokens == 0);

    // Synchronize shared memory
    sync_rdma_sender_smem();
    if (lane_id == 0 && channel_id == 0 && rank == 0) {
      (void)0;
    }

    // Get number of tokens to send for each RDMA rank
    int num_tokens_to_send = 0;
    if (lane_id < kNumRDMARanks) {
      num_tokens_to_send = rdma_channel_prefix_matrix[lane_id * num_channels + channel_id];
      if (channel_id > 0) num_tokens_to_send -= rdma_channel_prefix_matrix[lane_id * num_channels + channel_id - 1];
    }

    // Iterate all RDMA ranks
    int last_issued_tail = 0;
    while (__any_sync(0xffffffff, num_tokens_to_send > 0)) {
#pragma unroll
      for (int i = 0; i < kNumRDMARanks; ++i, __syncwarp()) {
        // To mitigate incast congestion, shuffle the starting index of target rank for different ranks and channels
        const int dst_rdma_rank = (i + channel_id + rdma_rank) % kNumRDMARanks;

        // -----------------------------------------------------------------
        // Phase 4 restructure: owner lane (lane_id == dst_rdma_rank) decides
        // whether to issue, then broadcasts num_tokens_to_issue and the
        // tail-base for slot indexing to all 32 lanes via shfl. All lanes
        // then participate in the cooperative cross-node fabric-IPC copy
        // (when peer_rdma_bases != nullptr). This replaces the legacy
        // single-lane handle.put(data) path which is broken on Azure CX-7.
        // -----------------------------------------------------------------
        const bool owner = (lane_id == dst_rdma_rank);
        int my_num_tokens_to_issue = 0;
        int my_issue_tail = last_issued_tail;
        if (owner && num_tokens_to_send > 0) {
          auto processed_tail = ld_acquire_cta(const_cast<const int*>(rdma_send_channel_tail + dst_rdma_rank));
          auto num_tokens_processed = processed_tail - last_issued_tail;
          if (num_tokens_processed == num_tokens_to_send || num_tokens_processed >= num_max_rdma_chunked_send_tokens) {
            int n = min(num_tokens_processed, num_max_rdma_chunked_send_tokens);
            EP_DEVICE_ASSERT(n >= 0 && n <= num_tokens_to_send);
            my_num_tokens_to_issue = n;
          }
        }

        const int n_issue = __shfl_sync(0xffffffff, my_num_tokens_to_issue, dst_rdma_rank);
        const int issue_tail = __shfl_sync(0xffffffff, my_issue_tail, dst_rdma_rank);
        if (n_issue == 0) continue;

        if (nvls_tail_mc != nullptr) {
          // Phase 3+4: NVLS counter fast path. For cross-node, payload
          // delivery uses warp-cooperative direct fabric-IPC writes when
          // peer_rdma_bases is available; otherwise falls back to single-
          // lane handle.put + flush.
          if (dst_rdma_rank != rdma_rank) {
            const auto dst_slot_idx = issue_tail % num_max_rdma_chunked_recv_tokens;
            const size_t num_bytes_per_msg = (size_t)num_bytes_per_rdma_token * (size_t)n_issue;
            const auto dst_offset = rdma_rank * (num_max_rdma_chunked_recv_tokens * num_bytes_per_rdma_token) +
                                    dst_slot_idx * num_bytes_per_rdma_token + data_recv_offset;
            const auto src_offset = dst_rdma_rank * (num_max_rdma_chunked_recv_tokens * num_bytes_per_rdma_token) +
                                    dst_slot_idx * num_bytes_per_rdma_token + data_send_offset;
            if (peer_rdma_bases != nullptr) {
              // Phase 4: warp-cooperative int4 stores via NVL72 fabric VA.
              // Uses NVLS counter (nvls_ctr_add) below to publish the
              // writes to the receiver — `multimem.red.RELAXED` + a
              // preceding `__threadfence_system()` is the validated
              // ordering pair (release semantics on multimem.red triggers
              // unspecified launch failure on Azure GB200).
              const int dst_rank_global = dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank;
              const int4* src_p =
                  reinterpret_cast<const int4*>(reinterpret_cast<uint8_t*>(rdma_buffer_ptr_base) + src_offset);
              int4* dst_p =
                  reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(peer_rdma_bases[dst_rank_global]) + dst_offset);
              const int n_int4 = (int)(num_bytes_per_msg / sizeof(int4));
              // Unrolled 8x to give the LSU pipeline more outstanding stores
              // per lane. Each lane handles k, k+32, ..., k+224 per iter
              // (stride 256) before looping. Tail handled with stride-32 loop.
              const int stride8 = 8 * 32;
              int k = lane_id;
              const int n_full = (n_int4 / stride8) * stride8;
              for (; k < n_full; k += stride8) {
                int4 v0 = src_p[k];
                int4 v1 = src_p[k + 32];
                int4 v2 = src_p[k + 64];
                int4 v3 = src_p[k + 96];
                int4 v4 = src_p[k + 128];
                int4 v5 = src_p[k + 160];
                int4 v6 = src_p[k + 192];
                int4 v7 = src_p[k + 224];
                dst_p[k] = v0;
                dst_p[k + 32] = v1;
                dst_p[k + 64] = v2;
                dst_p[k + 96] = v3;
                dst_p[k + 128] = v4;
                dst_p[k + 160] = v5;
                dst_p[k + 192] = v6;
                dst_p[k + 224] = v7;
              }
              for (; k < n_int4; k += 32) {
                dst_p[k] = src_p[k];
              }
              __syncwarp();
              __threadfence_system();
            } else if (owner) {
              const auto port_channel_idx =
                  kLowLatencyMode ? (channel_id * kNumRDMARanks + dst_rdma_rank)
                                  : (channel_id * num_ranks + dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank);
              auto& handle = port_channel_handles[port_channel_idx];
              handle.put(dst_offset, src_offset, num_bytes_per_msg);
              handle.flush();
            }
          }
          // Owner advances tail counter for this peer.
          // Phase 4 fix: cross-node tail goes through direct fabric-VA
          // store on peer's rdma_channel_tail slot. Self-loop tail goes
          // through plain local atomicAdd on rdma_channel_tail.buffer(rdma_rank)
          // — NVLS multicast is WRONG for self-loop because it fans out
          // to all bound NVL peers' buffers (4 NVL ranks × n_issue ⇒ 4x
          // over-count on each consumer's read).
          if (owner) {
            if (peer_rdma_bases != nullptr && dst_rdma_rank != rdma_rank) {
              const int dst_rank_global = dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank;
              const uintptr_t my_tail_off = reinterpret_cast<uintptr_t>(rdma_channel_tail.buffer(rdma_rank)) -
                                            reinterpret_cast<uintptr_t>(rdma_buffer_ptr_base);
              uint64_t* peer_tail = reinterpret_cast<uint64_t*>(
                  reinterpret_cast<uint8_t*>(peer_rdma_bases[dst_rank_global]) + my_tail_off);
              const uint64_t new_tail = (uint64_t)issue_tail + (uint64_t)n_issue;
              asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(peer_tail), "l"(new_tail) : "memory");
            } else {
              // Self-loop: plain release atomic on local slot (no multicast).
              mscclpp::atomicFetchAdd(reinterpret_cast<uint64_t*>(rdma_channel_tail.buffer(rdma_rank)),
                                      (uint64_t)n_issue, mscclpp::memoryOrderRelease);
            }
          }
        } else if (owner) {
          // Legacy non-NVLS path (single-lane).
          if (dst_rdma_rank == rdma_rank) {
            // Update tails
            mscclpp::atomicFetchAdd(reinterpret_cast<uint64_t*>(rdma_channel_tail.buffer(rdma_rank)), (uint64_t)n_issue,
                                    mscclpp::memoryOrderRelease);
          } else {
            const auto dst_slot_idx = issue_tail % num_max_rdma_chunked_recv_tokens;
            const size_t num_bytes_per_msg = (size_t)num_bytes_per_rdma_token * (size_t)n_issue;
            const auto dst_offset = rdma_rank * (num_max_rdma_chunked_recv_tokens * num_bytes_per_rdma_token) +
                                    dst_slot_idx * num_bytes_per_rdma_token + data_recv_offset;
            const auto src_offset = dst_rdma_rank * (num_max_rdma_chunked_recv_tokens * num_bytes_per_rdma_token) +
                                    dst_slot_idx * num_bytes_per_rdma_token + data_send_offset;
            const auto port_channel_idx = kLowLatencyMode
                                              ? (channel_id * kNumRDMARanks + dst_rdma_rank)
                                              : (channel_id * num_ranks + dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank);
            auto& handle = port_channel_handles[port_channel_idx];
            handle.put(dst_offset, src_offset, num_bytes_per_msg);

            // HW atomicAdd is broken on Azure CX-7 RoCE (IBV_ATOMIC_NONE).
            // Write the new absolute tail to a local scratch slot, then RDMA
            // WRITE it to the peer's tail recv slot.
            const uint64_t new_tail = (uint64_t)issue_tail + (uint64_t)n_issue;
            *rdma_channel_tail_send_src.buffer(dst_rdma_rank) = new_tail;
            __threadfence_system();
            const auto src_off_tail = reinterpret_cast<uintptr_t>(rdma_channel_tail_send_src.buffer(dst_rdma_rank)) -
                                      reinterpret_cast<uintptr_t>(rdma_buffer_ptr_base);
            handle.put(rdma_rank * sizeof(uint64_t) + tail_send_offset, src_off_tail, sizeof(uint64_t));
          }
        }
        if (owner) {
          last_issued_tail += n_issue;
          num_tokens_to_send -= n_issue;
        }
      }
    }
    // Phase 4 diag: report final issued tail per (channel, dst_rdma_rank).
    if (lane_id < kNumRDMARanks && rank == 0) {
      (void)0;
    }
  } else if (warp_role == WarpRole::kRDMAAndNVLForwarder) {
    // RDMA consumers and NVL producers
    const auto dst_nvl_rank = target_rank;
    const auto dst_rank = rdma_rank * NUM_MAX_NVL_PEERS + dst_nvl_rank;
    const auto dst_rank_expert_begin = dst_rank * (num_experts / num_ranks);
    const auto dst_rank_expert_end = dst_rank_expert_begin + (num_experts / num_ranks);

    // Wait counters to arrive
    int num_tokens_to_recv_from_rdma = 0, src_rdma_channel_prefix = 0;
    EP_DEVICE_ASSERT(kNumRDMARanks <= 32);
    auto start_time = clock64();
    if (lane_id < kNumRDMARanks) {
      while (true) {
        auto meta_0 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + dst_nvl_rank);
        auto meta_1 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS + dst_nvl_rank);
        auto meta_2 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS * 2);
        auto meta_3 = ld_volatile_global(rdma_channel_meta.recv_buffer(lane_id) + NUM_MAX_NVL_PEERS * 2 + 1);
        if (meta_0 < 0 and meta_1 < 0 and meta_2 < 0 and meta_3 < 0) {
          // Notify NVL ranks
          int start_sum = -meta_0 - 1, end_sum = -meta_1 - 1;
          EP_DEVICE_ASSERT(start_sum >= 0 and end_sum >= 0 and end_sum >= start_sum);
          st_relaxed_sys_global(nvl_channel_prefix_start.buffer() + lane_id, -start_sum - 1);
          st_relaxed_sys_global(nvl_channel_prefix_end.buffer() + lane_id, -end_sum - 1);

          // Save RDMA channel received token count
          src_rdma_channel_prefix = -meta_2 - 1;
          auto src_rdma_channel_prefix_1 = -meta_3 - 1;
          num_tokens_to_recv_from_rdma = src_rdma_channel_prefix_1 - src_rdma_channel_prefix;
          if (not kCachedMode)
            recv_rdma_channel_prefix_matrix[lane_id * num_channels + channel_id] = src_rdma_channel_prefix_1;
          src_rdma_channel_prefix += lane_id == 0 ? 0 : recv_rdma_rank_prefix_sum[lane_id - 1];
          EP_DEVICE_ASSERT(num_tokens_to_recv_from_rdma >= 0);
          // Phase 4 diag: report received expected token count per (channel, src).
          if (rank == 4) {
            (void)0;
          }
          break;
        }

        // Timeout check
        if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
          printf(
              "DeepEP dispatch forwarder timeout (RDMA meta), channel: %d, RDMA: %d, nvl: %d, src RDMA lane: %d, dst "
              "NVL: %d, meta: %d, %d, %d, %d\n",
              channel_id, rdma_rank, nvl_rank, lane_id, dst_nvl_rank, meta_0, meta_1, meta_2, meta_3);
          trap();
        }
      }
    }
    __syncwarp();

    // Shift cached head
    send_nvl_head += src_rdma_channel_prefix * NUM_MAX_NVL_PEERS + dst_nvl_rank;

    // Wait shared memory to be cleaned
    sync_forwarder_smem();

    // Forward tokens from RDMA buffer
    // NOTES: always start from the local rank
    int src_rdma_rank = sm_id % kNumRDMARanks;
    int cached_rdma_channel_head = 0, cached_rdma_channel_tail = 0;
    int cached_nvl_channel_head = 0, cached_nvl_channel_tail = 0, rdma_nvl_token_idx = 0;
    while (__any_sync(0xffffffff, num_tokens_to_recv_from_rdma > 0)) {
      // Check destination queue emptiness, or wait a buffer to be released
      start_time = clock64();
      while (lane_id == 0) {
        int num_used_slots = cached_nvl_channel_tail - cached_nvl_channel_head;
        if (num_max_nvl_chunked_recv_tokens - num_used_slots >= num_max_nvl_chunked_send_tokens) break;
        cached_nvl_channel_head = ld_volatile_global(nvl_channel_head.buffer());

        // Timeout check
        if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
          printf(
              "DeepEP dispatch forwarder timeout (NVL check), channel: %d, RDMA: %d, nvl: %d, dst NVL: %d, head: %d, "
              "tail: %d\n",
              channel_id, rdma_rank, nvl_rank, dst_nvl_rank, ld_volatile_global(nvl_channel_head.buffer()),
              cached_nvl_channel_tail);
          trap();
        }
      }
      __syncwarp();

      // Find next source RDMA rank (round-robin)
      start_time = clock64();
      while (true) {
        src_rdma_rank = (src_rdma_rank + 1) % kNumRDMARanks;
        if (__shfl_sync(0xffffffff, num_tokens_to_recv_from_rdma, src_rdma_rank) > 0) {
          if (lane_id == src_rdma_rank and cached_rdma_channel_head == cached_rdma_channel_tail) {
            // Phase 4 fix: cross-node tail is now delivered via direct
            // fabric-VA store to local rdma_channel_tail.buffer(src_rdma_rank),
            // so prefer the legacy ld_acquire read which sees those stores.
            // NVLS counter only used for self path (single rank).
            // Phase 4 fix: cross-node tail comes via direct fabric-VA store
            // (sender writes peer's rdma_channel_tail slot). Self-loop tail
            // is a plain local atomic. Both end up in rdma_channel_tail —
            // single read path via ld_acquire_sys_global covers them.
            cached_rdma_channel_tail = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(src_rdma_rank)));
          }
          if (__shfl_sync(0xffffffff, cached_rdma_channel_tail > cached_rdma_channel_head, src_rdma_rank)) break;
        }

        // Timeout check
        if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks) {
          printf(
              "DeepEP dispatch forwarder timeout (RDMA check), channel: %d, RDMA: %d, nvl: %d, dst NVL: %d, src RDMA "
              "lane: %d, head: %d, tail: %d, expected: %d\n",
              channel_id, rdma_rank, nvl_rank, dst_nvl_rank, lane_id, cached_rdma_channel_head,
              cached_rdma_channel_tail, num_tokens_to_recv_from_rdma);
          trap();
        }
      }
      auto src_rdma_head = __shfl_sync(0xffffffff, cached_rdma_channel_head, src_rdma_rank);
      auto src_rdma_tail = __shfl_sync(0xffffffff, cached_rdma_channel_tail, src_rdma_rank);

      if (rank == 4 && lane_id == 0 && dst_nvl_rank == 0 && channel_id == 0) {
        (void)0;
      }

      // Iterate over every token from the RDMA buffer
      for (int i = src_rdma_head, num_tokens_sent = 0; i < src_rdma_tail; ++i) {
        auto rdma_slot_idx = i % num_max_rdma_chunked_recv_tokens;
        void* shifted = rdma_channel_data.recv_buffer(src_rdma_rank) + rdma_slot_idx * num_bytes_per_rdma_token;
        auto src_meta = ld_nc_global(reinterpret_cast<SourceMeta*>(reinterpret_cast<int8_t*>(shifted) + hidden_bytes));
        lane_id == src_rdma_rank ? (num_tokens_to_recv_from_rdma -= 1) : 0;
        bool is_in_dst_nvl_rank = src_meta.is_token_in_nvl_rank(dst_nvl_rank);
        if (lane_id == src_rdma_rank) {
          auto cached_head = is_in_dst_nvl_rank ? rdma_nvl_token_idx : -1;
          rdma_nvl_token_idx += is_in_dst_nvl_rank;
          if (not kCachedMode) send_nvl_head[i * NUM_MAX_NVL_PEERS] = cached_head;
        }
        if (not is_in_dst_nvl_rank) continue;

        // Get an empty slot
        int dst_slot_idx = (cached_nvl_channel_tail++) % num_max_nvl_chunked_recv_tokens;

        // Copy data
        UNROLLED_WARP_COPY(5, lane_id, hidden_int4, nvl_channel_x.buffer() + dst_slot_idx * hidden_int4,
                           reinterpret_cast<int4*>(shifted), ld_nc_global, st_na_global);
        shifted = reinterpret_cast<int4*>(shifted) + hidden_int4;

        // Copy source meta
        if (lane_id == 0) st_na_global(nvl_channel_src_meta.buffer() + dst_slot_idx, src_meta);
        shifted = reinterpret_cast<SourceMeta*>(shifted) + 1;

        // Copy `x_scales`
        UNROLLED_WARP_COPY(1, lane_id, num_scales, nvl_channel_x_scales.buffer() + dst_slot_idx * num_scales,
                           reinterpret_cast<float*>(shifted), ld_nc_global, st_na_global);
        shifted = reinterpret_cast<float*>(shifted) + num_scales;

        // Copy `topk_idx` and `topk_weights`
        // NOTES: do not use `shifted` after this `if`, because only several lanes are shifted
        if (lane_id < num_topk) {
          // Read
          auto idx_value = ld_nc_global(reinterpret_cast<int*>(shifted) + lane_id);
          shifted = reinterpret_cast<int*>(shifted) + num_topk;
          auto weight_value = ld_nc_global(reinterpret_cast<float*>(shifted) + lane_id);

          // Transform and write
          idx_value = (idx_value >= dst_rank_expert_begin and idx_value < dst_rank_expert_end)
                          ? idx_value - dst_rank_expert_begin
                          : -1;
          st_na_global(nvl_channel_topk_idx.buffer() + dst_slot_idx * num_topk + lane_id, idx_value);
          weight_value = idx_value >= 0 ? weight_value : 0.0f;
          st_na_global(nvl_channel_topk_weights.buffer() + dst_slot_idx * num_topk + lane_id, weight_value);
        }

        // In case of insufficient NVL buffers, early stopping
        if ((++num_tokens_sent) == num_max_nvl_chunked_send_tokens) src_rdma_tail = i + 1;
      }

      // Sync head index
      if (lane_id == src_rdma_rank)
        forward_channel_head[dst_nvl_rank][src_rdma_rank] = (cached_rdma_channel_head = src_rdma_tail);

      // Move tail index
      __syncwarp();
      if (lane_id == 0) st_release_sys_global(nvl_channel_tail.buffer(), cached_nvl_channel_tail);
    }

    // Retired
    __syncwarp();
    if (lane_id == 0) {
      forward_channel_retired[dst_nvl_rank] = true;
      if (channel_id == 0) {
        (void)0;
      }
    }
  } else if (warp_role == WarpRole::kForwarderCoordinator) {
    // Extra warps for forwarder coordinator should exit directly
    if (target_rank > 0) return;

    // Forward warp coordinator
    EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");

    // Clean shared memory
    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= 32, "Invalid number of NVL peers");
#pragma unroll
    for (int i = lane_id; i < kNumRDMARanks * NUM_MAX_NVL_PEERS; i += 32)
      forward_channel_head[i % NUM_MAX_NVL_PEERS][i / NUM_MAX_NVL_PEERS] = 0;
    if (lane_id < NUM_MAX_NVL_PEERS) forward_channel_retired[lane_id] = false;
    sync_forwarder_smem();

    int last_head = 0, target_rdma = lane_id < kNumRDMARanks ? lane_id : 0;
    while (true) {
      // Find minimum head
      int min_head = std::numeric_limits<int>::max();
#pragma unroll
      for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i)
        if (not forward_channel_retired[i]) min_head = min(min_head, forward_channel_head[i][target_rdma]);
      if (__all_sync(0xffffffff, min_head == std::numeric_limits<int>::max())) break;

      // Update remote head
      // Phase 4 perf: lower the lazy-update threshold from chunk_send → chunk_send/4
      // (or 1 if retired) so the sender's receive-buffer-space window
      // advances more frequently. The original threshold caused partial
      // last chunks to never trigger head feedback, which deadlocked at
      // larger chunk_send values where 4096 tokens / 10 channels / 2 peers
      // ≈ 205 tokens ⇒ 3 full chunks + 1 partial.
      const bool any_retired = forward_channel_retired[0] || forward_channel_retired[1] || forward_channel_retired[2] ||
                               forward_channel_retired[3] || forward_channel_retired[4] || forward_channel_retired[5] ||
                               forward_channel_retired[6] || forward_channel_retired[7];
      const int head_update_threshold = any_retired ? 1 : max(1, num_max_rdma_chunked_send_tokens / 4);
      if (min_head != std::numeric_limits<int>::max() and min_head >= last_head + head_update_threshold and
          lane_id < kNumRDMARanks) {
        if (peer_rdma_bases != nullptr && lane_id != rdma_rank) {
          // Phase 4 fix: cross-node head feedback via direct fabric-VA
          // store on peer's rdma_channel_head slot (single writer per
          // (channel, my_rdma_rank) on peer's side: producer is me, slot
          // is `peer.rdma_channel_head.buffer(my_rdma_rank)`). Bypasses
          // both broken port_channel.put and the unreliable NVLS counter.
          const int dst_rank_global = lane_id * NUM_MAX_NVL_PEERS + nvl_rank;
          const uintptr_t my_head_off = reinterpret_cast<uintptr_t>(rdma_channel_head.buffer(rdma_rank)) -
                                        reinterpret_cast<uintptr_t>(rdma_buffer_ptr_base);
          uint64_t* peer_head =
              reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(peer_rdma_bases[dst_rank_global]) + my_head_off);
          asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(peer_head), "l"((uint64_t)min_head) : "memory");
        } else if (lane_id == rdma_rank) {
          // Self-loop: plain release atomic on local slot. Cannot use NVLS
          // multimem here \u2014 it fans out to all NVL peers' local buffers
          // and over-counts (4 NVL ranks \u00d7 add \u21d2 4x increment).
          mscclpp::atomicFetchAdd(static_cast<uint64_t*>(rdma_channel_head.buffer(rdma_rank)),
                                  (uint64_t)(min_head - last_head), mscclpp::memoryOrderRelease);
        } else {
          auto dst_offset = rdma_rank * sizeof(uint64_t) + head_send_offset;
          auto port_channel_idx = kLowLatencyMode ? (channel_id * kNumRDMARanks + lane_id)
                                                  : (channel_id * num_ranks + lane_id * NUM_MAX_NVL_PEERS + nvl_rank);
          auto& handle = port_channel_handles[port_channel_idx];
          // Absolute-value RDMA WRITE replaces broken HW atomicAdd (see note above).
          *rdma_channel_head_send_src.buffer(lane_id) = (uint64_t)min_head;
          __threadfence_system();
          const auto src_off_head = reinterpret_cast<uintptr_t>(rdma_channel_head_send_src.buffer(lane_id)) -
                                    reinterpret_cast<uintptr_t>(rdma_buffer_ptr_base);
          handle.put(dst_offset, src_off_head, sizeof(uint64_t));
        }
        last_head = min_head;
      }

      // Nanosleep and let other warps work
      __nanosleep(NUM_WAIT_NANOSECONDS);
    }
  } else {
    // NVL consumers
    // Retrieve rank offset from barrier results (each lane's register stores an RDMA rank)
    int src_nvl_rank = target_rank, total_offset = 0;
    EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
    if (lane_id < kNumRDMARanks and lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank > 0)
      total_offset = recv_gbl_rank_prefix_sum[lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank - 1];

    // Receive channel offsets
    int start_offset = 0, end_offset = 0, num_tokens_to_recv;
    auto start_time = clock64();
    while (lane_id < kNumRDMARanks) {
      start_offset = ld_volatile_global(nvl_channel_prefix_start.buffer() + lane_id);
      end_offset = ld_volatile_global(nvl_channel_prefix_end.buffer() + lane_id);
      if (start_offset < 0 and end_offset < 0) {
        start_offset = -start_offset - 1, end_offset = -end_offset - 1;
        total_offset += start_offset;
        break;
      }

      // Timeout check
      if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
        printf(
            "DeepEP dispatch NVL receiver timeout, channel: %d, RDMA: %d, nvl: %d, src RDMA: %d, src nvl: %d, start: "
            "%d, end: %d\n",
            channel_id, rdma_rank, nvl_rank, lane_id, src_nvl_rank, start_offset, end_offset);
        trap();
      }
    }
    num_tokens_to_recv = warp_reduce_sum(end_offset - start_offset);

    // Save for combine usage
    if (lane_id < kNumRDMARanks and not kCachedMode)
      recv_gbl_channel_prefix_matrix[(lane_id * NUM_MAX_NVL_PEERS + src_nvl_rank) * num_channels + channel_id] =
          total_offset;
    __syncwarp();

    int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
    while (num_tokens_to_recv > 0) {
      // Check channel status by lane 0
      start_time = clock64();
      while (lane_id == 0) {
        // Ready to copy
        if (cached_channel_head_idx != cached_channel_tail_idx) break;
        cached_channel_tail_idx = ld_acquire_sys_global(nvl_channel_tail.buffer());

        // Timeout check
        if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
          printf(
              "DeepEP dispatch NVL receiver timeout, channel: %d, RDMA: %d, nvl: %d, src NVL: %d, head: %d, tail: %d\n",
              channel_id, rdma_rank, nvl_rank, src_nvl_rank, cached_channel_head_idx, cached_channel_tail_idx);
          trap();
        }
      }

      // Sync queue tail
      cached_channel_tail_idx = __shfl_sync(0xffffffff, cached_channel_tail_idx, 0);

      // Copy data
      int num_recv_tokens = cached_channel_tail_idx - cached_channel_head_idx;
      for (int chunk_idx = 0; chunk_idx < num_recv_tokens; ++chunk_idx, --num_tokens_to_recv) {
        int token_idx_in_buffer = (cached_channel_head_idx++) % num_max_nvl_chunked_recv_tokens;
        auto meta = ld_nc_global(nvl_channel_src_meta.buffer() + token_idx_in_buffer);
        int64_t recv_token_idx = __shfl_sync(0xffffffff, total_offset, meta.src_rdma_rank);
        (lane_id == meta.src_rdma_rank) ? (total_offset += 1) : 0;

        // Copy data
        UNROLLED_WARP_COPY(5, lane_id, hidden_int4, recv_x + recv_token_idx * hidden_int4,
                           nvl_channel_x.buffer() + token_idx_in_buffer * hidden_int4, ld_nc_global, st_na_global);

        // Copy source meta
        if (lane_id == 0 and not kCachedMode) st_na_global(recv_src_meta + recv_token_idx, meta);

        // Copy scales
        UNROLLED_WARP_COPY(1, lane_id, num_scales, recv_x_scales + recv_token_idx * num_scales,
                           nvl_channel_x_scales.buffer() + token_idx_in_buffer * num_scales, ld_nc_global,
                           st_na_global);

        // Copy `topk_idx` and `topk_weights`
        if (lane_id < num_topk) {
          auto recv_idx = recv_token_idx * num_topk + lane_id;
          auto buffer_idx = token_idx_in_buffer * num_topk + lane_id;
          st_na_global(recv_topk_idx + recv_idx,
                       static_cast<int64_t>(ld_nc_global(nvl_channel_topk_idx.buffer() + buffer_idx)));
          st_na_global(recv_topk_weights + recv_idx, ld_nc_global(nvl_channel_topk_weights.buffer() + buffer_idx));
        }
      }

      // Move queue
      __syncwarp();
      if (lane_id == 0) st_relaxed_sys_global(nvl_channel_head.buffer(), cached_channel_head_idx);
    }
  }
  if (thread_id == 0 && channel_id == 0) {
    (void)0;
  }
}


#endif  // EP_DISPATCH_NCCLEP
#endif  // MSCCLPP_EP_INTERNODE_NCCLEP_CUH_
