// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// HT dispatch and combine kernels for directly mapped peers.

#include <limits>

#include "buffer.cuh"
#include "constants.cuh"
#include "device_helpers.cuh"
#include "exception.cuh"
#include "launch.cuh"

namespace mscclpp {
namespace ep {

namespace intranode {

template <int kNumRanks>
__global__ void notify_dispatch(const int* num_tokens_per_rank, int* moe_recv_counter_mapped,
                                const int* num_tokens_per_expert, int* moe_recv_expert_counter_mapped, int num_experts,
                                int num_tokens, int num_channels, const bool* is_token_in_rank,
                                int* channel_prefix_matrix, int* rank_prefix_matrix_copy, int num_memset_int,
                                int expert_alignment, void** buffer_ptrs, int** task_fifo_ptrs, int head, int rank) {
  auto sm_id = static_cast<int>(blockIdx.x);
  auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);
  auto lane_id = thread_id % 32, warp_id = thread_id / 32, num_warps = num_threads / 32;

  if (sm_id == 0) {
    // Barrier first
    barrier_device<kNumRanks>(task_fifo_ptrs, head, rank);
    move_fifo_slots<kNumRanks>(head);
    __syncthreads();

    int *per_rank_buffer, *per_expert_buffer;
    if (thread_id < kNumRanks) {
      per_rank_buffer = reinterpret_cast<int*>(buffer_ptrs[thread_id]);
      per_expert_buffer = per_rank_buffer + kNumRanks * kNumRanks;
    }

    // After this loop:
    //  - `per_rank_buffer[rank][i, j]` means the number of tokens from rank i to rank j
    //  - `per_expert_buffer[rank][i, j]` means the number of tokens from rank i to local expert j
    int num_experts_per_rank = num_experts / kNumRanks;
    if (thread_id < kNumRanks) {
#pragma unroll
      for (int i = 0; i < kNumRanks; ++i) per_rank_buffer[rank * kNumRanks + i] = num_tokens_per_rank[i];
#pragma unroll
      for (int i = 0; i < num_experts_per_rank; ++i)
        per_expert_buffer[rank * num_experts_per_rank + i] =
            num_tokens_per_expert[thread_id * num_experts_per_rank + i];
    }
    __syncthreads();

    // Wait for all ranks to be finished
    barrier_device<kNumRanks>(task_fifo_ptrs, head, rank);
    move_fifo_slots<kNumRanks>(head);
    __syncthreads();

    // Sum per-rank counts and return to CPU
    // Also pre-compute the prefix sum for data sending
    auto local_per_rank_buffer = reinterpret_cast<int*>(buffer_ptrs[rank]);
    if (thread_id < kNumRanks) {
#pragma unroll
      for (int i = 1; i < kNumRanks; ++i)
        local_per_rank_buffer[i * kNumRanks + thread_id] += local_per_rank_buffer[(i - 1) * kNumRanks + thread_id];
      if (thread_id == rank) *moe_recv_counter_mapped = local_per_rank_buffer[(kNumRanks - 1) * kNumRanks + rank];
    }

    // Sum per-experts counts and return to CPU
    auto local_per_expert_buffer = local_per_rank_buffer + kNumRanks * kNumRanks;
    if (thread_id < num_experts_per_rank) {
      int sum = 0;
#pragma unroll
      for (int i = 0; i < kNumRanks; ++i) sum += local_per_expert_buffer[i * num_experts_per_rank + thread_id];
      sum = (sum + expert_alignment - 1) / expert_alignment * expert_alignment;
      moe_recv_expert_counter_mapped[thread_id] = sum;
    }
    __syncthreads();

// Copy rank size prefix matrix to another tensor
#pragma unroll
    for (int i = thread_id; i < kNumRanks * kNumRanks; i += num_threads)
      rank_prefix_matrix_copy[i] = local_per_rank_buffer[i];

// Extra memset for later communication queue
#pragma unroll
    for (int i = thread_id; i < num_memset_int; i += num_threads) local_per_expert_buffer[i] = 0;

    // Barrier
    memory_fence();
    __syncthreads();
    barrier_device<kNumRanks>(task_fifo_ptrs, head, rank);
  } else {
    int dst_rank = sm_id - 1;
    for (int channel_id = warp_id; channel_id < num_channels; channel_id += num_warps) {
      int token_start_idx, token_end_idx;
      get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

      // Iterate over tokens
      int count = 0;
      for (int64_t i = token_start_idx + lane_id; i < token_end_idx; i += 32)
        count += is_token_in_rank[i * kNumRanks + dst_rank];
      count = warp_reduce_sum(count);
      if (lane_id == 0) channel_prefix_matrix[dst_rank * num_channels + channel_id] = count;
    }
    __syncthreads();

    // Pre-compute prefix sum for all channels
    if (thread_id == 0) {
#pragma unroll
      for (int i = 1; i < num_channels; ++i)
        channel_prefix_matrix[dst_rank * num_channels + i] += channel_prefix_matrix[dst_rank * num_channels + i - 1];
    }
  }
}

void notify_dispatch(const int* num_tokens_per_rank, int* moe_recv_counter_mapped, int num_ranks,
                     const int* num_tokens_per_expert, int* moe_recv_expert_counter_mapped, int num_experts,
                     int num_tokens, const bool* is_token_in_rank, int* channel_prefix_matrix,
                     int* rank_prefix_matrix_copy, int num_memset_int, int expert_alignment, void** buffer_ptrs,
                     int** task_fifo_ptrs, int head, int rank, cudaStream_t stream, int num_channels) {
#define NOTIFY_DISPATCH_LAUNCH_CASE(ranks)                                                                         \
  LAUNCH_KERNEL(&cfg, notify_dispatch<ranks>, num_tokens_per_rank, moe_recv_counter_mapped, num_tokens_per_expert, \
                moe_recv_expert_counter_mapped, num_experts, num_tokens, num_channels, is_token_in_rank,           \
                channel_prefix_matrix, rank_prefix_matrix_copy, num_memset_int, expert_alignment, buffer_ptrs,     \
                task_fifo_ptrs, head, rank);                                                                       \
  break

  constexpr int kNumThreads = 128;
  EP_HOST_ASSERT(num_experts % num_ranks == 0);
  EP_HOST_ASSERT(num_experts / num_ranks <= kNumThreads and num_ranks <= kNumThreads);

  SETUP_LAUNCH_CONFIG(1 + num_ranks, kNumThreads, stream);
  SWITCH_RANKS(NOTIFY_DISPATCH_LAUNCH_CASE);
#undef NOTIFY_DISPATCH_LAUNCH_CASE
}

template <int kNumRanks>
__global__ void cached_notify_dispatch(const int* rank_prefix_matrix, int num_memset_int, void** buffer_ptrs,
                                       int** task_fifo_ptrs, int head, int rank) {
  // A simplified version for cached handles
  barrier_device<kNumRanks>(task_fifo_ptrs, head, rank);
  move_fifo_slots<kNumRanks>(head);
  __syncthreads();

  // Copy and clean
  auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);
  auto ptr = reinterpret_cast<int*>(buffer_ptrs[rank]);
#pragma unroll
  for (int i = thread_id; i < kNumRanks * kNumRanks; i += num_threads) ptr[i] = rank_prefix_matrix[i];
#pragma unroll
  for (int i = thread_id; i < num_memset_int; i += num_threads) ptr[kNumRanks * kNumRanks + i] = 0;
  memory_fence();
  __syncthreads();

  // Barrier after cleaning
  barrier_device<kNumRanks>(task_fifo_ptrs, head, rank);
}

void cached_notify_dispatch(const int* rank_prefix_matrix, int num_memset_int, void** buffer_ptrs, int** task_fifo_ptrs,
                            int head, int rank, int num_ranks, cudaStream_t stream) {
#define CACHED_NOTIFY_DISPATCH_LAUNCH_CASE(ranks)                                                                     \
  LAUNCH_KERNEL(&cfg, cached_notify_dispatch<ranks>, rank_prefix_matrix, num_memset_int, buffer_ptrs, task_fifo_ptrs, \
                head, rank);                                                                                          \
  break

  SETUP_LAUNCH_CONFIG(1, 128, stream);
  SWITCH_RANKS(CACHED_NOTIFY_DISPATCH_LAUNCH_CASE);
#undef CACHED_NOTIFY_DISPATCH_LAUNCH_CASE
}

template <int kNumRanks, int kNumThreads>
__global__ void __launch_bounds__(kNumThreads, 1)
    dispatch(int4* recv_x, float* recv_x_scales, int* recv_src_idx, int64_t* recv_topk_idx, float* recv_topk_weights,
             int* recv_channel_offset, int* send_head, const int4* x, const float* x_scales, const int64_t* topk_idx,
             const float* topk_weights, const bool* is_token_in_rank, const int* channel_prefix_matrix, int num_tokens,
             int hidden_int4, int num_topk, int num_experts, int num_scales, void** buffer_ptrs, int rank,
             int num_max_send_tokens, int num_recv_buffer_tokens, void** recv_pool_ptrs, int64_t recv_pool_header_bytes,
             int* ep_combine_recv_idx) {
  const auto num_sms = static_cast<int>(gridDim.x), sm_id = static_cast<int>(blockIdx.x);
  const auto thread_id = static_cast<int>(threadIdx.x);
  const bool is_sender = sm_id % 2 == 0;
  EP_DEVICE_ASSERT(num_sms % 2 == 0);

  // Several warps are response for a single rank
  const auto num_threads_per_rank = kNumThreads / kNumRanks;
  const auto num_channels = num_sms / 2;
  const auto responsible_rank = (static_cast<int>(thread_id)) / num_threads_per_rank;
  // Even-numbered blocks for sending, odd-numbered blocks for receiving.
  const auto responsible_channel = sm_id / 2;

  int num_experts_per_rank = num_experts / kNumRanks;
  EP_DEVICE_ASSERT(num_experts_per_rank > 0 or num_topk == 0);
  EP_DEVICE_ASSERT(num_topk <= 32);
  EP_DEVICE_ASSERT((topk_idx == nullptr) == (topk_weights == nullptr));
  EP_DEVICE_ASSERT((recv_topk_idx == nullptr) == (recv_topk_weights == nullptr));

  // Calculate pointers by the specific layout
  // `rank_prefix_matrix`: kNumRanks * kNumRanks * sizeof(int)
  auto ptr = reinterpret_cast<void*>(reinterpret_cast<int8_t*>(buffer_ptrs[is_sender ? responsible_rank : rank]) +
                                     kNumRanks * kNumRanks * sizeof(int));
  int target_rank = is_sender ? rank : responsible_rank;
  auto num_channels_total = num_channels * kNumRanks;
  auto channel_rank_offset = responsible_channel * kNumRanks + target_rank;

  // Channel buffer metadata
  // Senders are responsible for tails, and receivers are responsible for heads
  // Stored on the receiver side
  // The retired signals are actually boolean flags, but to align with 16 bytes, we make it `int64_t`
  // `start_offset`: kNumChannels * kNumRanks * sizeof(int)
  // `end_offset`: kNumChannels * kNumRanks * sizeof(int)
  // `head_idx`: kNumChannels * kNumRanks * sizeof(int)
  // `tail_idx`: kNumChannels * kNumRanks * sizeof(int)
  auto channel_start_offset = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
  auto channel_end_offset = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
  auto channel_head_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
  auto channel_tail_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);

  // Channel data buffers, stored on the receiver side
  // `x_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * hidden_int4 * sizeof(int4)
  // `src_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * sizeof(int)
  // `topk_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(int64_t)
  // `topk_weights_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(float)
  // `x_scales_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_scales * sizeof(float)
  auto channel_x_buffers = Buffer<int4>(ptr, num_channels_total * num_recv_buffer_tokens * hidden_int4,
                                        channel_rank_offset * num_recv_buffer_tokens * hidden_int4);
  auto channel_src_idx_buffers =
      Buffer<int>(ptr, num_channels_total * num_recv_buffer_tokens, channel_rank_offset * num_recv_buffer_tokens);
  auto channel_topk_idx_buffers = Buffer<int64_t>(ptr, num_channels_total * num_recv_buffer_tokens * num_topk,
                                                  channel_rank_offset * num_recv_buffer_tokens * num_topk);
  auto channel_topk_weights_buffers = Buffer<float>(ptr, num_channels_total * num_recv_buffer_tokens * num_topk,
                                                    channel_rank_offset * num_recv_buffer_tokens * num_topk);
  auto channel_x_scales_buffers = Buffer<float>(ptr, num_channels_total * num_recv_buffer_tokens * num_scales,
                                                channel_rank_offset * num_recv_buffer_tokens * num_scales);

  if (is_sender) {
    // Workers for sending
    constexpr int num_send_warps = kNumThreads / 32;
    constexpr int num_send_warps_per_rank = num_send_warps / kNumRanks;
    const auto send_thread_id = thread_id;
    const auto send_lane_id = send_thread_id % 32;
    const auto send_warp_id_in_rank = send_thread_id % num_threads_per_rank / 32;
    EP_DEVICE_ASSERT(kNumRanks <= 32);
    EP_DEVICE_ASSERT(num_send_warps % kNumRanks == 0);

    // Send offset by `-value - 1`, e.g. 0 -> -1, 1 -> -2
    // NOTES: this is for distinguishing zero tokens
    if (send_lane_id == 0 and send_warp_id_in_rank == 0) {
      int value = responsible_channel > 0
                      ? channel_prefix_matrix[responsible_rank * num_channels + responsible_channel - 1]
                      : 0;
      st_relaxed_sys_global(channel_start_offset.buffer(), -value - 1);
      value = channel_prefix_matrix[responsible_rank * num_channels + responsible_channel];
      st_relaxed_sys_global(channel_end_offset.buffer(), -value - 1);
    }
    __syncwarp();

    // Sender-direct (MSCCLPP_EP_INTRA_DIRECT): precompute this (dst, channel)'s base
    // index into the destination's peer-mapped recv-output pool, so the sender can
    // write hidden straight to recv_x's final slot (no ring slot, no receiver drain).
    // Mirrors the receiver's `total_offset` = rank_prefix_matrix[(src-1)*R+dst] +
    // channel_prefix_matrix[dst*num_channels + ch-1].
    int64_t direct_base = 0;
    int4* direct_dst_pool = nullptr;
    if (recv_pool_ptrs != nullptr) {
      const int* dst_rank_prefix = reinterpret_cast<const int*>(buffer_ptrs[responsible_rank]);
      int rank_off = rank > 0 ? dst_rank_prefix[(rank - 1) * kNumRanks + responsible_rank] : 0;
      int ch_start = responsible_channel > 0
                         ? channel_prefix_matrix[responsible_rank * num_channels + responsible_channel - 1]
                         : 0;
      direct_base = static_cast<int64_t>(rank_off + ch_start);
      direct_dst_pool = reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(recv_pool_ptrs[responsible_rank]) +
                                                recv_pool_header_bytes);
    }

    // Get tasks
    int token_start_idx, token_end_idx;
    get_channel_task_range(num_tokens, num_channels, responsible_channel, token_start_idx, token_end_idx);

    // Iterate over all tokens and send by chunks
    int cached_channel_tail_idx = 0;
    for (int64_t token_idx = token_start_idx; token_idx < token_end_idx;) {
      // Check destination queue emptiness, or wait a buffer to be released (rare cases)
      // NOTES: the head index received by different warps may not be the same
      auto start_time = clock64();
      while (send_lane_id == 0) {
        // NOTES: we only consider the worst case, because counting the real numbers are time-consuming
        int num_used_slots = cached_channel_tail_idx - ld_volatile_global(channel_head_idx.buffer());
        if (num_recv_buffer_tokens - num_used_slots >= num_max_send_tokens) break;

        // Rare cases to loop again
        if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
          printf("DeepEP timeout for dispatch senders, rank %d, responsible_channel = %d\n", rank, responsible_channel);
          trap();
        }
      }
      __syncwarp();

      int chunk_token_idx = 0;
      while (chunk_token_idx < num_max_send_tokens and token_idx < token_end_idx) {
        // NOTES: for the same token, the warp assigned to save `send_head` may be different from the warp assigned to
        // send subsequent data
        if (send_lane_id == 0 and token_idx % num_send_warps_per_rank == send_warp_id_in_rank)
          send_head[token_idx * kNumRanks + responsible_rank] =
              is_token_in_rank[token_idx * kNumRanks + responsible_rank] ? cached_channel_tail_idx : -1;

        // Skip if not selected
        if (not is_token_in_rank[token_idx * kNumRanks + responsible_rank]) {
          token_idx++;
          continue;
        }

        // Get an empty slot
        const int abs_pos = cached_channel_tail_idx;  // running position within (src, dst, channel) = recv_x slot
        int dst_slot_idx = (cached_channel_tail_idx++) % num_recv_buffer_tokens;
        if (cached_channel_tail_idx % num_send_warps_per_rank == send_warp_id_in_rank) {
          // Copy data. Sender-direct writes hidden straight to recv_x's final slot in the
          // destination's pool; otherwise the 2-hop ring slot (receiver drains it later).
          auto shifted_x = x + token_idx * hidden_int4;
          if (recv_pool_ptrs != nullptr) {
            auto shifted_dst = direct_dst_pool + (direct_base + abs_pos) * hidden_int4;
            UNROLLED_WARP_COPY(5, send_lane_id, hidden_int4, shifted_dst, shifted_x, __ldg, st_na_global);
            // inc5 combine-direct: record this token's final recv-pool slot in the dst's
            // pool so the direct-gather combine can read it straight back (per (token, dst)).
            if (ep_combine_recv_idx != nullptr and send_lane_id == 0)
              ep_combine_recv_idx[token_idx * kNumRanks + responsible_rank] = static_cast<int>(direct_base + abs_pos);
          } else {
            auto shifted_channel_x_buffers = channel_x_buffers.buffer() + dst_slot_idx * hidden_int4;
            UNROLLED_WARP_COPY(5, send_lane_id, hidden_int4, shifted_channel_x_buffers, shifted_x, __ldg, st_na_global);
          }

          // Copy source index
          if (send_lane_id == 0) channel_src_idx_buffers[dst_slot_idx] = static_cast<int>(token_idx);

          // Copy `topk_idx` and `topk_weights` with transformed index
          if (send_lane_id < num_topk) {
            // Top-k index
            int recv_expert_begin = responsible_rank * num_experts_per_rank,
                recv_expert_end = (responsible_rank + 1) * num_experts_per_rank;
            auto idx_value = __ldg(topk_idx + token_idx * num_topk + send_lane_id);
            idx_value =
                (idx_value >= recv_expert_begin and idx_value < recv_expert_end) ? idx_value - recv_expert_begin : -1;
            channel_topk_idx_buffers[dst_slot_idx * num_topk + send_lane_id] = idx_value;

            // Top-k weights
            auto weight_value = __ldg(topk_weights + token_idx * num_topk + send_lane_id);
            weight_value = (idx_value >= 0) ? weight_value : 0.0f;
            channel_topk_weights_buffers[dst_slot_idx * num_topk + send_lane_id] = weight_value;
          }

// Copy `x_scales`
#pragma unroll
          for (int i = send_lane_id; i < num_scales; i += 32)
            channel_x_scales_buffers[dst_slot_idx * num_scales + i] = __ldg(x_scales + token_idx * num_scales + i);
        }

        // Move token index
        chunk_token_idx++, token_idx++;
      }

      // Move tail index
      // NOTES: here all warps should share the same new tail
      asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
      if (send_warp_id_in_rank == 0 and send_lane_id == 0)
        st_release_sys_global(channel_tail_idx.buffer(), cached_channel_tail_idx);
    }
  } else {
    // Workers for receiving and copying into buffer
    constexpr int num_recv_warps = kNumThreads / 32;
    constexpr int num_recv_warps_per_rank = num_recv_warps / kNumRanks;
    const auto recv_thread_id = thread_id;
    const auto recv_lane_id = recv_thread_id % 32;
    const auto recv_thread_id_in_rank = recv_thread_id % num_threads_per_rank;
    const auto recv_warp_id_in_rank = recv_thread_id_in_rank / 32;
    EP_DEVICE_ASSERT(kNumRanks <= 32);
    EP_DEVICE_ASSERT(recv_thread_id >= 0 and num_recv_warps % kNumRanks == 0);

    // Calculate offset first
    auto rank_prefix_matrix = reinterpret_cast<int*>(buffer_ptrs[rank]);
    int rank_offset = responsible_rank > 0 ? rank_prefix_matrix[(responsible_rank - 1) * kNumRanks + rank] : 0;

    // Receive channel offset
    int total_offset, num_tokens_to_recv;
    while (recv_lane_id == 0 and (total_offset = ld_volatile_global(channel_start_offset.buffer())) == 0);
    while (recv_lane_id == 0 and (num_tokens_to_recv = ld_volatile_global(channel_end_offset.buffer())) == 0);
    if (recv_lane_id == 0) {
      total_offset = -total_offset - 1, num_tokens_to_recv = -num_tokens_to_recv - 1;
      if (recv_warp_id_in_rank == 0)
        recv_channel_offset[responsible_rank * num_channels + responsible_channel] = total_offset;
      num_tokens_to_recv -= total_offset;
    }
    total_offset = __shfl_sync(0xffffffff, total_offset, 0);
    total_offset += rank_offset;
    num_tokens_to_recv = __shfl_sync(0xffffffff, num_tokens_to_recv, 0);

    // Shared tail indices for different warps
    __shared__ volatile int shared_channel_tail_idx[kNumRanks];

    auto start_time = clock64();
    int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
    while (num_tokens_to_recv > 0) {
      // NOTES: unlike the sender, the receiver must ensure that the tail indices hold by different warps are same
      while (recv_thread_id_in_rank == 0) {
        cached_channel_tail_idx = ld_acquire_sys_global(channel_tail_idx.buffer());
        ;

        // Ready to copy
        if (cached_channel_head_idx != cached_channel_tail_idx) {
          shared_channel_tail_idx[responsible_rank] = cached_channel_tail_idx;
          break;
        }

        // Timeout check
        if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
          printf("DeepEP timeout for dispatch receivers, rank %d, responsible_channel = %d, tokens remained: %d\n",
                 rank, responsible_channel, num_tokens_to_recv);
          trap();
        }
      }

      // Synchronize queue tail
      asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
      cached_channel_tail_idx = shared_channel_tail_idx[responsible_rank];

      // Copy data
      int num_recv_tokens = cached_channel_tail_idx - cached_channel_head_idx;
      // Sender-direct: hidden was already written straight to recv_x by the sender; skip the drain.
      if (recv_pool_ptrs == nullptr)
        for (int chunk_idx = recv_warp_id_in_rank; chunk_idx < num_recv_tokens; chunk_idx += num_recv_warps_per_rank) {
          int token_idx_in_buffer = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
          auto shifted_buffer_x_int4 = channel_x_buffers.buffer() + token_idx_in_buffer * hidden_int4;
          auto shifted_recv_x_int4 = recv_x + static_cast<int64_t>(total_offset + chunk_idx) * hidden_int4;
          UNROLLED_WARP_COPY(5, recv_lane_id, hidden_int4, shifted_recv_x_int4, shifted_buffer_x_int4, ld_nc_global,
                             st_na_global);
        }

// Copy `src_idx`
#pragma unroll 4
      for (int chunk_idx = cached_channel_head_idx + recv_thread_id_in_rank; chunk_idx < cached_channel_tail_idx;
           chunk_idx += 32 * num_recv_warps_per_rank)
        recv_src_idx[total_offset + chunk_idx - cached_channel_head_idx] =
            ld_nc_global(channel_src_idx_buffers.buffer() + chunk_idx % num_recv_buffer_tokens);

// Copy `topk_idx` and `topk_weights`
#pragma unroll 4
      for (int idx = recv_thread_id_in_rank; idx < num_recv_tokens * num_topk; idx += 32 * num_recv_warps_per_rank) {
        int chunk_idx = idx / num_topk, token_topk_idx = idx % num_topk;
        int token_idx_in_buffer = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
        auto recv_idx = static_cast<int64_t>(total_offset + chunk_idx) * num_topk + token_topk_idx;
        auto buffer_idx = token_idx_in_buffer * num_topk + token_topk_idx;
        recv_topk_idx[recv_idx] = ld_nc_global(channel_topk_idx_buffers.buffer() + buffer_idx);
        recv_topk_weights[recv_idx] = ld_nc_global(channel_topk_weights_buffers.buffer() + buffer_idx);
      }

// Copy `x_scales`
#pragma unroll 4
      for (int i = recv_thread_id_in_rank; i < num_recv_tokens * num_scales; i += 32 * num_recv_warps_per_rank) {
        int chunk_idx = i / num_scales, scales_idx = i % num_scales;
        int token_idx_in_buffer = (cached_channel_head_idx + chunk_idx) % num_recv_buffer_tokens;
        recv_x_scales[static_cast<int64_t>(total_offset + chunk_idx) * num_scales + scales_idx] =
            ld_nc_global(channel_x_scales_buffers.buffer() + token_idx_in_buffer * num_scales + scales_idx);
      }

      // Move queue
      cached_channel_head_idx += num_recv_tokens;
      total_offset += num_recv_tokens;
      asm volatile("bar.sync %0, %1;" ::"r"(responsible_rank), "r"(num_threads_per_rank));
      if (recv_warp_id_in_rank == num_recv_warps_per_rank - 1 and recv_lane_id == 0)
        st_relaxed_sys_global(channel_head_idx.buffer(), cached_channel_head_idx);

      // Exit
      num_tokens_to_recv -= num_recv_tokens;
    }
  }
}

void dispatch(void* recv_x, float* recv_x_scales, int* recv_src_idx, int64_t* recv_topk_idx, float* recv_topk_weights,
              int* recv_channel_offset, int* send_head, const void* x, const float* x_scales, const int64_t* topk_idx,
              const float* topk_weights, const bool* is_token_in_rank, const int* channel_prefix_matrix, int num_tokens,
              int hidden_int4, int num_topk, int num_experts, int num_scales, void** buffer_ptrs, int rank,
              int num_ranks, cudaStream_t stream, int num_sms, int num_max_send_tokens, int num_recv_buffer_tokens,
              void** recv_pool_ptrs, int64_t recv_pool_header_bytes, int* ep_combine_recv_idx) {
  constexpr int kNumThreads = 512;

#define DISPATCH_LAUNCH_CASE(ranks)                                                                                 \
  LAUNCH_KERNEL(&cfg, (dispatch<ranks, kNumThreads>), reinterpret_cast<int4*>(recv_x), recv_x_scales, recv_src_idx, \
                recv_topk_idx, recv_topk_weights, recv_channel_offset, send_head, reinterpret_cast<const int4*>(x), \
                x_scales, topk_idx, topk_weights, is_token_in_rank, channel_prefix_matrix, num_tokens, hidden_int4, \
                num_topk, num_experts, num_scales, buffer_ptrs, rank, num_max_send_tokens, num_recv_buffer_tokens,  \
                recv_pool_ptrs, recv_pool_header_bytes, ep_combine_recv_idx);                                       \
  break

  // Even-numbered blocks for sending, odd-numbered blocks for receiving.
  EP_HOST_ASSERT(num_sms % 2 == 0);
  SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
  SWITCH_RANKS(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

// ---------------------------------------------------------------------------
// All-sender intranode dispatch (MSCCLPP_EP_INTRA_ALLSENDER, INTRA_DIRECT only).
//
// The 2-hop intranode dispatch splits its grid 50/50 into sender/receiver blocks
// (is_sender = sm_id % 2). Under INTRA_DIRECT the sender writes hidden straight to
// the destination pool, so the receiver blocks no longer drain hidden and sit
// nearly idle -- wasting half the SMs. At NSM=16 only 8 blocks move the 14 KB/token
// payload, vs NCCL-EP's 16 all-sender blocks (HYBRIDEP_DISPATCH_NUM_OF_BLOCKS=16).
//
// This kernel makes EVERY block a sender (one channel per block, num_channels =
// gridDim.x). For each of its channel's tokens, the rank-group warps write the
// hidden row directly to each destination's recv-pool slot, write send_head + the
// combine gather map, and pack per-token metadata (src_idx, rebased topk_idx,
// topk_weights, scales) into the destination pool's META region at the final recv
// slot. A separate intranode_meta_drain kernel then unpacks the local pool META
// region into the recv_* output tensors.
// No ring, no head/tail flow control, no receiver. Pairs with the TMA combine,
// which is token-parallel and ignores the channel prefix matrix.
template <int kNumRanks, int kNumThreads>
__global__ void __launch_bounds__(kNumThreads, 1)
    dispatch_allsender(int* send_head, const int4* x, const int64_t* topk_idx, const float* topk_weights,
                       const float* x_scales, const bool* is_token_in_rank, const int* channel_prefix_matrix,
                       int num_tokens, int hidden_int4, int num_topk, int num_experts, int num_scales,
                       void** buffer_ptrs, int rank, void** recv_pool_ptrs, int64_t recv_pool_header_bytes,
                       int64_t recv_pool_meta_base, int64_t meta_slot_bytes, int* ep_combine_recv_idx) {
  const int num_channels = static_cast<int>(gridDim.x);  // all-sender: one channel per block
  const int channel = static_cast<int>(blockIdx.x);
  const int thread_id = static_cast<int>(threadIdx.x);
  const int num_threads_per_rank = kNumThreads / kNumRanks;
  const int responsible_rank = thread_id / num_threads_per_rank;  // destination rank
  const int tg_lane = thread_id % num_threads_per_rank;           // 0..num_threads_per_rank-1
  const int lane = thread_id % 32;
  const int experts_per_rank = num_experts / kNumRanks;
  EP_DEVICE_ASSERT(kNumRanks <= 32);
  EP_DEVICE_ASSERT(kNumThreads % kNumRanks == 0);

  // Destination pool + this (src, dst) base recv slot. Mirrors the receiver's
  // total_offset = rank_prefix_matrix[(src-1)*R+dst] + channel_prefix_matrix[dst*C + ch-1].
  const int* dst_rank_prefix = reinterpret_cast<const int*>(buffer_ptrs[responsible_rank]);
  const int rank_off = rank > 0 ? dst_rank_prefix[(rank - 1) * kNumRanks + responsible_rank] : 0;
  const int ch_start = channel > 0 ? channel_prefix_matrix[responsible_rank * num_channels + channel - 1] : 0;
  const int64_t direct_base = static_cast<int64_t>(rank_off + ch_start);
  uint8_t* dst_pool_u8 = reinterpret_cast<uint8_t*>(recv_pool_ptrs[responsible_rank]);
  int4* direct_dst_pool = reinterpret_cast<int4*>(dst_pool_u8 + recv_pool_header_bytes);
  uint8_t* dst_meta_base = dst_pool_u8 + recv_pool_meta_base;

  int token_start_idx, token_end_idx;
  get_channel_task_range(num_tokens, num_channels, channel, token_start_idx, token_end_idx);

  int abs_pos = 0;  // running count of selected tokens for this (channel, dst); all rank-group threads agree
  for (int token_idx = token_start_idx; token_idx < token_end_idx; ++token_idx) {
    const bool in_rank = is_token_in_rank[token_idx * kNumRanks + responsible_rank];
    const int64_t pos = direct_base + abs_pos;

    // send_head membership marker (one thread per (token, dst)).
    if (tg_lane == 0) send_head[token_idx * kNumRanks + responsible_rank] = in_rank ? static_cast<int>(pos) : -1;

    if (in_rank) {
      // Hidden: all num_threads_per_rank threads of this rank group cooperate on one row.
      const int4* src_row = x + static_cast<int64_t>(token_idx) * hidden_int4;
      int4* dst_row = direct_dst_pool + pos * hidden_int4;
      for (int i = tg_lane; i < hidden_int4; i += num_threads_per_rank) st_na_global(dst_row + i, __ldg(src_row + i));

      uint8_t* m = dst_meta_base + pos * meta_slot_bytes;
      // src_idx + combine gather map (one thread).
      if (tg_lane == 0) {
        *reinterpret_cast<int*>(m) = token_idx;
        if (ep_combine_recv_idx != nullptr)
          ep_combine_recv_idx[token_idx * kNumRanks + responsible_rank] = static_cast<int>(pos);
      }
      // topk idx (rebased to this dst's local expert range) + weights into the meta slot.
      if (topk_idx != nullptr and lane < num_topk and (thread_id % num_threads_per_rank) < 32) {
        int* tidx = reinterpret_cast<int*>(m + sizeof(int));
        float* tw = reinterpret_cast<float*>(m + sizeof(int) + static_cast<size_t>(num_topk) * sizeof(int));
        const int recv_expert_begin = responsible_rank * experts_per_rank;
        const int recv_expert_end = recv_expert_begin + experts_per_rank;
        auto idx_value = __ldg(topk_idx + token_idx * num_topk + lane);
        const int rebased = (idx_value >= recv_expert_begin and idx_value < recv_expert_end)
                                ? static_cast<int>(idx_value) - recv_expert_begin
                                : -1;
        tidx[lane] = rebased;
        auto w = __ldg(topk_weights + token_idx * num_topk + lane);
        tw[lane] = (rebased >= 0) ? w : 0.0f;
      }
      // scales into the meta slot (after src_idx + topk_idx + topk_weights).
      if (x_scales != nullptr and (thread_id % num_threads_per_rank) < 32) {
        float* sdat =
            reinterpret_cast<float*>(m + sizeof(int) + static_cast<size_t>(num_topk) * (sizeof(int) + sizeof(float)));
        for (int s = lane; s < num_scales; s += 32)
          sdat[s] = __ldg(x_scales + static_cast<int64_t>(token_idx) * num_scales + s);
      }
      ++abs_pos;
    }
  }
}

// Unpack the local recv-pool META region (filled by the all-sender dispatch) into the
// recv_* output tensors. One thread per recv token.
__global__ void intranode_meta_drain_kernel(const uint8_t* __restrict__ pool_base, int64_t meta_base,
                                            int num_recv_tokens, int* __restrict__ recv_src_idx,
                                            int64_t* __restrict__ recv_topk_idx, float* __restrict__ recv_topk_weights,
                                            float* __restrict__ recv_x_scales, int num_topk, int num_scales,
                                            int64_t meta_slot_bytes) {
  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < num_recv_tokens; t += gridDim.x * blockDim.x) {
    const uint8_t* m = pool_base + meta_base + static_cast<int64_t>(t) * meta_slot_bytes;
    if (recv_src_idx != nullptr) recv_src_idx[t] = *reinterpret_cast<const int*>(m);
    if (recv_topk_idx != nullptr) {
      const int* tidx = reinterpret_cast<const int*>(m + sizeof(int));
      const float* tw = reinterpret_cast<const float*>(m + sizeof(int) + static_cast<size_t>(num_topk) * sizeof(int));
      for (int k = 0; k < num_topk; ++k) {
        recv_topk_idx[static_cast<int64_t>(t) * num_topk + k] = static_cast<int64_t>(tidx[k]);
        recv_topk_weights[static_cast<int64_t>(t) * num_topk + k] = tw[k];
      }
    }
    if (recv_x_scales != nullptr) {
      const float* sdat = reinterpret_cast<const float*>(m + sizeof(int) +
                                                         static_cast<size_t>(num_topk) * (sizeof(int) + sizeof(float)));
      for (int s = 0; s < num_scales; ++s) recv_x_scales[static_cast<int64_t>(t) * num_scales + s] = sdat[s];
    }
  }
}

void intranode_meta_drain(void* pool_base, int64_t meta_base, int num_recv_tokens, int* recv_src_idx,
                          int64_t* recv_topk_idx, float* recv_topk_weights, float* recv_x_scales, int num_topk,
                          int num_scales, int64_t meta_slot_bytes, cudaStream_t stream) {
  if (num_recv_tokens <= 0) return;
  constexpr int kThreads = 256;
  const int kBlocks = (num_recv_tokens + kThreads - 1) / kThreads;
  intranode_meta_drain_kernel<<<kBlocks, kThreads, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(pool_base), meta_base, num_recv_tokens, recv_src_idx, recv_topk_idx,
      recv_topk_weights, recv_x_scales, num_topk, num_scales, meta_slot_bytes);
}

// Host launcher for the all-sender intranode dispatch. num_sms == num_channels (every
// block is a sender). Requires the direct recv pools; metadata lands in the pool META
// region and must be unpacked by intranode_meta_drain afterward.
void dispatch_allsender(int* send_head, const void* x, const int64_t* topk_idx, const float* topk_weights,
                        const float* x_scales, const bool* is_token_in_rank, const int* channel_prefix_matrix,
                        int num_tokens, int hidden_int4, int num_topk, int num_experts, int num_scales,
                        void** buffer_ptrs, int rank, int num_ranks, cudaStream_t stream, int num_sms,
                        void** recv_pool_ptrs, int64_t recv_pool_header_bytes, int64_t recv_pool_meta_base,
                        int64_t meta_slot_bytes, int* ep_combine_recv_idx) {
  constexpr int kNumThreads = 512;
  EP_HOST_ASSERT(recv_pool_ptrs != nullptr);
  // Meta slot must hold src_idx + topk_idx(int) + topk_weights(float) + scales(float).
  EP_HOST_ASSERT(static_cast<int64_t>(sizeof(int)) +
                     static_cast<int64_t>(num_topk) * static_cast<int64_t>(sizeof(int) + sizeof(float)) +
                     static_cast<int64_t>(num_scales) * static_cast<int64_t>(sizeof(float)) <=
                 meta_slot_bytes);
#define DISPATCH_ALLSENDER_LAUNCH_CASE(ranks)                                                                          \
  LAUNCH_KERNEL(&cfg, (dispatch_allsender<ranks, kNumThreads>), send_head, reinterpret_cast<const int4*>(x), topk_idx, \
                topk_weights, x_scales, is_token_in_rank, channel_prefix_matrix, num_tokens, hidden_int4, num_topk,    \
                num_experts, num_scales, buffer_ptrs, rank, recv_pool_ptrs, recv_pool_header_bytes,                    \
                recv_pool_meta_base, meta_slot_bytes, ep_combine_recv_idx);                                            \
  break

  SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
  SWITCH_RANKS(DISPATCH_ALLSENDER_LAUNCH_CASE);
#undef DISPATCH_ALLSENDER_LAUNCH_CASE
}

template <int kNumRanks>
__global__ void cached_notify_combine(void** buffer_ptrs, int* send_head, int num_channels, int num_recv_tokens,
                                      int num_memset_int, int** task_fifo_ptrs, int head, int rank) {
  const auto sm_id = static_cast<int>(blockIdx.x);
  if (sm_id == 0) {
    // Barrier before cleaning
    barrier_device<kNumRanks>(task_fifo_ptrs, head, rank);
    move_fifo_slots<kNumRanks>(head);
    __syncthreads();

    // Clean
    auto thread_id = static_cast<int>(threadIdx.x), num_threads = static_cast<int>(blockDim.x);
    auto ptr = reinterpret_cast<int*>(buffer_ptrs[rank]);
#pragma unroll
    for (int i = thread_id; i < num_memset_int; i += num_threads) ptr[i] = 0;
    memory_fence();
    __syncthreads();

    // Barrier after cleaning
    barrier_device<kNumRanks>(task_fifo_ptrs, head, rank);
  } else {
    const auto channel_id = sm_id - 1;
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto rank_id = thread_id / 32;
    const auto lane_id = thread_id % 32;
    if (rank_id >= kNumRanks) return;

    int token_start_idx, token_end_idx;
    get_channel_task_range(num_recv_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

    // NOTES: `1 << 25` is a heuristic large number
    int last_head = 1 << 25;
#pragma unroll
    for (int token_idx_tail = token_end_idx - 1; token_idx_tail >= token_start_idx; token_idx_tail -= 32) {
      int token_idx = token_idx_tail - lane_id, expected_head = 0;
      auto current_head = (token_idx >= token_start_idx) ? __ldg(send_head + token_idx * kNumRanks + rank_id) : -1;
      for (int i = 0; i < min(32, token_idx_tail - token_start_idx + 1); ++i) {
        head = __shfl_sync(0xffffffff, current_head, i);
        if (head < 0) {
          if (lane_id == i) expected_head = -last_head - 1;
        } else {
          last_head = head;
        }
      }
      if (current_head < 0 and token_idx >= token_start_idx) send_head[token_idx * kNumRanks + rank_id] = expected_head;
    }
  }
}

void cached_notify_combine(void** buffer_ptrs, int* send_head, int num_channels, int num_recv_tokens,
                           int num_memset_int, int** task_fifo_ptrs, int head, int rank, int num_ranks,
                           cudaStream_t stream) {
#define CACHED_NOTIFY_COMBINE(ranks)                                                                       \
  LAUNCH_KERNEL(&cfg, cached_notify_combine<ranks>, buffer_ptrs, send_head, num_channels, num_recv_tokens, \
                num_memset_int, task_fifo_ptrs, head, rank);                                               \
  break

  const int num_threads = std::max(128, 32 * num_ranks);
  EP_HOST_ASSERT(num_ranks <= num_threads);
  EP_HOST_ASSERT(num_threads <= 1024);
  EP_HOST_ASSERT(1 + num_channels <= num_channels * 2);
  SETUP_LAUNCH_CONFIG(1 + num_channels, num_threads, stream);
  SWITCH_RANKS(CACHED_NOTIFY_COMBINE);
#undef CACHED_NOTIFY_COMBINE
}

template <typename dtype_t, int kNumRanks, int kNumThreads>
__global__ void __launch_bounds__(kNumThreads, 1)
    combine(dtype_t* recv_x, float* recv_topk_weights, const dtype_t* x, const float* topk_weights, const int* src_idx,
            const int* rank_prefix_matrix, const int* channel_prefix_matrix, int* send_head, int num_tokens,
            int num_recv_tokens, int hidden, int num_topk, void** buffer_ptrs, int rank, int num_max_send_tokens,
            int num_recv_buffer_tokens) {
  const auto num_sms = static_cast<int>(gridDim.x);
  const auto thread_id = static_cast<int>(threadIdx.x);
  const auto sm_id = static_cast<int>(blockIdx.x);
  const auto num_channels = num_sms / 2;
  const bool is_sender = sm_id % 2 == 0;
  const int responsible_channel = sm_id / 2;
  EP_DEVICE_ASSERT(num_topk <= 32);

  constexpr int kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);
  int hidden_int4 = hidden * sizeof(dtype_t) / sizeof(int4);
  auto x_int4 = reinterpret_cast<const int4*>(x);
  auto recv_int4 = reinterpret_cast<int4*>(recv_x);

  if (is_sender) {
    // Workers for sending
    // Several warps are responsible for a single rank
    constexpr int num_send_warps = kNumThreads / 32;
    constexpr int num_send_warps_per_rank = num_send_warps / kNumRanks;
    const auto num_threads_per_rank = num_send_warps_per_rank * 32;
    const auto send_thread_id = thread_id;
    const auto send_lane_id = send_thread_id % 32;
    const auto send_rank_id = thread_id / num_threads_per_rank;
    const auto send_warp_id_in_rank = send_thread_id % num_threads_per_rank / 32;

    // Calculate pointers by the specific layout
    auto ptr = reinterpret_cast<void*>(reinterpret_cast<int8_t*>(buffer_ptrs[send_rank_id]));
    auto num_channels_total = num_channels * kNumRanks;
    auto channel_rank_offset = responsible_channel * kNumRanks + rank;

    // Channel meta data
    // `head_idx`: kNumChannels * kNumRanks * sizeof(int)
    // `tail_idx`: kNumChannels * kNumRanks * sizeof(int)
    // `x_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * hidden_int4 * sizeof(int4)
    // `src_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * sizeof(int)
    // `topk_weights_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(float)
    auto channel_head_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_tail_idx = Buffer<int>(ptr, num_channels_total, channel_rank_offset);
    auto channel_x_buffers = Buffer<int4>(ptr, num_channels_total * num_recv_buffer_tokens * hidden_int4,
                                          channel_rank_offset * num_recv_buffer_tokens * hidden_int4);
    auto channel_src_idx_buffers =
        Buffer<int>(ptr, num_channels_total * num_recv_buffer_tokens, channel_rank_offset * num_recv_buffer_tokens);
    auto channel_topk_weights_buffers = Buffer<float>(ptr, num_channels_total * num_recv_buffer_tokens * num_topk,
                                                      channel_rank_offset * num_recv_buffer_tokens * num_topk);

    // Get tasks
    // NOTES: `channel_offset` is already shifted
    int rank_offset = send_rank_id > 0 ? rank_prefix_matrix[(send_rank_id - 1) * kNumRanks + rank] : 0;
    int num_rank_tokens = rank_prefix_matrix[send_rank_id * kNumRanks + rank] - rank_offset;
    int channel_offset = channel_prefix_matrix[send_rank_id * num_channels + responsible_channel];
    int num_channel_tokens = (responsible_channel == num_channels - 1
                                  ? num_rank_tokens
                                  : channel_prefix_matrix[send_rank_id * num_channels + responsible_channel + 1]) -
                             channel_offset;
    int token_start_idx = rank_offset + channel_offset,
        token_end_idx = rank_offset + channel_offset + num_channel_tokens;

    // Iterate over all tokens and send by chunks
    int current_channel_tail_idx = 0;
    for (int64_t token_idx = token_start_idx; token_idx < token_end_idx;) {
      // Check destination queue emptiness, or wait a buffer to be released (rare cases)
      auto start_time = clock64();
      int num_round_tokens = min(num_max_send_tokens, token_end_idx - static_cast<int>(token_idx));
      while (send_lane_id == 0) {
        // NOTES: we only consider the worst case, because counting the real numbers are time-consuming
        int num_used_slots = current_channel_tail_idx - ld_volatile_global(channel_head_idx.buffer());
        if (num_recv_buffer_tokens - num_used_slots >= num_round_tokens) break;

        // Rare cases to loop again
        if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
          printf("DeepEP timeout for combine senders, rank %d, responsible_channel = %d\n", rank, responsible_channel);
          trap();
        }
      }
      __syncwarp();

// Send by chunk
#pragma unroll
      for (int i = send_warp_id_in_rank; i < num_round_tokens; i += num_send_warps_per_rank) {
        // Get an empty slot
        int dst_slot_idx = (current_channel_tail_idx + i) % num_recv_buffer_tokens;

        // Copy data
        auto shifted_x_buffers = channel_x_buffers.buffer() + dst_slot_idx * hidden_int4;
        auto shifted_x = x_int4 + (token_idx + i) * hidden_int4;
        UNROLLED_WARP_COPY(4, send_lane_id, hidden_int4, shifted_x_buffers, shifted_x, ld_nc_global, st_na_global);

        // Send source index
        if (send_lane_id == 0) channel_src_idx_buffers[dst_slot_idx] = __ldg(src_idx + token_idx + i);

        // Send `topk_weights`
        if (num_topk > 0 and send_lane_id < num_topk)
          channel_topk_weights_buffers[dst_slot_idx * num_topk + send_lane_id] =
              __ldg(topk_weights + (token_idx + i) * num_topk + send_lane_id);
      }
      token_idx += num_round_tokens;
      current_channel_tail_idx += num_round_tokens;

      // Move tail index
      asm volatile("bar.sync %0, %1;" ::"r"(send_rank_id), "r"(num_threads_per_rank));
      if (send_lane_id == 0 and send_warp_id_in_rank == 0)
        st_release_sys_global(channel_tail_idx.buffer(), current_channel_tail_idx);
    }
  } else {
    // Workers for receiving
    // One warp for moving the queue head, others for reduction
    constexpr int num_recv_warps = kNumThreads / 32;
    const auto recv_warp_id = thread_id / 32;
    const auto recv_lane_id = thread_id % 32;
    EP_DEVICE_ASSERT(kNumRanks <= 32 and kNumThreads > 32);
    EP_DEVICE_ASSERT(thread_id >= 0 and kNumThreads % 32 == 0);

    // Shared head, tail and retired flags for receiver warps
    __shared__ volatile int warp_channel_head_idx[num_recv_warps][kNumRanks];
    __shared__ volatile int channel_tail_idx[kNumRanks];
    __shared__ volatile bool warp_retired[num_recv_warps];
    if (thread_id < num_recv_warps) warp_retired[thread_id] = false;
    if (recv_lane_id < kNumRanks) warp_channel_head_idx[recv_warp_id][recv_lane_id] = 0;
    if (thread_id < kNumRanks) channel_tail_idx[thread_id] = 0;
    asm volatile("bar.sync 0, %0;" ::"r"(kNumThreads));

    if (thread_id < 32) {
      int* channel_head_idx_ptr =
          reinterpret_cast<int*>(buffer_ptrs[rank]) + responsible_channel * kNumRanks + recv_lane_id;
      int* channel_tail_idx_ptr = channel_head_idx_ptr + num_channels * kNumRanks;

      // Queue head updater
      int last_head = 0;
      while (recv_lane_id < kNumRanks) {
        // Check retired
        bool retired = true;
#pragma unroll
        for (int i = 1; i < num_recv_warps; ++i) retired = retired and warp_retired[i];
        if (retired) break;

        // Update queue tail
        channel_tail_idx[recv_lane_id] = ld_acquire_sys_global(channel_tail_idx_ptr);

        // Update minimum head
        int min_head = std::numeric_limits<int>::max();
#pragma unroll
        for (int i = 1; i < num_recv_warps; ++i)
          if (not warp_retired[i]) min_head = min(min_head, warp_channel_head_idx[i][recv_lane_id]);
        if (min_head != std::numeric_limits<int>::max() and min_head > last_head)
          st_relaxed_sys_global(channel_head_idx_ptr, last_head = min_head);
      }
    } else {
      // Receivers
      // Channel metadata
      // All lanes will use data buffer, but only rank lane will use `head/tail/src_idx`
      Buffer<int4> channel_x_buffers[kNumRanks];
      Buffer<float> channel_topk_weights_buffers[kNumRanks];

// Calculate pointers by the specific layout
#pragma unroll
      for (int i = 0; i < kNumRanks; ++i) {
        auto channel_rank_offset = responsible_channel * kNumRanks + i;
        auto num_channels_total = num_channels * kNumRanks;
        // `head_idx` & `tail_idx`: kNumChannels * kNumRanks * sizeof(int)
        auto ptr = reinterpret_cast<void*>(reinterpret_cast<int8_t*>(buffer_ptrs[rank]) +
                                           2 * num_channels * kNumRanks * sizeof(int));

        // `x_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * hidden_int4 * sizeof(int4)
        channel_x_buffers[i] = Buffer<int4>(ptr, num_channels_total * num_recv_buffer_tokens * hidden_int4,
                                            channel_rank_offset * num_recv_buffer_tokens * hidden_int4);

        // `src_idx_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * sizeof(int)
        ptr = reinterpret_cast<void*>(reinterpret_cast<int8_t*>(ptr) +
                                      num_channels_total * num_recv_buffer_tokens * sizeof(int));

        // `topk_weights_buffers`: kNumChannels * kNumRanks * num_recv_buffer_tokens * num_topk * sizeof(float)
        channel_topk_weights_buffers[i] = Buffer<float>(ptr, num_channels_total * num_recv_buffer_tokens * num_topk,
                                                        channel_rank_offset * num_recv_buffer_tokens * num_topk);
      }

      // The same tokens as the dispatch process
      int token_start_idx, token_end_idx;
      get_channel_task_range(num_recv_tokens, num_channels, responsible_channel, token_start_idx, token_end_idx);

      // Iterate over all tokens and combine
      for (int64_t token_idx = token_start_idx + recv_warp_id - 1; token_idx < token_end_idx;
           token_idx += num_recv_warps - 1) {
        // Read expected head
        int expected_head = -1;
        if (recv_lane_id < kNumRanks) expected_head = ld_nc_global(send_head + token_idx * kNumRanks + recv_lane_id);

        auto start_time = clock64();
        while (expected_head >= 0 and channel_tail_idx[recv_lane_id] <= expected_head) {
          // Timeout check
          if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
            printf("DeepEP timeout for combine receivers, rank %d, responsible_channel = %d, expect = %d\n", rank,
                   responsible_channel, expected_head);
            trap();
          }
        }
        __syncwarp();

        // Broadcast current heads
        int num_topk_ranks = 0, topk_ranks[kNumRanks], slot_indices[kNumRanks];
#pragma unroll
        for (int i = 0; i < kNumRanks; ++i) {
          auto expected_head_i = __shfl_sync(0xffffffff, expected_head, i);
          if (expected_head_i >= 0) {
            slot_indices[num_topk_ranks] = expected_head_i % num_recv_buffer_tokens;
            topk_ranks[num_topk_ranks++] = i;
          }
        }

// Reduce data
#pragma unroll
        for (int i = recv_lane_id; i < hidden_int4; i += 32) {
          // Read buffers
          int4 recv_value_int4[kNumRanks];
#pragma unroll
          for (int j = 0; j < num_topk_ranks; ++j)
            recv_value_int4[j] =
                ld_nc_global(channel_x_buffers[topk_ranks[j]].buffer() + slot_indices[j] * hidden_int4 + i);

          // Reduce all-to-all results
          float values[kDtypePerInt4] = {0};
#pragma unroll
          for (int j = 0; j < num_topk_ranks; ++j) {
            auto recv_value_dtypes = reinterpret_cast<const dtype_t*>(&recv_value_int4[j]);
#pragma unroll
            for (int k = 0; k < kDtypePerInt4; ++k) values[k] += static_cast<float>(recv_value_dtypes[k]);
          }

          // Cast back to `dtype_t` and write
          int4 out_int4;
          auto out_dtypes = reinterpret_cast<dtype_t*>(&out_int4);
#pragma unroll
          for (int j = 0; j < kDtypePerInt4; ++j) out_dtypes[j] = static_cast<dtype_t>(values[j]);
          recv_int4[token_idx * hidden_int4 + i] = out_int4;
        }

        // Reduce `topk_weights`
        if (recv_lane_id < num_topk) {
          float value = 0;
#pragma unroll
          for (int i = 0; i < num_topk_ranks; ++i)
            value += ld_nc_global(channel_topk_weights_buffers[topk_ranks[i]].buffer() + slot_indices[i] * num_topk +
                                  recv_lane_id);
          recv_topk_weights[token_idx * num_topk + recv_lane_id] = value;
        }

        // Update head
        if (recv_lane_id < kNumRanks)
          warp_channel_head_idx[recv_warp_id][recv_lane_id] =
              (expected_head < 0) ? -expected_head - 1 : expected_head + 1;
      }

      // Retired
      __syncwarp();
      if (recv_lane_id == 0) warp_retired[recv_warp_id] = true;
    }
  }
}

// ---------------------------------------------------------------------------
// Intranode TMA-staged direct-gather combine for the single-node peer-mapped path.
//
// For each combined output token, it discovers the contributing ranks and
// gathers each contributor's hidden row
// straight from that rank's IPC-mapped recv-output pool (recv_pool_ptrs[r] at
// slot ep_combine_recv_idx[t,r]) through a kStages-deep cp.async.bulk (TMA) SMEM
// pipeline, reduces from SMEM, and writes the summed row to combined_x. No
// 2-hop ring, no channel partitioning -> the grid is pure token-parallel so the
// block count can be set independently via MSCCLPP_EP_COMBINE_NSM.
//
// Contributor discovery uses send_head (>=0 == token routed to that rank during
// dispatch); the per-(token,rank) recv-pool slot comes from ep_combine_recv_idx,
// which the dispatch sender-direct path fills. combined_topk_weights is zeroed
// (the current intranode test validates only combined_x).
#ifndef EP_ICMB_TMA_CHUNK_INT4
#define EP_ICMB_TMA_CHUNK_INT4 64  // hidden chunk in int4 (1KB TMA descriptors)
#endif
#ifndef EP_ICMB_TMA_STAGES
#define EP_ICMB_TMA_STAGES 2  // pipeline depth (outstanding chunks in flight)
#endif
#ifndef EP_ICMB_TMA_WARPS
#define EP_ICMB_TMA_WARPS 16  // token-parallel warps per block
#endif

template <typename dtype_t, int kNumRanks, int kWarps>
__global__ void __launch_bounds__(kWarps * 32, 1)
    combine_intranode_gather_tma(int4* combined_x, float* combined_topk_weights, const int* send_head,
                                 int num_combined_tokens, int hidden, int num_topk, int num_ranks,
                                 void** recv_pool_ptrs, const int* ep_combine_recv_idx,
                                 int64_t recv_pool_header_bytes) {
  constexpr int kMaxContrib = kNumRanks;
  constexpr int kChunkInt4 = EP_ICMB_TMA_CHUNK_INT4;
  constexpr int kStages = EP_ICMB_TMA_STAGES;
  constexpr int kChunkBytes = kChunkInt4 * static_cast<int>(sizeof(int4));
  constexpr auto kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);

  const auto lane_id = get_lane_id();
  const int warp_id = static_cast<int>(threadIdx.x) / 32;
  const auto hidden_int4 = hidden / static_cast<int>(sizeof(int4) / sizeof(dtype_t));

  // Dynamic SMEM: [kWarps][kStages][kMaxContrib][kChunkBytes] staging tiles, then [kWarps][kStages] mbarriers.
  extern __shared__ uint8_t smem_raw[];
  const size_t per_warp_stage_bytes = static_cast<size_t>(kStages) * kMaxContrib * kChunkBytes;
  uint8_t* my_stage = smem_raw + warp_id * per_warp_stage_bytes;
  uint64_t* mbar_base = reinterpret_cast<uint64_t*>(smem_raw + static_cast<size_t>(kWarps) * per_warp_stage_bytes);
  uint64_t* my_mbar = mbar_base + warp_id * kStages;
  auto stage_buf = [&](int s, int j) -> uint8_t* {
    return my_stage + (static_cast<size_t>(s) * kMaxContrib + j) * kChunkBytes;
  };

  const int global_warp = static_cast<int>(blockIdx.x) * kWarps + warp_id;
  const int total_warps = static_cast<int>(gridDim.x) * kWarps;
  const int nchunks = (hidden_int4 + kChunkInt4 - 1) / kChunkInt4;

  for (int t = global_warp; t < num_combined_tokens; t += total_warps) {
    // ---- discovery: which ranks contributed to this token + their recv slots ----
    int topk_ranks[kMaxContrib], slot_indices[kMaxContrib], num_topk_ranks = 0;
    for (int base = 0; base < kNumRanks; base += 32) {
      const int r = base + lane_id;
      const bool is_in = (r < kNumRanks) and (send_head[static_cast<int64_t>(t) * num_ranks + r] >= 0);
      const int slot = is_in ? ep_combine_recv_idx[static_cast<int64_t>(t) * num_ranks + r] : 0;
      unsigned ballot = __ballot_sync(0xffffffffu, is_in);
      while (ballot != 0u) {
        const int l = __ffs(static_cast<int>(ballot)) - 1;
        if (num_topk_ranks < kMaxContrib) {
          topk_ranks[num_topk_ranks] = base + l;
          slot_indices[num_topk_ranks] = __shfl_sync(0xffffffffu, slot, l);
          ++num_topk_ranks;
        }
        ballot &= ballot - 1u;
      }
    }

    int4* combined_row = combined_x + static_cast<int64_t>(t) * hidden_int4;

    // lane 0 posts all contributors' TMA G2S loads for one chunk into stage s.
    auto issue_loads = [&](int s, int c0, int csize_int4) {
      if (lane_id == 0) {
        const uint32_t mbar_a = static_cast<uint32_t>(__cvta_generic_to_shared(&my_mbar[s]));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(mbar_a));
        fenceProxyAsyncSharedCta();
        const uint32_t cbytes = static_cast<uint32_t>(csize_int4 * static_cast<int>(sizeof(int4)));
        for (int j = 0; j < num_topk_ranks; ++j) {
          const uint8_t* src =
              reinterpret_cast<const uint8_t*>(recv_pool_ptrs[topk_ranks[j]]) + recv_pool_header_bytes +
              static_cast<int64_t>(slot_indices[j]) * hidden_int4 * static_cast<int64_t>(sizeof(int4)) +
              static_cast<int64_t>(c0) * sizeof(int4);
          const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(stage_buf(s, j)));
          asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
                       "l"(src), "r"(cbytes), "r"(mbar_a)
                       : "memory");
        }
        uint64_t st;
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;"
                     : "=l"(st)
                     : "r"(mbar_a), "r"(static_cast<uint32_t>(cbytes * num_topk_ranks)));
      }
    };

    // all lanes wait for stage s's loads, then fence so generic reads see the async-proxy writes.
    auto wait_stage = [&](int s) {
      if (lane_id == 0) {
        const uint32_t mbar_a = static_cast<uint32_t>(__cvta_generic_to_shared(&my_mbar[s]));
        uint32_t done = 0;
        while (!done) {
          asm volatile(
              "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0;"
              " selp.u32 %0, 1, 0, p; }"
              : "=r"(done)
              : "r"(mbar_a));
        }
      }
      __syncwarp();
      fenceProxyAsyncSharedCta();
    };

    auto reduce_store = [&](int s, int c0, int csize_int4) {
      for (int i = lane_id; i < csize_int4; i += 32) {
        float values[kDtypePerInt4];
#pragma unroll
        for (int k = 0; k < static_cast<int>(kDtypePerInt4); ++k) values[k] = 0.0f;
#pragma unroll
        for (int j = 0; j < kMaxContrib; ++j) {
          if (j >= num_topk_ranks) break;
          const int4 v = *reinterpret_cast<const int4*>(stage_buf(s, j) + i * static_cast<int>(sizeof(int4)));
          auto vd = reinterpret_cast<const dtype_t*>(&v);
#pragma unroll
          for (int k = 0; k < static_cast<int>(kDtypePerInt4); ++k) values[k] += static_cast<float>(vd[k]);
        }
        int4 out_int4;
        auto out_d = reinterpret_cast<dtype_t*>(&out_int4);
#pragma unroll
        for (int k = 0; k < static_cast<int>(kDtypePerInt4); ++k) out_d[k] = static_cast<dtype_t>(values[k]);
        st_na_global(combined_row + c0 + i, out_int4);
      }
    };

    // Software pipeline: prologue issues the first kStages-1 chunks; each iteration issues
    // chunk c+(kStages-1) while waiting on + reducing chunk c.
#pragma unroll
    for (int p = 0; p < kStages - 1; ++p) {
      if (p < nchunks) {
        const int c0 = p * kChunkInt4;
        const int csize = (c0 + kChunkInt4 <= hidden_int4) ? kChunkInt4 : (hidden_int4 - c0);
        issue_loads(p % kStages, c0, csize);
      }
    }
    for (int c = 0; c < nchunks; ++c) {
      const int s = c % kStages;
      const int c0 = c * kChunkInt4;
      const int csize = (c0 + kChunkInt4 <= hidden_int4) ? kChunkInt4 : (hidden_int4 - c0);
      const int cn = c + (kStages - 1);
      if (cn < nchunks) {
        const int sn = cn % kStages;
        const int c0n = cn * kChunkInt4;
        const int csizen = (c0n + kChunkInt4 <= hidden_int4) ? kChunkInt4 : (hidden_int4 - c0n);
        issue_loads(sn, c0n, csizen);
      }
      wait_stage(s);
      reduce_store(s, c0, csize);
      __syncwarp();
    }
    if (lane_id < num_topk) st_na_global(combined_topk_weights + static_cast<int64_t>(t) * num_topk + lane_id, 0.0f);
  }
}

// Host launcher for the intranode TMA direct-gather combine. Returns false (and
// launches nothing) when the direct-gather inputs are unavailable, so the caller
// falls back to the 2-hop ring combine.
bool combine_tma(cudaDataType_t type, void* combined_x, float* combined_topk_weights, int* send_head, int num_tokens,
                 int hidden, int num_topk, int num_ranks, void** recv_pool_ptrs, const int* ep_combine_recv_idx,
                 int64_t recv_pool_header_bytes, int combine_sms, cudaStream_t stream) {
  if (recv_pool_ptrs == nullptr or ep_combine_recv_idx == nullptr) return false;
  EP_HOST_ASSERT(type == CUDA_R_16BF);
  constexpr int kStages = EP_ICMB_TMA_STAGES;
  constexpr int kChunkInt4 = EP_ICMB_TMA_CHUNK_INT4;
  const int num_blocks = std::max(1, combine_sms);

  // SMEM/block = kWarps*kStages*kMaxContrib(=ranks)*kChunkBytes + mbars. kChunkBytes=1KB,
  // kStages=2. Keep it under the GB200 ~227KB opt-in cap: 16 warps for <=4 ranks (128KB),
  // 12 warps for 8 ranks (192KB).
#define COMBINE_INTRANODE_TMA_LAUNCH(ranks, WARPS)                                                                   \
  {                                                                                                                  \
    auto tma_func = combine_intranode_gather_tma<nv_bfloat16, ranks, WARPS>;                                         \
    const size_t tma_smem = static_cast<size_t>(WARPS) * kStages * (ranks) * kChunkInt4 * sizeof(int4) +             \
                            static_cast<size_t>(WARPS) * kStages * sizeof(uint64_t);                                 \
    CUDA_CHECK(                                                                                                      \
        cudaFuncSetAttribute(tma_func, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(tma_smem)));    \
    cudaLaunchConfig_t cfg = {                                                                                       \
        static_cast<unsigned>(num_blocks), static_cast<unsigned>((WARPS) * 32), tma_smem, stream, nullptr, 0};       \
    LAUNCH_KERNEL(&cfg, tma_func, reinterpret_cast<int4*>(combined_x), combined_topk_weights, send_head, num_tokens, \
                  hidden, num_topk, num_ranks, recv_pool_ptrs, ep_combine_recv_idx, recv_pool_header_bytes);         \
  }                                                                                                                  \
  break

  switch (num_ranks) {
    case 2:
      COMBINE_INTRANODE_TMA_LAUNCH(2, EP_ICMB_TMA_WARPS);
    case 4:
      COMBINE_INTRANODE_TMA_LAUNCH(4, EP_ICMB_TMA_WARPS);
    case 8:
      COMBINE_INTRANODE_TMA_LAUNCH(8, 12);
    default:
      EP_HOST_ASSERT(false and "Unsupported ranks");
  }
#undef COMBINE_INTRANODE_TMA_LAUNCH
  return true;
}

void combine(cudaDataType_t type, void* recv_x, float* recv_topk_weights, const void* x, const float* topk_weights,
             const int* src_idx, const int* rank_prefix_matrix, const int* channel_prefix_matrix, int* send_head,
             int num_tokens, int num_recv_tokens, int hidden, int num_topk, void** buffer_ptrs, int rank, int num_ranks,
             cudaStream_t stream, int num_sms, int num_max_send_tokens, int num_recv_buffer_tokens) {
  constexpr int kNumThreads = 768;

#define COMBINE_LAUNCH_CASE(dtype, ranks)                                                                            \
  LAUNCH_KERNEL(&cfg, (combine<dtype, ranks, kNumThreads>), reinterpret_cast<dtype*>(recv_x), recv_topk_weights,     \
                reinterpret_cast<const dtype*>(x), topk_weights, src_idx, rank_prefix_matrix, channel_prefix_matrix, \
                send_head, num_tokens, num_recv_tokens, hidden, num_topk, buffer_ptrs, rank, num_max_send_tokens,    \
                num_recv_buffer_tokens);                                                                             \
  break
#define COMBINE_DTYPE_LAUNCH_CASE(dtype)               \
  SWITCH_RANKS_WITH_DTYPE(dtype, COMBINE_LAUNCH_CASE); \
  break

  // Even-numbered blocks for sending, odd-numbered blocks for receiving
  EP_HOST_ASSERT(num_sms % 2 == 0);
  EP_HOST_ASSERT(kNumThreads >= num_ranks * 32);
  SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
  SWITCH_TYPES(COMBINE_DTYPE_LAUNCH_CASE);
#undef COMBINE_DTYPE_LAUNCH_CASE
#undef COMBINE_LAUNCH_CASE
}

}  // namespace intranode

}  // namespace ep
}  // namespace mscclpp
