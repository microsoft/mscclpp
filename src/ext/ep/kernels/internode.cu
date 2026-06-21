// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#include <limits>
#include <mscclpp/memory_channel_device.hpp>
#include <mscclpp/port_channel_device.hpp>
#include <type_traits>

#include "buffer.cuh"
#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"

namespace mscclpp {
namespace ep {

// Packed type for loading NUM_MAX_NVL_PEERS bools as a single integer.
// Supports the two configurations the rest of the kernel logic assumes:
//   8 peers -> uint64_t (8 bools fit exactly)
//   4 peers -> uint32_t (4 bools fit exactly)
// All packed loads/stores below use this alias instead of a hardcoded uint64_t.
using NvlPackT = std::conditional_t<NUM_MAX_NVL_PEERS == 8, uint64_t, uint32_t>;
static_assert(NUM_MAX_NVL_PEERS == 8 || NUM_MAX_NVL_PEERS == 4,
              "NUM_MAX_NVL_PEERS must be 4 or 8 for HT internode kernel");
static_assert(NUM_MAX_NVL_PEERS * sizeof(bool) == sizeof(NvlPackT), "NvlPackT size must match NUM_MAX_NVL_PEERS bools");

namespace internode {

// ===========================================================================
// NVLS multimem PTX wrappers. `multimem.*` ops require sm_90+ (Hopper /
// Blackwell) and an active NVLink-SHARP multicast handle. Hosts gate
// every call site behind `nvls_mc_ptr != nullptr`, which is set only when
// `mscclpp::isNvlsSupported()` returned true at Buffer construction, so
// these macros are unreachable on older archs at runtime — the `__trap()`
// fallback is purely a build-time guard to keep PTX assembly clean on
// sm_80 / sm_86 / sm_89 targets.
// ===========================================================================
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define MSCCLPP_EP_MULTIMEM_RED_ADD_RELAXED_U64(ptr, delta) \
  asm volatile("multimem.red.relaxed.sys.global.add.u64 [%0], %1;" ::"l"(ptr), "l"((uint64_t)(delta)) : "memory")
#define MSCCLPP_EP_MULTIMEM_RED_ADD_RELEASE_U64_ONE(ptr) \
  asm volatile("multimem.red.release.sys.global.add.u64 [%0], 1;" ::"l"(ptr) : "memory")
#define MSCCLPP_EP_MULTIMEM_ST_RELAXED_U32(ptr, val) \
  asm volatile("multimem.st.relaxed.sys.global.u32 [%0], %1;" ::"l"(ptr), "r"((uint32_t)(val)) : "memory")
#else
#define MSCCLPP_EP_MULTIMEM_RED_ADD_RELAXED_U64(ptr, delta) __trap()
#define MSCCLPP_EP_MULTIMEM_RED_ADD_RELEASE_U64_ONE(ptr) __trap()
#define MSCCLPP_EP_MULTIMEM_ST_RELAXED_U32(ptr, val) __trap()
#endif

// ===========================================================================
// NVLS HT counter helpers (Phase 3).
//
// All cross-node tail/head counter updates that previously used HW
// atomicAdd (broken on Azure CX-7 RoCE: IBV_ATOMIC_NONE) and were
// patched to absolute-value RDMA WRITE — go through the NVLS multicast
// fabric instead. multimem.red.add is hardware-atomic across all ranks
// in the multicast group (validated cross-node Phase 2). For the legacy
// IB-platform path, callers pass `nvls_*_mc/dev == nullptr` and the
// existing PortChannel path runs.
//
// Layout (matches `nvls_off_tail/head` allocation in buffer.cc):
//   region[kind][channel][src_rdma][dst_rdma]   (uint64 each)
// Sender writes to slot(channel, my_rdma_rank, dst_rdma_rank).
// Receiver reads from slot(channel, src_rdma_rank, my_rdma_rank).
// Self-loop (src == dst == my_rdma_rank) goes through the same path.
// ===========================================================================

__device__ __forceinline__ size_t nvls_ctr_slot_index(int channel_id, int src_rdma, int dst_rdma) {
  // NUM_MAX_RDMA_PEERS² slots per channel; uint64 each → slot stride is 8B
  // but we return slot index (multiply by sizeof(uint64) at use site via
  // pointer arithmetic on uint64_t*).
  return (size_t)channel_id * (size_t)NUM_MAX_RDMA_PEERS * (size_t)NUM_MAX_RDMA_PEERS +
         (size_t)src_rdma * (size_t)NUM_MAX_RDMA_PEERS + (size_t)dst_rdma;
}

__device__ __forceinline__ void nvls_ctr_add(void* mc_base, int channel_id, int src_rdma, int dst_rdma,
                                             uint64_t delta) {
  uint64_t* slot = reinterpret_cast<uint64_t*>(mc_base) + nvls_ctr_slot_index(channel_id, src_rdma, dst_rdma);
  // NOTE: must be `relaxed`, not `release`. Empirically on Azure GB200,
  // `multimem.red.release.sys.global.add.u64` issued concurrently from
  // many channels (warps) triggers `unspecified launch failure`. Use a
  // preceding `__threadfence_system()` at the call site to publish data
  // before bumping the counter.
  MSCCLPP_EP_MULTIMEM_RED_ADD_RELAXED_U64(slot, delta);
}

__device__ __forceinline__ uint64_t nvls_ctr_load(void* dev_base, int channel_id, int src_rdma, int dst_rdma) {
  uint64_t* slot = reinterpret_cast<uint64_t*>(dev_base) + nvls_ctr_slot_index(channel_id, src_rdma, dst_rdma);
  uint64_t v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(slot) : "memory");
  return v;
}

template <int kNumThreads, int kNumExpertsPerSM, int kNumRanksPerSM>
__global__ void __launch_bounds__(kNumThreads, 1)
    get_dispatch_layout(const int64_t* topk_idx, int* num_tokens_per_rank, int* num_tokens_per_rdma_rank,
                        int* num_tokens_per_expert, bool* is_token_in_rank, int num_tokens, int num_topk, int num_ranks,
                        int num_experts) {
  auto sm_id = static_cast<int>(blockIdx.x);
  auto thread_id = static_cast<int>(threadIdx.x);

  // Count expert statistics
  __shared__ int num_tokens_per_expert_per_thread[kNumThreads][kNumExpertsPerSM];
  int expert_begin_idx = sm_id * kNumExpertsPerSM,
      expert_end_idx = min(expert_begin_idx + kNumExpertsPerSM, num_experts);
  if (expert_begin_idx < expert_end_idx) {
// Per-thread count
#pragma unroll
    for (int i = 0; i < kNumExpertsPerSM; ++i) num_tokens_per_expert_per_thread[thread_id][i] = 0;
#pragma unroll
    for (int i = thread_id; i < num_tokens; i += kNumThreads) {
      auto shifted_topk_idx = topk_idx + i * num_topk;
#pragma unroll
      for (int j = 0, expert_idx; j < num_topk; ++j) {
        expert_idx = static_cast<int>(shifted_topk_idx[j]);
        if (expert_begin_idx <= expert_idx and expert_idx < expert_end_idx)
          ++num_tokens_per_expert_per_thread[thread_id][expert_idx - expert_begin_idx];
      }
    }
    __syncthreads();

    // Sum up
    EP_STATIC_ASSERT(kNumExpertsPerSM <= kNumThreads, "Too many experts per SM");
    if (expert_begin_idx + thread_id < expert_end_idx) {
      int sum = 0;
#pragma unroll
      for (int i = 0; i < kNumThreads; ++i) sum += num_tokens_per_expert_per_thread[i][thread_id];
      num_tokens_per_expert[expert_begin_idx + thread_id] = sum;
    }
    return;
  }

  if (num_tokens_per_rdma_rank != nullptr)
    EP_DEVICE_ASSERT(num_ranks % NUM_MAX_NVL_PEERS == 0 and num_ranks > NUM_MAX_NVL_PEERS);

  // Count rank statistics
  constexpr int kNumRDMARanksPerSM = kNumRanksPerSM / NUM_MAX_NVL_PEERS;
  __shared__ int num_tokens_per_rank_per_thread[kNumThreads][kNumRanksPerSM];
  __shared__ int num_tokens_per_rdma_rank_per_thread[kNumThreads][kNumRDMARanksPerSM];
  auto sm_begin = (num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM;
  int rank_begin_idx = (sm_id - sm_begin) * kNumRanksPerSM,
      rank_end_idx = min(rank_begin_idx + kNumRanksPerSM, num_ranks);
  int rdma_rank_begin_idx = rank_begin_idx / NUM_MAX_NVL_PEERS, rdma_rank_end_idx = rank_end_idx / NUM_MAX_NVL_PEERS;
  if (rank_begin_idx < rank_end_idx) {
    const auto num_expert_per_rank = num_experts / num_ranks;
    auto expert_begin = rank_begin_idx * num_expert_per_rank;
    auto expert_end = rank_end_idx * num_expert_per_rank;

// Per-thread count
#pragma unroll
    for (int i = 0; i < kNumRanksPerSM; ++i) num_tokens_per_rank_per_thread[thread_id][i] = 0;
#pragma unroll
    for (int i = 0; i < kNumRDMARanksPerSM; ++i) num_tokens_per_rdma_rank_per_thread[thread_id][i] = 0;
#pragma unroll
    for (int i = thread_id; i < num_tokens; i += kNumThreads) {
      auto shifted_topk_idx = topk_idx + i * num_topk;
      int is_in_rank[kNumRanksPerSM] = {0}, is_in_rdma_rank[kNumRDMARanksPerSM] = {0};
#pragma unroll
      for (int j = 0, expert_idx, rank_idx; j < num_topk; ++j) {
        expert_idx = static_cast<int>(shifted_topk_idx[j]);
        if (expert_begin <= expert_idx and expert_idx < expert_end) {
          // Count single rank
          rank_idx = expert_idx / num_expert_per_rank - rank_begin_idx;
          is_in_rank[rank_idx]++, is_in_rdma_rank[rank_idx / NUM_MAX_NVL_PEERS]++;
        }
      }

      auto shifted_is_token_in_rank = is_token_in_rank + i * num_ranks;
#pragma unroll
      for (int j = 0; j + rank_begin_idx < rank_end_idx; ++j) {
        shifted_is_token_in_rank[j + rank_begin_idx] = (is_in_rank[j] > 0);
        num_tokens_per_rank_per_thread[thread_id][j] += (is_in_rank[j] > 0);
      }

#pragma unroll
      for (int j = 0; j + rdma_rank_begin_idx < rdma_rank_end_idx; ++j)
        num_tokens_per_rdma_rank_per_thread[thread_id][j] += (is_in_rdma_rank[j] > 0);
    }
    __syncthreads();

    // Sum up
    EP_STATIC_ASSERT(kNumRanksPerSM <= kNumThreads, "Too many ranks per SM");
    if (rank_begin_idx + thread_id < rank_end_idx) {
      int sum = 0;
#pragma unroll
      for (int i = 0; i < kNumThreads; ++i) sum += num_tokens_per_rank_per_thread[i][thread_id];
      num_tokens_per_rank[rank_begin_idx + thread_id] = sum;
    }

    if (num_tokens_per_rdma_rank != nullptr and rdma_rank_begin_idx + thread_id < rdma_rank_end_idx) {
      int sum = 0;
#pragma unroll
      for (int i = 0; i < kNumThreads; ++i) sum += num_tokens_per_rdma_rank_per_thread[i][thread_id];
      num_tokens_per_rdma_rank[rdma_rank_begin_idx + thread_id] = sum;
    }
  }
}

void get_dispatch_layout(const int64_t* topk_idx, int* num_tokens_per_rank, int* num_tokens_per_rdma_rank,
                         int* num_tokens_per_expert, bool* is_token_in_rank, int num_tokens, int num_topk,
                         int num_ranks, int num_experts, cudaStream_t stream) {
  constexpr int kNumThreads = 256, kNumExpertsPerSM = 32, kNumRanksPerSM = 8;
  int num_sms =
      ((num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM) + (num_ranks + kNumRanksPerSM - 1) / kNumRanksPerSM;
  EP_STATIC_ASSERT(kNumExpertsPerSM % NUM_MAX_NVL_PEERS == 0, "Invalid number of experts per SM");

  SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
  LAUNCH_KERNEL(&cfg, (get_dispatch_layout<kNumThreads, kNumExpertsPerSM, kNumRanksPerSM>), topk_idx,
                num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, num_tokens,
                num_topk, num_ranks, num_experts);
}

struct SourceMeta {
  int src_rdma_rank, is_token_in_nvl_rank_bits;

  EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS == 8 || NUM_MAX_NVL_PEERS == 4, "Invalid number of maximum NVL peers");

  __forceinline__ SourceMeta() = default;

  // TODO: faster encoding
  __device__ __forceinline__ SourceMeta(int rdma_rank, const bool* is_token_in_nvl_ranks) {
    src_rdma_rank = rdma_rank;
    is_token_in_nvl_rank_bits = is_token_in_nvl_ranks[0];
#pragma unroll
    for (int i = 1; i < NUM_MAX_NVL_PEERS; ++i) is_token_in_nvl_rank_bits |= is_token_in_nvl_ranks[i] << i;
  }

  __device__ __forceinline__ bool is_token_in_nvl_rank(int nvl_rank) const {
    return (is_token_in_nvl_rank_bits >> nvl_rank) & 1;
  }
};

EP_STATIC_ASSERT(sizeof(SourceMeta) % sizeof(int) == 0, "Invalid size of `SourceMeta`");

int get_source_meta_bytes() { return sizeof(SourceMeta); }

// Increment 6 (kEpFlat): post-dispatch metadata drain kernel. Under the flat path
// the sender wrote each token's metadata (SourceMeta + scales + topk_idx/weights,
// topk RAW) into the destination pool's META region at the token's final recv slot.
// This kernel runs on comm_stream right after the dispatch kernel (which, with the
// receiver still present, guarantees all senders' pool writes are visible by the
// time it exits), and copies the pool meta into the recv_* output tensors with the
// topk expert-range rebase the receiver would have done. One thread per recv token.
__global__ void flat_meta_drain_kernel(const uint8_t* __restrict__ pool_base, int64_t meta_base, int num_recv_tokens,
                                       SourceMeta* __restrict__ recv_src_meta, float* __restrict__ recv_x_scales,
                                       int64_t* __restrict__ recv_topk_idx, float* __restrict__ recv_topk_weights,
                                       int num_scales, int num_topk, int expert_begin, int expert_end,
                                       int64_t meta_slot_bytes) {
  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < num_recv_tokens; t += gridDim.x * blockDim.x) {
    const uint8_t* m = pool_base + meta_base + static_cast<int64_t>(t) * meta_slot_bytes;
    if (recv_src_meta != nullptr) recv_src_meta[t] = *reinterpret_cast<const SourceMeta*>(m);
    const float* sdat = reinterpret_cast<const float*>(m + sizeof(SourceMeta));
    if (recv_x_scales != nullptr)
      for (int s = 0; s < num_scales; ++s) recv_x_scales[static_cast<int64_t>(t) * num_scales + s] = sdat[s];
    const int* idat = reinterpret_cast<const int*>(m + sizeof(SourceMeta) + num_scales * sizeof(float));
    const float* wdat = reinterpret_cast<const float*>(idat + num_topk);
    if (recv_topk_idx != nullptr) {
      for (int k = 0; k < num_topk; ++k) {
        const int idx = idat[k];
        const int local = (idx >= expert_begin and idx < expert_end) ? idx - expert_begin : -1;
        recv_topk_idx[static_cast<int64_t>(t) * num_topk + k] = static_cast<int64_t>(local);
        recv_topk_weights[static_cast<int64_t>(t) * num_topk + k] = (local >= 0) ? wdat[k] : 0.0f;
      }
    }
  }
}

void flat_meta_drain(void* pool_base, int64_t meta_base, int num_recv_tokens, void* recv_src_meta, float* recv_x_scales,
                     int64_t* recv_topk_idx, float* recv_topk_weights, int num_scales, int num_topk, int num_experts,
                     int num_ranks, int rank, int64_t meta_slot_bytes, cudaStream_t stream) {
  if (num_recv_tokens <= 0) return;
  const int num_local_experts = num_experts / num_ranks;
  const int expert_begin = rank * num_local_experts;
  const int expert_end = expert_begin + num_local_experts;
  constexpr int kThreads = 256;
  const int kBlocks = (num_recv_tokens + kThreads - 1) / kThreads;
  flat_meta_drain_kernel<<<kBlocks, kThreads, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(pool_base), meta_base, num_recv_tokens,
      reinterpret_cast<SourceMeta*>(recv_src_meta), recv_x_scales, recv_topk_idx, recv_topk_weights, num_scales,
      num_topk, expert_begin, expert_end, meta_slot_bytes);
}

__host__ __device__ __forceinline__ int get_num_bytes_per_rdma_token(int hidden_int4, int num_scales, int num_topk_idx,
                                                                     int num_topk_weights) {
  return static_cast<int>(align(hidden_int4 * sizeof(int4) + sizeof(SourceMeta) + num_scales * sizeof(float) +
                                    num_topk_idx * sizeof(int) + num_topk_weights * sizeof(float),
                                sizeof(int4)));
}

__host__ __device__ __forceinline__ std::pair<int, int> get_rdma_clean_meta(int hidden_int4, int num_scales,
                                                                            int num_topk_idx, int num_topk_weights,
                                                                            int num_rdma_ranks,
                                                                            int num_rdma_recv_buffer_tokens,
                                                                            int num_sms) {
  // Return `int32_t` offset and count to clean
  return {(get_num_bytes_per_rdma_token(hidden_int4, num_scales, num_topk_idx, num_topk_weights) *
           num_rdma_recv_buffer_tokens * num_rdma_ranks * 2 * num_sms) /
              sizeof(int),
          (NUM_MAX_NVL_PEERS * 2 + 4) * num_rdma_ranks * 2 * num_sms};
}

__host__ __device__ __forceinline__ std::pair<int, int> get_nvl_clean_meta(int hidden_int4, int num_scales,
                                                                           int num_topk_idx, int num_topk_weights,
                                                                           int num_rdma_ranks, int num_nvl_ranks,
                                                                           int num_nvl_recv_buffer_tokens,
                                                                           int num_sms) {
  // Return `int32_t` offset and to clean
  EP_STATIC_ASSERT(sizeof(SourceMeta) % sizeof(int) == 0, "Invalid size of `SourceMeta`");
  return {
      (num_nvl_recv_buffer_tokens *
       (hidden_int4 * sizeof(int4) + num_scales * sizeof(float) + num_topk_idx * sizeof(int) +
        num_topk_weights * sizeof(float) + sizeof(SourceMeta)) *
       num_nvl_ranks * num_sms) /
          sizeof(int),
      num_nvl_ranks * (2 * num_rdma_ranks + 2) * num_sms,
  };
}

template <bool kLowLatencyMode, int kNumRDMARanks>
__global__ void notify_dispatch(
    const int* num_tokens_per_rank, int* moe_recv_counter_mapped, int num_ranks, const int* num_tokens_per_rdma_rank,
    int* moe_recv_rdma_counter_mapped, const int* num_tokens_per_expert, int* moe_recv_expert_counter_mapped,
    int num_experts, const bool* is_token_in_rank, int num_tokens, int num_channels, int expert_alignment,
    const int rdma_clean_offset, const int rdma_num_int_clean, const int nvl_clean_offset, const int nvl_num_int_clean,
    int* rdma_channel_prefix_matrix, int* recv_rdma_rank_prefix_sum, int* gbl_channel_prefix_matrix,
    int* recv_gbl_rank_prefix_sum, void* rdma_buffer_ptr, void** buffer_ptrs, int** task_fifo_ptrs, int head, int rank,
    mscclpp::PortChannelDeviceHandle* port_channel_handles, mscclpp::MemoryChannelDeviceHandle* memory_channel_handles,
    // NVLS Phase 2 — replaces port_channel signal/wait + putWithSignal.
    // When `nvls_mc_ptr == nullptr` the legacy PortChannel path runs
    // unchanged (fallback for non-NVLS IB platforms).
    void* nvls_mc_ptr, void* nvls_dev_ptr, size_t nvls_off_barrier, size_t nvls_off_data, uint64_t nvls_epoch,
    int nvls_per_peer_bytes) {
  auto sm_id = static_cast<int>(blockIdx.x);
  auto thread_id = static_cast<int>(threadIdx.x), warp_id = thread_id / 32, lane_id = get_lane_id();
  auto num_threads = static_cast<int>(blockDim.x), num_warps = num_threads / 32;

  auto rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
  auto num_rdma_experts = num_experts / kNumRDMARanks, num_nvl_experts = num_rdma_experts / NUM_MAX_NVL_PEERS;

  if (sm_id == 0) {
    // Communication with others
    // Global barrier: the first warp do intra-node sync, the second warp do internode sync
    EP_DEVICE_ASSERT(num_warps > 1);
    EP_DEVICE_ASSERT(kNumRDMARanks + 32 <= num_threads);
    const auto barrier_thread_id = thread_id - 32;
    const bool run_barrier =
        (barrier_thread_id >= 0) && (barrier_thread_id < kNumRDMARanks) && (barrier_thread_id != rdma_rank);
    const auto barrier_channel_idx =
        kLowLatencyMode ? barrier_thread_id : (barrier_thread_id * NUM_MAX_NVL_PEERS + nvl_rank);
    if (nvls_mc_ptr != nullptr) {
      // NVLS epoch barrier #0: every rank in the multicast group does
      // `multimem.red.add 1` to slot[0] (over the NVL72 fabric — bypasses
      // broken IB atomics on Azure CX-7 RoCE), then spins on its local
      // mapping until the slot reaches `epoch * num_ranks` (everyone
      // arrived). Replaces the pairwise PortChannel signal/wait above.
      if (thread_id == 0) {
        if (rank == 0) (void)0;
        uint64_t* mc_b0 = reinterpret_cast<uint64_t*>(static_cast<char*>(nvls_mc_ptr) + nvls_off_barrier);
        uint64_t* dev_b0 = reinterpret_cast<uint64_t*>(static_cast<char*>(nvls_dev_ptr) + nvls_off_barrier);
        uint64_t pre;
        asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(pre) : "l"(dev_b0) : "memory");
        MSCCLPP_EP_MULTIMEM_RED_ADD_RELEASE_U64_ONE(mc_b0);
        (void)0;
        const uint64_t expected = nvls_epoch * static_cast<uint64_t>(num_ranks);
        uint64_t v;
        int spin = 0;
        do {
          asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(dev_b0) : "memory");
          if ((++spin & 0xFFFFFFF) == 0) {
            (void)0;
          }
        } while (v < expected);
        if (rank == 0) (void)0;
      }
    } else if (run_barrier) {
      port_channel_handles[barrier_channel_idx].signal();
      port_channel_handles[barrier_channel_idx].wait();
    }
    if constexpr (!kLowLatencyMode) {
      // kLowLatencyMode==false requires sync of all ranks, which can be done by running intra-node sync
      // after the inter-node sync is done.
      __syncthreads();
    }
#if 1
    barrier_device<NUM_MAX_NVL_PEERS>(task_fifo_ptrs, head, nvl_rank);
    move_fifo_slots<NUM_MAX_NVL_PEERS>(head);
#else
    // TODO(chhwang): make memory channels work
    if (thread_id < NUM_MAX_NVL_PEERS && thread_id != nvl_rank) {
      memory_channel_handles[thread_id].relaxedSignal();
      memory_channel_handles[thread_id].relaxedWait();
    }
#endif
    __syncthreads();

    // Send numbers of tokens per rank/expert to RDMA ranks
    auto rdma_buffer_ptr_int = reinterpret_cast<int*>(rdma_buffer_ptr);
    auto num_elems = NUM_MAX_NVL_PEERS + num_rdma_experts + 1;
    auto num_bytes = num_elems * sizeof(int);
    auto per_channel_bytes = num_bytes * kNumRDMARanks;
    auto rdma_recv_num_tokens_mixed = SymBuffer<int>(rdma_buffer_ptr, num_elems, kNumRDMARanks);

    // Clean up for later data dispatch
    EP_DEVICE_ASSERT(rdma_recv_num_tokens_mixed.total_bytes <= rdma_clean_offset * sizeof(int));
#pragma unroll
    for (int i = thread_id; i < rdma_num_int_clean; i += num_threads) rdma_buffer_ptr_int[rdma_clean_offset + i] = 0;

// Copy to send buffer
#pragma unroll
    for (int i = thread_id; i < num_ranks; i += num_threads)
      rdma_recv_num_tokens_mixed.send_buffer(i / NUM_MAX_NVL_PEERS)[i % NUM_MAX_NVL_PEERS] = num_tokens_per_rank[i];
#pragma unroll
    for (int i = thread_id; i < num_experts; i += num_threads)
      rdma_recv_num_tokens_mixed.send_buffer(i / num_rdma_experts)[NUM_MAX_NVL_PEERS + i % num_rdma_experts] =
          num_tokens_per_expert[i];
    if (thread_id < kNumRDMARanks)
      rdma_recv_num_tokens_mixed.send_buffer(thread_id)[NUM_MAX_NVL_PEERS + num_rdma_experts] =
          num_tokens_per_rdma_rank[thread_id];
    __syncthreads();

    // Issue send
    // TODO: more light fence or barrier or signaling
    // TODO: overlap EP barrier and NVL cleaning
    if (nvls_mc_ptr != nullptr) {
      // NVLS Phase 2 data path: every rank broadcasts its full send blob
      // (kNumRDMARanks × num_bytes contiguous in send_buffer(0..)) into
      // its own slot in the multicast data region using `multimem.st.u32`.
      // After a second NVLS epoch barrier, every receiver copies the
      // sub-block destined to it into the legacy `recv_buffer(s)` location
      // so the rest of the kernel reads the same layout as before.
      EP_DEVICE_ASSERT(num_bytes <= static_cast<size_t>(nvls_per_peer_bytes));
      const int my_global_rank = kLowLatencyMode ? rdma_rank : (rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank);
      const size_t slot_stride_bytes =
          static_cast<size_t>(NUM_MAX_RDMA_PEERS) * static_cast<size_t>(nvls_per_peer_bytes);
      // Sender: write our blob (num_elems × kNumRDMARanks ints) to our slot.
      // SymBuffer<int>::send_buffer(i) lives at base + i*num_elems ints in
      // the local send region — contiguous from send_buffer(0).
      int* src_ints = rdma_recv_num_tokens_mixed.send_buffer(0);
      const int total_ints = num_elems * kNumRDMARanks;
      int* mc_slot = reinterpret_cast<int*>(static_cast<char*>(nvls_mc_ptr) + nvls_off_data +
                                            static_cast<size_t>(my_global_rank) * slot_stride_bytes);
      for (int i = thread_id; i < total_ints; i += num_threads) {
        int val = src_ints[i];
        MSCCLPP_EP_MULTIMEM_ST_RELAXED_U32(mc_slot + i, val);
      }
      __syncthreads();
      // NVLS epoch barrier #1: ensure every sender's multimem.st has been
      // delivered to all peers before we read.
      if (thread_id == 0) {
        if (rank == 0) (void)0;
        uint64_t* mc_b1 = reinterpret_cast<uint64_t*>(static_cast<char*>(nvls_mc_ptr) + nvls_off_barrier + 8);
        uint64_t* dev_b1 = reinterpret_cast<uint64_t*>(static_cast<char*>(nvls_dev_ptr) + nvls_off_barrier + 8);
        MSCCLPP_EP_MULTIMEM_RED_ADD_RELEASE_U64_ONE(mc_b1);
        const uint64_t expected = nvls_epoch * static_cast<uint64_t>(num_ranks);
        uint64_t v;
        do {
          asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(dev_b1) : "memory");
        } while (v < expected);
        if (rank == 0) (void)0;
      }
      __syncthreads();
      // Receiver: for each sender s, copy num_elems ints from the NVLS slot
      // (sub-block at offset rdma_rank * num_bytes within the slot) into
      // the legacy SymBuffer recv_buffer(s) location.
      for (int s = 0; s < kNumRDMARanks; ++s) {
        const int sender_global = kLowLatencyMode ? s : (s * NUM_MAX_NVL_PEERS + nvl_rank);
        const int* nvls_src = reinterpret_cast<const int*>(static_cast<char*>(nvls_dev_ptr) + nvls_off_data +
                                                           static_cast<size_t>(sender_global) * slot_stride_bytes +
                                                           static_cast<size_t>(rdma_rank) * num_bytes);
        int* dst = rdma_recv_num_tokens_mixed.recv_buffer(s);
        for (int i = thread_id; i < num_elems; i += num_threads) {
          dst[i] = nvls_src[i];
        }
      }
    } else if (thread_id < kNumRDMARanks) {
      auto dst_offset = rdma_rank * num_bytes + per_channel_bytes;
      auto src_offset = thread_id * num_bytes;
      auto peer_rank = kLowLatencyMode ? thread_id : (thread_id * NUM_MAX_NVL_PEERS + nvl_rank);
      port_channel_handles[peer_rank].putWithSignal(dst_offset, src_offset, num_bytes);
      port_channel_handles[peer_rank].wait();
    }
    __syncthreads();

    // NVL buffers
    auto nvl_send_buffer = thread_id < NUM_MAX_NVL_PEERS ? buffer_ptrs[thread_id] : nullptr;
    auto nvl_recv_buffer = buffer_ptrs[nvl_rank];
    auto nvl_reduced_num_tokens_per_expert =
        Buffer<int>(nvl_recv_buffer, num_rdma_experts).advance_also(nvl_send_buffer);
    auto nvl_send_num_tokens_per_rank = AsymBuffer<int>(nvl_send_buffer, kNumRDMARanks, NUM_MAX_NVL_PEERS);
    auto nvl_send_num_tokens_per_expert = AsymBuffer<int>(nvl_send_buffer, num_nvl_experts, NUM_MAX_NVL_PEERS);
    auto nvl_recv_num_tokens_per_rank = AsymBuffer<int>(nvl_recv_buffer, kNumRDMARanks, NUM_MAX_NVL_PEERS);
    auto nvl_recv_num_tokens_per_expert = AsymBuffer<int>(nvl_recv_buffer, num_nvl_experts, NUM_MAX_NVL_PEERS);

    // Clean up for later data dispatch
    auto nvl_buffer_ptr_int = reinterpret_cast<int*>(buffer_ptrs[nvl_rank]);
    EP_DEVICE_ASSERT(nvl_reduced_num_tokens_per_expert.total_bytes + nvl_send_num_tokens_per_rank.total_bytes +
                         nvl_send_num_tokens_per_expert.total_bytes <=
                     nvl_clean_offset * sizeof(int));
#pragma unroll
    for (int i = thread_id; i < nvl_num_int_clean; i += num_threads) nvl_buffer_ptr_int[nvl_clean_offset + i] = 0;

    // Reduce number of tokens per expert into the NVL send buffer
    // TODO: may use NVSHMEM reduction
    EP_DEVICE_ASSERT(num_rdma_experts <= num_threads);
    if (thread_id < num_rdma_experts) {
      int sum = 0;
#pragma unroll
      for (int i = 0; i < kNumRDMARanks; ++i)
        sum += rdma_recv_num_tokens_mixed.recv_buffer(i)[NUM_MAX_NVL_PEERS + thread_id];
      nvl_reduced_num_tokens_per_expert[thread_id] = sum;
    }
    __syncthreads();

    // Reduce RDMA received tokens
    if (thread_id == 0) {
      int sum = 0;
#pragma unroll
      for (int i = 0; i < kNumRDMARanks; ++i) {
        sum += rdma_recv_num_tokens_mixed.recv_buffer(i)[NUM_MAX_NVL_PEERS + num_rdma_experts];
        recv_rdma_rank_prefix_sum[i] = sum;
      }
      while (ld_volatile_global(moe_recv_rdma_counter_mapped) != -1)
        ;
      *moe_recv_rdma_counter_mapped = sum;
    }

    // Send numbers of tokens per rank/expert to NVL ranks
    EP_DEVICE_ASSERT(NUM_MAX_NVL_PEERS <= num_threads);
    if (thread_id < NUM_MAX_NVL_PEERS) {
#pragma unroll
      for (int i = 0; i < kNumRDMARanks; ++i)
        nvl_send_num_tokens_per_rank.buffer(nvl_rank)[i] = rdma_recv_num_tokens_mixed.recv_buffer(i)[thread_id];
#pragma unroll
      for (int i = 0; i < num_nvl_experts; ++i)
        nvl_send_num_tokens_per_expert.buffer(nvl_rank)[i] =
            nvl_reduced_num_tokens_per_expert[thread_id * num_nvl_experts + i];
    }
    memory_fence();
    __syncthreads();
    barrier_device<NUM_MAX_NVL_PEERS>(task_fifo_ptrs, head, nvl_rank);
    move_fifo_slots<NUM_MAX_NVL_PEERS>(head);
    __syncthreads();

    // Reduce number of tokens per rank/expert
    EP_DEVICE_ASSERT(num_nvl_experts <= num_threads);
    if (thread_id == 0) {
      int sum = 0;
#pragma unroll
      for (int i = 0; i < num_ranks; ++i) {
        int src_rdma_rank = i / NUM_MAX_NVL_PEERS, src_nvl_rank = i % NUM_MAX_NVL_PEERS;
        sum += nvl_recv_num_tokens_per_rank.buffer(src_nvl_rank)[src_rdma_rank];
        recv_gbl_rank_prefix_sum[i] = sum;
      }
      while (ld_volatile_global(moe_recv_counter_mapped) != -1)
        ;
      *moe_recv_counter_mapped = sum;
    }
    if (thread_id < num_nvl_experts) {
      int sum = 0;
#pragma unroll
      for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i) sum += nvl_recv_num_tokens_per_expert.buffer(i)[thread_id];
      sum = (sum + expert_alignment - 1) / expert_alignment * expert_alignment;
      while (ld_volatile_global(moe_recv_expert_counter_mapped + thread_id) != -1)
        ;
      moe_recv_expert_counter_mapped[thread_id] = sum;
    }

    // Finally barrier
    __syncthreads();
    if (thread_id == 0) (void)0;

    if (nvls_mc_ptr != nullptr) {
      // Phase 4 / NVLS HT "B2": replace the cross-node port_channel
      // signal/wait pair with an epoch-monotonic NVLS multimem barrier
      // (slot at +16). Bypasses the broken IB signal/wait path on Azure
      // CX-7 RoCE that hangs the entire CTA at the !kLowLatencyMode
      // __syncthreads below (thread 33's wait never completes).
      if (thread_id == 0) {
        if (rank == 0) (void)0;
        uint64_t* mc_b2 = reinterpret_cast<uint64_t*>(static_cast<char*>(nvls_mc_ptr) + nvls_off_barrier + 16);
        uint64_t* dev_b2 = reinterpret_cast<uint64_t*>(static_cast<char*>(nvls_dev_ptr) + nvls_off_barrier + 16);
        MSCCLPP_EP_MULTIMEM_RED_ADD_RELEASE_U64_ONE(mc_b2);
        const uint64_t expected = nvls_epoch * static_cast<uint64_t>(num_ranks);
        uint64_t v;
        do {
          asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(dev_b2) : "memory");
        } while (v < expected);
        if (rank == 0) (void)0;
      }
    } else if (run_barrier) {
      port_channel_handles[barrier_channel_idx].signal();
      port_channel_handles[barrier_channel_idx].wait();
    }
    if (thread_id == 0) (void)0;
    if constexpr (!kLowLatencyMode) {
      // kLowLatencyMode==false requires sync of all ranks, which can be done by running intra-node sync
      // after the inter-node sync is done.
      __syncthreads();
    }
    barrier_device<NUM_MAX_NVL_PEERS>(task_fifo_ptrs, head, nvl_rank);
    move_fifo_slots<NUM_MAX_NVL_PEERS>(head);
    if (thread_id == 0) (void)0;
  } else {
    // Calculate meta data
    int dst_rdma_rank = sm_id - 1;
    for (int channel_id = warp_id; channel_id < num_channels; channel_id += num_warps) {
      int token_start_idx, token_end_idx;
      get_channel_task_range(num_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

      // Iterate over tokens
      int total_count = 0, per_nvl_rank_count[NUM_MAX_NVL_PEERS] = {0};
      for (int64_t i = token_start_idx + lane_id; i < token_end_idx; i += 32) {
        auto is_token_in_rank_packed =
            *reinterpret_cast<const NvlPackT*>(is_token_in_rank + i * num_ranks + dst_rdma_rank * NUM_MAX_NVL_PEERS);
        auto is_token_in_rank_values = reinterpret_cast<const bool*>(&is_token_in_rank_packed);
#pragma unroll
        for (int j = 0; j < NUM_MAX_NVL_PEERS; ++j) per_nvl_rank_count[j] += is_token_in_rank_values[j];
        total_count += (is_token_in_rank_packed != 0);
      }

      // Warp reduce
      total_count = warp_reduce_sum(total_count);
#pragma unroll
      for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i) per_nvl_rank_count[i] = warp_reduce_sum(per_nvl_rank_count[i]);

      // Write into channel matrix
      if (lane_id == 0) {
#pragma unroll
        for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i)
          gbl_channel_prefix_matrix[(dst_rdma_rank * NUM_MAX_NVL_PEERS + i) * num_channels + channel_id] =
              per_nvl_rank_count[i];
        rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id] = total_count;
      }
    }

    // Calculate prefix sum
    __syncthreads();
    if (thread_id == 0) {
      auto prefix_row = rdma_channel_prefix_matrix + dst_rdma_rank * num_channels;
#pragma unroll
      for (int i = 1; i < num_channels; ++i) prefix_row[i] += prefix_row[i - 1];
    }

    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= 32, "Invalid number of NVL peers");
    if (thread_id < NUM_MAX_NVL_PEERS) {
      auto prefix_row = gbl_channel_prefix_matrix + (dst_rdma_rank * NUM_MAX_NVL_PEERS + thread_id) * num_channels;
#pragma unroll
      for (int i = 1; i < num_channels; ++i) prefix_row[i] += prefix_row[i - 1];
    }
  }
}

void notify_dispatch(const int* num_tokens_per_rank, int* moe_recv_counter_mapped, int num_ranks,
                     const int* num_tokens_per_rdma_rank, int* moe_recv_rdma_counter_mapped,
                     const int* num_tokens_per_expert, int* moe_recv_expert_counter_mapped, int num_experts,
                     const bool* is_token_in_rank, int num_tokens, int num_channels, int hidden_int4, int num_scales,
                     int num_topk, int expert_alignment, int* rdma_channel_prefix_matrix,
                     int* recv_rdma_rank_prefix_sum, int* gbl_channel_prefix_matrix, int* recv_gbl_rank_prefix_sum,
                     void* rdma_buffer_ptr, int num_max_rdma_chunked_recv_tokens, void** buffer_ptrs,
                     int num_max_nvl_chunked_recv_tokens, int** task_fifo_ptrs, int head, int rank, cudaStream_t stream,
                     int64_t num_rdma_bytes, int64_t num_nvl_bytes, bool low_latency_mode,
                     mscclpp::PortChannelDeviceHandle* port_channel_handles,
                     mscclpp::MemoryChannelDeviceHandle* memory_channel_handles, void* nvls_mc_ptr, void* nvls_dev_ptr,
                     size_t nvls_off_barrier, size_t nvls_off_data, uint64_t nvls_epoch, int nvls_per_peer_bytes) {
#define NOTIFY_DISPATCH_LAUNCH_CASE(num_rdma_ranks)                                                                  \
  {                                                                                                                  \
    auto notify_dispatch_func =                                                                                      \
        low_latency_mode ? notify_dispatch<true, num_rdma_ranks> : notify_dispatch<false, num_rdma_ranks>;           \
    LAUNCH_KERNEL(&cfg, notify_dispatch_func, num_tokens_per_rank, moe_recv_counter_mapped, num_ranks,               \
                  num_tokens_per_rdma_rank, moe_recv_rdma_counter_mapped, num_tokens_per_expert,                     \
                  moe_recv_expert_counter_mapped, num_experts, is_token_in_rank, num_tokens, num_channels,           \
                  expert_alignment, rdma_clean_meta.first, rdma_clean_meta.second, nvl_clean_meta.first,             \
                  nvl_clean_meta.second, rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum,                      \
                  gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum, rdma_buffer_ptr, buffer_ptrs, task_fifo_ptrs, \
                  head, rank, port_channel_handles, memory_channel_handles, nvls_mc_ptr, nvls_dev_ptr,               \
                  nvls_off_barrier, nvls_off_data, nvls_epoch, nvls_per_peer_bytes);                                 \
  }                                                                                                                  \
  break

  constexpr int kNumThreads = 512;
  const auto num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;

  // Get clean meta. inc5: under MSCCLPP_EP_DIRECT the rdma ring slot excludes
  // hidden (kEpDirect), so the clean region must match the SMALL slot the kernel
  // uses, else the cleaned head/tail region != the one the kernel reads.
  const bool ep_direct = []() { const char* e = std::getenv("MSCCLPP_EP_DIRECT"); return e && std::atoi(e) != 0; }();
  auto rdma_clean_meta = get_rdma_clean_meta(ep_direct ? 0 : hidden_int4, num_scales, num_topk, num_topk, num_rdma_ranks,
                                             num_max_rdma_chunked_recv_tokens, num_channels);
  auto nvl_clean_meta = get_nvl_clean_meta(hidden_int4, num_scales, num_topk, num_topk, num_rdma_ranks,
                                           NUM_MAX_NVL_PEERS, num_max_nvl_chunked_recv_tokens, num_channels);
  EP_HOST_ASSERT((rdma_clean_meta.first + rdma_clean_meta.second) * sizeof(int) <= num_rdma_bytes);
  EP_HOST_ASSERT((nvl_clean_meta.first + nvl_clean_meta.second) * sizeof(int) <= num_nvl_bytes);
  EP_HOST_ASSERT(num_rdma_bytes < std::numeric_limits<int>::max());
  EP_HOST_ASSERT(num_nvl_bytes < std::numeric_limits<int>::max());

  // Launch kernel
  SETUP_LAUNCH_CONFIG(1 + num_rdma_ranks, kNumThreads, stream);
  SWITCH_RDMA_RANKS(NOTIFY_DISPATCH_LAUNCH_CASE);
#undef NOTIFY_DISPATCH_LAUNCH_CASE
}

// At most 8 RDMA ranks to be sent
constexpr int get_num_topk_rdma_ranks(int num_rdma_ranks) { return num_rdma_ranks < 8 ? num_rdma_ranks : 8; }

template <bool kLowLatencyMode, int kNumRDMARanks, bool kCachedMode, int kNumDispatchRDMASenderWarps,
          int kNumTopkRDMARanks = get_num_topk_rdma_ranks(kNumRDMARanks)>
__global__ void __launch_bounds__(((kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS) * 32), 1)
    dispatch(int4* recv_x, float* recv_x_scales, int64_t* recv_topk_idx, float* recv_topk_weights,
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

#ifdef EP_DISPATCH_NCCLEP
#include "internode_ncclep.cuh"  // warp-specialized NCCL-EP-ported dispatch_ncclep<>
#define EP_DISPATCH_KERNEL dispatch_ncclep
#define EP_DISPATCH_EXTRA_ARGS , recv_pool_ptrs, recv_pool_global_ptrs, ep_combine_recv_idx
#else
#define EP_DISPATCH_KERNEL dispatch
#define EP_DISPATCH_EXTRA_ARGS
#endif

void dispatch(void* recv_x, float* recv_x_scales, int64_t* recv_topk_idx, float* recv_topk_weights, void* recv_src_meta,
              const void* x, const float* x_scales, const int64_t* topk_idx, const float* topk_weights,
              int* send_rdma_head, int* send_nvl_head, int* recv_rdma_channel_prefix_matrix,
              int* recv_gbl_channel_prefix_matrix, const int* rdma_channel_prefix_matrix,
              const int* recv_rdma_rank_prefix_sum, const int* gbl_channel_prefix_matrix,
              const int* recv_gbl_rank_prefix_sum, int num_tokens, int hidden_int4, int num_scales, int num_topk,
              int num_experts, const bool* is_token_in_rank, void* rdma_buffer_ptr,
              int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens, void** buffer_ptrs,
              int num_max_nvl_chunked_send_tokens, int num_max_nvl_chunked_recv_tokens, int rank, int num_ranks,
              bool is_cached_dispatch, cudaStream_t stream, int num_channels, bool low_latency_mode,
              mscclpp::PortChannelDeviceHandle* port_channel_handles,
              mscclpp::MemoryChannelDeviceHandle* memory_channel_handles, void* nvls_head_mc, void* nvls_head_dev,
              void* nvls_tail_mc, void* nvls_tail_dev, void* const* peer_rdma_bases, void* const* recv_pool_ptrs,
              void* const* recv_pool_global_ptrs, int* ep_combine_recv_idx) {
  constexpr int kNumDispatchRDMASenderWarps = 6;

#ifdef EP_DISPATCH_NCCLEP
  // Increment 5 (inc5): upload MSCCLPP_EP_DIRECT -> __constant__ kEpDirect once.
  {
    static bool s_done_direct = false;
    if (!s_done_direct) {
      const char* e = std::getenv("MSCCLPP_EP_DIRECT");
      int v = (e && std::atoi(e) != 0) ? 1 : 0;
      cudaMemcpyToSymbol(kEpDirect, &v, sizeof(int));
      s_done_direct = true;
    }
  }
  // Increment 6 (inc6): upload MSCCLPP_EP_FLAT -> __constant__ kEpFlat once.
  // Flat all-sender path requires kEpDirect; treated as off if direct is off.
  {
    static bool s_done_flat = false;
    if (!s_done_flat) {
      const char* ed = std::getenv("MSCCLPP_EP_DIRECT");
      const char* ef = std::getenv("MSCCLPP_EP_FLAT");
      int v = (ef && std::atoi(ef) != 0 && ed && std::atoi(ed) != 0) ? 1 : 0;
      cudaMemcpyToSymbol(kEpFlat, &v, sizeof(int));
      s_done_flat = true;
    }
  }
#endif

#define DISPATCH_LAUNCH_CASE(num_rdma_ranks)                                                                           \
  {                                                                                                                    \
    auto dispatch_func =                                                                                               \
        low_latency_mode ? (is_cached_dispatch ? EP_DISPATCH_KERNEL<true, num_rdma_ranks, true, kNumDispatchRDMASenderWarps>     \
                                               : EP_DISPATCH_KERNEL<true, num_rdma_ranks, false, kNumDispatchRDMASenderWarps>)   \
                         : (is_cached_dispatch ? EP_DISPATCH_KERNEL<false, num_rdma_ranks, true, kNumDispatchRDMASenderWarps>    \
                                               : EP_DISPATCH_KERNEL<false, num_rdma_ranks, false, kNumDispatchRDMASenderWarps>); \
    if (EP_NCCLEP_TMA) {                                                                                               \
      /* B-depth-3: opt in to the >48KB dynamic-shared half-token TMA send ring (EP_TMA_SND_*). */                     \
      const size_t ep_dyn_bytes =                                                                                      \
          (size_t)kNumDispatchRDMASenderWarps * EP_TMA_SND_NSTAGE * EP_TMA_SND_CHUNK_BYTES;                            \
      static bool s_ep_dyn_attr = false;                                                                               \
      if (!s_ep_dyn_attr) {                                                                                            \
        CUDA_CHECK(cudaFuncSetAttribute(dispatch_func, cudaFuncAttributeMaxDynamicSharedMemorySize,                    \
                                        static_cast<int>(ep_dyn_bytes)));                                              \
        s_ep_dyn_attr = true;                                                                                          \
      }                                                                                                                \
      cfg.dynamicSmemBytes = ep_dyn_bytes;                                                                             \
    }                                                                                                                  \
    LAUNCH_KERNEL(&cfg, dispatch_func, reinterpret_cast<int4*>(recv_x), recv_x_scales, recv_topk_idx,                  \
                  recv_topk_weights, reinterpret_cast<SourceMeta*>(recv_src_meta), reinterpret_cast<const int4*>(x),   \
                  x_scales, topk_idx, topk_weights, send_rdma_head, send_nvl_head, recv_rdma_channel_prefix_matrix,    \
                  recv_gbl_channel_prefix_matrix, rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum,               \
                  gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum, num_tokens, hidden_int4, num_scales, num_topk,  \
                  num_experts, is_token_in_rank, rdma_buffer_ptr, num_max_rdma_chunked_send_tokens,                    \
                  num_max_rdma_chunked_recv_tokens, buffer_ptrs, num_max_nvl_chunked_send_tokens,                      \
                  num_max_nvl_chunked_recv_tokens, rank, num_ranks, port_channel_handles, memory_channel_handles,      \
                  nvls_head_mc, nvls_head_dev, nvls_tail_mc, nvls_tail_dev, peer_rdma_bases EP_DISPATCH_EXTRA_ARGS);                          \
  }                                                                                                                    \
  break

  EP_HOST_ASSERT((topk_idx == nullptr) == (topk_weights == nullptr));
  EP_HOST_ASSERT((recv_topk_idx == nullptr) == (recv_topk_weights == nullptr));

  // inc6 (kEpFlat): the FLAT all-sender path launches 1 block per channel (no
  // forwarder blocks); the default inc5 2-hop path launches 2 blocks per channel.
  int ep_grid_blocks = num_channels * 2;
#ifdef EP_DISPATCH_NCCLEP
  {
    const char* ed = std::getenv("MSCCLPP_EP_DIRECT");
    const char* ef = std::getenv("MSCCLPP_EP_FLAT");
    if (ef && std::atoi(ef) != 0 && ed && std::atoi(ed) != 0) ep_grid_blocks = num_channels;
  }
#endif
  SETUP_LAUNCH_CONFIG(ep_grid_blocks, (kNumDispatchRDMASenderWarps + 1 + NUM_MAX_NVL_PEERS) * 32, stream);
  SWITCH_RDMA_RANKS(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

template <bool kLowLatencyMode>
__global__ void cached_notify(const int rdma_clean_offset, const int rdma_num_int_clean, const int nvl_clean_offset,
                              const int nvl_num_int_clean, int* combined_rdma_head, int num_combined_tokens,
                              int num_channels, const int* rdma_channel_prefix_matrix, const int* rdma_rank_prefix_sum,
                              int* combined_nvl_head, void* rdma_buffer_ptr, void** buffer_ptrs, int** task_fifo_ptrs,
                              int head, int rank, int num_ranks, bool is_cached_dispatch,
                              mscclpp::PortChannelDeviceHandle* port_channel_handles,
                              mscclpp::MemoryChannelDeviceHandle* memory_channel_handles,
                              // Phase 4: NVLS multimem barrier path — bypasses port_channel signal/wait
                              // (broken on Azure CX-7 RoCE) by using multimem.red.add + ld.acquire on a
                              // pair of barrier slots (offsets +24 and +32). Both nullptr ⇒ fall back to IB.
                              void* nvls_mc_ptr, void* nvls_dev_ptr, size_t nvls_off_barrier, uint64_t nvls_epoch) {
  auto sm_id = static_cast<int>(blockIdx.x);
  auto thread_id = static_cast<int>(threadIdx.x);
  auto num_threads = static_cast<int>(blockDim.x);
  auto num_warps = num_threads / 32;
  auto warp_id = thread_id / 32;
  auto lane_id = get_lane_id();

  auto rdma_rank = rank / NUM_MAX_NVL_PEERS;
  auto nvl_rank = rank % NUM_MAX_NVL_PEERS;
  auto num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;

  // Using two SMs, which clean the RDMA/NVL buffer respectively
  if (sm_id == 0) {
    if (thread_id == 0) {
      (void)0;
    }
    // Barrier for RDMA — Phase 4: NVLS multimem.red.add fast path replaces
    // the port_channel signal/wait pair (broken on Azure CX-7).
    if (nvls_mc_ptr != nullptr) {
      if (thread_id == 0) {
        uint64_t* mc_b3 = reinterpret_cast<uint64_t*>(static_cast<char*>(nvls_mc_ptr) + nvls_off_barrier + 24);
        uint64_t* dev_b3 = reinterpret_cast<uint64_t*>(static_cast<char*>(nvls_dev_ptr) + nvls_off_barrier + 24);
        MSCCLPP_EP_MULTIMEM_RED_ADD_RELEASE_U64_ONE(mc_b3);
        const uint64_t expected = nvls_epoch * static_cast<uint64_t>(num_ranks);
        uint64_t v;
        do {
          asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(dev_b3) : "memory");
        } while (v < expected);
      }
      __syncthreads();
    } else {
      // TODO(chhwang): it should be a global barrier when kLowLatencyMode is false
      const bool run_barrier = (threadIdx.x < num_rdma_ranks) && (threadIdx.x != rdma_rank);
      const auto barrier_channel_idx = kLowLatencyMode ? threadIdx.x : (threadIdx.x * NUM_MAX_NVL_PEERS + nvl_rank);
      if (run_barrier) {
        port_channel_handles[barrier_channel_idx].signal();
        port_channel_handles[barrier_channel_idx].wait();
      }
      __syncthreads();
    }

    // Clean
    auto rdma_buffer_ptr_int = reinterpret_cast<int*>(rdma_buffer_ptr);
#pragma unroll
    for (int i = thread_id; i < rdma_num_int_clean; i += num_threads) rdma_buffer_ptr_int[rdma_clean_offset + i] = 0;
    // Make the cleanup visible to the proxy + remote peers before the barrier.
    // DeepEP used `nvshmem_fence()` here; we fall back to a system-scope
    // threadfence because the actual remote visibility is provided by the
    // subsequent port-channel barrier (signal + flush + wait).
    __threadfence_system();
    __syncthreads();

    // Barrier again — Phase 4: second NVLS multimem.red barrier on slot +32.
    if (nvls_mc_ptr != nullptr) {
      if (thread_id == 0) {
        uint64_t* mc_b4 = reinterpret_cast<uint64_t*>(static_cast<char*>(nvls_mc_ptr) + nvls_off_barrier + 32);
        uint64_t* dev_b4 = reinterpret_cast<uint64_t*>(static_cast<char*>(nvls_dev_ptr) + nvls_off_barrier + 32);
        MSCCLPP_EP_MULTIMEM_RED_ADD_RELEASE_U64_ONE(mc_b4);
        const uint64_t expected = nvls_epoch * static_cast<uint64_t>(num_ranks);
        uint64_t v;
        do {
          asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(dev_b4) : "memory");
        } while (v < expected);
      }
      __syncthreads();
    } else {
      const bool run_barrier = (threadIdx.x < num_rdma_ranks) && (threadIdx.x != rdma_rank);
      const auto barrier_channel_idx = kLowLatencyMode ? threadIdx.x : (threadIdx.x * NUM_MAX_NVL_PEERS + nvl_rank);
      if (run_barrier) {
        port_channel_handles[barrier_channel_idx].signal();
        port_channel_handles[barrier_channel_idx].flush();
        port_channel_handles[barrier_channel_idx].wait();
      }
    }
  } else if (sm_id == 1) {
    // Barrier for NVL
    barrier_device<NUM_MAX_NVL_PEERS>(task_fifo_ptrs, head, nvl_rank);
    move_fifo_slots<NUM_MAX_NVL_PEERS>(head);
    __syncthreads();

    // Clean
    auto nvl_buffer_ptr_int = reinterpret_cast<int*>(buffer_ptrs[nvl_rank]);
#pragma unroll
    for (int i = thread_id; i < nvl_num_int_clean; i += num_threads) nvl_buffer_ptr_int[nvl_clean_offset + i] = 0;
    memory_fence();
    __syncthreads();

    // Barrier again
    barrier_device<NUM_MAX_NVL_PEERS>(task_fifo_ptrs, head, nvl_rank);
    move_fifo_slots<NUM_MAX_NVL_PEERS>(head);
  } else if (sm_id == 2) {
    if (is_cached_dispatch) return;

    EP_DEVICE_ASSERT(num_rdma_ranks <= 32);

    // Iterate in reverse order. Stride over channels in warp granularity so we
    // support num_channels > num_warps (per-block thread cap on GB200 is 1024).
    if (lane_id < num_rdma_ranks) {
      for (int ch = warp_id; ch < num_channels; ch += num_warps) {
        int token_start_idx, token_end_idx;
        get_channel_task_range(num_combined_tokens, num_channels, ch, token_start_idx, token_end_idx);

        // NOTES: `1 << 25` is a heuristic large number
        int last_head = 1 << 25;
        for (int token_idx = token_end_idx - 1; token_idx >= token_start_idx; --token_idx) {
          auto current_head = __ldg(combined_rdma_head + token_idx * num_rdma_ranks + lane_id);
          if (current_head < 0) {
            combined_rdma_head[token_idx * num_rdma_ranks + lane_id] = -last_head - 1;
          } else {
            last_head = current_head;
          }
        }
      }
    }
  } else {
    if (is_cached_dispatch) return;

    EP_DEVICE_ASSERT(rdma_channel_prefix_matrix != nullptr and rdma_rank_prefix_sum != nullptr);
    EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= 32, "Too many NVL peers");

    // Stride over channels per warp so num_channels > num_warps works.
    if (lane_id < NUM_MAX_NVL_PEERS) {
      for (int dst_rdma_rank = sm_id - 3; dst_rdma_rank < num_rdma_ranks; dst_rdma_rank += num_channels * 2 - 3) {
        for (int ch = warp_id; ch < num_channels; ch += num_warps) {
          // Iterate in reverse order
          int token_start_idx = ch == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + ch - 1];
          int token_end_idx = rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + ch];
          int shift = dst_rdma_rank == 0 ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1];
          token_start_idx += shift, token_end_idx += shift;

          // NOTES: `1 << 25` is a heuristic large number
          int last_head = 1 << 25;
          for (int token_idx = token_end_idx - 1; token_idx >= token_start_idx; --token_idx) {
            auto current_head = __ldg(combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + lane_id);
            if (current_head < 0) {
              combined_nvl_head[token_idx * NUM_MAX_NVL_PEERS + lane_id] = -last_head - 1;
            } else {
              last_head = current_head;
            }
          }
        }
      }
    }
  }
}

void cached_notify(int hidden_int4, int num_scales, int num_topk_idx, int num_topk_weights, int num_ranks,
                   int num_channels, int num_combined_tokens, int* combined_rdma_head,
                   const int* rdma_channel_prefix_matrix, const int* rdma_rank_prefix_sum, int* combined_nvl_head,
                   void* rdma_buffer_ptr, int num_max_rdma_chunked_recv_tokens, void** buffer_ptrs,
                   int num_max_nvl_chunked_recv_tokens, int** task_fifo_ptrs, int head, int rank, cudaStream_t stream,
                   int64_t num_rdma_bytes, int64_t num_nvl_bytes, bool is_cached_dispatch, bool low_latency_mode,
                   mscclpp::PortChannelDeviceHandle* port_channel_handles,
                   mscclpp::MemoryChannelDeviceHandle* memory_channel_handles, void* nvls_mc_ptr, void* nvls_dev_ptr,
                   size_t nvls_off_barrier, uint64_t nvls_epoch) {
  const int num_threads = std::max(128, 32 * num_channels);
  const auto num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;

  // Get clean meta. inc5: only the DISPATCH ring is shrunk under
  // MSCCLPP_EP_DIRECT (hidden goes direct to the pool); the COMBINE ring still
  // carries hidden through the rdma channel, so its clean must stay full-slot.
  // is_cached_dispatch distinguishes the two callers (true=cached dispatch,
  // false=combine).
  const bool ep_direct = []() { const char* e = std::getenv("MSCCLPP_EP_DIRECT"); return e && std::atoi(e) != 0; }();
  const int clean_hidden_int4 = (ep_direct && is_cached_dispatch) ? 0 : hidden_int4;
  auto rdma_clean_meta = get_rdma_clean_meta(clean_hidden_int4, num_scales, num_topk_idx, num_topk_weights, num_rdma_ranks,
                                             num_max_rdma_chunked_recv_tokens, num_channels);
  auto nvl_clean_meta = get_nvl_clean_meta(hidden_int4, num_scales, num_topk_idx, num_topk_weights, num_rdma_ranks,
                                           NUM_MAX_NVL_PEERS, num_max_nvl_chunked_recv_tokens, num_channels);
  EP_HOST_ASSERT((rdma_clean_meta.first + rdma_clean_meta.second) * sizeof(int) <= num_rdma_bytes);
  EP_HOST_ASSERT((nvl_clean_meta.first + nvl_clean_meta.second) * sizeof(int) <= num_nvl_bytes);
  EP_HOST_ASSERT(num_rdma_bytes < std::numeric_limits<int>::max());
  EP_HOST_ASSERT(num_nvl_bytes < std::numeric_limits<int>::max());
  EP_HOST_ASSERT(num_channels * 2 > 3);

  // Launch kernel.
  // NOTE: cached_notify's sm_id>=2 work iterates `warp_id < num_channels`, so the
  // kernel originally requested 32*num_channels threads per block. On GB200 the
  // per-block thread limit is 1024 (= 32 warps), so nc>32 silently failed launch
  // (cudaLaunchKernelEx returns cudaErrorInvalidValue / "invalid argument").
  // Cap at 1024 here and add a strided fallback in the kernel (warps loop over
  // channels in stride-num_warps) so nc up to the SM-count is supported.
  auto cached_notify_func = low_latency_mode ? cached_notify<true> : cached_notify<false>;
  const int launch_threads = std::min(num_threads, 1024);
  SETUP_LAUNCH_CONFIG(num_channels * 2, launch_threads, stream);
  LAUNCH_KERNEL(&cfg, cached_notify_func, rdma_clean_meta.first, rdma_clean_meta.second, nvl_clean_meta.first,
                nvl_clean_meta.second, combined_rdma_head, num_combined_tokens, num_channels,
                rdma_channel_prefix_matrix, rdma_rank_prefix_sum, combined_nvl_head, rdma_buffer_ptr, buffer_ptrs,
                task_fifo_ptrs, head, rank, num_ranks, is_cached_dispatch, port_channel_handles, memory_channel_handles,
                nvls_mc_ptr, nvls_dev_ptr, nvls_off_barrier, nvls_epoch);
}

template <int kNumRanks, typename dtype_t, int kMaxNumRanks, typename ReceiveFn, typename ReceiveTWFn>
__device__ int combine_token(bool is_token_in_rank, int head_idx, int lane_id, int hidden_int4, int num_topk,
                             int4* combined_row, float* combined_topk_weights, int num_max_recv_tokens,
                             const ReceiveFn& recv_fn, const ReceiveTWFn& recv_tw_fn) {
  constexpr auto kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);

  // Broadcast current heads
  // Lane `i` holds the head of rank `i` and `is_token_in_rank`
  EP_STATIC_ASSERT(kMaxNumRanks <= 32, "Too many ranks");
  int num_topk_ranks = 0, topk_ranks[kMaxNumRanks], slot_indices[kMaxNumRanks];
#pragma unroll
  for (int i = 0; i < kNumRanks; ++i)
    if (__shfl_sync(0xffffffff, is_token_in_rank, i)) {
      slot_indices[num_topk_ranks] = __shfl_sync(0xffffffff, head_idx, i) % num_max_recv_tokens;
      topk_ranks[num_topk_ranks++] = i;
    }
  EP_DEVICE_ASSERT(num_topk_ranks <= kMaxNumRanks);

// Reduce data
#pragma unroll
  for (int i = lane_id; i < hidden_int4; i += 32) {
    // Read buffers
    // TODO: maybe too many registers here
    int4 recv_value_int4[kMaxNumRanks];
#pragma unroll
    for (int j = 0; j < num_topk_ranks; ++j) recv_value_int4[j] = recv_fn(topk_ranks[j], slot_indices[j], i);

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
    st_na_global(combined_row + i, out_int4);
  }

  // Reduce `topk_weights`
  if (lane_id < num_topk) {
    float value = 0;
#pragma unroll
    for (int i = 0; i < num_topk_ranks; ++i) value += recv_tw_fn(topk_ranks[i], slot_indices[i], lane_id);
    st_na_global(combined_topk_weights + lane_id, value);
  }

  // Return the minimum top-k rank
  return topk_ranks[0];
}

// inc8 (combine extract): the FLAT direct-gather combine path, extracted from the
// unified `combine` kernel into its own lean kernel. RATIONALE: the unified kernel
// carries the heavy 2-hop forwarder code, so it compiles to ~80 registers and is
// register-limited to 1 resident block/SM (25 warps). Under flat, only this gather
// runs (the forwarder is dead code) yet it inherits that 80-reg / 1-block ceiling,
// starving the NVLink pull-gather of the warp-level parallelism it needs to hide
// the (latency-, not bandwidth-bound) remote ld_nc_global reads. Extracting it lets
// the gather compile to far fewer registers, so a full 1024-thread block (32 warps)
// fits per SM at the same SM budget -> more warps in flight -> better latency hiding.
// The token-strided loop is grid/block-agnostic, so correctness is independent of
// the launch geometry. Logic is byte-for-byte the same reduction as the unified
// kernel's flat branch (combine_token for <=8 nodes, chunked-ballot for >8).
template <typename dtype_t, int kNumRDMARanks>
__global__ void __launch_bounds__(1024, 1)
    combine_flat_gather(int4* combined_x, float* combined_topk_weights, const bool* is_combined_token_in_rank,
                        int num_combined_tokens, int hidden, int num_topk, int num_ranks,
                        void* const* recv_pool_global_ptrs, const int* ep_combine_recv_idx) {
  const auto lane_id = get_lane_id();
  const auto hidden_int4 = hidden / static_cast<int>(sizeof(int4) / sizeof(dtype_t));
  constexpr int kEpNumRanks = kNumRDMARanks * NUM_MAX_NVL_PEERS;
  const int64_t ep_pool_header_bytes = ((static_cast<int64_t>(num_ranks) * sizeof(int) + 127) / 128) * 128;
  const int warp_id_flat = static_cast<int>(threadIdx.x) / 32;
  const int num_warps_per_block = static_cast<int>(blockDim.x) / 32;
  const int global_warp = static_cast<int>(blockIdx.x) * num_warps_per_block + warp_id_flat;
  const int total_warps = static_cast<int>(gridDim.x) * num_warps_per_block;
  auto recv_fn = [&](int r, int slot, int hi) -> int4 {
    return ld_nc_global(reinterpret_cast<const int4*>(reinterpret_cast<const uint8_t*>(recv_pool_global_ptrs[r]) +
                                                      ep_pool_header_bytes) +
                        static_cast<int64_t>(slot) * hidden_int4 + hi);
  };
  auto recv_tw_fn = [&](int, int, int) -> float { return 0.0f; };
  if constexpr (kEpNumRanks <= 32) {
    for (int t = global_warp; t < num_combined_tokens; t += total_warps) {
      const bool is_in =
          (lane_id < kEpNumRanks) and is_combined_token_in_rank[static_cast<int64_t>(t) * num_ranks + lane_id];
      const int recv_idx = is_in ? ep_combine_recv_idx[static_cast<int64_t>(t) * num_ranks + lane_id] : 0;
      combine_token<kEpNumRanks, dtype_t, 8>(is_in, recv_idx, lane_id, hidden_int4, num_topk,
                                             combined_x + static_cast<int64_t>(t) * hidden_int4,
                                             combined_topk_weights + static_cast<int64_t>(t) * num_topk, 1 << 30,
                                             recv_fn, recv_tw_fn);
    }
  } else {
    constexpr int kEpMaxContrib = 8;
    constexpr auto kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);
    for (int t = global_warp; t < num_combined_tokens; t += total_warps) {
      int topk_ranks[kEpMaxContrib], slot_indices[kEpMaxContrib], num_topk_ranks = 0;
      for (int base = 0; base < kEpNumRanks; base += 32) {
        const int r = base + lane_id;
        const bool is_in =
            (r < kEpNumRanks) and is_combined_token_in_rank[static_cast<int64_t>(t) * num_ranks + r];
        const int slot = is_in ? ep_combine_recv_idx[static_cast<int64_t>(t) * num_ranks + r] : 0;
        unsigned ballot = __ballot_sync(0xffffffffu, is_in);
        while (ballot != 0u) {
          const int l = __ffs(static_cast<int>(ballot)) - 1;
          if (num_topk_ranks < kEpMaxContrib) {
            topk_ranks[num_topk_ranks] = base + l;
            slot_indices[num_topk_ranks] = __shfl_sync(0xffffffffu, slot, l);
            ++num_topk_ranks;
          }
          ballot &= ballot - 1u;
        }
      }
      int4* combined_row = combined_x + static_cast<int64_t>(t) * hidden_int4;
#pragma unroll
      for (int i = lane_id; i < hidden_int4; i += 32) {
        int4 recv_value_int4[kEpMaxContrib];
#pragma unroll
        for (int j = 0; j < num_topk_ranks; ++j) recv_value_int4[j] = recv_fn(topk_ranks[j], slot_indices[j], i);
        float values[kDtypePerInt4] = {0};
#pragma unroll
        for (int j = 0; j < num_topk_ranks; ++j) {
          auto vd = reinterpret_cast<const dtype_t*>(&recv_value_int4[j]);
#pragma unroll
          for (int k = 0; k < kDtypePerInt4; ++k) values[k] += static_cast<float>(vd[k]);
        }
        int4 out_int4;
        auto out_dtypes = reinterpret_cast<dtype_t*>(&out_int4);
#pragma unroll
        for (int k = 0; k < kDtypePerInt4; ++k) out_dtypes[k] = static_cast<dtype_t>(values[k]);
        st_na_global(combined_row + i, out_int4);
      }
      if (lane_id < num_topk)
        st_na_global(combined_topk_weights + static_cast<int64_t>(t) * num_topk + lane_id, 0.0f);
    }
  }
}

// TMA-staged flat-combine gather. Stages each contributor's hidden-chunk from the remote
// NVLink recv pools into shared memory via cp.async.bulk (TMA), then reduces from SMEM and
// stores the combined token locally. This is the default flat direct-gather combine path.
//
// RATIONALE: the synchronous variant (combine_flat_gather) hides remote-NVLink read latency
// purely with memory-level parallelism + occupancy (per-thread int4[kMaxContrib] register
// batching x warps), which caps occupancy at ~50% and regresses once the grid has many
// blocks. Staging through TMA instead lets the async copy engine track many outstanding bulk
// loads at low register cost, so latency hiding no longer competes with occupancy -- it wins
// at every channel count (no crossover) and at every node scale. Topology is single-hop
// NVLink (one MNNVL LSA domain), so this is an apples-to-apples gather, not a 2-hop reduce.
//
// SHAPE: token-per-warp; the hidden row is split into kChunkInt4-sized chunks streamed through
// a kStages-deep software pipeline (prefetch kStages-1 chunks ahead). Per chunk, lane 0 posts
// one cp.async.bulk G2S per contributor into a per-warp SMEM tile; all lanes then reduce the
// chunk from SMEM. SMEM/block = kWarps * kStages * kMaxContrib * kChunkInt4 * 16 bytes
// (independent of hidden), so it requires the >48KB dynamic-shared opt-in (cudaFuncSetAttribute
// at launch). Defaults (chunk=64 int4 = 1KB descriptors, 12 warps, 2 stages) tuned on GB200
// sm_100 @ hidden=7168 bf16; overridable at compile time via -D for other shapes.
#ifndef EP_CMB_TMA_CHUNK_INT4
#define EP_CMB_TMA_CHUNK_INT4 64  // hidden chunk in int4 (1KB TMA descriptors; 896 int4 / 64 = 14 chunks @ hidden=7168 bf16)
#endif
#ifndef EP_CMB_TMA_WARPS
#define EP_CMB_TMA_WARPS 12  // warps (tokens) per block; SMEM = WARPS*STAGES*8*CHUNK_INT4*16 bytes
#endif
#ifndef EP_CMB_TMA_STAGES
#define EP_CMB_TMA_STAGES 2  // pipeline depth (outstanding chunks in flight)
#endif

template <typename dtype_t, int kNumRDMARanks>
__global__ void __launch_bounds__(EP_CMB_TMA_WARPS * 32, 1)
    combine_flat_gather_tma(int4* combined_x, float* combined_topk_weights,
                            const bool* is_combined_token_in_rank, int num_combined_tokens, int hidden, int num_topk,
                            int num_ranks, void* const* recv_pool_global_ptrs, const int* ep_combine_recv_idx) {
  constexpr int kMaxContrib = 8;
  constexpr int kChunkInt4 = EP_CMB_TMA_CHUNK_INT4;
  constexpr int kWarps = EP_CMB_TMA_WARPS;
  constexpr int kStages = EP_CMB_TMA_STAGES;  // pipeline depth (outstanding chunks in flight)
  constexpr int kChunkBytes = kChunkInt4 * static_cast<int>(sizeof(int4));
  constexpr auto kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);
  constexpr int kEpNumRanks = kNumRDMARanks * NUM_MAX_NVL_PEERS;

  const auto lane_id = get_lane_id();
  const int warp_id = static_cast<int>(threadIdx.x) / 32;
  const auto hidden_int4 = hidden / static_cast<int>(sizeof(int4) / sizeof(dtype_t));
  const int64_t ep_pool_header_bytes = ((static_cast<int64_t>(num_ranks) * sizeof(int) + 127) / 128) * 128;

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
    for (int base = 0; base < kEpNumRanks; base += 32) {
      const int r = base + lane_id;
      const bool is_in =
          (r < kEpNumRanks) and is_combined_token_in_rank[static_cast<int64_t>(t) * num_ranks + r];
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
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        const uint32_t cbytes = static_cast<uint32_t>(csize_int4 * static_cast<int>(sizeof(int4)));
        for (int j = 0; j < num_topk_ranks; ++j) {
          const uint8_t* src = reinterpret_cast<const uint8_t*>(recv_pool_global_ptrs[topk_ranks[j]]) +
                               ep_pool_header_bytes +
                               static_cast<int64_t>(slot_indices[j]) * hidden_int4 * static_cast<int64_t>(sizeof(int4)) +
                               static_cast<int64_t>(c0) * sizeof(int4);
          const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(stage_buf(s, j)));
          asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                           dst),
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
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
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
    // chunk c+(kStages-1) while waiting on + reducing chunk c. Each stage has its own SMEM
    // buffer + mbarrier, so up to kStages-1 chunks (x num_topk_ranks TMA loads each) stay in
    // flight, giving the async copy engine the memory-level parallelism cmbext gets from its
    // per-thread int4[8] batching -- but at low register cost.
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
    if (lane_id < num_topk)
      st_na_global(combined_topk_weights + static_cast<int64_t>(t) * num_topk + lane_id, 0.0f);
  }
}

template <bool kLowLatencyMode, int kNumRDMARanks, typename dtype_t, int kNumCombineForwarderWarps,
          int kNumTopkRDMARanks = get_num_topk_rdma_ranks(kNumRDMARanks),
          int kNumWarpsPerForwarder =
              (kNumCombineForwarderWarps / kNumRDMARanks > 0) ? kNumCombineForwarderWarps / kNumRDMARanks : 1,
          int kNumForwarders = kNumRDMARanks* kNumWarpsPerForwarder,
          int kNumRDMAReceivers = kNumForwarders + NUM_MAX_NVL_PEERS>
__global__ void __launch_bounds__((NUM_MAX_NVL_PEERS + 1 + kNumForwarders) * 32, 1)
    combine(int4* combined_x, float* combined_topk_weights, const bool* is_combined_token_in_rank, const int4* x,
            const float* topk_weights, const int* combined_rdma_head, const int* combined_nvl_head,
            const SourceMeta* src_meta, const int* rdma_channel_prefix_matrix, const int* rdma_rank_prefix_sum,
            const int* gbl_channel_prefix_matrix, int num_tokens, int num_combined_tokens, int hidden, int num_topk,
            void* rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens,
            void** buffer_ptrs, int num_max_nvl_chunked_send_tokens, int num_max_nvl_chunked_recv_tokens, int rank,
            int num_ranks, mscclpp::PortChannelDeviceHandle* port_channel_handles,
            mscclpp::MemoryChannelDeviceHandle* memory_channel_handles,
            // Phase 3 NVLS counter pointers (nullptr → fall back to PortChannel/atomicAdd path).
            void* nvls_head_mc, void* nvls_head_dev, void* nvls_tail_mc, void* nvls_tail_dev,
            // Phase 4 fabric-IPC peer base pointers (nullptr → fall back to handle.put for cross-node data).
            void* const* peer_rdma_bases,
            // Increment 5 combine-direct: domain-wide recv-pool bases (by global rank) +
            // the dispatch gather map. Non-null + kEpDirect => combine gathers each token's
            // contributions straight from the peer pools and reduces locally (no 2-hop).
            void* const* recv_pool_global_ptrs, const int* ep_combine_recv_idx) {
  enum class WarpRole { kNVLSender, kNVLAndRDMAForwarder, kRDMAReceiver, kCoordinator };

  const auto sm_id = static_cast<int>(blockIdx.x);
  const auto thread_id = static_cast<int>(threadIdx.x), lane_id = get_lane_id();
  const auto num_channels = static_cast<int>(gridDim.x) / 2, channel_id = sm_id / 2;
  const bool is_rdma_receiver_sm = sm_id % 2 == 1;

  EP_DEVICE_ASSERT(num_topk <= 32);
  EP_DEVICE_ASSERT(hidden % (sizeof(int4) / sizeof(dtype_t)) == 0);
  const auto hidden_int4 = hidden / (sizeof(int4) / sizeof(dtype_t));

  // NOTES: we decouple a channel into 2 SMs
  const auto rdma_rank = rank / NUM_MAX_NVL_PEERS, nvl_rank = rank % NUM_MAX_NVL_PEERS;
  if (rank == 0 && thread_id == 0 && channel_id == 0) {
    (void)0;
  }
#ifdef EP_DISPATCH_NCCLEP
  // inc5 combine-direct: under MSCCLPP_EP_DIRECT, gather each combined token's
  // contributions DIRECTLY from the peer recv pools (recv_pool_global_ptrs[r] +
  // header + recv_idx*hidden) and reduce locally, skipping nvl_channel +
  // forwarder + rdma_channel. recv_idx is the dispatch gather map
  // (ep_combine_recv_idx[t*num_ranks+r]); this mirrors dispatch's sender-direct
  // write in reverse. Assumes the combine input lives in the recv pool (DeepEP
  // contract; the flat test passes recv_x).
  if (kEpDirect and recv_pool_global_ptrs != nullptr and ep_combine_recv_idx != nullptr) {
    // kEpNumRanks is the world size for this kernel instantiation (compile-time:
    // kNumRDMARanks * NUM_MAX_NVL_PEERS, == num_ranks). The fast combine_token<>
    // discovery maps one warp lane per rank; that is correct AND fast only while
    // kEpNumRanks <= 32. For >= 16 nodes (kEpNumRanks > 32) its __shfl-by-rank source
    // lane wraps mod 32 (ranks >= 32 alias 0..31 -> double-count / garbage), so we fall
    // back to a chunked-ballot discovery that scans ranks in 32-wide chunks. The branch
    // is `if constexpr`, so <= 8 nodes compile to exactly the original fast path.
    constexpr int kEpNumRanks = kNumRDMARanks * NUM_MAX_NVL_PEERS;
    const int64_t ep_pool_header_bytes = ((static_cast<int64_t>(num_ranks) * sizeof(int) + 127) / 128) * 128;
    const int warp_id_flat = thread_id / 32;
    const int num_warps_per_block = static_cast<int>(blockDim.x) / 32;
    const int global_warp = sm_id * num_warps_per_block + warp_id_flat;
    const int total_warps = static_cast<int>(gridDim.x) * num_warps_per_block;
    auto recv_fn = [&](int r, int slot, int hi) -> int4 {
      return ld_nc_global(reinterpret_cast<const int4*>(reinterpret_cast<const uint8_t*>(recv_pool_global_ptrs[r]) +
                                                        ep_pool_header_bytes) +
                          static_cast<int64_t>(slot) * hidden_int4 + hi);
    };
    auto recv_tw_fn = [&](int, int, int) -> float { return 0.0f; };
    if constexpr (kEpNumRanks <= 32) {
      // Original fast path: one lane per rank; combine_token does compile-time-unrolled
      // discovery + register-array reduction. Correct because kEpNumRanks <= 32.
      for (int t = global_warp; t < num_combined_tokens; t += total_warps) {
        const bool is_in =
            (lane_id < kEpNumRanks) and is_combined_token_in_rank[static_cast<int64_t>(t) * num_ranks + lane_id];
        const int recv_idx = is_in ? ep_combine_recv_idx[static_cast<int64_t>(t) * num_ranks + lane_id] : 0;
        combine_token<kEpNumRanks, dtype_t, 8>(is_in, recv_idx, lane_id, hidden_int4, num_topk,
                                               combined_x + static_cast<int64_t>(t) * hidden_int4,
                                               combined_topk_weights + static_cast<int64_t>(t) * num_topk, 1 << 30,
                                               recv_fn, recv_tw_fn);
      }
    } else {
      // >= 16 nodes (kEpNumRanks > 32): chunked-ballot discovery (lane-per-rank aliases
      // past 32). A token routes to <= num_topk (<= 8) distinct ranks => <= 8 contributors.
      constexpr int kEpMaxContrib = 8;
      constexpr auto kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);
      for (int t = global_warp; t < num_combined_tokens; t += total_warps) {
        // Scan ranks in chunks of 32 and compact the set lanes via ballot. Discovery is
        // warp-uniform: every lane ends with the same topk_ranks/slot_indices/num_topk_ranks.
        int topk_ranks[kEpMaxContrib], slot_indices[kEpMaxContrib], num_topk_ranks = 0;
        for (int base = 0; base < kEpNumRanks; base += 32) {
          const int r = base + lane_id;
          const bool is_in =
              (r < kEpNumRanks) and is_combined_token_in_rank[static_cast<int64_t>(t) * num_ranks + r];
          const int slot = is_in ? ep_combine_recv_idx[static_cast<int64_t>(t) * num_ranks + r] : 0;
          unsigned ballot = __ballot_sync(0xffffffffu, is_in);
          while (ballot != 0u) {
            const int l = __ffs(static_cast<int>(ballot)) - 1;
            if (num_topk_ranks < kEpMaxContrib) {
              topk_ranks[num_topk_ranks] = base + l;
              slot_indices[num_topk_ranks] = __shfl_sync(0xffffffffu, slot, l);
              ++num_topk_ranks;
            }
            ballot &= ballot - 1u;
          }
        }
        // Reduce the contributors' hidden rows into combined_x (lane-strided), mirroring
        // combine_token: #pragma unroll the outer hidden loop + pre-load contributors.
        int4* combined_row = combined_x + static_cast<int64_t>(t) * hidden_int4;
#pragma unroll
        for (int i = lane_id; i < hidden_int4; i += 32) {
          int4 recv_value_int4[kEpMaxContrib];
#pragma unroll
          for (int j = 0; j < num_topk_ranks; ++j) recv_value_int4[j] = recv_fn(topk_ranks[j], slot_indices[j], i);
          float values[kDtypePerInt4] = {0};
#pragma unroll
          for (int j = 0; j < num_topk_ranks; ++j) {
            auto vd = reinterpret_cast<const dtype_t*>(&recv_value_int4[j]);
#pragma unroll
            for (int k = 0; k < kDtypePerInt4; ++k) values[k] += static_cast<float>(vd[k]);
          }
          int4 out_int4;
          auto out_dtypes = reinterpret_cast<dtype_t*>(&out_int4);
#pragma unroll
          for (int k = 0; k < kDtypePerInt4; ++k) out_dtypes[k] = static_cast<dtype_t>(values[k]);
          st_na_global(combined_row + i, out_int4);
        }
        // combined_topk_weights: gather path contributes 0 (parity with recv_tw_fn).
        if (lane_id < num_topk)
          st_na_global(combined_topk_weights + static_cast<int64_t>(t) * num_topk + lane_id, 0.0f);
      }
    }
    return;
  }
#endif
  auto role_meta = [=]() -> std::pair<WarpRole, int> {
    auto warp_id = thread_id / 32;
    if (not is_rdma_receiver_sm) {
      if (warp_id < NUM_MAX_NVL_PEERS) {
        auto shuffled_warp_id = warp_id;
        shuffled_warp_id = (shuffled_warp_id + channel_id) % NUM_MAX_NVL_PEERS;
        return {WarpRole::kNVLSender, shuffled_warp_id};
      } else if (warp_id < NUM_MAX_NVL_PEERS + kNumForwarders) {
        auto shuffled_warp_id = warp_id - NUM_MAX_NVL_PEERS;
        shuffled_warp_id = (shuffled_warp_id + channel_id) % kNumForwarders;
        return {WarpRole::kNVLAndRDMAForwarder, shuffled_warp_id};
      } else {
        return {WarpRole::kCoordinator, 0};
      }
    } else {
      if (warp_id < NUM_MAX_NVL_PEERS + kNumForwarders) {
        return {WarpRole::kRDMAReceiver, warp_id};
      } else {
        return {WarpRole::kCoordinator, 0};
      }
    }
  }();
  auto warp_role = role_meta.first;
  auto warp_id = role_meta.second;

  EP_DEVICE_ASSERT(num_warps == NUM_MAX_NVL_PEERS + kNumForwarders + 1);
  auto num_max_nvl_chunked_recv_tokens_per_rdma = num_max_nvl_chunked_recv_tokens / kNumRDMARanks;
  if (thread_id == 0 && channel_id == 0 && rank == 0) {
    (void)0;
  }

  if (warp_role == WarpRole::kNVLSender) {
    // NVL producers
    const auto dst_nvl_rank = warp_id;

    // NVL layouts
    // NOTES: to avoid deadlocks, we use separate NVL buffers for different RDMA sources
    auto dst_buffer_ptr = buffer_ptrs[dst_nvl_rank], local_buffer_ptr = buffer_ptrs[nvl_rank];
    auto nvl_channel_x = AsymBuffer<int4>(dst_buffer_ptr, num_max_nvl_chunked_recv_tokens * hidden_int4,
                                          NUM_MAX_NVL_PEERS, channel_id, num_channels, nvl_rank)
                             .advance_also(local_buffer_ptr);
    auto nvl_channel_src_meta = AsymBuffer<SourceMeta>(dst_buffer_ptr, num_max_nvl_chunked_recv_tokens,
                                                       NUM_MAX_NVL_PEERS, channel_id, num_channels, nvl_rank)
                                    .advance_also(local_buffer_ptr);
    auto nvl_channel_topk_weights = AsymBuffer<float>(dst_buffer_ptr, num_max_nvl_chunked_recv_tokens * num_topk,
                                                      NUM_MAX_NVL_PEERS, channel_id, num_channels, nvl_rank)
                                        .advance_also(local_buffer_ptr);
    auto nvl_channel_head =
        AsymBuffer<int>(local_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, dst_nvl_rank)
            .advance_also(dst_buffer_ptr);
    auto nvl_channel_tail =
        AsymBuffer<int>(dst_buffer_ptr, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels, nvl_rank)
            .advance_also(local_buffer_ptr);

    // Get tasks for each RDMA lane
    int token_start_idx = 0, token_end_idx = 0;
    if (lane_id < kNumRDMARanks) {
      int prefix_idx = (lane_id * NUM_MAX_NVL_PEERS + dst_nvl_rank) * num_channels + channel_id;
      token_start_idx = gbl_channel_prefix_matrix[prefix_idx];
      token_end_idx =
          (prefix_idx == num_channels * num_ranks - 1) ? num_tokens : gbl_channel_prefix_matrix[prefix_idx + 1];
    }
    __syncwarp();

    // NOTES: here the cached value of each lane is only responsible for a single RDMA buffer
    int cached_channel_head_idx = 0, cached_channel_tail_idx = 0;
    EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");

    // Iterate over all tokens and send by chunks
    while (true) {
      // Exit if possible
      if (__all_sync(0xffffffff, token_start_idx >= token_end_idx)) break;

      // Decide next RDMA buffer to send
      bool is_lane_ready = false;
      auto start_time = clock64();
      while (true) {
        int num_used_slots = cached_channel_tail_idx - cached_channel_head_idx;
        is_lane_ready = lane_id < kNumRDMARanks and token_start_idx < token_end_idx and
                        num_max_nvl_chunked_recv_tokens_per_rdma - num_used_slots >= num_max_nvl_chunked_send_tokens;
        if (__any_sync(0xffffffff, is_lane_ready)) break;

        // Retry
        if (lane_id < kNumRDMARanks and token_start_idx < token_end_idx)
          cached_channel_head_idx = ld_volatile_global(nvl_channel_head.buffer() + lane_id);

        // Timeout check
        if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < kNumRDMARanks) {
          printf(
              "DeepEP combine NVL sender timeout, channel: %d, RDMA: %d, nvl: %d, dst NVL: %d, RDMA lane: %d, head: "
              "%d, tail: %d, start: %d, end: %d\n",
              channel_id, rdma_rank, nvl_rank, dst_nvl_rank, lane_id,
              ld_volatile_global(nvl_channel_head.buffer() + lane_id), cached_channel_tail_idx, token_start_idx,
              token_end_idx);
          trap();
        }
      }

      // Sync token start index and count
      for (int current_rdma_idx = 0; current_rdma_idx < kNumRDMARanks; ++current_rdma_idx) {
        if (__shfl_sync(0xffffffff, (token_start_idx >= token_end_idx) or (not is_lane_ready), current_rdma_idx))
          continue;

        // Sync token start index
        auto token_idx = static_cast<int64_t>(__shfl_sync(0xffffffff, token_start_idx, current_rdma_idx));
        int num_tokens_in_chunk = __shfl_sync(
            0xffffffff, min(num_max_nvl_chunked_send_tokens, token_end_idx - token_start_idx), current_rdma_idx);

        // Send by chunk
        for (int chunk_idx = 0; chunk_idx < num_tokens_in_chunk; ++chunk_idx, ++token_idx) {
          // Get an empty slot
          int dst_slot_idx = 0;
          if (lane_id == current_rdma_idx) {
            dst_slot_idx = (cached_channel_tail_idx++) % num_max_nvl_chunked_recv_tokens_per_rdma;
            dst_slot_idx = current_rdma_idx * num_max_nvl_chunked_recv_tokens_per_rdma + dst_slot_idx;
          }
          dst_slot_idx = __shfl_sync(0xffffffff, dst_slot_idx, current_rdma_idx);

          // Copy data
          auto shifted_x_buffers = nvl_channel_x.buffer() + dst_slot_idx * hidden_int4;
          auto shifted_x = x + token_idx * hidden_int4;
          UNROLLED_WARP_COPY(5, lane_id, hidden_int4, shifted_x_buffers, shifted_x, ld_nc_global, st_na_global);

          // Copy source meta
          if (lane_id == 0)
            st_na_global(nvl_channel_src_meta.buffer() + dst_slot_idx, ld_nc_global(src_meta + token_idx));

          // Copy `topk_weights`
          if (lane_id < num_topk)
            st_na_global(nvl_channel_topk_weights.buffer() + dst_slot_idx * num_topk + lane_id,
                         ld_nc_global(topk_weights + token_idx * num_topk + lane_id));
        }
        lane_id == current_rdma_idx ? (token_start_idx = static_cast<int>(token_idx)) : 0;
      }

      // Move queue tail
      __syncwarp();
      if (lane_id < kNumRDMARanks and is_lane_ready)
        st_release_sys_global(nvl_channel_tail.buffer() + lane_id, cached_channel_tail_idx);
    }
  } else {
    // Combiners and coordinators
    // RDMA symmetric layout (snapshot the base before SymBuffer advances it).
    void* const rdma_buffer_ptr_base = rdma_buffer_ptr;
    auto hidden_bytes = hidden_int4 * sizeof(int4);
    auto num_bytes_per_rdma_token = get_num_bytes_per_rdma_token(hidden_int4, 0, 0, num_topk);
    auto rdma_channel_data =
        SymBuffer<int8_t>(rdma_buffer_ptr, num_max_rdma_chunked_recv_tokens * num_bytes_per_rdma_token, kNumRDMARanks,
                          channel_id, num_channels);
    auto rdma_channel_head = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);
    auto rdma_channel_tail = SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);
    // Scratch slots for absolute-value RDMA WRITE replacement of broken HW
    // atomicAdd on Azure CX-7 RoCE (see dispatch kernel for full rationale).
    auto rdma_channel_tail_send_src =
        SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);
    auto rdma_channel_head_send_src =
        SymBuffer<uint64_t, false>(rdma_buffer_ptr, 1, kNumRDMARanks, channel_id, num_channels);

    auto data_send_offset =
        sizeof(int8_t) * (num_max_rdma_chunked_recv_tokens * num_bytes_per_rdma_token) * kNumRDMARanks * channel_id;
    auto data_recv_offset = sizeof(int8_t) * (num_max_rdma_chunked_recv_tokens * num_bytes_per_rdma_token) *
                            kNumRDMARanks * (channel_id + num_channels);
    auto head_offset = sizeof(int8_t) * (num_max_rdma_chunked_recv_tokens * num_bytes_per_rdma_token) * kNumRDMARanks *
                       num_channels * 2;
    auto head_send_offset = head_offset + sizeof(uint64_t) * kNumRDMARanks * channel_id;
    auto tail_offset = head_offset + sizeof(uint64_t) * kNumRDMARanks * num_channels;
    auto tail_send_offset = tail_offset + sizeof(uint64_t) * kNumRDMARanks * channel_id;

    // NVL layouts
    void* local_nvl_buffer = buffer_ptrs[nvl_rank];
    void* nvl_buffers[NUM_MAX_NVL_PEERS];
#pragma unroll
    for (int i = 0; i < NUM_MAX_NVL_PEERS; ++i) nvl_buffers[i] = buffer_ptrs[i];
    auto nvl_channel_x = AsymBuffer<int4>(local_nvl_buffer, num_max_nvl_chunked_recv_tokens * hidden_int4,
                                          NUM_MAX_NVL_PEERS, channel_id, num_channels)
                             .advance_also<NUM_MAX_NVL_PEERS>(nvl_buffers);
    auto nvl_channel_src_meta = AsymBuffer<SourceMeta>(local_nvl_buffer, num_max_nvl_chunked_recv_tokens,
                                                       NUM_MAX_NVL_PEERS, channel_id, num_channels)
                                    .advance_also<NUM_MAX_NVL_PEERS>(nvl_buffers);
    auto nvl_channel_topk_weights = AsymBuffer<float>(local_nvl_buffer, num_max_nvl_chunked_recv_tokens * num_topk,
                                                      NUM_MAX_NVL_PEERS, channel_id, num_channels)
                                        .advance_also<NUM_MAX_NVL_PEERS>(nvl_buffers);
    auto nvl_channel_head = AsymBuffer<int, NUM_MAX_NVL_PEERS>(nvl_buffers, kNumRDMARanks, NUM_MAX_NVL_PEERS,
                                                               channel_id, num_channels, nvl_rank)
                                .advance_also(local_nvl_buffer);
    auto nvl_channel_tail =
        AsymBuffer<int>(local_nvl_buffer, kNumRDMARanks, NUM_MAX_NVL_PEERS, channel_id, num_channels)
            .advance_also<NUM_MAX_NVL_PEERS>(nvl_buffers);

    // Combiner warp synchronization
    __shared__ volatile int forwarder_nvl_head[kNumForwarders][NUM_MAX_NVL_PEERS];
    __shared__ volatile bool forwarder_retired[kNumForwarders];
    __shared__ volatile int rdma_receiver_rdma_head[kNumRDMAReceivers][kNumRDMARanks];
    __shared__ volatile bool rdma_receiver_retired[kNumRDMAReceivers];
    auto sync_forwarder_smem = [=]() { asm volatile("bar.sync 0, %0;" ::"r"((kNumForwarders + 1) * 32)); };
    auto sync_rdma_receiver_smem = [=]() { asm volatile("bar.sync 1, %0;" ::"r"((kNumRDMAReceivers + 1) * 32)); };

    if (warp_role == WarpRole::kNVLAndRDMAForwarder) {
      // Receive from NVL ranks and forward to RDMA ranks
      // NOTES: this part is using "large warps" for each RDMA ranks
      const auto dst_rdma_rank = warp_id / kNumWarpsPerForwarder;
      const auto sub_warp_id = warp_id % kNumWarpsPerForwarder;
      auto send_buffer = dst_rdma_rank == rdma_rank ? rdma_channel_data.recv_buffer(dst_rdma_rank)
                                                    : rdma_channel_data.send_buffer(dst_rdma_rank);
      auto sync_large_warp = [=]() {
        if (kNumWarpsPerForwarder == 1) {
          __syncwarp();
        } else {
          asm volatile("bar.sync %0, %1;" ::"r"(dst_rdma_rank + 2), "r"(kNumWarpsPerForwarder * 32));
        }
      };
      EP_STATIC_ASSERT(kNumWarpsPerForwarder == 1 or kNumRDMARanks + 2 <= 16, "Barriers are not enough");

      // Advance to the corresponding NVL buffer
      nvl_channel_x.advance(dst_rdma_rank * num_max_nvl_chunked_recv_tokens_per_rdma * hidden_int4);
      nvl_channel_src_meta.advance(dst_rdma_rank * num_max_nvl_chunked_recv_tokens_per_rdma);
      nvl_channel_topk_weights.advance(dst_rdma_rank * num_max_nvl_chunked_recv_tokens_per_rdma * num_topk);
      nvl_channel_head.advance(dst_rdma_rank);
      nvl_channel_tail.advance(dst_rdma_rank);

      // Clean shared memory and sync
      EP_STATIC_ASSERT(NUM_MAX_NVL_PEERS <= 32, "Invalid number of NVL peers");
      lane_id < NUM_MAX_NVL_PEERS ? (forwarder_nvl_head[warp_id][lane_id] = 0) : 0;
      lane_id == 0 ? (forwarder_retired[warp_id] = false) : false;
      sync_forwarder_smem();

      // Get count and cached head
      int cached_nvl_channel_tail_idx = 0;
      int num_tokens_to_combine = rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id];
      int num_tokens_prefix =
          channel_id == 0 ? 0 : rdma_channel_prefix_matrix[dst_rdma_rank * num_channels + channel_id - 1];
      num_tokens_to_combine -= num_tokens_prefix;
      num_tokens_prefix += dst_rdma_rank == 0 ? 0 : rdma_rank_prefix_sum[dst_rdma_rank - 1];
      combined_nvl_head += num_tokens_prefix * NUM_MAX_NVL_PEERS;

      // Iterate over all tokens and combine by chunks
      for (int token_start_idx = 0; token_start_idx < num_tokens_to_combine;
           token_start_idx += num_max_rdma_chunked_send_tokens) {
        // Check destination queue emptiness, or wait a buffer to be released
        auto token_end_idx = min(token_start_idx + num_max_rdma_chunked_send_tokens, num_tokens_to_combine);
        auto num_chunked_tokens = token_end_idx - token_start_idx;
        auto start_time = clock64();
        while (sub_warp_id == 0 and lane_id == 0) {
          // Inequality: `num_max_rdma_chunked_recv_tokens - (tail - head) >= num_chunked_tokens`
          // Here, `token_start_idx` is the actual tail
          // Phase 4: head read — cross-node head feedback comes from peer
          // via fabric-VA store, self-loop head from local atomic. Both end
          // up in rdma_channel_head; one read path covers them.
          int num_used_slots =
              token_start_idx - static_cast<int>(ld_volatile_global(rdma_channel_head.buffer(dst_rdma_rank)));
          if (num_max_rdma_chunked_recv_tokens - num_used_slots >= num_chunked_tokens) break;

          // Timeout check
          if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
            printf(
                "DeepEP combine forwarder (RDMA check) timeout, channel: %d, RDMA: %d, nvl: %d, dst RDMA: %d, head: "
                "%ld, tail: %d, chunked: %d\n",
                channel_id, rdma_rank, nvl_rank, dst_rdma_rank,
                ld_volatile_global(rdma_channel_head.buffer(dst_rdma_rank)), token_start_idx, num_chunked_tokens);
            trap();
          }
        }
        sync_large_warp();

        // Combine and write to the RDMA buffer
        for (int token_idx = token_start_idx + sub_warp_id; token_idx < token_end_idx;
             token_idx += kNumWarpsPerForwarder) {
          // Read expected head
          EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
          int expected_head = -1;
          if (lane_id < NUM_MAX_NVL_PEERS)
            expected_head = ld_nc_global(combined_nvl_head + token_idx * NUM_MAX_NVL_PEERS + lane_id);

          // Wait lanes to be ready
          start_time = clock64();
          while (cached_nvl_channel_tail_idx <= expected_head) {
            cached_nvl_channel_tail_idx = ld_acquire_sys_global(nvl_channel_tail.buffer(lane_id));

            // Timeout check
            if (clock64() - start_time > NUM_TIMEOUT_CYCLES and lane_id < NUM_MAX_NVL_PEERS) {
              printf(
                  "DeepEP combine forwarder (NVL check) timeout, channel: %d, RDMA: %d, nvl: %d, src NVL: %d, dst "
                  "RDMA: %d, tail: %d, waiting: %d, total: %d, sub: %d, large: %d, expected: %d\n",
                  channel_id, rdma_rank, nvl_rank, lane_id, dst_rdma_rank, cached_nvl_channel_tail_idx, token_idx,
                  num_tokens_to_combine, sub_warp_id, kNumWarpsPerForwarder, expected_head);
              trap();
            }
          }

          // Combine current token
          auto rdma_slot_idx = token_idx % num_max_rdma_chunked_recv_tokens;
          void* shifted = send_buffer + rdma_slot_idx * num_bytes_per_rdma_token;
          auto recv_fn = [&](int src_nvl_rank, int slot_idx, int hidden_int4_idx) -> int4 {
            return ld_nc_global(nvl_channel_x.buffer(src_nvl_rank) + slot_idx * hidden_int4 + hidden_int4_idx);
          };
          auto recv_tw_fn = [&](int src_nvl_rank, int slot_idx, int topk_idx) -> float {
            return ld_nc_global(nvl_channel_topk_weights.buffer(src_nvl_rank) + slot_idx * num_topk + topk_idx);
          };
          combine_token<NUM_MAX_NVL_PEERS, dtype_t, NUM_MAX_NVL_PEERS>(
              expected_head >= 0, expected_head, lane_id, hidden_int4, num_topk, reinterpret_cast<int4*>(shifted),
              reinterpret_cast<float*>(reinterpret_cast<int8_t*>(shifted) + hidden_bytes + sizeof(SourceMeta)),
              num_max_nvl_chunked_recv_tokens_per_rdma, recv_fn, recv_tw_fn);

          // Update head
          if (lane_id < NUM_MAX_NVL_PEERS)
            expected_head < 0 ? (forwarder_nvl_head[warp_id][lane_id] = -expected_head - 1)
                              : (forwarder_nvl_head[warp_id][lane_id] = expected_head + 1);
        }
        sync_large_warp();

        // Issue RDMA send
        if (sub_warp_id == kNumWarpsPerForwarder - 1) {
          if (nvls_tail_mc != nullptr) {
            // Phase 3+4: NVLS counter fast path. Cross-node payload uses
            // warp-cooperative direct fabric-IPC writes when peer_rdma_bases
            // is available; otherwise single-lane handle.put + flush.
            if (dst_rdma_rank != rdma_rank) {
              const auto rdma_slot_idx = token_start_idx % num_max_rdma_chunked_recv_tokens;
              const size_t num_bytes_per_msg = (size_t)num_chunked_tokens * (size_t)num_bytes_per_rdma_token;
              const auto dst_offset = rdma_rank * (num_max_rdma_chunked_recv_tokens * num_bytes_per_rdma_token) +
                                      rdma_slot_idx * num_bytes_per_rdma_token + data_recv_offset;
              const auto src_offset = dst_rdma_rank * (num_max_rdma_chunked_recv_tokens * num_bytes_per_rdma_token) +
                                      rdma_slot_idx * num_bytes_per_rdma_token + data_send_offset;
              if (peer_rdma_bases != nullptr) {
                const int dst_rank_global = dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank;
                const int4* src_p =
                    reinterpret_cast<const int4*>(reinterpret_cast<uint8_t*>(rdma_buffer_ptr_base) + src_offset);
                int4* dst_p =
                    reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(peer_rdma_bases[dst_rank_global]) + dst_offset);
                const int n_int4 = (int)(num_bytes_per_msg / sizeof(int4));
                // Same 8x unroll as dispatch sender for LSU pipelining.
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
              } else if (lane_id == 0) {
                const auto port_channel_idx =
                    kLowLatencyMode ? (channel_id * kNumRDMARanks + dst_rdma_rank)
                                    : (channel_id * num_ranks + dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank);
                auto& handle = port_channel_handles[port_channel_idx];
                handle.put(dst_offset, src_offset, num_bytes_per_msg);
                handle.flush();
              }
            }
            // Phase 4: tail counter publish.
            //   cross-node → direct fabric-VA st.release on peer's tail slot
            //   self-loop → plain local atomicAdd (NVLS multicast over-counts
            //                across NVL peers — see dispatch fix).
            if (dst_rdma_rank != rdma_rank) {
              if (peer_rdma_bases != nullptr && lane_id == 0) {
                const int dst_rank_global = dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank;
                const uintptr_t my_tail_off = reinterpret_cast<uintptr_t>(rdma_channel_tail.buffer(rdma_rank)) -
                                              reinterpret_cast<uintptr_t>(rdma_buffer_ptr_base);
                uint64_t* peer_tail = reinterpret_cast<uint64_t*>(
                    reinterpret_cast<uint8_t*>(peer_rdma_bases[dst_rank_global]) + my_tail_off);
                const uint64_t new_tail = (uint64_t)(token_start_idx + num_chunked_tokens);
                asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(peer_tail), "l"(new_tail) : "memory");
              }
            } else if (lane_id == 0) {
              mscclpp::atomicFetchAdd(reinterpret_cast<uint64_t*>(rdma_channel_tail.buffer(rdma_rank)),
                                      (uint64_t)num_chunked_tokens, mscclpp::memoryOrderRelease);
            }
          } else if (lane_id == 0) {
            if (dst_rdma_rank == rdma_rank) {
              mscclpp::atomicFetchAdd(reinterpret_cast<uint64_t*>(rdma_channel_tail.buffer(rdma_rank)),
                                      (uint64_t)num_chunked_tokens, mscclpp::memoryOrderRelease);
            } else {
              auto rdma_slot_idx = token_start_idx % num_max_rdma_chunked_recv_tokens;
              const size_t num_bytes_per_msg = num_chunked_tokens * num_bytes_per_rdma_token;
              auto dst_offset = rdma_rank * (num_max_rdma_chunked_recv_tokens * num_bytes_per_rdma_token) +
                                rdma_slot_idx * num_bytes_per_rdma_token + data_recv_offset;
              auto src_offset = dst_rdma_rank * (num_max_rdma_chunked_recv_tokens * num_bytes_per_rdma_token) +
                                rdma_slot_idx * num_bytes_per_rdma_token + data_send_offset;
              auto port_channel_idx = kLowLatencyMode
                                          ? (channel_id * kNumRDMARanks + dst_rdma_rank)
                                          : (channel_id * num_ranks + dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank);
              auto& handle = port_channel_handles[port_channel_idx];
              handle.put(dst_offset, src_offset, num_bytes_per_msg);

              // Absolute-value RDMA WRITE replaces broken HW atomicAdd.
              const uint64_t new_tail = (uint64_t)(token_start_idx + num_chunked_tokens);
              *rdma_channel_tail_send_src.buffer(dst_rdma_rank) = new_tail;
              __threadfence_system();
              const auto src_off_tail = reinterpret_cast<uintptr_t>(rdma_channel_tail_send_src.buffer(dst_rdma_rank)) -
                                        reinterpret_cast<uintptr_t>(rdma_buffer_ptr_base);
              handle.put(rdma_rank * sizeof(uint64_t) + tail_send_offset, src_off_tail, sizeof(uint64_t));
            }
          }
          __syncwarp();
        }
      }

      // Retired
      __syncwarp();
      if (lane_id == 0) forwarder_retired[warp_id] = true;
    } else if (warp_role == WarpRole::kRDMAReceiver) {
      // Receive from RDMA ranks and write to the output tensor
      // Clean shared memory and sync
      EP_DEVICE_ASSERT(kNumRDMARanks <= 32);
      lane_id < kNumRDMARanks ? (rdma_receiver_rdma_head[warp_id][lane_id] = 0) : 0;
      lane_id == 0 ? (rdma_receiver_retired[warp_id] = false) : 0;
      sync_rdma_receiver_smem();

      // The same tokens as the dispatch process
      int token_start_idx, token_end_idx;
      get_channel_task_range(num_combined_tokens, num_channels, channel_id, token_start_idx, token_end_idx);

      // Iterate over all tokens and combine
      int cached_channel_tail_idx = 0;
      for (int64_t token_idx = token_start_idx + warp_id; token_idx < token_end_idx; token_idx += kNumRDMAReceivers) {
        // Read expected head
        EP_STATIC_ASSERT(kNumRDMARanks <= 32, "Invalid number of RDMA peers");
        int expected_head = -1;
        if (lane_id < kNumRDMARanks) {
          expected_head = ld_nc_global(combined_rdma_head + token_idx * kNumRDMARanks + lane_id);
          (expected_head < 0) ? (rdma_receiver_rdma_head[warp_id][lane_id] = -expected_head - 1)
                              : (rdma_receiver_rdma_head[warp_id][lane_id] = expected_head);
        }

        // Wait lanes to be ready
        auto start_time = clock64();
        while (cached_channel_tail_idx <= expected_head) {
          // Phase 4: receiver waits on rdma_channel_tail. Cross-node sender
          // wrote it via fabric-VA, self sender wrote it via local atomic.
          // One read path via ld_acquire_sys_global covers both.
          cached_channel_tail_idx = static_cast<int>(ld_acquire_sys_global(rdma_channel_tail.buffer(lane_id)));

          // Timeout check
          if (clock64() - start_time > NUM_TIMEOUT_CYCLES) {
            printf(
                "DeepEP combine RDMA receiver timeout, channel: %d, RDMA: %d, nvl: %d, src RDMA: %d, tail: %d, "
                "waiting: %ld, expect: %d\n",
                channel_id, rdma_rank, nvl_rank, lane_id, cached_channel_tail_idx, token_idx, expected_head);
            trap();
          }
        }
        __syncwarp();

        // Combine current token
        auto recv_fn = [&](int src_rdma_rank, int slot_idx, int hidden_int4_idx) -> int4 {
          return ld_nc_global(reinterpret_cast<const int4*>(rdma_channel_data.recv_buffer(src_rdma_rank) +
                                                            slot_idx * num_bytes_per_rdma_token) +
                              hidden_int4_idx);
        };
        auto recv_tw_fn = [&](int src_rdma_rank, int slot_idx, int topk_idx) -> float {
          return ld_nc_global(reinterpret_cast<const float*>(rdma_channel_data.recv_buffer(src_rdma_rank) +
                                                             slot_idx * num_bytes_per_rdma_token + hidden_bytes +
                                                             sizeof(SourceMeta)) +
                              topk_idx);
        };
        combine_token<kNumRDMARanks, dtype_t, kNumTopkRDMARanks>(
            expected_head >= 0, expected_head, lane_id, hidden_int4, num_topk, combined_x + token_idx * hidden_int4,
            combined_topk_weights + token_idx * num_topk, num_max_rdma_chunked_recv_tokens, recv_fn, recv_tw_fn);
      }

      // Retired
      __syncwarp();
      if (lane_id == 0) rdma_receiver_retired[warp_id] = true;
    } else {
      // Coordinator
      // Sync shared memory status
      is_rdma_receiver_sm ? sync_rdma_receiver_smem() : sync_forwarder_smem();
      const auto num_warps_per_rdma_rank = kNumForwarders / kNumRDMARanks;

      int last_rdma_head = 0;
      int last_nvl_head[kNumRDMARanks] = {0};
      int dst_rdma_rank = lane_id < kNumRDMARanks ? lane_id : 0;
      int dst_nvl_rank = lane_id < NUM_MAX_NVL_PEERS ? lane_id : 0;
      EP_STATIC_ASSERT(kNumCombineForwarderWarps <= 32, "Invalid number of forwarder warps");
      while (true) {
        // Retired
        if (is_rdma_receiver_sm and
            __all_sync(0xffffffff, lane_id >= kNumRDMAReceivers or rdma_receiver_retired[lane_id]))
          break;
        if (not is_rdma_receiver_sm and __all_sync(0xffffffff, lane_id >= kNumForwarders or forwarder_retired[lane_id]))
          break;

        // Find minimum head for RDMA ranks
        if (is_rdma_receiver_sm) {
          int min_head = std::numeric_limits<int>::max();
#pragma unroll
          for (int i = 0; i < kNumRDMAReceivers; ++i)
            if (not rdma_receiver_retired[i]) min_head = min(min_head, rdma_receiver_rdma_head[i][dst_rdma_rank]);
          if (min_head != std::numeric_limits<int>::max() and
              min_head >= last_rdma_head + num_max_rdma_chunked_send_tokens and lane_id < kNumRDMARanks) {
            // Phase 4: head feedback path.
            //   cross-node → fabric-VA st.release on peer's head slot
            //   self-loop → local atomicAdd
            if (peer_rdma_bases != nullptr && dst_rdma_rank != rdma_rank) {
              const int dst_rank_global = dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank;
              const uintptr_t my_head_off = reinterpret_cast<uintptr_t>(rdma_channel_head.buffer(rdma_rank)) -
                                            reinterpret_cast<uintptr_t>(rdma_buffer_ptr_base);
              uint64_t* peer_head = reinterpret_cast<uint64_t*>(
                  reinterpret_cast<uint8_t*>(peer_rdma_bases[dst_rank_global]) + my_head_off);
              asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(peer_head), "l"((uint64_t)min_head) : "memory");
            } else if (dst_rdma_rank == rdma_rank) {
              mscclpp::atomicFetchAdd(static_cast<uint64_t*>(rdma_channel_head.buffer(rdma_rank)),
                                      (uint64_t)(min_head - last_rdma_head), mscclpp::memoryOrderRelease);
            } else {
              auto dst_offset = rdma_rank * sizeof(uint64_t) + head_send_offset;
              auto port_channel_idx = kLowLatencyMode
                                          ? (channel_id * kNumRDMARanks + dst_rdma_rank)
                                          : (channel_id * num_ranks + dst_rdma_rank * NUM_MAX_NVL_PEERS + nvl_rank);
              auto& handle = port_channel_handles[port_channel_idx];
              // Absolute-value RDMA WRITE replaces broken HW atomicAdd.
              *rdma_channel_head_send_src.buffer(dst_rdma_rank) = (uint64_t)min_head;
              __threadfence_system();
              const auto src_off_head = reinterpret_cast<uintptr_t>(rdma_channel_head_send_src.buffer(dst_rdma_rank)) -
                                        reinterpret_cast<uintptr_t>(rdma_buffer_ptr_base);
              handle.put(dst_offset, src_off_head, sizeof(uint64_t));
            }
            last_rdma_head = min_head;
          }
        } else {
// Find minimum head for NVL ranks
#pragma unroll
          for (int i = 0; i < kNumRDMARanks; ++i) {
            int min_head = std::numeric_limits<int>::max();
#pragma unroll
            for (int j = 0; j < num_warps_per_rdma_rank; ++j)
              if (not forwarder_retired[i * num_warps_per_rdma_rank + j])
                min_head = min(min_head, forwarder_nvl_head[i * num_warps_per_rdma_rank + j][dst_nvl_rank]);
            if (min_head != std::numeric_limits<int>::max() and min_head > last_nvl_head[i] and
                lane_id < NUM_MAX_NVL_PEERS)
              st_relaxed_sys_global(nvl_channel_head.buffer_by(dst_nvl_rank) + i, last_nvl_head[i] = min_head);
          }
        }

        // Nanosleep and let other warps work
        __nanosleep(NUM_WAIT_NANOSECONDS);
      }
    }
  }
}

void combine(cudaDataType_t type, void* combined_x, float* combined_topk_weights, const bool* is_combined_token_in_rank,
             const void* x, const float* topk_weights, const int* combined_rdma_head, const int* combined_nvl_head,
             const void* src_meta, const int* rdma_channel_prefix_matrix, const int* rdma_rank_prefix_sum,
             const int* gbl_channel_prefix_matrix, int num_tokens, int num_combined_tokens, int hidden, int num_topk,
             void* rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens,
             void** buffer_ptrs, int num_max_nvl_chunked_send_tokens, int num_max_nvl_chunked_recv_tokens, int rank,
             int num_ranks, cudaStream_t stream, int num_channels, bool low_latency_mode,
             mscclpp::PortChannelDeviceHandle* port_channel_handles,
             mscclpp::MemoryChannelDeviceHandle* memory_channel_handles, void* nvls_head_mc, void* nvls_head_dev,
             void* nvls_tail_mc, void* nvls_tail_dev, void* const* peer_rdma_bases, void* const* recv_pool_global_ptrs,
             const int* ep_combine_recv_idx) {
  constexpr int kNumCombineForwarderWarps = 16;

#define COMBINE_LAUNCH_CASE(num_rdma_ranks)                                                                           \
  {                                                                                                                   \
    auto combine_func = low_latency_mode ? combine<true, num_rdma_ranks, nv_bfloat16, kNumCombineForwarderWarps>      \
                                         : combine<false, num_rdma_ranks, nv_bfloat16, kNumCombineForwarderWarps>;    \
    LAUNCH_KERNEL(&cfg, combine_func, reinterpret_cast<int4*>(combined_x), combined_topk_weights,                     \
                  is_combined_token_in_rank, reinterpret_cast<const int4*>(x), topk_weights, combined_rdma_head,      \
                  combined_nvl_head, reinterpret_cast<const SourceMeta*>(src_meta), rdma_channel_prefix_matrix,       \
                  rdma_rank_prefix_sum, gbl_channel_prefix_matrix, num_tokens, num_combined_tokens, hidden, num_topk, \
                  rdma_buffer_ptr, num_max_rdma_chunked_send_tokens, num_max_rdma_chunked_recv_tokens, buffer_ptrs,   \
                  num_max_nvl_chunked_send_tokens, num_max_nvl_chunked_recv_tokens, rank, num_ranks,                  \
                  port_channel_handles, memory_channel_handles, nvls_head_mc, nvls_head_dev, nvls_tail_mc,            \
                  nvls_tail_dev, peer_rdma_bases, recv_pool_global_ptrs, ep_combine_recv_idx);                        \
  }                                                                                                                   \
  break

  int num_rdma_ranks = num_ranks / NUM_MAX_NVL_PEERS;
  auto num_warps_per_forwarder = std::max(kNumCombineForwarderWarps / num_rdma_ranks, 1);
  int num_forwarder_warps = num_rdma_ranks * num_warps_per_forwarder;
  EP_HOST_ASSERT(num_forwarder_warps > 0 and num_forwarder_warps % num_rdma_ranks == 0);
  EP_HOST_ASSERT(num_max_nvl_chunked_recv_tokens % num_rdma_ranks == 0);
  EP_HOST_ASSERT(num_max_nvl_chunked_recv_tokens / num_rdma_ranks >
                 std::max(num_max_rdma_chunked_send_tokens, num_max_nvl_chunked_send_tokens));
  EP_HOST_ASSERT(type == CUDA_R_16BF);

#ifdef EP_DISPATCH_NCCLEP
  // FLAT direct-gather combine path. recv_pool_global_ptrs + ep_combine_recv_idx are non-null
  // only on the flat direct path (MSCCLPP_EP_DIRECT); every other configuration falls through
  // to the unified 2-hop combine below, byte-for-byte unchanged.
  if (recv_pool_global_ptrs != nullptr and ep_combine_recv_idx != nullptr) {
    // Default: TMA-staged gather (combine_flat_gather_tma). The async copy engine hides the
    // remote-NVLink read latency at low register cost, so it wins at every channel count (no
    // crossover) and every node scale, and beats NCCL-EP. Escape hatch MSCCLPP_EP_COMBINE_TMA=0
    // falls back to the synchronous lean gather.
    static const bool ep_combine_tma_disabled = [] {
      const char* e = std::getenv("MSCCLPP_EP_COMBINE_TMA");
      return e != nullptr and std::atoi(e) == 0;
    }();
    if (not ep_combine_tma_disabled) {
      constexpr int kCmbTmaWarps = EP_CMB_TMA_WARPS;
      constexpr int kCmbTmaChunkInt4 = EP_CMB_TMA_CHUNK_INT4;
      constexpr int kCmbTmaStages = EP_CMB_TMA_STAGES;
      constexpr int kCmbTmaMaxContrib = 8;
      // SMEM is independent of hidden (staging tiles + per-stage mbarriers). >48KB -> opt in.
      const size_t cmb_tma_smem =
          static_cast<size_t>(kCmbTmaWarps) * kCmbTmaStages * kCmbTmaMaxContrib * kCmbTmaChunkInt4 * sizeof(int4) +
          static_cast<size_t>(kCmbTmaWarps) * kCmbTmaStages * sizeof(uint64_t);
#define COMBINE_FLAT_GATHER_TMA_CASE(num_rdma_ranks)                                                       \
  {                                                                                                       \
    auto tma_func = combine_flat_gather_tma<nv_bfloat16, num_rdma_ranks>;                                 \
    CUDA_CHECK(cudaFuncSetAttribute(tma_func, cudaFuncAttributeMaxDynamicSharedMemorySize,                \
                                    static_cast<int>(cmb_tma_smem)));                                      \
    cudaLaunchConfig_t cfg = {static_cast<unsigned>(num_channels * 2),                                    \
                              static_cast<unsigned>(kCmbTmaWarps * 32), cmb_tma_smem, stream, nullptr,    \
                              0};                                                                          \
    cudaLaunchAttribute a[1];                                                                             \
    a[0].id = cudaLaunchAttributeCooperative;                                                             \
    a[0].val.cooperative = 1;                                                                             \
    cfg.attrs = a;                                                                                        \
    cfg.numAttrs = 1;                                                                                     \
    LAUNCH_KERNEL(&cfg, tma_func, reinterpret_cast<int4*>(combined_x), combined_topk_weights,             \
                  is_combined_token_in_rank, num_combined_tokens, hidden, num_topk, num_ranks,            \
                  recv_pool_global_ptrs, ep_combine_recv_idx);                                            \
    break;                                                                                                \
  }
      SWITCH_RDMA_RANKS(COMBINE_FLAT_GATHER_TMA_CASE);
#undef COMBINE_FLAT_GATHER_TMA_CASE
      return;
    }
    // Fallback (MSCCLPP_EP_COMBINE_TMA=0): synchronous lean gather. Its full 1024-thread blocks
    // hide latency with register MLP + occupancy, which wins when blocks are few but regresses
    // at high SM -- so it is gated to a low channel count; above it, the unified kernel's flat
    // gather branch runs.
    constexpr int kEpCombineLeanMaxChannels = 14;
    if (num_channels <= kEpCombineLeanMaxChannels) {
#define COMBINE_FLAT_GATHER_CASE(num_rdma_ranks)                                                           \
  {                                                                                                       \
    auto gather_func = combine_flat_gather<nv_bfloat16, num_rdma_ranks>;                                  \
    LAUNCH_KERNEL(&cfg, gather_func, reinterpret_cast<int4*>(combined_x), combined_topk_weights,          \
                  is_combined_token_in_rank, num_combined_tokens, hidden, num_topk, num_ranks,            \
                  recv_pool_global_ptrs, ep_combine_recv_idx);                                            \
  }                                                                                                       \
  break
      SETUP_LAUNCH_CONFIG(num_channels * 2, 1024, stream);
      SWITCH_RDMA_RANKS(COMBINE_FLAT_GATHER_CASE);
#undef COMBINE_FLAT_GATHER_CASE
      return;
    }
  }
#endif

  SETUP_LAUNCH_CONFIG(num_channels * 2, (NUM_MAX_NVL_PEERS + num_forwarder_warps + 1) * 32, stream);
  SWITCH_RDMA_RANKS(COMBINE_LAUNCH_CASE);
#undef COMBINE_LAUNCH_CASE
}

}  // namespace internode

}  // namespace ep
}  // namespace mscclpp
