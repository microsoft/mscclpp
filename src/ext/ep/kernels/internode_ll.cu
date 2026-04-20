// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Low-latency internode dispatch/combine kernels ported from DeepEP
// `csrc/kernels/internode_ll.cu` (branch `chhwang/dev-atomic-add-cleanup`).
//
// NVSHMEM/IBGDA device calls are replaced with MSCCL++ PortChannel device
// operations:
//
//   nvshmemx_barrier_all_block()              -> port-channel signal/wait ring
//   nvshmemi_ibgda_put_nbi_warp(...)          -> port_channel.put(...) (lane 0)
//   nvshmemi_ibgda_amo_nonfetch_add(...)      -> port_channel.atomicAdd(...)
//
// Addressing convention:
//   - `rdma_buffer_ptr` is the base of the locally-registered RDMA buffer.
//   - Remote counter/buffer pointers written by the kernel are virtual
//     addresses that alias the corresponding offset inside each peer's
//     symmetric RDMA buffer. MSCCL++ needs those as offsets; we derive them
//     via `ptr - rdma_buffer_ptr`.
//   - Port-channel layout built by `Buffer::sync()` in low-latency mode is
//     `handles[qp * num_peers + peer_idx]` where `peer_idx` is the dst rank's
//     position in the connected-peer map. In the recommended 1-GPU-per-node
//     LL topology, `peer_idx == dst_rank`; see src/ext/ep/README.md.
//
// WARNING: This port is untested on multi-node H100; performance will NOT
// match IBGDA (host-proxy adds latency). Functional correctness needs
// validation on real hardware.

#include "configs.cuh"
#include "exception.cuh"
#include "launch.cuh"
#include "utils.cuh"

#include <cooperative_groups.h>
#include <mscclpp/port_channel_device.hpp>

namespace cg = cooperative_groups;

namespace mscclpp {
namespace ep {

namespace internode_ll {

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

// Pointer-to-offset helper for MSCCL++ port channels. Both src and dst
// pointers passed to the DeepEP LL kernels are virtual addresses aliased into
// the caller's symmetric RDMA buffer; MSCCL++ expects offsets.
__device__ __forceinline__ uint64_t rdma_offset_of(uint64_t ptr, void* rdma_buffer_ptr) {
    return ptr - reinterpret_cast<uint64_t>(rdma_buffer_ptr);
}

// Cross-rank barrier via port-channel signal/wait ring.
// Uses port channel `qp=0` across all connected peers.
__device__ __forceinline__ void port_channel_barrier_block(
        mscclpp::PortChannelDeviceHandle* port_channel_handles,
        int rank, int num_ranks) {
    const int tid = threadIdx.x;
    if (tid < num_ranks && tid != rank) {
        // Index: qp 0, peer = tid's rank (assumes peer_idx == rank in LL topology).
        port_channel_handles[tid].signal();
        port_channel_handles[tid].wait();
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// clean_low_latency_buffer
// ---------------------------------------------------------------------------

template <int kNumThreads> __launch_bounds__(kNumThreads, 1)
__global__ void clean_low_latency_buffer(int* clean_0, int num_clean_int_0,
                                         int* clean_1, int num_clean_int_1,
                                         mscclpp::PortChannelDeviceHandle* port_channel_handles,
                                         int rank, int num_ranks) {
    // Barrier before cleaning (in case of unfinished chunked EP)
    port_channel_barrier_block(port_channel_handles, rank, num_ranks);

    // Clean
    auto thread_id = static_cast<int>(threadIdx.x);
    #pragma unroll
    for (int i = thread_id; i < num_clean_int_0; i += kNumThreads)
        clean_0[i] = 0;
    #pragma unroll
    for (int i = thread_id; i < num_clean_int_1; i += kNumThreads)
        clean_1[i] = 0;

    // Barrier after cleaning (make sure low-latency mode work fine)
    port_channel_barrier_block(port_channel_handles, rank, num_ranks);
}

void clean_low_latency_buffer(int* clean_0, int num_clean_int_0,
                              int* clean_1, int num_clean_int_1,
                              int rank, int num_ranks,
                              mscclpp::PortChannelDeviceHandle* port_channel_handles,
                              cudaStream_t stream) {
    constexpr int kNumThreads = 256;

    SETUP_LAUNCH_CONFIG(1, kNumThreads, stream);
    LAUNCH_KERNEL(&cfg, clean_low_latency_buffer<kNumThreads>,
                  clean_0, num_clean_int_0, clean_1, num_clean_int_1,
                  port_channel_handles, rank, num_ranks);
}

// ---------------------------------------------------------------------------
// dispatch
// ---------------------------------------------------------------------------

template <bool kUseFP8, int kNumWarpGroups, int kNumWarpsPerGroup, int kHidden>
__global__ __launch_bounds__(kNumWarpGroups * kNumWarpsPerGroup * 32, 1) void
dispatch(void* packed_recv_x, float* packed_recv_x_scales,
         int* packed_recv_src_info, int64_t* packed_recv_layout_range,
         int* packed_recv_count,
         void* rdma_recv_x, int* rdma_recv_count, void* rdma_x,
         const void* x, const int64_t* topk_idx,
         int* atomic_counter_per_expert, int* atomic_finish_counter_per_expert,
         int* next_clean, int num_next_clean_int,
         int num_tokens, int num_max_dispatch_tokens_per_rank,
         int num_topk, int num_experts, int rank, int num_ranks,
         int phases,
         void* rdma_buffer_ptr,
         mscclpp::PortChannelDeviceHandle* port_channel_handles) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto num_warps = kNumWarpGroups * kNumWarpsPerGroup;
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / kNumWarpsPerGroup;
    const auto sub_warp_id = warp_id % kNumWarpsPerGroup;
    const auto responsible_expert_idx = sm_id * kNumWarpGroups + warp_group_id;

    // FP8 staffs
    constexpr int kNumPerChannels = 128;
    constexpr float kFP8Margin = 1e-4, kFP8Amax = 448, kFP8AmaxInv = 1.0f / 448.0f;
    const int num_scales = kHidden / kNumPerChannels;
    const size_t hidden_bytes = kHidden * (kUseFP8 ? sizeof(__nv_fp8_storage_t) : sizeof(nv_bfloat16));
    const size_t hidden_int4 = hidden_bytes / sizeof(int4);

    // Message package: hidden data, FP8 scales, index at source
    using vec_t = typename std::conditional<kUseFP8, int2, int4>::type;
    const size_t num_bytes_per_msg = sizeof(int4) + (kUseFP8 ? (kHidden + num_scales * sizeof(float)) : (kHidden * sizeof(nv_bfloat16)));
    const size_t num_int4_per_msg = num_bytes_per_msg / sizeof(int4);
    EP_DEVICE_ASSERT(num_bytes_per_msg % sizeof(int4) == 0);

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_DISPATCH_RECV;

    // Expert counts
    __shared__ int shared_num_tokens_sent_per_expert[kNumWarpGroups];

    if (warp_id < num_warps - 1) {
        constexpr int kNumElemsPerRead = sizeof(int4) / sizeof(nv_bfloat16);
        EP_DEVICE_ASSERT(kHidden % kNumElemsPerRead == 0);
        EP_STATIC_ASSERT(kNumElemsPerRead * 32 % kNumPerChannels == 0, "Invalid vectorization");
        const auto num_threads = (num_warps - 1) * 32;
        const size_t hidden_bf16_int4 = kHidden / kNumElemsPerRead;

        for (int token_idx = sm_id; token_idx < num_tokens; token_idx += num_sms) {
            const auto x_int4 = reinterpret_cast<const int4*>(x) + token_idx * hidden_bf16_int4;
            const auto rdma_x_src_idx = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(rdma_x) + token_idx * num_bytes_per_msg);
            const auto rdma_x_vec = reinterpret_cast<vec_t*>(reinterpret_cast<uint8_t*>(rdma_x_src_idx) + sizeof(int4));
            const auto rdma_x_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(rdma_x_vec) + hidden_bytes);

            auto dst_expert_idx = warp_id < num_topk ? static_cast<int>(__ldg(topk_idx + token_idx * num_topk + warp_id)) : -1;
            thread_id == 0 ? (*rdma_x_src_idx = token_idx) : 0;

            // FP8 cast
            #pragma unroll
            for (int i = thread_id; i < hidden_bf16_int4; i += num_threads) {
                auto int4_value = __ldg(x_int4 + i);

                if (kUseFP8) {
                    auto bf16_values = reinterpret_cast<nv_bfloat16*>(&int4_value);
                    float fp32_values[kNumElemsPerRead];
                    float amax = kFP8Margin, scale, scale_inv;
                    #pragma unroll
                    for (int j = 0; j < kNumElemsPerRead; ++ j) {
                        fp32_values[j] = static_cast<float>(bf16_values[j]);
                        amax = fmaxf(amax, fabsf(fp32_values[j]));
                    }

                    EP_STATIC_ASSERT(kNumElemsPerRead * 32 / kNumPerChannels == 2, "Invalid vectorization");
                    amax = half_warp_reduce_max(amax), scale = kFP8Amax / amax, scale_inv = amax * kFP8AmaxInv;
                    if (lane_id == 0 or lane_id == 16)
                        rdma_x_scales[i * kNumElemsPerRead / 128] = scale_inv;

                    vec_t int2_value;
                    auto fp8x2_values = reinterpret_cast<__nv_fp8x2_storage_t*>(&int2_value);
                    #pragma unroll
                    for (int j = 0; j < kNumElemsPerRead; j += 2) {
                        float2 fp32x2 = {fp32_values[j] * scale, fp32_values[j + 1] * scale};
                        fp8x2_values[j / 2] = __nv_cvt_float2_to_fp8x2(fp32x2, __NV_SATFINITE, __NV_E4M3);
                    }
                    rdma_x_vec[i] = int2_value;
                } else {
                    rdma_x_vec[i] = *reinterpret_cast<vec_t*>(&int4_value);
                }
            }
            asm volatile("bar.sync 1, %0;" :: "r"(num_threads));

            // Issue sends
            if (dst_expert_idx >= 0) {
                int slot_idx = lane_id == 0 ? atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1) : 0;
                slot_idx = __shfl_sync(0xffffffff, slot_idx, 0);
                const auto dst_rank = dst_expert_idx / num_local_experts;
                const auto dst_expert_local_idx = dst_expert_idx % num_local_experts;
                const auto src_ptr = reinterpret_cast<uint64_t>(rdma_x_src_idx);
                const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) +
                                     dst_expert_local_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                                     rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                                     slot_idx * num_bytes_per_msg;
                if (dst_rank != rank) {
                    // MSCCL++ port-channel PUT (lane 0 issues one request).
                    if (lane_id == 0) {
                        const auto dst_off = rdma_offset_of(dst_ptr, rdma_buffer_ptr);
                        const auto src_off = rdma_offset_of(src_ptr, rdma_buffer_ptr);
                        port_channel_handles[dst_expert_local_idx * num_ranks + dst_rank]
                            .put(dst_off, src_off, num_bytes_per_msg);
                    }
                    __syncwarp();
                } else {
                    const auto* src_int4_ptr = reinterpret_cast<const int4*>(src_ptr);
                    const auto* dst_int4_ptr = reinterpret_cast<int4*>(dst_ptr);
                    UNROLLED_WARP_COPY(8, lane_id, num_int4_per_msg, dst_int4_ptr, src_int4_ptr, ld_nc_global, st_na_global);
                }

                __syncwarp();
                lane_id == 0 ? atomic_add_release_global(atomic_finish_counter_per_expert + dst_expert_idx, 1) : 0;
            }
        }
    } else if (warp_id == num_warps - 1) {
        EP_DEVICE_ASSERT(num_sms > 1);
        if (sm_id == 0) {
            // NOTE: DeepEP asserts `ibgda_get_state()->num_rc_per_pe >= num_local_experts`
            // here. The MSCCL++ port relies on Buffer::sync() provisioning enough QPs
            // (see `num_ib_connections_per_rank` / `num_port_channels_per_rank`).

            #pragma unroll
            for (int i = lane_id; i < num_next_clean_int; i += 32)
                next_clean[i] = 0;

            __syncwarp();
            #pragma unroll
            for (int i = lane_id; i < num_experts; i += 32)
                atomic_add_release_global(atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG);
        }

        int expert_count[kNumWarpGroups] = {0};
        const auto expert_begin_idx = sm_id * kNumWarpGroups;
        const auto expert_end_idx = min(expert_begin_idx + kNumWarpGroups, num_experts);

        #pragma unroll 8
        for (int i = lane_id; i < num_tokens * num_topk; i += 32) {
            auto idx = static_cast<int>(__ldg(topk_idx + i));
            if (idx >= expert_begin_idx and idx < expert_end_idx)
                expert_count[idx - expert_begin_idx] ++;
        }

        #pragma unroll
        for (int i = expert_begin_idx; i < expert_end_idx; ++ i) {
            auto sum = warp_reduce_sum(expert_count[i - expert_begin_idx]);
            if (lane_id == 0) {
                shared_num_tokens_sent_per_expert[i - expert_begin_idx] = sum;
                atomic_add_release_global(atomic_finish_counter_per_expert + i, FINISHED_SUM_TAG - sum);
            }
        }
    }
    __syncthreads();

    // Issue count sends
    if (responsible_expert_idx < num_experts and sub_warp_id == 0 and lane_id == 0) {
        const auto dst_rank = responsible_expert_idx / num_local_experts;
        const auto dst_expert_local_idx = responsible_expert_idx % num_local_experts;
        const auto num_tokens_sent = shared_num_tokens_sent_per_expert[responsible_expert_idx - sm_id * kNumWarpGroups];

        while (ld_acquire_global(atomic_finish_counter_per_expert + responsible_expert_idx) != FINISHED_SUM_TAG * 2);
        if (dst_rank != rank) {
            auto* counter_ptr = rdma_recv_count + dst_expert_local_idx * num_ranks + rank;
            const auto off = rdma_offset_of(reinterpret_cast<uint64_t>(counter_ptr), rdma_buffer_ptr);
            port_channel_handles[dst_expert_local_idx * num_ranks + dst_rank]
                .atomicAdd(off, static_cast<int64_t>(-num_tokens_sent - 1));
        } else {
            st_na_release(rdma_recv_count + dst_expert_local_idx * num_ranks + rank, -num_tokens_sent - 1);
        }

        atomic_counter_per_expert[responsible_expert_idx] = 0;
        atomic_finish_counter_per_expert[responsible_expert_idx] = 0;

        if (dst_rank == 0)
            packed_recv_count[dst_expert_local_idx] = 0;
    }
    __syncwarp();

    // Receiving phase
    LOW_LATENCY_DISPATCH_RECV:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0)
        return;

    if (phases & LOW_LATENCY_SEND_PHASE)
        cg::this_grid().sync();

    if (responsible_expert_idx < num_experts) {
        const auto src_rank = responsible_expert_idx / num_local_experts;
        const auto local_expert_idx = responsible_expert_idx % num_local_experts;
        const auto rdma_recv_x_uint8 = reinterpret_cast<uint8_t*>(rdma_recv_x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_msg +
                src_rank * num_max_dispatch_tokens_per_rank * num_bytes_per_msg;
        const auto recv_x_int4 = reinterpret_cast<int4*>(packed_recv_x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_int4;
        const auto recv_x_scales = packed_recv_x_scales + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_scales;
        const auto recv_src_info = packed_recv_src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
        const auto recv_range = packed_recv_layout_range + local_expert_idx * num_ranks;

        __shared__ int shared_num_recv_tokens[kNumWarpGroups], shared_recv_token_begin_idx[kNumWarpGroups];

        int num_recv_tokens, recv_token_begin_idx;
        EP_STATIC_ASSERT(kNumWarpsPerGroup > 1, "Requires more than one warp per group");
        if (sub_warp_id == 1 and lane_id == 0) {
            while ((num_recv_tokens = ld_acquire_sys_global(rdma_recv_count + local_expert_idx * num_ranks + src_rank)) == 0);
            num_recv_tokens = -num_recv_tokens - 1;
            recv_token_begin_idx = atomicAdd(packed_recv_count + local_expert_idx, num_recv_tokens);
            shared_num_recv_tokens[warp_group_id] = num_recv_tokens;
            shared_recv_token_begin_idx[warp_group_id] = recv_token_begin_idx;
            recv_range[src_rank] = pack2<int, int64_t>(num_recv_tokens, recv_token_begin_idx);
        }
        asm volatile("bar.sync %0, %1;" :: "r"(warp_group_id + 2), "r"(kNumWarpsPerGroup * 32));
        num_recv_tokens = shared_num_recv_tokens[warp_group_id];
        recv_token_begin_idx = shared_recv_token_begin_idx[warp_group_id];

        EP_DEVICE_ASSERT(num_scales <= 64);
        for (int i = sub_warp_id; i < num_recv_tokens; i += kNumWarpsPerGroup) {
            const auto src_src_idx = reinterpret_cast<int*>(rdma_recv_x_uint8 + i * num_bytes_per_msg);
            if (lane_id == 0)
                recv_src_info[recv_token_begin_idx + i] = ld_nc_global(src_src_idx);
            __syncwarp();

            const auto src_data = reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(src_src_idx) + sizeof(int4));
            const auto dst_data = recv_x_int4 + (recv_token_begin_idx + i) * hidden_int4;
            UNROLLED_WARP_COPY(7, lane_id, hidden_int4, dst_data, src_data, ld_nc_global, st_na_global);

            if (kUseFP8) {
                const auto src_scales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(src_data) + hidden_bytes);
                const auto dst_scales = reinterpret_cast<float*>(recv_x_scales + recv_token_begin_idx + i);
                const auto scale_stride = num_ranks * num_max_dispatch_tokens_per_rank;
                auto scale_0 = lane_id < num_scales ? ld_nc_global(src_scales + lane_id) : 0;
                auto scale_1 = (lane_id + 32) < num_scales ? ld_nc_global(src_scales + lane_id + 32) : 0;
                lane_id < num_scales ? dst_scales[lane_id * scale_stride] = scale_0 : 0.0f;
                (lane_id + 32) < num_scales ? dst_scales[(lane_id + 32) * scale_stride] = scale_1 : 0.0f;
            }
        }
    }
}

void dispatch(void* packed_recv_x, float* packed_recv_x_scales,
              int* packed_recv_src_info, int64_t* packed_recv_layout_range,
              int* packed_recv_count,
              void* rdma_recv_x, int* rdma_recv_count, void* rdma_x,
              const void* x, const int64_t* topk_idx,
              int* next_clean, int num_next_clean_int,
              int num_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
              int num_topk, int num_experts, int rank, int num_ranks, bool use_fp8,
              void* workspace, cudaStream_t stream, int phases,
              void* rdma_buffer_ptr,
              mscclpp::PortChannelDeviceHandle* port_channel_handles) {
    constexpr int kNumMaxTopK = 9;
    constexpr int kNumWarpsPerGroup = 10;
    constexpr int kNumWarpGroups = 3;
    EP_STATIC_ASSERT(kNumMaxTopK + 1 <= kNumWarpGroups * kNumWarpsPerGroup, "Too many top-k selections");

    const auto num_warps = kNumWarpGroups * kNumWarpsPerGroup;
    const auto num_sms = cell_div(num_experts, kNumWarpGroups);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopK);

    auto atomic_counter_per_expert = reinterpret_cast<int*>(workspace);
    auto atomic_finish_counter_per_expert = atomic_counter_per_expert + num_experts;
    EP_HOST_ASSERT(num_experts * sizeof(int) * 2 <= NUM_WORKSPACE_BYTES);

#define DISPATCH_LAUNCH_CASE(hidden_case) { \
auto dispatch_func = use_fp8 ? dispatch<true, kNumWarpGroups, kNumWarpsPerGroup, hidden_case> : \
                               dispatch<false, kNumWarpGroups, kNumWarpsPerGroup, hidden_case>; \
LAUNCH_KERNEL(&cfg, dispatch_func, \
              packed_recv_x, packed_recv_x_scales, \
              packed_recv_src_info, packed_recv_layout_range, \
              packed_recv_count, \
              rdma_recv_x, rdma_recv_count, rdma_x, \
              x, topk_idx, \
              atomic_counter_per_expert, atomic_finish_counter_per_expert, \
              next_clean, num_next_clean_int, \
              num_tokens, num_max_dispatch_tokens_per_rank, \
              num_topk, num_experts, rank, num_ranks, phases, \
              rdma_buffer_ptr, port_channel_handles); } break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
    SWITCH_HIDDEN(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

// ---------------------------------------------------------------------------
// combine
// ---------------------------------------------------------------------------

template <int kNumWarpGroups, int kNumWarpsPerGroup, int kHidden, int kNumMaxTopk>
__global__ __launch_bounds__(kNumWarpGroups * kNumWarpsPerGroup * 32, 1) void
combine(void* combined_x,
        void* rdma_recv_x, int* rdma_recv_flag, void* rdma_send_x,
        const void* x, const int64_t* topk_idx, const float* topk_weights,
        const int* src_info, const int64_t* layout_range,
        int* next_clean, int num_next_clean_int,
        int* atomic_clean_flag,
        int num_combined_tokens, int hidden, int num_topk,
        int num_max_dispatch_tokens_per_rank,
        int num_experts, int rank, int num_ranks,
        int phases, bool zero_copy,
        void* rdma_buffer_ptr,
        mscclpp::PortChannelDeviceHandle* port_channel_handles) {
    const auto sm_id = static_cast<int>(blockIdx.x);
    const auto num_sms = static_cast<int>(gridDim.x);
    const auto thread_id = static_cast<int>(threadIdx.x);
    const auto num_threads = static_cast<int>(blockDim.x);
    const auto warp_id = thread_id / 32, lane_id = get_lane_id();
    const auto num_local_experts = num_experts / num_ranks;
    const auto warp_group_id = warp_id / kNumWarpsPerGroup;
    const auto sub_warp_id = warp_id % kNumWarpsPerGroup;
    const auto responsible_expert_idx = sm_id * kNumWarpGroups + warp_group_id;

    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
    const size_t hidden_bf16_int4 = kHidden / kNumElemsPerInt4;

    constexpr size_t num_bytes_per_slot = kHidden * sizeof(nv_bfloat16);
    EP_STATIC_ASSERT(num_bytes_per_slot % sizeof(int4) == 0, "Invalid vectorization");

    if ((phases & LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_COMBINE_RECV;

    if (sm_id == 0 and warp_group_id == 0 and sub_warp_id == 0) {
        #pragma unroll
        for (int i = lane_id; i < num_next_clean_int; i += 32)
            next_clean[i] = 0;

        __syncwarp();
        if (lane_id == 0)
            atomic_add_release_global(atomic_clean_flag, num_experts);
    }

    if (responsible_expert_idx < num_experts) {
        const auto dst_rank = responsible_expert_idx / num_local_experts;
        const auto local_expert_idx = responsible_expert_idx % num_local_experts;
        const auto global_expert_idx = rank * num_local_experts + local_expert_idx;
        const auto layout = __ldg(layout_range + local_expert_idx * num_ranks + dst_rank);
        const auto local_x = reinterpret_cast<const int4*>(x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * hidden_bf16_int4;
        const auto local_src_info = src_info + local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank;
        const auto rdma_send_x_vec = reinterpret_cast<uint8_t*>(rdma_send_x) +
                local_expert_idx * num_ranks * num_max_dispatch_tokens_per_rank * num_bytes_per_slot;

        int offset, num_tokens_to_send;
        unpack2(layout, num_tokens_to_send, offset);

        for (int token_idx = offset + sub_warp_id; token_idx < offset + num_tokens_to_send; token_idx += kNumWarpsPerGroup) {
            const auto x_int4 = local_x + token_idx * hidden_bf16_int4;
            const auto rdma_send_type_row = reinterpret_cast<int*>(rdma_send_x_vec + token_idx * num_bytes_per_slot);
            const auto rdma_send_x_vec_row = reinterpret_cast<uint8_t*>(rdma_send_type_row);

            auto src_idx = __ldg(local_src_info + token_idx);
            const auto buf_ptr = reinterpret_cast<int64_t>(rdma_send_x_vec_row);
            const auto dst_ptr = reinterpret_cast<uint64_t>(rdma_recv_x) + (global_expert_idx * num_max_dispatch_tokens_per_rank + src_idx) * num_bytes_per_slot;
            if (dst_rank == rank) {
                const auto dst_int4_ptr = reinterpret_cast<int4*>(dst_ptr);
                UNROLLED_WARP_COPY(7, lane_id, hidden_bf16_int4, dst_int4_ptr, x_int4, ld_nc_global, st_na_global);
            } else {
                const auto buf_int4_ptr = reinterpret_cast<int4*>(buf_ptr);
                if (not zero_copy)
                    UNROLLED_WARP_COPY(7, lane_id, hidden_bf16_int4, buf_int4_ptr, x_int4, ld_nc_global, st_na_global);
                // MSCCL++ port-channel PUT.
                if (lane_id == 0) {
                    const auto dst_off = rdma_offset_of(dst_ptr, rdma_buffer_ptr);
                    const auto src_off = rdma_offset_of(static_cast<uint64_t>(buf_ptr), rdma_buffer_ptr);
                    port_channel_handles[local_expert_idx * num_ranks + dst_rank]
                        .put(dst_off, src_off, hidden * sizeof(nv_bfloat16));
                }
                __syncwarp();
            }
        }

        EP_STATIC_ASSERT(kNumWarpsPerGroup > 1, "Requires more than one warp per group");
        asm volatile("bar.sync %0, %1;" :: "r"(warp_group_id + 1), "r"(kNumWarpsPerGroup * 32));
        if (sub_warp_id == 1 and lane_id == 0) {
            while (ld_acquire_global(atomic_clean_flag) == 0);
            if (dst_rank != rank) {
                auto* flag_ptr = rdma_recv_flag + global_expert_idx;
                const auto off = rdma_offset_of(reinterpret_cast<uint64_t>(flag_ptr), rdma_buffer_ptr);
                port_channel_handles[local_expert_idx * num_ranks + dst_rank]
                    .atomicAdd(off, static_cast<int64_t>(1));
            } else {
                st_na_release(rdma_recv_flag + global_expert_idx, 1);
            }
            atomic_add_release_global(atomic_clean_flag, -1);
        }
        __syncwarp();
    }

    LOW_LATENCY_COMBINE_RECV:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0)
        return;

    if (responsible_expert_idx < num_experts) {
        EP_STATIC_ASSERT(kNumWarpsPerGroup > 1, "Invalid number of warps per group");
        if (sub_warp_id == 0 and lane_id == 0)
            while (ld_acquire_sys_global(rdma_recv_flag + responsible_expert_idx) == 0);
    }
    cg::this_grid().sync();

    EP_DEVICE_ASSERT(num_topk <= 32 and hidden_bf16_int4 <= num_threads);
    EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerInt4) == 0, "Invalid vectorization");
    if (thread_id < hidden_bf16_int4) {
        for (int token_idx = sm_id; token_idx < num_combined_tokens; token_idx += num_sms) {
            int reg_topk_idx[kNumMaxTopk];
            float reg_topk_weights[kNumMaxTopk];
            #pragma unroll
            for (int i = 0; i < num_topk; ++ i) {
                reg_topk_idx[i] = static_cast<int>(__ldg(topk_idx + token_idx * num_topk + i));
                reg_topk_weights[i] = __ldg(topk_weights + token_idx * num_topk + i);
            }

            float combined_values[kNumElemsPerInt4] = {0.0f};
            #pragma unroll
            for (int i = 0; i < num_topk; ++ i) if (reg_topk_idx[i] >= 0) {
                auto rdma_buffer_type = reinterpret_cast<const int*>(reinterpret_cast<uint8_t*>(rdma_recv_x) + (reg_topk_idx[i] * num_max_dispatch_tokens_per_rank + token_idx) * num_bytes_per_slot);
                auto rdma_buffer_row = reinterpret_cast<const uint8_t*>(rdma_buffer_type);

                auto x_vec = ld_nc_global(reinterpret_cast<const int4*>(rdma_buffer_row) + thread_id);
                const auto x_bf16 = reinterpret_cast<nv_bfloat16*>(&x_vec);
                #pragma unroll
                for (int j = 0; j < kNumElemsPerInt4; ++ j)
                    combined_values[j] += static_cast<float>(x_bf16[j]) * reg_topk_weights[i];
            }

            int4& combined_int4 = *reinterpret_cast<int4*>(combined_values);
            auto combined_bf16 = reinterpret_cast<nv_bfloat16*>(&combined_values);
            #pragma unroll
            for (int j = 0; j < kNumElemsPerInt4; ++ j)
                combined_bf16[j] = static_cast<nv_bfloat16>(combined_values[j]);
            (reinterpret_cast<int4*>(combined_x) + token_idx * hidden_bf16_int4)[thread_id] = combined_int4;
        }
    }
}

void combine(void* combined_x,
             void* rdma_recv_x, int* rdma_recv_flag, void* rdma_send_x,
             const void* x, const int64_t* topk_idx, const float* topk_weights,
             const int* src_info, const int64_t* layout_range,
             int* next_clean, int num_next_clean_int,
             int num_combined_tokens, int hidden, int num_max_dispatch_tokens_per_rank,
             int num_topk, int num_experts, int rank, int num_ranks,
             void* workspace, cudaStream_t stream,
             int phases, bool zero_copy,
             void* rdma_buffer_ptr,
             mscclpp::PortChannelDeviceHandle* port_channel_handles) {
    constexpr int kNumWarpsPerGroup = 10;
    constexpr int kNumWarpGroups = 3;
    constexpr int kNumMaxTopk = 9;

    const auto num_warps = kNumWarpGroups * kNumWarpsPerGroup;
    const auto num_sms = cell_div(num_experts, kNumWarpGroups);

    auto atomic_clean_flag = reinterpret_cast<int*>(workspace);
    EP_HOST_ASSERT(sizeof(int) <= NUM_WORKSPACE_BYTES);
    EP_HOST_ASSERT(num_topk <= kNumMaxTopk);

#define COMBINE_LAUNCH_CASE(hidden_case) { \
auto combine_func = combine<kNumWarpGroups, kNumWarpsPerGroup, hidden_case, kNumMaxTopk>; \
LAUNCH_KERNEL(&cfg, combine_func, \
              combined_x, \
              rdma_recv_x, rdma_recv_flag, rdma_send_x, \
              x, topk_idx, topk_weights, src_info, layout_range, \
              next_clean, num_next_clean_int, \
              atomic_clean_flag, \
              num_combined_tokens, hidden, num_topk, \
              num_max_dispatch_tokens_per_rank, \
              num_experts, rank, num_ranks, \
              phases, zero_copy, \
              rdma_buffer_ptr, port_channel_handles); } break

    SETUP_LAUNCH_CONFIG(num_sms, num_warps * 32, stream);
    SWITCH_HIDDEN(COMBINE_LAUNCH_CASE);
#undef COMBINE_LAUNCH_CASE
}

}  // namespace internode_ll
}  // namespace ep
}  // namespace mscclpp
