// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP),
// branch `chhwang/dev-atomic-add-cleanup`. Licensed under the MIT License.
//
// Private host-callable API exposed by the EP CUDA kernels. One-to-one port of
// DeepEP `csrc/kernels/api.cuh` minus the NVSHMEM-only internode entrypoints,
// which are still to be migrated.

#pragma once

#include <cuda_runtime.h>
#include <library_types.h>
#include <vector>

#include <mscclpp/memory_channel_device.hpp>
#include <mscclpp/port_channel_device.hpp>

namespace mscclpp {
namespace ep {

// ===========================================================================
// Intranode (NVLink) runtime barrier.
// ===========================================================================
namespace intranode {

void barrier(int** task_fifo_ptrs, int head, int rank, int num_ranks, cudaStream_t stream);

void notify_dispatch(const int* num_tokens_per_rank, int* moe_recv_counter_mapped, int num_ranks,
                     const int* num_tokens_per_expert, int* moe_recv_expert_counter_mapped, int num_experts,
                     int num_tokens, const bool* is_token_in_rank, int* channel_prefix_matrix,
                     int* rank_prefix_matrix_copy, int num_memset_int, int expert_alignment,
                     void** buffer_ptrs, int** task_fifo_ptrs, int head, int rank,
                     cudaStream_t stream, int num_sms);

void cached_notify_dispatch(const int* rank_prefix_matrix, int num_memset_int,
                            void** buffer_ptrs, int** task_fifo_ptrs, int head, int rank, int num_ranks,
                            cudaStream_t stream);

void dispatch(void* recv_x, float* recv_x_scales, int* recv_src_idx, int64_t* recv_topk_idx, float* recv_topk_weights,
              int* recv_channel_offset, int* send_head, const void* x, const float* x_scales, const int64_t* topk_idx,
              const float* topk_weights, const bool* is_token_in_rank, const int* channel_prefix_matrix,
              int num_tokens, int hidden_int4, int num_topk, int num_experts, int num_scales,
              void** buffer_ptrs, int rank, int num_ranks, cudaStream_t stream, int num_sms,
              int num_max_send_tokens, int num_recv_buffer_tokens);

void cached_notify_combine(void** buffer_ptrs, int* send_head, int num_channels, int num_recv_tokens,
                           int num_memset_int, int** task_fifo_ptrs, int head, int rank, int num_ranks,
                           cudaStream_t stream);

void combine(cudaDataType_t type, void* recv_x, float* recv_topk_weights, const void* x, const float* topk_weights,
             const int* src_idx, const int* rank_prefix_matrix, const int* channel_prefix_matrix, int* send_head,
             int num_tokens, int num_recv_tokens, int hidden, int num_topk,
             void** buffer_ptrs, int rank, int num_ranks, cudaStream_t stream, int num_sms,
             int num_max_send_tokens, int num_recv_buffer_tokens);

}  // namespace intranode

// ===========================================================================
// Internode (NVLink + RDMA) high-throughput kernels. Ported from DeepEP
// `csrc/kernels/internode.cu` on branch `chhwang/dev-atomic-add-cleanup`.
// NVSHMEM dependencies in the kernel were replaced with MSCCL++ port-channel
// atomic adds (see src/ext/ep/README.md for the translation table).
// ===========================================================================
namespace internode {

int get_source_meta_bytes();

void get_dispatch_layout(const int64_t* topk_idx, int* num_tokens_per_rank, int* num_tokens_per_rdma_rank,
                         int* num_tokens_per_expert, bool* is_token_in_rank, int num_tokens, int num_topk,
                         int num_ranks, int num_experts, cudaStream_t stream);

void notify_dispatch(const int* num_tokens_per_rank, int* moe_recv_counter_mapped, int num_ranks,
                     const int* num_tokens_per_rdma_rank, int* moe_recv_rdma_counter_mapped,
                     const int* num_tokens_per_expert, int* moe_recv_expert_counter_mapped, int num_experts,
                     const bool* is_token_in_rank, int num_tokens, int num_channels,
                     int hidden_int4, int num_scales, int num_topk, int expert_alignment,
                     int* rdma_channel_prefix_matrix, int* recv_rdma_rank_prefix_sum,
                     int* gbl_channel_prefix_matrix, int* recv_gbl_rank_prefix_sum,
                     void* rdma_buffer_ptr, int num_max_rdma_chunked_recv_tokens,
                     void** buffer_ptrs, int num_max_nvl_chunked_recv_tokens,
                     int** task_fifo_ptrs, int head, int rank,
                     cudaStream_t stream, int64_t num_rdma_bytes, int64_t num_nvl_bytes,
                     bool low_latency_mode,
                     mscclpp::PortChannelDeviceHandle* port_channel_handles,
                     mscclpp::MemoryChannelDeviceHandle* memory_channel_handles);

void dispatch(void* recv_x, float* recv_x_scales, int64_t* recv_topk_idx, float* recv_topk_weights, void* recv_src_meta,
              const void* x, const float* x_scales, const int64_t* topk_idx, const float* topk_weights,
              int* send_rdma_head, int* send_nvl_head,
              int* recv_rdma_channel_prefix_matrix, int* recv_gbl_channel_prefix_matrix,
              const int* rdma_channel_prefix_matrix, const int* recv_rdma_rank_prefix_sum,
              const int* gbl_channel_prefix_matrix, const int* recv_gbl_rank_prefix_sum,
              int num_tokens, int hidden_int4, int num_scales, int num_topk, int num_experts,
              const bool* is_token_in_rank,
              void* rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens,
              void** buffer_ptrs, int num_max_nvl_chunked_send_tokens, int num_max_nvl_chunked_recv_tokens,
              int rank, int num_ranks, bool is_cached_dispatch,
              cudaStream_t stream, int num_channels, bool low_latency_mode,
              mscclpp::PortChannelDeviceHandle* port_channel_handles,
              mscclpp::MemoryChannelDeviceHandle* memory_channel_handles);

void cached_notify(int hidden_int4, int num_scales, int num_topk_idx, int num_topk_weights,
                   int num_ranks, int num_channels, int num_combined_tokens, int* combined_rdma_head,
                   const int* rdma_channel_prefix_matrix, const int* rdma_rank_prefix_sum, int* combined_nvl_head,
                   void* rdma_buffer_ptr, int num_max_rdma_chunked_recv_tokens,
                   void** buffer_ptrs, int num_max_nvl_chunked_recv_tokens,
                   int** task_fifo_ptrs, int head, int rank, cudaStream_t stream,
                   int64_t num_rdma_bytes, int64_t num_nvl_bytes,
                   bool is_cached_dispatch, bool low_latency_mode,
                   mscclpp::PortChannelDeviceHandle* port_channel_handles,
                   mscclpp::MemoryChannelDeviceHandle* memory_channel_handles);

void combine(cudaDataType_t type,
             void* combined_x, float* combined_topk_weights,
             const bool* is_combined_token_in_rank,
             const void* x, const float* topk_weights,
             const int* combined_rdma_head, const int* combined_nvl_head,
             const void* src_meta, const int* rdma_channel_prefix_matrix, const int* rdma_rank_prefix_sum,
             const int* gbl_channel_prefix_matrix,
             int num_tokens, int num_combined_tokens, int hidden, int num_topk,
             void* rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens,
             void** buffer_ptrs, int num_max_nvl_chunked_send_tokens, int num_max_nvl_chunked_recv_tokens,
             int rank, int num_ranks, cudaStream_t stream, int num_channels, bool low_latency_mode,
             mscclpp::PortChannelDeviceHandle* port_channel_handles,
             mscclpp::MemoryChannelDeviceHandle* memory_channel_handles);

}  // namespace internode

// ===========================================================================
// Internode low-latency (pure RDMA) kernels. Ported from DeepEP
// `csrc/kernels/internode_ll.cu` with NVSHMEM/IBGDA device ops replaced by
// MSCCL++ port-channel primitives (`put`, `atomicAdd`, signal/wait barrier).
// ===========================================================================
namespace internode_ll {

void clean_low_latency_buffer(int* clean_0, int num_clean_int_0,
                              int* clean_1, int num_clean_int_1,
                              int rank, int num_ranks,
                              mscclpp::PortChannelDeviceHandle* port_channel_handles,
                              cudaStream_t stream);

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
              mscclpp::PortChannelDeviceHandle* port_channel_handles);

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
             mscclpp::PortChannelDeviceHandle* port_channel_handles);

}  // namespace internode_ll

}  // namespace ep
}  // namespace mscclpp
