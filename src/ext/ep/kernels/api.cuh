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

#include <mscclpp/memory_channel_device.hpp>
#include <mscclpp/port_channel_device.hpp>
#include <vector>

namespace mscclpp {
namespace ep {

// ===========================================================================
// Archived HT intranode (NVLink) runtime barrier.
// Implementations live under `src/ext/ep/ht/` and are not compiled into the active
// `mscclpp_ep_cpp` target.
// ===========================================================================
namespace intranode {

void barrier(int** task_fifo_ptrs, int head, int rank, int num_ranks, cudaStream_t stream);

void notify_dispatch(const int* num_tokens_per_rank, int* moe_recv_counter_mapped, int num_ranks,
                     const int* num_tokens_per_expert, int* moe_recv_expert_counter_mapped, int num_experts,
                     int num_tokens, const bool* is_token_in_rank, int* channel_prefix_matrix,
                     int* rank_prefix_matrix_copy, int num_memset_int, int expert_alignment, void** buffer_ptrs,
                     int** task_fifo_ptrs, int head, int rank, cudaStream_t stream, int num_sms);

void cached_notify_dispatch(const int* rank_prefix_matrix, int num_memset_int, void** buffer_ptrs, int** task_fifo_ptrs,
                            int head, int rank, int num_ranks, cudaStream_t stream);

void dispatch(void* recv_x, float* recv_x_scales, int* recv_src_idx, int64_t* recv_topk_idx, float* recv_topk_weights,
              int* recv_channel_offset, int* send_head, const void* x, const float* x_scales, const int64_t* topk_idx,
              const float* topk_weights, const bool* is_token_in_rank, const int* channel_prefix_matrix, int num_tokens,
              int hidden_int4, int num_topk, int num_experts, int num_scales, void** buffer_ptrs, int rank,
              int num_ranks, cudaStream_t stream, int num_sms, int num_max_send_tokens, int num_recv_buffer_tokens,
              void** recv_pool_ptrs = nullptr, int64_t recv_pool_header_bytes = 0);

void cached_notify_combine(void** buffer_ptrs, int* send_head, int num_channels, int num_recv_tokens,
                           int num_memset_int, int** task_fifo_ptrs, int head, int rank, int num_ranks,
                           cudaStream_t stream);

void combine(cudaDataType_t type, void* recv_x, float* recv_topk_weights, const void* x, const float* topk_weights,
             const int* src_idx, const int* rank_prefix_matrix, const int* channel_prefix_matrix, int* send_head,
             int num_tokens, int num_recv_tokens, int hidden, int num_topk, void** buffer_ptrs, int rank, int num_ranks,
             cudaStream_t stream, int num_sms, int num_max_send_tokens, int num_recv_buffer_tokens);

}  // namespace intranode

// ===========================================================================
// Archived internode (NVLink + RDMA) high-throughput kernels. Ported from DeepEP
// `csrc/kernels/internode.cu` on branch `chhwang/dev-atomic-add-cleanup`. The
// implementations live under `src/ext/ep/ht/` and are not compiled into the
// active `mscclpp_ep_cpp` target.
// ===========================================================================
namespace internode {

int get_source_meta_bytes();

void get_dispatch_layout(const int64_t* topk_idx, int* num_tokens_per_rank, int* num_tokens_per_rdma_rank,
                         int* num_tokens_per_expert, bool* is_token_in_rank, int num_tokens, int num_topk,
                         int num_ranks, int num_experts, cudaStream_t stream);

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
                     size_t nvls_off_barrier, size_t nvls_off_data, uint64_t nvls_epoch, int nvls_per_peer_bytes);

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
              void* nvls_tail_mc, void* nvls_tail_dev, void* const* peer_rdma_bases,
              // Increment 4: per-peer base pointers of the VMM-allocated recv-output pool
              // (non-null enables cross-GPU forwarder direct-write to recv_x; nullptr = legacy path).
              void* const* recv_pool_ptrs = nullptr,
              // Increment 5 (inc5): domain-wide recv-pool bases indexed by GLOBAL rank
              // (sender direct-write under kEpDirect; nullptr = inactive).
              void* const* recv_pool_global_ptrs = nullptr,
              // Increment 5 combine-direct (Stage 1): per-(token, dst global rank) recv-pool
              // slot index written by the sender; consumed by combine's gather path.
              int* ep_combine_recv_idx = nullptr);

void cached_notify(int hidden_int4, int num_scales, int num_topk_idx, int num_topk_weights, int num_ranks,
                   int num_channels, int num_combined_tokens, int* combined_rdma_head,
                   const int* rdma_channel_prefix_matrix, const int* rdma_rank_prefix_sum, int* combined_nvl_head,
                   void* rdma_buffer_ptr, int num_max_rdma_chunked_recv_tokens, void** buffer_ptrs,
                   int num_max_nvl_chunked_recv_tokens, int** task_fifo_ptrs, int head, int rank, cudaStream_t stream,
                   int64_t num_rdma_bytes, int64_t num_nvl_bytes, bool is_cached_dispatch, bool low_latency_mode,
                   mscclpp::PortChannelDeviceHandle* port_channel_handles,
                   mscclpp::MemoryChannelDeviceHandle* memory_channel_handles, void* nvls_mc_ptr = nullptr,
                   void* nvls_dev_ptr = nullptr, size_t nvls_off_barrier = 0, uint64_t nvls_epoch = 0);

void combine(cudaDataType_t type, void* combined_x, float* combined_topk_weights, const bool* is_combined_token_in_rank,
             const void* x, const float* topk_weights, const int* combined_rdma_head, const int* combined_nvl_head,
             const void* src_meta, const int* rdma_channel_prefix_matrix, const int* rdma_rank_prefix_sum,
             const int* gbl_channel_prefix_matrix, int num_tokens, int num_combined_tokens, int hidden, int num_topk,
             void* rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens,
             void** buffer_ptrs, int num_max_nvl_chunked_send_tokens, int num_max_nvl_chunked_recv_tokens, int rank,
             int num_ranks, cudaStream_t stream, int num_channels, bool low_latency_mode,
             mscclpp::PortChannelDeviceHandle* port_channel_handles,
             mscclpp::MemoryChannelDeviceHandle* memory_channel_handles, void* nvls_head_mc, void* nvls_head_dev,
             void* nvls_tail_mc, void* nvls_tail_dev, void* const* peer_rdma_bases,
             // Increment 5 combine-direct: peer recv-pool bases + dispatch gather map
             // (non-null + kEpDirect => combine gathers from pools; nullptr = legacy 2-hop).
             void* const* recv_pool_global_ptrs = nullptr, const int* ep_combine_recv_idx = nullptr);

}  // namespace internode

// ===========================================================================
// Low-latency kernels for RDMA and IPC paths. Ported from DeepEP
// `csrc/kernels/internode_ll.cu` with NVSHMEM/IBGDA device ops replaced by
// MSCCL++ channel primitives (`put`, `atomicAdd`, direct IPC stores, barriers).
// ===========================================================================
namespace low_latency {

/// Element type used by low-latency dispatch data path.
enum class DType {
  /// NVIDIA bfloat16.
  BF16,
  /// NVIDIA FP8 E4M3.
  F8E4M3
};

/// Logical dispatch output layout. Low-latency mode uses the same contiguous
/// local-expert-major physical order for both layouts; FLAT is a 2D view.
enum class DispatchLayout {
  /// [num_local_experts, num_ranks * max_tokens_per_rank, hidden].
  EXPERT_MAJOR,
  /// [num_local_experts * num_ranks * max_tokens_per_rank, hidden].
  FLAT
};

/// Transport context that encapsulates all transport-related state.
struct TransportContext {
  /// Base address of the locally-registered RDMA buffer.
  void* rdmaBufferBase_;
  /// Port channel device handles for RDMA transport.
  mscclpp::PortChannelDeviceHandle* portChannels_;
  /// Base memory channel handles for IPC transport barrier (nullable).
  mscclpp::BaseMemoryChannelDeviceHandle* memoryChannels_;
  /// Peer-mapped base addresses for IPC path (nullable).
  void* const* peerBases_;
  /// True if IPC path is ready, false to use RDMA.
  bool ipcReady_;
  /// Current rank ID.
  int rank_;
  /// Total number of ranks.
  int numRanks_;
  /// Number of ranks in one IPC-reachable domain (0 when IPC is unavailable).
  int ranksPerIpcDomain_;
};

/// Buffer set that encapsulates ping-pong buffer layout.
struct BufferSet {
  /// Send data buffer.
  void* sendDataBuffer_;
  /// Send count buffer.
  int64_t* sendCountBuffer_;
  /// Receive data buffer.
  void* recvDataBuffer_;
  /// Receive count buffer.
  int64_t* recvCountBuffer_;
  /// Cleanup region for next iteration.
  int64_t* cleanupRegion_;
  /// Size of cleanup region in int64_t elements.
  int cleanupSize_;
};

/// Configuration for dispatch operation.
struct DispatchConfig {
  /// Number of input tokens to dispatch.
  int numTokens_;
  /// Hidden dimension size.
  int hidden_;
  /// Number of top-k experts per token.
  int numTopk_;
  /// Total number of experts.
  int numExperts_;
  /// Maximum tokens per rank in packed layout.
  int numMaxTokensPerRank_;
  /// Input dtype for source tokens.
  DType inputDType_;
  /// Output dtype for packed receive buffer.
  DType outputDType_;
  /// Logical output layout.
  DispatchLayout outputLayout_ = DispatchLayout::EXPERT_MAJOR;
};

/// Configuration for combine operation.
struct CombineConfig {
  /// Number of tokens to combine.
  int numCombinedTokens_;
  /// Hidden dimension size.
  int hidden_;
  /// Number of top-k experts per token.
  int numTopk_;
  /// Total number of experts.
  int numExperts_;
  /// Maximum tokens per rank in packed layout.
  int numMaxTokensPerRank_;
  /// Input dtype for expert outputs.
  DType inputDType_;
  /// Output dtype for combined tokens.
  DType outputDType_;
  /// True to use zero-copy optimization.
  bool zeroCopy_;
};

/// Phase control for send/recv operations.
enum Phase {
  /// Execute send phase only.
  SEND_ONLY = 0x1,
  /// Execute recv phase only.
  RECV_ONLY = 0x2,
  /// Execute both send and recv phases.
  SEND_AND_RECV = 0x3
};

/// Clean low-latency buffers (both ping-pong buffers).
/// @param buffer0 First cleanup region pointer.
/// @param numInt0 Size of first cleanup region in int64_t elements.
/// @param buffer1 Second cleanup region pointer.
/// @param numInt1 Size of second cleanup region in int64_t elements.
/// @param transport Transport context with channel handles and topology info.
/// @param stream CUDA stream to launch the kernel on.
void cleanBuffers(int64_t* buffer0, int numInt0, int64_t* buffer1, int numInt1, const TransportContext& transport,
                  cudaStream_t stream);

/// Low-latency dispatch kernel that distributes tokens to experts across ranks.
/// @param output Output packed data buffer. EXPERT_MAJOR shape is
/// [num_local_experts, num_ranks*max_tokens, hidden]; FLAT is the same
/// local-expert-major storage viewed as [num_local_experts*num_ranks*max_tokens, hidden].
/// @param outputScales FP8 scales (nullable if not using FP8).
/// @param outputSrcInfo Source rank info per token.
/// @param outputLayout Layout range [expert, rank] -> (offset, count).
/// @param outputCount Total count per expert.
/// @param input Input tokens [num_tokens, hidden].
/// @param topkIdx Expert indices [num_tokens, num_topk].
/// @param config Dispatch configuration.
/// @param currentBuffer Current iteration buffer set.
/// @param nextBuffer Next iteration buffer set (for cleanup).
/// @param transport Transport context.
/// @param workspace Temporary workspace buffer.
/// @param stream CUDA stream to launch the kernel on.
/// @param phase Phase control (default: SEND_AND_RECV).
void dispatch(void* output, float* outputScales, int* outputSrcInfo, int64_t* outputLayout, int* outputCount,
              const void* input, const int64_t* topkIdx, const DispatchConfig& config, const BufferSet& currentBuffer,
              const BufferSet& nextBuffer, const TransportContext& transport, void* workspace, cudaStream_t stream,
              Phase phase = SEND_AND_RECV);

/// Low-latency combine kernel that aggregates expert outputs back to tokens.
/// @param output Combined output [num_combined_tokens, hidden].
/// @param input Expert outputs [num_local_experts * num_ranks * max_tokens, hidden].
/// @param inputScales Optional FP8 scales for expert outputs.
/// @param topkIdx Expert indices [num_combined_tokens, num_topk].
/// @param topkWeights Weights [num_combined_tokens, num_topk].
/// @param srcInfo Source rank info.
/// @param layoutRange Layout range.
/// @param config Combine configuration.
/// @param currentBuffer Current iteration buffer set.
/// @param nextBuffer Next iteration buffer set (for cleanup).
/// @param transport Transport context.
/// @param workspace Temporary workspace buffer.
/// @param stream CUDA stream to launch the kernel on.
/// @param phase Phase control (default: SEND_AND_RECV).
void combine(void* output, const void* input, const float* inputScales, const int64_t* topkIdx,
             const float* topkWeights, const int* srcInfo, const int64_t* layoutRange, const CombineConfig& config,
             const BufferSet& currentBuffer, const BufferSet& nextBuffer, const TransportContext& transport,
             void* workspace, cudaStream_t stream, Phase phase = SEND_AND_RECV);

}  // namespace low_latency

}  // namespace ep
}  // namespace mscclpp
