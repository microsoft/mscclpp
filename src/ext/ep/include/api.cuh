// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP),
// branch `chhwang/dev-atomic-add-cleanup`. Licensed under the MIT License.
//
// Private host-callable API exposed by the EP CUDA kernels.

#pragma once

#include <cuda_runtime.h>
#include <library_types.h>

#include <mscclpp/memory_channel_device.hpp>
#include <vector>

namespace mscclpp {
namespace ep {

/// Expert-parallel backend mode.
enum class MoEMode {
  /// Low-latency dispatch/combine backend.
  LOW_LATENCY,
  /// Archived high-throughput backend.
  HIGH_THROUGHPUT
};

/// Logical dispatch output layout.
enum class DispatchLayout {
  /// [num_local_experts, num_ranks * max_tokens_per_rank, hidden].
  EXPERT_MAJOR,
  /// Token-major rows. Low latency uses
  /// [num_ranks * max_tokens_per_rank, hidden], grouped by source rank; high
  /// throughput uses [num_recv_tokens, hidden].
  TOKEN_MAJOR
};

// ===========================================================================
// High-throughput intranode kernels.
// ===========================================================================
namespace intranode {

void get_dispatch_layout(const int64_t* topk_idx, int* num_tokens_per_rank, int* num_tokens_per_expert,
                         bool* is_token_in_rank, int num_tokens, int num_topk, int num_ranks, int num_experts,
                         cudaStream_t stream);

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
              void** recv_pool_ptrs = nullptr, int64_t recv_pool_header_bytes = 0, int* ep_combine_recv_idx = nullptr);

void cached_notify_combine(void** buffer_ptrs, int* send_head, int num_channels, int num_recv_tokens,
                           int num_memset_int, int** task_fifo_ptrs, int head, int rank, int num_ranks,
                           cudaStream_t stream);

void combine(cudaDataType_t type, void* recv_x, float* recv_topk_weights, const void* x, const float* topk_weights,
             const int* src_idx, const int* rank_prefix_matrix, const int* channel_prefix_matrix, int* send_head,
             int num_tokens, int num_recv_tokens, int hidden, int num_topk, void** buffer_ptrs, int rank, int num_ranks,
             cudaStream_t stream, int num_sms, int num_max_send_tokens, int num_recv_buffer_tokens);

// Intranode TMA-staged direct-gather combine (MSCCLPP_EP_COMBINE_TMA). Gathers each
// token's contributions straight from peer recv pools through a cp.async.bulk SMEM
// pipeline; token-parallel grid so combine_sms is independent of dispatch channels.
// Returns false (launches nothing) when recv_pool_ptrs/ep_combine_recv_idx are null,
// so the caller falls back to the 2-hop ring combine.
bool combine_tma(cudaDataType_t type, void* combined_x, float* combined_topk_weights, int* send_head, int num_tokens,
                 int hidden, int num_topk, int num_ranks, void** recv_pool_ptrs, const int* ep_combine_recv_idx,
                 int64_t recv_pool_header_bytes, int combine_sms, cudaStream_t stream);

// All-sender intranode dispatch (MSCCLPP_EP_INTRA_ALLSENDER, INTRA_DIRECT only): every
// block sends hidden directly to the dest pool (num_sms == num_channels, no receiver
// blocks). Metadata lands in the dest pool META region; unpack it with intranode_meta_drain.
void dispatch_allsender(int* send_head, const void* x, const int64_t* topk_idx, const float* topk_weights,
                        const float* x_scales, const bool* is_token_in_rank, const int* channel_prefix_matrix,
                        int num_tokens, int hidden_int4, int num_topk, int num_experts, int num_scales,
                        void** buffer_ptrs, int rank, int num_ranks, cudaStream_t stream, int num_sms,
                        void** recv_pool_ptrs, int64_t recv_pool_header_bytes, int64_t recv_pool_meta_base,
                        int64_t meta_slot_bytes, int* ep_combine_recv_idx);

// Unpack the local recv-pool META region (filled by dispatch_allsender) into the recv_*
// output tensors. recv_topk_idx/recv_x_scales may be null (skipped). One thread per token.
void intranode_meta_drain(void* pool_base, int64_t meta_base, int num_recv_tokens, int* recv_src_idx,
                          int64_t* recv_topk_idx, float* recv_topk_weights, float* recv_x_scales, int num_topk,
                          int num_scales, int64_t meta_slot_bytes, cudaStream_t stream);

}  // namespace intranode

// ===========================================================================
// Low-latency kernels for RDMA and IPC paths. Ported from DeepEP
// `csrc/kernels/internode_ll.cu` with NVSHMEM/IBGDA device ops replaced by
// MSCCL++ channel primitives (`put`, `atomicAdd`, direct IPC stores, barriers).
// ===========================================================================
namespace low_latency {

/// Number of non-worker blocks in the dispatch grid.
inline constexpr int DispatchControlBlocks = 2;
/// Maximum worker blocks used by dispatch or combine.
inline constexpr int MaxWorkerBlocks = 128;
/// Maximum total dispatch grid size.
inline constexpr int MaxDispatchBlocks = MaxWorkerBlocks + DispatchControlBlocks;

/// Low-latency combine algorithm.
enum class CombineMode {
  /// Reduce expert rows on each destination rank before sending one partial per rank and token.
  RANK_LOCAL_REDUCE,
  /// Send every expert row directly and perform the full weighted reduction on the source rank.
  DIRECT_SEND
};

/// Dispatch payload data format.
enum class DispatchDataType {
  /// Unquantized BF16 payload.
  BF16,
  /// FP8 E4M3 payload with one floating-point scale per 128 hidden elements.
  FP8_E4M3,
  /// Reserved for MXFP8 E4M3 payloads with micro-scales.
  MXFP8_E4M3
};

/// Per-call low-latency workload dimensions.
struct Workload {
  /// Number of local input or output tokens.
  int numTokens_;
  /// Hidden dimension size.
  int hidden_;
  /// Number of top-k experts per token.
  int numTopk_;
  /// Total number of experts.
  int numExperts_;
  /// Maximum tokens per rank in the packed layout.
  int maxTokensPerRank_;
  /// User-visible dispatch output layout.
  DispatchLayout outputLayout_;
  /// Dispatch payload data format.
  DispatchDataType dispatchDataType_;
};

/// Persistent communication resources shared by low-latency operations.
struct CommContext {
  /// Base address of the local symmetric communication buffer.
  void* symmetricBufferBase_;
  /// Base memory channel handles used only for signal/wait synchronization.
  mscclpp::BaseMemoryChannelDeviceHandle* baseMemoryChannels_;
  /// Directly mapped symmetric-buffer bases for all participating peers.
  void* const* peerMappedBufferBases_;
  /// Maximum shared memory available to one block after opt-in.
  int maxSharedMemoryPerBlock_;
  /// Number of streaming multiprocessors on the device.
  int numSms_;
  /// CUDA device ID associated with this communicator.
  int deviceId_;
  /// Current rank ID.
  int rank_;
  /// Total number of ranks.
  int numRanks_;
};

/// Return the optimized low-latency workspace size.
/// @param[in] numRanks Total number of ranks.
/// @param[in] numExperts Total number of experts.
/// @return Required workspace bytes.
size_t workspaceSize(int numRanks, int numExperts);

/// Low-latency dispatch that distributes tokens to experts across ranks.
/// @param[out] output Expert-major or token-major packed output selected by
/// Workload::outputLayout_.
/// @param[out] outputScales Layout-matched FP8 block scales, or nullptr for BF16 dispatch.
/// @param[out] outputSrcInfo Original source-token index for every output row.
/// @param[out] outputTopkIdx Token-major local expert indices [num_ranks * max_tokens_per_rank, num_topk], or nullptr.
/// @param[out] outputTopkWeights Token-major routing weights
/// [num_ranks * max_tokens_per_rank, num_topk], or nullptr.
/// @param[out] outputLayout Per-[local expert, source rank] packed count and offset for expert-major output, or
/// nullptr.
/// @param[out] outputCount Per-local-expert counts for expert-major output or per-source-rank counts for token-major.
/// @param[in] input Local input tokens [num_tokens, hidden].
/// @param[in] topkIdx Global expert indices [num_tokens, num_topk].
/// @param[in] topkWeights Routing weights [num_tokens, num_topk], or nullptr for unit weights.
/// @param[in] workload Per-call workload dimensions.
/// @param[in,out] recvBuffer Current symmetric ping-pong buffer used for incoming payloads and rewritten metadata.
/// @param[in] comm Persistent communication context.
/// @param[in,out] workspace Persistent counters, task storage, semaphores, and device barriers.
/// @param[in] numBlocks Total dispatch grid size, including one scheduler and one metadata-notify block.
/// @param[in] stream CUDA stream.
void dispatch(void* output, float* outputScales, int* outputSrcInfo, int* outputTopkIdx, float* outputTopkWeights,
              int64_t* outputLayout, int* outputCount, const void* input, const int64_t* topkIdx,
              const float* topkWeights, const Workload& workload, void* recvBuffer, const CommContext& comm,
              void* workspace, int numBlocks, cudaStream_t stream);

/// Low-latency combine that aggregates expert outputs back to tokens.
/// @param[out] output Combined local tokens [num_tokens, hidden].
/// @param[in] input Expert-major expert outputs or token-major pre-weighted
/// rank-local partials, matching Workload::outputLayout_.
/// @param[in] topkIdx Global expert indices [num_tokens, num_topk].
/// @param[in] topkWeights Routing weights [num_tokens, num_topk], or nullptr for unit weights.
/// @param[in] srcInfo Original source-token index for every packed expert row.
/// @param[in] layoutRange Per-[local expert, source rank] packed count and offset.
/// @param[in] workload Per-call workload dimensions.
/// @param[in,out] recvBuffer Current symmetric ping-pong buffer receiving partials or expert rows.
/// @param[in] dispatchRecvBuffer Previous dispatch buffer containing rewritten routing metadata.
/// @param[in] comm Persistent communication context.
/// @param[in,out] workspace Persistent dispatch metadata plus the combine device barrier.
/// @param[in] numBlocks Number of combine blocks.
/// @param[in] mode Combine algorithm.
/// @param[in] stream CUDA stream.
void combine(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights, const int* srcInfo,
             const int64_t* layoutRange, const Workload& workload, void* recvBuffer, void* dispatchRecvBuffer,
             const CommContext& comm, void* workspace, int numBlocks, CombineMode mode, cudaStream_t stream);

}  // namespace low_latency

}  // namespace ep
}  // namespace mscclpp
