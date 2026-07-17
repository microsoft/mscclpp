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
  /// Direct high-throughput dispatch/combine backend.
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
// High-throughput kernels.
// ===========================================================================
namespace high_throughput {

void getDispatchLayout(const int64_t* topkIdx, int* numTokensPerRank, int* numTokensPerExpert, bool* isTokenInRank,
                       int numTokens, int numTopk, int numRanks, int numExperts, cudaStream_t stream);

void barrier(int** taskFifoPtrs, int head, int rank, int numRanks, cudaStream_t stream);

void notifyDispatch(const int* numTokensPerRank, int* mappedRecvCounter, int numRanks, const int* numTokensPerExpert,
                    int* mappedRecvExpertCounters, int numExperts, int numTokens, const bool* isTokenInRank,
                    int* channelPrefixMatrix, int* rankPrefixMatrix, int expertAlignment, void** bufferPtrs,
                    int** taskFifoPtrs, int head, int rank, cudaStream_t stream, int numChannels);

void cachedNotifyDispatch(const int* rankPrefixMatrix, void** bufferPtrs, int** taskFifoPtrs, int head, int rank,
                          int numRanks, cudaStream_t stream);

void dispatch(int* sendHead, const void* input, const int64_t* topkIdx, const float* topkWeights,
              const float* inputScales, const bool* isTokenInRank, const int* channelPrefixMatrix, int numTokens,
              int numRecvTokens, int hiddenInt4, int numTopk, int numExperts, int numScales, int64_t* recvTopkIdx,
              float* recvTopkWeights, float* recvXScales, void** bufferPtrs, int** taskFifoPtrs, int head, int rank,
              int numRanks, cudaStream_t stream, int numBlocks, void** recvPoolPtrs, int64_t recvPoolHeaderBytes,
              int64_t recvPoolMetadataOffset, int64_t metadataSlotBytes, int* combineRecvIdx);

void combine(void* output, float* outputTopkWeights, const int* sendHead, int numOutputTokens, int hidden, int numTopk,
             int numRanks, void** recvPoolPtrs, const int* combineRecvIdx, int** taskFifoPtrs, int head, int rank,
             int64_t recvPoolHeaderBytes, int64_t recvPoolMetadataOffset, int64_t metadataSlotBytes, int numBlocks,
             cudaStream_t stream);

}  // namespace high_throughput

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
  /// Whether token-major padding metadata is initialized to sentinel values.
  bool initializeTokenMajorPadding_;
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
/// token-major exclusive source-rank offsets [num_ranks + 1].
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
/// @param[in] layoutRange Per-[local expert, source rank] packed count and offset for expert-major input, or
/// token-major exclusive source-rank offsets [num_ranks + 1].
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
