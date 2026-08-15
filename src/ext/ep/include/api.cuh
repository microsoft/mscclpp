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

#include "device_context.cuh"

namespace mscclpp {
namespace ep {

/// Expert-parallel backend mode.
enum class MoEMode {
  /// Algorithms optimized for minimum standalone latency.
  LATENCY,
  /// Resource-bounded algorithms optimized for compute/communication overlap.
  OVERLAP
};

/// Logical dispatch output layout.
enum class DispatchLayout {
  /// [num_local_experts, num_ranks * max_tokens_per_rank, hidden].
  EXPERT_MAJOR,
  /// Dynamically sized token-major rows [num_recv_tokens, hidden]. Overlap mode only.
  TOKEN_MAJOR,
  /// Fixed-stride [num_ranks, max_tokens_per_rank, hidden], grouped by source rank.
  RANK_MAJOR
};

namespace dispatch {

/// Build local routing metadata for dynamically sized token-major dispatch.
///
/// Each token is counted once per destination rank even when several selected
/// experts are on that rank. This preparation phase does not move payload data.
/// @param[in] topkIdx Global expert indices [numTokens, numTopk].
/// @param[out] numTokensPerRank Number of local tokens sent to each rank [numRanks].
/// @param[out] numTokensPerExpert Number of local routes for each expert [numExperts].
/// @param[out] isTokenInRank Token-to-destination-rank membership [numTokens, numRanks].
/// @param[in] context Persistent runtime context, including its kernel-visible device pointer.
/// @param[in] stream CUDA stream.
void tokenMajorPrepare(const int64_t* topkIdx, int* numTokensPerRank, int* numTokensPerExpert, bool* isTokenInRank,
                       int numTokens, int numTopk, int numExperts, const DeviceContext& context, cudaStream_t stream);

/// Exchange token-major routing counts and construct send prefixes.
///
/// The kernel publishes total and per-local-expert receive counts to mapped
/// host-visible counters. The host waits for them before exposing the exact-size
/// receive-pool view.
/// @param[in] numTokensPerRank Local destination-rank counts from tokenMajorPrepare().
/// @param[in] numTokensPerExpert Local expert counts from tokenMajorPrepare().
/// @param[in] isTokenInRank Token-to-rank membership from tokenMajorPrepare().
/// @param[out] channelPrefixMatrix Per-rank, per-channel send prefixes.
/// @param[out] rankPrefixMatrix Per-source-rank token prefixes.
/// @param[in] expertAlignment Alignment applied to per-expert receive counts.
void tokenMajorExchangeCounts(const int* numTokensPerRank, const int* numTokensPerExpert, int numExperts, int numTokens,
                              const bool* isTokenInRank, int* channelPrefixMatrix, int* rankPrefixMatrix,
                              int expertAlignment, const DeviceContext& context, cudaStream_t stream, int numChannels);

/// Re-publish cached token-major rank prefixes and rendezvous with peers.
///
/// Use this instead of tokenMajorExchangeCounts() when routing metadata and the
/// receive size are reused from a previous dispatch.
void tokenMajorPublishCachedPrefix(const int* rankPrefixMatrix, const DeviceContext& context, cudaStream_t stream);

/// Dispatch payload and routing metadata into peer token-major receive pools.
///
/// This data-movement phase follows tokenMajorExchangeCounts(), or
/// tokenMajorPublishCachedPrefix() for cached routing.
/// @param[out] sendHead Opaque per-token, per-rank state consumed by tokenMajorReduceCombine().
/// @param[in] input Local token payload [numTokens, hiddenInt4].
/// @param[in] topkIdx Global expert indices, or nullptr in cached mode.
/// @param[in] topkWeights Routing weights, or nullptr when omitted or cached.
/// @param[in] inputScales Optional input scales.
/// @param[in] isTokenInRank Token-to-rank membership from tokenMajorPrepare().
/// @param[in] channelPrefixMatrix Channel prefixes from tokenMajorExchangeCounts().
/// @param[out] recvTopkIdx Received expert indices [numRecvTokens, numTopk].
/// @param[out] recvTopkWeights Received routing weights [numRecvTokens, numTopk].
/// @param[out] recvXScales Optional received scales.
void tokenMajorDispatch(int* sendHead, const void* input, const int64_t* topkIdx, const float* topkWeights,
                        const float* inputScales, const bool* isTokenInRank, const int* channelPrefixMatrix,
                        int numTokens, int numRecvTokens, int hiddenInt4, int numTopk, int numExperts, int numScales,
                        int64_t* recvTopkIdx, float* recvTopkWeights, float* recvXScales, int numBlocks,
                        int64_t recvPoolHeaderBytes, int64_t recvPoolMetadataOffset, int64_t metadataSlotBytes,
                        const DeviceContext& context, cudaStream_t stream);

}  // namespace dispatch

namespace combine {

/// Return token-major expert outputs to source ranks and reduce contributions.
///
/// Expert output and optional weights must be staged in the local receive pool.
/// @param[out] output Combined source tokens [numOutputTokens, hidden].
/// @param[out] outputTopkWeights Optional combined routing weights.
/// @param[in] sendHead Opaque routing state returned by tokenMajorDispatch().
/// @param[in] context Persistent runtime context, including its kernel-visible device pointer.
/// @param[in] stream CUDA stream.
void tokenMajorReduceCombine(void* output, float* outputTopkWeights, const int* sendHead, int numOutputTokens,
                             int hidden, int numTopk, int64_t recvPoolHeaderBytes, int64_t recvPoolMetadataOffset,
                             int64_t metadataSlotBytes, int numBlocks, const DeviceContext& context,
                             cudaStream_t stream);

}  // namespace combine

/// Number of non-worker blocks in the dispatch grid.
inline constexpr int DispatchControlBlocks = 2;
/// Maximum worker blocks used by dispatch or combine.
inline constexpr int MaxWorkerBlocks = 128;
/// Maximum total dispatch grid size.
inline constexpr int MaxDispatchBlocks = MaxWorkerBlocks + DispatchControlBlocks;

/// Combine algorithm.
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
  FP8_E4M3
};

/// Per-call dispatch or combine workload dimensions.
struct Workload {
  /// Number of local input or output tokens.
  int numTokens_;
  /// Hidden dimension size.
  int hidden_;
  /// Number of top-k experts per token.
  int numTopk_;
  /// Total number of experts.
  int numExperts_;
  /// Sentinel used for rank-major padding and non-local expert entries.
  int invalidTokenExpertId_;
  /// Maximum tokens per rank in the packed layout.
  int maxTokensPerRank_;
  /// User-visible dispatch output layout.
  DispatchLayout outputLayout_;
  /// Dispatch payload data format.
  DispatchDataType dispatchDataType_;
};

/// Return workspace bytes required by dispatch and combine.
size_t workspaceSize(int numRanks, int numExperts, int maxTokensPerRank, int numTopk);

namespace dispatch {

/// Dispatch tokens into a fixed-capacity expert-major output.
///
/// Rows are grouped by local expert and packed by source rank. BF16 and FP8
/// E4M3 payloads are supported.
/// @param[out] output Expert-major rows
/// [numLocalExperts, numRanks * maxTokensPerRank, hidden].
/// @param[out] outputScales Layout-matched FP32 scales for FP8_E4M3, or nullptr for BF16.
/// @param[out] outputSrcInfo Original source-token index for every output row.
/// @param[out] outputTopkIdx Unused; must be nullptr.
/// @param[out] outputTopkWeights Unused; must be nullptr.
/// @param[out] outputLayout Packed count and offset for each [local expert, source rank].
/// @param[out] outputCount Valid row count for each local expert.
/// @param[in] input Local input tokens [num_tokens, hidden].
/// @param[in] topkIdx Global expert indices [num_tokens, num_topk].
/// @param[in] topkWeights Routing weights [num_tokens, num_topk], or nullptr for unit weights.
/// @param[in] workload Per-call workload dimensions.
/// @param[in,out] recvBuffer Symmetric scratch buffer for incoming payloads and metadata.
/// @param[in] context Persistent runtime context, including its kernel-visible device pointer.
/// @param[in] numBlocks Total dispatch grid size, including one scheduler and one metadata-notify block.
/// @param[in] stream CUDA stream.
void expertMajorDispatch(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                         float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                         const int64_t* topkIdx, const float* topkWeights, const Workload& workload, void* recvBuffer,
                         const DeviceContext& context, int numBlocks, cudaStream_t stream);

/// Dispatch tokens into a fixed-stride rank-major output.
///
/// Each source token is transferred at most once per destination rank. Rows are
/// grouped by source rank and padded to maxTokensPerRank. The payload format is
/// BF16.
/// @param[out] output Rank-major rows [numRanks * maxTokensPerRank, hidden].
/// @param[out] outputScales Unused; must be nullptr.
/// @param[out] outputSrcInfo Unused; must be nullptr.
/// @param[out] outputTopkIdx Global expert IDs [numRanks * maxTokensPerRank, numTopk].
/// Non-local and padding entries use Workload::invalidTokenExpertId_.
/// @param[out] outputTopkWeights Routing weights with zeros for non-local and padding entries.
/// @param[out] outputLayout Unused; must be nullptr.
/// @param[out] outputCount Valid row count for each source rank [numRanks].
/// @param[in] input Local BF16 tokens [num_tokens, hidden].
/// @param[in] topkIdx Global expert indices [num_tokens, num_topk].
/// @param[in] topkWeights Routing weights, or nullptr for unit weights.
/// @param[in] workload Per-call workload dimensions.
/// @param[in,out] recvBuffer Symmetric metadata and synchronization scratch buffer.
/// @param[in] context Persistent runtime context, including its kernel-visible device pointer.
/// @param[in] numBlocks Total dispatch grid size, including control blocks.
/// @param[in] stream CUDA stream.
void rankMajorDispatch(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                       float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                       const int64_t* topkIdx, const float* topkWeights, const Workload& workload, void* recvBuffer,
                       const DeviceContext& context, int numBlocks, cudaStream_t stream);

}  // namespace dispatch

namespace combine {

/// Reduce expert-major rows locally before returning one partial per rank.
///
/// Each destination rank applies routing weights while reducing its local expert
/// rows, then sends one BF16 partial per source token back to the source rank.
/// @param[out] output Combined local tokens [num_tokens, hidden].
/// @param[in] input Expert-major expert outputs.
/// @param[in] topkIdx Global expert indices [num_tokens, num_topk].
/// @param[in] topkWeights Routing weights, or nullptr for unit weights.
/// @param[in] srcInfo Source-token index for every expert-major row.
/// @param[in] layoutRange Packed count/offset metadata for each [local expert, source rank].
/// @param[in] workload Per-call workload dimensions.
/// @param[in,out] recvBuffer Symmetric scratch buffer receiving partials.
/// @param[in] dispatchRecvBuffer Dispatch metadata rewritten for combine.
/// @param[in] context Persistent runtime context, including its kernel-visible device pointer.
/// @param[in] numBlocks Number of combine blocks.
/// @param[in] stream CUDA stream.
void expertMajorLocalReduceCombine(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights,
                                   const int* srcInfo, const int64_t* layoutRange, const Workload& workload,
                                   void* recvBuffer, void* dispatchRecvBuffer, const DeviceContext& context,
                                   int numBlocks, cudaStream_t stream);

/// Gather rank-major weighted partials from peers and reduce them locally.
///
/// The MLP output already contains one weighted partial per destination rank
/// and source token. Each source rank reads only the peer rows named by its
/// routing metadata and sums them into the final token output.
/// @param[out] output Combined local tokens [num_tokens, hidden].
/// @param[in] input Registered rank-major MLP output.
/// @param[in] topkIdx Global expert indices [num_tokens, num_topk].
/// @param[in] topkWeights Unused; must be nullptr.
/// @param[in] srcInfo Unused; must be nullptr.
/// @param[in] layoutRange Unused; must be nullptr.
/// @param[in] workload Per-call workload dimensions.
/// @param[in,out] recvBuffer Symmetric scratch buffer.
/// @param[in] dispatchRecvBuffer Dispatch metadata rewritten for combine.
/// @param[in] context Persistent runtime context, including its kernel-visible device pointer.
/// @param[in] numBlocks Number of combine worker blocks.
/// @param[in] stream CUDA stream.
void rankMajorGatherReduceCombine(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights,
                                  const int* srcInfo, const int64_t* layoutRange, const Workload& workload,
                                  void* recvBuffer, void* dispatchRecvBuffer, const DeviceContext& context,
                                  int numBlocks, cudaStream_t stream);

/// Send every expert-major row directly and reduce on each source rank.
///
/// This algorithm supports EXPERT_MAJOR input only. Destination ranks send
/// individual expert rows back to the source, where routing weights are applied
/// during the final reduction.
/// @param[out] output Combined local tokens [num_tokens, hidden].
/// @param[in] input Expert-major MLP output.
/// @param[in] topkIdx Global expert indices [num_tokens, num_topk].
/// @param[in] topkWeights Routing weights, or nullptr for unit weights.
/// @param[in] srcInfo Source-token index for every packed expert row.
/// @param[in] layoutRange Packed count and offset for each [local expert, source rank].
/// @param[in] workload Per-call workload dimensions.
/// @param[in,out] recvBuffer Symmetric scratch buffer receiving expert rows.
/// @param[in] dispatchRecvBuffer Dispatch metadata rewritten for combine.
/// @param[in] context Persistent runtime context, including its kernel-visible device pointer.
/// @param[in] numBlocks Number of combine blocks.
/// @param[in] stream CUDA stream.
void expertMajorDirectSendCombine(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights,
                                  const int* srcInfo, const int64_t* layoutRange, const Workload& workload,
                                  void* recvBuffer, void* dispatchRecvBuffer, const DeviceContext& context,
                                  int numBlocks, cudaStream_t stream);

}  // namespace combine

}  // namespace ep
}  // namespace mscclpp
