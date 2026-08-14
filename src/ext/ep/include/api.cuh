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
  /// Token-major rows: [num_recv_tokens, hidden]. High throughput only.
  TOKEN_MAJOR,
  /// Fixed-stride [num_ranks, max_tokens_per_rank, hidden], grouped by source rank.
  RANK_MAJOR
};

namespace dispatch {

/// Compute per-rank/per-expert routing counts and token-to-rank membership.
/// This is a sizing phase and is unrelated to the dispatch output layout.
void prepareTokenMajorOverlap(const int64_t* topkIdx, int* numTokensPerRank, int* numTokensPerExpert,
                              bool* isTokenInRank, int numTokens, int numTopk, int numExperts,
                              const DeviceContext& context, const DeviceContext* deviceContext, cudaStream_t stream);

/// Exchange routing counts, build prefix matrices, and publish receive counts.
/// The host consumes the mapped counters before allocating the dynamic receive view.
void exchangeTokenMajorCountsOverlap(const int* numTokensPerRank, const int* numTokensPerExpert, int numExperts,
                                     int numTokens, const bool* isTokenInRank, int* channelPrefixMatrix,
                                     int* rankPrefixMatrix, int expertAlignment, const DeviceContext& context,
                                     const DeviceContext* deviceContext, cudaStream_t stream, int numChannels);

/// Re-publish a cached rank-prefix matrix and rendezvous before cached dispatch.
void publishCachedTokenMajorPrefixOverlap(const int* rankPrefixMatrix, const DeviceContext& context,
                                          const DeviceContext* deviceContext, cudaStream_t stream);

/// Move token payload and routing metadata after the sizing phase completes.
void tokenMajorOverlap(int* sendHead, const void* input, const int64_t* topkIdx, const float* topkWeights,
                       const float* inputScales, const bool* isTokenInRank, const int* channelPrefixMatrix,
                       int numTokens, int numRecvTokens, int hiddenInt4, int numTopk, int numExperts, int numScales,
                       int64_t* recvTopkIdx, float* recvTopkWeights, float* recvXScales, int numBlocks,
                       int64_t recvPoolHeaderBytes, int64_t recvPoolMetadataOffset, int64_t metadataSlotBytes,
                       const DeviceContext& context, const DeviceContext* deviceContext, cudaStream_t stream);

}  // namespace dispatch

namespace combine {

/// Return expert outputs to their source ranks and reduce routed contributions.
void tokenMajorReduceOverlap(void* output, float* outputTopkWeights, const int* sendHead, int numOutputTokens,
                             int hidden, int numTopk, int64_t recvPoolHeaderBytes, int64_t recvPoolMetadataOffset,
                             int64_t metadataSlotBytes, int numBlocks, const DeviceContext& context,
                             const DeviceContext* deviceContext, cudaStream_t stream);

}  // namespace combine

/// Number of non-worker blocks in the dispatch grid.
inline constexpr int FixedBufferDispatchControlBlocks = 2;
/// Maximum worker blocks used by dispatch or combine.
inline constexpr int FixedBufferMaxWorkerBlocks = 128;
/// Maximum total dispatch grid size.
inline constexpr int FixedBufferMaxDispatchBlocks = FixedBufferMaxWorkerBlocks + FixedBufferDispatchControlBlocks;

/// Fixed-buffer combine algorithm.
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
  /// Sentinel used for token-major entries that do not name a valid expert.
  int invalidTokenExpertId_;
  /// Maximum tokens per rank in the packed layout.
  int maxTokensPerRank_;
  /// User-visible dispatch output layout.
  DispatchLayout outputLayout_;
  /// Dispatch payload data format.
  DispatchDataType dispatchDataType_;
};

size_t fixedBufferWorkspaceSize(int numRanks, int numExperts, int maxTokensPerRank, int numTopk);

namespace dispatch {

/// Latency-optimized dispatch that distributes tokens to experts across ranks.
/// @param[out] output Expert-major or token-major packed output selected by
/// Workload::outputLayout_.
/// @param[out] outputScales Layout-matched FP32 scales for FP8_E4M3, or nullptr for BF16.
/// @param[out] outputSrcInfo Original source-token index for every output row.
/// @param[out] outputTopkIdx Token-major global expert indices [num_ranks * max_tokens_per_rank, num_topk], or nullptr.
/// Non-local and padding entries use Workload::invalidTokenExpertId_.
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
void expertMajorLatency(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                        float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                        const int64_t* topkIdx, const float* topkWeights, const Workload& workload, void* recvBuffer,
                        const DeviceContext& context, const DeviceContext* deviceContext, int numBlocks,
                        cudaStream_t stream);

void rankMajorLatency(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                      float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                      const int64_t* topkIdx, const float* topkWeights, const Workload& workload, void* recvBuffer,
                      const DeviceContext& context, const DeviceContext* deviceContext, int numBlocks,
                      cudaStream_t stream);

}  // namespace dispatch

namespace combine {

/// Latency-optimized combine that aggregates expert outputs back to tokens.
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
void rankLocalReduceLatency(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights,
                            const int* srcInfo, const int64_t* layoutRange, const Workload& workload, void* recvBuffer,
                            void* dispatchRecvBuffer, const DeviceContext& context, const DeviceContext* deviceContext,
                            int numBlocks, cudaStream_t stream);

void directSendLatency(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights,
                       const int* srcInfo, const int64_t* layoutRange, const Workload& workload, void* recvBuffer,
                       void* dispatchRecvBuffer, const DeviceContext& context, const DeviceContext* deviceContext,
                       int numBlocks, cudaStream_t stream);

}  // namespace combine

}  // namespace ep
}  // namespace mscclpp
