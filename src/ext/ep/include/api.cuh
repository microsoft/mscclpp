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

#include <cstdint>
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
  /// Token-major rows: [num_recv_tokens, hidden]. High throughput only.
  TOKEN_MAJOR,
  /// Fixed-stride [num_ranks, max_tokens_per_rank, hidden], grouped by source rank.
  RANK_MAJOR
};

// ===========================================================================
// High-throughput kernels.
// ===========================================================================
namespace high_throughput {

struct DispatchCountPublication;

/// Compute per-rank/per-expert routing counts and token-to-rank membership.
/// This is a sizing phase and is unrelated to the dispatch output layout.
void computeDispatchCounts(const int64_t* topkIdx, int* numTokensPerRank, int* numTokensPerExpert, bool* isTokenInRank,
                           int numTokens, int numTopk, int numRanks, int numExperts, cudaStream_t stream);

/// Exchange routing counts, build prefix matrices, and publish receive counts.
/// The host consumes only a publication matching expectedGeneration.
void exchangeDispatchCounts(const int* numTokensPerRank, DispatchCountPublication* mappedPublication,
                            uint64_t expectedGeneration, int numRanks,
                            const int* numTokensPerExpert, int numExperts, int numTokens,
                            const bool* isTokenInRank, int* channelPrefixMatrix, int* rankPrefixMatrix,
                            int expertAlignment, void** bufferPtrs,
                            mscclpp::BaseMemoryChannelDeviceHandle* barrierChannels, int rank, cudaStream_t stream,
                            int numChannels);

/// Re-publish a cached rank-prefix matrix and rendezvous before cached dispatch.
void publishCachedRankPrefix(const int* rankPrefixMatrix, void** bufferPtrs,
                             mscclpp::BaseMemoryChannelDeviceHandle* barrierChannels, int rank, int numRanks,
                             cudaStream_t stream);

/// Move token payload and routing metadata after the sizing phase completes.
void dispatch(int* sendHead, const void* input, const int64_t* topkIdx, const float* topkWeights,
              const float* inputScales, const bool* isTokenInRank, const int* channelPrefixMatrix, int numTokens,
              int numRecvTokens, int hiddenInt4, int numTopk, int numExperts, int numScales, int64_t* recvTopkIdx,
              float* recvTopkWeights, float* recvXScales, void** bufferPtrs,
              mscclpp::BaseMemoryChannelDeviceHandle* barrierChannels, int rank, int numRanks, cudaStream_t stream,
              int numBlocks, void** recvPoolPtrs, int64_t recvPoolHeaderBytes, int64_t recvPoolMetadataOffset,
              int64_t metadataSlotBytes, int* combineRecvIdx);

/// Return expert outputs to their source ranks and reduce routed contributions.
void combine(void* output, float* outputTopkWeights, const int* sendHead, int numOutputTokens, int hidden, int numTopk,
             int numRanks, void** recvPoolPtrs, const int* combineRecvIdx,
             mscclpp::BaseMemoryChannelDeviceHandle* barrierChannels, int rank, int64_t recvPoolHeaderBytes,
             int64_t recvPoolMetadataOffset, int64_t metadataSlotBytes, int numBlocks, cudaStream_t stream);

}  // namespace high_throughput

// ===========================================================================
// Low-latency kernels for RDMA and IPC paths. Ported from DeepEP
// `csrc/kernels/internode_ll.cu` with NVSHMEM/IBGDA device ops replaced by
// MSCCL++ channel primitives (`put`, `atomicAdd`, direct IPC stores, barriers).
// ===========================================================================
namespace low_latency {

/// Native FP8 contract consumed by PR40 and Blackwell DeepGEMM.
inline constexpr int Fp8DeepGemmAbi = 1;
inline constexpr int Fp8DeepGemmScaleBlockSize = 128;

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
  FP8_E4M3
};

/// Device-authored receipt. Counters include CUDA graph replays.
struct ExecutionReceipt {
  uint64_t dispatches_;
  uint64_t fp8Dispatches_;
  uint64_t combines_;
  uint32_t lastDispatchEpoch_;
  int32_t abiVersion_;
  int32_t lastHidden_;
  int32_t lastScaleBlockSize_;
  int32_t lastDispatchDataType_;
  uint64_t lastScaleStrideExpert_;
  uint64_t lastScaleStrideToken_;
  uint64_t lastScaleStrideKBlock_;
  int32_t lastScaleContiguous_;
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
  /// Sentinel used for token-major entries that do not name a valid expert.
  int invalidTokenExpertId_;
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
/// @param[in] maxTokensPerRank Maximum local token capacity.
/// @param[in] numTopk Number of routed experts per token.
/// @return Required workspace bytes.
size_t workspaceSize(int numRanks, int numExperts, int maxTokensPerRank, int numTopk);

/// Low-latency dispatch that distributes tokens to experts across ranks.
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
void dispatch(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx, float* outputTopkWeights,
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
