// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EXT_EP_TYPES_HPP_
#define MSCCLPP_EXT_EP_TYPES_HPP_

#include <cuda_runtime.h>

#include <cstdint>
#include <utility>
#include <variant>

namespace mscclpp {
namespace ep {

class MoERuntime;

/// Expert-parallel runtime mode.
enum class MoEMode {
  /// Algorithms optimized for minimum standalone latency.
  LATENCY,
  /// Resource-bounded algorithms optimized for end-to-end throughput.
  THROUGHPUT
};

/// Logical dispatch output layout.
enum class DispatchLayout {
  /// Rows grouped by local expert.
  EXPERT_MAJOR,
  /// Dynamically sized token-major rows used by throughput mode.
  TOKEN_MAJOR,
  /// Fixed-stride rows grouped by source rank.
  RANK_MAJOR
};

/// Combine algorithm.
enum class CombineMode {
  /// Reduce local expert rows before sending one partial per rank and token.
  RANK_LOCAL_REDUCE,
  /// Reduce all contributions on the source rank.
  ///
  /// Expert-major sends every expert row. Rank-major consumes one weighted
  /// route row per top-k lane and performs the top-k reduction in combine.
  DIRECT_SEND
};

/// Dispatch payload data format.
enum class DispatchDataType {
  /// Unquantized BF16 payload.
  BF16,
  /// FP8 E4M3 payload with one floating-point scale per 128 hidden elements.
  FP8_E4M3
};

/// Arguments for latency-mode dispatch.
struct LatencyDispatchRequest {
  /// Dispatch output buffer.
  void* output;
  /// Optional dispatch scale output.
  void* outputScales;
  /// Optional source-token metadata output.
  int* outputSrcInfo;
  /// Optional dispatched top-k expert IDs.
  int* outputTopkIdx;
  /// Optional dispatched top-k weights.
  float* outputTopkWeights;
  /// Optional packed layout metadata.
  int64_t* outputLayoutRange;
  /// Per-expert or per-rank output counts.
  int* outputCount;
  /// Input token payload.
  const void* input;
  /// Input top-k expert IDs.
  const int64_t* topkIdx;
  /// Optional input top-k weights.
  const float* topkWeights;
  /// Number of input tokens.
  int numTokens;
  /// Hidden dimension.
  int hidden;
  /// Number of routed experts per token.
  int numTopk;
  /// Active per-rank token capacity.
  int maxTokensPerRank;
  /// Global expert count.
  int numExperts;
  /// Expert ID used for invalid rank-major entries.
  int invalidTokenExpertId;
  /// Requested dispatch output layout.
  DispatchLayout dispatchLayout;
  /// Requested dispatch payload format.
  DispatchDataType dispatchDataType;
  /// Dispatch grid block count.
  int numBlocks;
  /// CUDA stream used for the operation.
  cudaStream_t stream;
};

/// Arguments for throughput-mode dispatch.
struct ThroughputDispatchRequest {
  /// Token receive buffer.
  void* recvX;
  /// Optional received scale output.
  float* recvXScales;
  /// Optional received top-k expert IDs.
  int64_t* recvTopkIdx;
  /// Optional received top-k weights.
  float* recvTopkWeights;
  /// Per-token routing state consumed by combine.
  int* sendHead;
  /// Input token payload.
  const void* input;
  /// Optional input scales.
  const float* inputScales;
  /// Optional input top-k expert IDs.
  const int64_t* topkIdx;
  /// Optional input top-k weights.
  const float* topkWeights;
  /// Token-to-destination-rank membership.
  const bool* isTokenInRank;
  /// Per-source-rank token prefixes.
  const int* rankPrefixMatrix;
  /// Per-channel token prefixes.
  const int* channelPrefixMatrix;
  /// Number of input tokens.
  int numTokens;
  /// Hidden dimension.
  int hidden;
  /// Number of routed experts per token.
  int numTopk;
  /// Number of scales per token.
  int numScales;
  /// Global expert count, or zero when cached metadata is reused.
  int numExperts;
  /// Input element size in bytes.
  int inputElementSize;
  /// Number of received tokens.
  int numRecvTokens;
  /// Whether cached routing metadata is reused.
  bool cachedMode;
  /// CUDA stream used for the operation.
  cudaStream_t stream;
};

/// Mode-specific dispatch request.
struct DispatchRequest {
  /// Construct a latency dispatch request.
  explicit DispatchRequest(LatencyDispatchRequest request) : value_(std::move(request)) {}
  /// Construct a throughput dispatch request.
  explicit DispatchRequest(ThroughputDispatchRequest request) : value_(std::move(request)) {}

 private:
  friend class MoERuntime;
  std::variant<LatencyDispatchRequest, ThroughputDispatchRequest> value_;
};

/// Arguments for latency-mode combine.
struct LatencyCombineRequest {
  /// Combined token output.
  void* output;
  /// Local expert output.
  const void* input;
  /// Input top-k expert IDs.
  const int64_t* topkIdx;
  /// Optional input top-k weights.
  const float* topkWeights;
  /// Optional source-token metadata.
  const int* srcInfo;
  /// Optional packed layout metadata.
  const int64_t* layoutRange;
  /// Number of output tokens.
  int numTokens;
  /// Hidden dimension.
  int hidden;
  /// Number of routed experts per token.
  int numTopk;
  /// Active per-rank token capacity.
  int maxTokensPerRank;
  /// Global expert count.
  int numExperts;
  /// Dispatch input layout.
  DispatchLayout dispatchLayout;
  /// Dispatch payload format.
  DispatchDataType dispatchDataType;
  /// Combine algorithm.
  CombineMode combineMode;
  /// Combine worker block count.
  int numBlocks;
  /// CUDA stream used for the operation.
  cudaStream_t stream;
};

/// Arguments for throughput-mode combine.
struct ThroughputCombineRequest {
  /// Combined token output.
  void* output;
  /// Optional combined top-k weights.
  float* outputTopkWeights;
  /// Local expert output.
  const void* input;
  /// Optional local top-k weights.
  const float* topkWeights;
  /// Routing state returned by throughput dispatch.
  const int* sendHead;
  /// Number of local expert-output rows.
  int numInputTokens;
  /// Number of combined output tokens.
  int numOutputTokens;
  /// Hidden dimension.
  int hidden;
  /// Number of routed experts per token.
  int numTopk;
  /// Input element size in bytes.
  int inputElementSize;
  /// CUDA stream used for the operation.
  cudaStream_t stream;
};

/// Mode-specific combine request.
struct CombineRequest {
  /// Construct a latency combine request.
  explicit CombineRequest(LatencyCombineRequest request) : value_(std::move(request)) {}
  /// Construct a throughput combine request.
  explicit CombineRequest(ThroughputCombineRequest request) : value_(std::move(request)) {}

 private:
  friend class MoERuntime;
  std::variant<LatencyCombineRequest, ThroughputCombineRequest> value_;
};

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EXT_EP_TYPES_HPP_
