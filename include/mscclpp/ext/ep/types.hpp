// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EXT_EP_TYPES_HPP_
#define MSCCLPP_EXT_EP_TYPES_HPP_

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
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
  /// Compact token-major valid rows within throughput's fixed-capacity storage.
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

/// Arguments for throughput-mode routing preparation.
///
/// Preparation computes local routing counts and exchanges them with peers;
/// it does not read or transfer token payloads.
struct PrepareRequest {
  /// Device-resident input top-k expert IDs.
  ///
  /// Keep this buffer alive for the GPU work using it. Its contents must remain
  /// unchanged between preparation and dispatch, unless a graph replay also
  /// recomputes preparation.
  const int64_t* topkIdx;
  /// Number of local input tokens, in [0, maxTokensPerRank].
  int numTokens;
  /// Active per-rank token capacity, positive and no greater than the runtime capacity.
  int maxTokensPerRank;
  /// Requested grid block count for subsequent dispatches.
  int numBlocks;
  /// CUDA stream on which preparation is enqueued asynchronously.
  cudaStream_t stream;
};

/// Opaque routing metadata returned by MoERuntime::prepare().
///
/// A handle borrows runtime-owned routing buffers without extending the
/// runtime's lifetime. It can be reused for unchanged routing until another
/// preparation, including an automatically prepared dispatch, starts on the
/// same runtime. The routing buffers remain private to the runtime. Dispatch
/// inserts a device-side dependency when consuming this handle.
class PrepareHandle {
 public:
  /// Construct an empty handle, which requests automatic preparation in dispatch.
  PrepareHandle() = default;

 private:
  friend class MoERuntime;

  struct Impl;
  explicit PrepareHandle(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}

  std::shared_ptr<const Impl> impl_;
};

/// Arguments for latency-mode dispatch.
///
/// The caller must keep all buffers referenced by this struct (inputs and
/// outputs) valid until the corresponding combine operation has finished.
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
  /// Number of input tokens, in [0, maxTokensPerRank].
  int numTokens;
  /// Active per-rank token capacity, positive and no greater than the runtime capacity.
  int maxTokensPerRank;
  /// Sentinel written to rank-major padding and non-local expert entries.
  /// Defaults to -1.
  /// Here numExperts is the runtime's configured global expert count.
  /// Must be outside [0, numExperts): use a negative int (e.g., -1) or an
  /// int >= numExperts (e.g., numExperts). All ranks must use the same sentinel.
  /// This is an output sentinel; input topkIdx entries must still be valid
  /// global expert IDs or negative values for dropped routes.
  int invalidTokenExpertId = -1;
  /// Requested dispatch payload format.
  DispatchDataType dispatchDataType;
  /// Dispatch grid block count.
  int numBlocks;
  /// CUDA stream used for the operation.
  cudaStream_t stream;
};

/// Arguments for throughput-mode dispatch.
///
/// The caller must keep input and output buffers valid until the GPU work
/// using them, including graph replays, has completed. The returned
/// DispatchHandle must stay alive until the matching combine has been enqueued.
struct ThroughputDispatchRequest {
  /// Dispatch output buffer.
  ///
  /// This may alias MoERuntime::dispatchOutputBuffer() to use the runtime-owned
  /// receive buffer directly.
  void* output;
  /// Optional dispatch scale output.
  void* outputScales;
  /// Optional dispatched local-expert IDs.
  int* outputTopkIdx;
  /// Optional dispatched top-k weights.
  float* outputTopkWeights;
  /// Per-expert or per-rank output counts.
  int* outputCount;
  /// Input token payload.
  const void* input;
  /// Optional input scale factors.
  const float* inputScales;
  /// Input top-k expert IDs.
  const int64_t* topkIdx;
  /// Optional input top-k weights.
  const float* topkWeights;
  /// Number of input tokens, in [0, maxTokensPerRank].
  int numTokens;
  /// Active per-rank token capacity, positive and no greater than the runtime capacity.
  int maxTokensPerRank;
  /// Requested dispatch payload format.
  DispatchDataType dispatchDataType;
  /// Dispatch grid block count.
  int numBlocks;
  /// CUDA stream used for the operation.
  cudaStream_t stream;
  /// Routing metadata returned by MoERuntime::prepare().
  ///
  /// An empty handle requests automatic preparation. For a non-empty handle,
  /// topkIdx, numTokens, maxTokensPerRank, and numBlocks must match the preparation.
  /// All ranks must agree on whether to reuse preparation or compute it automatically.
  /// Routing IDs must remain unchanged; their device contents are not validated
  /// on the host. A graph may either reuse this preparation or capture automatic
  /// preparation with dispatch.
  PrepareHandle prepareHandle;
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

/// Opaque metadata returned by a successful dispatch.
///
/// A handle borrows the dispatch's routing buffers and runtime resources.
class DispatchHandle {
 public:
  /// Construct an empty handle. Passing it to combine raises EPException.
  DispatchHandle() = default;

 private:
  friend class MoERuntime;

  struct Impl;
  explicit DispatchHandle(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}

  std::shared_ptr<const Impl> impl_;
};

/// Arguments for latency-mode combine.
struct LatencyCombineRequest {
  /// Combined token output.
  void* output;
  /// Local expert output.
  const void* input;
  /// Handle returned by the matching dispatch.
  DispatchHandle handle;
  /// Combine grid block count.
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
  /// Local expert output in the dispatch output layout.
  ///
  /// A null pointer is valid only when the device receive count is zero; this
  /// data-dependent condition is checked on the GPU.
  const void* input;
  /// Handle returned by the matching dispatch.
  DispatchHandle handle;
  /// Combine grid block count.
  int numBlocks;
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
