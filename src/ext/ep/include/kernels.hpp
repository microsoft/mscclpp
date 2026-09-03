// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
// Private host-callable API exposed by the EP CUDA kernels.

#ifndef MSCCLPP_EP_KERNELS_HPP_
#define MSCCLPP_EP_KERNELS_HPP_

#include <cuda_runtime.h>

#include <mscclpp/ext/ep/types.hpp>

#include "device_context.hpp"

namespace mscclpp {
namespace ep {

inline constexpr int DispatchControlBlocks = 2;
inline constexpr int MaxWorkerBlocks = 128;
inline constexpr int MaxDispatchBlocks = MaxWorkerBlocks + DispatchControlBlocks;

struct Workload {
  /// Host-assigned epoch shared by the matching dispatch and combine calls.
  uint32_t epoch_;
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

size_t workspaceSize(int numRanks, int numExperts, int maxTokensPerRank, int numTopk);

void expertMajorDispatch(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                         float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                         const int64_t* topkIdx, const float* topkWeights, const Workload& workload, void* recvBuffer,
                         const DeviceContext& context, int numBlocks, cudaStream_t stream);

void rankMajorDispatch(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                       float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                       const int64_t* topkIdx, const float* topkWeights, const Workload& workload, void* recvBuffer,
                       const DeviceContext& context, int numBlocks, cudaStream_t stream);

void expertMajorLocalReduceCombine(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights,
                                   const int* srcInfo, const int64_t* layoutRange, const Workload& workload,
                                   void* recvBuffer, void* dispatchRecvBuffer, const DeviceContext& context,
                                   int numBlocks, cudaStream_t stream);

void rankMajorGatherReduceCombine(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights,
                                  const int* srcInfo, const int64_t* layoutRange, const Workload& workload,
                                  void* recvBuffer, void* dispatchRecvBuffer, const DeviceContext& context,
                                  int numBlocks, cudaStream_t stream);

void rankMajorDirectSendCombine(void* output, const void* input, const int64_t* topkIdx, const Workload& workload,
                                void* recvBuffer, void* dispatchRecvBuffer, const DeviceContext& context, int numBlocks,
                                cudaStream_t stream);

void expertMajorDirectSendCombine(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights,
                                  const int* srcInfo, const int64_t* layoutRange, const Workload& workload,
                                  void* recvBuffer, void* dispatchRecvBuffer, const DeviceContext& context,
                                  int numBlocks, cudaStream_t stream);

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_KERNELS_HPP_
