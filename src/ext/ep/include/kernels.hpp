// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
// Private host-callable API exposed by the EP CUDA kernels.

#ifndef MSCCLPP_EP_KERNELS_HPP_
#define MSCCLPP_EP_KERNELS_HPP_

#include <cuda_runtime.h>

#include <mscclpp/ext/ep/types.hpp>

#include "config.hpp"
#include "device_context.hpp"

namespace mscclpp {
namespace ep {

inline constexpr int ThroughputCountThreads = 128;
inline constexpr int DispatchNWarps = 16;
inline constexpr int DispatchMinNWarpsPerGroup = 8;

MSCCLPP_HOST_DEVICE_INLINE constexpr int dispatchNWarpsPerGroup(int nTokens, int nBlocks) {
  return nTokens <= nBlocks ? DispatchNWarps
                            : (nTokens <= 2 * nBlocks ? DispatchNWarps / 2 : DispatchMinNWarpsPerGroup);
}

inline constexpr int DispatchControlBlocks = 2;
inline constexpr int MaxWorkerBlocks = 128;
inline constexpr int MaxDispatchBlocks = MaxWorkerBlocks + DispatchControlBlocks;
inline constexpr int MaxNumTopk = 8;

inline constexpr bool isSupportedHidden(int hidden) {
  return hidden == 4096 || hidden == 4352 || hidden == 6656 || hidden == 7168 || hidden == 8192 || hidden == 8704 ||
         hidden == 9216;
}

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

struct KernelConfigCache {
  int deviceId_ = -1;
  size_t dynamicSharedBytes_ = 0;
  int residentBlocks_ = 0;
};

template <typename Kernel>
inline int configureKernel(Kernel kernel, int nThreads, size_t dynamicSharedBytes, const DeviceContext& context,
                           KernelConfigCache& cache) {
  if (cache.deviceId_ != context.deviceId_ || cache.dynamicSharedBytes_ < dynamicSharedBytes) {
    cudaFuncAttributes attributes;
    MSCCLPP_CUDATHROW(cudaFuncGetAttributes(&attributes, kernel));
    EP_HOST_ASSERT(dynamicSharedBytes + attributes.sharedSizeBytes <=
                   static_cast<size_t>(context.maxSharedMemoryPerBlock_));
    MSCCLPP_CUDATHROW(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           static_cast<int>(dynamicSharedBytes)));
    int blocksPerSm;
    MSCCLPP_CUDATHROW(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSm, kernel, nThreads, dynamicSharedBytes));
    cache.deviceId_ = context.deviceId_;
    cache.dynamicSharedBytes_ = dynamicSharedBytes;
    cache.residentBlocks_ = blocksPerSm * context.numSms_;
  }
  return cache.residentBlocks_;
}

// Local preparation: count routes and assign stable per-destination token offsets.
void throughputCountRoutes(const int64_t* topkIdx, const ThroughputWorkspaceLayout& workspace, const Workload& workload,
                           const DeviceContext& context, cudaStream_t stream);

// Collective preparation: exchange counts and determine source-rank receive ranges.
void throughputExchangeCounts(const ThroughputWorkspaceLayout& workspace, const Workload& workload,
                              const DeviceContext& context, cudaStream_t stream);

// Wait until peers have finished consuming the previous payload before overwriting it.
void throughputSynchronizePeers(const DeviceContext& context, cudaStream_t stream);

int maxCooperativeThroughputDispatchBlocks(DispatchLayout layout, const DeviceContext& context);

void throughputDispatch(void* output, int* outputTopkIdx, float* outputTopkWeights, float* outputScales,
                        const void* input, const int64_t* topkIdx, const float* topkWeights, const float* inputScales,
                        const Workload& workload, const ThroughputWorkspaceLayout& workspace,
                        const ThroughputPayloadView& payload, void* recvBuffer, const DeviceContext& context,
                        int numBlocks, cudaStream_t stream);

void throughputReduceCombine(void* output, float* outputTopkWeights, const void* input, const Workload& workload,
                             const ThroughputWorkspaceLayout& workspace, const ThroughputPayloadView& payload,
                             void* recvBuffer, const DeviceContext& context, int numBlocks, cudaStream_t stream);

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

void rankMajorGatherReduceCombine(void* output, const void* input, const int64_t* topkIdx, const Workload& workload,
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
