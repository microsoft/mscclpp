// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
// Private host-callable API of the hybrid NVLink/IPC + GPUNetIO latency kernels.
//
// The hybrid path serves latency RANK_MAJOR (when GPUNetIO connects ranks outside the
// CUDA IPC domain) and RANK_MAJOR_TOPK_EXPANDED. It reuses the shared Workload and
// launch limits from the main EP kernels and keeps its own device context and layout.

#ifndef MSCCLPP_EP_HYBRID_KERNELS_HPP_
#define MSCCLPP_EP_HYBRID_KERNELS_HPP_

#include <cuda_runtime.h>

#include <mscclpp/ext/ep/types.hpp>

#include "hybrid/include/device_context.hpp"
#include "include/kernels.hpp"

namespace mscclpp {
namespace ep {
namespace hybrid {

size_t workspaceSize(int numRanks, int numExperts, int maxTokensPerRank, int numTopk);

namespace topk_expanded {
void dispatch(void* output, int* outputIds, float* outputWeights, int* outputCount, const void* input,
              const int64_t* topkIds, const float* weights, const Workload& work, const DeviceContext& context,
              int numBlocks, cudaStream_t stream);
void combine(void* output, const void* input, const int64_t* topkIds, const float* weights, const Workload& work,
             const DeviceContext& context, int numBlocks, cudaStream_t stream);
}  // namespace topk_expanded

void rankMajorDispatch(void* output, int* outputTopkIdx, float* outputTopkWeights, int* outputCount, const void* input,
                       const int64_t* topkIdx, const float* topkWeights, const Workload& workload, void* recvBuffer,
                       const DeviceContext& context, int numBlocks, cudaStream_t stream);

void rankMajorGatherReduceCombine(void* output, const void* input, const int64_t* topkIdx, const Workload& workload,
                                  void* recvBuffer, void* dispatchRecvBuffer, const DeviceContext& context,
                                  int numBlocks, cudaStream_t stream);

}  // namespace hybrid
}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_HYBRID_KERNELS_HPP_
