// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include "api.cuh"

namespace mscclpp::ep::low_latency::topk_expanded {
// Fixed source-rank/token/top-k layout with BF16 input and BF16 or static-scaled
// E4M3 dispatch output. Combine remains BF16-only.
void dispatch(void* output, int* outputIds, float* outputWeights, int* outputCount, const void* input,
              const int64_t* topkIds, const float* weights, const Workload& work, const CommContext& comm,
              void* workspace, int numBlocks, cudaStream_t stream);
void combine(void* output, const void* input, const int64_t* topkIds, const float* weights, const Workload& work,
             const CommContext& comm, void* workspace, int numBlocks, cudaStream_t stream);
}  // namespace mscclpp::ep::low_latency::topk_expanded
