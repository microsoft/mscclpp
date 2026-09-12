// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include "api.cuh"

namespace mscclpp::ep::low_latency::topk_expanded {
// Ported fixed source-rank/token/top-k layout; BF16 and fixed capacity only.
void dispatch(void* output, int* outputIds, float* outputWeights, int* outputCount, const void* input,
              const int64_t* topkIds, const float* weights, const Workload& work, const CommContext& comm,
              void* workspace, int numBlocks, cudaStream_t stream);
void combine(void* output, const void* input, const int64_t* topkIds, const float* weights, const Workload& work,
             const CommContext& comm, void* workspace, int numBlocks, cudaStream_t stream);
}  // namespace mscclpp::ep::low_latency::topk_expanded
