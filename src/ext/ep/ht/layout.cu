// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP)
// branch `chhwang/dev-atomic-add-cleanup`. Licensed under the MIT License.
//
// HT routing layout construction.

#include "exception.cuh"
#include "launch.cuh"

namespace mscclpp {
namespace ep {
namespace intranode {

template <int NumThreads, int NumExpertsPerBlock, int NumRanksPerBlock>
__global__ void __launch_bounds__(NumThreads, 1)
    get_dispatch_layout(const int64_t* topk_idx, int* num_tokens_per_rank, int* num_tokens_per_expert,
                        bool* is_token_in_rank, int num_tokens, int num_topk, int num_ranks, int num_experts) {
  const int block_id = static_cast<int>(blockIdx.x);
  const int thread_id = static_cast<int>(threadIdx.x);

  union SharedStorage {
    int perExpert[NumThreads][NumExpertsPerBlock];
    int perRank[NumThreads][NumRanksPerBlock];
  };
  __shared__ SharedStorage shared;
  const int expert_begin = block_id * NumExpertsPerBlock;
  const int expert_end = min(expert_begin + NumExpertsPerBlock, num_experts);
  if (expert_begin < expert_end) {
#pragma unroll
    for (int i = 0; i < NumExpertsPerBlock; ++i) shared.perExpert[thread_id][i] = 0;
#pragma unroll
    for (int token = thread_id; token < num_tokens; token += NumThreads) {
      const int64_t* token_topk = topk_idx + token * num_topk;
#pragma unroll
      for (int i = 0; i < num_topk; ++i) {
        const int expert = static_cast<int>(token_topk[i]);
        if (expert_begin <= expert and expert < expert_end) ++shared.perExpert[thread_id][expert - expert_begin];
      }
    }
    __syncthreads();

    EP_STATIC_ASSERT(NumExpertsPerBlock <= NumThreads, "Too many experts per block");
    if (expert_begin + thread_id < expert_end) {
      int sum = 0;
#pragma unroll
      for (int i = 0; i < NumThreads; ++i) sum += shared.perExpert[i][thread_id];
      num_tokens_per_expert[expert_begin + thread_id] = sum;
    }
    return;
  }

  const int rank_blocks = (num_experts + NumExpertsPerBlock - 1) / NumExpertsPerBlock;
  const int rank_begin = (block_id - rank_blocks) * NumRanksPerBlock;
  const int rank_end = min(rank_begin + NumRanksPerBlock, num_ranks);
  if (rank_begin >= rank_end) return;

  const int experts_per_rank = num_experts / num_ranks;
  const int rank_expert_begin = rank_begin * experts_per_rank;
  const int rank_expert_end = rank_end * experts_per_rank;
#pragma unroll
  for (int i = 0; i < NumRanksPerBlock; ++i) shared.perRank[thread_id][i] = 0;
#pragma unroll
  for (int token = thread_id; token < num_tokens; token += NumThreads) {
    const int64_t* token_topk = topk_idx + token * num_topk;
    int is_in_rank[NumRanksPerBlock] = {0};
#pragma unroll
    for (int i = 0; i < num_topk; ++i) {
      const int expert = static_cast<int>(token_topk[i]);
      if (rank_expert_begin <= expert and expert < rank_expert_end)
        ++is_in_rank[expert / experts_per_rank - rank_begin];
    }

    bool* token_in_rank = is_token_in_rank + token * num_ranks;
#pragma unroll
    for (int i = 0; rank_begin + i < rank_end; ++i) {
      token_in_rank[rank_begin + i] = is_in_rank[i] > 0;
      shared.perRank[thread_id][i] += is_in_rank[i] > 0;
    }
  }
  __syncthreads();

  EP_STATIC_ASSERT(NumRanksPerBlock <= NumThreads, "Too many ranks per block");
  if (rank_begin + thread_id < rank_end) {
    int sum = 0;
#pragma unroll
    for (int i = 0; i < NumThreads; ++i) sum += shared.perRank[i][thread_id];
    num_tokens_per_rank[rank_begin + thread_id] = sum;
  }
}

void get_dispatch_layout(const int64_t* topk_idx, int* num_tokens_per_rank, int* num_tokens_per_expert,
                         bool* is_token_in_rank, int num_tokens, int num_topk, int num_ranks, int num_experts,
                         cudaStream_t stream) {
  constexpr int NumThreads = 256;
  constexpr int NumExpertsPerBlock = 32;
  constexpr int NumRanksPerBlock = 8;
  const int num_blocks = (num_experts + NumExpertsPerBlock - 1) / NumExpertsPerBlock +
                         (num_ranks + NumRanksPerBlock - 1) / NumRanksPerBlock;

  SETUP_LAUNCH_CONFIG(num_blocks, NumThreads, stream);
  LAUNCH_KERNEL(&cfg, (get_dispatch_layout<NumThreads, NumExpertsPerBlock, NumRanksPerBlock>), topk_idx,
                num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank, num_tokens, num_topk, num_ranks,
                num_experts);
}

}  // namespace intranode
}  // namespace ep
}  // namespace mscclpp
