// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP)
// branch `chhwang/dev-atomic-add-cleanup`. Licensed under the MIT License.
//
// High-throughput routing layout construction.

#include "api.cuh"
#include "exception.cuh"
#include "launch.cuh"

namespace mscclpp {
namespace ep {
namespace high_throughput {
namespace detail {

template <int NumThreads, int NumExpertsPerBlock, int NumRanksPerBlock>
__global__ void __launch_bounds__(NumThreads, 1)
    getDispatchLayoutKernel(const int64_t* topkIdx, int* numTokensPerRank, int* numTokensPerExpert, bool* isTokenInRank,
                            int numTokens, int numTopk, int numRanks, int numExperts) {
  const int blockId = static_cast<int>(blockIdx.x);
  const int threadId = static_cast<int>(threadIdx.x);

  union SharedStorage {
    int perExpert[NumThreads][NumExpertsPerBlock];
    int perRank[NumThreads][NumRanksPerBlock];
  };
  __shared__ SharedStorage shared;
  const int expertBegin = blockId * NumExpertsPerBlock;
  const int expertEnd = min(expertBegin + NumExpertsPerBlock, numExperts);
  if (expertBegin < expertEnd) {
#pragma unroll
    for (int i = 0; i < NumExpertsPerBlock; ++i) shared.perExpert[threadId][i] = 0;
#pragma unroll
    for (int token = threadId; token < numTokens; token += NumThreads) {
      const int64_t* tokenTopk = topkIdx + token * numTopk;
#pragma unroll
      for (int i = 0; i < numTopk; ++i) {
        const int expert = static_cast<int>(tokenTopk[i]);
        if (expertBegin <= expert && expert < expertEnd) ++shared.perExpert[threadId][expert - expertBegin];
      }
    }
    __syncthreads();

    EP_STATIC_ASSERT(NumExpertsPerBlock <= NumThreads, "Too many experts per block");
    if (expertBegin + threadId < expertEnd) {
      int sum = 0;
#pragma unroll
      for (int i = 0; i < NumThreads; ++i) sum += shared.perExpert[i][threadId];
      numTokensPerExpert[expertBegin + threadId] = sum;
    }
    return;
  }

  const int rankBlocks = (numExperts + NumExpertsPerBlock - 1) / NumExpertsPerBlock;
  const int rankBegin = (blockId - rankBlocks) * NumRanksPerBlock;
  const int rankEnd = min(rankBegin + NumRanksPerBlock, numRanks);
  if (rankBegin >= rankEnd) return;

  const int expertsPerRank = numExperts / numRanks;
  const int rankExpertBegin = rankBegin * expertsPerRank;
  const int rankExpertEnd = rankEnd * expertsPerRank;
#pragma unroll
  for (int i = 0; i < NumRanksPerBlock; ++i) shared.perRank[threadId][i] = 0;
#pragma unroll
  for (int token = threadId; token < numTokens; token += NumThreads) {
    const int64_t* tokenTopk = topkIdx + token * numTopk;
    int isInRank[NumRanksPerBlock] = {0};
#pragma unroll
    for (int i = 0; i < numTopk; ++i) {
      const int expert = static_cast<int>(tokenTopk[i]);
      if (rankExpertBegin <= expert && expert < rankExpertEnd) ++isInRank[expert / expertsPerRank - rankBegin];
    }

    bool* tokenInRank = isTokenInRank + token * numRanks;
#pragma unroll
    for (int i = 0; rankBegin + i < rankEnd; ++i) {
      tokenInRank[rankBegin + i] = isInRank[i] > 0;
      shared.perRank[threadId][i] += isInRank[i] > 0;
    }
  }
  __syncthreads();

  EP_STATIC_ASSERT(NumRanksPerBlock <= NumThreads, "Too many ranks per block");
  if (rankBegin + threadId < rankEnd) {
    int sum = 0;
#pragma unroll
    for (int i = 0; i < NumThreads; ++i) sum += shared.perRank[i][threadId];
    numTokensPerRank[rankBegin + threadId] = sum;
  }
}

}  // namespace detail

void getDispatchLayout(const int64_t* topkIdx, int* numTokensPerRank, int* numTokensPerExpert, bool* isTokenInRank,
                       int numTokens, int numTopk, int numRanks, int numExperts, cudaStream_t stream) {
  constexpr int NumThreads = 256;
  constexpr int NumExpertsPerBlock = 32;
  constexpr int NumRanksPerBlock = 8;
  const int numBlocks =
      (numExperts + NumExpertsPerBlock - 1) / NumExpertsPerBlock + (numRanks + NumRanksPerBlock - 1) / NumRanksPerBlock;

  SETUP_LAUNCH_CONFIG(numBlocks, NumThreads, stream);
  LAUNCH_KERNEL(&cfg, (detail::getDispatchLayoutKernel<NumThreads, NumExpertsPerBlock, NumRanksPerBlock>), topkIdx,
                numTokensPerRank, numTokensPerExpert, isTokenInRank, numTokens, numTopk, numRanks, numExperts);
}

}  // namespace high_throughput
}  // namespace ep
}  // namespace mscclpp
