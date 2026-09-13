// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP)
//
// Throughput routing-count construction.

#include <cub/block/block_scan.cuh>

#include "exception.hpp"
#include "kernels.hpp"

namespace mscclpp {
namespace ep {
namespace {

struct RankPrefix {
  int total = 0;

  MSCCLPP_DEVICE_INLINE int operator()(int aggregate) {
    const int prefix = total;
    total += aggregate;
    return prefix;
  }
};

}  // namespace

template <int NumThreads, int NumExpertsPerBlock>
__global__ void __launch_bounds__(NumThreads, 1)
    countThroughputRoutesKernel(const int64_t* topkIdx, ThroughputWorkspaceLayout workspace, Workload workload,
                                const DeviceContext* context) {
  const int numTokens = workload.numTokens_;
  const int numTopk = workload.numTopk_;
  const int numExperts = workload.numExperts_;
  const int blockId = static_cast<int>(blockIdx.x);
  const int threadId = static_cast<int>(threadIdx.x);

  using BlockScan = cub::BlockScan<int, NumThreads>;
  union SharedStorage {
    int perExpert[NumThreads][NumExpertsPerBlock];
    typename BlockScan::TempStorage rankScan;
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
      workspace.numTokensPerExpert_[expertBegin + threadId] = sum;
    }
    return;
  }

  const int expertBlocks = (numExperts + NumExpertsPerBlock - 1) / NumExpertsPerBlock;
  const int destinationRank = blockId - expertBlocks;
  if (destinationRank >= context->numRanks_) return;
  const int expertsPerRank = numExperts / context->numRanks_;
  const int rankExpertBegin = destinationRank * expertsPerRank;
  const int rankExpertEnd = rankExpertBegin + expertsPerRank;
  RankPrefix prefix;
  // Count and number each destination's tokens in parallel tiles of the same pass.
  for (int tokenBase = 0; tokenBase < numTokens; tokenBase += NumThreads) {
    const int token = tokenBase + threadId;
    int selected = 0;
    if (token < numTokens) {
#pragma unroll
      for (int topk = 0; topk < numTopk; ++topk) {
        const int expert = static_cast<int>(__ldg(topkIdx + static_cast<size_t>(token) * numTopk + topk));
        selected |= rankExpertBegin <= expert && expert < rankExpertEnd;
      }
    }
    int offset;
    BlockScan(shared.rankScan).ExclusiveSum(selected, offset, prefix);
    if (token < numTokens) {
      workspace.recvTokenOffsets_[static_cast<size_t>(token) * context->numRanks_ + destinationRank] =
          selected ? offset : -1;
    }
    __syncthreads();
  }
  if (threadId == 0) {
    workspace.numTokensPerRank_[destinationRank] = prefix.total;
  }
}

void throughputCountRoutes(const int64_t* topkIdx, const ThroughputWorkspaceLayout& workspace, const Workload& workload,
                           const DeviceContext& context, cudaStream_t stream) {
  constexpr int NumThreads = 256;
  constexpr int NumExpertsPerBlock = 32;
  const int numBlocks = (workload.numExperts_ + NumExpertsPerBlock - 1) / NumExpertsPerBlock + context.numRanks_;

  countThroughputRoutesKernel<NumThreads, NumExpertsPerBlock>
      <<<numBlocks, NumThreads, 0, stream>>>(topkIdx, workspace, workload, context.devicePtr_);
  MSCCLPP_CUDATHROW(cudaGetLastError());
}

}  // namespace ep
}  // namespace mscclpp
