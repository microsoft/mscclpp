// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP)
//
// Throughput routing preparation: token destination maps, stable offsets, and peer count exchange.

#include <cub/block/block_scan.cuh>

#include "common/device_helpers.cuh"
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
    for (int token = threadId; token < numTokens; token += NumThreads) {
      const int64_t* tokenTopk = topkIdx + token * numTopk;
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
  const bool powerOfTwoExperts = (expertsPerRank & (expertsPerRank - 1)) == 0;
  const int expertRankShift = __ffs(expertsPerRank) - 1;
  const uint64_t rankBit = uint64_t{1} << destinationRank;
  RankPrefix prefix;
  // Count and number each destination's tokens in parallel tiles of the same pass.
  for (int tokenBase = 0; tokenBase < numTokens; tokenBase += NumThreads) {
    const int token = tokenBase + threadId;
    uint64_t destinationMask = 0;
    if (token < numTokens) {
      for (int topk = 0; topk < numTopk; ++topk) {
        const int64_t expert = __ldg(topkIdx + static_cast<size_t>(token) * numTopk + topk);
        if (expert >= 0 && expert < numExperts) {
          const int rank = powerOfTwoExperts ? static_cast<int>(expert) >> expertRankShift
                                             : static_cast<int>(expert) / expertsPerRank;
          destinationMask |= uint64_t{1} << rank;
        }
      }
    }
    const int selected = (destinationMask & rankBit) != 0;
    int offset;
    BlockScan(shared.rankScan).ExclusiveSum(selected, offset, prefix);
    if (token < numTokens) {
      auto* routes = workspace.tokenRoutes_ + static_cast<size_t>(token) * numTopk;
      if (selected) {
        // Each destination owns one rank-ordered entry, independent of its top-k positions.
        routes[__popcll(destinationMask & (rankBit - 1))] = {destinationRank, offset};
      }
      if (destinationRank == 0) {
        for (int route = __popcll(destinationMask); route < numTopk; ++route) routes[route] = {-1, -1};
      }
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

__global__ void exchangeThroughputCountsKernel(ThroughputWorkspaceLayout workspace, Workload workload,
                                               const DeviceContext* context) {
  const int numRanks = context->numRanks_;
  const int threadId = static_cast<int>(threadIdx.x);
  blockPeerBarrier(context->channels_, context->rank_, numRanks);

  const int numExpertsPerRank = workload.numExperts_ / numRanks;
  if (threadId < numRanks) {
    auto* peerRankCounts = reinterpret_cast<int*>(context->peerBufferBases_[threadId]);
    auto* peerExpertCounts = peerRankCounts + numRanks * numRanks;
    for (int dstRank = 0; dstRank < numRanks; ++dstRank) {
      peerRankCounts[context->rank_ * numRanks + dstRank] = workspace.numTokensPerRank_[dstRank];
    }
    for (int localExpert = 0; localExpert < numExpertsPerRank; ++localExpert) {
      peerExpertCounts[context->rank_ * numExpertsPerRank + localExpert] =
          workspace.numTokensPerExpert_[threadId * numExpertsPerRank + localExpert];
    }
  }
  __syncthreads();

  blockPeerBarrier(context->channels_, context->rank_, numRanks);

  auto* localRankCounts = reinterpret_cast<int*>(context->peerBufferBases_[context->rank_]);
  if (threadId < numRanks) {
    for (int srcRank = 1; srcRank < numRanks; ++srcRank) {
      localRankCounts[srcRank * numRanks + threadId] += localRankCounts[(srcRank - 1) * numRanks + threadId];
    }
    const int rankPrefix = context->rank_ > 0 ? localRankCounts[(context->rank_ - 1) * numRanks + threadId] : 0;
    workspace.rankOffsets_[threadId] =
        workload.outputLayout_ == DispatchLayout::RANK_MAJOR ? context->rank_ * workload.maxTokensPerRank_ : rankPrefix;
    if (threadId == context->rank_)
      *workspace.numRecvTokens_ = localRankCounts[(numRanks - 1) * numRanks + context->rank_];
  }

  auto* localExpertCounts = localRankCounts + numRanks * numRanks;
  if (threadId < numExpertsPerRank) {
    int count = 0;
    for (int srcRank = 0; srcRank < numRanks; ++srcRank) {
      count += localExpertCounts[srcRank * numExpertsPerRank + threadId];
    }
    if (workload.outputLayout_ == DispatchLayout::TOKEN_MAJOR) workspace.recvCounts_[threadId] = count;
  }
  __syncthreads();

  if (workload.outputLayout_ == DispatchLayout::RANK_MAJOR && threadId < numRanks) {
    const int prefix = localRankCounts[threadId * numRanks + context->rank_];
    const int previous = threadId == 0 ? 0 : localRankCounts[(threadId - 1) * numRanks + context->rank_];
    workspace.recvCounts_[threadId] = prefix - previous;
  }
  // Dispatch and combine use local rank offsets; peers no longer read these prefixes.
}

void throughputExchangeCounts(const ThroughputWorkspaceLayout& workspace, const Workload& workload,
                              const DeviceContext& context, cudaStream_t stream) {
  constexpr int NumThreads = ThroughputCountThreads;
  EP_HOST_ASSERT(workload.numExperts_ % context.numRanks_ == 0);
  EP_HOST_ASSERT(workload.numExperts_ / context.numRanks_ <= NumThreads && context.numRanks_ <= NumThreads);

  exchangeThroughputCountsKernel<<<1, NumThreads, 0, stream>>>(workspace, workload, context.devicePtr_);
  MSCCLPP_CUDATHROW(cudaGetLastError());
}

}  // namespace ep
}  // namespace mscclpp
