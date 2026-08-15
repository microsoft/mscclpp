// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "moe_runtime.hpp"

#include <algorithm>
#include <stdexcept>

#include "exception.cuh"
#include "runtime/resources.hpp"

namespace mscclpp {
namespace ep {

MoERuntime::MoERuntime(mscclpp::Communicator& communicator, MoEMode mode, int maxTokensPerRank, int hidden,
                       int numExperts, int numTopk, int64_t maxHiddenBytes, int numBlocks, DispatchLayout outputLayout)
    : bootstrap_(communicator.bootstrap()),
      mode_(mode),
      rank_(bootstrap_->getRank()),
      numRanks_(bootstrap_->getNranks()),
      numNvlRanks_(std::min(numRanks_, bootstrap_->getNranksPerNode())),
      numRanksPerIpcDomain_(std::max(numNvlRanks_, std::min(numRanks_, bootstrap_->getNranksPerIpcDomain()))) {
  EP_HOST_ASSERT(rank_ >= 0 && rank_ < numRanks_);
  EP_HOST_ASSERT(numNvlRanks_ > 0);

  if (mode_ == MoEMode::LATENCY) {
    fixedBuffer_ = std::make_unique<detail::FixedBufferResources>(communicator, rank_, numRanks_, numNvlRanks_,
                                                                  numRanksPerIpcDomain_, maxTokensPerRank, hidden,
                                                                  numExperts, numTopk, outputLayout);
    available_ = fixedBuffer_->available();
  } else {
    recvPool_ = std::make_unique<detail::RecvPoolResources>(
        communicator, rank_, numRanks_, numNvlRanks_, numRanksPerIpcDomain_, maxHiddenBytes, RecvPoolConfig(numBlocks));
    available_ = recvPool_->available();
  }
}

MoERuntime::~MoERuntime() noexcept(false) = default;

void MoERuntime::requireMode(MoEMode expected) const {
  if (mode_ != expected) {
    throw std::runtime_error(expected == MoEMode::LATENCY ? "MoE runtime was not created with MoEMode::LATENCY"
                                                          : "MoE runtime was not created with MoEMode::OVERLAP");
  }
}

void* MoERuntime::outputTopkIdsBuffer() const {
  requireMode(MoEMode::LATENCY);
  return fixedBuffer_->outputTopkIdsBuffer();
}

void* MoERuntime::outputTopkWeightsBuffer() const {
  requireMode(MoEMode::LATENCY);
  return fixedBuffer_->outputTopkWeightsBuffer();
}

void* MoERuntime::dispatchOutputBuffer() const {
  requireMode(MoEMode::LATENCY);
  return fixedBuffer_->dispatchOutputBuffer();
}

void* MoERuntime::combineInputBuffer() const {
  requireMode(MoEMode::LATENCY);
  return fixedBuffer_->combineInputBuffer();
}

void MoERuntime::latencyDispatch(void* output, void* outputScales, int* outputSrcInfo, int* outputTopkIdx,
                                 float* outputTopkWeights, int64_t* outputLayout, int* outputCount, const void* input,
                                 const int64_t* topkIdx, const float* topkWeights, int numTokens, int hidden,
                                 int numTopk, int maxTokensPerRank, int numExperts, int invalidTokenExpertId,
                                 DispatchLayout dispatchLayout, DispatchDataType dispatchDataType, int numBlocks,
                                 cudaStream_t stream) {
  requireMode(MoEMode::LATENCY);
  fixedBuffer_->dispatch(output, outputScales, outputSrcInfo, outputTopkIdx, outputTopkWeights, outputLayout,
                         outputCount, input, topkIdx, topkWeights, numTokens, hidden, numTopk, maxTokensPerRank,
                         numExperts, invalidTokenExpertId, dispatchLayout, dispatchDataType, numBlocks, stream);
}

void MoERuntime::latencyCombine(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights,
                                const int* srcInfo, const int64_t* layoutRange, int numTokens, int hidden, int numTopk,
                                int maxTokensPerRank, int numExperts, DispatchLayout dispatchLayout,
                                DispatchDataType dispatchDataType, CombineMode mode, int numBlocks,
                                cudaStream_t stream) {
  requireMode(MoEMode::LATENCY);
  fixedBuffer_->combine(output, input, topkIdx, topkWeights, srcInfo, layoutRange, numTokens, hidden, numTopk,
                        maxTokensPerRank, numExperts, dispatchLayout, dispatchDataType, mode, numBlocks, stream);
}

void MoERuntime::tokenMajorPrepare(int* numTokensPerRank, int* numTokensPerExpert, bool* isTokenInRank,
                                   const int64_t* topkIdx, int numTokens, int numTopk, int numExperts,
                                   cudaStream_t stream) {
  requireMode(MoEMode::OVERLAP);
  recvPool_->prepare(numTokensPerRank, numTokensPerExpert, isTokenInRank, topkIdx, numTokens, numTopk, numExperts,
                     stream);
}

int MoERuntime::tokenMajorNumChannels(int xElementSize) const {
  requireMode(MoEMode::OVERLAP);
  return recvPool_->numChannels(xElementSize);
}

void* MoERuntime::tokenMajorResolveRecvBuffer(int numTokens, int numRecvTokens, int hidden, int xElementSize) const {
  requireMode(MoEMode::OVERLAP);
  return recvPool_->resolveRecvBuffer(numTokens, numRecvTokens, hidden, xElementSize);
}

int MoERuntime::tokenMajorNotify(int* rankPrefixMatrix, int* channelPrefixMatrix, int* numRecvTokensPerExpert,
                                 const int* numTokensPerRank, const int* numTokensPerExpert, const bool* isTokenInRank,
                                 int numTokens, int numExperts, int xElementSize, int expertAlignment,
                                 cudaStream_t stream) {
  requireMode(MoEMode::OVERLAP);
  return recvPool_->notify(rankPrefixMatrix, channelPrefixMatrix, numRecvTokensPerExpert, numTokensPerRank,
                           numTokensPerExpert, isTokenInRank, numTokens, numExperts, xElementSize, expertAlignment,
                           stream);
}

void MoERuntime::tokenMajorDispatch(void* recvX, float* recvXScales, int64_t* recvTopkIdx, float* recvTopkWeights,
                                    int* sendHead, const void* x, const float* xScales, const int64_t* topkIdx,
                                    const float* topkWeights, const bool* isTokenInRank, const int* rankPrefixMatrix,
                                    const int* channelPrefixMatrix, int numTokens, int hidden, int numTopk,
                                    int numScales, int numExperts, int xElementSize, int numRecvTokens, bool cachedMode,
                                    cudaStream_t stream) {
  requireMode(MoEMode::OVERLAP);
  recvPool_->dispatch(recvX, recvXScales, recvTopkIdx, recvTopkWeights, sendHead, x, xScales, topkIdx, topkWeights,
                      isTokenInRank, rankPrefixMatrix, channelPrefixMatrix, numTokens, hidden, numTopk, numScales,
                      numExperts, xElementSize, numRecvTokens, cachedMode, stream);
}

void MoERuntime::tokenMajorCombine(void* combinedX, float* combinedTopkWeights, const void* x, const float* topkWeights,
                                   const int* sendHead, int numInputTokens, int numOutputTokens, int hidden,
                                   int numTopk, int xElementSize, cudaStream_t stream) {
  requireMode(MoEMode::OVERLAP);
  recvPool_->combine(combinedX, combinedTopkWeights, x, topkWeights, sendHead, numInputTokens, numOutputTokens, hidden,
                     numTopk, xElementSize, stream);
}

std::shared_ptr<MoERuntime> createMoERuntime(mscclpp::Communicator& communicator, MoEMode mode, int maxTokensPerRank,
                                             int hidden, int numExperts, int numTopk, int64_t maxHiddenBytes,
                                             int numBlocks, DispatchLayout outputLayout) {
  return std::make_shared<MoERuntime>(communicator, mode, maxTokensPerRank, hidden, numExperts, numTopk, maxHiddenBytes,
                                      numBlocks, outputLayout);
}

}  // namespace ep
}  // namespace mscclpp
