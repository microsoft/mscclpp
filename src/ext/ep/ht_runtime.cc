// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP)
// branch `chhwang/dev-atomic-add-cleanup`. Licensed under the MIT License.

#include "ht_runtime.hpp"

#include <algorithm>
#include <chrono>
#include <future>
#include <mscclpp/gpu_utils.hpp>
#include <stdexcept>

#include "api.cuh"
#include "constants.cuh"
#include "exception.cuh"

namespace mscclpp {
namespace ep {

MoEHighThroughputRuntime::MoEHighThroughputRuntime(mscclpp::Communicator& communicator, int64_t maxHiddenBytes,
                                                   const high_throughput::Config& config)
    : bootstrap_(communicator.bootstrap()),
      rank_(bootstrap_->getRank()),
      numRanks_(bootstrap_->getNranks()),
      numNvlRanks_(std::min(numRanks_, bootstrap_->getNranksPerNode())),
      numRanksPerIpcDomain_(std::max(numNvlRanks_, std::min(numRanks_, bootstrap_->getNranksPerIpcDomain()))),
      maxHiddenBytes_(maxHiddenBytes),
      config_(config) {
  EP_HOST_ASSERT(rank_ >= 0 && rank_ < numRanks_);
  EP_HOST_ASSERT(numNvlRanks_ > 0);
  EP_HOST_ASSERT(maxHiddenBytes_ > 0);

  if ((numRanks_ != 2 && numRanks_ != 4 && numRanks_ != 8 && numRanks_ != 16) || numRanksPerIpcDomain_ < numRanks_)
    return;

  controlBufferBytes_ = config_.controlBufferBytes(numRanks_);
  taskFifoOffset_ = controlBufferBytes_;
  symmetricBufferBytes_ =
      configAlign<size_t>(taskFifoOffset_ + sizeof(int) * NUM_MAX_FIFO_SLOTS, NUM_BUFFER_ALIGNMENT_BYTES);
  physicalControlBuffer_ = numRanks_ > numNvlRanks_;
  recvPoolBytes_ = high_throughput::Config::recvPoolBytes(numRanks_);
  setup(communicator);
}

MoEHighThroughputRuntime::~MoEHighThroughputRuntime() noexcept(false) {
  if (!available_) return;

  CUDA_CHECK(cudaDeviceSynchronize());
  high_throughput::barrier(taskFifoPtrsGpu_, head_, rank_, numRanks_, nullptr);
  moveFifoSlots();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaFree(combineRecvIdxGpu_));
  CUDA_CHECK(cudaFree(recvPoolPtrsGpu_));
  CUDA_CHECK(cudaFree(taskFifoPtrsGpu_));
  CUDA_CHECK(cudaFree(bufferPtrsGpu_));
  CUDA_CHECK(cudaFreeHost(const_cast<int*>(moeRecvExpertCounter_)));
  CUDA_CHECK(cudaFreeHost(const_cast<int*>(moeRecvCounter_)));

  recvPoolMemories_.clear();
  peerMemories_.clear();
  mscclpp::detail::gpuFreePhysical(recvPool_);
  if (physicalControlBuffer_)
    mscclpp::detail::gpuFreePhysical(symmetricBuffer_);
  else
    CUDA_CHECK(cudaFree(symmetricBuffer_));
}

bool MoEHighThroughputRuntime::isAvailable() const { return available_; }

bool MoEHighThroughputRuntime::isInternodeAvailable() const { return isAvailable() && numRanks_ > numNvlRanks_; }

void MoEHighThroughputRuntime::setup(mscclpp::Communicator& communicator) {
  EP_HOST_ASSERT(!available_);
  if (physicalControlBuffer_) {
    symmetricBuffer_ = mscclpp::detail::gpuCallocPhysical(symmetricBufferBytes_);
  } else {
    CUDA_CHECK(cudaMalloc(&symmetricBuffer_, symmetricBufferBytes_));
    CUDA_CHECK(cudaMemset(symmetricBuffer_, 0, symmetricBufferBytes_));
  }
  recvPool_ = mscclpp::detail::gpuCallocPhysical(recvPoolBytes_);

  constexpr int ControlBufferTag = 17;
  constexpr int RecvPoolTag = 18;
  const auto transport = mscclpp::Transport::CudaIpc;
  peerMemories_.resize(numRanks_);
  peerMemories_[rank_] = communicator.registerMemory(symmetricBuffer_, symmetricBufferBytes_, transport);
  std::vector<std::shared_future<mscclpp::RegisteredMemory>> remoteMemories(numRanks_);
  recvPoolMemories_.resize(numRanks_);
  recvPoolMemories_[rank_] = communicator.registerMemory(recvPool_, recvPoolBytes_, transport);
  std::vector<std::shared_future<mscclpp::RegisteredMemory>> remoteRecvPools(numRanks_);
  for (int peer = 0; peer < numRanks_; ++peer) {
    if (peer == rank_) continue;
    communicator.sendMemory(peerMemories_[rank_], peer, ControlBufferTag);
    remoteMemories[peer] = communicator.recvMemory(peer, ControlBufferTag);
    communicator.sendMemory(recvPoolMemories_[rank_], peer, RecvPoolTag);
    remoteRecvPools[peer] = communicator.recvMemory(peer, RecvPoolTag);
  }

  bufferPtrs_.resize(numRanks_);
  taskFifoPtrs_.resize(numRanks_);
  recvPoolPtrs_.resize(numRanks_);
  for (int peer = 0; peer < numRanks_; ++peer) {
    if (peer != rank_) {
      peerMemories_[peer] = remoteMemories[peer].get();
      recvPoolMemories_[peer] = remoteRecvPools[peer].get();
    }
    void* base = peer == rank_ ? symmetricBuffer_ : peerMemories_[peer].data();
    bufferPtrs_[peer] = base;
    taskFifoPtrs_[peer] = reinterpret_cast<int*>(static_cast<uint8_t*>(base) + taskFifoOffset_);
    recvPoolPtrs_[peer] = peer == rank_ ? recvPool_ : recvPoolMemories_[peer].data();
  }

  CUDA_CHECK(cudaMalloc(&bufferPtrsGpu_, sizeof(void*) * numRanks_));
  CUDA_CHECK(cudaMalloc(&taskFifoPtrsGpu_, sizeof(int*) * numRanks_));
  CUDA_CHECK(cudaMemcpy(bufferPtrsGpu_, bufferPtrs_.data(), sizeof(void*) * numRanks_, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(taskFifoPtrsGpu_, taskFifoPtrs_.data(), sizeof(int*) * numRanks_, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&recvPoolPtrsGpu_, sizeof(void*) * numRanks_));
  CUDA_CHECK(cudaMemcpy(recvPoolPtrsGpu_, recvPoolPtrs_.data(), sizeof(void*) * numRanks_, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMalloc(&combineRecvIdxGpu_,
                        sizeof(int) * static_cast<size_t>(high_throughput::Config::RecvPoolMaxTokens) * numRanks_));
  CUDA_CHECK(cudaMallocHost(&moeRecvCounter_, sizeof(int), cudaHostAllocMapped));
  CUDA_CHECK(cudaHostGetDevicePointer(&moeRecvCounterMapped_, const_cast<int*>(moeRecvCounter_), 0));
  CUDA_CHECK(cudaMallocHost(&moeRecvExpertCounter_, sizeof(int) * NUM_MAX_LOCAL_EXPERTS, cudaHostAllocMapped));
  CUDA_CHECK(cudaHostGetDevicePointer(&moeRecvExpertCounterMapped_, const_cast<int*>(moeRecvExpertCounter_), 0));
  *moeRecvCounter_ = -1;
  for (int i = 0; i < NUM_MAX_LOCAL_EXPERTS; ++i) moeRecvExpertCounter_[i] = -1;
  available_ = true;
}

void MoEHighThroughputRuntime::moveFifoSlots(int numSlots) {
  head_ = (head_ + numRanks_ * numSlots) % NUM_MAX_FIFO_SLOTS;
}

int MoEHighThroughputRuntime::dispatchBlockCount(int xElementSize) const {
  EP_HOST_ASSERT(xElementSize == 2);
  return config_.numSms_;
}

bool MoEHighThroughputRuntime::canUseDirectRecvPool(int numTokens, int numRecvTokens, int hidden,
                                                    int xElementSize) const {
  if (!collectiveDirectReady_ || numTokens < 0 || numRecvTokens < 0 || hidden <= 0 || xElementSize != 2 ||
      numTokens > high_throughput::Config::RecvPoolMaxTokens ||
      numRecvTokens > high_throughput::Config::RecvPoolMaxTokens)
    return false;
  const int64_t hiddenBytes = static_cast<int64_t>(hidden) * xElementSize;
  return hiddenBytes <= maxHiddenBytes_ && static_cast<size_t>(numRecvTokens) * static_cast<size_t>(hiddenBytes) <=
                                               high_throughput::Config::recvPoolHiddenBytes(numRanks_);
}

void MoEHighThroughputRuntime::layout(int* numTokensPerRank, int* numTokensPerExpert, bool* isTokenInRank,
                                      const int64_t* topkIdx, int numTokens, int numTopk, int numExperts,
                                      cudaStream_t stream) {
  EP_HOST_ASSERT(available_);
  EP_HOST_ASSERT(numExperts > 0 && numExperts % numRanks_ == 0);
  EP_HOST_ASSERT(numTopk > 0 && numTopk <= 32);
  high_throughput::getDispatchLayout(topkIdx, numTokensPerRank, numTokensPerExpert, isTokenInRank, numTokens, numTopk,
                                     numRanks_, numExperts, stream);
}

int MoEHighThroughputRuntime::getDispatchNumChannels(int xElementSize) const {
  return dispatchBlockCount(xElementSize);
}

void* MoEHighThroughputRuntime::resolveRecvXBuffer(int numTokens, int numRecvTokens, int hidden,
                                                   int xElementSize) const {
  if (!available_ || !canUseDirectRecvPool(numTokens, numRecvTokens, hidden, xElementSize)) return nullptr;
  return static_cast<uint8_t*>(recvPoolPtrs_[rank_]) + high_throughput::Config::recvPoolHeaderBytes(numRanks_);
}

int MoEHighThroughputRuntime::notifyDispatch(int* rankPrefixMatrix, int* channelPrefixMatrix,
                                             int* numRecvTokensPerExpert, const int* numTokensPerRank,
                                             const int* numTokensPerExpert, const bool* isTokenInRank, int numTokens,
                                             int numExperts, int xElementSize, int expertAlignment,
                                             cudaStream_t stream) {
  EP_HOST_ASSERT(available_);
  EP_HOST_ASSERT(numExperts > 0 && numExperts % numRanks_ == 0);
  const int numLocalExperts = numExperts / numRanks_;
  EP_HOST_ASSERT(numLocalExperts <= NUM_MAX_LOCAL_EXPERTS);

  const int numChannels = dispatchBlockCount(xElementSize);

  *moeRecvCounter_ = -1;
  for (int i = 0; i < numLocalExperts; ++i) moeRecvExpertCounter_[i] = -1;
  high_throughput::notifyDispatch(numTokensPerRank, moeRecvCounterMapped_, numRanks_, numTokensPerExpert,
                                  moeRecvExpertCounterMapped_, numExperts, numTokens, isTokenInRank,
                                  channelPrefixMatrix, rankPrefixMatrix, expertAlignment, bufferPtrsGpu_,
                                  taskFifoPtrsGpu_, head_, rank_, stream, numChannels);
  moveFifoSlots(3);

  int numRecvTokens = -1;
  const auto start = std::chrono::high_resolution_clock::now();
  while (true) {
    numRecvTokens = static_cast<int>(*moeRecvCounter_);
    bool ready = numRecvTokens >= 0;
    for (int i = 0; i < numLocalExperts && ready; ++i) ready &= moeRecvExpertCounter_[i] >= 0;
    if (ready) break;
    if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() >
        NUM_CPU_TIMEOUT_SECS)
      throw std::runtime_error("DeepEP error: CPU recv timeout");
  }
  for (int i = 0; i < numLocalExperts; ++i) numRecvTokensPerExpert[i] = moeRecvExpertCounter_[i];

  const bool localDirectReady = numTokens >= 0 && numTokens <= high_throughput::Config::RecvPoolMaxTokens &&
                                numRecvTokens >= 0 && numRecvTokens <= high_throughput::Config::RecvPoolMaxTokens &&
                                static_cast<size_t>(numRecvTokens) * static_cast<size_t>(maxHiddenBytes_) <=
                                    high_throughput::Config::recvPoolHiddenBytes(numRanks_);
  std::vector<int> directReadyByRank(numRanks_, 0);
  directReadyByRank[rank_] = localDirectReady ? 1 : 0;
  bootstrap_->allGather(directReadyByRank.data(), sizeof(int));
  collectiveDirectReady_ =
      std::all_of(directReadyByRank.begin(), directReadyByRank.end(), [](int ready) { return ready != 0; });
  return numRecvTokens;
}

void MoEHighThroughputRuntime::dispatch(void* recvX, float* recvXScales, int64_t* recvTopkIdx, float* recvTopkWeights,
                                        int* sendHead, const void* x, const float* xScales, const int64_t* topkIdx,
                                        const float* topkWeights, const bool* isTokenInRank,
                                        const int* rankPrefixMatrix, const int* channelPrefixMatrix, int numTokens,
                                        int hidden, int numTopk, int numScales, int numExperts, int xElementSize,
                                        int numRecvTokens, bool cachedMode, cudaStream_t stream) {
  EP_HOST_ASSERT(available_);
  EP_HOST_ASSERT(hidden > 0 && xElementSize == 2);
  EP_HOST_ASSERT(static_cast<int64_t>(hidden) * xElementSize <= maxHiddenBytes_);
  EP_HOST_ASSERT((hidden * xElementSize) % sizeof(int4) == 0);
  EP_HOST_ASSERT(numTopk >= 0 && numTopk <= high_throughput::Config::MaxTopk);
  EP_HOST_ASSERT(numScales >= 0 && numScales <= high_throughput::Config::MaxScales);

  const int numChannels = dispatchBlockCount(xElementSize);
  const int effectiveNumExperts = cachedMode ? 0 : numExperts;
  if (cachedMode) {
    high_throughput::cachedNotifyDispatch(rankPrefixMatrix, bufferPtrsGpu_, taskFifoPtrsGpu_, head_, rank_, numRanks_,
                                          stream);
    moveFifoSlots(2);
  }

  dispatchReady_ = canUseDirectRecvPool(numTokens, numRecvTokens, hidden, xElementSize);
  EP_HOST_ASSERT(dispatchReady_ && "high-throughput direct dispatch capacity exceeded");
  const size_t poolHeaderBytes = high_throughput::Config::recvPoolHeaderBytes(numRanks_);
  EP_HOST_ASSERT(recvX == static_cast<uint8_t*>(recvPoolPtrs_[rank_]) + poolHeaderBytes);

  const int hiddenInt4 = hidden * xElementSize / sizeof(int4);
  dispatchMetadataReady_ = true;
  if (recvTopkWeights != nullptr) recvTopkWeights_ = recvTopkWeights;
  high_throughput::dispatch(sendHead, x, topkIdx, topkWeights, xScales, isTokenInRank, channelPrefixMatrix, numTokens,
                            numRecvTokens, hiddenInt4, numTopk, effectiveNumExperts, numScales, recvTopkIdx,
                            recvTopkWeights, recvXScales, bufferPtrsGpu_, taskFifoPtrsGpu_, head_, rank_, numRanks_,
                            stream, numChannels, recvPoolPtrsGpu_, static_cast<int64_t>(poolHeaderBytes),
                            static_cast<int64_t>(high_throughput::Config::recvPoolMetadataOffset(numRanks_)),
                            high_throughput::Config::RecvPoolMetaBytes, combineRecvIdxGpu_);
  moveFifoSlots();
}

void MoEHighThroughputRuntime::combine(void* combinedX, float* combinedTopkWeights, const void* x,
                                       const float* topkWeights, const int* sendHead, int numInputTokens,
                                       int numOutputTokens, int hidden, int numTopk, int xElementSize,
                                       cudaStream_t stream) {
  EP_HOST_ASSERT(available_);
  EP_HOST_ASSERT(dispatchReady_);
  EP_HOST_ASSERT(xElementSize == 2);
  EP_HOST_ASSERT(static_cast<int64_t>(hidden) * xElementSize <= maxHiddenBytes_);
  EP_HOST_ASSERT((hidden * xElementSize) % sizeof(int4) == 0);
  EP_HOST_ASSERT(numTopk >= 0 && numTopk <= high_throughput::Config::MaxTopk);
  EP_HOST_ASSERT((combinedTopkWeights == nullptr) == (topkWeights == nullptr));

  EP_HOST_ASSERT(numInputTokens <= high_throughput::Config::RecvPoolMaxTokens);
  const size_t recvPoolHeaderBytes = high_throughput::Config::recvPoolHeaderBytes(numRanks_);
  const size_t recvPoolMetadataOffset = high_throughput::Config::recvPoolMetadataOffset(numRanks_);
  auto* localRecvPool = static_cast<uint8_t*>(recvPoolPtrs_[rank_]);
  void* localRecvPoolX = localRecvPool + recvPoolHeaderBytes;
  if (numInputTokens > 0 && x != localRecvPoolX) {
    CUDA_CHECK(cudaMemcpyAsync(localRecvPoolX, x, static_cast<size_t>(numInputTokens) * hidden * xElementSize,
                               cudaMemcpyDeviceToDevice, stream));
  }
  const bool weightsAlreadyStaged = dispatchMetadataReady_ && topkWeights != nullptr && topkWeights == recvTopkWeights_;
  if (numInputTokens > 0 && numTopk > 0 && topkWeights != nullptr && !weightsAlreadyStaged) {
    const size_t weightBytes = static_cast<size_t>(numTopk) * sizeof(float);
    void* localMetadataWeights = localRecvPool + recvPoolMetadataOffset + static_cast<size_t>(numTopk) * sizeof(int);
    CUDA_CHECK(cudaMemcpy2DAsync(localMetadataWeights, high_throughput::Config::RecvPoolMetaBytes, topkWeights,
                                 weightBytes, weightBytes, numInputTokens, cudaMemcpyDeviceToDevice, stream));
  }

  const int numBlocks = config_.numSms_;
  high_throughput::combine(combinedX, combinedTopkWeights, sendHead, numOutputTokens, hidden, numTopk, numRanks_,
                           recvPoolPtrsGpu_, combineRecvIdxGpu_, taskFifoPtrsGpu_, head_, rank_,
                           static_cast<int64_t>(recvPoolHeaderBytes), static_cast<int64_t>(recvPoolMetadataOffset),
                           high_throughput::Config::RecvPoolMetaBytes, numBlocks, stream);
  moveFifoSlots();
}

}  // namespace ep
}  // namespace mscclpp
