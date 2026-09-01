// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP)
// branch `chhwang/dev-atomic-add-cleanup`. Licensed under the MIT License.

#include <algorithm>
#include <chrono>
#include <future>
#include <mscclpp/ext/ep/moe_runtime.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <stdexcept>

#include "exception.hpp"
#include "kernels.hpp"
#include "moe_runtime_context.hpp"

namespace mscclpp {
namespace ep {
namespace {

constexpr auto ReceiveCountTimeout = std::chrono::seconds(100);

}  // namespace

ThroughputContext::ThroughputContext(mscclpp::Communicator& communicator, int rank, int numRanks, int numNvlRanks,
                                     int numRanksPerIpcDomain, int maxTokensPerRank, int64_t maxHiddenBytes,
                                     DispatchLayout outputLayout)
    : rank_(rank),
      numRanks_(numRanks),
      numNvlRanks_(numNvlRanks),
      numRanksPerIpcDomain_(numRanksPerIpcDomain),
      bootstrap_(communicator.bootstrap()),
      maxTokensPerRank_(maxTokensPerRank),
      maxHiddenBytes_(maxHiddenBytes),
      outputLayout_(outputLayout),
      communicator_(&communicator) {
  EP_HOST_ASSERT(maxHiddenBytes_ > 0);
  EP_HOST_ASSERT(outputLayout_ == DispatchLayout::TOKEN_MAJOR || outputLayout_ == DispatchLayout::RANK_MAJOR);
  if (outputLayout_ == DispatchLayout::RANK_MAJOR) EP_HOST_ASSERT(maxTokensPerRank_ > 0);

  if ((numRanks_ != 2 && numRanks_ != 4 && numRanks_ != 8 && numRanks_ != 16) || numRanksPerIpcDomain_ < numRanks_)
    return;

  controlBufferBytes_ = RecvPoolConfig::controlBufferBytes(numRanks_);
  symmetricBufferBytes_ = configAlign<size_t>(controlBufferBytes_, BufferAlignmentBytes);
  physicalControlBuffer_ = numRanks_ > numNvlRanks_;
  recvPoolBytes_ = RecvPoolConfig::recvPoolBytes(numRanks_);
  available_ = true;
}

ThroughputContext::~ThroughputContext() noexcept(false) {
  if (deviceContext_.devicePtr_ == nullptr) return;

  CUDA_CHECK(cudaDeviceSynchronize());
  bootstrap_->barrier();

  CUDA_CHECK(cudaFree(deviceContext_.devicePtr_));
  CUDA_CHECK(cudaFree(combineRecvIdxGpu_));
  CUDA_CHECK(cudaFree(recvPoolPtrsGpu_));
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

void ThroughputContext::initialize() {
  EP_HOST_ASSERT(available_);
  EP_HOST_ASSERT(symmetricBuffer_ == nullptr);
  EP_HOST_ASSERT(communicator_ != nullptr);
  AvoidCudaGraphCaptureGuard captureGuard;
  auto& communicator = *communicator_;
  if (physicalControlBuffer_) {
    symmetricBuffer_ = mscclpp::detail::gpuCallocPhysical(symmetricBufferBytes_);
  } else {
    symmetricBuffer_ = mscclpp::detail::gpuCalloc(symmetricBufferBytes_);
  }
  recvPool_ = mscclpp::detail::gpuCallocPhysical(recvPoolBytes_);

  constexpr int ControlBufferTag = 17;
  constexpr int RecvPoolTag = 18;
  constexpr int BarrierConnectionTag = 19;
  const auto transport = mscclpp::Transport::CudaIpc;
  const mscclpp::EndpointConfig ipcConfig(transport);
  peerMemories_.resize(numRanks_);
  peerMemories_[rank_] = communicator.registerMemory(symmetricBuffer_, symmetricBufferBytes_, transport);
  std::vector<std::shared_future<mscclpp::RegisteredMemory>> remoteMemories(numRanks_);
  recvPoolMemories_.resize(numRanks_);
  recvPoolMemories_[rank_] = communicator.registerMemory(recvPool_, recvPoolBytes_, transport);
  std::vector<std::shared_future<mscclpp::RegisteredMemory>> remoteRecvPools(numRanks_);
  std::vector<std::shared_future<mscclpp::Connection>> barrierConnections(numRanks_);
  for (int peer = 0; peer < numRanks_; ++peer) {
    if (peer == rank_) continue;
    communicator.sendMemory(peerMemories_[rank_], peer, ControlBufferTag);
    remoteMemories[peer] = communicator.recvMemory(peer, ControlBufferTag);
    communicator.sendMemory(recvPoolMemories_[rank_], peer, RecvPoolTag);
    remoteRecvPools[peer] = communicator.recvMemory(peer, RecvPoolTag);
    barrierConnections[peer] = communicator.connect(ipcConfig, peer, BarrierConnectionTag);
  }

  bufferPtrs_.resize(numRanks_);
  recvPoolPtrs_.resize(numRanks_);
  barrierChannels_.reserve(numRanks_ - 1);
  std::vector<mscclpp::BaseMemoryChannelDeviceHandle> barrierChannelHandles(numRanks_);
  for (int peer = 0; peer < numRanks_; ++peer) {
    if (peer != rank_) {
      peerMemories_[peer] = remoteMemories[peer].get();
      recvPoolMemories_[peer] = remoteRecvPools[peer].get();
    }
    void* base = peer == rank_ ? symmetricBuffer_ : peerMemories_[peer].data();
    bufferPtrs_[peer] = base;
    recvPoolPtrs_[peer] = peer == rank_ ? recvPool_ : recvPoolMemories_[peer].data();
    if (peer != rank_) {
      auto semaphore =
          std::make_shared<mscclpp::MemoryDevice2DeviceSemaphore>(communicator, barrierConnections[peer].get());
      barrierChannels_.emplace_back(semaphore);
      barrierChannelHandles[peer] = barrierChannels_.back().deviceHandle();
    }
  }

  bufferPtrsGpu_ = static_cast<void**>(mscclpp::detail::gpuCalloc(sizeof(void*) * static_cast<size_t>(numRanks_)));
  mscclpp::gpuMemcpy<void*>(bufferPtrsGpu_, bufferPtrs_.data(), numRanks_, cudaMemcpyHostToDevice);
  recvPoolPtrsGpu_ = static_cast<void**>(mscclpp::detail::gpuCalloc(sizeof(void*) * static_cast<size_t>(numRanks_)));
  mscclpp::gpuMemcpy<void*>(recvPoolPtrsGpu_, recvPoolPtrs_.data(), numRanks_, cudaMemcpyHostToDevice);
  barrierChannelHandles_ = mscclpp::detail::gpuCallocShared<mscclpp::BaseMemoryChannelDeviceHandle>(numRanks_);
  mscclpp::gpuMemcpy<mscclpp::BaseMemoryChannelDeviceHandle>(barrierChannelHandles_.get(), barrierChannelHandles.data(),
                                                             numRanks_, cudaMemcpyHostToDevice);
  combineRecvIdxGpu_ = static_cast<int*>(
      mscclpp::detail::gpuCalloc(sizeof(int) * static_cast<size_t>(RecvPoolConfig::RecvPoolMaxTokens) * numRanks_));
  moeRecvCounter_ = static_cast<volatile int*>(mscclpp::detail::gpuCallocHost(sizeof(int), cudaHostAllocMapped));
  CUDA_CHECK(cudaHostGetDevicePointer(&moeRecvCounterMapped_, const_cast<int*>(moeRecvCounter_), 0));
  moeRecvExpertCounter_ = static_cast<volatile int*>(
      mscclpp::detail::gpuCallocHost(sizeof(int) * RecvPoolConfig::MaxLocalExperts, cudaHostAllocMapped));
  CUDA_CHECK(cudaHostGetDevicePointer(&moeRecvExpertCounterMapped_, const_cast<int*>(moeRecvExpertCounter_), 0));
  *moeRecvCounter_ = -1;
  for (int i = 0; i < RecvPoolConfig::MaxLocalExperts; ++i) moeRecvExpertCounter_[i] = -1;

  int deviceId;
  int maxSharedMemoryPerBlock;
  int numSms;
  CUDA_CHECK(cudaGetDevice(&deviceId));
  CUDA_CHECK(cudaDeviceGetAttribute(&maxSharedMemoryPerBlock, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceId));
  CUDA_CHECK(cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, deviceId));
  deviceContext_ = {.localBufferBase_ = symmetricBuffer_,
                    .peerBufferBases_ = bufferPtrsGpu_,
                    .peerPayloadBases_ = recvPoolPtrsGpu_,
                    .channels_ = barrierChannelHandles_.get(),
                    .workspace_ = nullptr,
                    .combineRecvIdx_ = combineRecvIdxGpu_,
                    .mappedRecvCounter_ = moeRecvCounterMapped_,
                    .mappedRecvExpertCounters_ = moeRecvExpertCounterMapped_,
                    .maxSharedMemoryPerBlock_ = maxSharedMemoryPerBlock,
                    .numSms_ = numSms,
                    .deviceId_ = deviceId,
                    .rank_ = rank_,
                    .numRanks_ = numRanks_};
  deviceContext_.devicePtr_ = static_cast<DeviceContext*>(mscclpp::detail::gpuCalloc(sizeof(DeviceContext)));
  mscclpp::gpuMemcpy<DeviceContext>(deviceContext_.devicePtr_, &deviceContext_, 1, cudaMemcpyHostToDevice);
}

bool ThroughputContext::canUseDirectRecvPool(int numTokens, int numRecvTokens, int hidden, int xElementSize) const {
  if (!collectiveDirectReady_ || numTokens < 0 || numRecvTokens < 0 || hidden <= 0 || xElementSize != 2 ||
      numTokens > RecvPoolConfig::RecvPoolMaxTokens || numRecvTokens > RecvPoolConfig::RecvPoolMaxTokens)
    return false;
  const int64_t hiddenBytes = static_cast<int64_t>(hidden) * xElementSize;
  return hiddenBytes <= maxHiddenBytes_ && static_cast<size_t>(numRecvTokens) * static_cast<size_t>(hiddenBytes) <=
                                               RecvPoolConfig::recvPoolHiddenBytes(numRanks_);
}

void MoERuntime::prepare(int* numTokensPerRank, int* numTokensPerExpert, bool* isTokenInRank, const int64_t* topkIdx,
                         int numTokens, int numTopk, int numExperts, cudaStream_t stream) {
  requireMode(MoEMode::THROUGHPUT);
  auto& context = *throughputContext_;
  EP_HOST_ASSERT(context.available_);
  EP_HOST_ASSERT(context.deviceContext_.devicePtr_ != nullptr);
  EP_HOST_ASSERT(numExperts > 0 && numExperts % context.numRanks_ == 0);
  EP_HOST_ASSERT(numTopk > 0 && numTopk <= 32);
  ep::throughputPrepare(topkIdx, numTokensPerRank, numTokensPerExpert, isTokenInRank, numTokens, numTopk, numExperts,
                        context.deviceContext_, stream);
}

int MoERuntime::notify(int* rankPrefixMatrix, int* channelPrefixMatrix, int* numRecvTokensPerExpert,
                       const int* numTokensPerRank, const int* numTokensPerExpert, const bool* isTokenInRank,
                       int numTokens, int numExperts, int expertAlignment, int numBlocks, cudaStream_t stream) {
  requireMode(MoEMode::THROUGHPUT);
  auto& context = *throughputContext_;
  EP_HOST_ASSERT(context.available_);
  EP_HOST_ASSERT(context.deviceContext_.devicePtr_ != nullptr);
  EP_HOST_ASSERT(numExperts > 0 && numExperts % context.numRanks_ == 0);
  EP_HOST_ASSERT(numBlocks > 0);
  const int numLocalExperts = numExperts / context.numRanks_;
  EP_HOST_ASSERT(numLocalExperts <= RecvPoolConfig::MaxLocalExperts);

  const int numChannels = numBlocks;

  *context.moeRecvCounter_ = -1;
  for (int i = 0; i < numLocalExperts; ++i) context.moeRecvExpertCounter_[i] = -1;
  throughputExchangeCounts(numTokensPerRank, numTokensPerExpert, numExperts, numTokens, isTokenInRank,
                           channelPrefixMatrix, rankPrefixMatrix, expertAlignment, context.deviceContext_, stream,
                           numChannels);

  int numRecvTokens = -1;
  const auto start = std::chrono::steady_clock::now();
  while (true) {
    numRecvTokens = static_cast<int>(*context.moeRecvCounter_);
    bool ready = numRecvTokens >= 0;
    for (int i = 0; i < numLocalExperts && ready; ++i) ready &= context.moeRecvExpertCounter_[i] >= 0;
    if (ready) break;
    if (std::chrono::steady_clock::now() - start >= ReceiveCountTimeout) {
      throw std::runtime_error("MSCCL++ EP throughput receive-count timeout");
    }
  }
  for (int i = 0; i < numLocalExperts; ++i) numRecvTokensPerExpert[i] = context.moeRecvExpertCounter_[i];

  const int outputRows = context.outputLayout_ == DispatchLayout::RANK_MAJOR
                             ? context.numRanks_ * context.maxTokensPerRank_
                             : numRecvTokens;
  const bool localDirectReady = numTokens >= 0 && numTokens <= RecvPoolConfig::RecvPoolMaxTokens && outputRows >= 0 &&
                                outputRows <= RecvPoolConfig::RecvPoolMaxTokens &&
                                static_cast<size_t>(outputRows) * static_cast<size_t>(context.maxHiddenBytes_) <=
                                    RecvPoolConfig::recvPoolHiddenBytes(context.numRanks_);
  std::vector<int> directReadyByRank(context.numRanks_, 0);
  directReadyByRank[context.rank_] = localDirectReady ? 1 : 0;
  context.bootstrap_->allGather(directReadyByRank.data(), sizeof(int));
  context.collectiveDirectReady_ =
      std::all_of(directReadyByRank.begin(), directReadyByRank.end(), [](int ready) { return ready != 0; });
  return numRecvTokens;
}

void MoERuntime::launchThroughputDispatch(const ThroughputDispatchRequest& request) {
  void* recvX = request.recvX;
  float* recvXScales = request.recvXScales;
  int64_t* recvTopkIdx = request.recvTopkIdx;
  float* recvTopkWeights = request.recvTopkWeights;
  int* sendHead = request.sendHead;
  const void* x = request.input;
  const float* xScales = request.inputScales;
  const int64_t* topkIdx = request.topkIdx;
  const float* topkWeights = request.topkWeights;
  const bool* isTokenInRank = request.isTokenInRank;
  const int* rankPrefixMatrix = request.rankPrefixMatrix;
  const int* channelPrefixMatrix = request.channelPrefixMatrix;
  const int numTokens = request.numTokens;
  const int hidden = request.hidden;
  const int numTopk = request.numTopk;
  const int numScales = request.numScales;
  const int numExperts = request.numExperts;
  const int xElementSize = request.inputElementSize;
  const int numRecvTokens = request.numRecvTokens;
  const bool cachedMode = request.cachedMode;
  const int numBlocks = request.numBlocks;
  const cudaStream_t stream = request.stream;

  auto& context = *throughputContext_;
  EP_HOST_ASSERT(context.available_);
  EP_HOST_ASSERT(context.deviceContext_.devicePtr_ != nullptr);
  EP_HOST_ASSERT(hidden > 0 && xElementSize == 2);
  EP_HOST_ASSERT(static_cast<int64_t>(hidden) * xElementSize <= context.maxHiddenBytes_);
  EP_HOST_ASSERT((hidden * xElementSize) % sizeof(int4) == 0);
  EP_HOST_ASSERT(numTopk >= 0 && numTopk <= RecvPoolConfig::MaxTopk);
  EP_HOST_ASSERT(numScales >= 0 && numScales <= RecvPoolConfig::MaxScales);
  EP_HOST_ASSERT(numBlocks > 0);

  const int numChannels = numBlocks;
  const int effectiveNumExperts = cachedMode ? 0 : numExperts;
  if (cachedMode) {
    throughputPublishCachedPrefix(rankPrefixMatrix, context.deviceContext_, stream);
  }

  context.dispatchReady_ = context.canUseDirectRecvPool(numTokens, numRecvTokens, hidden, xElementSize);
  EP_HOST_ASSERT(context.dispatchReady_ && "throughput dispatch capacity exceeded");
  const size_t poolHeaderBytes = RecvPoolConfig::recvPoolHeaderBytes(context.numRanks_);
  EP_HOST_ASSERT(recvX == static_cast<uint8_t*>(context.recvPoolPtrs_[context.rank_]) + poolHeaderBytes);

  const int hiddenInt4 = static_cast<int>(static_cast<int64_t>(hidden) * xElementSize / sizeof(int4));
  context.dispatchMetadataReady_ = true;
  if (recvTopkWeights != nullptr) context.recvTopkWeights_ = recvTopkWeights;
  throughputDispatch(sendHead, x, topkIdx, topkWeights, xScales, isTokenInRank, channelPrefixMatrix, numTokens,
                     numRecvTokens, hiddenInt4, numTopk, effectiveNumExperts, numScales, recvTopkIdx, recvTopkWeights,
                     recvXScales, numChannels, static_cast<int64_t>(poolHeaderBytes),
                     static_cast<int64_t>(RecvPoolConfig::recvPoolMetadataOffset(context.numRanks_)),
                     RecvPoolConfig::RecvPoolMetaBytes, context.outputLayout_, context.maxTokensPerRank_,
                     context.deviceContext_, stream);
}

void MoERuntime::launchThroughputCombine(const ThroughputCombineRequest& request) {
  void* combinedX = request.output;
  float* combinedTopkWeights = request.outputTopkWeights;
  const void* x = request.input;
  const float* topkWeights = request.topkWeights;
  const int* sendHead = request.sendHead;
  const int numInputTokens = request.numInputTokens;
  const int numOutputTokens = request.numOutputTokens;
  const int hidden = request.hidden;
  const int numTopk = request.numTopk;
  const int xElementSize = request.inputElementSize;
  const int numBlocks = request.numBlocks;
  const cudaStream_t stream = request.stream;

  auto& context = *throughputContext_;
  EP_HOST_ASSERT(context.available_);
  EP_HOST_ASSERT(context.deviceContext_.devicePtr_ != nullptr);
  EP_HOST_ASSERT(context.dispatchReady_);
  EP_HOST_ASSERT(xElementSize == 2);
  EP_HOST_ASSERT(static_cast<int64_t>(hidden) * xElementSize <= context.maxHiddenBytes_);
  EP_HOST_ASSERT((hidden * xElementSize) % sizeof(int4) == 0);
  EP_HOST_ASSERT(numTopk >= 0 && numTopk <= RecvPoolConfig::MaxTopk);
  EP_HOST_ASSERT((combinedTopkWeights == nullptr) == (topkWeights == nullptr));
  EP_HOST_ASSERT(numBlocks > 0);

  EP_HOST_ASSERT(numInputTokens <= RecvPoolConfig::RecvPoolMaxTokens);
  const size_t recvPoolHeaderBytes = RecvPoolConfig::recvPoolHeaderBytes(context.numRanks_);
  const size_t recvPoolMetadataOffset = RecvPoolConfig::recvPoolMetadataOffset(context.numRanks_);
  auto* localRecvPool = static_cast<uint8_t*>(context.recvPoolPtrs_[context.rank_]);
  void* localRecvPoolX = localRecvPool + recvPoolHeaderBytes;
  if (numInputTokens > 0 && x != localRecvPoolX) {
    CUDA_CHECK(cudaMemcpyAsync(localRecvPoolX, x, static_cast<size_t>(numInputTokens) * hidden * xElementSize,
                               cudaMemcpyDeviceToDevice, stream));
  }
  const bool weightsAlreadyStaged =
      context.dispatchMetadataReady_ && topkWeights != nullptr && topkWeights == context.recvTopkWeights_;
  if (numInputTokens > 0 && numTopk > 0 && topkWeights != nullptr && !weightsAlreadyStaged) {
    const size_t weightBytes = static_cast<size_t>(numTopk) * sizeof(float);
    void* localMetadataWeights = localRecvPool + recvPoolMetadataOffset + static_cast<size_t>(numTopk) * sizeof(int);
    CUDA_CHECK(cudaMemcpy2DAsync(localMetadataWeights, RecvPoolConfig::RecvPoolMetaBytes, topkWeights, weightBytes,
                                 weightBytes, numInputTokens, cudaMemcpyDeviceToDevice, stream));
  }

  throughputReduceCombine(combinedX, combinedTopkWeights, sendHead, numOutputTokens, hidden, numTopk,
                          static_cast<int64_t>(recvPoolHeaderBytes), static_cast<int64_t>(recvPoolMetadataOffset),
                          RecvPoolConfig::RecvPoolMetaBytes, numBlocks, context.deviceContext_, stream);
}

}  // namespace ep
}  // namespace mscclpp
