// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
//
// Portions adapted from DeepEP (https://github.com/deepseek-ai/DeepEP)
// branch `chhwang/dev-atomic-add-cleanup`. Licensed under the MIT License.

#include "ht_runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <future>
#include <limits>
#include <mscclpp/gpu_utils.hpp>
#include <stdexcept>

#include "api.cuh"
#include "constants.cuh"
#include "exception.cuh"

namespace mscclpp {
namespace ep {

MoEHighThroughputRuntime::MoEHighThroughputRuntime(mscclpp::Communicator& communicator, int64_t maxHiddenBytes,
                                                   const Config& config)
    : rank_(communicator.bootstrap()->getRank()),
      numRanks_(communicator.bootstrap()->getNranks()),
      numNvlRanks_(std::min(numRanks_, communicator.bootstrap()->getNranksPerNode())),
      numRanksPerIpcDomain_(
          std::max(numNvlRanks_, std::min(numRanks_, communicator.bootstrap()->getNranksPerIpcDomain()))),
      maxHiddenBytes_(maxHiddenBytes),
      config_(config) {
  EP_HOST_ASSERT(rank_ >= 0 and rank_ < numRanks_);
  EP_HOST_ASSERT(numNvlRanks_ > 0);
  EP_HOST_ASSERT(maxHiddenBytes_ > 0);

  if ((numRanks_ != 2 and numRanks_ != 4 and numRanks_ != 8) or numRanksPerIpcDomain_ < numRanks_) return;

  ringBufferBytes_ = config_.get_nvl_buffer_size_hint(static_cast<size_t>(maxHiddenBytes_), numRanks_);
  EP_HOST_ASSERT(ringBufferBytes_ <= static_cast<size_t>(std::numeric_limits<int>::max()));
  taskFifoOffset_ = ringBufferBytes_;
  symmetricBufferBytes_ =
      configAlign<size_t>(taskFifoOffset_ + sizeof(int) * NUM_MAX_FIFO_SLOTS, NUM_BUFFER_ALIGNMENT_BYTES);
  physicalRingBuffer_ = numRanks_ > numNvlRanks_;
  if (const char* direct = std::getenv("MSCCLPP_EP_INTRA_DIRECT"); direct != nullptr and std::atoi(direct) != 0)
    recvPoolBytes_ = Config::recv_pool_bytes_static(numRanks_);
  setup(communicator);
}

MoEHighThroughputRuntime::~MoEHighThroughputRuntime() noexcept(false) {
  CUDA_CHECK(cudaDeviceSynchronize());
  if (available_) {
    intranode::barrier(taskFifoPtrsGpu_, head_, rank_, numRanks_, nullptr);
    moveFifoSlots();
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  if (combineRecvIdxGpu_ != nullptr) CUDA_CHECK(cudaFree(combineRecvIdxGpu_));
  if (recvPoolPtrsGpu_ != nullptr) CUDA_CHECK(cudaFree(recvPoolPtrsGpu_));
  if (taskFifoPtrsGpu_ != nullptr) CUDA_CHECK(cudaFree(taskFifoPtrsGpu_));
  if (bufferPtrsGpu_ != nullptr) CUDA_CHECK(cudaFree(bufferPtrsGpu_));
  if (moeRecvExpertCounter_ != nullptr) CUDA_CHECK(cudaFreeHost(const_cast<int*>(moeRecvExpertCounter_)));
  if (moeRecvCounter_ != nullptr) CUDA_CHECK(cudaFreeHost(const_cast<int*>(moeRecvCounter_)));

  recvPoolMemories_.clear();
  peerMemories_.clear();
  if (recvPool_ != nullptr) mscclpp::detail::gpuFreePhysical(recvPool_);
  if (symmetricBuffer_ != nullptr) {
    if (physicalRingBuffer_)
      mscclpp::detail::gpuFreePhysical(symmetricBuffer_);
    else
      CUDA_CHECK(cudaFree(symmetricBuffer_));
  }
}

bool MoEHighThroughputRuntime::isAvailable() const { return available_; }

bool MoEHighThroughputRuntime::isInternodeAvailable() const { return isAvailable() and numRanks_ > numNvlRanks_; }

void MoEHighThroughputRuntime::setup(mscclpp::Communicator& communicator) {
  EP_HOST_ASSERT(!available_);
  if (physicalRingBuffer_) {
    symmetricBuffer_ = mscclpp::detail::gpuCallocPhysical(symmetricBufferBytes_);
  } else {
    CUDA_CHECK(cudaMalloc(&symmetricBuffer_, symmetricBufferBytes_));
    CUDA_CHECK(cudaMemset(symmetricBuffer_, 0, symmetricBufferBytes_));
  }
  if (recvPoolBytes_ > 0) recvPool_ = mscclpp::detail::gpuCallocPhysical(recvPoolBytes_);

  constexpr int RingBufferTag = 17;
  constexpr int RecvPoolTag = 18;
  const auto transport = mscclpp::Transport::CudaIpc;
  peerMemories_.resize(numRanks_);
  peerMemories_[rank_] = communicator.registerMemory(symmetricBuffer_, symmetricBufferBytes_, transport);
  std::vector<std::shared_future<mscclpp::RegisteredMemory>> remoteMemories(numRanks_);
  std::vector<std::shared_future<mscclpp::RegisteredMemory>> remoteRecvPools;
  if (recvPool_ != nullptr) {
    recvPoolMemories_.resize(numRanks_);
    recvPoolMemories_[rank_] = communicator.registerMemory(recvPool_, recvPoolBytes_, transport);
    remoteRecvPools.resize(numRanks_);
  }
  for (int peer = 0; peer < numRanks_; ++peer) {
    if (peer == rank_) continue;
    communicator.sendMemory(peerMemories_[rank_], peer, RingBufferTag);
    remoteMemories[peer] = communicator.recvMemory(peer, RingBufferTag);
    if (recvPool_ != nullptr) {
      communicator.sendMemory(recvPoolMemories_[rank_], peer, RecvPoolTag);
      remoteRecvPools[peer] = communicator.recvMemory(peer, RecvPoolTag);
    }
  }

  bufferPtrs_.resize(numRanks_);
  taskFifoPtrs_.resize(numRanks_);
  recvPoolPtrs_.resize(numRanks_);
  for (int peer = 0; peer < numRanks_; ++peer) {
    if (peer != rank_) {
      peerMemories_[peer] = remoteMemories[peer].get();
      if (recvPool_ != nullptr) recvPoolMemories_[peer] = remoteRecvPools[peer].get();
    }
    void* base = peer == rank_ ? symmetricBuffer_ : peerMemories_[peer].data();
    bufferPtrs_[peer] = base;
    taskFifoPtrs_[peer] = reinterpret_cast<int*>(static_cast<uint8_t*>(base) + taskFifoOffset_);
    recvPoolPtrs_[peer] = recvPool_ == nullptr ? nullptr : (peer == rank_ ? recvPool_ : recvPoolMemories_[peer].data());
  }

  CUDA_CHECK(cudaMalloc(&bufferPtrsGpu_, sizeof(void*) * numRanks_));
  CUDA_CHECK(cudaMalloc(&taskFifoPtrsGpu_, sizeof(int*) * numRanks_));
  CUDA_CHECK(cudaMemcpy(bufferPtrsGpu_, bufferPtrs_.data(), sizeof(void*) * numRanks_, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(taskFifoPtrsGpu_, taskFifoPtrs_.data(), sizeof(int*) * numRanks_, cudaMemcpyHostToDevice));
  if (recvPool_ != nullptr) {
    CUDA_CHECK(cudaMalloc(&recvPoolPtrsGpu_, sizeof(void*) * numRanks_));
    CUDA_CHECK(cudaMemcpy(recvPoolPtrsGpu_, recvPoolPtrs_.data(), sizeof(void*) * numRanks_, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMalloc(&combineRecvIdxGpu_, sizeof(int) * static_cast<size_t>(Config::RecvPoolMaxTokens) * numRanks_));
  }
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

void MoEHighThroughputRuntime::computeDispatchChannels(int xElementSize, int& dispatchNumSms, bool& allSender,
                                                       int& numChannels) const {
  int dispatch_num_sms = config_.num_sms;
  if (const char* env = std::getenv("MSCCLPP_EP_DISPATCH_NSM")) {
    int value = std::atoi(env) & ~1;
    if (value >= 2) dispatch_num_sms = std::min(value, config_.num_sms);
  }

  static const bool AllSenderEnabled = [] {
    const char* direct = std::getenv("MSCCLPP_EP_INTRA_DIRECT");
    if (not(direct != nullptr and std::atoi(direct) != 0)) return false;
    const char* tma = std::getenv("MSCCLPP_EP_COMBINE_TMA");
    if (tma != nullptr and std::atoi(tma) == 0) return false;
    const char* all_sender = std::getenv("MSCCLPP_EP_INTRA_ALLSENDER");
    return all_sender == nullptr or std::atoi(all_sender) != 0;
  }();

  dispatchNumSms = dispatch_num_sms;
  allSender = AllSenderEnabled and xElementSize == 2;
  numChannels = allSender ? dispatch_num_sms : dispatch_num_sms / 2;
}

bool MoEHighThroughputRuntime::canUseDirectRecvPool(int numTokens, int numRecvTokens, int hidden,
                                                    int xElementSize) const {
  const char* env = std::getenv("MSCCLPP_EP_INTRA_DIRECT");
  if (recvPool_ == nullptr or recvPoolPtrsGpu_ == nullptr or combineRecvIdxGpu_ == nullptr or env == nullptr or
      std::atoi(env) == 0 or numTokens < 0 or numRecvTokens < 0 or hidden <= 0 or xElementSize <= 0 or
      numTokens > Config::RecvPoolMaxTokens or numRecvTokens > Config::RecvPoolMaxTokens)
    return false;
  const int64_t hidden_bytes = static_cast<int64_t>(hidden) * xElementSize;
  return hidden_bytes <= maxHiddenBytes_ and static_cast<size_t>(numRecvTokens) * static_cast<size_t>(hidden_bytes) <=
                                                 Config::get_recv_pool_hidden_bytes(numRanks_);
}

void MoEHighThroughputRuntime::layout(int* numTokensPerRank, int* numTokensPerExpert, bool* isTokenInRank,
                                      const int64_t* topkIdx, int numTokens, int numTopk, int numExperts,
                                      cudaStream_t stream) {
  EP_HOST_ASSERT(available_);
  EP_HOST_ASSERT(numExperts > 0 and numExperts % numRanks_ == 0);
  EP_HOST_ASSERT(numTopk > 0 and numTopk <= 32);
  intranode::get_dispatch_layout(topkIdx, numTokensPerRank, numTokensPerExpert, isTokenInRank, numTokens, numTopk,
                                 numRanks_, numExperts, stream);
}

int MoEHighThroughputRuntime::getDispatchNumChannels(int xElementSize) const {
  int dispatch_num_sms = 0;
  int num_channels = 0;
  bool all_sender = false;
  computeDispatchChannels(xElementSize, dispatch_num_sms, all_sender, num_channels);
  return num_channels;
}

void* MoEHighThroughputRuntime::resolveRecvXBuffer(int numTokens, int numRecvTokens, int hidden,
                                                   int xElementSize) const {
  if (not available_ or not canUseDirectRecvPool(numTokens, numRecvTokens, hidden, xElementSize)) return nullptr;
  return static_cast<uint8_t*>(recvPoolPtrs_[rank_]) + Config::get_recv_pool_header_bytes(numRanks_);
}

int MoEHighThroughputRuntime::notifyDispatch(int* rankPrefixMatrix, int* channelPrefixMatrix,
                                             int* numRecvTokensPerExpert, const int* numTokensPerRank,
                                             const int* numTokensPerExpert, const bool* isTokenInRank, int numTokens,
                                             int numExperts, int xElementSize, int expertAlignment,
                                             cudaStream_t stream) {
  EP_HOST_ASSERT(available_);
  EP_HOST_ASSERT(numExperts > 0 and numExperts % numRanks_ == 0);
  const int num_local_experts = numExperts / numRanks_;
  EP_HOST_ASSERT(num_local_experts <= NUM_MAX_LOCAL_EXPERTS);

  int dispatch_num_sms = 0;
  int num_channels = 0;
  bool all_sender = false;
  computeDispatchChannels(xElementSize, dispatch_num_sms, all_sender, num_channels);
  const int num_memset_int = num_channels * numRanks_ * 4;

  *moeRecvCounter_ = -1;
  for (int i = 0; i < num_local_experts; ++i) moeRecvExpertCounter_[i] = -1;
  intranode::notify_dispatch(numTokensPerRank, moeRecvCounterMapped_, numRanks_, numTokensPerExpert,
                             moeRecvExpertCounterMapped_, numExperts, numTokens, isTokenInRank, channelPrefixMatrix,
                             rankPrefixMatrix, num_memset_int, expertAlignment, bufferPtrsGpu_, taskFifoPtrsGpu_, head_,
                             rank_, stream, num_channels);
  moveFifoSlots(3);

  int num_recv_tokens = -1;
  const auto start = std::chrono::high_resolution_clock::now();
  while (true) {
    num_recv_tokens = static_cast<int>(*moeRecvCounter_);
    bool ready = num_recv_tokens >= 0;
    for (int i = 0; i < num_local_experts and ready; ++i) ready &= moeRecvExpertCounter_[i] >= 0;
    if (ready) break;
    if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start).count() >
        NUM_CPU_TIMEOUT_SECS)
      throw std::runtime_error("DeepEP error: CPU recv timeout");
  }
  for (int i = 0; i < num_local_experts; ++i) numRecvTokensPerExpert[i] = moeRecvExpertCounter_[i];
  return num_recv_tokens;
}

void MoEHighThroughputRuntime::dispatch(void* recvX, float* recvXScales, int64_t* recvTopkIdx, float* recvTopkWeights,
                                        int* recvSrcIdx, int* sendHead, int* recvChannelPrefixMatrix, const void* x,
                                        const float* xScales, const int64_t* topkIdx, const float* topkWeights,
                                        const bool* isTokenInRank, const int* rankPrefixMatrix,
                                        const int* channelPrefixMatrix, int numTokens, int hidden, int numTopk,
                                        int numScales, int numExperts, int xElementSize, int numRecvTokens,
                                        bool cachedMode, cudaStream_t stream) {
  EP_HOST_ASSERT(available_);
  EP_HOST_ASSERT(hidden > 0 and xElementSize > 0);
  EP_HOST_ASSERT(static_cast<int64_t>(hidden) * xElementSize <= maxHiddenBytes_);
  EP_HOST_ASSERT((hidden * xElementSize) % sizeof(int4) == 0);
  EP_HOST_ASSERT(numTopk >= 0 and numTopk <= Config::MaxTopk);
  EP_HOST_ASSERT(numScales >= 0 and numScales <= Config::MaxScales);

  int dispatch_num_sms = 0;
  int num_channels = 0;
  bool all_sender = false;
  computeDispatchChannels(xElementSize, dispatch_num_sms, all_sender, num_channels);
  const int num_experts = cachedMode ? 0 : numExperts;
  const int num_memset_int = num_channels * numRanks_ * 4;
  if (cachedMode) {
    intranode::cached_notify_dispatch(rankPrefixMatrix, num_memset_int, bufferPtrsGpu_, taskFifoPtrsGpu_, head_, rank_,
                                      numRanks_, stream);
    moveFifoSlots(2);
  }

  const bool direct = canUseDirectRecvPool(numTokens, numRecvTokens, hidden, xElementSize);
  directDispatchReady_ = direct;
  void** direct_pool_ptrs = direct ? recvPoolPtrsGpu_ : nullptr;
  const size_t pool_header_bytes = Config::get_recv_pool_header_bytes(numRanks_);
  if (direct) EP_HOST_ASSERT(recvX == static_cast<uint8_t*>(recvPoolPtrs_[rank_]) + pool_header_bytes);

  const int hidden_int4 = hidden * xElementSize / sizeof(int4);
  if (all_sender) {
    EP_HOST_ASSERT(direct);
    CUDA_CHECK(cudaMemsetAsync(recvChannelPrefixMatrix, 0, static_cast<size_t>(numRanks_) * num_channels * sizeof(int),
                               stream));
    intranode::dispatch_allsender(
        sendHead, x, topkIdx, topkWeights, xScales, isTokenInRank, channelPrefixMatrix, numTokens, hidden_int4, numTopk,
        num_experts, numScales, bufferPtrsGpu_, rank_, numRanks_, stream, num_channels, recvPoolPtrsGpu_,
        static_cast<int64_t>(pool_header_bytes), static_cast<int64_t>(Config::get_recv_pool_meta_base(numRanks_)),
        Config::RecvPoolMetaBytes, combineRecvIdxGpu_);
    intranode::intranode_meta_drain(
        recvPoolPtrs_[rank_], static_cast<int64_t>(Config::get_recv_pool_meta_base(numRanks_)), numRecvTokens,
        recvSrcIdx, recvTopkIdx, recvTopkWeights, recvXScales, numTopk, numScales, Config::RecvPoolMetaBytes, stream);
    return;
  }

  const size_t required_bytes =
      static_cast<size_t>(numRanks_) * numRanks_ * sizeof(int) +
      static_cast<size_t>(num_channels) * numRanks_ * sizeof(int) * 4 +
      static_cast<size_t>(num_channels) * numRanks_ * config_.num_max_nvl_chunked_recv_tokens *
          (static_cast<size_t>(hidden) * xElementSize + sizeof(int) + static_cast<size_t>(numTopk) * sizeof(int64_t) +
           static_cast<size_t>(numTopk) * sizeof(float) + static_cast<size_t>(numScales) * sizeof(float));
  EP_HOST_ASSERT(required_bytes <= ringBufferBytes_);
  intranode::dispatch(recvX, recvXScales, recvSrcIdx, recvTopkIdx, recvTopkWeights, recvChannelPrefixMatrix, sendHead,
                      x, xScales, topkIdx, topkWeights, isTokenInRank, channelPrefixMatrix, numTokens, hidden_int4,
                      numTopk, num_experts, numScales, bufferPtrsGpu_, rank_, numRanks_, stream, dispatch_num_sms,
                      config_.num_max_nvl_chunked_send_tokens, config_.num_max_nvl_chunked_recv_tokens,
                      direct_pool_ptrs, static_cast<int64_t>(pool_header_bytes), direct ? combineRecvIdxGpu_ : nullptr);
}

void MoEHighThroughputRuntime::combine(void* combinedX, float* combinedTopkWeights, const void* x,
                                       const float* topkWeights, const int* srcIdx, const int* rankPrefixMatrix,
                                       const int* channelPrefixMatrix, const int* sendHead, int numTokens,
                                       int numRecvTokens, int hidden, int numTopk, int xElementSize,
                                       int ringNumChannels, cudaStream_t stream) {
  EP_HOST_ASSERT(available_);
  EP_HOST_ASSERT(static_cast<int64_t>(hidden) * xElementSize <= maxHiddenBytes_);
  EP_HOST_ASSERT((hidden * xElementSize) % sizeof(int4) == 0);
  EP_HOST_ASSERT(numTopk >= 0 and numTopk <= Config::MaxTopk);

  bool used_tma_combine = false;
  static const bool TmaDisabled = [] {
    const char* env = std::getenv("MSCCLPP_EP_COMBINE_TMA");
    return env != nullptr and std::atoi(env) == 0;
  }();
  if (not TmaDisabled and directDispatchReady_ and xElementSize == 2) {
    EP_HOST_ASSERT(numTokens <= Config::RecvPoolMaxTokens);
    int combine_sms = config_.num_sms;
    if (const char* env = std::getenv("MSCCLPP_EP_COMBINE_NSM"))
      if (std::atoi(env) >= 1) combine_sms = std::atoi(env);
    used_tma_combine = intranode::combine_tma(
        CUDA_R_16BF, combinedX, combinedTopkWeights, const_cast<int*>(sendHead), numRecvTokens, hidden, numTopk,
        numRanks_, recvPoolPtrsGpu_, combineRecvIdxGpu_,
        static_cast<int64_t>(Config::get_recv_pool_header_bytes(numRanks_)), combine_sms, stream);
  }

  if (used_tma_combine) return;

  const int max_ring_channels = config_.num_sms / 2;
  EP_HOST_ASSERT(ringNumChannels > 0 and ringNumChannels <= max_ring_channels);
  intranode::cached_notify_combine(bufferPtrsGpu_, const_cast<int*>(sendHead), ringNumChannels, numRecvTokens,
                                   ringNumChannels * numRanks_ * 2, taskFifoPtrsGpu_, head_, rank_, numRanks_, stream);
  moveFifoSlots(2);

  const size_t required_bytes =
      static_cast<size_t>(ringNumChannels) * numRanks_ * sizeof(int) * 2 +
      static_cast<size_t>(ringNumChannels) * numRanks_ * config_.num_max_nvl_chunked_recv_tokens *
          (static_cast<size_t>(hidden) * xElementSize + sizeof(int) + static_cast<size_t>(numTopk) * sizeof(float));
  EP_HOST_ASSERT(required_bytes <= ringBufferBytes_);
  intranode::combine(CUDA_R_16BF, combinedX, combinedTopkWeights, x, topkWeights, srcIdx, rankPrefixMatrix,
                     channelPrefixMatrix, const_cast<int*>(sendHead), numTokens, numRecvTokens, hidden, numTopk,
                     bufferPtrsGpu_, rank_, numRanks_, stream, ringNumChannels * 2,
                     config_.num_max_nvl_chunked_send_tokens, config_.num_max_nvl_chunked_recv_tokens);
}

}  // namespace ep
}  // namespace mscclpp
