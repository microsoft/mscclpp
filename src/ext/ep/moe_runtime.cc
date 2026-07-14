// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "moe_runtime.hpp"

#include <cuda.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <future>
#include <limits>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/utils.hpp>

#include "api.cuh"
#include "constants.cuh"
#include "exception.cuh"

namespace mscclpp {
namespace ep {

MoERuntime::MoERuntime(mscclpp::Communicator& communicator, int64_t numNvlBytes, int64_t symmetricBufferBytes,
                       MoEMode mode)
    : rank_(communicator.bootstrap()->getRank()),
      numRanks_(communicator.bootstrap()->getNranks()),
      numNvlBytes_(numNvlBytes),
      symmetricBufferBytes_(symmetricBufferBytes),
      mode_(mode),
      communicator_(&communicator) {
  EP_HOST_ASSERT(mode_ == MoEMode::LOW_LATENCY);
  EP_HOST_ASSERT(communicator_ != nullptr);
  EP_HOST_ASSERT(numNvlBytes_ == 0);
  EP_HOST_ASSERT(symmetricBufferBytes_ % NUM_BUFFER_ALIGNMENT_BYTES == 0);
  EP_HOST_ASSERT(rank_ >= 0 && rank_ < numRanks_);

  CUDA_CHECK(cudaGetDevice(&deviceId_));
  numNvlRanks_ = std::min(numRanks_, communicator.bootstrap()->getNranksPerNode());
  numRanksPerIpcDomain_ =
      std::max(numNvlRanks_, std::min(numRanks_, communicator.bootstrap()->getNranksPerIpcDomain()));
  directIpcDomainSize_ = numRanksPerIpcDomain_;
  if (const char* env = std::getenv("MSCCLPP_EP_TEST_IPC_DOMAIN_SIZE")) {
    const int testDomainSize = std::atoi(env);
    EP_HOST_ASSERT(testDomainSize > 0 && testDomainSize <= numNvlRanks_ && numRanks_ % testDomainSize == 0);
    directIpcDomainSize_ = testDomainSize;
  }
  EP_HOST_ASSERT(numNvlRanks_ > 0 && numRanks_ % numNvlRanks_ == 0);
  EP_HOST_ASSERT(numRanks_ % numRanksPerIpcDomain_ == 0);
  rdmaRank_ = rank_ / numNvlRanks_;
  nvlRank_ = rank_ % numNvlRanks_;
  numRdmaRanks_ = numRanks_ / numNvlRanks_;

  CUDA_CHECK(cudaMalloc(&workspace_, NUM_WORKSPACE_BYTES));
  CUDA_CHECK(cudaMemset(workspace_, 0, NUM_WORKSPACE_BYTES));
  setup();
}

MoERuntime::~MoERuntime() noexcept(false) {
  CUDA_CHECK(cudaDeviceSynchronize());
  for (auto& proxyService : proxyServices_) {
    if (proxyService != nullptr) proxyService->stopProxy();
  }
  portChannels_.clear();
  proxyServices_.clear();
  baseMemoryChannels_.clear();
  peerBufferMemories_.clear();
  if (peerMappedBufferBasesGpu_ != nullptr) CUDA_CHECK(cudaFree(peerMappedBufferBasesGpu_));
  if (workspace_ != nullptr) CUDA_CHECK(cudaFree(workspace_));
  if (symmetricBuffer_ != nullptr) {
    // gpuCallocPhysical allocations are intentionally process-lifetime in the
    // existing EP runtime. cudaFree is safe for the fallback cudaMalloc path.
    // Ignore failures from fabric allocations during process teardown.
    cudaFree(symmetricBuffer_);
  }
}

bool MoERuntime::isAvailable() const { return available_; }
bool MoERuntime::isInternodeAvailable() const { return isAvailable() && numRdmaRanks_ > 1; }
int MoERuntime::getNumRdmaRanks() const { return numRdmaRanks_; }
int MoERuntime::getRdmaRank() const { return rdmaRank_; }
int MoERuntime::getRootRdmaRank(bool global) const { return global ? nvlRank_ : 0; }
int MoERuntime::getLocalDeviceId() const { return deviceId_; }
std::string MoERuntime::getLocalIpcHandle() const { return {}; }

void MoERuntime::setup() {
  EP_HOST_ASSERT(!available_);
  EP_HOST_ASSERT(communicator_ != nullptr);

  const auto ipcTransport = mscclpp::Transport::CudaIpc;
  const bool testDomainOverride = directIpcDomainSize_ != numRanksPerIpcDomain_;
  const bool spansHosts = !testDomainOverride && numRanksPerIpcDomain_ > numNvlRanks_;
  std::vector<int> fabricCapabilities(numRanks_, 0);
  fabricCapabilities[rank_] = spansHosts && mscclpp::isFabricMemHandleAvailable() && mscclpp::isNvlsSupported();
  communicator_->bootstrap()->allGather(fabricCapabilities.data(), sizeof(int));
  const bool useFabricIpcAlloc =
      spansHosts && std::all_of(fabricCapabilities.begin(), fabricCapabilities.end(), [](int value) { return value; });
  directIpcDomainSize_ = useFabricIpcAlloc ? numRanksPerIpcDomain_ : std::min(directIpcDomainSize_, numNvlRanks_);
  const bool hasPortPeers = directIpcDomainSize_ < numRanks_;
  EP_HOST_ASSERT(!hasPortPeers || symmetricBufferBytes_ < static_cast<int64_t>(std::numeric_limits<uint32_t>::max()));

  static constexpr std::array<mscclpp::Transport, 8> IbTransports = {
      mscclpp::Transport::IB0, mscclpp::Transport::IB1, mscclpp::Transport::IB2, mscclpp::Transport::IB3,
      mscclpp::Transport::IB4, mscclpp::Transport::IB5, mscclpp::Transport::IB6, mscclpp::Transport::IB7};
  EP_HOST_ASSERT(!hasPortPeers || deviceId_ < static_cast<int>(IbTransports.size()));
  const auto portTransport = hasPortPeers ? IbTransports[deviceId_] : mscclpp::Transport::Unknown;

  if (useFabricIpcAlloc) {
    symmetricBuffer_ = mscclpp::detail::gpuCallocPhysical(symmetricBufferBytes_);
  } else {
    CUDA_CHECK(cudaMalloc(&symmetricBuffer_, symmetricBufferBytes_));
  }
  CUDA_CHECK(cudaMemset(symmetricBuffer_, 0, symmetricBufferBytes_));
  communicator_->bootstrap()->barrier();
  CUDA_CHECK(cudaDeviceSynchronize());

  const mscclpp::EndpointConfig ipcConfig(ipcTransport);
  auto isMappedPeer = [&](int peer) {
    return peer != rank_ && directIpcDomainSize_ > 1 && rank_ / directIpcDomainSize_ == peer / directIpcDomainSize_;
  };

  mscclpp::TransportFlags memoryTransports(ipcTransport);
  if (hasPortPeers) memoryTransports |= portTransport;

  constexpr int MemoryTag = 1;
  constexpr int ConnectionTag = 2;
  peerBufferMemories_.resize(numRanks_);
  peerBufferMemories_[rank_] = communicator_->registerMemory(symmetricBuffer_, symmetricBufferBytes_, memoryTransports);
  std::vector<std::shared_future<mscclpp::RegisteredMemory>> remoteFutures(numRanks_);
  std::vector<std::shared_future<mscclpp::Connection>> connectionFutures(numRanks_);
  for (int r = 0; r < numRanks_; ++r) {
    if (r == rank_) continue;
    communicator_->sendMemory(peerBufferMemories_[rank_], r, MemoryTag);
    remoteFutures[r] = communicator_->recvMemory(r, MemoryTag);
    if (isMappedPeer(r)) {
      connectionFutures[r] = communicator_->connect(ipcConfig, r, ConnectionTag);
    } else {
      mscclpp::EndpointConfig portConfig(portTransport);
      portConfig.ib.mode = mscclpp::EndpointConfig::Ib::Mode::Host;
      connectionFutures[r] = communicator_->connect(portConfig, r, ConnectionTag);
    }
  }

  peerMappedBufferBases_.assign(numRanks_, nullptr);
  peerMappedBufferBases_[rank_] = symmetricBuffer_;
  std::vector<mscclpp::BaseMemoryChannelDeviceHandle> baseMemoryChannelHandles(numRanks_);
  std::vector<mscclpp::PortChannelDeviceHandle> portChannelHandles(numRanks_);
  if (hasPortPeers) proxyServices_.resize(numRanks_);
  for (int r = 0; r < numRanks_; ++r) {
    if (r == rank_) continue;
    peerBufferMemories_[r] = remoteFutures[r].get();
    auto connection = connectionFutures[r].get();
    if (isMappedPeer(r)) {
      peerMappedBufferBases_[r] = peerBufferMemories_[r].data();
      auto semaphore = std::make_shared<mscclpp::MemoryDevice2DeviceSemaphore>(*communicator_, connection);
      baseMemoryChannels_.emplace_back(semaphore);
      baseMemoryChannelHandles[r] = baseMemoryChannels_.back().deviceHandle();
    } else {
      auto proxyService = std::make_shared<mscclpp::ProxyService>(NUM_MAX_FIFO_SLOTS);
      const auto localMemoryId = proxyService->addMemory(peerBufferMemories_[rank_]);
      const auto remoteMemoryId = proxyService->addMemory(peerBufferMemories_[r]);
      const auto semaphoreId = proxyService->buildAndAddSemaphore(*communicator_, connection);
      portChannels_.emplace_back(proxyService->portChannel(semaphoreId, remoteMemoryId, localMemoryId));
      portChannelHandles[r] = portChannels_.back().deviceHandle();
      proxyServices_[r] = std::move(proxyService);
    }
  }
  for (auto& proxyService : proxyServices_) {
    if (proxyService != nullptr) proxyService->startProxy(true);
  }

  CUDA_CHECK(cudaMalloc(&peerMappedBufferBasesGpu_, sizeof(void*) * numRanks_));
  CUDA_CHECK(cudaMemcpy(peerMappedBufferBasesGpu_, peerMappedBufferBases_.data(), sizeof(void*) * numRanks_,
                        cudaMemcpyHostToDevice));
  baseMemoryChannelHandles_ = mscclpp::detail::gpuCallocShared<mscclpp::BaseMemoryChannelDeviceHandle>(numRanks_);
  mscclpp::gpuMemcpy<mscclpp::BaseMemoryChannelDeviceHandle>(
      baseMemoryChannelHandles_.get(), baseMemoryChannelHandles.data(), numRanks_, cudaMemcpyHostToDevice);
  portChannelHandles_ = mscclpp::detail::gpuCallocShared<mscclpp::PortChannelDeviceHandle>(numRanks_);
  mscclpp::gpuMemcpy<mscclpp::PortChannelDeviceHandle>(portChannelHandles_.get(), portChannelHandles.data(), numRanks_,
                                                       cudaMemcpyHostToDevice);

  int maxSharedMemoryPerBlock;
  int numSms;
  CUDA_CHECK(cudaDeviceGetAttribute(&maxSharedMemoryPerBlock, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceId_));
  CUDA_CHECK(cudaDeviceGetAttribute(&numSms, cudaDevAttrMultiProcessorCount, deviceId_));
  commContext_ = {.symmetricBufferBase_ = symmetricBuffer_,
                  .baseMemoryChannels_ = baseMemoryChannelHandles_.get(),
                  .portChannels_ = portChannelHandles_.get(),
                  .peerMappedBufferBases_ = peerMappedBufferBasesGpu_,
                  .ranksPerIpcDomain_ = directIpcDomainSize_,
                  .maxSharedMemoryPerBlock_ = maxSharedMemoryPerBlock,
                  .numSms_ = numSms,
                  .deviceId_ = deviceId_,
                  .rank_ = rank_,
                  .numRanks_ = numRanks_};
  available_ = true;
}

void MoERuntime::dispatch(void* output, float* outputScales, int* outputSrcInfo, int64_t* outputLayout,
                          int* outputCount, const void* input, const int64_t* topkIdx, const float* topkWeights,
                          int numTokens, int hidden, int numTopk, int maxTokensPerRank, int numExperts,
                          low_latency::DispatchDataType dispatchDataType, int numBlocks, cudaStream_t stream) {
  EP_HOST_ASSERT(mode_ == MoEMode::LOW_LATENCY);
  EP_HOST_ASSERT(available_);
  EP_HOST_ASSERT(numTokens <= maxTokensPerRank);
  EP_HOST_ASSERT(numExperts % numRanks_ == 0);
  EP_HOST_ASSERT(numBlocks - low_latency::DispatchControlBlocks >= numRanks_ &&
                 numBlocks <= low_latency::MaxDispatchBlocks);

  low_latency::Layout layout(symmetricBuffer_, maxTokensPerRank, hidden, numRanks_, numExperts, numTopk);
  EP_HOST_ASSERT(layout.totalBytes_ <= static_cast<size_t>(symmetricBufferBytes_));
  void* recvBuffer = layout.buffers_[lowLatencyBufferIdx_].data_;
  lowLatencyBufferIdx_ ^= 1;

  const low_latency::Workload workload{.numTokens_ = numTokens,
                                       .hidden_ = hidden,
                                       .numTopk_ = numTopk,
                                       .numExperts_ = numExperts,
                                       .maxTokensPerRank_ = maxTokensPerRank,
                                       .dispatchDataType_ = dispatchDataType};
  const size_t workspaceBytes = low_latency::workspaceSize(numRanks_, numExperts);
  EP_HOST_ASSERT(workspaceBytes <= NUM_WORKSPACE_BYTES);
  low_latency::dispatch(output, outputScales, outputSrcInfo, outputLayout, outputCount, input, topkIdx, topkWeights,
                        workload, recvBuffer, commContext_, workspace_, numBlocks, stream);
}

void MoERuntime::combine(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights,
                         const int* srcInfo, const int64_t* layoutRange, int numTokens, int hidden, int numTopk,
                         int maxTokensPerRank, int numExperts, low_latency::DispatchDataType dispatchDataType,
                         low_latency::CombineMode mode, int numBlocks, cudaStream_t stream) {
  EP_HOST_ASSERT(mode_ == MoEMode::LOW_LATENCY);
  EP_HOST_ASSERT(available_);
  EP_HOST_ASSERT(numExperts % numRanks_ == 0);
  EP_HOST_ASSERT(numBlocks > 0 && numBlocks <= low_latency::MaxWorkerBlocks);

  low_latency::Layout layout(symmetricBuffer_, maxTokensPerRank, hidden, numRanks_, numExperts, numTopk);
  EP_HOST_ASSERT(layout.totalBytes_ <= static_cast<size_t>(symmetricBufferBytes_));
  void* recvBuffer = layout.buffers_[lowLatencyBufferIdx_].data_;
  lowLatencyBufferIdx_ ^= 1;
  void* dispatchRecvBuffer = layout.buffers_[lowLatencyBufferIdx_].data_;

  const low_latency::Workload workload{.numTokens_ = numTokens,
                                       .hidden_ = hidden,
                                       .numTopk_ = numTopk,
                                       .numExperts_ = numExperts,
                                       .maxTokensPerRank_ = maxTokensPerRank,
                                       .dispatchDataType_ = dispatchDataType};
  low_latency::combine(output, input, topkIdx, topkWeights, srcInfo, layoutRange, workload, recvBuffer,
                       dispatchRecvBuffer, commContext_, workspace_, numBlocks, mode, stream);
}

}  // namespace ep
}  // namespace mscclpp
