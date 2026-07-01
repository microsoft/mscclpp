// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "moe_runtime.hpp"

#include <cuda.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <future>
#include <map>
#include <string>

#include "kernels/api.cuh"
#include "kernels/configs.cuh"
#include "kernels/exception.cuh"

namespace mscclpp {
namespace ep {

namespace {

using EPProxyService = mscclpp::ProxyService;

int localWorldSize() {
  int localWorldSize = NUM_MAX_NVL_PEERS;
  if (const char* env = std::getenv("MSCCLPP_EP_LOCAL_WORLD_SIZE")) {
    int v = std::atoi(env);
    if (v > 0 && v <= NUM_MAX_NVL_PEERS) localWorldSize = v;
  }
  return localWorldSize;
}

int resolveNumProxyServices(int numRanks, int localWorldSize) {
  if (const char* env = std::getenv("MSCCLPP_EP_NUM_PROXIES")) {
    int v = std::atoi(env);
    return v > 0 ? v : 1;
  }
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return 8;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return 8;
  int lws = localWorldSize > 0 ? localWorldSize : 1;
  int numRdmaRanksLocal = std::max(1, numRanks / lws);
  if (prop.major >= 10) return numRdmaRanksLocal > 1 ? lws : 1;
  return 8;
}

bool resolveFabricIpcSupported() {
  if (const char* env = std::getenv("MSCCLPP_EP_FABRIC_IPC")) {
    std::string v(env);
    for (auto& c : v) c = std::tolower(static_cast<unsigned char>(c));
    if (v == "0" || v == "off" || v == "false" || v == "no") return false;
    if (v == "1" || v == "on" || v == "true" || v == "yes" || v == "force") return true;
  }
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return false;
  CUdevice cu_dev;
  if (cuDeviceGet(&cu_dev, dev) != CUDA_SUCCESS) return false;
  int supported = 0;
  if (cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, cu_dev) != CUDA_SUCCESS) {
    return false;
  }
  if (!supported) return false;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return false;
  return prop.major >= 10;
}

}  // namespace

MoERuntime::MoERuntime(mscclpp::Communicator& communicator, int64_t numNvlBytes, int64_t numRdmaBytes, MoEMode mode)
    : rank_(communicator.bootstrap()->getRank()),
      numRanks_(communicator.bootstrap()->getNranks()),
      numNvlBytes_(numNvlBytes),
      numRdmaBytes_(numRdmaBytes),
      mode_(mode),
      communicator_(&communicator) {
  EP_HOST_ASSERT(mode_ == MoEMode::LOW_LATENCY);
  EP_HOST_ASSERT(communicator_ != nullptr);
  EP_HOST_ASSERT(numNvlBytes_ == 0);
  EP_HOST_ASSERT(numRdmaBytes_ % NUM_BUFFER_ALIGNMENT_BYTES == 0);
  EP_HOST_ASSERT(rank_ >= 0 && rank_ < numRanks_);

  CUDA_CHECK(cudaGetDevice(&deviceId_));
  int lws = localWorldSize();
  rdmaRank_ = rank_ / lws;
  nvlRank_ = rank_ % lws;
  numRdmaRanks_ = std::max(1, numRanks_ / lws);
  numNvlRanks_ = std::min(numRanks_, lws);

  numProxyServices_ = resolveNumProxyServices(numRanks_, lws);
  proxyServices_.reserve(numProxyServices_);
  for (int i = 0; i < numProxyServices_; ++i) proxyServices_.emplace_back(std::make_shared<EPProxyService>());

  CUDA_CHECK(cudaStreamCreateWithFlags(&commStream_, cudaStreamNonBlocking));
  CUDA_CHECK(cudaMalloc(&workspace_, NUM_WORKSPACE_BYTES));
  CUDA_CHECK(cudaMemsetAsync(workspace_, 0, NUM_WORKSPACE_BYTES, commStream_));
  for (auto& ps : proxyServices_) ps->startProxy();
  setup();
}

MoERuntime::~MoERuntime() noexcept(false) {
  CUDA_CHECK(cudaDeviceSynchronize());
  if (peerRdmaBasesGpu_ != nullptr) CUDA_CHECK(cudaFree(peerRdmaBasesGpu_));
  for (auto& ps : proxyServices_) ps->stopProxy();
  if (workspace_ != nullptr) CUDA_CHECK(cudaFree(workspace_));
  if (rdmaBufferPtr_ != nullptr) {
    // gpuCallocPhysical allocations are intentionally process-lifetime in the
    // existing EP runtime. cudaFree is safe for the fallback cudaMalloc path.
    // Ignore failures from fabric allocations during process teardown.
    cudaFree(rdmaBufferPtr_);
  }
  if (commStream_ != nullptr) CUDA_CHECK(cudaStreamDestroy(commStream_));
}

bool MoERuntime::isAvailable() const { return available_; }
bool MoERuntime::isInternodeAvailable() const { return isAvailable() && numRanks_ > NUM_MAX_NVL_PEERS; }
int MoERuntime::getNumRdmaRanks() const { return numRdmaRanks_; }
int MoERuntime::getRdmaRank() const { return rdmaRank_; }
int MoERuntime::getRootRdmaRank(bool global) const { return global ? nvlRank_ : 0; }
int MoERuntime::getLocalDeviceId() const { return deviceId_; }
std::string MoERuntime::getLocalIpcHandle() const { return {}; }

void MoERuntime::setup() {
  EP_HOST_ASSERT(!available_);
  EP_HOST_ASSERT(communicator_ != nullptr);

  const std::vector<mscclpp::Transport> ibTransports = {
      mscclpp::Transport::IB0, mscclpp::Transport::IB1, mscclpp::Transport::IB2, mscclpp::Transport::IB3,
      mscclpp::Transport::IB4, mscclpp::Transport::IB5, mscclpp::Transport::IB6, mscclpp::Transport::IB7};
  const auto ipcTransport = mscclpp::Transport::CudaIpc;
  const auto ibTransport = ibTransports[deviceId_];
  const mscclpp::TransportFlags allTransport = ipcTransport | ibTransport;

  const bool fabricIpcSupported = resolveFabricIpcSupported();
  const bool useFabricIpcAlloc = mscclpp::isNvlsSupported() && fabricIpcSupported;
  if (useFabricIpcAlloc) {
    rdmaBufferPtr_ = mscclpp::detail::gpuCallocPhysical(numRdmaBytes_);
  } else {
    CUDA_CHECK(cudaMalloc(&rdmaBufferPtr_, numRdmaBytes_));
  }
  CUDA_CHECK(cudaMemset(rdmaBufferPtr_, 0, numRdmaBytes_));
  communicator_->bootstrap()->barrier();
  CUDA_CHECK(cudaDeviceSynchronize());

  std::map<int, mscclpp::MemoryId> memoryIds;
  auto addMemoryToAll = [&](mscclpp::RegisteredMemory mem) -> mscclpp::MemoryId {
    mscclpp::MemoryId id = static_cast<mscclpp::MemoryId>(-1);
    for (auto& ps : proxyServices_) {
      auto cur = ps->addMemory(mem);
      if (id == static_cast<mscclpp::MemoryId>(-1)) id = cur;
      EP_HOST_ASSERT(cur == id);
    }
    return id;
  };

  auto localRdmaBufferMem = communicator_->registerMemory(rdmaBufferPtr_, numRdmaBytes_, allTransport);
  memoryIds[rank_] = addMemoryToAll(localRdmaBufferMem);

  constexpr int kRdmaTag = 1;
  for (int r = 0; r < numRanks_; ++r) {
    if (r != rank_) communicator_->sendMemory(localRdmaBufferMem, r, kRdmaTag);
  }
  for (int r = 0; r < numRanks_; ++r) {
    if (r == rank_) continue;
    auto mem = communicator_->recvMemory(r, kRdmaTag).get();
    memoryIds[r] = addMemoryToAll(std::move(mem));
  }

  std::unordered_map<int, std::vector<mscclpp::Connection>> connections;
  const mscclpp::EndpointConfig ipcConfig(ipcTransport);
  const mscclpp::EndpointConfig ibConfig(ibTransport);
  connections[rank_].emplace_back(communicator_->connect(ipcConfig, rank_, kRdmaTag).get());

  constexpr int kNumIbConnectionsPerRank = 12;
  for (int r = 0; r < numRanks_; ++r) {
    if (r == rank_) continue;
    std::vector<std::shared_future<mscclpp::Connection>> futures;
    futures.reserve(kNumIbConnectionsPerRank);
    for (int i = 0; i < kNumIbConnectionsPerRank; ++i)
      futures.emplace_back(communicator_->connect(ibConfig, r, kRdmaTag));
    for (auto& f : futures) connections[r].emplace_back(f.get());
  }

  std::unordered_map<int, std::vector<std::pair<int, mscclpp::SemaphoreId>>> semaphoreIds;
  constexpr int kNumSemaphoresPerRank = 16;
  for (int i = 0; i < kNumSemaphoresPerRank; ++i) {
    for (int r = 0; r < numRanks_; ++r) {
      auto& conns = connections[r];
      auto& conn = conns[i % conns.size()];
      int proxyIdx = (i * numRanks_ + r) % numProxyServices_;
      auto semaphoreId = proxyServices_[proxyIdx]->buildAndAddSemaphore(*communicator_, conn);
      semaphoreIds[r].emplace_back(proxyIdx, semaphoreId);
    }
  }

  std::vector<mscclpp::PortChannelDeviceHandle> portChannelHandles;
  for (int i = 0; i < kNumSemaphoresPerRank; ++i) {
    for (int r = 0; r < numRanks_; ++r) {
      auto [proxyIdx, semaphoreId] = semaphoreIds[r][i % semaphoreIds[r].size()];
      auto portChannel = proxyServices_[proxyIdx]->portChannel(semaphoreId, memoryIds[r], memoryIds[rank_]);
      portChannels_.emplace_back(std::move(portChannel));
      portChannelHandles.emplace_back(portChannels_.rbegin()->deviceHandle());
    }
  }
  portChannelHandlesDevicePtr_ =
      mscclpp::detail::gpuCallocShared<mscclpp::PortChannelDeviceHandle>(portChannelHandles.size());
  mscclpp::gpuMemcpy<mscclpp::PortChannelDeviceHandle>(portChannelHandlesDevicePtr_.get(), portChannelHandles.data(),
                                                       portChannelHandles.size(), cudaMemcpyHostToDevice);

  const int ipcDomainSize = useFabricIpcAlloc ? numRanks_ : numNvlRanks_;
  auto isIpcPeer = [&](int peer) {
    return peer != rank_ && ipcDomainSize > 1 && rank_ / ipcDomainSize == peer / ipcDomainSize;
  };
  const bool wantPeerIpc = useFabricIpcAlloc || (mode_ == MoEMode::LOW_LATENCY && ipcDomainSize > 1);
  if (wantPeerIpc) {
    constexpr int kLlIpcTag = 2;
    auto rdmaMemIpc = communicator_->registerMemory(rdmaBufferPtr_, numRdmaBytes_, ipcTransport);
    std::vector<std::shared_future<mscclpp::RegisteredMemory>> remoteFutures(numRanks_);
    for (int r = 0; r < numRanks_; ++r) {
      if (r == rank_ || !isIpcPeer(r)) continue;
      communicator_->sendMemory(rdmaMemIpc, r, kLlIpcTag);
      remoteFutures[r] = communicator_->recvMemory(r, kLlIpcTag);
    }

    std::vector<mscclpp::Connection> llIpcConnections(numRanks_);
    std::vector<std::shared_future<mscclpp::Connection>> connectionFutures(numRanks_);
    for (int r = 0; r < numRanks_; ++r) {
      if (r == rank_ || !isIpcPeer(r)) continue;
      connectionFutures[r] = communicator_->connect(ipcConfig, r, kLlIpcTag);
    }
    for (int r = 0; r < numRanks_; ++r) {
      if (r == rank_ || !isIpcPeer(r)) continue;
      llIpcConnections[r] = connectionFutures[r].get();
    }

    peerRdmaBases_.assign(numRanks_, nullptr);
    peerRdmaBases_[rank_] = rdmaBufferPtr_;
    std::vector<mscclpp::RegisteredMemory> remoteMemories(numRanks_);
    for (int r = 0; r < numRanks_; ++r) {
      if (r == rank_ || !isIpcPeer(r)) continue;
      remoteMemories[r] = remoteFutures[r].get();
      peerRdmaBases_[r] = remoteMemories[r].data();
    }
    CUDA_CHECK(cudaMalloc(&peerRdmaBasesGpu_, sizeof(void*) * numRanks_));
    CUDA_CHECK(cudaMemcpy(peerRdmaBasesGpu_, peerRdmaBases_.data(), sizeof(void*) * numRanks_, cudaMemcpyHostToDevice));

    std::vector<mscclpp::BaseMemoryChannelDeviceHandle> llHandles(numRanks_);
    for (int r = 0; r < numRanks_; ++r) {
      if (r == rank_ || !isIpcPeer(r)) continue;
      auto sema = std::make_shared<mscclpp::MemoryDevice2DeviceSemaphore>(*communicator_, llIpcConnections[r]);
      llMemoryChannels_.emplace_back(sema, remoteMemories[r], rdmaMemIpc);
      llHandles[r] = llMemoryChannels_.rbegin()->deviceHandle();
    }
    llMemoryChannelHandlesDevicePtr_ =
        mscclpp::detail::gpuCallocShared<mscclpp::BaseMemoryChannelDeviceHandle>(numRanks_);
    mscclpp::gpuMemcpy<mscclpp::BaseMemoryChannelDeviceHandle>(llMemoryChannelHandlesDevicePtr_.get(), llHandles.data(),
                                                               numRanks_, cudaMemcpyHostToDevice);
    llRanksPerIpcDomain_ = ipcDomainSize;
    llIpcReady_ = ipcDomainSize >= numRanks_;
  }

  available_ = true;
}

void MoERuntime::dispatch(void* output, float* outputScales, int* outputSrcInfo, int64_t* outputLayout,
                          int* outputCount, const void* input, const int64_t* topkIdx, int numTokens, int hidden,
                          int numTopk, int numMaxDispatchTokensPerRank, int numExperts, int quantMode,
                          DispatchLayout dispatchLayout, cudaStream_t stream) {
  EP_HOST_ASSERT(mode_ == MoEMode::LOW_LATENCY);
  EP_HOST_ASSERT(hidden % sizeof(int4) == 0 && hidden % 128 == 0);
  EP_HOST_ASSERT(numTokens <= numMaxDispatchTokensPerRank);
  EP_HOST_ASSERT(numExperts % numRanks_ == 0);
  EP_HOST_ASSERT(dispatchLayout == DispatchLayout::EXPERT_MAJOR || dispatchLayout == DispatchLayout::FLAT);

  low_latency::DType outputDType = low_latency::DType::BF16;
  if (quantMode == 1) {
    outputDType = low_latency::DType::F8E4M3;
  } else if (quantMode == 2) {
    outputDType = low_latency::DType::MXF8E4M3;
  } else {
    EP_HOST_ASSERT(quantMode == 0 && "Unsupported low-latency dispatch quant mode");
  }

  LowLatencyLayout layout(rdmaBufferPtr_, numMaxDispatchTokensPerRank, hidden, numRanks_, numExperts);
  EP_HOST_ASSERT(layout.totalBytes <= static_cast<size_t>(numRdmaBytes_));
  auto buffer = layout.buffers[lowLatencyBufferIdx_];
  auto nextBuffer = layout.buffers[lowLatencyBufferIdx_ ^= 1];
  auto nextCleanMeta = nextBuffer.cleanMeta();

  low_latency::DispatchConfig config{.numTokens_ = numTokens,
                                     .hidden_ = hidden,
                                     .numTopk_ = numTopk,
                                     .numExperts_ = numExperts,
                                     .numMaxTokensPerRank_ = numMaxDispatchTokensPerRank,
                                     .inputDType_ = low_latency::DType::BF16,
                                     .outputDType_ = outputDType,
                                     .outputLayout_ = dispatchLayout};
  low_latency::BufferSet currentBuffer{.sendDataBuffer_ = buffer.dispatchRdmaSendBuffer,
                                       .sendCountBuffer_ = nullptr,
                                       .recvDataBuffer_ = buffer.dispatchRdmaRecvDataBuffer,
                                       .recvCountBuffer_ = buffer.dispatchRdmaRecvCountBuffer,
                                       .cleanupRegion_ = nullptr,
                                       .cleanupSize_ = 0};
  low_latency::BufferSet nextBufferSet{.sendDataBuffer_ = nullptr,
                                       .sendCountBuffer_ = nullptr,
                                       .recvDataBuffer_ = nullptr,
                                       .recvCountBuffer_ = nullptr,
                                       .cleanupRegion_ = nextCleanMeta.first,
                                       .cleanupSize_ = nextCleanMeta.second};
  low_latency::TransportContext transport{
      .rdmaBufferBase_ = rdmaBufferPtr_,
      .portChannels_ = portChannelHandlesDevicePtr_.get(),
      .memoryChannels_ = llMemoryChannelHandlesDevicePtr_ ? llMemoryChannelHandlesDevicePtr_.get() : nullptr,
      .peerBases_ = peerRdmaBasesGpu_,
      .ipcReady_ = llIpcReady_,
      .rank_ = rank_,
      .numRanks_ = numRanks_,
      .ranksPerIpcDomain_ = llRanksPerIpcDomain_};
  low_latency::dispatch(output, outputScales, outputSrcInfo, outputLayout, outputCount, input, topkIdx, config,
                        currentBuffer, nextBufferSet, transport, workspace_, stream, low_latency::SEND_AND_RECV);
}

void MoERuntime::combine(void* output, const void* input, const float* inputScales, const int64_t* topkIdx,
                         const float* topkWeights, const int* srcInfo, const int64_t* layoutRange, int numTokens,
                         int hidden, int numTopk, int numMaxDispatchTokensPerRank, int numExperts, int quantMode,
                         cudaStream_t stream) {
  EP_HOST_ASSERT(mode_ == MoEMode::LOW_LATENCY);
  EP_HOST_ASSERT(hidden % sizeof(int4) == 0 && hidden % 128 == 0);
  EP_HOST_ASSERT(numExperts % numRanks_ == 0);

  low_latency::DType inputDType = low_latency::DType::BF16;
  if (quantMode == 1) {
    inputDType = low_latency::DType::F8E4M3;
  } else if (quantMode == 2) {
    inputDType = low_latency::DType::MXF8E4M3;
  } else {
    EP_HOST_ASSERT(quantMode == 0 && "Unsupported low-latency combine quant mode");
  }

  LowLatencyLayout layout(rdmaBufferPtr_, numMaxDispatchTokensPerRank, hidden, numRanks_, numExperts);
  EP_HOST_ASSERT(layout.totalBytes <= static_cast<size_t>(numRdmaBytes_));
  auto buffer = layout.buffers[lowLatencyBufferIdx_];
  auto nextBuffer = layout.buffers[lowLatencyBufferIdx_ ^= 1];
  auto nextCleanMeta = nextBuffer.cleanMeta();

  low_latency::CombineConfig config{.numCombinedTokens_ = numTokens,
                                    .hidden_ = hidden,
                                    .numTopk_ = numTopk,
                                    .numExperts_ = numExperts,
                                    .numMaxTokensPerRank_ = numMaxDispatchTokensPerRank,
                                    .inputDType_ = inputDType,
                                    .outputDType_ = low_latency::DType::BF16,
                                    .zeroCopy_ = false};
  low_latency::BufferSet currentBuffer{.sendDataBuffer_ = buffer.combineRdmaSendBuffer,
                                       .sendCountBuffer_ = nullptr,
                                       .recvDataBuffer_ = buffer.combineRdmaRecvDataBuffer,
                                       .recvCountBuffer_ = buffer.combineRdmaRecvFlagBuffer,
                                       .cleanupRegion_ = nullptr,
                                       .cleanupSize_ = 0};
  low_latency::BufferSet nextBufferSet{.sendDataBuffer_ = nullptr,
                                       .sendCountBuffer_ = nullptr,
                                       .recvDataBuffer_ = nullptr,
                                       .recvCountBuffer_ = nullptr,
                                       .cleanupRegion_ = nextCleanMeta.first,
                                       .cleanupSize_ = nextCleanMeta.second};
  low_latency::TransportContext transport{
      .rdmaBufferBase_ = rdmaBufferPtr_,
      .portChannels_ = portChannelHandlesDevicePtr_.get(),
      .memoryChannels_ = llMemoryChannelHandlesDevicePtr_ ? llMemoryChannelHandlesDevicePtr_.get() : nullptr,
      .peerBases_ = peerRdmaBasesGpu_,
      .ipcReady_ = llIpcReady_,
      .rank_ = rank_,
      .numRanks_ = numRanks_,
      .ranksPerIpcDomain_ = llRanksPerIpcDomain_};
  low_latency::combine(output, input, inputScales, topkIdx, topkWeights, srcInfo, layoutRange, config, currentBuffer,
                       nextBufferSet, transport, workspace_, stream, low_latency::SEND_AND_RECV);
}

}  // namespace ep
}  // namespace mscclpp
