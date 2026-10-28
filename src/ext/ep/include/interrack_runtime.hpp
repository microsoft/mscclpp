// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
// Network service and pipeline state for NVLink/IPC + GPUNetIO latency kernels.
//
// LatencyRuntimeContext owns the symmetric buffer, workspace and IPC mappings. When the
// network path is selected, it sizes them from this runtime and hands them over during
// initialize(); this runtime then owns the GPUNetIO service and the per-dispatch
// pipeline state, and launches the rank-major network algorithms.

#ifndef MSCCLPP_EP_INTERRACK_RUNTIME_HPP_
#define MSCCLPP_EP_INTERRACK_RUNTIME_HPP_

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/ext/ep/types.hpp>
#include <mscclpp/memory_channel_device.hpp>
#include <vector>

#include "config.hpp"
#include "device_context.hpp"

namespace mscclpp {
namespace ep {

class EpGpuNetIoService;
struct Workload;

namespace interrack {

/// Whether the network path supports @p hidden.
bool isSupportedHidden(int hidden);

/// Whether GPUNetIO cross-domain transport is requested and supported for this configuration.
bool gpuNetIoRequested(int numRanks, int numRanksPerIpcDomain, DispatchLayout outputLayout, CombineMode combineMode);

class LatencyRuntime {
 public:
  LatencyRuntime(int rank, int numRanks, int numRanksPerIpcDomain, int numRanksPerNode, int maxTokensPerRank,
                 int hidden, int numExperts, int numTopk, DispatchLayout outputLayout, CombineMode combineMode,
                 bool useGpuNetIo);
  ~LatencyRuntime() noexcept(false);

  LatencyRuntime(const LatencyRuntime&) = delete;
  LatencyRuntime& operator=(const LatencyRuntime&) = delete;

  size_t symmetricBufferBytes() const { return symmetricBufferBytes_; }
  size_t workspaceBytes() const { return workspaceBytes_; }

  /// Collectively set up GPUNetIO and populate the shared device context.
  void initialize(mscclpp::Communicator& communicator, void* symmetricBuffer,
                  const std::vector<void*>& peerMappedBufferBases, void** peerMappedBufferBasesGpu,
                  mscclpp::BaseMemoryChannelDeviceHandle* channels, void* workspace, int maxSharedMemoryPerBlock,
                  int numSms, int deviceId);

  /// Storage layout over @p symmetricBuffer.
  LatencyStorageLayout layout(void* symmetricBuffer) const;

  void dispatch(void* output, int* outputTopkIdx, float* outputTopkWeights, int* outputCount, const void* input,
                const int64_t* topkIdx, const float* topkWeights, const Workload& workload, void* dispatchRecvBuffer,
                int numBlocks, cudaStream_t stream);

  void combine(void* output, const void* input, const int64_t* topkIdx, const Workload& workload,
               void* combineRecvBuffer, void* dispatchRecvBuffer, int numBlocks, cudaStream_t stream);

 private:
  int rank_;
  int numRanks_;
  int numRanksPerIpcDomain_;
  int numRanksPerNode_;
  int maxTokensPerRank_;
  int hidden_;
  int numExperts_;
  int numTopk_;
  DispatchLayout outputLayout_;
  CombineMode combineMode_;
  bool useGpuNetIo_;
  bool sendDistribute_;
  bool aggregateCombine_;
  size_t symmetricBufferBytes_;
  size_t workspaceBytes_;
  DeviceContext deviceContext_{};
  std::shared_ptr<EpGpuNetIoService> gpuNetIoService_;
  std::shared_ptr<int> chunkReady_;
  std::shared_ptr<uint64_t> dispatchTickets_;
  std::shared_ptr<uint64_t> combineChunkBase_;
  std::shared_ptr<int> dispatchDone_;
  std::shared_ptr<GpuNetIoDispatchPeerState> dispatchPeers_;
  std::shared_ptr<int> distributeScratch_;
  std::shared_ptr<int> aggregateMetadata_;
  std::shared_ptr<float> aggregateLocal_;
  std::shared_ptr<int> aggregateChunkReady_;
};

}  // namespace interrack
}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_INTERRACK_RUNTIME_HPP_
