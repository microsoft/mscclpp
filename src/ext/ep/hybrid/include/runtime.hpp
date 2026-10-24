// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
// Host state of the hybrid NVLink/IPC + GPUNetIO latency path.
//
// LatencyRuntimeContext owns the symmetric buffer, workspace and IPC mappings. When the
// hybrid path is selected, it sizes them from this runtime and hands them over during
// initialize(); this runtime then owns the GPUNetIO service and the per-dispatch
// pipeline state, and launches the hybrid kernels.

#ifndef MSCCLPP_EP_HYBRID_RUNTIME_HPP_
#define MSCCLPP_EP_HYBRID_RUNTIME_HPP_

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/ext/ep/types.hpp>
#include <mscclpp/memory_channel_device.hpp>
#include <vector>

#include "hybrid/include/config.hpp"
#include "hybrid/include/device_context.hpp"

namespace mscclpp {
namespace ep {

class EpGpuNetIoService;
struct Workload;

namespace hybrid {

/// Whether the hybrid path supports @p hidden.
bool isSupportedHidden(int hidden);

/// Whether a latency runtime with this configuration must use the hybrid path.
/// @p useGpuNetIo selects GPUNetIO for ranks outside the CUDA IPC domain.
bool requiresHybrid(DispatchLayout outputLayout, bool useGpuNetIo);

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
  bool useGpuNetIo() const { return useGpuNetIo_; }

  /// Collectively set up the GPUNetIO backend and the hybrid device context.
  void initialize(mscclpp::Communicator& communicator, void* symmetricBuffer,
                  const std::vector<void*>& peerMappedBufferBases, void** peerMappedBufferBasesGpu,
                  mscclpp::BaseMemoryChannelDeviceHandle* channels, void* workspace, int maxSharedMemoryPerBlock,
                  int numSms, int deviceId);

  /// Storage layout over @p symmetricBuffer.
  LatencyStorageLayout layout(void* symmetricBuffer) const;

  void dispatch(void* output, int* outputTopkIdx, float* outputTopkWeights, int* outputCount, const void* input,
                const int64_t* topkIdx, const float* topkWeights, const Workload& workload, void* dispatchRecvBuffer,
                int numBlocks, cudaStream_t stream);

  void combine(void* output, const void* input, const int64_t* topkIdx, const float* topkWeights,
               const Workload& workload, void* combineRecvBuffer, void* dispatchRecvBuffer, int numBlocks,
               cudaStream_t stream);

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
  size_t symmetricBufferBytes_;
  size_t workspaceBytes_;
  DeviceContext deviceContext_{};
  std::shared_ptr<EpGpuNetIoService> gpuNetIoService_;
  std::shared_ptr<int> chunkReady_;
  std::shared_ptr<uint64_t> dispatchTickets_;
  std::shared_ptr<uint64_t> combineChunkBase_;
  std::shared_ptr<int> dispatchDone_;
  std::shared_ptr<GpuNetIoDispatchPeerState> dispatchPeers_;
};

}  // namespace hybrid
}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_HYBRID_RUNTIME_HPP_
