// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_GPU_NET_IO_SERVICE_HPP_
#define MSCCLPP_GPU_NET_IO_SERVICE_HPP_

#include <cstddef>
#include <memory>
#include <string>

#include "core.hpp"
#include "port_channel_gpunetio_device.hpp"

namespace mscclpp {

/// Collective setup of GPUNetIO queue pairs per remote bootstrap rank.
/// Unlike ProxyService, the GPU posts RDMA operations and rings NIC doorbells directly.
/// Requires GPU_SM_DB, valid DBRs and GPU-resident non-collapsed CQs. CPU-assisted
/// fallback is disabled for the pinned upstream version; unsupported systems fail setup.
/// All ranks must call setup with the same buffer size and offset layout.
/// Synchronize every using stream before destruction, and destroy the service
/// before freeing its symmetric buffer or any channel signal counters.
class GpuNetIoService {
 public:
  /// @param bootstrap Bootstrap used for QP and registered-memory exchange.
  /// @param ibDeviceName Explicit local IB device name, for example "mlx5_0".
  /// @param cudaDeviceId CUDA device ordinal owning the symmetric buffer.
  GpuNetIoService(std::shared_ptr<Bootstrap> bootstrap, const std::string& ibDeviceName, int cudaDeviceId);
  /// Multi-QP variant; other arguments have the same meaning as above.
  /// @param numQpsPerPeer QPs per remote rank in [1, 64], identical on all ranks.
  /// Validated collectively by setup. The three-argument constructor uses one QP.
  GpuNetIoService(std::shared_ptr<Bootstrap> bootstrap, const std::string& ibDeviceName, int cudaDeviceId,
                  int numQpsPerPeer);
  ~GpuNetIoService();
  GpuNetIoService(const GpuNetIoService&) = delete;
  GpuNetIoService& operator=(const GpuNetIoService&) = delete;

  /// Register a CUDA buffer and publish the device context. Call exactly once.
  /// @param symmetricBuffer This rank's buffer, which must outlive the service.
  /// @param bytes Common buffer size across all ranks.
  void setup(void* symmetricBuffer, size_t bytes);

  /// Device context for low-level operations, or nullptr until setup succeeds.
  /// Host PortChannel construction takes the service itself for peer validation.
  GpuNetIoDeviceContext* deviceContext() const;

  /// Validate a remote bootstrap rank using host metadata and return the device context.
  /// Rejects self, out-of-range peers and incomplete setup in every build mode.
  GpuNetIoDeviceContext* deviceContext(int peer) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

}  // namespace mscclpp

#endif