// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_GPU_NET_IO_SERVICE_HPP_
#define MSCCLPP_GPU_NET_IO_SERVICE_HPP_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "core.hpp"
#include "port_channel_gpunetio_device.hpp"

namespace mscclpp {

namespace detail {
struct GpuNetIoConnectionState;
struct GpuNetIoMemoryState;
struct GpuNetIoSemaphoreState;
struct GpuNetIoChannelState;
}  // namespace detail
struct BasePortChannel;
struct PortChannel;

/// An already-connected QP, selected by remote rank and queue index.
/// Copies retain the transport resources. Created by GpuNetIoService::connect.
class GpuNetIoConnection {
 public:
  GpuNetIoConnection() = default;

 private:
  std::shared_ptr<detail::GpuNetIoConnectionState> state_;
  friend class GpuNetIoService;
};

/// A local registration or exchanged remote registration, independent of a QP.
/// Copies retain the registration and any optional buffer owner.
class GpuNetIoMemory {
 public:
  GpuNetIoMemory() = default;

 private:
  std::shared_ptr<detail::GpuNetIoMemoryState> state_;
  friend class GpuNetIoService;
  friend struct BasePortChannel;
  friend struct PortChannel;
};

/// Owned inbound/expected counters and peer signal metadata bound to one QP.
/// Constructed by GpuNetIoService::buildSemaphore on the two connected peers.
class GpuNetIoSemaphore {
 public:
  GpuNetIoSemaphore() = default;

 private:
  std::shared_ptr<detail::GpuNetIoSemaphoreState> state_;
  friend class GpuNetIoService;
  friend struct BasePortChannel;
};

/// Collective setup of GPUNetIO queue pairs per remote bootstrap rank.
/// Unlike ProxyService, the GPU posts RDMA operations and rings NIC doorbells directly.
/// Requires GPU_SM_DB, valid DBRs and GPU-resident non-collapsed CQs. CPU-assisted
/// fallback is disabled for the pinned upstream version; unsupported systems fail setup.
/// All ranks call setup with reciprocal per-peer counts, or use the legacy uniform plan.
/// Channels select requested connections,
/// semaphores and separately registered buffers after transport setup.
/// Handles retain the transport and registrations; synchronize every using
/// stream before releasing the last channel/registration/buffer owner.
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

  /// Legacy symmetric-buffer setup for low-level device-context operations.
  /// New PortChannels should use a sparse plan and separate memory registrations.
  /// @param symmetricBuffer This rank's buffer, which must outlive the service.
  /// @param bytes Common buffer size across all ranks.
  void setup(void* symmetricBuffer, size_t bytes);

  /// Legacy full-mesh setup without registering a symmetric payload; uses bootstrap tag zero.
  /// All ranks call once with the same QP count. Channel buffers are registered separately.
  void setup();

  /// Collectively create only the QPs requested by this rank's connection plan.
  /// @param peerQpCounts One count per bootstrap rank, in [0, 64], with zero for self.
  /// Each pair must request the same count; different peers may have different counts.
  /// Ranks with no connections still participate with all-zero counts. No payload is registered.
  /// @param tag Common nonnegative bootstrap tag reserved for this setup's QP metadata exchange.
  /// All ranks call once in matching order; do not overlap setup with other traffic using this tag.
  void setup(const std::vector<int>& peerQpCounts, int tag);

  /// Select a requested QP after setup; rejects self, unrequested peers and queue indices.
  GpuNetIoConnection connect(int peer, int qpIndex = 0) const;

  /// Register a local CUDA buffer with this transport's protection domain.
  /// If owner is omitted, the caller must keep the allocation alive while any handles use it.
  GpuNetIoMemory registerMemory(void* buffer, size_t bytes, std::shared_ptr<void> owner = {}) const;

  /// Exchange one local registration with the connection's peer using a caller-chosen tag.
  /// Both peers call in matching order with matching QP indices and unique active tags.
  /// Sizes and addresses may differ. Serialize setup calls; do not share tags with other bootstrap traffic.
  GpuNetIoMemory exchangeMemory(const GpuNetIoConnection& connection, const GpuNetIoMemory& local, int tag) const;

  /// Allocate, zero, register and exchange private semaphore counters for a connection.
  /// Both peers call with the same tag. No payload buffer or external signal pointers are needed.
  GpuNetIoSemaphore buildSemaphore(const GpuNetIoConnection& connection, int tag) const;

  /// Device context for low-level operations, or nullptr until setup succeeds.
  /// Host PortChannel construction takes a service-created semaphore binding.
  GpuNetIoDeviceContext* deviceContext() const;

  /// Validate a remote bootstrap rank using host metadata and return the device context.
  /// Rejects self, unrequested/out-of-range peers and incomplete setup in every build mode.
  GpuNetIoDeviceContext* deviceContext(int peer) const;

 private:
  struct Impl;
  std::shared_ptr<Impl> pimpl_;
  void setupImpl(void* symmetricBuffer, size_t bytes, const std::vector<int>& peerQpCounts, int tag);
  GpuNetIoMemory exchangeMemoryImpl(const GpuNetIoConnection&, const GpuNetIoMemory&, int tag, uint32_t kind) const;
  friend struct detail::GpuNetIoConnectionState;
  friend struct detail::GpuNetIoMemoryState;
  friend struct detail::GpuNetIoChannelState;
  friend struct BasePortChannel;
};

}  // namespace mscclpp

#endif