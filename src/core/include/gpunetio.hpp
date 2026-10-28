// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_GPUNETIO_HPP_
#define MSCCLPP_GPUNETIO_HPP_

#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/port_channel_device.hpp>
#include <mscclpp/semaphore.hpp>
#include <vector>

#include "connection.hpp"
#include "context.hpp"
#include "ib.hpp"

namespace mscclpp {

struct GpuNetIoContext;

/// A GPU-initiated RC queue pair (DOCA GPUNetIO) created on a Context's IB device and protection domain, so memory
/// registered through the same Context is directly addressable by it.
class GpuNetIoQp {
 public:
  /// @param ibCtx The IB device context whose protection domain the QP is created on.
  /// @param sharedContext DOCA device handles shared by the QPs of a Context; created on first use.
  GpuNetIoQp(IbCtx& ibCtx, std::shared_ptr<GpuNetIoContext>& sharedContext, Transport transport, int gpuId, int port,
             int gidIndex);
  ~GpuNetIoQp();

  /// Addressing info that the remote endpoint needs to connect to this QP.
  const IbQpInfo& info() const;

  /// Transition the QP to RTS, connected to the given remote QP.
  void connect(const IbQpInfo& remote);

  /// The GPU-resident QP (`doca_gpu_dev_verbs_qp*`) used by device code.
  void* deviceQp() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

/// A connection whose operations are posted directly by GPU threads through PortChannel device handles.
/// Host-side Connection operations are rejected.
class GpuNetIoConnection : public BaseConnection {
 public:
  GpuNetIoConnection(std::shared_ptr<Context> context, const Endpoint& localEndpoint, const Endpoint& remoteEndpoint);
  ~GpuNetIoConnection();

  Transport transport() const override;

  Transport remoteTransport() const override;

  void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
             uint64_t size) override;
  void updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src, uint64_t newValue) override;
  void accumulate(RegisteredMemory dst, uint64_t dstOffset, int64_t value) override;
  void flush(int64_t timeoutUsec) override;

  /// Build the device-side state for a channel on this connection.
  /// @param signal The remote semaphore token targeted by signals.
  /// @param outboundToken The device-resident outbound counter of the channel's semaphore.
  /// @param memories The memory table indexed by MemoryId. Entries that are neither local nor owned by the remote
  /// peer of this connection, or are not registered for the corresponding IB transport, are marked unusable.
  GpuNetIoDeviceHandle deviceHandle(const RegisteredMemory& signal, uint64_t* outboundToken,
                                    std::vector<GpuNetIoMemoryDeviceHandle>& table,
                                    const std::vector<RegisteredMemory>& memories) const;

 private:
  Transport transport_;
  Transport remoteTransport_;
  uint64_t remoteHostHash_;
  uint64_t remotePidHash_;
  std::shared_ptr<GpuNetIoQp> qp_;
  std::shared_ptr<uint64_t> atomicScratch_;
  RegisteredMemory atomicScratchMem_;
  uint32_t wqeSlot_;
};

}  // namespace mscclpp

#endif  // MSCCLPP_GPUNETIO_HPP_
