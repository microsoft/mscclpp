// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_GPUNETIO_HPP_
#define MSCCLPP_GPUNETIO_HPP_

#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/semaphore.hpp>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "connection.hpp"
#include "context.hpp"
#include "ib.hpp"

#if defined(MSCCLPP_USE_GPUNETIO)
extern "C" {
#include "host/doca_gpunetio.h"
#include "host/doca_gpunetio_high_level.h"
#include "host/doca_verbs.h"
}
#endif

namespace mscclpp {

/// DOCA device handles shared by the GPUNetIO QPs of an IB transport.
struct GpuNetIoContext {
  explicit GpuNetIoContext(IbCtx& ibCtx);
  ~GpuNetIoContext();

#if defined(MSCCLPP_USE_GPUNETIO)
  doca_gpu_t* gpu(int gpuId);

  std::mutex mutex_;
  doca_dev_t* dev_ = nullptr;
  std::unordered_map<int, doca_gpu_t*> gpus_;
#endif
};

/// A GPU-initiated RC queue pair (DOCA GPUNetIO) created on a Context's IB device and protection domain, so memory
/// registered through the same Context is directly addressable by it.
class GpuNetIoQp {
 public:
  /// @param ibCtx The IB device context whose protection domain the QP is created on.
  /// @param context DOCA device handles for this IB transport.
  GpuNetIoQp(IbCtx& ibCtx, std::shared_ptr<GpuNetIoContext> context, int gpuId, int port, int gidIndex);
  ~GpuNetIoQp();

  /// Addressing info that the remote endpoint needs to connect to this QP.
  const IbQpInfo& info() const { return info_; }

  /// Transition the QP to RTS, connected to the given remote QP.
  void connect(const IbQpInfo& remote);

  /// The GPU-resident QP (`doca_gpu_dev_verbs_qp*`) used by device code.
  void* deviceQp() const { return deviceQp_; }

 private:
  std::shared_ptr<GpuNetIoContext> context_;
#if defined(MSCCLPP_USE_GPUNETIO)
  doca_gpu_verbs_qp_hl* qp_ = nullptr;
#endif
  void* deviceQp_ = nullptr;
  int gpuId_ = 0;
  int port_ = 0;
  int gidIndex_ = 0;
  IbQpInfo info_{};
};

/// A connection whose operations are posted directly by GPU threads through PortChannel device handles.
/// Host-side Connection operations are rejected.
class GpuNetIoConnection : public BaseConnection {
 public:
  GpuNetIoConnection(std::shared_ptr<Context> context, const Endpoint& localEndpoint, const Endpoint& remoteEndpoint);
  ~GpuNetIoConnection() = default;

  Transport transport() const override { return transport_; }

  Transport remoteTransport() const override { return remoteTransport_; }

  bool isDeviceBackend() const override { return true; }

  DeviceBackendHandle getDeviceBackendInfo(const RegisteredMemory& signal, uint64_t* outboundToken) const override;

  DeviceMemoryHandle getDeviceMemoryInfo(const RegisteredMemory& memory) const override;

  void write(RegisteredMemory dst, uint64_t dstOffset, RegisteredMemory src, uint64_t srcOffset,
             uint64_t size) override;
  void updateAndSync(RegisteredMemory dst, uint64_t dstOffset, uint64_t* src, uint64_t newValue) override;
  void accumulate(RegisteredMemory dst, uint64_t dstOffset, int64_t value) override;
  void flush(int64_t timeoutUsec) override;

 private:
  Transport transport_;
  Transport remoteTransport_;
  std::shared_ptr<GpuNetIoQp> qp_;
  std::shared_ptr<uint64_t> atomicFetchBuff_;
  RegisteredMemory atomicFetchMem_;
};

}  // namespace mscclpp

#endif  // MSCCLPP_GPUNETIO_HPP_
