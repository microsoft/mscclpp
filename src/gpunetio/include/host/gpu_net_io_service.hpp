// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_GPUNETIO_HOST_GPU_NET_IO_SERVICE_HPP_
#define MSCCLPP_GPUNETIO_HOST_GPU_NET_IO_SERVICE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/port_channel_gpunetio_device.hpp>
#include <string>

namespace mscclpp {

/// Host-side bring-up for the GPU-initiated networking (GPUNetIO / GDAKI)
/// PortChannel backend. This is the counterpart to `ProxyService`: where
/// `ProxyService` runs a CPU thread that consumes a FIFO and issues verbs,
/// `GpuNetIoService` instead sets up the per-peer GDAKI queue pairs and memory
/// registration so that the *device* can issue RDMA directly (no CPU on the
/// data path), then publishes a `GpuNetIoDeviceContext` to GPU memory.
///
/// Usage (all ranks, symmetric):
///   GpuNetIoService svc(bootstrap, ibDeviceNames, cudaDeviceId);
///   svc.setup(symmetricBuffer, symmetricBytes);
///   auto* ctx = svc.deviceContext();   // device pointer for the channel handle
///
/// The symmetric buffer must be the same size and identically offset-addressed
/// on every rank (the EP runtimes already guarantee this).
class GpuNetIoService {
 public:
  /// @param bootstrap Bootstrap used for the QP-info / rkey all-gather.
  /// @param ibDeviceNames Comma-separated IB devices to use (e.g.
  /// "mlx5_0,mlx5_1"). Logical QPs are striped across the devices in order.
  /// An empty string enables automatic PCI/NUMA-aware assignment across the
  /// active HCAs and GPU ranks on each node.
  /// @param cudaDeviceId CUDA device ordinal that owns the symmetric buffer.
  GpuNetIoService(std::shared_ptr<Bootstrap> bootstrap, const std::string& ibDeviceNames, int cudaDeviceId);

  ~GpuNetIoService();

  GpuNetIoService(const GpuNetIoService&) = delete;
  GpuNetIoService& operator=(const GpuNetIoService&) = delete;

  /// Register the symmetric buffer on every configured HCA, create + connect
  /// GDAKI QPs per remote rank, exchange rkeys / base addresses, and build the
  /// device context.
  /// Idempotent guard: must be called exactly once.
  /// @param symmetricBuffer Device pointer to this rank's symmetric buffer.
  /// @param bytes Size of the symmetric buffer.
  void setup(void* symmetricBuffer, size_t bytes);

  /// Device pointer to the published `GpuNetIoDeviceContext` (valid after
  /// `setup`). Embed this into a PortChannel device handle whose backend is
  /// `PortChannelBackend::GpuNetIo`.
  GpuNetIoDeviceContext* deviceContext() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> pimpl_;
};

}  // namespace mscclpp

#endif  // MSCCLPP_GPUNETIO_HOST_GPU_NET_IO_SERVICE_HPP_
