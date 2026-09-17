// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef MSCCLPP_GPUNETIO_HOST_GPU_NET_IO_SERVICE_HPP_
#define MSCCLPP_GPUNETIO_HOST_GPU_NET_IO_SERVICE_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <mscclpp/core.hpp>
#include <mscclpp/port_channel_gpunetio_device.hpp>

namespace mscclpp {

/// Host-side bring-up for the GPU-initiated networking (GPUNetIO / GDAKI)
/// PortChannel backend. This is the counterpart to `ProxyService`: where
/// `ProxyService` runs a CPU thread that consumes a FIFO and issues verbs,
/// `GpuNetIoService` instead sets up the per-peer GDAKI queue pairs and memory
/// registration so that the *device* can issue RDMA directly (no CPU on the
/// data path), then publishes a `GpuNetIoDeviceContext` to GPU memory.
///
/// Usage (all ranks, symmetric):
///   GpuNetIoService svc(bootstrap, ibDeviceName, cudaDeviceId);
///   svc.setup(symmetricBuffer, symmetricBytes);
///   auto* ctx = svc.deviceContext();   // device pointer for the channel handle
///
/// The symmetric buffer must be the same size and identically offset-addressed
/// on every rank (the EP runtimes already guarantee this).
class GpuNetIoService {
 public:
  /// @param bootstrap Bootstrap used for the QP-info / rkey all-gather.
  /// @param ibDeviceName Comma-separated IB devices; QPs are striped across them.
  /// An empty string selects all active HCAs with the best PCI/NUMA affinity
  /// for this GPU, sorted by device name. GPUs may share the same local HCA set;
  /// it is not divided among local ranks. Explicit device lists are unchanged.
  /// @param cudaDeviceId CUDA device ordinal that owns the symmetric buffer.
  GpuNetIoService(std::shared_ptr<Bootstrap> bootstrap, const std::string& ibDeviceName, int cudaDeviceId);

  ~GpuNetIoService();

  GpuNetIoService(const GpuNetIoService&) = delete;
  GpuNetIoService& operator=(const GpuNetIoService&) = delete;

  /// Register the symmetric buffer on every selected HCA, connect per-peer QPs,
  /// exchange per-HCA rkeys and base addresses, and build the device context.
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
