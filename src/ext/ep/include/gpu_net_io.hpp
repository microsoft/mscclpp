// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_GPU_NET_IO_HPP_
#define MSCCLPP_EP_GPU_NET_IO_HPP_

#include <memory>
#include <mscclpp/atomic_device.hpp>
#include <mscclpp/core.hpp>
#include <mscclpp/port_channel_device.hpp>
#include <string>

namespace mscclpp::ep {

struct EpGpuNetIoSignalState {
  unsigned int lock;
  unsigned int pad;
  uint64_t offsets[4];
  uint64_t tokens[4];
};

/// Device view of the EP GPUNetIO channels: one PortChannel per (peer, queue), each bound to its own queue pair over
/// an EndpointConfig::Ib::Mode::GpuNetIo connection. Every channel's dst_/src_ are the peer's / local symmetric buffer,
/// so offsets below are symmetric-buffer offsets. As with GPUNetIO PortChannel operations, callers establish
/// source-data visibility; DOCA supplies queue-publication ordering (see the EP row-fence modes).
struct EpGpuNetIoDeviceContext {
  PortChannelDeviceHandle* channels;
  EpGpuNetIoSignalState* signalStates;
  int numPeers;
  int numQpsPerPeer;
  int numHcas;

#if defined(MSCCLPP_DEVICE_COMPILE)
  MSCCLPP_DEVICE_INLINE PortChannelDeviceHandle channel(int peer, int queue) const {
    GpuNetIoDeviceHandle::require(channels != nullptr && peer >= 0 && peer < numPeers && queue >= 0 &&
                                  queue < numQpsPerPeer);
    const auto handle = channels[static_cast<size_t>(peer) * numQpsPerPeer + queue];
    GpuNetIoDeviceHandle::require(handle.backend_ == PortChannelBackend::GpuNetIo && handle.gpuNetIo_.qp != nullptr);
    return handle;
  }

  /// Mutable notification state; callers must own the QP or hold its lock.
  MSCCLPP_DEVICE_INLINE EpGpuNetIoSignalState& signalState(int peer, int queue) const {
    GpuNetIoDeviceHandle::require(signalStates != nullptr && peer >= 0 && peer < numPeers && queue >= 0 &&
                                  queue < numQpsPerPeer);
    return signalStates[static_cast<size_t>(peer) * numQpsPerPeer + queue];
  }

#if defined(MSCCLPP_USE_GPUNETIO)
  /// Execute an operation or posting phase with exclusive ownership of the existing PortChannel's QP.
  template <typename Fn>
  MSCCLPP_DEVICE_INLINE void withChannelLock(int peer, int queue, Fn&& fn) const {
    auto& state = signalState(peer, queue);
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> lock(state.lock);
    bool done = false;
    do {
      unsigned int expected = 0;
      if (lock.compare_exchange_weak(expected, 1u, cuda::memory_order_acquire, cuda::memory_order_relaxed)) {
        fn(state);
        lock.store(0u, cuda::memory_order_release);
        done = true;
      }
    } while (!done);
  }

  MSCCLPP_DEVICE_INLINE static uint64_t nextSignalToken(EpGpuNetIoSignalState& state, uint64_t signalOffset,
                                                        uint64_t signalValue) {
    for (int i = 0; i < 4; ++i) {
      if (state.tokens[i] == 0 || state.offsets[i] == signalOffset) {
        state.offsets[i] = signalOffset;
        return state.tokens[i] += signalValue;
      }
    }
    GpuNetIoDeviceHandle::require(false);
    return 0;
  }
#endif
#endif
};

class EpGpuNetIoService {
 public:
  EpGpuNetIoService(std::shared_ptr<Bootstrap> bootstrap, const std::string& devices, int cudaDevice);
  ~EpGpuNetIoService();
  EpGpuNetIoService(const EpGpuNetIoService&) = delete;
  EpGpuNetIoService& operator=(const EpGpuNetIoService&) = delete;
  void setup(void* buffer, size_t bytes);
  EpGpuNetIoDeviceContext* deviceContext() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mscclpp::ep

#endif