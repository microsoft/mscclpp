// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_HYBRID_GPU_NET_IO_HPP_
#define MSCCLPP_EP_HYBRID_GPU_NET_IO_HPP_

#include <memory>
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
/// so offsets below are symmetric-buffer offsets. Unlike PortChannelDeviceHandle::put(), these helpers do not fence;
/// callers order their source writes (see the EP row-fence modes).
struct EpGpuNetIoDeviceContext {
  PortChannelDeviceHandle* channels;
  EpGpuNetIoSignalState* signalStates;
  int numPeers;
  int numQpsPerPeer;
  int numHcas;

#if defined(MSCCLPP_DEVICE_COMPILE)
  MSCCLPP_DEVICE_INLINE static void require(bool valid) { GpuNetIoDeviceHandle::require(valid); }

  MSCCLPP_DEVICE_INLINE PortChannelDeviceHandle channel(int peer, int queue) const {
    require(channels != nullptr && peer >= 0 && peer < numPeers && queue >= 0 && queue < numQpsPerPeer);
    const auto handle = channels[static_cast<size_t>(peer) * numQpsPerPeer + queue];
    require(handle.backend_ == PortChannelBackend::GpuNetIo && handle.gpuNetIo_.qp != nullptr);
    return handle;
  }

  MSCCLPP_DEVICE_INLINE EpGpuNetIoSignalState& signalState(int peer, int queue) const {
    require(signalStates != nullptr && peer >= 0 && peer < numPeers && queue >= 0 && queue < numQpsPerPeer);
    return signalStates[static_cast<size_t>(peer) * numQpsPerPeer + queue];
  }

#if defined(MSCCLPP_USE_GPUNETIO)
  MSCCLPP_DEVICE_INLINE static doca_gpu_dev_verbs_qp* qp(const PortChannelDeviceHandle& handle) {
    return static_cast<doca_gpu_dev_verbs_qp*>(handle.gpuNetIo_.qp);
  }

  MSCCLPP_DEVICE_INLINE static doca_gpu_dev_verbs_addr remoteAddr(const PortChannelDeviceHandle& handle,
                                                                  uint64_t offset, uint64_t bytes) {
    const auto memory = handle.gpuNetIo_.memory(handle.dst_, false, offset, bytes);
    return {memory.addr + offset, memory.key};
  }

  MSCCLPP_DEVICE_INLINE static doca_gpu_dev_verbs_addr localAddr(const PortChannelDeviceHandle& handle, uint64_t offset,
                                                                 uint64_t bytes) {
    const auto memory = handle.gpuNetIo_.memory(handle.src_, true, offset, bytes);
    return {memory.addr + offset, memory.key};
  }

  template <typename Fn>
  MSCCLPP_DEVICE_INLINE void withChannelLock(int peer, int queue, Fn&& fn) const {
    auto& state = signalState(peer, queue);
    bool done = false;
    do {
      if (atomicCAS(&state.lock, 0u, 1u) == 0u) {
        doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
        fn(state);
        doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
        atomicExch(&state.lock, 0u);
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
    require(false);
    return 0;
  }
#endif

  MSCCLPP_DEVICE_INLINE void put(int peer, uint64_t destination, uint64_t source, uint64_t bytes, int queue = 0) {
#if defined(MSCCLPP_USE_GPUNETIO)
    auto handle = channel(peer, queue);
    withChannelLock(peer, queue, [&](EpGpuNetIoSignalState&) { handle.put(destination, source, bytes); });
#else
    require(false);
#endif
  }

  MSCCLPP_DEVICE_INLINE void atomicAdd(int peer, uint64_t destination, int64_t value, int queue = 0) {
#if defined(MSCCLPP_USE_GPUNETIO)
    auto handle = channel(peer, queue);
    const auto dst = handle.gpuNetIo_.memory(handle.dst_, false, destination, sizeof(uint64_t));
    withChannelLock(peer, queue, [&](EpGpuNetIoSignalState& state) {
      handle.gpuNetIo_.signalAt(dst, destination, nextSignalToken(state, destination, static_cast<uint64_t>(value)));
    });
#else
    require(false);
#endif
  }

  /// RDMA write followed by an ordered inline semaphore update of `signalValue` at symmetric offset `signalOffset`.
  MSCCLPP_DEVICE_INLINE void putWithSignal(int peer, uint64_t destination, uint64_t source, uint64_t bytes,
                                           uint64_t signalOffset, uint64_t signalValue, int queue = 0) {
#if defined(MSCCLPP_USE_GPUNETIO)
    auto handle = channel(peer, queue);
    const auto dst = handle.gpuNetIo_.memory(handle.dst_, false, destination, bytes);
    const auto src = handle.gpuNetIo_.memory(handle.src_, true, source, bytes);
    const auto sig = handle.gpuNetIo_.memory(handle.dst_, false, signalOffset, sizeof(uint64_t));
    withChannelLock(peer, queue, [&](EpGpuNetIoSignalState& state) {
      handle.gpuNetIo_.putWithSignalAt(dst, destination, src, source, bytes, sig, signalOffset,
                                       nextSignalToken(state, signalOffset, signalValue));
    });
#else
    require(false);
#endif
  }

  MSCCLPP_DEVICE_INLINE void flush(int peer, int queue = 0) { channel(peer, queue).flush(-1); }

  /// Poll for completion of every WQE posted so far on (peer, queue) for at most `spins` iterations.
  /// @return 0 when drained, EBUSY if still pending, or the completion error status.
  MSCCLPP_DEVICE_INLINE int tryFlush(int peer, uint64_t spins, int queue = 0) {
#if defined(MSCCLPP_USE_GPUNETIO)
    return channel(peer, queue).gpuNetIo_.tryFlush(spins);
#else
    require(false);
    return 0;
#endif
  }

  /// Send-queue reservation index of (peer, queue): every WQE posted so far lies below it.
  MSCCLPP_DEVICE_INLINE uint64_t reservedTicket(int peer, int queue = 0) {
#if defined(MSCCLPP_USE_GPUNETIO)
    return channel(peer, queue).gpuNetIo_.reservedTicket();
#else
    require(false);
    return 0;
#endif
  }

  /// Wait until every WQE below `ticket` (from reservedTicket) on (peer, queue) has completed.
  MSCCLPP_DEVICE_INLINE void waitTicket(int peer, uint64_t ticket, int queue = 0) {
#if defined(MSCCLPP_USE_GPUNETIO)
    if (ticket == 0) return;
    auto handle = channel(peer, queue);
    auto& state = signalState(peer, queue);
    while (!handle.gpuNetIo_.isTicketDone(ticket)) {
      if (atomicCAS(&state.lock, 0u, 1u) == 0u) {
        doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
        if (!handle.gpuNetIo_.isTicketDone(ticket)) {
          handle.gpuNetIo_.waitTicket(ticket);
        }
        doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
        atomicExch(&state.lock, 0u);
        return;
      }
    }
#else
    require(false);
#endif
  }

  MSCCLPP_DEVICE_INLINE void putBatched3(int peer, int queue, uint64_t dst0, uint64_t src0, uint64_t size0,
                                         uint64_t dst1, uint64_t src1, uint64_t size1, uint64_t dst2, uint64_t src2,
                                         uint64_t size2) {
#if defined(MSCCLPP_USE_GPUNETIO)
    auto handle = channel(peer, queue);
    const uint64_t dstOffsets[3] = {dst0, dst1, dst2};
    const uint64_t srcOffsets[3] = {src0, src1, src2};
    const uint64_t sizes[3] = {size0, size1, size2};
    GpuNetIoMemoryDeviceHandle dst[3], src[3];
    for (int index = 0; index < 3; ++index) {
      dst[index] = handle.gpuNetIo_.memory(handle.dst_, false, dstOffsets[index], sizes[index]);
      src[index] = handle.gpuNetIo_.memory(handle.src_, true, srcOffsets[index], sizes[index]);
    }
    withChannelLock(peer, queue, [&](EpGpuNetIoSignalState&) {
      handle.gpuNetIo_.putBatched3(dst, dstOffsets, src, srcOffsets, sizes);
    });
#else
    require(false);
#endif
  }
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