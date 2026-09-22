// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_PORT_CHANNEL_DEVICE_HPP_
#define MSCCLPP_PORT_CHANNEL_DEVICE_HPP_

#include "fifo_device.hpp"
#include "port_channel_gpunetio_device.hpp"
#include "semaphore_device.hpp"

namespace mscclpp {

/// Backend that services a PortChannel's device-side operations.
/// Mirrors NCCL GIN's backend selection: the same device API (`put`, `signal`,
/// `putWithSignal`, `flush`, `atomicAdd`) is serviced either by the CPU proxy
/// (FIFO + ProxyService thread) or by GPU-initiated networking (GPUNetIO/GDAKI,
/// kernel-issued RDMA). The backend is chosen at channel creation.
enum class PortChannelBackend : uint8_t {
  Proxy = 0,     ///< CPU proxy: device pushes ProxyTrigger to the host FIFO.
  GpuNetIo = 1,  ///< GPU-initiated: device issues RDMA WQEs directly (DOCA GDAKI).
};

/// Numeric ID of Semaphore. ProxyService has an internal array indexed by these handles mapping to the
/// actual semaphores.
using SemaphoreId = uint32_t;

/// Numeric ID of RegisteredMemory. ProxyService has an internal array indexed by these handles mapping to the
/// actual.
using MemoryId = uint32_t;

namespace detail {
#if defined(MSCCLPP_DEVICE_COMPILE)
/// Wait until the proxy has processed and drained the TriggerSync at FIFO position `fifoPos`.
/// The proxy publishes `flushDonePos = latestCompletedPos + 1` when the CQ drains, so the
/// wait condition `flushDonePos > fifoPos` is satisfied exactly when our own request has
/// been completed. Using the FIFO push position as the wait target couples the wait to the
/// FIFO order, avoiding races when multiple GPU threads concurrently flush the same channel.
MSCCLPP_DEVICE_INLINE void waitFlush(uint64_t* flushDonePos, uint64_t fifoPos, [[maybe_unused]] int64_t maxSpinCount) {
  POLL_MAYBE_JAILBREAK((atomicLoad<uint64_t, scopeSystem>(flushDonePos, memoryOrderAcquire) <= fifoPos), maxSpinCount);
}
#endif  // defined(MSCCLPP_DEVICE_COMPILE)
}  // namespace detail

struct BasePortChannelDeviceHandle {
  SemaphoreId semaphoreId_;

  Host2DeviceSemaphoreDeviceHandle semaphore_;

  // this is a concurrent fifo which is multiple threads from the device
  // can produce for and the sole proxy thread consumes it.
  FifoDeviceHandle fifo_;

  // One past the highest FIFO position with a completed flush on this connection.
  // Host-pinned: proxy writes after CQ drain, GPU reads in waitFlush().
  uint64_t* flushDonePos_;

  // Backend that services this channel's device operations. Defaults to the CPU
  // proxy so existing (FIFO-based) construction paths are unchanged.
  PortChannelBackend backend_ = PortChannelBackend::Proxy;

  // GPU-initiated networking context; only used when backend_ == GpuNetIo.
  GpuNetIoDeviceContext* gin_ = nullptr;

  /// Peer rank in gin_'s QP table; independent of proxy memory/semaphore IDs.
  int ginPeer_ = -1;

  /// Registered remote inbound counter offset in the peer's symmetric buffer.
  uint64_t ginSignalOffset_ = UINT64_MAX;

  MSCCLPP_INLINE BasePortChannelDeviceHandle() = default;

  MSCCLPP_HOST_DEVICE_INLINE BasePortChannelDeviceHandle(SemaphoreId semaphoreId,
                                                         Host2DeviceSemaphoreDeviceHandle semaphore,
                                                         FifoDeviceHandle fifo, uint64_t* flushDonePos)
      : semaphoreId_(semaphoreId), semaphore_(semaphore), fifo_(fifo), flushDonePos_(flushDonePos) {}

  /// Construct a GPUNetIO channel using symmetric-buffer-relative data offsets.
  /// @param gin Device networking context, valid for the channel lifetime.
  /// @param peer Peer rank, not a proxy memory or semaphore ID.
  /// @param remoteSignalOffset Aligned offset of the peer's registered uint64_t inbound counter.
  /// @param inboundSignal Local registered uint64_t counter written by this peer.
  /// @param expectedSignal Local device counter tracking consumed signals; initially zero.
  /// Both inbound counters must start at zero and be disjoint from payload storage.
  /// MemoryId arguments are ignored for this backend; all data offsets are relative to gin's buffers.
  MSCCLPP_HOST_DEVICE_INLINE BasePortChannelDeviceHandle(GpuNetIoDeviceContext* gin, int peer,
                                                         uint64_t remoteSignalOffset, uint64_t* inboundSignal,
                                                         uint64_t* expectedSignal)
      : semaphoreId_(0),
        semaphore_{inboundSignal, expectedSignal},
        fifo_{},
        flushDonePos_(nullptr),
        backend_(PortChannelBackend::GpuNetIo),
        gin_(gin),
        ginPeer_(peer),
        ginSignalOffset_(remoteSignalOffset) {}

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Validate the explicitly configured GPUNetIO peer and signal resources.
  MSCCLPP_DEVICE_INLINE void validateGpuNetIo() const {
    MSCCLPP_ASSERT_DEVICE(gin_ != nullptr, "GPUNetIO channel requires a context");
    MSCCLPP_ASSERT_DEVICE(ginPeer_ >= 0 && ginPeer_ < gin_->numPeers, "GPUNetIO channel requires a valid peer rank");
    MSCCLPP_ASSERT_DEVICE(ginSignalOffset_ != UINT64_MAX && ginSignalOffset_ % sizeof(uint64_t) == 0,
                          "GPUNetIO channel requires an aligned registered signal offset");
    MSCCLPP_ASSERT_DEVICE(semaphore_.inboundToken != nullptr && semaphore_.expectedInboundToken != nullptr,
                          "GPUNetIO channel requires receive signal counters");
  }

  /// Push a TriggerData to the FIFO.
  /// @param dstId The ID of destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcId The ID of source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void put(MemoryId dstId, uint64_t dstOffset, MemoryId srcId, uint64_t srcOffset,
                                 uint64_t size) {
    if (backend_ == PortChannelBackend::GpuNetIo) {
      validateGpuNetIo();
      gin_->put(ginPeer_, dstOffset, srcOffset, size);
      return;
    }
    fifo_.push({TriggerData, dstId, dstOffset, srcId, srcOffset, size, semaphoreId_});
  }

  /// Push a TriggerData to the FIFO.
  /// @param dstId The ID of destination memory region.
  /// @param srcId The ID of source memory region.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void put(MemoryId dstId, MemoryId srcId, uint64_t offset, uint64_t size) {
    put(dstId, offset, srcId, offset, size);
  }

  /// Push a TriggerFlag to the FIFO.
  MSCCLPP_DEVICE_INLINE void signal() {
    if (backend_ == PortChannelBackend::GpuNetIo) {
      validateGpuNetIo();
      gin_->atomicAdd(ginPeer_, ginSignalOffset_, 1);
      return;
    }
    fifo_.push({TriggerFlag, 0, 0, 0, 0, 0, semaphoreId_});
  }

  /// Push a TriggerData and a TriggerFlag at the same time to the FIFO.
  /// @param dstId The ID of destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcId The ID of source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void putWithSignal(MemoryId dstId, uint64_t dstOffset, MemoryId srcId, uint64_t srcOffset,
                                           uint64_t size) {
    if (backend_ == PortChannelBackend::GpuNetIo) {
      validateGpuNetIo();
      gin_->putWithSignal(ginPeer_, dstOffset, srcOffset, size, ginSignalOffset_, 1);
      return;
    }
    fifo_.push({TriggerData | TriggerFlag, dstId, dstOffset, srcId, srcOffset, size, semaphoreId_});
  }

  /// Push a TriggerData and a TriggerFlag at the same time to the FIFO.
  /// @param dstId The ID of destination memory region.
  /// @param srcId The ID of source memory region.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void putWithSignal(MemoryId dstId, MemoryId srcId, uint64_t offset, uint64_t size) {
    putWithSignal(dstId, offset, srcId, offset, size);
  }

  /// Push a TriggerData, a TriggerFlag, and a TriggerSync at the same time to the FIFO.
  /// @param dstId The ID of destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcId The ID of source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(MemoryId dstId, uint64_t dstOffset, MemoryId srcId,
                                                   uint64_t srcOffset, uint64_t size, int64_t maxSpinCount = 1000000) {
    if (backend_ == PortChannelBackend::GpuNetIo) {
      putWithSignal(dstId, dstOffset, srcId, srcOffset, size);
      flush(maxSpinCount);
      return;
    }
    uint64_t pos =
        fifo_.push({TriggerData | TriggerFlag | TriggerSync, dstId, dstOffset, srcId, srcOffset, size, semaphoreId_});
    detail::waitFlush(flushDonePos_, pos, maxSpinCount);
  }

  /// Push a TriggerData, a TriggerFlag, and a TriggerSync at the same time to the FIFO.
  /// @param dstId The ID of destination memory region.
  /// @param srcId The ID of source memory region.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(MemoryId dstId, MemoryId srcId, uint64_t offset, uint64_t size,
                                                   int64_t maxSpinCount = 1000000) {
    putWithSignalAndFlush(dstId, offset, srcId, offset, size, maxSpinCount);
  }

  /// Push a TriggerSync to the FIFO.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  MSCCLPP_DEVICE_INLINE void flush(int64_t maxSpinCount = 1000000) {
    if (backend_ == PortChannelBackend::GpuNetIo) {
      validateGpuNetIo();
      if (maxSpinCount < 0) {
        gin_->flush(ginPeer_);
      } else {
        const int status = gin_->tryFlush(ginPeer_, static_cast<uint64_t>(maxSpinCount));
        MSCCLPP_ASSERT_DEVICE(status == 0, "GPUNetIO flush timed out or reported a CQ error");
      }
      return;
    }
    uint64_t pos = fifo_.push({TriggerSync, 0, 0, 0, 0, 0, semaphoreId_});
    detail::waitFlush(flushDonePos_, pos, maxSpinCount);
  }

  /// Push an atomic add trigger to the FIFO to perform a remote atomic add on a 64-bit value.
  /// Uses type == 0 to indicate an atomic add operation.
  /// @param dstId The ID of destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param value The 64-bit signed value to atomically add.
  MSCCLPP_DEVICE_INLINE void atomicAdd(MemoryId dstId, uint64_t dstOffset, int64_t value) {
    if (backend_ == PortChannelBackend::GpuNetIo) {
      validateGpuNetIo();
      gin_->atomicAdd(ginPeer_, dstOffset, value);
      return;
    }
    ProxyTrigger trigger;
    // Encode the full 64-bit add value in fst (size + srcOffset fields).
    trigger.fst = static_cast<uint64_t>(value);
    // Build snd with dstOffset, dstMemoryId, type=0 (atomic add), semaphoreId.
    trigger.snd = 0;
    trigger.fields.dstOffset = dstOffset;
    trigger.fields.dstMemoryId = dstId;
    trigger.fields.type = 0;
    trigger.fields.semaphoreId = semaphoreId_;
    fifo_.push(trigger);
  }

  /// Check if the port channel has been signaled.
  /// @return true if the port channel has been signaled.
  MSCCLPP_DEVICE_INLINE bool poll() { return semaphore_.poll(); }

  /// Wait for the port channel to be signaled.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  MSCCLPP_DEVICE_INLINE void wait(int64_t maxSpinCount = 10000000) { semaphore_.wait(maxSpinCount); }

#endif  // defined(MSCCLPP_DEVICE_COMPILE)
};

struct PortChannelDeviceHandle : public BasePortChannelDeviceHandle {
  MemoryId dst_;
  MemoryId src_;

  MSCCLPP_INLINE PortChannelDeviceHandle() = default;

  MSCCLPP_HOST_DEVICE_INLINE PortChannelDeviceHandle(SemaphoreId semaphoreId,
                                                     Host2DeviceSemaphoreDeviceHandle semaphore, FifoDeviceHandle fifo,
                                                     MemoryId dst, MemoryId src, uint64_t* flushDonePos)
      : BasePortChannelDeviceHandle(semaphoreId, semaphore, fifo, flushDonePos), dst_(dst), src_(src) {}

  /// Construct a symmetric-buffer GPUNetIO channel; signal arguments match BasePortChannelDeviceHandle.
  MSCCLPP_HOST_DEVICE_INLINE PortChannelDeviceHandle(GpuNetIoDeviceContext* gin, int peer, uint64_t remoteSignalOffset,
                                                     uint64_t* inboundSignal, uint64_t* expectedSignal)
      : BasePortChannelDeviceHandle(gin, peer, remoteSignalOffset, inboundSignal, expectedSignal), dst_(0), src_(0) {}

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Push a TriggerData to the FIFO.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void put(uint64_t dstOffset, uint64_t srcOffset, uint64_t size) {
    BasePortChannelDeviceHandle::put(dst_, dstOffset, src_, srcOffset, size);
  }

  /// Push a TriggerData to the FIFO.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void put(uint64_t offset, uint64_t size) { put(offset, offset, size); }

  /// Push a TriggerData and a TriggerFlag at the same time to the FIFO.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void putWithSignal(uint64_t dstOffset, uint64_t srcOffset, uint64_t size) {
    BasePortChannelDeviceHandle::putWithSignal(dst_, dstOffset, src_, srcOffset, size);
  }

  /// Push a TriggerData and a TriggerFlag at the same time to the FIFO.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void putWithSignal(uint64_t offset, uint64_t size) { putWithSignal(offset, offset, size); }

  /// Push a TriggerData, a TriggerFlag, and a TriggerSync at the same time to the FIFO.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(uint64_t dstOffset, uint64_t srcOffset, uint64_t size,
                                                   int64_t maxSpinCount = 1000000) {
    BasePortChannelDeviceHandle::putWithSignalAndFlush(dst_, dstOffset, src_, srcOffset, size, maxSpinCount);
  }

  /// Push a TriggerData, a TriggerFlag, and a TriggerSync at the same time to the FIFO.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(uint64_t offset, uint64_t size) {
    putWithSignalAndFlush(offset, offset, size);
  }

  /// Push an atomic add trigger to the FIFO to perform a remote atomic add on a 64-bit value.
  /// @param dstOffset The offset into the destination memory region.
  /// @param value The 64-bit signed value to atomically add.
  MSCCLPP_DEVICE_INLINE void atomicAdd(uint64_t dstOffset, int64_t value) {
    BasePortChannelDeviceHandle::atomicAdd(dst_, dstOffset, value);
  }
#endif  // defined(MSCCLPP_DEVICE_COMPILE)
};

}  // namespace mscclpp

#endif  // MSCCLPP_PORT_CHANNEL_DEVICE_HPP_
