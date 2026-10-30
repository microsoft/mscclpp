// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_PORT_CHANNEL_DEVICE_HPP_
#define MSCCLPP_PORT_CHANNEL_DEVICE_HPP_

#include "fifo_device.hpp"
#include "gpunetio_device.hpp"
#include "semaphore_device.hpp"

namespace mscclpp {

/// Backend that services a PortChannel's device-side operations.
enum class PortChannelBackend : uint8_t {
  Host = 0,    ///< The device pushes triggers to a FIFO that a host proxy thread consumes.
  Device = 1,  ///< The device posts RDMA work requests to the connection's queue pair directly.
};

/// Numeric ID of Semaphore. ProxyService has an internal array indexed by these handles mapping to the
/// actual semaphores.
using SemaphoreId = uint32_t;

/// Numeric ID of RegisteredMemory. ProxyService has an internal array indexed by these handles mapping to the
/// actual.
using MemoryId = uint32_t;

namespace detail {
#if defined(MSCCLPP_DEVICE_COMPILE)
/// Wait until the proxy has processed and drained the TriggerFlush at FIFO position `fifoPos`.
/// The proxy publishes `flushDonePos = latestCompletedPos + 1` when the CQ drains, so the
/// wait condition `flushDonePos > fifoPos` is satisfied exactly when our own request has
/// been completed. Using the FIFO push position as the wait target couples the wait to the
/// FIFO order, avoiding races when multiple GPU threads concurrently flush the same channel.
MSCCLPP_DEVICE_INLINE void waitFlush(uint64_t* flushDonePos, uint64_t fifoPos, [[maybe_unused]] int64_t maxSpinCount) {
  POLL_MAYBE_JAILBREAK((atomicLoad<uint64_t, scopeSystem>(flushDonePos, memoryOrderAcquire) <= fifoPos), maxSpinCount);
}
#endif  // defined(MSCCLPP_DEVICE_COMPILE)
}  // namespace detail

/// Device-side handle template for BasePortChannel.
/// A non-templated alias `BasePortChannelDeviceHandle` is provided for the default `PortChannelBackend::Host` backend
/// so existing code can use `BasePortChannelDeviceHandle` without `<>`.
template <PortChannelBackend Backend = PortChannelBackend::Host>
struct BasePortChannelDeviceHandleT {
  SemaphoreId semaphoreId_;

  Host2DeviceSemaphoreDeviceHandle semaphore_;

  // this is a concurrent fifo which is multiple threads from the device
  // can produce for and the sole proxy thread consumes it.
  FifoDeviceHandle fifo_;

  // One past the highest FIFO position with a completed flush on this connection.
  // Host-pinned: proxy writes after CQ drain, GPU reads in waitFlush().
  uint64_t* flushDonePos_;

  // State of the device-initiated backend; only used when Backend == PortChannelBackend::Device.
  DeviceBackendHandle device_;

  MSCCLPP_INLINE BasePortChannelDeviceHandleT() = default;

  MSCCLPP_HOST_DEVICE_INLINE BasePortChannelDeviceHandleT(SemaphoreId semaphoreId,
                                                          Host2DeviceSemaphoreDeviceHandle semaphore,
                                                          FifoDeviceHandle fifo, uint64_t* flushDonePos)
      : semaphoreId_(semaphoreId), semaphore_(semaphore), fifo_(fifo), flushDonePos_(flushDonePos), device_{} {}

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Push a TriggerPut to the FIFO.
  /// @param dstId The ID of destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcId The ID of source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void put(MemoryId dstId, uint64_t dstOffset, MemoryId srcId, uint64_t srcOffset,
                                 uint64_t size) {
    fifo_.push({TriggerPut, dstId, dstOffset, srcId, srcOffset, size, semaphoreId_});
  }

  /// Push a TriggerPut to the FIFO.
  /// @param dstId The ID of destination memory region.
  /// @param srcId The ID of source memory region.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void put(MemoryId dstId, MemoryId srcId, uint64_t offset, uint64_t size) {
    put(dstId, offset, srcId, offset, size);
  }

  /// Signal the remote peer.
  template <PortChannelSharingMode Mode = PortChannelSharingMode::Gpu>
  MSCCLPP_DEVICE_INLINE void signal() {
    if constexpr (Backend == PortChannelBackend::Device) {
      device_.signal<Mode>();
    } else {
      fifo_.push({TriggerSignal, 0, 0, 0, 0, 0, semaphoreId_});
    }
  }

  /// Push a TriggerPutWithSignal to the FIFO.
  /// @param dstId The ID of destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcId The ID of source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void putWithSignal(MemoryId dstId, uint64_t dstOffset, MemoryId srcId, uint64_t srcOffset,
                                           uint64_t size) {
    fifo_.push({TriggerPutWithSignal, dstId, dstOffset, srcId, srcOffset, size, semaphoreId_});
  }

  /// Push a TriggerPutWithSignal to the FIFO.
  /// @param dstId The ID of destination memory region.
  /// @param srcId The ID of source memory region.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void putWithSignal(MemoryId dstId, MemoryId srcId, uint64_t offset, uint64_t size) {
    putWithSignal(dstId, offset, srcId, offset, size);
  }

  /// Push a TriggerPutWithSignalAndFlush to the FIFO.
  /// @param dstId The ID of destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcId The ID of source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(MemoryId dstId, uint64_t dstOffset, MemoryId srcId,
                                                   uint64_t srcOffset, uint64_t size, int64_t maxSpinCount = 1000000) {
    uint64_t pos = fifo_.push({TriggerPutWithSignalAndFlush, dstId, dstOffset, srcId, srcOffset, size, semaphoreId_});
    detail::waitFlush(flushDonePos_, pos, maxSpinCount);
  }

  /// Push a TriggerPutWithSignalAndFlush to the FIFO.
  /// @param dstId The ID of destination memory region.
  /// @param srcId The ID of source memory region.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(MemoryId dstId, MemoryId srcId, uint64_t offset, uint64_t size,
                                                   int64_t maxSpinCount = 1000000) {
    putWithSignalAndFlush(dstId, offset, srcId, offset, size, maxSpinCount);
  }

  /// Flush pending operations on the channel.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  template <PortChannelSharingMode Mode = PortChannelSharingMode::Gpu>
  MSCCLPP_DEVICE_INLINE void flush(int64_t maxSpinCount = 1000000) {
    if constexpr (Backend == PortChannelBackend::Device) {
      device_.flush<Mode>(maxSpinCount);
    } else {
      uint64_t pos = fifo_.push({TriggerFlush, 0, 0, 0, 0, 0, semaphoreId_});
      detail::waitFlush(flushDonePos_, pos, maxSpinCount);
    }
  }

  /// Push an accumulate trigger to the FIFO: add a 64-bit value to remote memory.
  /// Connection::accumulate() documents how many concurrent writers each transport allows.
  /// @param dstId The ID of destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param value The 64-bit signed value to add.
  MSCCLPP_DEVICE_INLINE void accumulate(MemoryId dstId, uint64_t dstOffset, int64_t value) {
    // The operand occupies fst, spanning the low size and high srcOffset fields.
    uint64_t operand = static_cast<uint64_t>(value);
    ProxyTrigger trigger(TriggerAccumulate, dstId, dstOffset, /*srcId=*/0, operand >> TriggerBitsSize,
                         static_cast<uint32_t>(operand), semaphoreId_);
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

/// Default Host-backend BasePortChannelDeviceHandle alias for backward compatibility without template brackets.
using BasePortChannelDeviceHandle = BasePortChannelDeviceHandleT<PortChannelBackend::Host>;

/// Device-side handle template for PortChannel.
/// A non-templated alias `PortChannelDeviceHandle` is provided for the default `PortChannelBackend::Host` backend
/// so existing code can use `PortChannelDeviceHandle` without `<>`.
template <PortChannelBackend Backend = PortChannelBackend::Host>
struct PortChannelDeviceHandleT : public BasePortChannelDeviceHandleT<Backend> {
  MemoryId dst_;
  MemoryId src_;
  DeviceMemoryHandle dstMemory_;
  DeviceMemoryHandle srcMemory_;

  MSCCLPP_INLINE PortChannelDeviceHandleT() = default;

  MSCCLPP_HOST_DEVICE_INLINE PortChannelDeviceHandleT(SemaphoreId semaphoreId,
                                                      Host2DeviceSemaphoreDeviceHandle semaphore, FifoDeviceHandle fifo,
                                                      MemoryId dst, MemoryId src, uint64_t* flushDonePos)
      : BasePortChannelDeviceHandleT<Backend>(semaphoreId, semaphore, fifo, flushDonePos),
        dst_(dst),
        src_(src),
        dstMemory_{},
        srcMemory_{} {}

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Put data to the destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  template <PortChannelSharingMode Mode = PortChannelSharingMode::Gpu>
  MSCCLPP_DEVICE_INLINE void put(uint64_t dstOffset, uint64_t srcOffset, uint64_t size) {
    if constexpr (Backend == PortChannelBackend::Device) {
      this->device_.template put<Mode>(dstMemory_, dstOffset, srcMemory_, srcOffset, size);
    } else {
      BasePortChannelDeviceHandleT<Backend>::put(dst_, dstOffset, src_, srcOffset, size);
    }
  }

  /// Put data to the destination memory region.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  template <PortChannelSharingMode Mode = PortChannelSharingMode::Gpu>
  MSCCLPP_DEVICE_INLINE void put(uint64_t offset, uint64_t size) {
    put<Mode>(offset, offset, size);
  }

  /// Put data to the destination memory region and signal the remote peer.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  template <PortChannelSharingMode Mode = PortChannelSharingMode::Gpu>
  MSCCLPP_DEVICE_INLINE void putWithSignal(uint64_t dstOffset, uint64_t srcOffset, uint64_t size) {
    if constexpr (Backend == PortChannelBackend::Device) {
      this->device_.template putWithSignal<Mode>(dstMemory_, dstOffset, srcMemory_, srcOffset, size);
    } else {
      BasePortChannelDeviceHandleT<Backend>::putWithSignal(dst_, dstOffset, src_, srcOffset, size);
    }
  }

  /// Put data to the destination memory region and signal the remote peer.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  template <PortChannelSharingMode Mode = PortChannelSharingMode::Gpu>
  MSCCLPP_DEVICE_INLINE void putWithSignal(uint64_t offset, uint64_t size) {
    putWithSignal<Mode>(offset, offset, size);
  }

  /// Put data to the destination memory region, signal the remote peer, and flush.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  template <PortChannelSharingMode Mode = PortChannelSharingMode::Gpu>
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(uint64_t dstOffset, uint64_t srcOffset, uint64_t size,
                                                   int64_t maxSpinCount = 1000000) {
    if constexpr (Backend == PortChannelBackend::Device) {
      this->device_.template putWithSignalAndFlush<Mode>(dstMemory_, dstOffset, srcMemory_, srcOffset, size,
                                                         maxSpinCount);
    } else {
      BasePortChannelDeviceHandleT<Backend>::putWithSignalAndFlush(dst_, dstOffset, src_, srcOffset, size,
                                                                   maxSpinCount);
    }
  }

  /// Put data to the destination memory region, signal the remote peer, and flush.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  template <PortChannelSharingMode Mode = PortChannelSharingMode::Gpu>
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(uint64_t offset, uint64_t size) {
    putWithSignalAndFlush<Mode>(offset, offset, size);
  }

  /// Add a 64-bit value to the destination memory.
  /// See Connection::accumulate() for transport support.
  /// @param dstOffset The offset into the destination memory region.
  /// @param value The 64-bit signed value to add.
  template <PortChannelSharingMode Mode = PortChannelSharingMode::Gpu>
  MSCCLPP_DEVICE_INLINE void accumulate(uint64_t dstOffset, int64_t value) {
    if constexpr (Backend == PortChannelBackend::Device) {
      this->device_.template accumulate<Mode>(dstMemory_, dstOffset, value);
    } else {
      BasePortChannelDeviceHandleT<Backend>::accumulate(dst_, dstOffset, value);
    }
  }
#endif  // defined(MSCCLPP_DEVICE_COMPILE)
};

/// Default Host-backend PortChannelDeviceHandle alias for backward compatibility without template brackets.
using PortChannelDeviceHandle = PortChannelDeviceHandleT<PortChannelBackend::Host>;

}  // namespace mscclpp

#endif  // MSCCLPP_PORT_CHANNEL_DEVICE_HPP_
