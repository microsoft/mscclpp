// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

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
/// @tparam SmemSlots Number of internal 64-byte shared-memory slots per CTA (0 disables BlueFlame, default 32).
template <PortChannelBackend Backend = PortChannelBackend::Host,
          uint32_t SmemSlots = DeviceBackendHandle::kDefaultSmemSlots>
struct BasePortChannelDeviceHandleT;

/// Host-backend specialization of BasePortChannelDeviceHandleT.
/// Posting and flush methods only support Concurrency::Device, enforced at compile time.
template <>
struct BasePortChannelDeviceHandleT<PortChannelBackend::Host> {
  /// Scope of concurrent callers posting and flushing through a PortChannel.
  enum class Concurrency : uint8_t {
    /// Only one thread accesses the queue pair at a time; the caller serializes access.
    Exclusive = 0,
    /// Concurrent callers are confined to one thread block.
    Block = 1,
    /// Concurrent callers may run across the device.
    Device = 2,
  };

  SemaphoreId semaphoreId_;

  Host2DeviceSemaphoreDeviceHandle semaphore_;

  // this is a concurrent fifo which is multiple threads from the device
  // can produce for and the sole proxy thread consumes it.
  FifoDeviceHandle fifo_;

  // One past the highest FIFO position with a completed flush on this connection.
  // Host-pinned: proxy writes after CQ drain, GPU reads in waitFlush().
  uint64_t* flushDonePos_;

  MSCCLPP_INLINE BasePortChannelDeviceHandleT() = default;

  MSCCLPP_HOST_DEVICE_INLINE BasePortChannelDeviceHandleT(SemaphoreId semaphoreId,
                                                          Host2DeviceSemaphoreDeviceHandle semaphore,
                                                          FifoDeviceHandle fifo, uint64_t* flushDonePos)
      : semaphoreId_(semaphoreId), semaphore_(semaphore), fifo_(fifo), flushDonePos_(flushDonePos) {}

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Push a TriggerPut to the FIFO.
  /// @param dstId The ID of destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcId The ID of source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void put(MemoryId dstId, uint64_t dstOffset, MemoryId srcId, uint64_t srcOffset,
                                 uint64_t size) {
    static_assert(Mode == Concurrency::Device, "Host-backed PortChannel only supports Concurrency::Device");
    fifo_.push({TriggerPut, dstId, dstOffset, srcId, srcOffset, size, semaphoreId_});
  }

  /// Push a TriggerPut to the FIFO.
  /// @param dstId The ID of destination memory region.
  /// @param srcId The ID of source memory region.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void put(MemoryId dstId, MemoryId srcId, uint64_t offset, uint64_t size) {
    put<Mode>(dstId, offset, srcId, offset, size);
  }

  /// Push a TriggerSignal to the FIFO.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void signal() {
    static_assert(Mode == Concurrency::Device, "Host-backed PortChannel only supports Concurrency::Device");
    fifo_.push({TriggerSignal, 0, 0, 0, 0, 0, semaphoreId_});
  }

  /// Push a TriggerPutWithSignal to the FIFO.
  /// @param dstId The ID of destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcId The ID of source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void putWithSignal(MemoryId dstId, uint64_t dstOffset, MemoryId srcId, uint64_t srcOffset,
                                           uint64_t size) {
    static_assert(Mode == Concurrency::Device, "Host-backed PortChannel only supports Concurrency::Device");
    fifo_.push({TriggerPutWithSignal, dstId, dstOffset, srcId, srcOffset, size, semaphoreId_});
  }

  /// Push a TriggerPutWithSignal to the FIFO.
  /// @param dstId The ID of destination memory region.
  /// @param srcId The ID of source memory region.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void putWithSignal(MemoryId dstId, MemoryId srcId, uint64_t offset, uint64_t size) {
    putWithSignal<Mode>(dstId, offset, srcId, offset, size);
  }

  /// Push a TriggerPutWithSignalAndFlush to the FIFO.
  /// @param dstId The ID of destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcId The ID of source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(MemoryId dstId, uint64_t dstOffset, MemoryId srcId,
                                                   uint64_t srcOffset, uint64_t size, int64_t maxSpinCount = 1000000) {
    static_assert(Mode == Concurrency::Device, "Host-backed PortChannel only supports Concurrency::Device");
    uint64_t pos = fifo_.push({TriggerPutWithSignalAndFlush, dstId, dstOffset, srcId, srcOffset, size, semaphoreId_});
    detail::waitFlush(flushDonePos_, pos, maxSpinCount);
  }

  /// Push a TriggerPutWithSignalAndFlush to the FIFO.
  /// @param dstId The ID of destination memory region.
  /// @param srcId The ID of source memory region.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(MemoryId dstId, MemoryId srcId, uint64_t offset, uint64_t size,
                                                   int64_t maxSpinCount = 1000000) {
    putWithSignalAndFlush<Mode>(dstId, offset, srcId, offset, size, maxSpinCount);
  }

  /// Push a TriggerFlush to the FIFO.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void flush(int64_t maxSpinCount = 1000000) {
    static_assert(Mode == Concurrency::Device, "Host-backed PortChannel only supports Concurrency::Device");
    uint64_t pos = fifo_.push({TriggerFlush, 0, 0, 0, 0, 0, semaphoreId_});
    detail::waitFlush(flushDonePos_, pos, maxSpinCount);
  }

  /// Push an accumulate trigger to the FIFO: add a 64-bit value to remote memory.
  /// Connection::accumulate() documents how many concurrent writers each transport allows.
  /// @param dstId The ID of destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param value The 64-bit signed value to add.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void accumulate(MemoryId dstId, uint64_t dstOffset, int64_t value) {
    static_assert(Mode == Concurrency::Device, "Host-backed PortChannel only supports Concurrency::Device");
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

/// Device-backend specialization of BasePortChannelDeviceHandleT.
template <uint32_t SmemSlots>
struct BasePortChannelDeviceHandleT<PortChannelBackend::Device, SmemSlots> {
  static_assert(SmemSlots <= 1024, "At most 1024 shared-memory slots per CTA are supported");

  /// Scope of concurrent callers posting and flushing through a PortChannel.
  using Concurrency = BasePortChannelDeviceHandleT<PortChannelBackend::Host>::Concurrency;

  PortDevice2DeviceSemaphoreDeviceHandle semaphore_;

  DeviceBackendHandle device_;

  MSCCLPP_INLINE BasePortChannelDeviceHandleT() = default;

  MSCCLPP_HOST_DEVICE_INLINE BasePortChannelDeviceHandleT(PortDevice2DeviceSemaphoreDeviceHandle semaphore,
                                                          DeviceBackendHandle device)
      : semaphore_(semaphore), device_(device) {}

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Put data to the destination memory region via the device backend.
  /// @param dst The destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param src The source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  /// @param slotIdx Shared-memory slot index, or -1 for the linear thread index. Out-of-range indices use doorbells.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void put(const DeviceMemoryHandle& dst, uint64_t dstOffset, const DeviceMemoryHandle& src,
                                 uint64_t srcOffset, uint64_t size, int slotIdx = -1) {
    device_.put<Mode, SmemSlots>(dst, dstOffset, src, srcOffset, size, slotIdx);
  }

  /// Put data to the destination memory region via the device backend.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  /// @param slotIdx Shared-memory slot index, or -1 for the linear thread index. Out-of-range indices use doorbells.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void put(const DeviceMemoryHandle& dst, const DeviceMemoryHandle& src, uint64_t offset,
                                 uint64_t size, int slotIdx = -1) {
    put<Mode>(dst, offset, src, offset, size, slotIdx);
  }

  /// Signal the remote peer via the device backend.
  /// @param slotIdx Shared-memory slot index, or -1 for the linear thread index. Out-of-range indices use doorbells.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void signal(int slotIdx = -1) {
    device_.accumulate<Mode, SmemSlots>(semaphore_.remoteInboundToken, 0, 1, slotIdx);
  }

  /// Put data to the destination memory region and signal the remote peer via the device backend.
  /// @param dst The destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param src The source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  /// @param slotIdx Shared-memory slot index, or -1 for the linear thread index. Out-of-range indices use doorbells.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void putWithSignal(const DeviceMemoryHandle& dst, uint64_t dstOffset,
                                           const DeviceMemoryHandle& src, uint64_t srcOffset, uint64_t size,
                                           int slotIdx = -1) {
    device_.putWithAccumulate<Mode, SmemSlots>(dst, dstOffset, src, srcOffset, size, semaphore_.remoteInboundToken, 0,
                                               1, slotIdx);
  }

  /// Put data to the destination memory region and signal the remote peer via the device backend.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  /// @param slotIdx Shared-memory slot index, or -1 for the linear thread index. Out-of-range indices use doorbells.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void putWithSignal(const DeviceMemoryHandle& dst, const DeviceMemoryHandle& src,
                                           uint64_t offset, uint64_t size, int slotIdx = -1) {
    putWithSignal<Mode>(dst, offset, src, offset, size, slotIdx);
  }

  /// Put data to the destination memory region, signal the remote peer, and flush via the device backend.
  /// @param dst The destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param src The source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  /// @param slotIdx Shared-memory slot index, or -1 for the linear thread index. Out-of-range indices use doorbells.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(const DeviceMemoryHandle& dst, uint64_t dstOffset,
                                                   const DeviceMemoryHandle& src, uint64_t srcOffset, uint64_t size,
                                                   int64_t maxSpinCount = 1000000, int slotIdx = -1) {
    putWithSignal<Mode>(dst, dstOffset, src, srcOffset, size, slotIdx);
    flush<Mode>(maxSpinCount);
  }

  /// Put data to the destination memory region, signal the remote peer, and flush via the device backend.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  /// @param slotIdx Shared-memory slot index, or -1 for the linear thread index. Out-of-range indices use doorbells.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(const DeviceMemoryHandle& dst, const DeviceMemoryHandle& src,
                                                   uint64_t offset, uint64_t size, int64_t maxSpinCount = 1000000,
                                                   int slotIdx = -1) {
    putWithSignalAndFlush<Mode>(dst, offset, src, offset, size, maxSpinCount, slotIdx);
  }

  /// Flush pending operations on the channel via the device backend.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void flush(int64_t maxSpinCount = 1000000) {
    device_.flush<Mode>(maxSpinCount);
  }

  /// Atomically add a 64-bit value to remote memory via the device backend. Observing the new value publishes the
  /// earlier puts on this channel.
  /// @param dst The destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param value The 64-bit signed value to add.
  /// @param slotIdx Shared-memory slot index, or -1 for the linear thread index. Out-of-range indices use doorbells.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void accumulate(const DeviceMemoryHandle& dst, uint64_t dstOffset, int64_t value,
                                        int slotIdx = -1) {
    device_.accumulate<Mode, SmemSlots>(dst, dstOffset, value, slotIdx);
  }

  /// Put data to the destination memory region, then atomically add a 64-bit value to remote memory. Observing the
  /// new value publishes the put.
  /// @param dst The destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param src The source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  /// @param acc The memory region of the accumulated word.
  /// @param accOffset The offset of the accumulated word in `acc`.
  /// @param value The 64-bit signed value to add.
  /// @param slotIdx Shared-memory slot index, or -1 for the linear thread index. Out-of-range indices use doorbells.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void putWithAccumulate(const DeviceMemoryHandle& dst, uint64_t dstOffset,
                                               const DeviceMemoryHandle& src, uint64_t srcOffset, uint64_t size,
                                               const DeviceMemoryHandle& acc, uint64_t accOffset, int64_t value,
                                               int slotIdx = -1) {
    device_.putWithAccumulate<Mode, SmemSlots>(dst, dstOffset, src, srcOffset, size, acc, accOffset, value, slotIdx);
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
/// @tparam SmemSlots Number of internal 64-byte shared-memory slots per CTA (0 disables BlueFlame, default 32).
template <PortChannelBackend Backend = PortChannelBackend::Host,
          uint32_t SmemSlots = DeviceBackendHandle::kDefaultSmemSlots>
struct PortChannelDeviceHandleT;

/// Host-backend specialization of PortChannelDeviceHandleT.
/// Posting and flush methods only support Concurrency::Device, enforced at compile time.
template <>
struct PortChannelDeviceHandleT<PortChannelBackend::Host>
    : public BasePortChannelDeviceHandleT<PortChannelBackend::Host> {
  MemoryId dst_;
  MemoryId src_;

  MSCCLPP_INLINE PortChannelDeviceHandleT() = default;

  MSCCLPP_HOST_DEVICE_INLINE PortChannelDeviceHandleT(SemaphoreId semaphoreId,
                                                      Host2DeviceSemaphoreDeviceHandle semaphore, FifoDeviceHandle fifo,
                                                      MemoryId dst, MemoryId src, uint64_t* flushDonePos)
      : BasePortChannelDeviceHandleT<PortChannelBackend::Host>(semaphoreId, semaphore, fifo, flushDonePos),
        dst_(dst),
        src_(src) {}

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Push a TriggerPut to the FIFO.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void put(uint64_t dstOffset, uint64_t srcOffset, uint64_t size) {
    BasePortChannelDeviceHandleT<PortChannelBackend::Host>::put<Mode>(dst_, dstOffset, src_, srcOffset, size);
  }

  /// Push a TriggerPut to the FIFO.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void put(uint64_t offset, uint64_t size) {
    put<Mode>(offset, offset, size);
  }

  /// Push a TriggerPutWithSignal to the FIFO.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void putWithSignal(uint64_t dstOffset, uint64_t srcOffset, uint64_t size) {
    BasePortChannelDeviceHandleT<PortChannelBackend::Host>::putWithSignal<Mode>(dst_, dstOffset, src_, srcOffset, size);
  }

  /// Push a TriggerPutWithSignal to the FIFO.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void putWithSignal(uint64_t offset, uint64_t size) {
    putWithSignal<Mode>(offset, offset, size);
  }

  /// Push a TriggerPutWithSignalAndFlush to the FIFO.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(uint64_t dstOffset, uint64_t srcOffset, uint64_t size,
                                                   int64_t maxSpinCount = 1000000) {
    BasePortChannelDeviceHandleT<PortChannelBackend::Host>::putWithSignalAndFlush<Mode>(dst_, dstOffset, src_,
                                                                                        srcOffset, size, maxSpinCount);
  }

  /// Push a TriggerPutWithSignalAndFlush to the FIFO.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(uint64_t offset, uint64_t size) {
    putWithSignalAndFlush<Mode>(offset, offset, size);
  }

  /// Push an accumulate trigger to the FIFO: add a 64-bit value to the destination memory.
  /// See Connection::accumulate() for transport support.
  /// @param dstOffset The offset into the destination memory region.
  /// @param value The 64-bit signed value to add.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void accumulate(uint64_t dstOffset, int64_t value) {
    BasePortChannelDeviceHandleT<PortChannelBackend::Host>::accumulate<Mode>(dst_, dstOffset, value);
  }
#endif  // defined(MSCCLPP_DEVICE_COMPILE)
};

/// Device-backend specialization of PortChannelDeviceHandleT.
template <uint32_t SmemSlots>
struct PortChannelDeviceHandleT<PortChannelBackend::Device, SmemSlots>
    : public BasePortChannelDeviceHandleT<PortChannelBackend::Device, SmemSlots> {
  /// Scope of concurrent callers accessing this channel's queue pair.
  using Concurrency = typename BasePortChannelDeviceHandleT<PortChannelBackend::Device, SmemSlots>::Concurrency;

  DeviceMemoryHandle dstMemory_;
  DeviceMemoryHandle srcMemory_;

  MSCCLPP_INLINE PortChannelDeviceHandleT() = default;

  MSCCLPP_HOST_DEVICE_INLINE PortChannelDeviceHandleT(PortDevice2DeviceSemaphoreDeviceHandle semaphore,
                                                      DeviceBackendHandle device, DeviceMemoryHandle dstMemory,
                                                      DeviceMemoryHandle srcMemory)
      : BasePortChannelDeviceHandleT<PortChannelBackend::Device, SmemSlots>(semaphore, device),
        dstMemory_(dstMemory),
        srcMemory_(srcMemory) {}

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Put data to the destination memory region via the device backend.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  /// @param slotIdx Shared-memory slot index, or -1 for the linear thread index. Out-of-range indices use doorbells.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void put(uint64_t dstOffset, uint64_t srcOffset, uint64_t size, int slotIdx = -1) {
    BasePortChannelDeviceHandleT<PortChannelBackend::Device, SmemSlots>::template put<Mode>(
        dstMemory_, dstOffset, srcMemory_, srcOffset, size, slotIdx);
  }

  /// Put data to the destination memory region via the device backend.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void put(uint64_t offset, uint64_t size) {
    put<Mode>(offset, offset, size);
  }

  /// Put data to the destination memory region and signal the remote peer via the device backend.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  /// @param slotIdx Shared-memory slot index, or -1 for the linear thread index. Out-of-range indices use doorbells.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void putWithSignal(uint64_t dstOffset, uint64_t srcOffset, uint64_t size, int slotIdx = -1) {
    BasePortChannelDeviceHandleT<PortChannelBackend::Device, SmemSlots>::template putWithSignal<Mode>(
        dstMemory_, dstOffset, srcMemory_, srcOffset, size, slotIdx);
  }

  /// Put data to the destination memory region and signal the remote peer via the device backend.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void putWithSignal(uint64_t offset, uint64_t size) {
    putWithSignal<Mode>(offset, offset, size);
  }

  /// Put data to the destination memory region, signal the remote peer, and flush via the device backend.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  /// @param slotIdx Shared-memory slot index, or -1 for the linear thread index. Out-of-range indices use doorbells.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(uint64_t dstOffset, uint64_t srcOffset, uint64_t size,
                                                   int64_t maxSpinCount = 1000000, int slotIdx = -1) {
    BasePortChannelDeviceHandleT<PortChannelBackend::Device, SmemSlots>::template putWithSignalAndFlush<Mode>(
        dstMemory_, dstOffset, srcMemory_, srcOffset, size, maxSpinCount, slotIdx);
  }

  /// Put data to the destination memory region, signal the remote peer, and flush via the device backend.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(uint64_t offset, uint64_t size) {
    putWithSignalAndFlush<Mode>(offset, offset, size);
  }

  /// Atomically add a 64-bit value to the destination memory via the device backend.
  /// @param dstOffset The offset into the destination memory region.
  /// @param value The 64-bit signed value to add.
  /// @param slotIdx Shared-memory slot index, or -1 for the linear thread index. Out-of-range indices use doorbells.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void accumulate(uint64_t dstOffset, int64_t value, int slotIdx = -1) {
    BasePortChannelDeviceHandleT<PortChannelBackend::Device, SmemSlots>::template accumulate<Mode>(
        dstMemory_, dstOffset, value, slotIdx);
  }

  /// Put data to the destination memory region, then atomically add a 64-bit value to the destination memory.
  /// Observing the new value publishes the put.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  /// @param accOffset The offset of the accumulated word in the destination memory region.
  /// @param value The 64-bit signed value to add.
  /// @param slotIdx Shared-memory slot index, or -1 for the linear thread index. Out-of-range indices use doorbells.
  template <Concurrency Mode = Concurrency::Device>
  MSCCLPP_DEVICE_INLINE void putWithAccumulate(uint64_t dstOffset, uint64_t srcOffset, uint64_t size,
                                               uint64_t accOffset, int64_t value, int slotIdx = -1) {
    BasePortChannelDeviceHandleT<PortChannelBackend::Device, SmemSlots>::template putWithAccumulate<Mode>(
        dstMemory_, dstOffset, srcMemory_, srcOffset, size, dstMemory_, accOffset, value, slotIdx);
  }
#endif  // defined(MSCCLPP_DEVICE_COMPILE)
};

/// Default Host-backend PortChannelDeviceHandle alias for backward compatibility without template brackets.
using PortChannelDeviceHandle = PortChannelDeviceHandleT<PortChannelBackend::Host>;

}  // namespace mscclpp

#endif  // MSCCLPP_PORT_CHANNEL_DEVICE_HPP_
