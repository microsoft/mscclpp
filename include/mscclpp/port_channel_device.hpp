// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_PORT_CHANNEL_DEVICE_HPP_
#define MSCCLPP_PORT_CHANNEL_DEVICE_HPP_

#include "fifo_device.hpp"
#include "semaphore_device.hpp"

namespace mscclpp {

/// Numeric ID of Semaphore. ProxyService has an internal array indexed by these handles mapping to the
/// actual semaphores.
using SemaphoreId = uint32_t;

/// Numeric ID of RegisteredMemory. ProxyService has an internal array indexed by these handles mapping to the
/// actual.
using MemoryId = uint32_t;

using TriggerType = uint64_t;
constexpr TriggerType TriggerData = 0x1;  // Trigger a data transfer.
constexpr TriggerType TriggerFlag = 0x2;  // Trigger a signaling.
constexpr TriggerType TriggerSync = 0x4;  // Trigger a flush.

constexpr unsigned int TriggerBitsSize = 32;
constexpr unsigned int TriggerBitsOffset = 32;
constexpr unsigned int TriggerBitsMemoryId = 9;
constexpr unsigned int TriggerBitsType = 3;
constexpr unsigned int TriggerBitsSemaphoreId = 10;
constexpr unsigned int TriggerBitsFifoReserved = 1;

/// Basic structure of each work element in the FIFO.
union ChannelTrigger {
  ProxyTrigger value;
  // The summation of number of bits must be 128 or less.
  struct {
    // First 64 bits: value[0]
    uint64_t size : TriggerBitsSize;
    uint64_t srcOffset : TriggerBitsOffset;
    uint64_t : (64 - TriggerBitsSize - TriggerBitsOffset);  // ensure 64-bit alignment
    // Second 64 bits: value[1]
    uint64_t dstOffset : TriggerBitsOffset;
    uint64_t srcMemoryId : TriggerBitsMemoryId;
    uint64_t dstMemoryId : TriggerBitsMemoryId;
    uint64_t type : TriggerBitsType;
    uint64_t semaphoreId : TriggerBitsSemaphoreId;
    uint64_t : (64 - TriggerBitsOffset - TriggerBitsMemoryId - TriggerBitsMemoryId - TriggerBitsType -
                TriggerBitsSemaphoreId - TriggerBitsFifoReserved);  // ensure 64-bit alignment
    uint64_t reserved : TriggerBitsFifoReserved;
  } fields;

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Default constructor.
  MSCCLPP_INLINE ChannelTrigger() = default;

  /// Copy constructor.
  MSCCLPP_DEVICE_INLINE ChannelTrigger(ProxyTrigger value) : value(value) {}

  /// Constructor.
  /// @param type The type of the trigger.
  /// @param dst The destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param src The source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param bytes The bytes of the transfer.
  /// @param semaphoreId The ID of the semaphore.
  MSCCLPP_DEVICE_INLINE ChannelTrigger(TriggerType type, MemoryId dst, uint64_t dstOffset, MemoryId src,
                                       uint64_t srcOffset, uint64_t bytes, int semaphoreId) {
    MSCCLPP_ASSERT_DEVICE(type < (1ULL << TriggerBitsType), "type is too large");
    MSCCLPP_ASSERT_DEVICE(dst < (1ULL << TriggerBitsMemoryId), "dst is too large");
    MSCCLPP_ASSERT_DEVICE(dstOffset < (1ULL << TriggerBitsOffset), "dstOffset is too large");
    MSCCLPP_ASSERT_DEVICE(src < (1ULL << TriggerBitsMemoryId), "src is too large");
    MSCCLPP_ASSERT_DEVICE(srcOffset < (1ULL << TriggerBitsOffset), "srcOffset is too large");
    MSCCLPP_ASSERT_DEVICE(bytes != 0, "bytes must not be zero");
    MSCCLPP_ASSERT_DEVICE(bytes < (1ULL << TriggerBitsSize), "bytes is too large");
    MSCCLPP_ASSERT_DEVICE(semaphoreId < (1ULL << TriggerBitsSemaphoreId), "semaphoreId is too large");
    constexpr uint64_t maskSize = (1ULL << TriggerBitsSize) - 1;
    constexpr uint64_t maskSrcOffset = (1ULL << TriggerBitsOffset) - 1;
    constexpr uint64_t maskDstOffset = (1ULL << TriggerBitsOffset) - 1;
    constexpr uint64_t maskSrcMemoryId = (1ULL << TriggerBitsMemoryId) - 1;
    constexpr uint64_t maskDstMemoryId = (1ULL << TriggerBitsMemoryId) - 1;
    constexpr uint64_t maskType = (1ULL << TriggerBitsType) - 1;
    constexpr uint64_t maskSemaphoreId = (1ULL << TriggerBitsSemaphoreId) - 1;
    value.fst = (((srcOffset & maskSrcOffset) << TriggerBitsSize) + (bytes & maskSize));
    value.snd = (((((((((semaphoreId & maskSemaphoreId) << TriggerBitsType) + ((uint64_t)type & maskType))
                      << TriggerBitsMemoryId) +
                     (dst & maskDstMemoryId))
                    << TriggerBitsMemoryId) +
                   (src & maskSrcMemoryId))
                  << TriggerBitsOffset) +
                 (dstOffset & maskDstOffset));
  }
#endif  // defined(MSCCLPP_DEVICE_COMPILE)
};

struct BasePortChannelDeviceHandle {
  SemaphoreId semaphoreId_;

  Host2DeviceSemaphoreDeviceHandle semaphore_;

  // this is a concurrent fifo which is multiple threads from the device
  // can produce for and the sole proxy thread consumes it.
  FifoDeviceHandle fifo_;

  MSCCLPP_INLINE BasePortChannelDeviceHandle() = default;

  MSCCLPP_HOST_DEVICE_INLINE BasePortChannelDeviceHandle(SemaphoreId semaphoreId,
                                                         Host2DeviceSemaphoreDeviceHandle semaphore,
                                                         FifoDeviceHandle fifo)
      : semaphoreId_(semaphoreId), semaphore_(semaphore), fifo_(fifo) {}

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Push a TriggerData to the FIFO.
  /// @param dst The destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param src The source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void put(MemoryId dst, uint64_t dstOffset, MemoryId src, uint64_t srcOffset, uint64_t size) {
    fifo_.push(ChannelTrigger(TriggerData, dst, dstOffset, src, srcOffset, size, semaphoreId_).value);
  }

  /// Push a TriggerData to the FIFO.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void put(MemoryId dst, MemoryId src, uint64_t offset, uint64_t size) {
    put(dst, offset, src, offset, size);
  }

  /// Push a TriggerFlag to the FIFO.
  MSCCLPP_DEVICE_INLINE void signal() { fifo_.push(ChannelTrigger(TriggerFlag, 0, 0, 0, 0, 1, semaphoreId_).value); }

  /// Push a TriggerData and a TriggerFlag at the same time to the FIFO.
  /// @param dst The destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param src The source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void putWithSignal(MemoryId dst, uint64_t dstOffset, MemoryId src, uint64_t srcOffset,
                                           uint64_t size) {
    fifo_.push(ChannelTrigger(TriggerData | TriggerFlag, dst, dstOffset, src, srcOffset, size, semaphoreId_).value);
  }

  /// Push a TriggerData and a TriggerFlag at the same time to the FIFO.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void putWithSignal(MemoryId dst, MemoryId src, uint64_t offset, uint64_t size) {
    putWithSignal(dst, offset, src, offset, size);
  }

  /// Push a TriggerData, a TriggerFlag, and a TriggerSync at the same time to the FIFO.
  /// @param dst The destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param src The source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(MemoryId dst, uint64_t dstOffset, MemoryId src, uint64_t srcOffset,
                                                   uint64_t size, int64_t maxSpinCount = 1000000) {
    uint64_t curFifoHead = fifo_.push(
        ChannelTrigger(TriggerData | TriggerFlag | TriggerSync, dst, dstOffset, src, srcOffset, size, semaphoreId_)
            .value);
    fifo_.sync(curFifoHead, maxSpinCount);
  }

  /// Push a TriggerData, a TriggerFlag, and a TriggerSync at the same time to the FIFO.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(MemoryId dst, MemoryId src, uint64_t offset, uint64_t size,
                                                   int64_t maxSpinCount = 1000000) {
    putWithSignalAndFlush(dst, offset, src, offset, size, maxSpinCount);
  }

  /// Push a TriggerSync to the FIFO.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  MSCCLPP_DEVICE_INLINE void flush(int64_t maxSpinCount = 1000000) {
    uint64_t curFifoHead = fifo_.push(ChannelTrigger(TriggerSync, 0, 0, 0, 0, 1, semaphoreId_).value);
    fifo_.sync(curFifoHead, maxSpinCount);
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
                                                     MemoryId dst, MemoryId src)
      : BasePortChannelDeviceHandle(semaphoreId, semaphore, fifo), dst_(dst), src_(src) {}

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
#endif  // defined(MSCCLPP_DEVICE_COMPILE)
};

}  // namespace mscclpp

#endif  // MSCCLPP_PORT_CHANNEL_DEVICE_HPP_
