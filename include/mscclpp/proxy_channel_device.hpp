// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_PROXY_CHANNEL_DEVICE_HPP_
#define MSCCLPP_PROXY_CHANNEL_DEVICE_HPP_

#include "fifo_device.hpp"
#include "semaphore_device.hpp"

namespace mscclpp {

using SemaphoreId = uint32_t;

/// Numeric ID of @ref RegisteredMemory. @ref ProxyService has an internal array indexed by these handles mapping to the
/// actual.
using MemoryId = uint32_t;

using TriggerType = uint64_t;
const TriggerType TriggerData = 0x1;  // Trigger a data transfer.
const TriggerType TriggerFlag = 0x2;  // Trigger a signaling.
const TriggerType TriggerSync = 0x4;  // Trigger a flush.

#define MSCCLPP_BITS_SIZE 32
#define MSCCLPP_BITS_OFFSET 32
#define MSCCLPP_BITS_REGMEM_HANDLE 8
#define MSCCLPP_BITS_TYPE 3
#define MSCCLPP_BITS_CONNID 10

/// Basic structure of each work element in the FIFO.
union ChannelTrigger {
  ProxyTrigger value;
  // The summation of number of bits must be 128 or less.
  struct {
    // First 64 bits: value[0]
    uint64_t size : MSCCLPP_BITS_SIZE;
    uint64_t srcOffset : MSCCLPP_BITS_OFFSET;
    uint64_t : (64 - MSCCLPP_BITS_SIZE - MSCCLPP_BITS_OFFSET);  // ensure 64-bit alignment
    // Second 64 bits: value[1]
    uint64_t dstOffset : MSCCLPP_BITS_OFFSET;
    uint64_t srcMemoryId : MSCCLPP_BITS_REGMEM_HANDLE;
    uint64_t dstMemoryId : MSCCLPP_BITS_REGMEM_HANDLE;
    uint64_t type : MSCCLPP_BITS_TYPE;
    uint64_t chanId : MSCCLPP_BITS_CONNID;
    uint64_t : (64 - MSCCLPP_BITS_OFFSET - MSCCLPP_BITS_REGMEM_HANDLE - MSCCLPP_BITS_REGMEM_HANDLE -
                MSCCLPP_BITS_TYPE);  // ensure 64-bit alignment
  } fields;

#ifdef __CUDACC__
  /// Default constructor.
  __forceinline__ __device__ ChannelTrigger() {}

  /// Copy constructor.
  __forceinline__ __device__ ChannelTrigger(ProxyTrigger value) : value(value) {}

  /// Constructor.
  /// @param type The type of the trigger.
  /// @param dst The destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param src The source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param bytes The bytes of the transfer.
  /// @param semaphoreId The ID of the semaphore.
  __forceinline__ __device__ ChannelTrigger(TriggerType type, MemoryId dst, uint64_t dstOffset, MemoryId src,
                                            uint64_t srcOffset, uint64_t bytes, int semaphoreId) {
    value.fst = ((srcOffset << MSCCLPP_BITS_SIZE) + bytes);
    value.snd = ((((((((semaphoreId << MSCCLPP_BITS_TYPE) + (uint64_t)type) << MSCCLPP_BITS_REGMEM_HANDLE) + dst)
                    << MSCCLPP_BITS_REGMEM_HANDLE) +
                   src)
                  << MSCCLPP_BITS_OFFSET) +
                 dstOffset);
  }
#endif  // __CUDACC__
};

struct ProxyChannelDeviceHandle {
  SemaphoreId semaphoreId_;

  Host2DeviceSemaphoreDeviceHandle semaphore_;

  // this is a concurrent fifo which is multiple threads from the device
  // can produce for and the sole proxy thread consumes it.
  FifoDeviceHandle fifo_;

#ifdef __CUDACC__
  /// Push a @ref TriggerData to the FIFO.
  /// @param dst The destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param src The source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  __forceinline__ __device__ void put(MemoryId dst, uint64_t dstOffset, MemoryId src, uint64_t srcOffset,
                                      uint64_t size) {
    fifo_.push(ChannelTrigger(TriggerData, dst, dstOffset, src, srcOffset, size, semaphoreId_).value);
  }

  /// Push a @ref TriggerData to the FIFO.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  __forceinline__ __device__ void put(MemoryId dst, MemoryId src, uint64_t offset, uint64_t size) {
    put(dst, offset, src, offset, size);
  }

  /// Push a @ref TriggerFlag to the FIFO.
  __forceinline__ __device__ void signal() {
    fifo_.push(ChannelTrigger(TriggerFlag, 0, 0, 0, 0, 1, semaphoreId_).value);
  }

  /// Push a @ref TriggerData and a @ref TriggerFlag at the same time to the FIFO.
  /// @param dst The destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param src The source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  __forceinline__ __device__ void putWithSignal(MemoryId dst, uint64_t dstOffset, MemoryId src, uint64_t srcOffset,
                                                uint64_t size) {
    fifo_.push(ChannelTrigger(TriggerData | TriggerFlag, dst, dstOffset, src, srcOffset, size, semaphoreId_).value);
  }

  /// Push a @ref TriggerData and a @ref TriggerFlag at the same time to the FIFO.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  __forceinline__ __device__ void putWithSignal(MemoryId dst, MemoryId src, uint64_t offset, uint64_t size) {
    putWithSignal(dst, offset, src, offset, size);
  }

  /// Push a @ref TriggerData, a @ref TriggerFlag, and a @ref TriggerSync at the same time to the FIFO.
  /// @param dst The destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param src The source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  __forceinline__ __device__ void putWithSignalAndFlush(MemoryId dst, uint64_t dstOffset, MemoryId src,
                                                        uint64_t srcOffset, uint64_t size) {
    uint64_t curFifoHead = fifo_.push(
        ChannelTrigger(TriggerData | TriggerFlag | TriggerSync, dst, dstOffset, src, srcOffset, size, semaphoreId_)
            .value);
    fifo_.sync(curFifoHead);
  }

  /// Push a @ref TriggerData, a @ref TriggerFlag, and a @ref TriggerSync at the same time to the FIFO.
  /// @param dst The destination memory region.
  /// @param src The source memory region.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  __forceinline__ __device__ void putWithSignalAndFlush(MemoryId dst, MemoryId src, uint64_t offset, uint64_t size) {
    putWithSignalAndFlush(dst, offset, src, offset, size);
  }

  /// Push a @ref TriggerSync to the FIFO.
  __forceinline__ __device__ void flush() {
    uint64_t curFifoHead = fifo_.push(ChannelTrigger(TriggerSync, 0, 0, 0, 0, 1, semaphoreId_).value);
    fifo_.sync(curFifoHead);
  }

  /// Wait for the proxy channel to be signaled.
  __forceinline__ __device__ void wait() { semaphore_.wait(); }

#endif  // __CUDACC__
};

struct SimpleProxyChannelDeviceHandle {
  ProxyChannelDeviceHandle proxyChan_;
  MemoryId dst_;
  MemoryId src_;

#ifdef __CUDACC__
  /// Push a @ref TriggerData to the FIFO.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  __forceinline__ __device__ void put(uint64_t dstOffset, uint64_t srcOffset, uint64_t size) {
    proxyChan_.put(dst_, dstOffset, src_, srcOffset, size);
  }

  /// Push a @ref TriggerData to the FIFO.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  __forceinline__ __device__ void put(uint64_t offset, uint64_t size) { put(offset, offset, size); }

  /// Push a @ref TriggerFlag to the FIFO.
  __forceinline__ __device__ void signal() { proxyChan_.signal(); }

  /// Push a @ref TriggerData and a @ref TriggerFlag at the same time to the FIFO.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  __forceinline__ __device__ void putWithSignal(uint64_t dstOffset, uint64_t srcOffset, uint64_t size) {
    proxyChan_.putWithSignal(dst_, dstOffset, src_, srcOffset, size);
  }

  /// Push a @ref TriggerData and a @ref TriggerFlag at the same time to the FIFO.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  __forceinline__ __device__ void putWithSignal(uint64_t offset, uint64_t size) { putWithSignal(offset, offset, size); }

  /// Push a @ref TriggerData, a @ref TriggerFlag, and a @ref TriggerSync at the same time to the FIFO.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  __forceinline__ __device__ void putWithSignalAndFlush(uint64_t dstOffset, uint64_t srcOffset, uint64_t size) {
    proxyChan_.putWithSignalAndFlush(dst_, dstOffset, src_, srcOffset, size);
  }

  /// Push a @ref TriggerData, a @ref TriggerFlag, and a @ref TriggerSync at the same time to the FIFO.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  __forceinline__ __device__ void putWithSignalAndFlush(uint64_t offset, uint64_t size) {
    putWithSignalAndFlush(offset, offset, size);
  }

  /// Push a @ref TriggerSync to the FIFO.
  __forceinline__ __device__ void flush() { proxyChan_.flush(); }

  /// Wait for the proxy channel to be signaled.
  __forceinline__ __device__ void wait() { proxyChan_.wait(); }
#endif  // __CUDACC__
};

}  // namespace mscclpp

#endif  // MSCCLPP_PROXY_CHANNEL_DEVICE_HPP_
