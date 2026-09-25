// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_PORT_CHANNEL_DEVICE_HPP_
#define MSCCLPP_PORT_CHANNEL_DEVICE_HPP_

#include "fifo_device.hpp"
#include "port_channel_gpunetio_device.hpp"
#include "semaphore_device.hpp"

namespace mscclpp {

/// Backend that services a PortChannel's device-side operations.
/// The same device API (`put`, `signal`,
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
  PortChannelBackend backend_;

  // GPU-initiated networking context; only used when backend_ == GpuNetIo.
  GpuNetIoDeviceContext* gpuNetIo_;
  /// Remote bootstrap rank, independent of proxy IDs.
  int gpuNetIoPeer_;
  /// Queue selected by the connection binding.
  int gpuNetIoQpIndex_;
  /// Owner of local memory registrations.
  int gpuNetIoLocalRank_;
  /// Peer's private registered signal counter.
  GpuNetIoMemoryDeviceHandle gpuNetIoSignal_;
  /// Per-channel table indexed by MemoryId.
  const GpuNetIoMemoryDeviceHandle* gpuNetIoMemories_;
  uint32_t gpuNetIoMemoryCount_;

  MSCCLPP_INLINE BasePortChannelDeviceHandle() = default;

  MSCCLPP_HOST_DEVICE_INLINE BasePortChannelDeviceHandle(SemaphoreId semaphoreId,
                                                         Host2DeviceSemaphoreDeviceHandle semaphore,
                                                         FifoDeviceHandle fifo, uint64_t* flushDonePos)
      : semaphoreId_(semaphoreId),
        semaphore_(semaphore),
        fifo_(fifo),
        flushDonePos_(flushDonePos),
        backend_(PortChannelBackend::Proxy),
        gpuNetIo_(nullptr),
        gpuNetIoPeer_(-1),
        gpuNetIoQpIndex_(0),
        gpuNetIoLocalRank_(-1),
        gpuNetIoSignal_{},
        gpuNetIoMemories_(nullptr),
        gpuNetIoMemoryCount_(0) {}

  /// Build a handle from an owned host connection/semaphore binding and memory table.
  /// MemoryIds index this table; all resources must outlive GPU use.
  MSCCLPP_HOST_DEVICE_INLINE BasePortChannelDeviceHandle(GpuNetIoDeviceContext* gpuNetIo, int peer, int qpIndex,
                                                         int localRank, GpuNetIoMemoryDeviceHandle signal,
                                                         uint64_t* inboundSignal, uint64_t* expectedSignal,
                                                         const GpuNetIoMemoryDeviceHandle* memories,
                                                         uint32_t memoryCount)
      : semaphoreId_(0),
        semaphore_{inboundSignal, expectedSignal},
        fifo_{},
        flushDonePos_(nullptr),
        backend_(PortChannelBackend::GpuNetIo),
        gpuNetIo_(gpuNetIo),
        gpuNetIoPeer_(peer),
        gpuNetIoQpIndex_(qpIndex),
        gpuNetIoLocalRank_(localRank),
        gpuNetIoSignal_(signal),
        gpuNetIoMemories_(memories),
        gpuNetIoMemoryCount_(memoryCount) {}

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Check the connection/semaphore binding in every device build mode.
  MSCCLPP_DEVICE_INLINE void validateGpuNetIo() const {
    requireGpuNetIo(gpuNetIo_ != nullptr);
    requireGpuNetIo(gpuNetIoPeer_ >= 0 && gpuNetIoPeer_ < gpuNetIo_->numPeers && gpuNetIoPeer_ != gpuNetIoLocalRank_ &&
                    gpuNetIoQpIndex_ >= 0 && gpuNetIoQpIndex_ < gpuNetIo_->numQpsPerPeer &&
                    gpuNetIoSignal_.rank == gpuNetIoPeer_ && gpuNetIoSignal_.base != 0 &&
                    gpuNetIoSignal_.base % sizeof(uint64_t) == 0 && gpuNetIoSignal_.bytes == sizeof(uint64_t) &&
                    semaphore_.inboundToken && semaphore_.expectedInboundToken);
  }

  /// Reject invalid bindings or accesses even when device assertions are disabled.
  MSCCLPP_DEVICE_INLINE static void requireGpuNetIo(bool valid) {
    if (!valid) {
      MSCCLPP_ASSERT_DEVICE(false, "Invalid GPUNetIO channel binding or memory access");
#if defined(MSCCLPP_DEVICE_CUDA)
      __trap();
#else
      __builtin_trap();
#endif
    }
  }

  /// Resolve a memory ID and validate its owner and range before issuing a WQE.
  MSCCLPP_DEVICE_INLINE GpuNetIoMemoryDeviceHandle gpuNetIoMemory(MemoryId id, int rank, uint64_t offset,
                                                                  uint64_t bytes) const {
    requireGpuNetIo(gpuNetIoMemories_ != nullptr && id < gpuNetIoMemoryCount_);
    const auto memory = gpuNetIoMemories_[id];
    requireGpuNetIo(memory.rank == rank && memory.base != 0 && offset <= memory.bytes &&
                    bytes <= memory.bytes - offset && memory.bytes <= UINTPTR_MAX - memory.base);
    return memory;
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
      const auto dst = gpuNetIoMemory(dstId, gpuNetIoPeer_, dstOffset, size);
      const auto src = gpuNetIoMemory(srcId, gpuNetIoLocalRank_, srcOffset, size);
      gpuNetIo_->putRegistered(gpuNetIoPeer_, gpuNetIoQpIndex_, dst, dstOffset, src, srcOffset, size);
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
      gpuNetIo_->atomicAddRegistered(gpuNetIoPeer_, gpuNetIoQpIndex_, gpuNetIoSignal_, 0, 1);
      return;
    }
    fifo_.push({TriggerFlag, 0, 0, 0, 0, 0, semaphoreId_});
  }

  /// Push a (TriggerData | TriggerFlag) to the FIFO.
  /// @param dstId The ID of destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcId The ID of source memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void putWithSignal(MemoryId dstId, uint64_t dstOffset, MemoryId srcId, uint64_t srcOffset,
                                           uint64_t size) {
    if (backend_ == PortChannelBackend::GpuNetIo) {
      validateGpuNetIo();
      const auto dst = gpuNetIoMemory(dstId, gpuNetIoPeer_, dstOffset, size);
      const auto src = gpuNetIoMemory(srcId, gpuNetIoLocalRank_, srcOffset, size);
      gpuNetIo_->putRegisteredWithSignal(gpuNetIoPeer_, gpuNetIoQpIndex_, dst, dstOffset, src, srcOffset, size,
                                         gpuNetIoSignal_);
      return;
    }
    fifo_.push({(TriggerData | TriggerFlag), dstId, dstOffset, srcId, srcOffset, size, semaphoreId_});
  }

  /// Push a (TriggerData | TriggerFlag) to the FIFO.
  /// @param dstId The ID of destination memory region.
  /// @param srcId The ID of source memory region.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void putWithSignal(MemoryId dstId, MemoryId srcId, uint64_t offset, uint64_t size) {
    putWithSignal(dstId, offset, srcId, offset, size);
  }

  /// Push a (TriggerData | TriggerFlag | TriggerSync) to the FIFO.
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
        fifo_.push({(TriggerData | TriggerFlag | TriggerSync), dstId, dstOffset, srcId, srcOffset, size, semaphoreId_});
    detail::waitFlush(flushDonePos_, pos, maxSpinCount);
  }

  /// Push a (TriggerData | TriggerFlag | TriggerSync) to the FIFO.
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
        gpuNetIo_->flush(gpuNetIoPeer_, gpuNetIoQpIndex_);
      } else {
        const int status = gpuNetIo_->tryFlush(gpuNetIoPeer_, static_cast<uint64_t>(maxSpinCount), gpuNetIoQpIndex_);
        if (status != 0) {
          MSCCLPP_ASSERT_DEVICE(false, "GPUNetIO flush timed out or reported a CQ error");
          gpuNetIo_->flush(gpuNetIoPeer_, gpuNetIoQpIndex_);
        }
      }
      return;
    }
    uint64_t pos = fifo_.push({TriggerSync, 0, 0, 0, 0, 0, semaphoreId_});
    detail::waitFlush(flushDonePos_, pos, maxSpinCount);
  }

  /// Push an atomicAdd trigger to the FIFO: add a 64-bit value to remote memory.
  /// Connection::atomicAdd() documents how many concurrent writers each transport allows.
  /// @param dstId The ID of destination memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param value The 64-bit signed value to add.
  MSCCLPP_DEVICE_INLINE void atomicAdd(MemoryId dstId, uint64_t dstOffset, int64_t value) {
    if (backend_ == PortChannelBackend::GpuNetIo) {
      validateGpuNetIo();
      const auto dst = gpuNetIoMemory(dstId, gpuNetIoPeer_, dstOffset, sizeof(uint64_t));
      requireGpuNetIo((dst.base + dstOffset) % sizeof(uint64_t) == 0);
      gpuNetIo_->atomicAddRegistered(gpuNetIoPeer_, gpuNetIoQpIndex_, dst, dstOffset, value);
      return;
    }
    // The operand occupies fst, spanning the low size and high srcOffset fields.
    uint64_t operand = static_cast<uint64_t>(value);
    ProxyTrigger trigger(0, dstId, dstOffset, /*srcId=*/0, operand >> TriggerBitsSize, static_cast<uint32_t>(operand),
                         semaphoreId_);
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

  /// Bind source/destination IDs from a base channel's memory table.
  MSCCLPP_HOST_DEVICE_INLINE PortChannelDeviceHandle(BasePortChannelDeviceHandle base, MemoryId dst, MemoryId src)
      : BasePortChannelDeviceHandle(base), dst_(dst), src_(src) {}

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

  /// Push a (TriggerData | TriggerFlag) to the FIFO.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void putWithSignal(uint64_t dstOffset, uint64_t srcOffset, uint64_t size) {
    BasePortChannelDeviceHandle::putWithSignal(dst_, dstOffset, src_, srcOffset, size);
  }

  /// Push a (TriggerData | TriggerFlag) to the FIFO.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void putWithSignal(uint64_t offset, uint64_t size) { putWithSignal(offset, offset, size); }

  /// Push a (TriggerData | TriggerFlag | TriggerSync) to the FIFO.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param size The size of the transfer.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(uint64_t dstOffset, uint64_t srcOffset, uint64_t size,
                                                   int64_t maxSpinCount = 1000000) {
    BasePortChannelDeviceHandle::putWithSignalAndFlush(dst_, dstOffset, src_, srcOffset, size, maxSpinCount);
  }

  /// Push a (TriggerData | TriggerFlag | TriggerSync) to the FIFO.
  /// @param offset The common offset into the destination and source memory regions.
  /// @param size The size of the transfer.
  MSCCLPP_DEVICE_INLINE void putWithSignalAndFlush(uint64_t offset, uint64_t size) {
    putWithSignalAndFlush(offset, offset, size);
  }
  /// Push an atomicAdd trigger to the FIFO: add a 64-bit value to the destination memory.
  /// See Connection::atomicAdd() for transport support.
  /// @param dstOffset The offset into the destination memory region.
  /// @param value The 64-bit signed value to add.
  MSCCLPP_DEVICE_INLINE void atomicAdd(uint64_t dstOffset, int64_t value) {
    BasePortChannelDeviceHandle::atomicAdd(dst_, dstOffset, value);
  }
#endif  // defined(MSCCLPP_DEVICE_COMPILE)
};

}  // namespace mscclpp

#endif  // MSCCLPP_PORT_CHANNEL_DEVICE_HPP_
