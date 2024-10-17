// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_SEMAPHORE_DEVICE_HPP_
#define MSCCLPP_SEMAPHORE_DEVICE_HPP_

#include "device.hpp"

#if defined(MSCCLPP_DEVICE_COMPILE)
#include "atomic_device.hpp"
#include "poll_device.hpp"
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

namespace mscclpp {

/// Device-side handle for @ref Host2DeviceSemaphore.
struct Host2DeviceSemaphoreDeviceHandle {
#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Poll if the host has signaled.
  /// @param max_poll The max number of signals to poll.
  /// @return number of signals up to max_poll that the remote device has signaled.
  MSCCLPP_DEVICE_INLINE uint64_t poll(const int64_t max_poll = 1) {
    if (max_poll <= 0) return 0;
    uint64_t count = (atomicLoad(inboundSemaphoreId, memoryOrderAcquire) - (*expectedInboundSemaphoreId));
    if (count <= 0) {
      return 0;
    } else {
      if (max_poll < count) count = max_poll;
      *expectedInboundSemaphoreId += count;
      return count;
    }
  }

  /// Wait for the host to signal.
  MSCCLPP_DEVICE_INLINE void wait(int64_t maxSpinCount = 100000000) {
    (*expectedInboundSemaphoreId) += 1;
    POLL_MAYBE_JAILBREAK((atomicLoad(inboundSemaphoreId, memoryOrderAcquire) < (*expectedInboundSemaphoreId)),
                         maxSpinCount);
  }
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

  uint64_t* inboundSemaphoreId;
  uint64_t* expectedInboundSemaphoreId;
};

/// Device-side handle for @ref SmDevice2DeviceSemaphore.
struct SmDevice2DeviceSemaphoreDeviceHandle {
#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Poll if the remote device has signaled.
  /// @param max_poll The max number of signals to poll.
  /// @return number of signals up to max_poll that the remote device has signaled.
  MSCCLPP_DEVICE_INLINE uint64_t poll(const int64_t max_poll = 1) {
    if (max_poll <= 0) return 0;
    uint64_t count = (atomicLoad(inboundSemaphoreId, memoryOrderAcquire) - (*expectedInboundSemaphoreId));
    if (count <= 0) {
      return 0;
    } else {
      if (max_poll < count) count = max_poll;
      *expectedInboundSemaphoreId += count;
      return count;
    }
  }

  /// Wait for the remote device to signal.
  MSCCLPP_DEVICE_INLINE void wait(int64_t maxSpinCount = 100000000) {
    (*expectedInboundSemaphoreId) += 1;
    POLL_MAYBE_JAILBREAK((atomicLoad(inboundSemaphoreId, memoryOrderAcquire) < (*expectedInboundSemaphoreId)),
                         maxSpinCount);
  }

  /// Signal the remote device.
  ///
  /// This function guarantees that all the memory operation before this function is completed before the remote
  /// semaphore is signaled.
  ///
  MSCCLPP_DEVICE_INLINE void signal(const uint64_t count = 1) {
    // This fence ensures that preceding writes are visible on the peer GPU before the incremented
    // `outboundSemaphoreId` is visible.
    semaphoreIncrement(count);
    atomicStore(remoteInboundSemaphoreId, semaphoreGetLocal(), memoryOrderSeqCst);
  }

  /// Signal the remote device.
  ///
  /// This function is a relaxed version of signal() and provides no guarantee on the completion of memory operations.
  /// User requires to call proper fencing before using this function.
  ///
  MSCCLPP_DEVICE_INLINE void relaxedSignal() {
    // This fence ensures that preceding writes are visible on the peer GPU before the incremented
    // `outboundSemaphoreId` is visible.
    semaphoreIncrement();
    atomicStore(remoteInboundSemaphoreId, semaphoreGetLocal(), memoryOrderRelaxed);
  }

  /// Signal the remote device for copied packets.
  ///
  /// Unlike @ref signal(), this function provides no guarantee on the completion of memory operations. This is
  /// intended to be used with @ref putPackets() and @ref getPackets() that use flags inside packets to indicate the
  /// completion of copies.
  ///
  MSCCLPP_DEVICE_INLINE void signalPacket() {
    semaphoreIncrement();
    *remoteInboundSemaphoreId = semaphoreGetLocal();
  }

  /// Increase the counter of the local semaphore.
  MSCCLPP_DEVICE_INLINE void semaphoreIncrement(const uint64_t count = 1) { *outboundSemaphoreId += count; }

  /// Get the value of the local semaphore.
  MSCCLPP_DEVICE_INLINE uint64_t semaphoreGetLocal() const { return *outboundSemaphoreId; }
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

  uint64_t* inboundSemaphoreId;
  uint64_t* outboundSemaphoreId;
  uint64_t* remoteInboundSemaphoreId;
  uint64_t* expectedInboundSemaphoreId;
};

}  // namespace mscclpp

#endif  // MSCCLPP_SEMAPHORE_DEVICE_HPP_
