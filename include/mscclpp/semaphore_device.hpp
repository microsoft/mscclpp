// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_SEMAPHORE_DEVICE_HPP_
#define MSCCLPP_SEMAPHORE_DEVICE_HPP_

#include "poll.hpp"

namespace mscclpp {

/// Device-side handle for @ref Host2DeviceSemaphore.
struct Host2DeviceSemaphoreDeviceHandle {
#ifdef __CUDACC__
  /// Wait for the host to signal.
  __forceinline__ __device__ void wait() {
    (*expectedInboundSemaphoreId) += 1;
    POLL_MAYBE_JAILBREAK(*(volatile uint64_t*)(inboundSemaphoreId) < (*expectedInboundSemaphoreId), 100000000);
  }
#endif  // __CUDACC__

  uint64_t* inboundSemaphoreId;
  uint64_t* expectedInboundSemaphoreId;
};

/// Device-side handle for @ref SmDevice2DeviceSemaphore.
struct SmDevice2DeviceSemaphoreDeviceHandle {
#ifdef __CUDACC__
  /// Wait for the remote device to signal.
  __forceinline__ __device__ void wait() {
    (*expectedInboundSemaphoreId) += 1;
    POLL_MAYBE_JAILBREAK(*inboundSemaphoreId < (*expectedInboundSemaphoreId), 100000000);
  }

  /// Signal the remote device.
  ///
  /// This function guarantees that all the memory operation before this function is completed before the remote
  /// semaphore is signaled.
  ///
  __forceinline__ __device__ void signal() {
    // This fence ensures that preceding writes are visible on the peer GPU before the incremented
    // `outboundSemaphoreId` is visible.
    __threadfence_system();
    semaphoreIncrement();
    *remoteInboundSemaphoreId = semaphoreGetLocal();
  }

  /// Signal the remote device for copied packets.
  ///
  /// Unlike @ref signal(), this function provides no guarantee on the completion of memory operations. This is
  /// intended to be used with @ref putPackets() and @ref getPackets() that use flags inside packets to indicate the
  /// completion of copies.
  ///
  __forceinline__ __device__ void signalPacket() {
    semaphoreIncrement();
    *remoteInboundSemaphoreId = semaphoreGetLocal();
  }

  /// Increase the counter of the local semaphore.
  __forceinline__ __device__ void semaphoreIncrement() { *outboundSemaphoreId += 1; }

  /// Get the value of the local semaphore.
  __forceinline__ __device__ uint64_t semaphoreGetLocal() const { return *outboundSemaphoreId; }
#endif  // __CUDACC__

  volatile uint64_t* inboundSemaphoreId;
  uint64_t* outboundSemaphoreId;
  volatile uint64_t* remoteInboundSemaphoreId;
  uint64_t* expectedInboundSemaphoreId;
};

}  // namespace mscclpp

#endif  // MSCCLPP_SEMAPHORE_DEVICE_HPP_
