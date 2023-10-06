// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_SEMAPHORE_DEVICE_HPP_
#define MSCCLPP_SEMAPHORE_DEVICE_HPP_

#include <cuda/atomic>

#include "poll.hpp"

namespace mscclpp {

/// Device-side handle for @ref Host2DeviceSemaphore.
struct Host2DeviceSemaphoreDeviceHandle {
#ifdef __CUDACC__
  /// Poll if the host has signaled.
  /// @return true if the host has signaled.
  __forceinline__ __device__ bool poll() {
    bool signaled = (cuda::atomic_ref<uint64_t, cuda::thread_scope_system>{*inboundSemaphoreId}.load(
                         cuda::memory_order_acquire) > (*expectedInboundSemaphoreId));
    if (signaled) (*expectedInboundSemaphoreId) += 1;
    return signaled;
  }

  /// Wait for the host to signal.
  __forceinline__ __device__ void wait(int64_t maxSpinCount = 10000000) {
    (*expectedInboundSemaphoreId) += 1;
    POLL_MAYBE_JAILBREAK((cuda::atomic_ref<uint64_t, cuda::thread_scope_system>{*inboundSemaphoreId}.load(
                              cuda::memory_order_acquire) < (*expectedInboundSemaphoreId)),
                         maxSpinCount);
  }
#endif  // __CUDACC__

  uint64_t* inboundSemaphoreId;
  uint64_t* expectedInboundSemaphoreId;
};

/// Device-side handle for @ref SmDevice2DeviceSemaphore.
struct SmDevice2DeviceSemaphoreDeviceHandle {
#ifdef __CUDACC__
  /// Poll if the remote device has signaled.
  /// @return true if the remote device has signaled.
  __forceinline__ __device__ bool poll() {
    bool signaled = (cuda::atomic_ref<uint64_t, cuda::thread_scope_system>{*inboundSemaphoreId}.load(
                         cuda::memory_order_acquire) > (*expectedInboundSemaphoreId));
    if (signaled) (*expectedInboundSemaphoreId) += 1;
    return signaled;
  }

  /// Wait for the remote device to signal.
  __forceinline__ __device__ void wait(int64_t maxSpinCount = 10000000) {
    (*expectedInboundSemaphoreId) += 1;
    POLL_MAYBE_JAILBREAK((cuda::atomic_ref<uint64_t, cuda::thread_scope_system>{*inboundSemaphoreId}.load(
                              cuda::memory_order_acquire) < (*expectedInboundSemaphoreId)),
                         maxSpinCount);
  }

  /// Signal the remote device.
  ///
  /// This function guarantees that all the memory operation before this function is completed before the remote
  /// semaphore is signaled.
  ///
  __forceinline__ __device__ void signal() {
    // This fence ensures that preceding writes are visible on the peer GPU before the incremented
    // `outboundSemaphoreId` is visible.
    semaphoreIncrement();
    cuda::atomic_ref<uint64_t, cuda::thread_scope_system>{*remoteInboundSemaphoreId}.store(semaphoreGetLocal(),
                                                                                           cuda::memory_order_release);
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

  uint64_t* inboundSemaphoreId;
  uint64_t* outboundSemaphoreId;
  uint64_t* remoteInboundSemaphoreId;
  uint64_t* expectedInboundSemaphoreId;
};

}  // namespace mscclpp

#endif  // MSCCLPP_SEMAPHORE_DEVICE_HPP_
