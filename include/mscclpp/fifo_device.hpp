// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_FIFO_DEVICE_HPP_
#define MSCCLPP_FIFO_DEVICE_HPP_

#include <cstdint>

#include "device.hpp"

#if defined(MSCCLPP_DEVICE_COMPILE)
#include "atomic_device.hpp"
#include "poll_device.hpp"
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

namespace mscclpp {

#if defined(MSCCLPP_DEVICE_COMPILE)
MSCCLPP_DEVICE_INLINE uint64_t hostLoadRelaxed(uint64_t* ptr) {
  uint64_t val;
#if defined(MSCCLPP_DEVICE_CUDA) && (__CUDA_ARCH__ == 800)
  // This is faster for A100.
  asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(val) : "l"(ptr));
#else   // !defined(MSCCLPP_DEVICE_CUDA) || (__CUDA_ARCH__ != 800)
  val = atomicLoad(ptr, memoryOrderRelaxed);
#endif  // !defined(MSCCLPP_DEVICE_CUDA) || (__CUDA_ARCH__ != 800)
  return val;
}
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

/// Pair of 64-bit unsigned integers used as a trigger for the proxy.
/// Used as a work element in the concurrent FIFO.
/// Most significant bit of snd is reserved.
struct alignas(16) ProxyTrigger {
  uint64_t fst, snd;
};

/// Concurrent FIFO for multiple device threads to push work elements and a single host proxy thread to consume them.
/// Head pointer is on device, tail pointer is on host (readable by device).
struct FifoDeviceHandle {
#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Push a trigger to the FIFO.
  /// @param trigger Trigger to push.
  /// @param maxSpinCount Max spin count before assert. Never assert if negative.
  /// @return Previous head of the FIFO where the trigger was pushed.
  MSCCLPP_DEVICE_INLINE uint64_t push(ProxyTrigger trigger, [[maybe_unused]] int64_t maxSpinCount = 1000000) {
    uint64_t prevHead = atomicFetchAdd<uint64_t, scopeDevice>(head, uint64_t{1}, memoryOrderRelaxed);
    int triggerIdx = prevHead % size;

    // Ensure that only one thread pushes into the same trigger slot at a time.
    int* triggerLock = &triggerLocks[triggerIdx];
    while (atomicFetchAdd<int, scopeDevice>(triggerLock, 1, memoryOrderRelaxed) != 0) {
      POLL_MAYBE_JAILBREAK((atomicLoad<int, scopeDevice>(triggerLock, memoryOrderRelaxed) != 0), maxSpinCount);
    }

    // Flip the last bit for safe polling; host will revert.
    constexpr uint64_t flipMask = uint64_t{1} << uint64_t{63};
    trigger.snd ^= flipMask;

    // Wait until the trigger is freed by the host.
    if (prevHead - atomicLoad<uint64_t, scopeDevice>(tailCache, memoryOrderRelaxed) >= size) {
      POLL_MAYBE_JAILBREAK((hostLoadRelaxed(&(triggers[triggerIdx].fst)) != 0), maxSpinCount);
      atomicStore<uint64_t, scopeDevice>(tailCache, prevHead + 1, memoryOrderRelaxed);
    }

    ProxyTrigger* triggerPtr = &(triggers[triggerIdx]);

    // Ensure data is visible to host before updating tail.
#if defined(MSCCLPP_DEVICE_CUDA)
#if __CUDA_ARCH__ == 800
    // This is faster than release for A100.
    __threadfence_system();
    asm volatile("st.global.relaxed.sys.v2.u64 [%0], {%1,%2};" ::"l"(triggerPtr), "l"(trigger.fst), "l"(trigger.snd));
#else
    asm volatile("st.global.release.sys.v2.u64 [%0], {%1,%2};" ::"l"(triggerPtr), "l"(trigger.fst), "l"(trigger.snd));
#endif
#else   // !defined(MSCCLPP_DEVICE_CUDA)
    // Store snd no later than fst.
    atomicStore(&(triggerPtr->snd), trigger.snd, memoryOrderRelaxed);
    atomicStore(&(triggerPtr->fst), trigger.fst, memoryOrderRelease);
#endif  // !defined(MSCCLPP_DEVICE_CUDA)

    // Free the trigger lock.
    atomicStore<int, scopeDevice>(triggerLock, 0, memoryOrderRelaxed);

    return prevHead;
  }

  /// Wait until a specific trigger is popped from the FIFO.
  /// @param fifoHead FIFO head where the trigger was pushed.
  /// @param maxSpinCount Max spin count before assert. Never assert if negative.
  MSCCLPP_DEVICE_INLINE void sync(uint64_t fifoHead, [[maybe_unused]] int64_t maxSpinCount = 1000000) {
    if (fifoHead < atomicLoad<uint64_t, scopeDevice>(tailCache, memoryOrderRelaxed)) return;
    POLL_MAYBE_JAILBREAK((hostLoadRelaxed(&(triggers[fifoHead % size].fst)) != 0), maxSpinCount);
    atomicStore<uint64_t, scopeDevice>(tailCache, fifoHead + 1, memoryOrderRelaxed);
  }
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

  /// FIFO buffer on host.
  ProxyTrigger* triggers;
  /// FIFO head on device.
  uint64_t* head;
  /// FIFO tail on host.
  uint64_t* tail;
  /// Cached tail value.
  uint64_t* tailCache;
  /// Array of flags to lock each trigger slot.
  int* triggerLocks;
  /// FIFO size.
  int size;
};

}  // namespace mscclpp

#endif  // MSCCLPP_FIFO_DEVICE_HPP_
