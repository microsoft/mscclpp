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
    uint64_t prevHead = atomicFetchAdd(head, uint64_t{1}, memoryOrderRelaxed);

    // Set last bit non-zero for safe polling; host will revert.
    trigger.snd ^= (uint64_t{1} << uint64_t{63});

    // Proceed if tail advanced or target slot is 0.
    // As tail value is cached on device, the device doesn't need to access host memory every time.
    uint64_t numInflights = prevHead - *tailCache;
    if (numInflights >= size / 2) {
      numInflights = prevHead - (*tailCache = atomicLoad(tail, memoryOrderRelaxed));
    }
    if (numInflights >= size) {
      POLL_MAYBE_JAILBREAK((atomicLoad(&(triggers[prevHead % size].fst), memoryOrderRelaxed) != 0), maxSpinCount);
      *tailCache = atomicLoad(tail, memoryOrderRelaxed);
    }

    ProxyTrigger* triggerPtr = &(triggers[prevHead % size]);

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

    return prevHead;
  }

  /// Wait until a specific trigger is popped from the FIFO.
  /// @param fifoHead FIFO head where the trigger was pushed.
  /// @param maxSpinCount Max spin count before assert. Never assert if negative.
  MSCCLPP_DEVICE_INLINE void sync(uint64_t fifoHead, [[maybe_unused]] int64_t maxSpinCount = 1000000) {
    if (fifoHead < *tailCache) return;
    POLL_MAYBE_JAILBREAK((atomicLoad(&(triggers[fifoHead % size].fst), memoryOrderRelaxed) != 0), maxSpinCount);
    *tailCache = fifoHead + 1;
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
  /// FIFO size.
  int size;
};

}  // namespace mscclpp

#endif  // MSCCLPP_FIFO_DEVICE_HPP_
