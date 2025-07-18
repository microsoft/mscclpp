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

/// Concurrent FIFO where multiple device threads (the number of threads should not exceed the FIFO size) to push
/// Head pointer is on device, tail pointer is on host (readable by device).
/// The FIFO’s capacity is limited only by MAX_UINT64—effectively infinite for practical use. Exceeding this limit will
/// overflow the counter and lead to undefined behavior.
struct FifoDeviceHandle {
#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Push a trigger to the FIFO.
  /// @param trigger Trigger to push.
  /// @param maxSpinCount Max spin count before assert. Never assert if negative.
  /// @return Previous head of the FIFO where the trigger was pushed.
  MSCCLPP_DEVICE_INLINE uint64_t push(ProxyTrigger trigger, [[maybe_unused]] int64_t maxSpinCount = 1000000) {
    uint64_t prevHead = atomicFetchAdd<uint64_t, scopeDevice>(head, 1, memoryOrderRelaxed);

    // Flip the last bit for safe polling; host will revert.
    constexpr uint64_t flipMask = uint64_t{1} << uint64_t{63};
    trigger.snd ^= flipMask;

    // Wait until the trigger is freed by the host.
    if (prevHead >= size + *tailCache) {
      sync(prevHead - size, maxSpinCount);
    }

    ProxyTrigger* triggerPtr = &(triggers[prevHead % size]);

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
    uint64_t val;
    POLL_MAYBE_JAILBREAK((fifoHead >= (val = atomicLoad(tail, memoryOrderAcquire))), maxSpinCount);
    // If multiple threads sync in parallel, this may write a stale value to tailCache.
    // This is fine, as the tailCache is for avoiding unnecessary syncs from the push(),
    // which can work as long as the tailCache is not stale by the length of the FIFO.
    *tailCache = val;
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
