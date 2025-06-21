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
///
/// This struct is used as a work element in the concurrent FIFO where multiple device threads can push
/// ProxyTrigger elements and a single host proxy thread consumes these work elements.
///
/// Do not use the most significant bit of snd as it is reserved for memory consistency purposes.
struct alignas(16) ProxyTrigger {
  uint64_t fst, snd;
};

/// A concurrent FIFO where multiple device threads (the number of threads should not exceed the FIFO size) can push
/// work elements and a single host proxy thread consumes them.
///
/// The FIFO has a head pointer allocated on the device which starts at 0 and goes up to 2^64-1, which is almost
/// infinity. If `env()->fifoUseTailReplica` is true, there are two copies of the tail, one on the device,
/// FifoDeviceHandle::tailReplica, and another on the host, namely, hostTail.
/// The host always has the "true" tail and occasionally pushes it to the copy on the device.
/// Therefore, most of the time, the device has a stale version. The invariants are: tailReplica <= hostTail <= head.
/// The push() function increments head, hostTail is updated in Fifo::pop(), and it occasionally flushes
/// it to tailReplica via Fifo::flushTail().
///
/// If `env()->fifoUseTailReplica` is false, FifoDeviceHandle::tailReplica points to the original tail on the host.
/// In this case, the tail is always up-to-date and there is no need to flush it to the device.
///
/// Duplicating the tail is a good idea because the FIFO is large enough, and we do not need frequent updates for the
/// tail as there is usually enough space for device threads to push their work into.
///
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

    // Only one of two conditions need to be met to proceed. Either the tail has advanced enough or where we need to
    // write to is 0. However, the first condition is faster to check since the tail is flushed periodically anyways but
    // for the second condition we need to read CPU memory.
    // As atomic access is slow, we first check using the bare pointer and then use the atomic load if the
    // condition is not met.
    if (prevHead >= size + *tailReplica) {
      OR_POLL_MAYBE_JAILBREAK((prevHead >= size + atomicLoad(tailReplica, memoryOrderRelaxed)),
                              (hostLoadRelaxed(&(triggers[prevHead % size].fst)) != 0),
                              maxSpinCount);
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
    // Same as push but in this case checking the first condition is probably faster since for tail to be pushed we need
    // to wait for cudaMemcpy to be done.
    if (fifoHead < *tailReplica) return;
    OR_POLL_MAYBE_JAILBREAK((fifoHead >= atomicLoad(tailReplica, memoryOrderRelaxed)),
                            (hostLoadRelaxed(&(triggers[fifoHead % size].fst)) != 0),
                            maxSpinCount);
  }
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

  /// FIFO buffer on host.
  ProxyTrigger* triggers;
  /// FIFO head on device.
  uint64_t* head;
  /// FIFO tail replica on device.
  uint64_t* tailReplica;
  /// FIFO size.
  int size;
};

}  // namespace mscclpp

#endif  // MSCCLPP_FIFO_DEVICE_HPP_
