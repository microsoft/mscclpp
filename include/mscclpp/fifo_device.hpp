// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_FIFO_DEVICE_HPP_
#define MSCCLPP_FIFO_DEVICE_HPP_

#include "poll.hpp"

namespace mscclpp {

/// A struct representing a pair of 64-bit unsigned integers used as a trigger for the proxy.
///
/// This struct is used as a work element in the concurrent FIFO where multiple device threads can push
/// ProxyTrigger elements and a single host proxy thread consumes these work elements.
///
/// Do not use the most significant bit of @ref snd as it is reserved for memory consistency purposes
struct alignas(16) ProxyTrigger {
  uint64_t fst, snd;
};

/// A concurrent FIFO where multiple device threads can push work elements and a single host proxy thread consumes them.
///
/// The FIFO has a head pointer allocated on the device which starts at 0 and goes up to 2^64-1, which is almost
/// infinity. There are two copies of the tail, one on the device, @ref FifoDeviceHandle::tailReplica, and another on
/// the host, namely, hostTail. The host always has the "true" tail and occasionally pushes it to the copy on the
/// device. Therefore, most of the time, the device has a stale version. The invariants are: tailReplica <= hostTail <=
/// head. The @ref push() function increments head, hostTail is updated in @ref Fifo::pop(), and it occasionally flushes
/// it to tailReplica via @ref Fifo::flushTail().
///
/// Duplicating the tail is a good idea because the FIFO is large enough, and we do not need frequent updates for the
/// tail as there is usually enough space for device threads to push their work into.
///
struct FifoDeviceHandle {
#ifdef __CUDACC__
  /// Push a trigger to the FIFO.
  ///
  /// @param trigger The trigger to push.
  /// @return The new head of the FIFO.
  __forceinline__ __device__ uint64_t push(ProxyTrigger trigger) {
    uint64_t curFifoHead = atomicAdd((unsigned long long int*)this->head, 1);
    // make the last bit intentionally non-zero so that we can safely poll. Don't worry, we will change it back in host
    // side
    trigger.snd ^= ((uint64_t)1 << (uint64_t)63);

    // Only one of two conditions need to be met to proceed. Either the tail has advanced enough or where we need to
    // write to is 0. However, the first condition is faster to check since the tail is flushed periodically anyways but
    // for the second condition we need to read CPU memory.
    // As volatile access is slow, we first check using the bare pointer and then use the volatile pointer if the
    // condition is not met.
    if (curFifoHead >= size + *(this->tailReplica)) {
      OR_POLL_MAYBE_JAILBREAK(curFifoHead >= size + *((volatile uint64_t*)this->tailReplica),
                              *(volatile uint64_t*)&this->triggers[curFifoHead % size] != 0, 1000000);
    }

    ProxyTrigger* triggerPtr = (ProxyTrigger*)&(this->triggers[curFifoHead % size]);
    asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" ::"l"(triggerPtr), "l"(trigger.fst), "l"(trigger.snd));
    return curFifoHead;
  }

  /// Wait until there is a place in the FIFO to push a trigger.
  ///
  /// @param curFifoHead The current head of the FIFO.
  __forceinline__ __device__ void sync(uint64_t curFifoHead) {
    // Same as push but in this case checking the fist condition is probably faster since for tail to be pushed we need
    // to wait for cudaMemcpy to be done.
    OR_POLL_MAYBE_JAILBREAK(*(volatile uint64_t*)&(this->triggers[curFifoHead % size]) != 0,
                            *(volatile uint64_t*)(this->tailReplica) <= curFifoHead, 1000000);
  }
#endif  // __CUDACC__

  /// The FIFO buffer that is allocated on the host via `cudaHostAlloc()`.
  ProxyTrigger* triggers;
  /// Replica of the FIFO tail that is allocated on device.
  uint64_t* tailReplica;
  /// The FIFO head. Allocated on the device and only accessed by the device.
  uint64_t* head;
  /// The FIFO size.
  int size;
};

}  // namespace mscclpp

#endif  // MSCCLPP_FIFO_DEVICE_HPP_
