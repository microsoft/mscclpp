// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_FIFO_HPP_
#define MSCCLPP_FIFO_HPP_

#include <cstdint>
#include <functional>
#include <memory>
#include <mscclpp/poll.hpp>

#define MSCCLPP_PROXY_FIFO_SIZE 128

namespace mscclpp {

/// A struct representing a pair of 64-bit unsigned integers used as a trigger for the proxy.
///
/// This struct is used as a work element in the concurrent FIFO where multiple device threads can push
/// ProxyTrigger elements and a single host proxy thread consumes these work elements.
///
struct alignas(16) ProxyTrigger {
  uint64_t fst, snd;
};

/// A concurrent FIFO where multiple device threads can push work elements and a single host proxy thread consumes them.
///
/// The FIFO has a head pointer allocated on the device which starts at 0 and goes up to 2^64-1, which is almost
/// infinity. There are two copies of the tail, one on the device, @ref DeviceProxyFifo::tailReplica, and another on the
/// host, namely, hostTail. The host always has the "true" tail and occasionally pushes it to the copy on the device.
/// Therefore, most of the time, the device has a stale version. The invariants are: tailReplica <= hostTail <= head.
/// The @ref push() function increments head, hostTail is updated in @ref HostProxyFifo::pop(), and it occasionally
/// flushes it to tailReplica via @ref HostProxyFifo::flushTail().
///
/// Duplicating the tail is a good idea because the FIFO is large enough, and we do not need frequent updates for the
/// tail as there is usually enough space for device threads to push their work into.
///
struct DeviceProxyFifo {
#ifdef __CUDACC__
  /// Push a trigger to the FIFO.
  ///
  /// @param trigger The trigger to push.
  /// @return The new head of the FIFO.
  __forceinline__ __device__ uint64_t push(ProxyTrigger trigger) {
    uint64_t curFifoHead = atomicAdd((unsigned long long int*)this->head, 1);

    // Only one of two conditions need to be met to proceed. Either the tail has advanced enough or where we need to
    // write to is 0. However, the first condition is faster to check since the tail is flushed periodically anyways but
    // for the second condition we need to read CPU memory.
    // As volatile access is slow, we first check using the bare pointer and then use the volatile pointer if the
    // condition is not met.
    if (curFifoHead >= MSCCLPP_PROXY_FIFO_SIZE + *(this->tailReplica)) {
      OR_POLL_MAYBE_JAILBREAK(curFifoHead >= MSCCLPP_PROXY_FIFO_SIZE + *((volatile uint64_t*)this->tailReplica),
                              *(volatile uint64_t*)&this->triggers[curFifoHead % MSCCLPP_PROXY_FIFO_SIZE] != 0,
                              1000000);
    }

    ProxyTrigger* triggerPtr = (ProxyTrigger*)&(this->triggers[curFifoHead % MSCCLPP_PROXY_FIFO_SIZE]);
    asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" ::"l"(triggerPtr), "l"(trigger.fst), "l"(trigger.snd));
    return curFifoHead;
  }

  /// Wait until there is a place in the FIFO to push a trigger.
  ///
  /// @param curFifoHead The current head of the FIFO.
  __forceinline__ __device__ void sync(uint64_t curFifoHead) {
    // Same as push but in this case checking the fist condition is probably faster since for tail to be pushed we need
    // to wait for cudaMemcpy to be done.
    OR_POLL_MAYBE_JAILBREAK(*(volatile uint64_t*)&(this->triggers[curFifoHead % MSCCLPP_PROXY_FIFO_SIZE]) != 0,
                            *(volatile uint64_t*)(this->tailReplica) <= curFifoHead, 1000000);
  }
#endif  // __CUDACC__

  /// The FIFO buffer that is allocated on the host via `cudaHostAlloc()`.
  ProxyTrigger* triggers;
  /// Replica of the FIFO tail that is allocated on device.
  uint64_t* tailReplica;
  /// The FIFO head. Allocated on the device and only accessed by the device.
  uint64_t* head;
};

/// A class representing a host proxy FIFO that can consume work elements pushed by device threads.
class HostProxyFifo {
 public:
  /// Constructs a new @ref HostProxyFifo object.
  HostProxyFifo();

  /// Destroys the @ref HostProxyFifo object.
  ~HostProxyFifo();

  /// Polls the FIFO for a trigger.
  ///
  /// @param trigger A pointer to the trigger to be filled.
  void poll(ProxyTrigger* trigger);

  /// Pops a trigger from the FIFO.
  void pop();

  /// Flushes the tail of the FIFO.
  ///
  /// @param sync If true, waits for the flush to complete before returning.
  void flushTail(bool sync = false);

  /// Returns a @ref DeviceProxyFifo object representing the device FIFO.
  ///
  /// @return A @ref DeviceProxyFifo object representing the device FIFO.
  DeviceProxyFifo deviceFifo();

 private:
  struct Impl;
  std::unique_ptr<Impl> pimpl;
};

}  // namespace mscclpp

#endif  // MSCCLPP_FIFO_HPP_
