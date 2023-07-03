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

struct alignas(16) ProxyTrigger {
  uint64_t fst, snd;
};

/* This is a concurrent fifo where multiple device threads can push mscclppTrigger work elements to
 * and a single host proxy thread consumes these work elements. There is a head pointer allocated on device
 * which starts with 0 and goes to 2^64-1 which is almost infinity. There are two copies of tail, one
 * that is on the deivce (tailReplica) and another that is on host (proxyState->fifoTailHost).
 * The host always has the "true" tail and occasionally, pushes it to the copy on the device.
 * Therefore, most of the time, the device has a stale version. The invariants are:
 * tailReplica <= proxyState->fifoTailHost <= head.
 * push() function increments head, proxyState->fifoTailHost is updated in proxy.cc:mscclppProxyService
 * and it occasionally flushes it to tailReplica via a cudaMemcpyAsync.
 *
 * Why duplicating the tail is a good idea? The fifo is large engouh and we do not need frequent updates
 * for the tail as there is usually enough space for device threads to push their work into.
 */
struct DeviceProxyFifo {
#ifdef __CUDACC__
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

  __forceinline__ __device__ void sync(uint64_t curFifoHead) {
    // same as push but in this case checking the fist condition is probably faster since for tail to be pushed we need
    // to wait for cudaMemcpy to be done.
    OR_POLL_MAYBE_JAILBREAK(*(volatile uint64_t*)&(this->triggers[curFifoHead % MSCCLPP_PROXY_FIFO_SIZE]) != 0,
                            *(volatile uint64_t*)(this->tailReplica) <= curFifoHead, 1000000);
  }
#endif  // __CUDACC__

  ProxyTrigger* triggers;  // Allocate on host via cudaHostAlloc. This space is used for pushing the workelements
  uint64_t* tailReplica;   // Allocated on device. proxyState->fifoTailHost is the true tail on host and pused
                           // occasionally to device
  uint64_t* head;          // Allocated on device. Only accessed by device
};

class HostProxyFifo {
 public:
  HostProxyFifo();

  ~HostProxyFifo();

  void poll(ProxyTrigger* trigger);

  void pop();

  void flushTail(bool sync = false);

  DeviceProxyFifo deviceFifo();

 private:
  struct Impl;
  std::unique_ptr<Impl> pimpl;
};

}  // namespace mscclpp

#endif  // MSCCLPP_FIFO_HPP_
