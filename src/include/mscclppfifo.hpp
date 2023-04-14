#ifndef MSCCLPPFIFO_HPP_
#define MSCCLPPFIFO_HPP_

#include <stdint.h>
#include <functional>

namespace mscclpp {

struct alignas(16) ProxyTrigger {
  uint64_t fst, snd;
};

/* This is a concurrent fifo where multiple device threads can push mscclppTrigger work elements to
 * and a single host proxy thread consumes these work elements. There is a head pointer allocated on device
 * which starts with 0 and goes to 2^64-1 which is almost infinity. There are two copies of tail, one
 * that is on the deivce (triggerFifoTail) and another that is on host (proxyState->fifoTailHost).
 * The host always has the "true" tail and occasionally, pushes it to the copy on the device.
 * Therefore, most of the time, the device has a stale version. The invariants are:
 * triggerFifoTail <= proxyState->fifoTailHost <= triggerFifoHead.
 * push() function increments triggerFifoHead, proxyState->fifoTailHost is updated in proxy.cc:mscclppProxyService
 * and it occasionally flushes it to triggerFifoTail via a cudaMemcpyAsync.
 *
 * Why duplicating the tail is a good idea? The fifo is large engouh and we do not need frequent updates
 * for the tail as there is usually enough space for device threads to push their work into.
 */
struct ProxyFifo {
#ifdef __CUDACC__
  __forceinline__ __device__ uint64_t push(ProxyTrigger element)
  {
    uint64_t curFifoHead = atomicAdd((unsigned long long int*)this->triggerFifoHead, 1);
    while (curFifoHead >= MSCCLPP_PROXY_FIFO_SIZE + *((volatile uint64_t*)this->triggerFifoTail))
      ;
    while (*(volatile uint64_t*)&this->triggerFifo[curFifoHead % MSCCLPP_PROXY_FIFO_SIZE] != 0)
      ;
    uint64_t* valptr = (uint64_t*)&(this->triggerFifo[curFifoHead % MSCCLPP_PROXY_FIFO_SIZE].value);
    asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" ::"l"(valptr),
                 "l"(element.value[0]), "l"(element.value[1]));
    return curFifoHead;
  }
#endif // __CUDACC__

  void startProxyThread(std::function<void(ProxyTrigger)> handler);
  void stopProxyThread();

  ProxyTrigger* triggerFifo; // Allocate on host via cudaHostAlloc. This space is used for pushing the workelements
  uint64_t* triggerFifoTail;   // Allocated on device. proxyState->fifoTailHost is the true tail on host and pused
                               // occasionally to device
  uint64_t* triggerFifoHead;   // Allocated on device. Only accessed by device
};

} // namespace mscclpp

#endif // MSCCLPPFIFO_H_
