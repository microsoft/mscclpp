#ifndef MSCCLPPFIFO_H_
#define MSCCLPPFIFO_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum : uint64_t
{
  mscclppData = 0x1,
  mscclppFlag = 0x2,
  mscclppSync = 0x4
} mscclppTriggerType_t;

#define MSCCLPP_BITS_SIZE 32
#define MSCCLPP_BITS_OFFSET 32
#define MSCCLPP_BITS_TYPE 3
#define MSCCLPP_BITS_CONNID 10

// this is the basic structure of each work element in the fifo
// the summation of number of bits must be 128 or less
union alignas(16) mscclppTrigger {
  uint64_t value[2];
  struct
  {
    // first 64 bits: value[0]
    uint64_t dataSize : MSCCLPP_BITS_SIZE;
    uint64_t srcDataOffset : MSCCLPP_BITS_OFFSET;
    uint64_t : (64 - MSCCLPP_BITS_SIZE - MSCCLPP_BITS_OFFSET); // ensure 64-bit alignment
    // second 64 bits: value[1]
    uint64_t dstDataOffset : MSCCLPP_BITS_OFFSET;
    uint64_t connId : MSCCLPP_BITS_CONNID;
    uint64_t type : MSCCLPP_BITS_TYPE;
    uint64_t : (64 - MSCCLPP_BITS_OFFSET - MSCCLPP_BITS_CONNID - MSCCLPP_BITS_TYPE); // ensure 64-bit alignment
  } fields;
};

typedef mscclppTrigger* mscclppTrigger_t;

/* This is a concurrent fifo where multiple device threads can push mscclppTrigger work elements to
 * and a single host proxy thread consumes these work elements. There is a head pointer allocated on device
 * which starts with 0 and goes to 2^64-1 which is almost infinity. There are two copies of tail, one
 * that is on the deivce (triggerFifoTail) and another that is on host (proxyState->fifoTailHost).
 * The host always has the "true" tail and occasionally, pushes it to the tail version.
 * Therefore, most of the time, the device has a stale version. The invariants are:
 * triggerFifoTail <= proxyState->fifoTailHost <= triggerFifoHead.
 * push function increments triggerFifoHead, proxyState->fifoTailHost is updated in proxy.cc:mscclppProxyService
 * and it occasionally flushes it to triggerFifoTail via a cudaMemcpyAsync.
 *
 * Why douplicating the tail is a good idea? The fifo is large engouh and we do not need frequent updates
 * for the tail as there is usually enough space for device threads to push their work into.
 */
struct mscclppConcurrentFifo
{
#ifdef __CUDACC__

  __forceinline__ __device__ uint64_t push(uint64_t type, uint64_t dstDataOffset, uint64_t srcDataOffset,
                                           uint64_t dataSize)
  {
    uint64_t curFifoHead = atomicAdd((unsigned long long int*)this->triggerFifoHead, 1);
    while (curFifoHead >= MSCCLPP_PROXY_FIFO_SIZE + *((volatile uint64_t*)this->triggerFifoTail))
      ;
    while (*(volatile uint64_t*)&this->triggerFifo[curFifoHead % MSCCLPP_PROXY_FIFO_SIZE] != 0)
      ;
    uint64_t* valptr = (uint64_t*)&(this->triggerFifo[curFifoHead % MSCCLPP_PROXY_FIFO_SIZE].value);
    asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" ::"l"(valptr),
                 "l"((srcDataOffset << MSCCLPP_BITS_SIZE) + dataSize),
                 "l"((((type << MSCCLPP_BITS_CONNID) + this->connId) << MSCCLPP_BITS_OFFSET) + dstDataOffset));
    return curFifoHead;
  }

#endif                         // __CUDACC__
  mscclppTrigger* triggerFifo; // Allocate on host via cudaHostAlloc. This space is used for pushing the workelements
  uint64_t* triggerFifoTail;   // Allocated on device. proxyState->fifoTailHost is the true tail on host and pused
                               // occasionally to device
  uint64_t* triggerFifoHead;   // Allocated on device. Only accessed by device
  int connId;
};

#ifdef __cplusplus
} // end extern "C"
#endif

#endif // MSCCLPPFIFO_H_
