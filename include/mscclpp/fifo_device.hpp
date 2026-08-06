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

/// Operation that a trigger asks the proxy to perform.
///
/// These are opcodes, not flags: compare one by equality, and never combine two. The encoding
/// enumerates the combinations the device API can produce rather than composing them, so a
/// combination nothing emits cannot be expressed, and a trigger whose type field is unset is not
/// a valid operation.
using TriggerType = uint64_t;
constexpr TriggerType TriggerNone = 0;                   // Not an operation; invalid for ProxyService.
constexpr TriggerType TriggerPut = 1;                    // Transfer data.
constexpr TriggerType TriggerSignal = 2;                 // Signal the remote semaphore.
constexpr TriggerType TriggerFlush = 3;                  // Flush the connection.
constexpr TriggerType TriggerPutWithSignal = 4;          // Transfer data, then signal.
constexpr TriggerType TriggerPutWithSignalAndFlush = 5;  // Transfer data, signal, then flush.
constexpr TriggerType TriggerAccumulate = 6;             // Add a value to remote memory.
// 7 is unassigned.

constexpr unsigned int TriggerBitsSize = 32;
constexpr unsigned int TriggerBitsOffset = 32;
constexpr unsigned int TriggerBitsMemoryId = 9;
constexpr unsigned int TriggerBitsType = 3;
constexpr unsigned int TriggerBitsSemaphoreId = 10;
// The FIFO uses the reserved bit to mark a slot as written, so a trigger must not carry data
// there. See FifoDeviceHandle::push().
constexpr unsigned int TriggerBitsFifoReserved = 1;

static_assert(TriggerAccumulate < (1ULL << TriggerBitsType), "trigger opcodes must fit in the type field");

/// Pair of 64-bit unsigned integers used as a trigger for the proxy.
/// Used as a work element in the concurrent FIFO.
/// Most significant bit of snd is reserved.
union alignas(16) ProxyTrigger {
  struct {
    uint64_t fst;
    uint64_t snd;
  };
  // The summation of number of bits must be 128 or less.
  struct {
    // First 64 bits: value[0]
    uint64_t size : TriggerBitsSize;
    uint64_t srcOffset : TriggerBitsOffset;
    uint64_t : (64 - TriggerBitsSize - TriggerBitsOffset);  // ensure 64-bit alignment
    // Second 64 bits: value[1]
    uint64_t dstOffset : TriggerBitsOffset;
    uint64_t srcMemoryId : TriggerBitsMemoryId;
    uint64_t dstMemoryId : TriggerBitsMemoryId;
    uint64_t type : TriggerBitsType;
    uint64_t semaphoreId : TriggerBitsSemaphoreId;
    // The fields above use 63 bits; the commit bit occupies bit 63. Do not insert a zero-width
    // bitfield here: C++ uses it to start a new allocation unit, moving reserved out of snd.
    uint64_t reserved : TriggerBitsFifoReserved;
  } fields;

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Default constructor.
  MSCCLPP_INLINE ProxyTrigger() = default;

  /// Constructor.
  /// @param type The type of the trigger.
  /// @param dstId The destination ID of memory region.
  /// @param dstOffset The offset into the destination memory region.
  /// @param srcId The source ID of memory region.
  /// @param srcOffset The offset into the source memory region.
  /// @param bytes The bytes of the transfer.
  /// @param semaphoreId The ID of the semaphore.
  MSCCLPP_DEVICE_INLINE ProxyTrigger(TriggerType type, uint32_t dstId, uint64_t dstOffset, uint32_t srcId,
                                     uint64_t srcOffset, uint64_t bytes, uint32_t semaphoreId) {
    MSCCLPP_ASSERT_DEVICE(type < (1ULL << TriggerBitsType), "type is too large");
    MSCCLPP_ASSERT_DEVICE(dstId < (1ULL << TriggerBitsMemoryId), "dstId is too large");
    MSCCLPP_ASSERT_DEVICE(dstOffset < (1ULL << TriggerBitsOffset), "dstOffset is too large");
    MSCCLPP_ASSERT_DEVICE(srcId < (1ULL << TriggerBitsMemoryId), "srcId is too large");
    MSCCLPP_ASSERT_DEVICE(srcOffset < (1ULL << TriggerBitsOffset), "srcOffset is too large");
    MSCCLPP_ASSERT_DEVICE(bytes < (1ULL << TriggerBitsSize), "bytes is too large");
    MSCCLPP_ASSERT_DEVICE(semaphoreId < (1ULL << TriggerBitsSemaphoreId), "semaphoreId is too large");
    constexpr uint64_t maskSize = (1ULL << TriggerBitsSize) - 1;
    constexpr uint64_t maskSrcOffset = (1ULL << TriggerBitsOffset) - 1;
    constexpr uint64_t maskDstOffset = (1ULL << TriggerBitsOffset) - 1;
    constexpr uint64_t maskSrcMemoryId = (1ULL << TriggerBitsMemoryId) - 1;
    constexpr uint64_t maskDstMemoryId = (1ULL << TriggerBitsMemoryId) - 1;
    constexpr uint64_t maskType = (1ULL << TriggerBitsType) - 1;
    constexpr uint64_t maskSemaphoreId = (1ULL << TriggerBitsSemaphoreId) - 1;
    fst = (((srcOffset & maskSrcOffset) << TriggerBitsSize) + (bytes & maskSize));
    snd = (((((((((semaphoreId & maskSemaphoreId) << TriggerBitsType) + ((uint64_t)type & maskType))
                << TriggerBitsMemoryId) +
               (dstId & maskDstMemoryId))
              << TriggerBitsMemoryId) +
             (srcId & maskSrcMemoryId))
            << TriggerBitsOffset) +
           (dstOffset & maskDstOffset));
  }
#endif  // defined(MSCCLPP_DEVICE_COMPILE)
};

static_assert(sizeof(ProxyTrigger) == 16, "ProxyTrigger must be exactly two 64-bit words");

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
  MSCCLPP_DEVICE_INLINE uint64_t push(ProxyTrigger trigger, int64_t maxSpinCount = 1000000) {
    uint64_t prevHead = atomicFetchAdd<uint64_t, scopeDevice>(head, 1, memoryOrderRelaxed);

    // Wait until the slot's previous occupant has been consumed. Lap parity identifies a stale
    // trigger but does not prevent overwriting a live one; this does.
    if (prevHead >= size + *tailCache) {
      sync(prevHead - size, maxSpinCount);
    }

    // Commit bit: the parity of the lap this slot is on, so that the value left by the previous
    // lap reads as stale. Lap 0 writes 1, which makes a zero-initialized buffer read as empty.
    trigger.fields.reserved = ((prevHead >> sizeShift) & 1ULL) ^ 1ULL;

    ProxyTrigger* triggerPtr = &(triggers[prevHead & sizeMask]);

    // snd is the commit word: a consumer that observes this lap's parity must also observe the
    // payload that goes with it.
#if defined(MSCCLPP_DEVICE_CUDA)
    // One 128-bit store publishes both words together, so no ordering is needed between them.
    // The proxy already relies on this: a torn store would let it read a new fst against the
    // previous lap's snd, and dispatch on a stale semaphoreId.
    //
    // sm_80 used __threadfence_system() plus a relaxed store here, which was once faster. On
    // A100 with CUDA 12.9 it is no longer, according to `FifoTest.Fifo`.
    asm volatile("st.global.release.sys.v2.u64 [%0], {%1,%2};" ::"l"(triggerPtr), "l"(trigger.fst), "l"(trigger.snd));
#else   // !defined(MSCCLPP_DEVICE_CUDA)
    // No vector store here, so order the payload ahead of the commit explicitly.
    atomicStore(&(triggerPtr->fst), trigger.fst, memoryOrderRelaxed);
    atomicStore(&(triggerPtr->snd), trigger.snd, memoryOrderRelease);
#endif  // !defined(MSCCLPP_DEVICE_CUDA)

    return prevHead;
  }

  /// Poll whether a specific trigger is popped from the FIFO.
  /// @param fifoHead FIFO head where the trigger was pushed.
  /// @return True if the trigger is popped; false otherwise.
  MSCCLPP_DEVICE_INLINE bool poll(uint64_t fifoHead) {
    uint64_t val;
    if (fifoHead < (val = atomicLoad(tail, memoryOrderAcquire))) {
      // Same as in sync(), this may write a stale value to tailCache.
      *tailCache = val;
      return true;
    }
    return false;
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
  /// FIFO size. Always a power of two.
  int size;
  /// size - 1, for mapping a position to a slot.
  uint64_t sizeMask;
  /// log2(size), for extracting the lap from a position.
  uint64_t sizeShift;
};

}  // namespace mscclpp

#endif  // MSCCLPP_FIFO_DEVICE_HPP_
