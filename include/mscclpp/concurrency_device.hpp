// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_CONCURRENCY_DEVICE_HPP_
#define MSCCLPP_CONCURRENCY_DEVICE_HPP_

#include "atomic_device.hpp"
#include "poll_device.hpp"

#define NUM_DEVICE_SYNCER_COUNTER 3

namespace mscclpp {

/// A device-wide barrier.
struct DeviceSyncer {
 public:
  /// Construct a new DeviceSyncer object.
  MSCCLPP_INLINE DeviceSyncer() = default;

  /// Destroy the DeviceSyncer object.
  MSCCLPP_INLINE ~DeviceSyncer() = default;

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Synchronize all threads inside a kernel. Guarantee that all previous work of all threads in cooperating blocks is
  /// finished.
  /// @param blockNum The number of blocks that will synchronize.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  MSCCLPP_DEVICE_INLINE void sync(int blockNum, int64_t maxSpinCount = 100000000) {
    unsigned int targetCnt = blockNum;
    __syncthreads();
    if (blockNum == 1) return;
    if (threadIdx.x == 0) {
      unsigned int tmp = (preFlag_ + 1) % NUM_DEVICE_SYNCER_COUNTER;
      unsigned int next = (tmp + 1) % NUM_DEVICE_SYNCER_COUNTER;
      unsigned int* count = &count_[tmp];
      count_[next] = 0;
      atomicFetchAdd<unsigned int, scopeDevice>(count, 1U, memoryOrderRelease);
      POLL_MAYBE_JAILBREAK((atomicLoad<unsigned int, scopeDevice>(count, memoryOrderAcquire) != targetCnt),
                           maxSpinCount);
      preFlag_ = tmp;
    }
    // We need sync here because only a single thread is checking whether
    // the flag is flipped.
    __syncthreads();
  }
#endif  // !defined(MSCCLPP_DEVICE_COMPILE)

 private:
  /// The counter of synchronized blocks.
  unsigned int count_[NUM_DEVICE_SYNCER_COUNTER];
  /// The flag to indicate whether to increase or decrease @ref flag_.
  unsigned int preFlag_;
};

struct DeviceSemaphore {
 public:
  /// Construct a new DeviceSemaphore object.
  MSCCLPP_INLINE DeviceSemaphore() = default;

  /// Destroy the DeviceSemaphore object.
  MSCCLPP_INLINE ~DeviceSemaphore() = default;

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// set the semaphore value.
  /// @param value The value to set.
  MSCCLPP_DEVICE_INLINE void set(int value) { atomicStore<int, scopeDevice>(&semaphore_, value, memoryOrderRelease); }

  /// Acquire the semaphore.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  MSCCLPP_DEVICE_INLINE void acquire(int maxSpinCount = -1) {
    if (atomicFetchAdd<int, scopeDevice>(&semaphore_, -1, memoryOrderAcquire) <= 0) {
      POLL_MAYBE_JAILBREAK((atomicLoad<int, scopeDevice>(&semaphore_, memoryOrderAcquire) < 0), maxSpinCount);
    }
  }

  /// Release the semaphore.
  MSCCLPP_DEVICE_INLINE void release() { atomicFetchAdd<int, scopeDevice>(&semaphore_, 1, memoryOrderRelease); }
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

 private:
  /// The semaphore value.
  int semaphore_;
};

}  // namespace mscclpp

#endif  // MSCCLPP_CONCURRENCY_DEVICE_HPP_
