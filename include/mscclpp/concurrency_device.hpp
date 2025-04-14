// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_CONCURRENCY_DEVICE_HPP_
#define MSCCLPP_CONCURRENCY_DEVICE_HPP_

#include "atomic_device.hpp"
#include "poll_device.hpp"

namespace mscclpp {

/// A device-wide barrier.
struct DeviceSyncer {
 public:
  /// Construct a new DeviceSyncer object.
  DeviceSyncer() = default;

  /// Destroy the DeviceSyncer object.
  ~DeviceSyncer() = default;

#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Synchronize all threads inside a kernel. Guarantee that all previous work of all threads in cooperating blocks is
  /// finished.
  /// @param blockNum The number of blocks that will synchronize.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  MSCCLPP_DEVICE_INLINE void sync(int blockNum, int64_t maxSpinCount = 100000000) {
    int targetCnt = blockNum;
    __syncthreads();
    if (blockNum == 1) return;
    if (threadIdx.x == 0) {
      unsigned int tmp = preFlag_ ^ 1;
      int val = (tmp << 1) - 1;
      targetCnt = val == 1 ? targetCnt : 0;
      atomicFetchAdd<int, scopeDevice>(&count_, val, memoryOrderRelease);
      POLL_MAYBE_JAILBREAK((atomicLoad<int, scopeDevice>(&count_, memoryOrderAcquire) != targetCnt), maxSpinCount);
      preFlag_ = tmp;
    }
    // We need sync here because only a single thread is checking whether
    // the flag is flipped.
    __syncthreads();
  }
#endif  // !defined(MSCCLPP_DEVICE_COMPILE)

 private:
  /// The counter of synchronized blocks.
  int count_;
  /// The flag to indicate whether to increase or decrease @ref flag_.
  unsigned int preFlag_;
};

}  // namespace mscclpp

#endif  // MSCCLPP_CONCURRENCY_DEVICE_HPP_
