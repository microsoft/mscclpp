// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_CONCURRENCY_HPP_
#define MSCCLPP_CONCURRENCY_HPP_

#include <stdint.h>

#include <mscclpp/poll.hpp>

namespace mscclpp {

/// A device-wide barrier.
struct DeviceSyncer {
 public:
  /// Construct a new DeviceSyncer object.
  DeviceSyncer() = default;

  /// Destroy the DeviceSyncer object.
  ~DeviceSyncer() = default;

#ifdef __CUDACC__
  /// Synchronize all threads inside a kernel. Guarantee that all previous work of all threads in cooperating blocks is
  /// finished.
  /// @param blockNum The number of blocks that will synchronize.
  __forceinline__ __device__ void sync(int blockNum) {
    int maxOldCnt = blockNum - 1;
    __syncthreads();
    if (blockNum == 1) return;
    if (threadIdx.x == 0) {
      // Need a `__threadfence()` before to flip `flag`.
      __threadfence();
      int tmpIsAdd = isAdd_ ^ 1;
      if (tmpIsAdd) {
        if (atomicAdd(&count_, 1) == maxOldCnt) {
          flag_ = 1;
        }
        POLL_MAYBE_JAILBREAK(!flag_, 1000000000);
      } else {
        if (atomicSub(&count_, 1) == 1) {
          flag_ = 0;
        }
        POLL_MAYBE_JAILBREAK(flag_, 1000000000);
      }
      isAdd_ = tmpIsAdd;
    }
    // We need sync here because only a single thread is checking whether
    // the flag is flipped.
    __syncthreads();
  }
#endif

 private:
  /// The flag to indicate whether the barrier is reached by the latest thread.
  volatile int flag_;
  /// The counter of synchronized blocks.
  int count_;
  /// The flag to indicate whether to increase or decrease @ref count_.
  int isAdd_;
};

}  // namespace mscclpp

#endif  // MSCCLPP_CONCURRENCY_HPP_
