#ifndef MSCCLPP_CONCURRENCY_HPP_
#define MSCCLPP_CONCURRENCY_HPP_

#include <stdint.h>

#include <mscclpp/poll.hpp>

namespace mscclpp {
struct DeviceSyncer {
 public:
  DeviceSyncer() = default;
  ~DeviceSyncer() = default;

#ifdef __CUDACC__
  // Synchronize multiple thread blocks inside a kernel. Guarantee that all
  // previous work of all threads in cooperating blocks is finished.
  __forceinline__ __device__ void sync(int blockNum) {
    int maxOldCnt = blockNum - 1;
    __threadfence();
    // Make sure that all threads in this block have done `__threadfence()`
    // before to flip `flag`.
    __syncthreads();
    if (threadIdx.x == 0) {
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
  volatile int flag_;
  int count_;
  int isAdd_;
};
}  // namespace mscclpp
#endif  // MSCCLPP_CONCURRENCY_HPP_
