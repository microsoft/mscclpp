#ifndef MSCCLPP_CONCURRENCY_HPP_
#define MSCCLPP_CONCURRENCY_HPP_

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
    __syncthreads();
    if (threadIdx.x == 0) {
      // Need a `__threadfence()` before to flip `flag`.
      __threadfence();
      int tmpIsAdd = isAdd_ ^ 1;
      if (tmpIsAdd) {
        if (atomicAdd(&count_, 1) == maxOldCnt) {
          flag_ = 1;
        }
        while (!flag_) {
        }
      } else {
        if (atomicSub(&count_, 1) == 1) {
          flag_ = 0;
        }
        while (flag_) {
        }
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
