// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#ifndef MSCCLPP_EP_COMMON_DEVICE_HELPERS_CUH_
#define MSCCLPP_EP_COMMON_DEVICE_HELPERS_CUH_

#include "exception.hpp"

#ifndef WARP_SIZE
#if defined(__HIP_PLATFORM_AMD__)
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif
#endif

namespace mscclpp {
namespace ep {

MSCCLPP_DEVICE_INLINE void syncNamedBarrier(int barrierId, int numThreads) {
  asm volatile("bar.sync %0, %1;" ::"r"(barrierId), "r"(numThreads) : "memory");
}

template <typename dtype_a_t, typename dtype_b_t>
MSCCLPP_DEVICE_INLINE dtype_b_t pack2(const dtype_a_t& x, const dtype_a_t& y) {
  EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
  dtype_b_t packed;
  auto unpacked_ptr = reinterpret_cast<dtype_a_t*>(&packed);
  unpacked_ptr[0] = x, unpacked_ptr[1] = y;
  return packed;
}

template <typename dtype_a_t, typename dtype_b_t>
MSCCLPP_DEVICE_INLINE void unpack2(const dtype_b_t& packed, dtype_a_t& x, dtype_a_t& y) {
  EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
  auto unpacked_ptr = reinterpret_cast<const dtype_a_t*>(&packed);
  x = unpacked_ptr[0], y = unpacked_ptr[1];
}

template <typename T>
MSCCLPP_DEVICE_INLINE T warpBroadcast(T value, int sourceLane) {
  EP_STATIC_ASSERT(sizeof(T) % sizeof(int) == 0, "");
  const auto* sourceValues = reinterpret_cast<const int*>(&value);
  T result;
  auto* resultValues = reinterpret_cast<int*>(&result);
#pragma unroll
  for (int i = 0; i < sizeof(T) / sizeof(int); ++i) {
    resultValues[i] = __shfl_sync(0xffffffff, sourceValues[i], sourceLane);
  }
  return result;
}

MSCCLPP_DEVICE_INLINE int warpInclusiveSum(int value, int laneId) {
#pragma unroll
  for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
    const int previous = __shfl_up_sync(0xffffffff, value, offset);
    if (laneId >= offset) value += previous;
  }
  return value;
}

MSCCLPP_DEVICE_INLINE bool isFirstLaneForRank(int rank, int laneId) {
  const unsigned matchMask = __match_any_sync(0xffffffff, rank);
  return (__ffs(matchMask) - 1) == laneId;
}

MSCCLPP_DEVICE_INLINE int getLaneId() {
  int laneId;
  asm("mov.s32 %0, %laneid;" : "=r"(laneId));
  return laneId;
}

}  // namespace ep
}  // namespace mscclpp

#endif  // MSCCLPP_EP_COMMON_DEVICE_HELPERS_CUH_
