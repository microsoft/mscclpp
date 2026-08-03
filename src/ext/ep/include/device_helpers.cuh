// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <cstdint>

#include "config.hpp"
#include "exception.cuh"

#ifndef WARP_SIZE
#if defined(__HIP_PLATFORM_AMD__)
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif
#endif

namespace mscclpp {
namespace ep {

__device__ __forceinline__ void trap() { asm("trap;"); }

__device__ __forceinline__ void memory_fence() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }

__device__ __forceinline__ void syncNamedBarrier(int barrierId, int numThreads) {
  asm volatile("bar.sync %0, %1;" ::"r"(barrierId), "r"(numThreads) : "memory");
}

#if defined(__CUDACC__)
__device__ __forceinline__ void fenceProxyAsyncSharedCta() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void initTmaLoadBarrier(uint64_t* sharedBarrier) {
  const uint32_t barrierAddress = static_cast<uint32_t>(__cvta_generic_to_shared(sharedBarrier));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(barrierAddress));
  fenceProxyAsyncSharedCta();
}

__device__ __forceinline__ void issueTmaLoad(const void* source, void* sharedTile, uint64_t* sharedBarrier,
                                             uint32_t nBytes) {
  const uint32_t tileAddress = static_cast<uint32_t>(__cvta_generic_to_shared(sharedTile));
  const uint32_t barrierAddress = static_cast<uint32_t>(__cvta_generic_to_shared(sharedBarrier));
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];" ::"r"(tileAddress),
      "l"(source), "r"(nBytes), "r"(barrierAddress)
      : "memory");
}

__device__ __forceinline__ void expectTmaLoad(uint64_t* sharedBarrier, uint32_t nBytes) {
  const uint32_t barrierAddress = static_cast<uint32_t>(__cvta_generic_to_shared(sharedBarrier));
  [[maybe_unused]] uint64_t state;
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;"
               : "=l"(state)
               : "r"(barrierAddress), "r"(nBytes));
}

__device__ __forceinline__ void issueTmaLoadAndExpect(const void* source, void* sharedTile, uint64_t* sharedBarrier,
                                                      uint32_t nBytes) {
  issueTmaLoad(source, sharedTile, sharedBarrier, nBytes);
  expectTmaLoad(sharedBarrier, nBytes);
}

__device__ __forceinline__ void waitTmaLoad(uint64_t* sharedBarrier, uint32_t& phase) {
  const uint32_t barrierAddress = static_cast<uint32_t>(__cvta_generic_to_shared(sharedBarrier));
  uint32_t done = 0;
  while (!done) {
    asm volatile(
        "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;"
        " selp.u32 %0, 1, 0, p; }"
        : "=r"(done)
        : "r"(barrierAddress), "r"(phase));
  }
  phase ^= 1;
}

__device__ __forceinline__ void issueTmaStore(void* destination, void* sharedTile, uint32_t nBytes) {
  const uint32_t tileAddress = static_cast<uint32_t>(__cvta_generic_to_shared(sharedTile));
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(destination), "r"(tileAddress),
               "r"(nBytes)
               : "memory");
  asm volatile("cp.async.bulk.commit_group;");
}

template <int NumPendingGroups = 0>
__device__ __forceinline__ void waitBulkGroupRead() {
  // Wait until at most NumPendingGroups committed bulk groups may still read shared memory.
  asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(NumPendingGroups) : "memory");
}

__device__ __forceinline__ void waitBulkGroup() {
  // Wait for every committed bulk group to complete.
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}
#endif

// `st.global.L1::no_allocate` will be translated into `ST.E.NA.[width]` in SASS
#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define ST_NA_FUNC "st.global.L1::no_allocate"
#else
#define ST_NA_FUNC "st.global"
#endif

__device__ __forceinline__ void st_na_global(const float* ptr, const float& value) {
  asm volatile(ST_NA_FUNC ".f32 [%0], %1;" ::"l"(ptr), "f"(value));
}

__device__ __forceinline__ void st_na_global(const int4* ptr, const int4& value) {
  asm volatile(ST_NA_FUNC ".v4.s32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z),
               "r"(value.w));
}

__forceinline__ __device__ void get_channel_task_range(int num_tokens, int num_sms, int sm_id, int& token_start_idx,
                                                       int& token_end_idx) {
  int num_tokens_per_sm = configCellDiv(num_tokens, num_sms);
  token_start_idx = min(num_tokens_per_sm * sm_id, num_tokens);
  token_end_idx = min(token_start_idx + num_tokens_per_sm, num_tokens);
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ dtype_b_t pack2(const dtype_a_t& x, const dtype_a_t& y) {
  EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
  dtype_b_t packed;
  auto unpacked_ptr = reinterpret_cast<dtype_a_t*>(&packed);
  unpacked_ptr[0] = x, unpacked_ptr[1] = y;
  return packed;
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ void unpack2(const dtype_b_t& packed, dtype_a_t& x, dtype_a_t& y) {
  EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
  auto unpacked_ptr = reinterpret_cast<const dtype_a_t*>(&packed);
  x = unpacked_ptr[0], y = unpacked_ptr[1];
}

template <typename T>
__device__ __forceinline__ T warpBroadcast(T value, int sourceLane) {
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

__forceinline__ __device__ int warp_reduce_sum(int value) {
  value += __shfl_xor_sync(0xffffffff, value, 16);
  value += __shfl_xor_sync(0xffffffff, value, 8);
  value += __shfl_xor_sync(0xffffffff, value, 4);
  value += __shfl_xor_sync(0xffffffff, value, 2);
  value += __shfl_xor_sync(0xffffffff, value, 1);
  return value;
}

__forceinline__ __device__ int warpInclusiveSum(int value, int laneId) {
#pragma unroll
  for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
    const int previous = __shfl_up_sync(0xffffffff, value, offset);
    if (laneId >= offset) value += previous;
  }
  return value;
}

__forceinline__ __device__ bool isFirstLaneForRank(int rank, int laneId) {
  const unsigned matchMask = __match_any_sync(0xffffffff, rank);
  return (__ffs(matchMask) - 1) == laneId;
}

__forceinline__ __device__ int get_lane_id() {
  int lane_id;
  asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
  return lane_id;
}

}  // namespace ep
}  // namespace mscclpp
