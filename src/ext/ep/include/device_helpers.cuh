// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <cstdint>
#include <type_traits>

#include "exception.cuh"

#ifndef WARP_SIZE
#if defined(__HIP_PLATFORM_AMD__)
#define WARP_SIZE 64
#else
#define WARP_SIZE 32
#endif
#endif

#define UNROLLED_WARP_COPY(UNROLL_FACTOR, LANE_ID, N, DST, SRC, LD_FUNC, ST_FUNC)                        \
  {                                                                                                      \
    constexpr int LoopStride = WARP_SIZE * (UNROLL_FACTOR);                                              \
    typename std::remove_reference<decltype(LD_FUNC((SRC) + 0))>::type unrolled_values[(UNROLL_FACTOR)]; \
    auto __src = (SRC);                                                                                  \
    auto __dst = (DST);                                                                                  \
    for (int __i = (LANE_ID); __i < ((N) / LoopStride) * LoopStride; __i += LoopStride) {                \
      _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j) unrolled_values[__j] =           \
          LD_FUNC(__src + __i + __j * WARP_SIZE);                                                        \
      _Pragma("unroll") for (int __j = 0; __j < (UNROLL_FACTOR); ++__j)                                  \
          ST_FUNC(__dst + __i + __j * WARP_SIZE, unrolled_values[__j]);                                  \
    }                                                                                                    \
    for (int __i = ((N) / LoopStride) * LoopStride + (LANE_ID); __i < (N); __i += WARP_SIZE)             \
      ST_FUNC(__dst + __i, LD_FUNC(__src + __i));                                                        \
  }

namespace mscclpp {
namespace ep {

template <int Bytes>
struct VecInt {};
template <>
struct VecInt<1> {
  using vec_t = int8_t;
};
template <>
struct VecInt<2> {
  using vec_t = int16_t;
};
template <>
struct VecInt<4> {
  using vec_t = int;
};
template <>
struct VecInt<8> {
  using vec_t = int64_t;
};
template <>
struct VecInt<16> {
  using vec_t = int4;
};

__device__ __forceinline__ void trap() { asm("trap;"); }

__device__ __forceinline__ void memory_fence() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }

__device__ __forceinline__ void memory_fence_gpu() { asm volatile("fence.acq_rel.gpu;" ::: "memory"); }

__device__ __forceinline__ void memory_fence_cta() { asm volatile("fence.acq_rel.cta;" ::: "memory"); }

__device__ __forceinline__ void syncNamedBarrier(int barrierId, int numThreads) {
  asm volatile("bar.sync %0, %1;" ::"r"(barrierId), "r"(numThreads) : "memory");
}

#if defined(__CUDACC__)
__device__ __forceinline__ void fenceProxyAsyncSharedCta() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ float reciprocalApproximateFtz(float value) {
  float result;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

__device__ __forceinline__ void initTmaLoadBarrier(uint64_t *sharedBarrier) {
  const uint32_t barrierAddress = static_cast<uint32_t>(__cvta_generic_to_shared(sharedBarrier));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(barrierAddress));
  fenceProxyAsyncSharedCta();
}

__device__ __forceinline__ void issueTmaLoad(const void *source, void *sharedTile, uint64_t *sharedBarrier,
                                             uint32_t nBytes) {
  const uint32_t tileAddress = static_cast<uint32_t>(__cvta_generic_to_shared(sharedTile));
  const uint32_t barrierAddress = static_cast<uint32_t>(__cvta_generic_to_shared(sharedBarrier));
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];" ::"r"(tileAddress),
      "l"(source), "r"(nBytes), "r"(barrierAddress)
      : "memory");
}

__device__ __forceinline__ void expectTmaLoad(uint64_t *sharedBarrier, uint32_t nBytes) {
  const uint32_t barrierAddress = static_cast<uint32_t>(__cvta_generic_to_shared(sharedBarrier));
  [[maybe_unused]] uint64_t state;
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;"
               : "=l"(state)
               : "r"(barrierAddress), "r"(nBytes));
}

__device__ __forceinline__ void issueTmaLoadAndExpect(const void *source, void *sharedTile, uint64_t *sharedBarrier,
                                                      uint32_t nBytes) {
  issueTmaLoad(source, sharedTile, sharedBarrier, nBytes);
  expectTmaLoad(sharedBarrier, nBytes);
}

__device__ __forceinline__ void waitTmaLoad(uint64_t *sharedBarrier, uint32_t &phase) {
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

__device__ __forceinline__ void issueTmaStore(void *destination, void *sharedTile, uint32_t nBytes) {
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

__device__ __forceinline__ void st_relaxed_sys_global(const int *ptr, int val) {
  asm volatile("st.relaxed.sys.global.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
}

__device__ __forceinline__ void st_release_sys_global(const int *ptr, int val) {
  asm volatile("st.release.sys.global.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
}

__device__ __forceinline__ void st_release_cta(const int *ptr, int val) {
  asm volatile("st.release.cta.s32 [%0], %1;" ::"l"(ptr), "r"(val) : "memory");
}

__device__ __forceinline__ int ld_acquire_sys_global(const int *ptr) {
  int ret;
  asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

__device__ __forceinline__ uint64_t ld_acquire_sys_global(const uint64_t *ptr) {
  uint64_t ret;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
  return ret;
}

__device__ __forceinline__ int64_t ld_acquire_sys_global(const int64_t *ptr) {
  int64_t ret;
  asm volatile("ld.acquire.sys.global.s64 %0, [%1];" : "=l"(ret) : "l"(ptr));
  return ret;
}

__device__ __forceinline__ int ld_acquire_global(const int *ptr) {
  int ret;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

__device__ __forceinline__ int atomic_add_release_sys_global(const int *ptr, int value) {
  int ret;
  asm volatile("atom.add.release.sys.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
  return ret;
}

__device__ __forceinline__ int atomic_add_release_global(const int *ptr, int value) {
  int ret;
  asm volatile("atom.add.release.gpu.global.s32 %0, [%1], %2;" : "=r"(ret) : "l"(ptr), "r"(value));
  return ret;
}

__device__ __forceinline__ int ld_acquire_cta(const int *ptr) {
  int ret;
  asm volatile("ld.acquire.cta.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

__device__ __forceinline__ uint8_t ld_na_relaxed(const uint8_t *ptr) {
  uint16_t ret;
  asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b8 %0, [%1];" : "=h"(ret) : "l"(ptr));
  return static_cast<uint8_t>(ret);
}

__device__ __forceinline__ uint16_t ld_na_relaxed(const uint16_t *ptr) {
  uint16_t ret;
  asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b16 %0, [%1];" : "=h"(ret) : "l"(ptr));
  return ret;
}

__device__ __forceinline__ uint32_t ld_na_relaxed(const uint32_t *ptr) {
  uint32_t ret;
  asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

__device__ __forceinline__ uint64_t ld_na_relaxed(const uint64_t *ptr) {
  uint64_t ret;
  asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b64 %0, [%1];" : "=l"(ret) : "l"(ptr));
  return ret;
}

__device__ __forceinline__ int ld_volatile_global(const int *ptr) {
  int ret;
  asm volatile("ld.volatile.global.s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

__device__ __forceinline__ float ld_volatile_global(const float *ptr) {
  float ret;
  asm volatile("ld.volatile.global.f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
  return ret;
}

__device__ __forceinline__ int64_t ld_volatile_global(const int64_t *ptr) {
  int64_t ret;
  asm volatile("ld.volatile.global.s64 %0, [%1];" : "=l"(ret) : "l"(ptr));
  return ret;
}

__device__ __forceinline__ int64_t ld_volatile_global(const uint64_t *ptr) {
  int64_t ret;
  asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(ret) : "l"(ptr));
  return ret;
}

#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define LD_NC_FUNC "ld.global.nc.L1::no_allocate.L2::256B"
#else
#define LD_NC_FUNC "ld.volatile.global"
#endif

// `ld.global.nc.L1::no_allocate` will be translated into `LDG.E.NA.[width].CONSTANT` in SASS
template <typename dtype_t>
__device__ __forceinline__ dtype_t ld_nc_global(const dtype_t *ptr) {
  auto ret = ld_nc_global(reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t *>(ptr));
  return *reinterpret_cast<dtype_t *>(&ret);
}

template <>
__device__ __forceinline__ uint8_t ld_nc_global(const uint8_t *ptr) {
  uint16_t ret;
  // NOTES: we must use `uint16_t` as inline ASM does not support 8-bit constraint letter (`h` below means unsigned
  // 16-bit)
  asm volatile(LD_NC_FUNC ".u8 %0, [%1];" : "=h"(ret) : "l"(ptr));
  return static_cast<uint8_t>(ret);
}

template <>
__device__ __forceinline__ int ld_nc_global(const int *ptr) {
  int ret;
  asm volatile(LD_NC_FUNC ".s32 %0, [%1];" : "=r"(ret) : "l"(ptr));
  return ret;
}

template <>
__device__ __forceinline__ int64_t ld_nc_global(const int64_t *ptr) {
  int64_t ret;
  asm volatile(LD_NC_FUNC ".s64 %0, [%1];" : "=l"(ret) : "l"(ptr));
  return ret;
}

template <>
__device__ __forceinline__ float ld_nc_global(const float *ptr) {
  float ret;
  asm volatile(LD_NC_FUNC ".f32 %0, [%1];" : "=f"(ret) : "l"(ptr));
  return ret;
}

template <>
__device__ __forceinline__ int2 ld_nc_global(const int2 *ptr) {
  int2 ret;
  asm volatile(LD_NC_FUNC ".v2.s32 {%0, %1}, [%2];" : "=r"(ret.x), "=r"(ret.y) : "l"(ptr));
  return ret;
}

template <>
__device__ __forceinline__ int4 ld_nc_global(const int4 *ptr) {
  int4 ret;
  asm volatile(LD_NC_FUNC ".v4.s32 {%0, %1, %2, %3}, [%4];"
               : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
               : "l"(ptr));
  return ret;
}

__device__ __forceinline__ void st_na_relaxed(const uint8_t *ptr, uint8_t val) {
  asm volatile("st.relaxed.gpu.global.L1::no_allocate.b8 [%0], %1;" : : "l"(ptr), "h"(static_cast<uint16_t>(val)));
}

__device__ __forceinline__ void st_na_relaxed(const uint16_t *ptr, uint16_t val) {
  asm volatile("st.relaxed.gpu.global.L1::no_allocate.b16 [%0], %1;" : : "l"(ptr), "h"(val));
}

__device__ __forceinline__ void st_na_relaxed(const uint32_t *ptr, uint32_t val) {
  asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
}

__device__ __forceinline__ void st_na_relaxed(const int *ptr, int val) {
  asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
}

__device__ __forceinline__ void st_na_relaxed(const int4 *ptr, int4 val) {
  asm volatile("st.relaxed.gpu.global.L1::no_allocate.v4.s32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
}

__device__ __forceinline__ void st_na_release(const int *ptr, int val) {
  asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
}

__device__ __forceinline__ void st_na_release(const uint32_t *ptr, uint32_t val) {
  asm volatile("st.release.gpu.global.L1::no_allocate.b32 [%0], %1;" : : "l"(ptr), "r"(val));
}

__device__ __forceinline__ void st_na_release(const uint64_t *ptr, uint64_t val) {
  asm volatile("st.release.gpu.global.L1::no_allocate.b64 [%0], %1;" : : "l"(ptr), "l"(val));
}

__device__ __forceinline__ void st_na_release(const int64_t *ptr, int64_t val) {
  asm volatile("st.release.gpu.global.L1::no_allocate.b64 [%0], %1;" : : "l"(ptr), "l"(val));
}

// `st.global.L1::no_allocate` will be translated into `ST.E.NA.[width]` in SASS
#ifndef DISABLE_AGGRESSIVE_PTX_INSTRS
#define ST_NA_FUNC "st.global.L1::no_allocate"
#else
#define ST_NA_FUNC "st.global"
#endif

template <typename dtype_t>
__device__ __forceinline__ void st_na_global(const dtype_t *ptr, const dtype_t &value) {
  st_na_global(reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t *>(ptr),
               *reinterpret_cast<const typename VecInt<sizeof(dtype_t)>::vec_t *>(&value));
}

template <>
__device__ __forceinline__ void st_na_global(const int *ptr, const int &value) {
  asm volatile(ST_NA_FUNC ".s32 [%0], %1;" ::"l"(ptr), "r"(value));
}

template <>
__device__ __forceinline__ void st_na_global(const int64_t *ptr, const int64_t &value) {
  asm volatile(ST_NA_FUNC ".s64 [%0], %1;" ::"l"(ptr), "l"(value));
}

template <>
__device__ __forceinline__ void st_na_global(const float *ptr, const float &value) {
  asm volatile(ST_NA_FUNC ".f32 [%0], %1;" ::"l"(ptr), "f"(value));
}

template <>
__device__ __forceinline__ void st_na_global(const int4 *ptr, const int4 &value) {
  asm volatile(ST_NA_FUNC ".v4.s32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), "r"(value.x), "r"(value.y), "r"(value.z),
               "r"(value.w));
}

template <typename dtype_t>
__host__ __device__ dtype_t cell_div(dtype_t a, dtype_t b) {
  return (a + b - 1) / b;
}

template <typename dtype_t>
__host__ __device__ dtype_t align(dtype_t a, dtype_t b) {
  return cell_div<dtype_t>(a, b) * b;
}

__forceinline__ __device__ void get_channel_task_range(int num_tokens, int num_sms, int sm_id, int &token_start_idx,
                                                       int &token_end_idx) {
  int num_tokens_per_sm = cell_div(num_tokens, num_sms);
  token_start_idx = min(num_tokens_per_sm * sm_id, num_tokens);
  token_end_idx = min(token_start_idx + num_tokens_per_sm, num_tokens);
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ dtype_b_t pack2(const dtype_a_t &x, const dtype_a_t &y) {
  EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
  dtype_b_t packed;
  auto unpacked_ptr = reinterpret_cast<dtype_a_t *>(&packed);
  unpacked_ptr[0] = x, unpacked_ptr[1] = y;
  return packed;
}

template <typename dtype_a_t, typename dtype_b_t>
__device__ __forceinline__ void unpack2(const dtype_b_t &packed, dtype_a_t &x, dtype_a_t &y) {
  EP_STATIC_ASSERT(sizeof(dtype_a_t) * 2 == sizeof(dtype_b_t), "Invalid dtypes");
  auto unpacked_ptr = reinterpret_cast<const dtype_a_t *>(&packed);
  x = unpacked_ptr[0], y = unpacked_ptr[1];
}

template <typename T>
__device__ __forceinline__ T warpBroadcast(T value, int sourceLane) {
  EP_STATIC_ASSERT(sizeof(T) % sizeof(int) == 0, "");
  const auto *sourceValues = reinterpret_cast<const int *>(&value);
  T result;
  auto *resultValues = reinterpret_cast<int *>(&result);
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

__forceinline__ __device__ float half_warp_reduce_max(float value) {
  auto mask = __activemask();
  // The mask be in `{0xffffffff, 0xffff}`
  value = max(value, __shfl_xor_sync(mask, value, 8));
  value = max(value, __shfl_xor_sync(mask, value, 4));
  value = max(value, __shfl_xor_sync(mask, value, 2));
  value = max(value, __shfl_xor_sync(mask, value, 1));
  return value;
}

__forceinline__ __device__ int get_lane_id() {
  int lane_id;
  asm("mov.s32 %0, %laneid;" : "=r"(lane_id));
  return lane_id;
}

template <int NumRanks>
__forceinline__ __device__ void move_fifo_slots(int &head) {
  head = (head + NumRanks) % NUM_MAX_FIFO_SLOTS;
}

template <int NumRanks>
__device__ __forceinline__ bool not_finished(int *task, int expected) {
  auto result = false;
  auto lane_id = threadIdx.x % WARP_SIZE;
  if (lane_id < NumRanks) result = ld_volatile_global(task + lane_id) != expected;
  return __any_sync(0xffffffff, result);
}

template <int NumRanks>
__forceinline__ __device__ void timeout_check(int **task_fifo_ptrs, int head, int rank, int expected, int tag = 0) {
  auto start_time = clock64();
  while (not_finished<NumRanks>(task_fifo_ptrs[rank] + head, expected)) {
    if (clock64() - start_time > NUM_TIMEOUT_CYCLES and threadIdx.x == 0) {
      printf("DeepEP timeout check failed: %d (rank = %d)\n", tag, rank);
      trap();
    }
  }
}

template <int NumRanks>
__forceinline__ __device__ void barrier_device(int **task_fifo_ptrs, int head, int rank, int tag = 0) {
  auto thread_id = static_cast<int>(threadIdx.x);
  EP_DEVICE_ASSERT(NumRanks <= WARP_SIZE);

  if (thread_id < NumRanks) {
    atomicAdd_system(task_fifo_ptrs[rank] + head + thread_id, FINISHED_SUM_TAG);
    memory_fence();
    atomicSub_system(task_fifo_ptrs[thread_id] + head + rank, FINISHED_SUM_TAG);
  }
  timeout_check<NumRanks>(task_fifo_ptrs, head, rank, 0, tag);
}

}  // namespace ep
}  // namespace mscclpp
