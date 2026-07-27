// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_SWITCH_CHANNEL_DEVICE_HPP_
#define MSCCLPP_SWITCH_CHANNEL_DEVICE_HPP_

#include <mscclpp/gpu.hpp>
#include <type_traits>

#if defined(MSCCLPP_DEVICE_CUDA)
#include <cuda_fp16.h>
#endif  // defined(MSCCLPP_DEVICE_CUDA)

#include <mscclpp/atomic_device.hpp>
#include <mscclpp/gpu_data_types.hpp>
#include <mscclpp/poll_device.hpp>

#include "device.hpp"

namespace mscclpp {

template <class>
constexpr bool dependentFalse = false;  // workaround before CWG2518/P2593R1

/// Device-side handle for SwitchChannel.
struct SwitchChannelDeviceHandle {
  void* devicePtr;
  void* mcPtr;
  size_t bufferSize;
  /// Multicast pointer to the shared arrival counter used by barrier(). A single multimem add on
  /// this pointer is reflected into every rank's copy of the counter by the switch. Null if the
  /// owning connection was created without barrier support.
  uint32_t* mcBarrierFlag;
  /// Local (unicast) pointer to this rank's own copy of the arrival counter used by barrier().
  /// This is the address barrier() spins on. Null if the connection has no barrier support.
  uint32_t* localBarrierFlag;
  /// Local pointer to this rank's persistent barrier generation counter. It advances by nRanks on
  /// every barrier() call and provides the per-rank wait target (see barrier()). Persisting it in
  /// GPU memory lets barrier() be called repeatedly within and across kernel launches without any
  /// host-side reset. Null if the connection has no barrier support.
  uint32_t* barrierGen;
  /// Number of ranks (devices) participating in the multicast group.
  int nRanks;

#if defined(MSCCLPP_DEVICE_CUDA)
  /// Cross-rank switch barrier, split into arrival (signal) and completion (wait) halves.
  ///
  /// These four methods implement a device-side cross-rank barrier over the switch's multimem
  /// atomics, without any memory-channel semaphores or host-side barrier. A full barrier is a
  /// `signal()`/`wait()` pair (or their relaxed variants): every rank issues one multimem add of 1
  /// on the shared arrival counter -- which the switch applies to every rank's copy -- and then
  /// spins on its own local copy until the counter reaches its private target. Splitting arrival
  /// from completion lets a kernel overlap independent work between the two halves.
  ///
  /// The protocol is monotonic and never reset: `wait()`/`relaxedWait()` advance this rank's private
  /// target by nRanks each call. Because every rank calls the pair the same number of times and
  /// advances its target identically, the targets stay in lock-step with the shared counter.
  ///
  /// Ordering is selected by which pair is used. The relaxed pair (`relaxedSignal`/`relaxedWait`) is
  /// a pure execution barrier: it synchronizes rank arrival but makes no cross-rank data-visibility
  /// guarantee. The ordered pair (`signal`/`wait`) additionally publishes memory: the arrival is a
  /// `.release` multimem add and the wait an `.acquire` load, so writes issued by any rank before its
  /// `signal()` are visible to all ranks after their `wait()` returns. This ordering is carried by
  /// scoped release/acquire on the counter itself (at `.sys` scope, on the counter only) rather than
  /// by a pair of `__threadfence_system()` calls, which is much cheaper than a full system fence
  /// (this matches NCCL's LSA switch barrier in `lsa_barrier__funcs.h`).
  ///
  /// @note Each method must be called by exactly one thread per rank (e.g. block 0, thread 0); the
  /// barrier counts ranks, not threads. For a grid-wide cross-rank barrier, converge the grid (e.g.
  /// via `mscclpp::DeviceSyncer::sync`) around the pair. Requires that the owning `NvlsConnection`
  /// was created with barrier support, i.e. the barrier pointers are non-null.

  /// Issue an ordered cross-rank arrival, publishing this rank's prior writes.
  ///
  /// Performs one `multimem.red.release.sys.add` of 1 on the shared counter. The `.release` ordering
  /// makes writes issued before this call visible to any peer that observes the arrival via an
  /// acquiring `wait()`. Pair with `wait()`.
  MSCCLPP_DEVICE_INLINE void signal() {
    asm volatile("multimem.red.release.sys.add.u32 [%0], %1;" ::"l"(mcBarrierFlag), "r"(1U) : "memory");
  }

  /// Issue a relaxed cross-rank arrival, without any data-visibility ordering.
  ///
  /// Relaxed variant of `signal()`: performs `multimem.red.relaxed.sys.add`, a pure execution arrival
  /// that synchronizes rank progress but makes no cross-rank memory-visibility guarantee. Pair with
  /// `relaxedWait()`.
  MSCCLPP_DEVICE_INLINE void relaxedSignal() {
    asm volatile("multimem.red.relaxed.sys.add.u32 [%0], %1;" ::"l"(mcBarrierFlag), "r"(1U) : "memory");
  }

  /// Wait until every rank has arrived, acquiring peers' published writes.
  ///
  /// Advances this rank's private target by nRanks, then spins on its local copy of the counter with
  /// an `.acquire` load until the counter reaches the target (the signed, wrap-safe compare means
  /// "counter is behind target"). The acquire pairs with peers' `signal()` release so their
  /// pre-arrival writes are visible after this returns. Pair with `signal()`.
  ///
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  MSCCLPP_DEVICE_INLINE void wait(int64_t maxSpinCount = 10000000) {
    MSCCLPP_ASSERT_DEVICE(barrierGen != nullptr, "SwitchChannel::wait() called without barrier support");
    const uint32_t target = (*barrierGen += static_cast<uint32_t>(nRanks));
    POLL_MAYBE_JAILBREAK(
        (static_cast<int32_t>(atomicLoad<uint32_t, scopeSystem>(localBarrierFlag, cuda::memory_order::acquire) -
                              target) < 0),
        maxSpinCount);
  }

  /// Wait until every rank has arrived, without any data-visibility ordering.
  ///
  /// Relaxed variant of `wait()`: advances this rank's private target by nRanks and spins on its local
  /// copy of the counter with a relaxed load until the counter reaches the target. Provides rank
  /// synchronization only (no cross-rank memory ordering). Pair with `relaxedSignal()`.
  ///
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  MSCCLPP_DEVICE_INLINE void relaxedWait(int64_t maxSpinCount = 10000000) {
    MSCCLPP_ASSERT_DEVICE(barrierGen != nullptr, "SwitchChannel::relaxedWait() called without barrier support");
    const uint32_t target = (*barrierGen += static_cast<uint32_t>(nRanks));
    POLL_MAYBE_JAILBREAK(
        (static_cast<int32_t>(atomicLoad<uint32_t, scopeSystem>(localBarrierFlag, cuda::memory_order::relaxed) -
                              target) < 0),
        maxSpinCount);
  }

  template <typename T>
  MSCCLPP_DEVICE_INLINE T reduce(uint64_t index) {
    return SwitchChannelDeviceHandle::multimemLoadReduce(reinterpret_cast<T*>(mcPtr) + index);
  }

  template <typename T>
  MSCCLPP_DEVICE_INLINE void broadcast(uint64_t index, const T& val) {
    SwitchChannelDeviceHandle::multimemStore(val, reinterpret_cast<T*>(mcPtr) + index);
  }

  /// Vectorized multimem load+reduce. The optional `AccumT` template parameter selects the
  /// accumulator: when `AccumT == __half` and `VectorType` is an FP8 vector type, the
  /// `.acc::f16` variant of the instruction is used (faster but lower precision than the
  /// default FP32 accumulator). For all other types `AccumT` is ignored.
  template <typename VectorType, typename AccumT = void>
  MSCCLPP_DEVICE_INLINE static VectorType multimemLoadReduce(VectorType* ptr) {
    VectorType val;
    if constexpr (std::is_same_v<VectorType, i32x1>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.s32 %0, [%1];" : "=r"(val.words[0]) : "l"(ptr) : "memory");
    } else if constexpr (std::is_same_v<VectorType, u32x1>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.u32 %0, [%1];" : "=r"(val.words[0]) : "l"(ptr) : "memory");
    } else if constexpr (std::is_same_v<VectorType, f32x1>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];" : "=r"(val.words[0]) : "l"(ptr) : "memory");
    } else if constexpr (std::is_same_v<VectorType, f32x2>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.v2.f32 {%0,%1}, [%2];"
          : "=r"(val.words[0]), "=r"(val.words[1])
          : "l"(ptr)
          : "memory");
    } else if constexpr (std::is_same_v<VectorType, f32x4>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.v4.f32 {%0,%1,%2,%3}, [%4];"
          : "=r"(val.words[0]), "=r"(val.words[1]), "=r"(val.words[2]), "=r"(val.words[3])
          : "l"(ptr)
          : "memory");
    } else if constexpr (std::is_same_v<VectorType, f16x2>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.f16x2 %0, [%1];" : "=r"(val.words[0]) : "l"(ptr) : "memory");
    } else if constexpr (std::is_same_v<VectorType, f16x4>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.v2.f16x2 {%0,%1}, [%2];"
          : "=r"(val.words[0]), "=r"(val.words[1])
          : "l"(ptr)
          : "memory");
    } else if constexpr (std::is_same_v<VectorType, f16x8>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.v4.f16x2 {%0,%1,%2,%3}, [%4];"
          : "=r"(val.words[0]), "=r"(val.words[1]), "=r"(val.words[2]), "=r"(val.words[3])
          : "l"(ptr)
          : "memory");
    } else if constexpr (std::is_same_v<VectorType, bf16x2>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.bf16x2 %0, [%1];" : "=r"(val.words[0]) : "l"(ptr) : "memory");
    } else if constexpr (std::is_same_v<VectorType, bf16x4>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.v2.bf16x2 {%0,%1}, [%2];"
          : "=r"(val.words[0]), "=r"(val.words[1])
          : "l"(ptr)
          : "memory");
    } else if constexpr (std::is_same_v<VectorType, bf16x8>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
          : "=r"(val.words[0]), "=r"(val.words[1]), "=r"(val.words[2]), "=r"(val.words[3])
          : "l"(ptr)
          : "memory");
    } else if constexpr (std::is_same_v<VectorType, f8_e4m3x4>) {
      if constexpr (std::is_same_v<AccumT, __half>) {
        asm("multimem.ld_reduce.relaxed.sys.global.add.acc::f16.e4m3x4 %0, [%1];"
            : "=r"(val.words[0])
            : "l"(ptr)
            : "memory");
      } else {
        asm("multimem.ld_reduce.relaxed.sys.global.add.e4m3x4 %0, [%1];" : "=r"(val.words[0]) : "l"(ptr) : "memory");
      }
    } else if constexpr (std::is_same_v<VectorType, f8_e4m3x8>) {
      if constexpr (std::is_same_v<AccumT, __half>) {
        asm("multimem.ld_reduce.relaxed.sys.global.add.acc::f16.v2.e4m3x4 {%0,%1}, [%2];"
            : "=r"(val.words[0]), "=r"(val.words[1])
            : "l"(ptr)
            : "memory");
      } else {
        asm("multimem.ld_reduce.relaxed.sys.global.add.v2.e4m3x4 {%0,%1}, [%2];"
            : "=r"(val.words[0]), "=r"(val.words[1])
            : "l"(ptr)
            : "memory");
      }
    } else if constexpr (std::is_same_v<VectorType, f8_e4m3x16>) {
      if constexpr (std::is_same_v<AccumT, __half>) {
        asm("multimem.ld_reduce.relaxed.sys.global.add.acc::f16.v4.e4m3x4 {%0,%1,%2,%3}, [%4];"
            : "=r"(val.words[0]), "=r"(val.words[1]), "=r"(val.words[2]), "=r"(val.words[3])
            : "l"(ptr)
            : "memory");
      } else {
        asm("multimem.ld_reduce.relaxed.sys.global.add.v4.e4m3x4 {%0,%1,%2,%3}, [%4];"
            : "=r"(val.words[0]), "=r"(val.words[1]), "=r"(val.words[2]), "=r"(val.words[3])
            : "l"(ptr)
            : "memory");
      }
    } else if constexpr (std::is_same_v<VectorType, f8_e5m2x4>) {
      if constexpr (std::is_same_v<AccumT, __half>) {
        asm("multimem.ld_reduce.relaxed.sys.global.add.acc::f16.e5m2x4 %0, [%1];"
            : "=r"(val.words[0])
            : "l"(ptr)
            : "memory");
      } else {
        asm("multimem.ld_reduce.relaxed.sys.global.add.e5m2x4 %0, [%1];" : "=r"(val.words[0]) : "l"(ptr) : "memory");
      }
    } else if constexpr (std::is_same_v<VectorType, f8_e5m2x8>) {
      if constexpr (std::is_same_v<AccumT, __half>) {
        asm("multimem.ld_reduce.relaxed.sys.global.add.acc::f16.v2.e5m2x4 {%0,%1}, [%2];"
            : "=r"(val.words[0]), "=r"(val.words[1])
            : "l"(ptr)
            : "memory");
      } else {
        asm("multimem.ld_reduce.relaxed.sys.global.add.v2.e5m2x4 {%0,%1}, [%2];"
            : "=r"(val.words[0]), "=r"(val.words[1])
            : "l"(ptr)
            : "memory");
      }
    } else if constexpr (std::is_same_v<VectorType, f8_e5m2x16>) {
      if constexpr (std::is_same_v<AccumT, __half>) {
        asm("multimem.ld_reduce.relaxed.sys.global.add.acc::f16.v4.e5m2x4 {%0,%1,%2,%3}, [%4];"
            : "=r"(val.words[0]), "=r"(val.words[1]), "=r"(val.words[2]), "=r"(val.words[3])
            : "l"(ptr)
            : "memory");
      } else {
        asm("multimem.ld_reduce.relaxed.sys.global.add.v4.e5m2x4 {%0,%1,%2,%3}, [%4];"
            : "=r"(val.words[0]), "=r"(val.words[1]), "=r"(val.words[2]), "=r"(val.words[3])
            : "l"(ptr)
            : "memory");
      }
    } else {
      static_assert(dependentFalse<VectorType>, "Not supported type");
    }
    return val;
  };

  template <typename VectorType, typename T>
  MSCCLPP_DEVICE_INLINE static void multimemStore(const VectorType& val, T* ptr) {
    if constexpr (std::is_same_v<VectorType, i32x1>) {
      asm volatile("multimem.st.relaxed.sys.global.s32 [%0], %1;" ::"l"(ptr), "r"(val.words[0]) : "memory");
    } else if constexpr (std::is_same_v<VectorType, u32x1>) {
      asm volatile("multimem.st.relaxed.sys.global.u32 [%0], %1;" ::"l"(ptr), "r"(val.words[0]) : "memory");
    } else if constexpr (std::is_same_v<VectorType, f64x1>) {
      asm volatile("multimem.st.relaxed.sys.global.f64 [%0], %1;" ::"l"(ptr), "d"(val.words[0]) : "memory");
    } else if constexpr (std::is_same_v<VectorType, f32x1>) {
      asm volatile("multimem.st.relaxed.sys.global.f32 [%0], %1;" ::"l"(ptr), "r"(val.words[0]) : "memory");
    } else if constexpr (std::is_same_v<VectorType, f32x2>) {
      asm volatile("multimem.st.relaxed.sys.global.v2.f32 [%0], {%1,%2};" ::"l"(ptr), "r"(val.words[0]),
                   "r"(val.words[1])
                   : "memory");
    } else if constexpr (std::is_same_v<VectorType, f32x4>) {
      asm volatile("multimem.st.relaxed.sys.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(val.words[0]),
                   "r"(val.words[1]), "r"(val.words[2]), "r"(val.words[3])
                   : "memory");
    } else if constexpr (std::is_same_v<VectorType, f16x2>) {
      asm volatile("multimem.st.relaxed.sys.global.f16x2 [%0], %1;" ::"l"(ptr), "r"(val.words[0]) : "memory");
    } else if constexpr (std::is_same_v<VectorType, f16x4>) {
      asm volatile("multimem.st.relaxed.sys.global.v2.f16x2 [%0], {%1,%2};" ::"l"(ptr), "r"(val.words[0]),
                   "r"(val.words[1])
                   : "memory");
    } else if constexpr (std::is_same_v<VectorType, f16x8>) {
      asm volatile("multimem.st.relaxed.sys.global.v4.f16x2 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(val.words[0]),
                   "r"(val.words[1]), "r"(val.words[2]), "r"(val.words[3])
                   : "memory");
    } else if constexpr (std::is_same_v<VectorType, bf16x2>) {
      asm volatile("multimem.st.relaxed.sys.global.bf16x2 [%0], %1;" ::"l"(ptr), "r"(val.words[0]) : "memory");
    } else if constexpr (std::is_same_v<VectorType, bf16x4>) {
      asm volatile("multimem.st.relaxed.sys.global.v2.bf16x2 [%0], {%1,%2};" ::"l"(ptr), "r"(val.words[0]),
                   "r"(val.words[1])
                   : "memory");
    } else if constexpr (std::is_same_v<VectorType, bf16x8>) {
      asm volatile("multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(val.words[0]),
                   "r"(val.words[1]), "r"(val.words[2]), "r"(val.words[3])
                   : "memory");
    } else if constexpr (std::is_same_v<VectorType, f8_e4m3x4>) {
      asm volatile("multimem.st.relaxed.sys.global.e4m3x4 [%0], %1;" ::"l"(ptr), "r"(val.words[0]) : "memory");
    } else if constexpr (std::is_same_v<VectorType, f8_e4m3x8>) {
      asm volatile("multimem.st.relaxed.sys.global.v2.e4m3x4  [%0], {%1,%2};" ::"l"(ptr), "r"(val.words[0]),
                   "r"(val.words[1])
                   : "memory");
    } else if constexpr (std::is_same_v<VectorType, f8_e4m3x16>) {
      asm volatile("multimem.st.relaxed.sys.global.v4.e4m3x4 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(val.words[0]),
                   "r"(val.words[1]), "r"(val.words[2]), "r"(val.words[3])
                   : "memory");
    } else if constexpr (std::is_same_v<VectorType, f8_e5m2x4>) {
      asm volatile("multimem.st.relaxed.sys.global.e5m2x4 [%0], %1;" ::"l"(ptr), "r"(val.words[0]) : "memory");
    } else if constexpr (std::is_same_v<VectorType, f8_e5m2x8>) {
      asm volatile("multimem.st.relaxed.sys.global.v2.e5m2x4  [%0], {%1,%2};" ::"l"(ptr), "r"(val.words[0]),
                   "r"(val.words[1])
                   : "memory");
    } else if constexpr (std::is_same_v<VectorType, f8_e5m2x16>) {
      asm volatile("multimem.st.relaxed.sys.global.v4.e5m2x4 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(val.words[0]),
                   "r"(val.words[1]), "r"(val.words[2]), "r"(val.words[3])
                   : "memory");
    } else {
      static_assert(dependentFalse<VectorType>, "Not supported type");
    }
  };

  template <typename TValue, typename T>
  MSCCLPP_DEVICE_INLINE static void multimemStoreReduce(const TValue& val, T* ptr) {
    if constexpr (std::is_same_v<TValue, float4> && std::is_same_v<T, float>) {
      asm volatile("multimem.red.relaxed.sys.global.add.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(val.x), "r"(val.y),
                   "r"(val.z), "r"(val.w)
                   : "memory");
    } else if constexpr (std::is_same_v<TValue, uint2> && std::is_same_v<T, float>) {
      asm volatile("multimem.red.relaxed.sys.global.add.v2.f32 [%0], {%1,%2};" ::"l"(ptr), "r"(val.x), "r"(val.y)
                   : "memory");
    } else if constexpr (std::is_same_v<TValue, uint1> && std::is_same_v<T, float>) {
      asm volatile("multimem.red.relaxed.sys.global.add.f32 [%0], {%1};" ::"l"(ptr), "r"(val.x) : "memory");
    } else if constexpr (std::is_same_v<TValue, uint4> && std::is_same_v<T, __half2>) {
      asm volatile("multimem.red.relaxed.sys.global.add.v4.f16x2 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(val.x),
                   "r"(val.y), "r"(val.z), "r"(val.w)
                   : "memory");
    } else if constexpr (std::is_same_v<TValue, uint2> && std::is_same_v<T, __half2>) {
      asm volatile("multimem.red.relaxed.sys.global.add.v2.f16x2 [%0], {%1,%2};" ::"l"(ptr), "r"(val.x), "r"(val.y)
                   : "memory");
    } else if constexpr (std::is_same_v<TValue, uint1> && std::is_same_v<T, __half2>) {
      asm volatile("multimem.red.relaxed.sys.global.add.f16x2 [%0], {%1};" ::"l"(ptr), "r"(val.x) : "memory");
    } else {
      static_assert(dependentFalse<T>, "Not supported type");
    }
  };
#endif  // defined(MSCCLPP_DEVICE_CUDA)
};

}  // namespace mscclpp

#endif  // MSCCLPP_SWITCH_CHANNEL_DEVICE_HPP_
