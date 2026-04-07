// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_SWITCH_CHANNEL_DEVICE_HPP_
#define MSCCLPP_SWITCH_CHANNEL_DEVICE_HPP_

#include <mscclpp/gpu.hpp>
#include <type_traits>

#if defined(MSCCLPP_DEVICE_CUDA)
#include <cuda_fp16.h>
#endif  // defined(MSCCLPP_DEVICE_CUDA)

#include <mscclpp/gpu_data_types.hpp>

#include "device.hpp"

#if defined(MSCCLPP_DEVICE_COMPILE)
#include "atomic_device.hpp"
#include "poll_device.hpp"
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

namespace mscclpp {

template <class>
constexpr bool dependentFalse = false;  // workaround before CWG2518/P2593R1

/// Device-side handle for SwitchChannel.
struct SwitchChannelDeviceHandle {
  void* devicePtr;
  void* mcPtr;
  size_t bufferSize;

#if defined(MSCCLPP_DEVICE_CUDA)
  template <typename T>
  MSCCLPP_DEVICE_INLINE T reduce(uint64_t index) {
    return SwitchChannelDeviceHandle::multimemLoadReduce(reinterpret_cast<T*>(mcPtr) + index);
  }

  template <typename T>
  MSCCLPP_DEVICE_INLINE void broadcast(uint64_t index, const T& val) {
    SwitchChannelDeviceHandle::multimemStore(val, reinterpret_cast<T*>(mcPtr) + index);
  }

  template <typename VectorType>
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
      asm("multimem.ld_reduce.relaxed.sys.global.add.e4m3x4 %0, [%1];" : "=r"(val.words[0]) : "l"(ptr) : "memory");
    } else if constexpr (std::is_same_v<VectorType, f8_e4m3x8>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.v2.e4m3x4 {%0,%1}, [%2];"
          : "=r"(val.words[0]), "=r"(val.words[1])
          : "l"(ptr)
          : "memory");
    } else if constexpr (std::is_same_v<VectorType, f8_e4m3x16>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.v4.e4m3x4 {%0,%1,%2,%3}, [%4];"
          : "=r"(val.words[0]), "=r"(val.words[1]), "=r"(val.words[2]), "=r"(val.words[3])
          : "l"(ptr)
          : "memory");
    } else if constexpr (std::is_same_v<VectorType, f8_e5m2x4>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.e5m2x4 %0, [%1];" : "=r"(val.words[0]) : "l"(ptr) : "memory");
    } else if constexpr (std::is_same_v<VectorType, f8_e5m2x8>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.v2.e5m2x4 {%0,%1}, [%2];"
          : "=r"(val.words[0]), "=r"(val.words[1])
          : "l"(ptr)
          : "memory");
    } else if constexpr (std::is_same_v<VectorType, f8_e5m2x16>) {
      asm("multimem.ld_reduce.relaxed.sys.global.add.v4.e5m2x4 {%0,%1,%2,%3}, [%4];"
          : "=r"(val.words[0]), "=r"(val.words[1]), "=r"(val.words[2]), "=r"(val.words[3])
          : "l"(ptr)
          : "memory");
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

/// Device-side handle for @ref SwitchGroupSemaphore.
///
/// Provides O(1) signal/wait synchronization across all devices in a multicast group
/// using `multimem.red` hardware reduction on the NVSwitch. Each signal atomically
/// increments a flag on all peers via a single multicast reduction, and each wait
/// polls the local flag until all devices have signaled.
struct SwitchGroupSemaphoreDeviceHandle {
#if defined(MSCCLPP_DEVICE_CUDA)
  /// Signal all devices in the multicast group. Ensures prior memory operations are visible.
  MSCCLPP_DEVICE_INLINE void signal() {
    asm volatile("multimem.red.release.sys.global.add.u32 [%0], %1;" ::"l"(mcFlag), "r"(1u) : "memory");
  }

  /// Relaxed signal; no prior memory completion guarantee. Use only for synchronizing execution, not data.
  MSCCLPP_DEVICE_INLINE void relaxedSignal() {
    asm volatile("multimem.red.relaxed.sys.global.add.u32 [%0], %1;" ::"l"(mcFlag), "r"(1u) : "memory");
  }

  /// Wait for all devices in the group to signal.
  /// @param maxSpinCount Maximum number of spin iterations before assertion. Never asserts if negative.
  MSCCLPP_DEVICE_INLINE void wait([[maybe_unused]] int64_t maxSpinCount = 100000000) {
    uint32_t expected = incExpectedInbound();
    POLL_MAYBE_JAILBREAK((loadInbound() < expected), maxSpinCount);
  }

  /// Relaxed wait; no memory completion guarantee. Use only for synchronizing execution, not data.
  /// @param maxSpinCount Maximum number of spin iterations before assertion. Never asserts if negative.
  MSCCLPP_DEVICE_INLINE void relaxedWait([[maybe_unused]] int64_t maxSpinCount = 100000000) {
    uint32_t expected = incExpectedInbound();
    POLL_MAYBE_JAILBREAK((loadInboundRelaxed() < expected), maxSpinCount);
  }

  /// Thread-safe read of expected inbound value.
  /// @return The expected inbound value.
  MSCCLPP_DEVICE_INLINE uint32_t loadExpectedInbound() {
    return atomicLoad<uint32_t, scopeDevice>(expectedInbound, memoryOrderRelaxed);
  }

  /// Thread-safe increment of expected inbound value by @ref numDevices.
  /// @return The incremented expected inbound value.
  MSCCLPP_DEVICE_INLINE uint32_t incExpectedInbound() {
    return atomicFetchAdd<uint32_t, scopeDevice>(expectedInbound, static_cast<uint32_t>(numDevices),
                                                 memoryOrderRelaxed) +
           static_cast<uint32_t>(numDevices);
  }

  /// Thread-safe read of inbound flag value with acquire ordering.
  /// @return The inbound flag value.
  MSCCLPP_DEVICE_INLINE uint32_t loadInbound() {
    return atomicLoad<uint32_t, scopeSystem>(deviceFlag, memoryOrderAcquire);
  }

  /// Thread-safe read of inbound flag value with relaxed ordering.
  /// @return The inbound flag value.
  MSCCLPP_DEVICE_INLINE uint32_t loadInboundRelaxed() {
    return atomicLoad<uint32_t, scopeSystem>(deviceFlag, memoryOrderRelaxed);
  }
#endif  // defined(MSCCLPP_DEVICE_CUDA)

  /// Multicast address for the flag (used for signaling via multimem.red).
  uint32_t* mcFlag;

  /// Local device address for the flag (used for polling during wait).
  uint32_t* deviceFlag;

  /// Local GPU memory where the expected inbound value is tracked.
  uint32_t* expectedInbound;

  /// Number of devices in the multicast group.
  int numDevices;
};

}  // namespace mscclpp

#endif  // MSCCLPP_SWITCH_CHANNEL_DEVICE_HPP_
