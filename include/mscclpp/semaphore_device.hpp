// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_SEMAPHORE_DEVICE_HPP_
#define MSCCLPP_SEMAPHORE_DEVICE_HPP_

#include "device.hpp"

#if defined(MSCCLPP_DEVICE_COMPILE)
#include "atomic_device.hpp"
#include "poll_device.hpp"
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

namespace mscclpp {

/// Device-side handle for Host2DeviceSemaphore.
struct Host2DeviceSemaphoreDeviceHandle {
#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Poll if the host has signaled.
  /// @return true if the host has signaled.
  MSCCLPP_DEVICE_INLINE bool poll() {
    bool signaled = (loadInbound() > loadExpectedInbound());
    if (signaled) incExpectedInbound();
    return signaled;
  }

  /// Wait for the host to signal.
  MSCCLPP_DEVICE_INLINE void wait([[maybe_unused]] int64_t maxSpinCount = 100000000) {
    auto expected = incExpectedInbound();
    POLL_MAYBE_JAILBREAK((loadInbound() < expected), maxSpinCount);
  }

  /// Thread-safe read of expected inbound value.
  /// @return The expected inbound value.
  MSCCLPP_DEVICE_INLINE uint64_t loadExpectedInbound() {
    return atomicLoad<uint64_t, scopeDevice>(expectedInboundToken, memoryOrderRelaxed);
  }

  /// Thread-safe increment of expected inbound value.
  /// @return The incremented expected inbound value.
  MSCCLPP_DEVICE_INLINE uint64_t incExpectedInbound() {
    return atomicFetchAdd<uint64_t, scopeDevice>(expectedInboundToken, 1, memoryOrderRelaxed) + 1;
  }

  /// Thread-safe read of inbound value.
  /// @return The inbound value.
  MSCCLPP_DEVICE_INLINE uint64_t loadInbound() {
    return atomicLoad<uint64_t, scopeSystem>(inboundToken, memoryOrderAcquire);
  }
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

  /// A local memory space where a host thread (on behalf of the remote device) will write its semaphore value
  /// and the local device will read it.
  uint64_t* inboundToken;

  /// A local memory space where the local device stores the expected value of the inboundToken to wait for.
  uint64_t* expectedInboundToken;
};

/// Device-side handle for MemoryDevice2DeviceSemaphore.
struct MemoryDevice2DeviceSemaphoreDeviceHandle {
#if defined(MSCCLPP_DEVICE_COMPILE)
  /// Poll if remote device has signaled.
  /// @return true if remote device has signaled.
  MSCCLPP_DEVICE_INLINE bool poll() {
    bool signaled = (loadInbound() > loadExpectedInbound());
    if (signaled) incExpectedInbound();
    return signaled;
  }

  /// Wait for remote device to signal.
  MSCCLPP_DEVICE_INLINE void wait([[maybe_unused]] int64_t maxSpinCount = 100000000) {
    auto expected = incExpectedInbound();
    POLL_MAYBE_JAILBREAK((loadInbound() < expected), maxSpinCount);
  }

  /// Relaxed wait; no memory completion guarantee. Use it only for synchronizing execution, not data.
  MSCCLPP_DEVICE_INLINE void relaxedWait([[maybe_unused]] int64_t maxSpinCount = 100000000) {
    auto expected = incExpectedInbound();
    POLL_MAYBE_JAILBREAK((loadInboundRelaxed() < expected), maxSpinCount);
  }

  /// Signal remote device, ensures prior memory ops complete.
  MSCCLPP_DEVICE_INLINE void signal() {
#if defined(MSCCLPP_DEVICE_CUDA)
    asm volatile("red.release.sys.global.add.u64 [%0], %1;" ::"l"(remoteInboundToken), "l"((uint64_t)1) : "memory");
#elif defined(MSCCLPP_DEVICE_HIP)
    (void)atomicFetchAdd(remoteInboundToken, (uint64_t)1, memoryOrderRelease);
#endif
  }

  /// Relaxed signal; no memory completion guarantee. Use it only for synchronizing execution, not data.
  MSCCLPP_DEVICE_INLINE void relaxedSignal() {
#if defined(MSCCLPP_DEVICE_CUDA)
    asm volatile("red.relaxed.sys.global.add.u64 [%0], %1;" ::"l"(remoteInboundToken), "l"((uint64_t)1) : "memory");
#elif defined(MSCCLPP_DEVICE_HIP)
    (void)atomicFetchAdd(remoteInboundToken, (uint64_t)1, memoryOrderRelaxed);
#endif
  }

  /// Thread-safe read of expected inbound value.
  /// @return The expected inbound value.
  MSCCLPP_DEVICE_INLINE uint64_t loadExpectedInbound() {
    return atomicLoad<uint64_t, scopeDevice>(expectedInboundToken, memoryOrderRelaxed);
  }

  /// Thread-safe increment of expected inbound value.
  /// @return The incremented expected inbound value.
  MSCCLPP_DEVICE_INLINE uint64_t incExpectedInbound() {
    return atomicFetchAdd<uint64_t, scopeDevice>(expectedInboundToken, 1, memoryOrderRelaxed) + 1;
  }

  /// Thread-safe read of inbound value.
  /// @return The inbound value.
  MSCCLPP_DEVICE_INLINE uint64_t loadInbound() {
    return atomicLoad<uint64_t, scopeSystem>(inboundToken, memoryOrderAcquire);
  }

  /// Thread-safe read of inbound value without memory completion guarantee.
  /// @return The inbound value.
  MSCCLPP_DEVICE_INLINE uint64_t loadInboundRelaxed() {
    return atomicLoad<uint64_t, scopeSystem>(inboundToken, memoryOrderRelaxed);
  }

#endif  // defined(MSCCLPP_DEVICE_COMPILE)

  /// A local memory space where the remote device will write its semaphore value and the local device will read it.
  uint64_t* inboundToken;

  /// A remote memory space where the local device atomically increments. This is inboundToken of the remote device.
  uint64_t* remoteInboundToken;

  /// A local memory space where the local device stores the expected value of the inboundToken to wait for.
  uint64_t* expectedInboundToken;
};

/// Device-side handle for a switch (NVLS multicast) cross-rank group barrier semaphore.
///
/// Unlike the point-to-point semaphores above, this is a group semaphore backed by the switch's
/// multimem atomics: a single `signal()` performs one multimem add on the shared arrival counter,
/// which the switch reflects into every participating rank's local copy. `wait()` then spins on this
/// rank's local copy until all `nRanks` peers have arrived. It is the barrier backend for
/// `SwitchChannelDeviceHandle` and needs no separate memory-channel semaphores or host-side barrier.
///
/// A full barrier is a `signal()`/`wait()` pair (or the relaxed variants). The protocol is monotonic
/// and never reset: `wait()`/`relaxedWait()` advance this rank's private generation target by
/// `nRanks` each call, so because every rank calls the pair the same number of times the targets stay
/// in lock-step with the shared counter. Splitting arrival (signal) from completion (wait) lets a
/// kernel overlap independent work between the two halves.
///
/// Ordering is selected by which pair is used. The relaxed pair (`relaxedSignal`/`relaxedWait`) is a
/// pure execution barrier: it synchronizes rank arrival but makes no cross-rank data-visibility
/// guarantee. The ordered pair (`signal`/`wait`) additionally publishes memory: the arrival is a
/// `.release` multimem add and the wait an `.acquire` load, so writes issued by any rank before its
/// `signal()` are visible to all ranks after their `wait()` returns. This ordering is carried by
/// scoped release/acquire on the counter itself (at `.sys` scope, on the counter only) rather than by
/// `__threadfence_system()`, which is much cheaper than a full system fence (this matches NCCL's LSA
/// switch barrier in `lsa_barrier__funcs.h`).
///
/// @note Each method must be called by exactly one thread per rank (e.g. block 0, thread 0); the
/// barrier counts ranks, not threads. For a grid-wide cross-rank barrier, converge the grid (e.g. via
/// `mscclpp::DeviceSyncer::sync`) around the pair. Requires that the owning connection was created
/// with barrier support, i.e. the pointers below are non-null.
struct SwitchDevice2DeviceSemaphoreDeviceHandle {
#if defined(MSCCLPP_DEVICE_CUDA)
  /// Issue an ordered cross-rank arrival, publishing this rank's prior writes.
  ///
  /// Performs one `multimem.red.release.sys.add` of 1 on the shared counter. The `.release` ordering
  /// makes writes issued before this call visible to any peer that observes the arrival via an
  /// acquiring `wait()`. Pair with `wait()`.
  MSCCLPP_DEVICE_INLINE void signal() {
    asm volatile("multimem.red.release.sys.add.u64 [%0], %1;" ::"l"(mcBarrierFlag), "l"((uint64_t)1) : "memory");
  }

  /// Issue a relaxed cross-rank arrival, without any data-visibility ordering.
  ///
  /// Relaxed variant of `signal()`: performs `multimem.red.relaxed.sys.add`, a pure execution arrival
  /// that synchronizes rank progress but makes no cross-rank memory-visibility guarantee. Pair with
  /// `relaxedWait()`.
  MSCCLPP_DEVICE_INLINE void relaxedSignal() {
    asm volatile("multimem.red.relaxed.sys.add.u64 [%0], %1;" ::"l"(mcBarrierFlag), "l"((uint64_t)1) : "memory");
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
    MSCCLPP_ASSERT_DEVICE(barrierGen != nullptr, "SwitchDevice2DeviceSemaphore::wait() called without barrier support");
    const uint64_t target = (*barrierGen += static_cast<uint64_t>(nRanks));
    POLL_MAYBE_JAILBREAK(
        (static_cast<int64_t>(atomicLoad<uint64_t, scopeSystem>(localBarrierFlag, memoryOrderAcquire) - target) < 0),
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
    MSCCLPP_ASSERT_DEVICE(barrierGen != nullptr,
                          "SwitchDevice2DeviceSemaphore::relaxedWait() called without barrier support");
    const uint64_t target = (*barrierGen += static_cast<uint64_t>(nRanks));
    POLL_MAYBE_JAILBREAK(
        (static_cast<int64_t>(atomicLoad<uint64_t, scopeSystem>(localBarrierFlag, memoryOrderRelaxed) - target) < 0),
        maxSpinCount);
  }
#endif  // defined(MSCCLPP_DEVICE_CUDA)

  /// Multicast pointer to the shared arrival counter. A single multimem add here is reflected into
  /// every rank's copy of the counter by the switch. Null if the owning connection has no barrier
  /// support.
  uint64_t* mcBarrierFlag;

  /// Local (unicast) pointer to this rank's own copy of the arrival counter; the address wait() spins
  /// on. Null if the owning connection has no barrier support.
  uint64_t* localBarrierFlag;

  /// Local pointer to this rank's persistent generation counter. It advances by nRanks on every
  /// wait() and provides the per-rank wait target. Persisting it in GPU memory lets the barrier be
  /// called repeatedly within and across kernel launches without any host-side reset. Null if the
  /// owning connection has no barrier support.
  uint64_t* barrierGen;

  /// Number of ranks (devices) participating in the multicast group.
  int nRanks;
};

/// @deprecated Use MemoryDevice2DeviceSemaphoreDeviceHandle instead.
[[deprecated("Use MemoryDevice2DeviceSemaphoreDeviceHandle instead.")]] typedef MemoryDevice2DeviceSemaphoreDeviceHandle
    SmDevice2DeviceSemaphoreDeviceHandle;

}  // namespace mscclpp

#endif  // MSCCLPP_SEMAPHORE_DEVICE_HPP_
