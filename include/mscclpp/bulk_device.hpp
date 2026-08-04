// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_BULK_DEVICE_HPP_
#define MSCCLPP_BULK_DEVICE_HPP_

#include <cstdint>
#include <type_traits>

#include "assert_device.hpp"
#include "device.hpp"
#include "poll_device.hpp"

/// 1 if bulk asynchronous copy is available on the current device compilation target, 0 otherwise.
/// The declarations in this header exist only where this is 1, so call sites must be guarded by it.
#if defined(MSCCLPP_DEVICE_CUDA) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define MSCCLPP_BULK_AVAILABLE 1
#else
#define MSCCLPP_BULK_AVAILABLE 0
#endif

#if MSCCLPP_BULK_AVAILABLE
#include <cuda_bf16.h>
#endif  // MSCCLPP_BULK_AVAILABLE

namespace mscclpp {

#if MSCCLPP_BULK_AVAILABLE

namespace detail {  // NOLINT

template <class>
constexpr bool bulkDependentFalse = false;

/// Bulk copies require 16-byte aligned addresses and a size that is a multiple of 16 bytes.
MSCCLPP_DEVICE_INLINE bool bulkAligned(const void* ptr) { return (reinterpret_cast<uintptr_t>(ptr) & 15) == 0; }

}  // namespace detail

struct BulkBarrier;

MSCCLPP_DEVICE_INLINE void bulkFence();
MSCCLPP_DEVICE_INLINE void bulkLoad(void* dstShared, const void* srcGlobal, uint32_t bytes, BulkBarrier& barrier);

#endif  // MSCCLPP_BULK_AVAILABLE

/// Completion barrier for bulk loads.
///
/// A bulk load completes asynchronously; its completion is tracked by counting transferred bytes
/// against this barrier. The barrier must live in shared memory and alternates between two phases,
/// so one barrier serves an unbounded number of batches without re-initialization.
///
/// Per batch the caller declares the bytes to wait for with expect(), signals participation with
/// arrive(), and blocks with wait(); arriveAndExpect() fuses the common single-issuer case. The
/// phase starts at 0 and wait() flips it, so the caller keeps the phase in a register and passes it
/// back on the next batch.
///
/// @warning Every expect() for a batch must be issued before the arrival that completes the arrival
/// count. Arriving while the outstanding byte count is momentarily zero completes the phase
/// immediately, and wait() then returns over tiles that have not been filled. Issuing a single
/// arriveAndExpect() carrying the batch total avoids this entirely and is the recommended form.
///
/// The storage is declared on every build target, including host compilation, so that host code can
/// size the shared memory that holds barriers. The operations exist only where MSCCLPP_BULK_AVAILABLE
/// is 1.
struct BulkBarrier {
#if MSCCLPP_BULK_AVAILABLE
  /// Initialize the barrier and publish it to the asynchronous proxy. Called by a single thread
  /// before any other method, and not while a bulk load tracked by this barrier is in flight.
  /// @param arriveCount Number of arrivals that complete a phase.
  MSCCLPP_DEVICE_INLINE void init(uint32_t arriveCount = 1) {
    relaxedInit(arriveCount);
    bulkFence();
  }

  /// Initialize the barrier without publishing it to the asynchronous proxy.
  ///
  /// The initialization is not visible to bulk copies until a bulkFence() runs on the initializing
  /// thread. Use this to initialize several barriers under a single fence; otherwise use init().
  /// @param arriveCount Number of arrivals that complete a phase.
  MSCCLPP_DEVICE_INLINE void relaxedInit(uint32_t arriveCount = 1) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(addr()), "r"(arriveCount));
  }

  /// Invalidate the barrier, releasing the underlying hardware state. Called by a single thread
  /// before the shared memory holding this barrier is reused for another purpose.
  MSCCLPP_DEVICE_INLINE void invalidate() { asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(addr())); }

  /// Add to the bytes the current phase waits for, without arriving.
  /// @param bytes Bytes expected to arrive.
  MSCCLPP_DEVICE_INLINE void expect(uint32_t bytes) {
    asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;" ::"r"(addr()), "r"(bytes));
  }

  /// Signal one arrival on the current phase.
  MSCCLPP_DEVICE_INLINE void arrive() { asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(addr())); }

  /// Signal one arrival and add to the bytes the current phase waits for. Equivalent to expect()
  /// followed by arrive(), issued as a single instruction.
  /// @param bytes Bytes expected to arrive.
  MSCCLPP_DEVICE_INLINE void arriveAndExpect(uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(addr()), "r"(bytes));
  }

  /// Check whether the given phase has completed, without blocking or advancing the phase.
  /// @param phase The phase to check.
  /// @return True if the phase has completed.
  MSCCLPP_DEVICE_INLINE bool poll(uint32_t phase) {
    uint32_t done;
    asm volatile("{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p;}"
                 : "=r"(done)
                 : "r"(addr()), "r"(phase & 1u));
    return done != 0;
  }

  /// Wait for the given phase to complete and advance @p phase to the next one.
  ///
  /// Does not order the delivered data against generic shared memory reads; run bulkFence() on each
  /// consuming thread, or a thread barrier that orders it against one that did, before reading the
  /// loaded tiles.
  ///
  /// @param phase The phase to wait for, flipped on return. Starts at 0 and is kept by the caller.
  /// @param maxSpinCount The maximum number of spins before asserting. Never assert if negative.
  MSCCLPP_DEVICE_INLINE void wait(uint32_t& phase, int64_t maxSpinCount = 10000000) {
    POLL_MAYBE_JAILBREAK(!poll(phase), maxSpinCount);
    phase ^= 1u;
  }

 private:
  friend MSCCLPP_DEVICE_INLINE void bulkLoad(void* dstShared, const void* srcGlobal, uint32_t bytes,
                                             BulkBarrier& barrier);

  MSCCLPP_DEVICE_INLINE uint32_t addr() const { return static_cast<uint32_t>(__cvta_generic_to_shared(&mbar_)); }
#endif  // MSCCLPP_BULK_AVAILABLE

 private:
  alignas(8) uint64_t mbar_;
};

#if MSCCLPP_BULK_AVAILABLE

/// Issue an asynchronous bulk load from global memory to shared memory.
///
/// Returns immediately; completion is tracked by @p barrier, which must expect at least @p bytes for
/// the phase being waited on. Issued by a single thread. Loaded data is visible to generic shared
/// memory reads only after the barrier's phase completes and bulkFence() has run.
///
/// @param dstShared Destination in shared memory. 16-byte aligned; 128 bytes is faster.
/// @param srcGlobal Source in global memory, local or peer-mapped. 16-byte aligned.
/// @param bytes Bytes to load. A multiple of 16.
/// @param barrier Barrier tracking completion of this load.
MSCCLPP_DEVICE_INLINE void bulkLoad(void* dstShared, const void* srcGlobal, uint32_t bytes, BulkBarrier& barrier) {
  MSCCLPP_ASSERT_DEVICE(detail::bulkAligned(dstShared), "bulkLoad destination is not 16-byte aligned");
  MSCCLPP_ASSERT_DEVICE(detail::bulkAligned(srcGlobal), "bulkLoad source is not 16-byte aligned");
  MSCCLPP_ASSERT_DEVICE((bytes & 15) == 0, "bulkLoad size is not a multiple of 16 bytes");
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                   static_cast<uint32_t>(__cvta_generic_to_shared(dstShared))),
               "l"(srcGlobal), "r"(bytes), "r"(barrier.addr())
               : "memory");
}

/// Issue an asynchronous bulk store from shared memory to global memory.
///
/// Returns immediately; the store joins the calling thread's open bulk group, which is closed by
/// bulkStoreCommit() and drained by bulkStoreWait() or bulkStoreWaitSource(). Issued by a single
/// thread. Data written to @p srcShared by generic shared memory writes is visible to the store only
/// after bulkFence().
///
/// @param dstGlobal Destination in global memory, local or peer-mapped. 16-byte aligned.
/// @param srcShared Source in shared memory. 16-byte aligned.
/// @param bytes Bytes to store. A multiple of 16.
MSCCLPP_DEVICE_INLINE void bulkStore(void* dstGlobal, const void* srcShared, uint32_t bytes) {
  MSCCLPP_ASSERT_DEVICE(detail::bulkAligned(dstGlobal), "bulkStore destination is not 16-byte aligned");
  MSCCLPP_ASSERT_DEVICE(detail::bulkAligned(srcShared), "bulkStore source is not 16-byte aligned");
  MSCCLPP_ASSERT_DEVICE((bytes & 15) == 0, "bulkStore size is not a multiple of 16 bytes");
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(dstGlobal),
               "r"(static_cast<uint32_t>(__cvta_generic_to_shared(srcShared))), "r"(bytes)
               : "memory");
}

/// Reduction operation applied by bulkReduceStore().
enum class BulkRedOp { Add };

/// Issue an asynchronous bulk reduction from shared memory into global memory.
///
/// Accumulates @p srcShared into @p dstGlobal elementwise, in the copy engine rather than on the
/// SM, so the destination is never read back across the interconnect. Works on peer-mapped
/// destinations, which makes it a remote accumulate. Otherwise behaves exactly like bulkStore():
/// returns immediately, joins the calling thread's open bulk group, and requires a preceding
/// bulkFence() to make generic writes to @p srcShared visible.
///
/// Measured on H200 at roughly 90% of the bulkStore() rate for the same payload, from a single
/// issuing thread, so the accumulate is close to free relative to the transfer.
///
/// @tparam T Element type. Currently `float`, `__nv_bfloat16`, and `uint32_t`. Other types are
/// rejected at compile time rather than silently mapped, because the underlying instruction accepts
/// only certain operation and type combinations.
/// @tparam Op Reduction operation.
/// @param dstGlobal Destination in global memory, local or peer-mapped. 16-byte aligned.
/// @param srcShared Source in shared memory. 16-byte aligned.
/// @param bytes Bytes to reduce. A multiple of 16.
template <typename T, BulkRedOp Op = BulkRedOp::Add>
MSCCLPP_DEVICE_INLINE void bulkReduceStore(void* dstGlobal, const void* srcShared, uint32_t bytes) {
  MSCCLPP_ASSERT_DEVICE(detail::bulkAligned(dstGlobal), "bulkReduceStore destination is not 16-byte aligned");
  MSCCLPP_ASSERT_DEVICE(detail::bulkAligned(srcShared), "bulkReduceStore source is not 16-byte aligned");
  MSCCLPP_ASSERT_DEVICE((bytes & 15) == 0, "bulkReduceStore size is not a multiple of 16 bytes");
  const uint32_t src = static_cast<uint32_t>(__cvta_generic_to_shared(srcShared));
  if constexpr (Op == BulkRedOp::Add && std::is_same_v<T, float>) {
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;" ::"l"(dstGlobal),
                 "r"(src), "r"(bytes)
                 : "memory");
  } else if constexpr (Op == BulkRedOp::Add && std::is_same_v<T, __nv_bfloat16>) {
    // The instruction requires an explicit .noftz for bf16 addition.
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 [%0], [%1], %2;" ::"l"(dstGlobal),
                 "r"(src), "r"(bytes)
                 : "memory");
  } else if constexpr (Op == BulkRedOp::Add && std::is_same_v<T, uint32_t>) {
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32 [%0], [%1], %2;" ::"l"(dstGlobal),
                 "r"(src), "r"(bytes)
                 : "memory");
  } else {
    static_assert(detail::bulkDependentFalse<T>, "Unsupported bulk reduction type or operation");
  }
}

/// Close the calling thread's open bulk group, committing every bulkStore() and bulkReduceStore()
/// issued since the previous bulkStoreCommit().
MSCCLPP_DEVICE_INLINE void bulkStoreCommit() { asm volatile("cp.async.bulk.commit_group;" ::: "memory"); }

/// Wait until at most @p PendingGroups committed bulk groups have yet to complete.
///
/// Completion means the data has reached the destination. Wait for this before signaling a peer that
/// the data is available. To reuse the source shared memory only, bulkStoreWaitSource() is cheaper.
/// @tparam PendingGroups Number of most recently committed groups allowed to remain outstanding.
template <uint32_t PendingGroups = 0>
MSCCLPP_DEVICE_INLINE void bulkStoreWait() {
  asm volatile("cp.async.bulk.wait_group %0;" ::"n"(PendingGroups) : "memory");
}

/// Wait until at most @p PendingGroups committed bulk groups may still read their source.
///
/// Weaker than bulkStoreWait(): it guarantees only that the source shared memory can be overwritten,
/// not that the data has reached the destination. This is what a double-buffered store pipeline
/// needs between refills, and it lets stores stay in flight while the next tile is being staged.
/// @tparam PendingGroups Number of most recently committed groups allowed to still read their source.
template <uint32_t PendingGroups = 0>
MSCCLPP_DEVICE_INLINE void bulkStoreWaitSource() {
  asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(PendingGroups) : "memory");
}

/// Order the calling thread's generic shared memory accesses against bulk copies.
///
/// Bulk copies reach shared memory through a separate proxy, so they are not ordered against generic
/// accesses by default. Run this after a bulk load completes and before reading its destination,
/// after writing a source and before issuing a bulk store from it, and after initializing a barrier
/// with relaxedInit(). Every thread that touches the shared memory must run it, unless a thread
/// barrier such as __syncthreads() already orders that thread against one that did.
MSCCLPP_DEVICE_INLINE void bulkFence() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

#endif  // MSCCLPP_BULK_AVAILABLE

}  // namespace mscclpp

#endif  // MSCCLPP_BULK_DEVICE_HPP_
