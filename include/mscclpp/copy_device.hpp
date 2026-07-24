// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_COPY_DEVICE_HPP_
#define MSCCLPP_COPY_DEVICE_HPP_

#include <cstdint>
#include <type_traits>

#include "device.hpp"

#if defined(MSCCLPP_DEVICE_COMPILE)
#include "packet_device.hpp"
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

namespace mscclpp {
#if defined(MSCCLPP_DEVICE_COMPILE)

namespace detail {

/// Copy aligned elements from the source memory to the destination memory.
///
/// This function is intended to be collectively called by multiple threads. Each thread copies a part of
/// elements.
///
/// @tparam T The type of the elements to be copied.
/// @param dst The destination address.
/// @param src The source address.
/// @param numElems The number of elements to be copied.
/// @param threadId The index of the current thread among all threads running this function.
/// Should be less than @p numThreads.
/// @param numThreads The total number of threads that run this function.
///
template <typename T>
MSCCLPP_DEVICE_INLINE void copy(T* dst, T* src, uint64_t numElems, uint32_t threadId, uint32_t numThreads) {
  T reg;
  for (size_t i = threadId; i < numElems; i += numThreads) {
    // Load to register first.
    reg = src[i];
    // Then store to destination.
    dst[i] = reg;
  }
}

}  // namespace detail

/// Helper function of mscclpp::copy(). Copy data from the source memory to the destination memory.
///
/// This function is intended to be collectively called by multiple threads. Each thread copies a part of
/// elements.
///
/// @note The source and destination addresses do not have to be aligned to the size of @p T, but the misalignment
/// to the size of @p T should be multiple of 4 bytes and should be the same for both source and destination addresses.
/// The behavior of this function is undefined otherwise.
/// @note The number of bytes to be copied should be a multiple of 4 bytes. If the number of bytes is not a multiple
/// of 4 bytes, the remainder bytes will not be copied.
///
/// @tparam T The type of the elements to be copied.
/// @tparam CopyRemainder If false, the function will not copy data that is unaligned to the size of @p T. If true,
/// the function will try to copy the unaligned data with conditions (see the notes).
/// @param dst The destination address.
/// @param src The source address.
/// @param bytes Bytes of the data to be copied. Should be a multiple of 4 bytes.
/// @param threadId The index of the current thread among all threads running this function.
/// Should be less than @p numThreads.
/// @param numThreads The total number of threads that run this function.
///
template <typename T, bool CopyRemainder = true>
MSCCLPP_DEVICE_INLINE void copyHelper(void* dst, void* src, uint64_t bytes, uint32_t threadId, uint32_t numThreads) {
  int* dstInt = reinterpret_cast<int*>(dst);
  int* srcInt = reinterpret_cast<int*>(src);
  const uintptr_t dstPtr = reinterpret_cast<uintptr_t>(dst);
  const uintptr_t srcPtr = reinterpret_cast<uintptr_t>(src);
  const uint64_t numInt = bytes / sizeof(int);
  T* dstElem = reinterpret_cast<T*>((dstPtr + sizeof(T) - 1) / sizeof(T) * sizeof(T));
  T* srcElem = reinterpret_cast<T*>((srcPtr + sizeof(T) - 1) / sizeof(T) * sizeof(T));
  uint64_t nFirstInt = (reinterpret_cast<uintptr_t>(dstElem) - dstPtr) / sizeof(int);
  if constexpr (CopyRemainder) {
    // Copy the remainder integers at the beginning.
    detail::copy<int>(dstInt, srcInt, nFirstInt, threadId, numThreads);
  }
  // Copy elements.
  constexpr uint64_t nIntPerElem = sizeof(T) / sizeof(int);
  uint64_t nElem = (numInt - nFirstInt) / nIntPerElem;
  detail::copy<T>(dstElem, srcElem, nElem, threadId, numThreads);
  if constexpr (CopyRemainder && nIntPerElem > 1) {
    // Copy the remainder integers at the end.
    uint64_t nLastInt = (numInt - nFirstInt) % nIntPerElem;
    detail::copy<int>(dstInt + nFirstInt + nElem * nIntPerElem, srcInt + nFirstInt + nElem * nIntPerElem, nLastInt,
                      threadId, numThreads);
  }
}

/// Copy data from the source memory to the destination memory.
///
/// This function is intended to be collectively called by multiple threads. Each thread copies a part of
/// elements.
///
/// @note The source and destination addresses do not have to be aligned to the @p Alignment value,
/// but the misalignment to @p Alignment should be multiple of 4 bytes and should be the same for both source
/// and destination addresses.
/// The behavior of this function is undefined otherwise.
/// @note The number of bytes to be copied should be a multiple of 4 bytes. If the number of bytes is not a multiple
/// of 4 bytes, the remainder bytes will not be copied.
///
/// @tparam Alignment The alignment of the data to be copied. A larger alignment value is more likely to achieve higher
/// copying throughput. Should be one of 4, 8, or 16.
/// @tparam CopyRemainder If false, the function will not copy data that is unaligned to the @p Alignment value.
/// If true, the function will try to copy the unaligned data with conditions (see the notes).
/// @param dst The destination address.
/// @param src The source address.
/// @param bytes Bytes of the data to be copied. Should be a multiple of 4 bytes.
/// @param threadId The index of the current thread among all threads running this function.
/// Should be less than @p numThreads.
/// @param numThreads The total number of threads that run this function.
///
template <int Alignment = 16, bool CopyRemainder = true>
MSCCLPP_DEVICE_INLINE void copy(void* dst, void* src, uint64_t bytes, uint32_t threadId, uint32_t numThreads) {
  if constexpr (Alignment == 4) {
    copyHelper<int, CopyRemainder>(dst, src, bytes, threadId, numThreads);
  } else if constexpr (Alignment == 8) {
    copyHelper<long long, CopyRemainder>(dst, src, bytes, threadId, numThreads);
  } else if constexpr (Alignment == 16) {
    copyHelper<longlong2, CopyRemainder>(dst, src, bytes, threadId, numThreads);
  } else {
    static_assert(Alignment == 4 || Alignment == 8 || Alignment == 16, "Unsupported alignment");
  }
}

/// Write a value to the destination memory at the specified index.
template <typename T>
MSCCLPP_DEVICE_INLINE void write(void* dst, uint64_t index, const T& v) {
  *(reinterpret_cast<T*>(dst) + index) = v;
}

/// Read a value from the source memory at the specified index.
template <typename T>
MSCCLPP_DEVICE_INLINE T read(void* src, uint64_t index) {
  return *(reinterpret_cast<T*>(src) + index);
}

/// Read data from the origin and write packets to the target buffer.
///
/// This function is intended to be collectively called by multiple threads. Each thread copies a part of
/// packets.
///
/// @tparam PacketType The packet type. It should be either LL16Packet or LL8Packet.
/// @param targetPtr The target buffer.
/// @param originPtr The origin buffer.
/// @param originBytes The number of bytes to write to the target buffer.
/// @param threadId The index of the current thread among all threads running this function.
/// Should be less than @p numThreads.
/// @param numThreads The total number of threads that run this function.
/// @param flag The flag to write in the packets.
///
template <typename PacketType = LL16Packet>
MSCCLPP_DEVICE_INLINE void copyToPackets(void* targetPtr, const void* originPtr, uint64_t originBytes,
                                         uint32_t threadId, uint32_t numThreads, uint32_t flag);

template <>
MSCCLPP_DEVICE_INLINE void copyToPackets<LL16Packet>(void* targetPtr, const void* originPtr, uint64_t originBytes,
                                                     uint32_t threadId, uint32_t numThreads, uint32_t flag) {
  // Offsets should be aligned to 8 bytes & size should be a multiple of 8 bytes
  const uint32_t* originBase = reinterpret_cast<const uint32_t*>(originPtr);
  LL16Packet* targetBase = reinterpret_cast<LL16Packet*>(targetPtr);
  size_t nElem = originBytes / sizeof(uint64_t);
  for (size_t i = threadId; i < nElem; i += numThreads) {
    LL16Packet* pkt = &targetBase[i];
    pkt->write(originBase[2 * i], originBase[2 * i + 1], flag);
  }
}

template <>
MSCCLPP_DEVICE_INLINE void copyToPackets<LL8Packet>(void* targetPtr, const void* originPtr, uint64_t originBytes,
                                                    uint32_t threadId, uint32_t numThreads, uint32_t flag) {
  // Offsets should be aligned to 4 bytes & size should be a multiple of 4 bytes
  const uint32_t* originBase = reinterpret_cast<const uint32_t*>(originPtr);
  LL8Packet* targetBase = reinterpret_cast<LL8Packet*>(targetPtr);
  size_t nElem = originBytes / sizeof(uint32_t);
  for (size_t i = threadId; i < nElem; i += numThreads) {
    LL8Packet* pkt = &targetBase[i];
    pkt->write(originBase[i], flag);
  }
}

/// Read packets from the target buffer and write retrieved data to the origin.
///
/// This function is intended to be collectively called by multiple threads. Each thread reads a part of
/// packets.
///
/// @tparam PacketType The packet type. It should be either LL16Packet or LL8Packet.
/// @param originPtr The origin buffer.
/// @param targetPtr The target buffer.
/// @param originBytes The number of bytes to read from the origin buffer.
/// @param threadId The index of the current thread among all threads running this function.
/// Should be less than @p numThreads.
/// @param numThreads The total number of threads that run this function.
/// @param flag The flag to write in the packets.
/// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
///
template <typename PacketType = LL16Packet>
MSCCLPP_DEVICE_INLINE void copyFromPackets(void* originPtr, const void* targetPtr, uint64_t originBytes,
                                           uint32_t threadId, uint32_t numThreads, uint32_t flag,
                                           int64_t maxSpinCount = -1);

template <>
MSCCLPP_DEVICE_INLINE void copyFromPackets<LL16Packet>(void* originPtr, const void* targetPtr, uint64_t originBytes,
                                                       uint32_t threadId, uint32_t numThreads, uint32_t flag,
                                                       int64_t maxSpinCount) {
  // Offsets should be aligned to 8 bytes & size should be a multiple of 8 bytes
  const LL16Packet* targetBase = reinterpret_cast<const LL16Packet*>(targetPtr);
  uint2* originBase = reinterpret_cast<uint2*>(originPtr);
  size_t nElem = originBytes / sizeof(uint2);
  for (size_t i = threadId; i < nElem; i += numThreads) {
    const LL16Packet* pkt = &targetBase[i];
    originBase[i] = pkt->read(flag, maxSpinCount);
  }
}

template <>
MSCCLPP_DEVICE_INLINE void copyFromPackets<LL8Packet>(void* originPtr, const void* targetPtr, uint64_t originBytes,
                                                      uint32_t threadId, uint32_t numThreads, uint32_t flag,
                                                      int64_t maxSpinCount) {
  // Offsets should be aligned to 4 bytes & size should be a multiple of 4 bytes
  const LL8Packet* targetBase = reinterpret_cast<const LL8Packet*>(targetPtr);
  uint32_t* originBase = reinterpret_cast<uint32_t*>(originPtr);
  size_t nElem = originBytes / sizeof(uint32_t);
  for (size_t i = threadId; i < nElem; i += numThreads) {
    const LL8Packet* pkt = &targetBase[i];
    originBase[i] = pkt->read(flag, maxSpinCount);
  }
}

// ------------------------------------------------------------------------------------------------
// Internal TMA (Tensor Memory Accelerator) bulk-load helpers (NVIDIA sm_90+).
//
// These live in `detail` and are NOT user-facing: TMA is NVIDIA-specific, so the public API exposes
// bulk loads only through channel methods with architecture-neutral names (e.g.
// MemoryChannelDeviceHandle::getBulk plus the mscclpp::BulkLoad completion object). These
// helpers offload contiguous global->shared loads to the async copy engine via `cp.async.bulk`,
// with completion tracked by a caller-owned mbarrier, so they compose into multi-source gathers and
// multi-stage pipelines.
//
// Availability: NVIDIA sm_90+ (Hopper and newer). On other targets the helpers are no-ops.
// All are leader-only: call from a single thread (e.g. one lane per warp).
// ------------------------------------------------------------------------------------------------

// True on device compilation targets where `cp.async.bulk` is available (NVIDIA sm_90+). Internal
// only: this token names a hardware feature, so it must never escape `detail` into user code. It is
// #undef-ed at the end of the internal region below.
#if defined(MSCCLPP_DEVICE_CUDA) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
#define MSCCLPP_DETAIL_BULK_LOAD_AVAILABLE 1
#else
#define MSCCLPP_DETAIL_BULK_LOAD_AVAILABLE 0
#endif

namespace detail {

#if defined(MSCCLPP_DEVICE_CUDA)

/// Initialize a shared-memory mbarrier with the given arrival count.
MSCCLPP_DEVICE_INLINE void mbarInit(uint64_t* mbar, uint32_t count) {
#if MSCCLPP_DETAIL_BULK_LOAD_AVAILABLE
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(addr), "r"(count));
#else
  (void)mbar;
  (void)count;
#endif
}

/// Arm the mbarrier to expect @p bytes of asynchronous transaction bytes.
MSCCLPP_DEVICE_INLINE void mbarExpectTx(uint64_t* mbar, uint32_t bytes) {
#if MSCCLPP_DETAIL_BULK_LOAD_AVAILABLE
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(addr), "r"(bytes));
#else
  (void)mbar;
  (void)bytes;
#endif
}

/// Issue an async bulk load: global (@p gptr) -> shared (@p smem), completion tracked by @p mbar.
MSCCLPP_DEVICE_INLINE void tmaLoad(void* smem, const void* gptr, uint32_t bytes, uint64_t* mbar) {
#if MSCCLPP_DETAIL_BULK_LOAD_AVAILABLE
  uint32_t smemAddr = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  uint32_t mbarAddr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(smemAddr),
      "l"(gptr), "r"(bytes), "r"(mbarAddr)
      : "memory");
#else
  (void)smem;
  (void)gptr;
  (void)bytes;
  (void)mbar;
#endif
}

/// Spin until the mbarrier reaches the given @p phase (0/1 parity).
MSCCLPP_DEVICE_INLINE void mbarWait(uint64_t* mbar, uint32_t phase) {
#if MSCCLPP_DETAIL_BULK_LOAD_AVAILABLE
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  uint32_t done = 0;
  do {
    asm volatile("{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p;}"
                 : "=r"(done)
                 : "r"(addr), "r"(phase));
  } while (!done);
#else
  (void)mbar;
  (void)phase;
#endif
}

/// Fence so that data delivered by an async (TMA) proxy into shared memory becomes visible to
/// subsequent generic shared-memory reads by the calling thread.
MSCCLPP_DEVICE_INLINE void asyncProxyFence() {
#if MSCCLPP_DETAIL_BULK_LOAD_AVAILABLE
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
}

/// Single source of truth for bulk-load availability on the current device target. The channel
/// handle's bulkSupported() forwards here so the sm_90 threshold lives in exactly one place.
MSCCLPP_DEVICE_INLINE constexpr bool bulkLoadAvailable() { return MSCCLPP_DETAIL_BULK_LOAD_AVAILABLE; }

#endif  // defined(MSCCLPP_DEVICE_CUDA)

#undef MSCCLPP_DETAIL_BULK_LOAD_AVAILABLE

}  // namespace detail

#endif  // defined(MSCCLPP_DEVICE_COMPILE)

}  // namespace mscclpp

#endif  // MSCCLPP_COPY_DEVICE_HPP_
