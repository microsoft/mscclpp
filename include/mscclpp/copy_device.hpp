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

#endif  // defined(MSCCLPP_DEVICE_COMPILE)

}  // namespace mscclpp

#endif  // MSCCLPP_COPY_DEVICE_HPP_
