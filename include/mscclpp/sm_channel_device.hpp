// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_SM_CHANNEL_DEVICE_HPP_
#define MSCCLPP_SM_CHANNEL_DEVICE_HPP_

#include "packet.hpp"
#include "poll.hpp"
#include "semaphore_device.hpp"

namespace mscclpp {

#ifdef __CUDACC__

namespace Element {

/// Load an element from DRAM.
///
/// This is a warpper of ld.volatile.global.* PTX instruction. Address alignment is not this function's
/// responsibility.
///
/// @param v The value to be loaded.
/// @param p The address of the value to be loaded.
///
template <typename T>
__forceinline__ __device__ void load(T& v, const T* p) {
  // We should only use the specialized functions.
  __assert_fail("Unsupported type", __FILE__, __LINE__, __PRETTY_FUNCTION__);
}

/// Write an element on DRAM.
///
/// This is a wrapper of st.volatile.global.* PTX instruction. Address alignment is not this function's
/// responsibility.
///
/// @param p The address of the value to be written.
/// @param v The value to be written.
///
template <typename T>
__forceinline__ __device__ void store(T* p, const T& v) {
  // We should only use the specialized functions.
  __assert_fail("Unsupported type", __FILE__, __LINE__, __PRETTY_FUNCTION__);
}

/// Copy aligned elements from the source memory to the destination memory.
///
/// This function is intended to be collectively called by multiple threads. Each thread copies a part of
/// elements.
///
/// @param dst The destination address.
/// @param src The source address.
/// @param numElems The number of elements to be copied.
/// @param threadId The index of the current thread among all threads running this function. This is different
/// from the `threadIdx` in CUDA.
/// @param numThreads The total number of threads that run this function.
///
template <typename T>
__forceinline__ __device__ void copy(T* dst, T* src, uint64_t numElems, uint32_t threadId, uint32_t numThreads) {
  T reg;
  for (size_t i = threadId; i < numElems; i += numThreads) {
    // Load to register first.
    load(reg, src + i);
    store(dst + i, reg);
  }
}

template <>
__forceinline__ __device__ void load<long long>(long long& v, const long long* p) {
  asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
}

template <>
__forceinline__ __device__ void store<long long>(long long* p, const long long& v) {
  asm volatile("st.volatile.global.u64 [%0], %1;" : : "l"(p), "l"(v) : "memory");
}

template <>
__forceinline__ __device__ void load<int>(int& v, const int* p) {
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
}

template <>
__forceinline__ __device__ void store<int>(int* p, const int& v) {
  asm volatile("st.volatile.global.u32 [%0], %1;" : : "l"(p), "r"(v) : "memory");
}

template <>
__forceinline__ __device__ void load<longlong2>(longlong2& v, const longlong2* p) {
  asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
}

template <>
__forceinline__ __device__ void store<longlong2>(longlong2* p, const longlong2& v) {
  asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" : : "l"(p), "l"(v.x), "l"(v.y) : "memory");
}

template <>
__forceinline__ __device__ void load<int4>(int4& v, const int4* p) {
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.w), "=r"(v.x), "=r"(v.y), "=r"(v.z)
               : "l"(p)
               : "memory");
}

template <>
__forceinline__ __device__ void store<int4>(int4* p, const int4& v) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};"
               :
               : "l"(p), "r"(v.w), "r"(v.x), "r"(v.y), "r"(v.z)
               : "memory");
}

}  // namespace Element

#endif  // __CUDACC__

/// Channel for accessing peer memory directly from SM.
struct SmChannelDeviceHandle {
  SmDevice2DeviceSemaphoreDeviceHandle semaphore_;
  void* src_;
  void* dst_;
  void* getPacketBuffer_;

#ifdef __CUDACC__
  /// Load a value from the remote memory.
  /// @tparam T The type of the value to be loaded.
  /// @param index The index of the value to be loaded. The offset in bytes is calculated as index * sizeof(T).
  /// @return The value loaded.
  template <typename T>
  __forceinline__ __device__ T read(uint64_t index) {
    T v;
    Element::load<T>(v, (T*)dst_ + index);
    return v;
  }

  /// Write a value to the remote memory.
  /// @tparam T The type of the value to be written.
  /// @param index The index of the value to be written. The offset in bytes is calculated as index * sizeof(T).
  /// @param v The value to be written.
  template <typename T>
  __forceinline__ __device__ void write(uint64_t index, const T& v) {
    Element::store<T>((T*)dst_ + index, v);
  }

  /// this is a helper for copy function
  template <typename T, bool CopyRemainder = true>
  __forceinline__ __device__ void copy_helper(void* dst, void* src, uint64_t bytes, uint32_t threadId,
                                              uint32_t numThreads) {
    int* dstInt = reinterpret_cast<int*>(dst);
    int* srcInt = reinterpret_cast<int*>(src);
    const uintptr_t dstPtr = reinterpret_cast<uintptr_t>(dst);
    const uintptr_t srcPtr = reinterpret_cast<uintptr_t>(src);
    const uint64_t numInt = bytes / sizeof(int);
    T* dstElem = reinterpret_cast<T*>((dstPtr + sizeof(T) - 1) / sizeof(T) * sizeof(T));
    T* srcElem = reinterpret_cast<T*>((srcPtr + sizeof(T) - 1) / sizeof(T) * sizeof(T));
    uint64_t nFirstInt = (reinterpret_cast<uintptr_t>(dstElem) - dstPtr) / sizeof(int);
    if (CopyRemainder) {
      // Copy the remainder integers at the beginning.
      Element::copy<int>(dstInt, srcInt, nFirstInt, threadId, numThreads);
    }
    // Copy elements.
    constexpr uint64_t nIntPerElem = sizeof(T) / sizeof(int);
    uint64_t nElem = (numInt - nFirstInt) / nIntPerElem;
    Element::copy<T>(dstElem, srcElem, nElem, threadId, numThreads);
    if (CopyRemainder && nIntPerElem > 1) {
      // Copy the remainder integers at the end.
      uint64_t nLastInt = (numInt - nFirstInt) % nIntPerElem;
      Element::copy<int>(dstInt + nFirstInt + nElem * nIntPerElem, srcInt + nFirstInt + nElem * nIntPerElem, nLastInt,
                         threadId, numThreads);
    }
  }

  /// Copy aligned data from the source memory to the destination memory.
  ///
  /// This function is a warpper of Element<T>::copy(). Unlike Element<T>::copy(), this function can copy remainder
  /// bytes when @p CopyRemainder is true. Still, the  16.
  /// @tparam CopyRemainder Whether to copy remainder bytes when the number of bytes is not a multiple of @p
  /// Alignment.
  /// @param dst The destination address. Should be aligned to @p Alignment in the same way as @p src.
  /// @param src The source address. Should be aligned to @p Alignment in the same way as @p dst.
  /// @param bytes Bytes of the data to be copied. Should be a multiple of @p Alignment.
  /// @param threadId The index of the current thread among all threads running this function. This is different from
  /// the `threadIdx` in CUDA.
  /// @param numThreads The total number of threads that run this function.
  ///
  template <int Alignment = 16, bool CopyRemainder = true>
  __forceinline__ __device__ void copy(void* dst, void* src, uint64_t bytes, uint32_t threadId, uint32_t numThreads) {
    if (Alignment == 4) {
      copy_helper<int, CopyRemainder>(dst, src, bytes, threadId, numThreads);
    } else if (Alignment == 8) {
      copy_helper<long long, CopyRemainder>(dst, src, bytes, threadId, numThreads);
    } else if (Alignment == 16) {
      copy_helper<longlong2, CopyRemainder>(dst, src, bytes, threadId, numThreads);
    } else {
      static_assert(Alignment == 4 || Alignment == 8 || Alignment == 16, "Unsupported alignment");
    }
  }

  /// Copy data from the local memory to the remote memory.
  ///
  /// This function is intended to be collectively called by multiple threads. Each thread copies a part of data.
  ///
  /// @tparam Alignment The alignment of the source and destination addresses. Should be 4, 8, or a multiple of 16.
  /// @tparam CopyRemainder Whether to copy remainder bytes when the number of bytes is not a multiple of @p
  /// Alignment.
  /// @param dstOffset The offset in bytes of the remote address. Should be a multiple of @p Alignment.
  /// @param srcOffset The offset in bytes of the local address. Should be a multiple of @p Alignment.
  /// @param bytes Bytes of the data to be copied. Should be a multiple of @p Alignment.
  /// @param threadId The index of the current thread among all threads running this function. This is different from
  /// the `threadIdx` in CUDA.
  /// @param numThreads The total number of threads that run this function.
  ///
  template <int Alignment = 16, bool CopyRemainder = true>
  __forceinline__ __device__ void put(uint64_t dstOffset, uint64_t srcOffset, uint64_t bytes, uint32_t threadId,
                                      uint32_t numThreads) {
    copy<Alignment, CopyRemainder>((char*)dst_ + dstOffset, (char*)src_ + srcOffset, bytes, threadId, numThreads);
  }

  /// Copy data from the remote memory to the local memory.
  ///
  /// This function is intended to be collectively called by multiple threads. Each thread copies a part of data.
  ///
  /// @tparam Alignment The alignment of the source and destination addresses. Should be 4, 8, or a multiple of 16.
  /// @tparam CopyRemainder Whether to copy remainder bytes when the number of bytes is not a multiple of @p
  /// Alignment.
  /// @param dstOffset The offset in bytes of the remote address. Should be a multiple of @p Alignment.
  /// @param srcOffset The offset in bytes of the local address. Should be a multiple of @p Alignment.
  /// @param bytes Bytes of the data to be copied. Should be a multiple of @p Alignment.
  /// @param threadId The index of the current thread among all threads running this function. This is different from
  /// the `threadIdx` in CUDA.
  /// @param numThreads The total number of threads that run this function.
  ///
  template <int Alignment = 16, bool CopyRemainder = true>
  __forceinline__ __device__ void get(uint64_t dstOffset, uint64_t srcOffset, uint64_t bytes, uint32_t threadId,
                                      uint32_t numThreads) {
    // Note that `dst` and `src` are swapped for `get()`.
    copy<Alignment, CopyRemainder>((char*)src_ + srcOffset, (char*)dst_ + dstOffset, bytes, threadId, numThreads);
  }

  /// Copy data from the local memory to the remote memory.
  ///
  /// This function is intended to be collectively called by multiple threads. Each thread copies a part of data.
  ///
  /// @tparam Alignment The alignment of the source and destination addresses. Should be 4, 8, or a multiple of 16.
  /// @tparam CopyRemainder Whether to copy remainder bytes when the number of bytes is not a multiple of @p
  /// Alignment.
  /// @param offset The offset in bytes of the local and remote addresses. Should be a multiple of @p Alignment.
  /// @param bytes Bytes of the data to be copied. Should be a multiple of @p Alignment.
  /// @param threadId The index of the current thread among all threads running this function. This is different from
  /// the `threadIdx` in CUDA.
  /// @param numThreads The total number of threads that run this function.
  ///
  template <int Alignment = 16, bool CopyRemainder = true>
  __forceinline__ __device__ void put(uint64_t offset, uint64_t size, uint32_t threadId, uint32_t numThreads) {
    put<Alignment, CopyRemainder>(offset, offset, size, threadId, numThreads);
  }

  /// Copy data from the remote memory to the local memory.
  ///
  /// This function is intended to be collectively called by multiple threads. Each thread copies a part of data.
  ///
  /// @tparam Alignment The alignment of the source and destination addresses. Should be 4, 8, or a multiple of 16.
  /// @tparam CopyRemainder Whether to copy remainder bytes when the number of bytes is not a multiple of @p
  /// Alignment.
  /// @param offset The offset in bytes of the local and remote addresses. Should be a multiple of @p Alignment.
  /// @param bytes Bytes of the data to be copied. Should be a multiple of @p Alignment.
  /// @param threadId The index of the current thread among all threads running this function. This is different from
  /// the `threadIdx` in CUDA.
  /// @param numThreads The total number of threads that run this function.
  ///
  template <int Alignment = 16, bool CopyRemainder = true>
  __forceinline__ __device__ void get(uint64_t offset, uint64_t size, uint32_t threadId, uint32_t numThreads) {
    get<Alignment, CopyRemainder>(offset, offset, size, threadId, numThreads);
  }

  /// Construct @ref LLPacket from the data in the local memory and write it on the remote memory.
  ///
  /// This function is intended to be collectively called by multiple threads. Each thread copies a part of packets.
  ///
  /// @param dstOffset The offset in bytes of the remote address.
  /// @param srcOffset The offset in bytes of the local address.
  /// @param bytes Bytes of the data to be copied.
  /// @param threadId The index of the current thread among all threads running this function. This is different from
  /// the `threadIdx` in CUDA.
  /// @param numThreads The total number of threads that run this function.
  ///
  __forceinline__ __device__ void putPackets(uint64_t dstOffset, uint64_t srcOffset, uint64_t bytes, uint32_t threadId,
                                             uint32_t numThreads, uint32_t flag) {
    mscclpp::putPackets(dst_, dstOffset, src_, srcOffset, bytes, threadId, numThreads, flag);
  }

  /// Retrieve data from @ref LLPacket in the local packet buffer and write it on the local memory.
  ///
  /// This function is intended to be collectively called by multiple threads. Each thread copies a part of data.
  ///
  /// @param dstOffset The offset in bytes of the local memory.
  /// @param srcOffset The offset in bytes of the local packet buffer.
  /// @param bytes Bytes of the data to be copied.
  /// @param threadId The index of the current thread among all threads running this function. This is different from
  /// the `threadIdx` in CUDA.
  /// @param numThreads The total number of threads that run this function.
  ///
  __forceinline__ __device__ void getPackets(uint64_t dstOffset, uint64_t srcOffset, uint64_t bytes, uint32_t threadId,
                                             uint32_t numThreads, uint32_t flag) {
    mscclpp::getPackets(src_, dstOffset, getPacketBuffer_, srcOffset, bytes, threadId, numThreads, flag);
  }

  /// Signal the remote semaphore.
  ///
  /// This function guarantees that all the memory operation before this function is completed before the remote
  /// semaphore is signaled.
  ///
  __forceinline__ __device__ void signal() { semaphore_.signal(); }

  /// Signal the remote semaphore for copied packets.
  ///
  /// Unlike @ref signal(), this function provides no guarantee on the completion of memory operations. This is
  /// intended to be used with @ref putPackets() and @ref getPackets() that use flags inside packets to indicate the
  /// completion of copies.
  ///
  __forceinline__ __device__ void signalPacket() { semaphore_.signalPacket(); }

  /// Increase the counter of the local semaphore.
  __forceinline__ __device__ void semaphoreIncrement() { semaphore_.semaphoreIncrement(); }

  /// Read the counter of the local semaphore.
  __forceinline__ __device__ uint64_t semaphoreGetLocal() const { return semaphore_.semaphoreGetLocal(); }

  /// Wait for the remote semaphore to send a signal.
  __forceinline__ __device__ void wait() { semaphore_.wait(); }
#endif  // __CUDACC__
};

}  // namespace mscclpp

#endif  // MSCCLPP_SM_CHANNEL_DEVICE_HPP_
