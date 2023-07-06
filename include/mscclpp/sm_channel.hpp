// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_SM_CHANNEL_HPP_
#define MSCCLPP_SM_CHANNEL_HPP_

#include <mscclpp/core.hpp>
#include <mscclpp/packet.hpp>
#include <mscclpp/semaphore.hpp>
#include <type_traits>

namespace mscclpp {

/// Channel for accessing peer memory directly from SM.
struct SmChannel {
 private:
  SmDevice2DeviceSemaphore::DeviceHandle semaphore_;
  void* src_;
  void* dst_;
  void* getPacketBuffer_;

#ifdef __CUDACC__
  /// Helper for aligned data type access.
  /// @tparam T The type to be checked.
  template <typename T>
  struct Element {
    static constexpr bool is4B = (sizeof(T) == 4);
    static constexpr bool is8B = (sizeof(T) == 8);
    static constexpr bool is4Bx2 =
        (std::is_same<T, int2>::value || std::is_same<T, uint2>::value || std::is_same<T, float2>::value);
    static constexpr bool is4Bx4 =
        (std::is_same<T, int4>::value || std::is_same<T, uint4>::value || std::is_same<T, float4>::value);
    static constexpr bool is8Bx2 =
        (std::is_same<T, longlong2>::value || std::is_same<T, ulonglong2>::value || std::is_same<T, double2>::value);
    // Note: we do not support long2 and ulong2 as their size may differ on different platforms.
    static constexpr bool isValid = (is4B || is8B || is4Bx2 || is4Bx4 || is8Bx2);

    /// Load an element from DRAM.
    ///
    /// This is a warpper of ld.volatile.global.* PTX instruction. Address alignment is not this function's
    /// responsibility.
    ///
    /// @tparam T The type of the value to be loaded.
    /// @param v The value to be loaded.
    /// @param p The address of the value to be loaded.
    ///
    static __forceinline__ __device__ void load(T& v, const T* p) {
      if constexpr (is4B) {
        asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
      } else if constexpr (is8B) {
        asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
      } else if constexpr (is4Bx2) {
        asm volatile("ld.volatile.global.v2.u32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p) : "memory");
      } else if constexpr (is4Bx4) {
        asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.w), "=r"(v.x), "=r"(v.y), "=r"(v.z)
                     : "l"(p)
                     : "memory");
      } else if constexpr (is8Bx2) {
        asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
      }
      static_assert(isValid, "Unsupported type T");
    }

    /// Write an element on DRAM.
    ///
    /// This is a wrapper of st.volatile.global.* PTX instruction. Address alignment is not this function's
    /// responsibility.
    ///
    /// @tparam T The type of the value to be written.
    /// @param p The address of the value to be written.
    /// @param v The value to be written.
    ///
    static __forceinline__ __device__ void store(T* p, const T& v) {
      if constexpr (is4B) {
        asm volatile("st.volatile.global.u32 [%0], %1;" : : "l"(p), "r"(v) : "memory");
      } else if constexpr (is8B) {
        asm volatile("st.volatile.global.u64 [%0], %1;" : : "l"(p), "l"(v) : "memory");
      } else if constexpr (is4Bx2) {
        asm volatile("st.volatile.global.v2.u32 [%0], {%1,%2};" : : "l"(p), "r"(v.x), "r"(v.y) : "memory");
      } else if constexpr (is4Bx4) {
        asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};"
                     :
                     : "l"(p), "r"(v.w), "r"(v.x), "r"(v.y), "r"(v.z)
                     : "memory");
      } else if constexpr (is8Bx2) {
        asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" : : "l"(p), "l"(v.x), "l"(v.y) : "memory");
      }
      static_assert(isValid, "Unsupported type T");
    }

    /// Copy aligned elements from the source memory to the destination memory.
    ///
    /// This function is intended to be collectively called by multiple threads. Each thread copies a part of elements.
    ///
    /// @tparam T The type of the elements to be copied.
    /// @param dst The destination address.
    /// @param src The source address.
    /// @param numElems The number of elements to be copied.
    /// @param threadId The index of the current thread among all threads running this function. This is different from
    /// the `threadIdx` in CUDA.
    /// @param numThreads The total number of threads that run this function.
    ///
    static __forceinline__ __device__ void copy(T* dst, T* src, uint64_t numElems, uint32_t threadId,
                                                uint32_t numThreads) {
      T reg;
      for (size_t i = threadId; i < numElems; i += numThreads) {
        // Load to register first.
        load(reg, src + i);
        store(dst + i, reg);
      }
    }
  };
#endif  // __CUDACC__

 public:
  SmChannel() = default;
  SmChannel(SmDevice2DeviceSemaphore::DeviceHandle semaphore, RegisteredMemory dst, void* src,
            void* getPacketBuffer = nullptr);

#ifdef __CUDACC__
  /// Load a value from the remote memory.
  /// @tparam T The type of the value to be loaded.
  /// @param index The index of the value to be loaded. The offset in bytes is calculated as index * sizeof(T).
  /// @return The value loaded.
  template <typename T>
  __forceinline__ __device__ T read(uint64_t index) {
    T v;
    Element<T>::load(v, (T*)dst_ + index);
    return v;
  }

  /// Write a value to the remote memory.
  /// @tparam T The type of the value to be written.
  /// @param index The index of the value to be written. The offset in bytes is calculated as index * sizeof(T).
  /// @param v The value to be written.
  template <typename T>
  __forceinline__ __device__ void write(uint64_t index, const T& v) {
    Element<T>::store((T*)dst_ + index, v);
  }

  /// Copy 4-byte aligned data from the source memory to the destination memory.
  ///
  /// This function is a warpper of @ref Element<T>::copy(). Unlike @ref Element<T>::copy(), this function does not
  /// require complete alignment of the source and destination addresses. Still, the source and destination addresses
  /// must be 4-byte aligned and the number of copying bytes must be a multiple of 4. @ref copy() tries to use larger
  /// data types (up to 16-byte types) to copy more data at once.
  ///
  /// @tparam T The type of the elements to be copied.
  /// @param dst The destination address. Should be 4-byte aligned.
  /// @param src The source address. Should be 4-byte aligned.
  /// @param bytes Bytes of the data to be copied. Should be a multiple of 4.
  /// @param threadId The index of the current thread among all threads running this function. This is different from
  /// the `threadIdx` in CUDA.
  /// @param numThreads The total number of threads that run this function.
  ///
  __forceinline__ __device__ void copy(void* dst, void* src, uint64_t bytes, uint32_t threadId, uint32_t numThreads) {
    // Align `dst` and `src` into 4-byte integers.
    int* dstInt = reinterpret_cast<int*>(dst);
    int* srcInt = reinterpret_cast<int*>(src);
    const uintptr_t dstPtr = reinterpret_cast<uintptr_t>(dst);
    const uintptr_t srcPtr = reinterpret_cast<uintptr_t>(src);
    const uint64_t numInt = bytes / sizeof(int);
    if (numInt <= 4) {
      // Handle this case separately to make the code for other cases simpler.
      Element<int>::copy(dstInt, srcInt, numInt, threadId, numThreads);
    } else if (dstPtr % 16 == srcPtr % 16) {
      // 16-byte aligned.
      longlong2* dstLongLong2 = reinterpret_cast<longlong2*>((dstPtr + 15) / 16 * 16);
      longlong2* srcLongLong2 = reinterpret_cast<longlong2*>((srcPtr + 15) / 16 * 16);
      // Copy the first 0-3 integers.
      uint64_t nFirstInt = (reinterpret_cast<uintptr_t>(dstLongLong2) - dstPtr) / sizeof(int);
      Element<int>::copy(dstInt, srcInt, nFirstInt, threadId, numThreads);
      // Copy 16-byte elements.
      uint64_t nElem = (numInt - nFirstInt) / 4;
      Element<longlong2>::copy(dstLongLong2, srcLongLong2, nElem, threadId, numThreads);
      // Copy the last 0-3 integers.
      uint64_t nLastInt = (numInt - nFirstInt) % 4;
      Element<int>::copy(dstInt + nFirstInt + nElem * 4, srcInt + nFirstInt + nElem * 4, nLastInt, threadId,
                         numThreads);
    } else if (dstPtr % 8 == srcPtr % 8) {
      // 8-byte aligned.
      uint64_t* dstInt64 = reinterpret_cast<uint64_t*>((dstPtr + 7) / 8 * 8);
      uint64_t* srcInt64 = reinterpret_cast<uint64_t*>((srcPtr + 7) / 8 * 8);
      // Copy the first 0-1 integer.
      uint64_t nFirstInt = (reinterpret_cast<uintptr_t>(dstInt64) - dstPtr) / sizeof(int);
      Element<int>::copy(dstInt, srcInt, nFirstInt, threadId, numThreads);
      // Copy 8-byte elements.
      uint64_t nElem = (numInt - nFirstInt) / 2;
      Element<uint64_t>::copy(dstInt64, srcInt64, nElem, threadId, numThreads);
      // Copy the last 0-1 integer.
      uint64_t nLastInt = (numInt - nFirstInt) % 2;
      Element<int>::copy(dstInt + nFirstInt + nElem * 2, srcInt + nFirstInt + nElem * 2, nLastInt, threadId,
                         numThreads);
    } else {
      // Only 4-byte aligned.
      Element<int>::copy(dstInt, srcInt, numInt, threadId, numThreads);
    }
  }

  /// Copy data from the local memory to the remote memory.
  ///
  /// This function is intended to be collectively called by multiple threads. Each thread copies a part of data.
  ///
  /// @param dstOffset The offset in bytes of the remote address. Should be a multiple of 4.
  /// @param srcOffset The offset in bytes of the local address. Should be a multiple of 4.
  /// @param bytes Bytes of the data to be copied. Should be a multiple of 4.
  /// @param threadId The index of the current thread among all threads running this function. This is different from
  /// the `threadIdx` in CUDA.
  /// @param numThreads The total number of threads that run this function.
  __forceinline__ __device__ void put(uint64_t dstOffset, uint64_t srcOffset, uint64_t bytes, uint32_t threadId,
                                      uint32_t numThreads) {
    copy((char*)dst_ + dstOffset, (char*)src_ + srcOffset, bytes, threadId, numThreads);
  }

  /// Copy data from the remote memory to the local memory.
  ///
  /// This function is intended to be collectively called by multiple threads. Each thread copies a part of data.
  ///
  /// @param dstOffset The offset in bytes of the remote address. Should be a multiple of 4.
  /// @param srcOffset The offset in bytes of the local address. Should be a multiple of 4.
  /// @param bytes Bytes of the data to be copied. Should be a multiple of 4.
  /// @param threadId The index of the current thread among all threads running this function. This is different from
  /// the `threadIdx` in CUDA.
  /// @param numThreads The total number of threads that run this function.
  __forceinline__ __device__ void get(uint64_t dstOffset, uint64_t srcOffset, uint64_t bytes, uint32_t threadId,
                                      uint32_t numThreads) {
    // Note that `dst` and `src` are swapped for `get()`.
    copy((char*)src_ + srcOffset, (char*)dst_ + dstOffset, bytes, threadId, numThreads);
  }

  __forceinline__ __device__ void put(uint64_t offset, uint64_t size, uint32_t threadId, uint32_t numThreads) {
    put(offset, offset, size, threadId, numThreads);
  }

  __forceinline__ __device__ void get(uint64_t offset, uint64_t size, uint32_t threadId, uint32_t numThreads) {
    get(offset, offset, size, threadId, numThreads);
  }

  __forceinline__ __device__ void putPackets(uint64_t dstOffset, uint64_t srcOffset, uint64_t size, uint32_t threadId,
                                             uint32_t numThreads, uint32_t flag) {
    mscclpp::putPackets(dst_, dstOffset, src_, srcOffset, size, threadId, numThreads, flag);
  }

  __forceinline__ __device__ void getPackets(uint64_t dstOffset, uint64_t srcOffset, uint64_t size, uint32_t threadId,
                                             uint32_t numThreads, uint32_t flag) {
    mscclpp::getPackets(src_, dstOffset, getPacketBuffer_, srcOffset, size, threadId, numThreads, flag);
  }

  __forceinline__ __device__ void signal() { semaphore_.signal(); }

  __forceinline__ __device__ void signalPacket() { semaphore_.signalPacket(); }

  __forceinline__ __device__ void semaphoreIncrement() { semaphore_.semaphoreIncrement(); }

  __forceinline__ __device__ uint64_t semaphoreGetLocal() const { return semaphore_.semaphoreGetLocal(); }

  __forceinline__ __device__ void wait() { semaphore_.wait(); }
#endif  // __CUDACC__
};

}  // namespace mscclpp

#endif  // MSCCLPP_SM_CHANNEL_HPP_
