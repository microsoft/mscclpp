// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_SM_CHANNEL_HPP_
#define MSCCLPP_SM_CHANNEL_HPP_

#include <mscclpp/core.hpp>
#include <mscclpp/packet.hpp>
#include <mscclpp/semaphore.hpp>

namespace mscclpp {

// A direct version of DeviceChannel only for CudaIpc
struct SmChannel {
 public:
  SmChannel() = default;
  SmChannel(SmDevice2DeviceSemaphore::DeviceHandle semaphore, RegisteredMemory dst, void* src,
            void* getPacketBuffer = nullptr);

#ifdef __CUDACC__

  __forceinline__ __device__ void fetch128(ulong2& v, const ulong2* p) {
    asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
  }
  __forceinline__ __device__ void store128(ulong2* p, ulong2& v) {
    asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" ::"l"(p), "l"(v.x), "l"(v.y) : "memory");
  }
#define UNROLL 4
  __forceinline__ __device__ void put(uint64_t dstOffset, uint64_t srcOffset, uint64_t size, uint32_t threadId,
                                      uint32_t numThreads) {
    constexpr int WARP_SIZE = 32;
    // assume the memory is aligned to 8 bytes
    ulong2* srcAddr = (ulong2*)((char*)src_ + srcOffset);
    ulong2* dstAddr = (ulong2*)((char*)dst_ + dstOffset);
    ulong2 ele[UNROLL];
    int warpId = threadId / WARP_SIZE;
    int tidInWarp = threadId % WARP_SIZE;
    size_t offset = warpId * WARP_SIZE * UNROLL + tidInWarp;
    size_t nElem = size % sizeof(ulong2) ? (size + sizeof(ulong2)) / sizeof(ulong2) : size / sizeof(ulong2);
    for (size_t i = offset; i < nElem; i += numThreads * UNROLL) {
// load to register first
#pragma unroll
      for (int j = 0; j < UNROLL; j++) {
        fetch128(ele[j], srcAddr + i + j * WARP_SIZE);
      }
#pragma unroll
      for (int j = 0; j < UNROLL; j++) {
        store128(dstAddr + i + j * WARP_SIZE, ele[j]);
      }
    }
  }

  __forceinline__ __device__ void get(uint64_t dstOffset, uint64_t srcOffset, uint64_t size, uint32_t threadId,
                                      uint32_t numThreads) {
    constexpr int WARP_SIZE = 32;
    // assume the memory is aligned to 8 bytes
    ulong2* srcAddr = (ulong2*)((char*)src_ + srcOffset);
    ulong2* dstAddr = (ulong2*)((char*)dst_ + dstOffset);
    ulong2 ele[UNROLL];
    int warpId = threadId / WARP_SIZE;
    int tidInWarp = threadId % WARP_SIZE;
    size_t offset = warpId * WARP_SIZE * UNROLL + tidInWarp;
    size_t nElem = size % sizeof(ulong2) ? (size + sizeof(ulong2)) / sizeof(ulong2) : size / sizeof(ulong2);
    for (size_t i = offset; i < nElem; i += numThreads * UNROLL) {
// load to register first
#pragma unroll
      for (int j = 0; j < UNROLL; j++) {
        fetch128(ele[j], dstAddr + i + j * WARP_SIZE);
      }
#pragma unroll
      for (int j = 0; j < UNROLL; j++) {
        store128(srcAddr + i + j * WARP_SIZE, ele[j]);
      }
    }
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
 private:
  SmDevice2DeviceSemaphore::DeviceHandle semaphore_;
  void* src_;
  void* dst_;
  void* getPacketBuffer_;
};

}  // namespace mscclpp

#endif  // MSCCLPP_SM_CHANNEL_HPP_
