// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_PACKET_HPP_
#define MSCCLPP_PACKET_HPP_

#include "poll.hpp"

namespace mscclpp {

/// LL (low latency) protocol packet.
union LLPacket {
  // Assume data is written with an atomicity of 8 bytes (IB/RDMA).
  struct {
    uint32_t data1;
    uint32_t flag1;
    uint32_t data2;
    uint32_t flag2;
  };

  struct {
    uint64_t x;
    uint64_t y;
  } vec;

  uint64_t v[2];

#ifdef __CUDACC__
  __forceinline__ __device__ LLPacket() {}

  /// Write 8 bytes of data to the packet.
  /// @param val1 The first 4-byte data to write.
  /// @param val2 The second 4-byte data to write.
  /// @param flag The flag to write.
  __forceinline__ __device__ void write(uint32_t val1, uint32_t val2, uint32_t flag) {
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(v), "r"(val1), "r"(flag), "r"(val2), "r"(flag));
  }

  /// Write 8 bytes of data to the packet.
  /// @param val The 8-byte data to write.
  /// @param flag The flag to write.
  __forceinline__ __device__ void write(uint64_t val, uint32_t flag) {
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(v), "r"((uint32_t)val), "r"(flag),
                 "r"((uint32_t)(val >> 32)), "r"(flag));
  }

  /// Helper of @ref read().
  /// @param flag The flag to read.
  /// @param data The 8-byte data read.
  /// @return True if the flag is not equal to the given flag.
  __forceinline__ __device__ bool readOnce(uint32_t flag, uint2& data) const {
    uint32_t flag1, flag2;
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(data.x), "=r"(flag1), "=r"(data.y), "=r"(flag2)
                 : "l"(v));
    return (flag1 != flag) || (flag2 != flag);
  }

  /// Read 8 bytes of data from the packet.
  /// @param flag The flag to read.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  /// @return The 8-byte data read.
  __forceinline__ __device__ uint2 read(uint32_t flag, int64_t maxSpinCount = 100000000) const {
    uint2 data;
    POLL_MAYBE_JAILBREAK(readOnce(flag, data), maxSpinCount);
    return data;
  }

  /// Clear the packet.
  __forceinline__ __device__ void clear() {
    vec.x = 0;
    vec.y = 0;
  }
#endif  // __CUDACC__
};

#ifdef __CUDACC__
/// Read from the origin and write to the target buffer.
__forceinline__ __device__ void putPackets(void* targetPtr, uint64_t targetOffset, const void* originPtr,
                                           uint64_t originOffset, uint64_t originBytes, uint32_t threadId,
                                           uint32_t numThreads, uint32_t flag) {
  // Offsets should be aligned to 8 bytes & size should be a multiple of 8 bytes
  const uint32_t* originBase = (const uint32_t*)((const char*)originPtr + originOffset);
  LLPacket* targetBase = (LLPacket*)((char*)targetPtr + targetOffset);
  size_t nElem = originBytes / sizeof(uint64_t);
  for (size_t i = threadId; i < nElem; i += numThreads) {
    LLPacket* pkt = &targetBase[i];
    pkt->write(originBase[2 * i], originBase[2 * i + 1], flag);
  }
}

/// Read from the target buffer and write to the origin.
__forceinline__ __device__ void getPackets(const void* targetPtr, uint64_t targetOffset, void* originPtr,
                                           uint64_t originOffset, uint64_t originBytes, uint32_t threadId,
                                           uint32_t numThreads, uint32_t flag) {
  // Offsets should be aligned to 8 bytes & size should be a multiple of 8 bytes
  const LLPacket* targetBase = (const LLPacket*)((const char*)targetPtr + targetOffset);
  uint2* originBase = (uint2*)((char*)originPtr + originOffset);
  size_t nElem = originBytes / sizeof(uint2);
  for (size_t i = threadId; i < nElem; i += numThreads) {
    const LLPacket* pkt = &targetBase[i];
    originBase[i] = pkt->read(flag);
  }
}
#endif  // __CUDACC__

};  // namespace mscclpp

#endif  // MSCCLPP_PACKET_HPP_
