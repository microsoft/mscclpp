// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_PACKET_HPP_
#define MSCCLPP_PACKET_HPP_

#include <mscclpp/core.hpp>

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

  MSCCLPP_DEVICE LLPacket() {}

  /// Write 8 bytes of data to the packet.
  /// @param val1 The first 4-byte data to write.
  /// @param val2 The second 4-byte data to write.
  /// @param flag The flag to write.
  MSCCLPP_DEVICE void write(uint32_t val1, uint32_t val2, uint32_t flag) {
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(v), "r"(val1), "r"(flag), "r"(val2), "r"(flag));
  }

  /// Write 8 bytes of data to the packet.
  /// @param val The 8-byte data to write.
  /// @param flag The flag to write.
  MSCCLPP_DEVICE void write(uint64_t val, uint32_t flag) {
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(v), "r"((uint32_t)val), "r"(flag),
                 "r"((uint32_t)(val >> 32)), "r"(flag));
  }

  /// Read 8 bytes of data from the packet.
  /// @param flag The flag to read.
  /// @return The 8-byte data read.
  MSCCLPP_DEVICE uint2 read(uint32_t flag) {
    uint2 data;
    uint32_t flag1, flag2;
    do {
      asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(data.x), "=r"(flag1), "=r"(data.y), "=r"(flag2)
                   : "l"(v));
    } while ((flag1 != flag) || (flag2 != flag));
    return data;
  }

  /// Clear the packet.
  MSCCLPP_DEVICE void clear() {
    vec.x = 0;
    vec.y = 0;
  }
};

MSCCLPP_DEVICE void putPackets(void* dst, uint64_t dstOffset, void* src, uint64_t srcOffset, uint64_t srcBytes,
                               uint32_t threadId, uint32_t numThreads, uint32_t flag) {
  // Offsets should be aligned to 8 bytes & size should be a multiple of 8 bytes
  uint32_t* srcBase = (uint32_t*)((char*)src + srcOffset);
  LLPacket* dstBase = (LLPacket*)((char*)dst + dstOffset);
  size_t nElem = srcBytes / sizeof(uint64_t);
  for (size_t i = threadId; i < nElem; i += numThreads) {
    LLPacket* pkt = &dstBase[i];
    pkt->write(srcBase[2 * i], srcBase[2 * i + 1], flag);
  }
}

MSCCLPP_DEVICE void getPackets(void* dst, uint64_t dstOffset, void* src, uint64_t srcOffset, uint64_t dstBytes,
                               uint32_t threadId, uint32_t numThreads, uint32_t flag) {
  // Offsets should be aligned to 8 bytes & size should be a multiple of 8 bytes
  LLPacket* srcBase = (LLPacket*)((char*)src + srcOffset);
  uint2* dstBase = (uint2*)((char*)dst + dstOffset);
  size_t nElem = dstBytes / sizeof(uint2);
  for (size_t i = threadId; i < nElem; i += numThreads) {
    LLPacket* pkt = &srcBase[i];
    dstBase[i] = pkt->read(flag);
  }
}

};  // namespace mscclpp

#endif  // MSCCLPP_PACKET_HPP_
