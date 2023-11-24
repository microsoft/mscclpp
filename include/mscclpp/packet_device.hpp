// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_PACKET_DEVICE_HPP_
#define MSCCLPP_PACKET_DEVICE_HPP_

#include "device.hpp"

#if defined(MSCCLPP_DEVICE_COMPILE)
#include "atomic_device.hpp"
#include "poll_device.hpp"
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

namespace mscclpp {

/// LL (low latency) protocol packet.
union alignas(16) LLPacket {
  // Assume data is written with an atomicity of 8 bytes (IB/RDMA).
  struct {
    uint32_t data1;
    uint32_t flag1;
    uint32_t data2;
    uint32_t flag2;
  };

#if defined(MSCCLPP_DEVICE_COMPILE)
  ulonglong2 raw_;

  MSCCLPP_DEVICE_INLINE LLPacket() {}

  /// Write 8 bytes of data to the packet.
  /// @param val1 The first 4-byte data to write.
  /// @param val2 The second 4-byte data to write.
  /// @param flag The flag to write.
  MSCCLPP_DEVICE_INLINE void write(uint32_t val1, uint32_t val2, uint32_t flag) {
#if defined(MSCCLPP_DEVICE_CUDA)
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(&raw_), "r"(val1), "r"(flag), "r"(val2),
                 "r"(flag));
#else  // !defined(MSCCLPP_DEVICE_CUDA)
    uint4 reg = make_uint4(val1, flag, val2, flag);
    ulonglong2* p = reinterpret_cast<ulonglong2*>(&reg);
    // TODO: revisit after changing clang++ to hipcc
    atomicStore(&(raw_.x), p->x, memoryOrderRelaxed);
    atomicStore(&(raw_.y), p->y, memoryOrderRelaxed);
#endif
  }

  /// Write 8 bytes of data to the packet.
  /// @param val The 8-byte data to write.
  /// @param flag The flag to write.
  MSCCLPP_DEVICE_INLINE void write(uint64_t val, uint32_t flag) { write((uint32_t)val, (uint32_t)(val >> 32), flag); }

  /// Helper of @ref read().
  /// @param flag The flag to read.
  /// @param data The 8-byte data read.
  /// @return True if the flag is not equal to the given flag.
  MSCCLPP_DEVICE_INLINE bool readOnce(uint32_t flag, uint2& data) const {
#if defined(MSCCLPP_DEVICE_CUDA)
    uint32_t flag1, flag2;
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(data.x), "=r"(flag1), "=r"(data.y), "=r"(flag2)
                 : "l"(&raw_));
    return (flag1 != flag) || (flag2 != flag);
#else  // !defined(MSCCLPP_DEVICE_CUDA)
    ulonglong2 reg;
    // TODO: revisit after changing clang++ to hipcc
    reg.x = atomicLoad(&(raw_.x), memoryOrderRelaxed);
    reg.y = atomicLoad(&(raw_.y), memoryOrderRelaxed);
    uint4* ptr = reinterpret_cast<uint4*>(&reg);
    data.x = ptr->x;
    data.y = ptr->z;
    return (ptr->y != flag) || (ptr->w != flag);
#endif
  }

  /// Read 8 bytes of data from the packet.
  /// @param flag The flag to read.
  /// @param maxSpinCount The maximum number of spin counts before asserting. Never assert if negative.
  /// @return The 8-byte data read.
  MSCCLPP_DEVICE_INLINE uint2 read(uint32_t flag, int64_t maxSpinCount = 100000000) const {
    uint2 data;
    POLL_MAYBE_JAILBREAK(readOnce(flag, data), maxSpinCount);
    return data;
  }

  /// Clear the packet.
  MSCCLPP_DEVICE_INLINE void clear() { raw_ = make_ulonglong2(0, 0); }
#endif  // defined(MSCCLPP_DEVICE_COMPILE)
};

#if defined(MSCCLPP_DEVICE_COMPILE)
/// Read from the origin and write to the target buffer.
MSCCLPP_DEVICE_INLINE void putPackets(void* targetPtr, uint64_t targetOffset, const void* originPtr,
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
MSCCLPP_DEVICE_INLINE void getPackets(const void* targetPtr, uint64_t targetOffset, void* originPtr,
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
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

};  // namespace mscclpp

#endif  // MSCCLPP_PACKET_DEVICE_HPP_
