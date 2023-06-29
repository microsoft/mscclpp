// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_PACKET_HPP_
#define MSCCLPP_PACKET_HPP_

namespace mscclpp {

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
  __forceinline__ __device__ void write(uint32_t val1, uint32_t val2, uint32_t flag) {
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(v), "r"(val1), "r"(flag), "r"(val2), "r"(flag));
  }
  __forceinline__ __device__ void write(uint32_t val1, uint32_t val2) {
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,1,%2,1};" ::"l"(v), "r"(val1), "r"(val2));
  }
  __forceinline__ __device__ void write(uint64_t val, uint32_t flag) {
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(v), "r"((uint32_t)val), "r"(flag),
                 "r"((uint32_t)(val >> 32)), "r"(flag));
  }
  __forceinline__ __device__ void write(uint64_t val) {
    asm volatile("st.volatile.global.v4.u32 [%0], {%1,1,%2,1};" ::"l"(v), "r"((uint32_t)val),
                 "r"((uint32_t)(val >> 32)));
  }
  __forceinline__ __device__ uint2 read(uint32_t flag) {
    uint2 data;
    uint32_t flag1, flag2;
    do {
      asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                   : "=r"(data.x), "=r"(flag1), "=r"(data.y), "=r"(flag2)
                   : "l"(v));
    } while ((flag1 != flag) || (flag2 != flag));
    return data;
  }
  __forceinline__ __device__ uint2 read() { return read(1); }
  __forceinline__ __device__ void clear() {
    vec.x = 0;
    vec.y = 0;
  }
#endif  // __CUDACC__
};

#ifdef __CUDACC__
__forceinline__ __device__ void putPackets(void* dst, uint64_t dstOffset, void* src, uint64_t srcOffset,
                                           uint64_t srcSize, uint32_t threadId, uint32_t numThreads, uint32_t flag) {
  // Offsets should be aligned to 8 bytes & size should be a multiple of 8 bytes
  uint32_t* srcBase = (uint32_t*)((char*)src + srcOffset);
  LLPacket* dstBase = (LLPacket*)((char*)dst + dstOffset);
  size_t nElem = srcSize / sizeof(uint64_t);
  for (size_t i = threadId; i < nElem; i += numThreads) {
    LLPacket* pkt = &dstBase[i];
    pkt->write(srcBase[2 * i], srcBase[2 * i + 1], flag);
  }
}

__forceinline__ __device__ void getPackets(void* dst, uint64_t dstOffset, void* src, uint64_t srcOffset,
                                           uint64_t dstSize, uint32_t threadId, uint32_t numThreads, uint32_t flag) {
  // Offsets should be aligned to 8 bytes & size should be a multiple of 8 bytes
  LLPacket* srcBase = (LLPacket*)((char*)src + srcOffset);
  uint2* dstBase = (uint2*)((char*)dst + dstOffset);
  size_t nElem = dstSize / sizeof(uint2);
  for (size_t i = threadId; i < nElem; i += numThreads) {
    LLPacket* pkt = &srcBase[i];
    dstBase[i] = pkt->read(flag);
  }
}
#endif  // __CUDACC__

};  // namespace mscclpp

#endif  // MSCCLPP_PACKET_HPP_
