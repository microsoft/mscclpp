// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_SM_CHANNEL_HPP_
#define MSCCLPP_SM_CHANNEL_HPP_

#include <mscclpp/core.hpp>
#include <mscclpp/epoch.hpp>
#include <mscclpp/packet.hpp>

namespace mscclpp {
namespace channel {
namespace sm {

// A direct version of DeviceChannel only for CudaIpc
struct SmChannel {
 public:
  SmChannel() = default;
  SmChannel(SmDevice2DeviceEpoch::DeviceHandle epoch, RegisteredMemory dst, void* src, void* getPacketBuffer = nullptr);

#ifdef __CUDACC__
  __forceinline__ __device__ void put(uint64_t dstOffset, uint64_t srcOffset, uint64_t size, uint32_t threadId,
                                      uint32_t numThreads) {
    // assume the memory is aligned to 8 bytes
    uint64_t* srcAddr = (uint64_t*)((char*)src_ + srcOffset);
    uint64_t* dstAddr = (uint64_t*)((char*)dst_ + dstOffset);
    uint64_t ele;
    size_t nElem = size % sizeof(uint64_t) ? (size + sizeof(uint64_t)) / sizeof(uint64_t) : size / sizeof(uint64_t);
    for (size_t i = threadId; i < nElem; i += numThreads) {
      // load to register first
      ele = srcAddr[i];
      dstAddr[i] = ele;
    }
  }

  __forceinline__ __device__ void putPackets(uint64_t dstOffset, uint64_t srcOffset, uint64_t size, uint32_t threadId,
                                             uint32_t numThreads, uint32_t flag) {
    mscclpp::packet::putPackets(dst_, dstOffset, src_, srcOffset, size, threadId, numThreads, flag);
  }

  __forceinline__ __device__ void getPackets(uint64_t dstOffset, uint64_t srcOffset, uint64_t size, uint32_t threadId,
                                             uint32_t numThreads, uint32_t flag) {
    mscclpp::packet::getPackets(src_, dstOffset, getPacketBuffer_, srcOffset, size, threadId, numThreads, flag);
  }

  __forceinline__ __device__ void signal() { epoch_.signal(); }

  __forceinline__ __device__ void signalPacket() { epoch_.signalPacket(); }

  __forceinline__ __device__ void epochIncrement() { epoch_.epochIncrement(); }

  __forceinline__ __device__ uint64_t epochGetLocal() const { return epoch_.epochGetLocal(); }

  __forceinline__ __device__ void wait() { epoch_.wait(); }
#endif  // __CUDACC__
 private:
  SmDevice2DeviceEpoch::DeviceHandle epoch_;
  void* src_;
  void* dst_;
  void* getPacketBuffer_;
};

}  // namespace sm
}  // namespace channel
}  // namespace mscclpp

#endif  // MSCCLPP_SM_CHANNEL_HPP_
