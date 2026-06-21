// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <mscclpp/atomic_device.hpp>
#include <mscclpp/memory_channel_device.hpp>
#include <mscclpp/port_channel_device.hpp>

#include "utils.cuh"

namespace mscclpp {
namespace ep {

// Pointer-to-offset helper for MSCCL++ port channels.
MSCCLPP_DEVICE_INLINE uint64_t portChannelOffsetOf(uint64_t ptr, void* rdmaBufferPtr) {
  return ptr - reinterpret_cast<uint64_t>(rdmaBufferPtr);
}

// Translate a local buffer pointer into a peer-mapped pointer at the same
// offset in `peerRank`'s buffer.
MSCCLPP_DEVICE_INLINE uint64_t peerMappedPtrOf(uint64_t localPtr, void* const* peerBases, void* localBufferBase,
                                               int peerRank) {
  const auto off = localPtr - reinterpret_cast<uint64_t>(localBufferBase);
  return reinterpret_cast<uint64_t>(peerBases[peerRank]) + off;
}

MSCCLPP_DEVICE_INLINE bool isIpcPeer(int rank, int peerRank, int ranksPerIpcDomain) {
  return ranksPerIpcDomain > 0 && rank / ranksPerIpcDomain == peerRank / ranksPerIpcDomain;
}

// Cross-rank barrier via either MemoryChannel (same IPC domain) or PortChannel.
MSCCLPP_DEVICE_INLINE void channelBarrierBlock(mscclpp::PortChannelDeviceHandle* portChannelHandles,
                                               mscclpp::BaseMemoryChannelDeviceHandle* memoryChannelHandles, int rank,
                                               int numRanks, int ranksPerIpcDomain) {
  const int peerRank = threadIdx.x;
  if (peerRank < numRanks && peerRank != rank) {
    if (memoryChannelHandles != nullptr && isIpcPeer(rank, peerRank, ranksPerIpcDomain)) {
      memoryChannelHandles[peerRank].signal();
      memoryChannelHandles[peerRank].wait();
    } else {
      // Index: qp 0, peer = peerRank (assumes peer_idx == rank in LL topology).
      portChannelHandles[peerRank].signal();
      portChannelHandles[peerRank].wait();
    }
  }
  __syncthreads();
}

template <typename T>
MSCCLPP_DEVICE_INLINE void storeRelease(T* ptr, T value) {
  st_na_release(ptr, value);
}

MSCCLPP_DEVICE_INLINE void atomicAddReleaseDevice(int* ptr, int value) {
#if defined(MSCCLPP_DEVICE_CUDA)
  asm volatile("red.release.gpu.global.add.s32 [%0], %1;" ::"l"(ptr), "r"(value) : "memory");
#else
  (void)mscclpp::atomicFetchAdd<int, mscclpp::scopeDevice>(ptr, value, mscclpp::memoryOrderRelease);
#endif
}

}  // namespace ep
}  // namespace mscclpp
