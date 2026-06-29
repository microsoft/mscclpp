// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "kernels/configs.cuh"
#include "kernels/exception.cuh"

namespace mscclpp {
namespace ep {

template <typename dtype_t>
dtype_t cellDiv(dtype_t a, dtype_t b) {
  return (a + b - 1) / b;
}

template <typename dtype_t>
dtype_t align(dtype_t a, dtype_t b) {
  return cellDiv<dtype_t>(a, b) * b;
}

struct LowLatencyBuffer {
  int numCleanInt = 0;

  void* dispatchRdmaSendBuffer = nullptr;
  void* dispatchRdmaRecvDataBuffer = nullptr;
  // NOTE: signaling buffers are int64_t (not int) so that IB atomic ops
  // (IBV_WR_ATOMIC_FETCH_AND_ADD is a 64-bit, 8-byte-aligned op) always
  // target an 8-byte-aligned address. Using int32 slots produced unaligned
  // atomics at odd indices that the NIC silently drops.
  int64_t* dispatchRdmaRecvCountBuffer = nullptr;

  void* combineRdmaSendBuffer = nullptr;
  void* combineRdmaRecvDataBuffer = nullptr;
  int64_t* combineRdmaRecvFlagBuffer = nullptr;

  void* combineRdmaSendBufferDataStart = nullptr;
  size_t numBytesPerCombineMsg = 0;

  std::pair<int64_t*, int> cleanMeta() {
    EP_HOST_ASSERT(dispatchRdmaRecvCountBuffer == combineRdmaRecvFlagBuffer);
    return {dispatchRdmaRecvCountBuffer, numCleanInt};
  }
};

struct LowLatencyLayout {
  size_t totalBytes = 0;
  LowLatencyBuffer buffers[2];

  template <typename out_ptr_t = void*, typename count_ptr_t = uint8_t*, typename in_ptr_t = void*>
  out_ptr_t advance(const in_ptr_t& ptr, size_t count) {
    return reinterpret_cast<out_ptr_t>(reinterpret_cast<count_ptr_t>(ptr) + count);
  }

  LowLatencyLayout(void* rdmaBuffer, int numMaxDispatchTokensPerRank, int hidden, int numRanks, int numExperts) {
    (void)numRanks;
    const int numScales = hidden / 128;

    // Dispatch and combine layout:
    //  - 2 symmetric odd/even send buffer
    //  - 2 symmetric odd/even receive buffers
    //  - 2 symmetric odd/even signaling buffers

    // Message sizes
    // NOTES: you should add a control `int4` for combine messages if you want to do data transformation
    EP_HOST_ASSERT(numScales * static_cast<int>(sizeof(float)) <= hidden);
    size_t numBytesPerDispatchMsg =
        sizeof(int4) + std::max(hidden * sizeof(nv_bfloat16), hidden + numScales * sizeof(float));
    size_t numBytesPerCombineMsg = hidden * sizeof(nv_bfloat16);

    // Send buffer
    size_t dispatchSendBufferBytes = numMaxDispatchTokensPerRank * numBytesPerDispatchMsg;
    size_t combineSendBufferBytes = numExperts * numMaxDispatchTokensPerRank * numBytesPerCombineMsg;
    size_t sendBufferBytes = std::max(dispatchSendBufferBytes, combineSendBufferBytes);
    EP_HOST_ASSERT(sendBufferBytes % sizeof(int4) == 0);
    totalBytes += sendBufferBytes * 2;

    // Symmetric receive buffers
    // TODO: optimize memory usages
    size_t dispatchRecvDataBufferBytes = numExperts * numMaxDispatchTokensPerRank * numBytesPerDispatchMsg;
    size_t combineRecvBufferBytes = numExperts * numMaxDispatchTokensPerRank * numBytesPerCombineMsg;
    size_t recvBufferBytes = std::max(dispatchRecvDataBufferBytes, combineRecvBufferBytes);
    EP_HOST_ASSERT(recvBufferBytes % sizeof(int4) == 0);
    totalBytes += recvBufferBytes * 2;

    // Symmetric signaling buffers (int64_t slots for 8-byte-aligned IB atomics).
    size_t dispatchRecvCountBufferBytes = numExperts * sizeof(int64_t);
    size_t combineRecvFlagBufferBytes = dispatchRecvCountBufferBytes;
    size_t signalingBufferBytes = std::max(dispatchRecvCountBufferBytes, combineRecvFlagBufferBytes);
    totalBytes += signalingBufferBytes * 2;

    // Assign pointers
    // NOTES: we still leave some space for distinguishing dispatch/combine buffer,
    // so you may see some parameters are duplicated
    for (int i = 0; i < 2; ++i) {
      buffers[i] = {static_cast<int>(signalingBufferBytes / sizeof(int64_t)),
                    advance(rdmaBuffer, sendBufferBytes * i),
                    advance(rdmaBuffer, sendBufferBytes * 2 + recvBufferBytes * i),
                    advance<int64_t*>(rdmaBuffer, sendBufferBytes * 2 + recvBufferBytes * 2 + signalingBufferBytes * i),
                    advance(rdmaBuffer, sendBufferBytes * i),
                    advance(rdmaBuffer, sendBufferBytes * 2 + recvBufferBytes * i),
                    advance<int64_t*>(rdmaBuffer, sendBufferBytes * 2 + recvBufferBytes * 2 + signalingBufferBytes * i),
                    advance(rdmaBuffer, sendBufferBytes * i),
                    numBytesPerCombineMsg};
    }
  }
};

inline size_t getLowLatencyRdmaSizeHint(int numMaxDispatchTokensPerRank, int hidden, int numRanks, int numExperts) {
  auto numBytes = LowLatencyLayout(nullptr, numMaxDispatchTokensPerRank, hidden, numRanks, numExperts).totalBytes;
  return ((numBytes + NUM_BUFFER_ALIGNMENT_BYTES) / NUM_BUFFER_ALIGNMENT_BYTES) * NUM_BUFFER_ALIGNMENT_BYTES;
}

}  // namespace ep
}  // namespace mscclpp
