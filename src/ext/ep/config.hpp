// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mscclpp/device.hpp>
#include <type_traits>

#include "kernels/configs.cuh"
#include "kernels/exception.cuh"

namespace mscclpp {
namespace ep {

template <typename dtype_t>
__host__ __device__ constexpr dtype_t configCellDiv(dtype_t a, dtype_t b) {
  return (a + b - 1) / b;
}

template <typename dtype_t>
__host__ __device__ constexpr dtype_t configAlign(dtype_t a, dtype_t b) {
  return configCellDiv<dtype_t>(a, b) * b;
}

// Low-latency rank-deduplicated dispatch payload layout:
//
//   [data: DataType[hidden]]
//   [optional scales: ScaleType[hidden / scale_block_size]]
//   [topKIndices: int[topK]]
//   [topKValues: float[topK]]
//   [srcTokenGlobalIdx: int]
//
// The payload is 32-byte aligned as a whole. ScaleType=void means the payload is
// not quantized and has no scale section.
template <typename DataType, typename ScaleType = void>
struct LowLatencyPayloadView {
  static constexpr bool kHasScales = !std::is_void_v<ScaleType>;

  int hidden;
  int topK;
  int scaleBlockSize;
  int numScales;
  size_t hiddenBytes_;
  size_t scaleOffset_;
  size_t metadataOffset_;
  size_t numBytes_;

  MSCCLPP_HOST_DEVICE_INLINE static int nScales([[maybe_unused]] int hidden, [[maybe_unused]] int scaleBlockSize) {
    if constexpr (kHasScales) {
      return hidden / scaleBlockSize;
    } else {
      return 0;
    }
  }

  MSCCLPP_HOST_DEVICE_INLINE static size_t hiddenBytes(int hidden) {
    return static_cast<size_t>(hidden) * sizeof(DataType);
  }

  MSCCLPP_HOST_DEVICE_INLINE static size_t scaleOffset(int hidden) {
    if constexpr (kHasScales) {
      return configAlign<size_t>(hiddenBytes(hidden), alignof(ScaleType));
    } else {
      return 0;
    }
  }

  MSCCLPP_HOST_DEVICE_INLINE static size_t scaleBytes([[maybe_unused]] int hidden,
                                                      [[maybe_unused]] int scaleBlockSize) {
    if constexpr (kHasScales) {
      return static_cast<size_t>(nScales(hidden, scaleBlockSize)) * sizeof(ScaleType);
    } else {
      return 0;
    }
  }

  MSCCLPP_HOST_DEVICE_INLINE static size_t metadataOffset(int hidden, [[maybe_unused]] int scaleBlockSize) {
    if constexpr (kHasScales) {
      return configAlign<size_t>(scaleOffset(hidden) + scaleBytes(hidden, scaleBlockSize), alignof(int));
    } else {
      return configAlign<size_t>(hiddenBytes(hidden), alignof(int));
    }
  }

  MSCCLPP_HOST_DEVICE_INLINE static size_t metadataBytes(int topK) {
    return static_cast<size_t>(topK) * sizeof(int) + static_cast<size_t>(topK) * sizeof(float) + sizeof(int);
  }

  MSCCLPP_HOST_DEVICE_INLINE static size_t numBytes(int hidden, int topK, int scaleBlockSize) {
    return configAlign<size_t>(metadataOffset(hidden, scaleBlockSize) + metadataBytes(topK), 32);
  }

  MSCCLPP_HOST_DEVICE_INLINE LowLatencyPayloadView(int hidden, int topK, int scaleBlockSize = (kHasScales ? 128 : 0))
      : hidden(hidden),
        topK(topK),
        scaleBlockSize(scaleBlockSize),
        numScales(nScales(hidden, scaleBlockSize)),
        hiddenBytes_(hiddenBytes(hidden)),
        scaleOffset_(scaleOffset(hidden)),
        metadataOffset_(metadataOffset(hidden, scaleBlockSize)),
        numBytes_(numBytes(hidden, topK, scaleBlockSize)) {}

  template <typename T>
  MSCCLPP_HOST_DEVICE_INLINE T* data(void* base) const {
    return reinterpret_cast<T*>(base);
  }

  MSCCLPP_HOST_DEVICE_INLINE ScaleType* scaleFactors(void* base) const {
    static_assert(kHasScales, "Payload has no scale factors");
    return reinterpret_cast<ScaleType*>(reinterpret_cast<uint8_t*>(base) + scaleOffset_);
  }

  MSCCLPP_HOST_DEVICE_INLINE const ScaleType* scaleFactors(const void* base) const {
    static_assert(kHasScales, "Payload has no scale factors");
    return reinterpret_cast<const ScaleType*>(reinterpret_cast<const uint8_t*>(base) + scaleOffset_);
  }

  MSCCLPP_HOST_DEVICE_INLINE int* topKIndices(void* base) const {
    return reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(base) + metadataOffset_);
  }

  MSCCLPP_HOST_DEVICE_INLINE const int* topKIndices(const void* base) const {
    return reinterpret_cast<const int*>(reinterpret_cast<const uint8_t*>(base) + metadataOffset_);
  }

  MSCCLPP_HOST_DEVICE_INLINE float* topKValues(void* base) const {
    return reinterpret_cast<float*>(topKIndices(base) + topK);
  }

  MSCCLPP_HOST_DEVICE_INLINE const float* topKValues(const void* base) const {
    return reinterpret_cast<const float*>(topKIndices(base) + topK);
  }

  MSCCLPP_HOST_DEVICE_INLINE int* srcTokenGlobalIdx(void* base) const {
    return reinterpret_cast<int*>(topKValues(base) + topK);
  }

  MSCCLPP_HOST_DEVICE_INLINE const int* srcTokenGlobalIdx(const void* base) const {
    return reinterpret_cast<const int*>(topKValues(base) + topK);
  }
};

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

  LowLatencyLayout(void* rdmaBuffer, int numMaxDispatchTokensPerRank, int hidden, int numRanks, int numExperts,
                   int numTopk) {
    // Dispatch and combine layout:
    //  - 2 symmetric odd/even send buffer
    //  - 2 symmetric odd/even receive buffers
    //  - 2 symmetric odd/even signaling buffers

    // Message sizes
    // NOTES: you should add a control `int4` for combine messages if you want to do data transformation
    const LowLatencyPayloadView<nv_bfloat16> bf16DispatchPayload(hidden, numTopk);
    const LowLatencyPayloadView<__nv_fp8_storage_t, float> fp8DispatchPayload(hidden, numTopk, 128);
    size_t numBytesPerDispatchMsg = std::max(bf16DispatchPayload.numBytes_, fp8DispatchPayload.numBytes_);
    const size_t optDispatchMetadataBytes =
        configAlign<size_t>(static_cast<size_t>(numRanks + numExperts) * sizeof(uint64_t), 128);
    const size_t optDispatchPayloadStride = configAlign<size_t>(bf16DispatchPayload.numBytes_, 128);
    size_t numBytesPerCombineMsg = hidden * sizeof(nv_bfloat16);

    // Send buffer
    size_t dispatchSendBufferBytes = std::max(
        static_cast<size_t>(numMaxDispatchTokensPerRank) * numBytesPerDispatchMsg,
        optDispatchMetadataBytes + static_cast<size_t>(numMaxDispatchTokensPerRank) * optDispatchPayloadStride);
    size_t combineSendBufferBytes = numExperts * numMaxDispatchTokensPerRank * numBytesPerCombineMsg;
    size_t sendBufferBytes = std::max(dispatchSendBufferBytes, combineSendBufferBytes);
    EP_HOST_ASSERT(sendBufferBytes % sizeof(int4) == 0);
    totalBytes += sendBufferBytes * 2;

    // Symmetric receive buffers
    // TODO: optimize memory usages
    size_t dispatchRecvDataBufferBytes =
        std::max(static_cast<size_t>(numRanks) * numMaxDispatchTokensPerRank * numBytesPerDispatchMsg,
                 optDispatchMetadataBytes +
                     static_cast<size_t>(numRanks) * numMaxDispatchTokensPerRank * optDispatchPayloadStride);
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

inline size_t getLowLatencyRdmaSizeHint(int numMaxDispatchTokensPerRank, int hidden, int numRanks, int numExperts,
                                        int numTopk) {
  auto numBytes =
      LowLatencyLayout(nullptr, numMaxDispatchTokensPerRank, hidden, numRanks, numExperts, numTopk).totalBytes;
  return configAlign<size_t>(numBytes, NUM_BUFFER_ALIGNMENT_BYTES);
}

}  // namespace ep
}  // namespace mscclpp
