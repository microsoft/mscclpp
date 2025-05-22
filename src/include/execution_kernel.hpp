// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_EXECUTION_KERNEL_HPP_
#define MSCCLPP_EXECUTION_KERNEL_HPP_

#include <mscclpp/executor.hpp>
#if defined(ENABLE_NPKIT)
#include <mscclpp/npkit/npkit.hpp>
#endif
#include <cstdint>
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/packet_device.hpp>
#include <mscclpp/port_channel.hpp>

#include "execution_common.hpp"
#if defined(MSCCLPP_DEVICE_COMPILE)
#include <mscclpp/gpu_data_types.hpp>
#include <mscclpp/nvls_device.hpp>
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

namespace {
#if defined(MSCCLPP_DEVICE_COMPILE)
template <typename To, typename From>
MSCCLPP_DEVICE_INLINE To bit_cast(const From& src) {
  static_assert(sizeof(To) == sizeof(From), "Size mismatch for bit_cast");

  union {
    From f;
    To t;
  } u;
  u.f = src;
  return u.t;
}

template <typename T>
MSCCLPP_DEVICE_INLINE T add_elements(T a, T b) {
  return a + b;
}

template <>
MSCCLPP_DEVICE_INLINE __half2 add_elements(__half2 a, __half2 b) {
  return __hadd2(a, b);
}

template <>
MSCCLPP_DEVICE_INLINE __bfloat16 add_elements(__bfloat16 a, __bfloat16 b) {
  return __hadd(a, b);
}

template <>
MSCCLPP_DEVICE_INLINE __bfloat162 add_elements(__bfloat162 a, __bfloat162 b) {
  return __hadd2(a, b);
}

template <typename T>
MSCCLPP_DEVICE_INLINE int4 add_vectors_helper(int4 a, int4 b) {
  int4 ret;
  ret.w = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.w), bit_cast<T, int>(b.w)));
  ret.x = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.x), bit_cast<T, int>(b.x)));
  ret.y = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.y), bit_cast<T, int>(b.y)));
  ret.z = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.z), bit_cast<T, int>(b.z)));
  return ret;
}

template <typename T>
MSCCLPP_DEVICE_INLINE int4 add_vectors(int4 a, int4 b) {
  return add_vectors_helper<T>(a, b);
}

template <>
MSCCLPP_DEVICE_INLINE int4 add_vectors<__half>(int4 a, int4 b) {
  return add_vectors_helper<__half2>(a, b);
}

template <>
MSCCLPP_DEVICE_INLINE int4 add_vectors<__bfloat16>(int4 a, int4 b) {
  return add_vectors_helper<__bfloat162>(a, b);
}

template <typename T>
MSCCLPP_DEVICE_INLINE uint2 add_vectors_helper(uint2 a, uint2 b) {
  uint2 ret;
  ret.x = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.x), bit_cast<T, int>(b.x)));
  ret.y = bit_cast<int, T>(add_elements(bit_cast<T, int>(a.y), bit_cast<T, int>(b.y)));
  return ret;
}

template <typename T>
MSCCLPP_DEVICE_INLINE uint2 add_vectors(uint2 a, uint2 b) {
  return add_vectors_helper<T>(a, b);
}

template <>
MSCCLPP_DEVICE_INLINE __attribute__((unused)) uint2 add_vectors<__half>(uint2 a, uint2 b) {
  return add_vectors_helper<__half2>(a, b);
}

template <>
MSCCLPP_DEVICE_INLINE __attribute__((unused)) uint2 add_vectors<__bfloat16>(uint2 a, uint2 b) {
  return add_vectors_helper<__bfloat162>(a, b);
}

template <typename T>
MSCCLPP_DEVICE_INLINE int add_vectors_helper(int a, int b) {
  return bit_cast<int, T>(add_elements(bit_cast<T, int>(a), bit_cast<T, int>(b)));
}

template <typename T>
MSCCLPP_DEVICE_INLINE int add_vectors(int a, int b) {
  return add_vectors_helper<T>(a, b);
}

template <>
MSCCLPP_DEVICE_INLINE __attribute__((unused)) int add_vectors<__half>(int a, int b) {
  return add_vectors_helper<__half2>(a, b);
}

template <>
MSCCLPP_DEVICE_INLINE __attribute__((unused)) int add_vectors<__bfloat16>(int a, int b) {
  return add_vectors_helper<__bfloat162>(a, b);
}

template <typename T>
MSCCLPP_DEVICE_INLINE uint32_t add_vectors_helper(uint32_t a, uint32_t b) {
  return bit_cast<uint32_t, T>(add_elements(bit_cast<T, uint32_t>(a), bit_cast<T, uint32_t>(b)));
}

template <typename T>
MSCCLPP_DEVICE_INLINE uint32_t add_vectors(uint32_t a, uint32_t b) {
  return add_vectors_helper<T>(a, b);
}

template <>
MSCCLPP_DEVICE_INLINE uint32_t add_vectors<__half>(uint32_t a, uint32_t b) {
  return add_vectors_helper<__half2>(a, b);
}

template <>
MSCCLPP_DEVICE_INLINE uint32_t add_vectors<__bfloat16>(uint32_t a, uint32_t b) {
  return add_vectors_helper<__bfloat162>(a, b);
}

template <typename T>
struct VectorType {
  using type = T;
  using nvls_type = T;
  using nvls_type2 = T;
};

template <>
struct VectorType<__half> {
  using type = __half2;
  using nvls_type = uint4;
  using nvls_type2 = uint1;
};

template <>
struct VectorType<__bfloat16> {
  using type = __bfloat162;
  using nvls_type = uint4;
  using nvls_type2 = uint1;
};

template <>
struct VectorType<float> {
  using type = float;
  using nvls_type = uint4;
  using nvls_type2 = uint1;
};
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

}  // namespace

namespace mscclpp {

#if defined(MSCCLPP_DEVICE_COMPILE)

#define MAX_DEVICE_SYNCERS 16
#define MAX_DEVICE_FUNCTIONS_IN_PIPELINE 16
__device__ DeviceSyncer deviceSyncers[MAX_DEVICE_SYNCERS];

__shared__ DeviceHandle<MemoryChannel>* memoryChannels;
__shared__ DeviceHandle<PortChannel>* portChannels;
__shared__ DeviceHandle<NvlsConnection::DeviceMulticastPointer>* nvlsChannels;
__shared__ int flag;
__shared__ uint32_t scratchSize;

typedef void (*DeviceFunction)(Operation2* op, void* src, void* dst, void* scratch);

template <typename T>
MSCCLPP_DEVICE_INLINE T* getBuffer(T* input, T* output, T* scratch, BufferType bufferType) {
  if (bufferType == BufferType::INPUT) {
    return input;
  }
  if (bufferType == BufferType::OUTPUT) {
    return output;
  }
  if (bufferType == BufferType::SCRATCH) {
    return scratch;
  }
  return nullptr;
}

template <typename T, typename PacketType>
MSCCLPP_DEVICE_INLINE DeviceFunction getDeviceFunction(const Operation2& op, uint8_t* nSteps);

MSCCLPP_DEVICE_INLINE void handleNop(Operation2* operation, void* input, void* output, void* scratch) {
  __syncthreads();
}

MSCCLPP_DEVICE_INLINE void handleBarrier(Operation2* operation, void* input, void* output, void* scratch) {
  DeviceSyncer* syncer = &deviceSyncers[operation->deviceSyncerIndex];
  syncer->sync(operation->nThreadBlocks);
}

MSCCLPP_DEVICE_INLINE void handleSignal(Operation2* operation, void* input, void* output, void* scratch) {
  int nChannels = operation->nOutputs;
  ChannelType chType = operation->channelType;
  uint8_t* channelIndex = operation->outputChannelIndexes;
  int tid = threadIdx.x;
  if (tid < nChannels && chType == ChannelType::MEMORY) {
    memoryChannels[channelIndex[tid]].signal();
    return;
  }
  if (tid < nChannels && chType == ChannelType::PORT) {
    portChannels[channelIndex[threadIdx.x]].signal();
  }
}

MSCCLPP_DEVICE_INLINE void handleWait(Operation2* operation, void* src, void* dst, void* scratch) {
  int nChannels = operation->nInputs;
  ChannelType chType = operation->channelType;
  uint8_t* channelIndex = operation->inputChannelIndexes;
  int tid = threadIdx.x;
  if (tid < nChannels && chType == ChannelType::MEMORY) {
    memoryChannels[channelIndex[tid]].wait();
    return;
  }
  if (tid < nChannels && chType == ChannelType::PORT) {
    portChannels[channelIndex[tid]].wait();
  }
}

MSCCLPP_DEVICE_INLINE void handleFlush(Operation2* operation, void* src, void* dst, void* scratch) {
  int nChannels = operation->nOutputs;
  uint8_t* channelIndexes = operation->outputChannelIndexes;
  int tid = threadIdx.x;
  if (tid < nChannels) {
    portChannels[channelIndexes[tid]].flush();
  }
}

MSCCLPP_DEVICE_INLINE void handleGet(Operation2* operation, void* src, void* dst, void* scratch) {
  uint32_t count = operation->nInputs;
  uint32_t size = operation->size;
  uint8_t* srcChannelIndexes = operation->inputChannelIndexes;
  uint32_t* srcOffsets = operation->inputOffsets;
  uint32_t* dstOffsets = operation->outputOffsets;
  for (int i = 0; i < count; i++) {
    uint32_t dstOffset = dstOffsets[i];
    uint32_t srcOffset = srcOffsets[i];
    memoryChannels[srcChannelIndexes[i]].get(srcOffset, dstOffset, size, threadIdx.x, blockDim.x);
  }
}

template <bool PutWithSignal = false, bool PutWithSignalAndFlush = false>
MSCCLPP_DEVICE_INLINE void handlePut(Operation2* operation, void* src, void* dst, void* scratch) {
  ChannelType chType = operation->channelType;
  uint32_t count = operation->nOutputs;
  uint32_t size = operation->size;
  uint8_t* dstChannelIndexes = operation->outputChannelIndexes;
  uint32_t* dstOffsets = operation->outputOffsets;
  uint32_t* srcOffsets = operation->inputOffsets;
  if (chType == ChannelType::MEMORY) {
    for (int i = 0; i < count; i++) {
      uint32_t dstOffset = dstOffsets[i];
      uint32_t srcOffset = srcOffsets[i];
      memoryChannels[dstChannelIndexes[i]].put(dstOffset, srcOffset, size, threadIdx.x, blockDim.x);
    }
    return;
  }
  if (chType == ChannelType::PORT) {
    int tid = threadIdx.x;
    if (tid < count) {
      if constexpr (PutWithSignal) {
        portChannels[dstChannelIndexes[tid]].putWithSignal(dstOffsets[tid], srcOffsets[tid], size);
      } else if constexpr (PutWithSignalAndFlush) {
        portChannels[dstChannelIndexes[tid]].putWithSignalAndFlush(dstOffsets[tid], srcOffsets[tid], size);
      } else {
        portChannels[dstChannelIndexes[tid]].put(dstOffsets[tid], srcOffsets[tid], size);
      }
    }
  }
}

template <typename T, bool SendToRemote = true>
MSCCLPP_DEVICE_INLINE void handleReadReduceCopySend(Operation2* operation, void* input, void* output, void* scratch) {
  const uint32_t size = operation->size;
  const uint32_t nInt4 = operation->size / sizeof(int4);
  const uint32_t inputOffset4 = operation->inputOffset / sizeof(int4);
  const uint32_t outputOffset4 = operation->outputOffset / sizeof(int4);
  uint8_t* srcChannelIndexes = operation->inputChannelIndexes;
  uint8_t* dstChannelIndexes = operation->outputChannelIndexes;
  uint8_t nSrcChannels = operation->nInputs;
  uint8_t nDstChannels = operation->nOutputs;
  uint32_t* srcOffsets = operation->inputOffsets;
  uint32_t* dstOffsets = operation->outputOffsets;
  int4* input4 = (int4*)input;
  int4* output4 = (int4*)output;
  for (size_t idx = threadIdx.x; idx < nInt4; idx += blockDim.x) {
    int4 tmp = input4[inputOffset4 + idx];
    for (int index = 0; index < nSrcChannels; ++index) {
      int4 val;
      uint32_t srcOffset = srcOffsets[index] / sizeof(int4);
      val = memoryChannels[srcChannelIndexes[index]].read<int4>(srcOffset + idx);
      tmp = add_vectors<T>(tmp, val);
    }
    output4[outputOffset4 + idx] = tmp;
    if constexpr (SendToRemote) {
      for (int index = 0; index < nDstChannels; ++index) {
        uint32_t dstOffset = dstOffsets[index] / sizeof(int4);
        memoryChannels[dstChannelIndexes[index]].write<int4>(dstOffset + idx, tmp);
      }
    }
  }
  // handle rest of data
  size_t processed = nInt4 * sizeof(int4);
  const size_t startIdx = (operation->inputOffset + processed) / sizeof(T);
  const size_t endIdx = (operation->inputOffset + size) / sizeof(T);
  for (size_t idx = threadIdx.x + startIdx; idx < endIdx; idx += blockDim.x) {
    T tmp = static_cast<T*>(input)[idx];
    for (int index = 0; index < nSrcChannels; ++index) {
      size_t srcOffset = srcOffsets[index] / sizeof(T);
      tmp = add_elements(tmp, memoryChannels[srcChannelIndexes[index]].read<T>(srcOffset + idx));
    }
    static_cast<T*>(output)[idx] = tmp;
    if constexpr (SendToRemote) {
      for (int index = 0; index < nDstChannels; ++index) {
        size_t dstOffset = dstOffsets[index] / sizeof(T);
        memoryChannels[dstChannelIndexes[index]].write<T>(dstOffset + idx, tmp);
      }
    }
  }
}

template <typename PacketType>
MSCCLPP_DEVICE_INLINE void handlePutPacket(Operation2* operation, void* input, void* output, void* scratch) {
  ChannelType chType = operation->channelType;
  uint16_t nDstChannels = operation->nOutputs;
  uint32_t* dstOffsets = operation->outputOffsets;
  uint32_t* srcOffsets = operation->inputOffsets;
  uint32_t size = operation->size;
  uint8_t* dstChannelIndexes = operation->outputChannelIndexes;
  const size_t scratchBaseOffset = flag & 0x1 ? 0 : scratchSize >> 1;
  if (chType == ChannelType::MEMORY) {
    for (int index = 0; index < nDstChannels; ++index) {
      memoryChannels[dstChannelIndexes[index]].putPackets<PacketType>(
          scratchBaseOffset + dstOffsets[index] * 2, srcOffsets[index], size, threadIdx.x, blockDim.x, flag);
    }
  }
  if (chType == ChannelType::PORT) {
    int tid = threadIdx.x;
    if (tid >= nDstChannels) {
      return;
    }
    // For port channel, we assume src and dst are in packet format
    // TODO: support non-packet format and remove packet format(packet format should be handle in handleReadPutPacket)
    uint32_t dstOffset = (dstOffsets[tid] << 1) + scratchBaseOffset;
    uint32_t srcOffset = (srcOffsets[tid] << 1) + scratchBaseOffset;
    portChannels[dstChannelIndexes[tid]].put(dstOffset, srcOffset, size << 1);
  }
}

template <typename PacketType>
MSCCLPP_DEVICE_INLINE void handleReadPutPacket(Operation2* operation, void* input, void* output, void* scratch) {
  uint32_t nDstChannels = operation->nOutputs;
  uint32_t* dstOffsets = operation->outputOffsets;
  uint32_t* srcOffsets = operation->inputOffsets;
  uint8_t* dstChannelIndexes = operation->outputChannelIndexes;
  uint32_t size = operation->size;
  ChannelType chType = operation->channelType;
  const size_t scratchBaseOffset = flag & 0x1 ? 0 : scratchSize >> 1;
  if (chType == ChannelType::MEMORY) {
    size_t nPackets = size * 2 / sizeof(PacketType);
    for (size_t pkt_idx = threadIdx.x; pkt_idx < nPackets; pkt_idx += blockDim.x) {
      for (int ch_idx = 0; ch_idx < nDstChannels; ++ch_idx) {
        PacketType* pkts = (PacketType*)((char*)scratch + scratchBaseOffset + srcOffsets[ch_idx] * 2);
        PacketPayload<PacketType> data = pkts[pkt_idx].read(flag);
        PacketType pkt(data, flag);
        size_t offset = (scratchBaseOffset + dstOffsets[ch_idx] * 2) / sizeof(PacketType);
        memoryChannels[dstChannelIndexes[ch_idx]].write(offset + pkt_idx, pkt);
      }
    }
  } else if (chType == ChannelType::PORT) {
    // Ensuring Data Is Ready
    size_t nPackets = size * 2 / sizeof(PacketType);
    for (size_t pkt_idx = threadIdx.x; pkt_idx < nPackets; pkt_idx += blockDim.x) {
      for (int ch_idx = 0; ch_idx < nDstChannels; ++ch_idx) {
        PacketType* pkts = (PacketType*)((char*)scratch + scratchBaseOffset + srcOffsets[ch_idx] * 2);
        PacketPayload<PacketType> data = pkts[pkt_idx].read(flag);
      }
    }
    __syncthreads();

    // Putting the data
    int ch_idx = threadIdx.x;
    if (ch_idx >= nDstChannels) {
      return;
    }
    uint32_t dstOffset = scratchBaseOffset + dstOffsets[ch_idx] * 2;
    uint32_t srcOffset = scratchBaseOffset + srcOffsets[ch_idx] * 2;
    portChannels[dstChannelIndexes[ch_idx]].put(dstOffset, srcOffset, size * 2);
  }
}

template <typename T, typename PacketType, bool SendToRemote = true>
MSCCLPP_DEVICE_INLINE void handleReduceSendPacket(Operation2* operation, void* dst, void* src, void* scratch) {
  const uint32_t size = operation->size;
  const uint32_t nSrcs = operation->nInputs;
  const uint32_t nDstChannels = operation->nOutputs;
  const uint32_t srcOffsetByBytes = operation->inputOffset;
  const uint32_t dstOffsetByBytes = operation->outputOffset;
  const uint32_t* inputOffsets = operation->inputOffsets;
  const uint8_t* outputChannelIndexes = operation->outputChannelIndexes;
  const uint32_t* outputOffsets = operation->outputOffsets;

  size_t nPackets = size * 2 / sizeof(PacketType);
  const size_t intputBaseOffset = flag & 0x1 ? 0 : scratchSize >> 1;
  const uint32_t srcOffset = srcOffsetByBytes / sizeof(PacketPayload<PacketType>);
  const uint32_t dstOffset = dstOffsetByBytes / sizeof(PacketPayload<PacketType>);
  PacketPayload<PacketType>* srcPacketPayload = (PacketPayload<PacketType>*)src + srcOffset;
  PacketPayload<PacketType>* dstPacketPayload = (PacketPayload<PacketType>*)dst + dstOffset;
  for (size_t idx = threadIdx.x; idx < nPackets; idx += blockDim.x) {
    PacketPayload<PacketType> data = {};
    for (int index = 0; index < nSrcs; ++index) {
      PacketType* pkt = (PacketType*)((char*)scratch + intputBaseOffset + 2 * inputOffsets[index]);
      PacketPayload<PacketType> val = pkt[idx].read(flag);
      data = add_vectors<T>(data, val);
    }
    data = add_vectors<T>(data, srcPacketPayload[idx]);
    dstPacketPayload[idx] = data;

    if constexpr (SendToRemote) {
      PacketType pkt(data, flag);
      for (int index = 0; index < nDstChannels; ++index) {
        size_t offset = (intputBaseOffset + outputOffsets[index] * 2) / sizeof(PacketType);
        memoryChannels[outputChannelIndexes[index]].write(offset + idx, pkt);
      }
    }
  }
}

template <typename PacketType>
MSCCLPP_DEVICE_INLINE void handleCopyPacket(Operation2* operation, void* dst, void* src, void* scratch) {
  const uint32_t size = operation->size;
  const uint32_t dstOffset = operation->outputOffset;
  const uint32_t srcOffset = operation->inputOffset;
  const size_t inputScratchBaseOffset = flag & 0x1 ? 0 : scratchSize >> 1;
  PacketType* srcPackets = (PacketType*)((char*)src + inputScratchBaseOffset + 2 * srcOffset);
  PacketPayload<PacketType>* result = (PacketPayload<PacketType>*)((char*)dst + dstOffset);
  size_t nPackets = size * 2 / sizeof(PacketType);
  for (size_t idx = threadIdx.x; idx < nPackets; idx += blockDim.x) {
    PacketPayload<PacketType> data = srcPackets[idx].read(flag);
    result[idx] = data;
  }
}

template <typename PacketType>
MSCCLPP_DEVICE_INLINE void handleTransformToPacket(Operation2* op, void* dst, void* src, void* scratch) {
  uint32_t size = op->size;
  uint32_t dstOffset = op->outputOffset;
  uint32_t srcOffset = op->inputOffset;
  const size_t outputScratchBaseOffset = flag & 0x1 ? 0 : scratchSize >> 1;
  dstOffset = dstOffset * 2 + outputScratchBaseOffset;
  mscclpp::copyToPackets<PacketType>((char*)dst + dstOffset, (char*)src + srcOffset, size, threadIdx.x, blockDim.x,
                                     flag);
}

template <typename T, bool SendToRemote = true>
MSCCLPP_DEVICE_INLINE void handleReduceSend(Operation2* op, void* dst, void* src, void* scratch) {
  const uint32_t size = op->size;
  const uint32_t nInt4 = size / sizeof(int4);
  int nSrcs = op->nInputs;
  int nOutChannels = op->nOutputs;
  uint32_t* inputOffsets = op->inputOffsets;
  uint32_t* outputOffsets = op->outputOffsets;
  uint8_t* outputChannelIndexes = op->outputChannelIndexes;
  T* input =
      getBuffer<T>(static_cast<T*>(src), static_cast<T*>(dst), static_cast<T*>(scratch), op->localInputBufferType);
  uint32_t srcOffsetByBytes = op->inputOffset;

  const uint32_t srcOffset4 = op->inputOffset / sizeof(int4);
  const uint32_t dstOffset4 = op->outputOffset / sizeof(int4);
  int4* src4 = (int4*)src;
  int4* dst4 = (int4*)dst;
  int4* input4 = (int4*)input; // need to fix
  for (size_t idx = threadIdx.x; idx < nInt4; idx += blockDim.x) {
    int4 tmp = src4[srcOffset4 + idx];
    for (int index = 0; index < nSrcs; ++index) {
      size_t offset = inputOffsets[index] / sizeof(int4);
      int4 val = input4[offset + idx];
      tmp = add_vectors<T>(tmp, val);
    }
    dst4[dstOffset4 + idx] = tmp;
    if constexpr (SendToRemote) {
      for (int index = 0; index < nOutChannels; ++index) {
        size_t offset = outputOffsets[index] / sizeof(int4);
        memoryChannels[outputChannelIndexes[index]].write<int4>(offset + idx, tmp);
      }
    }
  }
  // handle rest of data
  size_t processed = nInt4 * sizeof(int4);
  const size_t startIdx = (srcOffsetByBytes + processed) / sizeof(T);
  const size_t endIdx = (srcOffsetByBytes + size) / sizeof(T);
  for (size_t idx = threadIdx.x + startIdx; idx < endIdx; idx += blockDim.x) {
    T tmp = static_cast<T*>(src)[idx];
    for (int index = 0; index < nSrcs; ++index) {
      size_t offset = inputOffsets[index] / sizeof(T);
      tmp = add_elements(tmp, input[offset + idx]);
    }
    static_cast<T*>(dst)[idx] = tmp;
    if constexpr (SendToRemote) {
      for (int index = 0; index < nOutChannels; ++index) {
        size_t offset = outputOffsets[index] / sizeof(T);
        memoryChannels[outputChannelIndexes[index]].write<T>(offset + idx, tmp);
      }
    }
  }
}

MSCCLPP_DEVICE_INLINE void handleCopy(Operation2* op, void* dst, void* src, void* scratch) {
  uint32_t size = op->size;
  uint32_t dstOffset = op->outputOffset;
  uint32_t srcOffset = op->inputOffset;
  char* srcData = (char*)src + srcOffset;
  char* dstData = (char*)dst + dstOffset;
  mscclpp::copy(dstData, srcData, size, threadIdx.x, blockDim.x);
}

// #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
// template <typename T>
// MSCCLPP_DEVICE_INLINE void handleMultiLoadReduceStore(T* dst, T* src, uint32_t dstOffset, uint32_t srcOffset,
//                                                       size_t size) {
//   using vectorType = typename VectorType<T>::type;
//   using nvlsType = typename VectorType<T>::nvls_type;
//   // nvls can only handle 4 bytes alignment
//   assert(size % sizeof(vectorType) == 0);
//   const size_t nInt4 = size / sizeof(nvlsType);
//   const size_t srcOffset4 = srcOffset / sizeof(nvlsType);
//   const size_t dstOffset4 = dstOffset / sizeof(nvlsType);
//   nvlsType* src4 = (nvlsType*)src;
//   nvlsType* dst4 = (nvlsType*)dst;
//   for (size_t idx = threadIdx.x; idx < nInt4; idx += blockDim.x) {
//     nvlsType val;
//     DeviceMulticastPointerDeviceHandle::multimemLoadReduce(val, (vectorType*)(src4 + srcOffset4 + idx));
//     DeviceMulticastPointerDeviceHandle::multimemStore(val, (vectorType*)(dst4 + dstOffset4 + idx));
//   }
//   // handle rest of data
//   size_t processed = nInt4 * sizeof(nvlsType);
//   using nvlsType2 = typename VectorType<T>::nvls_type2;
//   const size_t startIdx = (srcOffset + processed) / sizeof(nvlsType2);
//   const size_t endIdx = (dstOffset + size) / sizeof(nvlsType2);
//   for (size_t idx = threadIdx.x + startIdx; idx < endIdx; idx += blockDim.x) {
//     nvlsType2 val;
//     DeviceMulticastPointerDeviceHandle::multimemLoadReduce(val, (vectorType*)src + idx);
//     DeviceMulticastPointerDeviceHandle::multimemStore(val, (vectorType*)dst + idx);
//   }
// }
// #endif

template <typename T, typename PacketType>
MSCCLPP_DEVICE_INLINE void handlePipeline(Operation2* operations, void* input, void* output, void* scratch) {
  uint16_t nIterations = operations->nIterations;
  uint16_t nOperations = operations->nOperations;
  uint32_t unitSize = operations->unitSize;
  for (uint16_t i = 0; i < nIterations; i++) {
    for (uint16_t opId = 0; opId < nOperations; opId++) {
      uint32_t size =
          (operations[opId].size - i * unitSize) > unitSize ? unitSize : max(operations[opId].size - i * unitSize, 0);
      operations[opId].size = size;
      if (size == 0) {
        continue;
      }
      DeviceFunction func = getDeviceFunction<T, PacketType>(operations[opId], nullptr);
      func(operations + opId, input, output, scratch);
    }
  }
}

template <typename T, typename PacketType>
MSCCLPP_DEVICE_INLINE DeviceFunction getDeviceFunction(const Operation2& op, uint8_t* nSteps) {
  *nSteps = 1;
  OperationType opType = op.type;
  if (opType == OperationType::NOP) {
    return handleNop;
  }
  if (opType == OperationType::BARRIER) {
    return handleBarrier;
  }
  if (opType == OperationType::SIGNAL) {
    return handleSignal;
  }
  if (opType == OperationType::WAIT) {
    return handleWait;
  }
  if (opType == OperationType::FLUSH) {
    return handleFlush;
  }
  if (opType == OperationType::PUT) {
    return handlePut;
  }
  if (opType == OperationType::PUT_WITH_SIGNAL) {
    return handlePut<true>;
  }
  if (opType == OperationType::PUT_WITH_SIGNAL_AND_FLUSH) {
    return handlePut<true, true>;
  }
  if (opType == OperationType::PUT_PACKET) {
    return handlePutPacket<PacketType>;
  }
  if (opType == OperationType::GET) {
    return handleGet;
  }
  if (opType == OperationType::READ_REDUCE_COPY_SEND) {
    return handleReadReduceCopySend<T, true>;
  }
  if (opType == OperationType::READ_REDUCE_COPY) {
    return handleReadReduceCopySend<T, false>;
  }
  if (opType == OperationType::COPY) {
    return handleCopy;
  }
  if (opType == OperationType::REDUCE_SEND) {
    return handleReduceSend<T>;
  }
  if (opType == OperationType::PIPELINE) {
    *nSteps = op.nIterations;
    return handlePipeline<T, PacketType>;
  }
  return nullptr;
}

template <typename T, typename PacketType = LL16Packet>
__global__ void executionKernel([[maybe_unused]] int rank /*for debug*/, T* input, T* output, T* scratch,
                                size_t scratchSize, DeviceExecutionPlan* plan, uint32_t flag
#if defined(ENABLE_NPKIT)
                                ,
                                NpKitEventCollectContext* npKitEventCollectContexts, uint64_t* cpuTimestamp) {
#else
) {
#endif
  extern __shared__ int4 sharedMem[];
  int bid = blockIdx.x;
  int tid = threadIdx.x;
#if defined(ENABLE_NPKIT)
  NpKitEvent* event_buffer = (NpKitEvent*)((char*)sharedMem + sizeof(DeviceExecutionPlan));
  uint64_t event_buffer_head = 0;
#if defined(ENABLE_NPKIT_EVENT_EXECUTOR_INIT_ENTRY) && defined(ENABLE_NPKIT_EVENT_EXECUTOR_INIT_EXIT)
  uint64_t npkit_timestamp_entry = 0;
  if (tid == 0) {
    npkit_timestamp_entry = NPKIT_GET_GPU_TIMESTAMP();
  }
#endif
#endif
  DeviceExecutionPlan* localPlan = plan + bid;
  for (size_t i = tid; i < sizeof(DeviceExecutionPlan) / sizeof(int4); i += blockDim.x) {
    sharedMem[i] = ((int4*)localPlan)[i];
  }
  __syncshm();
  localPlan = (DeviceExecutionPlan*)sharedMem;
  int nOperations = localPlan->nOperations;
  Operation2* operations = (Operation2*)localPlan->operations;
  memoryChannels = localPlan->channels.memoryChannels;
  portChannels = localPlan->channels.portChannels;
  [[maybe_unused]] DeviceHandle<NvlsConnection::DeviceMulticastPointer>* nvlsChannels =
      localPlan->channels.nvlsChannels;

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_CPU)
#if defined(MSCCLPP_DEVICE_HIP)
  NpKit::CollectGpuEventShm(NPKIT_EVENT_TIME_SYNC_CPU, 0, 0, NPKIT_LOAD_CPU_TIMESTAMP_PER_BLOCK(cpuTimestamp, bid),
#else
  NpKit::CollectGpuEventShm(NPKIT_EVENT_TIME_SYNC_CPU, 0, 0, *cpuTimestamp,
#endif
                            event_buffer, &event_buffer_head);
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_TIME_SYNC_GPU)
  NpKit::CollectGpuEventShm(NPKIT_EVENT_TIME_SYNC_GPU, 0, 0, NPKIT_GET_GPU_TIMESTAMP(), event_buffer,
                            &event_buffer_head);
#endif

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_EXECUTOR_INIT_ENTRY) && \
    defined(ENABLE_NPKIT_EVENT_EXECUTOR_INIT_EXIT)
  NpKit::CollectGpuEventShm(NPKIT_EVENT_EXECUTOR_INIT_ENTRY, 0, 0, npkit_timestamp_entry, event_buffer,
                            &event_buffer_head);
  NpKit::CollectGpuEventShm(NPKIT_EVENT_EXECUTOR_INIT_EXIT, 0, 0, NPKIT_GET_GPU_TIMESTAMP(), event_buffer,
                            &event_buffer_head);
#endif

  for (int i = 0; i < nOperations;) {
    Operation2& op = operations[i];

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_EXECUTOR_OP_BASE_ENTRY)
    NpKit::CollectGpuEventShm(NPKIT_EVENT_EXECUTOR_OP_BASE_ENTRY + (int)op.type, op.size, 0, NPKIT_GET_GPU_TIMESTAMP(),
                              event_buffer, &event_buffer_head);
#endif
    uint8_t nSteps = 0;
    DeviceFunction function = getDeviceFunction<T, PacketType>(op, &nSteps);
    function(operations + i, input, output, scratch);
    i += nSteps;

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_EXECUTOR_OP_BASE_EXIT)
    NpKit::CollectGpuEventShm(NPKIT_EVENT_EXECUTOR_OP_BASE_EXIT + (int)op.type, op.size, 0, NPKIT_GET_GPU_TIMESTAMP(),
                              event_buffer, &event_buffer_head);
#endif
  }

#if defined(ENABLE_NPKIT)
  NpKit::StoreGpuEventShm(npKitEventCollectContexts, event_buffer, event_buffer_head);
#endif
}
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

class ExecutionKernel {
 public:
#if defined(MSCCLPP_DEVICE_HIP)
  template <typename PacketType>
  static void launchKernel(int rank, int nthreadblocks, int nthreads, void* src, void* dst, void* scratch,
                           size_t scratchSize, DataType dataType, DeviceExecutionPlan* plan, size_t sharedMemSize,
                           cudaStream_t stream, uint32_t flag = 0) {
    switch (dataType) {
      case DataType::INT32:
        executionKernel<int32_t, PacketType><<<nthreadblocks, nthreads, sharedMemSize, stream>>>(
            rank, (int32_t*)src, (int32_t*)dst, (int32_t*)scratch, scratchSize, plan, flag
#if defined(ENABLE_NPKIT)
            ,
            NpKit::GetGpuEventCollectContexts(), NpKit::GetCpuTimestamp());
#else
        );
#endif
        break;
      case DataType::UINT32:
        executionKernel<uint32_t, PacketType><<<nthreadblocks, nthreads, sharedMemSize, stream>>>(
            rank, (uint32_t*)src, (uint32_t*)dst, (uint32_t*)scratch, scratchSize, plan, flag
#if defined(ENABLE_NPKIT)
            ,
            NpKit::GetGpuEventCollectContexts(), NpKit::GetCpuTimestamp());
#else
        );
#endif
        break;
      case DataType::FLOAT16:
        executionKernel<half, PacketType><<<nthreadblocks, nthreads, sharedMemSize, stream>>>(
            rank, (half*)src, (half*)dst, (half*)scratch, scratchSize, plan, flag
#if defined(ENABLE_NPKIT)
            ,
            NpKit::GetGpuEventCollectContexts(), NpKit::GetCpuTimestamp());
#else
        );
#endif
        break;
      case DataType::FLOAT32:
        executionKernel<float, PacketType><<<nthreadblocks, nthreads, sharedMemSize, stream>>>(
            rank, (float*)src, (float*)dst, (float*)scratch, scratchSize, plan, flag
#if defined(ENABLE_NPKIT)
            ,
            NpKit::GetGpuEventCollectContexts(), NpKit::GetCpuTimestamp());
#else
        );
#endif
        break;
      case DataType::BFLOAT16:
        executionKernel<__bfloat16, PacketType><<<nthreadblocks, nthreads, sharedMemSize, stream>>>(
            rank, (__bfloat16*)src, (__bfloat16*)dst, (__bfloat16*)scratch, scratchSize, plan, flag
#if defined(ENABLE_NPKIT)
            ,
            NpKit::GetGpuEventCollectContexts(), NpKit::GetCpuTimestamp());
#else
        );
#endif
        break;
    }
  }
#else   // !defined(MSCCLPP_DEVICE_HIP)
  template <typename PacketType>
  static void launchKernel(int rank, int nthreadblocks, int nthreads, void* src, void* dst, void* scratch,
                           size_t scratchSize, DataType dataType, DeviceExecutionPlan* plan, size_t sharedMemSize,
                           cudaStream_t stream, uint32_t flag = 0);
#endif  // !defined(MSCCLPP_DEVICE_HIP)
};
}  // namespace mscclpp

#endif  // MSCCLPP_EXECUTION_KERNEL_HPP_
