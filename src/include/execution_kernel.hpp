// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_EXECUTION_KERNEL_HPP_
#define MSCCLPP_EXECUTION_KERNEL_HPP_

#include <mscclpp/executor.hpp>
#if defined(ENABLE_NPKIT)
#include <mscclpp/npkit/npkit.hpp>
#endif
#include <mscclpp/concurrency_device.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/packet_device.hpp>
#include <mscclpp/port_channel.hpp>

#include "execution_common.hpp"

#if defined(MSCCLPP_DEVICE_COMPILE)
#include <mscclpp/gpu_data_types.hpp>
#include <mscclpp/switch_channel_device.hpp>

namespace {
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

}  // namespace
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

namespace mscclpp {

#define MAX_DEVICE_SYNCERS 16
__device__ DeviceSyncer deviceSyncers[MAX_DEVICE_SYNCERS];

#if defined(MSCCLPP_DEVICE_COMPILE)

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

MSCCLPP_DEVICE_INLINE void handleSignal(DeviceHandle<MemoryChannel>* memoryChannels,
                                        DeviceHandle<PortChannel>* portChannels, uint8_t* channelIndex, int nChannels,
                                        ChannelType chType) {
  int tid = threadIdx.x;
  if (tid < nChannels && chType == ChannelType::MEMORY) {
    memoryChannels[channelIndex[tid]].signal();
    return;
  }
  if (tid < nChannels && chType == ChannelType::PORT) {
    portChannels[channelIndex[threadIdx.x]].signal();
  }
}

MSCCLPP_DEVICE_INLINE void handleWait(DeviceHandle<MemoryChannel>* memoryChannels,
                                      DeviceHandle<PortChannel>* portChannels, uint8_t* channelIndexes, int nChannels,
                                      ChannelType chType) {
  int tid = threadIdx.x;
  if (tid < nChannels && chType == ChannelType::MEMORY) {
    memoryChannels[channelIndexes[tid]].wait();
    return;
  }
  if (tid < nChannels && chType == ChannelType::PORT) {
    portChannels[channelIndexes[tid]].wait();
  }
}

MSCCLPP_DEVICE_INLINE void handleFlush(DeviceHandle<PortChannel>* portChannels, uint8_t* channelIndexes,
                                       int nChannels) {
  int tid = threadIdx.x;
  if (tid < nChannels) {
    portChannels[channelIndexes[tid]].flush();
  }
}

MSCCLPP_DEVICE_INLINE void handleGet(DeviceHandle<MemoryChannel>* memoryChannel, uint8_t* srcChannelIndexes,
                                     uint32_t* dstOffsets, uint32_t* srcOffsets, int count, uint32_t size) {
  for (int i = 0; i < count; i++) {
    uint32_t dstOffset = dstOffsets[i];
    uint32_t srcOffset = srcOffsets[i];
    memoryChannel[srcChannelIndexes[i]].get(srcOffset, dstOffset, size, threadIdx.x, blockDim.x);
  }
}

template <bool PutWithSignal = false, bool PutWithSignalAndFlush = false>
MSCCLPP_DEVICE_INLINE void handlePut(DeviceHandle<MemoryChannel>* memoryChannel,
                                     DeviceHandle<PortChannel>* portChannels, uint8_t* dstChannelIndexes,
                                     uint32_t* dstOffsets, uint32_t* srcOffsets, int count, uint32_t size,
                                     ChannelType chType) {
  if (chType == ChannelType::MEMORY) {
    for (int i = 0; i < count; i++) {
      uint32_t dstOffset = dstOffsets[i];
      uint32_t srcOffset = srcOffsets[i];
      memoryChannel[dstChannelIndexes[i]].put(dstOffset, srcOffset, size, threadIdx.x, blockDim.x);
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

template <typename T>
MSCCLPP_DEVICE_INLINE void handleReadReduceCopySend(T* output, uint32_t outputOffsetByBytes, T* input,
                                                    uint32_t inputOffsetByBytes,
                                                    DeviceHandle<MemoryChannel>* memoryChannels,
                                                    uint8_t* dstChannelIndexes, uint8_t* srcChannelIndexes,
                                                    uint32_t* dstOffsets, uint32_t* srcOffsets, int nDstChannels,
                                                    int nSrcChannels, uint32_t size, bool sendToRemote = true) {
  const size_t nInt4 = size / sizeof(int4);
  const size_t inputOffset4 = inputOffsetByBytes / sizeof(int4);
  const size_t outputOffset4 = outputOffsetByBytes / sizeof(int4);
  int4* input4 = (int4*)input;
  int4* output4 = (int4*)output;
  for (size_t idx = threadIdx.x; idx < nInt4; idx += blockDim.x) {
    int4 tmp = input4[inputOffset4 + idx];
    for (int index = 0; index < nSrcChannels; ++index) {
      int4 val;
      size_t srcOffset = srcOffsets[index] / sizeof(int4);
      val = memoryChannels[srcChannelIndexes[index]].read<int4>(srcOffset + idx);
      tmp = add_vectors<T>(tmp, val);
    }
    output4[outputOffset4 + idx] = tmp;
    if (sendToRemote) {
      for (int index = 0; index < nDstChannels; ++index) {
        size_t dstOffset = dstOffsets[index] / sizeof(int4);
        memoryChannels[dstChannelIndexes[index]].write<int4>(dstOffset + idx, tmp);
      }
    }
  }
  // handle rest of data
  size_t processed = nInt4 * sizeof(int4);
  const size_t startIdx = (inputOffsetByBytes + processed) / sizeof(T);
  const size_t endIdx = (inputOffsetByBytes + size) / sizeof(T);
  for (size_t idx = threadIdx.x + startIdx; idx < endIdx; idx += blockDim.x) {
    T tmp = input[idx];
    for (int index = 0; index < nSrcChannels; ++index) {
      size_t srcOffset = srcOffsets[index] / sizeof(T);
      tmp = add_elements(tmp, memoryChannels[srcChannelIndexes[index]].read<T>(srcOffset + idx));
    }
    output[idx] = tmp;
    if (sendToRemote) {
      for (int index = 0; index < nDstChannels; ++index) {
        size_t dstOffset = dstOffsets[index] / sizeof(T);
        memoryChannels[dstChannelIndexes[index]].write<T>(dstOffset + idx, tmp);
      }
    }
  }
}

template <typename PacketType>
MSCCLPP_DEVICE_INLINE void handlePutPacket(size_t scratchSize, DeviceHandle<MemoryChannel>* memoryChannels,
                                           DeviceHandle<PortChannel>* portChannels, uint8_t* dstChannelIndexes,
                                           uint32_t* dstOffsets, uint32_t* srcOffsets, int nDstChannels, uint32_t size,
                                           ChannelType chType, uint32_t flag) {
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
MSCCLPP_DEVICE_INLINE void handleReadPutPacket(int rank, void* scratch, size_t scratchSize,
                                               DeviceHandle<SmChannel>* smChannels,
                                               DeviceHandle<ProxyChannel>* proxyChannels, uint8_t* dstChannelIndexes,
                                               uint32_t* dstOffsets, uint32_t* srcOffsets, int nDstChannels,
                                               uint32_t size, ChannelType chType, uint32_t flag) {
  const size_t scratchBaseOffset = flag & 0x1 ? 0 : scratchSize >> 1;
  if (chType == ChannelType::MEMORY) {
    size_t nPackets = size * 2 / sizeof(PacketType);
    for (size_t pkt_idx = threadIdx.x; pkt_idx < nPackets; pkt_idx += blockDim.x) {
      for (int ch_idx = 0; ch_idx < nDstChannels; ++ch_idx) {
        PacketType* pkts = (PacketType*)((char*)scratch + scratchBaseOffset + srcOffsets[ch_idx] * 2);
        PacketPayload<PacketType> data = pkts[pkt_idx].read(flag);
        PacketType pkt(data, flag);
        size_t offset = (scratchBaseOffset + dstOffsets[ch_idx] * 2) / sizeof(PacketType);
        smChannels[dstChannelIndexes[ch_idx]].write(offset + pkt_idx, pkt);
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
    proxyChannels[dstChannelIndexes[ch_idx]].put(dstOffset, srcOffset, size * 2);
  }
}

template <typename T, typename PacketType, bool SendToRemote = true>
MSCCLPP_DEVICE_INLINE void handleReduceSendPacket(T* dst, uint32_t dstOffsetByBytes, T* src, uint32_t srcOffsetByBytes,
                                                  T* inputBuff, size_t inputBuffSize, uint32_t* inputOffsets, int nSrcs,
                                                  DeviceHandle<MemoryChannel>* memoryChannels,
                                                  uint8_t* outputChannelIndexes, uint32_t* outputOffsets,
                                                  int nDstChannels, size_t size, uint32_t flag) {
  size_t nPackets = size * 2 / sizeof(PacketType);
  const size_t intputBaseOffset = flag & 0x1 ? 0 : inputBuffSize >> 1;
  const uint32_t srcOffset = srcOffsetByBytes / sizeof(PacketPayload<PacketType>);
  const uint32_t dstOffset = dstOffsetByBytes / sizeof(PacketPayload<PacketType>);
  PacketPayload<PacketType>* srcPacketPayload = (PacketPayload<PacketType>*)src + srcOffset;
  PacketPayload<PacketType>* dstPacketPayload = (PacketPayload<PacketType>*)dst + dstOffset;
  for (size_t idx = threadIdx.x; idx < nPackets; idx += blockDim.x) {
    PacketPayload<PacketType> data = {};
    for (int index = 0; index < nSrcs; ++index) {
      PacketType* pkt = (PacketType*)((char*)inputBuff + intputBaseOffset + 2 * inputOffsets[index]);
      PacketPayload<PacketType> val = pkt[idx].read(flag);
      data = add_vectors<T>(data, val);
    }
    data = add_vectors<T>(data, srcPacketPayload[idx]);
    dstPacketPayload[idx] = data;

    if (SendToRemote) {
      PacketType pkt(data, flag);
      for (int index = 0; index < nDstChannels; ++index) {
        size_t offset = (intputBaseOffset + outputOffsets[index] * 2) / sizeof(PacketType);
        memoryChannels[outputChannelIndexes[index]].write(offset + idx, pkt);
      }
    }
  }
}

template <typename PacketType>
MSCCLPP_DEVICE_INLINE void handleCopyPacket(void* dst, void* src, size_t srcSize, uint32_t dstOffset,
                                            uint32_t srcOffset, size_t size, uint32_t flag) {
  const size_t inputScratchBaseOffset = flag & 0x1 ? 0 : srcSize >> 1;
  PacketType* srcPackets = (PacketType*)((char*)src + inputScratchBaseOffset + 2 * srcOffset);
  PacketPayload<PacketType>* result = (PacketPayload<PacketType>*)((char*)dst + dstOffset);
  size_t nPackets = size * 2 / sizeof(PacketType);
  for (size_t idx = threadIdx.x; idx < nPackets; idx += blockDim.x) {
    PacketPayload<PacketType> data = srcPackets[idx].read(flag);
    result[idx] = data;
  }
}

template <typename PacketType>
MSCCLPP_DEVICE_INLINE void handleTransformToPacket(void* dst, void* src, size_t dstSize, uint32_t dstOffset,
                                                   uint32_t srcOffset, size_t size, uint32_t flag) {
  const size_t outputScratchBaseOffset = flag & 0x1 ? 0 : dstSize >> 1;
  dstOffset = dstOffset * 2 + outputScratchBaseOffset;
  mscclpp::copyToPackets<PacketType>((char*)dst + dstOffset, (char*)src + srcOffset, size, threadIdx.x, blockDim.x,
                                     flag);
}

template <typename T, bool SendToRemote = true>
MSCCLPP_DEVICE_INLINE void handleReduceSend(T* dst, uint32_t dstOffsetByBytes, T* src, uint32_t srcOffsetByBytes,
                                            T* input, uint32_t* inputOffsets, int nSrcs,
                                            DeviceHandle<MemoryChannel>* memoryChannels, uint8_t* outputChannelIndexes,
                                            uint32_t* outputOffsets, int nOutChannels, uint32_t size) {
  const size_t nInt4 = size / sizeof(int4);
  const size_t srcOffset4 = srcOffsetByBytes / sizeof(int4);
  const size_t dstOffset4 = dstOffsetByBytes / sizeof(int4);
  int4* src4 = (int4*)src;
  int4* dst4 = (int4*)dst;
  int4* input4 = (int4*)input;
  for (size_t idx = threadIdx.x; idx < nInt4; idx += blockDim.x) {
    int4 tmp = src4[srcOffset4 + idx];
    for (int index = 0; index < nSrcs; ++index) {
      size_t offset = inputOffsets[index] / sizeof(int4);
      int4 val = input4[offset + idx];
      tmp = add_vectors<T>(tmp, val);
    }
    dst4[dstOffset4 + idx] = tmp;
    if (SendToRemote) {
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
    T tmp = src[idx];
    for (int index = 0; index < nSrcs; ++index) {
      size_t offset = inputOffsets[index] / sizeof(T);
      tmp = add_elements(tmp, input[offset + idx]);
    }
    dst[idx] = tmp;
    if (SendToRemote) {
      for (int index = 0; index < nOutChannels; ++index) {
        size_t offset = outputOffsets[index] / sizeof(T);
        memoryChannels[outputChannelIndexes[index]].write<T>(offset + idx, tmp);
      }
    }
  }
}

MSCCLPP_DEVICE_INLINE void handleCopy(void* dst, void* src, uint32_t dstOffset, uint32_t srcOffset, size_t size) {
  char* srcData = (char*)src + srcOffset;
  char* dstData = (char*)dst + dstOffset;
  mscclpp::copy(dstData, srcData, size, threadIdx.x, blockDim.x);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
template <typename T>
MSCCLPP_DEVICE_INLINE void handleMultiLoadReduceStore(T* dst, T* src, uint32_t dstOffset, uint32_t srcOffset,
                                                      size_t size) {
  static_assert(sizeof(T) <= 8, "Only support type with size <= 8 bytes");
  // TODO: use `nelems` instead of `size` to avoid the size check
  assert(size % sizeof(T) == 0);
  assert(srcOffset % sizeof(T) == 0);
  assert(dstOffset % sizeof(T) == 0);
  if constexpr (std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>) {
    const size_t nElem = size / sizeof(T);
    const size_t srcOffsetElem = srcOffset / sizeof(T);
    const size_t dstOffsetElem = dstOffset / sizeof(T);
    VectorType<T, 1>* srcElem = reinterpret_cast<VectorType<T, 1>*>(src + srcOffsetElem);
    VectorType<T, 1>* dstElem = reinterpret_cast<VectorType<T, 1>*>(dst + dstOffsetElem);
    for (size_t idx = threadIdx.x; idx < nElem; idx += blockDim.x) {
      auto val = SwitchChannelDeviceHandle::multimemLoadReduce(srcElem + idx);
      SwitchChannelDeviceHandle::multimemStore(val, dstElem + idx);
    }
  } else {
    // handle data in 16-byte unit
    using Type16 = typename mscclpp::VectorType<T, 16 / sizeof(T)>;
    const size_t nType16 = size / sizeof(Type16);
    const size_t srcOffset16 = srcOffset / sizeof(Type16);
    const size_t dstOffset16 = dstOffset / sizeof(Type16);
    Type16* src16 = reinterpret_cast<Type16*>(src) + srcOffset16;
    Type16* dst16 = reinterpret_cast<Type16*>(dst) + dstOffset16;
    for (size_t idx = threadIdx.x; idx < nType16; idx += blockDim.x) {
      Type16 val = SwitchChannelDeviceHandle::multimemLoadReduce(src16 + idx);
      SwitchChannelDeviceHandle::multimemStore(val, dst16 + idx);
    }
    // handle rest of data
    constexpr int RedBytes = (sizeof(T) == 8) ? 8 : 4;
    using TypeRest = typename mscclpp::VectorType<T, RedBytes / sizeof(T)>;
    const size_t processed = nType16 * sizeof(Type16);
    const size_t nRest = (size - processed) / sizeof(TypeRest);
    TypeRest* srcR = reinterpret_cast<TypeRest*>(src + srcOffset + processed);
    TypeRest* dstR = reinterpret_cast<TypeRest*>(dst + dstOffset + processed);
    for (size_t idx = threadIdx.x; idx < nRest; idx += blockDim.x) {
      TypeRest val = SwitchChannelDeviceHandle::multimemLoadReduce(srcR + idx);
      SwitchChannelDeviceHandle::multimemStore(val, dstR + idx);
    }
  }
}
#endif

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
  Operation* operations = localPlan->operations;
  DeviceHandle<MemoryChannel>* memoryChannels = localPlan->channels.memoryChannels;
  DeviceHandle<PortChannel>* portChannels = localPlan->channels.portChannels;
  [[maybe_unused]] DeviceHandle<SwitchChannel>* nvlsChannels = localPlan->channels.nvlsChannels;

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

  for (int i = 0; i < nOperations; i++) {
    Operation& op = operations[i];

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_EXECUTOR_OP_BASE_ENTRY)
    NpKit::CollectGpuEventShm(NPKIT_EVENT_EXECUTOR_OP_BASE_ENTRY + (int)op.type, op.size, 0, NPKIT_GET_GPU_TIMESTAMP(),
                              event_buffer, &event_buffer_head);
#endif

    if (op.type == OperationType::NOP) {
      __syncthreads();
    } else if (op.type == OperationType::BARRIER) {
      int nThreadBlocks = op.nThreadBlocks;
      int syncStateIndex = op.deviceSyncerIndex;
      deviceSyncers[syncStateIndex].sync(nThreadBlocks);
    } else if (op.type == OperationType::SIGNAL) {
      handleSignal(memoryChannels, portChannels, op.outputChannelIndexes, op.nOutputs, op.channelType);
    } else if (op.type == OperationType::WAIT) {
      handleWait(memoryChannels, portChannels, op.inputChannelIndexes, op.nInputs, op.channelType);
    } else if (op.type == OperationType::FLUSH) {
      handleFlush(portChannels, op.outputChannelIndexes, op.nOutputs);
    } else if (op.type == OperationType::PUT) {
      handlePut(memoryChannels, portChannels, op.outputChannelIndexes, op.outputOffsets, op.inputOffsets, op.nOutputs,
                op.size, op.channelType);
    } else if (op.type == OperationType::PUT_WITH_SIGNAL) {
      handlePut<true>(memoryChannels, portChannels, op.outputChannelIndexes, op.outputOffsets, op.inputOffsets,
                      op.nOutputs, op.size, op.channelType);
    } else if (op.type == OperationType::PUT_WITH_SIGNAL_AND_FLUSH) {
      handlePut<false, true>(memoryChannels, portChannels, op.outputChannelIndexes, op.outputOffsets, op.inputOffsets,
                             op.nOutputs, op.size, op.channelType);
    } else if (op.type == OperationType::GET) {
      handleGet(memoryChannels, op.inputChannelIndexes, op.outputOffsets, op.inputOffsets, op.nInputs, op.size);
    } else if (op.type == OperationType::COPY) {
      T* dst = getBuffer(input, output, scratch, op.dstBufferType);
      T* src = getBuffer(input, output, scratch, op.srcBufferType);
      handleCopy(dst, src, op.dstOffset, op.srcOffset, op.size);
    } else if (op.type == OperationType::READ_REDUCE_COPY_SEND) {
      T* dst = getBuffer(input, output, scratch, op.dstBufferType);
      T* src = getBuffer(input, output, scratch, op.srcBufferType);
      handleReadReduceCopySend(dst, op.dstOffset, src, op.srcOffset, memoryChannels, op.outputChannelIndexes,
                               op.inputChannelIndexes, op.outputOffsets, op.inputOffsets, op.nOutputs, op.nInputs,
                               op.size);
    } else if (op.type == OperationType::READ_REDUCE_COPY) {
      T* dst = getBuffer(input, output, scratch, op.dstBufferType);
      T* src = getBuffer(input, output, scratch, op.srcBufferType);

      handleReadReduceCopySend(dst, op.dstOffset, src, op.srcOffset, memoryChannels, op.outputChannelIndexes,
                               op.inputChannelIndexes, op.outputOffsets, op.inputOffsets, op.nOutputs, op.nInputs,
                               op.size, false);
    } else if (op.type == OperationType::READ_PUT_PACKET) {
      handleReadPutPacket<PacketType>(rank, scratch, scratchSize, memoryChannels, portChannels, op.outputChannelIndexes,
                                      op.outputOffsets, op.inputOffsets, op.nOutputs, op.size, op.channelType, flag);
    } else if (op.type == OperationType::PUT_PACKET) {
      handlePutPacket<PacketType>(scratchSize, memoryChannels, portChannels, op.outputChannelIndexes, op.outputOffsets,
                                  op.inputOffsets, op.nOutputs, op.size, op.channelType, flag);
    } else if (op.type == OperationType::REDUCE_SEND_PACKET) {
      T* dst = getBuffer(input, output, scratch, op.dstBufferType);
      T* src = getBuffer(input, output, scratch, op.srcBufferType);
      handleReduceSendPacket<T, PacketType>(dst, op.dstOffset, src, op.srcOffset, scratch, scratchSize, op.inputOffsets,
                                            op.nInputs, memoryChannels, op.outputChannelIndexes, op.outputOffsets,
                                            op.nOutputs, op.size, flag);
    } else if (op.type == OperationType::REDUCE) {
      T* dst = getBuffer(input, output, scratch, op.dstBufferType);
      T* src = getBuffer(input, output, scratch, op.srcBufferType);
      T* tmp = getBuffer(input, output, scratch, op.inputBufferType);
      handleReduceSend<T, false>(dst, op.dstOffset, src, op.srcOffset, tmp, op.inputOffsets, op.nInputs, memoryChannels,
                                 op.outputChannelIndexes, op.outputOffsets, op.nOutputs, op.size);
    } else if (op.type == OperationType::REDUCE_PACKET) {
      T* dst = getBuffer(input, output, scratch, op.dstBufferType);
      T* src = getBuffer(input, output, scratch, op.srcBufferType);
      handleReduceSendPacket<T, PacketType, false>(dst, op.dstOffset, src, op.srcOffset, scratch, scratchSize,
                                                   op.inputOffsets, op.nInputs, memoryChannels, op.outputChannelIndexes,
                                                   op.outputOffsets, op.nOutputs, op.size, flag);
    } else if (op.type == OperationType::COPY_PACKET) {
      T* dst = getBuffer(input, output, scratch, op.dstBufferType);
      T* src = getBuffer(input, output, scratch, op.srcBufferType);
      handleCopyPacket<PacketType>(dst, src, scratchSize, op.dstOffset, op.srcOffset, op.size, flag);
    } else if (op.type == OperationType::TRANSFORM_TO_PACKET) {
      T* dst = getBuffer(input, output, scratch, op.dstBufferType);
      T* src = getBuffer(input, output, scratch, op.srcBufferType);
      handleTransformToPacket<PacketType>(dst, src, scratchSize, op.dstOffset, op.srcOffset, op.size, flag);
    } else if (op.type == OperationType::REDUCE_SEND) {
      T* dst = getBuffer(input, output, scratch, op.dstBufferType);
      T* src = getBuffer(input, output, scratch, op.srcBufferType);
      T* tmp = getBuffer(input, output, scratch, op.inputBufferType);
      handleReduceSend(dst, op.dstOffset, src, op.srcOffset, tmp, op.inputOffsets, op.nInputs, memoryChannels,
                       op.outputChannelIndexes, op.outputOffsets, op.nOutputs, op.size);
    }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    else if (op.type == OperationType::MULTI_LOAD_REDUCE_STORE) {
      T* dst = (T*)(nvlsChannels[op.nvlsOutputIndex].mcPtr);
      T* src = (T*)(nvlsChannels[op.nvlsInputIndex].mcPtr);
      handleMultiLoadReduceStore(dst, src, op.dstOffset, op.srcOffset, op.size);
    }
#endif

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
