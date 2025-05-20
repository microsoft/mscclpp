// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_EXECUTION_OPERATION_HPP_
#define MSCCLPP_EXECUTION_OPERATION_HPP_

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
#include <mscclpp/nvls_device.hpp>

__shared__ DeviceHandle<MemoryChannel>* memoryChannels;
__shared__ DeviceHandle<PortChannel>* portChannels;
__shared__ DeviceHandle<NvlsConnection::DeviceMulticastPointer>* nvlsChannels;


namespace mscclpp {

MSCCLPP_DEVICE_INLINE void handleNop(Operation2* operations, void* src, void* dst, void* scratch) { __syncthreads(); }

MSCCLPP_DEVICE_INLINE void handleBarrier(Operation2* operations, void* src, void* dst, void* scratch) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  DeviceSyncer* syncer = &deviceSyncers[operations[bid].syncerId];
  syncer->sync(operations[bid].blockNum);
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

}  // namespace mscclpp
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

#endif  // MSCCLPP_EXECUTION_OPERATION_HPP_