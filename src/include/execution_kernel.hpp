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

#define MAX_DEVICE_FUNCTIONS_IN_PIPELINE 16
__device__ DeviceSyncer deviceSyncers[MAX_DEVICE_SYNCERS];
__device__ DeviceSemaphore* deviceSemaphores;

__shared__ DeviceHandle<BaseMemoryChannel>* memoryChannels_;
__shared__ DeviceHandle<BasePortChannel>* portChannels_;
__shared__ DeviceHandle<NvlsConnection::DeviceMulticastPointer>* nvlsChannels_;
__shared__ void** remoteMemoriesViaMemoryChan_;
__shared__ MemoryId* remoteMemoriesViaPortChan_;
__shared__ BufferType* remoteBufferTypes_;
__shared__ uint32_t flag_;
__shared__ uint32_t scratchChunkSize_;

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

template <bool ReuseScratch>
MSCCLPP_DEVICE_INLINE uint32_t getOffset(BufferType bufferType, uint32_t offset) {
  if constexpr (!ReuseScratch) {
    return offset;
  } else if (bufferType != BufferType::SCRATCH) {
    return offset;
  } else {
    return offset % scratchChunkSize_;
  }
}

template <typename T, typename PacketType, bool ReuseScratch = false>
MSCCLPP_DEVICE_INLINE void executeDeviceFunction(const Operation& op, T* input, T* output, T* scratch,
                                                 uint8_t* nSteps = nullptr, uint32_t offset = 0,
                                                 uint32_t unitSize = UINT32_MAX);

MSCCLPP_DEVICE_INLINE void handleNop() { __syncthreads(); }

MSCCLPP_DEVICE_INLINE void handleBarrier(const Operation& op) {
  DeviceSyncer* syncer = &deviceSyncers[op.deviceSyncerIndex];
  syncer->sync(op.nThreadBlocks);
}

template <bool Relaxed = false>
MSCCLPP_DEVICE_INLINE void handleSignal(const Operation& op) {
  int nChannels = op.nChannels;
  ChannelType chType = op.channelType;
  const uint8_t* channelIndex = op.channelIndexes;
  int tid = threadIdx.x;
  if (tid < nChannels && chType == ChannelType::MEMORY) {
    if constexpr (Relaxed) {
      memoryChannels_[channelIndex[tid]].relaxedSignal();
    } else {
      memoryChannels_[channelIndex[tid]].signal();
    }
    return;
  }
  if (tid < nChannels && chType == ChannelType::PORT) {
    portChannels_[channelIndex[tid]].signal();
  }
}

template <bool Relaxed = false>
MSCCLPP_DEVICE_INLINE void handleWait(const Operation& op) {
  int nChannels = op.nChannels;
  ChannelType chType = op.channelType;
  const uint8_t* channelIndex = op.channelIndexes;
  int tid = threadIdx.x;
  if (tid < nChannels && chType == ChannelType::MEMORY) {
    if constexpr (Relaxed) {
      memoryChannels_[channelIndex[tid]].relaxedWait();
    } else {
      memoryChannels_[channelIndex[tid]].wait();
    }
    return;
  }
  if (tid < nChannels && chType == ChannelType::PORT) {
    portChannels_[channelIndex[tid]].wait();
  }
}

MSCCLPP_DEVICE_INLINE void handleFlush(const Operation& op) {
  int nChannels = op.nOutputs;
  const uint8_t* channelIndexes = op.channelIndexes;
  int tid = threadIdx.x;
  if (tid < nChannels) {
    portChannels_[channelIndexes[tid]].flush();
  }
}

template <bool ReuseScratch>
MSCCLPP_DEVICE_INLINE void handleGet(const Operation& op, void* input, void* output, void* scratch, uint32_t offset,
                                     uint32_t unitSize) {
  const uint32_t count = op.nInputs;
  const uint32_t* sizes = op.inputBufferSizes;
  const uint32_t* srcOffsets = op.inputOffsets;
  const uint32_t* dstOffsets = op.outputOffsets;
  for (int i = 0; i < count; i++) {
    uint32_t dstOffset = dstOffsets[i] + getOffset<ReuseScratch>(op.outputBufferRefs[i].type, offset);
    uint32_t srcOffset = srcOffsets[i] + getOffset<ReuseScratch>(remoteBufferTypes_[op.inputBufferRefs[i].id], offset);
    uint32_t size = min(sizes[i] - offset, unitSize);
    char* remoteMemory = static_cast<char*>(remoteMemoriesViaMemoryChan_[op.inputBufferRefs[i].id]);
    mscclpp::copy(static_cast<char*>(getBuffer(input, output, scratch, op.outputBufferRefs[i].type)) + srcOffset,
                  remoteMemory + dstOffset, size, threadIdx.x, blockDim.x);
  }
}

template <bool ReuseScratch, bool PutWithSignal = false, bool PutWithSignalAndFlush = false>
MSCCLPP_DEVICE_INLINE void handlePut(const Operation& op, void* input, void* output, void* scratch, uint32_t offset,
                                     uint32_t unitSize) {
  ChannelType chType = op.channelType;
  uint32_t count = op.nOutputs;
  const uint8_t* channelIndexes = op.channelIndexes;
  const uint32_t* dstOffsets = op.outputOffsets;
  const uint32_t* srcOffsets = op.inputOffsets;
  const uint32_t* outputSizes = op.outputBufferSizes;
  char* src = static_cast<char*>(getBuffer(input, output, scratch, op.inputBufferRefs[0].type));
  if (chType == ChannelType::MEMORY) {
    for (int i = 0; i < count; i++) {
      uint32_t dstOffset =
          dstOffsets[i] + getOffset<ReuseScratch>(remoteBufferTypes_[op.outputBufferRefs[i].id], offset);
      uint32_t srcOffset = srcOffsets[i] + getOffset<ReuseScratch>(op.inputBufferRefs[i].type, offset);
      uint32_t size = min(outputSizes[i] - offset, unitSize);
      char* remoteMemory = static_cast<char*>(remoteMemoriesViaMemoryChan_[op.outputBufferRefs[i].id]);
      mscclpp::copy(remoteMemory + dstOffset, src + srcOffset, size, threadIdx.x, blockDim.x);
    }
    return;
  }
  if (chType == ChannelType::PORT) {
    int tid = threadIdx.x;
    if (tid < count) {
      uint32_t size = min(outputSizes[tid] - offset, unitSize);
      MemoryId dstMemoryId = remoteMemoriesViaPortChan_[op.outputBufferRefs[tid].id];
      MemoryId srcMemoryId = static_cast<MemoryId>(op.inputBufferRefs[tid].type);
      uint32_t dstOffset =
          dstOffsets[tid] + getOffset<ReuseScratch>(remoteBufferTypes_[op.outputBufferRefs[tid].id], offset);
      uint32_t srcOffset = srcOffsets[tid] + getOffset<ReuseScratch>(op.inputBufferRefs[tid].type, offset);
      if constexpr (PutWithSignal) {
        portChannels_[channelIndexes[tid]].putWithSignal(dstMemoryId, dstOffset, srcMemoryId, srcOffset, size);
      } else if constexpr (PutWithSignalAndFlush) {
        portChannels_[channelIndexes[tid]].putWithSignalAndFlush(dstMemoryId, (uint64_t)dstOffset, srcMemoryId,
                                                                 (uint64_t)srcOffsets, size);
      } else {
        portChannels_[channelIndexes[tid]].put(dstMemoryId, dstOffset, srcMemoryId, srcOffset, size);
      }
    }
  }
}

template <typename T, bool ReuseScratch, bool SendToRemote = true>
MSCCLPP_DEVICE_INLINE void handleReadReduceSend(const Operation& op, void* input, void* output, void* scratch,
                                                uint32_t offset, uint32_t unitSize) {
  const uint32_t size = min(op.inputBufferSizes[0] - offset, unitSize);
  const uint32_t nInt4 = size / sizeof(int4);
  const uint32_t inputOffset4 =
      (op.inputOffsets[0] + getOffset<ReuseScratch>(op.inputBufferRefs[0].type, offset)) / sizeof(int4);
  const uint32_t outputOffset4 =
      (op.outputOffsets[0] + getOffset<ReuseScratch>(op.outputBufferRefs[0].type, offset)) / sizeof(int4);
  const uint8_t nRemoteInputs = op.nInputs - 1;
  const uint8_t nRemoteOutputs = op.nOutputs - 1;
  const uint32_t* srcOffsets = op.inputOffsets + 1;
  const uint32_t* dstOffsets = op.outputOffsets + 1;
  int4* input4 = static_cast<int4*>(getBuffer(input, output, scratch, op.inputBufferRefs[0].type));
  int4* output4 = static_cast<int4*>(getBuffer(input, output, scratch, op.outputBufferRefs[0].type));
  for (uint32_t idx = threadIdx.x; idx < nInt4; idx += blockDim.x) {
    int4 tmp = input4[inputOffset4 + idx];
    for (int index = 0; index < nRemoteInputs; ++index) {
      int4 val;
      uint32_t srcOffset =
          (srcOffsets[index] + getOffset<ReuseScratch>(remoteBufferTypes_[op.inputBufferRefs[index + 1].id], offset)) /
          sizeof(int4);
      void* remoteMemory = static_cast<char*>(remoteMemoriesViaMemoryChan_[op.inputBufferRefs[index + 1].id]);
      val = mscclpp::read<int4>(remoteMemory, srcOffset + idx);
      tmp = add_vectors<T>(tmp, val);
    }
    output4[outputOffset4 + idx] = tmp;
    if constexpr (SendToRemote) {
      for (int index = 0; index < nRemoteOutputs; ++index) {
        uint32_t dstOffset = (dstOffsets[index] +
                              getOffset<ReuseScratch>(remoteBufferTypes_[op.outputBufferRefs[index + 1].id], offset)) /
                             sizeof(int4);
        void* remoteMemory = static_cast<char*>(remoteMemoriesViaMemoryChan_[op.outputBufferRefs[index + 1].id]);
        mscclpp::write<int4>(remoteMemory, dstOffset + idx, tmp);
      }
    }
  }
  // handle rest of data
  uint32_t processed = nInt4 * sizeof(int4);
  const uint32_t startIdx =
      (op.inputOffsets[0] + getOffset<ReuseScratch>(op.inputBufferRefs[0].type, offset) + processed) / sizeof(T);
  const uint32_t endIdx =
      (op.inputOffsets[0] + getOffset<ReuseScratch>(op.inputBufferRefs[0].type, offset) + size) / sizeof(T);
  for (uint32_t idx = threadIdx.x + startIdx; idx < endIdx; idx += blockDim.x) {
    T tmp = static_cast<T*>(input)[idx];
    for (int index = 0; index < nRemoteInputs; ++index) {
      uint32_t srcOffset =
          (srcOffsets[index] + getOffset<ReuseScratch>(remoteBufferTypes_[op.inputBufferRefs[index + 1].id], offset)) /
          sizeof(T);
      void* remoteMemory = static_cast<char*>(remoteMemoriesViaMemoryChan_[op.inputBufferRefs[index + 1].id]);
      tmp = add_elements(tmp, mscclpp::read<T>(remoteMemory, srcOffset + idx));
    }
    static_cast<T*>(output)[idx] = tmp;
    if constexpr (SendToRemote) {
      for (int index = 0; index < nRemoteOutputs; ++index) {
        uint32_t dstOffset = (dstOffsets[index] +
                              getOffset<ReuseScratch>(remoteBufferTypes_[op.outputBufferRefs[index + 1].id], offset)) /
                             sizeof(T);
        void* remoteMemory = static_cast<char*>(remoteMemoriesViaMemoryChan_[op.outputBufferRefs[index + 1].id]);
        mscclpp::write<T>(remoteMemory, dstOffset + idx, tmp);
      }
    }
  }
}

template <typename PacketType>
MSCCLPP_DEVICE_INLINE void handlePutPacket(const Operation& op, void* input, void* output, void* scratch) {
  ChannelType chType = op.channelType;
  uint16_t nDstChannels = op.nOutputs;
  const uint32_t* dstOffsets = op.outputOffsets;
  const uint32_t* srcOffsets = op.inputOffsets;
  const uint32_t* sizes = op.inputBufferSizes;
  const uint8_t* channelIndexes = op.channelIndexes;
  void* inputBuff = getBuffer(input, output, scratch, op.inputBufferRefs[0].type);
  if (chType == ChannelType::MEMORY) {
    for (int index = 0; index < nDstChannels; ++index) {
      uint32_t size = sizes[index];
      mscclpp::copyToPackets<PacketType>(
          (char*)remoteMemoriesViaMemoryChan_[op.outputBufferRefs[index].id] + dstOffsets[index] * 2,
          (char*)inputBuff + srcOffsets[index], size, threadIdx.x, blockDim.x, flag_);
    }
  }
  if (chType == ChannelType::PORT) {
    int tid = threadIdx.x;
    if (tid >= nDstChannels) {
      return;
    }
    // For port channel, we assume src and dst are in packet format
    // TODO: support non-packet format and remove packet format(packet format should be handle in handleReadPutPacket)
    uint32_t size = sizes[tid];
    uint32_t dstOffset = (dstOffsets[tid] << 1);
    uint32_t srcOffset = (srcOffsets[tid] << 1);
    MemoryId srcMemoryId = static_cast<MemoryId>(op.inputBufferRefs[tid].type);
    MemoryId dstMemoryId = remoteMemoriesViaPortChan_[op.outputBufferRefs[tid].id];
    portChannels_[channelIndexes[tid]].put(dstMemoryId, dstOffset, srcMemoryId, srcOffset, size << 1);
  }
}

template <typename PacketType>
MSCCLPP_DEVICE_INLINE void handleReadPutPacket(const Operation& op, void* scratch) {
  uint32_t nOutput = op.nOutputs;
  const uint32_t* dstOffsets = op.outputOffsets;
  const uint32_t* srcOffsets = op.inputOffsets;
  const uint8_t* channelIndexes = op.channelIndexes;
  uint32_t size = op.inputBufferSizes[0];
  ChannelType chType = op.channelType;
  if (chType == ChannelType::MEMORY) {
    size_t nPackets = size / sizeof(PacketPayload<PacketType>);
    for (size_t pktIdx = threadIdx.x; pktIdx < nPackets; pktIdx += blockDim.x) {
      for (int idx = 0; idx < nOutput; ++idx) {
        PacketType* pkts = (PacketType*)((char*)scratch + srcOffsets[idx] * 2);
        PacketPayload<PacketType> data = pkts[pktIdx].read(flag_);
        PacketType pkt(data, flag_);
        size_t offset = (dstOffsets[idx] * 2) / sizeof(PacketType);
        void* remoteMemory = static_cast<char*>(remoteMemoriesViaMemoryChan_[op.outputBufferRefs[idx].id]);
        mscclpp::write<PacketType>(remoteMemory, offset + pktIdx, pkt);
      }
    }
  } else if (chType == ChannelType::PORT) {
    // Ensuring Data Is Ready
    size_t nPackets = size / sizeof(PacketPayload<PacketType>);
    for (size_t pktIdx = threadIdx.x; pktIdx < nPackets; pktIdx += blockDim.x) {
      for (int idx = 0; idx < nOutput; ++idx) {
        PacketType* pkts = (PacketType*)((char*)scratch + srcOffsets[idx] * 2);
        PacketPayload<PacketType> data = pkts[pktIdx].read(flag_);
      }
    }
    __syncthreads();

    // Putting the data
    int chIdx = threadIdx.x;
    if (chIdx >= nOutput) {
      return;
    }
    uint32_t dstOffset = dstOffsets[chIdx] << 1;
    uint32_t srcOffset = srcOffsets[chIdx] << 1;
    MemoryId dstMemoryId = remoteMemoriesViaPortChan_[op.outputBufferRefs[chIdx].id];
    portChannels_[channelIndexes[chIdx]].put(dstMemoryId, dstOffset, srcOffset, size << 1);
  }
}

template <typename T, typename PacketType, bool SendToRemote = true>
MSCCLPP_DEVICE_INLINE void handleReduceSendPacket(const Operation& op, void* input, void* output, void* scratch) {
  uint32_t size = op.inputBufferSizes[0];
  const uint32_t nSrcs = op.nInputs - 1;
  const uint32_t nDstChannels = op.nOutputs - 1;
  const uint32_t srcOffsetByBytes = op.inputOffsets[0];
  const uint32_t dstOffsetByBytes = op.outputOffsets[0];
  const uint32_t* inputOffsets = op.inputOffsets + 1;
  const uint32_t* outputOffsets = op.outputOffsets + 1;
  const BufferRef* outputBufferRefs = op.outputBufferRefs + 1;

  uint32_t nPackets = size / sizeof(PacketPayload<PacketType>);
  const uint32_t intputBaseOffset = 0;
  const uint32_t srcOffset = srcOffsetByBytes / sizeof(PacketPayload<PacketType>);
  const uint32_t dstOffset = dstOffsetByBytes / sizeof(PacketPayload<PacketType>);
  PacketPayload<PacketType>* srcPacketPayload =
      (PacketPayload<PacketType>*)getBuffer(input, output, scratch, op.inputBufferRefs[0].type) + srcOffset;
  PacketPayload<PacketType>* dstPacketPayload =
      (PacketPayload<PacketType>*)getBuffer(input, output, scratch, op.outputBufferRefs[0].type) + dstOffset;
  for (uint32_t idx = threadIdx.x; idx < nPackets; idx += blockDim.x) {
    PacketPayload<PacketType> data = {};
    for (int index = 0; index < nSrcs; ++index) {
      PacketType* pkt = (PacketType*)((char*)scratch + intputBaseOffset + 2 * inputOffsets[index]);
      PacketPayload<PacketType> val = pkt[idx].read(flag_);
      data = add_vectors<T>(data, val);
    }
    data = add_vectors<T>(data, srcPacketPayload[idx]);
    dstPacketPayload[idx] = data;

    if constexpr (SendToRemote) {
      PacketType pkt(data, flag_);
      for (uint32_t index = 0; index < nDstChannels; ++index) {
        uint32_t offset = (intputBaseOffset + outputOffsets[index] * 2) / sizeof(PacketType);
        void* remoteMemory = static_cast<char*>(remoteMemoriesViaMemoryChan_[outputBufferRefs[index].id]);
        mscclpp::write<PacketType>(remoteMemory, offset + idx, pkt);
      }
    }
  }
}

template <typename PacketType>
MSCCLPP_DEVICE_INLINE void handleCopyPacket(const Operation& op, void* input, void* output, void* scratch) {
  const uint32_t size = op.inputBufferSizes[0];
  const uint32_t dstOffset = op.outputOffsets[0];
  const uint32_t srcOffset = op.inputOffsets[0];
  PacketType* srcPackets = (PacketType*)(static_cast<char*>(scratch) + 2 * srcOffset);
  PacketPayload<PacketType>* result =
      (PacketPayload<PacketType>*)(static_cast<char*>(getBuffer(input, output, scratch, op.outputBufferRefs[0].type)) +
                                   dstOffset);
  uint32_t nPackets = size / sizeof(PacketPayload<PacketType>);
  for (uint32_t idx = threadIdx.x; idx < nPackets; idx += blockDim.x) {
    PacketPayload<PacketType> data = srcPackets[idx].read(flag_);
    result[idx] = data;
  }
}

template <typename PacketType>
MSCCLPP_DEVICE_INLINE void handleTransformToPacket(const Operation& op, void* input, void* output, void* scratch) {
  uint32_t size = op.inputBufferSizes[0];
  uint32_t dstOffset = op.outputOffsets[0];
  uint32_t srcOffset = op.inputOffsets[0];
  dstOffset = dstOffset * 2;
  char* dst = static_cast<char*>(getBuffer(input, output, scratch, op.outputBufferRefs[0].type)) + dstOffset;
  char* src = static_cast<char*>(getBuffer(input, output, scratch, op.inputBufferRefs[0].type)) + srcOffset;
  mscclpp::copyToPackets<PacketType>(dst, src, size, threadIdx.x, blockDim.x, flag_);
}

template <typename T, bool ReuseScratch, bool SendToRemote = true>
MSCCLPP_DEVICE_INLINE void handleReduceSend(const Operation& op, void* input, void* output, void* scratch,
                                            uint32_t offset, uint32_t unitSize) {
  const uint32_t size = min(op.inputBufferSizes[0] - offset, unitSize);
  const uint32_t nInt4 = size / sizeof(int4);
  int nInput = op.nInputs - 1;
  int nOutput = op.nOutputs - 1;
  const uint32_t* inputOffsets = op.inputOffsets + 1;
  const uint32_t* outputOffsets = op.outputOffsets + 1;
  const BufferRef* inputBufferRefs = op.inputBufferRefs + 1;
  const BufferRef* outputBufferRefs = op.outputBufferRefs + 1;
  uint32_t srcOffsetByBytes = op.inputOffsets[0] + getOffset<ReuseScratch>(op.inputBufferRefs[0].type, offset);
  uint32_t dstOffsetByBytes = op.outputOffsets[0] + getOffset<ReuseScratch>(op.outputBufferRefs[0].type, offset);

  const uint32_t srcOffset4 = srcOffsetByBytes / sizeof(int4);
  const uint32_t dstOffset4 = dstOffsetByBytes / sizeof(int4);
  int4* src4 = (int4*)getBuffer(input, output, scratch, op.inputBufferRefs[0].type);
  int4* dst4 = (int4*)getBuffer(input, output, scratch, op.outputBufferRefs[0].type);
  for (size_t idx = threadIdx.x; idx < nInt4; idx += blockDim.x) {
    int4 tmp = src4[srcOffset4 + idx];
    for (int index = 0; index < nInput; ++index) {
      int4* buff4 = static_cast<int4*>(getBuffer(input, output, scratch, inputBufferRefs[index].type));
      size_t buffOffset =
          (inputOffsets[index] + getOffset<ReuseScratch>(outputBufferRefs[index].type, offset)) / sizeof(int4);
      int4 val = buff4[buffOffset + idx];
      tmp = add_vectors<T>(tmp, val);
    }
    dst4[dstOffset4 + idx] = tmp;
    if constexpr (SendToRemote) {
      for (int index = 0; index < nOutput; ++index) {
        size_t outOffset =
            (outputOffsets[index] + getOffset<ReuseScratch>(remoteBufferTypes_[outputBufferRefs[index].id], offset)) /
            sizeof(int4);
        void* remoteMemory = remoteMemoriesViaMemoryChan_[outputBufferRefs[index].id];
        mscclpp::write(remoteMemory, outOffset + idx, tmp);
      }
    }
  }
  // handle rest of data
  uint32_t processed = nInt4 * sizeof(int4);
  const uint32_t startIdx = (srcOffsetByBytes + processed) / sizeof(T);
  const uint32_t endIdx = (srcOffsetByBytes + size) / sizeof(T);
  T* src = (T*)src4;
  T* dst = (T*)dst4;
  for (uint32_t idx = threadIdx.x + startIdx; idx < endIdx; idx += blockDim.x) {
    T tmp = src[idx];
    for (int index = 0; index < nInput; ++index) {
      T* buff = static_cast<T*>(getBuffer(input, output, scratch, inputBufferRefs[index].type));
      uint32_t buffOffset =
          (inputOffsets[index] + getOffset<ReuseScratch>(inputBufferRefs[index].type, offset)) / sizeof(T);
      tmp = add_elements(tmp, buff[buffOffset + idx]);
    }
    dst[idx] = tmp;
    if constexpr (SendToRemote) {
      for (int index = 0; index < nOutput; ++index) {
        uint32_t outOffset =
            (outputOffsets[index] + getOffset<ReuseScratch>(remoteBufferTypes_[outputBufferRefs[index].id], offset)) /
            sizeof(T);
        void* remoteMemory = remoteMemoriesViaMemoryChan_[outputBufferRefs[index].id];
        mscclpp::write<T>(remoteMemory, outOffset + idx, tmp);
      }
    }
  }
}

template <bool ReuseScratch>
MSCCLPP_DEVICE_INLINE void handleCopy(const Operation& op, void* input, void* output, void* scratch, uint32_t offset,
                                      uint32_t unitSize) {
  uint32_t size = min(op.inputBufferSizes[0] - offset, unitSize);
  if (size <= 0) {
    return;
  }
  uint32_t dstOffset = op.outputOffsets[0] + getOffset<ReuseScratch>(op.outputBufferRefs[0].type, offset);
  uint32_t srcOffset = op.inputOffsets[0] + getOffset<ReuseScratch>(op.inputBufferRefs[0].type, offset);
  char* srcData = static_cast<char*>(getBuffer(input, output, scratch, op.inputBufferRefs[0].type)) + srcOffset;
  char* dstData = static_cast<char*>(getBuffer(input, output, scratch, op.outputBufferRefs[0].type)) + dstOffset;
  mscclpp::copy(dstData, srcData, size, threadIdx.x, blockDim.x);
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
template <typename T, bool ReuseScratch>
MSCCLPP_DEVICE_INLINE void handleMultiLoadReduceStore(const Operation& op, uint32_t offset, uint32_t unitSize) {
  using vectorType = typename VectorType<T>::type;
  using nvlsType = typename VectorType<T>::nvls_type;
  // nvls can only handle 4 bytes alignment
  const uint32_t size = min(op.inputBufferSizes[0] - offset, unitSize);
  if (size <= 0) {
    return;
  }

  const uint32_t nInt4 = size / sizeof(nvlsType);
  const uint32_t srcOffset = op.inputOffsets[0] + getOffset<ReuseScratch>(op.nvlsInputBufferType, offset);
  const uint32_t dstOffset = op.outputOffsets[0] + getOffset<ReuseScratch>(op.nvlsOutputBufferType, offset);
  assert(srcOffset % sizeof(vectorType) == 0 && dstOffset % sizeof(vectorType) == 0);

  const uint32_t srcOffset4 = srcOffset / sizeof(nvlsType);
  const uint32_t dstOffset4 = dstOffset / sizeof(nvlsType);
  nvlsType* src4 = (nvlsType*)nvlsChannels_[op.nvlsInputIndex].mcPtr;
  nvlsType* dst4 = (nvlsType*)nvlsChannels_[op.nvlsOutputIndex].mcPtr;
  for (uint32_t idx = threadIdx.x; idx < nInt4; idx += blockDim.x) {
    nvlsType val;
    DeviceMulticastPointerDeviceHandle::multimemLoadReduce(val, (vectorType*)(src4 + srcOffset4 + idx));
    DeviceMulticastPointerDeviceHandle::multimemStore(val, (vectorType*)(dst4 + dstOffset4 + idx));
  }
  // handle rest of data
  uint32_t processed = nInt4 * sizeof(nvlsType);
  using nvlsType2 = typename VectorType<T>::nvls_type2;
  const uint32_t startIdx = (srcOffset + processed) / sizeof(nvlsType2);
  const uint32_t endIdx = (dstOffset + size) / sizeof(nvlsType2);
  for (uint32_t idx = threadIdx.x + startIdx; idx < endIdx; idx += blockDim.x) {
    nvlsType2 val;
    DeviceMulticastPointerDeviceHandle::multimemLoadReduce(val,
                                                           (vectorType*)nvlsChannels_[op.nvlsInputIndex].mcPtr + idx);
    DeviceMulticastPointerDeviceHandle::multimemStore(val, (vectorType*)nvlsChannels_[op.nvlsOutputIndex].mcPtr + idx);
  }
}
#endif

template <typename T, typename PacketType, bool ReuseScratch>
MSCCLPP_DEVICE_INLINE void handlePipeline(const Operation& op, T* input, T* output, T* scratch) {
  uint16_t nIterations = op.nIterations;
  uint16_t nOperations = op.nOperations;
  uint32_t unitSize = op.unitSize;
  const Operation* operations = &op + 1;
  for (uint16_t i = 0; i < nIterations; i++) {
    uint32_t offset = i * unitSize;
    for (uint8_t opId = 0; opId < nOperations; opId++) {
      executeDeviceFunction<T, PacketType, ReuseScratch>(operations[opId], input, output, scratch, nullptr, offset,
                                                         unitSize);
    }
  }
}

MSCCLPP_DEVICE_INLINE void handleSemRelease(const Operation& op) {
  int tid = threadIdx.x;
  if (tid < op.nDeviceSemaphores) {
    DeviceSemaphore* sem = &deviceSemaphores[op.deviceSemaphoreIds[tid]];
    sem->release();
  }
}

MSCCLPP_DEVICE_INLINE void handleSemAquire(const Operation& op) {
  int tid = threadIdx.x;
  if (tid < op.nDeviceSemaphores) {
    DeviceSemaphore* sem = &deviceSemaphores[op.deviceSemaphoreIds[tid]];
    sem->acquire();
  }
}

template <typename T, typename PacketType, bool ReuseScratch>
MSCCLPP_DEVICE_INLINE void executeDeviceFunction(const Operation& op, T* input, T* output, T* scratch, uint8_t* nSteps,
                                                 uint32_t offset, uint32_t unitSize) {
  if (nSteps != nullptr) {
    *nSteps = 1;
  }
  OperationType opType = op.type;
  if (opType == OperationType::NOP) {
    return handleNop();
  }
  if (opType == OperationType::BARRIER) {
    return handleBarrier(op);
  }
  if (opType == OperationType::SIGNAL) {
    return handleSignal(op);
  }
  if (opType == OperationType::WAIT) {
    return handleWait(op);
  }
  if (opType == OperationType::RELAXED_SIGNAL) {
    return handleSignal<true>(op);
  }
  if (opType == OperationType::RELAXED_WAIT) {
    return handleWait<true>(op);
  }
  if (opType == OperationType::FLUSH) {
    return handleFlush(op);
  }
  if (opType == OperationType::PUT) {
    return handlePut<ReuseScratch>(op, input, output, scratch, offset, unitSize);
  }
  if (opType == OperationType::PUT_WITH_SIGNAL) {
    return handlePut<ReuseScratch, true>(op, input, output, scratch, offset, unitSize);
  }
  if (opType == OperationType::PUT_WITH_SIGNAL_AND_FLUSH) {
    return handlePut<ReuseScratch, true, true>(op, input, output, scratch, offset, unitSize);
  }
  if (opType == OperationType::PUT_PACKET) {
    return handlePutPacket<PacketType>(op, input, output, scratch);
  }
  if (opType == OperationType::READ_PUT_PACKET) {
    return handleReadPutPacket<PacketType>(op, scratch);
  }
  if (opType == OperationType::GET) {
    return handleGet<ReuseScratch>(op, input, output, scratch, offset, unitSize);
  }
  if (opType == OperationType::READ_REDUCE_SEND) {
    return handleReadReduceSend<T, ReuseScratch, true>(op, input, output, scratch, offset, unitSize);
  }
  if (opType == OperationType::READ_REDUCE) {
    return handleReadReduceSend<T, ReuseScratch, false>(op, input, output, scratch, offset, unitSize);
  }
  if (opType == OperationType::COPY) {
    return handleCopy<ReuseScratch>(op, input, output, scratch, offset, unitSize);
  }
  if (opType == OperationType::REDUCE_SEND) {
    return handleReduceSend<T, ReuseScratch>(op, input, output, scratch, offset, unitSize);
  }
  if (opType == OperationType::REDUCE_SEND_PACKET) {
    return handleReduceSendPacket<T, PacketType>(op, input, output, scratch);
  }
  if (opType == OperationType::REDUCE_PACKET) {
    return handleReduceSendPacket<T, PacketType, false>(op, input, output, scratch);
  }
  if (opType == OperationType::COPY_PACKET) {
    return handleCopyPacket<PacketType>(op, input, output, scratch);
  }
  if (opType == OperationType::TRANSFORM_TO_PACKET) {
    return handleTransformToPacket<PacketType>(op, input, output, scratch);
  }
  if (opType == OperationType::SEM_ACQUIRE) {
    return handleSemAquire(op);
  }
  if (opType == OperationType::SEM_RELEASE) {
    return handleSemRelease(op);
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (opType == OperationType::MULTI_LOAD_REDUCE_STORE) {
    return handleMultiLoadReduceStore<T, ReuseScratch>(op, offset, unitSize);
  }
#endif
  if (opType == OperationType::PIPELINE) {
    *nSteps = op.nOperations + 1;
    return handlePipeline<T, PacketType, ReuseScratch>(op, input, output, scratch);
  }
  return;
}

template <typename T, typename PacketType = LL16Packet, bool ReuseScratch = false>
__global__ __launch_bounds__(1024, 1) void executionKernel([[maybe_unused]] int rank /*for debug*/, T* input, T* output,
                                                           T* scratch, uint32_t scratchChunkSize,
                                                           DeviceExecutionPlan* plan, DeviceSemaphore* semaphores,
                                                           uint32_t flag
#if defined(ENABLE_NPKIT)
                                                           ,
                                                           NpKitEventCollectContext* npKitEventCollectContexts,
                                                           uint64_t* cpuTimestamp) {
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
  deviceSemaphores = semaphores;
  localPlan = (DeviceExecutionPlan*)sharedMem;
  int nOperations = localPlan->nOperations;
  Operation* operations = (Operation*)localPlan->operations;
  memoryChannels_ = localPlan->channels.memoryChannels;
  portChannels_ = localPlan->channels.portChannels;
  nvlsChannels_ = localPlan->channels.nvlsChannels;
  remoteMemoriesViaMemoryChan_ = localPlan->remoteBuffers.remoteBuffersViaMemoryChannel;
  remoteMemoriesViaPortChan_ = localPlan->remoteBuffers.remoteBuffersViaPortChannel;
  remoteBufferTypes_ = localPlan->remoteBuffers.remoteBufferTypes;
  flag_ = flag;
  scratchChunkSize_ = scratchChunkSize;

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
    Operation& op = operations[i];

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_EXECUTOR_OP_BASE_ENTRY)
    NpKit::CollectGpuEventShm(NPKIT_EVENT_EXECUTOR_OP_BASE_ENTRY + (int)op.type, op.size, 0, NPKIT_GET_GPU_TIMESTAMP(),
                              event_buffer, &event_buffer_head);
#endif
    uint8_t nSteps = 0;
    executeDeviceFunction<T, PacketType, ReuseScratch>(op, input, output, scratch, &nSteps);
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
  template <typename PacketType, bool ReuseScratch>
  static void launchKernel(int rank, int nthreadblocks, int nthreads, void* src, void* dst, void* scratch,
                           uint32_t scratchChunkSize, DataType dataType, DeviceExecutionPlan* plan,
                           DeviceSemaphore* semaphores, uint32_t sharedMemSize, cudaStream_t stream,
                           uint32_t flag = 0) {
    switch (dataType) {
      case DataType::INT32:
        executionKernel<int32_t, PacketType, ReuseScratch><<<nthreadblocks, nthreads, sharedMemSize, stream>>>(
            rank, (int32_t*)src, (int32_t*)dst, (int32_t*)scratch, scratchChunkSize, plan, semaphores, flag
#if defined(ENABLE_NPKIT)
            ,
            NpKit::GetGpuEventCollectContexts(), NpKit::GetCpuTimestamp());
#else
        );
#endif
        break;
      case DataType::UINT32:
        executionKernel<uint32_t, PacketType, ReuseScratch><<<nthreadblocks, nthreads, sharedMemSize, stream>>>(
            rank, (uint32_t*)src, (uint32_t*)dst, (uint32_t*)scratch, scratchChunkSize, plan, semaphores, flag
#if defined(ENABLE_NPKIT)
            ,
            NpKit::GetGpuEventCollectContexts(), NpKit::GetCpuTimestamp());
#else
        );
#endif
        break;
      case DataType::FLOAT16:
        executionKernel<half, PacketType, ReuseScratch><<<nthreadblocks, nthreads, sharedMemSize, stream>>>(
            rank, (half*)src, (half*)dst, (half*)scratch, scratchChunkSize, plan, semaphores, flag
#if defined(ENABLE_NPKIT)
            ,
            NpKit::GetGpuEventCollectContexts(), NpKit::GetCpuTimestamp());
#else
        );
#endif
        break;
      case DataType::FLOAT32:
        executionKernel<float, PacketType, ReuseScratch><<<nthreadblocks, nthreads, sharedMemSize, stream>>>(
            rank, (float*)src, (float*)dst, (float*)scratch, scratchChunkSize, plan, semaphores, flag
#if defined(ENABLE_NPKIT)
            ,
            NpKit::GetGpuEventCollectContexts(), NpKit::GetCpuTimestamp());
#else
        );
#endif
        break;
      case DataType::BFLOAT16:
        executionKernel<__bfloat16, PacketType, ReuseScratch><<<nthreadblocks, nthreads, sharedMemSize, stream>>>(
            rank, (__bfloat16*)src, (__bfloat16*)dst, (__bfloat16*)scratch, scratchChunkSize, plan, semaphores, flag
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
  template <typename PacketType, bool ReuseScratch>
  static void launchKernel(int rank, int nthreadblocks, int nthreads, void* src, void* dst, void* scratch,
                           uint32_t scratchChunkSize, DataType dataType, DeviceExecutionPlan* plan,
                           DeviceSemaphore* semaphores, uint32_t sharedMemSize, cudaStream_t stream, uint32_t flag = 0);
#endif  // !defined(MSCCLPP_DEVICE_HIP)
};
}  // namespace mscclpp

#endif  // MSCCLPP_EXECUTION_KERNEL_HPP_
