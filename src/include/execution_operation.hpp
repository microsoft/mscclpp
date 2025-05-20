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

}  // namespace

namespace mscclpp {

#define MAX_DEVICE_SYNCERS 16
#define MAX_DEVICE_FUNCTIONS_IN_PIPELINE 16
__device__ DeviceSyncer deviceSyncers[MAX_DEVICE_SYNCERS];

__shared__ DeviceHandle<MemoryChannel>* memoryChannels;
__shared__ DeviceHandle<PortChannel>* portChannels;
__shared__ DeviceHandle<NvlsConnection::DeviceMulticastPointer>* nvlsChannels;

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

// MSCCLPP_DEVICE_INLINE void handlePipeline(Operation2* operations, uint16_t numOperations, int maxNumIterations,
//                                           uint32_t unitSize) {
//   for (uint16_t i = 0; i < maxNumIterations; i++) {
//     for (uint16_t opId = 0; opId < numOperations; opId++) {
//       uint32_t size =
//           (operations[opId].size - i * unitSize) > unitSize ? unitSize : max(operations[opId].size - i * unitSize,
//           0);
//       operations[opId].size = size;
//       if (size == 0) {
//         continue;
//       }
//       if (i == 0) {
//         DeviceFunction function = getDeviceFunction(operations[opId].type, nullptr);
//         function(operations + opId);
//         deviceFunctions[opId] = function;
//       } else {
//         if (deviceFunctions[opId] != nullptr) {
//           deviceFunctions[opId](operations + opId);
//         }
//       }
//     }
//   }
// }

// MSCCLPP_DEVICE_INLINE void handleFlush(DeviceHandle<PortChannel>* portChannels, uint8_t* channelIndexes,
//                                        int nChannels) {
//   int tid = threadIdx.x;
//   if (tid < nChannels) {
//     portChannels[channelIndexes[tid]].flush();
//   }
// }

// MSCCLPP_DEVICE_INLINE void handleGet(DeviceHandle<MemoryChannel>* memoryChannel, uint8_t* srcChannelIndexes,
//                                      uint32_t* dstOffsets, uint32_t* srcOffsets, int count, uint32_t size) {
//   for (int i = 0; i < count; i++) {
//     uint32_t dstOffset = dstOffsets[i];
//     uint32_t srcOffset = srcOffsets[i];
//     memoryChannel[srcChannelIndexes[i]].get(srcOffset, dstOffset, size, threadIdx.x, blockDim.x);
//   }
// }

// template <bool PutWithSignal = false, bool PutWithSignalAndFlush = false>
// MSCCLPP_DEVICE_INLINE void handlePut(DeviceHandle<MemoryChannel>* memoryChannel,
//                                      DeviceHandle<PortChannel>* portChannels, uint8_t* dstChannelIndexes,
//                                      uint32_t* dstOffsets, uint32_t* srcOffsets, int count, uint32_t size,
//                                      ChannelType chType) {
//   if (chType == ChannelType::MEMORY) {
//     for (int i = 0; i < count; i++) {
//       uint32_t dstOffset = dstOffsets[i];
//       uint32_t srcOffset = srcOffsets[i];
//       memoryChannel[dstChannelIndexes[i]].put(dstOffset, srcOffset, size, threadIdx.x, blockDim.x);
//     }
//     return;
//   }
//   if (chType == ChannelType::PORT) {
//     int tid = threadIdx.x;
//     if (tid < count) {
//       if constexpr (PutWithSignal) {
//         portChannels[dstChannelIndexes[tid]].putWithSignal(dstOffsets[tid], srcOffsets[tid], size);
//       } else if constexpr (PutWithSignalAndFlush) {
//         portChannels[dstChannelIndexes[tid]].putWithSignalAndFlush(dstOffsets[tid], srcOffsets[tid], size);
//       } else {
//         portChannels[dstChannelIndexes[tid]].put(dstOffsets[tid], srcOffsets[tid], size);
//       }
//     }
//   }
// }

template <typename T, bool SendToRemote = true>
MSCCLPP_DEVICE_INLINE void handleReadReduceCopySend(Operation2* operation, T* input, T* output, T* scratch) {
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
    T tmp = input[idx];
    for (int index = 0; index < nSrcChannels; ++index) {
      size_t srcOffset = srcOffsets[index] / sizeof(T);
      tmp = add_elements(tmp, memoryChannels[srcChannelIndexes[index]].read<T>(srcOffset + idx));
    }
    output[idx] = tmp;
    if constexpr (SendToRemote) {
      for (int index = 0; index < nDstChannels; ++index) {
        size_t dstOffset = dstOffsets[index] / sizeof(T);
        memoryChannels[dstChannelIndexes[index]].write<T>(dstOffset + idx, tmp);
      }
    }
  }
}

// template <typename PacketType>
// MSCCLPP_DEVICE_INLINE void handlePutPacket(size_t scratchSize, DeviceHandle<MemoryChannel>* memoryChannels,
//                                            DeviceHandle<PortChannel>* portChannels, uint8_t* dstChannelIndexes,
//                                            uint32_t* dstOffsets, uint32_t* srcOffsets, int nDstChannels, uint32_t
//                                            size, ChannelType chType, uint32_t flag) {
//   const size_t scratchBaseOffset = flag & 0x1 ? 0 : scratchSize >> 1;
//   if (chType == ChannelType::MEMORY) {
//     for (int index = 0; index < nDstChannels; ++index) {
//       memoryChannels[dstChannelIndexes[index]].putPackets<PacketType>(
//           scratchBaseOffset + dstOffsets[index] * 2, srcOffsets[index], size, threadIdx.x, blockDim.x, flag);
//     }
//   }
//   if (chType == ChannelType::PORT) {
//     int tid = threadIdx.x;
//     if (tid >= nDstChannels) {
//       return;
//     }
//     // For port channel, we assume src and dst are in packet format
//     // TODO: support non-packet format and remove packet format(packet format should be handle in
//     handleReadPutPacket) uint32_t dstOffset = (dstOffsets[tid] << 1) + scratchBaseOffset; uint32_t srcOffset =
//     (srcOffsets[tid] << 1) + scratchBaseOffset; portChannels[dstChannelIndexes[tid]].put(dstOffset, srcOffset, size
//     << 1);
//   }
// }

// template <typename PacketType>
// MSCCLPP_DEVICE_INLINE void handleReadPutPacket(int rank, void* scratch, size_t scratchSize,
//                                                DeviceHandle<SmChannel>* smChannels,
//                                                DeviceHandle<ProxyChannel>* proxyChannels, uint8_t* dstChannelIndexes,
//                                                uint32_t* dstOffsets, uint32_t* srcOffsets, int nDstChannels,
//                                                uint32_t size, ChannelType chType, uint32_t flag) {
//   const size_t scratchBaseOffset = flag & 0x1 ? 0 : scratchSize >> 1;
//   if (chType == ChannelType::MEMORY) {
//     size_t nPackets = size * 2 / sizeof(PacketType);
//     for (size_t pkt_idx = threadIdx.x; pkt_idx < nPackets; pkt_idx += blockDim.x) {
//       for (int ch_idx = 0; ch_idx < nDstChannels; ++ch_idx) {
//         PacketType* pkts = (PacketType*)((char*)scratch + scratchBaseOffset + srcOffsets[ch_idx] * 2);
//         PacketPayload<PacketType> data = pkts[pkt_idx].read(flag);
//         PacketType pkt(data, flag);
//         size_t offset = (scratchBaseOffset + dstOffsets[ch_idx] * 2) / sizeof(PacketType);
//         smChannels[dstChannelIndexes[ch_idx]].write(offset + pkt_idx, pkt);
//       }
//     }
//   } else if (chType == ChannelType::PORT) {
//     // Ensuring Data Is Ready
//     size_t nPackets = size * 2 / sizeof(PacketType);
//     for (size_t pkt_idx = threadIdx.x; pkt_idx < nPackets; pkt_idx += blockDim.x) {
//       for (int ch_idx = 0; ch_idx < nDstChannels; ++ch_idx) {
//         PacketType* pkts = (PacketType*)((char*)scratch + scratchBaseOffset + srcOffsets[ch_idx] * 2);
//         PacketPayload<PacketType> data = pkts[pkt_idx].read(flag);
//       }
//     }
//     __syncthreads();

//     // Putting the data
//     int ch_idx = threadIdx.x;
//     if (ch_idx >= nDstChannels) {
//       return;
//     }
//     uint32_t dstOffset = scratchBaseOffset + dstOffsets[ch_idx] * 2;
//     uint32_t srcOffset = scratchBaseOffset + srcOffsets[ch_idx] * 2;
//     proxyChannels[dstChannelIndexes[ch_idx]].put(dstOffset, srcOffset, size * 2);
//   }
// }

// template <typename T, typename PacketType, bool SendToRemote = true>
// MSCCLPP_DEVICE_INLINE void handleReduceSendPacket(T* dst, uint32_t dstOffsetByBytes, T* src, uint32_t
// srcOffsetByBytes,
//                                                   T* inputBuff, size_t inputBuffSize, uint32_t* inputOffsets, int
//                                                   nSrcs, DeviceHandle<MemoryChannel>* memoryChannels, uint8_t*
//                                                   outputChannelIndexes, uint32_t* outputOffsets, int nDstChannels,
//                                                   size_t size, uint32_t flag) {
//   size_t nPackets = size * 2 / sizeof(PacketType);
//   const size_t intputBaseOffset = flag & 0x1 ? 0 : inputBuffSize >> 1;
//   const uint32_t srcOffset = srcOffsetByBytes / sizeof(PacketPayload<PacketType>);
//   const uint32_t dstOffset = dstOffsetByBytes / sizeof(PacketPayload<PacketType>);
//   PacketPayload<PacketType>* srcPacketPayload = (PacketPayload<PacketType>*)src + srcOffset;
//   PacketPayload<PacketType>* dstPacketPayload = (PacketPayload<PacketType>*)dst + dstOffset;
//   for (size_t idx = threadIdx.x; idx < nPackets; idx += blockDim.x) {
//     PacketPayload<PacketType> data = {};
//     for (int index = 0; index < nSrcs; ++index) {
//       PacketType* pkt = (PacketType*)((char*)inputBuff + intputBaseOffset + 2 * inputOffsets[index]);
//       PacketPayload<PacketType> val = pkt[idx].read(flag);
//       data = add_vectors<T>(data, val);
//     }
//     data = add_vectors<T>(data, srcPacketPayload[idx]);
//     dstPacketPayload[idx] = data;

//     if (SendToRemote) {
//       PacketType pkt(data, flag);
//       for (int index = 0; index < nDstChannels; ++index) {
//         size_t offset = (intputBaseOffset + outputOffsets[index] * 2) / sizeof(PacketType);
//         memoryChannels[outputChannelIndexes[index]].write(offset + idx, pkt);
//       }
//     }
//   }
// }

// template <typename PacketType>
// MSCCLPP_DEVICE_INLINE void handleCopyPacket(void* dst, void* src, size_t srcSize, uint32_t dstOffset,
//                                             uint32_t srcOffset, size_t size, uint32_t flag) {
//   const size_t inputScratchBaseOffset = flag & 0x1 ? 0 : srcSize >> 1;
//   PacketType* srcPackets = (PacketType*)((char*)src + inputScratchBaseOffset + 2 * srcOffset);
//   PacketPayload<PacketType>* result = (PacketPayload<PacketType>*)((char*)dst + dstOffset);
//   size_t nPackets = size * 2 / sizeof(PacketType);
//   for (size_t idx = threadIdx.x; idx < nPackets; idx += blockDim.x) {
//     PacketPayload<PacketType> data = srcPackets[idx].read(flag);
//     result[idx] = data;
//   }
// }

// template <typename PacketType>
// MSCCLPP_DEVICE_INLINE void handleTransformToPacket(void* dst, void* src, size_t dstSize, uint32_t dstOffset,
//                                                    uint32_t srcOffset, size_t size, uint32_t flag) {
//   const size_t outputScratchBaseOffset = flag & 0x1 ? 0 : dstSize >> 1;
//   dstOffset = dstOffset * 2 + outputScratchBaseOffset;
//   mscclpp::copyToPackets<PacketType>((char*)dst + dstOffset, (char*)src + srcOffset, size, threadIdx.x, blockDim.x,
//                                      flag);
// }

// template <typename T, bool SendToRemote = true>
// MSCCLPP_DEVICE_INLINE void handleReduceSend(T* dst, uint32_t dstOffsetByBytes, T* src, uint32_t srcOffsetByBytes,
//                                             T* input, uint32_t* inputOffsets, int nSrcs,
//                                             DeviceHandle<MemoryChannel>* memoryChannels, uint8_t*
//                                             outputChannelIndexes, uint32_t* outputOffsets, int nOutChannels, uint32_t
//                                             size) {
//   const size_t nInt4 = size / sizeof(int4);
//   const size_t srcOffset4 = srcOffsetByBytes / sizeof(int4);
//   const size_t dstOffset4 = dstOffsetByBytes / sizeof(int4);
//   int4* src4 = (int4*)src;
//   int4* dst4 = (int4*)dst;
//   int4* input4 = (int4*)input;
//   for (size_t idx = threadIdx.x; idx < nInt4; idx += blockDim.x) {
//     int4 tmp = src4[srcOffset4 + idx];
//     for (int index = 0; index < nSrcs; ++index) {
//       size_t offset = inputOffsets[index] / sizeof(int4);
//       int4 val = input4[offset + idx];
//       tmp = add_vectors<T>(tmp, val);
//     }
//     dst4[dstOffset4 + idx] = tmp;
//     if (SendToRemote) {
//       for (int index = 0; index < nOutChannels; ++index) {
//         size_t offset = outputOffsets[index] / sizeof(int4);
//         memoryChannels[outputChannelIndexes[index]].write<int4>(offset + idx, tmp);
//       }
//     }
//   }
//   // handle rest of data
//   size_t processed = nInt4 * sizeof(int4);
//   const size_t startIdx = (srcOffsetByBytes + processed) / sizeof(T);
//   const size_t endIdx = (srcOffsetByBytes + size) / sizeof(T);
//   for (size_t idx = threadIdx.x + startIdx; idx < endIdx; idx += blockDim.x) {
//     T tmp = src[idx];
//     for (int index = 0; index < nSrcs; ++index) {
//       size_t offset = inputOffsets[index] / sizeof(T);
//       tmp = add_elements(tmp, input[offset + idx]);
//     }
//     dst[idx] = tmp;
//     if (SendToRemote) {
//       for (int index = 0; index < nOutChannels; ++index) {
//         size_t offset = outputOffsets[index] / sizeof(T);
//         memoryChannels[outputChannelIndexes[index]].write<T>(offset + idx, tmp);
//       }
//     }
//   }
// }

// MSCCLPP_DEVICE_INLINE void handleCopy(void* dst, void* src, uint32_t dstOffset, uint32_t srcOffset, size_t size) {
//   char* srcData = (char*)src + srcOffset;
//   char* dstData = (char*)dst + dstOffset;
//   mscclpp::copy(dstData, srcData, size, threadIdx.x, blockDim.x);
// }

}  // namespace mscclpp
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

#endif  // MSCCLPP_EXECUTION_OPERATION_HPP_