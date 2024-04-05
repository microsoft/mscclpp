// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef MSCCLPP_EXECUTION_KERNEL_HPP_
#define MSCCLPP_EXECUTION_KERNEL_HPP_

#include <mscclpp/executor.hpp>
#include <mscclpp/proxy_channel.hpp>
#include <mscclpp/sm_channel.hpp>

#include "execution_common.hpp"

#if defined(MSCCLPP_DEVICE_COMPILE)
#if defined(MSCCLPP_DEVICE_HIP)
#define __synclds() asm volatile("s_waitcnt lgkmcnt(0) \n s_barrier");
#endif  // defined(MSCCLPP_DEVICE_HIP)

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

template <typename T>
MSCCLPP_DEVICE_INLINE uint32_t add_vectors_helper(uint32_t a, uint32_t b) {
  return bit_cast<uint32_t, T>(add_elements(bit_cast<T, uint32_t>(a), bit_cast<T, uint32_t>(b)));
}

template <typename T>
MSCCLPP_DEVICE_INLINE uint32_t add_vectors(uint32_t a, uint32_t b) {
  return add_vectors_helper<T>(a, b);
}

template <>
MSCCLPP_DEVICE_INLINE __attribute__((unused)) uint32_t add_vectors<__half>(uint32_t a, uint32_t b) {
  return add_vectors_helper<__half2>(a, b);
}

}  // namespace
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

namespace mscclpp {

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

MSCCLPP_DEVICE_INLINE void handleSignal(int tid, DeviceHandle<SmChannel>* smChannels,
                                        DeviceHandle<SimpleProxyChannel>* proxyChannels, uint8_t* channelIndex,
                                        int nChannels, ChannelType chType) {
  if (tid < nChannels) {
    if (chType == ChannelType::SM) {
      smChannels[channelIndex[tid]].signal();
    }
    if (chType == ChannelType::PROXY) {
      proxyChannels[channelIndex[tid]].signal();
    }
  }
}

MSCCLPP_DEVICE_INLINE void handleWait(int tid, DeviceHandle<SmChannel>* smChannels,
                                      DeviceHandle<SimpleProxyChannel>* proxyChannels, uint8_t* channelIndex,
                                      int nChannels, ChannelType chType) {
  if (tid < nChannels) {
    if (chType == ChannelType::SM) {
      smChannels[channelIndex[tid]].wait();
    }
    if (chType == ChannelType::PROXY) {
      proxyChannels[channelIndex[tid]].wait();
    }
  }
}

template <typename T>
MSCCLPP_DEVICE_INLINE void handleReadReduceCopySend(T* input, uint32_t inputOffsetByBytes, T* output,
                                                    uint32_t outputOffsetByBytes, DeviceHandle<SmChannel>* smChannels,
                                                    uint8_t* srcChannelIndex, uint8_t* dstChannelIndex,
                                                    uint32_t* srcOffsets, uint32_t* dstOffsets, int nSrcChannels,
                                                    int nDstChannels, uint32_t size) {
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
      val = smChannels[srcChannelIndex[index]].read<int4>(srcOffset + idx);
      tmp = add_vectors<T>(tmp, val);
    }
    output4[outputOffset4 + idx] = tmp;
    for (int index = 0; index < nDstChannels; ++index) {
      size_t dstOffset = dstOffsets[index] / sizeof(int4);
      smChannels[dstChannelIndex[index]].write<int4>(dstOffset + idx, tmp);
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
      tmp += smChannels[srcChannelIndex[index]].read<T>(srcOffset + idx);
    }
    output[idx] = tmp;
    for (int index = 0; index < nDstChannels; ++index) {
      size_t dstOffset = dstOffsets[index] / sizeof(T);
      smChannels[dstChannelIndex[index]].write<T>(dstOffset + idx, tmp);
    }
  }
}

template <typename T>
__global__ void executionKernel([[maybe_unused]] int rank /*for debug*/, T* input, T* output, T* scratch,
                                DeviceExecutionPlan* plan) {
  extern __shared__ int sharedMem[];
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  DeviceExecutionPlan* localPlan = plan + bid;
  for (size_t i = tid; i < sizeof(DeviceExecutionPlan) / sizeof(int); i += blockDim.x) {
    sharedMem[i] = ((int*)localPlan)[i];
  }
#if defined(MSCCLPP_DEVICE_HIP)
  __synclds();
#else   // !defined(MSCCLPP_DEVICE_HIP)
  __syncthreads();
#endif  // !defined(MSCCLPP_DEVICE_HIP)
  Operation* operations = localPlan->operations;
  DeviceHandle<SmChannel>* smChannels = localPlan->channels.smChannels;
  DeviceHandle<SimpleProxyChannel>* proxyChannels = localPlan->channels.proxyChannels;
  T* src = nullptr;
  T* dst = nullptr;
  for (int i = 0; i < localPlan->nOperations; i++) {
    switch (operations[i].type) {
      case OperationType::BARRIER:
        __syncthreads();
        break;
      case OperationType::SIGNAL:
        handleSignal(tid, smChannels, proxyChannels, operations[i].outputChannelIndexes, operations[i].nOutputChannels,
                     operations[i].channelType);
        break;
      case OperationType::WAIT:
        handleWait(tid, smChannels, proxyChannels, operations[i].inputChannelIndexes, operations[i].nInputChannels,
                   operations[i].channelType);
        break;
      case OperationType::READ_REDUCE_COPY_SEND:
        src = getBuffer(input, output, scratch, operations[i].srcBufferType);
        dst = getBuffer(input, output, scratch, operations[i].dstBufferType);
        handleReadReduceCopySend(src, operations[i].srcOffset, dst, operations[i].dstOffset, smChannels,
                                 operations[i].inputChannelIndexes, operations[i].outputChannelIndexes,
                                 operations[i].inputOffsets, operations[i].outputOffsets, operations[i].nInputChannels,
                                 operations[i].nOutputChannels, operations[i].size);
        break;
      default:
        break;
    }
  }
}
#endif  // defined(MSCCLPP_DEVICE_COMPILE)

class ExecutionKernel {
 public:
#if defined(MSCCLPP_DEVICE_HIP)
  static void launchKernel(int rank, int nthreadblocks, int nthreads, void* src, void* dst, void* scratch,
                           DataType dataType, DeviceExecutionPlan* plan, size_t sharedMemSize, cudaStream_t stream) {
    switch (dataType) {
      case DataType::INT32:
        executionKernel<int32_t><<<nthreadblocks, nthreads, sharedMemSize, stream>>>(rank, (int32_t*)src, (int32_t*)dst,
                                                                                     (int32_t*)scratch, plan);
        break;
      case DataType::UINT32:
        executionKernel<uint32_t><<<nthreadblocks, nthreads, sharedMemSize, stream>>>(
            rank, (uint32_t*)src, (uint32_t*)dst, (uint32_t*)scratch, plan);
        break;
      case DataType::FLOAT16:
        executionKernel<half>
            <<<nthreadblocks, nthreads, sharedMemSize, stream>>>(rank, (half*)src, (half*)dst, (half*)scratch, plan);
        break;
      case DataType::FLOAT32:
        executionKernel<float>
            <<<nthreadblocks, nthreads, sharedMemSize, stream>>>(rank, (float*)src, (float*)dst, (float*)scratch, plan);
        break;
    }
  }
#else   // !defined(MSCCLPP_DEVICE_HIP)
  static void launchKernel(int rank, int nthreadblocks, int nthreads, void* src, void* dst, void* scratch,
                           DataType dataType, DeviceExecutionPlan* plan, size_t sharedMemSize, cudaStream_t stream);
#endif  // !defined(MSCCLPP_DEVICE_HIP)
};
}  // namespace mscclpp

#endif  // MSCCLPP_EXECUTION_KERNEL_HPP_
