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

namespace mscclpp {

#define MAX_DEVICE_SYNCERS 16
#define MAX_DEVICE_FUNCTIONS_IN_PIPELINE 16
__device__ DeviceSyncer deviceSyncers[MAX_DEVICE_SYNCERS];

#if defined(MSCCLPP_DEVICE_COMPILE)

typedef void (*DeviceFunction) (Operation2 * op, void* src, void* dst, void* scratch);
__shared__ DeviceFunction deviceFunctions[MAX_DEVICE_FUNCTIONS_IN_PIPELINE];

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

template <typename T>
MSCCLPP_DEVICE_INLINE DeviceFunction getDeviceFunction(OperationType opType, uint8_t* nSteps) {
  *nSteps = 1;
  return nullptr;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
template <typename T>
MSCCLPP_DEVICE_INLINE void handleMultiLoadReduceStore(T* dst, T* src, uint32_t dstOffset, uint32_t srcOffset,
                                                      size_t size) {
  using vectorType = typename VectorType<T>::type;
  using nvlsType = typename VectorType<T>::nvls_type;
  // nvls can only handle 4 bytes alignment
  assert(size % sizeof(vectorType) == 0);
  const size_t nInt4 = size / sizeof(nvlsType);
  const size_t srcOffset4 = srcOffset / sizeof(nvlsType);
  const size_t dstOffset4 = dstOffset / sizeof(nvlsType);
  nvlsType* src4 = (nvlsType*)src;
  nvlsType* dst4 = (nvlsType*)dst;
  for (size_t idx = threadIdx.x; idx < nInt4; idx += blockDim.x) {
    nvlsType val;
    DeviceMulticastPointerDeviceHandle::multimemLoadReduce(val, (vectorType*)(src4 + srcOffset4 + idx));
    DeviceMulticastPointerDeviceHandle::multimemStore(val, (vectorType*)(dst4 + dstOffset4 + idx));
  }
  // handle rest of data
  size_t processed = nInt4 * sizeof(nvlsType);
  using nvlsType2 = typename VectorType<T>::nvls_type2;
  const size_t startIdx = (srcOffset + processed) / sizeof(nvlsType2);
  const size_t endIdx = (dstOffset + size) / sizeof(nvlsType2);
  for (size_t idx = threadIdx.x + startIdx; idx < endIdx; idx += blockDim.x) {
    nvlsType2 val;
    DeviceMulticastPointerDeviceHandle::multimemLoadReduce(val, (vectorType*)src + idx);
    DeviceMulticastPointerDeviceHandle::multimemStore(val, (vectorType*)dst + idx);
  }
}
#endif

MSCCLPP_DEVICE_INLINE void handlePipeline(Operation2* operations, uint16_t numOperations, int maxNumIterations,
                                          uint32_t unitSize) {
  for (uint16_t i = 0; i < maxNumIterations; i++) {
    for (uint16_t opId = 0; opId < numOperations; opId++) {
      uint32_t size =
          (operations[opId].size - i * unitSize) > unitSize ? unitSize : max(operations[opId].size - i * unitSize, 0);
      operations[opId].size = size;
      if (size == 0) {
        continue;
      }
      if (i == 0) {
        DeviceFunction function = getDeviceFunction(operations[opId].type, nullptr);
        function(operations + opId);
        deviceFunctions[opId] = function;
      } else {
        if (deviceFunctions[opId] != nullptr) {
          deviceFunctions[opId](operations + opId);
        }
      }
    }
  }
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
  Operation* operations = localPlan->operations;
  DeviceHandle<MemoryChannel>* memoryChannels = localPlan->channels.memoryChannels;
  DeviceHandle<PortChannel>* portChannels = localPlan->channels.portChannels;
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
    Operation& op = operations[i];

#if defined(ENABLE_NPKIT) && defined(ENABLE_NPKIT_EVENT_EXECUTOR_OP_BASE_ENTRY)
    NpKit::CollectGpuEventShm(NPKIT_EVENT_EXECUTOR_OP_BASE_ENTRY + (int)op.type, op.size, 0, NPKIT_GET_GPU_TIMESTAMP(),
                              event_buffer, &event_buffer_head);
#endif
    uint8_t nSteps = 0;
    DeviceFunction function = getDeviceFunction(op.type, &nSteps);
    function(operations + i, src, dst);
    i += nSteps;

    // if (op.type == OperationType::NOP) {
    //   __syncthreads();
    // } else if (op.type == OperationType::BARRIER) {
    //   int nThreadBlocks = op.nThreadBlocks;
    //   int syncStateIndex = op.deviceSyncerIndex;
    //   deviceSyncers[syncStateIndex].sync(nThreadBlocks);
    // } else if (op.type == OperationType::SIGNAL) {
    //   handleSignal(memoryChannels, portChannels, op.outputChannelIndexes, op.nOutputs, op.channelType);
    // } else if (op.type == OperationType::WAIT) {
    //   handleWait(memoryChannels, portChannels, op.inputChannelIndexes, op.nInputs, op.channelType);
    // } else if (op.type == OperationType::FLUSH) {
    //   handleFlush(portChannels, op.outputChannelIndexes, op.nOutputs);
    // } else if (op.type == OperationType::PUT) {
    //   handlePut(memoryChannels, portChannels, op.outputChannelIndexes, op.outputOffsets, op.inputOffsets, op.nOutputs,
    //             op.size, op.channelType);
    // } else if (op.type == OperationType::PUT_WITH_SIGNAL) {
    //   handlePut<true>(memoryChannels, portChannels, op.outputChannelIndexes, op.outputOffsets, op.inputOffsets,
    //                   op.nOutputs, op.size, op.channelType);
    // } else if (op.type == OperationType::PUT_WITH_SIGNAL_AND_FLUSH) {
    //   handlePut<false, true>(memoryChannels, portChannels, op.outputChannelIndexes, op.outputOffsets, op.inputOffsets,
    //                          op.nOutputs, op.size, op.channelType);
    // } else if (op.type == OperationType::GET) {
    //   handleGet(memoryChannels, op.inputChannelIndexes, op.outputOffsets, op.inputOffsets, op.nInputs, op.size);
    // } else if (op.type == OperationType::COPY) {
    //   T* dst = getBuffer(input, output, scratch, op.dstBufferType);
    //   T* src = getBuffer(input, output, scratch, op.srcBufferType);
    //   handleCopy(dst, src, op.dstOffset, op.srcOffset, op.size);
    // } else if (op.type == OperationType::READ_REDUCE_COPY_SEND) {
    //   T* dst = getBuffer(input, output, scratch, op.dstBufferType);
    //   T* src = getBuffer(input, output, scratch, op.srcBufferType);
    //   handleReadReduceCopySend(dst, op.dstOffset, src, op.srcOffset, memoryChannels, op.outputChannelIndexes,
    //                            op.inputChannelIndexes, op.outputOffsets, op.inputOffsets, op.nOutputs, op.nInputs,
    //                            op.size);
    // } else if (op.type == OperationType::READ_REDUCE_COPY) {
    //   T* dst = getBuffer(input, output, scratch, op.dstBufferType);
    //   T* src = getBuffer(input, output, scratch, op.srcBufferType);

    //   handleReadReduceCopySend(dst, op.dstOffset, src, op.srcOffset, memoryChannels, op.outputChannelIndexes,
    //                            op.inputChannelIndexes, op.outputOffsets, op.inputOffsets, op.nOutputs, op.nInputs,
    //                            op.size, false);
    // } else if (op.type == OperationType::READ_PUT_PACKET) {
    //   handleReadPutPacket<PacketType>(rank, scratch, scratchSize, memoryChannels, portChannels, op.outputChannelIndexes,
    //                                   op.outputOffsets, op.inputOffsets, op.nOutputs, op.size, op.channelType, flag);
    // } else if (op.type == OperationType::PUT_PACKET) {
    //   handlePutPacket<PacketType>(scratchSize, memoryChannels, portChannels, op.outputChannelIndexes, op.outputOffsets,
    //                               op.inputOffsets, op.nOutputs, op.size, op.channelType, flag);
    // } else if (op.type == OperationType::REDUCE_SEND_PACKET) {
    //   T* dst = getBuffer(input, output, scratch, op.dstBufferType);
    //   T* src = getBuffer(input, output, scratch, op.srcBufferType);
    //   handleReduceSendPacket<T, PacketType>(dst, op.dstOffset, src, op.srcOffset, scratch, scratchSize, op.inputOffsets,
    //                                         op.nInputs, memoryChannels, op.outputChannelIndexes, op.outputOffsets,
    //                                         op.nOutputs, op.size, flag);
    // } else if (op.type == OperationType::REDUCE) {
    //   T* dst = getBuffer(input, output, scratch, op.dstBufferType);
    //   T* src = getBuffer(input, output, scratch, op.srcBufferType);
    //   T* tmp = getBuffer(input, output, scratch, op.inputBufferType);
    //   handleReduceSend<T, false>(dst, op.dstOffset, src, op.srcOffset, tmp, op.inputOffsets, op.nInputs, memoryChannels,
    //                              op.outputChannelIndexes, op.outputOffsets, op.nOutputs, op.size);
    // } else if (op.type == OperationType::REDUCE_PACKET) {
    //   T* dst = getBuffer(input, output, scratch, op.dstBufferType);
    //   T* src = getBuffer(input, output, scratch, op.srcBufferType);
    //   handleReduceSendPacket<T, PacketType, false>(dst, op.dstOffset, src, op.srcOffset, scratch, scratchSize,
    //                                                op.inputOffsets, op.nInputs, memoryChannels, op.outputChannelIndexes,
    //                                                op.outputOffsets, op.nOutputs, op.size, flag);
    // } else if (op.type == OperationType::COPY_PACKET) {
    //   T* dst = getBuffer(input, output, scratch, op.dstBufferType);
    //   T* src = getBuffer(input, output, scratch, op.srcBufferType);
    //   handleCopyPacket<PacketType>(dst, src, scratchSize, op.dstOffset, op.srcOffset, op.size, flag);
    // } else if (op.type == OperationType::TRANSFORM_TO_PACKET) {
    //   T* dst = getBuffer(input, output, scratch, op.dstBufferType);
    //   T* src = getBuffer(input, output, scratch, op.srcBufferType);
    //   handleTransformToPacket<PacketType>(dst, src, scratchSize, op.dstOffset, op.srcOffset, op.size, flag);
    // } else if (op.type == OperationType::REDUCE_SEND) {
    //   T* dst = getBuffer(input, output, scratch, op.dstBufferType);
    //   T* src = getBuffer(input, output, scratch, op.srcBufferType);
    //   T* tmp = getBuffer(input, output, scratch, op.inputBufferType);
    //   handleReduceSend(dst, op.dstOffset, src, op.srcOffset, tmp, op.inputOffsets, op.nInputs, memoryChannels,
    //                    op.outputChannelIndexes, op.outputOffsets, op.nOutputs, op.size);
    // }
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
