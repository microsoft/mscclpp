// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "execution_kernel.hpp"

#if defined(MSCCLPP_DEVICE_CUDA)
namespace mscclpp {

template <typename PacketType, bool ReuseScratch>
void ExecutionKernel::launchKernel(int rank, int nthreadblocks, int nthreads, void* src, void* dst, void* scratch,
                                   uint32_t scratchOffset, uint32_t scratchChunkSize, DataType dataType,
                                   DeviceExecutionPlan* plan, DeviceSemaphore* semaphores, uint32_t localMemoryIdBegin,
                                   uint32_t sharedMemSize, cudaStream_t stream, uint32_t flag) {
  switch (dataType) {
    case DataType::INT32:
      executionKernel<int32_t, PacketType, ReuseScratch><<<nthreadblocks, nthreads, sharedMemSize, stream>>>(
          rank, (int32_t*)src, (int32_t*)dst, (int32_t*)scratch, scratchOffset, scratchChunkSize, plan, semaphores,
          localMemoryIdBegin, flag
#if defined(ENABLE_NPKIT)
          ,
          NpKit::GetGpuEventCollectContexts(), NpKit::GetCpuTimestamp());
#else
      );
#endif
      break;
    case DataType::UINT32:
      executionKernel<uint32_t, PacketType, ReuseScratch><<<nthreadblocks, nthreads, sharedMemSize, stream>>>(
          rank, (uint32_t*)src, (uint32_t*)dst, (uint32_t*)scratch, scratchOffset, scratchChunkSize, plan, semaphores,
          localMemoryIdBegin, flag
#if defined(ENABLE_NPKIT)
          ,
          NpKit::GetGpuEventCollectContexts(), NpKit::GetCpuTimestamp());
#else
      );
#endif
      break;
    case DataType::UINT8:
      executionKernel<uint8_t, PacketType, ReuseScratch><<<nthreadblocks, nthreads, sharedMemSize, stream>>>(
          rank, (uint8_t*)src, (uint8_t*)dst, (uint8_t*)scratch, scratchOffset, scratchChunkSize, plan, semaphores,
          localMemoryIdBegin, flag
#if defined(ENABLE_NPKIT)
          ,
          NpKit::GetGpuEventCollectContexts(), NpKit::GetCpuTimestamp());
#else
      );
#endif
      break;
    case DataType::FLOAT16:
      executionKernel<half, PacketType, ReuseScratch><<<nthreadblocks, nthreads, sharedMemSize, stream>>>(
          rank, (half*)src, (half*)dst, (half*)scratch, scratchOffset, scratchChunkSize, plan, semaphores,
          localMemoryIdBegin, flag
#if defined(ENABLE_NPKIT)
          ,
          NpKit::GetGpuEventCollectContexts(), NpKit::GetCpuTimestamp());
#else
      );
#endif
      break;
    case DataType::FLOAT32:
      executionKernel<float, PacketType, ReuseScratch><<<nthreadblocks, nthreads, sharedMemSize, stream>>>(
          rank, (float*)src, (float*)dst, (float*)scratch, scratchOffset, scratchChunkSize, plan, semaphores,
          localMemoryIdBegin, flag
#if defined(ENABLE_NPKIT)
          ,
          NpKit::GetGpuEventCollectContexts(), NpKit::GetCpuTimestamp());
#else
      );
#endif
      break;
    case DataType::BFLOAT16:
      executionKernel<__bfloat16, PacketType, ReuseScratch><<<nthreadblocks, nthreads, sharedMemSize, stream>>>(
          rank, (__bfloat16*)src, (__bfloat16*)dst, (__bfloat16*)scratch, scratchOffset, scratchChunkSize, plan,
          semaphores, localMemoryIdBegin, flag
#if defined(ENABLE_NPKIT)
          ,
          NpKit::GetGpuEventCollectContexts(), NpKit::GetCpuTimestamp());
#else
      );
#endif
      break;
    case DataType::FLOAT8_E4M3FN:
    case DataType::FLOAT8_E4M3FNUZ:
    case DataType::FLOAT8_E5M2:
    case DataType::FLOAT8_E5M2FNUZ:
    case DataType::FLOAT8_E4M3B15:
      // The CUDA execution kernel has no FP8 instantiation. Returning here would look like success with the output
      // buffer left untouched, so fail loudly instead.
      THROW(LogSubsys::EXEC, Error, ErrorCode::InvalidUsage,
            "The CUDA execution kernel does not support FP8 data types");
      break;
    case DataType::AUTO:
      THROW(LogSubsys::EXEC, Error, ErrorCode::InvalidUsage, "DataType::AUTO must be resolved before kernel launch");
      break;
  }
}

#define INSTANTIATE_LAUNCH(PKT, REUSE)                                                                        \
  template void ExecutionKernel::launchKernel<PKT, REUSE>(                                                    \
      int rank, int nthreadblocks, int nthreads, void* src, void* dst, void* scratch, uint32_t scratchOffset, \
      uint32_t scratchChunkSize, DataType dataType, DeviceExecutionPlan* plan, DeviceSemaphore* semaphores,   \
      uint32_t localMemoryIdBegin, uint32_t sharedMemSize, cudaStream_t stream, uint32_t flag);

INSTANTIATE_LAUNCH(LL16Packet, true)
INSTANTIATE_LAUNCH(LL8Packet, true)
INSTANTIATE_LAUNCH(LL16Packet, false)
INSTANTIATE_LAUNCH(LL8Packet, false)
#undef INSTANTIATE_LAUNCH

}  // namespace mscclpp
#endif
