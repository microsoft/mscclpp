// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/nvls_device.hpp>

#define MULTIMEM_ST(val, ptr)                                                                                   \
  asm volatile("multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), \
               "r"(val.w)                                                                                       \
               : "memory");
// specific PTX for fp16 reduction. bf16 would be multimem.ld_reduce.global.add.v4.bf16x2 etc
#define MULTIMEM_LD(val, ptr)                                     \
  asm("multimem.ld_reduce.global.add.v4.f32 {%0,%1,%2,%3}, [%4];" \
      : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)        \
      : "l"(ptr)                                                  \
      : "memory");


extern "C" __global__ void __launch_bounds__(1024, 1)
    nvls_test(mscclpp::DeviceMulticastPointerDeviceHandle nvlsPtrs, int my_rank, int nranks) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  if (tid == 0 && bid == 0) {
    float* devPtr = (float*)nvlsPtrs.devicePtr;
    devPtr[0] = 3;
    devPtr[1] = 4;
    devPtr[2] = 5;
    devPtr[3] = 6;
    __threadfence_system();
  }
  if (tid == 0 && bid == 0 && my_rank == 0) {
    float* devPtr = (float*)nvlsPtrs.devicePtr;

    float* mcPtr = (float*)nvlsPtrs.mcPtr;
    uint4 val;
    MULTIMEM_LD(val, mcPtr);
    MULTIMEM_ST(val, mcPtr);
    __threadfence_system();

    float tmp = *(float*)&val.x;

    printf("RRR %f %f\n", *devPtr, tmp);
  }
}
