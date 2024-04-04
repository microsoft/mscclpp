// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/device.hpp>

#include "execution_kernel.hpp"

#if defined(MSCCLPP_DEVICE_HIP)
#define __synclds() asm volatile("s_waitcnt lgkmcnt(0) \n s_barrier");
#endif  // defined(MSCCLPP_DEVICE_HIP)

namespace mscclpp {

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

__global__ void kernel(int rank, DeviceExecutionPlan* plan) {
  extern __shared__ int sharedMem[];
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  DeviceExecutionPlan* localPlan = plan + bid;
  for (int i = tid; i < sizeof(DeviceExecutionPlan) / sizeof(int); i += blockDim.x) {
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
  if (bid > 0) {
    return;
  }
  for (int i = 0; i < localPlan->nOperations; i++) {
    switch (operations[i].type) {
      case OperationType::BARRIER:
        __syncthreads();
        break;
      case OperationType::SIGNAL:
        // if (tid == 0) {
        //   printf("rank: %d bid: %d, noutputchannels: %d outputChannelIndex %d\n", rank, bid,
        //          operations[i].nOutputChannels, operations[i].outputChannelIndex[0]);
        // }
        handleSignal(tid, smChannels, proxyChannels, operations[i].outputChannelIndex, operations[i].nOutputChannels,
                     operations[i].channelType);
        break;
      case OperationType::WAIT:
        // if (tid == 0) {
        //   printf("rank: %d bid: %d, ninputchannels: %d inputChannelIndex %d\n", rank, bid,
        //   operations[i].nInputChannels,
        //          operations[i].inputChannelIndex[0]);
        // }
        handleWait(tid, smChannels, proxyChannels, operations[i].inputChannelIndex, operations[i].nInputChannels,
                   operations[i].channelType);
        break;
      default:
        break;
    }
  }
}

void ExecutionKernel::launchKernel(int rank, int nthreadblocks, int nthreads, DeviceExecutionPlan* plan,
                                   size_t sharedMemSize, cudaStream_t stream) {
  kernel<<<nthreadblocks, nthreads, sharedMemSize, stream>>>(rank, plan);
}
}  // namespace mscclpp
