// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "execution_plan.hpp"

extern __shared__ mscclpp::DeviceExecutionPlan sharedMem[];

__global__ void commnuication_kernel(void* sendbuff, void* recvbuff, void* scratchbuff, size_t chunkSize) {
  // read data from shared memory
  // 1. get the number of command from shared memory
  int nOps = sharedMem->nOperations;
  mscclpp::DeviceHandle<mscclpp::SmChannel>* smChannel = sharedMem->channels.smChannels;
  mscclpp::DeviceHandle<mscclpp::ProxyChannel>* proxyChannel = sharedMem->channels.proxyChannels;
  for (int opId = 0; opId < nOps; opId++) {
    // 2. get the command
    mscclpp::Operation* op = sharedMem->operations + opId;
    // 3. execute the command
    switch (op->type) {
      default:
        break;
    }
  }
}
