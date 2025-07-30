// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <iostream>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/memory_channel.hpp>
#include <mscclpp/memory_channel_device.hpp>

__global__ void gpuKernel0(mscclpp::BaseMemoryChannelDeviceHandle *devHandle, int iter) {
  if (threadIdx.x + blockIdx.x * gridDim.x == 0) {
    for (int i = 0; i < iter; ++i) {
      devHandle->relaxedWait();
      // sleep (roughly) 1ms
      __nanosleep(1e6);
      devHandle->relaxedSignal();
    }
  }
}

__global__ void gpuKernel1(mscclpp::BaseMemoryChannelDeviceHandle *devHandle, int iter) {
  if (threadIdx.x + blockIdx.x * gridDim.x == 0) {
    for (int i = 0; i < iter; ++i) {
      devHandle->relaxedSignal();
      devHandle->relaxedWait();
    }
  }
}

int main() {
  // Optional: check if we have at least two GPUs
  int deviceCount;
  MSCCLPP_CUDATHROW(cudaGetDeviceCount(&deviceCount));
  if (deviceCount < 2) {
    std::cout << "Error: At least two GPUs are required." << std::endl;
    return 1;
  }

  // Optional: check if the two GPUs can peer-to-peer access each other
  int canAccessPeer;
  MSCCLPP_CUDATHROW(cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1));
  if (!canAccessPeer) {
    std::cout
        << "Error: GPU 0 cannot access GPU 1. Make sure that the GPUs are connected peer-to-peer. You can check this "
           "by running `nvidia-smi topo -m` (the connection between GPU 0 and 1 should be either NV# or PIX)."
        << std::endl;
    return 1;
  }

  const int iter = 100;
  const mscclpp::Transport transport = mscclpp::Transport::CudaIpc;

  std::cout << "Creating endpoints ..." << std::endl;

  auto ctx = mscclpp::Context::create();
  mscclpp::Endpoint ep0 = ctx->createEndpoint({transport, {mscclpp::DeviceType::GPU, 0}});
  mscclpp::Endpoint ep1 = ctx->createEndpoint({transport, {mscclpp::DeviceType::GPU, 1}});

  std::cout << "GPU 0: Creating a connection and a semaphore stub ..." << std::endl;

  MSCCLPP_CUDATHROW(cudaSetDevice(0));
  std::shared_ptr<mscclpp::Connection> conn0 = ctx->connect(/*localEndpoint*/ ep0, /*remoteEndpoint*/ ep1);
  mscclpp::SemaphoreStub semaStub0(conn0);

  std::cout << "GPU 1: Creating a connection and a semaphore stub ..." << std::endl;

  MSCCLPP_CUDATHROW(cudaSetDevice(1));
  std::shared_ptr<mscclpp::Connection> conn1 = ctx->connect(/*localEndpoint*/ ep1, /*remoteEndpoint*/ ep0);
  mscclpp::SemaphoreStub semaStub1(conn1);

  std::cout << "GPU 0: Creating a semaphore and a memory channel ..." << std::endl;

  MSCCLPP_CUDATHROW(cudaSetDevice(0));
  mscclpp::Semaphore sema0(/*localSemaphoreStub*/ semaStub0, /*remoteSemaphoreStub*/ semaStub1);
  mscclpp::BaseMemoryChannel memChan0(sema0);
  mscclpp::BaseMemoryChannelDeviceHandle memChanHandle0 = memChan0.deviceHandle();
  void *devHandle0;
  MSCCLPP_CUDATHROW(cudaMalloc(&devHandle0, sizeof(mscclpp::BaseMemoryChannelDeviceHandle)));
  MSCCLPP_CUDATHROW(cudaMemcpy(devHandle0, &memChanHandle0, sizeof(memChanHandle0), cudaMemcpyHostToDevice));

  std::cout << "GPU 1: Creating a semaphore and a memory channel ..." << std::endl;

  MSCCLPP_CUDATHROW(cudaSetDevice(1));
  mscclpp::Semaphore sema1(/*localSemaphoreStub*/ semaStub1, /*remoteSemaphoreStub*/ semaStub0);
  mscclpp::BaseMemoryChannel memChan1(sema1);
  mscclpp::BaseMemoryChannelDeviceHandle memChanHandle1 = memChan1.deviceHandle();
  void *devHandle1;
  MSCCLPP_CUDATHROW(cudaMalloc(&devHandle1, sizeof(mscclpp::BaseMemoryChannelDeviceHandle)));
  MSCCLPP_CUDATHROW(cudaMemcpy(devHandle1, &memChanHandle1, sizeof(memChanHandle1), cudaMemcpyHostToDevice));

  std::cout << "GPU 0: Launching gpuKernel0 ..." << std::endl;

  MSCCLPP_CUDATHROW(cudaSetDevice(0));
  gpuKernel0<<<1, 1>>>(reinterpret_cast<mscclpp::BaseMemoryChannelDeviceHandle *>(devHandle0), iter);
  MSCCLPP_CUDATHROW(cudaGetLastError());

  std::cout << "GPU 1: Launching gpuKernel1 ..." << std::endl;

  MSCCLPP_CUDATHROW(cudaSetDevice(1));
  cudaEvent_t start, end;
  MSCCLPP_CUDATHROW(cudaEventCreate(&start));
  MSCCLPP_CUDATHROW(cudaEventCreate(&end));
  MSCCLPP_CUDATHROW(cudaEventRecord(start));
  gpuKernel1<<<1, 1>>>(reinterpret_cast<mscclpp::BaseMemoryChannelDeviceHandle *>(devHandle1), iter);
  MSCCLPP_CUDATHROW(cudaGetLastError());
  MSCCLPP_CUDATHROW(cudaEventRecord(end));
  MSCCLPP_CUDATHROW(cudaEventSynchronize(end));

  float elapsedMs;
  MSCCLPP_CUDATHROW(cudaEventElapsedTime(&elapsedMs, start, end));

  MSCCLPP_CUDATHROW(cudaSetDevice(0));
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  float msPerIter = elapsedMs / iter;
  std::cout << "Elapsed " << msPerIter << " ms per iteration (" << iter << ")" << std::endl;
  if (msPerIter < 1.0f) {
    std::cout << "Failed: the elapsed time per iteration is less than 1 ms, which may indicate that the relaxedSignal "
                 "and relaxedWait are not working as expected."
              << std::endl;
    return 1;
  }
  std::cout << "Succeed!" << std::endl;
  return 0;
}
