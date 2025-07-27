// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/port_channel_device.hpp>
#include <iostream>

#define MAGIC_CONST 777

__constant__ mscclpp::PortChannelDeviceHandle gPortChannel;

__global__ void kernelLocalPortChannelTest(void *dst, void *src, size_t bytes, int *ret) {
  if (blockIdx.x == 0) {
    // sender
    int *ptr = reinterpret_cast<int *>(src);
    for (size_t idx = threadIdx.x; idx < bytes / sizeof(int); idx += blockDim.x) {
      ptr[idx] = MAGIC_CONST;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      gPortChannel.putWithSignal(0, bytes);
    }
  } else if (blockIdx.x == 1) {
    // receiver
    if (threadIdx.x == 0) {
      gPortChannel.wait();
    }
    __syncthreads();
    int *ptr = reinterpret_cast<int *>(dst);
    for (size_t idx = threadIdx.x; idx < bytes / sizeof(int); idx += blockDim.x) {
      if (ptr[idx] != MAGIC_CONST) {
        *ret = 1;  // Error: value mismatch
        return;
      }
    }
  }
}

int main() {
  const mscclpp::Transport transport = mscclpp::Transport::CudaIpc;
  const size_t bufSize = 4 * 1024 * 1024;

  auto ctx = mscclpp::Context::create();
  mscclpp::Endpoint ep0 = ctx->createEndpoint({transport, {mscclpp::DeviceType::GPU, 0}});
  mscclpp::Endpoint ep1 = ctx->createEndpoint({transport, {mscclpp::DeviceType::GPU, 1}});

  MSCCLPP_CUDATHROW(cudaSetDevice(1));
  void *remoteBuf;
  MSCCLPP_CUDATHROW(cudaMalloc(&remoteBuf, bufSize));
  mscclpp::RegisteredMemory remoteRegMem = ctx->registerMemory(remoteBuf, bufSize, transport);

  std::shared_ptr<mscclpp::Connection> conn = ctx->connect(ep0, ep1);

  auto sema0 = mscclpp::Semaphore(conn0, conn1);
  auto sema1 = mscclpp::Semaphore(conn1, conn0);

  auto proxyService = std::make_shared<mscclpp::ProxyService>();
  auto rmemId0 = proxyService->addMemory(rmem0);
  auto rmemId1 = proxyService->addMemory(rmem1);
  auto semaId0 = proxyService->addSemaphore(sema0);
  auto semaId1 = proxyService->addSemaphore(sema1);

  auto portChannel0 = proxyService->portChannel(semaId0, rmemId1, rmemId0);
  auto portChannel1 = proxyService->portChannel(semaId1, rmemId0, rmemId1);
  auto portChannelHandle = portChannel.deviceHandle();

  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gPortChannel, &portChannelHandle, sizeof(portChannelHandle)));

  std::shared_ptr<int> ret = mscclpp::detail::gpuCallocHostShared<int>();

  proxyService->startProxy();
  kernelLocalPortChannelTest<<<2, 1024>>>(dstBuff.get(), srcBuff.get(), bytes, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  proxyService->stopProxy();

  std::cout << *ret << std::endl;

  return (*ret == 0) ? 0 : 1;
}
