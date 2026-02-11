// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "../framework.hpp"

#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/port_channel_device.hpp>

#define MAGIC_CONST 777

__constant__ mscclpp::PortChannelDeviceHandle gPortChannel;

__global__ void kernelLocalPortChannelTest(void* dst, void* src, size_t bytes, int* ret) {
  if (blockIdx.x == 0) {
    // sender
    int* ptr = reinterpret_cast<int*>(src);
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
    int* ptr = reinterpret_cast<int*>(dst);
    for (size_t idx = threadIdx.x; idx < bytes / sizeof(int); idx += blockDim.x) {
      if (ptr[idx] != MAGIC_CONST) {
        *ret = 1;  // Error: value mismatch
        return;
      }
    }
  }
}

static void localPortChannelTest(mscclpp::Transport transport) {
  MSCCLPP_CUDATHROW(cudaSetDevice(0));

  auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(/*rank*/ 0, /*nRanks*/ 1);
  bootstrap->initialize(mscclpp::TcpBootstrap::createUniqueId());
  auto communicator = std::make_shared<mscclpp::Communicator>(bootstrap);

  auto connection = communicator->connect(transport, /*remoteRank*/ 0).get();

  const size_t bytes = 4 * 1024 * 1024;
  auto srcBuff = mscclpp::GpuBuffer(bytes).memory();
  auto dstBuff = mscclpp::GpuBuffer(bytes).memory();

  auto srcMem = communicator->registerMemory(srcBuff.get(), bytes, transport);
  auto dstMem = communicator->registerMemory(dstBuff.get(), bytes, transport);

  auto proxyService = std::make_shared<mscclpp::ProxyService>();
  auto srcMemId = proxyService->addMemory(srcMem);
  auto dstMemId = proxyService->addMemory(dstMem);

  auto sid = proxyService->buildAndAddSemaphore(*communicator, connection);
  auto portChannel = proxyService->portChannel(sid, dstMemId, srcMemId);
  auto portChannelHandle = portChannel.deviceHandle();

  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gPortChannel, &portChannelHandle, sizeof(portChannelHandle)));

  std::shared_ptr<int> ret = mscclpp::detail::gpuCallocHostShared<int>();

  proxyService->startProxy();
  kernelLocalPortChannelTest<<<2, 1024>>>(dstBuff.get(), srcBuff.get(), bytes, ret.get());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  proxyService->stopProxy();

  EXPECT_EQ(*ret, 0);
}

TEST(LocalChannelTest, PortChannel) { localPortChannelTest(mscclpp::Transport::CudaIpc); }
