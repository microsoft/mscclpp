// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <chrono>
#include <iostream>
#include <memory>
#include <mscclpp/core.hpp>
#include <mscclpp/gpu_utils.hpp>
#include <mscclpp/port_channel.hpp>
#include <thread>
#include <vector>

__global__ void transfer(mscclpp::PortChannelDeviceHandle channel, size_t bytes) {
  channel.putWithSignalAndFlush(0, 0, bytes, -1);
  channel.wait(-1);
}

int main() {
  try {
    MSCCLPP_CUDATHROW(cudaSetDevice(0));
    auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(0, 1);
    bootstrap->initialize(mscclpp::TcpBootstrap::createUniqueId());
    mscclpp::Communicator communicator(bootstrap);
    auto connection = communicator.connect(mscclpp::Transport::CudaIpc, 0).get();
    constexpr size_t count = 1024;
    constexpr size_t bytes = count * sizeof(int);
    auto src = mscclpp::GpuBuffer(bytes).memory();
    auto dst = mscclpp::GpuBuffer(bytes).memory();
    auto srcMemory = communicator.registerMemory(src.get(), bytes, mscclpp::Transport::CudaIpc);
    auto dstMemory = communicator.registerMemory(dst.get(), bytes, mscclpp::Transport::CudaIpc);
    std::vector<int> expected(count, 777);
    std::vector<int> actual(count);
    MSCCLPP_CUDATHROW(cudaMemcpy(src.get(), expected.data(), bytes, cudaMemcpyHostToDevice));

    {
      mscclpp::ProxyService service;
      service.stopProxy();
      service.stopProxy();
    }
    std::cout << "unstarted and repeated stop passed" << std::endl;

    for (int iteration = 0; iteration < 12; ++iteration) {
      auto service = std::make_unique<mscclpp::ProxyService>();
      auto srcId = service->addMemory(srcMemory);
      auto dstId = service->addMemory(dstMemory);
      auto semaphoreId = service->buildAndAddSemaphore(communicator, connection);
      auto channel = std::make_unique<mscclpp::PortChannel>(service->portChannel(semaphoreId, dstId, srcId));
      for (int run = 0; run < 2; ++run) {
        MSCCLPP_CUDATHROW(cudaMemset(dst.get(), 0, bytes));
        service->startProxy(true);
        transfer<<<1, 1>>>(channel->deviceHandle(), bytes);
        MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
        MSCCLPP_CUDATHROW(cudaMemcpy(actual.data(), dst.get(), bytes, cudaMemcpyDeviceToHost));
        if (actual != expected) {
          std::cerr << "port channel transfer mismatch at iteration " << iteration << ", run " << run << std::endl;
          return 1;
        }
        if (run == 0) {
          service->stopProxy();
          service->stopProxy();
        }
      }
      if (iteration == 0) std::cout << "explicit stop, repeated stop, and restart passed" << std::endl;
      if (iteration % 3 == 0) {
        service->stopProxy();
        service->stopProxy();
        service.reset();
      } else if (iteration % 3 == 1) {
        channel.reset();
        service.reset();
      } else {
        // Retain the channel while destroying the service through its virtual base.
        std::unique_ptr<mscclpp::BaseProxyService> base = std::move(service);
        base.reset();
        // Give stale callbacks time to execute under host memory sanitizers.
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        channel.reset();
      }
    }
    std::cout << "automatic destruction with and without a retained channel passed" << std::endl;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
