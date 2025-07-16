// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <connection.hpp>
#include <random>
#include <vector>

class IndirectConnectionTest : public ::testing::Test {
 protected:
  void SetUp() override { ctx = mscclpp::Context::create(); }
  std::shared_ptr<mscclpp::Context> ctx;
};

TEST_F(IndirectConnectionTest, CPUGPUDataTransfer) {
  mscclpp::Device dst(mscclpp::DeviceType::GPU, 1);
  mscclpp::Device fwd(mscclpp::DeviceType::GPU, 2);
  size_t granularity = 20'000'000;
  size_t n = size_t(granularity * 15);

  // generate random data
  int* dummy;
  cudaMallocHost((void**)&dummy, n * sizeof(int));
  std::mt19937 gen(std::random_device{}());
  std::generate(dummy, dummy + n, [&]() { return gen() % (1 << 30); });

  int* dummy_device;
  cudaSetDevice(dst.id);
  cudaMalloc((void**)&dummy_device, n * sizeof(int));

  // generate IO tasks
  std::vector<mscclpp::IOTask> tasks;
  for (size_t t = 0; t < n; t += granularity) {
    auto bytes = std::min(granularity, n - t) * sizeof(int);
    auto cpu_memory = ctx->registerMemory(dummy + t, bytes, mscclpp::NoTransports);
    auto gpu_memory = ctx->registerMemory(dummy_device + t, bytes, mscclpp::Transport::CudaIpc);
    tasks.push_back(mscclpp::IOTask{gpu_memory, cpu_memory});
  }

  // register buffers
  int *buf1, *buf2;
  cudaSetDevice(fwd.id);
  int canAccess;
  cudaDeviceCanAccessPeer(&canAccess, fwd.id, dst.id);
  if (canAccess) {
    std::cout << "Enabling peer access from " << fwd.id << " to " << dst.id << std::endl;
    cudaDeviceEnablePeerAccess(dst.id, 0);
  }
  cudaMalloc((void**)&buf1, granularity * sizeof(int));
  cudaMalloc((void**)&buf2, granularity * sizeof(int));
  auto fwd_buf1 = ctx->registerMemory(buf1, granularity * sizeof(int), mscclpp::Transport::CudaIpc);
  auto fwd_buf2 = ctx->registerMemory(buf2, granularity * sizeof(int), mscclpp::Transport::CudaIpc);
  auto buf_ptr = std::make_shared<mscclpp::DoubleBuffer>(fwd_buf1, fwd_buf2);

  // register scheduler
  auto scheduler_ptr = std::make_shared<mscclpp::VortexScheduler>(buf_ptr, granularity * sizeof(int));

  // launch tasks
  for (size_t _ = 0; _ < 4; _ ++) {
    auto connection = mscclpp::IndirectConnection(buf_ptr, scheduler_ptr);
    auto start = std::chrono::high_resolution_clock::now();
    connection.launch(tasks);
    connection.sync();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end - start);

    std::cout << "Time: " << duration.count() << " seconds" << std::endl;
    std::cout << "Bandwidth: "
              << (n * sizeof(int)) / duration.count() / (1e9) << " GB/s" << std::endl;

  }

  // validate
  int* validate;
  cudaMallocHost((void**)&validate, n * sizeof(int));
  cudaMemcpy(validate, dummy_device, n * sizeof(int), cudaMemcpyDefault);

  for (size_t i = 0; i < n; ++i) {
    EXPECT_EQ(validate[i], dummy[i]);
  }

  // cleanup
  cudaFree(dummy_device);
  cudaFree(buf1);
  cudaFree(buf2);
  cudaFreeHost(dummy);
  cudaFreeHost(validate);
}

TEST_F(IndirectConnectionTest, GPUCPUDataTransfer) {
  mscclpp::Device src(mscclpp::DeviceType::GPU, 1);
  mscclpp::Device fwd(mscclpp::DeviceType::GPU, 2);
  size_t granularity = 20'000'000;
  size_t n = size_t(granularity * 15);

  // generate random data
  int* dst_host, * dummy;
  cudaMallocHost((void**)&dummy, n * sizeof(int));
  cudaMallocHost((void**)&dst_host, n * sizeof(int));
  std::mt19937 gen(std::random_device{}());
  std::generate(dummy, dummy + n, [&]() { return gen() % (1 << 30); });

  
  int* dummy_device;
  cudaSetDevice(src.id);
  cudaMalloc((void**)&dummy_device, n * sizeof(int));
  cudaMemcpy(dummy_device, dummy, n * sizeof(int), cudaMemcpyHostToDevice);

  // generate IO tasks
  std::vector<mscclpp::IOTask> tasks;
  for (size_t t = 0; t < n; t += granularity) {
    auto bytes = std::min(granularity, n - t) * sizeof(int);
    auto gpu_memory = ctx->registerMemory(dummy_device + t, bytes, mscclpp::Transport::CudaIpc);
    auto cpu_memory = ctx->registerMemory(dst_host + t, bytes, mscclpp::NoTransports);
    tasks.push_back(mscclpp::IOTask{cpu_memory, gpu_memory});
  }

  // register buffers
  int *buf1, *buf2;
  cudaSetDevice(src.id);
  int canAccess;
  cudaDeviceCanAccessPeer(&canAccess, src.id, fwd.id);
  if (canAccess) {
    std::cout << "Enabling peer access from " << src.id << " to " << fwd.id << std::endl;
    cudaDeviceEnablePeerAccess(fwd.id, 0);
  }
  cudaMalloc((void**)&buf1, granularity * sizeof(int));
  cudaMalloc((void**)&buf2, granularity * sizeof(int));
  auto fwd_buf1 = ctx->registerMemory(buf1, granularity * sizeof(int), mscclpp::Transport::CudaIpc);
  auto fwd_buf2 = ctx->registerMemory(buf2, granularity * sizeof(int), mscclpp::Transport::CudaIpc);
  auto buf_ptr = std::make_shared<mscclpp::DoubleBuffer>(fwd_buf1, fwd_buf2);

  // register scheduler
  auto scheduler_ptr = std::make_shared<mscclpp::VortexScheduler>(buf_ptr, granularity * sizeof(int));

  // launch tasks
  for (size_t _ = 0; _ < 4; _ ++) {
    auto connection = mscclpp::IndirectConnection(buf_ptr, scheduler_ptr);
    auto start = std::chrono::high_resolution_clock::now();
    connection.launch(tasks);
    connection.sync();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end - start);

    std::cout << "Time: " << duration.count() << " seconds" << std::endl;
    std::cout << "Bandwidth: "
              << (n * sizeof(int)) / duration.count() / (1e9) << " GB/s" << std::endl;
  }

  // validate

  for (size_t i = 0; i < n; ++i) {
    EXPECT_EQ(dummy[i], dst_host[i]);
  }

  // cleanup
  cudaFree(dummy_device);
  cudaFree(buf1);
  cudaFree(buf2);
  cudaFreeHost(dummy);
  cudaFreeHost(dst_host);
}
