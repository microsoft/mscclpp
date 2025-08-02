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
  bool validate_answer = false;
};

TEST_F(IndirectConnectionTest, CPUGPUDataTransfer) {
  mscclpp::Device dst(mscclpp::DeviceType::GPU, 1);
  mscclpp::Device fwd(mscclpp::DeviceType::GPU, 2);
  uint64_t granularity = 20'000'000;
  uint64_t n = uint64_t(granularity * 30);

  // generate random data
  int* dummy;
  cudaMallocHost((void**)&dummy, n * sizeof(int));
  std::mt19937 gen(std::random_device{}());
  std::generate(dummy, dummy + n, [&]() { return gen() % (1 << 30); });

  // reserve memory on destination GPU
  int* dummy_device;
  cudaSetDevice(dst.id);
  cudaMalloc((void**)&dummy_device, n * sizeof(int));

  // enable GPU peer access
  cudaSetDevice(fwd.id);
  int canAccess;
  cudaDeviceCanAccessPeer(&canAccess, fwd.id, dst.id);
  if (canAccess) {
    std::cout << "Enabling peer access from " << fwd.id << " to " << dst.id << std::endl;
    cudaDeviceEnablePeerAccess(dst.id, 0);
  }

  // create local endpoint
  auto localEndpoint = ctx->createEndpoint(mscclpp::EndpointConfig());

  // register scheduler
  auto scheduler_ptr = std::make_shared<mscclpp::VortexScheduler>(ctx, granularity, fwd);

  // launch writes and measure performance
  for (uint64_t _ = 0; _ < 4; _ ++) {
    auto connection = mscclpp::IndirectConnection(ctx, localEndpoint, scheduler_ptr);
    auto start = std::chrono::high_resolution_clock::now();
    connection.write(
      ctx->registerMemory(dummy_device, n * sizeof(int), mscclpp::Transport::CudaIpc),
      0,
      ctx->registerMemory(dummy, n * sizeof(int), mscclpp::NoTransports),
      0,
      n * sizeof(int)
    );
    connection.flush();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end - start);

    std::cout << "Time: " << duration.count() << " seconds" << std::endl;
    std::cout << "Bandwidth: "
              << (n * sizeof(int)) / duration.count() / (1e9) << " GB/s" << std::endl;

  }

  // validate
  if (validate_answer) {
    int* validate;
    cudaMallocHost((void**)&validate, n * sizeof(int));
    cudaMemcpy(validate, dummy_device, n * sizeof(int), cudaMemcpyDefault);

    for (uint64_t i = 0; i < n; ++i) {
      EXPECT_EQ(validate[i], dummy[i]) << "Mismatch at index " << i;
    }
    cudaFreeHost(validate);
  }
  // cleanup
  cudaFree(dummy_device);
  cudaFreeHost(dummy);
}

TEST_F(IndirectConnectionTest, GPUCPUDataTransfer) {
  mscclpp::Device src(mscclpp::DeviceType::GPU, 1);
  mscclpp::Device fwd(mscclpp::DeviceType::GPU, 2);
  uint64_t granularity = 20'000'000;
  uint64_t n = uint64_t(granularity * 30);

  // generate random data
  int* dst_host, * dummy;
  cudaMallocHost((void**)&dummy, n * sizeof(int));
  cudaMallocHost((void**)&dst_host, n * sizeof(int));
  std::mt19937 gen(std::random_device{}());
  std::generate(dummy, dummy + n, [&]() { return gen() % (1 << 30); });

  // reserve memory on source GPU
  int* dummy_device;
  cudaSetDevice(src.id);
  cudaMalloc((void**)&dummy_device, n * sizeof(int));
  cudaMemcpy(dummy_device, dummy, n * sizeof(int), cudaMemcpyHostToDevice);
  
  // enable GPU peer access
  cudaSetDevice(src.id);
  int canAccess;
  cudaDeviceCanAccessPeer(&canAccess, src.id, fwd.id);
  if (canAccess) {
    std::cout << "Enabling peer access from " << src.id << " to " << fwd.id << std::endl;
    cudaDeviceEnablePeerAccess(fwd.id, 0);
  }
  auto localEndpoint = ctx->createEndpoint(mscclpp::EndpointConfig());

  // register scheduler
  auto scheduler_ptr = std::make_shared<mscclpp::VortexScheduler>(ctx, granularity, fwd);

  // launch writes and measure performance
  for (uint64_t _ = 0; _ < 4; _ ++) {
    auto connection = mscclpp::IndirectConnection(ctx, localEndpoint, scheduler_ptr);
    auto start = std::chrono::high_resolution_clock::now();
    connection.write(
      ctx->registerMemory(dst_host, n * sizeof(int), mscclpp::NoTransports),
      0,
      ctx->registerMemory(dummy_device, n * sizeof(int), mscclpp::Transport::CudaIpc),
      0,
      n * sizeof(int)
    );
    connection.flush();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<double>(end - start);

    std::cout << "Time: " << duration.count() << " seconds" << std::endl;
    std::cout << "Bandwidth: "
              << (n * sizeof(int)) / duration.count() / (1e9) << " GB/s" << std::endl;
  }

  // validate
  if (validate_answer) {
    for (uint64_t i = 0; i < n; ++i) {
      EXPECT_EQ(dummy[i], dst_host[i]) << "Mismatch at index " << i;
    }
  }
  
  // cleanup
  cudaFree(dummy_device);
  cudaFreeHost(dummy);
  cudaFreeHost(dst_host);
}
