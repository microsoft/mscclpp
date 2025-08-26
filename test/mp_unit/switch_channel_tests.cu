// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <algorithm>
#include <mscclpp/switch_channel.hpp>
#include <mscclpp/switch_channel_device.hpp>

#include "mp_unit_tests.hpp"

void SwitchChannelTest::SetUp() {
  // Need at least two ranks within a node
  if (gEnv->nRanksPerNode < 2) {
    GTEST_SKIP();
  }
  if (!mscclpp::isNvlsSupported()) {
    GTEST_SKIP();
  }
  // Use only two ranks
  setNumRanksToUse(2);
  CommunicatorTestBase::SetUp();
}

void SwitchChannelTest::TearDown() { CommunicatorTestBase::TearDown(); }

__constant__ mscclpp::SwitchChannelDeviceHandle gConstSwitchChan;

__global__ void kernelSwitchReduce() {
#if (CUDA_NVLS_API_AVAILABLE) && (__CUDA_ARCH__ >= 900)
  auto val = gConstSwitchChan.reduce<mscclpp::f32x1>(0);
  gConstSwitchChan.broadcast(0, val);
#endif  // (CUDA_NVLS_API_AVAILABLE) && (__CUDA_ARCH__ >= 900)
}

TEST_F(SwitchChannelTest, SimpleAllReduce) {
  if (gEnv->rank >= numRanksToUse) return;

  std::vector<int> ranks;
  for (int i = 0; i < numRanksToUse; i++) {
    ranks.push_back(i);
  }

  auto buffer = mscclpp::GpuBuffer<float>(1024 * 1024);
  float data = gEnv->rank + 1.0f;
  MSCCLPP_CUDATHROW(cudaMemcpy(buffer.data(), &data, sizeof(data), cudaMemcpyHostToDevice));

  std::vector<std::shared_ptr<mscclpp::Connection>> rootConns;
  std::shared_ptr<mscclpp::Connection> myConn;

  mscclpp::EndpointConfig cfg;
  cfg.transport = mscclpp::Transport::CudaIpc;
  cfg.device = mscclpp::DeviceType::GPU;
  cfg.nvls.numDevices = numRanksToUse;
  cfg.nvls.bufferSize = buffer.bytes();
  if (communicator->bootstrap()->getRank() == 0) {
    cfg.nvls.isRoot = true;
    auto rootEndpoint = communicator->context()->createEndpoint(cfg);
    for (int peer = 1; peer < numRanksToUse; ++peer) {
      rootConns.emplace_back(communicator->connect(rootEndpoint, peer).get());
    }
    cfg.nvls.isRoot = false;
    auto endpoint = communicator->context()->createEndpoint(cfg);
    auto rootSelfConn = communicator->context()->connect(rootEndpoint, endpoint);
    myConn = communicator->context()->connect(endpoint, rootEndpoint);
  } else {
    cfg.nvls.isRoot = false;
    auto endpoint = communicator->context()->createEndpoint(cfg);
    myConn = communicator->connect(endpoint, 0).get();
  }

  // auto nvlsConnection = mscclpp::connectNvlsCollective(communicator, ranks, buffer.bytes());
  // nvlsConnection->bindMemory(CUdeviceptr(buffer.data()), buffer.bytes());
  // mscclpp::SwitchChannel switchChannel(nvlsConnection);
  mscclpp::SwitchChannel switchChannel(myConn, buffer.data(), buffer.bytes());
  auto deviceHandle = switchChannel.deviceHandle();

  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gConstSwitchChan, &deviceHandle, sizeof(deviceHandle)));
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  communicator->bootstrap()->barrier();

  if (gEnv->rank == 0) {
    kernelSwitchReduce<<<1, 1>>>();
    MSCCLPP_CUDATHROW(cudaGetLastError());
    MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  }
  communicator->bootstrap()->barrier();

  float result;
  MSCCLPP_CUDATHROW(cudaMemcpy(&result, buffer.data(), sizeof(result), cudaMemcpyDeviceToHost));

  float expected = 0.0f;
  for (int i = 0; i < numRanksToUse; i++) {
    expected += i + 1.0f;
  }
  ASSERT_EQ(result, expected) << "Expected " << expected << " but got " << result << " for rank " << gEnv->rank;
}
