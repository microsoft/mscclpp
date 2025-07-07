// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include <algorithm>
#include <mscclpp/nvls.hpp>
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
  auto mcPtr = reinterpret_cast<mscclpp::f32x1*>(gConstSwitchChan.mcPtr);
  auto val = mscclpp::SwitchChannelDeviceHandle::multimemLoadReduce(mcPtr);
  mscclpp::SwitchChannelDeviceHandle::multimemStore(val, mcPtr);
}

TEST_F(SwitchChannelTest, SimpleReduce) {
  if (gEnv->rank >= numRanksToUse) return;

  std::vector<int> ranks;
  for (int i = 0; i < numRanksToUse; i++) {
    ranks.push_back(i);
  }

  auto buffer = mscclpp::GpuBuffer<float>(1024);
  float data = gEnv->rank + 1.0f;
  MSCCLPP_CUDATHROW(cudaMemcpy(buffer.data(), &data, sizeof(data), cudaMemcpyHostToDevice));

  auto nvlsConnection = mscclpp::connectNvlsCollective(communicator, ranks, 1024);
  auto deviceMulticastPointer = nvlsConnection->bindAllocatedMemory(CUdeviceptr(buffer.data()), 1024);
  auto deviceHandle = deviceMulticastPointer.deviceHandle();

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
