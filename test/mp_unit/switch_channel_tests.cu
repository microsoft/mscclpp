// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <algorithm>
#include <mscclpp/switch_channel.hpp>
#include <mscclpp/switch_channel_device.hpp>

#include "mp_unit_tests.hpp"

void SwitchChannelTest::SetUp() {
  // Need at least two ranks within a node
  if (gEnv->nRanksPerNode < 2) {
    SKIP_TEST();
  }
  if (!mscclpp::isNvlsSupported()) {
    SKIP_TEST();
  }
  // Use only two ranks
  setNumRanksToUse(2);
  CommunicatorTestBase::SetUp();
}

void SwitchChannelTest::TearDown() { CommunicatorTestBase::TearDown(); }

__constant__ mscclpp::SwitchChannelDeviceHandle gConstSwitchChan;
__constant__ mscclpp::SwitchChannelDeviceHandle gConstSwitchChan1;
__constant__ mscclpp::SwitchChannelDeviceHandle gConstSwitchChan2;
__constant__ mscclpp::SwitchGroupSemaphoreDeviceHandle gConstSwitchGroupSema;

__global__ void kernelSwitchReduce() {
#if (CUDA_NVLS_API_AVAILABLE) && (__CUDA_ARCH__ >= 900)
  auto val = gConstSwitchChan.reduce<mscclpp::f32x1>(0);
  gConstSwitchChan.broadcast(0, val);
#endif  // (CUDA_NVLS_API_AVAILABLE) && (__CUDA_ARCH__ >= 900)
}

__global__ void kernelSwitchReduceTwo() {
#if (CUDA_NVLS_API_AVAILABLE) && (__CUDA_ARCH__ >= 900)
  auto val1 = gConstSwitchChan1.reduce<mscclpp::f32x1>(0);
  gConstSwitchChan1.broadcast(0, val1);
  auto val2 = gConstSwitchChan2.reduce<mscclpp::f32x1>(0);
  gConstSwitchChan2.broadcast(0, val2);
#endif  // (CUDA_NVLS_API_AVAILABLE) && (__CUDA_ARCH__ >= 900)
}

TEST(SwitchChannelTest, SimpleAllReduce) {
  if (gEnv->rank >= numRanksToUse) return;

  std::vector<int> ranks;
  for (int i = 0; i < numRanksToUse; i++) {
    ranks.push_back(i);
  }

  auto buffer = mscclpp::GpuBuffer<float>(1024);
  float data = gEnv->rank + 1.0f;
  MSCCLPP_CUDATHROW(cudaMemcpy(buffer.data(), &data, sizeof(data), cudaMemcpyHostToDevice));

  auto nvlsConnection = mscclpp::connectNvlsCollective(communicator, ranks, 1024);
  auto switchChannel = nvlsConnection->bindAllocatedMemory(CUdeviceptr(buffer.data()), 1024);
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
  if (result != expected) {
    std::cerr << "Expected " << expected << " but got " << result << " for rank " << gEnv->rank << std::endl;
  }
  ASSERT_EQ(result, expected);
}

TEST(SwitchChannelTest, TwoChannelsSameConnection) {
  if (gEnv->rank >= numRanksToUse) return;

  std::vector<int> ranks;
  for (int i = 0; i < numRanksToUse; i++) {
    ranks.push_back(i);
  }

  const size_t bufSize = 1024;
  auto buffer1 = mscclpp::GpuBuffer<float>(bufSize / sizeof(float));
  auto buffer2 = mscclpp::GpuBuffer<float>(bufSize / sizeof(float));
  float data1 = (gEnv->rank + 1.0f) * 1.0f;
  float data2 = (gEnv->rank + 1.0f) * 10.0f;
  MSCCLPP_CUDATHROW(cudaMemcpy(buffer1.data(), &data1, sizeof(data1), cudaMemcpyHostToDevice));
  MSCCLPP_CUDATHROW(cudaMemcpy(buffer2.data(), &data2, sizeof(data2), cudaMemcpyHostToDevice));

  const size_t connSize = buffer1.bytes() + buffer2.bytes();
  auto nvlsConnection = mscclpp::connectNvlsCollective(communicator, ranks, connSize);

  auto switchChannel1 = nvlsConnection->bindAllocatedMemory(CUdeviceptr(buffer1.data()), bufSize);
  auto switchChannel2 = nvlsConnection->bindAllocatedMemory(CUdeviceptr(buffer2.data()), bufSize);

  auto deviceHandle1 = switchChannel1.deviceHandle();
  auto deviceHandle2 = switchChannel2.deviceHandle();

  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gConstSwitchChan1, &deviceHandle1, sizeof(deviceHandle1)));
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gConstSwitchChan2, &deviceHandle2, sizeof(deviceHandle2)));
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  communicator->bootstrap()->barrier();

  if (gEnv->rank == 0) {
    kernelSwitchReduceTwo<<<1, 1>>>();
    MSCCLPP_CUDATHROW(cudaGetLastError());
    MSCCLPP_CUDATHROW(cudaDeviceSynchronize());
  }
  communicator->bootstrap()->barrier();

  float result1, result2;
  MSCCLPP_CUDATHROW(cudaMemcpy(&result1, buffer1.data(), sizeof(result1), cudaMemcpyDeviceToHost));
  MSCCLPP_CUDATHROW(cudaMemcpy(&result2, buffer2.data(), sizeof(result2), cudaMemcpyDeviceToHost));

  float expected1 = 0.0f;
  float expected2 = 0.0f;
  for (int i = 0; i < numRanksToUse; i++) {
    expected1 += (i + 1.0f) * 1.0f;
    expected2 += (i + 1.0f) * 10.0f;
  }
  ASSERT_EQ(result1, expected1);
  ASSERT_EQ(result2, expected2);
}

__global__ void kernelSwitchGroupSignalWait() {
#if (CUDA_NVLS_API_AVAILABLE) && (__CUDA_ARCH__ >= 900)
  // All GPUs signal and wait - acts as a barrier ensuring all data is ready
  gConstSwitchGroupSema.signal();
  gConstSwitchGroupSema.wait();

  // After barrier, reduce and broadcast (safe because all data is visible)
  auto val = gConstSwitchChan.reduce<mscclpp::f32x1>(0);
  gConstSwitchChan.broadcast(0, val);

  // Signal and wait again to ensure broadcast is complete before host reads
  gConstSwitchGroupSema.signal();
  gConstSwitchGroupSema.wait();
#endif  // (CUDA_NVLS_API_AVAILABLE) && (__CUDA_ARCH__ >= 900)
}

TEST(SwitchChannelTest, GroupSignalWait) {
  if (gEnv->rank >= numRanksToUse) return;

  std::vector<int> ranks;
  for (int i = 0; i < numRanksToUse; i++) {
    ranks.push_back(i);
  }

  auto buffer = mscclpp::GpuBuffer<float>(256);
  float data = gEnv->rank + 1.0f;
  MSCCLPP_CUDATHROW(cudaMemcpy(buffer.data(), &data, sizeof(data), cudaMemcpyHostToDevice));

  auto flagBuffer = mscclpp::GpuBuffer<uint32_t>(1);

  size_t connSize = buffer.bytes() + flagBuffer.bytes();
  auto nvlsConnection = mscclpp::connectNvlsCollective(communicator, ranks, connSize);

  auto switchChannel = nvlsConnection->bindAllocatedMemory(CUdeviceptr(buffer.data()), buffer.bytes());
  auto flagChannel = nvlsConnection->bindAllocatedMemory(CUdeviceptr(flagBuffer.data()), flagBuffer.bytes());

  auto semaphore = mscclpp::SwitchGroupSemaphore(flagChannel, numRanksToUse);

  auto deviceHandle = switchChannel.deviceHandle();
  auto semaHandle = semaphore.deviceHandle();

  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gConstSwitchChan, &deviceHandle, sizeof(deviceHandle)));
  MSCCLPP_CUDATHROW(cudaMemcpyToSymbol(gConstSwitchGroupSema, &semaHandle, sizeof(semaHandle)));
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  communicator->bootstrap()->barrier();

  // All ranks launch the kernel - signal/wait replaces the host barrier for GPU synchronization
  kernelSwitchGroupSignalWait<<<1, 1>>>();
  MSCCLPP_CUDATHROW(cudaGetLastError());
  MSCCLPP_CUDATHROW(cudaDeviceSynchronize());

  communicator->bootstrap()->barrier();

  float result;
  MSCCLPP_CUDATHROW(cudaMemcpy(&result, buffer.data(), sizeof(result), cudaMemcpyDeviceToHost));

  float expected = 0.0f;
  for (int i = 0; i < numRanksToUse; i++) {
    expected += i + 1.0f;
  }
  if (result != expected) {
    std::cerr << "Expected " << expected << " but got " << result << " for rank " << gEnv->rank << std::endl;
  }
  ASSERT_EQ(result, expected);
}
